"""q27fuse.draftscreen - the MTP drafter's draft head on a rotated 1-bit screen (track "draftscreen").

On by default whenever the plugin is on (Q27_FUSE) and the server runs 1Cat's MTP refined draft head; Q27_DS=0 keeps
1Cat's refined head.

1Cat's draft head (qwen3_5_mtp_refined_head.MTPRefinedHead) streams a full-vocabulary NVFP4 copy of the target LM head
(179 MB per rank per pass), takes each rank's top-32 and reranks them exactly against a resident FP16 copy of the head
(636 MB per rank); the draft token is the best exact score over the four ranks. This module keeps the exact rerank and
1Cat's cross-device top-1, and replaces the proposal pass with a rotated 1-bit sign screen (40 MB per rank):
  codes     r_i = blockdiag_5(H_1024 D_b) w_i / 32 (w_i = the effective FP16 head row, D_b fixed random signs), sign bits
            of r_i and the per-row factor f_i = ||r_i||^2 / ||r_i||_1; built at load from the head weights (~0.6 s per
            rank), cached on disk under $Q27_DS_CACHE (default ~/.cache/q27fuse/draftscreen; an unwritable or
            missing directory only means a rebuild at every start) keyed by the checkpoint file and the shard
            (csrc/draft_screen.cu has the bit layout);
  screen    s_i = f_i <sign(r_i), q'>, q' = the same rotation of the input row (in-kernel); tensor cores on +-1 FP16
            operands (3+ rows per pass), or popcounts against 4-bit planes of q' (1-2 rows, Q27_DS_PCMAX); per-CTA
            top-NC (Q27_DS_NC, default 8) per row;
  merge     the KEEP (= 1Cat's 32) best screen candidates of the rank, reranked exactly (FP32 dot of the FP16 input with the
            FP16 head row, rounded to FP16, lowest id among ties) -> (score, start + id) -> 1Cat's top-1 exchange.
Rerank rows (Q27_DS_RERANK): "fp8" (default) = the checkpoint's FP8 E4M3 rows (318 MB per rank) x row scale, decoded to
exactly fp16(code x scale) = 1Cat's dense rows (checked against 1Cat's own reconstruction when the cache is built);
"dense" = 1Cat's FP16 reconstruction (636 MB) with 1Cat's Triton rerank kernels (bitwise 1Cat's scores).
Rows: the screen serves 1..Q27_DS_MAXR (default 40 = 1Cat's limit) rows per pass; with MAXR >= 40 1Cat's refined head is
not built at all (its 179 MB NVFP4 copy and 636 MB FP16 copy are never allocated), below that it is built and serves
the rows the screen does not. Only drafts change; the target model's outputs do not depend on the draft head."""
import hashlib
import json
import logging
import os

import torch

log = logging.getLogger("q27fuse")

HD = 5120
NG = 10
SEED = 1234
VERSION = 2
CACHE = os.environ.get("Q27_DS_CACHE", "").strip() or os.path.join(
    os.path.expanduser("~"), ".cache", "q27fuse", "draftscreen"
)


def _env_int(name, default):
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


NC = _env_int("Q27_DS_NC", 8)              # candidates per CTA and row
KEEP = _env_int("Q27_DS_KEEP", 32)         # candidates reranked per rank and row (1Cat: 32)
MAXR = min(64, _env_int("Q27_DS_MAXR", 40))  # rows per pass the screen serves
WARPS = _env_int("Q27_DS_WARPS", 12)
PCMAX = _env_int("Q27_DS_PCMAX", 2)        # rows per pass up to which the popcount kernel runs (0: never; max 2)
RERANK = os.environ.get("Q27_DS_RERANK", "fp8").strip() or "fp8"

_X = None


def ext():
    global _X
    if _X is None:
        from .ops import _load
        _X = _load("q27fuse_ds", "draft_screen.cu", "_ds")
    return _X


def num_ctas(dev=None) -> int:
    """screen CTAs per query tile: one per SM (rows per CTA must stay <= 1024)"""
    return torch.cuda.get_device_properties(dev or torch.cuda.current_device()).multi_processor_count


# ---------------------------------------------------------------------------------------------------------- codes
def rotation_signs() -> torch.Tensor:
    """D: 5 x 1024 random signs (the drafthead track's rbq rotation without its permutation, which does not change a
    full-width inner product)"""
    g = torch.Generator().manual_seed(SEED)
    return torch.cat([(torch.randint(0, 2, (1024,), generator=g) * 2 - 1).float() for _ in range(5)])


def sign_bits(D: torch.Tensor) -> torch.Tensor:
    """D [5120] +-1 -> int32 [160]: word w bit i = (D[32 w + i] < 0)"""
    b = (D.view(160, 32) < 0).to(torch.int64) << torch.arange(32, device=D.device).view(1, 32)
    w = b.sum(1)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


def hadamard(n=1024) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def rotate(w: torch.Tensor, D: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """[n, 5120] -> [n, 5120] FP32 rotated rows"""
    n = w.shape[0]
    return ((w.float().view(n, 5, 1024) * D.view(5, 1024)) @ H.T / 32.0).reshape(n, HD)


def pack_codes(neg: torch.Tensor) -> torch.Tensor:
    """sign bits [V, 5120] bool (True = negative) -> int32 [V/8, 10, 32 lanes, 4 words] (draft_screen.cu layout)"""
    V = neg.shape[0]
    b = neg.view(V, NG, 4, 8, 4, 2, 2).to(torch.int64)               # [V, g, t, i', q, m0, hi]
    sh = (16 * torch.arange(2).view(1, 2) + 2 * torch.arange(8).view(8, 1, 1) + torch.arange(2).view(1, 2, 1))
    sh = sh.view(8, 1, 2, 2).to(neg.device)                           # [i', 1, m0, hi] -> 16 hi + 2 i' + m0
    words = (b << sh).sum(dim=(3, 5, 6))                              # [V, g, t, q]
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    u = V // 8
    words = words.view(u, 2, 4, NG, 4, 4)                             # [u, jh, jl, g, t, q]
    return words.permute(0, 3, 1, 5, 2, 4).reshape(u, NG, 32, 4)      # [u, g, jh, q, jl, t] = [u, g, lane, t]


def build_codes(rows_fn, V: int, dev):
    """rows_fn(begin, end) -> exact FP16 head rows [end - begin, 5120] on dev; returns codes int32, rowf f32 [V]
    (||r||^2 / ||r||_1), rowpb f32 [V] (number of negative signs)"""
    D, H = rotation_signs().to(dev), hadamard().to(dev)
    codes = torch.empty(V // 8, NG, 32, 4, dtype=torch.int32, device=dev)
    rowf = torch.empty(V, dtype=torch.float32, device=dev)
    rowpb = torch.empty(V, dtype=torch.float32, device=dev)
    step = 4096
    for b in range(0, V, step):
        e = min(V, b + step)
        r = rotate(rows_fn(b, e), D, H)
        rowf[b:e] = r.pow(2).sum(1) / r.abs().sum(1).clamp_min(1e-30)
        neg = r < 0
        rowpb[b:e] = neg.sum(1).float()
        codes[b // 8:e // 8] = pack_codes(neg)
    return codes, rowf, rowpb


def _ckpt_key(vllm_config, start: int, end: int) -> tuple[str, str, str]:
    """(cache dir, checkpoint file holding lm_head, key) for this shard"""
    model = vllm_config.model_config.model
    path = os.path.realpath(model)
    idx = os.path.join(path, "model.safetensors.index.json")
    fname = "model.safetensors"
    if os.path.exists(idx):
        with open(idx) as f:
            fname = json.load(f)["weight_map"].get("lm_head.weight", fname)
    fp = os.path.join(path, fname)
    st = os.stat(fp)
    key = hashlib.sha1(f"{fp}|{st.st_size}|{int(st.st_mtime)}|{start}|{end}|v{VERSION}|s{SEED}".encode()).hexdigest()[:16]
    return os.path.join(CACHE, key), fp, key


def _load_fp8_rows(fp: str, start: int, end: int, dev):
    from safetensors import safe_open
    with safe_open(fp, "pt", device="cpu") as f:
        w = f.get_slice("lm_head.weight")[start:end]
        s = f.get_slice("lm_head.weight_scale")[start:end]
    assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (end - start, HD), (w.dtype, w.shape)
    return w.to(dev).view(torch.uint8).contiguous(), s.float().reshape(-1).to(dev).contiguous()


def load_or_build(cfile: str, dense_fn, w8: torch.Tensor, s8: torch.Tensor, dev) -> dict:
    """codes, rowf, rowpb and exact_fp8 of one shard: from cfile if it exists, else built from dense_fn() (-> the exact
    FP16 head rows [V, 5120] on dev, 1Cat's reconstruction) and saved to cfile (written to a temporary name, then
    renamed: concurrent ranks / servers never read a partial file; a failed save is logged and ignored).
    exact_fp8: fp16(checkpoint FP8 code x row scale) equals the dense rows in value (signed zeros may differ: -0 x scale
    vs the kernel's +0, irrelevant to a dot product)."""
    out = {}
    if os.path.exists(cfile):
        c = torch.load(cfile, map_location=dev, weights_only=True)
        out["codes"], out["rowf"], out["rowpb"] = c["codes"], c["rowf"], c["rowpb"]
        out["exact_fp8"] = bool(c["exact_fp8"])
        out["cache"] = "loaded"
        log.warning("q27fuse draftscreen: codes loaded from %s", cfile)
        return out
    import time
    t0 = time.time()
    dense = dense_fn()
    V = dense.shape[0]
    exact = all(torch.equal(dense[b:b + 8192].float(),
                            (w8[b:b + 8192].view(torch.float8_e4m3fn).float() * s8[b:b + 8192, None]).half().float())
                for b in range(0, V, 8192))
    codes, rowf, rowpb = build_codes(lambda b, e: dense[b:e], V, dev)
    out["dense"] = dense
    out["codes"], out["rowf"], out["rowpb"], out["exact_fp8"] = codes, rowf, rowpb, exact
    out["cache"] = "built"
    try:
        os.makedirs(os.path.dirname(cfile), exist_ok=True)
        tmp = cfile + f".tmp{os.getpid()}"
        torch.save({"codes": codes.cpu(), "rowf": rowf.cpu(), "rowpb": rowpb.cpu(), "exact_fp8": exact,
                    "version": VERSION, "seed": SEED}, tmp)
        os.replace(tmp, cfile)
        where = cfile
    except OSError as e:
        where = f"not cached ({e})"
    log.warning("q27fuse draftscreen: built codes in %.1f s (fp8 rows == 1Cat dense rows: %s) -> %s",
                time.time() - t0, exact, where)
    return out


def prepare(head, vllm_config, rerank: str):
    """codes, rowf (and the rerank rows) for this rank's shard; cached on disk"""
    from vllm.model_executor.models.qwen3_5_mtp_sm70_head import _hot_rows
    dev = head.weight.device
    start = head.shard_indices.org_vocab_start_index
    end = head.shard_indices.org_vocab_end_index
    V = end - start
    cdir, fp, key = _ckpt_key(vllm_config, start, end)
    cfile = os.path.join(cdir, f"shard_{start}_{end}.pt")
    hc = getattr(head, "q27_head", None)          # [head] the unified FP8 head copy (head.py): the rerank reads it
    if hc is not None and rerank == "fp8" and os.path.exists(cfile):
        out = load_or_build(cfile, None, None, None, dev)
        if out["exact_fp8"]:
            out.update(rerank="fp8q", hc=hc, start=start)
            return out
    w8, s8 = _load_fp8_rows(fp, start, end, dev)
    # the rows 1Cat reranks against (its _hot_rows: the head's own kernel on identity blocks); only built on a cache miss
    # ([head] with the unified head, _hot_rows runs our fp8_head.cu logits kernel, so the build-time check below also
    # proves that kernel's weights equal fp16(checkpoint code x scale))
    out = load_or_build(cfile, lambda: _hot_rows(head, torch.arange(V, device=dev, dtype=torch.int64)), w8, s8, dev)
    if hc is not None and rerank == "fp8" and out["exact_fp8"]:          # [head] no second FP8 copy
        del w8, s8
        out.update(rerank="fp8q", hc=hc, start=start)
        torch.cuda.empty_cache()
        return out
    if rerank == "fp8" and not out["exact_fp8"]:
        log.warning("q27fuse draftscreen: checkpoint FP8 rows differ from 1Cat's dense rows; rerank falls back to dense")
        rerank = "dense"
    if rerank == "fp8":
        out["w8"], out["s8"] = w8, s8
        out.pop("dense", None)
    else:
        del w8, s8
        if "dense" not in out:
            out["dense"] = _hot_rows(head, torch.arange(V, device=dev, dtype=torch.int64))
    out["rerank"] = rerank
    out["start"] = start
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------------------- op
def draft_head(x: torch.Tensor, codes: torch.Tensor, rowf: torch.Tensor, rowpb: torch.Tensor, dbits: torch.Tensor,
               rows: torch.Tensor, scales: torch.Tensor, ws: torch.Tensor, start: int, mode: int, G: int, nc: int,
               keep: int, warps: int, pcmax: int) -> torch.Tensor:
    """x [R, 5120] fp16 -> pairs [R, 2] f32 (best exact score of this rank's shard, start + id); mode 1 = FP16 rows +
    1Cat's Triton rerank, 2 = FP8 rows + fused rerank, 3 = FP16 rows + fused rerank"""
    x = x.contiguous()
    R = x.shape[0]
    X = ext()
    cand = torch.empty(R, G, nc, dtype=torch.int32, device=x.device)
    use16 = R > 8 and (R + 15) // 16 * 17 < (R + 7) // 8 * 10      # as draft_screen.cu
    qg = torch.empty(R * HD if use16 else 0, dtype=torch.float16, device=x.device)
    X.screen(x, dbits, codes, rowf, rowpb, cand, G, nc, warps, R <= pcmax, 0, qg)
    if mode == 1:
        from vllm.model_executor.models.qwen3_5_mtp_refined_head import _rerank_candidates
        ids = torch.empty(R, keep, dtype=torch.int32, device=x.device)
        X.merge(cand, G, nc, keep, x, 0, rows, scales, start, ids, ids, ws)
        return _rerank_candidates(x, rows, ids, start)
    pairs = torch.empty(R, 2, dtype=torch.float32, device=x.device)
    X.merge(cand, G, nc, keep, x, {2: 2, 4: 3}.get(mode, 1), rows, scales, start, pairs, pairs, ws)   # [head] 4: QPN8
    return pairs


def draft_head_fake(x, codes, rowf, rowpb, dbits, rows, scales, ws, start, mode, G, nc, keep, warps, pcmax):
    return x.new_empty((x.shape[0], 2), dtype=torch.float32)


_REGISTERED = False


def register_op():
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="q27_draft_head", op_func=draft_head, mutates_args=[], fake_impl=draft_head_fake)
    _REGISTERED = True


class ScreenHead(torch.nn.Module):
    """stands where 1Cat's MTPRefinedHead stands (Qwen3_5MTP._sm70_draft_head): get_top_tokens asks it for the local
    (score, id) pair; rows it does not serve go to 1Cat's refined head (if built) or return None (full head path)"""

    @torch.inference_mode()
    def __init__(self, head, fallback, data):
        super().__init__()
        self.head = head
        self.fallback = fallback
        self.shard_indices = head.shard_indices
        self.start = int(data["start"])
        self.register_buffer("codes", data["codes"], persistent=False)
        self.register_buffer("rowf", data["rowf"], persistent=False)
        self.register_buffer("rowpb", data["rowpb"], persistent=False)
        self.register_buffer("dbits", sign_bits(rotation_signs()).to(data["codes"].device), persistent=False)
        # merge workspace (per-row atomic max key + counter), zero at rest; rows <= 64
        self.register_buffer("ws", torch.zeros(128, dtype=torch.int64, device=data["codes"].device), persistent=False)
        ext()
        if data["rerank"] == "fp8q":                 # [head] the unified head copy: QPN8 codes + f32 row scales
            self.rows, self.scales = data["hc"].codes, data["hc"].s32     # (not buffers: the head owns them)
            self.mode = 4
        elif data["rerank"] == "fp8":
            self.register_buffer("rows", data["w8"], persistent=False)
            self.register_buffer("scales", data["s8"], persistent=False)
            self.mode = 2
        else:
            self.register_buffer("rows", data["dense"], persistent=False)
            self.register_buffer("scales", data["rowf"][:1], persistent=False)
            self.mode = 1 if os.environ.get("Q27_DS_TRITON_RERANK", "1") == "1" else 3
        self.G = num_ctas(data["codes"].device)
        self.maxr = MAXR

    @property
    def weight(self):
        return self.head.weight

    @property
    def quant_method(self):
        return self

    def apply(self, layer, x, bias=None):
        return self.head.quant_method.apply(self.head, x, bias)

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        return None

    def maybe_get_sm70_lm_head_top1_pair(self, hidden_states, bias=None):
        if (bias is not None or hidden_states.ndim != 2 or hidden_states.dtype != torch.float16
                or not 1 <= hidden_states.shape[0] <= self.maxr):
            if self.fallback is not None:
                return self.fallback.maybe_get_sm70_lm_head_top1_pair(hidden_states, bias)
            return None
        return torch.ops.vllm.q27_draft_head(hidden_states, self.codes, self.rowf, self.rowpb, self.dbits, self.rows,
                                             self.scales, self.ws, self.start, self.mode, self.G, NC, KEEP, WARPS, PCMAX)


def enabled() -> bool:
    """on unless Q27_DS=0 (the plugin itself is on: q27fuse.register calls apply() only then)"""
    return (os.environ.get("Q27_DS", "") or "1").strip() != "0"


def apply():
    """wrap 1Cat's refined-head construction (called by Qwen3_5MTP.prepare_sm70_draft_head before KV sizing)"""
    if not enabled():
        return
    from vllm.model_executor.models import qwen3_5_mtp_refined_head as RH
    if getattr(RH, "_q27_ds", False):
        return
    register_op()
    orig = RH.prepare_mtp_refined_head

    def prepare_mtp_refined_head(head, vllm_config):
        from vllm.model_executor.models.qwen3_5_mtp_sm70_head import _mtp_head_shard_rows
        w = getattr(head, "weight", None)
        sc = vllm_config.speculative_config
        from vllm import envs
        from vllm.platforms import current_platform
        # 1Cat's own conditions (prepare_mtp_refined_head) plus ours: a quantised (non-FP16) head of width 5120
        ok = (isinstance(w, torch.Tensor) and w.is_cuda and getattr(head, "shard_indices", None) is not None
              and current_platform.is_device_capability(70) and not envs.VLLM_BATCH_INVARIANT
              and vllm_config.lora_config is None
              and w.dtype != torch.float16 and vllm_config.model_config.dtype == torch.float16
              and sc is not None and sc.method == "mtp" and sc.draft_sample_method != "probabilistic"
              and getattr(head, "embedding_dim", 0) == HD and _mtp_head_shard_rows(head) is not None)
        if not ok:
            log.warning("q27fuse draftscreen: head not eligible; 1Cat's refined head unchanged")
            return orig(head, vllm_config)
        fallback = orig(head, vllm_config) if MAXR < 40 else None
        data = prepare(head, vllm_config, RERANK)
        h = ScreenHead(head, fallback, data)
        log.warning("q27fuse draftscreen: draft head = 1-bit screen (%d CTAs, %d warps, NC %d, KEEP %d, popcount up "
                    "to %d rows, rerank %s, rows 1..%d%s)", h.G, WARPS, NC, KEEP, PCMAX, data["rerank"], MAXR,
                    "; 1Cat's refined head above" if fallback is not None else "; 1Cat's refined head not built")
        return h

    RH.prepare_mtp_refined_head = prepare_mtp_refined_head
    RH._q27_ds = True
