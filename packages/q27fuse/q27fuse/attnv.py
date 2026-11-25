"""q27fuse.attnv - track attnv: every decode-time call of 1Cat's E4M3 long-context attention operator
(_vllm_fa2_C.sm70_grouped_long_fwd = grouped_e4m3_2cta_partial + parallel_combine) runs csrc/attn_mq.cu.

1Cat reaches that operator through two module-global names, each looked up at call time:
- `sm70_grouped_long.run_six_head_groups`, called by `wrap_long_attention.run` (the E4M3 grouped verifier route of
  `VerificationExecutor.call_smallq_decode_paged`): MTP verify rows and the drafter's first pass, q [B Q, 6, 256] with
  Q = K + 1 rows per request in 2..8, per-request block table [B, W], per-row lengths [B Q]. 1Cat's wrapper has already
  applied its admission test (B = 1 with 2..8 rows, or B requests with equal rows in 2..8, page, context bound) when it
  calls this name, so whatever reaches it is a call 1Cat would serve with its own kernels; everything the wrapper
  rejects keeps 1Cat's fallback (flash_attn_v100 kernels).
- `sm70_e4m3_decode.run_six_head_groups` (its own imported binding), called by `wrap_e4m3_decode_attention.run`:
  single-query decode (target-only decode, the drafter's passes 2..K, draft-free steps), q [B, 6, 256], B >= 1, after
  1Cat's admission. patch.apply_attn already sends B >= 2 there to csrc/attn_dec.cu; this module takes B >= 2 over
  (B = 1 stays on 1Cat, see MQ_CFG).
Shapes our kernel does not take (head dim, dtypes, layouts) fall through to 1Cat's operator.
Configuration per (Q, B): MQ_CFG below, from `test_attn.py mqbench` on one GPU (notes/attnv.md)."""
import logging
import os
import torch

log = logging.getLogger("q27fuse")

_MQ = None
_WS = {}
_SEEN = set()


def mq_ext():
    """attn_mq.cu: split-KV E4M3 decode attention, Q query rows per request, in-kernel combine."""
    global _MQ
    if _MQ is None:
        from .ops import _load
        _MQ = _load("q27fuse_attnmq", "attn_mq.cu", "_attnmq")
    return _MQ


# (warps per set nw, splits per request S, min tiles per set, sets per CTA ng, tiles in flight in registers per set pd,
# further tiles per set prefetched into L2 pf, row split rs, separate combine kernel sep) by query rows per request Q
# and batch B:
# MQ_CFG[Q] = [(max B, cfg or None = keep 1Cat's operator), ...]
MQ_CFG = {
    # (test_attn.py mqbench, one GPU, same process; results/r1/attnv/jh_sep.log) every entry combines in a second
    # kernel (sep): no split waits on another, outputs bitwise reproducible, and as fast or faster than the in-kernel
    # split barrier.
    # B = 1 (C = 1, every Q) keeps 1Cat's operator: in the server our B = 1 kernels only tie 1Cat (nsys, 8K C1: verify
    # k_mq 31.6 + k_comb 13.9 us vs 1Cat 39.7 + 7.2 per layer; the separate combine costs 11-14 us in the server against
    # ~5 in the bench) and the unprofiled step was 0.3-0.4% slower in three alternated pairs (srv2 / srv3), while srv4
    # (this routing) equals the base tree at C = 1 (8K 12.367 vs 12.350 ms, 1K 12.129 vs 12.128, greedy outputs 16/16
    # identical; notes/attnv.md "C = 1 attribution").
    # single query (drafter passes 2..K, draft-free steps, target-only decode), 2 sets of 4 warps: B 2..15 64 splits as
    # in the server A/B (srv2 / srv4; in the MTP mapping single-query calls are drafter passes at B <= 15 only, K = 1
    # from C 16), B >= 16 32 splits (target-only decode; results/r1/attnv/jm_singlequery.log, pages 1568 and 2048
    # alike: 8K B 32 / 64 170-172 / 329-330 us vs 64 splits 200-202 / 386-388, 1Cat 237-238 / 440-442, attn_dec.cu
    # 175.0 / 337.0; 1K B 64 54.4 vs attn_dec.cu 51.0, the only point attn_dec.cu is ahead). Not server-tested from B 16.
    1: [(1, None), (15, (4, 64, 1, 2, 1, 0, 0, 1)), (10 ** 9, (4, 32, 1, 2, 1, 0, 0, 1))],
    # MTP verify / drafter first pass, K = 1 (C 16..32 under 4@1,3@2,1@16): 8K B 16 / 32 100.4 / 182.9 vs 141.2 / 308.0
    2: [(1, None), (10 ** 9, (4, 64, 1, 1, 1, 0, 0, 1))],
    3: [(1, None), (10 ** 9, (4, 128, 1, 2, 1, 0, 0, 1))],
    # K = 3 (C 2..15): 8K B 2 / 4 / 8 31.1 / 43.9 / 68.2 vs 38.6 / 72.1 / 126.1
    4: [(1, None), (10 ** 9, (4, 128, 1, 2, 1, 0, 0, 1))],
    # K = 4 (C 1): B 1 keeps 1Cat (above); one GPU 8K B 1 25.4 vs 33.8
    5: [(1, None), (10 ** 9, (4, 128, 1, 2, 1, 0, 0, 1))],
    6: [(1, None), (10 ** 9, (8, 64, 1, 1, 1, 0, 0, 1))],
    7: [(1, None), (10 ** 9, (8, 64, 1, 1, 1, 0, 0, 1))],
    8: [(1, None), (10 ** 9, (8, 64, 1, 1, 1, 0, 0, 1))],
}
MAX_B = 256


def mq_cfg(q_rows: int, b: int):
    for m, c in MQ_CFG[q_rows]:
        if b <= m:
            return c
    return None


def workspace(dev, hkv: int, tag=None):
    """Per (device, KV heads): partial numerators + row stats sized for the largest launch any (Q, B <= MAX_B)
    configuration can make (one resident wave), and per-(request, head) counters (zeroed once, left zeroed)."""
    key = (dev.type, dev.index, hkv)
    if key not in _WS:
        if torch.cuda.is_current_stream_capturing():
            # Never allocate a persistent workspace inside CUDA-graph capture (graph memory pool aliasing).
            if ("cap",) + key not in _SEEN:
                _SEEN.add(("cap",) + key)
                log.error("q27fuse attnv: workspace %s first needed inside CUDA graph capture; falling back", key)
            return None
        ext = mq_ext()
        n_po = n_pml = 1
        for qr, cfgs in MQ_CFG.items():
            for _, c in cfgs:
                if c is None:
                    continue
                nw, S, _mt, ng, pd, _pf, rs, _sep = c
                for b in range(1, MAX_B + 1):
                    a, c2 = ext.mq_workspace(b * hkv, S, nw, qr, ng, pd, rs)
                    n_po, n_pml = max(n_po, a), max(n_pml, c2)
        _WS[key] = (torch.empty(n_po, dtype=torch.float32, device=dev),
                    torch.empty(n_pml, dtype=torch.float32, device=dev),
                    torch.zeros(max(8192, MAX_B * hkv), dtype=torch.int32, device=dev))
        log.info("q27fuse: attn_mq workspace %.1f MB on %s", (n_po + n_pml) * 4 / 2 ** 20, dev)
    return _WS[key]


PREALLOC = (os.environ.get("Q27_ATTNV_PREALLOC", "") or "1").strip() != "0"     # [head] Job 0 fix


def _kv_heads_per_rank(cfg=None):
    if cfg is None:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
    tp = cfg.parallel_config.tensor_parallel_size
    return max(1, cfg.model_config.get_total_num_kv_heads() // tp)


def install_prealloc():
    """[head] Job 0 root-cause fix: the workspace must not be allocated inside a CUDA-graph capture. 1Cat's grouped
    long-attention wrapper admits a call only with the captured batch descriptor's context bucket, so the warm-up runs
    never reach run_mq and its first call (which used to allocate the workspace lazily) happened INSIDE the first FULL
    capture: po / pml / cnt then came out of the CUDA-graph private memory pool (and the counter zero-fill was a captured
    kernel). Allocate it on the capture's device before every capture instead (no-op once it exists)."""
    from vllm.v1.worker.gpu import cudagraph_utils as CU
    M = CU.CudaGraphManager
    if getattr(M, "_q27_attnv_prealloc", False):
        return
    prev = M.capture

    def capture(self, *a, **kw):
        try:
            dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
            if dev.type == "cuda":
                if dev.index is None:
                    dev = torch.device("cuda", torch.cuda.current_device())
                workspace(dev, _kv_heads_per_rank(getattr(self, "vllm_config", None)))
        except Exception as e:  # noqa: BLE001 - never block capture; the lazy path stays as the fallback
            log.warning("q27fuse attnv: workspace pre-allocation failed (%r)", e)
        return prev(self, *a, **kw)

    M.capture = capture
    M._q27_attnv_prealloc = True


def _shapes_ok(q, k, v, out, table, lens) -> bool:
    return (q.dtype == torch.float16 and q.is_contiguous() and q.dim() == 3 and q.shape[2] == 256
            and k.dim() == 4 and k.dtype == torch.uint8 and v.dtype == torch.uint8 and k.shape == v.shape
            and k.shape[3] == 256 and k.stride(3) == 1 and v.stride(3) == 1 and q.shape[1] == 6 * k.shape[2]
            and k.shape[1] % 32 == 0 and table.dtype == torch.int32 and table.dim() == 2 and table.stride(1) == 1
            and 1 <= table.shape[0] <= MAX_B and lens.dtype == torch.int32 and lens.dim() == 1
            and lens.is_contiguous() and lens.shape[0] == q.shape[0] and out.is_contiguous() and out.shape == q.shape
            and out.dtype == torch.float16 and q.shape[0] % table.shape[0] == 0 and 1 <= q.shape[0] // table.shape[0] <= 8)


def run_mq(q, k, v, out, table, lens, scale, k_scale, v_scale) -> bool:
    """attn_mq.cu for this call if a configuration exists; False -> caller keeps 1Cat's operator."""
    if not _shapes_ok(q, k, v, out, table, lens):
        return False
    B = table.shape[0]
    Q = q.shape[0] // B
    cfg = mq_cfg(Q, B)
    if cfg is None:
        return False
    nw, S, mt, ng, pd, pf, rs, sep = cfg
    ws = workspace(q.device, k.shape[2], Q)
    if ws is None:
        return False
    po, pml, cnt = ws
    mq_ext().attn_mq(q, k, v, table, lens, out, po, pml, cnt, float(scale), float(k_scale), float(v_scale), Q, nw, S,
                     mt, ng, pd, pf, rs, sep)
    if (Q, B) not in _SEEN and len(_SEEN) < 64:
        _SEEN.add((Q, B))
        log.warning("q27fuse: attention on attn_mq.cu (B=%d, Q=%d, page %d, kv heads %d, cfg %s)",
                    B, Q, k.shape[1], k.shape[2], cfg)
    return True


def apply():
    from vllm.v1.attention.ops import sm70_grouped_long as GL
    from vllm.v1.attention.ops import sm70_e4m3_decode as D
    mq_ext()
    if PREALLOC:
        install_prealloc()
    # [merge] Q27_ATTNV_ROUTES: which of the two hooks to install ("gl" = verify rows / drafter first pass through
    # sm70_grouped_long, "d" = single-query decode through sm70_e4m3_decode); default both
    routes = {r.strip() for r in (os.environ.get("Q27_ATTNV_ROUTES", "") or "gl,d").split(",")}
    if "gl" in routes and not getattr(GL, "_q27_attnv_patched", False):
        orig_gl = GL.run_six_head_groups

        def run_six_head_groups(operator, q, k, v, out, table, row_lengths, partial, lse, scale, k_scale, v_scale,
                                *rest, **kw):
            if not rest and run_mq(q, k, v, out, table, row_lengths, scale, k_scale, v_scale):
                return out
            return orig_gl(operator, q, k, v, out, table, row_lengths, partial, lse, scale, k_scale, v_scale,
                           *rest, **kw)

        GL.run_six_head_groups = run_six_head_groups
        GL._q27_attnv_patched = True
    if "d" in routes and not getattr(D, "_q27_attnv_patched", False):
        prev = D.run_six_head_groups          # patch.apply_attn's hook (B >= 2 -> attn_dec.cu), then 1Cat's

        def run_six_head_groups_dec(operator, q, k, v, out, table, seq_lens, partial, lse, scale, k_scale, v_scale,
                                    *rest, **kw):
            if (not rest and table.dim() == 2 and q.dim() == 3 and q.shape[0] == table.shape[0]
                    and mq_cfg(1, table.shape[0]) is not None
                    and run_mq(q, k, v, out, table, seq_lens, scale, k_scale, v_scale)):
                return out
            return prev(operator, q, k, v, out, table, seq_lens, partial, lse, scale, k_scale, v_scale, *rest, **kw)

        D.run_six_head_groups = run_six_head_groups_dec
        D._q27_attnv_patched = True
