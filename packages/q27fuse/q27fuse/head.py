"""q27fuse.head (track head) - ONE FP8 copy of the target LM head per rank, served by csrc/fp8_head.cu.

HeadCopy = the head shard in the QPN8 layout (built from the checkpoint rows by 1Cat's fp8_qpn8_prepare_sm70, the same
code layout our FP8 projections read) + fp16 scales (scale * 256) + f32 scales. Its kernels: logits (fp16 [rows, N],
1..64 rows per launch, more rows in 64-row chunks) and top-1 (per row the (max, global index) pair torch's
logits.max(-1) gives on those fp16 logits, written straight to a device address - e.g. the registered all-reduce
staging buffer that 1Cat's cross-device top-1 reads - without storing logits)."""
import logging
import torch

log = logging.getLogger("q27fuse")

_E = None


def ext():
    global _E
    if _E is None:
        from .ops import _load
        _E = _load("q27fuse_head", "fp8_head.cu", "_head")
    return _E


# fp8_head.cu configuration per row bucket and epilogue: [(max rows, name)] (test_head.py bench, one V100, cold
# weights; notes/head.md "Job 1 bench")
HEAD_CFG = [(8, "h 1x1 wk8 col1"), (16, "h 2x1 wk8 col2"), (32, "h 4x1 wk4 col4"), (48, "h 6x1 wk4 col2"),
            (64, "h 8x1 wk2 col4 pr0")]                                               # logits
HEAD_CFG_TOP1 = [(8, "h 1x1 wk8 col1"), (16, "h 2x1 wk8 col2"), (32, "h 4x1 wk2 col4 pr0"), (48, "h 6x1 wk4 col2"),
                 (64, "h 8x1 wk2 col4 pr0")]                                          # top-1 epilogue
CHUNK = 64
WS = 1          # per-weight fp16(code x scale) = TurboMind's weights
_IDX = {}


def cfg_for(m: int, top1: bool = False) -> int:
    if not _IDX:
        _IDX.update({c.split(":", 1)[1]: i for i, c in enumerate(ext().head_configs())})
    return _IDX[next(n for r, n in (HEAD_CFG_TOP1 if top1 else HEAD_CFG) if m <= r)]


class HeadCopy:
    def __init__(self, codes: torch.Tensor, s16: torch.Tensor, s32: torch.Tensor, vstart: int, nvalid: int):
        self.codes, self.s16, self.s32 = codes, s16, s32
        self.N = s16.numel()
        self.K = codes.numel() // self.N
        self.vstart, self.nvalid = int(vstart), int(nvalid)
        dev = codes.device
        self.keys = torch.zeros(32 * CHUNK, dtype=torch.int32, device=dev)  # [32 slots][64 rows], zero at rest (kernel resets)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)

    @classmethod
    def from_checkpoint(cls, w8: torch.Tensor, scale: torch.Tensor, vstart: int, nvalid: int) -> "HeadCopy":
        """w8 [N, K] float8_e4m3fn (or its uint8 bits), scale [N] / [N, 1] float (one per row)"""
        import vllm._C  # noqa: F401
        if w8.dtype == torch.uint8:
            w8 = w8.view(torch.float8_e4m3fn)
        s = scale.float().reshape(-1, 1).contiguous()
        codes, s16 = torch.ops._C.fp8_qpn8_prepare_sm70(w8.contiguous(), s)[:2]
        s16 = s16.reshape(-1).contiguous()
        s32 = s.reshape(-1).contiguous()
        if not torch.equal(s16, (s32 * 256).half()):
            raise RuntimeError("q27fuse head: QPN8 fp16 scales != fp16(scale * 256)")
        return cls(codes.reshape(-1).contiguous(), s16, s32, vstart, nvalid)

    def clone(self) -> "HeadCopy":
        return HeadCopy(self.codes.clone(), self.s16.clone(), self.s32.clone(), self.vstart, self.nvalid)

    def nbytes(self) -> int:
        return self.codes.numel() + self.s16.numel() * 2 + self.s32.numel() * 4

    def logits(self, x: torch.Tensor, out: torch.Tensor | None = None, ws: int = WS, cfg: int | None = None):
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, self.N), dtype=torch.float16, device=x.device)
        if m == 0:
            return out
        if x.stride(-1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
            x = x.contiguous()
        e = ext()
        for b in range(0, m, CHUNK):
            xe = x[b:b + CHUNK]
            e.head_logits(out[b:b + CHUNK], xe, self.codes, self.s16, self.s32,
                          cfg if cfg is not None else cfg_for(xe.shape[0]), ws)
        return out

    def top1(self, x: torch.Tensor, out_ptr: int | None = None, ws: int = WS, cfg: int | None = None):
        """pairs (value f32, global index f32) per row; at out_ptr (device address, 8 B per row) if given, else in a
        new [rows, 2] f32 tensor (returned)"""
        m = x.shape[0]
        pairs = None
        if out_ptr is None:
            pairs = torch.empty((m, 2), dtype=torch.float32, device=x.device)
            out_ptr = pairs.data_ptr()
        if m == 0:
            return pairs
        if x.stride(-1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
            x = x.contiguous()
        e = ext()
        for b in range(0, m, CHUNK):
            xe = x[b:b + CHUNK]
            e.head_top1(out_ptr + 8 * b, xe, self.codes, self.s16, self.s32, self.keys, self.counter, self.vstart,
                        self.nvalid, cfg if cfg is not None else cfg_for(xe.shape[0], top1=True), ws)
        return pairs


# ------------------------------------------------------------------------------------------------- integration
def enabled() -> bool:
    import os
    return (os.environ.get("Q27_HEAD", "") or "1").strip() != "0"


_LOGGED = set()


def _log_once(key, msg, *args):
    if key not in _LOGGED:
        _LOGGED.add(key)
        log.warning(msg, *args)


def _eligible(scheme, layer) -> str | None:
    """why this channel-FP8 layer is not taken as the unified head (None = it is)"""
    prefix = getattr(layer, "prefix", "") or ""
    if not prefix.endswith("lm_head"):
        return "not an lm_head"
    if not getattr(scheme, "use_sm70_fp8_turbomind", False):
        return "not 1Cat's SM70 TurboMind FP8 route"
    w, s = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
    sh = getattr(layer, "shard_indices", None)
    if not (isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.float8_e4m3fn and w.dim() == 2
            and w.shape[1] % 32 == 0 and w.shape[0] % 64 == 0 and isinstance(s, torch.Tensor)
            and s.numel() == w.shape[0] and sh is not None):
        return f"weight {getattr(w, 'dtype', None)} {tuple(getattr(w, 'shape', ()))} / scale / shard indices"
    if getattr(layer, "orig_dtype", torch.float16) != torch.float16:
        return "not an fp16 model"
    try:    # 1Cat's DraftVocab (speculative draft_token_map) index_selects lm_head.weight rows: needs the row-major copy
        from vllm.config import get_current_vllm_config
        sc = get_current_vllm_config().speculative_config
        if sc is not None and getattr(sc, "draft_token_map", None) is not None:
            return "speculative draft_token_map set (DraftVocab reads row-major lm_head.weight rows)"
    except Exception:   # noqa: BLE001 - no config in scope (tests): nothing to check
        pass
    return None


def apply():
    """the lm_head's weight processing builds the HeadCopy instead of TurboMind's packed copy (load time, before vLLM
    sizes the KV cache); its apply_weights (every full-logits consumer: compute_logits, the top-k / probe paths, the
    drafter's full-head fallback, 1Cat's _hot_rows) runs fp8_head.cu's logits kernel; LogitsProcessor.get_top_tokens
    (greedy verify / decode) runs its top-1 epilogue into the registered staging buffer of 1Cat's cross-device top-1"""
    if not enabled():
        return
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a16_fp8 as S
    cls = S.CompressedTensorsW8A16Fp8
    if getattr(cls, "_q27_head_patched", False):
        return
    ext()
    prev_pw, prev_aw = cls.process_weights_after_loading, cls.apply_weights

    def process_weights_after_loading(self, layer):
        why = _eligible(self, layer)
        if why is not None:
            if (getattr(layer, "prefix", "") or "").endswith("lm_head"):
                log.warning("q27fuse head: %s keeps 1Cat's head (%s)", layer.prefix, why)
            return prev_pw(self, layer)
        sh = layer.shard_indices
        w, s = layer.weight.data, layer.weight_scale.data
        N, K = w.shape
        hc = HeadCopy.from_checkpoint(w, s, vstart=int(sh.org_vocab_start_index),
                                      nvalid=N - int(sh.num_org_vocab_padding))
        from torch.nn import Parameter
        del layer._parameters["weight"]
        layer.weight = Parameter(hc.codes.view(N, K), requires_grad=False)     # the QPN8 codes = the only copy
        del layer._parameters["weight_scale"]
        layer.input_scale = None
        layer.q27_head = hc
        del w, s
        torch.cuda.empty_cache()
        log.warning("q27fuse head: %s = one FP8 copy in the QPN8 layout (%d x %d, %.1f MB; vocab %d.., %d valid), "
                    "fp8_head.cu logits + top-1; TurboMind's copy not built", layer.prefix, N, K, hc.nbytes() / 1e6,
                    hc.vstart, hc.nvalid)

    def apply_weights(self, layer, x, bias=None, output=None):
        hc = getattr(layer, "q27_head", None)
        if hc is None:
            return prev_aw(self, layer, x, bias, output)
        if x.dtype != torch.float16:
            raise RuntimeError(f"q27fuse head: fp16 activations required, got {x.dtype}")
        x2 = x.reshape(-1, x.shape[-1])
        out2 = output.reshape(-1, hc.N) if output is not None else None
        out2 = hc.logits(x2, out=out2)
        if bias is not None:
            out2.add_(bias)
        _log_once(("logits", x2.shape[0] > CHUNK), "q27fuse head: logits on fp8_head.cu (first call %d rows)", x2.shape[0])
        return out2.reshape(*x.shape[:-1], hc.N)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    cls._q27_head_patched = True
    apply_top1()


def apply_top1():
    from vllm.model_executor.layers import logits_processor as LP
    from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
    from . import tail
    P = LP.LogitsProcessor
    if getattr(P, "_q27_head_top1", False):
        return
    prev = P.get_top_tokens

    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        """the local (max, index) pair from the head kernel's top-1 epilogue (no logits); then 1Cat's cross-device
        top-1 exactly as tail's argmax / regtop1 pieces call it. Anything else: the previous get_top_tokens."""
        hc = getattr(lm_head, "q27_head", None)
        rows = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
        tp = get_tensor_model_parallel_world_size()
        if (hc is None or tp == 1 or embedding_bias is not None or self.soft_cap is not None or self.scale != 1.0
                or not 1 <= rows <= 128 or hidden_states.dtype != torch.float16 or not hidden_states.is_cuda
                or getattr(lm_head, "maybe_get_sm70_lm_head_top1_pair", None) is not None
                or torch.cuda.is_current_stream_capturing()):
            return prev(self, lm_head, hidden_states, embedding_bias)
        ca = tail._regtop1_comm(self, rows) if "regtop1" in tail.ACTIVE else None
        if ca is not None:
            staging = ca.buffer_ptrs[ca.rank]
            hc.top1(hidden_states, out_ptr=staging)
            local_pair = tail.ext().from_ptr(staging, 2 * rows, hidden_states.new_empty(0, dtype=torch.float32)).view(rows, 2)
            LP._maybe_sync_top1_all_gather(self, local_pair)
            _log_once("top1r", "q27fuse head: greedy top-1 from fp8_head.cu into the registered staging buffer "
                      "(first call %d rows)", rows)
            if self._top1_exchange is None:
                try:
                    out = torch.empty((rows,), dtype=torch.int64, device=hidden_states.device)
                    ca._native.top1_argmax(ca._ptr, local_pair.reshape(-1), out, 0, 0)
                    self._maybe_dump_top_token_margin(lm_head, hidden_states, embedding_bias, out)
                    return out
                except Exception as e:
                    log.warning("q27fuse head: registered top-1 failed (%r); copy mode from now on", e)
                    tail.ACTIVE.discard("regtop1")
            local_pair = local_pair.clone()
        else:
            local_pair = hc.top1(hidden_states)
            _log_once("top1", "q27fuse head: greedy top-1 from fp8_head.cu (first call %d rows)", rows)
        LP._maybe_sync_top1_all_gather(self, local_pair)
        if self._top1_exchange is not None:
            exchanged = self._top1_exchange(local_pair)
            if exchanged is not None:
                return exchanged
        custom = self._maybe_custom_top1_argmax(local_pair)
        if custom is not None:
            self._maybe_dump_top_token_margin(lm_head, hidden_states, embedding_bias, custom)
            return custom
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1).view(rows, tp, 2)
        top = gathered[:, :, 1].gather(dim=-1, index=gathered[:, :, 0].argmax(dim=-1, keepdim=True)).squeeze(-1)
        top = top.to(torch.int64)
        self._maybe_dump_top_token_margin(lm_head, hidden_states, embedding_bias, top)
        return top

    P.get_top_tokens = get_top_tokens
    P._q27_head_top1 = True
