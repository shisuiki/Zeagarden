"""q27fuse.pf2 - Unified exact prefill all-reduce + Gemma norm (csrc/arn.cu) and FP8 prefill GEMMs (csrc/pf_gemm.cu).

1. Q27_ARN (default ON, csrc/arn.cu):
   Replaces both legacy pf_arn.cu and short_arn.cu with ONE 45.0 MiB/rank NVLink IPC buffer, ONE device-side monotonic
   u64 epoch (100% CUDA-graph-safe), and ONE CudaCommunicator hook covering M = 65..4608 rows (M <= 64 stays on 1Cat's
   37 us push-norm):
     - M = 65..1024 (and bare all_reduce): complete FP32 residual, CUDA-graph-safe (1.2x-2.5x vs 1Cat v1.5.4).
     - M = 1025..4608: row-sharded FP32 residual (1.9x-2.2x vs 1Cat v1.5.4), plus optional H_32/FP4 fused quant.
2. Q27_PF_FP8 (default ON, csrc/pf_gemm.cu):
   cuBLAS algo 6 FP8 prefill GEMMs for large prefill projections (M >= 257).
"""
import inspect
import logging
import os
import torch

log = logging.getLogger("q27fuse")


def _on(knob: str, default: str = "1") -> bool:
    return (os.environ.get(knob, "") or default).strip() != "0"


def _int(knob: str, default: int) -> int:
    v = os.environ.get(knob, "").strip()
    return int(v) if v else default


# -------------------------------------------------------------------------------- 1. Unified ARN + AR (csrc/arn.cu)
ARN_MIN = _int("Q27_ARN_MIN", 65)
ARN_SHARD_MIN = _int("Q27_ARN_SHARD_MIN", 1025)
AR_MIN = _int("Q27_AR_MIN", 65)
AR_MAX = _int("Q27_AR_MAX", 1024)

# Hardware-tuned config indices into kCfgs[0..8] in csrc/arn.cu (all <= 80 CTAs co-resident on 80-SM V100)
_SHORT_ARN_CFG = [(256, 3), (1024, 5)]
_SHORT_AR_CFG = [(128, 3), (192, 5), (256, 3), (1024, 5)]
_LONG_ARN_CFG = [(1535, 6), (3071, 7), (4608, 8)]

_AX = None
# Norms whose boundary must return the COMPLETE residual (e.g., last layer's post-attn norm on target-only route).
FULL_RESIDUAL = []
_FR = {}


def _pick(table, m: int) -> int:
    for hi, v in table:
        if m <= hi:
            return v
    return table[-1][1]


def _aligned16(t: torch.Tensor) -> bool:
    return (t.data_ptr() & 15) == 0


def _gather_residual(comm, res):
    """Gather the complete FP32 residual from the four row quarters left on their owners."""
    M, H = res.shape
    MQ = (M + 3) // 4
    r = comm.rank_in_group
    lo, hi = r * MQ, min(M, (r + 1) * MQ)
    part = res.new_zeros((MQ, H))
    if hi > lo:
        part[:hi - lo] = res[lo:hi]
    full = comm.all_gather(part, dim=0)
    if not _FR.get("logged") and r == 0:
        _FR["logged"] = True
        log.warning("q27fuse arn: complete residual gathered after final sharded boundary (%d rows)", M)
    return full[:M].contiguous()


def _full_residual(weight) -> bool:
    if not FULL_RESIDUAL:
        return False
    if _FR.get("n") != len(FULL_RESIDUAL):
        _FR["ptrs"] = {m.weight.data_ptr() for m in FULL_RESIDUAL}
        _FR["n"] = len(FULL_RESIDUAL)
    return weight.data_ptr() in _FR["ptrs"]


def arn_ext():
    global _AX
    if _AX is None:
        from torch.utils.cpp_extension import load
        from .ops import default_build_dir
        here = os.path.dirname(os.path.abspath(__file__))
        csrc = os.environ.get("Q27_CSRC_DIR", os.path.join(here, "csrc"))
        build = default_build_dir("_arn")
        os.makedirs(build, exist_ok=True)
        flags = ["-O3", "-gencode=arch=compute_70,code=sm_70", "-std=c++17"]
        _AX = load(name="q27fuse_arn", sources=[os.path.join(csrc, "arn.cu")],
                   build_directory=build, extra_cuda_cflags=flags, verbose=False)
    return _AX


def _arn_setup(comm) -> None:
    import torch.distributed as dist
    X = arn_ext()
    with torch.cuda.device(comm.device):
        mine = X.alloc_buf()
        hs = [None] * comm.world_size
        dist.all_gather_object(hs, X.ipc_handle(mine), group=comm.cpu_group)
        r = comm.rank_in_group
        X.init([mine if q == r else X.ipc_open(hs[q]) for q in range(4)], r)
    comm._q27_arn = {"maxm": int(X.max_rows()), "calls": 0, "ar_calls": 0, "sharded": False}
    if r == 0:
        log.warning(
            "q27fuse arn: unified arn.cu active (ARN %d..%d [full<=%d, shard>%d], AR %d..%d, %.1f MiB/rank, graph-safe)",
            ARN_MIN, X.max_rows(), ARN_SHARD_MIN - 1, ARN_SHARD_MIN - 1,
            AR_MIN, min(AR_MAX, int(X.max_rows())), X.buf_bytes() / (1 << 20),
        )


def _fuse_qmax(weight, m: int):
    from . import ops
    if not ops.PF_PK:
        return 0.0, None, None
    from . import pf3
    return pf3.fuse_qmax(weight, m)


def apply_arn() -> None:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator as C
    if getattr(C, "_q27_arn_patched", False):
        return
    orig_init = C.__init__
    orig_arn = C.all_reduce_gemma_rms_norm
    orig_ar = C.all_reduce
    orig_has_shard = "allow_shard_residual" in inspect.signature(orig_arn).parameters

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._q27_arn = None
        if self.world_size == 4 and "tp" in (self.unique_name or "") and self.device.type == "cuda":
            _arn_setup(self)

    def all_reduce_gemma_rms_norm(self, input_, residual, weight, epsilon, allow_shard_residual=False):
        st = getattr(self, "_q27_arn", None)
        if (st is not None and input_.dim() == 2 and ARN_MIN <= input_.shape[0] <= st["maxm"]
                and input_.shape[1] == 5120 and input_.dtype == torch.float16
                and residual.dtype == torch.float32 and weight.dtype == torch.float16
                and residual.shape == input_.shape and input_.is_contiguous()
                and residual.is_contiguous() and weight.is_contiguous()
                and _aligned16(input_) and _aligned16(residual) and _aligned16(weight)):
            m = int(input_.shape[0])
            capturing = torch.cuda.is_current_stream_capturing()
            st["calls"] += 1
            if st["calls"] == 1 and self.rank_in_group == 0:
                log.warning("q27fuse arn: first unified ARN call (%d rows, capturing=%s)", m, capturing)

            # Opt-in fused H_32 quantize path (Q27_PF_PK=1, eager long prefill only)
            qm = 0.0
            u = R = None
            if not capturing:
                qm, u, R = _fuse_qmax(weight, m)

            if qm:
                cfg = _pick(_LONG_ARN_CFG, m) if m > 1024 else _pick(_SHORT_ARN_CFG, m)
                out, res, xq, sx = arn_ext().arn(input_, residual, weight, float(epsilon), cfg, True, qm, u, R)
                from . import pf3
                pf3.stash(out, xq, sx, qm, R is not None, u)
                need_full = _full_residual(weight) or (orig_has_shard and not allow_shard_residual)
                if need_full:
                    res = _gather_residual(self, res)
                    st["sharded"] = False
                else:
                    st["sharded"] = True
                return out, res

            from . import ops
            if ops.PF_PK:
                from . import pf3
                pf3.clear_stash()

            # On 1Cat-vLLM v1.5.4, delegate eager M=128..1024 sharded-residual prefills to 1Cat's native
            # 0-VRAM tiled RS+Norm+AG (eliminates dynamic S2a/S2b TTFT/ITL spin-barrier overhead while
            # keeping arn.cu for PIECEWISE graphs / M<128 and for M>1024 long prefills + H_32 fusion).
            if (orig_has_shard and not capturing and not st["sharded"] and 128 <= m < ARN_SHARD_MIN
                    and (allow_shard_residual or st.get("cat_sharded"))):
                st["cat_sharded"] = bool(allow_shard_residual)
                return orig_arn(self, input_, residual, weight, epsilon, allow_shard_residual)
            st["cat_sharded"] = False

            # Short/medium regime (65..1024) or capturing: always full residual (CUDA-graph-safe)
            if m < ARN_SHARD_MIN or capturing:
                if not st["sharded"] and m <= 1024:
                    cfg = _pick(_SHORT_ARN_CFG, m)
                    out, res = arn_ext().arn(input_, residual, weight, float(epsilon), cfg, False)
                    return out, res

            # Long regime (1025..4608): row-sharded residual when allowed or completing a sharded chain
            cfg = _pick(_LONG_ARN_CFG, m) if m > 1024 else _pick(_SHORT_ARN_CFG, m)
            can_shard = (not orig_has_shard) or bool(allow_shard_residual)
            if can_shard:
                out, res = arn_ext().arn(input_, residual, weight, float(epsilon), cfg, True)
                if _full_residual(weight):
                    res = _gather_residual(self, res)
                    st["sharded"] = False
                else:
                    st["sharded"] = True
                return out, res
            if st["sharded"]:
                out, res = arn_ext().arn(input_, residual, weight, float(epsilon), cfg, True)
                res = _gather_residual(self, res)
                st["sharded"] = False
                return out, res
            # Unsharded caller with allow_shard_residual=False: complete residual directly, zero gather
            out, res = arn_ext().arn(input_, residual, weight, float(epsilon), cfg, False)
            return out, res

        if orig_has_shard:
            return orig_arn(self, input_, residual, weight, epsilon, allow_shard_residual)
        return orig_arn(self, input_, residual, weight, epsilon)

    def all_reduce(self, input_):
        st = getattr(self, "_q27_arn", None)
        if (st is not None and input_.dim() == 2 and AR_MIN <= input_.shape[0] <= min(AR_MAX, st["maxm"])
                and input_.shape[1] == 5120 and input_.dtype == torch.float16
                and input_.is_contiguous() and _aligned16(input_)):
            st["ar_calls"] += 1
            if st["ar_calls"] == 1 and self.rank_in_group == 0:
                log.warning("q27fuse arn: first unified bare AR (%d rows, capturing=%s)",
                            input_.shape[0], torch.cuda.is_current_stream_capturing())
            cfg = _pick(_SHORT_AR_CFG, int(input_.shape[0]))
            return arn_ext().ar(input_, cfg)
        return orig_ar(self, input_)

    C.__init__ = __init__
    C.all_reduce_gemma_rms_norm = all_reduce_gemma_rms_norm
    C.sm70_tp4_all_reduce_gemma_rms_norm = all_reduce_gemma_rms_norm
    C.all_reduce = all_reduce
    C._q27_arn_patched = True


# --------------------------------------------------------------------------- 2. FP8 projections at prefill rows (pf_gemm.cu)
# Microbench-verified cutovers against 1Cat-vLLM v1.5.4 + fp8_short.cu on V100:
#   - (5120, 4096): fp8_short <=448, algo 6 at 512..768 & >=3072, 1Cat v1.5.4 at 769..3071
#   - (5120, 3584): fp8_short <=512, algo 6 at 513..768 & >=1536, 1Cat v1.5.4 at 769..1535
#   - (4352, 5120): fp8_short <=512, 1Cat v1.5.4 at 513..767, algo 6 at >=768
#   - (1536, 5120): fp8_short <=640, algo 6 at 641..1536
FP8_MIN = int(os.environ.get("Q27_PF_FP8_MIN", "512") or 512)
FP8_ALGO = {
    (5120, 4096): [(512, 769, 0, 6), (3072, 1 << 30, 0, 6)],
    (5120, 3584): [(513, 769, 0, 6), (1536, 1 << 30, 0, 6)],
    (4352, 5120): [(768, 1 << 30, 0, 6)],
    (1536, 5120): [(641, 1536, 0, 6)],
}
_GX = None


def gemm_ext():
    global _GX
    if _GX is None:
        from .ops import _load
        _GX = _load("q27fuse_pfgemm", "pf_gemm.cu", "_pfgemm", ldflags=["-lcublas"])
    return _GX


def apply_fp8() -> None:
    import vllm.model_executor.kernels.linear.qpn.fp8 as F
    if getattr(F, "_q27_pf2_patched", False):
        return
    from vllm import _sm70_ops as S
    orig = F.dispatch_channel_rows
    seen = set()

    def dispatch_channel_rows(out, x, codes, scales, split_k, accumulator_chains, prefetch_codes, gated_silu,
                              native_policy=None):
        m = x.shape[0]
        if m >= FP8_MIN and not gated_silu and codes.dim() == 2 and x.dtype == torch.float16 and out.is_contiguous():
            K, N = int(codes.shape[0]), int(codes.shape[1])
            cfg = next(((lay, al) for lo, hi, lay, al in FP8_ALGO.get((K, N), ()) if lo <= m < hi), None)
            if cfg is not None and x.shape[1] == K and out.shape[-1] == N:
                ws = F._get_sm70_fp8_prefill_exact_dense_workspace(codes)
                w = ws[:K * N].view(K, N)
                S.fp8_qpn8_dequantize_sm70_out(w, codes, scales)
                st = gemm_ext().gemm(w, x if x.is_contiguous() else x.contiguous(), out, cfg[0], cfg[1])
                if st == 0:
                    if (K, N) not in seen:
                        seen.add((K, N))
                        log.warning("q27fuse prefill2: FP8 projection %dx%d at %d rows on cublasGemmEx (layout %d, algo %d)",
                                    K, N, m, cfg[0], cfg[1])
                    return
        return orig(out, x, codes, scales, split_k, accumulator_chains, prefetch_codes, gated_silu, native_policy)

    F.dispatch_channel_rows = dispatch_channel_rows
    F._q27_pf2_patched = True


def apply() -> None:
    # Single Q27_ARN switch (default ON); also honors legacy Q27_PF_ARN=0 if Q27_ARN is unset
    arn_on = _on("Q27_ARN", "1") if os.environ.get("Q27_ARN", "").strip() else _on("Q27_PF_ARN", "1")
    if arn_on:
        apply_arn()
    if _on("Q27_PF_FP8") and FP8_ALGO:
        apply_fp8()


def exts():
    """(name, loader) of every extension of this module, for the build script"""
    return [("arn", arn_ext), ("pf_gemm", gemm_ext)]
