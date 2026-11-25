"""q27fuse.pf2 - track prefill2: faster prefill (rows above the decode / CUDA-graph range) with the target model's maths
unchanged up to summation-order rounding. On by default; opt-out knobs per piece.

1. Q27_PF_ARN (all-reduce + norm boundary, csrc/pf_arn.cu). Above 64 rows 1Cat's fused all-reduce + Gemma RMSNorm op
   falls back to a pynccl all-reduce (572 us at 4,096 rows) plus a separate add + norm pass over all rows (312 us,
   252 MB of HBM traffic). For ARN_MIN..max_rows() rows (default from 257: smaller row counts can be captured in 1Cat's
   CUDA graphs, capture sizes up to 256) the boundary is one kernel: reduce-scatter by row quarters over NVLink-mapped
   buffers, the add + norm on the owner's quarter, all-gather of the normalised rows (bench: 535 us at 4,096 rows).
   The four fp16 partials are summed in FP32 in rank order and rounded to fp16 (NCCL's ring rounds every hop in fp16),
   then the FP32 residual add and the Gemma norm as before. The residual stays ROW-SHARDED between these calls: every
   rank updates (and later reads) only its own quarter; the residual is consumed by nothing but these boundaries (the
   final norm's residual output is discarded). Both communicator entry points are patched (all_reduce_gemma_rms_norm
   and its alias sm70_tp4_all_reduce_gemma_rms_norm), which serve 1Cat's MTP decoder forward and our target-only one.
"""
import logging
import os
import torch

log = logging.getLogger("q27fuse")


def _on(knob: str) -> bool:
    return (os.environ.get(knob, "") or "1").strip() != "0"


# ------------------------------------------------------------------------------------- 1. all-reduce + norm (pf_arn.cu)
ARN_MIN = int(os.environ.get("Q27_PF_ARN_MIN", "257") or 257)
# (min rows, (rows per block, pusher CTAs, reducer CTAs, puller CTAs)); test_pf_arn.py sweeps (us at 4,096 / 2,048 /
# 1,024 rows against 1Cat's 882 / 473 / 264): rb32 p12 r40 a12 535, rb24 p12 r48 a12 309, rb16 p12 r40 a12 193
ARN_CFG = [(3072, (32, 12, 40, 12)), (1536, (24, 12, 48, 12)), (768, (16, 12, 40, 12)), (0, (8, 12, 40, 12))]
_AX = None
# [prefill3] norms whose boundary must return the COMPLETE residual: the last layer's post-attention norm on the target-only
# route (patch.py), whose residual the model's final norm reads on every row (1Cat's MTP route fuses that final norm into
# the communicator op, which reads only the owned quarter). After those boundaries the four FP32 residual quarters are
# all-gathered (pynccl, ~0.6 ms at 4,096 rows, once per forward).
FULL_RESIDUAL = []
_FR = {}


def _gather_residual(comm, res):
    """[prefill3] the complete FP32 residual from the four row quarters pf_arn left on their owners"""
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
        log.warning("q27fuse prefill3: complete residual gathered after the last layer's prefill boundary (%d rows)", M)
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
        from .ops import _load
        _AX = _load("q27fuse_pfarn", "pf_arn.cu", "_pfarn")
    return _AX


def _arn_cfg(m: int):
    return next(c for lo, c in ARN_CFG if m >= lo)


def _arn_setup(comm) -> None:
    """cudaMalloc + exchange of the NVLink buffers of the TP4 group (collective over the group's CPU group; runs while
    the communicator is built, on every rank, before any memory profiling or graph capture)."""
    import torch.distributed as dist
    X = arn_ext()
    with torch.cuda.device(comm.device):
        mine = X.alloc_buf()
        hs = [None] * comm.world_size
        dist.all_gather_object(hs, X.ipc_handle(mine), group=comm.cpu_group)
        r = comm.rank_in_group
        X.init([mine if q == r else X.ipc_open(hs[q]) for q in range(4)], r)
    comm._q27_arn = {"epoch": 0, "maxm": int(X.max_rows()), "calls": 0}
    if r == 0:
        log.warning("q27fuse prefill2: fused prefill all-reduce + Gemma norm (pf_arn.cu) for %d..%d rows, %.1f MB of NVLink "
                    "buffers per rank", ARN_MIN, X.max_rows(), X.buf_bytes() / 2 ** 20)


def _fuse_qmax(weight, m: int):
    """[prefill3/4] Job 5: (qmax, u, R, is_h32) to quantise this norm's output inside pf_arn.cu (0 = no fused quantisation):
    only for the post-attention norm of an NVFP4 MLP layer whose gate+up runs on the packed kernel at this row count"""
    from . import ops
    if not ops.PF_PK:
        return 0.0, None, None, False
    from . import pf3
    return pf3.fuse_qmax(weight, m)


def apply_arn() -> None:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator as C
    if getattr(C, "_q27_arn_patched", False):
        return
    orig_init, orig = C.__init__, C.all_reduce_gemma_rms_norm

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._q27_arn = None
        if self.world_size == 4 and "tp" in (self.unique_name or "") and self.device.type == "cuda":
            _arn_setup(self)

    def all_reduce_gemma_rms_norm(self, input_, residual, weight, epsilon):
        st = getattr(self, "_q27_arn", None)
        if (st is not None and input_.dim() == 2 and ARN_MIN <= input_.shape[0] <= st["maxm"] and input_.shape[1] == 5120
                and input_.dtype == torch.float16 and residual.dtype == torch.float32 and weight.dtype == torch.float16
                and residual.shape == input_.shape and input_.is_contiguous() and residual.is_contiguous()
                and weight.is_contiguous() and not torch.cuda.is_current_stream_capturing()):
            st["epoch"] += 1
            st["calls"] += 1
            if st["calls"] == 1 and self.rank_in_group == 0:
                log.warning("q27fuse prefill2: first fused prefill all-reduce + norm (%d rows)", input_.shape[0])
            rb, gp, gr, ga = _arn_cfg(input_.shape[0])
            qm, u, R, is_h32 = _fuse_qmax(weight, input_.shape[0])  # [prefill3/4] Job 5: + packed MLP input (FP4-331 or H_32)
            if qm:
                out, res, xq, sx = arn_ext().arn(input_, residual, weight, float(epsilon), st["epoch"], rb, gp, gr, ga,
                                                 qm, u, R, is_h32)
                from . import pf3
                pf3.stash(out, xq, sx, qm, R is not None, is_h32, u)
                if _full_residual(weight):                          # [prefill3]
                    res = _gather_residual(self, res)
                return out, res
            from . import ops
            if ops.PF_PK:
                from . import pf3
                pf3.clear_stash()
            out, res = arn_ext().arn(input_, residual, weight, float(epsilon), st["epoch"], rb, gp, gr, ga)
            if _full_residual(weight):                              # [prefill3] the final norm reads every row
                res = _gather_residual(self, res)
            return out, res
        return orig(self, input_, residual, weight, epsilon)

    C.__init__ = __init__
    C.all_reduce_gemma_rms_norm = all_reduce_gemma_rms_norm
    C.sm70_tp4_all_reduce_gemma_rms_norm = all_reduce_gemma_rms_norm
    C._q27_arn_patched = True


# --------------------------------------------------------------------------- 2. FP8 projections at prefill rows (pf_gemm.cu)
# 1Cat's dense prefill route for the channel-FP8 projections (QPN8 codes) = its exact dequantisation into a shared fp16
# workspace W[K][N] = fp16(E4M3 * s16 / 256) + a cuBLAS GEMM whose heuristic picks volta_fp16_s884gemm_fp16_64x128 "nn"
# for the GDN qkvz / attention qkv projections (74 TF in the server). Same dequantisation, then cublasGemmEx with the
# algorithm measured best per (K, N) (bench_pfgemm.py); identical fp16 operands, FP32 accumulation: summation order only.
FP8_MIN = int(os.environ.get("Q27_PF_FP8_MIN", "257") or 257)
# {(K, N): [(min rows, max rows, layout, algo), ...]}; layout 0 = W[K][N] as dequantised ("nn"); algo -1 =
# CUBLAS_GEMM_DEFAULT_TENSOR_OP (= 1Cat's call, bitwise), i = CUBLAS_GEMM_ALGO<i>_TENSOR_OP. bench_pfgemm.py (one GPU,
# us at 4,096 / 2,048 / 1,024 rows, default -> chosen): GDN qkvz 5120x4096 2,080 -> 2,012 (algo 6 only from 3,072 rows:
# it loses at 2,048), attention qkv 5120x3584 1,778 -> 1,734 / 927 -> 879, FP8 MLP down 4352x5120 2,065 -> 2,037 /
# 1,044 -> 1,023, out-projections 1536x5120 tie at 2-4K and 199 -> 189 at 1K.
FP8_ALGO = {(5120, 4096): [(3072, 1 << 30, 0, 6)], (5120, 3584): [(1536, 1 << 30, 0, 6)],
            (4352, 5120): [(FP8_MIN, 1 << 30, 0, 6)], (1536, 5120): [(FP8_MIN, 1536, 0, 6)]}
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
    if _on("Q27_PF_ARN"):
        apply_arn()
    if _on("Q27_PF_FP8") and FP8_ALGO:
        apply_fp8()


def exts():
    """(name, loader) of every extension of this module, for the build script"""
    return [("pf_arn", arn_ext), ("pf_gemm", gemm_ext)]
