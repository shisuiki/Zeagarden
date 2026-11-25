"""q27fuse.shortpf (track shortpf) - Exact short/agentic prefill code path (M = 65..768) for 4x V100-SXM2 TP4.

Targets the exact-path cliffs identified in shortpf_anat.txt:
  1. Q27_SPF_ARN / Q27_SPF_AR (csrc/short_arn.cu):
     CUDA-graph-safe 1-set zero-epoch 0x7f7f sentinel TP4 Reduce-Scatter + All-Gather returning the COMPLETE FP32
     residual on every rank. Eliminates CustomAllreduce cross_device_reduce_2stage (497-990 us/call, 48-60% of
     cold_128/256!) inside piecewise CUDA graphs (80..256) and eager calls up to SPF_ARN_MAX / SPF_AR_MAX.
  2. Q27_SPF_MLP (csrc/mlp_short.cu):
     Exact single-pass 2D-tiled W4A16 HMMA.884 reading q27_w directly (zero extra weight copies, zero 64-row chunking
     at 65..256, zero DRAM dequant + k_silu_pair at 257..768).
  3. Q27_SPF_FP8 (csrc/fp8_short.cu):
     Exact single-pass 2D-tiled W8A16 HMMA.884 reading 1Cat's QPN8 codes+scales directly (eliminates 10.8 ms
     fp8_qpn8_dequantize_sm70_kernel + small-M cuBLAS at 65..512).

Zero impact on decode (M <= 64) or long prefill (M > max). Every pillar independently toggleable via env vars.
"""
import logging
import os
import torch

log = logging.getLogger("q27fuse.shortpf")


def _on(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, "") or default).strip() != "0"


def _int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


SPF_ARN = _on("Q27_SPF_ARN", "1")
SPF_AR = _on("Q27_SPF_AR", "1")
SPF_MLP = _on("Q27_SPF_MLP", "1")
SPF_FP8 = _on("Q27_SPF_FP8", "1")

SPF_ARN_MIN = _int("Q27_SPF_ARN_MIN", 65)
SPF_ARN_MAX = _int("Q27_SPF_ARN_MAX", 1024)
SPF_AR_MIN = _int("Q27_SPF_AR_MIN", 65)
SPF_AR_MAX = _int("Q27_SPF_AR_MAX", 1024)
SPF_MLP_MIN = _int("Q27_SPF_MLP_MIN", 129)
SPF_MLP_MAX = _int("Q27_SPF_MLP_MAX", 448)
SPF_FP8_MIN = _int("Q27_SPF_FP8_MIN", 65)
SPF_FP8_MAX = _int("Q27_SPF_FP8_MAX", 512)
SPF_FP8_GATED_MAX = _int("Q27_SPF_FP8_GATED_MAX", 448)

# Hardware-tuned per-M config tables from standalone V100 sweeps:
# short_arn.cu v2 (test_short_arn.py TP4 sweep: 1.03-1.52x vs pf_arn, 1.36-2.83x vs nccl+1cat_norm, 9-14x vs 2stage)
ARN_CFG = [(256, 3), (1024, 5)]
AR_CFG = [(128, 3), (192, 5), (256, 3), (1024, 5)]
# mlp_short.cu (test_short_gemm.py cold-weight sweep: 1.02-1.16x over M=129..448)
MLP_CFG = [(192, (1, 0)), (256, (3, 1)), (320, (7, 10)), (448, (10, 1))]
# fp8_short.cu linear (test_short_gemm.py cold-weight sweep: 1.13-2.65x over M=65..512)
# [(max_M, {(K, N): cfg}, default_cfg)]
FP8_CFG = [
    (128, {}, 11),
    (192, {}, 0),
    (256, {(5120, 4096): 11, (5120, 3584): 0}, 1),
    (384, {}, 1),
    (512, {(5120, 4096): 0, (5120, 3584): 1}, 4),
]
# fp8_short_gated (test_short_gemm.py cold-weight sweep: 1.14-7.04x over M=65..448)
FP8_GATED_CFG = [(192, 0), (256, 1), (448, 0)]

_SARN = None
_SMLP = None
_SFP8 = None


def _pick(table, m: int):
    for hi, v in table:
        if m <= hi:
            return v
    return table[-1][1]


def _pick_fp8(m: int, k: int, n: int) -> int:
    for hi, by_kn, default in FP8_CFG:
        if m <= hi:
            return by_kn.get((k, n), default)
    return FP8_CFG[-1][2]


def _spf_build_root() -> str:
    env_b = os.environ.get("Q27_SHORTPF_BUILD", "").strip()
    if env_b:
        return env_b
    from .ops import default_build_dir
    return default_build_dir("_spf")


def _spf_csrc(cu_name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", cu_name)


def _load_spf(name: str, cu_name: str, suffix: str, fast_math: bool = True):
    from torch.utils.cpp_extension import load
    src = _spf_csrc(cu_name)
    bdir = _spf_build_root() + suffix
    os.makedirs(bdir, exist_ok=True)
    flags = ["-O3", "-gencode=arch=compute_70,code=sm_70", "-std=c++17"]
    if fast_math:
        flags.append("--use_fast_math")
    return load(name=name, sources=[src], build_directory=bdir, extra_cuda_cflags=flags, verbose=False)


def short_arn_ext():
    global _SARN
    if _SARN is None:
        _SARN = _load_spf("short_arn_ext", "short_arn.cu", "_short_arn", fast_math=False)
    return _SARN


def mlp_short_ext():
    global _SMLP
    if _SMLP is None:
        _SMLP = _load_spf("q27shortpf_mlp", "mlp_short.cu", "_mlp_short", fast_math=True)
    return _SMLP


def fp8_short_ext():
    global _SFP8
    if _SFP8 is None:
        _SFP8 = _load_spf("q27shortpf_fp8", "fp8_short.cu", "_fp8_short", fast_math=True)
    return _SFP8


def _aligned16(t: torch.Tensor) -> bool:
    return (t.data_ptr() & 15) == 0


def _aligned4(t: torch.Tensor) -> bool:
    return (t.data_ptr() & 3) == 0


def _sarn_setup(comm) -> None:
    import torch.distributed as dist
    X = short_arn_ext()
    with torch.cuda.device(comm.device):
        mine = X.alloc_buf()
        hs = [None] * comm.world_size
        dist.all_gather_object(hs, X.ipc_handle(mine), group=comm.cpu_group)
        r = comm.rank_in_group
        X.init([mine if q == r else X.ipc_open(hs[q]) for q in range(4)], r)
    comm._q27_sarn = {"maxm": int(X.max_rows()), "arn_calls": 0, "ar_calls": 0}
    if r == 0:
        log.warning("q27fuse shortpf: short_arn.cu active (ARN %d..%d, AR %d..%d, %.1f MiB/rank, graph-safe)",
                    SPF_ARN_MIN, min(SPF_ARN_MAX, int(X.max_rows())),
                    SPF_AR_MIN, min(SPF_AR_MAX, int(X.max_rows())), X.buf_bytes() / (1 << 20))


def apply_arn() -> None:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator as C
    if getattr(C, "_q27_sarn_patched", False):
        return
    # Ensure pf2.ARN_MIN yields 65..eff_arn_max to short_arn (yielding PK_MIN.. to pf_arn_h32 when Q27_PF_PK=1 & FUSEQ=1)
    from . import ops, pf2
    eff_arn_max = SPF_ARN_MAX
    if ops.PF_PK:
        from . import pf3
        if pf3.FUSEQ and pf3.PK_MIN <= eff_arn_max:
            eff_arn_max = max(SPF_ARN_MIN - 1, pf3.PK_MIN - 1)
    if SPF_ARN and pf2.ARN_MIN <= eff_arn_max:
        pf2.ARN_MIN = eff_arn_max + 1

    orig_init = C.__init__
    orig_arn = C.all_reduce_gemma_rms_norm
    orig_ar = C.all_reduce

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._q27_sarn = None
        if self.world_size == 4 and "tp" in (self.unique_name or "") and self.device.type == "cuda":
            _sarn_setup(self)

    def all_reduce_gemma_rms_norm(self, input_, residual, weight, epsilon):
        st = getattr(self, "_q27_sarn", None)
        if (SPF_ARN and st is not None and input_.dim() == 2
                and SPF_ARN_MIN <= input_.shape[0] <= min(eff_arn_max, st["maxm"])
                and input_.shape[1] == 5120 and input_.dtype == torch.float16
                and residual.dtype == torch.float32 and weight.dtype == torch.float16
                and residual.shape == input_.shape and input_.is_contiguous()
                and residual.is_contiguous() and weight.is_contiguous()
                and _aligned16(input_) and _aligned16(residual) and _aligned16(weight)):
            st["arn_calls"] += 1
            if st["arn_calls"] == 1 and self.rank_in_group == 0:
                log.warning("q27fuse shortpf: first short_arn (%d rows, capturing=%s)",
                            input_.shape[0], torch.cuda.is_current_stream_capturing())
            cfg = _pick(ARN_CFG, int(input_.shape[0]))
            out, res = short_arn_ext().short_arn(input_, residual, weight, float(epsilon), cfg)
            return out, res
        return orig_arn(self, input_, residual, weight, epsilon)

    def all_reduce(self, input_):
        st = getattr(self, "_q27_sarn", None)
        if (SPF_AR and st is not None and input_.dim() == 2
                and SPF_AR_MIN <= input_.shape[0] <= min(SPF_AR_MAX, st["maxm"])
                and input_.shape[1] == 5120 and input_.dtype == torch.float16
                and input_.is_contiguous() and _aligned16(input_)):
            st["ar_calls"] += 1
            if st["ar_calls"] == 1 and self.rank_in_group == 0:
                log.warning("q27fuse shortpf: first short_ar (%d rows, capturing=%s)",
                            input_.shape[0], torch.cuda.is_current_stream_capturing())
            cfg = _pick(AR_CFG, int(input_.shape[0]))
            return short_arn_ext().short_ar(input_, cfg)
        return orig_ar(self, input_)

    C.__init__ = __init__
    C.all_reduce_gemma_rms_norm = all_reduce_gemma_rms_norm
    C.sm70_tp4_all_reduce_gemma_rms_norm = all_reduce_gemma_rms_norm
    C.all_reduce = all_reduce
    C._q27_sarn_patched = True


def apply_mlp() -> None:
    from . import ops
    if getattr(ops, "_q27_smlp_patched", False):
        return
    mlp_short_ext()
    orig_run = ops._mlp_run
    seen = {"n": 0}

    def _mlp_run(mlp, x: torch.Tensor, reduce: bool) -> torch.Tensor:
        m = x.shape[0]
        gu, dn = mlp.gate_up_proj, mlp.down_proj
        ir = gu.output_size_per_partition // 2
        if (SPF_MLP and SPF_MLP_MIN <= m <= SPF_MLP_MAX and x.dtype == torch.float16
                and x.dim() == 2 and x.shape[1] == gu.q27_k and x.shape[1] % 32 == 0
                and ir % 32 == 0 and dn.output_size_per_partition % 32 == 0):
            xc = x if (x.is_contiguous() and _aligned16(x)) else x.contiguous()
            if _aligned16(xc) and _aligned16(gu.q27_w) and _aligned16(dn.q27_w):
                gu_cfg, dn_cfg = _pick(MLP_CFG, m)
                out = mlp_short_ext().mlp_short(xc, gu.q27_w, dn.q27_w, float(gu.q27_epi), float(dn.q27_epi),
                                                ir, gu_cfg, dn_cfg)
                if seen["n"] == 0:
                    seen["n"] = 1
                    log.warning("q27fuse shortpf: NVFP4 MLP on mlp_short.cu (%d rows, cfg=%s)", m, (gu_cfg, dn_cfg))
                if reduce and dn.reduce_results and dn.tp_size > 1:
                    from vllm.distributed import tensor_model_parallel_all_reduce
                    out = tensor_model_parallel_all_reduce(out)
                return out
        return orig_run(mlp, x, reduce)

    ops._mlp_run = _mlp_run
    ops._q27_smlp_patched = True


def apply_fp8() -> None:
    import vllm.model_executor.kernels.linear.qpn.fp8 as F
    if getattr(F, "_q27_sfp8_patched", False):
        return
    ext = fp8_short_ext()
    has_gated = hasattr(ext, "fp8_short_gated")
    orig = F.dispatch_channel_rows
    seen = set()

    def dispatch_channel_rows(out, x, codes, scales, split_k, accumulator_chains, prefetch_codes, gated_silu,
                              native_policy=None):
        m = x.shape[0]
        if (SPF_FP8 and SPF_FP8_MIN <= m <= SPF_FP8_MAX and codes.dim() == 2
                and x.dtype == torch.float16 and out.dtype == torch.float16
                and codes.dtype == torch.uint8 and scales.dtype == torch.float16
                and out.is_contiguous() and _aligned16(out) and _aligned16(codes) and _aligned4(scales)):
            K, N = int(codes.shape[0]), int(codes.shape[1])
            xc = x if (x.is_contiguous() and _aligned16(x)) else x.contiguous()
            if (xc.shape[1] == K and K % 32 == 0 and N % 32 == 0 and _aligned16(xc)
                    and scales.numel() == N):
                if not gated_silu and out.shape[-1] == N:
                    cfg = _pick_fp8(m, K, N)
                    ext.fp8_short(xc, codes.view(-1), scales.view(-1), out, cfg)
                    if (K, N, False) not in seen:
                        seen.add((K, N, False))
                        log.warning("q27fuse shortpf: FP8 %dx%d on fp8_short.cu (%d rows, cfg=%d)", K, N, m, cfg)
                    return
                if gated_silu and has_gated and m <= SPF_FP8_GATED_MAX and out.shape[-1] * 2 == N and (N // 2) % 32 == 0:
                    cfg = _pick(FP8_GATED_CFG, m)
                    ext.fp8_short_gated(xc, codes.view(-1), scales.view(-1), out, cfg)
                    if (K, N, True) not in seen:
                        seen.add((K, N, True))
                        log.warning("q27fuse shortpf: FP8 gated %dx%d on fp8_short_gated (%d rows, cfg=%d)", K, N, m, cfg)
                    return
        return orig(out, x, codes, scales, split_k, accumulator_chains, prefetch_codes, gated_silu, native_policy)

    F.dispatch_channel_rows = dispatch_channel_rows
    F._q27_sfp8_patched = True


def _install_preinit():
    """Pre-initialize mlp_short / fp8_short SMEM attributes on each rank's GPU before CUDA graph capture."""
    from vllm.v1.worker.gpu import cudagraph_utils as CU
    M = CU.CudaGraphManager
    if getattr(M, "_q27_spf_preinit", False):
        return
    prev = M.capture

    def capture(self, *a, **kw):
        try:
            dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
            if dev.type == "cuda":
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
                if SPF_MLP and _SMLP is not None and hasattr(_SMLP, "init"):
                    _SMLP.init(idx)
                if SPF_FP8 and _SFP8 is not None and hasattr(_SFP8, "init"):
                    _SFP8.init(idx)
        except Exception as e:  # noqa: BLE001
            log.warning("q27fuse shortpf: pre-capture init failed (%r)", e)
        return prev(self, *a, **kw)

    M.capture = capture
    M._q27_spf_preinit = True


def apply() -> None:
    if SPF_ARN or SPF_AR:
        apply_arn()
    if SPF_MLP:
        apply_mlp()
    if SPF_FP8:
        apply_fp8()
    if SPF_MLP or SPF_FP8:
        _install_preinit()
