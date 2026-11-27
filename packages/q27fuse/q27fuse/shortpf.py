"""q27fuse.shortpf - Exact short/agentic prefill GEMM kernels (M = 65..512) for 4x V100-SXM2 TP4.

  1. Q27_SPF_MLP (csrc/mlp_short.cu):
     Exact single-pass 2D-tiled W4A16 HMMA.884 reading q27_w directly (zero extra weight copies, zero 64-row chunking
     at 65..256, zero DRAM dequant + k_silu_pair at 257..448).
  2. Q27_SPF_FP8 (csrc/fp8_short.cu):
     Exact single-pass 2D-tiled W8A16 HMMA.884 reading 1Cat's QPN8 codes+scales directly (eliminates FP8 dequant
     + small-M cuBLAS overhead at 65..512).

Note: short- and long-prefill all-reduce + Gemma norm are consolidated into pf2.py (csrc/arn.cu, Q27_ARN=1).
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


SPF_MLP = _on("Q27_SPF_MLP", "1")
SPF_FP8 = _on("Q27_SPF_FP8", "1")

SPF_MLP_MIN = _int("Q27_SPF_MLP_MIN", 169)
SPF_MLP_MAX = _int("Q27_SPF_MLP_MAX", 448)
SPF_FP8_MIN = _int("Q27_SPF_FP8_MIN", 65)
SPF_FP8_MAX = _int("Q27_SPF_FP8_MAX", 640)
SPF_FP8_GATED_MAX = _int("Q27_SPF_FP8_GATED_MAX", 448)
# Per-shape upper row limits tuned against 1Cat-vLLM v1.5.4 (M=65..1024 microbench on V100):
#   (5120, 4096) hands off to pf_gemm algo 6 at M >= 512 (+1.8 ms at 512, +1.4 ms at 620)
#   (1536, 5120) keeps fp8_short through 640 (+1.8 ms at 620 vs 1Cat/algo 6)
#   (5120, 3584) / (4352, 5120) hand off to 1Cat v1.5.4 / pf_gemm above 512 (+1.0 ms at 620)
FP8_MAX_BY_KN = {(5120, 4096): 448, (5120, 3584): 512, (4352, 5120): 512, (1536, 5120): 640}

MLP_CFG = [(200, (1, 0)), (256, (3, 1)), (320, (7, 10)), (448, (10, 1))]
FP8_CFG = [
    (128, {}, 11),
    (200, {}, 0),
    (256, {(5120, 4096): 11, (5120, 3584): 0}, 1),
    (384, {}, 1),
    (640, {(5120, 4096): 0, (5120, 3584): 1}, 4),
]
FP8_GATED_CFG = [(200, 0), (256, 1), (448, 0)]

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
                if not gated_silu and out.shape[-1] == N and m <= min(SPF_FP8_MAX, FP8_MAX_BY_KN.get((K, N), 512)):
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
    if SPF_MLP:
        apply_mlp()
    if SPF_FP8:
        apply_fp8()
    if SPF_MLP or SPF_FP8:
        _install_preinit()
