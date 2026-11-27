# SPDX-License-Identifier: Apache-2.0
"""1-launch persistent Volta (sm_70) EP=4 NVFP4 MoE extension for M <= 8 decode/verify batches.

Fuses `router GEMV + top-8 ep4_select + W13+softcapped SiTU + W2 + slot-order token sum`
into a single 80-CTA x 1024-thread persistent kernel with 2 self-resetting grid barriers
(zero host sync, zero memset, 100% CUDA-graph safe). Output `[M, 2304]` FP16 is this rank's
partial EP=4 MoE sum, ready to pass directly into `all_reduce_conv_mix_cuda(x=out, ...)`.
"""

from __future__ import annotations

import fcntl
import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType

import torch

_EXT_NAME = "inkling_ep4_moe_fused"
_EXT: ModuleType | None = None

_E = 128
_E_LOCAL = 32
_H = 2304
_I = 1152
_TOPK = 8
MAX_TOKENS = 8
_MIN_SMS = 80

# Per-device persistent scratch buffers (zero allocs during CUDA graph capture):
#   (logits [8, 128] fp32, act [64, 1152] fp16, counters [2] int32)
_SCRATCH: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
# Cached device eligibility (sm_70 and >= 80 SMs) per device index.
_DEVICE_OK: dict[int, bool] = {}


def is_ep4_moe_fused_enabled() -> bool:
    """Return True when the 1-launch persistent EP=4 NVFP4 MoE kernel is enabled."""
    return os.environ.get("INKLING_USE_CUDA_EP4_MOE_FUSED", "1") != "0"


def _pkg_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_ext_dir() -> Path:
    env = os.environ.get("INKLING_V100_EXT_DIR")
    return Path(env) if env else (_pkg_dir() / "_ext")


def _ensure_cuda_in_path() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for cand in ("/usr/local/cuda-12.9", "/usr/local/cuda-12", "/usr/local/cuda"):
        if (Path(cand) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = cand
            path = os.environ.get("PATH", "")
            bin_dir = str(Path(cand) / "bin")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
            return


def _load_so(so_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(_EXT_NAME, str(so_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ep4_moe_fused_ext(verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    """Compile (once, without --use_fast_math) and import `ep4_moe_fused.cu`."""
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT

    csrc = _pkg_dir() / "csrc" / "ep4_moe_fused.cu"
    ext_dir = _default_ext_dir()
    ext_dir.mkdir(parents=True, exist_ok=True)
    so_path = ext_dir / f"{_EXT_NAME}.so"

    rebuild = force_rebuild or os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc.stat().st_mtime:
        _EXT = _load_so(so_path)
        return _EXT

    lock_path = ext_dir / f".{_EXT_NAME}.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check after acquiring lock so concurrent EP=4 ranks build at most once.
            if (
                not rebuild
                and so_path.exists()
                and so_path.stat().st_mtime >= csrc.stat().st_mtime
            ):
                _EXT = _load_so(so_path)
                return _EXT

            _ensure_cuda_in_path()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
            from torch.utils.cpp_extension import load

            build_dir = ext_dir / f"_build_{_EXT_NAME}"
            build_dir.mkdir(parents=True, exist_ok=True)
            built_so = build_dir / f"{_EXT_NAME}.so"
            if rebuild:
                for p in (*build_dir.glob("*.o"), built_so, so_path):
                    if p.exists():
                        p.unlink()

            load(
                name=_EXT_NAME,
                sources=[str(csrc)],
                build_directory=str(build_dir),
                extra_cuda_cflags=[
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-std=c++17",
                    "--ptxas-options=-v",
                ],
                extra_cflags=["-O3", "-std=c++17"],
                verbose=verbose,
            )
            tmp_so = ext_dir / f".{_EXT_NAME}.{os.getpid()}.tmp.so"
            shutil.copy2(built_so, tmp_so)
            os.replace(tmp_so, so_path)
            _EXT = _load_so(so_path)
            return _EXT
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _device_index(device: torch.device | str | int | None) -> tuple[torch.device, int]:
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    assert dev.type == "cuda" and dev.index is not None, f"Expected CUDA device, got {dev}"
    return dev, int(dev.index)


def _is_device_supported(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ok = _DEVICE_OK.get(idx)
    if ok is None:
        props = torch.cuda.get_device_properties(idx)
        ok = (props.major == 7 and props.minor == 0 and props.multi_processor_count >= _MIN_SMS)
        _DEVICE_OK[idx] = ok
    return ok


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev, idx = _device_index(device)
    buf = _SCRATCH.get(idx)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "ep4_moe_fused_cuda scratch buffer was not pre-allocated before CUDA graph capture; "
                "call preallocate_ep4_moe_fused_scratch(device) or run a warmup step first."
            )
        logits = torch.empty((MAX_TOKENS, _E), dtype=torch.float32, device=dev)
        act = torch.empty((MAX_TOKENS * _TOPK, _I), dtype=torch.float16, device=dev)
        counters = torch.zeros((2,), dtype=torch.int32, device=dev)
        buf = (logits, act, counters)
        _SCRATCH[idx] = buf
    return buf


def preallocate_ep4_moe_fused_scratch(device: torch.device | str | int | None = None) -> None:
    """Pre-allocate per-device scratch and configure/verify kernel occupancy outside graph capture."""
    dev, idx = _device_index(device)
    _get_scratch(dev)
    if _is_device_supported(dev):
        ext = load_ep4_moe_fused_ext()
        ext.init_ep4_moe_fused(idx)


def ep4_moe_fused_kernel_info(device: torch.device | str | int | None = None) -> dict[str, int]:
    """Return kernel register count, static/dynamic SMEM bytes, local spill bytes, and occupancy."""
    _, idx = _device_index(device)
    ext = load_ep4_moe_fused_ext()
    regs, static_smem, local_bytes, max_dyn, occ, req_dyn, max_routes = (
        ext.ep4_moe_fused_kernel_info(idx)
    )
    return {
        "num_regs": int(regs),
        "static_smem_bytes": int(static_smem),
        "local_bytes": int(local_bytes),
        "max_dyn_smem_bytes": int(max_dyn),
        "occupancy": int(occ),
        "req_dyn_smem_bytes": int(req_dyn),
        "max_stage_routes": int(max_routes),
    }


def is_ep4_moe_fused_eligible(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    w13_qpn: torch.Tensor,
    w2_qpn: torch.Tensor,
    expert_lo: int,
    expert_hi: int,
) -> bool:
    """Fast eligibility check for the 1-launch persistent EP=4 NVFP4 MoE kernel."""
    if not is_ep4_moe_fused_enabled():
        return False
    if not (
        x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == 2
        and 1 <= x.shape[0] <= MAX_TOKENS
        and x.shape[1] == _H
    ):
        return False
    if not _is_device_supported(x.device):
        return False
    return (
        gate_weight.dtype == torch.float16
        and tuple(gate_weight.shape) == (_E, _H)
        and tuple(w13_qpn.shape) == (_E_LOCAL, _H, _H // 2)
        and tuple(w2_qpn.shape) == (_E_LOCAL, _H, _I // 2)
        and expert_lo >= 0
        and expert_hi <= _E
        and (expert_hi - expert_lo) == _E_LOCAL
    )


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return t.contiguous()


def ep4_nvfp4_moe_fused_cuda(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w13_qpn: torch.Tensor,
    s13_qpn: torch.Tensor,
    g13_t: torch.Tensor,
    w2_qpn: torch.Tensor,
    s2_qpn: torch.Tensor,
    g2_t: torch.Tensor,
    route_scale: float,
    gate_cap: float,
    up_cap: float,
    expert_lo: int,
    expert_hi: int,
    is_padding: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run 1-launch persistent Volta sm_70 EP=4 NVFP4 MoE for `M <= 8` tokens.

    Args:
        x: `[M, 2304]` FP16 hidden states.
        gate_weight: `[128, 2304]` FP16 router weights (`Inkling2Gate.weight`).
        gate_bias: `[128]` FP32 router bias (`Inkling2Gate.bias`).
        w13_qpn: `[32, 2304, 1152]` uint8 in-place QPN-prepacked gate/up weights.
        s13_qpn: `[32, 2304, 144]` uint8 (or float8_e4m3fn) QPN E4M3FN scales.
        g13_t: `[32]` FP32 per-expert global scales for W13.
        w2_qpn: `[32, 2304, 576]` uint8 in-place QPN-prepacked down weights.
        s2_qpn: `[32, 2304, 72]` uint8 (or float8_e4m3fn) QPN E4M3FN scales.
        g2_t: `[32]` FP32 per-expert global scales for W2.
        route_scale: Router output scale factor.
        gate_cap: Gate softcap for exact FP32 interleaved SiTU.
        up_cap: Up softcap for exact FP32 interleaved SiTU.
        expert_lo: Inclusive start global expert index for this EP=4 rank.
        expert_hi: Exclusive end global expert index (`expert_hi - expert_lo == 32`).
        is_padding: Optional `[M]` bool tensor marking padding tokens (produce zeros).
        out: Optional preallocated `[M, 2304]` FP16 output buffer.

    Returns:
        `[M, 2304]` FP16 partial EP=4 MoE output summed over local experts for each token.
    """
    m = int(x.shape[0])
    if out is None:
        out = torch.empty((m, _H), dtype=torch.float16, device=x.device)
    if m == 0:
        return out

    ext = load_ep4_moe_fused_ext()
    logits, act, counters = _get_scratch(x.device)
    pad = is_padding.contiguous() if is_padding is not None else None

    ext.ep4_nvfp4_moe_fused_forward(
        x.contiguous(),
        gate_weight.contiguous(),
        gate_bias.contiguous(),
        pad,
        _as_u8(w13_qpn),
        _as_u8(s13_qpn),
        g13_t.contiguous(),
        _as_u8(w2_qpn),
        _as_u8(s2_qpn),
        g2_t.contiguous(),
        logits,
        act,
        counters,
        out,
        float(route_scale),
        float(gate_cap),
        float(up_cap),
        int(expert_lo),
        int(expert_hi),
    )
    return out
