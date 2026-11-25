# SPDX-License-Identifier: Apache-2.0
"""Fused 2-kernel Volta (sm_70) Inkling2 loop message controller (`blend`)."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType

import torch

from .loop import blend_rms_norm, pair_rms_norm

_EXT_NAME = "inkling_loop_blend"
_EXT: ModuleType | None = None

# Empirically tuned crossovers on Tesla V100-SXM2 (R=64 & R=128, D=2304):
#   - S == 1: fused wins up to T <= 16 (at T=32 cuBLAS bmm wins 18.9 vs 20.4 us)
#   - S >= 2 (e.g. S=3 attn streams): fused wins strongly up to T <= 32 (41.3 vs 108.5 us = 2.63x!)
# Tunable via INKLING_LOOP_BLEND_MAX_T to override both (max 32).
_BLEND_MAX_T_S1_DEFAULT = 16
_BLEND_MAX_T_MULTI_S_DEFAULT = 32
_BLEND_MAX_S = 4
_BLEND_MAX_T_CAP = 32

# Per-device persistent scratch buffers (zero allocs during CUDA graph capture):
#   (hidden[4*32*128] fp16, x[4*32*D] fp32, sum_sq[4*32] fp32 zeros, done[4*4] int32 zeros, D)
_SCRATCH: dict[
    int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
] = {}


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


def load_loop_blend_ext(verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT
    csrc = _pkg_dir() / "csrc" / "loop_blend.cu"
    ext_dir = _default_ext_dir()
    ext_dir.mkdir(parents=True, exist_ok=True)
    so_path = ext_dir / f"{_EXT_NAME}.so"

    rebuild = force_rebuild or os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc.stat().st_mtime:
        _EXT = _load_so(so_path)
        return _EXT

    _ensure_cuda_in_path()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from torch.utils.cpp_extension import load

    # Build in an isolated subdir to avoid ninja races, then stage directly under `_ext/`.
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
        verbose=verbose,
    )
    tmp_so = ext_dir / f".{_EXT_NAME}.{os.getpid()}.tmp.so"
    shutil.copy2(built_so, tmp_so)
    os.replace(tmp_so, so_path)
    _EXT = _load_so(so_path)
    return _EXT


def _get_scratch(
    device: torch.device, d: int = 2304
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    entry = _SCRATCH.get(idx)
    if entry is not None and entry[4] >= d:
        return entry[0], entry[1], entry[2], entry[3]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "loop_blend_cuda scratch buffer was not pre-allocated before CUDA graph capture; "
            "call preallocate_loop_blend_scratch(device, d) or run a warmup step first."
        )
    d_alloc = max(d, 2304)
    hidden = torch.empty(_BLEND_MAX_S * _BLEND_MAX_T_CAP * 128, dtype=torch.float16, device=device)
    x = torch.empty(_BLEND_MAX_S * _BLEND_MAX_T_CAP * d_alloc, dtype=torch.float32, device=device)
    sum_sq = torch.zeros(_BLEND_MAX_S * _BLEND_MAX_T_CAP, dtype=torch.float32, device=device)
    done = torch.zeros(_BLEND_MAX_S * 8, dtype=torch.int32, device=device)
    _SCRATCH[idx] = (hidden, x, sum_sq, done, d_alloc)
    return hidden, x, sum_sq, done


def preallocate_loop_blend_scratch(
    device: torch.device | str | int | None = None, d: int = 2304
) -> None:
    """Pre-allocate per-device persistent scratch outside CUDA graph capture."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    _get_scratch(dev, d)


def _blend_max_t(num_streams: int) -> int:
    env = os.environ.get("INKLING_LOOP_BLEND_MAX_T")
    if env is not None:
        return max(0, min(_BLEND_MAX_T_CAP, int(env)))
    return _BLEND_MAX_T_S1_DEFAULT if num_streams == 1 else _BLEND_MAX_T_MULTI_S_DEFAULT


def _loop_blend_reference_fallback(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    num_tokens = dest.shape[1]
    if num_tokens <= 1024:
        pair = pair_rms_norm(dest, message, out_dtype=torch.float16)
        hidden = torch.bmm(pair, down.transpose(1, 2))
        del pair
        gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
        del hidden
        return blend_rms_norm(
            dest, message, gain_forget, norm_weight, norm_eps, out_dtype=torch.float16
        )
    out = message.new_empty(message.shape, dtype=torch.float16)
    for t0 in range(0, num_tokens, 1024):
        t1 = min(t0 + 1024, num_tokens)
        d_c = dest[:, t0:t1]
        m_c = message[:, t0:t1].contiguous()
        pair = pair_rms_norm(d_c, m_c, out_dtype=torch.float16)
        hidden = torch.bmm(pair, down.transpose(1, 2))
        del pair
        gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
        del hidden
        out[:, t0:t1] = blend_rms_norm(
            d_c, m_c, gain_forget, norm_weight, norm_eps, out_dtype=torch.float16
        )
    return out


def loop_blend_cuda(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    force_cuda: bool | None = None,
) -> torch.Tensor:
    """Fused loop message controller blend (`dest`, `message` FP32 -> `out` FP16 `(S, T, D)`).

    Drop-in replacement for `Inkling2LoopDecoderLayer.blend` body. Supports broadcast
    `dest.stride(0) == 0` and `norm_weight.stride(0) == 0` (local layers).
    """
    num_streams, num_tokens, d = message.shape
    if num_streams == 0 or num_tokens == 0:
        return message.new_empty(message.shape, dtype=torch.float16)

    rank = down.shape[1]
    cuda_eligible = (
        0 < num_streams <= _BLEND_MAX_S
        and 0 < num_tokens <= _BLEND_MAX_T_CAP
        and rank in (64, 128)
        and (d % 32) == 0
        and dest.dtype == torch.float32
        and message.dtype == torch.float32
        and down.dtype == torch.float16
        and up.dtype == torch.float16
        and norm_weight.dtype == torch.float16
        and message.is_contiguous()
        and down.is_contiguous()
        and up.is_contiguous()
        and dest.stride(1) == d
        and dest.stride(2) == 1
        and (dest.stride(0) % 4) == 0
        and norm_weight.stride(1) == 1
        and (norm_weight.stride(0) % 4) == 0
    )
    use_cuda = (
        (force_cuda and cuda_eligible)
        if force_cuda is not None
        else (cuda_eligible and num_tokens <= _blend_max_t(num_streams))
    )
    if not use_cuda:
        return _loop_blend_reference_fallback(dest, message, down, up, norm_weight, norm_eps)

    ext = load_loop_blend_ext()
    scratch_hidden, scratch_x, scratch_sum_sq, scratch_done = _get_scratch(message.device, d)
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.float16)
    ext.loop_blend_sm70(
        dest,
        message,
        down,
        up,
        norm_weight,
        out,
        scratch_hidden,
        scratch_x,
        scratch_sum_sq,
        scratch_done,
        float(norm_eps),
    )
    return out
