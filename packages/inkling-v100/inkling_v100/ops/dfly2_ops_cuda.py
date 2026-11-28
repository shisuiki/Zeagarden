# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 CUDA extension loader & workspace helpers for Inkling-2-Mini DFly2-KV."""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any

import torch

_EXT: Any = None
_WS: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_EMPTY_F32: dict[tuple[str, int | None], torch.Tensor] = {}


def _ensure_cuda_in_path() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for cand in ("/usr/local/cuda", "/usr/local/cuda-12.9", "/usr/local/cuda-12.8", "/usr/local/cuda-12"):
        if (pathlib.Path(cand) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = cand
            path = os.environ.get("PATH", "")
            bin_dir = str(pathlib.Path(cand) / "bin")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
            return


def _ext_dir() -> pathlib.Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    p = pathlib.Path(env_dir) if env_dir else (pathlib.Path(__file__).resolve().parents[1] / "_ext")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dfly2_ops_ext(verbose: bool = False) -> Any:
    """Build (if needed) and import `inkling_dfly2_ops` from `csrc/dfly2_ops.cu`."""
    global _EXT
    if _EXT is not None:
        return _EXT

    csrc_path = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "dfly2_ops.cu"
    build_dir = _ext_dir() / "dfly2_ops"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "inkling_dfly2_ops.so"

    rebuild = os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc_path.stat().st_mtime:
        spec = importlib.util.spec_from_file_location("inkling_dfly2_ops", str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    _ensure_cuda_in_path()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from torch.utils.cpp_extension import load

    _EXT = load(
        name="inkling_dfly2_ops",
        sources=[str(csrc_path)],
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
    return _EXT


def preallocate_dfly2_workspace(
    slots: int = 32768,
    head_dim: int = 256,
    device: torch.device | str | int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-allocate persistent split-KV workspace `(o_part, ml_part)` outside CUDA graph capture."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (dev.type, dev.index)
    cur = _WS.get(key)
    need_o = slots * head_dim
    need_ml = slots * 2
    if cur is not None and cur[0].numel() >= need_o and cur[1].numel() >= need_ml:
        return cur
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"dfly2_ops_cuda: workspace ({slots} slots x {head_dim}) cannot be allocated inside CUDA graph capture."
        )
    alloc_o = max(need_o, 32768 * 256)
    alloc_ml = max(need_ml, 32768 * 2)
    o_part = torch.empty((alloc_o,), dtype=torch.float32, device=dev)
    ml_part = torch.empty((alloc_ml,), dtype=torch.float32, device=dev)
    _WS[key] = (o_part, ml_part)
    return o_part, ml_part


def empty_f32(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    t = _EMPTY_F32.get(key)
    if t is None:
        t = torch.empty((0,), dtype=torch.float32, device=device)
        if not torch.cuda.is_current_stream_capturing():
            _EMPTY_F32[key] = t
    return t


def plan_dfly2_splits(is_local: bool, num_reqs: int, q: int, is_gla: bool = False) -> int:
    """Wave-aligned split-KV count on 80-SM Volta V100."""
    work = max(1, num_reqs * 2)  # 2 KV heads (or 2 m_tiles for GLA)
    if is_local:
        # SWA-1024 = at most 32 tiles of 32 tokens.
        if work <= 4:
            return 8 if q > 1 else 4
        if work <= 16:
            return 4 if q > 1 else 2
        return 1
    # Global L23: align CTAs to 80 SMs (keep S <= 32 for single-pass warp combine)
    if work <= 2:
        return 32
    if work <= 4:
        return 20
    if work <= 16:
        return 10
    if work <= 32:
        return 5
    return 2
