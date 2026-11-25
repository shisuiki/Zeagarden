# SPDX-License-Identifier: Apache-2.0
"""Volta (sm_70) fused 1-shot TP4 NVLink P2P AllReduce + MoE combine + short_conv + AttnRes."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist

_EXT_NAME = "inkling_ar_conv_mix"
_EXT: ModuleType | None = None
_INIT_ATTEMPTED: bool = False
_INIT_OK: bool = False
_LOCAL_IPC_PTR: int = 0  # kept alive for process lifetime


def _pkg_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_ext_dir() -> Path:
    env = os.environ.get("INKLING_V100_EXT_DIR")
    if env:
        return Path(env) / "ar_conv_mix"
    return _pkg_dir() / "_ext" / "ar_conv_mix"


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


def load_ar_conv_mix_ext(verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    """Compile (once, without --use_fast_math) and import `all_reduce_conv_mix.cu`."""
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT

    csrc = _pkg_dir() / "csrc" / "all_reduce_conv_mix.cu"
    ext_dir = _default_ext_dir()
    ext_dir.mkdir(parents=True, exist_ok=True)
    so_path = ext_dir / f"{_EXT_NAME}.so"

    rebuild = force_rebuild or os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc.stat().st_mtime:
        spec = importlib.util.spec_from_file_location(_EXT_NAME, str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    _ensure_cuda_in_path()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from torch.utils.cpp_extension import load

    if rebuild and so_path.exists():
        so_path.unlink()

    mod = load(
        name=_EXT_NAME,
        sources=[str(csrc)],
        build_directory=str(ext_dir),
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_70,code=sm_70",
            "-std=c++17",
            "--ptxas-options=-v",
        ],
        verbose=verbose,
    )
    _EXT = mod
    return _EXT


def init_tp4_ar_conv_mix(group: dist.ProcessGroup | None = None) -> bool:
    """Initialize TP4 NVLink IPC buffers once across the 4 TP ranks (idempotent)."""
    global _INIT_ATTEMPTED, _INIT_OK, _LOCAL_IPC_PTR
    if _INIT_ATTEMPTED:
        return _INIT_OK
    _INIT_ATTEMPTED = True
    if os.environ.get("INKLING_USE_CUDA_AR_CONV_MIX", "1") == "0":
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    if group is None:
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        group = tp.cpu_group
        world_size = tp.world_size
        rank = tp.rank_in_group
    else:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
    if world_size != 4:
        return False

    ext = load_ar_conv_mix_ext()
    if ext.is_initialized():
        _INIT_OK = True
        return True

    # Ensure IPC memory allocation happens outside expandable_segments for CUDA IPC compatibility.
    local_ptr = ext.alloc_buf()
    _LOCAL_IPC_PTR = local_ptr
    my_handle = ext.ipc_handle(local_ptr)
    handles: list[bytes | None] = [None] * world_size
    dist.all_gather_object(handles, my_handle, group=group)
    bases: list[int] = []
    for r in range(world_size):
        if r == rank:
            bases.append(local_ptr)
        else:
            h = handles[r]
            assert h is not None
            bases.append(ext.ipc_open(h))
    ext.init(bases, rank)
    dist.barrier(group=group)
    _INIT_OK = True
    return True


def is_ar_conv_mix_ready(num_tokens: int) -> bool:
    return _INIT_OK and 0 < num_tokens <= 64


def all_reduce_conv_mix_cuda(
    *,
    x: torch.Tensor | None = None,
    combine: tuple[torch.Tensor, torch.Tensor] | None = None,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    addend: torch.Tensor | None,
    residual: torch.Tensor | None,
    conv_out: torch.Tensor,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    ext = load_ar_conv_mix_ext()
    down = None if combine is None else combine[0]
    src2dst = None if combine is None else combine[1]
    return ext.ar_conv_mix_forward(
        x,
        down,
        src2dst,
        conv_weight,
        cache,
        positions,
        block_table,
        seq_idx,
        slot_mapping,
        query_start,
        int(stream_offset),
        int(stream_width),
        addend,
        residual,
        conv_out,
        stack,
        last,
        queries,
        norm_weight,
        float(eps),
        float(norm_eps),
    )
