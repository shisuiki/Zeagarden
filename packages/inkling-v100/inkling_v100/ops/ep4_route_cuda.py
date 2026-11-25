# SPDX-License-Identifier: Apache-2.0
"""Fused Volta (sm_70) Inkling2 MoE gate GEMV + router + zero-sync EP=4 sentinel grouping."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType

import torch

_EXT_NAME = "inkling_ep4_route"
_EXT: ModuleType | None = None

# Maximum token count using the single-kernel fused gate GEMV + route path.
# Empirically tuned on Tesla V100-SXM2: fused wins strongly at T=1..8 (6.7-10.4 vs 10.5-17.5 us),
# while cuBLAS HMMA mm + 16-warp bitmask route wins at T>=16 (15.4 vs 15.7 us).
# Tunable via INKLING_ROUTE_GEMV_MAX_T (max 32).
_GEMV_MAX_T_DEFAULT = 8

# Per-device persistent scratch buffers (zero allocs during CUDA graph capture):
#   (counts[64], base[64], meta[2], gemv_logits[32*128], gemv_done[1], dummy_rank[1])
_SCRATCH: dict[
    int,
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
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


def load_ep4_route_ext(verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT
    csrc = _pkg_dir() / "csrc" / "ep4_route.cu"
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

    # Build in an isolated subdir to avoid ninja races with parallel extension builds,
    # then atomically stage the final .so directly under `_ext/inkling_ep4_route.so`.
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
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    buf = _SCRATCH.get(idx)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "ep4_route_cuda scratch buffer was not pre-allocated before CUDA graph capture; "
                "call preallocate_ep4_route_scratch(device) or run a warmup step first."
            )
        counts = torch.zeros(64, dtype=torch.int32, device=device)
        base = torch.empty(64, dtype=torch.int32, device=device)
        meta = torch.empty(2, dtype=torch.int32, device=device)
        gemv_logits = torch.empty(32 * 128, dtype=torch.float32, device=device)
        gemv_done = torch.zeros(1, dtype=torch.int32, device=device)
        dummy_rank = torch.empty(1, dtype=torch.int32, device=device)
        buf = (counts, base, meta, gemv_logits, gemv_done, dummy_rank)
        _SCRATCH[idx] = buf
    return buf


def preallocate_ep4_route_scratch(device: torch.device | str | int | None = None) -> None:
    """Pre-allocate per-device persistent scratch outside CUDA graph capture."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    _get_scratch(dev)


def _gemv_max_t() -> int:
    env = os.environ.get("INKLING_ROUTE_GEMV_MAX_T")
    if env is not None:
        return max(0, min(32, int(env)))
    return _GEMV_MAX_T_DEFAULT


def select_and_route_ep4_cuda(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    lo: int,
    hi: int,
    is_padding: torch.Tensor | None = None,
    *,
    force_gemv: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused gate GEMV (small T) + router + zero-sync EP=4 sentinel grouping.

    Returns `(ids [T, topk], weights [S], src2dst [S], perm [S], gids [S], goff [S+1])`.
    """
    ext = load_ep4_route_ext()
    tokens, hidden = x.shape
    num_experts = gate_weight.shape[0]
    e_local = hi - lo
    S = tokens * topk
    device = x.device
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    weights = torch.empty(S, dtype=torch.float32, device=device)
    src2dst = torch.empty(S, dtype=torch.int32, device=device)
    perm = torch.empty(S, dtype=torch.int32, device=device)
    gids = torch.empty(S, dtype=torch.int32, device=device)
    goff = torch.empty(S + 1, dtype=torch.int32, device=device)
    if tokens == 0:
        goff.zero_()
        return ids, weights, src2dst, perm, gids, goff

    counts, base, meta, gemv_logits, gemv_done, dummy_rank = _get_scratch(device)

    gemv_eligible = (
        tokens <= 32
        and num_experts == 128
        and topk == 8
        and 0 < e_local <= 32
        and (hidden % 8) == 0
        and x.dtype == torch.float16
        and gate_weight.dtype == torch.float16
        and x.is_contiguous()
        and gate_weight.is_contiguous()
    )
    use_gemv = (
        (force_gemv and gemv_eligible)
        if force_gemv is not None
        else (gemv_eligible and tokens <= _gemv_max_t())
    )

    if use_gemv:
        ext.ep4_gemv_select_and_route(
            x,
            gate_weight,
            bias,
            is_padding,
            ids,
            weights,
            src2dst,
            perm,
            gids,
            goff,
            gemv_logits,
            gemv_done,
            int(topk),
            float(route_scale),
            int(lo),
            int(hi),
        )
        return ids, weights, src2dst, perm, gids, goff

    logits = torch.mm(x, gate_weight.t(), out_dtype=torch.float32)
    need_rank = not (
        (tokens <= 32 and num_experts <= 128 and topk == 8 and e_local <= 32)
        or (tokens <= 16 and num_experts <= 128 and S <= 256)
    )
    rank_buf = torch.empty(S, dtype=torch.int32, device=device) if need_rank else dummy_rank
    ext.ep4_select_and_route(
        logits,
        bias,
        is_padding,
        ids,
        weights,
        src2dst,
        perm,
        gids,
        goff,
        counts,
        base,
        meta,
        rank_buf,
        int(topk),
        float(route_scale),
        int(lo),
        int(hi),
    )
    return ids, weights, src2dst, perm, gids, goff
