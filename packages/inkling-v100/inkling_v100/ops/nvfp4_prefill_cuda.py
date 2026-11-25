# SPDX-License-Identifier: Apache-2.0
"""Exact native NVFP4 prefill MoE for Volta (sm_70): fused large-M HMMA.884 + SiTU.

Eliminates `skinny_moe_qpn_sm70`'s lack of large-M Tensor Core tiling on prefill chunks
while preserving 100% exact native NVFP4 math (zero lossy weight packing, zero extra
permanent weight copies, zero FP16 weight materialization, zero host synchronization).
Reads directly from in-place QPN-prepacked NVFP4 weights.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import torch

_EXT_NAME = "inkling_nvfp4_prefill"
_EXT: ModuleType | None = None

# Execution modes:
#   0 = Optimal zero-sync HMMA.884 auto-dispatch (T<=192: 16x64, T<=448: 32x64, T>448: 3-bucket) [DEFAULT]
#   1 = Single-BM HMMA.884 16x64
#   2 = Single-BM HMMA.884 32x64
#   3 = Single-BM HMMA.884 64x64
#   4 = Exact-fit 3-bucket zero-sync HMMA.884 (64x64 + 32x64 + 16x64)
#   5 = Dequant + FP16 cuBLAS TN GEMM two-phase (ablation/reference)
DEFAULT_PREFILL_MODE = 0

# Per-device reusable scratch buffers (zero per-layer allocs; CUDA-graph safe after warmup).
_SCRATCH: dict[int, dict[str, torch.Tensor]] = {}


def _pkg_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_ext_dir() -> Path:
    env = os.environ.get("INKLING_V100_EXT_DIR")
    base = Path(env) if env else (_pkg_dir() / "_ext")
    return base / "nvfp4_prefill"


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


def load_nvfp4_prefill_ext(verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    """Compile (once, into `inkling_v100/_ext/nvfp4_prefill/`) and import `nvfp4_prefill_moe.cu`."""
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT

    csrc = _pkg_dir() / "csrc" / "nvfp4_prefill_moe.cu"
    build_dir = _default_ext_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / f"{_EXT_NAME}.so"

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

    if rebuild:
        for stale in (so_path, build_dir / "nvfp4_prefill_moe.cuda.o"):
            if stale.exists():
                stale.unlink()

    _EXT = load(
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
    return _EXT


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return t.contiguous()


def _get_scratch(
    device: torch.device,
    n13: int,
    h: int,
    inter: int,
    e_local: int,
    num_slots: int,
    mode: int,
    combine: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    entry = _SCRATCH.get(idx)
    if entry is None:
        entry = {}
        _SCRATCH[idx] = entry

    need_wb = (n13 * h) if mode == 5 else 1
    wb0 = entry.get("w_buf0")
    if wb0 is None or wb0.numel() < need_wb:
        wb0 = torch.empty((need_wb,), dtype=torch.float16, device=device)
        entry["w_buf0"] = wb0

    need_tiles = max((num_slots // 8 + 2 * e_local + 2) * 4, 2048)
    tile_desc = entry.get("tile_desc")
    if tile_desc is None or tile_desc.numel() < need_tiles:
        tile_desc = torch.empty((need_tiles,), dtype=torch.int32, device=device)
        entry["tile_desc"] = tile_desc

    tile_count = entry.get("tile_count")
    if tile_count is None or tile_count.numel() < 4:
        tile_count = torch.empty((4,), dtype=torch.int32, device=device)
        entry["tile_count"] = tile_count

    need_inter = (num_slots * inter) if mode != 5 else 1
    inter_buf = entry.get("inter_buf")
    if inter_buf is None or inter_buf.numel() < need_inter:
        inter_buf = torch.empty((need_inter,), dtype=torch.float16, device=device)
        entry["inter_buf"] = inter_buf

    need_y2 = (num_slots * h) if (mode != 5 and combine) else 1
    y2_buf = entry.get("y2_buf")
    if y2_buf is None or y2_buf.numel() < need_y2:
        y2_buf = torch.empty((need_y2,), dtype=torch.float16, device=device)
        entry["y2_buf"] = y2_buf

    need_inv = num_slots if (mode != 5 and combine) else 1
    inv_perm = entry.get("inv_perm")
    if inv_perm is None or inv_perm.numel() < need_inv:
        inv_perm = torch.empty((need_inv,), dtype=torch.int32, device=device)
        entry["inv_perm"] = inv_perm

    need_e = max(e_local, 64)
    counts = entry.get("counts")
    if counts is None or counts.numel() < need_e:
        counts = torch.zeros((need_e,), dtype=torch.int32, device=device)
        base = torch.empty((need_e,), dtype=torch.int32, device=device)
        entry["counts"] = counts
        entry["base"] = base
    else:
        base = entry["base"]

    return wb0, tile_desc, tile_count, inter_buf, y2_buf, inv_perm, counts, base


def preallocate_nvfp4_prefill_scratch(
    device: torch.device,
    h: int = 2304,
    inter: int = 1152,
    e_local: int = 32,
    max_tokens: int = 2048,
    topk: int = 8,
) -> None:
    _get_scratch(device, 2 * inter, h, inter, e_local, max_tokens * topk, 0, True)


def _infer_nk(w_qpn: torch.Tensor, s_qpn: torch.Tensor) -> tuple[int, int]:
    if w_qpn.ndim == 3:
        return int(w_qpn.shape[1]), int(w_qpn.shape[2]) * 2
    if w_qpn.ndim == 5:
        return int(w_qpn.shape[1]) * 32, int(w_qpn.shape[2]) * 16
    raise ValueError(f"Unsupported w_qpn shape: {tuple(w_qpn.shape)}")


def nvfp4_qpn_dequant_expert(
    w_qpn: torch.Tensor,
    s_qpn: torch.Tensor,
    g_t: torch.Tensor,
    expert_idx: int,
    out: torch.Tensor | None = None,
    *,
    use_direct: bool = False,
) -> torch.Tensor:
    """Bit-exact dequantization of one QPN-prepacked NVFP4 expert -> `[N, K]` FP16."""
    ext = load_nvfp4_prefill_ext()
    n, k = _infer_nk(w_qpn, s_qpn)
    if out is None:
        out = torch.empty((n, k), dtype=torch.float16, device=w_qpn.device)
    ext.nvfp4_qpn_dequant_expert(
        _as_u8(w_qpn),
        _as_u8(s_qpn),
        g_t.contiguous(),
        int(expert_idx),
        out,
        bool(use_direct),
    )
    return out


def nvfp4_qpn_dequant_batch(
    w_qpn: torch.Tensor,
    s_qpn: torch.Tensor,
    g_t: torch.Tensor,
    expert_ids_dev: torch.Tensor,
    n_experts: int | None = None,
    out_buf: torch.Tensor | None = None,
    *,
    use_direct: bool = False,
) -> torch.Tensor:
    """Bit-exact batched dequantization of `n_experts` QPN NVFP4 experts -> `[B, N, K]` FP16."""
    ext = load_nvfp4_prefill_ext()
    n, k = _infer_nk(w_qpn, s_qpn)
    b = int(expert_ids_dev.numel()) if n_experts is None else int(n_experts)
    if out_buf is None:
        out_buf = torch.empty((b, n, k), dtype=torch.float16, device=w_qpn.device)
    if b == 0:
        return out_buf
    ext.nvfp4_qpn_dequant_batch(
        _as_u8(w_qpn),
        _as_u8(s_qpn),
        g_t.contiguous(),
        expert_ids_dev.to(torch.int32).contiguous(),
        b,
        out_buf,
        bool(use_direct),
    )
    return out_buf


def run_nvfp4_experts_prefill_cuda(
    x: torch.Tensor,
    w13_qpn: torch.Tensor,
    s13_qpn: torch.Tensor,
    g13_t: torch.Tensor,
    w2_qpn: torch.Tensor,
    s2_qpn: torch.Tensor,
    g2_t: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor | None,
    gate_cap: float,
    up_cap: float,
    *,
    precomputed_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    combine: bool = True,
    mode: int | None = None,
    use_direct_dequant: bool = False,
) -> torch.Tensor:
    """Exact native NVFP4 prefill MoE (Volta sm_70 fused large-M HMMA.884).

    Args:
        x: `[T, H]` FP16 activations.
        w13_qpn, s13_qpn, g13_t: In-place QPN-prepacked gate/up weights, E4M3FN scales, FP32 global scales.
        w2_qpn, s2_qpn, g2_t: In-place QPN-prepacked down weights, E4M3FN scales, FP32 global scales.
        ids: `[T, topk]` int32 local expert ids (`-1` for non-local / padding).
        weights: `[T * topk]` FP32 routing weights.
        src2dst: Unused (kept for drop-in signature parity with `run_nvfp4_experts_sm70`).
        gate_cap, up_cap: Softcap values for exact FP32 interleaved SiTU.
        precomputed_route: Optional `(perm, gids, goff)` from `select_and_route_ep4_cuda`.
        combine: If True (default), returns combined `[T, H]` FP16 directly.
            If False, writes directly to scattered `down_all [T*topk, H]` FP16 matching
            `run_nvfp4_experts_sm70` for use with `combine_sm70`.
        mode: Execution mode override:
            0 = Optimal zero-sync HMMA.884 auto-dispatch (default)
            1 = Single-BM HMMA.884 16x64
            2 = Single-BM HMMA.884 32x64
            3 = Single-BM HMMA.884 64x64
            4 = Exact-fit 3-bucket HMMA.884 (64x64 + 32x64 + 16x64)
            5 = Dequant + cuBLAS TN GEMM two-phase (reference)
        use_direct_dequant: Only applies to mode 5.

    Returns:
        `[T, H]` FP16 tensor if `combine=True`, else `[T * topk, H]` FP16 tensor.
    """
    ext = load_nvfp4_prefill_ext()
    num_tokens, hidden = x.shape
    topk = int(ids.shape[1])
    num_local = int(g13_t.numel())
    n13, k13 = _infer_nk(w13_qpn, s13_qpn)
    inter = n13 // 2
    assert k13 == hidden, f"w13 K={k13} != H={hidden}"
    assert x.dtype == torch.float16, f"Expected FP16 x, got {x.dtype}"

    out_shape = (num_tokens, hidden) if combine else (num_tokens * topk, hidden)
    out = torch.empty(out_shape, dtype=torch.float16, device=x.device)
    if num_tokens == 0:
        return out

    exec_mode = DEFAULT_PREFILL_MODE if mode is None else int(mode)
    S = num_tokens * topk
    wb0, tile_desc, tile_count, inter_buf, y2_buf, inv_perm, counts, base = _get_scratch(
        x.device, n13, hidden, inter, num_local, S, exec_mode, combine
    )

    if precomputed_route is not None:
        perm, gids, goff = precomputed_route
        perm = perm.contiguous()
        gids = gids.contiguous()
        goff = goff.contiguous()
    else:
        max_g = min(S, num_local)
        perm = torch.empty((S,), dtype=torch.int32, device=x.device)
        gids = torch.empty((max_g,), dtype=torch.int32, device=x.device)
        goff = torch.empty((max_g + 1,), dtype=torch.int32, device=x.device)
        rank_buf = torch.empty((S,), dtype=torch.int32, device=x.device)
        ext.nvfp4_route_from_ids(
            ids.contiguous(),
            perm,
            gids,
            goff,
            counts,
            base,
            rank_buf,
            num_local,
        )

    ext.nvfp4_prefill_moe_forward(
        x.contiguous(),
        _as_u8(w13_qpn),
        _as_u8(s13_qpn),
        g13_t.contiguous(),
        _as_u8(w2_qpn),
        _as_u8(s2_qpn),
        g2_t.contiguous(),
        perm,
        gids,
        goff,
        weights.reshape(-1).contiguous(),
        wb0,
        tile_desc,
        tile_count,
        inter_buf,
        y2_buf,
        inv_perm,
        out,
        topk,
        float(gate_cap),
        float(up_cap),
        bool(combine),
        exec_mode,
        bool(use_direct_dequant),
    )
    return out
