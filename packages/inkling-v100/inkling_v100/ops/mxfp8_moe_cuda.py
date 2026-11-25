# SPDX-License-Identifier: Apache-2.0
"""Volta (sm_70) exact native MXFP8 grouped MoE extension loader, QPN8 prepack, and wrappers.

Zero host synchronization, 100% CUDA-graph safe, zero lossy weight packing.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import torch

from .moe_sm70 import ep4_route_to_skinny_groups

_EXT_NAME = "inkling_mxfp8_moe"
_EXT: ModuleType | None = None

# Exact Volta QPN8 B-fragment k-order per 16-element half-chunk (matches fp8_tile.cuh / fp8_lin.cu).
_QPN8_KORDER16 = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)

# Maximum per-expert UE8M0 span (hi - lo) for the fast FP16 power-of-two hmul2 path:
#   - <= 22: s_reb in [112..134] -> 2^(s_reb - 119) in [2^-7..2^+15], 100% bit-exact for ALL 254
#     finite E4M3FN codes including subnormals (zero rounding, zero underflow, zero overflow).
#   - > 22: automatically uses exact FP32 per-chunk UE8M0 scaling (supports full span 0..254).
_DEFAULT_MAX_FAST_SPAN = 22


def _pkg_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_ext_dir() -> Path:
    env = os.environ.get("INKLING_V100_EXT_DIR")
    if env:
        return Path(env)
    return _pkg_dir() / "_ext"


def _ensure_cuda_in_path() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for cand in ("/usr/local/cuda-12.9", "/usr/local/cuda-12", "/usr/local/cuda"):
        nvcc = Path(cand) / "bin" / "nvcc"
        if nvcc.exists():
            os.environ["CUDA_HOME"] = cand
            path = os.environ.get("PATH", "")
            bin_dir = str(Path(cand) / "bin")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
            return


def load_mxfp8_moe_ext(verbose: bool = False, force_rebuild: bool = False) -> ModuleType:
    """Compile (once, into `inkling_v100/_ext/`) and import `mlp_mxfp8_moe.cu`."""
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT

    csrc = _pkg_dir() / "csrc" / "mlp_mxfp8_moe.cu"
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
        if torch.cuda.is_available():
            _EXT.init()
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
    if torch.cuda.is_available():
        _EXT.init()
    return _EXT


def mxfp8_qpn8_prepack(
    w_u8: torch.Tensor,
    s_u8: torch.Tensor,
    *,
    is_w13: bool = False,
    inplace: bool = False,
    max_fast_span: int = _DEFAULT_MAX_FAST_SPAN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Permute MXFP8 expert weights & UE8M0 scales into exact Volta QPN8 layout.

    Byte-equal permutation (zero lossy packing):
      - `W_qpn`: `[E, N/32, K/32, 32, 32]` uint8 (logically `[E, N/32, K/32, 2, 32, 16]` where
        half 0 holds k=0..15 and half 1 holds k=16..31, lane l holds output col
        `8*((l>>2)&3) + (l&3) + 4*(l>>4)`, bytes ordered `0,2,4,6,1,3,5,7` per 8).
      - `S_qpn`: `[E, N/32, K/32, 32]` uint8 (1 byte per `(tile, chunk_k32, lane)`).
      - When `is_w13=True`, interleaved `(gate_0, up_0, gate_1, up_1, ...)` rows are de-interleaved
        into gate tiles `0..I/32-1` followed by up tiles `I/32..2*I/32-1` so `w13_situ` fuses
        gate+up with exact FP32 softcapped SiTU directly in registers.
      - Per-expert UE8M0 scale rebase:
        If every expert's non-255 UE8M0 span `hi - lo <= max_fast_span` (default 22, guaranteed
        100% bit-exact in FP16 for all 254 E4M3FN codes including subnormals; up to 29 fits FP16
        normal exponent range), sets `shift_e = 134 - hi_e`, stores `s_reb = s + shift_e`
        (255 preserved as 255) in `S_qpn` and `g_scale[e] = 2^(-shift_e)` (FP32), returning
        `wide_scale=False`. Otherwise preserves raw UE8M0 bytes with `g_scale=1.0` and returns
        `wide_scale=True` for the exact FP32 per-chunk scale kernel.

    Args:
        w_u8: `[E, N, K]` uint8 or float8_e4m3fn contiguous tensor.
        s_u8: `[E, N, K//32]` uint8 contiguous tensor.
        is_w13: True for interleaved Inkling2 `w13` weights (`N = 2 * I`, `I % 32 == 0`).
        inplace: If True, overwrites `w_u8` and `s_u8` storage in-place (expert-by-expert) so
            permanent VRAM overhead is zero.
        max_fast_span: Max allowed `hi - lo` per expert for the fast FP16 `hmul2` path (<= 29).

    Returns:
        `(W_qpn, S_qpn, g_scale, wide_scale)`
    """
    if w_u8.dtype == torch.float8_e4m3fn:
        w_raw = w_u8.view(torch.uint8)
    else:
        assert w_u8.dtype == torch.uint8, f"Expected uint8 or float8_e4m3fn, got {w_u8.dtype}"
        w_raw = w_u8
    assert s_u8.dtype == torch.uint8, f"Expected uint8 scales, got {s_u8.dtype}"
    assert w_raw.ndim == 3 and s_u8.ndim == 3, f"Expected 3D [E, N, K], got {w_raw.shape} / {s_u8.shape}"
    assert 0 <= max_fast_span <= 29, f"max_fast_span must be in [0, 29], got {max_fast_span}"

    E, N, K = w_raw.shape
    assert K % 32 == 0 and N % 32 == 0, f"N and K must be multiples of 32, got N={N}, K={K}"
    assert s_u8.shape == (E, N, K // 32), f"scale shape mismatch: {s_u8.shape} vs {(E, N, K // 32)}"
    if is_w13:
        assert N % 64 == 0, f"w13 requires I = N/2 to be a multiple of 32 (N % 64 == 0), got N={N}"

    dev = w_raw.device
    NT = N // 32
    NC = K // 32

    # Step 1: Analyze per-expert non-255 UE8M0 spans.
    shifts: list[int] = []
    max_span = 0
    for e in range(E):
        se = s_u8[e]
        valid = se != 255
        if bool(valid.any()):
            se_v = se[valid]
            lo = int(se_v.min().item())
            hi = int(se_v.max().item())
            span = hi - lo
            if span > max_span:
                max_span = span
            shifts.append(134 - hi)
        else:
            shifts.append(0)

    wide_scale = max_span > max_fast_span
    if wide_scale:
        g_scale = torch.ones((E,), dtype=torch.float32, device=dev)
        eff_shifts = [0] * E
    else:
        g_scale = torch.tensor([float(2.0 ** (-sh)) for sh in shifts], dtype=torch.float32, device=dev)
        eff_shifts = shifts

    # Step 2: Build QPN8 lane->col and k-order index tables.
    lane = torch.arange(32, dtype=torch.long, device=dev)
    col_map = ((lane >> 2) & 3) * 8 + (lane & 3) + (lane >> 4) * 4
    korder16 = torch.tensor(_QPN8_KORDER16, dtype=torch.long, device=dev)
    k_perm = torch.cat([korder16, korder16 + 16], dim=0)  # [32] -> (half 0: 0..15, half 1: 16..31)

    if is_w13:
        inter = N // 2
        row_order = torch.cat(
            [
                torch.arange(0, N, 2, dtype=torch.long, device=dev),
                torch.arange(1, N, 2, dtype=torch.long, device=dev),
            ],
            dim=0,
        )
        assert row_order.numel() == 2 * inter
    else:
        row_order = torch.arange(N, dtype=torch.long, device=dev)

    # Map (tile, lane) -> original row index in [0..N-1]
    tile_base = torch.arange(NT, dtype=torch.long, device=dev).view(NT, 1) * 32 + col_map.view(1, 32)
    row_gather = row_order[tile_base]  # [NT, 32]

    if inplace:
        assert w_raw.is_contiguous() and s_u8.is_contiguous(), "inplace prepack requires contiguous tensors"
        for e in range(E):
            we = w_raw[e]
            se = s_u8[e]
            # [NT, 32, K] -> [NT, 32, NC, 32] -> permute k -> [NT, NC, 2, 32, 16]
            wt = we.index_select(0, row_gather.reshape(-1)).view(NT, 32, NC, 32)
            wt = wt.index_select(3, k_perm).view(NT, 32, NC, 2, 16).permute(0, 2, 3, 1, 4).contiguous()
            w_raw[e].view(-1).copy_(wt.view(-1))

            st = se.index_select(0, row_gather.reshape(-1)).view(NT, 32, NC).permute(0, 2, 1).contiguous()
            if not wide_scale and eff_shifts[e] != 0:
                st = torch.where(st == 255, st, (st.to(torch.int32) + eff_shifts[e]).to(torch.uint8))
            s_u8[e].view(-1).copy_(st.view(-1))
        return w_raw.view(E, NT, NC, 32, 32), s_u8.view(E, NT, NC, 32), g_scale, wide_scale

    w_qpn = torch.empty((E, NT, NC, 32, 32), dtype=torch.uint8, device=dev)
    s_qpn = torch.empty((E, NT, NC, 32), dtype=torch.uint8, device=dev)
    for e in range(E):
        we = w_raw[e]
        se = s_u8[e]
        wt = we.index_select(0, row_gather.reshape(-1)).view(NT, 32, NC, 32)
        wt = wt.index_select(3, k_perm).view(NT, 32, NC, 2, 16).permute(0, 2, 3, 1, 4).contiguous()
        w_qpn[e].copy_(wt.view(NT, NC, 32, 32))

        st = se.index_select(0, row_gather.reshape(-1)).view(NT, 32, NC).permute(0, 2, 1).contiguous()
        if not wide_scale and eff_shifts[e] != 0:
            st = torch.where(st == 255, st, (st.to(torch.int32) + eff_shifts[e]).to(torch.uint8))
        s_qpn[e].copy_(st)
    return w_qpn, s_qpn, g_scale, wide_scale


def mxfp8_moe_w13_situ(
    x: torch.Tensor,
    w13_qpn: torch.Tensor,
    s13_qpn: torch.Tensor,
    g13_scale: torch.Tensor,
    perm: torch.Tensor,
    gids: torch.Tensor,
    goff: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    gate_cap: float,
    up_cap: float,
    *,
    out: torch.Tensor | None = None,
    wide_scale: bool = False,
    cfg: int = -1,
    max_groups: int = 0,
) -> torch.Tensor:
    """Zero-sync grouped MXFP8 w13 + fused exact FP32 softcapped SiTU -> `[T*topk, I]` fp16."""
    ext = load_mxfp8_moe_ext()
    num_tokens, hidden = x.shape
    num_slots = num_tokens * topk
    if out is None:
        if w13_qpn.ndim == 5:
            inter = (w13_qpn.shape[1] * 32) // 2
        else:
            inter = w13_qpn.shape[1] // 2
        out = torch.empty((num_slots, inter), dtype=torch.float16, device=x.device)
    if num_tokens == 0:
        return out
    ext.mxfp8_moe_w13_situ(
        x.contiguous(),
        w13_qpn.view(torch.uint8) if w13_qpn.dtype == torch.float8_e4m3fn else w13_qpn,
        s13_qpn,
        g13_scale,
        perm,
        gids,
        goff,
        weights,
        out,
        int(topk),
        float(gate_cap),
        float(up_cap),
        bool(wide_scale),
        int(cfg),
        int(max_groups),
    )
    return out


def mxfp8_moe_w2(
    inter_act: torch.Tensor,
    w2_qpn: torch.Tensor,
    s2_qpn: torch.Tensor,
    g2_scale: torch.Tensor,
    perm: torch.Tensor,
    gids: torch.Tensor,
    goff: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    wide_scale: bool = False,
    cfg: int = -1,
    max_groups: int = 0,
    num_tokens_hint: int = 0,
) -> torch.Tensor:
    """Zero-sync grouped MXFP8 w2 down projection -> `[T*topk, H]` fp16."""
    ext = load_mxfp8_moe_ext()
    num_slots = inter_act.shape[0]
    if out is None:
        if w2_qpn.ndim == 5:
            hidden = w2_qpn.shape[1] * 32
        else:
            hidden = w2_qpn.shape[1]
        out = torch.empty((num_slots, hidden), dtype=torch.float16, device=inter_act.device)
    if num_slots == 0:
        return out
    ext.mxfp8_moe_w2(
        inter_act.contiguous(),
        w2_qpn.view(torch.uint8) if w2_qpn.dtype == torch.float8_e4m3fn else w2_qpn,
        s2_qpn,
        g2_scale,
        perm,
        gids,
        goff,
        out,
        bool(wide_scale),
        int(cfg),
        int(max_groups),
        int(num_tokens_hint),
    )
    return out


def run_mxfp8_experts_cuda(
    x: torch.Tensor,
    w13_qpn: torch.Tensor,
    s13_qpn: torch.Tensor,
    g13_scale: torch.Tensor,
    w2_qpn: torch.Tensor,
    s2_qpn: torch.Tensor,
    g2_scale: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    *,
    wide_scale_13: bool = False,
    wide_scale_2: bool = False,
    cfg13: int = -1,
    cfg2: int = -1,
    precomputed_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Zero-sync, CUDA-graph-safe EP=4 MXFP8 routed experts (Phase 1 Volta sm_70 CUDA)."""
    num_tokens, hidden = x.shape
    topk = ids.shape[1]
    num_local = w13_qpn.shape[0]
    inter = (w13_qpn.shape[1] * 32) // 2 if w13_qpn.ndim == 5 else w13_qpn.shape[1] // 2
    out_dim = w2_qpn.shape[1] * 32 if w2_qpn.ndim == 5 else w2_qpn.shape[1]
    assert x.dtype == torch.float16
    down_all = torch.empty((num_tokens * topk, out_dim), dtype=torch.float16, device=x.device)
    if num_tokens == 0:
        return down_all

    # Bound transient inter_act memory for very large prefill batches while staying zero-sync.
    bytes_per_token = max(topk * 2 * inter, 1)
    chunk = max(1, (128 << 20) // bytes_per_token)
    for t0 in range(0, num_tokens, chunk):
        t1 = min(t0 + chunk, num_tokens)
        tc = t1 - t0
        xc = x[t0:t1].contiguous()
        ids_c = ids[t0:t1]
        w_c = weights[t0 * topk : t1 * topk].contiguous()
        if precomputed_route is not None and t0 == 0 and t1 == num_tokens:
            perm, gids32, goff32 = precomputed_route
        else:
            perm, gids32, goff32 = ep4_route_to_skinny_groups(ids_c, num_local)
        inter_act = mxfp8_moe_w13_situ(
            xc,
            w13_qpn,
            s13_qpn,
            g13_scale,
            perm,
            gids32,
            goff32,
            w_c,
            topk,
            gate_cap,
            up_cap,
            wide_scale=wide_scale_13,
            cfg=cfg13,
        )
        y2 = down_all[t0 * topk : t1 * topk]
        mxfp8_moe_w2(
            inter_act,
            w2_qpn,
            s2_qpn,
            g2_scale,
            perm,
            gids32,
            goff32,
            out=y2,
            wide_scale=wide_scale_2,
            cfg=cfg2,
            num_tokens_hint=tc,
        )
    return down_all
