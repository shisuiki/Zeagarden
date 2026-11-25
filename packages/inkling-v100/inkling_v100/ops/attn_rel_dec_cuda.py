# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 CUDA extension loader & wrapper for Inkling2 relative-bias decode/verify attention.

Provides a drop-in replacement for `rel_decode_attention` in `attention_sm70.py` backed by
`csrc/attn_rel_dec.cu` (zero-spill HMMA.884, FP8 E4M3FN & FP16 paged KV, fused `gate_sink`,
CUDA-graph-safe persistent split-KV workspace).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
from typing import Any

import torch

log = logging.getLogger(__name__)

_EXT: Any = None
_WORKSPACES: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_DEFAULT_SLOTS = 32768  # 32768 * (128 + 2) * 4 B ~= 16.25 MB per device
_EMPTY_F32_BY_DEV: dict[tuple[str, int | None], torch.Tensor] = {}


def _ext_dir() -> pathlib.Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
    else:
        p = pathlib.Path(__file__).resolve().parents[1] / "_ext"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_attn_rel_dec_ext(verbose: bool = False) -> Any:
    """Build (if needed) and import `inkling_attn_rel_dec` from `csrc/attn_rel_dec.cu`."""
    global _EXT
    if _EXT is not None:
        return _EXT

    csrc_path = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "attn_rel_dec.cu"
    build_dir = _ext_dir() / "attn_rel_dec"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "inkling_attn_rel_dec.so"

    rebuild = os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc_path.stat().st_mtime:
        spec = importlib.util.spec_from_file_location("inkling_attn_rel_dec", str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    from torch.utils.cpp_extension import load

    cuda_flags = [
        "-O3",
        "-arch=sm_70",
        "-std=c++17",
        "--ptxas-options=-v",
    ]
    _EXT = load(
        name="inkling_attn_rel_dec",
        sources=[str(csrc_path)],
        build_directory=str(build_dir),
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXT


def preallocate_workspace(
    slots: int = _DEFAULT_SLOTS,
    device: torch.device | str | int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-allocate persistent split-KV workspace `(o_part, ml_part)` outside CUDA graph capture."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (dev.type, dev.index)
    cur = _WORKSPACES.get(key)
    needed_o = slots * 128
    needed_ml = slots * 2
    if cur is not None and cur[0].numel() >= needed_o and cur[1].numel() >= needed_ml:
        return cur
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"attn_rel_dec_cuda: split-KV workspace ({slots} slots) cannot be allocated inside "
            f"CUDA graph capture; call preallocate_workspace(slots>={slots}) before capture."
        )
    alloc_slots = max(slots, _DEFAULT_SLOTS)
    o_part = torch.empty((alloc_slots * 128,), dtype=torch.float32, device=dev)
    ml_part = torch.empty((alloc_slots * 2,), dtype=torch.float32, device=dev)
    _WORKSPACES[key] = (o_part, ml_part)
    return o_part, ml_part


def _get_empty_f32(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    t = _EMPTY_F32_BY_DEV.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            # Zero-element tensor does not allocate device memory
            return torch.empty((0,), dtype=torch.float32, device=device)
        t = torch.empty((0,), dtype=torch.float32, device=device)
        _EMPTY_F32_BY_DEV[key] = t
    return t


def plan_rel_decode_splits(
    is_local: bool,
    num_reqs: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int = 1024,
    num_sms: int = 80,
) -> int:
    """Choose wave-aligned split-KV count for Volta sm_70 `attn_rel_dec.cu` (80 SMs)."""
    work = max(1, num_reqs * num_kv_heads)
    group = max(1, num_heads // max(1, num_kv_heads))
    max_nr = max(1, max_query_len * group)
    if is_local:
        # SWA-1024 = 32 tiles of 32 tokens. Keep CTAs wave-aligned on 80 SMs (avoid tail waves).
        max_tiles = max(1, (rel_extent + 31) // 32)
        if work <= 4:
            target = 16 if max_nr > 8 else 8
            return min(target, max(1, max_tiles // 2))
        if work <= 16:
            return min(8, max(1, max_tiles // 4))
        if work <= 64:
            return min(2, max(1, max_tiles // 8))
        return 1
    # Global attention: keep S <= 32 (fast single-pass warp-register combine) and align to 80 SMs
    # (128-160 CTAs = 1.6-2 CTAs/SM, avoiding 240+ tail-wave quantization).
    if work <= 2:
        return 32
    if work <= 4:
        return 20
    if work <= 16:
        return 10
    if work <= 32:
        return 5
    if work <= 64:
        return 2
    return 1


def rel_decode_attention_cuda(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    is_local: bool,
    num_splits: int,
    window: int | None = None,
    t_warps_override: int = 0,
) -> None:
    """Volta sm_70 CUDA relative-bias GQA decode & verify attention with fused `gate_sink`."""
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    assert head_dim == 128, f"attn_rel_dec_cuda requires head_dim=128, got {head_dim}"

    extent = rel_logits.shape[2] - 256
    eff_window = (extent if is_local else 0) if window is None else window

    is_fp8_kv = key_cache.element_size() == 1
    kc_arg = key_cache.view(torch.uint8) if is_fp8_kv else key_cache
    vc_arg = value_cache.view(torch.uint8) if is_fp8_kv else value_cache

    if num_splits <= 0:
        num_splits = plan_rel_decode_splits(
            is_local=is_local,
            num_reqs=int(cache_seqlens.shape[0]),
            max_query_len=max_query_len,
            num_heads=num_heads,
            num_kv_heads=int(key_cache.shape[2]),
            rel_extent=extent,
        )

    ext = load_attn_rel_dec_ext()
    if num_splits > 1:
        slots = num_tokens * num_heads * num_splits
        o_part, ml_part = preallocate_workspace(slots=slots, device=q.device)
    else:
        empty = _get_empty_f32(q.device)
        o_part, ml_part = empty, empty

    ext.attn_rel_dec_forward(
        q,
        kc_arg,
        vc_arg,
        rel_logits,
        gate,
        sinks,
        out,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        o_part,
        ml_part,
        int(max_query_len),
        int(num_actual_tokens),
        float(softmax_scale),
        int(eff_window),
        int(num_splits),
        int(t_warps_override),
    )
