# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 3-head-shared WMMA prefill attention loader & wrapper with on-the-fly R @ rel_proj.

Backed by `csrc/attn_rel_pf_wmma.cu` (WMMA m16n16k16, 3 query heads sharing each staged
KV tile, separate paged FP16 / FP8 E4M3FN KV caches, on-the-fly Tensor-Core relative bias
`R @ rel_proj`, fused `gate_sink` epilogue).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any

import torch

from vllm.triton_utils import tl, triton

from .norm import _l2_clip

_EXT: Any = None
_VARIANT = int(os.environ.get("INKLING_PF_WMMA_VARIANT", "0"))


def _ext_dir() -> pathlib.Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
    else:
        p = pathlib.Path(__file__).resolve().parents[1] / "_ext"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_attn_rel_pf_wmma_ext(verbose: bool = False) -> Any:
    """Build (if needed) and import `inkling_attn_rel_pf_wmma` from `csrc/attn_rel_pf_wmma.cu`."""
    global _EXT
    if _EXT is not None:
        return _EXT

    csrc_path = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "attn_rel_pf_wmma.cu"
    build_dir = _ext_dir() / "attn_rel_pf_wmma"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "inkling_attn_rel_pf_wmma.so"

    rebuild = os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc_path.stat().st_mtime:
        spec = importlib.util.spec_from_file_location("inkling_attn_rel_pf_wmma", str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from torch.utils.cpp_extension import load

    cuda_flags = [
        "-O3",
        "-gencode=arch=compute_70,code=sm_70",
        "-std=c++17",
        "--ptxas-options=-v",
    ]
    _EXT = load(
        name="inkling_attn_rel_pf_wmma",
        sources=[str(csrc_path)],
        build_directory=str(build_dir),
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXT


def supports_prefill_wmma(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    dtype: torch.dtype,
) -> bool:
    """Return True if the layer geometry and dtype are supported by `attn_rel_pf_wmma`."""
    return (
        num_kv_heads > 0
        and num_heads == 3 * num_kv_heads
        and head_dim == 128
        and d_rel == 16
        and dtype == torch.float16
    )


@triton.jit
def _qr_clip_kernel(
    Q,
    R,
    QScale,
    RScale,
    Tau,
    QOut,
    ROut,
    stride_q,
    stride_r,
    q_threshold,
    r_threshold,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    q_cols = tl.arange(0, D)[None, :]
    r_cols = tl.arange(0, D_REL)[None, :]
    tau = tl.load(Tau + t) if HAS_TAU else 1.0
    q = _l2_clip(
        Q + t * stride_q, QScale, heads, q_cols, h_mask, q_threshold, eps, tau, HAS_TAU
    )
    tl.store(QOut + t * H * D + heads * D + q_cols, q, h_mask)
    r = _l2_clip(
        R + t * stride_r, RScale, heads, r_cols, h_mask, r_threshold, eps, tau, HAS_TAU
    )
    tl.store(ROut + t * H * D_REL + heads * D_REL + r_cols, r, h_mask)


def clip_qr(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    *,
    clip_threshold: float,
    rel_threshold: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-clip `q * q_scale` `(T, H, D)` and `r * r_scale` `(T, H, d_rel)` into contiguous outputs.

    Matches `qr_clip_rel_tokens` in `norm.py` (without materializing sheared `rel_logits`),
    rounding to `q.dtype`/`r.dtype` and scaling by per-token `tau` when provided.
    """
    num_tokens, num_heads, head_dim = q.shape
    d_rel = r.shape[-1]
    assert q.dim() == 3 and r.dim() == 3
    assert r.shape[0] == num_tokens and r.shape[1] == num_heads
    assert q.stride(2) == 1 and q.stride(1) == head_dim
    assert r.stride(2) == 1 and r.stride(1) == d_rel
    assert q_scale.is_contiguous() and r_scale.is_contiguous()
    if tau is not None:
        assert tau.is_contiguous() and tau.numel() >= num_tokens

    q_out = torch.empty((num_tokens, num_heads, head_dim), dtype=q.dtype, device=q.device)
    r_out = torch.empty((num_tokens, num_heads, d_rel), dtype=r.dtype, device=r.device)
    if num_tokens:
        has_tau = tau is not None
        # Pass a valid tensor placeholder when HAS_TAU=False so Triton never receives None.
        tau_arg = tau if has_tau else q_scale
        _qr_clip_kernel[(num_tokens,)](
            q,
            r,
            q_scale,
            r_scale,
            tau_arg,
            q_out,
            r_out,
            q.stride(0),
            r.stride(0),
            clip_threshold,
            rel_threshold,
            eps,
            H=num_heads,
            D=head_dim,
            D_REL=d_rel,
            HAS_TAU=has_tau,
            BLOCK_H=triton.next_power_of_2(num_heads),
            num_warps=4,
        )
    return q_out, r_out


def rel_prefill_attention_wmma_cuda(
    q: torch.Tensor,
    r: torch.Tensor,
    rel_proj: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_actual_tokens: int | None = None,
    variant: int | None = None,
) -> None:
    """Volta sm_70 3-head-shared WMMA prefill attention with on-the-fly `R @ rel_proj`."""
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return

    assert head_dim == 128, f"attn_rel_pf_wmma requires head_dim=128, got {head_dim}"
    assert r.shape == (num_tokens, num_heads, 16), (
        f"attn_rel_pf_wmma requires r shape {(num_tokens, num_heads, 16)}, got {tuple(r.shape)}"
    )
    assert rel_proj.is_contiguous() and rel_proj.dim() == 2 and rel_proj.shape[0] == 16, (
        f"attn_rel_pf_wmma requires contiguous rel_proj [16, E], got {tuple(rel_proj.shape)}"
    )
    extent = rel_proj.shape[1]
    assert extent >= 128 and extent % 16 == 0, (
        f"attn_rel_pf_wmma requires rel_proj extent >= 128 and multiple of 16, got {extent}"
    )
    assert key_cache.dim() == 4 and value_cache.dim() == 4 and key_cache.shape == value_cache.shape
    assert key_cache.shape[2] > 0 and num_heads == 3 * key_cache.shape[2], (
        f"attn_rel_pf_wmma requires num_heads == 3 * num_kv_heads, "
        f"got num_heads={num_heads}, num_kv_heads={key_cache.shape[2]}"
    )

    is_fp8_kv = key_cache.element_size() == 1
    kc_arg = key_cache.view(torch.uint8) if is_fp8_kv else key_cache
    vc_arg = value_cache.view(torch.uint8) if is_fp8_kv else value_cache
    act_tokens = num_tokens if num_actual_tokens is None else int(num_actual_tokens)
    var = _VARIANT if variant is None else int(variant)

    ext = load_attn_rel_pf_wmma_ext()
    ext.attn_rel_pf_wmma_forward(
        q,
        r,
        rel_proj,
        kc_arg,
        vc_arg,
        gate,
        sinks,
        out,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        int(max_query_len),
        int(act_tokens),
        float(softmax_scale),
        int(window),
        int(var),
    )
