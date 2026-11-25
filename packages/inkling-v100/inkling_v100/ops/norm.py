# SPDX-License-Identifier: Apache-2.0
"""Inkling2 row-wise kernels for Volta sm_70: RMSNorm and q/r L2 clipping + relative logits."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def rsqrt(x):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    stride_x,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask, other=0.0).to(tl.float32)
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), N) + eps)
    y = x * rstd
    y = y * tl.load(W + cols, mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dim of a row-major (possibly row-strided) tensor."""
    n = x.shape[-1]
    assert x.stride(-1) == 1
    x2d = x if x.dim() == 2 else x.reshape(-1, n)
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    if x2d.shape[0]:
        block = triton.next_power_of_2(n)
        _rms_norm_kernel[(x2d.shape[0],)](
            x2d,
            weight,
            out,
            x2d.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=min(max(block // 512, 1), 8),
        )
    return out


@triton.jit
def _l2_clip(X, Scale, heads, cols, mask, threshold, eps, tau, HAS_TAU):
    x = tl.load(X + heads * cols.shape[1] + cols, mask, other=0.0).to(tl.float32)
    x = x * tl.load(Scale + heads * cols.shape[1] + cols, mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=1)
    norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
    y = x * tl.minimum(tl.div_rn(threshold, norm + eps), 1.0)[:, None]
    if HAS_TAU:
        y = y.to(X.dtype.element_ty).to(tl.float32) * tau
    return y.to(X.dtype.element_ty)


@triton.jit
def qr_clip_rel_tokens(
    t0,
    num_t: tl.constexpr,
    chunk,
    T,
    Q,
    R,
    QScale,
    RScale,
    Tau,
    RelProj,
    QOut,
    RelOut,
    SeqIdx,
    CuSeqlensQ,
    SeqLens,
    stride_q,
    stride_r,
    q_threshold,
    r_threshold,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    E_PAD: tl.constexpr = E + 256
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    q_cols = tl.arange(0, D)[None, :]
    r_cols = tl.arange(0, D_REL)[None, :]
    e_idx = tl.arange(0, BLOCK_E)[None, :]
    for i in tl.static_range(num_t):
        t = (t0 + i).to(tl.int64)
        if t < T and chunk < 0:
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            q = _l2_clip(
                Q + t * stride_q,
                QScale,
                heads,
                q_cols,
                h_mask,
                q_threshold,
                eps,
                tau,
                HAS_TAU,
            )
            tl.store(QOut + t * H * D + heads * D + q_cols, q, h_mask)
        elif t < T:
            seq = tl.load(SeqIdx + t)
            q_start = tl.load(CuSeqlensQ + seq)
            q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
            q_idx = t - q_start
            pos = tl.load(SeqLens + seq) - q_len + q_idx
            tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
            diag = pos + E_PAD - tl.cdiv(pos - q_idx + tile_end, 128) * 128
            diag = tl.minimum(tl.maximum(diag, E + 1), E_PAD - 1)
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            r = _l2_clip(
                R + t * stride_r,
                RScale,
                heads,
                r_cols,
                h_mask,
                r_threshold,
                eps,
                tau,
                HAS_TAU,
            )
            diag8 = diag // 8 * 8
            table = RelProj + (diag - diag8) * (D_REL + 1) * W + E + 256 - diag8
            col = chunk * BLOCK_E + e_idx
            proj = tl.load(table + tl.trans(r_cols) * W + col)
            pad = tl.load(table + D_REL * W + col)
            rel = tl.dot(r, proj) + pad.to(tl.float32)
            tl.store(
                RelOut + t * H * E_PAD + heads * E_PAD + col,
                rel.to(RelOut.dtype.element_ty),
                h_mask,
            )
