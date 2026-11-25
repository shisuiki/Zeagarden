# SPDX-License-Identifier: Apache-2.0
"""Volta (sm_70) FP16/FP32 Triton kernels for Inkling2 DSpark speculative drafting.

Eliminates all bf16 dependencies and adds Volta-optimized TP-local Markov argmax
(reduces O(201K) vocab all-gather per step to 8 bytes/req across TP=4 NVLink).
"""

from __future__ import annotations

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _mix_rmsnorm_kernel(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mask = h < HIDDEN
    mixed = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, mask=mask, other=0.0).to(tl.float32)
    out = (mixed * inv_rms * weight).to(OUT.dtype.element_ty)
    tl.store(OUT + row * HIDDEN + h, out, mask=mask)


def mix_rmsnorm(
    targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm of the `alpha`-weighted sum of `targets[:, i]` in FP32, stored in `targets.dtype`."""
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    if rows:
        _mix_rmsnorm_kernel[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out


@triton.jit
def _qk_norm_rope_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.maximum(tl.load(POS + t).to(tl.int64), 0)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
    sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)

    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    row = X + t * row_stride + col0
    x1 = tl.load(row + half_ar).to(tl.float32)
    x2 = tl.load(row + HALF + half_ar).to(tl.float32)
    if NORM:
        ms = (tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D
        inv = tl.rsqrt(ms + EPS)
        if is_q:
            w1 = tl.load(Q_WEIGHT + half_ar).to(tl.float32)
            w2 = tl.load(Q_WEIGHT + HALF + half_ar).to(tl.float32)
        else:
            w1 = tl.load(K_WEIGHT + half_ar).to(tl.float32)
            w2 = tl.load(K_WEIGHT + HALF + half_ar).to(tl.float32)
        x1 = x1 * inv * w1
        x2 = x2 * inv * w2
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(row + half_ar, o1.to(X.dtype.element_ty))
    tl.store(row + HALF + half_ar, o2.to(X.dtype.element_ty))


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
) -> None:
    """In-place per-head Q/K RMSNorm + neox RoPE in FP32 on leading Q+K heads of `qkv`."""
    head_dim = cos_sin_cache.shape[-1]
    if qkv.shape[0]:
        _qk_norm_rope_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            qkv.stride(0),
            num_q_heads * head_dim,
            NHQ=num_q_heads,
            D=head_dim,
            EPS=eps,
            NORM=True,
            num_warps=2,
        )


def rope_(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> None:
    """In-place neox RoPE on `x` of shape `(T, H, D)` in FP32 math."""
    num_tokens, num_heads, head_dim = x.shape
    assert x.stride(1) == head_dim and x.stride(2) == 1
    if num_tokens:
        _qk_norm_rope_kernel[(num_tokens, num_heads)](
            x,
            x,
            x,
            cos_sin_cache,
            positions,
            x.stride(0),
            0,
            NHQ=0,
            D=head_dim,
            EPS=0.0,
            NORM=False,
            num_warps=2,
        )


@triton.jit
def _silu_and_mul_kernel(X, Y, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
    tl.store(Y + row * N + cols, (silu * up).to(Y.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up` in FP32, rounded once to `x.dtype`."""
    num_tokens, width = x.shape
    n = width // 2
    out = x.new_empty((num_tokens, n))
    if num_tokens:
        _silu_and_mul_kernel[(num_tokens, triton.cdiv(n, 1024))](
            x, out, x.stride(0), n, 1024, num_warps=4
        )
    return out


@triton.jit
def _add_softcap_kernel(A, B, OUT, stride_a, stride_b, softcap, N, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * stride_a + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    x = a + b
    if softcap > 0.0:
        x = tldevice.tanh(x / softcap) * softcap
    tl.store(OUT + row * N + cols, x.to(OUT.dtype.element_ty), mask=mask)


def add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float) -> torch.Tensor:
    """`softcap * tanh((a + b) / softcap)` in FP32, stored in `a.dtype`."""
    rows, n = a.shape
    assert a.stride(1) == 1 and b.stride(1) == 1
    out = a.new_empty((rows, n))
    if rows:
        _add_softcap_kernel[(triton.cdiv(n, 1024), rows)](
            a, b, out, a.stride(0), b.stride(0), float(softcap or 0.0), n, 1024, num_warps=4
        )
    return out


@triton.jit
def _markov_gate_kernel(RAW, STATE, OUT, LATENT, RANK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    raw = RAW + row * 3 * RANK + c
    gate_in = tl.load(raw).to(tl.float32)
    cand_in = tl.load(raw + RANK).to(tl.float32)
    out_in = tl.load(raw + 2 * RANK).to(tl.float32)
    state = tl.load(STATE + row * RANK + c).to(tl.float32)
    gate = tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate_in))
    cand = tldevice.tanh(cand_in)
    new_state = gate * state + (1.0 - gate) * cand
    latent = tldevice.tanh(out_in)
    tl.store(OUT + row * RANK + c, new_state.to(OUT.dtype.element_ty))
    tl.store(LATENT + row * RANK + c, latent.to(LATENT.dtype.element_ty))


def markov_gate(
    raw: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent Markov head gating in FP32 -> FP16 `(new_state, latent)`."""
    rows, rank = state.shape
    assert raw.shape == (rows, 3 * rank) and raw.is_contiguous()
    assert state.is_contiguous() and (rank & (rank - 1)) == 0
    out, latent = torch.empty_like(state), torch.empty_like(state)
    if rows:
        _markov_gate_kernel[(rows,)](raw, state, out, latent, rank, num_warps=4)
    return out, latent


@triton.jit
def _markov_local_argmax_stage1_kernel(
    A,
    B,
    PART_VAL,
    PART_IDX,
    stride_a,
    stride_b,
    local_n,
    vocab_offset,
    org_vocab_size,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    global_cols = vocab_offset + cols
    valid = (cols < local_n) & (global_cols < org_vocab_size)
    a = tl.load(A + row * stride_a + cols, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask=valid, other=0.0).to(tl.float32)
    val = a + b
    # Treat NaN and out-of-vocab columns as -inf (matching greedy_argmax semantics).
    is_ok = valid & (val == val)
    val = tl.where(is_ok, val, float("-inf"))
    best_val, best_local = tl.max(val, axis=0, return_indices=True)
    best_global = (vocab_offset + tile * BLOCK + best_local).to(tl.int32)
    # If all values in this tile are -inf, record max int32 so valid tiles win ties.
    best_global = tl.where(best_val == float("-inf"), 0x7FFFFFFF, best_global)
    out_off = row * NUM_TILES + tile
    tl.store(PART_VAL + out_off, best_val)
    tl.store(PART_IDX + out_off, best_global)


@triton.jit
def _markov_local_argmax_stage2_kernel(
    PART_VAL,
    PART_IDX,
    OUT_PACKED,  # [rows, 2] int32: col 0 = float32 bits of max_val, col 1 = global_idx
    NUM_TILES: tl.constexpr,
    PAD_TILES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, PAD_TILES)
    mask = t < NUM_TILES
    vals = tl.load(PART_VAL + row * NUM_TILES + t, mask=mask, other=float("-inf"))
    idxs = tl.load(PART_IDX + row * NUM_TILES + t, mask=mask, other=0x7FFFFFFF)
    max_val = tl.max(vals, axis=0)
    # Deterministic lowest global_idx tie-break.
    is_max = mask & (vals == max_val)
    best_idx = tl.min(tl.where(is_max, idxs, 0x7FFFFFFF), axis=0)
    best_idx = tl.where(best_idx == 0x7FFFFFFF, 0, best_idx)
    val_bits = max_val.to(tl.int32, bitcast=True)
    tl.store(OUT_PACKED + row * 2 + 0, val_bits)
    tl.store(OUT_PACKED + row * 2 + 1, best_idx)


@triton.jit
def _tp_reduce_argmax_kernel(
    GATHERED,  # [rows, TP_SIZE * 2] int32
    OUT_IDS,  # [rows] int64
    TP_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = GATHERED + row * (TP_SIZE * 2)
    best_val = float("-inf")
    best_idx = 0x7FFFFFFF
    for r in tl.static_range(TP_SIZE):
        val = tl.load(base + r * 2 + 0).to(tl.float32, bitcast=True)
        idx = tl.load(base + r * 2 + 1)
        better = (val > best_val) | ((val == best_val) & (idx < best_idx))
        best_val = tl.where(better, val, best_val)
        best_idx = tl.where(better, idx, best_idx)
    best_idx = tl.where(best_idx == 0x7FFFFFFF, 0, best_idx)
    tl.store(OUT_IDS + row, best_idx.to(tl.int64))


def tp_local_markov_argmax(
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    tp_rank: int,
    vocab_shard_size: int,
    org_vocab_size: int,
) -> torch.Tensor:
    """Exact greedy `argmax(softcap(base_logits + bias))` via TP-local reduction.

    Because `30 * tanh(x / 30)` is strictly increasing on R, argmax commutes with
    softcapping. Reducing locally per rank shrinks NVLink communication per Markov
    step from `[B, 201024]` FP16 (402 KB/req) to `[B, 2]` int32 (8 B/req) — a
    ~50,000x reduction in all-gather volume and zero tanh/full-vocab scans!
    """
    rows, local_n = base_logits.shape
    assert bias.shape == (rows, local_n)
    assert base_logits.stride(1) == 1 and bias.stride(1) == 1
    if rows == 0:
        return torch.empty((0,), dtype=torch.int64, device=base_logits.device)

    BLOCK = 1024
    num_tiles = triton.cdiv(local_n, BLOCK)
    pad_tiles = triton.next_power_of_2(num_tiles)
    part_val = torch.empty((rows, num_tiles), dtype=torch.float32, device=base_logits.device)
    part_idx = torch.empty((rows, num_tiles), dtype=torch.int32, device=base_logits.device)
    local_packed = torch.empty((rows, 2), dtype=torch.int32, device=base_logits.device)

    vocab_offset = tp_rank * vocab_shard_size
    _markov_local_argmax_stage1_kernel[(num_tiles, rows)](
        base_logits,
        bias,
        part_val,
        part_idx,
        base_logits.stride(0),
        bias.stride(0),
        local_n,
        vocab_offset,
        org_vocab_size,
        NUM_TILES=num_tiles,
        BLOCK=BLOCK,
        num_warps=4,
    )
    _markov_local_argmax_stage2_kernel[(rows,)](
        part_val,
        part_idx,
        local_packed,
        NUM_TILES=num_tiles,
        PAD_TILES=pad_tiles,
        num_warps=2,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return local_packed[:, 1].to(torch.int64)

    gathered = tensor_model_parallel_all_gather(local_packed, dim=-1)
    out_ids = torch.empty((rows,), dtype=torch.int64, device=base_logits.device)
    _tp_reduce_argmax_kernel[(rows,)](gathered, out_ids, TP_SIZE=tp_size, num_warps=1)
    return out_ids
