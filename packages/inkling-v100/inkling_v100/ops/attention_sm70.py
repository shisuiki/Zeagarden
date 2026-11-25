# SPDX-License-Identifier: Apache-2.0
"""Inkling2 relative-bias paged attention for Volta sm_70 (Triton, FP16 & uint8-backed FP8 KV)."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton

from .fp8_sm70 import e4m3_u8_to_f16

_SHEAR_TILE = tl.constexpr(128)
_FA4_BLOCK_N = 128
_VERIFY_BLOCK_M = 64

MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _rel_attention_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    Gate,
    Sink,
    softmax_scale,
    num_tokens,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    DRAFT: tl.constexpr,
    IS_FP8_KV: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // num_splits
    split = tl.program_id(1) % num_splits
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    if DRAFT:
        pos = tl.zeros([BLOCK_M], tl.int32) + seqlen_k - 1
        pos_lo = seqlen_k - 1
        pos_hi = seqlen_k - 1
    else:
        pos = past + tok
        pos_lo = past + first_tok
        pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1

    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    if not DRAFT:
        tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h
    for i in range(num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if IS_FP8_KV:
            k_u8 = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0,
            )
            k = e4m3_u8_to_f16(k_u8)
        else:
            k = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0.0,
            ).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        if WINDOW > 0:
            visible &= dist < WINDOW
        if not DRAFT:
            in_extent = visible & (dist < E) & row_valid[:, None]
            bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
            s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if IS_FP8_KV:
            v_u8 = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0,
            )
            v = e4m3_u8_to_f16(v_u8)
        else:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if MODE == 0:
        tl.store(
            Out
            + token[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :],
            o,
            mask=row_valid[:, None],
        )
        tl.store(
            Lse + head * stride_lse_h + token * stride_lse_t,
            lse2 * 0.6931471805599453,
            mask=row_valid,
        )
    elif MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        if not DRAFT:
            gate = tl.load(
                Gate + token[:, None] * stride_gate + cols,
                mask=row_valid[:, None],
                other=0.0,
            )
            sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
            o = o * _sigmoid(gate.to(tl.float32))
            o = o * _sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * stride_out_t + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


@triton.jit
def _rel_decode_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    num_tokens,
    num_actual_tokens,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GATED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    y_ptrs = Y + row * H * D + head * D + offs_d
    if row >= num_actual_tokens:
        tl.store(y_ptrs, tl.zeros([BLOCK_D], tl.float32).to(Y.dtype.element_ty))
        return
    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = tl.exp2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    if GATED:
        gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
        sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)
        lse = (tl.log2(tl.where(w_sum == 0.0, 1.0, w_sum)) + lse_max) * 0.6931471805599453
        o = o * _sigmoid(gate)
        o = o * _sigmoid(lse - sink)
    tl.store(y_ptrs, o.to(Y.dtype.element_ty))


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    stride_x,
    stride_gate,
    stride_lse_t,
    stride_lse_h,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * stride_x + col, mask, other=0.0).to(tl.float32)
    gate = tl.load(Gate + row * stride_gate + col, mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + row * stride_lse_t + head * stride_lse_h, mask, other=0.0).to(tl.float32)
    sink = tl.load(Sink + row * stride_sink + head, mask, other=0.0).to(tl.float32)
    result = x * _sigmoid(gate)
    result = result * _sigmoid(lse - sink)
    tl.store(Y + row * H * D + col, result.to(Y.dtype.element_ty), mask)


def gate_sink(
    output: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
) -> None:
    num_tokens, num_heads = sinks.shape
    if num_tokens == 0:
        return
    width = gate.shape[1]
    _gate_sink_kernel[(num_tokens, triton.cdiv(width, 512))](
        output,
        gate,
        lse,
        sinks,
        result,
        output.stride(0),
        gate.stride(0),
        lse.stride(0),
        lse.stride(1),
        sinks.stride(0),
        num_heads,
        width // num_heads,
        512,
        enable_fp_fusion=False,
    )


@triton.jit
def _log_scaling_tau_kernel(
    Pos, Out, n, FLOOR: tl.constexpr, ALPHA: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tldevice.log(ratio)
    tl.store(Out + i, tau, i < n)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    n = positions.numel()
    if n:
        _log_scaling_tau_kernel[(triton.cdiv(n, 256),)](
            positions, output, n, n_floor, alpha, 256, enable_fp_fusion=False
        )
    return output


def plan_decode_attention(
    is_local: bool,
    num_reqs: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
    num_sms: int = 80,
) -> int:
    if max_query_len > _VERIFY_BLOCK_M // (num_heads // num_kv_heads):
        return 0
    work = num_reqs * num_kv_heads
    if is_local:
        if work <= (16 if max_query_len <= 2 else 8):
            return min(8, triton.cdiv(rel_extent - 1, _FA4_BLOCK_N) + 1)
        return 4 if work <= 16 else 2 if work <= 32 else 1
    return max(1, min(32, max(num_sms // max(work, 1), min(num_sms // 4, 4 * num_sms // max(work, 1)))))


def _block_m_sm70(max_query_len: int, group: int) -> int:
    rows = max_query_len * group
    return 16 if rows <= 16 else 32 if rows <= 32 else 64


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
) -> None:
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    is_fp8_kv = key_cache.element_size() == 1
    kc_arg = key_cache.view(torch.uint8) if is_fp8_kv else key_cache
    vc_arg = value_cache.view(torch.uint8) if is_fp8_kv else value_cache
    assert kc_arg.stride() == vc_arg.stride() and kc_arg.stride(3) == 1
    assert q.stride(2) == 1
    assert mode != MODE_GATED or num_splits == 1
    draft = rel_logits is None
    extent = 0 if rel_logits is None else rel_logits.shape[2] - 256
    rel = q if rel_logits is None else rel_logits
    gate_arg = q if gate is None else gate
    sinks_arg = q if sinks is None else sinks
    lse_arg = out if lse is None else lse

    block_m = _block_m_sm70(max_query_len, group)
    block_n = 64
    tokens_per_tile = block_m // group
    num_reqs = cache_seqlens.shape[0]
    grid = (
        triton.cdiv(max_query_len, tokens_per_tile),
        num_kv_heads * num_splits,
        num_reqs,
    )
    if mode == MODE_OUT_LSE:
        out_strides = (out.stride(0), out.stride(1), lse_arg.stride(0), lse_arg.stride(1))
    elif mode == MODE_GATED:
        assert out.is_contiguous()
        out_strides = (out.stride(0), 0, 0, 0)
    else:
        assert out.is_contiguous() and lse_arg.is_contiguous()
        out_strides = (0, 0, 0, 0)
    _rel_attention_kernel[grid](
        q,
        kc_arg,
        vc_arg,
        rel,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse_arg,
        gate_arg,
        sinks_arg,
        softmax_scale,
        num_tokens,
        num_splits,
        q.stride(0),
        q.stride(1),
        kc_arg.stride(0),
        kc_arg.stride(1),
        kc_arg.stride(2),
        rel.stride(0),
        block_table.stride(0),
        *out_strides,
        gate_arg.stride(0),
        sinks_arg.stride(0),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW=window,
        DRAFT=draft,
        IS_FP8_KV=is_fp8_kv,
        MODE=mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1 if block_m >= 64 else 2,
    )


def rel_decode_attention(
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
) -> None:
    num_tokens, num_heads, head_dim = q.shape
    extent = rel_logits.shape[2] - 256
    window = extent if is_local else 0
    fuse_gate = num_splits == 1
    if fuse_gate:
        rel_attention(
            q,
            key_cache,
            value_cache,
            out,
            None,
            rel_logits,
            gate,
            sinks,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max_query_len,
            softmax_scale=softmax_scale,
            window=window,
            num_splits=1,
            mode=MODE_GATED,
        )
        if num_actual_tokens < num_tokens:
            out[num_actual_tokens:].zero_()
        return

    o_part = q.new_empty((num_splits, num_tokens, num_heads, head_dim), dtype=torch.float32)
    lse_part = q.new_empty((num_splits, num_tokens, num_heads), dtype=torch.float32)
    rel_attention(
        q,
        key_cache,
        value_cache,
        o_part,
        lse_part,
        rel_logits,
        None,
        None,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max_query_len,
        softmax_scale=softmax_scale,
        window=window,
        num_splits=num_splits,
        mode=MODE_PARTIAL,
    )
    block_d = head_dim if num_tokens * num_heads >= 512 else 32
    num_warps = (2 if num_splits > 32 else 1) if block_d == head_dim else 4
    _rel_decode_combine_kernel[(num_tokens, num_heads, head_dim // block_d)](
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        num_tokens,
        num_actual_tokens,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
        GATED=True,
        num_warps=num_warps,
    )
