# SPDX-License-Identifier: Apache-2.0
"""Inkling2 short convolution & attention prologue kernels for Volta sm_70.

Supports FP16 and uint8-backed unit-scale FP8 E4M3FN KV caches without Hopper PDL.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from .fp8_sm70 import f32_to_e4m3_u8
from .norm import qr_clip_rel_tokens, rsqrt


@triton.jit
def _seq_metadata_kernel(
    qsl_ptr,
    seq_idx_ptr,
    query_start_ptr,
    num_reqs,
    num_actual_tokens,
    num_padded_tokens,
    n_iters,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tok = offs.to(tl.int32)
    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.full([BLOCK], num_reqs - 1, tl.int32)
    for _ in range(n_iters):
        mid = (lo + hi + 1) // 2
        below = tl.load(qsl_ptr + mid) <= tok
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    actual = offs < num_actual_tokens
    padded = offs < num_padded_tokens
    query_start = tl.load(qsl_ptr + lo)
    tl.store(seq_idx_ptr + offs, tl.where(actual, lo, 0), mask=padded)
    tl.store(
        query_start_ptr + offs,
        tl.where(actual, query_start, 0),
        mask=padded,
    )


def sconv_seq_metadata(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    seq_idx_out: torch.Tensor,
    query_start_out: torch.Tensor,
    num_padded_tokens: int | None = None,
) -> None:
    if num_padded_tokens is None:
        num_padded_tokens = num_actual_tokens
    if num_padded_tokens == 0:
        return
    BLOCK = 256
    n_iters = max(0, (num_reqs - 1).bit_length())
    grid = (triton.cdiv(num_padded_tokens, BLOCK),)
    _seq_metadata_kernel[grid](
        query_start_loc,
        seq_idx_out,
        query_start_out,
        num_reqs,
        num_actual_tokens,
        num_padded_tokens,
        n_iters,
        BLOCK=BLOCK,
    )


@triton.jit
def _ring_conv_meta_kernel(
    qsl_ptr,
    pos_ptr,
    bt_ptr,
    seq_idx_ptr,
    query_start_ptr,
    slot_mapping_ptr,
    num_reqs,
    num_actual_tokens,
    num_padded_tokens,
    n_iters,
    RING_N: tl.constexpr,
    MAX_INPLACE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tok = offs.to(tl.int32)
    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.full([BLOCK], num_reqs - 1, tl.int32)
    for _ in range(n_iters):
        mid = (lo + hi + 1) // 2
        below = tl.load(qsl_ptr + mid) <= tok
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    actual = offs < num_actual_tokens
    padded = offs < num_padded_tokens
    safe_r = tl.where(actual, lo, 0)
    qstart = tl.load(qsl_ptr + safe_r)
    qend = tl.load(qsl_ptr + safe_r + 1)
    req_len = qend - qstart
    req_slot = tl.load(bt_ptr + safe_r).to(tl.int64)
    pos = tl.load(pos_ptr + offs, mask=actual, other=0).to(tl.int64)
    inplace_slot = req_slot * RING_N + (pos & (RING_N - 1))
    # -1 => padded (invalid, no read/write); -2 => long prefill (valid read, deferred tail write)
    slot = tl.where(actual, tl.where(req_len <= MAX_INPLACE_LEN, inplace_slot, -2), -1)
    tl.store(seq_idx_ptr + offs, safe_r, mask=padded)
    tl.store(query_start_ptr + offs, tl.where(actual, qstart, 0), mask=padded)
    tl.store(slot_mapping_ptr + offs, slot, mask=padded)


def ring_conv_metadata(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table_1d: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    seq_idx_out: torch.Tensor,
    query_start_out: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    num_padded_tokens: int,
    ring_n: int = 16,
    max_inplace_len: int = 12,
) -> None:
    if num_padded_tokens == 0:
        return
    BLOCK = 256
    n_iters = max(0, (max(num_reqs, 1) - 1).bit_length())
    grid = (triton.cdiv(num_padded_tokens, BLOCK),)
    _ring_conv_meta_kernel[grid](
        query_start_loc,
        positions,
        block_table_1d,
        seq_idx_out,
        query_start_out,
        slot_mapping_out,
        max(num_reqs, 1),
        num_actual_tokens,
        num_padded_tokens,
        n_iters,
        RING_N=ring_n,
        MAX_INPLACE_LEN=max_inplace_len,
        BLOCK=BLOCK,
    )


@triton.jit
def _ring_commit_tail_kernel(
    x_ptr,
    stride_x,
    cache_ptr,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    pos_ptr,
    tail_toks_ptr,
    tail_slots_ptr,
    num_tail,
    RING_N: tl.constexpr,
    OFF: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    idx = tl.program_id(0)
    if idx >= num_tail:
        return
    coff = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    C = H * WS
    c_mask = coff < C
    src_tok = tl.load(tail_toks_ptr + idx).to(tl.int64)
    req_slot = tl.load(tail_slots_ptr + idx).to(tl.int64)
    pos = tl.load(pos_ptr + src_tok).to(tl.int64)
    ring_pos = pos & (RING_N - 1)
    val = tl.load(x_ptr + src_tok * stride_x + coff, mask=c_mask, other=0.0)
    head = tl.minimum(coff // WS, H - 1)
    col = OFF + (coff % WS)
    dst = (
        cache_ptr
        + req_slot * stride_c_blk
        + head * stride_c_h
        + ring_pos * stride_c_n
        + col
    )
    tl.store(dst, val.to(cache_ptr.dtype.element_ty), mask=c_mask)


def ring_commit_tail(
    x: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    tail_src_toks: torch.Tensor,
    tail_req_slots: torch.Tensor,
    num_tail_writes: int,
    stream_offset: int,
    stream_width: int,
) -> None:
    if num_tail_writes <= 0:
        return
    num_heads = cache.shape[1]
    ring_n = cache.shape[2]
    channels = num_heads * stream_width
    block_c = min(triton.next_power_of_2(channels), 512)
    grid = (num_tail_writes, triton.cdiv(channels, block_c))
    _ring_commit_tail_kernel[grid](
        x,
        x.stride(0),
        cache,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        positions,
        tail_src_toks,
        tail_req_slots,
        num_tail_writes,
        RING_N=ring_n,
        OFF=stream_offset,
        WS=stream_width,
        H=num_heads,
        BLOCK_C=block_c,
        num_warps=4,
    )


@triton.jit
def _cached_taps(
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    c_mask,
    block_table_ptr,
    stride_bt,
    max_blocks,
    req,
    src,
    cached,
):
    safe_src = tl.maximum(src, 0)
    lblk = tl.minimum(safe_src // N, max_blocks - 1)
    blk = tl.load(block_table_ptr + req * stride_bt + lblk, mask=cached, other=0)
    return tl.load(
        cache_ptr
        + blk.to(tl.int64)[:, None] * stride_c_blk
        + (safe_src % N)[:, None] * stride_c_n
        + cache_col[None, :],
        mask=c_mask[None, :] & cached[:, None],
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _tap_sum(
    x_ptr,
    stride_x,
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    weight_ptr,
    stride_w0,
    stride_w1,
    block_table_ptr,
    stride_bt,
    max_blocks,
    toff,
    coff,
    c_mask,
    tc_mask,
    pos,
    req,
    qstart,
    valid,
    cur,
    W: tl.constexpr,
    dyn_ptr,
    stride_dyn,
    HEAD_DIM: tl.constexpr,
    READ_CACHE: tl.constexpr = True,
):
    dtype = cache_ptr.dtype.element_ty
    acc = tl.zeros(cur.shape, dtype=tl.float32)
    for k in tl.static_range(W):
        lag = k if dyn_ptr is not None else W - 1 - k
        src = pos - lag
        row = toff - lag
        in_win = valid & (src >= 0)
        intra = in_win & (row >= qstart)
        cached = in_win & (row < qstart)
        xt = (
            tl.load(
                x_ptr + tl.maximum(row, 0)[:, None] * stride_x + coff[None, :],
                mask=c_mask[None, :] & intra[:, None],
                other=0.0,
            )
            .to(dtype)
            .to(tl.float32)
        )
        if READ_CACHE:
            xt += _cached_taps(
                cache_ptr,
                cache_col,
                stride_c_blk,
                stride_c_n,
                N,
                c_mask,
                block_table_ptr,
                stride_bt,
                max_blocks,
                req,
                src,
                cached,
            )
        value = xt
        if dyn_ptr is not None:
            w = tl.load(
                weight_ptr + lag * stride_w0 + coff * stride_w1, mask=c_mask, other=0.0
            ).to(tl.float32)
            dyn = tl.load(
                dyn_ptr
                + toff[:, None] * stride_dyn
                + (coff // HEAD_DIM)[None, :] * W
                + lag,
                mask=tc_mask,
                other=0.0,
            ).to(tl.float32)
            w = w[None, :] + dyn
        else:
            w = tl.load(
                weight_ptr + coff * stride_w0 + k * stride_w1, mask=c_mask, other=0.0
            )
            w = w.to(dtype).to(tl.float32)[None, :]
        acc = value * w if dyn_ptr is not None and k == 0 else tl.fma(value, w, acc)
    return acc


@triton.jit
def _conv(
    x_ptr,
    stride_x,
    cache_ptr,
    cache_col,
    stride_c_blk,
    stride_c_n,
    N,
    weight_ptr,
    stride_w0,
    stride_w1,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    stride_bt,
    max_blocks,
    qstart_ptr,
    scale,
    toff,
    coff,
    t_mask,
    c_mask,
    W: tl.constexpr,
    dyn_ptr=None,
    stride_dyn=0,
    HEAD_DIM: tl.constexpr = 1,
    SKIP_IDLE_CACHE: tl.constexpr = False,
    tile_reads_cache=True,
):
    tc_mask = t_mask[:, None] & c_mask[None, :]
    slot = tl.load(slot_ptr + toff, mask=t_mask, other=-1)
    valid = slot != -1
    write = slot >= 0
    pos = tl.load(pos_ptr + toff, mask=t_mask, other=0)
    req = tl.load(seq_idx_ptr + toff, mask=t_mask, other=0)
    qstart = tl.load(qstart_ptr + toff, mask=t_mask, other=0)

    dtype = cache_ptr.dtype.element_ty
    cur = tl.load(x_ptr + toff[:, None] * stride_x + coff[None, :], mask=tc_mask, other=0.0)
    cur = cur.to(dtype)
    safe_slot = tl.maximum(slot, 0)
    dst = (
        cache_ptr
        + (safe_slot // N)[:, None] * stride_c_blk
        + (safe_slot % N)[:, None] * stride_c_n
        + cache_col[None, :]
    )
    tl.store(dst, cur, mask=tc_mask & write[:, None])

    tap_args = (
        x_ptr,
        stride_x,
        cache_ptr,
        cache_col,
        stride_c_blk,
        stride_c_n,
        N,
        weight_ptr,
        stride_w0,
        stride_w1,
        block_table_ptr,
        stride_bt,
        max_blocks,
        toff,
        coff,
        c_mask,
        tc_mask,
        pos,
        req,
        qstart,
        valid,
        cur,
    )
    if not SKIP_IDLE_CACHE or tile_reads_cache:
        acc = _tap_sum(*tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM, READ_CACHE=True)
    else:
        acc = _tap_sum(*tap_args, W, dyn_ptr, stride_dyn, HEAD_DIM, READ_CACHE=False)
    return (acc + cur.to(tl.float32)) * scale


@triton.jit
def _sconv_kernel(
    x_ptr,
    cache_ptr,
    weight_ptr,
    out_ptr,
    res_ptr,
    addend_ptr,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_x,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_w0,
    stride_w1,
    stride_bt,
    max_blocks,
    N,
    scale,
    W: tl.constexpr,
    OFF: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    HAS_RES: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SKIP_IDLE_CACHE: tl.constexpr,
):
    first = tl.program_id(0) * BT
    tile_reads_cache = True
    if SKIP_IDLE_CACHE:
        last = tl.minimum(first + BT, T) - 1
        tile_reads_cache = tl.load(qstart_ptr + last) > first - (W - 1)
    toff = first + tl.arange(0, BT)
    coff = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    C = H * WS
    t_mask = toff < T
    c_mask = coff < C
    y = _conv(
        x_ptr,
        stride_x,
        cache_ptr,
        tl.minimum(coff // WS, H - 1) * stride_c_h + OFF + coff % WS,
        stride_c_blk,
        stride_c_n,
        N,
        weight_ptr,
        stride_w0,
        stride_w1,
        pos_ptr,
        seq_idx_ptr,
        slot_ptr,
        block_table_ptr,
        stride_bt,
        max_blocks,
        qstart_ptr,
        scale,
        toff,
        coff,
        t_mask,
        c_mask,
        W,
        SKIP_IDLE_CACHE=SKIP_IDLE_CACHE,
        tile_reads_cache=tile_reads_cache,
    )
    offs = toff[:, None] * C + coff[None, :]
    mask = t_mask[:, None] & c_mask[None, :]
    if HAS_ADDEND:
        acc = tl.load(addend_ptr + offs, mask, other=0.0).to(tl.float32)
        if HAS_RES:
            acc = tl.load(res_ptr + offs, mask, other=0.0).to(tl.float32) + acc
        y = acc + y
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    tail_src_toks: torch.Tensor | None = None,
    tail_req_slots: torch.Tensor | None = None,
    num_tail_writes: int = 0,
) -> torch.Tensor:
    """Static depthwise causal conv over paged conv cache; supports FP32 `out`/`addend`/`residual`."""
    num_tokens, channels = x.shape
    if out is None:
        out = torch.empty((num_tokens, channels), dtype=cache.dtype, device=x.device)
    if num_tokens == 0:
        return out
    assert x.stride(1) == 1 and cache.stride(3) == 1 and out.is_contiguous()
    assert residual is None or addend is not None
    for t in (addend, residual):
        assert t is None or (t.is_contiguous() and t.shape == out.shape)
    num_heads = cache.shape[1]
    assert channels == num_heads * stream_width
    width = weight.shape[1]
    prefill = num_tokens > 256
    if num_tokens <= 8:
        block_t, block_c = 1, 512
    else:
        block_t, block_c = (4, 256) if prefill else (2, 512)
    block_c = min(triton.next_power_of_2(channels), block_c)
    grid = (triton.cdiv(num_tokens, block_t), triton.cdiv(channels, block_c))
    _sconv_kernel[grid](
        x,
        cache,
        weight,
        out,
        residual if residual is not None else out,
        addend if addend is not None else out,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        x.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        weight.stride(0),
        weight.stride(1),
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width + 1) ** -0.5,
        W=width,
        OFF=stream_offset,
        WS=stream_width,
        H=num_heads,
        HAS_RES=residual is not None,
        HAS_ADDEND=addend is not None,
        BT=block_t,
        BLOCK_C=block_c,
        SKIP_IDLE_CACHE=prefill,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if num_tail_writes > 0 and tail_src_toks is not None and tail_req_slots is not None:
        ring_commit_tail(
            x,
            cache,
            positions,
            tail_src_toks,
            tail_req_slots,
            num_tail_writes,
            stream_offset,
            stream_width,
        )
    return out


_SHIFT_PAD = 512


def shift_rel_proj(rel_proj: torch.Tensor, is_local: bool) -> torch.Tensor:
    d_rel, extent = rel_proj.shape
    shift = torch.arange(8, device=rel_proj.device)[:, None]
    y = torch.arange(extent + _SHIFT_PAD, device=rel_proj.device)
    dist = shift + extent + 256 - y
    outside = (dist < 0) | (dist >= extent)
    proj = rel_proj[:, dist.clamp(0, extent - 1)].transpose(0, 1)
    pad = (dist >= extent) if is_local else torch.zeros_like(outside)
    pad = (dist < 0) | pad
    pad = torch.zeros_like(proj[:, :1]).masked_fill(pad[:, None], float("-inf"))
    return torch.cat([proj.masked_fill(outside[:, None], 0), pad], 1).contiguous()


@triton.jit
def _kv_conv_tile(
    k_ptr,
    v_ptr,
    wk_ptr,
    wv_ptr,
    dk_ptr,
    dv_ptr,
    cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    kv_slot_ptr,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_k,
    stride_v,
    stride_w0,
    stride_w1,
    stride_dk,
    stride_dv,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    kv_block_size,
    stride_bt,
    max_blocks,
    N,
    scale,
    clip_threshold,
    clip_eps,
    norm_eps,
    W: tl.constexpr,
    D: tl.constexpr,
    DYNAMIC: tl.constexpr,
    IS_FP8_KV: tl.constexpr,
    BT: tl.constexpr,
):
    toff = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    is_v = tl.program_id(2) == 1
    d_idx = tl.arange(0, D)
    coff = head * D + d_idx
    t_mask = toff < T
    if is_v:
        x_ptr = v_ptr
        w_ptr = wv_ptr
        dyn_ptr = dv_ptr
        stride_x = stride_v
        stride_dyn = stride_dv
    else:
        x_ptr = k_ptr
        w_ptr = wk_ptr
        dyn_ptr = dk_ptr
        stride_x = stride_k
        stride_dyn = stride_dk
    y = _conv(
        x_ptr,
        stride_x,
        cache_ptr,
        head * stride_c_h + tl.program_id(2) * D + d_idx,
        stride_c_blk,
        stride_c_n,
        N,
        w_ptr,
        stride_w0,
        stride_w1,
        pos_ptr,
        seq_idx_ptr,
        slot_ptr,
        block_table_ptr,
        stride_bt,
        max_blocks,
        qstart_ptr,
        scale,
        toff,
        coff,
        t_mask,
        d_idx < D,
        W,
        dyn_ptr if DYNAMIC else None,
        stride_dyn,
        D,
    )
    y = y.to(k_ptr.dtype.element_ty).to(tl.float32)
    sum_sq = tl.sum(y * y, axis=1)
    if is_v:
        y = y * rsqrt(tl.div_rn(sum_sq, D) + norm_eps)[:, None]
        cache = value_cache_ptr
    else:
        norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
        y = y * tl.minimum(tl.div_rn(clip_threshold, norm + clip_eps), 1.0)[:, None]
        cache = key_cache_ptr
    kv_slot = tl.load(kv_slot_ptr + toff, mask=t_mask, other=-1)
    safe = tl.maximum(kv_slot, 0)
    dst = (
        cache
        + (safe // kv_block_size)[:, None] * stride_kv_blk
        + (safe % kv_block_size)[:, None] * stride_kv_n
        + head * stride_kv_h
        + d_idx[None, :]
    )
    if IS_FP8_KV:
        tl.store(dst, f32_to_e4m3_u8(y), mask=(kv_slot >= 0)[:, None])
    else:
        tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])


@triton.jit
def _attn_prologue_kernel(
    k_ptr,
    v_ptr,
    wk_ptr,
    wv_ptr,
    dk_ptr,
    dv_ptr,
    cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    kv_slot_ptr,
    pos_ptr,
    seq_idx_ptr,
    slot_ptr,
    block_table_ptr,
    qstart_ptr,
    T,
    stride_k,
    stride_v,
    stride_w0,
    stride_w1,
    stride_dk,
    stride_dv,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_kv_blk,
    stride_kv_n,
    stride_kv_h,
    kv_block_size,
    stride_bt,
    max_blocks,
    N,
    scale,
    clip_threshold,
    clip_eps,
    norm_eps,
    q_ptr,
    r_ptr,
    q_scale_ptr,
    r_scale_ptr,
    tau_ptr,
    rel_proj_ptr,
    q_out_ptr,
    rel_out_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    stride_q,
    stride_r,
    rel_threshold,
    W: tl.constexpr,
    D: tl.constexpr,
    DYNAMIC: tl.constexpr,
    IS_FP8_KV: tl.constexpr,
    BT: tl.constexpr,
    HQ: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    SHIFT_W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_E: tl.constexpr,
    QR_TOKENS: tl.constexpr,
    FUSE_QR: tl.constexpr,
):
    if FUSE_QR and tl.program_id(2) >= 2:
        if tl.program_id(1) * QR_TOKENS < BT:
            qr_clip_rel_tokens(
                tl.program_id(0) * BT + tl.program_id(1) * QR_TOKENS,
                QR_TOKENS,
                tl.program_id(2) - 3,
                T,
                q_ptr,
                r_ptr,
                q_scale_ptr,
                r_scale_ptr,
                tau_ptr,
                rel_proj_ptr,
                q_out_ptr,
                rel_out_ptr,
                seq_idx_ptr,
                cu_seqlens_q_ptr,
                seq_lens_ptr,
                stride_q,
                stride_r,
                clip_threshold,
                rel_threshold,
                clip_eps,
                HQ,
                D,
                D_REL,
                E,
                SHIFT_W,
                HAS_TAU,
                BLOCK_HQ,
                BLOCK_E,
            )
    else:
        _kv_conv_tile(
            k_ptr,
            v_ptr,
            wk_ptr,
            wv_ptr,
            dk_ptr,
            dv_ptr,
            cache_ptr,
            key_cache_ptr,
            value_cache_ptr,
            kv_slot_ptr,
            pos_ptr,
            seq_idx_ptr,
            slot_ptr,
            block_table_ptr,
            qstart_ptr,
            T,
            stride_k,
            stride_v,
            stride_w0,
            stride_w1,
            stride_dk,
            stride_dv,
            stride_c_blk,
            stride_c_h,
            stride_c_n,
            stride_kv_blk,
            stride_kv_n,
            stride_kv_h,
            kv_block_size,
            stride_bt,
            max_blocks,
            N,
            scale,
            clip_threshold,
            clip_eps,
            norm_eps,
            W,
            D,
            DYNAMIC,
            IS_FP8_KV,
            BT,
        )


@triton.jit
def _qr_rel_kernel(
    q_ptr,
    r_ptr,
    q_scale_ptr,
    r_scale_ptr,
    tau_ptr,
    rel_proj_ptr,
    q_out_ptr,
    rel_out_ptr,
    seq_idx_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    T,
    stride_q,
    stride_r,
    clip_threshold,
    rel_threshold,
    clip_eps,
    HQ: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    SHIFT_W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_E: tl.constexpr,
    QR_TOKENS: tl.constexpr,
):
    qr_clip_rel_tokens(
        tl.program_id(0) * QR_TOKENS,
        QR_TOKENS,
        tl.program_id(1) - 1,
        T,
        q_ptr,
        r_ptr,
        q_scale_ptr,
        r_scale_ptr,
        tau_ptr,
        rel_proj_ptr,
        q_out_ptr,
        rel_out_ptr,
        seq_idx_ptr,
        cu_seqlens_q_ptr,
        seq_lens_ptr,
        stride_q,
        stride_r,
        clip_threshold,
        rel_threshold,
        clip_eps,
        HQ,
        D,
        D_REL,
        E,
        SHIFT_W,
        HAS_TAU,
        BLOCK_HQ,
        BLOCK_E,
    )


_PROLOGUE_SPLIT_MIN_TOKENS = 512


def attention_prologue(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_proj: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    q_out: torch.Tensor,
    rel_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    clip_threshold: float,
    rel_threshold: float,
    norm_eps: float,
    clip_eps: float = 1e-6,
    tail_src_toks: torch.Tensor | None = None,
    tail_req_slots: torch.Tensor | None = None,
    num_tail_writes: int = 0,
) -> None:
    num_tokens = kv_slot_mapping.shape[0]
    if num_tokens == 0:
        return
    num_heads, head_dim = key_cache.shape[2:]
    num_q_heads, d_rel = r.shape[1:]
    extent = rel_proj.shape[2] - _SHIFT_PAD
    assert rel_proj.shape[:2] == (8, d_rel + 1)
    assert k.stride(1) == v.stride(1) == 1 and cache.stride(3) == 1
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    is_fp8_kv = key_cache.element_size() == 1
    kc_arg = key_cache.view(torch.uint8) if is_fp8_kv else key_cache
    vc_arg = value_cache.view(torch.uint8) if is_fp8_kv else value_cache

    block_e = min(extent, 256)
    dynamic = k_dynamic is not None
    if dynamic:
        assert k_dynamic is not None and v_dynamic is not None
        width = k_weight.shape[0]
        dyn_strides = (k_dynamic.stride(0), v_dynamic.stride(0))
    else:
        width = k_weight.shape[1]
        dyn_strides = (0, 0)
    block_t = 8 if num_tokens >= _PROLOGUE_SPLIT_MIN_TOKENS else 4
    qr_tokens = 8 if num_tokens >= _PROLOGUE_SPLIT_MIN_TOKENS else 0
    qr_blocks = 1 + (extent + 256) // block_e
    grid = (
        triton.cdiv(num_tokens, block_t),
        num_heads,
        2 + (0 if qr_tokens else qr_blocks),
    )
    tau_arg = tau if tau is not None else q_scale  # unused placeholder when HAS_TAU=False
    _attn_prologue_kernel[grid](
        k,
        v,
        k_weight,
        v_weight,
        k_dynamic if dynamic else k_weight,
        v_dynamic if dynamic else v_weight,
        cache,
        kc_arg,
        vc_arg,
        kv_slot_mapping,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        k.stride(0),
        v.stride(0),
        k_weight.stride(0),
        k_weight.stride(1),
        *dyn_strides,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        kc_arg.stride(0),
        kc_arg.stride(1),
        kc_arg.stride(2),
        kc_arg.shape[1],
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width * (2 if dynamic else 1) + 1) ** -0.5,
        clip_threshold,
        clip_eps,
        norm_eps,
        q,
        r,
        q_scale,
        r_scale,
        tau_arg,
        rel_proj,
        q_out,
        rel_out,
        cu_seqlens_q,
        seq_lens,
        q.stride(0),
        r.stride(0),
        rel_threshold,
        W=width,
        D=head_dim,
        DYNAMIC=dynamic,
        IS_FP8_KV=is_fp8_kv,
        BT=block_t,
        HQ=num_q_heads,
        D_REL=d_rel,
        E=extent,
        SHIFT_W=rel_proj.shape[2],
        HAS_TAU=tau is not None,
        BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
        BLOCK_E=block_e,
        QR_TOKENS=triton.cdiv(block_t, num_heads),
        FUSE_QR=not qr_tokens,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if qr_tokens:
        _qr_rel_kernel[(triton.cdiv(num_tokens, qr_tokens), qr_blocks)](
            q,
            r,
            q_scale,
            r_scale,
            tau_arg,
            rel_proj,
            q_out,
            rel_out,
            seq_idx,
            cu_seqlens_q,
            seq_lens,
            num_tokens,
            q.stride(0),
            r.stride(0),
            clip_threshold,
            rel_threshold,
            clip_eps,
            HQ=num_q_heads,
            D=head_dim,
            D_REL=d_rel,
            E=extent,
            SHIFT_W=rel_proj.shape[2],
            HAS_TAU=tau is not None,
            BLOCK_HQ=max(triton.next_power_of_2(num_q_heads), 16),
            BLOCK_E=block_e,
            QR_TOKENS=qr_tokens,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if num_tail_writes > 0 and tail_src_toks is not None and tail_req_slots is not None:
        ring_commit_tail(
            k, cache, positions, tail_src_toks, tail_req_slots, num_tail_writes, 0, head_dim
        )
        ring_commit_tail(
            v, cache, positions, tail_src_toks, tail_req_slots, num_tail_writes, head_dim, head_dim
        )
