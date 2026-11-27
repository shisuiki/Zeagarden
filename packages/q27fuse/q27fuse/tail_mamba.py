"""q27fuse.tail_mamba - align-mode Mamba/GDN state migration kernels with a fixed grid (track "tail", piece "mamba").

Ported for 1Cat-vLLM v1.5.3 + v1.5.4: preserves v1.5.4 64-token decode-output prefix cache checkpoints
(decode_ckpt_slots + spec_verified) while keeping the fixed 96 x 16 launch grid.
"""
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu import mamba_align as _MA
from vllm.v1.worker.gpu.mamba_align import _TEMPORAL_TILES, _copy_mamba_state_block

_copy_to_block_id = getattr(_MA, "_copy_mamba_state_to_block_id", None)


@triton.jit
def _q27_precopy_kernel(
    state_idx_ptr, src_col_ptr, token_bias_ptr, block_table_ptrs_ptr, block_table_stride_req, state_base_addrs_ptr,
    state_block_strides_ptr, state_elem_sizes_ptr, state_inner_sizes_ptr, state_conv_widths_ptr,
    state_group_indices_ptr, idx_mapping_ptr, query_start_loc_ptr, num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr, TEMPORAL_TILES: tl.constexpr, TEMPORAL_REPLAY: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    state_idx = pid // TEMPORAL_TILES
    tile_idx = pid % TEMPORAL_TILES
    r = tl.arange(0, BLOCK_R)
    m = r < num_reqs
    req = tl.load(idx_mapping_ptr + r, mask=m, other=-1)
    valid = m & (req >= 0)
    reqc = tl.where(valid, req, 0)
    src = tl.load(src_col_ptr + reqc, mask=valid, other=-1)
    dst = tl.load(state_idx_ptr + reqc, mask=valid, other=-1)
    tb = tl.load(token_bias_ptr + reqc, mask=valid, other=0)
    qs = tl.load(query_start_loc_ptr + r, mask=m, other=0)
    qe = tl.load(query_start_loc_ptr + r + 1, mask=m, other=0)
    stt = ((qe - qs) == 1) & (tb > 0)
    need = valid & (src >= 0) & ((src != dst) | stt)
    if tl.sum(need.to(tl.int32), axis=0) == 0:
        return
    for b in range(num_reqs):
        req_idx = tl.load(idx_mapping_ptr + b)
        if req_idx >= 0:
            src_col = tl.load(src_col_ptr + req_idx)
            dst_col = tl.load(state_idx_ptr + req_idx)
            token_bias = tl.load(token_bias_ptr + req_idx)
            query_start = tl.load(query_start_loc_ptr + b)
            query_end = tl.load(query_start_loc_ptr + b + 1)
            single_token_tail = query_end - query_start == 1 and token_bias > 0
            if not (src_col < 0 or (src_col == dst_col and not single_token_tail)):
                _copy_mamba_state_block(
                    state_idx, b, src_col, dst_col, token_bias, block_table_ptrs_ptr, block_table_stride_req,
                    state_base_addrs_ptr, state_block_strides_ptr, state_elem_sizes_ptr, state_inner_sizes_ptr,
                    state_conv_widths_ptr, state_group_indices_ptr, tile_idx, COPY_BLOCK_SIZE, TEMPORAL_TILES,
                    TEMPORAL_REPLAY)


@triton.jit
def _q27_postprocess_kernel(
    num_accepted_snapshot_ptr, num_accepted_out_ptr, state_idx_ptr, new_num_computed_tokens_ptr,
    block_table_ptrs_ptr, block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr,
    state_elem_sizes_ptr, state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, idx_mapping_ptr,
    decode_ckpt_slots_ptr, spec_verified_ptr, num_reqs,
    MAMBA_BLOCK_SIZE: tl.constexpr, COPY_BLOCK_SIZE: tl.constexpr, TEMPORAL_TILES: tl.constexpr,
    TEMPORAL_REPLAY: tl.constexpr, BLOCK_R: tl.constexpr,
    HAS_DECODE_CKPT: tl.constexpr = False, DECODE_CKPT_MAX_REQS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    state_meta_idx = pid // TEMPORAL_TILES
    tile_idx = pid % TEMPORAL_TILES
    r = tl.arange(0, BLOCK_R)
    m = r < num_reqs
    req = tl.load(idx_mapping_ptr + r, mask=m, other=-1)
    valid = m & (req >= 0)
    reqc = tl.where(valid, req, 0)
    num_accepted = tl.load(num_accepted_snapshot_ptr + reqc, mask=valid, other=1)
    src = tl.load(state_idx_ptr + reqc, mask=valid, other=-1)
    nnc = tl.load(new_num_computed_tokens_ptr + reqc, mask=valid, other=0)
    running = nnc - num_accepted + 1
    aligned = (nnc // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    go = valid & (aligned >= running)
    tb = aligned - running
    dst = aligned // MAMBA_BLOCK_SIZE - 1
    same = src == dst
    if state_meta_idx == 0 and tile_idx == 0:
        tl.store(num_accepted_out_ptr + reqc, tl.full([BLOCK_R], 1, tl.int32), mask=go & same)
    copy_full = go & ~(same & (tb == 0))
    if TEMPORAL_REPLAY:
        is_temporal = tl.load(state_conv_widths_ptr + state_meta_idx) == 0
        copy_full = copy_full & (~is_temporal)
    if tl.sum(copy_full.to(tl.int32), axis=0) != 0:
        for b in range(num_reqs):
            req_idx = tl.load(idx_mapping_ptr + b)
            if req_idx >= 0:
                n_acc = tl.load(num_accepted_snapshot_ptr + req_idx)
                src_col = tl.load(state_idx_ptr + req_idx)
                new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
                num_tokens_running_state = new_num_computed - n_acc + 1
                aligned_new_computed = (new_num_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
                if aligned_new_computed >= num_tokens_running_state:
                    token_bias = aligned_new_computed - num_tokens_running_state
                    dst_col = aligned_new_computed // MAMBA_BLOCK_SIZE - 1
                    skip_f = src_col == dst_col and token_bias == 0
                    if TEMPORAL_REPLAY:
                        skip_f = skip_f | (tl.load(state_conv_widths_ptr + state_meta_idx) == 0)
                    if not skip_f:
                        _copy_mamba_state_block(
                            state_meta_idx, b, src_col, dst_col, token_bias, block_table_ptrs_ptr,
                            block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr, state_elem_sizes_ptr,
                            state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, tile_idx,
                            COPY_BLOCK_SIZE, TEMPORAL_TILES, TEMPORAL_REPLAY)
    if HAS_DECODE_CKPT:
        aligned_64 = (nnc // 64) * 64
        need_64 = (valid & (aligned_64 >= running) & (aligned_64 > 0)
                   & ((aligned_64 % MAMBA_BLOCK_SIZE) != 0) & (num_accepted < 64) & (src >= 0))
        if tl.sum(need_64.to(tl.int32), axis=0) != 0:
            group_idx = tl.load(state_group_indices_ptr + state_meta_idx).to(tl.int64)
            for b in range(num_reqs):
                req_idx = tl.load(idx_mapping_ptr + b)
                if req_idx >= 0:
                    n_acc = tl.load(num_accepted_snapshot_ptr + req_idx)
                    src_col = tl.load(state_idx_ptr + req_idx)
                    new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
                    num_tokens_running_state = new_num_computed - n_acc + 1
                    a64 = (new_num_computed // 64) * 64
                    if (a64 >= num_tokens_running_state and a64 > 0
                            and a64 % MAMBA_BLOCK_SIZE != 0 and n_acc < 64 and src_col >= 0):
                        parity = (a64 // 64) & 1
                        ckpt_dst_id = tl.load(
                            decode_ckpt_slots_ptr + (group_idx * DECODE_CKPT_MAX_REQS + req_idx) * 2 + parity
                        ).to(tl.int64)
                        if ckpt_dst_id >= 0:
                            bias_64 = a64 - num_tokens_running_state
                            skip_ckpt = False
                            if TEMPORAL_REPLAY:
                                was_spec = tl.load(spec_verified_ptr + req_idx) != 0
                                if was_spec:
                                    skip_ckpt = tl.load(state_conv_widths_ptr + state_meta_idx) == 0
                            if not skip_ckpt:
                                _copy_to_block_id(
                                    state_meta_idx, b, src_col, ckpt_dst_id, bias_64, block_table_ptrs_ptr,
                                    block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr,
                                    state_elem_sizes_ptr, state_inner_sizes_ptr, state_conv_widths_ptr,
                                    state_group_indices_ptr, tile_idx, COPY_BLOCK_SIZE, TEMPORAL_TILES, TEMPORAL_REPLAY)


def _block_r(n):
    return max(16, triton.next_power_of_2(n))


def run_precopy(ctx, num_reqs, state_idx, src_col, token_bias, idx_mapping, query_start_loc):
    if num_reqs == 0 or not ctx.is_initialized:
        return
    temporal_replay = bool(getattr(ctx, "temporal_replay", False))
    _q27_precopy_kernel[(ctx.num_states * _TEMPORAL_TILES,)](
        state_idx, src_col, token_bias, ctx.block_table_ptrs, ctx.block_table_stride_req, ctx.state_base_addrs,
        ctx.state_block_strides, ctx.state_elem_sizes, ctx.state_inner_sizes, ctx.state_conv_widths,
        ctx.state_group_indices, idx_mapping, query_start_loc, num_reqs,
        COPY_BLOCK_SIZE=1024, TEMPORAL_TILES=_TEMPORAL_TILES, TEMPORAL_REPLAY=temporal_replay,
        BLOCK_R=_block_r(num_reqs))


def run_postprocess(ctx, num_reqs, num_accepted_tokens, state_idx, new_num_computed_tokens, idx_mapping,
                    decode_ckpt_slots=None, spec_verified=None):
    if num_reqs == 0 or not ctx.is_initialized:
        return
    snapshot = ctx.num_accepted_tokens_out
    snapshot.copy_(num_accepted_tokens)
    temporal_replay = bool(getattr(ctx, "temporal_replay", False))
    has_decode_ckpt = (
        _copy_to_block_id is not None
        and decode_ckpt_slots is not None
        and ctx.block_size >= 128
        and (not temporal_replay or spec_verified is not None)
    )
    _q27_postprocess_kernel[(ctx.num_states * _TEMPORAL_TILES,)](
        snapshot, num_accepted_tokens, state_idx, new_num_computed_tokens, ctx.block_table_ptrs,
        ctx.block_table_stride_req, ctx.state_base_addrs, ctx.state_block_strides, ctx.state_elem_sizes,
        ctx.state_inner_sizes, ctx.state_conv_widths, ctx.state_group_indices, idx_mapping,
        decode_ckpt_slots if has_decode_ckpt else state_idx,
        spec_verified if (has_decode_ckpt and spec_verified is not None) else state_idx,
        num_reqs,
        MAMBA_BLOCK_SIZE=ctx.block_size, COPY_BLOCK_SIZE=1024, TEMPORAL_TILES=_TEMPORAL_TILES,
        TEMPORAL_REPLAY=temporal_replay, BLOCK_R=_block_r(num_reqs),
        HAS_DECODE_CKPT=has_decode_ckpt,
        DECODE_CKPT_MAX_REQS=(decode_ckpt_slots.shape[1] if decode_ckpt_slots is not None else 0))


@triton.jit
def _q27_accept_kernel(
    num_sampled_ptr, num_accepted_ptr, replay_accepted_ptr, state_idx_ptr, new_num_computed_tokens_ptr,
    block_table_ptrs_ptr, block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr,
    state_elem_sizes_ptr, state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, idx_mapping_ptr,
    decode_ckpt_slots_ptr, spec_verified_ptr, num_reqs,
    HAS_REPLAY: tl.constexpr, MAMBA_BLOCK_SIZE: tl.constexpr, COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr, TEMPORAL_REPLAY: tl.constexpr, BLOCK_R: tl.constexpr,
    HAS_DECODE_CKPT: tl.constexpr = False, DECODE_CKPT_MAX_REQS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    state_meta_idx = pid // TEMPORAL_TILES
    tile_idx = pid % TEMPORAL_TILES
    r = tl.arange(0, BLOCK_R)
    m = r < num_reqs
    req = tl.load(idx_mapping_ptr + r, mask=m, other=-1)
    valid = m & (req >= 0)
    reqc = tl.where(valid, req, 0)
    ns = tl.load(num_sampled_ptr + r, mask=valid, other=1)
    num_accepted = tl.maximum(ns, 1)
    src = tl.load(state_idx_ptr + reqc, mask=valid, other=-1)
    nnc = tl.load(new_num_computed_tokens_ptr + reqc, mask=valid, other=0)
    running = nnc - num_accepted + 1
    aligned = (nnc // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    go = valid & (aligned >= running)
    tb = aligned - running
    dst = aligned // MAMBA_BLOCK_SIZE - 1
    same = src == dst
    if state_meta_idx == 0 and tile_idx == 0:
        tl.store(num_accepted_ptr + reqc, tl.where(go & same, 1, num_accepted), mask=valid)
        if HAS_REPLAY:
            tl.store(replay_accepted_ptr + reqc, num_accepted, mask=valid)
    copy_full = go & ~(same & (tb == 0))
    if TEMPORAL_REPLAY:
        is_temporal = tl.load(state_conv_widths_ptr + state_meta_idx) == 0
        copy_full = copy_full & (~is_temporal)
    if tl.sum(copy_full.to(tl.int32), axis=0) != 0:
        for b in range(num_reqs):
            req_idx = tl.load(idx_mapping_ptr + b)
            if req_idx >= 0:
                n_acc = tl.maximum(tl.load(num_sampled_ptr + b), 1)
                src_col = tl.load(state_idx_ptr + req_idx)
                new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
                num_tokens_running_state = new_num_computed - n_acc + 1
                aligned_new_computed = (new_num_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
                if aligned_new_computed >= num_tokens_running_state:
                    token_bias = aligned_new_computed - num_tokens_running_state
                    dst_col = aligned_new_computed // MAMBA_BLOCK_SIZE - 1
                    skip_f = src_col == dst_col and token_bias == 0
                    if TEMPORAL_REPLAY:
                        skip_f = skip_f | (tl.load(state_conv_widths_ptr + state_meta_idx) == 0)
                    if not skip_f:
                        _copy_mamba_state_block(
                            state_meta_idx, b, src_col, dst_col, token_bias, block_table_ptrs_ptr,
                            block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr, state_elem_sizes_ptr,
                            state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, tile_idx,
                            COPY_BLOCK_SIZE, TEMPORAL_TILES, TEMPORAL_REPLAY)
    if HAS_DECODE_CKPT:
        aligned_64 = (nnc // 64) * 64
        need_64 = (valid & (aligned_64 >= running) & (aligned_64 > 0)
                   & ((aligned_64 % MAMBA_BLOCK_SIZE) != 0) & (num_accepted < 64) & (src >= 0))
        if tl.sum(need_64.to(tl.int32), axis=0) != 0:
            group_idx = tl.load(state_group_indices_ptr + state_meta_idx).to(tl.int64)
            for b in range(num_reqs):
                req_idx = tl.load(idx_mapping_ptr + b)
                if req_idx >= 0:
                    n_acc = tl.maximum(tl.load(num_sampled_ptr + b), 1)
                    src_col = tl.load(state_idx_ptr + req_idx)
                    new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
                    num_tokens_running_state = new_num_computed - n_acc + 1
                    a64 = (new_num_computed // 64) * 64
                    if (a64 >= num_tokens_running_state and a64 > 0
                            and a64 % MAMBA_BLOCK_SIZE != 0 and n_acc < 64 and src_col >= 0):
                        parity = (a64 // 64) & 1
                        ckpt_dst_id = tl.load(
                            decode_ckpt_slots_ptr + (group_idx * DECODE_CKPT_MAX_REQS + req_idx) * 2 + parity
                        ).to(tl.int64)
                        if ckpt_dst_id >= 0:
                            bias_64 = a64 - num_tokens_running_state
                            skip_ckpt = False
                            if TEMPORAL_REPLAY:
                                was_spec = tl.load(spec_verified_ptr + req_idx) != 0
                                if was_spec:
                                    skip_ckpt = tl.load(state_conv_widths_ptr + state_meta_idx) == 0
                            if not skip_ckpt:
                                _copy_to_block_id(
                                    state_meta_idx, b, src_col, ckpt_dst_id, bias_64, block_table_ptrs_ptr,
                                    block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr,
                                    state_elem_sizes_ptr, state_inner_sizes_ptr, state_conv_widths_ptr,
                                    state_group_indices_ptr, tile_idx, COPY_BLOCK_SIZE, TEMPORAL_TILES, TEMPORAL_REPLAY)


def run_accept(ctx, num_reqs, num_sampled, num_accepted_tokens, replay_accepted, state_idx, new_num_computed_tokens,
               idx_mapping, decode_ckpt_slots=None, spec_verified=None):
    temporal_replay = bool(getattr(ctx, "temporal_replay", False))
    has_decode_ckpt = (
        _copy_to_block_id is not None
        and decode_ckpt_slots is not None
        and ctx.block_size >= 128
        and (not temporal_replay or spec_verified is not None)
    )
    _q27_accept_kernel[(ctx.num_states * _TEMPORAL_TILES,)](
        num_sampled, num_accepted_tokens, replay_accepted if replay_accepted is not None else num_accepted_tokens,
        state_idx, new_num_computed_tokens, ctx.block_table_ptrs, ctx.block_table_stride_req, ctx.state_base_addrs,
        ctx.state_block_strides, ctx.state_elem_sizes, ctx.state_inner_sizes, ctx.state_conv_widths,
        ctx.state_group_indices, idx_mapping,
        decode_ckpt_slots if has_decode_ckpt else state_idx,
        spec_verified if (has_decode_ckpt and spec_verified is not None) else state_idx,
        num_reqs,
        HAS_REPLAY=replay_accepted is not None, MAMBA_BLOCK_SIZE=ctx.block_size, COPY_BLOCK_SIZE=1024,
        TEMPORAL_TILES=_TEMPORAL_TILES, TEMPORAL_REPLAY=temporal_replay, BLOCK_R=_block_r(num_reqs),
        HAS_DECODE_CKPT=has_decode_ckpt,
        DECODE_CKPT_MAX_REQS=(decode_ckpt_slots.shape[1] if decode_ckpt_slots is not None else 0))
