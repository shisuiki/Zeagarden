"""q27fuse.tail_mamba - align-mode Mamba/GDN state migration kernels with a fixed grid (track "tail", piece "mamba").

1Cat's run_mamba_align_precopy / run_mamba_align_postprocess (vllm/v1/worker/gpu/mamba_align.py) launch one program per
(request, state tensor, temporal tile) = num_reqs x 96 x 16 programs every step (24,576 at 16 requests), and almost every
program returns after a few loads: a block-boundary copy is needed about once per 1,568 tokens per request. In the
server that costs 36 us per kernel at C = 16 and grows with the batch (trace r1_bl_map_c16; untraced: tail.md).
These versions launch one program per (state tensor, tile) - fixed at 96 x 16 - and test all requests of the step with
one vector load per field; only when a request needs a copy do they walk the requests, and the copy itself is 1Cat's
_copy_mamba_state_block with the same (state, request, tile) arguments, so the bytes moved are identical. Different
requests copy disjoint blocks, so doing them in one program instead of several changes no result.
"""
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.mamba_align import _TEMPORAL_TILES, _copy_mamba_state_block


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
    num_reqs,
    MAMBA_BLOCK_SIZE: tl.constexpr, COPY_BLOCK_SIZE: tl.constexpr, TEMPORAL_TILES: tl.constexpr,
    TEMPORAL_REPLAY: tl.constexpr, BLOCK_R: tl.constexpr,
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
    src = tl.load(state_idx_ptr + reqc, mask=valid, other=0)
    nnc = tl.load(new_num_computed_tokens_ptr + reqc, mask=valid, other=0)
    running = nnc - num_accepted + 1
    aligned = (nnc // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    go = valid & (aligned >= running)
    tb = aligned - running
    dst = aligned // MAMBA_BLOCK_SIZE - 1
    same = src == dst
    if state_meta_idx == 0 and tile_idx == 0:
        tl.store(num_accepted_out_ptr + reqc, tl.full([BLOCK_R], 1, tl.int32), mask=go & same)
    copy = go & ~(same & (tb == 0))
    if TEMPORAL_REPLAY:  # noqa: SIM102
        if tl.load(state_conv_widths_ptr + state_meta_idx) == 0:
            return
    if tl.sum(copy.to(tl.int32), axis=0) == 0:
        return
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
                if not (src_col == dst_col and token_bias == 0):
                    _copy_mamba_state_block(
                        state_meta_idx, b, src_col, dst_col, token_bias, block_table_ptrs_ptr,
                        block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr, state_elem_sizes_ptr,
                        state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, tile_idx,
                        COPY_BLOCK_SIZE, TEMPORAL_TILES, TEMPORAL_REPLAY)


def _block_r(n):
    return max(16, triton.next_power_of_2(n))


def run_precopy(ctx, num_reqs, state_idx, src_col, token_bias, idx_mapping, query_start_loc):
    """= mamba_align.run_mamba_align_precopy"""
    if num_reqs == 0 or not ctx.is_initialized:
        return
    _q27_precopy_kernel[(ctx.num_states * _TEMPORAL_TILES,)](
        state_idx, src_col, token_bias, ctx.block_table_ptrs, ctx.block_table_stride_req, ctx.state_base_addrs,
        ctx.state_block_strides, ctx.state_elem_sizes, ctx.state_inner_sizes, ctx.state_conv_widths,
        ctx.state_group_indices, idx_mapping, query_start_loc, num_reqs,
        COPY_BLOCK_SIZE=1024, TEMPORAL_TILES=_TEMPORAL_TILES, TEMPORAL_REPLAY=ctx.temporal_replay,
        BLOCK_R=_block_r(num_reqs))


def run_postprocess(ctx, num_reqs, num_accepted_tokens, state_idx, new_num_computed_tokens, idx_mapping):
    """= mamba_align.run_mamba_align_postprocess (same snapshot copy first)"""
    if num_reqs == 0 or not ctx.is_initialized:
        return
    snapshot = ctx.num_accepted_tokens_out
    snapshot.copy_(num_accepted_tokens)
    _q27_postprocess_kernel[(ctx.num_states * _TEMPORAL_TILES,)](
        snapshot, num_accepted_tokens, state_idx, new_num_computed_tokens, ctx.block_table_ptrs,
        ctx.block_table_stride_req, ctx.state_base_addrs, ctx.state_block_strides, ctx.state_elem_sizes,
        ctx.state_inner_sizes, ctx.state_conv_widths, ctx.state_group_indices, idx_mapping, num_reqs,
        MAMBA_BLOCK_SIZE=ctx.block_size, COPY_BLOCK_SIZE=1024, TEMPORAL_TILES=_TEMPORAL_TILES,
        TEMPORAL_REPLAY=ctx.temporal_replay, BLOCK_R=_block_r(num_reqs))


@triton.jit
def _q27_accept_kernel(
    num_sampled_ptr, num_accepted_ptr, replay_accepted_ptr, state_idx_ptr, new_num_computed_tokens_ptr,
    block_table_ptrs_ptr, block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr,
    state_elem_sizes_ptr, state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, idx_mapping_ptr,
    num_reqs,
    HAS_REPLAY: tl.constexpr, MAMBA_BLOCK_SIZE: tl.constexpr, COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr, TEMPORAL_REPLAY: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """MambaHybridModelState.postprocess_state in one launch: _scatter_num_accepted_kernel (num_accepted[req] =
    max(num_sampled, 1)), SpecReplayState.record_accepted (replay.accepted[req] = that value) and the align-mode
    postprocess (its snapshot of num_accepted is exactly that value for the step's requests), same grid as
    _q27_postprocess_kernel"""
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
    src = tl.load(state_idx_ptr + reqc, mask=valid, other=0)
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
    copy = go & ~(same & (tb == 0))
    if TEMPORAL_REPLAY:  # noqa: SIM102
        if tl.load(state_conv_widths_ptr + state_meta_idx) == 0:
            return
    if tl.sum(copy.to(tl.int32), axis=0) == 0:
        return
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
                if not (src_col == dst_col and token_bias == 0):
                    _copy_mamba_state_block(
                        state_meta_idx, b, src_col, dst_col, token_bias, block_table_ptrs_ptr,
                        block_table_stride_req, state_base_addrs_ptr, state_block_strides_ptr, state_elem_sizes_ptr,
                        state_inner_sizes_ptr, state_conv_widths_ptr, state_group_indices_ptr, tile_idx,
                        COPY_BLOCK_SIZE, TEMPORAL_TILES, TEMPORAL_REPLAY)


def run_accept(ctx, num_reqs, num_sampled, num_accepted_tokens, replay_accepted, state_idx, new_num_computed_tokens,
               idx_mapping):
    """scatter + record_accepted + run_mamba_align_postprocess in one kernel (ctx initialized, num_reqs > 0)"""
    _q27_accept_kernel[(ctx.num_states * _TEMPORAL_TILES,)](
        num_sampled, num_accepted_tokens, replay_accepted if replay_accepted is not None else num_accepted_tokens,
        state_idx, new_num_computed_tokens, ctx.block_table_ptrs, ctx.block_table_stride_req, ctx.state_base_addrs,
        ctx.state_block_strides, ctx.state_elem_sizes, ctx.state_inner_sizes, ctx.state_conv_widths,
        ctx.state_group_indices, idx_mapping, num_reqs,
        HAS_REPLAY=replay_accepted is not None, MAMBA_BLOCK_SIZE=ctx.block_size, COPY_BLOCK_SIZE=1024,
        TEMPORAL_TILES=_TEMPORAL_TILES, TEMPORAL_REPLAY=ctx.temporal_replay, BLOCK_R=_block_r(num_reqs))
