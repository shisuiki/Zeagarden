# SPDX-License-Identifier: Apache-2.0
"""Shard-local Markov head of the Inkling2 DSpark draft and shard-local target argmax (sm_70).

Avoids full-vocabulary TP all-gathers (201K logits) during both greedy and probabilistic
DSpark drafting by running the folded LM head + Gumbel-max locally on each TP shard and
exchanging only 8 exact-FP16-integer words per request per step (64 bytes/req).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

# FP16 words per (value, token) pair: 5 x 11 + 9 bits of FP64 value, 11 + 7 bits of token id.
WORDS = 8
VOCAB_ROWS = 32
ARGMAX_CHUNK = 1024


@triton.jit
def _winner(G, row, num_rows, WORLD: tl.constexpr, WP: tl.constexpr):
    """Token of the best pair over the ranks (ties: lowest rank / lowest token id)."""
    r = tl.arange(0, WP)
    m = r < WORLD
    p = G + (r * num_rows + row) * 8
    bits = tl.load(p, mask=m, other=0.0).to(tl.uint64)
    bits |= tl.load(p + 1, mask=m, other=0.0).to(tl.uint64) << 11
    bits |= tl.load(p + 2, mask=m, other=0.0).to(tl.uint64) << 22
    bits |= tl.load(p + 3, mask=m, other=0.0).to(tl.uint64) << 33
    bits |= tl.load(p + 4, mask=m, other=0.0).to(tl.uint64) << 44
    bits |= tl.load(p + 5, mask=m, other=0.0).to(tl.uint64) << 55
    value = tl.where(m, bits.to(tl.float64, bitcast=True), float("-inf"))
    token = tl.load(p + 6, mask=m, other=0.0).to(tl.int64)
    token |= tl.load(p + 7, mask=m, other=0.0).to(tl.int64) << 11
    _, best = tl.max(value, axis=0, return_indices=True)
    return tl.sum(tl.where(r == best, token, 0), axis=0)


@triton.jit
def _pick_pack_kernel(
    PART_V,
    PART_I,
    OUT,
    n_parts,
    num_rows,
    rank,
    NP: tl.constexpr,
    WORLD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, NP)
    m = c < n_parts
    v = tl.load(PART_V + row * n_parts + c, mask=m, other=float("-inf"))
    ids = tl.load(PART_I + row * n_parts + c, mask=m, other=0).to(tl.int64)
    best, j = tl.max(v, axis=0, return_indices=True)
    token = tl.sum(tl.where(c == j, ids, 0), axis=0).to(tl.uint64)
    bits = best.to(tl.float64).to(tl.uint64, bitcast=True)
    for r in tl.static_range(WORLD):
        dst = OUT + (r * num_rows + row) * 8
        mine = r == rank
        tl.store(dst, tl.where(mine, (bits & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 1, tl.where(mine, ((bits >> 11) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 2, tl.where(mine, ((bits >> 22) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 3, tl.where(mine, ((bits >> 33) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 4, tl.where(mine, ((bits >> 44) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 5, tl.where(mine, (bits >> 55).to(tl.float16), 0.0))
        tl.store(dst + 6, tl.where(mine, (token & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 7, tl.where(mine, (token >> 11).to(tl.float16), 0.0))


def pick_pack(
    part_v: torch.Tensor, part_i: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """Pack per-shard best `(value, token)` into `(world, rows, 8)` exact FP16 integers."""
    rows, parts = part_v.shape
    assert part_v.is_contiguous() and part_i.is_contiguous() and part_i.dtype == torch.int32
    out = torch.empty((world, rows, WORDS), dtype=torch.float16, device=part_v.device)
    if rows:
        _pick_pack_kernel[(rows,)](
            part_v,
            part_i,
            out,
            parts,
            rows,
            rank,
            NP=triton.next_power_of_2(parts),
            WORLD=world,
            num_warps=8 if parts > 512 else 4,
        )
    return out


@triton.jit
def _unpack_kernel(G, OUT, num_rows, WORLD: tl.constexpr, WP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tl.store(OUT + row, _winner(G, row, num_rows, WORLD, WP))


def unpack_tokens(gathered: torch.Tensor) -> torch.Tensor:
    """Unpack `int64` winning token id per row from a reduced `(world, rows, 8)` stack."""
    world, rows, _ = gathered.shape
    assert gathered.is_contiguous() and gathered.dtype == torch.float16
    out = torch.empty(rows, dtype=torch.int64, device=gathered.device)
    if rows:
        _unpack_kernel[(rows,)](
            gathered,
            out,
            rows,
            WORLD=world,
            WP=triton.next_power_of_2(world),
            num_warps=1,
        )
    return out


@triton.jit
def _markov_rnn_kernel(
    STATE,
    LATENT,
    CTX,
    W,
    W1,
    PREV,
    TOKENS,
    tok_stride,
    ANCHORS,
    anchor_stride,
    step,
    num_reqs,
    RANK: tl.constexpr,
    U: tl.constexpr,
    WORLD: tl.constexpr,
    WP: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    if step == 0:
        prev = tl.load(ANCHORS + b * anchor_stride).to(tl.int64)
    else:
        prev = _winner(PREV, b, num_reqs, WORLD, WP)
        if tl.program_id(1) == 0:
            tl.store(TOKENS + b * tok_stride + (step - 1), prev)
    u = tl.program_id(1) * U + tl.arange(0, U)
    cols = tl.arange(0, RANK)
    xp = tl.load(W1 + prev * RANK + cols).to(tl.float32)
    src = STATE + (step * num_reqs + b) * RANK
    xs = tl.load(src + cols).to(W.dtype.element_ty).to(tl.float32)
    state = tl.load(src + u)
    ctx = CTX + (step * num_reqs + b) * (3 * RANK)
    wg = W + u[:, None] * (2 * RANK) + cols[None, :]
    wc = W + (RANK + u)[:, None] * (2 * RANK) + cols[None, :]
    wo = W + (2 * RANK + u)[:, None] * (2 * RANK) + cols[None, :]
    raw_g = tl.sum(
        tl.load(wg).to(tl.float32) * xs[None, :]
        + tl.load(wg + RANK).to(tl.float32) * xp[None, :],
        axis=1,
    ) + tl.load(ctx + u)
    raw_c = tl.sum(
        tl.load(wc).to(tl.float32) * xs[None, :]
        + tl.load(wc + RANK).to(tl.float32) * xp[None, :],
        axis=1,
    ) + tl.load(ctx + RANK + u)
    raw_o = tl.sum(
        tl.load(wo).to(tl.float32) * xs[None, :]
        + tl.load(wo + RANK).to(tl.float32) * xp[None, :],
        axis=1,
    ) + tl.load(ctx + 2 * RANK + u)
    gate = 1.0 / (1.0 + tldevice.exp(-raw_g))
    new_state = gate * state + (1.0 - gate) * tldevice.tanh(raw_c)
    tl.store(STATE + ((step + 1) * num_reqs + b) * RANK + u, new_state)
    tl.store(LATENT + b * RANK + u, tldevice.tanh(raw_o).to(LATENT.dtype.element_ty))


@triton.jit
def _markov_vocab_kernel(
    BASE,
    base_stride,
    FOLD_B,
    FW,
    LATENT,
    DENSE,
    dense_stride,
    PART_V,
    PART_I,
    IDX_MAP,
    POS,
    TEMP,
    SEEDS,
    step,
    num_reqs,
    num_valid,
    id_base,
    n_chunks,
    softcap,
    steps,
    S_ALL: tl.constexpr,
    RANK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_DENSE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    chunk = tl.program_id(0)
    rows = chunk * ROWS + tl.arange(0, ROWS)
    mask = rows < num_valid
    cols = tl.arange(0, RANK)
    w = tl.load(
        FW + rows.to(tl.int64)[:, None] * RANK + cols[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    fold_b = tl.load(FOLD_B + rows, mask=mask, other=0.0).to(tl.float32)
    keys = id_base + rows
    for b in range(num_reqs):
        lat = tl.load(LATENT + b * RANK + cols).to(tl.float32)
        bias = tl.sum(w * lat[None, :], axis=1) + fold_b
        row = (b * steps + step).to(tl.int64)
        base = tl.load(BASE + row * base_stride + rows, mask=mask, other=0.0)
        logits = tldevice.tanh((base.to(tl.float32) + bias) / softcap) * softcap
        sample = b * S_ALL + step
        rs = tl.load(IDX_MAP + sample).to(tl.int64)
        valid = rs >= 0
        temp = tl.load(TEMP + rs, mask=valid, other=0.0).to(tl.float32)
        if temp != 0.0:
            logits = logits / temp
        logits = tl.where(mask, logits, float("-inf"))
        if HAS_DENSE:
            tl.store(
                DENSE + (rs * S_ALL + step) * dense_stride + rows,
                logits,
                mask=mask & valid,
            )
        seed = tl.load(SEEDS + rs, mask=valid, other=0)
        pos = tl.load(POS + sample) - 1
        value, idx = gumbel_noised_argmax(
            logits,
            keys,
            mask,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
        tl.store(PART_V + b * n_chunks + chunk, value)
        tl.store(PART_I + b * n_chunks + chunk, id_base + chunk * ROWS + idx)


@triton.jit
def _markov_last_kernel(
    PREV, TOKENS, tok_stride, step, num_reqs, WORLD: tl.constexpr, WP: tl.constexpr
):
    b = tl.program_id(0).to(tl.int64)
    tl.store(TOKENS + b * tok_stride + step, _winner(PREV, b, num_reqs, WORLD, WP))


def markov_sample(
    base: torch.Tensor,
    context: torch.Tensor,
    w_state: torch.Tensor,
    w1: torch.Tensor,
    fold_weight: torch.Tensor,
    fold_bias: torch.Tensor,
    tokens: torch.Tensor,
    anchors: torch.Tensor,
    anchor_stride: int,
    idx_map: torch.Tensor,
    sample_pos: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    dense: torch.Tensor | None,
    num_spec: int,
    num_valid: int,
    id_base: int,
    rank: int,
    world: int,
    softcap: float,
    use_fp64: bool,
    all_reduce,
) -> None:
    """Shard-local recurrent Markov head sampling `steps` draft tokens into `tokens[:reqs, :steps]`."""
    steps, num_reqs, width = context.shape
    rank_r = width // 3
    shard = base.shape[1]
    assert base.shape[0] == num_reqs * steps and base.stride(1) == 1
    assert context.is_contiguous() and context.dtype == torch.float32
    assert w_state.shape == (3 * rank_r, 2 * rank_r) and w_state.is_contiguous()
    assert w1.is_contiguous() and fold_weight.is_contiguous()
    assert fold_weight.shape == (shard, rank_r) and fold_bias.shape == (shard,)
    assert rank_r & (rank_r - 1) == 0 and tokens.dtype == torch.int64
    assert steps <= num_spec and id_base + shard < (1 << 18)
    if not num_reqs or not steps:
        return
    device = base.device
    n_chunks = triton.cdiv(shard, VOCAB_ROWS)
    state = torch.zeros((steps + 1, num_reqs, rank_r), dtype=torch.float32, device=device)
    latent = torch.empty((num_reqs, rank_r), dtype=w_state.dtype, device=device)
    part_v = torch.empty(
        (num_reqs, n_chunks),
        dtype=torch.float64 if use_fp64 else torch.float32,
        device=device,
    )
    part_i = torch.empty((num_reqs, n_chunks), dtype=torch.int32, device=device)
    has_dense = dense is not None
    if has_dense:
        assert dense.shape[1] == shard and dense.stride(1) == 1
        assert dense.dtype == torch.float32 and dense.shape[0] % num_spec == 0
    wp = triton.next_power_of_2(world)
    units = 8
    prev = latent  # unused at step 0
    for step in range(steps):
        _markov_rnn_kernel[(num_reqs, rank_r // units)](
            state,
            latent,
            context,
            w_state,
            w1,
            prev,
            tokens,
            tokens.stride(0),
            anchors,
            anchor_stride,
            step,
            num_reqs,
            RANK=rank_r,
            U=units,
            WORLD=world,
            WP=wp,
            num_warps=4,
        )
        _markov_vocab_kernel[(n_chunks,)](
            base,
            base.stride(0),
            fold_bias,
            fold_weight,
            latent,
            dense if has_dense else base,
            dense.stride(0) if has_dense else 0,
            part_v,
            part_i,
            idx_map,
            sample_pos,
            temperature,
            seeds,
            step,
            num_reqs,
            num_valid,
            id_base,
            n_chunks,
            softcap,
            steps,
            S_ALL=num_spec,
            RANK=rank_r,
            ROWS=VOCAB_ROWS,
            HAS_DENSE=has_dense,
            USE_FP64=use_fp64,
            num_warps=4,
        )
        prev = pick_pack(part_v, part_i, rank, world)
        if world > 1:
            prev = all_reduce(prev)
    _markov_last_kernel[(num_reqs,)](
        prev,
        tokens,
        tokens.stride(0),
        steps - 1,
        num_reqs,
        WORLD=world,
        WP=wp,
        num_warps=1,
    )


@triton.jit
def _argmax_chunk_kernel(
    LOGITS,
    stride,
    PART_V,
    PART_I,
    num_valid,
    id_base,
    n_chunks,
    softcap,
    CHUNK: tl.constexpr,
    HAS_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    m = cols < num_valid
    x = tl.load(LOGITS + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    if HAS_CAP:
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
    x = tl.where(m, x, float("-inf"))
    value, idx = tl.max(x, axis=0, return_indices=True)
    tl.store(PART_V + row * n_chunks + chunk, value)
    tl.store(PART_I + row * n_chunks + chunk, id_base + chunk * CHUNK + idx)


def shard_argmax(
    logits: torch.Tensor,
    softcap: float | None,
    num_valid: int,
    id_base: int,
    rank: int,
    world: int,
    all_reduce,
) -> torch.Tensor:
    """Shard-local greedy argmax over raw FP32 `(rows, shard)` logits without vocab all-gather."""
    rows, shard = logits.shape
    assert logits.stride(1) == 1 and id_base + shard < (1 << 18)
    n_chunks = triton.cdiv(shard, ARGMAX_CHUNK)
    part_v = torch.empty((rows, n_chunks), dtype=torch.float32, device=logits.device)
    part_i = torch.empty((rows, n_chunks), dtype=torch.int32, device=logits.device)
    if rows:
        _argmax_chunk_kernel[(rows, n_chunks)](
            logits,
            logits.stride(0),
            part_v,
            part_i,
            num_valid,
            id_base,
            n_chunks,
            softcap if softcap is not None else 1.0,
            CHUNK=ARGMAX_CHUNK,
            HAS_CAP=softcap is not None,
            num_warps=4,
        )
    packed = pick_pack(part_v, part_i, rank, world)
    if world > 1:
        packed = all_reduce(packed)
    return unpack_tokens(packed)
