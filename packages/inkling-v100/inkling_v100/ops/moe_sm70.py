# SPDX-License-Identifier: Apache-2.0
"""Exact native-precision Inkling2 MoE ops for Volta sm_70 (EP=4, zero lossy packing).

- Router: FP16 x @ W^T -> FP32 logits + sigmoid + bias top-k + renormalized weights * route_scale.
- L0-19 NVFP4 experts: zero-sync EP=4 sentinel routing into 1Cat-vLLM's compiled
  `torch.ops._C.skinny_moe_qpn_sm70` (native Volta mma.m8n8k4 W4A16).
- Activation: FP16-safe interleaved (gate, up) softcapped SiTU using `tldevice.tanh`
  (legal on sm_70, unlike sm_75+ `tanh.approx.f32`).
- L20-23 MXFP8 experts: exact UE8M0 x E4M3FN -> FP16 dequant + cuBLAS FP16 GEMM (Phase 0),
  upgradable to fused sm_70 CUDA in Phase 1.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    grouped_splitk,
    qpn_prepack,
)
from vllm.triton_utils import tl, tldevice, triton

from .fp8_sm70 import e4m3_u8_to_f16_scaled

_CUDA_GRID_Y_LIMIT = 65535


@triton.jit
def _sum8(x):
    low, high = tl.split(tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2)), (0, 1, 3, 2)))
    low, high = tl.split(low + high)
    low, high = tl.split(low + high)
    return low + high


@triton.jit
def _selection_keys(logits, bias, experts, E: tl.constexpr):
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-logits))
    selection = sigmoid + bias
    bits = selection.to(tl.uint32, bitcast=True)
    bits = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    keys = (bits.to(tl.uint64) << 16) | (E - experts).to(tl.uint64)
    return tl.where(experts < E, keys, 0)


@triton.jit
def _log_sigmoid(logits):
    return tl.minimum(logits, 0.0) - tldevice.log1p(tldevice.exp(-tl.abs(logits)))


@triton.jit
def _routing_weights(log_probs, route_scale, TOPK: tl.constexpr):
    max_log_probs = tl.max(log_probs, 1)
    shifted = tldevice.exp(log_probs - max_log_probs[:, None])
    total = _sum8(shifted) if TOPK == 8 else tl.sum(shifted, 1)
    logsumexp = max_log_probs + tldevice.log(total)
    return tldevice.exp(log_probs - logsumexp[:, None]) * route_scale


@triton.jit
def _select_kernel_sm70(
    Logits,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    token = tl.program_id(0) + tl.arange(0, 1)
    offs = tl.arange(0, E_POW2)[None, :]
    mask = offs < E
    bias = tl.load(Bias + offs, mask, other=0.0)
    logits = tl.load(Logits + token[:, None] * E + offs, mask, other=0.0).to(tl.float32)
    keys = _selection_keys(logits, bias, offs, E)
    slots = tl.arange(0, TOPK)[None, :]
    ids = tl.zeros((1, TOPK), tl.int32)
    selected = tl.zeros((1, TOPK), tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(keys, 1)[:, None]
        best_id = E - (best & 0xFFFF).to(tl.int32)
        hit = keys == best
        best_logit = tl.max(tl.where(hit, logits, float("-inf")), 1)[:, None]
        ids = tl.where(slots == k, best_id, ids)
        selected = tl.where(slots == k, best_logit, selected)
        keys = tl.where(hit, 0, keys)

    weights = _routing_weights(_log_sigmoid(selected), route_scale, TOPK)
    lo = tl.load(ExpertBounds)
    hi = tl.load(ExpertBounds + 1)
    local = tl.where((ids >= lo) & (ids < hi), ids - lo, -1)
    if HAS_PADDING:
        pad = tl.load(IsPadding + token)
        local = tl.where(pad[:, None], -1, local)
    rows = token[:, None] * TOPK + slots
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))


def select_experts_sm70(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    logits = torch.mm(x, gate_weight.t(), out_dtype=torch.float32)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    if tokens:
        _select_kernel_sm70[(tokens,)](
            logits,
            bias,
            expert_bounds,
            is_padding if is_padding is not None else ids,
            ids,
            weights,
            src2dst,
            tokens,
            route_scale,
            E=experts,
            E_POW2=triton.next_power_of_2(experts),
            TOPK=topk,
            HAS_PADDING=is_padding is not None,
            num_warps=4,
        )
    return ids, weights, src2dst


@triton.jit
def _situ_interleaved_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """Exact softcapped interleaved SiTU for FP16/BF16/FP32 inputs on sm_70.

    `GateUp` has interleaved columns `(gate_0, up_0, gate_1, up_1, ...)` of width `2*N`.
    Computes `softcap(gate, GATE_CAP) * sigmoid(gate) * softcap(up, UP_CAP) * gamma`
    in FP32 using sm_70-safe `tldevice.tanh` (NOT `tanh.approx.f32`).
    """
    row = tl.program_id(0).to(tl.int64)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    cols = start + tl.arange(0, BLOCK)
    base = GateUp + row * (2 * N)
    gate = tl.load(base + 2 * cols).to(tl.float32)
    up = tl.load(base + 2 * cols + 1).to(tl.float32)
    gamma = tl.load(Weights + row).to(tl.float32)
    sig = tl.div_rn(1.0, 1.0 + tldevice.exp(-gate))
    if GATE_CAP > 0:
        gate = GATE_CAP * tldevice.tanh(tl.div_rn(gate, GATE_CAP))
    if UP_CAP > 0:
        up = UP_CAP * tldevice.tanh(tl.div_rn(up, UP_CAP))
    val = ((gate * sig) * up) * gamma
    tl.store(Out + row * N + cols, val.to(Out.dtype.element_ty))


def situ_interleaved(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    rows, n2 = gate_up.shape
    n = n2 // 2
    out = torch.empty((rows, n), dtype=gate_up.dtype, device=gate_up.device)
    if rows == 0:
        return out
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    _situ_interleaved_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def ep4_route_to_skinny_groups(
    ids: torch.Tensor, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-sync, CUDA-graph-safe EP=4 sentinel routing for `skinny_moe_qpn_sm70`.

    Maps non-local `-1` slots to sentinel `num_local_experts`, sorts stably so all local
    slots occupy `[0, n_local)`, and caps `goff` at `n_local` so every sentinel/unused
    group has zero length and is skipped by `skinny_moe_qpn_sm70`.
    """
    flat_in = ids.reshape(-1)
    S = flat_in.numel()
    device = ids.device
    flat = torch.where(flat_in >= 0, flat_in, num_local_experts).to(torch.int64)
    perm = torch.argsort(flat, stable=True).to(torch.int32)
    sorted_e = flat[perm.long()]
    valid = sorted_e < num_local_experts
    n_local = valid.sum().to(torch.int64)
    new_grp = torch.zeros(S, dtype=torch.bool, device=device)
    new_grp[0] = valid[0]
    new_grp[1:] = valid[1:] & (sorted_e[1:] != sorted_e[:-1])
    gidx = torch.cumsum(new_grp, 0) - 1
    goff = torch.full((S + 1,), S, dtype=torch.int64, device=device)
    pos = torch.arange(S, dtype=torch.int64, device=device)
    valid_idx = torch.where(valid, gidx, S)
    goff.scatter_reduce_(0, valid_idx, pos, reduce="amin")
    goff = torch.minimum(goff, n_local)
    gids = torch.zeros(S, dtype=torch.int64, device=device)
    gids.scatter_(0, torch.where(valid, gidx, S - 1), torch.where(valid, sorted_e, 0))
    return perm, gids.to(torch.int32), goff.to(torch.int32)


def run_nvfp4_experts_sm70(
    x: torch.Tensor,
    w13_qpn: torch.Tensor,
    s13_qpn: torch.Tensor,
    g13_t: torch.Tensor,
    w2_qpn: torch.Tensor,
    s2_qpn: torch.Tensor,
    g2_t: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    precomputed_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run exact native NVFP4 experts via `torch.ops._C.skinny_moe_qpn_sm70` (EP=4 safe)."""
    num_tokens, hidden = x.shape
    topk = ids.shape[1]
    num_local = w13_qpn.shape[0]
    n13 = w13_qpn.shape[1]
    inter = n13 // 2
    n2 = w2_qpn.shape[1]
    assert x.dtype == torch.float16
    down_all = torch.empty((num_tokens * topk, n2), dtype=torch.float16, device=x.device)
    if num_tokens == 0:
        return down_all

    # Chunk tokens to obey skinny_moe_qpn_sm70's grid-y (65535) and transient memory cap.
    bytes_per_token = topk * (2 * (n13 + inter + n2))
    chunk = min(_CUDA_GRID_Y_LIMIT // topk, max(1, (128 << 20) // max(bytes_per_token, 1)))
    sk13 = grouped_splitk(hidden, 16)
    sk2 = grouped_splitk(inter, 8)
    for t0 in range(0, num_tokens, chunk):
        t1 = min(t0 + chunk, num_tokens)
        tc = t1 - t0
        xc = x[t0:t1].contiguous()
        ids_c = ids[t0:t1]
        w_c = weights[t0 * topk : t1 * topk]
        if precomputed_route is not None and t0 == 0 and t1 == num_tokens:
            perm, gids32, goff32 = precomputed_route
            s2d_c = src2dst
        else:
            s2d_c = torch.where(
                ids_c.reshape(-1) >= 0,
                torch.arange(tc * topk, dtype=torch.int32, device=x.device),
                -1,
            )
            perm, gids32, goff32 = ep4_route_to_skinny_groups(ids_c, num_local)
        y13 = torch.empty((tc * topk, n13), dtype=torch.float16, device=x.device)
        torch.ops._C.skinny_moe_qpn_sm70(
            xc, w13_qpn, s13_qpn, g13_t, perm, gids32, goff32, topk, y13,
            False, tc, sk13, 1, 0,
        )
        inter_act = situ_interleaved(y13, w_c, s2d_c, gate_cap, up_cap)
        del y13
        y2 = down_all[t0 * topk : t1 * topk]
        torch.ops._C.skinny_moe_qpn_sm70(
            inter_act, w2_qpn, s2_qpn, g2_t, perm, gids32, goff32, topk, y2,
            True, tc, sk2, 1, 0,
        )
    return down_all


@triton.jit
def _mxfp8_dequant_kernel(
    W_u8,
    S_u8,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Exact UE8M0 x E4M3FN -> FP16 dequantization of one [N, K] MXFP8 expert weight.

    Uses exact 3-op E4M3FN bit-unpack (`val * 2^-8`) combined in FP32 with UE8M0 exponent
    `2^(eb - 127 + 8) = 2^(eb - 119)` (or `exp2(eb - 119)`) so the full UE8M0 dynamic range
    is preserved cleanly before rounding once to FP16.
    """
    n_idx = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    G: tl.constexpr = K // 32

    # Process each 32-element group with bit-exact IEEE FP32 power-of-two scales.
    # `e4m3_u8_to_f16_scaled` returns `w * 2^-8`.
    # UE8M0 byte `s` represents `2^(s - 127)` (s == 255 is NaN -> 0).
    # For s <= 246: `2^(s - 127) * 2^8 = 2^(s - 119)` -> exact FP32 bit pattern `(s + 8) << 23`!
    # For 247 <= s <= 254: multiply by `256.0 * bitcast(s << 23)`.
    k_sub = tl.arange(0, 32)
    for gi in tl.static_range(BLOCK_G):
        g_cur = tl.program_id(1) * BLOCK_G + gi
        if g_cur < G:
            s_col = tl.load(S_u8 + n_idx * G + g_cur, mask=n_mask, other=127).to(tl.int32)
            k_offs = g_cur * 32 + k_sub
            w_u8 = tl.load(
                W_u8 + n_idx[:, None] * K + k_offs[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            w_f16_s = e4m3_u8_to_f16_scaled(w_u8).to(tl.float32)
            hi_exp = (s_col >= 247) & (s_col < 255)
            sf_lo = ((tl.minimum(s_col, 246) + 8) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
            sf_hi = ((tl.minimum(s_col, 254)) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
            val_f32 = tl.where(
                s_col[:, None] == 255,
                0.0,
                tl.where(hi_exp[:, None], (w_f16_s * 256.0) * sf_hi[:, None], w_f16_s * sf_lo[:, None]),
            )
            tl.store(
                Out + n_idx[:, None] * K + k_offs[None, :],
                val_f32.to(tl.float16),
                mask=n_mask[:, None],
            )


def mxfp8_dequant_expert(w_u8: torch.Tensor, s_u8: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize one [N, K] MXFP8 expert (`uint8` E4M3 codes, `[N, K/32]` `uint8` UE8M0) -> FP16."""
    N, K = w_u8.shape
    if out is None:
        out = torch.empty((N, K), dtype=torch.float16, device=w_u8.device)
    BLOCK_N = 16
    BLOCK_G = 4
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K // 32, BLOCK_G))
    _mxfp8_dequant_kernel[grid](
        w_u8, s_u8, out, N=N, K=K, BLOCK_N=BLOCK_N, BLOCK_G=BLOCK_G, num_warps=4
    )
    return out


def run_mxfp8_experts_sm70(
    x: torch.Tensor,
    w13_u8: torch.Tensor,
    s13_u8: torch.Tensor,
    w2_u8: torch.Tensor,
    s2_u8: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    scratch_w13: torch.Tensor | None = None,
    scratch_w2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase-0 exact MXFP8 experts (L20-23, 4/48 visits): dequant active experts -> FP16 cuBLAS.

    Uses zero host sync inside graph capture? Note: Phase-0 uses active-expert loop (eager/piecewise);
    Phase-1 replaces this with a zero-sync grouped CUDA kernel (`mlp_mxfp8_moe.cu`).
    """
    num_tokens, hidden = x.shape
    topk = ids.shape[1]
    n13 = w13_u8.shape[1]
    inter = n13 // 2
    n2 = w2_u8.shape[1]
    down_all = torch.empty((num_tokens * topk, n2), dtype=torch.float16, device=x.device)
    if num_tokens == 0:
        return down_all

    flat_ids = ids.reshape(-1)
    local_mask = flat_ids >= 0
    if not bool(local_mask.any()):
        return down_all

    w13_buf = scratch_w13 if scratch_w13 is not None else torch.empty((n13, hidden), dtype=torch.float16, device=x.device)
    w2_buf = scratch_w2 if scratch_w2 is not None else torch.empty((n2, inter), dtype=torch.float16, device=x.device)

    local_slots = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
    local_e = flat_ids[local_slots]
    order = torch.argsort(local_e, stable=True)
    sorted_slots = local_slots[order]
    sorted_e = local_e[order]
    uniq_e, counts = torch.unique_consecutive(sorted_e, return_counts=True)
    uniq_list = uniq_e.tolist()
    cnt_list = counts.tolist()

    pos = 0
    for e, cnt in zip(uniq_list, cnt_list):
        slots_e = sorted_slots[pos : pos + cnt]
        pos += cnt
        tok_idx = slots_e // topk
        xe = x[tok_idx]
        mxfp8_dequant_expert(w13_u8[e], s13_u8[e], w13_buf)
        y13 = torch.mm(xe, w13_buf.t())
        s2d_e = torch.arange(cnt, dtype=torch.int32, device=x.device)
        act_e = situ_interleaved(y13, weights[slots_e], s2d_e, gate_cap, up_cap)
        mxfp8_dequant_expert(w2_u8[e], s2_u8[e], w2_buf)
        down_all[slots_e] = torch.mm(act_e, w2_buf.t())
    return down_all


@triton.jit
def _combine_kernel_sm70(
    Down,
    Src2Dst,
    Out,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(TOPK):
        row = tl.load(Src2Dst + token * TOPK + slot).to(tl.int64)
        acc += tl.load(Down + row * H + cols, mask & (row >= 0), other=0.0).to(tl.float32)
    tl.store(Out + token * H + cols, acc.to(Out.dtype.element_ty), mask=mask)


def combine_sm70(
    down: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    tokens = src2dst.numel() // topk
    hidden = down.shape[1]
    out = torch.empty((tokens, hidden), dtype=out_dtype, device=down.device)
    if tokens == 0:
        return out
    block = 256
    _combine_kernel_sm70[(tokens, triton.cdiv(hidden, block))](
        down, src2dst, out, H=hidden, TOPK=topk, BLOCK=block, num_warps=4
    )
    return out
