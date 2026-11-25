# SPDX-License-Identifier: Apache-2.0
"""Comprehensive GPU unit tests for inkling_v100 sm_70 kernels & checkpoint mapping.

Run on a V100 (sm_70) GPU:
    CUDA_VISIBLE_DEVICES=0 python3 -m pytest -s tests/test_ops_sm70.py
or directly:
    CUDA_VISIBLE_DEVICES=0 python3 tests/test_ops_sm70.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

from inkling_v100.ops.attention_sm70 import (
    MODE_GATED,
    MODE_OUT_LSE,
    compute_log_scaling_tau,
    gate_sink,
    rel_attention,
    rel_decode_attention,
)
from inkling_v100.ops.attn_res import attn_res
from inkling_v100.ops.fp8_sm70 import (
    e4m3_u8_to_f16,
    e4m3_u8_to_f16_scaled,
    f32_to_e4m3_u8,
)
from inkling_v100.ops.loop import blend_rms_norm, pair_rms_norm
from inkling_v100.ops.moe_sm70 import (
    combine_sm70,
    mxfp8_dequant_expert,
    qpn_prepack,
    run_mxfp8_experts_sm70,
    run_nvfp4_experts_sm70,
    select_experts_sm70,
    situ_interleaved,
)
from inkling_v100.ops.norm import rms_norm
from inkling_v100.ops.sconv import (
    attention_prologue,
    sconv_seq_metadata,
    shift_rel_proj,
    short_conv,
)

DEVICE = torch.device("cuda:0")


@triton.jit
def _test_unpack_kernel(In, Out1, Out2, N: tl.constexpr):
    i = tl.arange(0, N)
    b = tl.load(In + i)
    tl.store(Out1 + i, e4m3_u8_to_f16_scaled(b))
    tl.store(Out2 + i, e4m3_u8_to_f16(b))


@triton.jit
def _test_pack_kernel(In, Out, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(In + i, mask=m, other=0.0)
    tl.store(Out + i, f32_to_e4m3_u8(x), mask=m)


def test_fp8_sm70() -> None:
    print("[1/6] test_fp8_sm70 ...", flush=True)
    all_bytes = torch.arange(256, dtype=torch.uint8, device=DEVICE)
    out_scaled = torch.empty(256, dtype=torch.float16, device=DEVICE)
    out_full = torch.empty(256, dtype=torch.float16, device=DEVICE)
    _test_unpack_kernel[(1,)](all_bytes, out_scaled, out_full, N=256)
    ref = all_bytes.view(torch.float8_e4m3fn).to(torch.float16)
    non_nan = (all_bytes != 0x7F) & (all_bytes != 0xFF)
    assert torch.equal(out_full[non_nan].view(torch.uint16), ref[non_nan].view(torch.uint16)), (
        f"e4m3_u8_to_f16 mismatch!"
    )
    scaled_ref = (ref[non_nan].float() / 256.0).half()
    assert torch.equal(out_scaled[non_nan].view(torch.uint16), scaled_ref.view(torch.uint16)), (
        f"e4m3_u8_to_f16_scaled mismatch!"
    )

    # Test f32_to_e4m3_u8 across all critical points + 100K random values
    torch.manual_seed(0)
    # All exact representable values + midpoints + subnormals + boundaries + >448
    rep = ref[non_nan].float()
    rep_pos = torch.sort(rep[rep >= 0].unique()).values
    mids = 0.5 * (rep_pos[:-1] + rep_pos[1:])
    eps = torch.nextafter(mids, torch.full_like(mids, float("inf")))
    em = torch.nextafter(mids, torch.full_like(mids, float("-inf")))
    edge = torch.tensor(
        [
            0.0, -0.0, 1e-12, -1e-12, 2**-10, -(2**-10),
            math.nextafter(2**-10, 1.0), math.nextafter(2**-10, 0.0),
            2**-9, 2**-7, 2**-6, 447.0, 448.0, 449.0, 464.0, 480.0, 1e4, -448.0, -1e4,
            float("nan"), -float("nan"),
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    rand = torch.randn(100_000, dtype=torch.float32, device=DEVICE) * 20.0
    tiny = torch.randn(20_000, dtype=torch.float32, device=DEVICE) * 0.02
    xs = torch.cat([rep, mids, -mids, eps, -eps, em, -em, edge, rand, tiny]).contiguous()
    got = torch.empty(xs.numel(), dtype=torch.uint8, device=DEVICE)
    _test_pack_kernel[(triton.cdiv(xs.numel(), 256),)](xs, got, xs.numel(), BLOCK=256, enable_fp_fusion=False)
    # Clamp xs to [-448, 448] for PyTorch reference (since PyTorch maps >448 to NaN or 448 depending on version)
    xs_clamped = torch.where(torch.isnan(xs), xs, xs.clamp(-448.0, 448.0))
    ref_u8 = xs_clamped.to(torch.float8_e4m3fn).view(torch.uint8)
    is_nan = torch.isnan(xs)
    # Non-NaN: exact byte equality (allow ±0 sign bit)
    diff = (got != ref_u8) & (~is_nan)
    if diff.any():
        idx = diff.nonzero(as_tuple=False)[:10].squeeze(-1)
        for k in idx.tolist():
            print(f"  DIFF x={xs[k].item()!r} got=0x{got[k].item():02x} ref=0x{ref_u8[k].item():02x}")
        raise AssertionError(f"f32_to_e4m3_u8 had {int(diff.sum())} mismatches")
    assert ((got[is_nan] & 0x7F) == 0x7F).all(), "NaN not encoded as 0x7F/0xFF"
    print("  PASS: all 256 E4M3 codes + 120K+ FP32->E4M3FN RNE conversions bit-exact!")


def test_norms_attnres_loop() -> None:
    print("[2/6] test_norms_attnres_loop ...", flush=True)
    torch.manual_seed(1)
    D = 2304
    eps = 1e-6

    # rms_norm
    x = torch.randn(17, D, dtype=torch.float32, device=DEVICE)
    w = torch.randn(D, dtype=torch.float16, device=DEVICE)
    y = rms_norm(x, w, eps, out_dtype=torch.float16)
    ref = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()).half()
    err = (y.float() - ref.float()).abs().max().item()
    assert err < 2e-3, f"rms_norm err={err}"

    # attn_res: both row kernel (T*Q <= 256) and tiled kernel (T*Q > 256)
    for T, Q, N_stack, has_add, has_norm in [
        (4, 1, 2, False, True),
        (4, 3, 3, True, True),
        (4, 4, 1, False, False),
        (300, 1, 3, True, True),
        (100, 3, 2, False, False),
    ]:
        stack = torch.randn(N_stack, T, D, dtype=torch.float32, device=DEVICE) * 5.0
        last = torch.randn(T, D, dtype=torch.float32, device=DEVICE) * 5.0
        addend = torch.randn(T, D, dtype=torch.float32, device=DEVICE) * 3.0 if has_add else None
        queries = torch.randn(Q, D, dtype=torch.float32, device=DEVICE) * (D**-0.5)
        nw = torch.randn(Q, D, dtype=torch.float16, device=DEVICE) if has_norm else None

        got = attn_res(stack, last, addend, queries, nw, eps, normalize_values=True)
        srcs = [*stack.unbind(0), last + addend if has_add else last]
        S = torch.stack(srcs, dim=0)  # [N+1, T, D]
        rstd = torch.rsqrt(S.square().mean(-1, keepdim=True) + eps)  # [N+1, T, 1]
        S_norm = S * rstd
        logits = torch.einsum("ntd,qd->qnt", S_norm, queries)
        probs = torch.softmax(logits, dim=1)
        mixed = torch.einsum("qnt,ntd->qtd", probs, S_norm)
        if has_norm:
            out_rstd = torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
            ref_ar = (mixed * out_rstd * nw[:, None, :].float()).half()
            assert got.dtype == torch.float16
            tol = 5e-3
        else:
            ref_ar = mixed
            assert got.dtype == torch.float32
            tol = 1e-4
        err = (got.float() - ref_ar.float()).abs().max().item()
        assert err < tol, f"attn_res T={T} Q={Q} err={err}"

    # pair_rms_norm & blend_rms_norm (broadcast dest & full dest)
    for broadcast in (False, True):
        S, T = 3, 19
        dest = (
            torch.randn(1, T, D, dtype=torch.float32, device=DEVICE).expand(S, T, D)
            if broadcast
            else torch.randn(S, T, D, dtype=torch.float32, device=DEVICE)
        )
        msg = torch.randn(S, T, D, dtype=torch.float32, device=DEVICE)
        pair = pair_rms_norm(dest, msg, out_dtype=torch.float16)
        d_n = dest * torch.rsqrt(dest.square().mean(-1, keepdim=True) + eps)
        m_n = msg * torch.rsqrt(msg.square().mean(-1, keepdim=True) + eps)
        ref_pair = torch.cat([d_n, m_n], dim=-1).half()
        assert (pair.float() - ref_pair.float()).abs().max().item() < 2e-3

        gf = torch.randn(S, T, 2 * D, dtype=torch.float32, device=DEVICE)
        nw = (
            torch.randn(1, D, dtype=torch.float16, device=DEVICE).expand(S, D)
            if broadcast
            else torch.randn(S, D, dtype=torch.float16, device=DEVICE)
        )
        got_b = blend_rms_norm(dest, msg, gf, nw, eps, out_dtype=torch.float16)
        gain = torch.tanh(gf[..., :D])
        forget = torch.tanh(gf[..., D:])
        blended = (1.0 - forget) * dest + gain * msg
        ref_b = (
            blended * torch.rsqrt(blended.square().mean(-1, keepdim=True) + eps) * nw[:, None, :].float()
        ).half()
        err = (got_b.float() - ref_b.float()).abs().max().item()
        assert err < 5e-3, f"blend_rms_norm broadcast={broadcast} err={err}"
    print("  PASS: rms_norm, attn_res (row+tiled, FP32/FP16), pair_rms_norm, blend_rms_norm!")


def _ref_sconv_static(seq_x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact reference for Inkling2 static depthwise causal conv:
    `out[t] = (x[t] + sum_{k=0..W-1} w[:, k] * x[t - (W-1-k)]) / sqrt(W + 1)`."""
    T, C = seq_x.shape
    W = w.shape[1]
    scale = (W + 1) ** -0.5
    out = torch.zeros(T, C, dtype=torch.float32, device=seq_x.device)
    xf = seq_x.half().float()
    wf = w.half().float()
    for t in range(T):
        acc = xf[t].clone()
        for k in range(W):
            lag = W - 1 - k
            if t - lag >= 0:
                acc += xf[t - lag] * wf[:, k]
        out[t] = acc * scale
    return out


def _ref_sconv_dynamic(
    seq_x: torch.Tensor, static_w: torch.Tensor, dyn: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Exact reference for global-layer dynamic K/V conv:
    `out[t] = (x[t] + sum_{lag=0..W-1} (static_w[lag] + dyn[t, head, lag]) * x[t - lag]) / sqrt(2*W + 1)`."""
    T, C = seq_x.shape
    W = static_w.shape[0]
    H = C // head_dim
    scale = (2 * W + 1) ** -0.5
    out = torch.zeros(T, C, dtype=torch.float32, device=seq_x.device)
    xf = seq_x.half().float()
    wf = static_w.float()
    df = dyn.float().view(T, H, W)
    for t in range(T):
        acc = xf[t].clone()
        for lag in range(W):
            if t - lag >= 0:
                w_eff = wf[lag] + df[t, :, lag].repeat_interleave(head_dim)
                acc += xf[t - lag] * w_eff
        out[t] = acc * scale
    return out


def test_sconv_and_prologue() -> None:
    print("[3/6] test_sconv_and_prologue ...", flush=True)
    torch.manual_seed(2)
    W = 4
    H_kv = 2
    D = 128
    H_q = 6
    D_rel = 16
    E = 1024
    C_hid = 2304
    ws_hid = C_hid // H_kv
    head_size = 1 << (2 * D + 2 * ws_hid - 1).bit_length()  # 4096

    # 2 requests: req 0 has 13 tokens (prefill 10 + decode 3), req 1 has 9 tokens (prefill 6 + decode 3)
    lens_pf = [10, 6]
    lens_dec = [3, 3]
    total_lens = [a + b for a, b in zip(lens_pf, lens_dec)]

    x_seqs = [torch.randn(L, C_hid, dtype=torch.float16, device=DEVICE) for L in total_lens]
    w_sconv = torch.randn(C_hid, W, dtype=torch.float16, device=DEVICE) * 0.25
    add_seqs = [torch.randn(L, C_hid, dtype=torch.float32, device=DEVICE) for L in total_lens]
    res_seqs = [torch.randn(L, C_hid, dtype=torch.float32, device=DEVICE) for L in total_lens]

    # Allocate paged conv cache: block_size = W = 4
    max_blks = 8
    conv_cache = torch.zeros((2 * max_blks, H_kv, W, head_size), dtype=torch.float16, device=DEVICE)
    block_table = torch.tensor(
        [list(range(0, max_blks)), list(range(max_blks, 2 * max_blks))],
        dtype=torch.int32,
        device=DEVICE,
    )

    # Step 1: Prefill batch
    pf_x = torch.cat([x_seqs[r][: lens_pf[r]] for r in range(2)], dim=0)
    pf_add = torch.cat([add_seqs[r][: lens_pf[r]] for r in range(2)], dim=0)
    pf_res = torch.cat([res_seqs[r][: lens_pf[r]] for r in range(2)], dim=0)
    pf_pos = torch.cat([torch.arange(lens_pf[r], dtype=torch.int32, device=DEVICE) for r in range(2)])
    pf_qsl = torch.tensor([0, lens_pf[0], sum(lens_pf)], dtype=torch.int32, device=DEVICE)
    pf_slots = torch.cat(
        [
            block_table[r, torch.arange(lens_pf[r], device=DEVICE) // W] * W
            + (torch.arange(lens_pf[r], device=DEVICE) % W)
            for r in range(2)
        ]
    ).to(torch.int64)
    seq_idx = torch.empty(sum(lens_pf), dtype=torch.int32, device=DEVICE)
    qstart = torch.empty(sum(lens_pf), dtype=torch.int32, device=DEVICE)
    sconv_seq_metadata(pf_qsl, 2, sum(lens_pf), seq_idx, qstart)

    pf_out = torch.empty(sum(lens_pf), C_hid, dtype=torch.float32, device=DEVICE)
    short_conv(
        pf_x, w_sconv, conv_cache, pf_pos, block_table, seq_idx, pf_slots, qstart,
        2 * D, ws_hid, addend=pf_add, residual=pf_res, out=pf_out,
    )

    # Step 2: 3 sequential decode steps reading cached history across block boundaries
    dec_outs = [[], []]
    for step in range(3):
        dx = torch.stack([x_seqs[r][lens_pf[r] + step] for r in range(2)], dim=0)
        dadd = torch.stack([add_seqs[r][lens_pf[r] + step] for r in range(2)], dim=0)
        dres = torch.stack([res_seqs[r][lens_pf[r] + step] for r in range(2)], dim=0)
        dpos = torch.tensor([lens_pf[0] + step, lens_pf[1] + step], dtype=torch.int32, device=DEVICE)
        dqsl = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
        dslots = torch.tensor(
            [
                int(block_table[r, (lens_pf[r] + step) // W]) * W + ((lens_pf[r] + step) % W)
                for r in range(2)
            ],
            dtype=torch.int64,
            device=DEVICE,
        )
        dsi = torch.empty(2, dtype=torch.int32, device=DEVICE)
        dqs = torch.empty(2, dtype=torch.int32, device=DEVICE)
        sconv_seq_metadata(dqsl, 2, 2, dsi, dqs)
        dout = torch.empty(2, C_hid, dtype=torch.float32, device=DEVICE)
        short_conv(
            dx, w_sconv, conv_cache, dpos, block_table, dsi, dslots, dqs,
            2 * D, ws_hid, addend=dadd, residual=dres, out=dout,
        )
        dec_outs[0].append(dout[0])
        dec_outs[1].append(dout[1])

    for r in range(2):
        ref = _ref_sconv_static(x_seqs[r], w_sconv) + add_seqs[r] + res_seqs[r]
        got = torch.cat(
            [pf_out[pf_qsl[r] : pf_qsl[r + 1]], torch.stack(dec_outs[r], dim=0)], dim=0
        )
        err = (got - ref).abs().max().item()
        assert err < 3e-3, f"short_conv req {r} err={err}"

    # Now test attention_prologue for BOTH (local static, global dynamic) x (FP16 KV, FP8 KV)
    for is_local in (True, False):
        for kv_dtype in (torch.float16, torch.float8_e4m3fn):
            T = 17
            q = torch.randn(T, H_q, D, dtype=torch.float16, device=DEVICE) * 2.0
            r = torch.randn(T, H_q, D_rel, dtype=torch.float16, device=DEVICE) * 2.0
            k = torch.randn(T, H_kv * D, dtype=torch.float16, device=DEVICE) * 2.0
            v = torch.randn(T, H_kv * D, dtype=torch.float16, device=DEVICE) * 2.0
            q_scale = torch.randn(H_q, D, dtype=torch.float16, device=DEVICE) * 0.5 + 1.0
            r_scale = torch.randn(H_q, D_rel, dtype=torch.float16, device=DEVICE) * 0.5 + 1.0
            rel_proj = torch.randn(D_rel, E, dtype=torch.float16, device=DEVICE) * 0.1
            rel_shifted = shift_rel_proj(rel_proj, is_local)
            positions = torch.arange(T, dtype=torch.int32, device=DEVICE)
            tau = None if is_local else compute_log_scaling_tau(positions, 8, 0.1)

            if is_local:
                wk = torch.randn(H_kv * D, W, dtype=torch.float16, device=DEVICE) * 0.2
                wv = torch.randn(H_kv * D, W, dtype=torch.float16, device=DEVICE) * 0.2
                dk = dv = None
            else:
                wk = torch.randn(W, H_kv * D, dtype=torch.float16, device=DEVICE) * 0.2
                wv = torch.randn(W, H_kv * D, dtype=torch.float16, device=DEVICE) * 0.2
                dk = torch.randn(T, H_kv * W, dtype=torch.float16, device=DEVICE) * 0.2
                dv = torch.randn(T, H_kv * W, dtype=torch.float16, device=DEVICE) * 0.2

            cc = torch.zeros((8, H_kv, W, head_size), dtype=torch.float16, device=DEVICE)
            bt = torch.arange(8, dtype=torch.int32, device=DEVICE)[None, :]
            slots_c = torch.arange(T, dtype=torch.int64, device=DEVICE)
            qsl = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
            seq_lens = torch.tensor([T], dtype=torch.int32, device=DEVICE)
            si = torch.zeros(T, dtype=torch.int32, device=DEVICE)
            qs = torch.zeros(T, dtype=torch.int32, device=DEVICE)

            page = 16
            kc = torch.zeros((4, page, H_kv, D), dtype=kv_dtype, device=DEVICE)
            vc = torch.zeros((4, page, H_kv, D), dtype=kv_dtype, device=DEVICE)
            kv_slots = torch.arange(T, dtype=torch.int64, device=DEVICE)
            q_out = torch.empty_like(q)
            rel_out = torch.empty((T + 128, H_q, E + 256), dtype=torch.float16, device=DEVICE)

            attention_prologue(
                q, r, k, v, q_scale, r_scale, tau, rel_shifted, wk, wv, dk, dv,
                cc, positions, bt, si, slots_c, qs, kc, vc, kv_slots, q_out, rel_out,
                qsl, seq_lens, clip_threshold=22.627416997969522, rel_threshold=8.0, norm_eps=1e-6,
            )

            # Verify clipped Q
            qf = q.float() * q_scale[None].float()
            q_norm = qf.norm(dim=-1, keepdim=True)
            q_ref = (qf * (22.627416997969522 / (q_norm + 1e-6)).clamp(max=1.0)).half()
            if tau is not None:
                q_ref = (q_ref.float() * tau[:, None, None]).half()
            assert (q_out.float() - q_ref.float()).abs().max().item() < 2e-2

            # Verify K & V caches
            kc_ref_raw = _ref_sconv_static(k, wk) if is_local else _ref_sconv_dynamic(k, wk, dk, D)
            kc_ref_raw = kc_ref_raw.half().float().view(T, H_kv, D)
            kn = kc_ref_raw.norm(dim=-1, keepdim=True)
            kc_ref = kc_ref_raw * (22.627416997969522 / (kn + 1e-6)).clamp(max=1.0)
            vc_ref_raw = _ref_sconv_static(v, wv) if is_local else _ref_sconv_dynamic(v, wv, dv, D)
            vc_ref_raw = vc_ref_raw.half().float().view(T, H_kv, D)
            vc_ref = vc_ref_raw * torch.rsqrt(vc_ref_raw.square().mean(-1, keepdim=True) + 1e-6)
            kc_got = kc.view(-1, H_kv, D)[:T].to(torch.float32)
            vc_got = vc.view(-1, H_kv, D)[:T].to(torch.float32)
            if kv_dtype == torch.float8_e4m3fn:
                kc_ref_q = kc_ref.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
                vc_ref_q = vc_ref.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
                assert (kc_got - kc_ref_q).abs().max().item() <= 0.25
                assert (vc_got - vc_ref_q).abs().max().item() <= 0.15
            else:
                assert (kc_got - kc_ref).abs().max().item() < 2e-2
                assert (vc_got - vc_ref).abs().max().item() < 2e-2

            # Verify sheared rel_logits at EVERY causal pair (t, j)
            rf = r.float() * r_scale[None].float()
            rn = rf.norm(dim=-1, keepdim=True)
            r_ref = (rf * (8.0 / (rn + 1e-6)).clamp(max=1.0)).half()
            if tau is not None:
                r_ref = (r_ref.float() * tau[:, None, None]).half()
            rel_full = torch.einsum("thd,de->the", r_ref.float(), rel_proj.float())
            rel_base = ((T + 127) // 128) * 128 - (E + 256)
            for t in range(T):
                for j in range(t + 1):
                    dist = t - j
                    got_b = rel_out[t, :, j - rel_base].float()
                    ref_b = rel_full[t, :, dist]
                    assert (got_b - ref_b).abs().max().item() < 2e-2
    print("  PASS: short_conv (prefill+multi-step decode) & attention_prologue (static/dynamic x FP16/FP8)!")


def _ref_rel_attention(
    q: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    rel_proj: torch.Tensor,
    r_clipped: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    past: int,
    softmax_scale: float,
    window: int,
) -> torch.Tensor:
    """Exact FP32 eager Inkling2 relative-bias gated-sink attention for one sequence."""
    Q_len, H, D = q.shape
    S_len, H_kv, _ = k_seq.shape
    G = H // H_kv
    E = rel_proj.shape[1]
    k_exp = k_seq.float().repeat_interleave(G, dim=1)  # [S, H, D]
    v_exp = v_seq.float().repeat_interleave(G, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_exp) * softmax_scale
    rel_all = torch.einsum("qhd,de->hqe", r_clipped.float(), rel_proj.float())
    q_pos = torch.arange(past, past + Q_len, device=q.device)[:, None]
    k_pos = torch.arange(S_len, device=q.device)[None, :]
    dist = q_pos - k_pos
    visible = dist >= 0
    if window > 0:
        visible = visible & (dist < window)
    in_ext = visible & (dist < E)
    idx = dist.clamp(0, E - 1).unsqueeze(0).expand(H, Q_len, S_len)
    bias = torch.gather(rel_all, 2, idx)
    scores = scores + torch.where(in_ext.unsqueeze(0), bias, torch.zeros_like(bias))
    scores = torch.where(visible.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
    lse = torch.logsumexp(scores, dim=-1)  # [H, Q]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v_exp)  # [Q, H, D]
    out = out * torch.sigmoid(gate.float().view(Q_len, H, D))
    out = out * torch.sigmoid(lse.t() - sinks.float()).unsqueeze(-1)
    return out.reshape(Q_len, H * D)


def test_rel_attention() -> None:
    print("[4/6] test_rel_attention ...", flush=True)
    torch.manual_seed(3)
    H, H_kv, D, D_rel, E, page = 6, 2, 128, 16, 1024, 16
    scale = D**-0.5

    cases = [
        # (name, is_local, kv_dtype, req_specs=[(past, q_len)...], splits)
        ("decode_local_fp16_s4", True, torch.float16, [(1200, 1), (37, 1)], 4),
        ("decode_local_fp8_s4", True, torch.float8_e4m3fn, [(1200, 1), (37, 1)], 4),
        ("decode_global_fp8_s8", False, torch.float8_e4m3fn, [(1500, 1), (400, 1)], 8),
        ("decode_global_fp8_s1", False, torch.float8_e4m3fn, [(300, 1), (50, 1)], 1),
        ("verify_q8_local_fp8_s4", True, torch.float8_e4m3fn, [(1100, 8), (200, 8)], 4),
        ("prefill_local_fp16", True, torch.float16, [(0, 160), (64, 130)], 0),
        ("prefill_global_fp8", False, torch.float8_e4m3fn, [(0, 200), (100, 140)], 0),
    ]
    for name, is_local, kv_dtype, specs, splits in cases:
        window = E if is_local else 0
        num_reqs = len(specs)
        max_seq = max(p + q for p, q in specs)
        max_blks = (max_seq + page - 1) // page
        total_blks = num_reqs * max_blks
        kc = torch.zeros((total_blks, page, H_kv, D), dtype=kv_dtype, device=DEVICE)
        vc = torch.zeros((total_blks, page, H_kv, D), dtype=kv_dtype, device=DEVICE)
        bt = torch.arange(total_blks, dtype=torch.int32, device=DEVICE).view(num_reqs, max_blks)

        k_seqs, v_seqs = [], []
        for r_i, (past, ql) in enumerate(specs):
            sl = past + ql
            kf = torch.randn(sl, H_kv, D, dtype=torch.float16, device=DEVICE) * 0.5
            vf = torch.randn(sl, H_kv, D, dtype=torch.float16, device=DEVICE) * 0.5
            if kv_dtype == torch.float8_e4m3fn:
                kq = kf.float().to(torch.float8_e4m3fn)
                vq = vf.float().to(torch.float8_e4m3fn)
                k_seqs.append(kq.float())
                v_seqs.append(vq.float())
            else:
                kq, vq = kf, vf
                k_seqs.append(kf.float())
                v_seqs.append(vf.float())
            for pos in range(sl):
                kc[bt[r_i, pos // page], pos % page] = kq[pos]
                vc[bt[r_i, pos // page], pos % page] = vq[pos]

        cu_q = [0]
        for _, ql in specs:
            cu_q.append(cu_q[-1] + ql)
        T = cu_q[-1]
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
        seq_lens_t = torch.tensor([p + q for p, q in specs], dtype=torch.int32, device=DEVICE)

        q = torch.randn(T, H, D, dtype=torch.float16, device=DEVICE) * 0.5
        r_vec = torch.randn(T, H, D_rel, dtype=torch.float16, device=DEVICE) * 0.5
        gate = torch.randn(T, H * D, dtype=torch.float16, device=DEVICE)
        sinks = torch.randn(T, H, dtype=torch.float16, device=DEVICE)
        rel_proj = torch.randn(D_rel, E, dtype=torch.float16, device=DEVICE) * 0.1
        rel_shifted = shift_rel_proj(rel_proj, is_local)

        # Build sheared rel_logits using the real qr_clip_rel_tokens path via unit scales + huge clip
        rel_logits = torch.zeros((T + 128, H, E + 256), dtype=torch.float16, device=DEVICE)
        # Populate sheared rel_logits directly matching attention_prologue convention
        for r_i, (past, ql) in enumerate(specs):
            q_s = cu_q[r_i]
            rel_all = torch.einsum("qhd,de->qhe", r_vec[q_s : q_s + ql].float(), rel_proj.float()).half()
            for tok in range(ql):
                tile_end = min((tok // 128 + 1) * 128, ql)
                rel_base = ((past + tile_end + 127) // 128) * 128 - (E + 256)
                pos = past + tok
                for d in range(min(E, pos + 1)):
                    k_idx = pos - d
                    rel_logits[q_s + tok, :, k_idx - rel_base] = rel_all[tok, :, d]

        out = torch.empty((T, H * D), dtype=torch.float16, device=DEVICE)
        max_q = max(ql for _, ql in specs)
        if splits > 0:
            rel_decode_attention(
                q, kc, vc, rel_logits, gate, sinks, out,
                block_table=bt, cache_seqlens=seq_lens_t, cu_seqlens_q=cu_q_t,
                max_query_len=max_q, num_actual_tokens=T, softmax_scale=scale,
                is_local=is_local, num_splits=splits,
            )
        else:
            pf_out = torch.empty((T, H, D), dtype=torch.float32, device=DEVICE)
            lse = torch.empty((H, T), dtype=torch.float32, device=DEVICE)
            rel_attention(
                q, kc, vc, pf_out, lse, rel_logits,
                block_table=bt, cache_seqlens=seq_lens_t, cu_seqlens_q=cu_q_t,
                max_query_len=max_q, softmax_scale=scale, window=window,
                num_splits=1, mode=MODE_OUT_LSE,
            )
            gate_sink(pf_out, gate, lse.t(), sinks, out)

        refs = []
        for r_i, (past, ql) in enumerate(specs):
            q_s = cu_q[r_i]
            refs.append(
                _ref_rel_attention(
                    q[q_s : q_s + ql], k_seqs[r_i], v_seqs[r_i], rel_proj,
                    r_vec[q_s : q_s + ql], gate[q_s : q_s + ql], sinks[q_s : q_s + ql],
                    past, scale, window,
                )
            )
        ref = torch.cat(refs, dim=0)
        err = (out.float() - ref).abs().max().item()
        assert err < 5e-3, f"rel_attention {name} max_err={err}"
        print(f"    {name:24s} max_err={err:.2e}", flush=True)
    print("  PASS: rel_attention (decode/verify/prefill x local/global x FP16/FP8)!")


_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_nvfp4_ref(codes_u8: torch.Tensor, scales_fp8: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Exact FP32 dequant of [E, N, K/2] uint8 + [E, N, K/16] fp8_e4m3fn + [E] fp32 global scale."""
    lut = _E2M1_LUT.to(codes_u8.device)
    lo = lut[(codes_u8 & 0xF).long()]
    hi = lut[(codes_u8 >> 4).long()]
    E, N, K2 = codes_u8.shape
    w = torch.stack([lo, hi], dim=-1).view(E, N, K2 * 2)
    s = scales_fp8.to(torch.float32).repeat_interleave(16, dim=-1)
    return w * s * g[:, None, None]


def test_moe_sm70() -> None:
    print("[5/6] test_moe_sm70 ...", flush=True)
    torch.manual_seed(4)
    T, H, I, E_all, E_loc, TOPK = 12, 2304, 1152, 128, 32, 8

    # 1. Router selection + tie breaking + padding
    x = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25
    gw = torch.randn(E_all, H, dtype=torch.float16, device=DEVICE) * 0.05
    bias = torch.randn(E_all, dtype=torch.float32, device=DEVICE) * 0.1
    bounds = torch.tensor([32, 64], dtype=torch.int32, device=DEVICE)
    is_pad = torch.zeros(T, dtype=torch.bool, device=DEVICE)
    is_pad[-1] = True
    ids, wts, s2d = select_experts_sm70(x, gw, bias, TOPK, 8.0, bounds, is_pad)

    logits_ref = torch.mm(x, gw.t(), out_dtype=torch.float32)
    sel_ref = torch.sigmoid(logits_ref) + bias[None, :]
    top_idx = torch.topk(sel_ref, TOPK, dim=-1).indices
    top_logits = logits_ref.gather(1, top_idx)
    log_sig = F.logsigmoid(top_logits)
    w_ref = torch.softmax(log_sig, dim=-1) * 8.0
    loc_ref = torch.where((top_idx >= 32) & (top_idx < 64), top_idx - 32, -1)
    loc_ref[-1] = -1
    assert torch.equal(ids, loc_ref.to(torch.int32)), "router local ids mismatch!"
    assert (wts.view(T, TOPK) - w_ref).abs().max().item() < 1e-5, "router weights mismatch!"
    assert (ids[-1] == -1).all() and (s2d[-TOPK:] == -1).all()

    # 2. situ_interleaved
    gu = torch.randn(T * TOPK, 2 * I, dtype=torch.float16, device=DEVICE) * 5.0
    act_got = situ_interleaved(gu, wts, s2d, 4.0, 25.0)
    g_ref = gu[:, 0::2].float()
    u_ref = gu[:, 1::2].float()
    act_ref = (
        (4.0 * torch.tanh(g_ref / 4.0)) * torch.sigmoid(g_ref) * (25.0 * torch.tanh(u_ref / 25.0)) * wts[:, None]
    ).half()
    valid_rows = s2d >= 0
    assert (act_got[valid_rows].float() - act_ref[valid_rows].float()).abs().max().item() < 5e-3

    # 3. NVFP4 experts via skinny_moe_qpn_sm70 against exact FP32 manual NVFP4 dequant!
    w13_c = torch.randint(0, 256, (E_loc, 2 * I, H // 2), dtype=torch.uint8, device=DEVICE)
    # Positive non-NaN E4M3FN block scales in [0.25 .. 4.0]
    s13_f = (torch.rand(E_loc, 2 * I, H // 16, device=DEVICE) * 2.0 + 0.25).to(torch.float8_e4m3fn)
    g13 = torch.rand(E_loc, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005
    w2_c = torch.randint(0, 256, (E_loc, H, I // 2), dtype=torch.uint8, device=DEVICE)
    s2_f = (torch.rand(E_loc, H, I // 16, device=DEVICE) * 2.0 + 0.25).to(torch.float8_e4m3fn)
    g2 = torch.rand(E_loc, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005

    w13_ref = _dequant_nvfp4_ref(w13_c, s13_f, g13)
    w2_ref = _dequant_nvfp4_ref(w2_c, s2_f, g2)

    # Prepack into QPN order
    w13_qpn = w13_c.clone()
    s13_qpn = s13_f.view(torch.uint8).clone()
    w2_qpn = w2_c.clone()
    s2_qpn = s2_f.view(torch.uint8).clone()
    for e in range(E_loc):
        qc, qs = qpn_prepack(w13_qpn[e], s13_qpn[e], 16)
        w13_qpn[e].view(-1).copy_(qc)
        s13_qpn[e].view(-1).copy_(qs)
        qc, qs = qpn_prepack(w2_qpn[e], s2_qpn[e], 16)
        w2_qpn[e].view(-1).copy_(qc)
        s2_qpn[e].view(-1).copy_(qs)

    down_got = run_nvfp4_experts_sm70(
        x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids, wts, s2d, 4.0, 25.0
    )
    out_got = combine_sm70(down_got, s2d, TOPK, torch.float16)

    out_ref = torch.zeros((T, H), dtype=torch.float32, device=DEVICE)
    for t in range(T):
        for k in range(TOPK):
            e = int(ids[t, k])
            if e < 0:
                continue
            y13 = (x[t].float() @ w13_ref[e].t()).half().float()
            gg = y13[0::2]
            uu = y13[1::2]
            act = ((4.0 * torch.tanh(gg / 4.0)) * torch.sigmoid(gg) * (25.0 * torch.tanh(uu / 25.0)) * wts[t * TOPK + k]).half().float()
            y2 = (act @ w2_ref[e].t()).half().float()
            out_ref[t] += y2
    err_nvfp4 = (out_got.float() - out_ref).abs().max().item()
    cos_nvfp4 = F.cosine_similarity(out_got.float().flatten(), out_ref.flatten(), dim=0).item()
    assert err_nvfp4 < 0.05 and cos_nvfp4 > 0.99999, f"NVFP4 MoE err={err_nvfp4} cos={cos_nvfp4}"

    # All-sentinel test (0 local experts routed on this EP rank!)
    ids_none = torch.full_like(ids, -1)
    s2d_none = torch.full_like(s2d, -1)
    down_none = run_nvfp4_experts_sm70(
        x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids_none, wts, s2d_none, 4.0, 25.0
    )
    out_none = combine_sm70(down_none, s2d_none, TOPK, torch.float16)
    assert torch.equal(out_none, torch.zeros_like(out_none))

    # 4. MXFP8 dequant (full UE8M0 range including low/high exponents & 0xFF NaN) + run_mxfp8_experts_sm70
    # Test dequant bit-exactness across s_u8 in [115..140] + extreme s=105 + s=255
    w_mx = (torch.randn(2 * I, H, device=DEVICE) * 2.0).clamp(-448, 448).to(torch.float8_e4m3fn)
    s_mx = torch.randint(115, 134, (2 * I, H // 32), dtype=torch.uint8, device=DEVICE)
    s_mx[0, 0] = 255
    s_mx[1, 0] = 105
    dq_got = mxfp8_dequant_expert(w_mx.view(torch.uint8), s_mx)
    exp_f = torch.where(s_mx == 255, torch.zeros_like(s_mx, dtype=torch.float32), torch.pow(2.0, s_mx.float() - 127.0))
    dq_ref = (w_mx.float() * exp_f.repeat_interleave(32, dim=-1)).half()
    assert torch.equal(dq_got, dq_ref), "MXFP8 dequant value mismatch!"
    non_nan_s = (s_mx != 255).repeat_interleave(32, dim=-1)
    assert torch.equal(dq_got.view(torch.int16)[non_nan_s], dq_ref.view(torch.int16)[non_nan_s]), "MXFP8 dequant not bit-exact on non-NaN scales!"

    E_mx = 8
    w13_mx = (torch.randn(E_mx, 2 * I, H, device=DEVICE) * 1.5).clamp(-448, 448).to(torch.float8_e4m3fn)
    s13_mx = torch.randint(120, 126, (E_mx, 2 * I, H // 32), dtype=torch.uint8, device=DEVICE)
    w2_mx = (torch.randn(E_mx, H, I, device=DEVICE) * 1.5).clamp(-448, 448).to(torch.float8_e4m3fn)
    s2_mx = torch.randint(120, 126, (E_mx, H, I // 32), dtype=torch.uint8, device=DEVICE)
    ids_mx = torch.where(ids >= 0, ids % E_mx, -1)
    down_mx = run_mxfp8_experts_sm70(
        x, w13_mx.view(torch.uint8), s13_mx, w2_mx.view(torch.uint8), s2_mx,
        ids_mx, wts, s2d, 4.0, 25.0,
    )
    out_mx = combine_sm70(down_mx, s2d, TOPK, torch.float16)
    ref_mx = torch.zeros((T, H), dtype=torch.float32, device=DEVICE)
    for t in range(T):
        for k in range(TOPK):
            e = int(ids_mx[t, k])
            if e < 0:
                continue
            W13 = w13_mx[e].float() * torch.pow(2.0, s13_mx[e].float() - 127.0).repeat_interleave(32, dim=-1)
            W2 = w2_mx[e].float() * torch.pow(2.0, s2_mx[e].float() - 127.0).repeat_interleave(32, dim=-1)
            y13 = (x[t].float() @ W13.t()).half().float()
            gg, uu = y13[0::2], y13[1::2]
            act = ((4.0 * torch.tanh(gg / 4.0)) * torch.sigmoid(gg) * (25.0 * torch.tanh(uu / 25.0)) * wts[t * TOPK + k]).half().float()
            ref_mx[t] += (act @ W2.t()).half().float()
    err_mx = (out_mx.float() - ref_mx).abs().max().item()
    rel_mx = err_mx / max(ref_mx.abs().max().item(), 1e-6)
    cos_mx = F.cosine_similarity(out_mx.float().flatten(), ref_mx.flatten(), dim=0).item()
    assert rel_mx < 2e-3 and cos_mx > 0.99999, f"MXFP8 MoE err={err_mx} rel={rel_mx} cos={cos_mx}"
    print(f"  PASS: router, situ, NVFP4 MoE (err={err_nvfp4:.2e}, cos={cos_nvfp4:.7f}), MXFP8 bit-exact dequant & MoE (err={err_mx:.2e}, rel={rel_mx:.2e}, cos={cos_mx:.7f})!")


def test_checkpoint_key_mapping() -> None:
    print("[6/6] test_checkpoint_key_mapping ...", flush=True)
    meta_dir = pathlib.Path(__file__).resolve().parents[2] / "model_meta"
    if not meta_dir.exists():
        print("  SKIP: model_meta directory not found beside inkling_v100")
        return
    cfg_dict = json.loads((meta_dir / "config.json").read_text())
    q_dict = json.loads((meta_dir / "hf_quant_config.json").read_text())
    idx_dict = json.loads((meta_dir / "model.safetensors.index.json").read_text())

    from inkling_v100 import register
    from inkling_v100.configs import Inkling2LoopMMConfig
    from inkling_v100.model import Inkling2LoopForConditionalGeneration
    from inkling_v100.quant_config import Inkling2QuantConfig

    register()
    hf_cfg = Inkling2LoopMMConfig(**cfg_dict)
    q_cfg = Inkling2QuantConfig(q_dict)
    assert [i for i in range(24) if q_cfg.is_mxfp8_layer(i)] == [20, 21, 22, 23]

    # Map every non-visual key in model.safetensors.index.json through _renames and _shards
    renames = Inkling2LoopForConditionalGeneration._renames
    shards = Inkling2LoopForConditionalGeneration._shards
    ckpt_keys = [k for k in idx_dict["weight_map"] if not k.startswith("model.visual.")]
    mapped = set()
    shard_hits: dict[str, set[int]] = {}
    for k in ckpt_keys:
        name = k
        for old, new in renames:
            name = name.replace(old, new)
        sid = None
        for old, (new, s) in shards.items():
            if old in name:
                name = name.replace(old, new)
                sid = s
                break
        if sid is None:
            mapped.add(name)
        else:
            shard_hits.setdefault(name, set()).add(sid)
    for name, sids in shard_hits.items():
        if ".attn.qkvr." in name:
            assert sids == {0, 1, 2, 3, 4, 5}, f"incomplete qkvr shards on {name}: {sids}"
        elif ".attn_norm." in name:
            assert sids == {0, 1, 2}, f"incomplete attn_norm shards on {name}: {sids}"
        mapped.add(name)

    # Verify expected parameter count (all 24 layers + embed + embed_norm + attn_res + norm + loop queries + lm_head)
    assert "lm_head.weight" in mapped
    assert "model.embed_tokens.weight" in mapped
    assert "model.embed_norm.weight" in mapped
    assert "model.norm.weight" in mapped
    assert "model.middle_cycle_cross_pass_queries" in mapped
    assert "model.middle_cycle_cross_pass_kv_queries" in mapped
    assert "model.attn_res.output_query" in mapped
    for i in range(24):
        assert f"model.layers.{i}.attn.qkvr.weight" in mapped
        assert f"model.layers.{i}.attn.wo_ud.weight" in mapped
        assert f"model.layers.{i}.attn_norm.weight" in mapped
        assert f"model.layers.{i}.mlp_norm.weight" in mapped
        assert f"model.layers.{i}.mlp.experts.w13_weight" in mapped
        assert f"model.layers.{i}.mlp.experts.w2_weight" in mapped
        if i in range(6, 18):
            for st in ("attn", "k", "v", "mlp"):
                assert f"model.layers.{i}.controllers.{st}.down_weight" in mapped
    print(f"  PASS: all {len(ckpt_keys)} text checkpoint keys map cleanly to {len(mapped)} model parameters!")


if __name__ == "__main__":
    test_fp8_sm70()
    test_norms_attnres_loop()
    test_sconv_and_prologue()
    test_rel_attention()
    test_moe_sm70()
    test_checkpoint_key_mapping()
    print("\nALL 6 SM70 OP TEST SUITES PASSED!", flush=True)
