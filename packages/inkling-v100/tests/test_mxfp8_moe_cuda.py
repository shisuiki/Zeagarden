# SPDX-License-Identifier: Apache-2.0
"""Comprehensive correctness, CUDA-graph, and warm-clock benchmark suite for mlp_mxfp8_moe.cu.

Run on a V100 (sm_70) GPU:
    CUDA_VISIBLE_DEVICES=0 python3 tests/test_mxfp8_moe_cuda.py
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from inkling_v100.ops.moe_sm70 import (
    combine_sm70,
    ep4_route_to_skinny_groups,
    run_mxfp8_experts_sm70,
)
from inkling_v100.ops.mxfp8_moe_cuda import (
    load_mxfp8_moe_ext,
    mxfp8_moe_w13_situ,
    mxfp8_moe_w2,
    mxfp8_qpn8_prepack,
    run_mxfp8_experts_cuda,
)

DEVICE = torch.device("cuda:0")
E_LOCAL = 32
H = 2304
I = 1152
TOPK = 8
GATE_CAP = 4.0
UP_CAP = 25.0


def dequant_mxfp8_fp32(w_u8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """Exact FP32 dequant of [N, K] uint8 E4M3FN + [N, K/32] uint8 UE8M0."""
    w_f32 = w_u8.view(torch.float8_e4m3fn).to(torch.float32)
    s_f32 = torch.where(
        s_u8 == 255,
        torch.zeros_like(s_u8, dtype=torch.float32),
        torch.pow(2.0, s_u8.to(torch.float32) - 127.0),
    )
    return w_f32 * s_f32.repeat_interleave(32, dim=-1)


def ref_mxfp8_moe_fp32(
    x: torch.Tensor,
    w13_u8: torch.Tensor,
    s13_u8: torch.Tensor,
    w2_u8: torch.Tensor,
    s2_u8: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    gate_cap: float = GATE_CAP,
    up_cap: float = UP_CAP,
) -> torch.Tensor:
    """Exact FP32 oracle for Inkling2 EP=4 MXFP8 MoE (interleaved w13 + softcapped SiTU + w2 + combine)."""
    T = x.shape[0]
    topk = ids.shape[1]
    out_dim = w2_u8.shape[1]
    out = torch.zeros((T, out_dim), dtype=torch.float32, device=x.device)
    flat_ids = ids.reshape(-1)
    local_mask = flat_ids >= 0
    if not bool(local_mask.any()):
        return out
    local_slots = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
    local_e = flat_ids[local_slots]
    uniq_e = torch.unique(local_e)
    xf = x.to(torch.float32)
    for e_t in uniq_e.tolist():
        slots_e = local_slots[local_e == e_t]
        tok_e = slots_e // topk
        xe = xf[tok_e]
        W13 = dequant_mxfp8_fp32(w13_u8[e_t], s13_u8[e_t])
        W2 = dequant_mxfp8_fp32(w2_u8[e_t], s2_u8[e_t])
        y13 = xe @ W13.t()
        g = y13[:, 0::2]
        u = y13[:, 1::2]
        sig = torch.sigmoid(g)
        if gate_cap > 0.0:
            g = gate_cap * torch.tanh(g / gate_cap)
        if up_cap > 0.0:
            u = up_cap * torch.tanh(u / up_cap)
        act = ((g * sig) * u) * weights[slots_e].unsqueeze(-1)
        y2 = act @ W2.t()
        out.index_add_(0, tok_e, y2)
    return out


def make_ep4_routing(
    T: int,
    e_local: int = E_LOCAL,
    topk: int = TOPK,
    local_prob: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate realistic EP=4 routing (unique experts per token, ~local_prob hitting this rank)."""
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    e_total = int(round(e_local / local_prob))
    scores = torch.rand((T, e_total), generator=gen, device=DEVICE)
    top_e = torch.topk(scores, topk, dim=-1).indices
    # Rank 0 owns [0, e_local)
    ids = torch.where(top_e < e_local, top_e, -1).to(torch.int32)
    raw_w = torch.rand((T, topk), generator=gen, device=DEVICE) + 0.1
    weights = (raw_w / raw_w.sum(dim=-1, keepdim=True) * 8.0).reshape(-1).to(torch.float32).contiguous()
    flat = ids.reshape(-1)
    src2dst = torch.where(flat >= 0, torch.arange(T * topk, dtype=torch.int32, device=DEVICE), -1)
    return ids, weights, src2dst


def make_mxfp8_weights(
    e_local: int = E_LOCAL,
    h: int = H,
    inter: int = I,
    s_lo: int = 120,
    s_hi: int = 126,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    w13 = (torch.randn((e_local, 2 * inter, h), generator=gen, device=DEVICE) * 1.5).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)
    s13 = torch.randint(s_lo, s_hi + 1, (e_local, 2 * inter, h // 32), generator=gen, dtype=torch.uint8, device=DEVICE)
    w2 = (torch.randn((e_local, h, inter), generator=gen, device=DEVICE) * 1.5).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)
    s2 = torch.randint(s_lo, s_hi + 1, (e_local, h, inter // 32), generator=gen, dtype=torch.uint8, device=DEVICE)
    return w13, s13, w2, s2


def check_metrics(name: str, got: torch.Tensor, ref: torch.Tensor, *, cos_min: float = 0.99999, rel_max: float = 2e-3) -> None:
    got_f = got.to(torch.float32)
    ref_f = ref.to(torch.float32)
    assert torch.isfinite(got_f).all(), f"[{name}] non-finite values in output!"
    ref_norm = ref_f.norm().item()
    diff_norm = (got_f - ref_f).norm().item()
    if ref_norm == 0.0:
        assert diff_norm == 0.0, f"[{name}] expected exact zero, got diff_norm={diff_norm}"
        print(f"  PASS {name:38s} | exact zero (0 local hits)", flush=True)
        return
    rel_l2 = diff_norm / ref_norm
    max_abs = (got_f - ref_f).abs().max().item()
    max_rel = max_abs / max(ref_f.abs().max().item(), 1e-12)
    cos = F.cosine_similarity(got_f.flatten(), ref_f.flatten(), dim=0).item()
    print(
        f"  PASS {name:38s} | cos={cos:.7f}  rel_l2={rel_l2:.2e}  max_rel={max_rel:.2e}  max_abs={max_abs:.2e}",
        flush=True,
    )
    assert cos > cos_min, f"[{name}] cos={cos:.7f} <= {cos_min}"
    assert rel_l2 < rel_max, f"[{name}] rel_l2={rel_l2:.3e} >= {rel_max}"


def test_correctness_suite() -> None:
    print("\n=== [1/4] Correctness vs Exact FP32 Oracle (T=1,4,8,32,128,512 + Edge Cases) ===", flush=True)
    ext = load_mxfp8_moe_ext()
    print("  w13 configs:", ext.w13_configs(), flush=True)
    print("  w2  configs:", ext.w2_configs(), flush=True)

    w13_u8, s13_u8, w2_u8, s2_u8 = make_mxfp8_weights(E_LOCAL, H, I, s_lo=118, s_hi=127, seed=101)
    # Inject a few s=255 (UE8M0 NaN -> 0) blocks even in the standard weight set!
    s13_u8[0, 0, 0] = 255
    s13_u8[3, 17, 5] = 255
    s2_u8[0, 5, 3] = 255
    s2_u8[7, 100, 11] = 255

    w13_qpn, s13_qpn, g13, wide13 = mxfp8_qpn8_prepack(w13_u8, s13_u8, is_w13=True, inplace=False)
    w2_qpn, s2_qpn, g2, wide2 = mxfp8_qpn8_prepack(w2_u8, s2_u8, is_w13=False, inplace=False)
    assert not wide13 and not wide2, f"Expected fast path, got wide13={wide13}, wide2={wide2}"

    # Verify inplace prepack is bit-identical and preserves data_ptr!
    w13_ip = w13_u8.clone()
    s13_ip = s13_u8.clone()
    ptr_w = w13_ip.data_ptr()
    ptr_s = s13_ip.data_ptr()
    w13_qpn_ip, s13_qpn_ip, g13_ip, wide13_ip = mxfp8_qpn8_prepack(w13_ip, s13_ip, is_w13=True, inplace=True)
    assert w13_qpn_ip.data_ptr() == ptr_w and s13_qpn_ip.data_ptr() == ptr_s
    assert torch.equal(w13_qpn_ip, w13_qpn) and torch.equal(s13_qpn_ip, s13_qpn) and torch.equal(g13_ip, g13)
    assert wide13_ip == wide13
    print("  PASS inplace prepack bit-exact + zero extra storage verified!", flush=True)

    # 1a. Standard EP=4 batches T = 1, 4, 8, 32, 128, 512
    for idx_t, T in enumerate((1, 4, 8, 32, 128, 512)):
        torch.manual_seed(1000 + T)
        x = (torch.randn((T, H), dtype=torch.float32, device=DEVICE) * 0.25).to(torch.float16)
        ids, wts, s2d = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=2000 + T)
        if T == 1 and int((ids >= 0).sum()) == 0:
            ids[0, 0] = 0
            ids[0, 3] = 7
            s2d[0] = 0
            s2d[3] = 3
        down = run_mxfp8_experts_cuda(x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids, wts, s2d, GATE_CAP, UP_CAP)
        got = combine_sm70(down, s2d, TOPK, torch.float16)
        ref = ref_mxfp8_moe_fp32(x, w13_u8, s13_u8, w2_u8, s2_u8, ids, wts, GATE_CAP, UP_CAP)
        n_loc = int((ids >= 0).sum())
        check_metrics(f"EP4 T={T:<3d} (local_slots={n_loc})", got, ref)

    # 1b. ALL-SENTINEL (0 local hits!)
    for T in (1, 16, 128):
        x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids_none = torch.full((T, TOPK), -1, dtype=torch.int32, device=DEVICE)
        wts_none = torch.ones(T * TOPK, dtype=torch.float32, device=DEVICE)
        s2d_none = torch.full((T * TOPK,), -1, dtype=torch.int32, device=DEVICE)
        down_none = run_mxfp8_experts_cuda(
            x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids_none, wts_none, s2d_none, GATE_CAP, UP_CAP
        )
        got_none = combine_sm70(down_none, s2d_none, TOPK, torch.float16)
        check_metrics(f"ALL-SENTINEL T={T}", got_none, torch.zeros_like(got_none))

    # 1c. Large single-expert batch (m_e = 40 > R8, and m_e = 100 > 48) across EVERY kernel cfg!
    for m_e in (40, 100):
        T_big = m_e
        torch.manual_seed(3000 + m_e)
        x_big = (torch.randn((T_big, H), device=DEVICE) * 0.25).half()
        ids_big = torch.full((T_big, TOPK), -1, dtype=torch.int32, device=DEVICE)
        ids_big[:, 0] = 5  # every token hits expert 5 -> single group of size m_e!
        wts_big = (torch.rand(T_big * TOPK, device=DEVICE) * 1.5 + 0.5).to(torch.float32)
        s2d_big = torch.where(
            ids_big.reshape(-1) >= 0,
            torch.arange(T_big * TOPK, dtype=torch.int32, device=DEVICE),
            -1,
        )
        ref_big = ref_mxfp8_moe_fp32(x_big, w13_u8, s13_u8, w2_u8, s2_u8, ids_big, wts_big, GATE_CAP, UP_CAP)
        for c13 in range(len(ext.w13_configs())):
            c2 = min(c13, len(ext.w2_configs()) - 1)
            down_big = run_mxfp8_experts_cuda(
                x_big, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2,
                ids_big, wts_big, s2d_big, GATE_CAP, UP_CAP,
                cfg13=c13, cfg2=c2,
            )
            got_big = combine_sm70(down_big, s2d_big, TOPK, torch.float16)
            check_metrics(f"single-expert m_e={m_e} cfg13={c13} cfg2={c2}", got_big, ref_big)
        # Also test w2 cfg 5 (NCOL=1 fallback)
        down_big5 = run_mxfp8_experts_cuda(
            x_big, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2,
            ids_big, wts_big, s2d_big, GATE_CAP, UP_CAP,
            cfg13=0, cfg2=5,
        )
        check_metrics(f"single-expert m_e={m_e} cfg2=5(col1)", combine_sm70(down_big5, s2d_big, TOPK, torch.float16), ref_big)


def test_ue8m0_wide_and_nan() -> None:
    print("\n=== [2/4] UE8M0 Wide Exponent Ranges + s=255 NaN + All-255 Expert ===", flush=True)
    T = 16
    E_s = 8
    torch.manual_seed(404)
    x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
    ids, wts, s2d = make_ep4_routing(T, E_s, TOPK, local_prob=0.5, seed=405)

    # Case A: Maximum bit-exact fast-path span = 22 (incl. s=134 -> 2^+15 in w13, low [95..117], and s=255)
    w13_a, s13_a, w2_a, s2_a = make_mxfp8_weights(E_s, H, I, s_lo=104, s_hi=126, seed=406)
    s13_a[:, 0:32, 0] = 255
    s2_a[:, 0:32, 0] = 255
    # Expert 0: entirely s=255 (all zeros)
    s13_a[0].fill_(255)
    # Expert 1: low exponents [95..117] (span=22, tests large positive shift_e=+17!)
    s13_a[1] = torch.randint(95, 118, s13_a[1].shape, dtype=torch.uint8, device=DEVICE)
    s2_a[1] = torch.randint(95, 118, s2_a[1].shape, dtype=torch.uint8, device=DEVICE)
    # Expert 2: high exponents [112..134] in w13 (span=22, shift_e=0, tests max FP16 pow2 2^+15!)
    s13_a[2] = torch.randint(112, 135, s13_a[2].shape, dtype=torch.uint8, device=DEVICE)
    w13_qa, s13_qa, g13_a, wide13_a = mxfp8_qpn8_prepack(w13_a, s13_a, is_w13=True)
    w2_qa, s2_qa, g2_a, wide2_a = mxfp8_qpn8_prepack(w2_a, s2_a, is_w13=False)
    assert not wide13_a and not wide2_a
    down_a = run_mxfp8_experts_cuda(
        x, w13_qa, s13_qa, g13_a, w2_qa, s2_qa, g2_a, ids, wts, s2d, GATE_CAP, UP_CAP,
        wide_scale_13=wide13_a, wide_scale_2=wide2_a,
    )
    ref_a = ref_mxfp8_moe_fp32(x, w13_a, s13_a, w2_a, s2_a, ids, wts, GATE_CAP, UP_CAP)
    check_metrics("UE8M0 fast-path span=22 + s=255", combine_sm70(down_a, s2d, TOPK, torch.float16), ref_a)

    # Case B: Wide span = 29 ([105..134] in both w13 and w2 + s=255; small weights keep FP16 out finite)
    w13_b, s13_b, w2_b, s2_b = make_mxfp8_weights(E_s, H, I, s_lo=118, s_hi=127, seed=407)
    s13_b[:, 0:64, :] = 105
    s13_b[:, 64:128, :] = 134
    s13_b[:, 128:160, 0:4] = 255
    s2_b[:, 0:64, :] = 105
    s2_b[:, 64:128, :] = 134
    s2_b[:, 128:160, 0:4] = 255
    wts_b = wts * 0.01  # keeps y2 finite in FP16 even on s2=134 (2^7) rows
    w13_qb, s13_qb, g13_b, wide13_b = mxfp8_qpn8_prepack(w13_b, s13_b, is_w13=True)
    w2_qb, s2_qb, g2_b, wide2_b = mxfp8_qpn8_prepack(w2_b, s2_b, is_w13=False)
    assert wide13_b and wide2_b
    down_b = run_mxfp8_experts_cuda(
        x, w13_qb, s13_qb, g13_b, w2_qb, s2_qb, g2_b, ids, wts_b, s2d, GATE_CAP, UP_CAP,
        wide_scale_13=wide13_b, wide_scale_2=wide2_b,
    )
    ref_b = ref_mxfp8_moe_fp32(x, w13_b, s13_b, w2_b, s2_b, ids, wts_b, GATE_CAP, UP_CAP)
    check_metrics("UE8M0 wide span=29 [105..134]+255", combine_sm70(down_b, s2d, TOPK, torch.float16), ref_b)

    # Case C: Extreme UE8M0 span = 65+ ([60..130] within same row + s=255)
    w13_c, s13_c, w2_c, s2_c = make_mxfp8_weights(E_s, H, I, s_lo=118, s_hi=124, seed=408)
    s13_c[:, 0:32, 0] = 80    # tiny 2^-47
    s13_c[:, 0:32, 1] = 130   # large 2^3 in same row!
    s13_c[:, 32:64, 0] = 255  # NaN -> 0
    s2_c[:, 0:32, 0] = 60     # tiny 2^-67
    s2_c[:, 0:32, 1] = 125    # 2^-2 in same row!
    s2_c[:, 32:64, 0] = 255
    w13_qc, s13_qc, g13_c, wide13_c = mxfp8_qpn8_prepack(w13_c, s13_c, is_w13=True)
    w2_qc, s2_qc, g2_c, wide2_c = mxfp8_qpn8_prepack(w2_c, s2_c, is_w13=False)
    assert wide13_c and wide2_c
    down_c = run_mxfp8_experts_cuda(
        x, w13_qc, s13_qc, g13_c, w2_qc, s2_qc, g2_c, ids, wts, s2d, GATE_CAP, UP_CAP,
        wide_scale_13=wide13_c, wide_scale_2=wide2_c,
    )
    ref_c = ref_mxfp8_moe_fp32(x, w13_c, s13_c, w2_c, s2_c, ids, wts, GATE_CAP, UP_CAP)
    check_metrics("UE8M0 wide span=65 [60..130]+255", combine_sm70(down_c, s2d, TOPK, torch.float16), ref_c)

    # Case D: All-255 expert produces exact zeros
    ids_z = torch.full((4, TOPK), -1, dtype=torch.int32, device=DEVICE)
    ids_z[:, 0] = 0  # expert 0 has all s13==255 -> inter_act==0 -> down==0!
    wts_z = torch.ones(4 * TOPK, dtype=torch.float32, device=DEVICE)
    s2d_z = torch.where(ids_z.reshape(-1) >= 0, torch.arange(4 * TOPK, dtype=torch.int32, device=DEVICE), -1)
    down_z = run_mxfp8_experts_cuda(
        x[:4], w13_qa, s13_qa, g13_a, w2_qa, s2_qa, g2_a, ids_z, wts_z, s2d_z, GATE_CAP, UP_CAP
    )
    check_metrics("All-255 expert -> exact zero", combine_sm70(down_z, s2d_z, TOPK, torch.float16), torch.zeros((4, H), device=DEVICE))

    # Case E: Non-multiple-of-64 output dim (H=96, I=64) exercising w2 NCOL=1 fallback
    w13_s, s13_s, w2_s, s2_s = make_mxfp8_weights(4, 96, 64, s_lo=120, s_hi=126, seed=409)
    w13_qs, s13_qs, g13_s, _ = mxfp8_qpn8_prepack(w13_s, s13_s, is_w13=True)
    w2_qs, s2_qs, g2_s, _ = mxfp8_qpn8_prepack(w2_s, s2_s, is_w13=False)
    x_s = (torch.randn((64, 96), device=DEVICE) * 0.25).half()
    ids_s, wts_s, s2d_s = make_ep4_routing(64, 4, TOPK, local_prob=0.5, seed=410)
    down_s = run_mxfp8_experts_cuda(x_s, w13_qs, s13_qs, g13_s, w2_qs, s2_qs, g2_s, ids_s, wts_s, s2d_s, GATE_CAP, UP_CAP)
    ref_s = ref_mxfp8_moe_fp32(x_s, w13_s, s13_s, w2_s, s2_s, ids_s, wts_s, GATE_CAP, UP_CAP)
    check_metrics("Non-64-aligned H=96 I=64 T=64", combine_sm70(down_s, s2d_s, TOPK, torch.float16), ref_s)


def test_cuda_graph_capture_replay() -> None:
    print("\n=== [3/4] CUDA Graph Capture & Replay (Zero Host Sync Proof) ===", flush=True)
    w13_u8, s13_u8, w2_u8, s2_u8 = make_mxfp8_weights(E_LOCAL, H, I, s_lo=119, s_hi=126, seed=505)
    w13_qpn, s13_qpn, g13, _ = mxfp8_qpn8_prepack(w13_u8, s13_u8, is_w13=True)
    w2_qpn, s2_qpn, g2, _ = mxfp8_qpn8_prepack(w2_u8, s2_u8, is_w13=False)

    for T in (1, 8, 32, 128):
        x_buf = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids_buf, wts_buf, s2d_buf = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=600 + T)
        if T == 1 and int((ids_buf >= 0).sum()) == 0:
            ids_buf[0, 0] = 2
            s2d_buf[0] = 0

        # Warmup on side stream before capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                d = run_mxfp8_experts_cuda(x_buf, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids_buf, wts_buf, s2d_buf, GATE_CAP, UP_CAP)
                _ = combine_sm70(d, s2d_buf, TOPK, torch.float16)
        torch.cuda.current_stream().wait_stream(stream)

        # Capture full routing + w13_situ + w2 + combine inside torch.cuda.CUDAGraph!
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            down_g = run_mxfp8_experts_cuda(
                x_buf, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids_buf, wts_buf, s2d_buf, GATE_CAP, UP_CAP
            )
            out_g = combine_sm70(down_g, s2d_buf, TOPK, torch.float16)

        # Replay 1: completely new x, ids, weights!
        torch.manual_seed(700 + T)
        x_new = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids_new, wts_new, s2d_new = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=800 + T)
        if T == 1 and int((ids_new >= 0).sum()) == 0:
            ids_new[0, 1] = 11
            ids_new[0, 5] = 29
            s2d_new[1] = 1
            s2d_new[5] = 5
        x_buf.copy_(x_new)
        ids_buf.copy_(ids_new)
        wts_buf.copy_(wts_new)
        s2d_buf.copy_(s2d_new)
        out_g.zero_()
        g.replay()
        torch.cuda.synchronize()
        ref_new = ref_mxfp8_moe_fp32(x_new, w13_u8, s13_u8, w2_u8, s2_u8, ids_new, wts_new, GATE_CAP, UP_CAP)
        check_metrics(f"CUDAGraph replay T={T:<3d} (new routing)", out_g, ref_new)

        # Replay 2: ALL-SENTINEL through the EXACT SAME captured graph!
        ids_buf.fill_(-1)
        s2d_buf.fill_(-1)
        out_g.fill_(123.0)
        g.replay()
        torch.cuda.synchronize()
        check_metrics(f"CUDAGraph replay T={T:<3d} (all-sentinel)", out_g, torch.zeros_like(out_g))


def _bench_fn(fn, warmup_s: float = 1.0, iters: int = 100) -> float:
    """Warm-clock CUDA event timer (returns median latency in microseconds)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times_us[len(times_us) // 2]


def run_benchmarks() -> None:
    print("\n=== [4/4] Warm-Clock (>= 1s warmup) Benchmark: Phase 1 CUDA vs Phase 0 Triton ===", flush=True)
    print(f"    1 layer: E_local={E_LOCAL}, H={H}, I={I}, topk={TOPK}, realistic EP=4 (~25% local hits)", flush=True)
    ext = load_mxfp8_moe_ext()

    w13_u8, s13_u8, w2_u8, s2_u8 = make_mxfp8_weights(E_LOCAL, H, I, s_lo=120, s_hi=126, seed=909)
    w13_qpn, s13_qpn, g13, _ = mxfp8_qpn8_prepack(w13_u8, s13_u8, is_w13=True)
    w2_qpn, s2_qpn, g2, _ = mxfp8_qpn8_prepack(w2_u8, s2_u8, is_w13=False)
    scratch_w13 = torch.empty((2 * I, H), dtype=torch.float16, device=DEVICE)
    scratch_w2 = torch.empty((H, I), dtype=torch.float16, device=DEVICE)

    # First: quick per-config sweep for w13 and w2 at T=1, 8, 32, 128, 512 (0.25s warmup each)
    print("\n  --- Config Sweep (pure kernel median us) ---", flush=True)
    for T in (1, 8, 32, 128, 512):
        torch.manual_seed(9000 + T)
        x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids, wts, s2d = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=9100 + T)
        if T == 1:
            # Exact expected EP=4 T=1 load: 2 local hits (25% of topk=8)
            ids.fill_(-1)
            ids[0, 0] = 3
            ids[0, 4] = 19
            s2d.fill_(-1)
            s2d[0] = 0
            s2d[4] = 4
        perm, gids, goff = ep4_route_to_skinny_groups(ids, E_LOCAL)
        inter_buf = torch.empty((T * TOPK, I), dtype=torch.float16, device=DEVICE)
        down_buf = torch.empty((T * TOPK, H), dtype=torch.float16, device=DEVICE)
        w13_res = []
        for c in range(len(ext.w13_configs())):
            us = _bench_fn(
                lambda c=c: mxfp8_moe_w13_situ(x, w13_qpn, s13_qpn, g13, perm, gids, goff, wts, TOPK, GATE_CAP, UP_CAP, out=inter_buf, cfg=c),
                warmup_s=0.25, iters=60,
            )
            w13_res.append(f"c{c}={us:.1f}")
        w2_res = []
        for c in range(len(ext.w2_configs())):
            us = _bench_fn(
                lambda c=c: mxfp8_moe_w2(inter_buf, w2_qpn, s2_qpn, g2, perm, gids, goff, out=down_buf, cfg=c, num_tokens_hint=T),
                warmup_s=0.25, iters=60,
            )
            w2_res.append(f"c{c}={us:.1f}")
        print(f"  T={T:<3d} | w13: {' '.join(w13_res)} | w2: {' '.join(w2_res)}", flush=True)

    # Second: full >=1.0s warm-clock head-to-head table
    print("\n  --- Head-to-Head Warm-Clock (>=1s warmup) Latency Table ---", flush=True)
    header = (
        f"  {'T':>4s} | {'Local':>5s} | {'ActE':>4s} | {'Triton P0 (us)':>14s} | "
        f"{'CUDA Eager (us)':>15s} | {'CUDA Graph (us)':>15s} | {'KernelsOnly (us)':>16s} | "
        f"{'Speedup(Eager)':>14s} | {'Speedup(Graph)':>14s}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    for T in (1, 8, 32, 128, 512):
        torch.manual_seed(9000 + T)
        x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids, wts, s2d = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=9100 + T)
        if T == 1:
            ids.fill_(-1)
            ids[0, 0] = 3
            ids[0, 4] = 19
            s2d.fill_(-1)
            s2d[0] = 0
            s2d[4] = 4

        local_mask = ids >= 0
        n_local = int(local_mask.sum().item())
        n_act_e = int(torch.unique(ids[local_mask]).numel()) if n_local > 0 else 0

        # Phase 0 Triton
        iters = 80 if T <= 128 else 40
        p0_us = _bench_fn(
            lambda: run_mxfp8_experts_sm70(
                x, w13_u8, s13_u8, w2_u8, s2_u8, ids, wts, s2d, GATE_CAP, UP_CAP, scratch_w13, scratch_w2
            ),
            warmup_s=1.0,
            iters=iters,
        )

        # Phase 1 CUDA eager (route + w13_situ + w2)
        p1_eager_us = _bench_fn(
            lambda: run_mxfp8_experts_cuda(
                x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids, wts, s2d, GATE_CAP, UP_CAP
            ),
            warmup_s=1.0,
            iters=iters,
        )

        # Phase 1 CUDA kernels-only (w13_situ + w2 given precomputed perm/gids/goff)
        perm, gids, goff = ep4_route_to_skinny_groups(ids, E_LOCAL)
        inter_buf = torch.empty((T * TOPK, I), dtype=torch.float16, device=DEVICE)
        down_buf = torch.empty((T * TOPK, H), dtype=torch.float16, device=DEVICE)
        p1_kern_us = _bench_fn(
            lambda: (
                mxfp8_moe_w13_situ(x, w13_qpn, s13_qpn, g13, perm, gids, goff, wts, TOPK, GATE_CAP, UP_CAP, out=inter_buf),
                mxfp8_moe_w2(inter_buf, w2_qpn, s2_qpn, g2, perm, gids, goff, out=down_buf, num_tokens_hint=T),
            ),
            warmup_s=1.0,
            iters=iters,
        )

        # Phase 1 CUDA Graph replay (route + w13_situ + w2 captured in CUDAGraph!)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _ = run_mxfp8_experts_cuda(
                    x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids, wts, s2d, GATE_CAP, UP_CAP
                )
        torch.cuda.current_stream().wait_stream(stream)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            _ = run_mxfp8_experts_cuda(
                x, w13_qpn, s13_qpn, g13, w2_qpn, s2_qpn, g2, ids, wts, s2d, GATE_CAP, UP_CAP
            )
        p1_graph_us = _bench_fn(lambda: cg.replay(), warmup_s=1.0, iters=iters)

        print(
            f"  {T:4d} | {n_local:5d} | {n_act_e:4d} | {p0_us:14.1f} | "
            f"{p1_eager_us:15.1f} | {p1_graph_us:15.1f} | {p1_kern_us:16.1f} | "
            f"{p0_us / p1_eager_us:13.2f}x | {p0_us / p1_graph_us:13.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    test_correctness_suite()
    test_ue8m0_wide_and_nan()
    test_cuda_graph_capture_replay()
    run_benchmarks()
    print("\nALL MXFP8 MOE CUDA TESTS & BENCHMARKS PASSED!", flush=True)
