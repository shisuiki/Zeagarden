# SPDX-License-Identifier: Apache-2.0
"""Correctness, bit-exactness, ptxas audit, and warm-clock benchmark for nvfp4_prefill_moe.cu.

Run on a V100 (sm_70) GPU:
    CUDA_VISIBLE_DEVICES=1 python3 tests/test_nvfp4_prefill_cuda.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    nvfp4_skinny_scale_reason,
    qpn_prepack,
)

from inkling_v100.ops.moe_sm70 import (
    combine_sm70,
    ep4_route_to_skinny_groups,
    run_nvfp4_experts_sm70,
)
from inkling_v100.ops.nvfp4_prefill_cuda import (
    load_nvfp4_prefill_ext,
    nvfp4_qpn_dequant_batch,
    nvfp4_qpn_dequant_expert,
    run_nvfp4_experts_prefill_cuda,
)

DEVICE = torch.device("cuda:0")  # Mapped to physical GPU 1 via CUDA_VISIBLE_DEVICES=1
E_LOCAL = 32
H = 2304
I = 1152
TOPK = 8
GATE_CAP = 4.0
UP_CAP = 25.0
NVFP4_VISITS = 44  # Inkling-2-Mini: 44 NVFP4 MoE visits (48 total - 4 MXFP8)

_E2M1_VALS = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _e4m3fn_lut_f32(device: torch.device) -> torch.Tensor:
    vals = []
    for u in range(256):
        sign = -1.0 if (u & 0x80) else 1.0
        exp = (u >> 3) & 0xF
        mant = u & 0x7
        if exp == 15 and mant == 7:
            vals.append(0.0)
        elif exp == 0:
            vals.append(sign * (mant * (2.0**-9)))
        else:
            vals.append(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))
    return torch.tensor(vals, dtype=torch.float32, device=device)


def dequant_nvfp4_ref_fp32(
    codes_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    g_val: float,
    e2m1_lut: torch.Tensor,
    e4m3_lut: torch.Tensor,
) -> torch.Tensor:
    """Exact FP32 dequant of raw (pre-QPN) [N, K/2] uint8 codes + [N, K/16] uint8 E4M3FN scales."""
    N, K2 = codes_u8.shape
    lo = e2m1_lut[(codes_u8 & 0xF).long()]
    hi = e2m1_lut[(codes_u8 >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).view(N, K2 * 2)
    s = (e4m3_lut[scales_u8.long()] * float(g_val)).repeat_interleave(16, dim=-1)
    return w * s


def make_nvfp4_weights(
    e_local: int = E_LOCAL,
    h: int = H,
    inter: int = I,
    seed: int = 42,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    w13_raw = torch.randint(0, 256, (e_local, 2 * inter, h // 2), generator=gen, dtype=torch.uint8, device=DEVICE)
    w2_raw = torch.randint(0, 256, (e_local, h, inter // 2), generator=gen, dtype=torch.uint8, device=DEVICE)

    s13_raw = torch.randint(0x18, 0x58, (e_local, 2 * inter, h // 16), generator=gen, dtype=torch.uint8, device=DEVICE)
    s2_raw = torch.randint(0x18, 0x58, (e_local, h, inter // 16), generator=gen, dtype=torch.uint8, device=DEVICE)

    g13_t = (torch.rand((e_local,), generator=gen, device=DEVICE, dtype=torch.float32) * 8e-4 + 2e-4).contiguous()
    g2_t = (torch.rand((e_local,), generator=gen, device=DEVICE, dtype=torch.float32) * 8e-4 + 2e-4).contiguous()
    reason = nvfp4_skinny_scale_reason(g13_t, g2_t)
    assert reason is None, f"Test global scales rejected by skinny_qpn: {reason}"

    w13_qpn = w13_raw.clone()
    s13_qpn = s13_raw.clone()
    w2_qpn = w2_raw.clone()
    s2_qpn = s2_raw.clone()
    for e in range(e_local):
        qc, qs = qpn_prepack(w13_raw[e], s13_raw[e], 16)
        w13_qpn[e].view(-1).copy_(qc)
        s13_qpn[e].view(-1).copy_(qs)
        qc, qs = qpn_prepack(w2_raw[e], s2_raw[e], 16)
        w2_qpn[e].view(-1).copy_(qc)
        s2_qpn[e].view(-1).copy_(qs)

    return w13_raw, s13_raw, g13_t, w2_raw, s2_raw, g2_t, w13_qpn, s13_qpn, w2_qpn, s2_qpn


def make_ep4_routing(
    T: int,
    e_local: int = E_LOCAL,
    topk: int = TOPK,
    local_prob: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    e_total = int(round(e_local / local_prob))
    scores = torch.rand((T, e_total), generator=gen, device=DEVICE)
    top_e = torch.topk(scores, topk, dim=-1).indices
    ids = torch.where(top_e < e_local, top_e, -1).to(torch.int32)
    raw_w = torch.rand((T, topk), generator=gen, device=DEVICE) + 0.1
    weights = (raw_w / raw_w.sum(dim=-1, keepdim=True) * 8.0).reshape(-1).to(torch.float32).contiguous()
    flat = ids.reshape(-1)
    src2dst = torch.where(flat >= 0, torch.arange(T * topk, dtype=torch.int32, device=DEVICE), -1)
    return ids, weights, src2dst


def ref_nvfp4_moe_fp32(
    x: torch.Tensor,
    w13_raw: torch.Tensor,
    s13_raw: torch.Tensor,
    g13_t: torch.Tensor,
    w2_raw: torch.Tensor,
    s2_raw: torch.Tensor,
    g2_t: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    gate_cap: float = GATE_CAP,
    up_cap: float = UP_CAP,
) -> torch.Tensor:
    T = x.shape[0]
    topk = ids.shape[1]
    out_dim = w2_raw.shape[1]
    out = torch.zeros((T, out_dim), dtype=torch.float32, device=x.device)
    flat_ids = ids.reshape(-1)
    local_mask = flat_ids >= 0
    if not bool(local_mask.any()):
        return out

    e2m1_lut = torch.tensor(_E2M1_VALS, dtype=torch.float32, device=x.device)
    e4m3_lut = _e4m3fn_lut_f32(x.device)

    local_slots = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
    local_e = flat_ids[local_slots]
    uniq_e = torch.unique(local_e)
    xf = x.to(torch.float32)
    g13_cpu = g13_t.tolist()
    g2_cpu = g2_t.tolist()
    for e_t in uniq_e.tolist():
        slots_e = local_slots[local_e == e_t]
        tok_e = slots_e // topk
        xe = xf[tok_e]
        W13 = dequant_nvfp4_ref_fp32(w13_raw[e_t], s13_raw[e_t], g13_cpu[e_t], e2m1_lut, e4m3_lut)
        W2 = dequant_nvfp4_ref_fp32(w2_raw[e_t], s2_raw[e_t], g2_cpu[e_t], e2m1_lut, e4m3_lut)
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


def check_metrics(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    cos_min: float = 0.99999,
    rel_max: float = 3e-3,
) -> tuple[float, float, float]:
    got_f = got.to(torch.float32)
    ref_f = ref.to(torch.float32)
    assert torch.isfinite(got_f).all(), f"[{name}] non-finite values in output!"
    ref_norm = ref_f.norm().item()
    diff_norm = (got_f - ref_f).norm().item()
    if ref_norm == 0.0:
        assert diff_norm == 0.0, f"[{name}] expected exact zero, got diff_norm={diff_norm}"
        print(f"  PASS {name:48s} | exact zero", flush=True)
        return 1.0, 0.0, 0.0
    rel_l2 = diff_norm / ref_norm
    max_abs = (got_f - ref_f).abs().max().item()
    cos = F.cosine_similarity(got_f.flatten(), ref_f.flatten(), dim=0).item()
    print(
        f"  PASS {name:48s} | cos={cos:.7f}  rel_l2={rel_l2:.2e}  max_abs={max_abs:.2e}",
        flush=True,
    )
    assert cos >= cos_min, f"[{name}] cos={cos:.7f} < {cos_min}"
    assert rel_l2 < rel_max, f"[{name}] rel_l2={rel_l2:.3e} >= {rel_max}"
    return cos, rel_l2, max_abs


def test_build_and_ptxas() -> None:
    print("\n=== [1/4] Build & PTXAS Register / Spill Audit (sm_70) ===", flush=True)
    cmd = [
        sys.executable,
        "-c",
        "from inkling_v100.ops.nvfp4_prefill_cuda import load_nvfp4_prefill_ext; "
        "load_nvfp4_prefill_ext(verbose=True, force_rebuild=True)",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"BUILD FAILED!\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}", flush=True)
        res.check_returncode()
    combined = res.stdout + "\n" + res.stderr
    ptx_lines = [ln.strip() for ln in combined.splitlines() if "ptxas" in ln]
    assert ptx_lines, f"No ptxas -v output captured!\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    for ln in ptx_lines:
        print(f"  {ln}", flush=True)

    for ln in ptx_lines:
        m = re.search(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads", ln)
        if m:
            stk, sst, sld = int(m.group(1)), int(m.group(2)), int(m.group(3))
            assert stk == 0 and sst == 0 and sld == 0, f"Volta discipline violated: {ln}"
        m_cum = re.search(r"(\d+)\s+bytes cumulative stack size", ln)
        if m_cum:
            assert int(m_cum.group(1)) == 0, f"Volta discipline violated (stack spill!): {ln}"

    load_nvfp4_prefill_ext(verbose=False, force_rebuild=False)
    print("  PASS: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads across ALL kernels!", flush=True)


def _bench_fn(fn, warmup_s: float = 1.0, iters: int = 80) -> float:
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
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times_ms[len(times_ms) // 2]


def test_dequant_bit_exact() -> None:
    print("\n=== [2/4] Bit-Exact Dequantization Verification + Bandwidth ===", flush=True)
    e2m1_lut = torch.tensor(_E2M1_VALS, dtype=torch.float32, device=DEVICE)
    e4m3_lut = _e4m3fn_lut_f32(DEVICE)

    # 2a. Exhaustive coverage of ALL 256 E4M3FN scale bytes x ALL 16 E2M1 codes!
    N_ex, K_ex = 256, 64
    codes_ex = torch.zeros((1, N_ex, K_ex // 2), dtype=torch.uint8, device=DEVICE)
    nibs = torch.arange(16, dtype=torch.uint8, device=DEVICE).repeat(4)
    packed_row = nibs[0::2] | (nibs[1::2] << 4)
    codes_ex[0] = packed_row.unsqueeze(0).expand(N_ex, -1)
    scales_ex = torch.arange(256, dtype=torch.uint8, device=DEVICE).view(1, 256, 1).expand(1, 256, K_ex // 16).contiguous()
    g_ex = torch.tensor([0.000732421875], dtype=torch.float32, device=DEVICE)

    qc_ex, qs_ex = qpn_prepack(codes_ex[0], scales_ex[0], 16)
    w_qpn_ex = qc_ex.view(1, N_ex, K_ex // 2).contiguous()
    s_qpn_ex = qs_ex.view(1, N_ex, K_ex // 16).contiguous()

    ref_ex = dequant_nvfp4_ref_fp32(codes_ex[0], scales_ex[0], g_ex[0].item(), e2m1_lut, e4m3_lut).to(torch.float16)
    for use_direct, label in ((False, "SMEM-swiz"), (True, "Direct")):
        got_ex = nvfp4_qpn_dequant_expert(w_qpn_ex, s_qpn_ex, g_ex, 0, use_direct=use_direct)
        mismatches = int((got_ex.view(torch.uint16) != ref_ex.view(torch.uint16)).sum().item())
        assert mismatches == 0, f"Exhaustive 256x16 dequant ({label}) had {mismatches} bit mismatches!"
        print(f"  PASS Exhaustive 256 E4M3FN (incl. subnormals/NaN) x 16 E2M1 ({label}): 0 bit mismatches", flush=True)

    # 2b. Full Inkling2 shapes: w13 [32, 2304, 2304] and w2 [32, 2304, 1152]
    w13_raw, s13_raw, g13_t, w2_raw, s2_raw, g2_t, w13_qpn, s13_qpn, w2_qpn, s2_qpn = make_nvfp4_weights(
        E_LOCAL, H, I, seed=123
    )
    s13_raw[0, :32, 0] = 0x7F
    s13_raw[0, 32:64, 0] = 0xFF
    s13_raw[0, 64:96, 1] = 0x03
    s13_raw[31, 0:32, 2] = 0x85
    for e in (0, 31):
        qc, qs = qpn_prepack(w13_raw[e], s13_raw[e], 16)
        w13_qpn[e].view(-1).copy_(qc)
        s13_qpn[e].view(-1).copy_(qs)

    for nm, w_raw, s_raw, w_qpn, s_qpn, g_t in (
        ("w13 [32, 2304, 2304]", w13_raw, s13_raw, w13_qpn, s13_qpn, g13_t),
        ("w2  [32, 2304, 1152]", w2_raw, s2_raw, w2_qpn, s2_qpn, g2_t),
    ):
        for use_direct, label in ((False, "SMEM"), (True, "Direct")):
            total_mismatches = 0
            for e in (0, 1, 15, 31):
                got = nvfp4_qpn_dequant_expert(w_qpn, s_qpn, g_t, e, use_direct=use_direct)
                ref = dequant_nvfp4_ref_fp32(w_raw[e], s_raw[e], g_t[e].item(), e2m1_lut, e4m3_lut).to(torch.float16)
                total_mismatches += int((got.view(torch.uint16) != ref.view(torch.uint16)).sum().item())
            assert total_mismatches == 0, f"{nm} ({label}) had {total_mismatches} bit mismatches!"
            print(f"  PASS {nm} ({label:6s}) experts (0,1,15,31): 0 bit mismatches", flush=True)

        eids = torch.tensor([0, 7, 19, 31], dtype=torch.int32, device=DEVICE)
        got_b = nvfp4_qpn_dequant_batch(w_qpn, s_qpn, g_t, eids)
        for bi, e in enumerate(eids.tolist()):
            ref = dequant_nvfp4_ref_fp32(w_raw[e], s_raw[e], g_t[e].item(), e2m1_lut, e4m3_lut).to(torch.float16)
            mm = int((got_b[bi].view(torch.uint16) != ref.view(torch.uint16)).sum().item())
            assert mm == 0, f"{nm} batched bi={bi} e={e} had {mm} bit mismatches!"
        print(f"  PASS {nm} batched [0,7,19,31]: 0 bit mismatches", flush=True)


def test_moe_correctness() -> None:
    print("\n=== [3/4] Full MoE Correctness vs Pure FP32 Oracle & skinny_moe_qpn_sm70 ===", flush=True)
    w13_raw, s13_raw, g13_t, w2_raw, s2_raw, g2_t, w13_qpn, s13_qpn, w2_qpn, s2_qpn = make_nvfp4_weights(
        E_LOCAL, H, I, seed=2026
    )

    # 3a. Standard EP=4 batches T = 64, 128, 256, 512, 1024, 2048 (finite realistic scales vs BOTH FP32 & skinny_qpn)
    for T in (64, 128, 256, 512, 1024, 2048):
        torch.manual_seed(3000 + T)
        x = (torch.randn((T, H), dtype=torch.float32, device=DEVICE) * 0.25).to(torch.float16)
        ids, wts, s2d = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=4000 + T)
        route = ep4_route_to_skinny_groups(ids, E_LOCAL)

        ref_fp32 = ref_nvfp4_moe_fp32(x, w13_raw, s13_raw, g13_t, w2_raw, s2_raw, g2_t, ids, wts, GATE_CAP, UP_CAP)
        down_skinny = run_nvfp4_experts_sm70(
            x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d, GATE_CAP, UP_CAP,
            precomputed_route=route,
        )
        out_skinny = combine_sm70(down_skinny, s2d, TOPK, torch.float16)

        got = run_nvfp4_experts_prefill_cuda(
            x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d, GATE_CAP, UP_CAP,
            precomputed_route=route, combine=True, mode=0,
        )
        check_metrics(f"T={T:<4d} fused HMMA vs FP32 oracle", got, ref_fp32, cos_min=0.99999)
        check_metrics(f"T={T:<4d} fused HMMA vs skinny_qpn", got, out_skinny, cos_min=0.99999)

        # At T=256 and T=1024, verify ALL modes (0..5), combine=False, internal route!
        if T in (256, 1024):
            for mode, label in (
                (0, "auto"), (1, "16x64"), (2, "32x64"), (3, "64x64"),
                (4, "3b-64"), (5, "cuBLAS"),
            ):
                got_m = run_nvfp4_experts_prefill_cuda(
                    x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d, GATE_CAP, UP_CAP,
                    precomputed_route=route, combine=True, mode=mode,
                )
                check_metrics(f"T={T:<4d} {label:6s} vs FP32 oracle", got_m, ref_fp32, cos_min=0.99999)
            down_nc = run_nvfp4_experts_prefill_cuda(
                x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d, GATE_CAP, UP_CAP,
                precomputed_route=None, combine=False, mode=0,
            )
            got_nc = combine_sm70(down_nc, s2d, TOPK, torch.float16)
            check_metrics(f"T={T:<4d} combine=False + internal route vs FP32", got_nc, ref_fp32, cos_min=0.99999)

    # 3b. All-sentinel empty-rank case (0 local hits!)
    for T in (64, 256, 1024):
        x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids_none = torch.full((T, TOPK), -1, dtype=torch.int32, device=DEVICE)
        wts_none = torch.ones(T * TOPK, dtype=torch.float32, device=DEVICE)
        s2d_none = torch.full((T * TOPK,), -1, dtype=torch.int32, device=DEVICE)
        route_none = ep4_route_to_skinny_groups(ids_none, E_LOCAL)
        for mode in (0, 1, 2, 3, 4, 5):
            got_none = run_nvfp4_experts_prefill_cuda(
                x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids_none, wts_none, s2d_none,
                GATE_CAP, UP_CAP, precomputed_route=route_none, combine=True, mode=mode,
            )
            assert torch.equal(got_none, torch.zeros_like(got_none)), f"All-sentinel T={T} mode={mode} non-zero!"
        print(f"  PASS ALL-SENTINEL T={T:<4d} (all 6 modes 0..5): exact zero", flush=True)

    # 3c. Skewed routing (3/32 active, 1/32 active, 2/32 active) + odd row counts!
    for n_active_case, active_experts in (
        ("skewed 3/32 experts", (0, 13, 29)),
        ("single 1/32 expert ", (7,)),
        ("pair   2/32 experts", (4, 21)),
    ):
        T_sk = 512
        torch.manual_seed(5555 + len(active_experts))
        x_sk = (torch.randn((T_sk, H), device=DEVICE) * 0.25).half()
        ids_sk = torch.full((T_sk, TOPK), -1, dtype=torch.int32, device=DEVICE)
        if len(active_experts) == 3:
            ids_sk[0, 0] = active_experts[0]
            ids_sk[1:18, 1] = active_experts[1]
            ids_sk[18:, 2] = active_experts[2]
            ids_sk[::2, 5] = active_experts[2]
        elif len(active_experts) == 1:
            ids_sk[:, 0] = active_experts[0]
        else:
            ids_sk[:100, 0] = active_experts[0]
            ids_sk[100:, 3] = active_experts[1]
        wts_sk = (torch.rand(T_sk * TOPK, device=DEVICE) * 1.5 + 0.5).to(torch.float32)
        flat_sk = ids_sk.reshape(-1)
        s2d_sk = torch.where(flat_sk >= 0, torch.arange(T_sk * TOPK, dtype=torch.int32, device=DEVICE), -1)
        route_sk = ep4_route_to_skinny_groups(ids_sk, E_LOCAL)

        ref_sk = ref_nvfp4_moe_fp32(
            x_sk, w13_raw, s13_raw, g13_t, w2_raw, s2_raw, g2_t, ids_sk, wts_sk, GATE_CAP, UP_CAP
        )
        down_sk_ref = run_nvfp4_experts_sm70(
            x_sk, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids_sk, wts_sk, s2d_sk,
            GATE_CAP, UP_CAP, precomputed_route=route_sk,
        )
        out_sk_ref = combine_sm70(down_sk_ref, s2d_sk, TOPK, torch.float16)
        for mode in (1, 2, 3, 4):
            got_sk = run_nvfp4_experts_prefill_cuda(
                x_sk, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids_sk, wts_sk, s2d_sk,
                GATE_CAP, UP_CAP, precomputed_route=route_sk, combine=True, mode=mode,
            )
            check_metrics(f"{n_active_case} T={T_sk} mode={mode} vs FP32", got_sk, ref_sk, cos_min=0.99999)
            if mode == 4:
                check_metrics(f"{n_active_case} T={T_sk} mode=4 vs skinny_qpn", got_sk, out_sk_ref, cos_min=0.99999)

    # 3d. Verify CUDA-graph safety of default fused zero-sync HMMA.884 path!
    T_cg = 256
    x_buf = (torch.randn((T_cg, H), device=DEVICE) * 0.25).half()
    ids_buf, wts_buf, s2d_buf = make_ep4_routing(T_cg, E_LOCAL, TOPK, 0.25, seed=7100)
    route_buf = ep4_route_to_skinny_groups(ids_buf, E_LOCAL)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _ = run_nvfp4_experts_prefill_cuda(
                x_buf, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids_buf, wts_buf, s2d_buf,
                GATE_CAP, UP_CAP, precomputed_route=route_buf, combine=True, mode=0,
            )
    torch.cuda.current_stream().wait_stream(stream)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        out_cg = run_nvfp4_experts_prefill_cuda(
            x_buf, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids_buf, wts_buf, s2d_buf,
            GATE_CAP, UP_CAP, precomputed_route=route_buf, combine=True, mode=0,
        )
    # Replay with brand new inputs & routing!
    x_new = (torch.randn((T_cg, H), device=DEVICE) * 0.25).half()
    ids_new, wts_new, s2d_new = make_ep4_routing(T_cg, E_LOCAL, TOPK, 0.25, seed=7200)
    r_new = ep4_route_to_skinny_groups(ids_new, E_LOCAL)
    x_buf.copy_(x_new)
    ids_buf.copy_(ids_new)
    wts_buf.copy_(wts_new)
    s2d_buf.copy_(s2d_new)
    route_buf[0].copy_(r_new[0])
    route_buf[1].copy_(r_new[1])
    route_buf[2].copy_(r_new[2])
    out_cg.zero_()
    cg.replay()
    torch.cuda.synchronize()
    ref_cg = ref_nvfp4_moe_fp32(
        x_new, w13_raw, s13_raw, g13_t, w2_raw, s2_raw, g2_t, ids_new, wts_new, GATE_CAP, UP_CAP
    )
    check_metrics("CUDA-Graph capture+replay T=256 (zero-sync!)", out_cg, ref_cg, cos_min=0.99999)

    # 3e. Subnormal + NaN (0x7F/0xFF -> 0) E4M3FN scales inside fused HMMA.884 vs FP32 oracle
    s13_edge = s13_raw.clone()
    s2_edge = s2_raw.clone()
    s13_edge[0, :64, 0] = 0x03   # subnormal 3 * 2^-9
    s13_edge[0, 64:128, 1] = 0x7F  # NaN -> 0
    s2_edge[0, :64, 0] = 0x05    # subnormal 5 * 2^-9
    s2_edge[0, 64:128, 1] = 0xFF   # NaN -> 0
    w13_qe = w13_qpn.clone()
    s13_qe = s13_qpn.clone()
    w2_qe = w2_qpn.clone()
    s2_qe = s2_qpn.clone()
    qc, qs = qpn_prepack(w13_raw[0], s13_edge[0], 16)
    w13_qe[0].view(-1).copy_(qc)
    s13_qe[0].view(-1).copy_(qs)
    qc, qs = qpn_prepack(w2_raw[0], s2_edge[0], 16)
    w2_qe[0].view(-1).copy_(qc)
    s2_qe[0].view(-1).copy_(qs)
    ids_e = torch.full((128, TOPK), -1, dtype=torch.int32, device=DEVICE)
    ids_e[:, 0] = 0
    wts_e = torch.ones(128 * TOPK, dtype=torch.float32, device=DEVICE)
    s2d_e = torch.where(ids_e.reshape(-1) >= 0, torch.arange(128 * TOPK, dtype=torch.int32, device=DEVICE), -1)
    x_e = (torch.randn((128, H), device=DEVICE) * 0.25).half()
    ref_e = ref_nvfp4_moe_fp32(x_e, w13_raw, s13_edge, g13_t, w2_raw, s2_edge, g2_t, ids_e, wts_e, GATE_CAP, UP_CAP)
    for mode in (1, 2, 3, 4):
        got_e = run_nvfp4_experts_prefill_cuda(
            x_e, w13_qe, s13_qe, g13_t, w2_qe, s2_qe, g2_t, ids_e, wts_e, s2d_e,
            GATE_CAP, UP_CAP, combine=True, mode=mode,
        )
        check_metrics(f"Subnormal+NaN E4M3FN scales mode={mode} vs FP32", got_e, ref_e, cos_min=0.99999)


def run_benchmarks() -> None:
    print("\n=== [4/4] Warm-Clock (>= 1.0s warmup) Benchmark Suite (GPU 1) ===", flush=True)
    print(f"    Inkling-2-Mini NVFP4: E_local={E_LOCAL}, H={H}, I={I}, topk={TOPK}, {NVFP4_VISITS} NVFP4 visits", flush=True)

    _, _, g13_t, _, _, g2_t, w13_qpn, s13_qpn, w2_qpn, s2_qpn = make_nvfp4_weights(E_LOCAL, H, I, seed=9090)

    # 4a. Tile-size & engine ablation (1:16x64, 2:32x64, 3:64x64, 4:3b-64, 5:cuBLAS)
    print("\n  --- Engine & Tile Ablation (per-layer median ms, 0.5s warmup) ---", flush=True)
    print(
        f"  {'T':>5s} | {'16x64':>8s} | {'32x64':>8s} | {'64x64':>8s} | "
        f"{'3b-64':>8s} | {'cuBLAS':>8s} | {'Best':>7s}",
        flush=True,
    )
    print("  " + "-" * 69, flush=True)
    for T in (64, 128, 192, 256, 384, 512, 1024, 2048):
        torch.manual_seed(8000 + T)
        x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids, wts, s2d = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=8100 + T)
        route = ep4_route_to_skinny_groups(ids, E_LOCAL)
        iters = 80 if T <= 512 else 40
        modes = (1, 2, 3, 4, 5)
        names = ("16x64", "32x64", "64x64", "3b-64", "cuBLAS")
        res = []
        for mode in modes:
            ms = _bench_fn(
                lambda m=mode: run_nvfp4_experts_prefill_cuda(
                    x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d,
                    GATE_CAP, UP_CAP, precomputed_route=route, combine=True, mode=m,
                ),
                warmup_s=0.5,
                iters=iters,
            )
            res.append(ms)
        best_i = min(range(len(modes)), key=lambda i: res[i])
        print(
            f"  {T:5d} | {res[0]:8.3f} | {res[1]:8.3f} | {res[2]:8.3f} | "
            f"{res[3]:8.3f} | {res[4]:8.3f} | {names[best_i]:>7s}",
            flush=True,
        )

    # 4b. Full >= 1.0s warm-clock head-to-head table vs run_nvfp4_experts_sm70
    print("\n  --- Head-to-Head Warm-Clock (>= 1.0s warmup): Fused HMMA Prefill vs Skinny QPN ---", flush=True)
    header = (
        f"  {'T':>5s} | {'Local':>5s} | {'ActE':>4s} | "
        f"{'Skinny+Comb (ms)':>16s} | {'SkinnyOnly (ms)':>15s} | {'Prefill+Comb (ms)':>17s} | "
        f"{'44x Skinny (ms)':>15s} | {'44x Prefill (ms)':>16s} | {'Speedup':>8s}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    crossover_t: int | None = None
    for T in (64, 128, 192, 256, 384, 512, 1024, 2048):
        torch.manual_seed(9000 + T)
        x = (torch.randn((T, H), device=DEVICE) * 0.25).half()
        ids, wts, s2d = make_ep4_routing(T, E_LOCAL, TOPK, local_prob=0.25, seed=9100 + T)
        route = ep4_route_to_skinny_groups(ids, E_LOCAL)

        local_mask = ids >= 0
        n_local = int(local_mask.sum().item())
        n_act = int(torch.unique(ids[local_mask]).numel()) if n_local > 0 else 0
        iters = 80 if T <= 512 else 40

        skinny_comb_ms = _bench_fn(
            lambda: combine_sm70(
                run_nvfp4_experts_sm70(
                    x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d,
                    GATE_CAP, UP_CAP, precomputed_route=route,
                ),
                s2d, TOPK, torch.float16,
            ),
            warmup_s=1.0,
            iters=iters,
        )
        skinny_only_ms = _bench_fn(
            lambda: run_nvfp4_experts_sm70(
                x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d,
                GATE_CAP, UP_CAP, precomputed_route=route,
            ),
            warmup_s=1.0,
            iters=iters,
        )
        prefill_ms = _bench_fn(
            lambda: run_nvfp4_experts_prefill_cuda(
                x, w13_qpn, s13_qpn, g13_t, w2_qpn, s2_qpn, g2_t, ids, wts, s2d,
                GATE_CAP, UP_CAP, precomputed_route=route, combine=True, mode=0,
            ),
            warmup_s=1.0,
            iters=iters,
        )
        speedup = skinny_comb_ms / prefill_ms
        if crossover_t is None and speedup >= 1.0:
            crossover_t = T
        print(
            f"  {T:5d} | {n_local:5d} | {n_act:4d} | "
            f"{skinny_comb_ms:16.3f} | {skinny_only_ms:15.3f} | {prefill_ms:17.3f} | "
            f"{skinny_comb_ms * NVFP4_VISITS:15.1f} | {prefill_ms * NVFP4_VISITS:16.1f} | {speedup:7.2f}x",
            flush=True,
        )

    print(f"\n  Recommended crossover threshold: T >= {crossover_t}", flush=True)


if __name__ == "__main__":
    test_build_and_ptxas()
    test_dequant_bit_exact()
    test_moe_correctness()
    run_benchmarks()
    print("\nALL NVFP4 PREFILL CUDA TESTS & BENCHMARKS PASSED!", flush=True)
