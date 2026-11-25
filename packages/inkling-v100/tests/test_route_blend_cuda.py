# SPDX-License-Identifier: Apache-2.0
"""Comprehensive verification, ptxas-spill audit, CUDA-graph safety, and warm-clock
benchmarks for:
  1. `ep4_route.cu` (`ep4_gemv_select_and_route` + `ep4_select_and_route`)
  2. `loop_blend.cu` (`loop_blend_sm70` for R=64 Inkling-2-Mini & R=128)

Run on GPU 2 ONLY:
    CUDA_VISIBLE_DEVICES=2 python3 tests/test_route_blend_cuda.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from inkling_v100.ops.ep4_route_cuda import (
    load_ep4_route_ext,
    preallocate_ep4_route_scratch,
    select_and_route_ep4_cuda,
)
from inkling_v100.ops.loop import blend_rms_norm, pair_rms_norm
from inkling_v100.ops.loop_blend_cuda import (
    load_loop_blend_ext,
    loop_blend_cuda,
    preallocate_loop_blend_scratch,
)
from inkling_v100.ops.moe_sm70 import ep4_route_to_skinny_groups, select_experts_sm70

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# 1. Build + ptxas -v spill audit
# ---------------------------------------------------------------------------

def build_and_audit_ptxas() -> None:
    print("=" * 78)
    print("1. COMPILING & AUDITING ptxas -v (0 B stack frame, 0 B spills)")
    print("=" * 78, flush=True)

    # Run builds in a subprocess so we capture nvcc/ptxas stderr cleanly AND stage `.so` files.
    code = (
        "from inkling_v100.ops.ep4_route_cuda import load_ep4_route_ext;"
        "from inkling_v100.ops.loop_blend_cuda import load_loop_blend_ext;"
        "load_ep4_route_ext(verbose=True, force_rebuild=True);"
        "load_loop_blend_ext(verbose=True, force_rebuild=True);"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(combined, flush=True)
        raise RuntimeError(f"CUDA extension build failed with rc={proc.returncode}")

    # Parse ptxas info lines
    cur_func = None
    kernels: list[tuple[str, int, int, int, int, int]] = []
    # (func, regs, stack, spill_st, spill_ld, smem)
    cur_stack = 0
    cur_sp_st = 0
    cur_sp_ld = 0
    for line in combined.splitlines():
        m_fn = re.search(r"Compiling entry function '([^']+)'", line)
        if m_fn:
            cur_func = m_fn.group(1)
            cur_stack = cur_sp_st = cur_sp_ld = 0
            continue
        m_sp = re.search(
            r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads",
            line,
        )
        if m_sp:
            cur_stack = int(m_sp.group(1))
            cur_sp_st = int(m_sp.group(2))
            cur_sp_ld = int(m_sp.group(3))
            continue
        m_rg = re.search(r"Used (\d+) registers(?:,\s*(\d+) bytes smem)?", line)
        if m_rg and cur_func is not None:
            regs = int(m_rg.group(1))
            smem = int(m_rg.group(2)) if m_rg.group(2) else 0
            kernels.append((cur_func, regs, cur_stack, cur_sp_st, cur_sp_ld, smem))
            cur_func = None

    assert kernels, "No ptxas kernel stats found in build output!"
    # Demangle symbol names for readability
    demangled_map: dict[str, str] = {}
    try:
        dm = subprocess.run(
            ["c++filt", *[k[0] for k in kernels]],
            capture_output=True,
            text=True,
            check=True,
        )
        for orig, pretty in zip([k[0] for k in kernels], dm.stdout.splitlines()):
            pretty = pretty.replace("(anonymous namespace)::", "")
            pretty = re.sub(r"\([^)]*\)$", "", pretty).strip()
            demangled_map[orig] = pretty
    except Exception:
        pass

    hot_spills = []
    for fn, regs, st, sp_st, sp_ld, smem in kernels:
        pretty = demangled_map.get(fn, fn)
        if len(pretty) > 58:
            pretty = pretty[:55] + "..."
        is_hot = (
            "k_gate_gemv_route_fused" in fn
            or "k_select_route_" in fn
            or "k_select_count_multi_fast8" in fn
            or "k_blend_down" in fn
            or "k_blend_up_norm" in fn
        )
        tag = "HOT" if is_hot else "fb "
        print(
            f"  [{tag}] {pretty:58s} regs={regs:3d} smem={smem:5d}B "
            f"stack={st}B spill=({sp_st}B/{sp_ld}B)",
            flush=True,
        )
        if is_hot and (st > 0 or sp_st > 0 or sp_ld > 0):
            hot_spills.append((fn, st, sp_st, sp_ld))

    # Reload into current process from staged .so
    load_ep4_route_ext()
    load_loop_blend_ext()
    ext_dir = Path(__file__).resolve().parents[1] / "inkling_v100" / "_ext"
    assert (ext_dir / "inkling_ep4_route.so").exists(), "Missing inkling_ep4_route.so"
    assert (ext_dir / "inkling_loop_blend.so").exists(), "Missing inkling_loop_blend.so"
    print(f"  Staged .so verified under {ext_dir}", flush=True)
    if hot_spills:
        raise AssertionError(f"Hot kernels have stack/spills: {hot_spills}")
    print("  PASS: 0 bytes stack frame and 0 bytes register spills across ALL hot kernels!\n", flush=True)


# ---------------------------------------------------------------------------
# 2. Correctness: ep4_route (FP32 oracle + Triton ref + edge cases + CUDAGraph)
# ---------------------------------------------------------------------------

def fp32_route_oracle(
    x: torch.Tensor,
    gw: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    lo: int,
    hi: int,
    is_padding: torch.Tensor | None = None,
):
    T, _ = x.shape
    E = gw.shape[0]
    logits = x.float() @ gw.float().t()  # exact FP32 matmul
    sig = torch.sigmoid(logits)
    sel = sig + bias.float().unsqueeze(0)
    # Deterministic tie-breaking: lower expert index wins.
    ids_g = torch.empty((T, topk), dtype=torch.int32, device=x.device)
    sel_logits = torch.empty((T, topk), dtype=torch.float32, device=x.device)
    for t in range(T):
        # Stable sort descending on sel (tie-breaking on ascending e via Python/argsort)
        vals = sel[t].cpu().tolist()
        order = sorted(range(E), key=lambda e: (-vals[e], e))[:topk]
        ids_g[t] = torch.tensor(order, dtype=torch.int32, device=x.device)
        sel_logits[t] = logits[t, ids_g[t].long()]
    lp = F.logsigmoid(sel_logits)
    w = torch.softmax(lp, dim=-1) * route_scale
    local = torch.where((ids_g >= lo) & (ids_g < hi), ids_g - lo, -1)
    if is_padding is not None:
        local = torch.where(is_padding[:, None], -1, local)
    rows = torch.arange(T * topk, dtype=torch.int32, device=x.device)
    s2d = torch.where(local.reshape(-1) >= 0, rows, -1)
    perm, gids, goff = ep4_route_to_skinny_groups(local, hi - lo)
    return local, w.reshape(-1), s2d, perm, gids, goff


def _check_route_outputs(tag: str, got, ref, T: int, e_local: int, exact_perm: bool, w_tol: float = 1e-5):
    g_ids, g_w, g_s2d, g_perm, g_gids, g_goff = got
    r_ids, r_w, r_s2d, r_perm, r_gids, r_goff = ref
    assert torch.equal(g_ids, r_ids), f"{tag}: ids mismatch\ngot={g_ids}\nref={r_ids}"
    assert torch.equal(g_s2d, r_s2d), f"{tag}: src2dst mismatch"
    w_err = (g_w - r_w).abs().max().item()
    assert w_err < w_tol, f"{tag}: weights max_err={w_err:.3e} >= {w_tol}"
    assert torch.equal(g_goff, r_goff), f"{tag}: goff mismatch\ngot={g_goff}\nref={r_goff}"
    assert torch.equal(g_gids, r_gids), f"{tag}: gids mismatch\ngot={g_gids}\nref={r_gids}"
    n_loc = int(r_goff[-1].item())
    if exact_perm:
        assert torch.equal(g_perm[:n_loc], r_perm[:n_loc]), f"{tag}: perm mismatch"
    else:
        for g in range(e_local):
            a, b = int(r_goff[g].item()), int(r_goff[g + 1].item())
            if a == b:
                break
            assert torch.equal(torch.sort(g_perm[a:b]).values, torch.sort(r_perm[a:b]).values), (
                f"{tag}: perm set mismatch at group {g}"
            )
    return w_err


def test_ep4_route_correctness() -> None:
    print("=" * 78)
    print("2. VERIFYING ep4_route_cuda CORRECTNESS & CUDA GRAPH SAFETY")
    print("=" * 78, flush=True)

    torch.manual_seed(0)
    H, E, TOPK, LO, HI = 2304, 128, 8, 32, 64
    gw = torch.randn(E, H, dtype=torch.float16, device=DEVICE) * 0.05
    bias = torch.randn(E, dtype=torch.float32, device=DEVICE) * 0.1
    bounds = torch.tensor([LO, HI], dtype=torch.int32, device=DEVICE)

    preallocate_ep4_route_scratch(DEVICE)

    # 2a. Sweep T = 1, 2, 4, 8, 16, 32, 128, 1024 (including partial padding & all-padding)
    cases = [
        (1, False, False),
        (2, False, False),
        (4, False, False),
        (8, False, False),
        (16, False, False),
        (32, False, False),
        (128, False, False),
        (1024, False, False),
        (1, True, False),    # T=1 all-padding (all sentinel)
        (8, True, False),    # T=8 all-padding (all sentinel)
        (16, True, False),   # T=16 all-padding
        (32, True, False),   # T=32 all-padding
        (128, True, False),  # T=128 all-padding
        (1, False, True),    # T=1 all-remote (all sentinel)
        (8, False, True),    # T=8 all-remote (all sentinel)
    ]
    for T, pad_all, all_remote in cases:
        x = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25
        is_pad = torch.ones(T, dtype=torch.bool, device=DEVICE) if pad_all else None
        if not pad_all and T >= 4 and not all_remote:
            is_pad = torch.zeros(T, dtype=torch.bool, device=DEVICE)
            is_pad[-1] = True
            if T >= 16:
                is_pad[0] = True

        bias_use = bias.clone()
        if all_remote:
            # Push all local experts [LO..HI) way down so zero local experts are chosen
            bias_use[LO:HI] = -100.0

        r_ids, r_w, r_s2d = select_experts_sm70(x, gw, bias_use, TOPK, 8.0, bounds, is_pad)
        r_perm, r_gids, r_goff = ep4_route_to_skinny_groups(r_ids, HI - LO)
        ref_triton = (r_ids, r_w, r_s2d, r_perm, r_gids, r_goff)

        # Test fused GEMV path (for T <= 32) AND mm+route path
        modes = [True, False] if T <= 32 else [False]
        errs = []
        for fg in modes:
            got = select_and_route_ep4_cuda(
                x, gw, bias_use, TOPK, 8.0, LO, HI, is_pad, force_gemv=fg
            )
            w_err = _check_route_outputs(
                f"T={T} fg={fg} pad={pad_all} rem={all_remote}",
                got,
                ref_triton,
                T,
                HI - LO,
                exact_perm=(T <= 32),
                w_tol=1e-5,
            )
            errs.append(w_err)
            if T <= 16 and fg:
                ref_fp32 = fp32_route_oracle(x, gw, bias_use, TOPK, 8.0, LO, HI, is_pad)
                _check_route_outputs(
                    f"FP32-oracle T={T}", got, ref_fp32, T, HI - LO, exact_perm=True, w_tol=1e-5
                )

        n_loc = int(r_goff[-1].item())
        print(
            f"  PASS T={T:4d} pad_all={str(pad_all):5s} all_rem={str(all_remote):5s} "
            f"n_local={n_loc:4d} max_w_err={max(errs):.2e}",
            flush=True,
        )

    # 2b. Explicit deterministic tie-breaking test (identical weight rows + identical bias)
    gw_tie = gw.clone()
    gw_tie[32:48].copy_(gw_tie[32:33].expand(16, H))  # 16 identical rows in local range
    bias_tie = torch.full((E,), -10.0, dtype=torch.float32, device=DEVICE)
    bias_tie[32:48] = 5.0  # all 16 tie identically at top -> experts 32..39 (local 0..7) MUST win!
    for T in (1, 2, 4, 8, 16, 32):
        x_tie = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25
        for fg in (True, False):
            got = select_and_route_ep4_cuda(x_tie, gw_tie, bias_tie, TOPK, 8.0, LO, HI, force_gemv=fg)
            expected_ids = torch.arange(8, dtype=torch.int32, device=DEVICE).unsqueeze(0).expand(T, 8)
            assert torch.equal(got[0], expected_ids), (
                f"Tie-breaking failed for T={T} fg={fg}: got {got[0]}"
            )
            ref_fp32 = fp32_route_oracle(x_tie, gw_tie, bias_tie, TOPK, 8.0, LO, HI)
            _check_route_outputs(f"tie T={T} fg={fg}", got, ref_fp32, T, HI - LO, exact_perm=True)
    print("  PASS tie-breaking (identical logits+bias -> lower expert index wins, T=1..32)!", flush=True)

    # 2c. CUDA Graph capture & multi-replay with mutated inputs
    for T in (1, 2, 4, 8, 16, 32, 128):
        x_g = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25
        fg = True if T <= 32 else False
        # warmup
        select_and_route_ep4_cuda(x_g, gw, bias, TOPK, 8.0, LO, HI, force_gemv=fg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_g = select_and_route_ep4_cuda(x_g, gw, bias, TOPK, 8.0, LO, HI, force_gemv=fg)
        for rep in range(5):
            x_g.copy_(torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25)
            g.replay()
            r_ids, r_w, r_s2d = select_experts_sm70(x_g, gw, bias, TOPK, 8.0, bounds)
            r_perm, r_gids, r_goff = ep4_route_to_skinny_groups(r_ids, HI - LO)
            _check_route_outputs(
                f"CUDAGraph T={T} rep={rep}",
                out_g,
                (r_ids, r_w, r_s2d, r_perm, r_gids, r_goff),
                T,
                HI - LO,
                exact_perm=(T <= 32),
                w_tol=1e-5,
            )
    print("  PASS CUDA Graph capture & 5x mutated replay (T=1, 2, 4, 8, 16, 32, 128)!\n", flush=True)


# ---------------------------------------------------------------------------
# 3. Correctness: loop_blend_cuda (R=64 Inkling-2-Mini & R=128, S=1/3, broadcast, CUDAGraph)
# ---------------------------------------------------------------------------

def ref_blend_triton_bmm(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    pair = pair_rms_norm(dest, message, out_dtype=torch.float16)
    hidden = torch.bmm(pair, down.transpose(1, 2))
    gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
    return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps, out_dtype=torch.float16)


def ref_blend_fp32_oracle(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    d_n = dest * torch.rsqrt(dest.square().mean(-1, keepdim=True) + 1e-6)
    m_n = message * torch.rsqrt(message.square().mean(-1, keepdim=True) + 1e-6)
    p32 = torch.cat([d_n, m_n], dim=-1)
    h16 = (p32 @ down.float().transpose(1, 2)).half().float()
    gf = h16 @ up.float().transpose(1, 2)
    D = message.shape[-1]
    x = (1.0 - torch.tanh(gf[..., D:])) * dest + torch.tanh(gf[..., :D]) * message
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    return (x * rstd * norm_weight.float().unsqueeze(1)).half()


def _check_blend_err(tag: str, got: torch.Tensor, ref_tri: torch.Tensor, ref_o32: torch.Tensor) -> tuple[float, float, float]:
    assert got.dtype == torch.float16 and got.shape == ref_tri.shape
    diff_o32 = (got.float() - ref_o32.float()).abs()
    max_o32 = diff_o32.max().item()
    cos_o32 = F.cosine_similarity(got.float().reshape(-1), ref_o32.float().reshape(-1), dim=0).item()

    diff_tri = (got.float() - ref_tri.float()).abs()
    max_tri = diff_tri.max().item()
    mean_tri = diff_tri.mean().item()
    cos_tri = F.cosine_similarity(got.float().reshape(-1), ref_tri.float().reshape(-1), dim=0).item()
    # 1. Exact FP32 oracle: strict <= 2e-3 everywhere |y| < 4.0 and <= 1 FP16 ULP everywhere |y| >= 4.0,
    #    mean <= 1e-4, cos >= 0.999999.
    o32_ulp = (torch.nextafter(ref_o32.abs(), torch.full_like(ref_o32, float("inf"))) - ref_o32.abs()).float()
    tol_o32 = torch.maximum(torch.full_like(diff_o32, 2e-3), o32_ulp * 1.01)

    assert (diff_o32 <= tol_o32).all().item() and cos_o32 >= 0.99999 and diff_o32.mean().item() <= 1e-4, (
        f"{tag} vs FP32-oracle FAIL: max_err={max_o32:.4e} mean={diff_o32.mean().item():.4e} cos={cos_o32:.7f}"
    )

    # 2. Triton+FP16-bmm reference: ref_tri itself has FP16 pair & bmm quantization error `tri_vs_o32`,
    #    so verify `got` stays within `tri_vs_o32 + tol_o32`, mean <= 5e-4, and cos >= 0.99999.
    tri_vs_o32 = (ref_tri.float() - ref_o32.float()).abs()
    assert (diff_tri <= tri_vs_o32 + tol_o32).all().item() and cos_tri >= 0.99999 and mean_tri <= 5e-4, (
        f"{tag} vs Triton-bmm FAIL: max_err={max_tri:.4e} mean={mean_tri:.4e} cos={cos_tri:.7f}"
    )
    return max_o32, max_tri, cos_tri


def test_loop_blend_correctness() -> None:
    print("=" * 78)
    print("3. VERIFYING loop_blend_cuda CORRECTNESS & CUDA GRAPH SAFETY")
    print("=" * 78, flush=True)

    torch.manual_seed(42)
    D = 2304
    EPS = 1e-6
    preallocate_loop_blend_scratch(DEVICE, D)

    for R in (64, 128):
        print(f"  --- Testing gate_rank R={R} ---", flush=True)
        for S, broadcast_dest, broadcast_nw in [
            (1, False, False),
            (3, False, False),
            (3, True, True),   # local layer: dest.stride(0)==0 and norm_weight.stride(0)==0!
            (3, True, False),
        ]:
            down = torch.randn(S, R, 2 * D, dtype=torch.float16, device=DEVICE) * (2.0 * D) ** -0.5
            up = torch.randn(S, 2 * D, R, dtype=torch.float16, device=DEVICE) * (R ** -0.5)
            nw_base = (
                torch.randn(1, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0
                if broadcast_nw
                else torch.randn(S, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0
            )
            nw = nw_base.expand(S, D) if broadcast_nw else nw_base
            assert (nw.stride(0) == 0) == (broadcast_nw and S > 1)

            worst_o32 = 0.0
            worst_tri = 0.0
            min_cos = 1.0
            for T in (1, 2, 4, 8, 16, 32):
                dest_base = (
                    torch.randn(1, T, D, dtype=torch.float32, device=DEVICE) * 0.5
                    if broadcast_dest
                    else torch.randn(S, T, D, dtype=torch.float32, device=DEVICE) * 0.5
                )
                dest = dest_base.expand(S, T, D) if broadcast_dest else dest_base
                assert (dest.stride(0) == 0) == (broadcast_dest and S > 1)
                msg = torch.randn(S, T, D, dtype=torch.float32, device=DEVICE) * 0.5

                ref_tri = ref_blend_triton_bmm(dest, msg, down, up, nw, EPS)
                ref_o32 = ref_blend_fp32_oracle(dest, msg, down, up, nw, EPS)
                got = loop_blend_cuda(dest, msg, down, up, nw, EPS, force_cuda=True)

                e_o32, e_tri, cos = _check_blend_err(
                    f"R={R} S={S} T={T} bcast=({broadcast_dest},{broadcast_nw})",
                    got,
                    ref_tri,
                    ref_o32,
                )
                worst_o32 = max(worst_o32, e_o32)
                worst_tri = max(worst_tri, e_tri)
                min_cos = min(min_cos, cos)
            print(
                f"  PASS R={R:3d} S={S} bcast=({str(broadcast_dest):5s},{str(broadcast_nw):5s}) "
                f"T in [1..32]: max_o32={worst_o32:.2e} max_tri={worst_tri:.2e} min_cos={min_cos:.7f}",
                flush=True,
            )

        # Extreme scales & zeros check (R=64 and R=128)
        S, T = 3, 4
        down = torch.randn(S, R, 2 * D, dtype=torch.float16, device=DEVICE) * 0.02
        up = torch.randn(S, 2 * D, R, dtype=torch.float16, device=DEVICE) * 0.08
        nw = (torch.randn(1, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0).expand(S, D)
        for scale_name, sc in [("zeros", 0.0), ("tiny_1e-3", 1e-3), ("large_10", 10.0)]:
            dest = (torch.randn(1, T, D, dtype=torch.float32, device=DEVICE) * sc).expand(S, T, D)
            msg = torch.randn(S, T, D, dtype=torch.float32, device=DEVICE) * sc
            ref_tri = ref_blend_triton_bmm(dest, msg, down, up, nw, EPS)
            ref_o32 = ref_blend_fp32_oracle(dest, msg, down, up, nw, EPS)
            got = loop_blend_cuda(dest, msg, down, up, nw, EPS, force_cuda=True)
            if sc == 0.0:
                assert (got.float() - ref_tri.float()).abs().max().item() == 0.0
            else:
                _check_blend_err(f"scale={scale_name} R={R}", got, ref_tri, ref_o32)
        print(f"  PASS R={R:3d} edge scales (zeros, 1e-3, 10.0)!", flush=True)

        # CUDA Graph capture & 5x mutated replay for S=1 and S=3 (broadcast) at T=1, 2, 4, 8, 16, 32
        for S, bcast in [(1, False), (3, True), (3, False)]:
            down = torch.randn(S, R, 2 * D, dtype=torch.float16, device=DEVICE) * 0.02
            up = torch.randn(S, 2 * D, R, dtype=torch.float16, device=DEVICE) * 0.08
            nw_b = torch.randn(1 if bcast else S, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0
            nw = nw_b.expand(S, D) if bcast else nw_b
            for T in (1, 2, 4, 8, 16, 32):
                dest_b = torch.randn(1 if bcast else S, T, D, dtype=torch.float32, device=DEVICE) * 0.5
                dest = dest_b.expand(S, T, D) if bcast else dest_b
                msg = torch.randn(S, T, D, dtype=torch.float32, device=DEVICE) * 0.5
                loop_blend_cuda(dest, msg, down, up, nw, EPS, force_cuda=True)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out_g = loop_blend_cuda(dest, msg, down, up, nw, EPS, force_cuda=True)
                for rep in range(5):
                    dest_b.copy_(torch.randn_like(dest_b) * 0.5)
                    msg.copy_(torch.randn_like(msg) * 0.5)
                    g.replay()
                    ref_tri = ref_blend_triton_bmm(dest, msg, down, up, nw, EPS)
                    ref_o32 = ref_blend_fp32_oracle(dest, msg, down, up, nw, EPS)
                    _check_blend_err(
                        f"CUDAGraph R={R} S={S} bcast={bcast} T={T} rep={rep}",
                        out_g,
                        ref_tri,
                        ref_o32,
                    )
        print(f"  PASS R={R:3d} CUDA Graph capture & 5x mutated replay (S=1/3, bcast, T=1..32)!", flush=True)

    # Default auto-dispatch + fallback check (T=0, T=1, T=16, T=32, T=64)
    R = 64
    for S in (1, 3):
        down = torch.randn(S, R, 2 * D, dtype=torch.float16, device=DEVICE) * 0.02
        up = torch.randn(S, 2 * D, R, dtype=torch.float16, device=DEVICE) * 0.08
        nw = torch.randn(S, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0
        for T in (0, 1, 16, 32, 64):
            dest = torch.randn(S, T, D, dtype=torch.float32, device=DEVICE) * 0.5
            msg = torch.randn(S, T, D, dtype=torch.float32, device=DEVICE) * 0.5
            got = loop_blend_cuda(dest, msg, down, up, nw, EPS)
            assert got.shape == (S, T, D) and got.dtype == torch.float16
            if T > 0:
                ref_tri = ref_blend_triton_bmm(dest, msg, down, up, nw, EPS)
                ref_o32 = ref_blend_fp32_oracle(dest, msg, down, up, nw, EPS)
                if (S == 1 and T <= 16) or (S > 1 and T <= 32):
                    _check_blend_err(f"auto S={S} T={T}", got, ref_tri, ref_o32)
                else:
                    assert torch.equal(got, ref_tri), f"fallback mismatch S={S} T={T}"
    print("  PASS auto-dispatch & fallback (T=0, 1, 16, 32, 64 for S=1/3)!\n", flush=True)


# ---------------------------------------------------------------------------
# 4. Warm-clock CUDA-Graph Benchmarks on GPU 2
# ---------------------------------------------------------------------------

def warm_gpu(seconds: float = 1.2) -> None:
    a = torch.randn(2048, 2048, dtype=torch.float16, device=DEVICE)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        _ = a @ a
    torch.cuda.synchronize()


def time_cuda_graph(fn, warmup: int = 20, iters: int = 400) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    # Replay a few times before timing
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def benchmark_all() -> None:
    print("=" * 78)
    print("4. WARM-CLOCK STANDALONE CUDA-GRAPH BENCHMARKS (GPU 2, >= 1.2s warmup)")
    print("=" * 78, flush=True)
    warm_gpu(1.2)

    # 4a. MoE Gate GEMV + EP4 Route (48 visits/step)
    print("\n[A] MoE Gate + TopK-8 + EP4 Grouping (H=2304, E=128, topk=8, EP=4; 48 visits/step)")
    print(f"  {'T':>3s} | {'mm+route(us)':>12s} | {'fused(us)':>10s} | {'speedup':>8s} | {'48x save(ms)':>12s} | {'vs Triton+PyT':>13s}")
    print("  " + "-" * 72, flush=True)

    torch.manual_seed(7)
    H, E, TOPK, LO, HI = 2304, 128, 8, 32, 64
    gw = torch.randn(E, H, dtype=torch.float16, device=DEVICE) * 0.05
    bias = torch.randn(E, dtype=torch.float32, device=DEVICE) * 0.1
    bounds = torch.tensor([LO, HI], dtype=torch.int32, device=DEVICE)

    for T in (1, 2, 4, 8, 16, 32):
        x = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25

        def run_triton():
            ids, w, s2d = select_experts_sm70(x, gw, bias, TOPK, 8.0, bounds)
            return ep4_route_to_skinny_groups(ids, HI - LO)

        def run_mm_route():
            return select_and_route_ep4_cuda(x, gw, bias, TOPK, 8.0, LO, HI, force_gemv=False)

        def run_fused():
            return select_and_route_ep4_cuda(x, gw, bias, TOPK, 8.0, LO, HI, force_gemv=True)

        t_tri = time_cuda_graph(run_triton, iters=200)
        t_mm = time_cuda_graph(run_mm_route, iters=400)
        t_fu = time_cuda_graph(run_fused, iters=400)
        save_ms = (t_mm - t_fu) * 48.0 / 1000.0
        print(
            f"  {T:3d} | {t_mm:12.2f} | {t_fu:10.2f} | {t_mm / t_fu:7.2f}x | "
            f"{save_ms:+11.3f}  | {t_tri:8.1f} us",
            flush=True,
        )

    # 4b. Loop Controller blend() for real Inkling-2-Mini R=64 (and R=128)
    D = 2304
    EPS = 1e-6
    for R in (64, 128):
        print(
            f"\n[B] Loop Controller blend() (D=2304, R={R}; 24 looped visits x (S=3 + S=1) / step)"
        )
        print(
            f"  {'T':>3s} | {'S=3 old':>9s} {'S=3 new':>9s} {'sp':>6s} | "
            f"{'S=1 old':>9s} {'S=1 new':>9s} {'sp':>6s} | {'24x old->new (save ms)':>24s}"
        )
        print("  " + "-" * 84, flush=True)
        for T in (1, 2, 4, 8, 16, 32):
            # S = 3 (broadcast local style, matching Inkling-2-Mini local layers)
            down3 = torch.randn(3, R, 2 * D, dtype=torch.float16, device=DEVICE) * 0.02
            up3 = torch.randn(3, 2 * D, R, dtype=torch.float16, device=DEVICE) * 0.08
            nw3 = (torch.randn(1, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0).expand(3, D)
            dest3 = (torch.randn(1, T, D, dtype=torch.float32, device=DEVICE) * 0.5).expand(3, T, D)
            msg3 = torch.randn(3, T, D, dtype=torch.float32, device=DEVICE) * 0.5

            # S = 1 (MLP stream)
            down1 = torch.randn(1, R, 2 * D, dtype=torch.float16, device=DEVICE) * 0.02
            up1 = torch.randn(1, 2 * D, R, dtype=torch.float16, device=DEVICE) * 0.08
            nw1 = torch.randn(1, D, dtype=torch.float16, device=DEVICE) * 0.2 + 1.0
            dest1 = torch.randn(1, T, D, dtype=torch.float32, device=DEVICE) * 0.5
            msg1 = torch.randn(1, T, D, dtype=torch.float32, device=DEVICE) * 0.5

            t3_old = time_cuda_graph(lambda: ref_blend_triton_bmm(dest3, msg3, down3, up3, nw3, EPS))
            t3_new = time_cuda_graph(lambda: loop_blend_cuda(dest3, msg3, down3, up3, nw3, EPS, force_cuda=True))
            t1_old = time_cuda_graph(lambda: ref_blend_triton_bmm(dest1, msg1, down1, up1, nw1, EPS))
            t1_new = time_cuda_graph(lambda: loop_blend_cuda(dest1, msg1, down1, up1, nw1, EPS, force_cuda=True))

            step_old = (t3_old + t1_old) * 24.0 / 1000.0
            step_new = (t3_new + t1_new) * 24.0 / 1000.0
            save_ms = step_old - step_new
            print(
                f"  {T:3d} | {t3_old:8.2f}u {t3_new:8.2f}u {t3_old/t3_new:5.2f}x | "
                f"{t1_old:8.2f}u {t1_new:8.2f}u {t1_old/t1_new:5.2f}x | "
                f"{step_old:5.2f} -> {step_new:5.2f} ({save_ms:+.2f} ms)",
                flush=True,
            )


if __name__ == "__main__":
    if "--no-rebuild" not in sys.argv:
        build_and_audit_ptxas()
    else:
        load_ep4_route_ext()
        load_loop_blend_ext()
    test_ep4_route_correctness()
    test_loop_blend_correctness()
    benchmark_all()
