# SPDX-License-Identifier: Apache-2.0
"""Verify and benchmark fused Volta ep4_route.cu against Triton select + PyTorch ep4_route."""

from __future__ import annotations

import time
import torch

from inkling_v100.ops.ep4_route_cuda import load_ep4_route_ext, select_and_route_ep4_cuda
from inkling_v100.ops.moe_sm70 import ep4_route_to_skinny_groups, select_experts_sm70

DEVICE = torch.device("cuda:0")


def test_correctness() -> None:
    torch.manual_seed(0)
    H, E, TOPK, LO, HI = 2304, 128, 8, 32, 64
    gw = torch.randn(E, H, dtype=torch.float16, device=DEVICE) * 0.05
    bias = torch.randn(E, dtype=torch.float32, device=DEVICE) * 0.1
    bounds = torch.tensor([LO, HI], dtype=torch.int32, device=DEVICE)

    for T, pad_all in [(1, False), (4, False), (8, False), (32, False), (64, False), (128, False), (512, False), (16, True), (128, True)]:
        x = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25
        is_pad = torch.ones(T, dtype=torch.bool, device=DEVICE) if pad_all else None
        if not pad_all and T >= 8:
            is_pad = torch.zeros(T, dtype=torch.bool, device=DEVICE)
            is_pad[-1] = True

        r_ids, r_w, r_s2d = select_experts_sm70(x, gw, bias, TOPK, 8.0, bounds, is_pad)
        r_perm, r_gids, r_goff = ep4_route_to_skinny_groups(r_ids, HI - LO)

        g_ids, g_w, g_s2d, g_perm, g_gids, g_goff = select_and_route_ep4_cuda(
            x, gw, bias, TOPK, 8.0, LO, HI, is_pad
        )
        assert torch.equal(g_ids, r_ids), f"T={T} ids mismatch"
        assert torch.equal(g_s2d, r_s2d), f"T={T} src2dst mismatch"
        assert (g_w - r_w).abs().max().item() < 1e-6, f"T={T} weights err"
        assert torch.equal(g_goff, r_goff), f"T={T} goff mismatch"
        assert torch.equal(g_gids, r_gids), f"T={T} gids mismatch"
        # Check perm sets per active group (exact ascending order for T<=64, same set for T>64)
        n_loc = int(r_goff[-1].item())
        if T <= 16:
            assert torch.equal(g_perm[:n_loc], r_perm[:n_loc]), f"T={T} perm mismatch"
        else:
            for g in range(HI - LO):
                a, b = int(r_goff[g].item()), int(r_goff[g + 1].item())
                if a == b:
                    break
                assert torch.equal(torch.sort(g_perm[a:b]).values, torch.sort(r_perm[a:b]).values), f"T={T} grp {g}"
        print(f"  PASS T={T:3d} pad_all={str(pad_all):5s} n_local={n_loc:4d}", flush=True)

    # CUDA graph capture & replay test at T=1, 32, 128
    for T in (1, 32, 128):
        x = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25
        # warmup
        select_and_route_ep4_cuda(x, gw, bias, TOPK, 8.0, LO, HI)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_t = select_and_route_ep4_cuda(x, gw, bias, TOPK, 8.0, LO, HI)
        x.copy_(torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25)
        g.replay()
        r_ids, r_w, r_s2d = select_experts_sm70(x, gw, bias, TOPK, 8.0, bounds)
        r_perm, r_gids, r_goff = ep4_route_to_skinny_groups(r_ids, HI - LO)
        assert torch.equal(out_t[0], r_ids) and torch.equal(out_t[4], r_gids) and torch.equal(out_t[5], r_goff)
    print("  PASS CUDA graph capture & replay (T=1, 32, 128)!", flush=True)


def bench() -> None:
    torch.manual_seed(1)
    H, E, TOPK, LO, HI = 2304, 128, 8, 32, 64
    gw = torch.randn(E, H, dtype=torch.float16, device=DEVICE) * 0.05
    bias = torch.randn(E, dtype=torch.float32, device=DEVICE) * 0.1
    bounds = torch.tensor([LO, HI], dtype=torch.int32, device=DEVICE)

    # 1s clock warmup
    a = torch.randn(2048, 2048, dtype=torch.float16, device=DEVICE)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        _ = a @ a
    torch.cuda.synchronize()

    print("\nWarm-clock latency (1 layer, CUDA Graph replay, 48x projected savings):", flush=True)
    for T in (1, 8, 32, 64, 128, 512):
        x = torch.randn(T, H, dtype=torch.float16, device=DEVICE) * 0.25

        def run_old():
            ids, w, s2d = select_experts_sm70(x, gw, bias, TOPK, 8.0, bounds)
            return ep4_route_to_skinny_groups(ids, HI - LO)

        def run_new():
            return select_and_route_ep4_cuda(x, gw, bias, TOPK, 8.0, LO, HI)

        def time_graph(fn, iters=200):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0 / iters

        t_old = time_graph(run_old)
        t_new = time_graph(run_new)
        print(
            f"  T={T:3d}: old={t_old:6.1f} us  new={t_new:5.1f} us  "
            f"speedup={t_old/t_new:5.2f}x  (48x save={(t_old-t_new)*48/1000:.2f} ms/step)",
            flush=True,
        )


if __name__ == "__main__":
    load_ep4_route_ext(verbose=True, force_rebuild=True)
    test_correctness()
    bench()
