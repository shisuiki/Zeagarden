#!/usr/bin/env python3
"""Multi-GPU TP4 correctness, CUDA-graph stress, and benchmark suite for `all_reduce_conv_mix.cu`."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from inkling_v100.ops.allreduce_conv_mix_cuda import (
    all_reduce_conv_mix_cuda,
    init_tp4_ar_conv_mix,
    load_ar_conv_mix_ext,
)
from inkling_v100.ops.attn_res import attn_res
from inkling_v100.ops.moe_sm70 import combine_sm70
from inkling_v100.ops.sconv import short_conv

H = 2304
W = 4
TOPK = 8
CACHE_H = 2
CACHE_N = 4
CACHE_D = 4096
STREAM_OFF = 256
STREAM_W = 1152


def _worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29671"
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cpu_g = dist.new_group(backend="gloo")

    if rank == 0:
        ext = load_ar_conv_mix_ext(verbose=True)
    dist.barrier(group=cpu_g)
    ext = load_ar_conv_mix_ext(verbose=False)
    ok = init_tp4_ar_conv_mix(group=cpu_g)
    assert ok, f"Rank {rank}: init_tp4_ar_conv_mix failed!"
    if rank == 0:
        print("=== Kernel diagnostics (regs / lmem / smem / occ) ===", flush=True)
        for line in ext.info():
            print(" ", line, flush=True)
            assert "lmem=0 " in line, f"Register spill detected: {line}"

    # 1. Exactness across all 12 template variants + Q=1 decode + Q=8 verify + padding + rank bit-identity
    cases = [
        # (name, T, Q_per_req, n_pad, has_combine, has_res, has_last, n_stack, Q_out, apply_norm)
        ("attn_T1_last0_norm1", 1, 1, 0, False, False, False, 1, 1, True),
        ("attn_T1_last1_norm1", 1, 1, 0, False, False, True, 3, 1, True),
        ("attn_T8_Q1_last1_norm0", 8, 1, 0, False, False, True, 2, 1, False),
        ("attn_T8_Q8_verify_last1_norm1", 8, 8, 0, False, False, True, 4, 1, True),
        ("attn_T16_Q1_pad3_last0_norm1", 16, 1, 3, False, False, False, 2, 1, True),
        ("moe_T1_res0_q1_norm1", 1, 1, 0, True, False, False, 1, 1, True),
        ("moe_T1_res1_q3_norm1", 1, 1, 0, True, True, False, 4, 3, True),
        ("moe_T8_Q8_verify_res1_q3_norm1", 8, 8, 0, True, True, False, 3, 3, True),
        ("moe_T16_Q8_verify_res0_q3_norm0", 16, 8, 0, True, False, False, 2, 3, False),
        ("moe_T32_Q1_pad5_res1_q1_norm1", 32, 1, 5, True, True, False, 4, 1, True),
        ("moe_T64_Q8_verify_res1_q3_norm1", 64, 8, 0, True, True, False, 4, 3, True),
    ]

    for idx, (name, T, qpr, n_pad, has_comb, has_res, has_last, n_stack, Q_out, apply_norm) in enumerate(cases):
        g_shared = torch.Generator(device=dev).manual_seed(1000 + idx)
        g_rank = torch.Generator(device=dev).manual_seed(5000 + idx * 10 + rank)

        n_actual = T - n_pad
        num_reqs = max(1, (n_actual + qpr - 1) // qpr)
        max_blocks = 16
        # Unique physical blocks per request
        bt = torch.arange(num_reqs * max_blocks, dtype=torch.int32, device=dev).view(num_reqs, max_blocks)
        positions = torch.zeros(T, dtype=torch.int64, device=dev)
        seq_idx = torch.zeros(T, dtype=torch.int32, device=dev)
        query_start = torch.zeros(T, dtype=torch.int32, device=dev)
        slot_mapping = torch.full((T,), -1, dtype=torch.int64, device=dev)
        for t in range(n_actual):
            r = t // qpr
            off = t % qpr
            base_pos = 10 + (r % 8) * 3  # exercises cached taps within max_blocks=16!
            pos = base_pos + off
            positions[t] = pos
            seq_idx[t] = r
            query_start[t] = r * qpr
            lblk = pos // CACHE_N
            blk = int(bt[r, lblk].item())
            slot_mapping[t] = blk * CACHE_N + (pos % CACHE_N)

        conv_w = torch.randn((H, W), dtype=torch.float16, device=dev, generator=g_shared) * 0.2
        cache_ref = torch.randn((num_reqs * max_blocks, CACHE_H, CACHE_N, CACHE_D), dtype=torch.float16, device=dev, generator=g_shared) * 0.3
        cache_tst = cache_ref.clone()
        stack = torch.randn((n_stack, T, H), dtype=torch.float32, device=dev, generator=g_shared) * 1.5
        last = torch.randn((T, H), dtype=torch.float32, device=dev, generator=g_shared) * 1.2 if has_last else None
        addend = torch.randn((T, H), dtype=torch.float32, device=dev, generator=g_shared) * 1.1 if has_comb else None
        residual = torch.randn((T, H), dtype=torch.float32, device=dev, generator=g_shared) * 1.3 if has_res else None
        queries = torch.randn((Q_out, H), dtype=torch.float32, device=dev, generator=g_shared) * (H ** -0.5)
        norm_w = (torch.randn((Q_out, H), dtype=torch.float16, device=dev, generator=g_shared) * 0.3 + 1.0) if apply_norm else None

        if has_comb:
            down = torch.randn((T * TOPK, H), dtype=torch.float16, device=dev, generator=g_rank) * 0.4
            mask = torch.rand((T * TOPK,), device=dev, generator=g_rank) < 0.25
            src2dst = torch.where(mask, torch.arange(T * TOPK, dtype=torch.int32, device=dev), -1)
            x_in = None
            comb_in = (down, src2dst)
            local_ref = combine_sm70(down, src2dst, TOPK, torch.float16)
        else:
            x_in = torch.randn((T, H), dtype=torch.float16, device=dev, generator=g_rank) * 0.5
            comb_in = None
            local_ref = x_in

        # Reference: exact rank-order 0..3 FP32 sum -> FP16, then short_conv, then attn_res
        gathered = [torch.empty_like(local_ref) for _ in range(world_size)]
        dist.all_gather(gathered, local_ref)
        ar_ref = sum(g.float() for g in gathered).half()
        conv_out_ref = torch.empty((T, H), dtype=torch.float32, device=dev)
        short_conv(
            ar_ref, conv_w, cache_ref, positions, bt, seq_idx, slot_mapping, query_start,
            STREAM_OFF, STREAM_W, addend, residual, conv_out_ref,
        )
        out_ref = attn_res(
            stack,
            conv_out_ref if last is None else last,
            None if last is None else conv_out_ref,
            queries,
            norm_w,
            1e-6 if norm_w is not None else 0.0,
            True,
        )

        # Fused kernel
        conv_out_tst = torch.empty((T, H), dtype=torch.float32, device=dev)
        out_tst = all_reduce_conv_mix_cuda(
            x=x_in,
            combine=comb_in,
            conv_weight=conv_w,
            cache=cache_tst,
            positions=positions,
            block_table=bt,
            seq_idx=seq_idx,
            slot_mapping=slot_mapping,
            query_start=query_start,
            stream_offset=STREAM_OFF,
            stream_width=STREAM_W,
            addend=addend,
            residual=residual,
            conv_out=conv_out_tst,
            stack=stack,
            last=last,
            queries=queries,
            norm_weight=norm_w,
            norm_eps=1e-6 if norm_w is not None else 0.0,
        )
        torch.cuda.synchronize()

        # Check bit-identical across all 4 ranks!
        outs_all = [torch.empty_like(out_tst) for _ in range(world_size)]
        dist.all_gather(outs_all, out_tst)
        rk_diff = max((outs_all[0] != outs_all[p]).sum().item() for p in range(1, world_size))
        cache_diff = (cache_tst != cache_ref).sum().item()
        conv_err = (conv_out_tst - conv_out_ref).abs().max().item()
        out_err = (out_tst.float() - out_ref.float()).abs().max().item()
        tol = 3e-3 if apply_norm else 2e-4
        assert rk_diff == 0, f"{name}: rank divergence {rk_diff}!"
        assert cache_diff == 0, f"{name}: cache bit diff {cache_diff}!"
        assert conv_err <= 2e-4, f"{name}: conv_err={conv_err}"
        assert out_err <= tol, f"{name}: out_err={out_err} > {tol}"
        if rank == 0:
            print(f"  PASS {name:34s} | rk_diff={rk_diff} cache_diff={cache_diff} conv_err={conv_err:.2e} out_err={out_err:.2e}", flush=True)

    # 2. CUDA-graph capture & 100-replay stress across odd chain (3 calls in graph) + varying T!
    if rank == 0:
        print("\n=== CUDA Graph Odd-Chain Stress (100 replays) ===", flush=True)
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    T = 8
    bt = torch.arange(8 * 4, dtype=torch.int32, device=dev).view(8, 4)
    pos = torch.full((T,), 5, dtype=torch.int64, device=dev)
    sidx = torch.arange(T, dtype=torch.int32, device=dev)
    qs = torch.arange(T, dtype=torch.int32, device=dev)
    sm = (bt[:, 1] * CACHE_N + 1).to(torch.int64)
    cw = torch.randn((H, W), dtype=torch.float16, device=dev) * 0.2
    cache = torch.randn((32, CACHE_H, CACHE_N, CACHE_D), dtype=torch.float16, device=dev) * 0.2
    stack = torch.randn((3, T, H), dtype=torch.float32, device=dev)
    q1 = torch.randn((1, H), dtype=torch.float32, device=dev) * (H ** -0.5)
    nw1 = torch.ones((1, H), dtype=torch.float16, device=dev)
    x_buf = torch.randn((T, H), dtype=torch.float16, device=dev) * 0.5
    co_buf = torch.empty((T, H), dtype=torch.float32, device=dev)

    # Warmup
    _ = all_reduce_conv_mix_cuda(
        x=x_buf, conv_weight=cw, cache=cache, positions=pos, block_table=bt,
        seq_idx=sidx, slot_mapping=sm, query_start=qs, stream_offset=STREAM_OFF,
        stream_width=STREAM_W, addend=None, residual=None, conv_out=co_buf,
        stack=stack, last=None, queries=q1, norm_weight=nw1, norm_eps=1e-6,
    )
    torch.cuda.synchronize()
    dist.barrier(group=cpu_g)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # 3 calls (odd chain!) inside graph
        for _ in range(3):
            out_g = all_reduce_conv_mix_cuda(
                x=x_buf, conv_weight=cw, cache=cache, positions=pos, block_table=bt,
                seq_idx=sidx, slot_mapping=sm, query_start=qs, stream_offset=STREAM_OFF,
                stream_width=STREAM_W, addend=None, residual=None, conv_out=co_buf,
                stack=stack, last=None, queries=q1, norm_weight=nw1, norm_eps=1e-6,
            )
    for it in range(100):
        x_buf.fill_(float((it % 7 + 1) * (rank + 1)) * 0.05)
        g.replay()
    torch.cuda.synchronize()
    outs_g = [torch.empty_like(out_g) for _ in range(world_size)]
    dist.all_gather(outs_g, out_g)
    assert all(torch.equal(outs_g[0], o) for o in outs_g[1:]), "CUDA graph rank divergence!"
    if rank == 0:
        print("  PASS 100x odd-chain CUDA graph replays (bit-identical across 4 GPUs!)", flush=True)

    # 3. Warm-clock CUDA-graph benchmark: fused vs (NCCL + Triton sconv + Triton attn_res + combine)
    if rank == 0:
        print("\n=== Warm-Clock TP4 Benchmark (96 sublayers/step equivalent) ===", flush=True)
    # Warmup clocks ~1s
    t_w = time.perf_counter()
    a_w = torch.randn((2048, 2048), dtype=torch.float16, device=dev)
    while time.perf_counter() - t_w < 1.0:
        a_w = torch.mm(a_w, a_w)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_g)

    for T_b in (1, 4, 8, 16, 32, 64):
        bt = torch.arange(T_b * 4, dtype=torch.int32, device=dev).view(T_b, 4)
        pos = torch.full((T_b,), 5, dtype=torch.int64, device=dev)
        sidx = torch.arange(T_b, dtype=torch.int32, device=dev)
        qs = torch.arange(T_b, dtype=torch.int32, device=dev)
        sm = (bt[:, 1] * CACHE_N + 1).to(torch.int64)
        cache_b = torch.randn((T_b * 4, CACHE_H, CACHE_N, CACHE_D), dtype=torch.float16, device=dev) * 0.2
        stack_b = torch.randn((3, T_b, H), dtype=torch.float32, device=dev)
        add_b = torch.randn((T_b, H), dtype=torch.float32, device=dev)
        res_b = torch.randn((T_b, H), dtype=torch.float32, device=dev)
        co_b = torch.empty((T_b, H), dtype=torch.float32, device=dev)
        down_b = torch.randn((T_b * TOPK, H), dtype=torch.float16, device=dev) * 0.2
        s2d_b = torch.where(torch.arange(T_b * TOPK, device=dev) % 4 == rank,
                            torch.arange(T_b * TOPK, dtype=torch.int32, device=dev), -1)
        q1_b = torch.randn((1, H), dtype=torch.float32, device=dev) * (H ** -0.5)
        nw1_b = torch.ones((1, H), dtype=torch.float16, device=dev)

        N_REP = 100
        # Fused graph (MoE combine + AR + sconv + attn_res)
        for _ in range(5):
            all_reduce_conv_mix_cuda(
                combine=(down_b, s2d_b), conv_weight=cw, cache=cache_b, positions=pos,
                block_table=bt, seq_idx=sidx, slot_mapping=sm, query_start=qs,
                stream_offset=STREAM_OFF, stream_width=STREAM_W, addend=add_b,
                residual=res_b, conv_out=co_b, stack=stack_b, last=None,
                queries=q1_b, norm_weight=nw1_b, norm_eps=1e-6,
            )
        torch.cuda.synchronize()
        dist.barrier(group=cpu_g)
        gf = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gf):
            for _ in range(N_REP):
                all_reduce_conv_mix_cuda(
                    combine=(down_b, s2d_b), conv_weight=cw, cache=cache_b, positions=pos,
                    block_table=bt, seq_idx=sidx, slot_mapping=sm, query_start=qs,
                    stream_offset=STREAM_OFF, stream_width=STREAM_W, addend=add_b,
                    residual=res_b, conv_out=co_b, stack=stack_b, last=None,
                    queries=q1_b, norm_weight=nw1_b, norm_eps=1e-6,
                )
        torch.cuda.synchronize()
        dist.barrier(group=cpu_g)
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        gf.replay()
        ev1.record()
        torch.cuda.synchronize()
        fused_us = ev0.elapsed_time(ev1) * 1000.0 / N_REP

        # Unfused baseline (combine + NCCL all_reduce + Triton sconv + Triton attn_res)
        for _ in range(5):
            c_out = combine_sm70(down_b, s2d_b, TOPK, torch.float16)
            dist.all_reduce(c_out)
            short_conv(c_out, cw, cache_b, pos, bt, sidx, sm, qs, STREAM_OFF, STREAM_W, add_b, res_b, co_b)
            attn_res(stack_b, co_b, None, q1_b, nw1_b, 1e-6, True)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_g)
        gb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb):
            for _ in range(N_REP):
                c_out = combine_sm70(down_b, s2d_b, TOPK, torch.float16)
                dist.all_reduce(c_out)
                short_conv(c_out, cw, cache_b, pos, bt, sidx, sm, qs, STREAM_OFF, STREAM_W, add_b, res_b, co_b)
                attn_res(stack_b, co_b, None, q1_b, nw1_b, 1e-6, True)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_g)
        ev0.record()
        gb.replay()
        ev1.record()
        torch.cuda.synchronize()
        base_us = ev0.elapsed_time(ev1) * 1000.0 / N_REP

        if rank == 0:
            print(
                f"  T={T_b:2d} | unfused={base_us:6.1f} us | fused={fused_us:6.1f} us | "
                f"speedup={base_us / fused_us:4.2f}x | 96-sublayer saved={(base_us - fused_us) * 0.096:5.2f} ms/step",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(_worker, args=(4,), nprocs=4, join=True)
