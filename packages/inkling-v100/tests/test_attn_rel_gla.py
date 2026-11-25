#!/usr/bin/env python3
"""Unit tests & V100 microbenchmark for zero-spill `attn_rel_gla.cu` (Post-Prologue GLA-g4 r=256)."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from inkling_v100.ops.attn_rel_dec_cuda import (
    preallocate_workspace as prealloc_std_ws,
    rel_decode_attention_cuda,
)
from inkling_v100.ops.attn_rel_gla_cuda import (
    gla_gate_epilogue_cuda,
    gla_pack_q_cuda,
    gla_scatter_fp8_cuda,
    load_attn_rel_gla_ext,
    plan_rel_gla_decode_splits,
    preallocate_gla_workspace,
    rel_gla_decode_attention_cuda,
    rel_gla_prefill_attention_cuda,
)


def ref_gla_attention_f32(
    q_lat: torch.Tensor,       # [2, T, 3, 256] fp16
    c_lo_u8: torch.Tensor,     # [NB, page, 1, 128] uint8
    c_hi_u8: torch.Tensor,     # [NB, page, 1, 128] uint8
    rel_logits: torch.Tensor,  # [T+128, 6, E+256] fp16
    sinks: torch.Tensor,       # [T, 6] fp16
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    act_tokens: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Exact FP32 reference matching sheared rel_logits + FP8 E4M3FN latent C + sink scaling."""
    T_total = q_lat.shape[1]
    page = c_lo_u8.shape[1]
    E_pad = rel_logits.shape[2]
    E = E_pad - 256
    B = cache_seqlens.shape[0]
    out = torch.zeros_like(q_lat, dtype=torch.float32)

    c_lo_f = c_lo_u8.view(torch.float8_e4m3fn).float()
    c_hi_f = c_hi_u8.view(torch.float8_e4m3fn).float()

    for b in range(B):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        q_len = qe - qs
        eff_q = max(0, min(q_len, act_tokens - qs))
        if eff_q == 0:
            continue
        sk = int(cache_seqlens[b].item())
        past = sk - q_len
        tb = block_table[b]

        # Reconstruct C [sk, 256]
        idx_k = torch.arange(sk, device=q_lat.device)
        blks = tb[idx_k // page].long()
        offs = (idx_k % page).long()
        c_full = torch.cat([c_lo_f[blks, offs, 0], c_hi_f[blks, offs, 0]], dim=-1)  # [sk, 256]

        for tok in range(eff_q):
            token = qs + tok
            pos = past + tok
            tile_end = min(((tok // 128) + 1) * 128, q_len)
            rel_base = ((past + tile_end + 127) // 128) * 128 - E_pad
            for head in range(6):
                g, sub = head // 3, head % 3
                qv = q_lat[g, token, sub].float()  # [256]
                scores = (c_full @ qv) * softmax_scale  # [sk]
                dist = pos - idx_k
                in_rel = (dist >= 0) & (dist < E)
                rel_row = rel_logits[token, head].float()
                rel_idx = (idx_k - rel_base).clamp(0, E_pad - 1)
                scores = scores + torch.where(in_rel, rel_row[rel_idx], torch.zeros_like(scores))
                scores = scores.masked_fill(dist < 0, float("-inf"))
                lse = torch.logsumexp(scores, dim=0)
                probs = torch.softmax(scores, dim=0)
                ov = probs @ c_full  # [256]
                sink_scale = torch.sigmoid(lse - sinks[token, head].float())
                out[g, token, sub] = ov * sink_scale
    return out.half()


def test_scatter_and_helpers() -> None:
    torch.manual_seed(42)
    dev = torch.device("cuda")
    T, NB, page = 137, 8, 64
    c_raw = (torch.randn(T, 256, device=dev, dtype=torch.float16) * 120.0)
    # Mix normal, subnormal, extreme saturation, and zero
    c_raw[0, :16] = torch.tensor(
        [0.0, -0.0, 1e-4, 2**-9, 2**-8, 2**-6, -2**-6, 0.5, -1.25, 100.0, 448.0, 500.0, -448.0, -999.0, 0.003, -0.007],
        dtype=torch.float16, device=dev,
    )
    slots = torch.randperm(NB * page, device=dev, dtype=torch.int64)[:T]
    slots[-7:] = -1  # CUDA-graph pad slots
    c_lo = torch.zeros((NB, page, 1, 128), dtype=torch.uint8, device=dev)
    c_hi = torch.zeros((NB, page, 1, 128), dtype=torch.uint8, device=dev)
    gla_scatter_fp8_cuda(c_raw, c_lo, c_hi, slots)

    ref_u8 = c_raw.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    for t in range(T):
        s = int(slots[t].item())
        if s < 0:
            continue
        blk, off = s // page, s % page
        assert torch.equal(c_lo[blk, off, 0], ref_u8[t, :128]), f"c_lo mismatch at t={t}"
        assert torch.equal(c_hi[blk, off, 0], ref_u8[t, 128:]), f"c_hi mismatch at t={t}"
    print("PASS: gla_scatter_fp8 bit-for-bit exact vs torch.float8_e4m3fn!")

    # Pack Q + Gate epilogue exactness
    q_in = torch.randn(T, 6, 128, device=dev, dtype=torch.float16)
    q_pack = torch.empty((2, T, 3, 128), device=dev, dtype=torch.float16)
    gla_pack_q_cuda(q_in, q_pack)
    q_ref = q_in.view(T, 2, 3, 128).transpose(0, 1).contiguous()
    assert torch.equal(q_pack, q_ref), "gla_pack_q mismatch!"

    v_out = torch.randn(2, T, 3, 128, device=dev, dtype=torch.float16)
    gate = torch.randn(T, 768, device=dev, dtype=torch.float16)
    out = torch.empty((T, 768), device=dev, dtype=torch.float16)
    gla_gate_epilogue_cuda(v_out, gate, out)
    out_ref = (v_out.transpose(0, 1).reshape(T, 768).float() * torch.sigmoid(gate.float())).half()
    err = (out.float() - out_ref.float()).abs().max().item()
    assert err <= 1e-3, f"gla_gate_epilogue err={err}"
    print(f"PASS: gla_pack_q exact, gla_gate_epilogue max_err={err:.6f}!")


def test_decode_and_prefill() -> None:
    torch.manual_seed(123)
    dev = torch.device("cuda")
    scale = 128.0 ** -0.5
    E = 1024
    E_pad = E + 256

    cases_dec = [
        # (B, Q, ctx, page, S, pad_toks)
        (1, 1, 37, 64, 1, 0),
        (1, 1, 2048, 64, 32, 0),
        (2, 6, 1500, 64, 20, 4),
        (4, 4, 4096, 64, 10, 0),
        (8, 4, 8192, 64, 10, 8),
        (1, 8, 513, 32, 16, 0),
        (2, 8, 1025, 128, 8, 0),
        (2, 4, 600, 256, 4, 0),
    ]
    preallocate_gla_workspace(slots=65536, device=dev)

    for B, Q, ctx, page, S, pad_toks in cases_dec:
        act_toks = B * Q
        T_total = act_toks + pad_toks
        n_blks_per_req = (ctx + page - 1) // page
        NB = B * n_blks_per_req + 4
        # Realistic latent values inside [-15, 15]
        c_lo = (torch.randn(NB, page, 1, 128, device=dev) * 2.5).to(torch.float8_e4m3fn).view(torch.uint8)
        c_hi = (torch.randn(NB, page, 1, 128, device=dev) * 2.5).to(torch.float8_e4m3fn).view(torch.uint8)
        q_lat = (torch.randn(2, T_total, 3, 256, device=dev, dtype=torch.float16) * 0.35)
        rel_logits = (torch.randn(T_total + 128, 6, E_pad, device=dev, dtype=torch.float16) * 0.4)
        sinks = (torch.randn(T_total, 6, device=dev, dtype=torch.float16) * 0.5)
        perm = torch.randperm(NB, device=dev, dtype=torch.int32)
        bt = perm[: B * n_blks_per_req].view(B, n_blks_per_req).contiguous()
        seqlens = torch.tensor([ctx - (b % 3) * 7 for b in range(B)], dtype=torch.int32, device=dev)
        cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * Q

        o_lat = torch.empty_like(q_lat)
        rel_gla_decode_attention_cuda(
            q_lat, c_lo, c_hi, rel_logits, sinks, o_lat,
            block_table=bt, cache_seqlens=seqlens, cu_seqlens_q=cu_q,
            max_query_len=Q, num_actual_tokens=act_toks, softmax_scale=scale, num_splits=S,
        )
        ref = ref_gla_attention_f32(q_lat, c_lo, c_hi, rel_logits, sinks, bt, seqlens, cu_q, act_toks, scale)
        diff = (o_lat.float() - ref.float()).abs()
        max_e = float(diff.max().item())
        mean_e = float(diff.mean().item())
        if pad_toks > 0:
            assert o_lat[:, act_toks:].abs().max().item() == 0.0, "Non-zero padded rows!"
        assert max_e < 0.02, f"DECODE FAIL B={B} Q={Q} ctx={ctx} page={page} S={S}: max_e={max_e}"
        print(f"PASS DEC: B={B:2d} Q={Q} ctx={ctx:4d} page={page:3d} S={S:2d} pad={pad_toks} | max_err={max_e:.5f} mean={mean_e:.6f}")

    # Prefill tests (unchunked + chunked + variable-length batch)
    cases_pf = [
        ([37], [37], 64),
        ([128, 73], [128, 73], 64),
        ([256, 190], [2304, 1500], 64),   # chunked prefill with past > 0!
        ([512], [4608], 64),              # 4K+ chunked prefill!
        ([65], [65], 32),
        ([130], [500], 256),
    ]
    for qlens, slens, page in cases_pf:
        B = len(qlens)
        act_toks = sum(qlens)
        T_total = act_toks + 5
        max_s = max(slens)
        n_blks_per_req = (max_s + page - 1) // page
        NB = B * n_blks_per_req + 4
        c_lo = (torch.randn(NB, page, 1, 128, device=dev) * 2.5).to(torch.float8_e4m3fn).view(torch.uint8)
        c_hi = (torch.randn(NB, page, 1, 128, device=dev) * 2.5).to(torch.float8_e4m3fn).view(torch.uint8)
        q_lat = (torch.randn(2, T_total, 3, 256, device=dev, dtype=torch.float16) * 0.35)
        rel_logits = (torch.randn(T_total + 128, 6, E_pad, device=dev, dtype=torch.float16) * 0.4)
        sinks = (torch.randn(T_total, 6, device=dev, dtype=torch.float16) * 0.5)
        bt = torch.randperm(NB, device=dev, dtype=torch.int32)[: B * n_blks_per_req].view(B, n_blks_per_req).contiguous()
        seqlens = torch.tensor(slens, dtype=torch.int32, device=dev)
        cu_q = torch.tensor([0] + [sum(qlens[: i + 1]) for i in range(B)], dtype=torch.int32, device=dev)

        o_lat = torch.empty_like(q_lat)
        rel_gla_prefill_attention_cuda(
            q_lat, c_lo, c_hi, rel_logits, sinks, o_lat,
            block_table=bt, cache_seqlens=seqlens, cu_seqlens_q=cu_q,
            max_query_len=max(qlens), num_actual_tokens=act_toks, softmax_scale=scale,
        )
        ref = ref_gla_attention_f32(q_lat, c_lo, c_hi, rel_logits, sinks, bt, seqlens, cu_q, act_toks, scale)
        diff = (o_lat.float() - ref.float()).abs()
        max_e = float(diff.max().item())
        mean_e = float(diff.mean().item())
        assert o_lat[:, act_toks:].abs().max().item() == 0.0, "Non-zero padded prefill rows!"
        assert max_e < 0.02, f"PREFILL FAIL qlens={qlens} slens={slens}: max_e={max_e}"
        print(f"PASS PF : qlens={qlens} slens={slens} page={page:3d} | max_err={max_e:.5f} mean={mean_e:.6f}")


def bench_decode_vs_std() -> None:
    dev = torch.device("cuda")
    scale = 128.0 ** -0.5
    E_pad = 1280
    preallocate_gla_workspace(slots=65536, device=dev)
    prealloc_std_ws(slots=65536, device=dev)

    print("\n=== Standalone V100 Global Decode Microbenchmark: GLA-g4 (r=256) vs Std GQA (G=3, H_kv=2) ===")
    for ctx in [1024, 8192, 16384, 32768]:
        for B, Q in [(1, 8), (8, 4), (16, 4)]:
            T = B * Q
            page_std, page_gla = 32, 64
            nb_s = B * ((ctx + page_std - 1) // page_std)
            nb_g = B * ((ctx + page_gla - 1) // page_gla)
            kc_s = torch.randint(0, 60, (nb_s, page_std, 2, 128), dtype=torch.uint8, device=dev)
            vc_s = torch.randint(0, 60, (nb_s, page_std, 2, 128), dtype=torch.uint8, device=dev)
            c_lo = torch.randint(0, 60, (nb_g, page_gla, 1, 128), dtype=torch.uint8, device=dev)
            c_hi = torch.randint(0, 60, (nb_g, page_gla, 1, 128), dtype=torch.uint8, device=dev)

            q = torch.randn(T, 6, 128, dtype=torch.float16, device=dev)
            q_pack = torch.empty((2, T, 3, 128), dtype=torch.float16, device=dev)
            u_k_t = torch.randn(2, 128, 256, dtype=torch.float16, device=dev) * 0.05
            u_v = torch.randn(2, 256, 128, dtype=torch.float16, device=dev) * 0.05
            w_down = torch.randn(512, 256, dtype=torch.float16, device=dev) * 0.05
            kv_post = torch.randn(T, 512, dtype=torch.float16, device=dev)
            rel = torch.randn(T + 128, 6, E_pad, dtype=torch.float16, device=dev)
            gate = torch.randn(T, 768, dtype=torch.float16, device=dev)
            sinks = torch.randn(T, 6, dtype=torch.float16, device=dev)
            out = torch.empty(T, 768, dtype=torch.float16, device=dev)
            o_lat = torch.empty(2, T, 3, 256, dtype=torch.float16, device=dev)
            slots = torch.arange(T, dtype=torch.int64, device=dev)

            bt_s = torch.arange(nb_s, dtype=torch.int32, device=dev).view(B, -1)
            bt_g = torch.arange(nb_g, dtype=torch.int32, device=dev).view(B, -1)
            seqlens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
            cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * Q

            def run_std():
                rel_decode_attention_cuda(
                    q, kc_s, vc_s, rel, gate, sinks, out,
                    block_table=bt_s, cache_seqlens=seqlens, cu_seqlens_q=cu_q,
                    max_query_len=Q, num_actual_tokens=T, softmax_scale=scale,
                    is_local=False, num_splits=0,
                )

            def run_gla():
                c_raw = torch.mm(kv_post, w_down)
                gla_scatter_fp8_cuda(c_raw, c_lo, c_hi, slots)
                gla_pack_q_cuda(q, q_pack)
                q_lat = torch.bmm(q_pack.view(2, 3 * T, 128), u_k_t).view(2, T, 3, 256)
                rel_gla_decode_attention_cuda(
                    q_lat, c_lo, c_hi, rel, sinks, o_lat,
                    block_table=bt_g, cache_seqlens=seqlens, cu_seqlens_q=cu_q,
                    max_query_len=Q, num_actual_tokens=T, softmax_scale=scale, num_splits=0,
                )
                v_out = torch.bmm(o_lat.view(2, 3 * T, 256), u_v).view(2, T, 3, 128)
                gla_gate_epilogue_cuda(v_out, gate, out)

            for _ in range(10):
                run_std()
                run_gla()
            torch.cuda.synchronize()

            # CUDA graph timing for clean comparison
            g_std, g_gla = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_std):
                run_std()
            with torch.cuda.graph(g_gla):
                run_gla()
            torch.cuda.synchronize()

            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            iters = 100
            e0.record()
            for _ in range(iters):
                g_std.replay()
            e1.record()
            torch.cuda.synchronize()
            t_std = e0.elapsed_time(e1) * 1000.0 / iters

            e0.record()
            for _ in range(iters):
                g_gla.replay()
            e1.record()
            torch.cuda.synchronize()
            t_gla = e0.elapsed_time(e1) * 1000.0 / iters

            bw_gla = (B * ctx * 256.0) / (t_gla * 1e-6) / 1e9
            print(
                f"  ctx={ctx:5d} B={B:2d} Q={Q} | std={t_std:6.1f} us | gla_full={t_gla:6.1f} us "
                f"({t_std/t_gla:.2f}x) | GLA_eff_BW={bw_gla:5.1f} GB/s"
            )


if __name__ == "__main__":
    print("Compiling & loading attn_rel_gla (sm_70, --ptxas-options=-v)...")
    load_attn_rel_gla_ext(verbose=True)
    test_scatter_and_helpers()
    test_decode_and_prefill()
    bench_decode_vs_std()
