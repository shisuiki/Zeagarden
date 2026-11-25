# SPDX-License-Identifier: Apache-2.0
"""Exhaustive correctness, CUDA-graph, and warm-clock benchmark suite for `attn_rel_dec.cu`.

Run on a V100 (sm_70) GPU:
    CUDA_VISIBLE_DEVICES=1 python3 tests/test_attn_rel_dec_cuda.py
"""

from __future__ import annotations

import math
import time

import torch

from inkling_v100.ops.attention_sm70 import plan_decode_attention, rel_decode_attention
from inkling_v100.ops.attn_rel_dec_cuda import (
    load_attn_rel_dec_ext,
    plan_rel_decode_splits,
    preallocate_workspace,
    rel_decode_attention_cuda,
)

DEVICE = torch.device("cuda:0")  # mapped to physical GPU 1 via CUDA_VISIBLE_DEVICES=1


def _ref_rel_attention_from_sheared(
    q: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    rel_logits_req: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    past: int,
    softmax_scale: float,
    window: int,
    extent: int,
) -> torch.Tensor:
    """Exact FP32 eager Inkling2 relative-bias gated-sink attention reading sheared rel_logits."""
    q_len, h_q, head_dim = q.shape
    seq_len, h_kv, _ = k_seq.shape
    g = h_q // h_kv
    e_pad = extent + 256

    k_exp = k_seq.float().repeat_interleave(g, dim=1)  # [S, H_q, D]
    v_exp = v_seq.float().repeat_interleave(g, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_exp) * softmax_scale  # [H_q, Q, S]

    q_pos = torch.arange(past, past + q_len, device=q.device)[:, None]
    k_pos = torch.arange(seq_len, device=q.device)[None, :]
    dist = q_pos - k_pos
    visible = dist >= 0
    if window > 0:
        visible = visible & (dist < window)
    in_ext = visible & (dist < extent)

    bias = torch.zeros((h_q, q_len, seq_len), dtype=torch.float32, device=q.device)
    for tok in range(q_len):
        tile_end = min((tok // 128 + 1) * 128, q_len)
        rel_base = ((past + tile_end + 127) // 128) * 128 - e_pad
        valid_k = in_ext[tok].nonzero(as_tuple=False).squeeze(-1)
        if valid_k.numel() > 0:
            bias[:, tok, valid_k] = rel_logits_req[tok, :, valid_k - rel_base].float()

    scores = scores + bias
    scores = torch.where(visible.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
    lse = torch.logsumexp(scores, dim=-1)  # [H_q, Q]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v_exp)  # [Q, H_q, D]
    out = out * torch.sigmoid(gate.float().view(q_len, h_q, head_dim))
    out = out * torch.sigmoid(lse.t() - sinks.float()).unsqueeze(-1)
    return out.reshape(q_len, h_q * head_dim)


def _make_problem(
    specs: list[tuple[int, int]],
    *,
    h_kv: int = 2,
    g: int = 3,
    head_dim: int = 128,
    extent: int = 1024,
    page: int = 64,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
    is_local: bool = True,
    stride_gate: int | None = None,
    stride_sink: int | None = None,
    interleaved_kv: bool = True,
    pad_tokens: int = 0,
    seed: int = 0,
):
    """Build realistic Inkling2 decode/verify inputs (including padded KV & strided gate/sinks)."""
    torch.manual_seed(seed)
    h_q = h_kv * g
    num_reqs = len(specs)
    max_seq = max((p + ql for p, ql in specs), default=1)
    max_blks = max(1, (max_seq + page - 1) // page)
    total_blks = num_reqs * max_blks

    # Interleaved [num_blocks, 2, page, H_kv, 128] matches Inkling2AttentionBackend.get_kv_cache_shape
    if interleaved_kv:
        kv_buf = torch.zeros((total_blks, 2, page, h_kv, head_dim), dtype=kv_dtype, device=DEVICE)
        kc, vc = kv_buf[:, 0], kv_buf[:, 1]
    else:
        kc = torch.zeros((total_blks, page, h_kv, head_dim), dtype=kv_dtype, device=DEVICE)
        vc = torch.zeros((total_blks, page, h_kv, head_dim), dtype=kv_dtype, device=DEVICE)

    # Permute block IDs to thoroughly test non-contiguous paged indirection
    perm = torch.randperm(total_blks, dtype=torch.int32, device=DEVICE)
    bt = perm.view(num_reqs, max_blks).contiguous()

    k_seqs: list[torch.Tensor] = []
    v_seqs: list[torch.Tensor] = []
    for r_i, (past, ql) in enumerate(specs):
        sl = past + ql
        kf = torch.randn(sl, h_kv, head_dim, dtype=torch.float16, device=DEVICE) * 0.5
        vf = torch.randn(sl, h_kv, head_dim, dtype=torch.float16, device=DEVICE) * 0.5
        if kv_dtype == torch.float8_e4m3fn:
            kq = kf.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            vq = vf.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            k_seqs.append(kq.float())
            v_seqs.append(vq.float())
        else:
            kq, vq = kf, vf
            k_seqs.append(kf.float())
            v_seqs.append(vf.float())
        # Vectorized scatter into paged cache
        pos = torch.arange(sl, device=DEVICE)
        blk_ids = bt[r_i, pos // page].long()
        slots = pos % page
        kc[blk_ids, slots] = kq
        vc[blk_ids, slots] = vq

    cu_q = [0]
    for _, ql in specs:
        cu_q.append(cu_q[-1] + ql)
    num_actual = cu_q[-1]
    total_t = num_actual + pad_tokens

    # If pad_tokens > 0, add dummy padded requests or leave cu_seqlens_q shorter (test both!)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
    seq_lens_t = torch.tensor([p + ql for p, ql in specs], dtype=torch.int32, device=DEVICE)

    q = torch.randn(total_t, h_q, head_dim, dtype=torch.float16, device=DEVICE) * 0.5

    # Construct strided gate & sinks (e.g. from qkvr projection with stride_gate=2152!)
    gate_width = h_q * head_dim
    if stride_gate is not None:
        assert stride_gate >= gate_width
        gate_backing = torch.randn(total_t, stride_gate, dtype=torch.float16, device=DEVICE)
        gate = gate_backing[:, :gate_width]
    else:
        gate = torch.randn(total_t, gate_width, dtype=torch.float16, device=DEVICE)

    if stride_sink is not None:
        assert stride_sink >= h_q
        sink_backing = torch.randn(total_t, stride_sink, dtype=torch.float16, device=DEVICE)
        sinks = sink_backing[:, :h_q]
    else:
        sinks = torch.randn(total_t, h_q, dtype=torch.float16, device=DEVICE)

    # Fill sheared rel_logits with -inf outside valid positions (just like shift_rel_proj!)
    # to catch ANY out-of-bounds rel_logits reads immediately!
    e_pad = extent + 256
    rel_logits = torch.full((total_t + 128, h_q, e_pad), float("-inf"), dtype=torch.float16, device=DEVICE)
    for r_i, (past, ql) in enumerate(specs):
        q_s = cu_q[r_i]
        # Vectorized populate for each token in this request
        for tok in range(ql):
            tile_end = min((tok // 128 + 1) * 128, ql)
            rel_base = ((past + tile_end + 127) // 128) * 128 - e_pad
            pos = past + tok
            d_max = min(extent, pos + 1)
            k_lo = pos - d_max + 1
            k_hi = pos + 1
            vals = torch.randn(h_q, d_max, dtype=torch.float16, device=DEVICE) * 0.25
            rel_logits[q_s + tok, :, k_lo - rel_base : k_hi - rel_base] = vals

    return {
        "q": q,
        "kc": kc,
        "vc": vc,
        "rel_logits": rel_logits,
        "gate": gate,
        "sinks": sinks,
        "bt": bt,
        "seq_lens": seq_lens_t,
        "cu_q": cu_q_t,
        "k_seqs": k_seqs,
        "v_seqs": v_seqs,
        "specs": specs,
        "cu_q_list": cu_q,
        "num_actual": num_actual,
        "total_t": total_t,
        "max_q": max((ql for _, ql in specs), default=1),
        "extent": extent,
        "window": extent if is_local else 0,
        "is_local": is_local,
    }


def _compute_ref(prob: dict, scale: float) -> torch.Tensor:
    refs = []
    for r_i, (past, ql) in enumerate(prob["specs"]):
        q_s = prob["cu_q_list"][r_i]
        refs.append(
            _ref_rel_attention_from_sheared(
                prob["q"][q_s : q_s + ql],
                prob["k_seqs"][r_i],
                prob["v_seqs"][r_i],
                prob["rel_logits"][q_s : q_s + ql],
                prob["gate"][q_s : q_s + ql],
                prob["sinks"][q_s : q_s + ql],
                past,
                scale,
                prob["window"],
                prob["extent"],
            )
        )
    ref = torch.cat(refs, dim=0)
    if prob["num_actual"] < prob["total_t"]:
        pad = torch.zeros(
            (prob["total_t"] - prob["num_actual"], ref.shape[1]),
            dtype=torch.float32,
            device=DEVICE,
        )
        ref = torch.cat([ref, pad], dim=0)
    return ref


def run_accuracy_tests() -> list[tuple[str, float, float]]:
    print("=== [1] Building extension & running exhaustive FP32-oracle accuracy tests ===", flush=True)
    load_attn_rel_dec_ext(verbose=False)
    scale = 128**-0.5
    results: list[tuple[str, float, float]] = []

    cases = [
        # (name, specs, h_kv, g, page, kv_dtype, is_local, num_splits, stride_gate, stride_sink, pad_tokens)
        # Decode Q=1: Local SWA-1024 across edge & long seqlens (1, 37, 1024, 1500, 4096)
        ("dec_q1_swa_s1_fp8_p64", [(0, 1), (36, 1), (1023, 1), (1499, 1), (4095, 1)], 2, 3, 64, torch.float8_e4m3fn, True, 1, None, None, 0),
        ("dec_q1_swa_s4_fp8_p64", [(0, 1), (36, 1), (1023, 1), (1499, 1), (4095, 1)], 2, 3, 64, torch.float8_e4m3fn, True, 4, None, None, 0),
        ("dec_q1_swa_s8_fp8_p16", [(0, 1), (15, 1), (36, 1), (1023, 1), (1499, 1), (4095, 1)], 2, 3, 16, torch.float8_e4m3fn, True, 8, 2152, 24, 0),
        ("dec_q1_swa_s16_fp8_p128", [(1023, 1), (1499, 1), (4095, 1)], 2, 3, 128, torch.float8_e4m3fn, True, 16, 2152, 24, 0),
        ("dec_q1_swa_s1_fp16_p16", [(0, 1), (16, 1), (36, 1), (1023, 1), (1499, 1), (4095, 1)], 2, 3, 16, torch.float16, True, 1, 2152, 24, 0),
        ("dec_q1_swa_s8_fp16_p128", [(0, 1), (36, 1), (1023, 1), (1499, 1), (4095, 1)], 2, 3, 128, torch.float16, True, 8, None, None, 0),

        # Decode Q=1: Global (1, 37, 1500, 8192)
        ("dec_q1_glb_s1_fp8_p64", [(0, 1), (36, 1), (1499, 1), (8191, 1)], 2, 3, 64, torch.float8_e4m3fn, False, 1, 2152, 16, 0),
        ("dec_q1_glb_s8_fp8_p16", [(0, 1), (36, 1), (1499, 1), (8191, 1)], 2, 3, 16, torch.float8_e4m3fn, False, 8, 2152, 16, 0),
        ("dec_q1_glb_s16_fp8_p128", [(8191, 1)], 2, 3, 128, torch.float8_e4m3fn, False, 16, None, None, 0),
        ("dec_q1_glb_s16_fp16_p64", [(0, 1), (36, 1), (1499, 1), (8191, 1)], 2, 3, 64, torch.float16, False, 16, 2152, 16, 0),
        ("dec_q1_glb_s1_fp16_p128", [(36, 1), (1499, 1)], 2, 3, 128, torch.float16, False, 1, None, None, 0),

        # Decode B=1, B=8, B=32 (large grid > resident CTAs when B=32, S=16 -> deadlock test!)
        ("dec_q1_B1_fp8_s16", [(4095, 1)], 2, 3, 64, torch.float8_e4m3fn, False, 16, 2152, 16, 0),
        ("dec_q1_B8_fp8_s8", [(100 + 250 * i, 1) for i in range(8)], 2, 3, 64, torch.float8_e4m3fn, False, 8, 2152, 16, 0),
        ("dec_q1_B32_fp8_s16_nodeadlock", [(200 + 40 * i, 1) for i in range(32)], 2, 3, 64, torch.float8_e4m3fn, False, 16, 2152, 16, 0),
        ("dec_q1_B32_fp16_s4", [(100 + 30 * i, 1) for i in range(32)], 2, 3, 64, torch.float16, True, 4, 2152, 16, 0),

        # Speculative Verify Q=2 (6 rows in 1 MMA-8!), Q=4 (12 rows, M_WARPS=2), Q=8 (24 rows, M_WARPS=3)
        ("ver_q2_swa_s4_fp8_p16", [(1020, 2), (35, 2), (0, 2)], 2, 3, 16, torch.float8_e4m3fn, True, 4, 2152, 16, 0),
        ("ver_q4_swa_s1_fp8_p64", [(0, 4), (33, 4), (1020, 4), (1496, 4), (4092, 4)], 2, 3, 64, torch.float8_e4m3fn, True, 1, 2152, 16, 0),
        ("ver_q4_glb_s8_fp8_p16", [(33, 4), (1496, 4), (4092, 4)], 2, 3, 16, torch.float8_e4m3fn, False, 8, 2152, 16, 0),
        ("ver_q4_glb_s8_fp16_p128", [(33, 4), (1496, 4)], 2, 3, 128, torch.float16, False, 8, 2152, 16, 0),
        ("ver_q8_swa_B1_s1_fp8", [(1492, 8)], 2, 3, 64, torch.float8_e4m3fn, True, 1, 2152, 16, 0),
        ("ver_q8_swa_B8_s4_fp8_p16", [(100 + 300 * i, 8) for i in range(8)], 2, 3, 16, torch.float8_e4m3fn, True, 4, 2152, 16, 0),
        ("ver_q8_glb_B8_s8_fp8_p64", [(30 + 600 * i, 8) for i in range(8)], 2, 3, 64, torch.float8_e4m3fn, False, 8, 2152, 16, 0),
        ("ver_q8_glb_B1_s16_fp16", [(4088, 8)], 2, 3, 64, torch.float16, False, 16, 2152, 16, 0),
        ("ver_q8_glb_B8_s1_fp16_p128", [(100 + 200 * i, 8) for i in range(8)], 2, 3, 128, torch.float16, False, 1, None, None, 0),

        # Ragged mixed Q lengths + TP=1 (H_kv=8) + general G (G=6, Q=8 -> 48 rows -> 2 m_tiles!) + S>32 (S=40)
        ("ragged_q_mixed_fp8_s8", [(0, 1), (50, 4), (1100, 8), (30, 2)], 2, 3, 16, torch.float8_e4m3fn, True, 8, 2152, 16, 0),
        ("tp1_hkv8_g3_fp8_s4", [(1200, 1), (37, 1), (2000, 1)], 8, 3, 64, torch.float8_e4m3fn, False, 4, 8608, 32, 0),
        ("general_g6_q8_mtiles2_fp8", [(500, 8), (1200, 8)], 2, 6, 64, torch.float8_e4m3fn, True, 4, None, None, 0),
        ("high_splits_s40_fp8", [(8191, 1)], 2, 3, 64, torch.float8_e4m3fn, False, 40, 2152, 16, 0),

        # Padded tokens (num_actual_tokens < num_tokens) for S=1 and S=8!
        ("pad_tokens_s1_fp8", [(500, 1), (1200, 1), (37, 1)], 2, 3, 64, torch.float8_e4m3fn, True, 1, 2152, 16, 5),
        ("pad_tokens_s8_fp8", [(500, 1), (1200, 1), (37, 1)], 2, 3, 64, torch.float8_e4m3fn, False, 8, 2152, 16, 5),
    ]

    for idx, (name, specs, h_kv, g, page, kv_dtype, is_local, splits, sg, ss, pad_t) in enumerate(cases):
        prob = _make_problem(
            specs, h_kv=h_kv, g=g, page=page, kv_dtype=kv_dtype, is_local=is_local,
            stride_gate=sg, stride_sink=ss, pad_tokens=pad_t, seed=100 + idx,
        )
        out = torch.full((prob["total_t"], h_kv * g * 128), float("nan"), dtype=torch.float16, device=DEVICE)
        rel_decode_attention_cuda(
            prob["q"], prob["kc"], prob["vc"], prob["rel_logits"], prob["gate"], prob["sinks"], out,
            block_table=prob["bt"], cache_seqlens=prob["seq_lens"], cu_seqlens_q=prob["cu_q"],
            max_query_len=prob["max_q"], num_actual_tokens=prob["num_actual"],
            softmax_scale=scale, is_local=is_local, num_splits=splits,
        )
        ref = _compute_ref(prob, scale)
        assert not torch.isnan(out).any(), f"{name}: NaN in output!"
        if pad_t > 0:
            assert torch.equal(out[prob["num_actual"] :], torch.zeros_like(out[prob["num_actual"] :])), (
                f"{name}: padded rows not zeroed!"
            )
        diff = (out.float() - ref).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        results.append((name, max_err, mean_err))
        status = "PASS" if max_err < 1e-3 else "FAIL"
        print(f"  [{status}] {name:30s} max_err={max_err:.3e}  mean_err={mean_err:.3e}", flush=True)
        assert max_err < 1e-3, f"{name}: max_err={max_err:.3e} >= 1e-3"

    # Also test BOTH T_WARPS=2 and T_WARPS=4 for M_WARPS=1, and BOTH T_WARPS=1 and T_WARPS=2 for M_WARPS=3
    print("  Checking explicit T_WARPS overrides (1/2/4) across FP8 & FP16 ...", flush=True)
    for kv_dtype in (torch.float8_e4m3fn, torch.float16):
        dt_tag = "fp8" if kv_dtype == torch.float8_e4m3fn else "fp16"
        for q_len, tw_list in [(1, [2, 4]), (8, [1, 2])]:
            for splits in (1, 8):
                prob = _make_problem(
                    [(1200, q_len), (37, q_len)], h_kv=2, g=3, page=64,
                    kv_dtype=kv_dtype, is_local=False, stride_gate=2152, stride_sink=16, seed=777,
                )
                ref = _compute_ref(prob, scale)
                for tw in tw_list:
                    out = torch.empty((prob["total_t"], 6 * 128), dtype=torch.float16, device=DEVICE)
                    rel_decode_attention_cuda(
                        prob["q"], prob["kc"], prob["vc"], prob["rel_logits"], prob["gate"], prob["sinks"], out,
                        block_table=prob["bt"], cache_seqlens=prob["seq_lens"], cu_seqlens_q=prob["cu_q"],
                        max_query_len=q_len, num_actual_tokens=prob["num_actual"],
                        softmax_scale=scale, is_local=False, num_splits=splits, t_warps_override=tw,
                    )
                    err = (out.float() - ref).abs().max().item()
                    assert err < 1e-3, f"{dt_tag} Q={q_len} S={splits} TW={tw} err={err}"
    # Explicit subnormal & full-E4M3FN code range test (all non-NaN subnormals 0x01..0x07, 0x81..0x87 + normals)
    sub_prob = _make_problem(
        [(255, 1), (127, 4)], h_kv=2, g=3, page=32, kv_dtype=torch.float8_e4m3fn,
        is_local=False, stride_gate=2152, stride_sink=16, seed=888,
    )
    # Inject subnormal E4M3 codes (0x01..0x07, 0x81..0x87) into half the tokens to verify exact 2^-8 unpacking
    sub_codes = torch.tensor([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                              0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87], dtype=torch.uint8, device=DEVICE)
    for r_i, (past, ql) in enumerate(sub_prob["specs"]):
        sl = past + ql
        # Put subnormals in first half of sequence, normal values in second half
        u8_k = sub_codes[torch.randint(0, 16, (sl // 2, 2, 128), device=DEVICE)]
        u8_v = sub_codes[torch.randint(0, 16, (sl // 2, 2, 128), device=DEVICE)]
        sub_prob["k_seqs"][r_i][: sl // 2] = u8_k.view(torch.float8_e4m3fn).float()
        sub_prob["v_seqs"][r_i][: sl // 2] = u8_v.view(torch.float8_e4m3fn).float()
        pos = torch.arange(sl // 2, device=DEVICE)
        blk_ids = sub_prob["bt"][r_i, pos // 32].long()
        slots = pos % 32
        sub_prob["kc"][blk_ids, slots] = u8_k.view(torch.float8_e4m3fn)
        sub_prob["vc"][blk_ids, slots] = u8_v.view(torch.float8_e4m3fn)
    sub_ref = _compute_ref(sub_prob, scale)
    for splits in (1, 4):
        sub_out = torch.empty((sub_prob["total_t"], 6 * 128), dtype=torch.float16, device=DEVICE)
        rel_decode_attention_cuda(
            sub_prob["q"], sub_prob["kc"], sub_prob["vc"], sub_prob["rel_logits"],
            sub_prob["gate"], sub_prob["sinks"], sub_out,
            block_table=sub_prob["bt"], cache_seqlens=sub_prob["seq_lens"], cu_seqlens_q=sub_prob["cu_q"],
            max_query_len=4, num_actual_tokens=sub_prob["num_actual"],
            softmax_scale=scale, is_local=False, num_splits=splits,
        )
        err = (sub_out.float() - sub_ref).abs().max().item()
        assert err < 1e-3, f"e4m3 subnormals S={splits} err={err}"
    print(f"  [PASS] e4m3_subnormals_exact (S=1, S=4) max_err={err:.3e}", flush=True)
    print("  [PASS] All explicit (M_WARPS, T_WARPS) x (FP8, FP16) x (S=1, S=8) overrides < 1e-3!", flush=True)
    return results


def run_cudagraph_tests() -> None:
    print("\n=== [2] CUDA Graph Capture & Replay Tests (with num_actual_tokens < num_tokens) ===", flush=True)
    scale = 128**-0.5
    preallocate_workspace(slots=32768, device=DEVICE)

    for splits in (1, 8):
        for kv_dtype in (torch.float8_e4m3fn, torch.float16):
            dt_tag = "fp8" if kv_dtype == torch.float8_e4m3fn else "fp16"
            # Capture with B=8 padded batch where only 5 tokens are actual, 3 tokens are padding
            # AND requests 5..7 have dummy cu_seqlens_q=[..., 6, 7, 8] + garbage block_table=-1!
            prob1 = _make_problem(
                [(1500, 1), (40, 1), (1024, 1), (3000, 1), (12, 1)],
                h_kv=2, g=3, page=64, kv_dtype=kv_dtype, is_local=False,
                stride_gate=2152, stride_sink=16, pad_tokens=3, seed=501,
            )
            # Extend cu_q, seq_lens, bt to 8 requests with poisonous -1 block_table on padded reqs 5..7
            cu_q = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=DEVICE)
            seq_lens = torch.cat([prob1["seq_lens"], torch.tensor([9999, 9999, 9999], dtype=torch.int32, device=DEVICE)])
            bt = torch.cat([prob1["bt"], torch.full((3, prob1["bt"].shape[1]), -1, dtype=torch.int32, device=DEVICE)], dim=0)
            out = torch.full((8, 6 * 128), 123.0, dtype=torch.float16, device=DEVICE)

            # Warmup outside graph
            rel_decode_attention_cuda(
                prob1["q"], prob1["kc"], prob1["vc"], prob1["rel_logits"], prob1["gate"], prob1["sinks"], out,
                block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                max_query_len=1, num_actual_tokens=5, softmax_scale=scale, is_local=False, num_splits=splits,
            )
            torch.cuda.synchronize()

            # Capture CUDA graph (verify zero allocations by rel_decode_attention_cuda inside capture!)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mem_before = torch.cuda.memory_allocated(DEVICE)
                rel_decode_attention_cuda(
                    prob1["q"], prob1["kc"], prob1["vc"], prob1["rel_logits"], prob1["gate"], prob1["sinks"], out,
                    block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                    max_query_len=1, num_actual_tokens=5, softmax_scale=scale, is_local=False, num_splits=splits,
                )
                mem_after = torch.cuda.memory_allocated(DEVICE)
            assert mem_after == mem_before, f"Memory allocated inside CUDA graph capture: {mem_after - mem_before} B"

            # Poison output, mutate inputs in-place, replay graph, verify against fresh FP32 reference!
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            ref1 = _compute_ref(prob1, scale)
            err1 = (out.float() - ref1).abs().max().item()
            assert not torch.isnan(out).any() and err1 < 1e-3, f"CG replay 1 {dt_tag} S={splits}: err={err1}"
            assert torch.equal(out[5:], torch.zeros_like(out[5:])), "CG padded rows not zero!"

            # Mutate Q & gate in-place and replay a second time to verify graph truly reads updated buffers
            prob2 = _make_problem(
                [(1500, 1), (40, 1), (1024, 1), (3000, 1), (12, 1)],
                h_kv=2, g=3, page=64, kv_dtype=kv_dtype, is_local=False,
                stride_gate=2152, stride_sink=16, pad_tokens=3, seed=999,
            )
            prob1["q"].copy_(prob2["q"])
            prob1["kc"].copy_(prob2["kc"])
            prob1["vc"].copy_(prob2["vc"])
            prob1["rel_logits"].copy_(prob2["rel_logits"])
            prob1["gate"].copy_(prob2["gate"])
            prob1["sinks"].copy_(prob2["sinks"])
            prob1["bt"].copy_(prob2["bt"])
            bt[:5].copy_(prob2["bt"])
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            ref2 = _compute_ref(prob2, scale)
            err2 = (out.float() - ref2).abs().max().item()
            assert not torch.isnan(out).any() and err2 < 1e-3, f"CG replay 2 {dt_tag} S={splits}: err={err2}"
            assert torch.equal(out[5:], torch.zeros_like(out[5:])), "CG padded rows not zero after replay 2!"
            print(f"  [PASS] CUDAGraph {dt_tag:4s} S={splits:<2d} (5 actual + 3 poisoned pad rows): "
                  f"replay1={err1:.3e}, replay2={err2:.3e}, alloc=0B", flush=True)


def _bench_us(fn, warmup_s: float = 1.0, iters: int = 200) -> float:
    """Warm-clock CUDA event microbenchmark (>= warmup_s warmup before timing)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    # Measure with CUDA events inside a tight loop
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_benchmarks() -> None:
    print("\n=== [3] Warm-Clock (>=1s warmup) Microbenchmark: attn_rel_dec.cu vs Phase 0 Triton ===", flush=True)
    scale = 128**-0.5
    h_kv, g, page, extent = 2, 3, 64, 1024
    h_q = h_kv * g
    preallocate_workspace(slots=65536, device=DEVICE)

    # First warm up GPU clocks thoroughly (>= 1.5s)
    dummy = torch.randn(2048, 2048, dtype=torch.float16, device=DEVICE)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        _ = dummy @ dummy
    torch.cuda.synchronize()

    configs = []
    # Full FP8 matrix: (B=1, 8, 32) x (SWA-1024, Global-4K, Global-16K) x (Q=1, Q=8)
    for kv_dtype, b_list in [(torch.float8_e4m3fn, [1, 8, 32]), (torch.float16, [1, 32])]:
        for is_local, ctx in [(True, 1024), (False, 4096), (False, 16384)]:
            for q_len in (1, 8):
                for b in b_list:
                    configs.append((kv_dtype, is_local, ctx, q_len, b))

    print(
        f"{'DType':5s} {'Layer':10s} {'Q':>2s} {'B':>3s} | "
        f"{'Triton(us)':>10s} {'S_tri':>5s} | "
        f"{'CUDA(us)':>10s} {'S_cu':>5s} | {'Speedup':>8s} | {'EffBW(GB/s)':>11s} | {'MaxDiff':>8s}",
        flush=True,
    )
    print("-" * 98, flush=True)

    for kv_dtype, is_local, ctx, q_len, b in configs:
        dt_str = "FP8" if kv_dtype == torch.float8_e4m3fn else "FP16"
        layer_str = "SWA-1024" if is_local else f"Glb-{ctx // 1024}K"
        past = ctx - q_len
        specs = [(past, q_len)] * b
        # Allocate minimal KV blocks needed (share/reuse page pool cleanly across batch)
        num_blks_per_req = (ctx + page - 1) // page
        total_blks = b * num_blks_per_req
        kv_buf = torch.zeros((total_blks, 2, page, h_kv, 128), dtype=kv_dtype, device=DEVICE)
        if kv_dtype == torch.float8_e4m3fn:
            # Realistic scaled FP8 KV distribution (clamped to E4M3FN range)
            kv_f = torch.randn(kv_buf.shape, dtype=torch.float16, device=DEVICE) * 0.5
            kv_buf.copy_(kv_f.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn))
            del kv_f
        else:
            kv_buf.normal_(0.0, 0.5)
        kc, vc = kv_buf[:, 0], kv_buf[:, 1]
        bt = torch.arange(total_blks, dtype=torch.int32, device=DEVICE).view(b, num_blks_per_req).contiguous()
        cu_q = torch.arange(0, b * q_len + 1, q_len, dtype=torch.int32, device=DEVICE)
        seq_lens = torch.full((b,), ctx, dtype=torch.int32, device=DEVICE)
        total_t = b * q_len
        q = torch.randn(total_t, h_q, 128, dtype=torch.float16, device=DEVICE) * 0.25
        gate_backing = torch.randn(total_t, 2152, dtype=torch.float16, device=DEVICE)
        gate = gate_backing[:, : h_q * 128]
        sinks = torch.randn(total_t, h_q, dtype=torch.float16, device=DEVICE)
        rel_logits = torch.randn(total_t + 128, h_q, extent + 256, dtype=torch.float16, device=DEVICE) * 0.1
        out_tri = torch.empty((total_t, h_q * 128), dtype=torch.float16, device=DEVICE)
        out_cu = torch.empty((total_t, h_q * 128), dtype=torch.float16, device=DEVICE)

        s_tri = plan_decode_attention(is_local, b, q_len, h_q, h_kv, extent)
        s_cu = plan_rel_decode_splits(is_local, b, q_len, h_q, h_kv, extent)

        def run_tri():
            rel_decode_attention(
                q, kc, vc, rel_logits, gate, sinks, out_tri,
                block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                max_query_len=q_len, num_actual_tokens=total_t,
                softmax_scale=scale, is_local=is_local, num_splits=s_tri,
            )

        def run_cu():
            rel_decode_attention_cuda(
                q, kc, vc, rel_logits, gate, sinks, out_cu,
                block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                max_query_len=q_len, num_actual_tokens=total_t,
                softmax_scale=scale, is_local=is_local, num_splits=s_cu,
            )

        # Quick parity check vs Triton
        run_tri()
        run_cu()
        torch.cuda.synchronize()
        max_diff = (out_cu.float() - out_tri.float()).abs().max().item()

        iters = 300 if (b * ctx <= 32768) else 150
        us_tri = _bench_us(run_tri, warmup_s=0.35, iters=iters)
        us_cu = _bench_us(run_cu, warmup_s=0.35, iters=iters)
        speedup = us_tri / us_cu

        # Effective KV + rel_logits memory bandwidth (GB/s)
        kv_elem_bytes = 1 if kv_dtype == torch.float8_e4m3fn else 2
        eff_k = min(ctx, extent) if is_local else ctx
        kv_bytes = b * h_kv * eff_k * 128 * 2 * kv_elem_bytes
        rel_bytes = b * q_len * h_q * min(ctx, extent) * 2
        bw_gbs = (kv_bytes + rel_bytes) / (us_cu * 1e-6) / 1e9

        print(
            f"{dt_str:5s} {layer_str:10s} {q_len:2d} {b:3d} | "
            f"{us_tri:10.1f} {s_tri:5d} | "
            f"{us_cu:10.1f} {s_cu:5d} | {speedup:7.2f}x | {bw_gbs:11.1f} | {max_diff:8.2e}",
            flush=True,
        )
        del kv_buf, kc, vc, bt, q, gate_backing, gate, sinks, rel_logits, out_tri, out_cu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_accuracy_tests()
    run_cudagraph_tests()
    run_benchmarks()
    print("\nALL TESTS AND BENCHMARKS PASSED!", flush=True)
