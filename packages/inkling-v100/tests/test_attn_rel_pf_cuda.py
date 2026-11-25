# SPDX-License-Identifier: Apache-2.0
"""Exhaustive correctness, ptxas-audit, CUDA-graph, and warm-clock benchmark suite for `attn_rel_pf.cu`.

Run on a V100 (sm_70) GPU:
    CUDA_VISIBLE_DEVICES=0 python3 tests/test_attn_rel_pf_cuda.py
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import time

import torch

from inkling_v100.ops.attention_sm70 import MODE_OUT_LSE, gate_sink, rel_attention
from inkling_v100.ops.attn_rel_pf_cuda import (
    load_attn_rel_pf_ext,
    rel_prefill_attention_cuda,
)

DEVICE = torch.device("cuda:0")  # mapped to physical GPU 0 via CUDA_VISIBLE_DEVICES=0
MAX_ERR_TOL = 5e-4


def _run_triton_prefill(
    q: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_actual_tokens: int | None = None,
) -> None:
    """Exact Triton prefill path matching `Inkling2Attention._attention` (rel_attention + gate_sink)."""
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    nt = num_tokens if num_actual_tokens is None else num_actual_tokens
    pf_out = q.new_empty((num_tokens, num_heads, head_dim), dtype=torch.float32)
    lse = q.new_empty((num_heads, num_tokens), dtype=torch.float32)
    if nt > 0:
        rel_attention(
            q[:nt],
            kc,
            vc,
            pf_out[:nt],
            lse[:, :nt],
            rel_logits,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max_query_len,
            softmax_scale=softmax_scale,
            window=window,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
    if nt < num_tokens:
        pf_out[nt:].zero_()
        lse[:, nt:].zero_()
    gate_sink(pf_out, gate, lse.t(), sinks, out)


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
    """Exact FP32 eager Inkling2 relative-bias gated-sink attention (chunked over Q for memory)."""
    q_len, h_q, head_dim = q.shape
    seq_len, h_kv, _ = k_seq.shape
    g = h_q // h_kv
    e_pad = extent + 256
    if q_len == 0:
        return torch.empty((0, h_q * head_dim), dtype=torch.float32, device=q.device)
    if seq_len == 0:
        return torch.zeros((q_len, h_q * head_dim), dtype=torch.float32, device=q.device)

    k_exp = k_seq.float().repeat_interleave(g, dim=1)  # [S, H_q, D]
    v_exp = v_seq.float().repeat_interleave(g, dim=1)
    k_pos = torch.arange(seq_len, device=q.device)[None, :]
    out_chunks = []
    chunk_q = 256
    for q0 in range(0, q_len, chunk_q):
        q1 = min(q0 + chunk_q, q_len)
        qc = q[q0:q1].float()  # [C, H_q, D]
        scores = torch.einsum("qhd,khd->hqk", qc, k_exp) * softmax_scale  # [H_q, C, S]
        q_pos = torch.arange(past + q0, past + q1, device=q.device)[:, None]
        dist = q_pos - k_pos
        visible = dist >= 0
        if window > 0:
            visible = visible & (dist < window)
        in_ext = visible & (dist < extent)

        bias = torch.zeros_like(scores)
        for tok_local in range(q1 - q0):
            tok = q0 + tok_local
            tile_end = min((tok // 128 + 1) * 128, q_len)
            rel_base = ((past + tile_end + 127) // 128) * 128 - e_pad
            valid_k = in_ext[tok_local].nonzero(as_tuple=False).squeeze(-1)
            if valid_k.numel() > 0:
                bias[:, tok_local, valid_k] = rel_logits_req[tok, :, valid_k - rel_base].float()

        scores = scores + bias
        scores = torch.where(visible.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
        has_any = visible.any(dim=-1)  # [C]
        safe_scores = torch.where(has_any[None, :, None], scores, torch.zeros_like(scores))
        lse = torch.where(
            has_any[None, :],
            torch.logsumexp(safe_scores, dim=-1),
            torch.full((h_q, q1 - q0), float("-inf"), dtype=torch.float32, device=q.device),
        )
        probs = torch.where(has_any[None, :, None], torch.softmax(safe_scores, dim=-1), torch.zeros_like(scores))
        oc = torch.einsum("hqk,khd->qhd", probs, v_exp)  # [C, H_q, D]
        oc = oc * torch.sigmoid(gate[q0:q1].float().view(q1 - q0, h_q, head_dim))
        oc = oc * torch.sigmoid(lse.t() - sinks[q0:q1].float()).unsqueeze(-1)
        out_chunks.append(oc.reshape(q1 - q0, h_q * head_dim))
    return torch.cat(out_chunks, dim=0)


def _populate_sheared_rel_logits(
    rel_logits: torch.Tensor,
    specs: list[tuple[int, int]],
    cu_q: list[int],
    extent: int,
) -> None:
    """Populate valid sheared rel_logits positions vectorized by 128-token shear tile."""
    h_q = rel_logits.shape[1]
    e_pad = extent + 256
    for r_i, (past, ql) in enumerate(specs):
        q_s = cu_q[r_i]
        for tile0 in range(0, ql, 128):
            tile_end = min(tile0 + 128, ql)
            rel_base = ((past + tile_end + 127) // 128) * 128 - e_pad
            t_len = tile_end - tile0
            vals = torch.randn((t_len, h_q, e_pad), dtype=torch.float16, device=rel_logits.device) * 0.25
            toks = torch.arange(tile0, tile_end, device=rel_logits.device)[:, None]
            pos = past + toks
            cols = torch.arange(e_pad, device=rel_logits.device)[None, :] + rel_base
            dist = pos - cols
            valid = (cols >= 0) & (dist >= 0) & (dist < extent)
            rel_logits[q_s + tile0 : q_s + tile_end] = torch.where(
                valid[:, None, :], vals, torch.full_like(vals, float("-inf"))
            )


def _make_problem(
    specs: list[tuple[int, int]],
    *,
    h_kv: int = 2,
    g: int = 3,
    head_dim: int = 128,
    extent: int = 1024,
    page: int = 128,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
    window: int = 1024,
    stride_gate: int | None = None,
    stride_sink: int | None = None,
    interleaved_kv: bool = True,
    pad_tokens: int = 0,
    seed: int = 0,
):
    torch.manual_seed(seed)
    h_q = h_kv * g
    num_reqs = len(specs)
    max_seq = max((p + ql for p, ql in specs), default=1)
    max_blks = max(1, (max_seq + page - 1) // page)
    total_blks = max(1, num_reqs * max_blks)

    if interleaved_kv:
        kv_buf = torch.zeros((total_blks, 2, page, h_kv, head_dim), dtype=kv_dtype, device=DEVICE)
        kc, vc = kv_buf[:, 0], kv_buf[:, 1]
    else:
        kc = torch.zeros((total_blks, page, h_kv, head_dim), dtype=kv_dtype, device=DEVICE)
        vc = torch.zeros((total_blks, page, h_kv, head_dim), dtype=kv_dtype, device=DEVICE)

    perm = torch.randperm(total_blks, dtype=torch.int32, device=DEVICE)
    bt = perm[: num_reqs * max_blks].view(num_reqs, max_blks).contiguous()

    k_seqs: list[torch.Tensor] = []
    v_seqs: list[torch.Tensor] = []
    for r_i, (past, ql) in enumerate(specs):
        sl = past + ql
        if sl == 0:
            k_seqs.append(torch.empty((0, h_kv, head_dim), dtype=torch.float32, device=DEVICE))
            v_seqs.append(torch.empty((0, h_kv, head_dim), dtype=torch.float32, device=DEVICE))
            continue
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

    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=DEVICE)
    seq_lens_t = torch.tensor([p + ql for p, ql in specs], dtype=torch.int32, device=DEVICE)

    q = torch.randn(total_t, h_q, head_dim, dtype=torch.float16, device=DEVICE) * 0.5

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

    e_pad = extent + 256
    rel_logits = torch.full((total_t + 128, h_q, e_pad), float("-inf"), dtype=torch.float16, device=DEVICE)
    _populate_sheared_rel_logits(rel_logits, specs, cu_q, extent)

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
        "max_q": max((ql for _, ql in specs), default=0),
        "extent": extent,
        "window": window,
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
    ref = torch.cat(refs, dim=0) if refs else torch.empty((0, prob["gate"].shape[1]), dtype=torch.float32, device=DEVICE)
    if prob["num_actual"] < prob["total_t"]:
        pad = torch.zeros(
            (prob["total_t"] - prob["num_actual"], ref.shape[1]),
            dtype=torch.float32,
            device=DEVICE,
        )
        ref = torch.cat([ref, pad], dim=0)
    return ref


def audit_ptxas_zero_spills() -> None:
    """Compile attn_rel_pf.cu with ptxas -v (via fresh build) and verify 0 B stack & 0 B spills."""
    print("=== [0] Compiling & Auditing ptxas -v (0 B stack, 0 B spills required) ===", flush=True)
    root = pathlib.Path(__file__).resolve().parents[1]
    csrc = root / "inkling_v100" / "csrc" / "attn_rel_pf.cu"
    build_dir = root / "inkling_v100" / "_ext" / "attn_rel_pf"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Remove old .o / .so so ptxas -v always emits full report on audit
    for p in list(build_dir.glob("*.o")) + list(build_dir.glob("*.so")):
        p.unlink(missing_ok=True)

    env = os.environ.copy()
    env["INKLING_REBUILD_EXT"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = "7.0"
    cmd = [
        sys.executable,
        "-c",
        "from inkling_v100.ops.attn_rel_pf_cuda import load_attn_rel_pf_ext; load_attn_rel_pf_ext(verbose=True)",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    combined = proc.stdout + "\n" + proc.stderr

    kernel_lines = []
    lines = combined.splitlines()
    for i, line in enumerate(lines):
        if "Compiling entry function" in line or "Function properties" in line or "stack frame" in line or "Used " in line:
            kernel_lines.append(line)
            print("  " + line, flush=True)

    stack_matches = re.findall(r"(\d+)\s+bytes\s+stack\s+frame,\s+(\d+)\s+bytes\s+spill\s+stores,\s+(\d+)\s+bytes\s+spill\s+loads", combined)
    assert len(stack_matches) > 0, "Failed to capture ptxas -v spill report!"
    for stack_b, spill_st, spill_ld in stack_matches:
        assert int(stack_b) == 0 and int(spill_st) == 0 and int(spill_ld) == 0, (
            f"NON-ZERO STACK/SPILLS DETECTED: stack={stack_b}B, spill_stores={spill_st}B, spill_loads={spill_ld}B"
        )
    print(f"  [PASS] Verified {len(stack_matches)} sm_70 kernels: ALL 0 B stack frame, 0 B spill stores, 0 B spill loads!", flush=True)
    # Load into current process and print Volta occupancy report
    ext = load_attn_rel_pf_ext(verbose=False)
    print("  Volta sm_70 Occupancy Report:\n" + ext.get_occupancy_report().rstrip(), flush=True)


def run_accuracy_tests() -> None:
    print("\n=== [1] Exhaustive FP32-Oracle & Triton Parity Tests (tol <= 5e-4) ===", flush=True)
    scale = 128**-0.5

    cases = [
        # (name, specs, h_kv, g, page, kv_dtype, window, stride_gate, stride_sink, pad_tokens, check_triton)
        # Required q_len sweep: 1, 7, 16, 31, 32, 64, 128, 134, 512, 1024, 3692 (both SWA & Global, FP8 & FP16)
        ("q1_swa_fp8_p128", [(0, 1)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q7_glb_fp8_p128", [(0, 7)], 2, 3, 128, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("q16_swa_fp8_p16", [(0, 16)], 2, 3, 16, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q31_glb_fp16_p16", [(0, 31)], 2, 3, 16, torch.float16, 0, 2152, 16, 0, True),
        ("q32_swa_fp8_p32", [(0, 32)], 2, 3, 32, torch.float8_e4m3fn, 1024, None, None, 0, True),
        ("q64_glb_fp8_p64", [(0, 64)], 2, 3, 64, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("q128_swa_fp8_p128", [(0, 128)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q128_glb_fp16_p128", [(0, 128)], 2, 3, 128, torch.float16, 0, 2152, 16, 0, True),
        ("q134_swa_fp8_p16", [(0, 134)], 2, 3, 16, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q134_glb_fp16_p64", [(0, 134)], 2, 3, 64, torch.float16, 0, None, None, 0, True),
        ("q512_swa_fp8_p128", [(0, 512)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q512_glb_fp8_p128", [(0, 512)], 2, 3, 128, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("q1024_swa_fp8_p128", [(0, 1024)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q1024_glb_fp16_p128", [(0, 1024)], 2, 3, 128, torch.float16, 0, 2152, 16, 0, True),
        ("q3692_swa_fp8_p128", [(0, 3692)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("q3692_glb_fp8_p128", [(0, 3692)], 2, 3, 128, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("q3692_swa_fp16_p128", [(0, 3692)], 2, 3, 128, torch.float16, 1024, 2152, 16, 0, True),
        ("q3692_glb_fp16_p128", [(0, 3692)], 2, 3, 128, torch.float16, 0, 2152, 16, 0, True),

        # Chunked prefill (past > 0)
        ("chunk_p512_q512_swa_fp8", [(512, 512)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("chunk_p512_q512_glb_fp8", [(512, 512)], 2, 3, 128, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("chunk_p2048_q1024_swa_fp8", [(2048, 1024)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("chunk_p2048_q1024_glb_fp8", [(2048, 1024)], 2, 3, 128, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("chunk_p2048_q1024_glb_fp16_p16", [(2048, 1024)], 2, 3, 16, torch.float16, 0, 2152, 16, 0, True),
        ("chunk_unaligned_p37_q200_swa", [(37, 200)], 2, 3, 16, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),

        # Ragged multi-request batches with mixed past
        ("ragged_swa_fp8_p128", [(0, 17), (150, 133), (900, 500), (2000, 1024)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 0, True),
        ("ragged_glb_fp8_p16", [(0, 17), (150, 133), (900, 500), (2000, 1024)], 2, 3, 16, torch.float8_e4m3fn, 0, 2152, 16, 0, True),
        ("ragged_glb_fp16_p64", [(55, 17), (0, 133), (1200, 500), (300, 1024)], 2, 3, 64, torch.float16, 0, 2152, 16, 0, True),

        # General G = 1, 2, 4 & padded tail rows (num_actual_tokens < num_tokens)
        ("g1_swa_fp8_p32", [(100, 130), (0, 65)], 4, 1, 32, torch.float8_e4m3fn, 1024, None, None, 0, True),
        ("g2_glb_fp8_p64", [(300, 200), (10, 90)], 3, 2, 64, torch.float8_e4m3fn, 0, None, None, 0, True),
        ("g4_swa_fp16_p128", [(600, 180), (0, 70)], 2, 4, 128, torch.float16, 1024, None, None, 0, True),
        ("pad_tail_swa_fp8", [(200, 135), (0, 40)], 2, 3, 128, torch.float8_e4m3fn, 1024, 2152, 16, 19, True),
        ("pad_tail_glb_fp16", [(1200, 260)], 2, 3, 16, torch.float16, 0, 2152, 16, 7, True),
    ]

    for idx, (name, specs, h_kv, g, page, kv_dtype, window, sg, ss, pad_t, check_tri) in enumerate(cases):
        prob = _make_problem(
            specs, h_kv=h_kv, g=g, page=page, kv_dtype=kv_dtype, window=window,
            stride_gate=sg, stride_sink=ss, pad_tokens=pad_t, seed=200 + idx,
        )
        out_cu = torch.full((prob["total_t"], h_kv * g * 128), float("nan"), dtype=torch.float16, device=DEVICE)
        rel_prefill_attention_cuda(
            prob["q"], prob["kc"], prob["vc"], prob["rel_logits"], prob["gate"], prob["sinks"], out_cu,
            block_table=prob["bt"], cache_seqlens=prob["seq_lens"], cu_seqlens_q=prob["cu_q"],
            max_query_len=prob["max_q"], softmax_scale=scale, window=window,
            num_actual_tokens=prob["num_actual"],
        )
        ref = _compute_ref(prob, scale)
        assert not torch.isnan(out_cu).any(), f"{name}: NaN in CUDA output!"
        if pad_t > 0:
            assert torch.equal(out_cu[prob["num_actual"] :], torch.zeros_like(out_cu[prob["num_actual"] :])), (
                f"{name}: padded rows not zeroed!"
            )
        err_ref = (out_cu.float() - ref).abs().max().item()
        err_tri_str = "n/a"
        if check_tri:
            out_tri = torch.full_like(out_cu, float("nan"))
            _run_triton_prefill(
                prob["q"], prob["kc"], prob["vc"], prob["rel_logits"], prob["gate"], prob["sinks"], out_tri,
                block_table=prob["bt"], cache_seqlens=prob["seq_lens"], cu_seqlens_q=prob["cu_q"],
                max_query_len=prob["max_q"], softmax_scale=scale, window=window,
                num_actual_tokens=prob["num_actual"],
            )
            err_tri = (out_cu.float() - out_tri.float()).abs().max().item()
            err_tri_str = f"{err_tri:.3e}"
            assert err_tri <= MAX_ERR_TOL, f"{name}: vs Triton max_err={err_tri:.3e} > {MAX_ERR_TOL}"

        status = "PASS" if err_ref <= MAX_ERR_TOL else "FAIL"
        print(f"  [{status}] {name:30s} vs_fp32={err_ref:.3e}  vs_triton={err_tri_str}", flush=True)
        assert err_ref <= MAX_ERR_TOL, f"{name}: vs FP32 max_err={err_ref:.3e} > {MAX_ERR_TOL}"
        del prob, out_cu, ref

    # Test ALL 6 template instantiations (cfg 1..6) across FP8 & FP16, SWA & Global, PAGE=16 & 128
    print("  Testing all 6 kernel configs (1x8 + 2x8) x FP8/FP16 ...", flush=True)
    for kv_dtype in (torch.float8_e4m3fn, torch.float16):
        dt_tag = "fp8" if kv_dtype == torch.float8_e4m3fn else "fp16"
        for page in (16, 128):
            for window in (1024, 0):
                prob = _make_problem(
                    [(1100, 145), (37, 73)], h_kv=2, g=3, page=page, kv_dtype=kv_dtype,
                    window=window, stride_gate=2152, stride_sink=16, seed=777,
                )
                ref = _compute_ref(prob, scale)
                for cfg in (1, 2, 3, 4, 5, 6):
                    out = torch.empty((prob["total_t"], 6 * 128), dtype=torch.float16, device=DEVICE)
                    rel_prefill_attention_cuda(
                        prob["q"], prob["kc"], prob["vc"], prob["rel_logits"], prob["gate"], prob["sinks"], out,
                        block_table=prob["bt"], cache_seqlens=prob["seq_lens"], cu_seqlens_q=prob["cu_q"],
                        max_query_len=prob["max_q"], softmax_scale=scale, window=window,
                        num_actual_tokens=prob["num_actual"], config_override=cfg,
                    )
                    err = (out.float() - ref).abs().max().item()
                    assert err <= MAX_ERR_TOL, f"cfg={cfg} {dt_tag} p={page} w={window}: err={err:.3e}"
    print("  [PASS] All 6 configs x FP8/FP16 x PAGE=16/128 x SWA/Global <= 5e-4!", flush=True)

    # E4M3FN subnormal exactness test
    sub_prob = _make_problem(
        [(200, 135)], h_kv=2, g=3, page=32, kv_dtype=torch.float8_e4m3fn,
        window=0, stride_gate=2152, stride_sink=16, seed=888,
    )
    sub_codes = torch.tensor(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87],
        dtype=torch.uint8,
        device=DEVICE,
    )
    sl = 335
    u8_k = sub_codes[torch.randint(0, 16, (sl // 2, 2, 128), device=DEVICE)]
    u8_v = sub_codes[torch.randint(0, 16, (sl // 2, 2, 128), device=DEVICE)]
    sub_prob["k_seqs"][0][: sl // 2] = u8_k.view(torch.float8_e4m3fn).float()
    sub_prob["v_seqs"][0][: sl // 2] = u8_v.view(torch.float8_e4m3fn).float()
    pos = torch.arange(sl // 2, device=DEVICE)
    blk_ids = sub_prob["bt"][0, pos // 32].long()
    slots = pos % 32
    sub_prob["kc"][blk_ids, slots] = u8_k.view(torch.float8_e4m3fn)
    sub_prob["vc"][blk_ids, slots] = u8_v.view(torch.float8_e4m3fn)
    sub_ref = _compute_ref(sub_prob, scale)
    sub_out = torch.empty((sub_prob["total_t"], 6 * 128), dtype=torch.float16, device=DEVICE)
    rel_prefill_attention_cuda(
        sub_prob["q"], sub_prob["kc"], sub_prob["vc"], sub_prob["rel_logits"],
        sub_prob["gate"], sub_prob["sinks"], sub_out,
        block_table=sub_prob["bt"], cache_seqlens=sub_prob["seq_lens"], cu_seqlens_q=sub_prob["cu_q"],
        max_query_len=135, softmax_scale=scale, window=0,
    )
    sub_err = (sub_out.float() - sub_ref).abs().max().item()
    assert sub_err <= MAX_ERR_TOL, f"subnormals err={sub_err:.3e}"
    print(f"  [PASS] e4m3_subnormals_exact max_err={sub_err:.3e}", flush=True)

    # Empty / zero edge cases: num_tokens=0, num_actual_tokens=0, q_len=0 request mixed in batch
    empty_q = torch.empty((0, 6, 128), dtype=torch.float16, device=DEVICE)
    empty_kv = torch.zeros((1, 128, 2, 128), dtype=torch.float8_e4m3fn, device=DEVICE)
    empty_rel = torch.zeros((128, 6, 1280), dtype=torch.float16, device=DEVICE)
    empty_gate = torch.empty((0, 6 * 128), dtype=torch.float16, device=DEVICE)
    empty_sinks = torch.empty((0, 6), dtype=torch.float16, device=DEVICE)
    empty_out = torch.empty((0, 6 * 128), dtype=torch.float16, device=DEVICE)
    rel_prefill_attention_cuda(
        empty_q, empty_kv, empty_kv, empty_rel, empty_gate, empty_sinks, empty_out,
        block_table=torch.zeros((0, 1), dtype=torch.int32, device=DEVICE),
        cache_seqlens=torch.zeros((0,), dtype=torch.int32, device=DEVICE),
        cu_seqlens_q=torch.zeros((1,), dtype=torch.int32, device=DEVICE),
        max_query_len=0, softmax_scale=scale, window=1024,
    )

    p_zero = _make_problem([(10, 32)], h_kv=2, g=3, page=128, kv_dtype=torch.float8_e4m3fn, window=1024, seed=901)
    out_zero = torch.full((32, 6 * 128), 7.0, dtype=torch.float16, device=DEVICE)
    rel_prefill_attention_cuda(
        p_zero["q"], p_zero["kc"], p_zero["vc"], p_zero["rel_logits"], p_zero["gate"], p_zero["sinks"], out_zero,
        block_table=p_zero["bt"], cache_seqlens=p_zero["seq_lens"], cu_seqlens_q=p_zero["cu_q"],
        max_query_len=32, softmax_scale=scale, window=1024, num_actual_tokens=0,
    )
    assert torch.equal(out_zero, torch.zeros_like(out_zero)), "num_actual_tokens=0 did not zero out!"

    p_mix0 = _make_problem([(0, 0), (50, 45), (0, 0), (10, 19)], h_kv=2, g=3, page=64, kv_dtype=torch.float8_e4m3fn, window=0, seed=902)
    out_mix0 = torch.full((p_mix0["total_t"], 6 * 128), float("nan"), dtype=torch.float16, device=DEVICE)
    rel_prefill_attention_cuda(
        p_mix0["q"], p_mix0["kc"], p_mix0["vc"], p_mix0["rel_logits"], p_mix0["gate"], p_mix0["sinks"], out_mix0,
        block_table=p_mix0["bt"], cache_seqlens=p_mix0["seq_lens"], cu_seqlens_q=p_mix0["cu_q"],
        max_query_len=p_mix0["max_q"], softmax_scale=scale, window=0,
    )
    ref_mix0 = _compute_ref(p_mix0, scale)
    err_mix0 = (out_mix0.float() - ref_mix0).abs().max().item()
    assert not torch.isnan(out_mix0).any() and err_mix0 <= MAX_ERR_TOL, f"mixed q_len=0 err={err_mix0:.3e}"
    print(f"  [PASS] empty/zero edge cases (T=0, act=0, q_len=0 in batch max_err={err_mix0:.3e})", flush=True)


def run_cudagraph_tests() -> None:
    print("\n=== [2] CUDA Graph Capture & Replay Tests ===", flush=True)
    scale = 128**-0.5
    for kv_dtype in (torch.float8_e4m3fn, torch.float16):
        dt_tag = "fp8" if kv_dtype == torch.float8_e4m3fn else "fp16"
        for window in (1024, 0):
            prob1 = _make_problem(
                [(300, 140), (50, 70)], h_kv=2, g=3, page=128, kv_dtype=kv_dtype,
                window=window, stride_gate=2152, stride_sink=16, pad_tokens=18, seed=601,
            )
            # Add a poisoned padded request with block_table=-1
            cu_q = torch.cat([prob1["cu_q"], torch.tensor([prob1["total_t"]], dtype=torch.int32, device=DEVICE)])
            seq_lens = torch.cat([prob1["seq_lens"], torch.tensor([99999], dtype=torch.int32, device=DEVICE)])
            bt = torch.cat([prob1["bt"], torch.full((1, prob1["bt"].shape[1]), -1, dtype=torch.int32, device=DEVICE)], dim=0)
            out = torch.full((prob1["total_t"], 6 * 128), 99.0, dtype=torch.float16, device=DEVICE)

            # Warmup outside graph
            rel_prefill_attention_cuda(
                prob1["q"], prob1["kc"], prob1["vc"], prob1["rel_logits"], prob1["gate"], prob1["sinks"], out,
                block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                max_query_len=140, softmax_scale=scale, window=window,
                num_actual_tokens=prob1["num_actual"],
            )
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mem_before = torch.cuda.memory_allocated(DEVICE)
                rel_prefill_attention_cuda(
                    prob1["q"], prob1["kc"], prob1["vc"], prob1["rel_logits"], prob1["gate"], prob1["sinks"], out,
                    block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                    max_query_len=140, softmax_scale=scale, window=window,
                    num_actual_tokens=prob1["num_actual"],
                )
                mem_after = torch.cuda.memory_allocated(DEVICE)
            assert mem_after == mem_before, f"Allocated {mem_after - mem_before} B inside CUDA graph!"

            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            ref1 = _compute_ref(prob1, scale)
            err1 = (out.float() - ref1).abs().max().item()
            assert not torch.isnan(out).any() and err1 <= MAX_ERR_TOL, f"CG replay 1: err={err1:.3e}"
            assert torch.equal(out[prob1["num_actual"] :], torch.zeros_like(out[prob1["num_actual"] :]))

            # Mutate inputs in-place and replay
            prob2 = _make_problem(
                [(300, 140), (50, 70)], h_kv=2, g=3, page=128, kv_dtype=kv_dtype,
                window=window, stride_gate=2152, stride_sink=16, pad_tokens=18, seed=999,
            )
            prob1["q"].copy_(prob2["q"])
            prob1["kc"].copy_(prob2["kc"])
            prob1["vc"].copy_(prob2["vc"])
            prob1["rel_logits"].copy_(prob2["rel_logits"])
            prob1["gate"].copy_(prob2["gate"])
            prob1["sinks"].copy_(prob2["sinks"])
            bt[:2].copy_(prob2["bt"])
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            ref2 = _compute_ref(prob2, scale)
            err2 = (out.float() - ref2).abs().max().item()
            assert not torch.isnan(out).any() and err2 <= MAX_ERR_TOL, f"CG replay 2: err={err2:.3e}"
            assert torch.equal(out[prob1["num_actual"] :], torch.zeros_like(out[prob1["num_actual"] :]))
            print(
                f"  [PASS] CUDAGraph {dt_tag:4s} window={window:<4d}: replay1={err1:.3e}, replay2={err2:.3e}, alloc=0B",
                flush=True,
            )


def _bench_us(fn, warmup_s: float = 0.5, iters: int = 100) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_benchmarks() -> None:
    print("\n=== [3] Warm-Clock (>=1s warmup) Benchmark: attn_rel_pf.cu vs Triton (rel_attention + gate_sink) ===", flush=True)
    scale = 128**-0.5
    h_kv, g, page, extent = 2, 3, 128, 1024
    h_q = h_kv * g

    # Warm up V100 boost clocks thoroughly (>= 1.5s)
    dummy = torch.randn(2048, 2048, dtype=torch.float16, device=DEVICE)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        _ = dummy @ dummy
    torch.cuda.synchronize()
    del dummy

    shapes = [
        ("q128", 0, 128),
        ("q512", 0, 512),
        ("q1024", 0, 1024),
        ("q3692 (4K)", 0, 3692),
        ("p3692+q3692 (8K-c2)", 3692, 3692),
    ]

    print(
        f"{'Shape':20s} {'Layer':8s} | {'Triton(us)':>11s} | {'CUDA(us)':>10s} | {'Speedup':>9s} | "
        f"{'TFLOPs':>7s} | {'MaxDiff':>8s} | {'cfg1/2/3/4/5/6(us)'}",
        flush=True,
    )
    print("-" * 124, flush=True)

    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for label, past, q_len in shapes:
        summary[label] = {}
        seq_len = past + q_len
        num_blks = (seq_len + page - 1) // page
        kv_buf = torch.zeros((num_blks, 2, page, h_kv, 128), dtype=torch.float8_e4m3fn, device=DEVICE)
        kv_f = torch.randn(kv_buf.shape, dtype=torch.float16, device=DEVICE) * 0.5
        kv_buf.copy_(kv_f.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn))
        del kv_f
        kc, vc = kv_buf[:, 0], kv_buf[:, 1]
        bt = torch.arange(num_blks, dtype=torch.int32, device=DEVICE).view(1, num_blks).contiguous()
        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        q = torch.randn(q_len, h_q, 128, dtype=torch.float16, device=DEVICE) * 0.25
        gate_backing = torch.randn(q_len, 2152, dtype=torch.float16, device=DEVICE)
        gate = gate_backing[:, : h_q * 128]
        sinks = torch.randn(q_len, h_q, dtype=torch.float16, device=DEVICE)
        rel_logits = torch.randn(q_len + 128, h_q, extent + 256, dtype=torch.float16, device=DEVICE) * 0.1
        out_tri = torch.empty((q_len, h_q * 128), dtype=torch.float16, device=DEVICE)
        out_cu = torch.empty((q_len, h_q * 128), dtype=torch.float16, device=DEVICE)

        for layer_name, window in [("SWA-1024", 1024), ("Global", 0)]:
            def run_tri():
                _run_triton_prefill(
                    q, kc, vc, rel_logits, gate, sinks, out_tri,
                    block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                    max_query_len=q_len, softmax_scale=scale, window=window,
                )

            def run_cu(cfg: int = 0):
                rel_prefill_attention_cuda(
                    q, kc, vc, rel_logits, gate, sinks, out_cu,
                    block_table=bt, cache_seqlens=seq_lens, cu_seqlens_q=cu_q,
                    max_query_len=q_len, softmax_scale=scale, window=window,
                    config_override=cfg,
                )

            run_tri()
            run_cu(0)
            torch.cuda.synchronize()
            max_diff = (out_cu.float() - out_tri.float()).abs().max().item()
            assert max_diff <= MAX_ERR_TOL, f"Bench parity fail {label} {layer_name}: {max_diff:.3e}"

            tri_iters = 10 if q_len >= 3692 else (30 if q_len >= 1024 else 100)
            cu_iters = 100 if q_len >= 3692 else 200
            us_tri = _bench_us(run_tri, warmup_s=0.3, iters=tri_iters)
            us_cu = _bench_us(lambda: run_cu(0), warmup_s=0.35, iters=cu_iters)

            # Profile all 6 configs for visibility
            cfg_us = []
            for c in (1, 2, 3, 4, 5, 6):
                cfg_us.append(_bench_us(lambda c=c: run_cu(c), warmup_s=0.1, iters=max(30, cu_iters // 2)))

            # Compute effective causal/SWA FLOPs
            pos_arr = torch.arange(past, past + q_len, dtype=torch.int64)
            vis_k = torch.minimum(pos_arr + 1, torch.full_like(pos_arr, window)) if window > 0 else (pos_arr + 1)
            pairs = int(vis_k.sum().item()) * h_q
            flops = 4.0 * 128.0 * pairs
            tflops = flops / (us_cu * 1e-6) / 1e12
            speedup = us_tri / us_cu
            summary[label][layer_name] = (us_tri, us_cu)

            cfg_str = "/".join(f"{x:.0f}" for x in cfg_us)
            print(
                f"{label:20s} {layer_name:8s} | {us_tri:11.1f} | {us_cu:10.1f} | {speedup:8.1f}x | "
                f"{tflops:7.2f} | {max_diff:8.2e} | {cfg_str}",
                flush=True,
            )

        del kv_buf, kc, vc, bt, q, gate_backing, gate, sinks, rel_logits, out_tri, out_cu
        torch.cuda.empty_cache()

    print("\n--- 48-Visit Layer Totals (40 SWA-1024 + 8 Global) ---", flush=True)
    for label, _, _ in shapes:
        tri_swa, cu_swa = summary[label]["SWA-1024"]
        tri_glb, cu_glb = summary[label]["Global"]
        tot_tri_ms = (40 * tri_swa + 8 * tri_glb) / 1000.0
        tot_cu_ms = (40 * cu_swa + 8 * cu_glb) / 1000.0
        print(
            f"  {label:20s}: Triton = {tot_tri_ms:8.2f} ms  |  CUDA = {tot_cu_ms:6.2f} ms  |  "
            f"Speedup = {tot_tri_ms / tot_cu_ms:6.1f}x",
            flush=True,
        )


if __name__ == "__main__":
    audit_ptxas_zero_spills()
    run_accuracy_tests()
    run_cudagraph_tests()
    run_benchmarks()
    print("\nALL PREFILL ATTENTION TESTS AND BENCHMARKS PASSED!", flush=True)
