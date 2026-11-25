#!/usr/bin/env python3
"""48-visit real-weights FP32 oracle evaluation of lossy optimizations on Inkling-2-Mini.

Evaluates on 1x V100-16GB from a read-only Inkling-2-Mini checkpoint:
  0. Baseline: Exact FP16 + FP8 KV (current Phase 3 quality baseline, reference = FP32 oracle)
  1. ELR Last-Loop KV Caching (arXiv:2610.12582v1 Sec 3.4/5.3):
     - All 12 core layers (L6-17): 2.0x total attention KV compression, 0 extra FLOPs
     - Global core layers only (L11, L17): 2.0x global KV compression, SWA exact
  2. LoopCD-Hidden (arXiv:2610.02185v1 Sec 3.1):
     - Full depth R=3 with omega in {0.15, 0.30}
     - Reduced depth R=2 (-25% target compute) with omega in {0.0, 0.25, 0.50, 0.75, 1.0}
  3. Post-Prologue TP4-Native GLA-g4 on the 8 global logical visits (L5, L11x3, L17x3, L23):
     - Fit Q/O-weighted & plain SVD on STRICTLY DISJOINT calibration texts (~1,000 tokens)
     - Evaluate r=256 (2.0x global KV, G: 3->6) and r=192 (2.67x global KV) with FP8 latent KV
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from eval_fp8_oracle import (
    DEVICE,
    MODEL_DIR,
    PROMPTS,
    WeightStore,
    attn_res_f32,
    dequant_mxfp8_expert,
    dequant_nvfp4_expert,
    dynamic_sconv_f32,
    l2_clip_f32,
    rms_norm_f32,
    static_sconv_f32,
)

# Strictly disjoint calibration corpus for post-prologue GLA-g4 SVD (zero overlap with PROMPTS)
CALIB_TEXTS = [
    (
        "Implement a lock-free single-producer single-consumer ring buffer in C++17 using "
        "std::atomic acquire-release memory ordering, cache-line padding to prevent false sharing, "
        "and power-of-two capacity masking:\n\n```cpp\n#include <atomic>\n#include <cstddef>\n"
        "#include <array>\n\ntemplate <typename T, std::size_t N>\nclass SpscRing {\n"
        "  static_assert((N & (N - 1)) == 0, \"N must be power of two\");\n"
        "  alignas(64) std::atomic<std::size_t> head_{0};\n"
        "  alignas(64) std::atomic<std::size_t> tail_{0};\n"
        "  std::array<T, N> buf_;\npublic:\n  bool push(const T& v) {\n"
        "    auto h = head_.load(std::memory_order_relaxed);\n"
        "    auto next = (h + 1) & (N - 1);\n"
        "    if (next == tail_.load(std::memory_order_acquire)) return false;\n"
        "    buf_[h] = v;\n    head_.store(next, std::memory_order_release);\n"
        "    return true;\n  }\n};\n```\n"
    ),
    (
        "Let A be a real symmetric positive definite n x n matrix and consider the Krylov "
        "subspace K_m(A, r_0) = span{r_0, A r_0, ..., A^{m-1} r_0}. By the optimality property "
        "of conjugate gradients, the m-th iterate x_m minimizes the A-norm of the error "
        "||x_m - x*||_A over x_0 + K_m(A, r_0). Expressing e_m = p_m(A) e_0 where p_m is a "
        "polynomial of degree at most m with p_m(0) = 1, we apply Chebyshev polynomials shifted "
        "to [lambda_min, lambda_max] to obtain the classic condition-number convergence rate "
        "2 * ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^m.\n"
    ),
    (
        "{\"service\": \"telemetry-ingest\", \"version\": \"2.4.1\", \"storage\": {\"engine\": "
        "\"clickhouse\", \"retention_days\": 90, \"compression\": \"zstd\", \"partitions\": "
        "[\"region\", \"day\"]}, \"limits\": {\"max_batch_bytes\": 4194304, \"rate_per_sec\": "
        "25000, \"retry_backoff_ms\": [100, 250, 1000, 5000]}, \"alerts\": [{\"name\": "
        "\"high_lag\", \"threshold_s\": 30, \"severity\": \"critical\"}]}\n"
    ),
    (
        "In deep-sea hydrothermal vent ecosystems, chemosynthetic bacteria oxidize hydrogen "
        "sulfide and methane emanating from basaltic fissures along mid-ocean ridges to fix "
        "inorganic carbon without sunlight. Giant tube worms such as Riftia pachyptila lack "
        "digestive tracts entirely, housing dense endosymbiont colonies inside a specialized "
        "trophosome supplied with dissolved gases via extracellular hemoglobins capable of "
        "reversibly binding both oxygen and sulfide simultaneously.\n"
    ),
    (
        "def dijkstra(graph, source):\n    import heapq\n    dist = {source: 0.0}\n"
        "    pq = [(0.0, source)]\n    while pq:\n        d, u = heapq.heappop(pq)\n"
        "        if d > dist.get(u, float('inf')):\n            continue\n"
        "        for v, w in graph.get(u, []):\n            nd = d + w\n"
        "            if nd < dist.get(v, float('inf')):\n                dist[v] = nd\n"
        "                heapq.heappush(pq, (nd, v))\n    return dist\n"
    ),
    (
        "Historical monetary systems transitioned from bimetallic commodity standards to "
        "fiat regimes governed by central bank balance-sheet operations. When short-term "
        "nominal interest rates reach the effective lower bound, policymakers rely on forward "
        "guidance and large-scale asset purchases to compress term premia across longer "
        "maturities and anchor inflation expectations near target.\n"
    ),
]


def eval_metrics(logits_ref: list[torch.Tensor], logits_test: list[torch.Tensor], ids_list: list[torch.Tensor]) -> dict:
    cos_vals, kl_vals, dnll_vals = [], [], []
    top1_match = 0
    total_pos = 0
    for lr, lt, ids in zip(logits_ref, logits_test, ids_list):
        # Compare all next-token positions [0 .. T-2] against ground truth ids[1:]
        r = lr[:-1].float()
        t = lt[:-1].float()
        tgt = ids[1:].long().to(r.device)
        cos_vals.append(F.cosine_similarity(r, t, dim=-1))
        log_pr = F.log_softmax(r, dim=-1)
        log_pt = F.log_softmax(t, dim=-1)
        pr = log_pr.exp()
        kl_vals.append((pr * (log_pr - log_pt)).sum(dim=-1))
        nll_r = -log_pr.gather(1, tgt.unsqueeze(1)).squeeze(1)
        nll_t = -log_pt.gather(1, tgt.unsqueeze(1)).squeeze(1)
        dnll_vals.append(nll_t - nll_r)
        top1_match += int((r.argmax(dim=-1) == t.argmax(dim=-1)).sum().item())
        total_pos += int(r.shape[0])
    cos_all = torch.cat(cos_vals)
    kl_all = torch.cat(kl_vals)
    dnll_all = torch.cat(dnll_vals)
    return {
        "cos": round(float(cos_all.mean().item()), 6),
        "kl_mean": round(float(kl_all.mean().item()), 5),
        "kl_p95": round(float(torch.quantile(kl_all, 0.95).item()), 5),
        "dnll": round(float(dnll_all.mean().item()), 5),
        "top1_pct": round(100.0 * top1_match / max(total_pos, 1), 2),
        "n_pos": total_pos,
    }


@torch.no_grad()
def run_inkling_variant(
    input_ids_list: list[torch.Tensor],
    cfg: dict,
    ws: WeightStore,
    *,
    fp8_kv: bool = True,
    fp16_compute: bool = True,
    num_passes_override: int | None = None,
    loopcd_omega: float = 0.0,
    last_loop_kv_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] | None = None,
    last_loop_kv_layers: set[int] | None = None,
    record_pass2_kv: bool = False,
    record_global_gla_stats: dict | None = None,
    gla_projs: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] | None = None,
    gla_fp8_latent: bool = True,
) -> tuple[list[torch.Tensor], dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]]:
    """Flexible 48-visit Inkling-2-Mini forward supporting ELR Last-Loop KV, LoopCD-Hidden, and GLA-g4."""
    H = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    local_ids = set(cfg["local_layer_ids"])
    core_start = cfg["loop_config"]["core_start"]
    core_end = cfg["loop_config"]["core_end"]
    num_passes = num_passes_override or cfg["loop_config"]["num_passes"]
    block_size = cfg["n_layers_per_attn_res_block"]
    eps = cfg["rms_norm_eps"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    d_rel = cfg["d_rel"]
    rel_extent = cfg["rel_extent"]
    sw = cfg["sliding_window_size"]
    qk_clip = cfg["qk_l2_clip_threshold"]
    rel_clip = cfg["rel_l2_clip_threshold"]
    gate_cap = cfg["mlp_gate_softcap"]
    up_cap = cfg["mlp_up_softcap"]
    route_scale = cfg["route_scale"]
    topk = cfg["num_experts_per_tok"]
    softcap = cfg["final_logit_softcapping"]
    unpadded_vocab = cfg["unpadded_vocab_size"]

    inv_sqrt_h = H ** -0.5
    attn_queries = [
        ws.get(f"model.llm.attn_res.attn_queries.{i}", torch.float32).view(-1, H) * inv_sqrt_h
        for i in range(num_layers - 1)
    ]
    mlp_queries = [
        ws.get(f"model.llm.attn_res.mlp_queries.{i}", torch.float32).view(1, H) * inv_sqrt_h
        for i in range(num_layers)
    ]
    output_query = ws.get("model.llm.attn_res.output_query", torch.float32).view(1, H) * inv_sqrt_h
    q_cp = ws.get("model.llm.middle_cycle_cross_pass_queries", torch.float32).view(-1, 2, H)
    kv_cp = ws.get("model.llm.middle_cycle_cross_pass_kv_queries", torch.float32).view(-1, 2, H)
    msg_queries = torch.stack((q_cp[:, 0], kv_cp[:, 0], kv_cp[:, 1], q_cp[:, 1]), dim=1) * inv_sqrt_h

    embed_w = ws.get("model.llm.embed.weight")
    embed_norm_w = ws.get("model.llm.embed_norm.weight")
    final_norm_w = ws.get("model.llm.norm.weight")

    states = []
    for p_idx, ids in enumerate(input_ids_list):
        T = ids.numel()
        pos = torch.arange(T, device=DEVICE, dtype=torch.float32)
        tau = 1.0 + cfg["log_scaling_alpha"] * torch.log(((pos + 1.0) / cfg["log_scaling_n_floor"]).clamp_min(1.0))
        emb = F.embedding(ids, embed_w)
        s0 = rms_norm_f32(emb, embed_norm_w, eps)
        stack = torch.zeros((6, T, H), dtype=torch.float32, device=DEVICE)
        stack[0].copy_(s0)
        states.append({
            "p_idx": p_idx,
            "T": T,
            "tau": tau,
            "stack": stack,
            "num_blocks": 0,
            "inputs": None,
            "bank": None,
            "bank_pass0": None,
            "prelude_blocks": None,
            "prelude_partial": None,
            "prelude_dests": None,
        })
    del embed_w

    visits = [(i, 0) for i in range(core_start)]
    visits += [(i, p) for p in range(num_passes) for i in range(core_start, core_end)]
    visits += [(i, 0) for i in range(core_end, num_layers)]

    recorded_pass2: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if fp16_compute:
            return torch.mm(x.half(), w.half().t(), out_dtype=torch.float32)
        return torch.mm(x.float(), w.float().t())

    for visit, (i, pass_idx) in enumerate(visits):
        is_local = i in local_ids
        looped = pass_idx > 0
        pfx = f"model.llm.layers.{i}"

        if is_local:
            an_w = ws.get(f"{pfx}.attn_norm.weight").view(1, H)
        else:
            an_w = torch.stack([
                ws.get(f"{pfx}.q_input_norm.weight"),
                ws.get(f"{pfx}.k_input_norm.weight"),
                ws.get(f"{pfx}.v_input_norm.weight"),
            ], dim=0)
        mn_w = ws.get(f"{pfx}.mlp_norm.weight").view(1, H)

        wq = ws.get(f"{pfx}.attn.wq_du.weight")
        wr = ws.get(f"{pfx}.attn.wr_du.weight")
        wg = ws.get(f"{pfx}.attn.gate_attn.weight")
        wsink = ws.get(f"{pfx}.attn.attn_sink_proj.weight")
        wk = ws.get(f"{pfx}.attn.wk_dv.weight")
        wv = ws.get(f"{pfx}.attn.wv_dv.weight")
        wo = ws.get(f"{pfx}.attn.wo_ud.weight")
        q_scale = ws.get(f"{pfx}.attn.q_l2_clip.scale").view(n_heads, hd)
        r_scale = ws.get(f"{pfx}.attn.rel_l2_clip.scale").view(n_heads, d_rel)
        rel_proj = ws.get(f"{pfx}.attn.rel_logits_proj.proj")
        if is_local:
            k_sw = ws.get(f"{pfx}.attn.k_sconv.weight")
            v_sw = ws.get(f"{pfx}.attn.v_sconv.weight")
            k_wp = v_wp = None
        else:
            k_sw = ws.get(f"{pfx}.attn.k_sconv.static_w")
            v_sw = ws.get(f"{pfx}.attn.v_sconv.static_w")
            k_wp = ws.get(f"{pfx}.attn.k_sconv.weight_proj")
            v_wp = ws.get(f"{pfx}.attn.v_sconv.weight_proj")
        attn_sw = ws.get(f"{pfx}.attn_sconv.weight")
        mlp_sw = ws.get(f"{pfx}.mlp_sconv.weight")

        if looped:
            c_down = torch.stack([ws.get(f"{pfx}.controllers.{s}.down_weight") for s in ("attn", "k", "v", "mlp")], 0)
            c_gain = torch.stack([ws.get(f"{pfx}.controllers.{s}.gain_weight") for s in ("attn", "k", "v", "mlp")], 0)
            c_fgt = torch.stack([ws.get(f"{pfx}.controllers.{s}.forget_weight") for s in ("attn", "k", "v", "mlp")], 0)
        else:
            c_down = c_gain = c_fgt = None

        gate_w = ws.get(f"{pfx}.mlp.gate.weight")
        gate_b = ws.get(f"{pfx}.mlp.gate.bias", torch.float32)

        is_nvfp4 = i < 20
        sf = ws.sf
        w13_slice = sf.get_slice(f"{pfx}.mlp.experts.w13_weight")
        s13_slice = sf.get_slice(f"{pfx}.mlp.experts.w13_weight_scale")
        w2_slice = sf.get_slice(f"{pfx}.mlp.experts.w2_weight")
        s2_slice = sf.get_slice(f"{pfx}.mlp.experts.w2_weight_scale")
        if is_nvfp4:
            g13_all = (sf.get_tensor(f"{pfx}.mlp.experts.w13_weight_amax").float()
                       / sf.get_tensor(f"{pfx}.mlp.experts.w13_weight_outer_scale_denominator").float())
            g2_all = (sf.get_tensor(f"{pfx}.mlp.experts.w2_weight_amax").float()
                      / sf.get_tensor(f"{pfx}.mlp.experts.w2_weight_outer_scale_denominator").float())

        if visit + 1 == len(visits):
            next_q = output_query
            next_nw = final_norm_w.view(1, H)
        else:
            ni, npass = visits[visit + 1]
            next_q = attn_queries[ni - 1]
            if npass:
                next_nw = None
            else:
                if ni in local_ids:
                    next_nw = ws.get(f"model.llm.layers.{ni}.attn_norm.weight").view(1, H)
                else:
                    next_nw = torch.stack([
                        ws.get(f"model.llm.layers.{ni}.q_input_norm.weight"),
                        ws.get(f"model.llm.layers.{ni}.k_input_norm.weight"),
                        ws.get(f"model.llm.layers.{ni}.v_input_norm.weight"),
                    ], dim=0)

        # Record W_O metric for global GLA-g4 Q/O-weighted SVD
        if record_global_gla_stats is not None and not is_local:
            key_v = (i, pass_idx)
            if key_v not in record_global_gla_stats:
                # wo: [H, n_heads * hd] -> reshape [H, 4, 6, hd] -> per TP4 group g in 0..3,
                # two KV heads: h0 gets Q heads 0..2, h1 gets Q heads 3..5
                wo_f = wo.float().view(H, 4, 2, 3, hd)
                mv_list = []
                for g in range(4):
                    m0 = torch.einsum("rqd,rqe->de", wo_f[:, g, 0], wo_f[:, g, 0]).double().cpu()
                    m1 = torch.einsum("rqd,rqe->de", wo_f[:, g, 1], wo_f[:, g, 1]).double().cpu()
                    mv_list.append((m0, m1))
                record_global_gla_stats[key_v] = {
                    "kv_list": [[] for _ in range(4)],
                    "cq_sum": [[torch.zeros(hd, hd, dtype=torch.float64), torch.zeros(hd, hd, dtype=torch.float64)] for _ in range(4)],
                    "mv": mv_list,
                }

        pre_mlp = []
        active_experts: set[int] = set()
        for st in states:
            T = st["T"]
            stack = st["stack"]
            nb = st["num_blocks"]
            inputs = st["inputs"]
            bank = st["bank"]

            if i == core_start and not looped:
                st["prelude_blocks"] = nb
                st["prelude_partial"] = stack[nb].clone()
            elif i == core_start:
                bank = stack[st["prelude_blocks"] + 1 : nb + 1].clone()
                if pass_idx == 1:
                    st["bank_pass0"] = bank.clone()
                st["bank"] = bank
                nb = st["prelude_blocks"]
                st["num_blocks"] = nb
                stack[nb].copy_(st["prelude_partial"])
                if st["prelude_dests"] is None:
                    st["prelude_dests"] = attn_res_f32(
                        stack[:nb], st["prelude_partial"], None, attn_queries[i - 1], None, 0.0, True
                    )
                inputs = st["prelude_dests"]
            elif i == core_end and loopcd_omega != 0.0 and st["bank_pass0"] is not None:
                # LoopCD-Hidden: extrapolate core output blocks away from Pass-0 bank right before coda (L18-23)!
                pb = st["prelude_blocks"]
                b_final = stack[pb + 1 : nb + 1]
                b_ref = st["bank_pass0"]
                stack[pb + 1 : nb + 1].copy_(b_final + loopcd_omega * (b_final - b_ref))
                # Recompute entry inputs to L18 from the extrapolated stack!
                inputs = attn_res_f32(
                    stack[:nb], stack[nb], None, attn_queries[i - 1], an_w, eps, True
                )

            partial = stack[nb]
            messages = None
            if looped:
                assert bank is not None and inputs is not None
                messages = attn_res_f32(bank[:1], bank[1], None, msg_queries[i - core_start], None, 0.0, True)
                dests = inputs.expand(3, -1, -1) if is_local else inputs
                nw = an_w.expand(3, -1) if is_local else an_w
                attn_in = []
                for s_idx in range(3):
                    d_norm = rms_norm_f32(dests[s_idx], None, 1e-6)
                    m_norm = rms_norm_f32(messages[s_idx], None, 1e-6)
                    pair = torch.cat([d_norm, m_norm], dim=-1)
                    hid = lin(pair, c_down[s_idx])
                    if fp16_compute:
                        hid = hid.half()
                    gain = torch.tanh(lin(hid, c_gain[s_idx]))
                    fgt = torch.tanh(lin(hid, c_fgt[s_idx]))
                    blended = (1.0 - fgt) * dests[s_idx] + gain * messages[s_idx]
                    attn_in.append(rms_norm_f32(blended, nw[s_idx], eps))
                x_q, x_k, x_v = attn_in[0], attn_in[1], attn_in[2]
            elif inputs is None:
                x_q = rms_norm_f32(partial, an_w[0], eps)
                x_k = x_v = x_q
            else:
                x_q = inputs[0]
                x_k = inputs[0] if is_local else inputs[1]
                x_v = inputs[0] if is_local else inputs[2]

            if fp16_compute:
                x_q, x_k, x_v = x_q.half(), x_k.half(), x_v.half()

            q = lin(x_q, wq).view(T, n_heads, hd)
            r = lin(x_q, wr).view(T, n_heads, d_rel)
            gate_a = lin(x_q, wg)
            sinks = lin(x_q, wsink)
            k = lin(x_k, wk)
            v = lin(x_v, wv)

            if is_local:
                k = static_sconv_f32(k, k_sw)
                v = static_sconv_f32(v, v_sw)
            else:
                dk = lin(x_k, k_wp)
                dv = lin(x_v, v_wp)
                k = dynamic_sconv_f32(k, k_sw, dk, hd)
                v = dynamic_sconv_f32(v, v_sw, dv, hd)

            q_c = l2_clip_f32(q * q_scale.float().unsqueeze(0), qk_clip, eps)
            r_c = l2_clip_f32(r * r_scale.float().unsqueeze(0), rel_clip, eps)
            if not is_local:
                if fp16_compute:
                    q_c = q_c.half().float()
                    r_c = r_c.half().float()
                q_c = q_c * st["tau"][:, None, None]
                r_c = r_c * st["tau"][:, None, None]
            k_c = l2_clip_f32(k.view(T, n_kv, hd), qk_clip, eps)
            v_c = rms_norm_f32(v.view(T, n_kv, hd), None, eps)

            # Collect calibration tensors for post-prologue GLA-g4 SVD (before FP8 KV)
            if record_global_gla_stats is not None and not is_local:
                rec = record_global_gla_stats[(i, pass_idx)]
                q_g = q_c.view(T, 4, 2, 3, hd)
                for g in range(4):
                    kv_g = torch.cat([k_c[:, 2 * g], k_c[:, 2 * g + 1], v_c[:, 2 * g], v_c[:, 2 * g + 1]], dim=-1)
                    rec["kv_list"][g].append(kv_g.double().cpu())
                    for sub in range(2):
                        qs = q_g[:, g, sub].reshape(-1, hd).double()
                        rec["cq_sum"][g][sub] += (qs.t() @ qs).cpu()

            # Apply post-prologue GLA-g4 projection (exact mathematical equivalent of absorbed GLA!)
            if gla_projs is not None and not is_local and (i, pass_idx) in gla_projs:
                projs_g = gla_projs[(i, pass_idx)]
                k_new = torch.empty_like(k_c)
                v_new = torch.empty_like(v_c)
                for g in range(4):
                    w_down, w_up, lat_scale = projs_g[g]
                    kv_g = torch.cat([k_c[:, 2 * g], k_c[:, 2 * g + 1], v_c[:, 2 * g], v_c[:, 2 * g + 1]], dim=-1)
                    lat = kv_g @ w_down  # [T, r]
                    if gla_fp8_latent:
                        # Per-group static scale so FP8 E4M3FN uses full dynamic range without saturation
                        lat_q = (lat / lat_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * lat_scale
                    else:
                        lat_q = lat.half().float() if fp16_compute else lat
                    rec_kv = lat_q @ w_up  # [T, 512]
                    k_new[:, 2 * g] = rec_kv[:, 0:hd]
                    k_new[:, 2 * g + 1] = rec_kv[:, hd : 2 * hd]
                    v_new[:, 2 * g] = rec_kv[:, 2 * hd : 3 * hd]
                    v_new[:, 2 * g + 1] = rec_kv[:, 3 * hd : 4 * hd]
                k_c, v_c = k_new, v_new
            elif fp8_kv:
                k_c = k_c.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
                v_c = v_c.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
            elif fp16_compute:
                k_c = k_c.half().float()
                v_c = v_c.half().float()

            if fp16_compute:
                q_c = q_c.half().float()
                r_c = r_c.half().float()

            # Record final-pass K/V for ELR Last-Loop KV evaluation
            if record_pass2_kv and core_start <= i < core_end and pass_idx == num_passes - 1:
                recorded_pass2[(i, st["p_idx"])] = (k_c.detach().cpu(), v_c.detach().cpu())

            rep = n_heads // n_kv
            # Check if ELR Last-Loop KV applies to this core layer
            use_llkv = (
                last_loop_kv_cache is not None
                and core_start <= i < core_end
                and (last_loop_kv_layers is None or i in last_loop_kv_layers)
                and (i, st["p_idx"]) in last_loop_kv_cache
            )
            if use_llkv:
                k_p2, v_p2 = last_loop_kv_cache[(i, st["p_idx"])]
                k_past = k_p2.to(DEVICE).repeat_interleave(rep, dim=1)
                v_past = v_p2.to(DEVICE).repeat_interleave(rep, dim=1)
                k_cur = k_c.repeat_interleave(rep, dim=1)
                v_cur = v_c.repeat_interleave(rep, dim=1)
                # Strict causal Last-Loop KV: past tokens s < t use Pass-2 K/V; self s == t uses current Pass-p K/V!
                scores_past = torch.einsum("thd,shd->hts", q_c, k_past) * (hd ** -0.5)
                scores_self = (q_c * k_cur).sum(dim=-1).t() * (hd ** -0.5)  # [H, T]
                idx = torch.arange(T, device=DEVICE)
                scores = scores_past.clone()
                scores[:, idx, idx] = scores_self
            else:
                k_exp = k_c.repeat_interleave(rep, dim=1)
                v_exp = v_c.repeat_interleave(rep, dim=1)
                scores = torch.einsum("thd,shd->hts", q_c, k_exp) * (hd ** -0.5)
                idx = torch.arange(T, device=DEVICE)

            rel_all = torch.einsum("thd,de->the", r_c, rel_proj.float())
            dist = idx[:, None] - idx[None, :]
            in_rel = (dist >= 0) & (dist < rel_extent)
            safe_dist = dist.clamp(0, rel_extent - 1)
            rel_b = rel_all.permute(1, 0, 2).gather(
                2, safe_dist.unsqueeze(0).expand(n_heads, T, T)
            )
            rel_b = torch.where(in_rel.unsqueeze(0), rel_b, torch.zeros_like(rel_b))
            scores = scores + rel_b
            causal_mask = dist >= 0
            if is_local:
                causal_mask = causal_mask & (dist < sw)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            lse = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            if use_llkv:
                # Strictly: sum_{s < t} probs[h,t,s] * v_past[s,h] + probs[h,t,t] * v_cur[t,h]
                p_self = probs[:, idx, idx]  # [H, T]
                p_past = probs.clone()
                p_past[:, idx, idx] = 0.0
                o = torch.einsum("hts,shd->thd", p_past, v_past) + p_self.t().unsqueeze(-1) * v_cur
            else:
                o = torch.einsum("hts,shd->thd", probs, v_exp)

            o = o * torch.sigmoid(gate_a.view(T, n_heads, hd))
            o = o * torch.sigmoid(lse.t() - sinks).unsqueeze(-1)
            o_flat = o.reshape(T, n_heads * hd)
            if fp16_compute:
                o_flat = o_flat.half()

            attn_partial = lin(o_flat, wo)
            attn_out = static_sconv_f32(attn_partial, attn_sw)

            first = (i % block_size == 0)
            mlp_q = mlp_queries[i]
            mlp_nw = None if messages is not None else mn_w
            if first:
                mlp_in = attn_res_f32(stack[: nb + 1], attn_out, None, mlp_q, mlp_nw, eps, True)
            else:
                mlp_in = attn_res_f32(stack[:nb], partial, attn_out, mlp_q, mlp_nw, eps, True)
            if messages is not None:
                d_norm = rms_norm_f32(mlp_in[0], None, 1e-6)
                m_norm = rms_norm_f32(messages[3], None, 1e-6)
                pair = torch.cat([d_norm, m_norm], dim=-1)
                hid = lin(pair, c_down[3])
                if fp16_compute:
                    hid = hid.half()
                gain = torch.tanh(lin(hid, c_gain[3]))
                fgt = torch.tanh(lin(hid, c_fgt[3]))
                blended = (1.0 - fgt) * mlp_in[0] + gain * messages[3]
                mlp_x = rms_norm_f32(blended, mn_w[0], eps)
            else:
                mlp_x = mlp_in[0]

            if fp16_compute:
                mlp_x = mlp_x.half()

            residual = None if first else partial
            if residual is None:
                nb += 1
                st["num_blocks"] = nb

            r_logits = (
                torch.mm(mlp_x.half(), gate_w.half().t(), out_dtype=torch.float32)
                if fp16_compute
                else torch.mm(mlp_x.float(), gate_w.float().t())
            )
            sel = torch.sigmoid(r_logits) + gate_b.unsqueeze(0)
            top_idx = torch.topk(sel, k=topk, dim=-1).indices
            top_logits = r_logits.gather(1, top_idx)
            r_w = torch.softmax(F.logsigmoid(top_logits), dim=-1) * route_scale

            active_experts.update(top_idx.unique().tolist())
            pre_mlp.append((st, mlp_x, top_idx, r_w, attn_out, residual))

        moe_outs = [torch.zeros((st["T"], H), dtype=torch.float32, device=DEVICE) for st in states]
        for e in sorted(active_experts):
            if is_nvfp4:
                we13 = dequant_nvfp4_expert(
                    w13_slice[e].to(DEVICE), s13_slice[e].to(DEVICE), float(g13_all[e].item())
                )
                we2 = dequant_nvfp4_expert(
                    w2_slice[e].to(DEVICE), s2_slice[e].to(DEVICE), float(g2_all[e].item())
                )
            else:
                we13 = dequant_mxfp8_expert(w13_slice[e].to(DEVICE), s13_slice[e].to(DEVICE))
                we2 = dequant_mxfp8_expert(w2_slice[e].to(DEVICE), s2_slice[e].to(DEVICE))
            if fp16_compute:
                we13_h, we2_h = we13.half(), we2.half()
            for bi, (st, mlp_x, top_idx, r_w, _, _) in enumerate(pre_mlp):
                mask = top_idx == e
                if not mask.any():
                    continue
                tok_idx, k_pos = mask.nonzero(as_tuple=True)
                xin = mlp_x[tok_idx]
                if fp16_compute:
                    gu = torch.mm(xin.half(), we13_h.t(), out_dtype=torch.float32)
                else:
                    gu = torch.mm(xin.float(), we13.t())
                g_val = gate_cap * torch.tanh(gu[:, 0::2] / gate_cap)
                u_val = up_cap * torch.tanh(gu[:, 1::2] / up_cap)
                act = F.silu(g_val) * u_val
                if fp16_compute:
                    dn = torch.mm(act.half(), we2_h.t(), out_dtype=torch.float32)
                else:
                    dn = torch.mm(act, we2.t())
                w_sc = r_w[tok_idx, k_pos].unsqueeze(-1)
                moe_outs[bi].index_add_(0, tok_idx, dn * w_sc)
            del we13, we2

        for bi, (st, _, _, _, attn_out, residual) in enumerate(pre_mlp):
            nb = st["num_blocks"]
            stack = st["stack"]
            m_conv = static_sconv_f32(moe_outs[bi], mlp_sw)
            acc = attn_out if residual is None else (residual + attn_out)
            stack[nb].copy_(acc + m_conv)
            st["inputs"] = attn_res_f32(stack[:nb], stack[nb], None, next_q, next_nw, eps, True)

    lm_w = ws.get("model.llm.unembed.weight")[:unpadded_vocab]
    logits_out = []
    for st in states:
        h_final = st["inputs"][0]
        if fp16_compute:
            logits = torch.mm(h_final.half(), lm_w.half().t(), out_dtype=torch.float32)
        else:
            logits = torch.mm(h_final.float(), lm_w.float().t())
        logits = softcap * torch.tanh(logits / softcap)
        logits_out.append(logits.cpu())
    del lm_w
    return logits_out, recorded_pass2


def fit_post_prologue_gla(
    stats: dict, rank: int, qo_weighted: bool = True
) -> tuple[dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], float]:
    """Fit post-prologue TP4-native GLA-g4 (512 -> rank -> 512) per global visit & GPU group."""
    projs = {}
    tail_energies = []
    hd = 128
    for key_v, rec in stats.items():
        g_list = []
        for g in range(4):
            X = torch.cat(rec["kv_list"][g], dim=0)  # [N_tok, 512] float64
            # Build block-diagonal metric M = diag(M_K0, M_K1, M_V0, M_V1)
            if qo_weighted:
                blocks_m = []
                blocks_inv = []
                for sub in range(2):
                    cq = rec["cq_sum"][g][sub]
                    cq = 0.5 * (cq + cq.t())
                    tr_c = float(torch.trace(cq).item()) / hd
                    cq = cq / max(tr_c, 1e-12) + 1e-5 * torch.eye(hd, dtype=torch.float64)
                    L = torch.linalg.cholesky(cq)  # cq = L L^T -> metric M = L^T
                    blocks_m.append(L.t())
                    blocks_inv.append(torch.linalg.inv(L.t()))
                for sub in range(2):
                    mv = rec["mv"][g][sub]
                    mv = 0.5 * (mv + mv.t())
                    tr_m = float(torch.trace(mv).item()) / hd
                    mv = mv / max(tr_m, 1e-12) + 1e-5 * torch.eye(hd, dtype=torch.float64)
                    L = torch.linalg.cholesky(mv)
                    blocks_m.append(L.t())
                    blocks_inv.append(torch.linalg.inv(L.t()))
                M = torch.block_diag(*blocks_m)  # [512, 512]
                Minv = torch.block_diag(*blocks_inv)
                Xm = X @ M.t()  # weighted data: ||X M^T - Xhat M^T||_F
                _, S, Vh = torch.linalg.svd(Xm, full_matrices=False)
                Vr = Vh[:rank]  # [r, 512]
                tail_e = float((S[rank:].square().sum() / S.square().sum()).item())
                tail_energies.append(tail_e)
                # Map back: Xhat = X (M^T Vr^T) (Vr M^-T)
                w_down = (M.t() @ Vr.t()).float().to(DEVICE)  # [512, r]
                w_up = (Vr @ Minv.t()).float().to(DEVICE)  # [r, 512]
            else:
                _, S, Vh = torch.linalg.svd(X, full_matrices=False)
                Vr = Vh[:rank]
                tail_e = float((S[rank:].square().sum() / S.square().sum()).item())
                tail_energies.append(tail_e)
                w_down = Vr.t().float().to(DEVICE)
                w_up = Vr.float().to(DEVICE)
            # Static FP8 latent scale (keep absmax <= 380 for safety margin inside +-448)
            lat_cal = (X.float().to(DEVICE) @ w_down).abs().amax().item()
            lat_scale = torch.tensor(max(lat_cal / 320.0, 1e-4), dtype=torch.float32, device=DEVICE)
            g_list.append((w_down, w_up, lat_scale))
        projs[key_v] = g_list
    mean_tail = 100.0 * sum(tail_energies) / max(len(tail_energies), 1)
    return projs, mean_tail


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Path to Inkling-2-Mini checkpoint")
    ap.add_argument("--fp8-path", default="", help="Optional offline FP8 dense safetensors path")
    ap.add_argument("--out", default="./lossy_oracle_results.json", help="Output JSON path")
    args = ap.parse_args()

    t0 = time.time()
    model_dir = Path(args.model_dir)
    cfg_all = json.loads((model_dir / "config.json").read_text())
    cfg = cfg_all["text_config"]
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    eval_ids = [torch.tensor(tok.encode(p), dtype=torch.long, device=DEVICE) for _, p in PROMPTS]
    calib_ids = [torch.tensor(tok.encode(p), dtype=torch.long, device=DEVICE) for p in CALIB_TEXTS]
    print(f"Eval tokens: {sum(x.numel() for x in eval_ids)} across {len(eval_ids)} prompts; "
          f"Disjoint GLA calib tokens: {sum(x.numel() for x in calib_ids)} across {len(calib_ids)} texts")

    ws_f32 = WeightStore(None, set(), torch.float32, model_dir=model_dir)
    fp8_p = Path(args.fp8_path) if args.fp8_path else None
    ws_p3 = WeightStore(fp8_p, {"attn", "lm_head"} if fp8_p else set(), torch.float16, model_dir=model_dir)

    print("\n[1/7] Running exact FP32 Oracle (48 visits, FP32 KV)...")
    logits_f32, _ = run_inkling_variant(eval_ids, cfg, ws_f32, fp8_kv=False, fp16_compute=False)

    print("[2/7] Running Phase 3 Baseline (FP8 attn+lm_head + FP8 KV) + recording Pass-2 KV...")
    logits_p3, pass2_kv_0 = run_inkling_variant(eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True, record_pass2_kv=True)
    m_p3 = eval_metrics(logits_f32, logits_p3, eval_ids)
    print(f"  Phase 3 Baseline (exact KV)         | {m_p3}")

    results = {"p3_baseline": m_p3}

    # --- Candidate 1: ELR Last-Loop KV Caching (2 fixed-point passes for exact autoregressive trajectory) ---
    print("\n[3/7] Evaluating ELR Last-Loop KV Caching (arXiv:2610.12582v1)...")
    global_core = {11, 17}
    for name, layer_subset in [("elr_llkv_all_core", None), ("elr_llkv_global_core", global_core)]:
        _, pass2_kv_1 = run_inkling_variant(
            eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
            last_loop_kv_cache=pass2_kv_0, last_loop_kv_layers=layer_subset, record_pass2_kv=True,
        )
        logits_ll, _ = run_inkling_variant(
            eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
            last_loop_kv_cache=pass2_kv_1, last_loop_kv_layers=layer_subset,
        )
        m = eval_metrics(logits_f32, logits_ll, eval_ids)
        m_vs_p3 = eval_metrics(logits_p3, logits_ll, eval_ids)
        results[name] = {"vs_fp32": m, "vs_p3": m_vs_p3}
        print(f"  {name:35s} | vs FP32: {m} | vs P3 top1={m_vs_p3['top1_pct']}% KL={m_vs_p3['kl_mean']}")

    # --- Candidate 2: LoopCD-Hidden (R=3 and R=2) ---
    print("\n[4/7] Evaluating LoopCD-Hidden (arXiv:2610.02185v1)...")
    for n_pass, omegas in [(3, [0.15, 0.30]), (2, [0.0, 0.25, 0.50, 0.75, 1.0])]:
        for om in omegas:
            name = f"loopcd_R{n_pass}_w{om:.2f}"
            logits_cd, _ = run_inkling_variant(
                eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
                num_passes_override=n_pass, loopcd_omega=om,
            )
            m = eval_metrics(logits_f32, logits_cd, eval_ids)
            results[name] = m
            print(f"  {name:35s} | vs FP32: {m}")

    # --- Candidate 3: Post-Prologue TP4-Native GLA-g4 on the 8 Global Logical Visits ---
    print("\n[5/7] Collecting strictly disjoint post-prologue global KV calibration stats...")
    gla_stats: dict = {}
    run_inkling_variant(
        calib_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        record_global_gla_stats=gla_stats,
    )

    print("[6/7] Fitting & evaluating Post-Prologue GLA-g4 (Q/O-weighted & plain, FP8 latent KV)...")
    for rank in [256, 192, 128]:
        for qow in ([True, False] if rank == 256 else [True]):
            tag = f"gla_g4_r{rank}_{'qo' if qow else 'plain'}_fp8"
            projs, tail_pct = fit_post_prologue_gla(gla_stats, rank=rank, qo_weighted=qow)
            logits_gla, _ = run_inkling_variant(
                eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
                gla_projs=projs, gla_fp8_latent=True,
            )
            m = eval_metrics(logits_f32, logits_gla, eval_ids)
            m["calib_tail_energy_pct"] = round(tail_pct, 2)
            m_vs_p3 = eval_metrics(logits_p3, logits_gla, eval_ids)
            results[tag] = {"vs_fp32": m, "vs_p3": m_vs_p3}
            print(f"  {tag:35s} | tail={tail_pct:5.2f}% | vs FP32: {m} | vs P3 top1={m_vs_p3['top1_pct']}% KL={m_vs_p3['kl_mean']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[7/7] Done in {time.time() - t0:.1f}s! Saved {out_path}")


if __name__ == "__main__":
    main()
