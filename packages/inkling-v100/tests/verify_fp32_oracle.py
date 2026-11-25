# SPDX-License-Identifier: Apache-2.0
"""Real-weights FP32 oracle vs Volta sm_70 ops + activation magnitude audit.

Runs all 48 logical layer visits of `thinkingmachines/Inkling-2-Mini` on GPU 0
layer-by-layer (streaming weights from safetensors / CPU RAM to keep VRAM < 4 GiB),
comparing:
  1. Pure FP32 reference (exact BF16->FP32 weights, exact NVFP4/MXFP8 FP32 dequant, FP32 math)
  2. Volta sm_70 ops (`inkling_v100` FP16 Tensor Cores + FP32 residual stack + skinny_moe_qpn_sm70)
     with FP16 KV cache AND with unit-scale FP8 E4M3FN KV cache!

Also audits real activation & weight-scale magnitudes across all 48 visits (conv state,
residual stack, pre-softcap w13_out, KV values, NVFP4 global scales, MXFP8 UE8M0 ranges).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    nvfp4_skinny_scale_reason,
    qpn_prepack,
)

from inkling_v100.ops.attention_sm70 import (
    MODE_OUT_LSE,
    compute_log_scaling_tau,
    gate_sink,
    rel_attention,
    rel_decode_attention,
)
from inkling_v100.ops.attn_res import attn_res
from inkling_v100.ops.loop import blend_rms_norm, pair_rms_norm
from inkling_v100.ops.moe_sm70 import (
    combine_sm70,
    run_mxfp8_experts_sm70,
    run_nvfp4_experts_sm70,
    select_experts_sm70,
)
from inkling_v100.ops.norm import rms_norm
from inkling_v100.ops.sconv import (
    attention_prologue,
    sconv_seq_metadata,
    shift_rel_proj,
    short_conv,
)

DEVICE = torch.device("cuda:0")
MODEL_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("INKLING_MODEL_DIR", "Inkling-2-Mini"))

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
    device=DEVICE,
)


def dequant_nvfp4_one(code_u8: torch.Tensor, scale_fp8: torch.Tensor, g: float) -> torch.Tensor:
    N, K2 = code_u8.shape
    lo = _E2M1_LUT[(code_u8 & 0xF).long()]
    hi = _E2M1_LUT[(code_u8 >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).view(N, K2 * 2)
    s = scale_fp8.to(torch.float32).repeat_interleave(16, dim=-1)
    return w * s * g


def dequant_mxfp8_one(w_fp8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    exp = torch.where(
        s_u8 == 255,
        torch.zeros_like(s_u8, dtype=torch.float32),
        torch.pow(2.0, s_u8.float() - 127.0),
    )
    return w_fp8.to(torch.float32) * exp.repeat_interleave(32, dim=-1)


def ref_rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def ref_attn_res(
    stack: list[torch.Tensor],
    last: torch.Tensor,
    query: torch.Tensor,
    norm_w: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    # stack: list of [T, D], last: [T, D], query: [Q, D] already scaled by D**-0.5
    S = torch.stack([*stack, last], dim=0)  # [N, T, D]
    Sn = S * torch.rsqrt(S.square().mean(-1, keepdim=True) + eps)
    logits = torch.einsum("ntd,qd->qnt", Sn, query)
    probs = torch.softmax(logits, dim=1)
    mixed = torch.einsum("qnt,ntd->qtd", probs, Sn)
    if norm_w is None:
        return mixed
    return mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps) * norm_w[:, None, :]


def ref_sconv_static(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: [T, C], w: [C, W]
    T, C = x.shape
    W = w.shape[1]
    scale = (W + 1) ** -0.5
    out = x.clone()
    for k in range(W):
        lag = W - 1 - k
        if lag == 0:
            out = out + x * w[:, k]
        elif lag < T:
            out[lag:] = out[lag:] + x[:-lag] * w[:, k]
    return out * scale


def ref_sconv_dynamic(x: torch.Tensor, static_w: torch.Tensor, dyn: torch.Tensor, head_dim: int) -> torch.Tensor:
    # x: [T, C], static_w: [W, C], dyn: [T, H*W]
    T, C = x.shape
    W = static_w.shape[0]
    H = C // head_dim
    scale = (2 * W + 1) ** -0.5
    df = dyn.view(T, H, W)
    out = x.clone()
    for lag in range(W):
        w_eff = static_w[lag][None, :] + df[:, :, lag].repeat_interleave(head_dim, dim=-1)
        if lag == 0:
            out = out + x * w_eff
        elif lag < T:
            out[lag:] = out[lag:] + x[:-lag] * w_eff[lag:]
    return out * scale


def ref_l2_clip(x: torch.Tensor, scale: torch.Tensor | None, thresh: float, eps: float = 1e-6) -> torch.Tensor:
    if scale is not None:
        x = x * scale[None, :, :]
    n = x.norm(dim=-1, keepdim=True)
    return x * (thresh / (n + eps)).clamp(max=1.0)


@torch.no_grad()
def main() -> None:
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    tcfg = cfg["text_config"]
    qcfg = json.loads((MODEL_DIR / "hf_quant_config.json").read_text())
    mxfp8_layers = set()
    for grp in qcfg["model"]["parameter_groups"]:
        if grp["compute_dtype"] == "mxfp8":
            for t in grp["tensors"]:
                parts = t.split(".")
                if len(parts) > 3 and parts[2] == "layers":
                    mxfp8_layers.add(int(parts[3]))
    print(f"Model: {MODEL_DIR} | MXFP8 layers: {sorted(mxfp8_layers)}", flush=True)

    sf_path = MODEL_DIR / "model-00001-of-00001.safetensors"
    f = safe_open(str(sf_path), framework="pt", device="cpu")
    all_keys = set(f.keys())
    print(f"Safetensors loaded handle: {len(all_keys)} tensors", flush=True)

    # 1. Scale & metadata audit across ALL 24 layers first!
    print("\n=== 1. Weight Scale Audit (all 24 layers, 128 experts/layer) ===", flush=True)
    nvfp4_g_min, nvfp4_g_max = float("inf"), 0.0
    mxfp8_s_min, mxfp8_s_max, mxfp8_nan = 255, 0, 0
    for i in range(tcfg["num_hidden_layers"]):
        if i in mxfp8_layers:
            for nm in ("w13", "w2"):
                s = f.get_tensor(f"model.llm.layers.{i}.mlp.experts.{nm}_weight_scale")
                mxfp8_nan += int((s == 255).sum())
                valid = s[s != 255]
                mxfp8_s_min = min(mxfp8_s_min, int(valid.min()))
                mxfp8_s_max = max(mxfp8_s_max, int(valid.max()))
        else:
            g13 = f.get_tensor(f"model.llm.layers.{i}.mlp.experts.w13_weight_amax") / f.get_tensor(
                f"model.llm.layers.{i}.mlp.experts.w13_weight_outer_scale_denominator"
            )
            g2 = f.get_tensor(f"model.llm.layers.{i}.mlp.experts.w2_weight_amax") / f.get_tensor(
                f"model.llm.layers.{i}.mlp.experts.w2_weight_outer_scale_denominator"
            )
            reason = nvfp4_skinny_scale_reason(g13, g2)
            assert reason is None, f"Layer {i} failed nvfp4_skinny_scale_reason: {reason}"
            nvfp4_g_min = min(nvfp4_g_min, float(g13.min()), float(g2.min()))
            nvfp4_g_max = max(nvfp4_g_max, float(g13.max()), float(g2.max()))
    print(
        f"  NVFP4 (L0..19): ALL 20x128 experts PASS skinny_qpn rebias check! "
        f"g in [{nvfp4_g_min:.6e}, {nvfp4_g_max:.6e}] (max*2^14={nvfp4_g_max*16384:.2f} << 65504)",
        flush=True,
    )
    print(
        f"  MXFP8 (L20..23): UE8M0 bytes in [{mxfp8_s_min}, {mxfp8_s_max}] "
        f"(exponents [{mxfp8_s_min-127}, {mxfp8_s_max-127}]), NaN(255) count={mxfp8_nan}",
        flush=True,
    )

    # Build a realistic 32-token prompt using the real tokenizer if available, else token IDs.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)
        prompt = (
            "Explain step by step why Volta sm_70 Tensor Cores accumulate FP16 "
            "HMMA.884 tiles in FP32 and derive the exact arithmetic intensity of "
            "a 2304x1152 MoE projection. Answer:"
        )
        input_ids_list = tok.encode(prompt)
        print(f"\nPrompt ({len(input_ids_list)} tokens): {prompt!r}", flush=True)
    except Exception as exc:
        print(f"Tokenizer fallback ({exc}); using deterministic token IDs", flush=True)
        tok = None
        input_ids_list = [1, 791, 6864, 315, 9822, 374, 12366, 13, 358, 1097, 264, 4221, 315, 279, 1917, 13]

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=DEVICE)
    T = input_ids.numel()
    H = tcfg["hidden_size"]
    H_q = tcfg["num_attention_heads"]
    H_kv = tcfg["num_key_value_heads"]
    D = tcfg["head_dim"]
    D_rel = tcfg["d_rel"]
    E_rel = tcfg["rel_extent"]
    W_sconv = tcfg["sconv_kernel_size"]
    eps = tcfg["rms_norm_eps"]
    local_ids = set(tcfg["local_layer_ids"])
    blk_sz = tcfg["n_layers_per_attn_res_block"]
    core_start = tcfg["loop_config"]["core_start"]
    core_end = tcfg["loop_config"]["core_end"]
    num_passes = tcfg["loop_config"]["num_passes"]
    gate_cap = tcfg["mlp_gate_softcap"]
    up_cap = tcfg["mlp_up_softcap"]
    route_scale = tcfg["route_scale"]
    topk = tcfg["num_experts_per_tok"]
    n_experts = tcfg["n_routed_experts"]
    q_clip = tcfg["qk_l2_clip_threshold"]
    r_clip = tcfg["rel_l2_clip_threshold"]
    inv_sqrt_H = H**-0.5

    visits = [(i, 0) for i in range(core_start)]
    visits += [(i, p) for p in range(num_passes) for i in range(core_start, core_end)]
    visits += [(i, 0) for i in range(core_end, tcfg["num_hidden_layers"])]
    assert len(visits) == 48

    # Load global small tensors
    embed_w = f.get_tensor("model.llm.embed.weight").to(DEVICE)
    embed_norm_w = f.get_tensor("model.llm.embed_norm.weight").to(DEVICE)
    out_norm_w = f.get_tensor("model.llm.norm.weight").to(DEVICE)
    attn_queries = [
        f.get_tensor(f"model.llm.attn_res.attn_queries.{i}").float().to(DEVICE).view(-1, H) * inv_sqrt_H
        for i in range(tcfg["num_hidden_layers"] - 1)
    ]
    mlp_queries = [
        f.get_tensor(f"model.llm.attn_res.mlp_queries.{i}").float().to(DEVICE).view(1, H) * inv_sqrt_H
        for i in range(tcfg["num_hidden_layers"])
    ]
    output_query = f.get_tensor("model.llm.attn_res.output_query").float().to(DEVICE).view(1, H) * inv_sqrt_H
    cp_q = f.get_tensor("model.llm.middle_cycle_cross_pass_queries").float().to(DEVICE).view(-1, 2, H)
    cp_kv = f.get_tensor("model.llm.middle_cycle_cross_pass_kv_queries").float().to(DEVICE).view(-1, 2, H)
    msg_queries = torch.stack((cp_q[:, 0], cp_kv[:, 0], cp_kv[:, 1], cp_q[:, 1]), 1) * inv_sqrt_H

    positions = torch.arange(T, dtype=torch.int32, device=DEVICE)
    tau = compute_log_scaling_tau(positions, tcfg["log_scaling_n_floor"], tcfg["log_scaling_alpha"])

    # Initialize 3 parallel execution states:
    #   (A) ref: pure FP32
    #   (B) sm70_fp16: Volta sm_70 kernels + FP32 residual stack + FP16 KV
    #   (C) sm70_fp8:  Volta sm_70 kernels + FP32 residual stack + FP8 E4M3FN KV
    x0_ref = ref_rms(embed_w[input_ids].float(), embed_norm_w.float(), eps)
    x0_sm70 = rms_norm(embed_w.half()[input_ids], embed_norm_w.half(), eps, out_dtype=torch.float32)

    del embed_w

    max_blks = 5  # _max_loop_blocks() + 1 = 4 + 1
    stack_ref: list[torch.Tensor] = [x0_ref]
    stack_16 = torch.zeros((max_blks, T, H), dtype=torch.float32, device=DEVICE)
    stack_8 = torch.zeros((max_blks, T, H), dtype=torch.float32, device=DEVICE)
    stack_16[0].copy_(x0_sm70)
    stack_8[0].copy_(x0_sm70)

    # Shared sconv / KV cache buffers for sm70 single-sequence prefill (reused per visit)
    ws_hid = H // H_kv  # 288
    cs_head_size = 1 << (2 * D + 2 * ws_hid - 1).bit_length()  # 1024
    c_blks = (T + W_sconv - 1) // W_sconv + 1
    conv_cache_16 = torch.zeros((c_blks, H_kv, W_sconv, cs_head_size), dtype=torch.float16, device=DEVICE)
    conv_cache_8 = torch.zeros((c_blks, H_kv, W_sconv, cs_head_size), dtype=torch.float16, device=DEVICE)
    c_bt = torch.arange(c_blks, dtype=torch.int32, device=DEVICE)[None, :]
    c_slots = torch.arange(T, dtype=torch.int64, device=DEVICE)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([T], dtype=torch.int32, device=DEVICE)
    seq_idx = torch.empty(T, dtype=torch.int32, device=DEVICE)
    qstart = torch.empty(T, dtype=torch.int32, device=DEVICE)
    sconv_seq_metadata(qsl, 1, T, seq_idx, qstart)

    kv_page = 16
    kv_blks = (T + kv_page - 1) // kv_page + 1
    kv_bt = torch.arange(kv_blks, dtype=torch.int32, device=DEVICE)[None, :]
    kv_slots = torch.arange(T, dtype=torch.int64, device=DEVICE)
    kc16 = torch.zeros((kv_blks, kv_page, H_kv, D), dtype=torch.float16, device=DEVICE)
    vc16 = torch.zeros((kv_blks, kv_page, H_kv, D), dtype=torch.float16, device=DEVICE)
    kc8 = torch.zeros((kv_blks, kv_page, H_kv, D), dtype=torch.float8_e4m3fn, device=DEVICE)
    vc8 = torch.zeros((kv_blks, kv_page, H_kv, D), dtype=torch.float8_e4m3fn, device=DEVICE)
    bounds_all = torch.tensor([0, n_experts], dtype=torch.int32, device=DEVICE)

    # Loop state trackers
    nb_ref = nb_16 = nb_8 = 0
    in_ref = in_16 = in_8 = None
    bank_ref = bank_16 = bank_8 = None
    pre_nb = pre_part_ref = pre_dest_ref = None
    pre_part_16 = pre_dest_16 = None
    pre_part_8 = pre_dest_8 = None

    # Activation magnitude stats
    stats = {
        "max_stack": 0.0,
        "max_attn_pre_sconv": 0.0,
        "max_mlp_pre_sconv": 0.0,
        "max_k_pre_sconv": 0.0,
        "max_v_pre_sconv": 0.0,
        "max_k_kv": 0.0,
        "max_v_kv": 0.0,
        "max_w13_out": 0.0,
        "max_router_diff_frac": 0.0,
    }

    print("\n=== 2. Running 48 Logical Visits (FP32 Oracle vs sm_70 FP16-KV & FP8-KV) ===", flush=True)
    t_start = time.time()

    # Cache loaded layer weights only while running that layer (to keep VRAM small)
    for visit, (i, pass_idx) in enumerate(visits):
        is_local = i in local_ids
        is_mxfp8 = i in mxfp8_layers
        looped = pass_idx > 0
        pfx = f"model.llm.layers.{i}"

        # Handle loop pass boundaries
        if i == core_start and not looped:
            pre_nb = nb_ref
            pre_part_ref = stack_ref[nb_ref].clone()
            pre_part_16 = stack_16[nb_16].clone()
            pre_part_8 = stack_8[nb_8].clone()
        elif i == core_start:
            bank_ref = [t.clone() for t in stack_ref[pre_nb + 1 : nb_ref + 1]]
            nb_ref = pre_nb
            stack_ref = stack_ref[: nb_ref + 1]
            stack_ref[nb_ref] = pre_part_ref.clone()
            if pre_dest_ref is None:
                pre_dest_ref = ref_attn_res(stack_ref[:nb_ref], pre_part_ref, attn_queries[i - 1], None, eps)
            in_ref = pre_dest_ref

            bank_16 = stack_16[pre_nb + 1 : nb_16 + 1].clone()
            nb_16 = pre_nb
            stack_16[nb_16].copy_(pre_part_16)
            if pre_dest_16 is None:
                pre_dest_16 = attn_res(stack_16[:nb_16], pre_part_16, None, attn_queries[i - 1], None, 0.0, True)
            in_16 = pre_dest_16

            bank_8 = stack_8[pre_nb + 1 : nb_8 + 1].clone()
            nb_8 = pre_nb
            stack_8[nb_8].copy_(pre_part_8)
            if pre_dest_8 is None:
                pre_dest_8 = attn_res(stack_8[:nb_8], pre_part_8, None, attn_queries[i - 1], None, 0.0, True)
            in_8 = pre_dest_8

        # Load this layer's weights onto GPU
        if is_local:
            attn_nw = f.get_tensor(f"{pfx}.attn_norm.weight").to(DEVICE).view(1, H)
        else:
            attn_nw = torch.stack(
                [
                    f.get_tensor(f"{pfx}.q_input_norm.weight"),
                    f.get_tensor(f"{pfx}.k_input_norm.weight"),
                    f.get_tensor(f"{pfx}.v_input_norm.weight"),
                ],
                dim=0,
            ).to(DEVICE)
        mlp_nw = f.get_tensor(f"{pfx}.mlp_norm.weight").to(DEVICE).view(1, H)
        wq = f.get_tensor(f"{pfx}.attn.wq_du.weight").to(DEVICE)
        wr = f.get_tensor(f"{pfx}.attn.wr_du.weight").to(DEVICE)
        wgate = f.get_tensor(f"{pfx}.attn.gate_attn.weight").to(DEVICE)
        wsink = f.get_tensor(f"{pfx}.attn.attn_sink_proj.weight").to(DEVICE)
        wk = f.get_tensor(f"{pfx}.attn.wk_dv.weight").to(DEVICE)
        wv = f.get_tensor(f"{pfx}.attn.wv_dv.weight").to(DEVICE)
        wo = f.get_tensor(f"{pfx}.attn.wo_ud.weight").to(DEVICE)
        q_sc = f.get_tensor(f"{pfx}.attn.q_l2_clip.scale").to(DEVICE)
        r_sc = f.get_tensor(f"{pfx}.attn.rel_l2_clip.scale").to(DEVICE)
        rel_proj = f.get_tensor(f"{pfx}.attn.rel_logits_proj.proj").to(DEVICE)
        if is_local:
            k_sw = f.get_tensor(f"{pfx}.attn.k_sconv.weight").squeeze(1).to(DEVICE)
            v_sw = f.get_tensor(f"{pfx}.attn.v_sconv.weight").squeeze(1).to(DEVICE)
            k_wp = v_wp = None
        else:
            k_sw = f.get_tensor(f"{pfx}.attn.k_sconv.static_w").to(DEVICE)
            v_sw = f.get_tensor(f"{pfx}.attn.v_sconv.static_w").to(DEVICE)
            k_wp = f.get_tensor(f"{pfx}.attn.k_sconv.weight_proj").to(DEVICE)
            v_wp = f.get_tensor(f"{pfx}.attn.v_sconv.weight_proj").to(DEVICE)
        attn_sw = f.get_tensor(f"{pfx}.attn_sconv.weight").squeeze(1).to(DEVICE)
        mlp_sw = f.get_tensor(f"{pfx}.mlp_sconv.weight").squeeze(1).to(DEVICE)

        ctrl_down = ctrl_up = None
        if looped:
            downs, ups = [], []
            for st in ("attn", "k", "v", "mlp"):
                downs.append(f.get_tensor(f"{pfx}.controllers.{st}.down_weight").to(DEVICE))
                g_w = f.get_tensor(f"{pfx}.controllers.{st}.gain_weight").to(DEVICE)
                f_w = f.get_tensor(f"{pfx}.controllers.{st}.forget_weight").to(DEVICE)
                ups.append(torch.cat([g_w, f_w], dim=0))
            ctrl_down = torch.stack(downs, dim=0)  # [4, 64, 2H]
            ctrl_up = torch.stack(ups, dim=0)  # [4, 2H, 64]

        # ---- Prepare Attention inputs ----
        if looped:
            mq = msg_queries[i - core_start]  # [4, H]
            msg_ref = ref_attn_res([bank_ref[0]], bank_ref[1], mq, None, eps)
            d_ref = in_ref.expand(3, -1, -1) if is_local else in_ref
            nw_ref = attn_nw.float().expand(3, -1) if is_local else attn_nw.float()
            # FP32 controller blend
            dn = d_ref * torch.rsqrt(d_ref.square().mean(-1, keepdim=True) + eps)
            mn = msg_ref[:3] * torch.rsqrt(msg_ref[:3].square().mean(-1, keepdim=True) + eps)
            pair_r = torch.cat([dn, mn], dim=-1)
            hid_r = torch.bmm(pair_r, ctrl_down[:3].float().transpose(1, 2))
            gf_r = torch.bmm(hid_r, ctrl_up[:3].float().transpose(1, 2))
            bl_r = (1.0 - torch.tanh(gf_r[..., H:])) * d_ref + torch.tanh(gf_r[..., :H]) * msg_ref[:3]
            a_in_ref = bl_r * torch.rsqrt(bl_r.square().mean(-1, keepdim=True) + eps) * nw_ref[:, None, :]

            def _sm70_blend_attn(bank_t, in_t):
                ms = attn_res(bank_t[:1], bank_t[1], None, mq, None, 0.0, True)
                dt = in_t.expand(3, -1, -1) if is_local else in_t
                nwt = attn_nw.half().expand(3, -1) if is_local else attn_nw.half()
                pr = pair_rms_norm(dt, ms[:3], out_dtype=torch.float16)
                hd = torch.bmm(pr, ctrl_down[:3].half().transpose(1, 2))
                gf = torch.bmm(hd, ctrl_up[:3].half().transpose(1, 2), out_dtype=torch.float32)
                return ms, blend_rms_norm(dt, ms[:3], gf, nwt, eps, out_dtype=torch.float16)

            msg_16, a_in_16 = _sm70_blend_attn(bank_16, in_16)
            msg_8, a_in_8 = _sm70_blend_attn(bank_8, in_8)
        elif in_ref is None:
            a_in_ref = [ref_rms(stack_ref[nb_ref], attn_nw[j].float(), eps) for j in range(attn_nw.shape[0])]
            a_in_16 = [rms_norm(stack_16[nb_16], attn_nw[j].half(), eps, torch.float16) for j in range(attn_nw.shape[0])]
            a_in_8 = [rms_norm(stack_8[nb_8], attn_nw[j].half(), eps, torch.float16) for j in range(attn_nw.shape[0])]
            msg_ref = msg_16 = msg_8 = None
        else:
            a_in_ref, a_in_16, a_in_8 = in_ref, in_16, in_8
            msg_ref = msg_16 = msg_8 = None

        # ---- Attention: Pure FP32 reference ----
        xq_r = a_in_ref[0]
        xk_r = a_in_ref[0] if (is_local and len(a_in_ref) == 1) else a_in_ref[1]
        xv_r = a_in_ref[0] if (is_local and len(a_in_ref) == 1) else a_in_ref[2]
        q_r = F.linear(xq_r, wq.float()).view(T, H_q, D)
        r_r = F.linear(xq_r, wr.float()).view(T, H_q, D_rel)
        g_r = F.linear(xq_r, wgate.float())
        sk_r = F.linear(xq_r, wsink.float())
        k_r = F.linear(xk_r, wk.float())
        v_r = F.linear(xv_r, wv.float())
        stats["max_k_pre_sconv"] = max(stats["max_k_pre_sconv"], float(k_r.abs().max()))
        stats["max_v_pre_sconv"] = max(stats["max_v_pre_sconv"], float(v_r.abs().max()))
        q_r = ref_l2_clip(q_r, q_sc.float(), q_clip, eps)
        r_r = ref_l2_clip(r_r, r_sc.float(), r_clip, eps)
        if not is_local:
            q_r = q_r * tau[:, None, None]
            r_r = r_r * tau[:, None, None]
            k_c = ref_sconv_dynamic(k_r, k_sw.float(), F.linear(xk_r, k_wp.float()), D).view(T, H_kv, D)
            v_c = ref_sconv_dynamic(v_r, v_sw.float(), F.linear(xv_r, v_wp.float()), D).view(T, H_kv, D)
        else:
            k_c = ref_sconv_static(k_r, k_sw.float()).view(T, H_kv, D)
            v_c = ref_sconv_static(v_r, v_sw.float()).view(T, H_kv, D)
        k_c = ref_l2_clip(k_c, None, q_clip, eps)
        v_c = ref_rms(v_c, None, eps)
        stats["max_k_kv"] = max(stats["max_k_kv"], float(k_c.abs().max()))
        stats["max_v_kv"] = max(stats["max_v_kv"], float(v_c.abs().max()))

        G = H_q // H_kv
        scores = torch.einsum("qhd,khd->hqk", q_r, k_c.repeat_interleave(G, dim=1)) * (D**-0.5)
        rel_all = torch.einsum("qhd,de->hqe", r_r, rel_proj.float())
        q_pos = torch.arange(T, device=DEVICE)[:, None]
        k_pos = torch.arange(T, device=DEVICE)[None, :]
        dist = q_pos - k_pos
        vis = dist >= 0
        if is_local:
            vis = vis & (dist < E_rel)
        in_ext = vis & (dist < E_rel)
        bias = torch.gather(rel_all, 2, dist.clamp(0, E_rel - 1).unsqueeze(0).expand(H_q, T, T))
        scores = scores + torch.where(in_ext.unsqueeze(0), bias, torch.zeros_like(bias))
        scores = torch.where(vis.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
        lse_r = torch.logsumexp(scores, dim=-1)
        probs_r = torch.softmax(scores, dim=-1)
        ao_r = torch.einsum("hqk,khd->qhd", probs_r, v_c.repeat_interleave(G, dim=1))
        ao_r = ao_r * torch.sigmoid(g_r.view(T, H_q, D)) * torch.sigmoid(lse_r.t() - sk_r).unsqueeze(-1)
        attn_pre_r = F.linear(ao_r.reshape(T, H_q * D), wo.float())
        stats["max_attn_pre_sconv"] = max(stats["max_attn_pre_sconv"], float(attn_pre_r.abs().max()))
        attn_out_ref = ref_sconv_static(attn_pre_r, attn_sw.float())

        # ---- Attention: Volta sm_70 (both FP16 KV and FP8 KV) ----
        rel_shifted = shift_rel_proj(rel_proj.half(), is_local)

        def _sm70_attn(a_in, conv_c, kc, vc):
            conv_c.zero_()
            xq = a_in[0]
            xk = a_in[0] if (is_local and len(a_in) == 1) else a_in[1]
            xv = a_in[0] if (is_local and len(a_in) == 1) else a_in[2]
            q = F.linear(xq, wq.half()).view(T, H_q, D)
            r = F.linear(xq, wr.half()).view(T, H_q, D_rel)
            gate = F.linear(xq, wgate.half())
            sink = F.linear(xq, wsink.half())
            k = F.linear(xk, wk.half())
            v = F.linear(xv, wv.half())
            dk = None if is_local else F.linear(xk, k_wp.half())
            dv = None if is_local else F.linear(xv, v_wp.half())
            q_out = torch.empty_like(q)
            rel_out = torch.empty((T + 128, H_q, E_rel + 256), dtype=torch.float16, device=DEVICE)
            attention_prologue(
                q, r, k, v, q_sc.half(), r_sc.half(), None if is_local else tau,
                rel_shifted, k_sw.half(), v_sw.half(), dk, dv,
                conv_c, positions, c_bt, seq_idx, c_slots, qstart,
                kc, vc, kv_slots, q_out, rel_out, qsl, seq_lens,
                clip_threshold=q_clip, rel_threshold=r_clip, norm_eps=eps,
            )
            pf_o = torch.empty((T, H_q, D), dtype=torch.float32, device=DEVICE)
            lse = torch.empty((H_q, T), dtype=torch.float32, device=DEVICE)
            rel_attention(
                q_out, kc, vc, pf_o, lse, rel_out,
                block_table=kv_bt, cache_seqlens=seq_lens, cu_seqlens_q=qsl,
                max_query_len=T, softmax_scale=D**-0.5, window=E_rel if is_local else 0,
                num_splits=1, mode=MODE_OUT_LSE,
            )
            ao = torch.empty((T, H_q * D), dtype=torch.float16, device=DEVICE)
            gate_sink(pf_o, gate, lse.t(), sink, ao)
            attn_pre = F.linear(ao, wo.half())
            attn_out = torch.empty((T, H), dtype=torch.float32, device=DEVICE)
            short_conv(
                attn_pre, attn_sw.half(), conv_c, positions, c_bt, seq_idx, c_slots, qstart,
                2 * D, ws_hid, out=attn_out,
            )
            return attn_out

        attn_out_16 = _sm70_attn(a_in_16, conv_cache_16, kc16, vc16)
        attn_out_8 = _sm70_attn(a_in_8, conv_cache_8, kc8, vc8)

        # ---- MLP input AttnRes (+ loop blend) ----
        first = i % blk_sz == 0
        mq_mlp = mlp_queries[i]
        if first:
            mlp_in_ref = ref_attn_res(stack_ref[: nb_ref + 1], attn_out_ref, mq_mlp, None if looped else mlp_nw.float(), eps)
            mlp_in_16 = attn_res(stack_16[: nb_16 + 1], attn_out_16, None, mq_mlp, None if looped else mlp_nw.half(), 0.0 if looped else eps, True)
            mlp_in_8 = attn_res(stack_8[: nb_8 + 1], attn_out_8, None, mq_mlp, None if looped else mlp_nw.half(), 0.0 if looped else eps, True)
        else:
            mlp_in_ref = ref_attn_res(stack_ref[:nb_ref], stack_ref[nb_ref] + attn_out_ref, mq_mlp, None if looped else mlp_nw.float(), eps)
            mlp_in_16 = attn_res(stack_16[:nb_16], stack_16[nb_16], attn_out_16, mq_mlp, None if looped else mlp_nw.half(), 0.0 if looped else eps, True)
            mlp_in_8 = attn_res(stack_8[:nb_8], stack_8[nb_8], attn_out_8, mq_mlp, None if looped else mlp_nw.half(), 0.0 if looped else eps, True)

        if looped:
            dn = mlp_in_ref * torch.rsqrt(mlp_in_ref.square().mean(-1, keepdim=True) + eps)
            mn = msg_ref[3:] * torch.rsqrt(msg_ref[3:].square().mean(-1, keepdim=True) + eps)
            pr = torch.cat([dn, mn], dim=-1)
            hd = torch.bmm(pr, ctrl_down[3:].float().transpose(1, 2))
            gf = torch.bmm(hd, ctrl_up[3:].float().transpose(1, 2))
            bl = (1.0 - torch.tanh(gf[..., H:])) * mlp_in_ref + torch.tanh(gf[..., :H]) * msg_ref[3:]
            mlp_in_ref = bl * torch.rsqrt(bl.square().mean(-1, keepdim=True) + eps) * mlp_nw.float()[:, None, :]

            def _sm70_blend_mlp(mi, ms):
                pr = pair_rms_norm(mi, ms[3:], out_dtype=torch.float16)
                hd = torch.bmm(pr, ctrl_down[3:].half().transpose(1, 2))
                gf = torch.bmm(hd, ctrl_up[3:].half().transpose(1, 2), out_dtype=torch.float32)
                return blend_rms_norm(mi, ms[3:], gf, mlp_nw.half(), eps, out_dtype=torch.float16)

            mlp_in_16 = _sm70_blend_mlp(mlp_in_16, msg_16)
            mlp_in_8 = _sm70_blend_mlp(mlp_in_8, msg_8)

        # ---- MoE: FP32 oracle vs Volta sm_70 ----
        gw = f.get_tensor(f"{pfx}.mlp.gate.weight").to(DEVICE)
        gb = f.get_tensor(f"{pfx}.mlp.gate.bias").float().to(DEVICE)
        w13_raw = f.get_tensor(f"{pfx}.mlp.experts.w13_weight").to(DEVICE)
        s13_raw = f.get_tensor(f"{pfx}.mlp.experts.w13_weight_scale").to(DEVICE)
        w2_raw = f.get_tensor(f"{pfx}.mlp.experts.w2_weight").to(DEVICE)
        s2_raw = f.get_tensor(f"{pfx}.mlp.experts.w2_weight_scale").to(DEVICE)
        if not is_mxfp8:
            g13 = (
                f.get_tensor(f"{pfx}.mlp.experts.w13_weight_amax").to(DEVICE)
                / f.get_tensor(f"{pfx}.mlp.experts.w13_weight_outer_scale_denominator").to(DEVICE)
            ).float()
            g2 = (
                f.get_tensor(f"{pfx}.mlp.experts.w2_weight_amax").to(DEVICE)
                / f.get_tensor(f"{pfx}.mlp.experts.w2_weight_outer_scale_denominator").to(DEVICE)
            ).float()

        # FP32 router
        xr = mlp_in_ref[0]
        logits_r = F.linear(xr, gw.float())
        sel_r = torch.sigmoid(logits_r) + gb[None, :]
        top_idx_r = torch.topk(sel_r, topk, dim=-1).indices
        top_log_r = logits_r.gather(1, top_idx_r)
        w_r = torch.softmax(F.logsigmoid(top_log_r), dim=-1) * route_scale
        moe_ref = torch.zeros((T, H), dtype=torch.float32, device=DEVICE)
        active_e = sorted(set(top_idx_r.cpu().flatten().tolist()))
        for e in active_e:
            mask_e = top_idx_r == e
            t_idx, k_idx = mask_e.nonzero(as_tuple=True)
            W13 = dequant_mxfp8_one(w13_raw[e], s13_raw[e]) if is_mxfp8 else dequant_nvfp4_one(w13_raw[e], s13_raw[e], float(g13[e]))
            W2 = dequant_mxfp8_one(w2_raw[e], s2_raw[e]) if is_mxfp8 else dequant_nvfp4_one(w2_raw[e], s2_raw[e], float(g2[e]))
            y13 = xr[t_idx] @ W13.t()
            stats["max_w13_out"] = max(stats["max_w13_out"], float(y13.abs().max()))
            gg, uu = y13[:, 0::2], y13[:, 1::2]
            act = (gate_cap * torch.tanh(gg / gate_cap)) * torch.sigmoid(gg) * (up_cap * torch.tanh(uu / up_cap)) * w_r[t_idx, k_idx][:, None]
            moe_ref.index_add_(0, t_idx, act @ W2.t())
        stats["max_mlp_pre_sconv"] = max(stats["max_mlp_pre_sconv"], float(moe_ref.abs().max()))

        # sm_70 MoE (only prepack active experts from both sm70 paths to stay super fast!)
        ids_16, wts_16, s2d_16 = select_experts_sm70(mlp_in_16[0], gw.half(), gb, topk, route_scale, bounds_all)
        ids_8, wts_8, s2d_8 = select_experts_sm70(mlp_in_8[0], gw.half(), gb, topk, route_scale, bounds_all)
        router_mismatch = float((ids_16 != top_idx_r.to(torch.int32)).float().mean())
        stats["max_router_diff_frac"] = max(stats["max_router_diff_frac"], router_mismatch)

        if is_mxfp8:
            d16 = run_mxfp8_experts_sm70(mlp_in_16[0], w13_raw.view(torch.uint8), s13_raw, w2_raw.view(torch.uint8), s2_raw, ids_16, wts_16, s2d_16, gate_cap, up_cap)
            d8 = run_mxfp8_experts_sm70(mlp_in_8[0], w13_raw.view(torch.uint8), s13_raw, w2_raw.view(torch.uint8), s2_raw, ids_8, wts_8, s2d_8, gate_cap, up_cap)
        else:
            # Remap active experts compactly into [0..E_act-1] so we only prepack active experts!
            act_sm70 = sorted(set(ids_16.cpu().flatten().tolist()) | set(ids_8.cpu().flatten().tolist()))
            e_map = {e: idx for idx, e in enumerate(act_sm70)}
            e_lut = torch.full((n_experts,), -1, dtype=torch.int32, device=DEVICE)
            for e, idx in e_map.items():
                e_lut[e] = idx
            E_act = len(act_sm70)
            w13_q = torch.empty((E_act, w13_raw.shape[1], w13_raw.shape[2]), dtype=torch.uint8, device=DEVICE)
            s13_q = torch.empty((E_act, s13_raw.shape[1], s13_raw.shape[2]), dtype=torch.uint8, device=DEVICE)
            w2_q = torch.empty((E_act, w2_raw.shape[1], w2_raw.shape[2]), dtype=torch.uint8, device=DEVICE)
            s2_q = torch.empty((E_act, s2_raw.shape[1], s2_raw.shape[2]), dtype=torch.uint8, device=DEVICE)
            g13_q = torch.empty(E_act, dtype=torch.float32, device=DEVICE)
            g2_q = torch.empty(E_act, dtype=torch.float32, device=DEVICE)
            for idx, e in enumerate(act_sm70):
                qc, qs = qpn_prepack(w13_raw[e], s13_raw[e].view(torch.uint8), 16)
                w13_q[idx].view(-1).copy_(qc)
                s13_q[idx].view(-1).copy_(qs)
                qc, qs = qpn_prepack(w2_raw[e], s2_raw[e].view(torch.uint8), 16)
                w2_q[idx].view(-1).copy_(qc)
                s2_q[idx].view(-1).copy_(qs)
                g13_q[idx] = g13[e]
                g2_q[idx] = g2[e]
            d16 = run_nvfp4_experts_sm70(mlp_in_16[0], w13_q, s13_q, g13_q, w2_q, s2_q, g2_q, e_lut[ids_16.long()], wts_16, s2d_16, gate_cap, up_cap)
            d8 = run_nvfp4_experts_sm70(mlp_in_8[0], w13_q, s13_q, g13_q, w2_q, s2_q, g2_q, e_lut[ids_8.long()], wts_8, s2d_8, gate_cap, up_cap)
            del w13_q, s13_q, w2_q, s2_q

        moe_16 = combine_sm70(d16, s2d_16, topk, torch.float16)
        moe_8 = combine_sm70(d8, s2d_8, topk, torch.float16)
        del w13_raw, s13_raw, w2_raw, s2_raw

        # ---- MLP sconv + residual update ----
        res_ref = None if first else stack_ref[nb_ref]
        res_16 = None if first else stack_16[nb_16]
        res_8 = None if first else stack_8[nb_8]
        if first:
            nb_ref += 1
            nb_16 += 1
            nb_8 += 1
            stack_ref.append(torch.empty_like(attn_out_ref))

        mlp_sc_ref = ref_sconv_static(moe_ref, mlp_sw.float())
        stack_ref[nb_ref] = (attn_out_ref if res_ref is None else res_ref + attn_out_ref) + mlp_sc_ref
        short_conv(
            moe_16, mlp_sw.half(), conv_cache_16, positions, c_bt, seq_idx, c_slots, qstart,
            2 * D + ws_hid, ws_hid, addend=attn_out_16, residual=res_16, out=stack_16[nb_16],
        )
        short_conv(
            moe_8, mlp_sw.half(), conv_cache_8, positions, c_bt, seq_idx, c_slots, qstart,
            2 * D + ws_hid, ws_hid, addend=attn_out_8, residual=res_8, out=stack_8[nb_8],
        )
        stats["max_stack"] = max(stats["max_stack"], float(stack_ref[nb_ref].abs().max()))

        # Compute next inputs
        if visit + 1 == len(visits):
            nq, nn_w = output_query, out_norm_w.view(1, H)
            nn_eps = eps
        else:
            ni, npass = visits[visit + 1]
            nq = attn_queries[ni - 1]
            if npass:
                nn_w, nn_eps = None, 0.0
            else:
                npfx = f"model.llm.layers.{ni}"
                if ni in local_ids:
                    nn_w = f.get_tensor(f"{npfx}.attn_norm.weight").to(DEVICE).view(1, H)
                else:
                    nn_w = torch.stack(
                        [
                            f.get_tensor(f"{npfx}.q_input_norm.weight"),
                            f.get_tensor(f"{npfx}.k_input_norm.weight"),
                            f.get_tensor(f"{npfx}.v_input_norm.weight"),
                        ],
                        dim=0,
                    ).to(DEVICE)
                nn_eps = eps

        in_ref = ref_attn_res(stack_ref[:nb_ref], stack_ref[nb_ref], nq, None if nn_w is None else nn_w.float(), eps)
        in_16 = attn_res(stack_16[:nb_16], stack_16[nb_16], None, nq, None if nn_w is None else nn_w.half(), nn_eps, True)
        in_8 = attn_res(stack_8[:nb_8], stack_8[nb_8], None, nq, None if nn_w is None else nn_w.half(), nn_eps, True)

        if visit in (0, 5, 11, 17, 29, 41, 43, 47):
            cos16 = F.cosine_similarity(stack_16[nb_16].flatten(), stack_ref[nb_ref].flatten(), dim=0).item()
            cos8 = F.cosine_similarity(stack_8[nb_8].flatten(), stack_ref[nb_ref].flatten(), dim=0).item()
            print(
                f"  visit {visit:2d} (L{i:2d} p{pass_idx} {'MXFP8' if is_mxfp8 else 'NVFP4'} {'LOC' if is_local else 'GLB'}): "
                f"cos_fp16kv={cos16:.6f}  cos_fp8kv={cos8:.6f}  max|stack|={stack_ref[nb_ref].abs().max().item():.2f}",
                flush=True,
            )

    # ---- Final LM Head + Softcap(30.0) ----
    unembed = f.get_tensor("model.llm.unembed.weight").to(DEVICE)
    vocab = tcfg["unpadded_vocab_size"]
    softcap = tcfg["final_logit_softcapping"]

    def _logits(h, w, fp32_mm: bool):
        if fp32_mm:
            lg = F.linear(h.float(), w.float())[:, :vocab]
        else:
            lg = torch.mm(h.half(), w.half().t(), out_dtype=torch.float32)[:, :vocab]
        return torch.tanh(lg / softcap) * softcap

    lg_ref = _logits(in_ref[0], unembed, True)
    lg_16 = _logits(in_16[0], unembed, False)
    lg_8 = _logits(in_8[0], unembed, False)

    print(f"\n=== 3. Final Logits Comparison ({time.time() - t_start:.1f}s) ===", flush=True)
    for label, lg in [("sm70 FP16-KV", lg_16), ("sm70 FP8-KV ", lg_8)]:
        cos_all = F.cosine_similarity(lg.flatten(), lg_ref.flatten(), dim=0).item()
        cos_last = F.cosine_similarity(lg[-1], lg_ref[-1], dim=0).item()
        max_diff = (lg - lg_ref).abs().max().item()
        p_ref = F.softmax(lg_ref, dim=-1)
        kl = F.kl_div(F.log_softmax(lg, dim=-1), p_ref, reduction="batchmean").item()
        top1_match = float((lg.argmax(-1) == lg_ref.argmax(-1)).float().mean())
        top5_ref = lg_ref[-1].topk(5)
        top5_got = lg[-1].topk(5)
        print(
            f"  {label}: cos_all={cos_all:.6f}  cos_last={cos_last:.6f}  "
            f"max|dlogit|={max_diff:.4f}  KL={kl:.6f}  top1_match={top1_match*100:.1f}%",
            flush=True,
        )
        print(f"    last-tok top5 ref: {top5_ref.indices.tolist()} ({[round(x,2) for x in top5_ref.values.tolist()]})", flush=True)
        print(f"    last-tok top5 got: {top5_got.indices.tolist()} ({[round(x,2) for x in top5_got.values.tolist()]})", flush=True)
        if tok is not None:
            print(f"    next token ref={tok.decode([top5_ref.indices[0].item()])!r} got={tok.decode([top5_got.indices[0].item()])!r}", flush=True)

    print("\n=== 4. Activation Magnitude & FP16 Safety Audit ===", flush=True)
    for k, v in stats.items():
        print(f"  {k:24s} = {v:.4f}", flush=True)

    assert not torch.isnan(lg_16).any() and not torch.isnan(lg_8).any(), "NaN in logits!"
    assert F.cosine_similarity(lg_16.flatten(), lg_ref.flatten(), dim=0).item() > 0.999, "FP16-KV cosine too low!"
    assert F.cosine_similarity(lg_8.flatten(), lg_ref.flatten(), dim=0).item() > 0.998, "FP8-KV cosine too low!"
    print("\nFP32 ORACLE VERIFICATION PASSED!", flush=True)


if __name__ == "__main__":
    main()
