#!/usr/bin/env python3
"""Offline quantization tool for Q27 DFly2-KV drafter checkpoints (BF16 -> NVFP4+FP8 or FP8-all).

Produces a compact safetensors checkpoint + config.json matching Qwen3.8-27B-NVFP4's
quantization hierarchy for Volta sm_70 (4x V100-SXM2-16GB, TP4):
  - mode 'nvfp4_fp8' (default, ~266 MB/GPU TP4, matches Qwen3.8-27B-NVFP4):
      * layers.{0..2}.mlp.{gate_proj,up_proj,down_proj} -> NVFP4 (E2M1 packed uint8 [N, K/2] +
        E4M3FN block-16 scales [N, K/16] + shared FP32 global_scale for gate+up, compatible
        directly with q27fuse/csrc/mlp_fp4.cu::prepack).
      * All remaining 2D projections (backbone self_attn.{q,k,v,o}_proj, mtp_fusion.eh_proj,
        hidden_correction.{gate,up,down}_proj, hidden_correction.attention.{q_in,k_blk_proj,
        v_blk_proj,o_proj}) -> per-channel FP8 E4M3FN (compatible directly with
        torch.ops._C.fp8_qpn8_prepare_sm70 and q27fuse/csrc/fp8_lin.cu).
      * Norms, biases, mask_embedding, mtp_fusion.gate, block_gate -> FP16 (or FP32 for tiny
        scalars/biases as needed; stored in FP16 for lossless Volta loading).
  - mode 'fp8_all' (~354 MB/GPU TP4):
      * All 2D projections (including backbone MLPs) -> per-channel FP8 E4M3FN.
      * Norms, biases, gates -> FP16.

Usage:
  python3 quantize_dfly2.py \
      --src /data/models/Qwen3.8-27B-DFly2-Pilot \
      --dst /data/models/Qwen3.8-27B-DFly2-Pilot-NVFP4-FP8 \
      --mode nvfp4_fp8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# Exact round-to-nearest-even midpoints between adjacent E2M1 magnitudes
E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

BACKBONE_MLP_RE_PARTS = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")
FP16_KEEP_EXACT = {
    "hidden_correction.attention.block_gate.weight",  # tiny [24, 10240] (~0.49 MB total)
}


def is_backbone_mlp(name: str) -> bool:
    return name.startswith("layers.") and any(name.endswith(s) for s in BACKBONE_MLP_RE_PARTS)


def is_quantized_linear(name: str, t: torch.Tensor) -> bool:
    if t.ndim != 2 or not name.endswith(".weight"):
        return False
    if name in FP16_KEEP_EXACT:
        return False
    return t.shape[0] >= 256 and t.shape[1] >= 256


def quantize_fp8_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Per-output-channel symmetric FP8 E4M3FN quantization (matches Q27 compressed-tensors group_0)."""
    wf = w.float()
    amax = wf.abs().amax(dim=1)
    scale = (amax / 448.0).clamp_min(1e-12)
    q = (wf / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dq = q.float() * scale[:, None]
    err = (wf - dq).abs()
    rel = err.pow(2).mean().sqrt() / wf.pow(2).mean().sqrt().clamp_min(1e-12)
    stats = {"rmse_rel": float(rel), "max_abs": float(err.max())}
    return q.contiguous(), scale.to(torch.float32).contiguous(), stats


def quantize_nvfp4(w: torch.Tensor, global_scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Block-16 NVFP4 (E2M1 codes + E4M3FN block scales + FP32 global scale g) matching mlp_fp4.cu.

    Dequant formula: W_hat = E2M1(code) * E4M3(scale) / g.
    codes: [N, K/2] uint8 (low nibble = even k, high nibble = odd k).
    scales: [N, K/16] float8_e4m3fn.
    weight_global_scale: [1] float32 (= g).
    """
    assert w.ndim == 2 and w.shape[0] % 32 == 0 and w.shape[1] % 32 == 0, f"bad NVFP4 shape {tuple(w.shape)}"
    dev = w.device
    wf = w.float()
    N, K = wf.shape
    if global_scale is None:
        amax_g = float(wf.abs().max().item())
        global_scale = (448.0 * 6.0) / max(amax_g, 1e-12)
    g = float(global_scale)

    blk = wf.view(N, K // 16, 16)
    blk_amax = blk.abs().amax(dim=-1)  # [N, K/16]
    # Scale in E4M3FN: (blk_amax * g) / 6.0
    sc_fp8 = ((blk_amax * g) / 6.0).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
    # Effective real-valued block scale = sc_fp8 / g
    eff_sc = sc_fp8.float() / g
    safe_sc = torch.where(eff_sc > 0, eff_sc, torch.ones_like(eff_sc))
    normed = torch.where(eff_sc.unsqueeze(-1) > 0, blk / safe_sc.unsqueeze(-1), torch.zeros_like(blk))

    sign = (normed < 0).to(torch.uint8)
    mag = normed.abs()
    mids = E2M1_MID.to(dev)
    # Round-to-nearest-even for exact midpoints (even codes are 0, 2, 4, 6 -> indices 0,2,4,6)
    # bucketize(..., right=False): <= mid goes to lower index; exact ties on odd codes (0.75->2, 1.75->4, 3.5->6)
    # should round up to even code! Let's compute exact RNE cleanly:
    pos_vals = E2M1_POS.to(dev)  # [8]
    # Distances to all 8 grid points: [N, K/16, 16, 8]
    # Process in row chunks to keep peak memory tiny even on CPU/GPU
    codes_u4 = torch.empty((N, K), dtype=torch.uint8, device=dev)
    dq_all = torch.empty((N, K), dtype=torch.float32, device=dev)
    chunk_rows = 1024
    for r0 in range(0, N, chunk_rows):
        r1 = min(N, r0 + chunk_rows)
        m_c = mag[r0:r1].reshape(-1, 1)  # [R*K, 1]
        s_c = sign[r0:r1].reshape(-1)
        # Nearest grid index with ties to even:
        # idx = count(m > mid) except at exact ties where odd idx rounds up to even
        idx = torch.zeros_like(s_c)
        for j in range(7):
            mid = float(mids[j].item())
            if j % 2 == 0:
                # j even -> j+1 odd: tie stays at j (strict >)
                idx = idx + (m_c[:, 0] > mid).to(torch.uint8)
            else:
                # j odd -> j+1 even: tie rounds up to j+1 (>=)
                idx = idx + (m_c[:, 0] >= mid).to(torch.uint8)
        # Avoid -0.0 code (sign=1, idx=0) -> canonical +0.0
        s_clean = torch.where(idx == 0, torch.zeros_like(s_c), s_c)
        nib = (s_clean << 3) | idx
        codes_u4[r0:r1] = nib.view(r1 - r0, K)
        vals = pos_vals[idx.long()] * torch.where(s_clean.bool(), -1.0, 1.0)
        dq_all[r0:r1] = (vals.view(r1 - r0, K // 16, 16) * eff_sc[r0:r1].unsqueeze(-1)).view(r1 - r0, K)

    # Pack pairs: low nibble = even k, high nibble = odd k
    packed = (codes_u4[:, 0::2] | (codes_u4[:, 1::2] << 4)).contiguous()
    err = (wf - dq_all).abs()
    rel = err.pow(2).mean().sqrt() / wf.pow(2).mean().sqrt().clamp_min(1e-12)
    stats = {"rmse_rel": float(rel), "max_abs": float(err.max()), "global_scale": g}
    g_tensor = torch.tensor([g], dtype=torch.float32, device=dev)
    return packed, sc_fp8.contiguous(), g_tensor, stats


def quantize_checkpoint(src_dir: Path, dst_dir: Path, mode: str, device: str) -> None:
    t0 = time.perf_counter()
    src_st = src_dir / "model.safetensors"
    src_cfg = src_dir / "config.json"
    assert src_st.is_file(), f"Missing {src_st}"
    assert src_cfg.is_file(), f"Missing {src_cfg}"

    dst_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    sd = load_file(str(src_st), device="cpu")
    cfg = json.loads(src_cfg.read_text())

    # Precompute shared global_scale per backbone layer for (gate_proj, up_proj)
    gate_up_g: dict[int, float] = {}
    if mode == "nvfp4_fp8":
        for li in range(int(cfg["num_hidden_layers"])):
            g_w = sd[f"layers.{li}.mlp.gate_proj.weight"].float()
            u_w = sd[f"layers.{li}.mlp.up_proj.weight"].float()
            amax = max(float(g_w.abs().max().item()), float(u_w.abs().max().item()))
            gate_up_g[li] = (448.0 * 6.0) / max(amax, 1e-12)

    out_sd: dict[str, torch.Tensor] = {}
    q_report: dict[str, dict] = {}
    bytes_nvfp4 = bytes_fp8 = bytes_fp16 = 0

    for name in sorted(sd.keys()):
        t = sd[name]
        if mode == "nvfp4_fp8" and is_backbone_mlp(name):
            li = int(name.split(".")[1])
            g_override = gate_up_g[li] if (name.endswith("gate_proj.weight") or name.endswith("up_proj.weight")) else None
            packed, sc_fp8, g_t, stats = quantize_nvfp4(t.to(dev), global_scale=g_override)
            base = name[: -len(".weight")]
            out_sd[f"{base}.weight_packed"] = packed.cpu()
            out_sd[f"{base}.weight_scale"] = sc_fp8.cpu()
            out_sd[f"{base}.weight_global_scale"] = g_t.cpu()
            bytes_nvfp4 += packed.numel() + sc_fp8.numel() + 4
            q_report[name] = {"scheme": "nvfp4", "shape": list(t.shape), **stats}
        elif is_quantized_linear(name, t):
            q, sc, stats = quantize_fp8_per_channel(t.to(dev))
            base = name[: -len(".weight")]
            out_sd[f"{base}.weight"] = q.cpu()
            out_sd[f"{base}.weight_scale"] = sc.cpu()
            bytes_fp8 += q.numel() + sc.numel() * 4
            q_report[name] = {"scheme": "fp8_channel", "shape": list(t.shape), **stats}
        else:
            # Keep norms, biases, mask_embedding, gates in FP16
            t16 = t.to(torch.float16).contiguous()
            out_sd[name] = t16
            bytes_fp16 += t16.numel() * 2
            q_report[name] = {"scheme": "fp16", "shape": list(t.shape)}

    dst_st = dst_dir / "model.safetensors"
    save_file(out_sd, str(dst_st), metadata={"q27_dfly2_quant_mode": mode})

    out_cfg = dict(cfg)
    out_cfg["torch_dtype"] = "float16"
    out_cfg["q27_dfly2_quantization"] = {
        "mode": mode,
        "nvfp4_mlp_layers": [0, 1, 2] if mode == "nvfp4_fp8" else [],
        "fp8_channel_linears": True,
        "fp16_keep": sorted(FP16_KEEP_EXACT),
        "total_bytes": bytes_nvfp4 + bytes_fp8 + bytes_fp16,
        "tp4_bytes_per_gpu_est": (bytes_nvfp4 + bytes_fp8) / 4.0 + bytes_fp16,
    }
    (dst_dir / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
    (dst_dir / "quant_report.json").write_text(json.dumps(q_report, indent=2) + "\n")
    for extra in ("HANDOFF.md", "reference_model.py", "eval_pilot_report.json"):
        if (src_dir / extra).is_file():
            shutil.copy2(src_dir / extra, dst_dir / extra)

    dt = time.perf_counter() - t0
    tot_mb = (bytes_nvfp4 + bytes_fp8 + bytes_fp16) / 1e6
    tp4_mb = ((bytes_nvfp4 + bytes_fp8) / 4.0 + bytes_fp16) / 1e6
    nv_rels = [v["rmse_rel"] for v in q_report.values() if v["scheme"] == "nvfp4"]
    fp8_rels = [v["rmse_rel"] for v in q_report.values() if v["scheme"] == "fp8_channel"]
    print(f"[{mode}] Saved {dst_st} in {dt:.2f}s | total={tot_mb:.1f} MB | TP4={tp4_mb:.1f} MB/GPU")
    if nv_rels:
        print(f"  NVFP4 ({len(nv_rels)} tensors): rel RMSE mean={sum(nv_rels)/len(nv_rels):.4f}, max={max(nv_rels):.4f}")
    if fp8_rels:
        print(f"  FP8   ({len(fp8_rels)} tensors): rel RMSE mean={sum(fp8_rels)/len(fp8_rels):.4f}, max={max(fp8_rels):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--mode", default="nvfp4_fp8", choices=["nvfp4_fp8", "fp8_all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    quantize_checkpoint(args.src, args.dst, args.mode, args.device)


if __name__ == "__main__":
    main()
