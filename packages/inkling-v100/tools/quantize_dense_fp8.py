#!/usr/bin/env python3
"""Offline per-channel OCP FP8 E4M3FN quantizer for Inkling-2-Mini dense projections.

Reads the Inkling-2-Mini checkpoint read-only and writes quantized per-channel FP8
E4M3FN weights, FP32 scales, and a manifest JSON for `INKLING_FP8_DENSE_PATH`.

Included tiers (selectable at serve time via `INKLING_FP8_TIERS`, default `attn,lm_head`):
  - `attn`:        layers.{0..23}.attn.{wq_du,wr_du,gate_attn,attn_sink_proj,wk_dv,wv_dv,wo_ud}.weight
  - `lm_head`:     model.llm.unembed.weight
  - `controllers`: layers.{6..17}.controllers.{attn,k,v,mlp}.{down_weight,gain_weight,forget_weight}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
MIN_SCALE = 1e-12


def quantize_per_channel_fp8(
    w: torch.Tensor,
    mode: str = "absmax",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Quantize 2D weight [N, K] to per-row OCP FP8 E4M3FN + FP32 scale [N]."""
    assert w.dim() == 2, f"Expected 2D weight, got {tuple(w.shape)}"
    wf = w.to(device=device, dtype=torch.float32)
    absmax = wf.abs().amax(dim=1).clamp_min(MIN_SCALE)
    base_scale = absmax / FP8_MAX

    if mode == "absmax":
        best_scale = base_scale
    elif mode == "mse":
        alphas = torch.linspace(0.80, 1.00, 21, device=device, dtype=torch.float32)
        best_err = torch.full((wf.shape[0],), float("inf"), device=device, dtype=torch.float32)
        best_scale = base_scale.clone()
        chunk = 4096
        for r0 in range(0, wf.shape[0], chunk):
            r1 = min(r0 + chunk, wf.shape[0])
            wc = wf[r0:r1]
            bs = base_scale[r0:r1]
            be = best_err[r0:r1]
            bsc = best_scale[r0:r1]
            for alpha in alphas:
                sc = (bs * alpha).clamp_min(MIN_SCALE)
                q = (wc / sc[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                dq = q.float() * sc[:, None]
                err = (wc - dq).square().sum(dim=1)
                better = err < be
                be = torch.where(better, err, be)
                bsc = torch.where(better, sc, bsc)
            best_err[r0:r1] = be
            best_scale[r0:r1] = bsc
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    q_chunks = []
    rel_num = 0.0
    rel_den = 0.0
    max_abs_err = 0.0
    chunk = 8192
    for r0 in range(0, wf.shape[0], chunk):
        r1 = min(r0 + chunk, wf.shape[0])
        wc = wf[r0:r1]
        sc = best_scale[r0:r1]
        q = (wc / sc[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        dq = q.float() * sc[:, None]
        diff = wc - dq
        rel_num += float(diff.square().sum().item())
        rel_den += float(wc.square().sum().item())
        max_abs_err = max(max_abs_err, float(diff.abs().max().item()))
        q_chunks.append(q.cpu())

    w_q = torch.cat(q_chunks, dim=0).contiguous()
    s32 = best_scale.cpu().contiguous()
    s16 = (s32 * 256.0).half()
    assert torch.isfinite(s16).all(), "s16 overflow"
    nz = s32 > 1e-10
    s16_min = float(s16[nz].abs().min().item()) if nz.any() else 0.0
    s16_max = float(s16.abs().max().item())
    assert s16_min > 0.0 or not nz.any(), f"s16 underflow to zero: min s32={s32[nz].min()}"

    stats = {
        "rel_l2": (rel_num / max(rel_den, 1e-30)) ** 0.5,
        "max_abs": max_abs_err,
        "s16_min": s16_min,
        "s16_max": s16_max,
        "shape": list(w.shape),
    }
    return w_q, s32, stats


def tier_of_key(key: str) -> str | None:
    if not key.startswith("model.llm."):
        return None
    if key == "model.llm.unembed.weight":
        return "lm_head"
    if ".attn." in key and key.endswith(".weight"):
        leaf = key.rsplit(".", 2)[-2]
        if leaf in {"wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv", "wo_ud"}:
            return "attn"
    if ".controllers." in key and (
        key.endswith(".down_weight")
        or key.endswith(".gain_weight")
        or key.endswith(".forget_weight")
    ):
        return "controllers"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Path to Inkling-2-Mini checkpoint")
    ap.add_argument("--out-dir", default="./fp8_dense", help="Output directory")
    ap.add_argument("--mode", choices=["absmax", "mse"], default="absmax")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    by_file: dict[str, list[tuple[str, str]]] = {}
    for k, fname in weight_map.items():
        tier = tier_of_key(k)
        if tier is not None:
            by_file.setdefault(fname, []).append((k, tier))

    out_tensors: dict[str, torch.Tensor] = {}
    tier_stats: dict[str, list[tuple[str, dict[str, float]]]] = {
        "attn": [],
        "lm_head": [],
        "controllers": [],
    }
    orig_bytes = {"attn": 0, "lm_head": 0, "controllers": 0}
    fp8_bytes = {"attn": 0, "lm_head": 0, "controllers": 0}

    for fname, items in sorted(by_file.items()):
        items.sort()
        with safe_open(model_dir / fname, framework="pt", device="cpu") as sf:
            for key, tier in items:
                w = sf.get_tensor(key)
                w_q, s32, st = quantize_per_channel_fp8(w, mode=args.mode, device=args.device)
                out_tensors[key] = w_q
                out_tensors[f"{key}_scale"] = s32
                tier_stats[tier].append((key, st))
                orig_bytes[tier] += w.numel() * 2
                fp8_bytes[tier] += w_q.numel() + s32.numel() * 4

    out_path = out_dir / f"inkling2_fp8_dense_{args.mode}.safetensors"
    save_file(out_tensors, str(out_path))

    summary = {}
    print(f"\nOffline FP8 E4M3FN Quantization Complete ({args.mode}, {time.time() - t0:.1f}s):")
    for tier in ("attn", "lm_head", "controllers"):
        items = tier_stats[tier]
        rels = [st["rel_l2"] for _, st in items]
        s16_lo = min(st["s16_min"] for _, st in items)
        s16_hi = max(st["s16_max"] for _, st in items)
        summary[tier] = {
            "num_tensors": len(items),
            "orig_mb": round(orig_bytes[tier] / 1e6, 2),
            "fp8_mb": round(fp8_bytes[tier] / 1e6, 2),
            "mean_rel_l2": sum(rels) / len(rels),
            "max_rel_l2": max(rels),
            "s16_range": [s16_lo, s16_hi],
            "keys": [k for k, _ in items],
        }
        print(
            f"  [{tier:11s}] {len(items):3d} tensors | "
            f"{orig_bytes[tier]/1e6:6.1f} -> {fp8_bytes[tier]/1e6:6.1f} MB | "
            f"rel_L2 mean={summary[tier]['mean_rel_l2']:.5f} max={summary[tier]['max_rel_l2']:.5f} | "
            f"s16=[{s16_lo:.4g}, {s16_hi:.4g}]"
        )

    manifest = {
        "source_model": str(model_dir),
        "format": "fp8_e4m3fn_per_channel",
        "calibration_mode": args.mode,
        "safetensors_file": out_path.name,
        "tiers": summary,
    }
    manifest_path = out_dir / f"manifest_{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB) + {manifest_path}")


if __name__ == "__main__":
    main()
