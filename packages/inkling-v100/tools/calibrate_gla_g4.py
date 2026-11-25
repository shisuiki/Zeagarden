#!/usr/bin/env python3
"""Offline Q/O-weighted Post-Prologue GLA-g4 (r=256, FP8 latent) calibrator & 4-combo FP32 oracle validator.

Reads an Inkling-2-Mini checkpoint read-only and uses a strictly disjoint ~3,700-token diverse
calibration corpus (zero overlap with eval prompts).

Produces `<out>.safetensors` (default `./inkling2_gla_g4_r256.safetensors`) containing:
  - Per-visit FP16 projectors for all 8 global visits:
      l5_p0, l11_p0, l11_p1, l11_p2, l17_p0, l17_p1, l17_p2, l23_p0
  - Shared ELR FP16 projectors for combined (ELR Global KV + GLA-g4) mode:
      l11_elr, l17_elr   (fitted on Pass-2 KV weighted by combined Passes 0+1+2 Q & W_O metrics)

Each key prefix `<k>` stores three FP16 tensors across TP4 ranks (g in 0..3):
  - `<k>.w_down`: [4, 512, 256] fp16   (pre-divided by static FP8 latent scale `s_lat`)
  - `<k>.u_k_t`:  [4, 2, 128, 256] fp16 (pre-multiplied by `s_lat`; q_lat[sub] = q[sub] @ u_k_t[g, sub])
  - `<k>.u_v`:    [4, 2, 256, 128] fp16 (pre-multiplied by `s_lat`; v_out[sub] = o_lat[sub] @ u_v[g, sub])

Then validates all 4 combinations using the ACTUAL SAVED FP16 tensors + FP8 latent against the
exact 48-visit FP32 oracle:
  1. (ELR=0, GLA=0) Phase 3 Baseline
  2. (ELR=1, GLA=0) ELR Last-Loop Global KV (L11, L17)
  3. (ELR=0, GLA=1) Post-Prologue GLA-g4 r=256 (FP16 folded + FP8 latent)
  4. (ELR=1, GLA=1) Combined ELR Global KV + Post-Prologue GLA-g4 r=256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from eval_fp8_oracle import DEVICE, PROMPTS, WeightStore
from eval_lossy_oracle import CALIB_TEXTS as BASE_CALIB, eval_metrics, run_inkling_variant

# Expanded strictly disjoint calibration corpus (~3,500+ tokens across code, math, systems, configs, science, prose)
EXTRA_CALIB_TEXTS = [
    # Rust systems programming
    (
        "Write a thread-safe LRU cache with sharded RwLocks in Rust that supports O(1) lookup, "
        "insertion, and eviction while avoiding lock convoying under skewed Zipfian workloads:\n\n"
        "```rust\nuse std::collections::HashMap;\nuse std::hash::{Hash, Hasher};\n"
        "use std::collections::hash_map::DefaultHasher;\nuse std::sync::RwLock;\n\n"
        "struct Entry<K, V> { key: K, val: V, prev: Option<usize>, next: Option<usize> }\n"
        "struct Shard<K, V> {\n    map: HashMap<K, usize>,\n    arena: Vec<Entry<K, V>>,\n"
        "    head: Option<usize>,\n    tail: Option<usize>,\n    cap: usize,\n}\n"
        "impl<K: Eq + Hash + Clone, V> Shard<K, V> {\n"
        "    fn touch(&mut self, idx: usize) {\n"
        "        if self.head == Some(idx) { return; }\n"
        "        let (p, n) = (self.arena[idx].prev, self.arena[idx].next);\n"
        "        if let Some(pi) = p { self.arena[pi].next = n; }\n"
        "        if let Some(ni) = n { self.arena[ni].prev = p; }\n"
        "        if self.tail == Some(idx) { self.tail = p; }\n"
        "        self.arena[idx].prev = None;\n"
        "        self.arena[idx].next = self.head;\n"
        "        if let Some(h) = self.head { self.arena[h].prev = Some(idx); }\n"
        "        self.head = Some(idx);\n"
        "        if self.tail.is_none() { self.tail = Some(idx); }\n"
        "    }\n}\n```\n"
    ),
    # SQL & query planning
    (
        "Consider a distributed analytical query over a 20-terabyte clickstream table partitioned "
        "by event_date and clustered by user_id:\n\n```sql\nWITH session_bounds AS (\n"
        "  SELECT user_id, event_ts, page_url,\n"
        "         LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts\n"
        "  FROM analytics.page_events\n"
        "  WHERE event_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-31'\n"
        "), tagged AS (\n"
        "  SELECT user_id, event_ts, page_url,\n"
        "         SUM(CASE WHEN prev_ts IS NULL OR EXTRACT(EPOCH FROM (event_ts - prev_ts)) > 1800\n"
        "                  THEN 1 ELSE 0 END)\n"
        "           OVER (PARTITION BY user_id ORDER BY event_ts ROWS UNBOUNDED PRECEDING) AS sid\n"
        "  FROM session_bounds\n)\n"
        "SELECT COUNT(DISTINCT (user_id, sid)) AS total_sessions,\n"
        "       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cnt) AS p95_pages\n"
        "FROM (SELECT user_id, sid, COUNT(*) AS cnt FROM tagged GROUP BY 1, 2) s;\n```\n"
        "Because the partition key aligns with the window PARTITION BY user_id clause, the optimizer "
        "can execute a partition-local sort-merge without a global shuffle, reducing network I/O by 8x.\n"
    ),
    # Probability & information theory
    (
        "Theorem (Sanov's Theorem for Finite Alphabets): Let X_1, X_2, ..., X_n be i.i.d. draws from "
        "a discrete probability distribution Q supported on a finite alphabet A. Let E be a subset of "
        "the probability simplex P(A) whose closure equals the closure of its interior. Denote by "
        "P_n the empirical type of the sample. Then\n"
        "  lim_{n -> inf} (1/n) log P(P_n in E) = -inf_{P in E} D_KL(P || Q),\n"
        "where D_KL(P || Q) = sum_{a in A} P(a) log(P(a) / Q(a)) is the Kullback-Leibler divergence.\n"
        "Proof sketch: The number of distinct types of length n is bounded polynomially by (n + 1)^{|A|}, "
        "whereas the probability of observing any specific type P decays exponentially as "
        "2^{-n D_KL(P || Q)} up to polynomial factors. Summing over all types in E, the exponential "
        "rate is governed solely by the I-projection P* = argmin_{P in E} D_KL(P || Q).\n"
    ),
    # Distributed consensus & Raft
    (
        "In the Raft consensus protocol, safety relies on two core invariants: the Election Restriction "
        "and the Leader Completeness Property. During RequestVote RPCs, a candidate includes its "
        "lastLogIndex and lastLogTerm; voters reject any candidate whose log is less up-to-date than "
        "their own (comparing last term first, then length). Consequently, any newly elected leader is "
        "guaranteed to hold every entry committed in prior terms. However, a leader must never commit "
        "log entries from previous terms merely by counting replicas; it may only commit entries from "
        "its current term by counting a majority, after which prior entries are committed indirectly "
        "via the Log Matching Property.\n"
    ),
    # Solid-state physics & semiconductors
    (
        "In a strained silicon MOSFET channel, biaxial tensile strain lifts the six-fold degeneracy of "
        "the conduction band Delta valleys, lowering the two out-of-plane Delta_2 valleys relative to "
        "the four in-plane Delta_4 valleys. This splitting suppresses intervalley phonon scattering "
        "and increases electron occupancy in valleys with smaller in-plane transport effective mass "
        "m_t = 0.19 m_0, yielding a substantial mobility enhancement at room temperature. At high "
        "vertical electric fields, surface roughness scattering at the Si/SiO2 interface dominates, "
        "following an empirical E_eff^{-2} roll-off that dictates gate-oxide reliability trade-offs.\n"
    ),
    # Compiler optimization & register allocation
    (
        "Linear-scan register allocation trades a modest loss in spill code quality for O(N log R) "
        "compile time compared to Chaitin-Briggs graph coloring. By sorting live intervals in order "
        "of increasing start point in SSA form, the allocator maintains an active set ordered by end "
        "point. When no physical register is free, it compares the current interval's end point "
        "against the longest-lived active interval, spilling whichever extends further into the "
        "future. Modern JITs extend this with live-range splitting around loop headers and cold "
        "exception paths so hot inner loops keep induction variables and accumulators strictly in "
        "registers without memory traffic.\n"
    ),
    # Kubernetes / YAML infrastructure config
    (
        "apiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: vector-index-shard\n"
        "  namespace: search-prod\nspec:\n  serviceName: vector-shard\n  replicas: 6\n"
        "  podManagementPolicy: Parallel\n  template:\n    spec:\n      topologySpreadConstraints:\n"
        "        - maxSkew: 1\n          topologyKey: topology.kubernetes.io/zone\n"
        "          whenUnsatisfiable: DoNotSchedule\n      containers:\n"
        "        - name: indexd\n          image: registry.internal/indexd:v3.14.2\n"
        "          resources:\n            requests: { cpu: '8', memory: '32Gi' }\n"
        "            limits: { cpu: '16', memory: '48Gi' }\n"
        "          readinessProbe:\n            httpGet: { path: /readyz, port: 8080 }\n"
        "            initialDelaySeconds: 15\n            periodSeconds: 5\n"
    ),
    # Molecular biology & CRISPR
    (
        "Prime editing couples a catalytically impaired Cas9 H840A nickase to an engineered reverse "
        "transcriptase guided by a prime editing guide RNA (pegRNA). Unlike canonical double-strand "
        "break repair, which triggers error-prone non-homologous end joining and unwanted indels, "
        "prime editing nicks only the non-target DNA strand. The 3' extension of the pegRNA contains "
        "a primer binding site that hybridizes to the nicked 3' flap, enabling direct reverse "
        "transcription of the encoded edit into the genomic locus. Subsequent 5' flap equilibration "
        "and cellular mismatch repair resolve the heteroduplex into a permanent edit.\n"
    ),
    # Python numerical / FFT implementation
    (
        "```python\nimport cmath\n\ndef fft_radix2(x: list[complex]) -> list[complex]:\n"
        "    n = len(x)\n    if n <= 1:\n        return list(x)\n"
        "    assert (n & (n - 1)) == 0, 'length must be power of two'\n"
        "    even = fft_radix2(x[0::2])\n    odd = fft_radix2(x[1::2])\n"
        "    out = [0j] * n\n    for k in range(n // 2):\n"
        "        twiddle = cmath.exp(-2j * cmath.pi * k / n) * odd[k]\n"
        "        out[k] = even[k] + twiddle\n        out[k + n // 2] = even[k] - twiddle\n"
        "    return out\n\n"
        "def convolve_exact(a: list[float], b: list[float]) -> list[float]:\n"
        "    size = 1 << (len(a) + len(b) - 2).bit_length()\n"
        "    fa = fft_radix2([complex(v) for v in a] + [0j] * (size - len(a)))\n"
        "    fb = fft_radix2([complex(v) for v in b] + [0j] * (size - len(b)))\n"
        "    prod = [u * v for u, v in zip(fa, fb)]\n"
        "    inv = fft_radix2([z.conjugate() for z in prod])\n"
        "    return [(z.conjugate().real / size) for z in inv[: len(a) + len(b) - 1]]\n```\n"
    ),
    # Atmospheric dynamics & climate physics
    (
        "Quasi-geostrophic potential vorticity conservation governs synoptic-scale midlatitude "
        "weather systems where the Rossby number Ro = U / (f L) is small compared to unity. "
        "Baroclinic instability arises when meridional temperature gradients, coupled through "
        "thermal wind balance to vertical wind shear, tilt developing troughs westward with height. "
        "This vertical phase tilt allows warm air to rise poleward and cold air to sink equatorward, "
        "converting available potential energy into eddy kinetic energy that drives extratropical "
        "cyclones and maintains the eddy-driven polar jet stream against surface friction.\n"
    ),
    # Longer multi-paragraph technical narrative (tests long-range global attention patterns!)
    (
        "Design Review: Low-Latency Storage Engine for Financial Order Books\n\n"
        "1. Requirements & Latency Envelope\n"
        "Our matching engine processes peak bursts of 1.2 million limit-order messages per second "
        "across 4,000 equity symbols with a strict 99.99th-percentile tick-to-trade budget of 35 "
        "microseconds. Kernel network stacks and garbage-collected runtimes introduce unpredictable "
        "millisecond tail spikes; therefore, all hot-path components run pinned to isolated physical "
        "cores with kernel bypass (DPDK/Solarflare ef_vi), hugepage-backed arena allocators, and "
        "pre-warmed instruction caches.\n\n"
        "2. Price-Time Priority Data Structures\n"
        "Each symbol maintains bid and ask books indexed by fixed-point integer ticks (1e-4 USD). "
        "Instead of balanced binary search trees whose pointer chasing stalls on L3 misses, active "
        "price levels within +-2048 ticks of the National Best Bid and Offer reside in a flat ring "
        "array paired with a two-level 64-bit occupancy bitmask. Finding the new best price after a "
        "sweep reduces to a single hardware `tzcnt` / `lzcnt` instruction (3 cycles), while each "
        "price level holds an intrusive doubly linked list of slab-allocated order structs so "
        "cancellation by order ID completes in O(1) via an open-addressed Robin Hood hash table.\n\n"
        "3. Deterministic Journaling & Failover\n"
        "Before acknowledging any fill, the sequenced input message is appended to an RDMA-replicated "
        "ring log on two hot-standby replicas. Because the matching state machine is strictly "
        "deterministic with zero wall-clock dependencies, standby nodes replay the exact event stream "
        "in lockstep and can promote to primary within 12 milliseconds upon heartbeat loss.\n"
    ),
]

ALL_CALIB_TEXTS = BASE_CALIB + EXTRA_CALIB_TEXTS


def fit_single_group_qo(
    X: torch.Tensor,
    cq_pair: list[torch.Tensor],
    mv_pair: tuple[torch.Tensor, torch.Tensor],
    rank: int,
    reg: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Fit Q/O-weighted SVD for one TP rank (512 -> rank -> 512) and fold static FP8 scale."""
    hd = 128
    blocks_m, blocks_inv = [], []
    for sub in range(2):
        cq = 0.5 * (cq_pair[sub] + cq_pair[sub].t())
        tr_c = float(torch.trace(cq).item()) / hd
        cq = cq / max(tr_c, 1e-12) + reg * torch.eye(hd, dtype=torch.float64)
        L = torch.linalg.cholesky(cq)
        blocks_m.append(L.t())
        blocks_inv.append(torch.linalg.inv(L.t()))
    for sub in range(2):
        mv = 0.5 * (mv_pair[sub] + mv_pair[sub].t())
        tr_m = float(torch.trace(mv).item()) / hd
        mv = mv / max(tr_m, 1e-12) + reg * torch.eye(hd, dtype=torch.float64)
        L = torch.linalg.cholesky(mv)
        blocks_m.append(L.t())
        blocks_inv.append(torch.linalg.inv(L.t()))

    M = torch.block_diag(*blocks_m)
    Minv = torch.block_diag(*blocks_inv)
    Xm = X @ M.t()
    _, S, Vh = torch.linalg.svd(Xm, full_matrices=False)
    Vr = Vh[:rank]
    tail_e = float((S[rank:].square().sum() / S.square().sum()).item())

    w_down_f64 = M.t() @ Vr.t()      # [512, r]
    w_up_f64 = Vr @ Minv.t()         # [r, 512]

    # Calibrate static FP8 latent scale with generous headroom (absmax <= 300 inside +-448)
    lat_cal = (X @ w_down_f64).abs().amax().item()
    s_lat = max(lat_cal / 300.0, 1e-4)

    w_down_s = (w_down_f64 / s_lat).to(torch.float16)  # [512, r]
    w_up_s = (w_up_f64 * s_lat).to(torch.float16)      # [r, 512]

    # Split w_up_s [r, 512] -> U_K^T [2, 128, r] and U_V [2, r, 128]
    u_k_t = torch.stack(
        [w_up_s[:, 0:hd].t().contiguous(), w_up_s[:, hd : 2 * hd].t().contiguous()], dim=0
    )
    u_v = torch.stack(
        [w_up_s[:, 2 * hd : 3 * hd].contiguous(), w_up_s[:, 3 * hd : 4 * hd].contiguous()], dim=0
    )
    max_w = max(float(w_down_s.abs().max()), float(u_k_t.abs().max()), float(u_v.abs().max()))
    return w_down_s, u_k_t, u_v, tail_e, max_w


def build_gla_projs_from_safetensors(
    st_dict: dict[str, torch.Tensor], use_elr: bool
) -> dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Convert saved FP16 folded tensors back into `run_inkling_variant` gla_projs format.

    Note: `w_down` is already pre-divided by `s_lat` and `u_k_t`/`u_v` are pre-multiplied by `s_lat`,
    so `lat_scale = 1.0` tests the EXACT runtime FP16 folded arithmetic + unit-scale FP8 E4M3FN!
    """
    hd = 128
    one = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    projs: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    visits = [(5, 0), (11, 0), (11, 1), (11, 2), (17, 0), (17, 1), (17, 2), (23, 0)]
    for layer_id, pass_idx in visits:
        if use_elr and layer_id in (11, 17):
            key = f"l{layer_id}_elr"
        else:
            key = f"l{layer_id}_p{pass_idx}"
        wd = st_dict[f"{key}.w_down"].to(DEVICE)   # [4, 512, 256] fp16
        ukt = st_dict[f"{key}.u_k_t"].to(DEVICE)   # [4, 2, 128, 256] fp16
        uv = st_dict[f"{key}.u_v"].to(DEVICE)      # [4, 2, 256, 128] fp16
        g_list = []
        for g in range(4):
            w_up = torch.cat(
                [ukt[g, 0].t(), ukt[g, 1].t(), uv[g, 0], uv[g, 1]], dim=-1
            ).float()  # [256, 512] (exact FP16 values!)
            g_list.append((wd[g].float(), w_up, one))
        projs[(layer_id, pass_idx)] = g_list
    return projs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Path to Inkling-2-Mini checkpoint")
    ap.add_argument("--out", default="./inkling2_gla_g4_r256.safetensors", help="Output GLA safetensors path")
    ap.add_argument("--fp8-path", default="", help="Optional offline FP8 dense safetensors path")
    ap.add_argument("--eval-out", default="", help="Optional JSON path to save 4-combo FP32 oracle metrics")
    ap.add_argument("--skip-eval", action="store_true", help="Skip 4-combo FP32 oracle validation after fitting")
    args = ap.parse_args()

    t0 = time.time()
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    eval_ids = [torch.tensor(tok.encode(p), dtype=torch.long, device=DEVICE) for _, p in PROMPTS]
    calib_ids = [torch.tensor(tok.encode(p), dtype=torch.long, device=DEVICE) for p in ALL_CALIB_TEXTS]
    print(
        f"Eval tokens: {sum(x.numel() for x in eval_ids)} across {len(eval_ids)} prompts | "
        f"Disjoint calibration tokens: {sum(x.numel() for x in calib_ids)} across {len(calib_ids)} texts"
    )

    ws_f32 = WeightStore(None, set(), torch.float32, model_dir=model_dir)
    fp8_p = Path(args.fp8_path) if args.fp8_path else None
    ws_p3 = WeightStore(fp8_p, {"attn", "lm_head"} if fp8_p else set(), torch.float16, model_dir=model_dir)

    print("\n[1/5] Collecting post-prologue global KV + Q/O stats on disjoint calibration corpus...")
    gla_stats: dict = {}
    run_inkling_variant(
        calib_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True, record_global_gla_stats=gla_stats
    )

    print("[2/5] Fitting Q/O-weighted GLA-g4 (r=256) per-visit + shared ELR FP16 folded projectors...")
    rank = 256
    reg = 1e-4
    tensors: dict[str, torch.Tensor] = {}
    max_fp16_abs = 0.0
    tails_std, tails_elr = [], []

    # 8 standard per-visit projectors
    for (layer_id, pass_idx), rec in sorted(gla_stats.items()):
        key = f"l{layer_id}_p{pass_idx}"
        wds, ukts, uvs = [], [], []
        for g in range(4):
            X = torch.cat(rec["kv_list"][g], dim=0)
            wd, ukt, uv, tail_e, mw = fit_single_group_qo(X, rec["cq_sum"][g], rec["mv"][g], rank, reg=reg)
            wds.append(wd)
            ukts.append(ukt)
            uvs.append(uv)
            tails_std.append(tail_e)
            max_fp16_abs = max(max_fp16_abs, mw)
        tensors[f"{key}.w_down"] = torch.stack(wds, dim=0).contiguous()
        tensors[f"{key}.u_k_t"] = torch.stack(ukts, dim=0).contiguous()
        tensors[f"{key}.u_v"] = torch.stack(uvs, dim=0).contiguous()

    # 2 shared ELR projectors (L11 & L17): Pass-2 KV distribution weighted by combined Passes 0+1+2 C_Q & M_V!
    for layer_id in (11, 17):
        key = f"l{layer_id}_elr"
        rec_p2 = gla_stats[(layer_id, 2)]
        wds, ukts, uvs = [], [], []
        for g in range(4):
            X_p2 = torch.cat(rec_p2["kv_list"][g], dim=0)
            cq_comb = [
                sum(gla_stats[(layer_id, p)]["cq_sum"][g][sub] for p in (0, 1, 2))
                for sub in range(2)
            ]
            # W_O is identical across passes for a given physical layer
            wd, ukt, uv, tail_e, mw = fit_single_group_qo(X_p2, cq_comb, rec_p2["mv"][g], rank, reg=reg)
            wds.append(wd)
            ukts.append(ukt)
            uvs.append(uv)
            tails_elr.append(tail_e)
            max_fp16_abs = max(max_fp16_abs, mw)
        tensors[f"{key}.w_down"] = torch.stack(wds, dim=0).contiguous()
        tensors[f"{key}.u_k_t"] = torch.stack(ukts, dim=0).contiguous()
        tensors[f"{key}.u_v"] = torch.stack(uvs, dim=0).contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    size_mb = out_path.stat().st_size / (1024**2)
    print(
        f"  Saved {out_path} ({size_mb:.2f} MiB, {len(tensors)} FP16 tensors) | "
        f"mean_tail_std={100*sum(tails_std)/len(tails_std):.2f}% | "
        f"mean_tail_elr={100*sum(tails_elr)/len(tails_elr):.2f}% | "
        f"max_|W_fp16|={max_fp16_abs:.2f}"
    )
    assert max_fp16_abs < 1000.0, f"Unexpectedly large FP16 projector weight: {max_fp16_abs}"
    if args.skip_eval:
        print(f"Done in {time.time() - t0:.1f}s (skipped oracle validation).")
        return

    print("\n[3/5] Running Exact FP32 Oracle & Baseline (ELR=0, GLA=0)...")
    logits_f32, _ = run_inkling_variant(eval_ids, cfg, ws_f32, fp8_kv=False, fp16_compute=False)
    logits_p3, pass2_kv_0 = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True, record_pass2_kv=True
    )
    m_p3 = eval_metrics(logits_f32, logits_p3, eval_ids)
    print(f"  (ELR=0, GLA=0) Baseline     | vs FP32: {m_p3}")

    saved_st = load_file(str(out_path))
    projs_std = build_gla_projs_from_safetensors(saved_st, use_elr=False)
    projs_elr = build_gla_projs_from_safetensors(saved_st, use_elr=True)

    print("\n[4/5] Evaluating all 3 lossy flag combinations using ACTUAL saved FP16 tensors + FP8 latent...")
    results = {
        "calib_tokens": int(sum(x.numel() for x in calib_ids)),
        "mean_tail_std_pct": round(100.0 * sum(tails_std) / len(tails_std), 3),
        "mean_tail_elr_pct": round(100.0 * sum(tails_elr) / len(tails_elr), 3),
        "max_fp16_abs": round(max_fp16_abs, 3),
        "elr0_gla0": m_p3,
    }

    # (ELR=1, GLA=0)
    global_core = {11, 17}
    _, p2_elr1 = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        last_loop_kv_cache=pass2_kv_0, last_loop_kv_layers=global_core, record_pass2_kv=True,
    )
    logits_elr, _ = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        last_loop_kv_cache=p2_elr1, last_loop_kv_layers=global_core,
    )
    m_elr = eval_metrics(logits_f32, logits_elr, eval_ids)
    m_elr_p3 = eval_metrics(logits_p3, logits_elr, eval_ids)
    results["elr1_gla0"] = {"vs_fp32": m_elr, "vs_p3": m_elr_p3}
    print(f"  (ELR=1, GLA=0) ELR Global   | vs FP32: {m_elr} | vs P3 top1={m_elr_p3['top1_pct']}% KL={m_elr_p3['kl_mean']}")

    # (ELR=0, GLA=1)
    logits_gla, _ = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        gla_projs=projs_std, gla_fp8_latent=True,
    )
    m_gla = eval_metrics(logits_f32, logits_gla, eval_ids)
    m_gla_p3 = eval_metrics(logits_p3, logits_gla, eval_ids)
    results["elr0_gla1"] = {"vs_fp32": m_gla, "vs_p3": m_gla_p3}
    print(f"  (ELR=0, GLA=1) GLA-g4 r=256 | vs FP32: {m_gla} | vs P3 top1={m_gla_p3['top1_pct']}% KL={m_gla_p3['kl_mean']}")

    # (ELR=1, GLA=1) Combined! (2 fixed-point passes using shared ELR basis on L11 & L17)
    _, p2_combo0 = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        gla_projs=projs_elr, gla_fp8_latent=True, record_pass2_kv=True,
    )
    _, p2_combo1 = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        gla_projs=projs_elr, gla_fp8_latent=True,
        last_loop_kv_cache=p2_combo0, last_loop_kv_layers=global_core, record_pass2_kv=True,
    )
    logits_combo, _ = run_inkling_variant(
        eval_ids, cfg, ws_p3, fp8_kv=True, fp16_compute=True,
        gla_projs=projs_elr, gla_fp8_latent=True,
        last_loop_kv_cache=p2_combo1, last_loop_kv_layers=global_core,
    )
    m_combo = eval_metrics(logits_f32, logits_combo, eval_ids)
    m_combo_p3 = eval_metrics(logits_p3, logits_combo, eval_ids)
    results["elr1_gla1"] = {"vs_fp32": m_combo, "vs_p3": m_combo_p3}
    print(f"  (ELR=1, GLA=1) COMBINED     | vs FP32: {m_combo} | vs P3 top1={m_combo_p3['top1_pct']}% KL={m_combo_p3['kl_mean']}")

    if args.eval_out:
        eval_out = Path(args.eval_out)
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        eval_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n[5/5] Done in {time.time() - t0:.1f}s! Saved {eval_out}")
    else:
        print(f"\n[5/5] Done in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
