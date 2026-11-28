# Inkling-2-Mini on 4× Tesla V100-SXM2 (`inkling_v100`)

`packages/inkling-v100` is a `vllm.general_plugins` package for serving **`thinkingmachines/Inkling-2-Mini`** (`24` physical / `48` logical looped layers, 128-expert top-8 MoE with native NVFP4 `L0–19` and MXFP8 `L20–23`, 20 local SWA-1024 + 4 global sheared relative-bias attention layers, `W=4` short convolutions, FP32 AttnRes, and native **DSpark** or zero-KV **DFly2-KV** 7-token drafters) on 4× Tesla V100-SXM2-16GB (`TP=4/EP=4`, `sm_70`) running unmodified `1Cat-vLLM v1.5.4`.

## Overview

- **13 Zero-Spill `sm_70` CUDA Extensions**:
  - `ep4_moe_fused.cu` (1-launch persistent 80-CTA × 1024-thread `M≤8` EP4 NVFP4 MoE, 64 regs / 0 spills / occupancy 1, `1.7×–5.6×` faster than 4-launch), `nvfp4_prefill_moe.cu` (`BM=8/16/32/64` for `M>8`, `chunk_cap=4096`, exact native NVFP4 with fused softcapped SiTU), `mlp_mxfp8_moe.cu` (exact in-place E8M0 MXFP8 MoE).
  - `fp8_dense.cu` (offline per-channel `absmax` E4M3FN QPN8 dense attention projections + FP32-output `lm_head`, zero-copy Pass-0/Pass-1–2 slices, `KLD=0.0038` vs 48-visit FP32 oracle).
  - `attn_rel_pf_wmma.cu` (3-head-shared WMMA prefill with on-the-fly Tensor-Core `R @ rel_proj`, eliminating `rel_logits` VRAM and accelerating short/local prefill `1.34×–1.47×`), `attn_rel_pf.cu` (`2×8` `HMMA.884` sheared relative-bias prefill fallback), `attn_rel_dec2.cu` (short-SASS 1-launch fused `kv_conv_prologue + decode_attention`, `3.5×–5.7×` faster on local `C=1`, active `C≤8` local / `C≤1` global), `attn_rel_dec.cu` (high-`C` split-KV FP8/FP16 decode/verify, **94% HBM2 peak**), `attn_rel_gla.cu` (opt-in `GLA-g4 r=256` decode/verify/prefill).
  - `ep4_route.cu` (fused top-8 gate + EP4 grouping for `M>8`), `loop_blend.cu` (1-launch cooperative blend for `S*T≤8` + zero-copy strided `chunk=2048` multi-token for `S*T>8`), `all_reduce_conv_mix.cu` (1-shot NVLink P2P AllReduce with L2 prefetch, conditional rehearsal, direct post-MoE `[M, H]` path + `W=4` conv + FP32 AttnRes + RMSNorm), and `dfly2_ops.cu` (all zero-spill Stage A/B/C + shared target FP16/FP8/GLA-g4 context-KV kernels for `DFly2-KV`).
  - **Long-Prefill Fast Path (`-70 ms` per 4K chunk, bit-identical)**: 2-pass FP32 `attn_res` (`-38.3 ms`), zero-copy strided `chunk=2048` `loop_blend` (`-15.8 ms`), and `chunk_cap=4096` MoE (`-15.9 ms`) push cold prefill to **`9,913 / 10,029 / 9,917 / 9,458 tok/s`** at `1×4K / 1×8K / 1×16K / 1×32K` — **faster than upstream `1Cat-vLLM` across all 8 regimes (`1×128 → 1×32K`)** while keeping full FP32 AttnRes precision.
- **Lossless `6.6×–16.0×` KV Capacity + Prefix Caching (`RingConv` + `RingAPC`)**:
  - Under `INKLING_RING_CONV=auto` (default), replaces 6 paged conv KV groups with a fixed 16-slot per-request FP16 ring buffer (`247.5 MiB/GPU`), yielding **`323.0K` (DSpark7) / `379.4K` (`392.8K` w/o APC, DFly2-KV) Exact FP16 KV tokens (`6.6×–8.0×` upstream `49.2K`)** and **`582.5K` (DSpark7) / `785.6K` (DFly2-KV) near-lossless FP8 KV tokens (`11.8×–16.0×` upstream)** at `util=0.88`.
  - **RingAPC** automatically activates when `--enable-prefix-caching` is enabled with `RingConv`: captures bit-exact 3-tap conv checkpoints every `INKLING_RING_APC_STRIDE` tokens (default `512`) into a lazy-allocated pinned host-RAM LRU pool (`1.41 MiB/ckpt`, `0 B` extra GPU KV groups), aligns EngineCore prefix-cache hits & SWA replay windows (`INKLING_RING_APC_RETENTION`, default `4096`) to stride boundaries (with speculative lookahead awareness), delivering **`0.0` bit-exact diff**, **95–99%** turn-2/3 hit rate (`65.3%` overall 3-turn agentic hit rate), **`377–392 tok/s`** `S=16` multi-user agentic throughput (`2.46×` upstream, `45–49 tok/s/user`, `18–22×` lower `S=16` p50 TTFT: `323–380 ms`), and **13×** TTFT speedup at 17.6K context with the full KV pool preserved. Also integrates natively with `--kv-offloading-backend native --kv-offloading-size <GiB>` via zero-copy refcounted GPU↔CPU checkpoint slot aliasing + retention-aligned SWA offload pruning (`~41.3 KB/tok` across 4 GPUs; `64 GiB` RAM adds `524,288` CPU KV blocks ≈ `1.62M` extra cached tokens).
- **Native `DSpark7` & Zero-KV `DFly2-KV` Speculative Decoding (`INKLING_SPEC_K="7@1,5@2,3@4"`)**:
  - **Native `DSpark7`**: Mounts `mtp.safetensors` zero-copy with CUDA-graphed draft/context-KV metadata, shard-local recurrent Markov drafting and shard-local greedy/probabilistic target rejection sampling (zero 201K-vocab all-gather for both `T=0` and `T>0`), and `VLLM_USE_SPINLOOP_EXT=1`.
  - **Zero-KV `DFly2-KV` (`dfly2.py`, `dfly2_ops.cu`)**: Drop-in 3-layer block-parallel (`block_size=8`, `γ=7`) + sequential SWA-1024 `AttentiveCorrection` drafter (`165 MB/GPU` FP16 TP4, or `83 MB/GPU` FP8 via `INKLING_DFLY2_DTYPE=fp8`, `cos = 0.999999` live TP4 parity vs BF16 reference across `ctx=0..10K`, `0 B` draft KV, `0 B` draft conv state, zero draft prefill pass) sharing target epilogue KV directly from `L20 (SWA), L23 (Global), L22 (SWA)`. Runs inside FULL CUDA graphs for both greedy (`T=0`) and shard-local probabilistic (`T>0`) drafting/verification, works seamlessly with **RingAPC** (`98.9%` prefix hit on 15.5K follow-up, `4.47×` end-to-end speedup), and reclaims 100% of DSpark7's 3-layer KV cache (**`379.4K` w/ RingAPC (`392.8K` w/o APC) Exact FP16 unlocking `S=16` & `C=16`, `+34.9%` FP8 → `785.6K` unlocking `C=32` at `2,057 tok/s` with zero lossy KV, `+44.7%` `GLA-g4 + ELR` → `1,135.2K` tokens zero-retrain!**).
- **Opt-In Composable Global-KV Compression (default OFF)**:
  - `INKLING_GLA_PATH=<safetensors>`: Post-prologue TP4-native `GLA-g4 r=256` (`2×` global KV, `Top-1=98.28%`, `KLD=0.0083`, `ΔNLL=-0.0089` vs FP32).
  - `INKLING_ELR_GLOBAL_KV=1`: ELR last-loop KV sharing on looped global layers `L11, L17`; combined `GLA-g4 + ELR` reaches **`784.5K` KV (`DSpark7`)** and **`1,135.2K` KV (`DFly2-KV`, `23.1×` upstream, `34.6× 32K`)** (`Top-1=96.56%`, `KLD=0.0241`, `ΔNLL=+0.0113`).

## Cross-Framework Benchmarks (`4× V100-SXM2-16GB`, `8K–32K` Mixed KV, `util=0.88`, `1Cat-vLLM v1.5.4`)

| Framework / Mode | Agentic `8K–32K` (`S=1 / 2 / 4 / 8 / 16` Out, APC ON) | Cold Prefill (`1×128 / 4×128 / 1×8K / 1×16K / 1×32K`) | `8K–32K` Steady Decode (`C=1 / 4 / 12 / 16 / 24 / 32`) | Served KV Capacity | 48-Visit FP32 Oracle (`ΔNLL / KLD / Top-1`) & Notes |
|---|---:|---:|---:|---:|---|
| `llama.cpp` (`UD-Q4_K_XL`, `-sm layer`) | `6.0 / — / — / — / — tok/s` | `255 / 254 / 275 / 248 / — tok/s` | `8.8 / — / — / — / — / — tok/s` | `24.6K` (`16.4K` dspark) | Overflows `C≥2`, `S≥2`, and `1×32K` |
| **Upstream `1Cat-vLLM` (`main` `#1616`/`#1625`, `TP4`)** | **`159`** `/ 157 / — / — / — tok/s` | `2,852 / 4,566 / 9,734 / 9,371 / 8,987 tok/s` | **`538`** `/ — / — / — / — / — tok/s` | `49.2K` (`1.5× 32K`) | Exact FP16 KV + 6 paged conv groups + APC + prob. DSpark7; overflows `C≥4`, `S≥4` |
| **`Zeagarden` Exact FP16 + `DFly2-KV` (`RingAPC`)** | `91 / 156 / 239 / 308 / 377 tok/s` | **`3,942`** `/ 6,318 / 9,888 / 9,704 / 9,196 tok/s` | `340 / 582 / 898 / 1,140 / — / — tok/s` | **`379.4K`** (`7.7×` Up) | **Exact FP16 KV + FP16 dense** + RingAPC (`65.3%` 3-turn hit); unlocks `S=16` (`46 t/s/u`) & `C=16` |
| **`Zeagarden` FP8 KV + `DSpark7` (`RingAPC`)** | `116 /` **`174 / 266 / 329 / 392 tok/s`** | `3,715 /` **`6,410`** `/ 10,004 / 9,911 /` **`9,459 tok/s`** | `437 / 617 /` **`1,023 / 1,165 / 1,640`** `/ — tok/s` | **`582.5K`** (`11.8×` Up) | `+0.0064 / 0.0038 / 97.94%`; **`2.46×` Upstream** agentic (`49 t/s/u @ S=16`, TTFT `380 ms`) |
| **`Zeagarden` FP8 KV + `DFly2-KV` (`RingAPC`)** | `110 / 165 / 257 / 317 / 379 tok/s` | `3,891 / 6,254 /` **`10,029 / 9,917`** `/ 9,458 tok/s` | `378 /` **`696`** `/ 1,011 / 1,142 / 1,621 /` **`2,057 tok/s`** | **`785.6K`** (`16.0×` Up) | Near-lossless FP8 KV (`+35%` KV vs DSpark7); **fits `C=32` @ `2,057 tok/s` (`3.82×` Up) with 0 lossy KV!** |
| **`Zeagarden` `GLA-g4 + ELR`\* + `DFly2-KV` (`RingAPC`)** | `107 / 167 / 241 / 303 / 359 tok/s` | `3,715 / 6,106 / 9,668 / 9,373 / 8,721 tok/s` | `291 / 533 / 891 / 1,142 / 1,358 / 1,789 tok/s` | **`1,135.2K`** (`23.1×` Up) | `+0.0113 / 0.0241 / 96.56%`; **`34.6× 32K` KV**, zero-retrain GLA-g4 + RingAPC |

> Note: `DFly2-KV` results above use the **WSD On-Policy Pilot checkpoint (`step 956`, `2.0B` pre-cooldown tokens)**. Because it shares target epilogue KV (`L20/L23/L22`) directly with zero draft KV cache, it unlocks **`379.4K` w/ RingAPC (`392.8K` w/o APC)** Exact FP16, **`785.6K` (`+35%`)** FP8 (fitting `C=32` at `2,057 tok/s` without lossy KV), and **`1,135.2K` (`+45%`)** `GLA-g4 + ELR` served capacity. With **RingAPC** enabled (`--enable-prefix-caching`), multi-turn `8K–32K` agentic throughput nearly doubles (`S=16`: `206–210 → 377–392 tok/s`, `44–49 tok/s/user`, `323–552 ms` p50 TTFT, `65.2–65.3%` overall 3-turn prefix hit ≈ `98%` on turns 2–3), surpassing Upstream `1Cat-vLLM` starting at `S=2` (`165–174` vs `157 tok/s`, `5–6×` lower p50 TTFT). On `T=0` chat-templated prompts, `DFly2-KV` Pilot matches pretrained `DSpark7` (`slot 0 = 81%`, `slot 1 = 77%`, `τ = 3.56` vs `3.61`), while trailing slightly on `C=1`/`S=1` raw completions until a post-cooldown checkpoint drops in with the exact same 71-tensor layout.

![Inkling-2-Mini cross-framework performance card](assets/inkling2_mini_performance.png)

![Inkling-2-Mini Volta sm_70 empirical roofline and zero-spill register audit](assets/inkling2_mini_roofline.png)

## Offline Tools & Configuration Reference

Offline artifact generators under [`packages/inkling-v100/tools/`](../packages/inkling-v100/tools/):
- `prepare_dspark.py --model-dir <ckpt> --out-dir <dir>` — creates the zero-copy DSpark draft directory (`config.json` + symlink to `mtp.safetensors`).
- `quantize_dense_fp8.py --model-dir <ckpt> --out-dir <dir> --mode absmax` — generates offline per-channel E4M3FN dense weights for `INKLING_FP8_DENSE_PATH`.
- `calibrate_gla_g4.py --model-dir <ckpt> --out <file.safetensors> [--fp8-path <fp8>]` — calibrates Post-Prologue `GLA-g4 r=256` (+ shared ELR) projectors for `INKLING_GLA_PATH`.
- `serve_inkling.sh` / `bench_inkling.py` — launch and benchmark helpers (`DFLY2_DIR=...` or `DSPARK_DIR=...`).

| Env Var | Default | Description |
|---|---|---|
| `INKLING_V100` | `1` | Master switch (`0` disables plugin registration). |
| `INKLING_DFLY2_CKPT` | `""` | Optional override path to a `DFly2-KV` drafter checkpoint (normally passed via `--speculative-config '{"method":"dflash","model":"<dir>",...}'` or `DFLY2_DIR`). |
| `INKLING_DFLY2_DTYPE` | `fp16` | `DFly2-KV` linear weight precision (`fp16`: `165 MB/GPU` TP4, highest `τ`; `fp8`: `83 MB/GPU` TP4 QPN8). |
| `INKLING_FP8_DENSE_PATH` | `""` | Path to offline `absmax` FP8 dense safetensors (`weights/inkling2-mini/inkling2_fp8_dense_absmax.safetensors`; empty = FP16 dense). |
| `INKLING_FP8_TIERS` | `attn,lm_head` | Dense tiers loaded in FP8 (`attn`, `lm_head`, `controllers`). |
| `INKLING_SPEC_K` | `""` | Adaptive speculative verify width by concurrency for both DSpark & DFly2-KV (recommended: `7@1,5@2,3@4`). |
| `INKLING_MOE_CHUNK_CAP` | `4096` | Max tokens per prefill MoE dispatch chunk (default `4096` for max long-prefill throughput). |
| `INKLING_RING_CONV` | `auto` | Lossless per-request ring conv buffer (`auto`/`1` enables `RingConv`, plus `RingAPC` automatically when `--enable-prefix-caching` is on; `0` forces legacy paged conv). |
| `INKLING_RING_APC_STRIDE` | `512` | Token checkpoint stride for `RingAPC` prefix caching (rounded up to a multiple of `lcm_block_size`). |
| `INKLING_RING_APC_RETENTION` | `4096` | SWA replay-window retention interval in tokens for branching prefixes (`0` keeps prompt/reply tail windows only for max multi-session capacity). |
| `INKLING_RING_APC_MAX_SLOTS` | `3072` (`6144` w/ offload) | Max pinned host-RAM conv checkpoints per TP rank for `RingAPC` (lazy-allocated in 64-slot chunks, `1.41 MiB/ckpt`). |
| `INKLING_GLA_PATH` | `""` | Opt-in calibrated `GLA-g4 r=256` projectors (`weights/inkling2-mini/inkling2_gla_g4_r256.safetensors`; works zero-retrain with `DFly2-KV`). |
| `INKLING_ELR_GLOBAL_KV` | `0` | Opt-in ELR last-loop KV sharing on `L11, L17` (`1` enables; composable with `INKLING_GLA_PATH` & `DFly2-KV`). |
| `INKLING_V100_EXT_DIR` | `<pkg>/_ext` | Directory for compiled CUDA extensions. |
