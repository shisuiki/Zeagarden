# Zeagarden

A collection of optimized multi-kernels and model-specific serving packages.

## Packages

| Package | Model | Target Hardware / Runtime | Highlights | Docs |
|---|---|---|---|---|
| [`packages/q27fuse`](packages/q27fuse/) | `Qwen3.8-27B-NVFP4` (hybrid GDN + Full Attn) | 4× Tesla V100-SXM2 (`sm_70`), stock [`1Cat-vLLM v1.5.4`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.4) | `1,742 tok/s` `8K–32K` `C=16` (`227 tok/s` agentic `S=8`), `1.11–1.98×` prefill, opt-in zero-KV `DFly2-KV` (`630K`/`889K` KV), `H_32` (`5.17K tok/s`), & `GLA-g4` (`854K–1.24M` KV) | [Qwen3.8-27B](docs/qwen3-27b.md) |
| [`packages/inkling-v100`](packages/inkling-v100/) | `thinkingmachines/Inkling-2-Mini` (looped 128-expert MoE + DSpark / DFly2-KV) | 4× Tesla V100-SXM2 (`sm_70`), stock [`1Cat-vLLM v1.5.4`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.4) | 13 zero-spill `sm_70` CUDA extensions, **`10.0K tok/s`** `1×8K` / **`9.46K`** `1×32K` prefill (beats upstream across all 8 regimes), **`2,057 tok/s`** `C=32` near-lossless FP8 KV + zero-KV `DFly2-KV` (`785.6K` KV, `16×` upstream), lossless `RingConv`+`RingAPC`, opt-in `GLA-g4`+`ELR` (`1.14M` KV, `23.1×` upstream) | [Inkling-2-Mini](docs/inkling2-mini.md) |

Both Volta packages register via standard `vllm.general_plugins` entry points on top of **unmodified `1Cat-vLLM v1.5.4`** (no source patches required; the two plugins coexist cleanly in the same environment).

## Repository Layout

```text
Zeagarden/
├── docs/                  # Centralized documentation & diagrams
│   ├── qwen3-27b.md
│   ├── inkling2-mini.md
│   └── assets/
├── packages/              # Self-contained, model/runtime-independent packages
│   ├── q27fuse/           # Qwen3.8-27B sm_70 plugin, CUDA kernels, tools, tests
│   └── inkling-v100/      # Inkling-2-Mini sm_70 plugin, CUDA kernels, tools, tests
└── weights/               # Calibrated / quantized auxiliary safetensors (Git LFS)
    ├── qwen3-27b/         # GLA-g4 r=192/256 projectors + H_32 1-bit selection planes
    └── inkling2-mini/     # Offline absmax FP8 dense weights + GLA-g4 r=256 projectors + DSpark config
```

## Quick Start (Volta `sm_70` + `1Cat-vLLM v1.5.4`)

1. Install stock [`1Cat-vLLM v1.5.4`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.4) (CUDA 12.8/12.9, Python 3.12):

   ```bash
   pip install https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.4/1cat_vllm-1.5.4%2Bcu128-cp312-cp312-linux_x86_64.whl
   ```

2. Install either or both packages (editable, no build isolation):

   ```bash
   pip install --no-build-isolation --no-deps -e packages/q27fuse
   pip install --no-build-isolation --no-deps -e packages/inkling-v100
   ```

3. *(Optional)* Pre-compile the CUDA extensions ahead of first launch:

   ```bash
   packages/q27fuse/build.sh
   packages/inkling-v100/build.sh
   ```

4. Serve:

   - **Qwen3.8-27B**:
     ```bash
     MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 packages/q27fuse/tools/serve_q27.sh
     # Optional zero-KV DFly2-KV drafter:
     MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 SPEC=dfly2 Q27_DFLY2_CKPT=/path/to/dfly2_nvfp4_fp8.safetensors packages/q27fuse/tools/serve_q27.sh
     ```
   - **Inkling-2-Mini** (uses prebuilt LFS weights under `weights/inkling2-mini/`):
     ```bash
     # Native DSpark7 drafter:
     python3 packages/inkling-v100/tools/prepare_dspark.py --model-dir /path/to/Inkling-2-Mini --out-dir ./dspark_draft
     MODEL_DIR=/path/to/Inkling-2-Mini DSPARK_DIR=./dspark_draft \
       INKLING_FP8_DENSE_PATH=weights/inkling2-mini/inkling2_fp8_dense_absmax.safetensors \
       packages/inkling-v100/tools/serve_inkling.sh

     # Or zero-KV DFly2-KV drafter (0 B draft KV, +22% Exact / +35% FP8 / +45% GLA+ELR KV capacity):
     MODEL_DIR=/path/to/Inkling-2-Mini DFLY2_DIR=/path/to/Inkling-2-Mini-DFly2-Pilot \
       INKLING_FP8_DENSE_PATH=weights/inkling2-mini/inkling2_fp8_dense_absmax.safetensors \
       packages/inkling-v100/tools/serve_inkling.sh
     ```

See [docs/qwen3-27b.md](docs/qwen3-27b.md) and [docs/inkling2-mini.md](docs/inkling2-mini.md) for architecture notes, benchmark tables, and optional flags.

## License

[GNU General Public License v3.0](LICENSE)
