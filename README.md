# Zeagarden

A collection of optimized multi-kernels and model-specific serving packages.

## Packages

| Package | Model | Target Hardware / Runtime | Highlights | Docs |
|---|---|---|---|---|
| [`packages/q27fuse`](packages/q27fuse/) | `Qwen3.8-27B-NVFP4` (hybrid GDN + Full Attn) | 4× Tesla V100-SXM2 (`sm_70`), stock [`1Cat-vLLM v1.5.3`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) | `2,261 tok/s` 8K MTP4 (`1.53×`), `1.36×` short prefill, opt-in `H_32` 3-for-2 HMMA (`478 TOPS`) & `GLA-g4` (`1.86–1.94×` KV) | [Qwen3.8-27B](docs/qwen3-27b.md) |
| [`packages/inkling-v100`](packages/inkling-v100/) | `thinkingmachines/Inkling-2-Mini` (looped 128-expert MoE + DSpark) | 4× Tesla V100-SXM2 (`sm_70`), stock [`1Cat-vLLM v1.5.3`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) | 9 zero-spill `sm_70` CUDA extensions, `344 tok/s` `C=1` code, `1,732 tok/s` `C=32`, lossless `16.2×` KV (`RingConv`), opt-in `GLA-g4` + `ELR` (`755K` KV) | [Inkling-2-Mini](docs/inkling2-mini.md) |

Both Volta packages register via standard `vllm.general_plugins` entry points on top of **unmodified `1Cat-vLLM v1.5.3`** (no source patches required; the two plugins coexist cleanly in the same environment).

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
    ├── qwen3-27b/         # GLA-g4 r=192 projectors + H_32 1-bit selection planes
    └── inkling2-mini/     # Offline absmax FP8 dense weights + GLA-g4 r=256 projectors + DSpark config
```

## Quick Start (Volta `sm_70` + `1Cat-vLLM v1.5.3`)

1. Install stock [`1Cat-vLLM v1.5.3`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) (CUDA 12.8/12.9, Python 3.12):

   ```bash
   pip install https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/1cat_vllm-1.5.3%2Bcu128-cp312-cp312-linux_x86_64.whl
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
     ```
   - **Inkling-2-Mini** (uses prebuilt LFS weights under `weights/inkling2-mini/`):
     ```bash
     python3 packages/inkling-v100/tools/prepare_dspark.py --model-dir /path/to/Inkling-2-Mini --out-dir ./dspark_draft
     MODEL_DIR=/path/to/Inkling-2-Mini DSPARK_DIR=./dspark_draft \
       INKLING_FP8_DENSE_PATH=weights/inkling2-mini/inkling2_fp8_dense_absmax.safetensors \
       packages/inkling-v100/tools/serve_inkling.sh
     ```

See [docs/qwen3-27b.md](docs/qwen3-27b.md) and [docs/inkling2-mini.md](docs/inkling2-mini.md) for architecture notes, benchmark tables, and optional flags.

## License

[GNU General Public License v3.0](LICENSE)
