#!/usr/bin/env bash
# Launch thinkingmachines/Inkling-2-Mini on 4x V100-SXM2-16GB with inkling_v100 (1Cat-vLLM v1.5.4).
#
# Usage:
#   # Native DSpark7 drafter:
#   MODEL_DIR=/path/to/Inkling-2-Mini DSPARK_DIR=/path/to/dspark_draft ./serve_inkling.sh
#   # Zero-KV DFly2-KV drafter (0 B draft KV, zero draft prefill, zero-retrain GLA-g4):
#   MODEL_DIR=/path/to/Inkling-2-Mini DFLY2_DIR=/path/to/Inkling-2-Mini-DFly2-Pilot ./serve_inkling.sh
#
# Optional env vars:
#   DFLY2_DIR=<path>               Zero-KV DFly2-KV drafter directory (takes precedence over DSPARK_DIR)
#   INKLING_DFLY2_DTYPE=fp16       DFly2-KV drafter linear dtype: fp16 (default, 165 MB/GPU) or fp8 (83 MB/GPU)
#   KV_DTYPE=fp8                   Target KV cache dtype: fp8 (default, near-lossless) or float16 (exact FP16 KV)
#   INKLING_FP8_DENSE_PATH=<path>  Offline absmax FP8 dense weights (from tools/quantize_dense_fp8.py)
#   INKLING_FP8_TIERS=attn,lm_head Dense tiers quantized to FP8 (default: attn,lm_head)
#   INKLING_SPEC_K="7@1,5@2,3@4"   Adaptive speculative verify width by concurrency
#   INKLING_GLA_PATH=<path>        Opt-in Post-Prologue GLA-g4 r=256 projectors (from tools/calibrate_gla_g4.py)
#   INKLING_ELR_GLOBAL_KV=0        Opt-in ELR Last-Loop KV sharing on L11/L17 (set 1 to enable; composable with GLA)
#   PREFIX_CACHING=1               Default 1 (--enable-prefix-caching auto-enables lossless RingAPC on top of RingConv; set 0 for --no-enable-prefix-caching)
#   KV_OFFLOAD_GB=0                Optional native Host RAM KV cache tier in GiB (e.g. 64 => --kv-offloading-backend native --kv-offloading-size 64)
#   INKLING_RING_APC_STRIDE=512    Token checkpoint stride for RingAPC prefix caching
#   INKLING_RING_APC_RETENTION=4096 SWA replay-window retention interval in tokens (0 = tail windows only)
#   INKLING_RING_APC_MAX_SLOTS=3072 Max pinned host-RAM checkpoints per TP rank (auto-6144 when KV_OFFLOAD_GB > 0)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-${1:-}}"
if [ -z "$MODEL_DIR" ]; then
  echo "Usage: MODEL_DIR=/path/to/Inkling-2-Mini [DFLY2_DIR=... | DSPARK_DIR=...] $0" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_SPINLOOP_EXT="${VLLM_USE_SPINLOOP_EXT:-1}"
export VLLM_V41_PIN_CORES="${VLLM_V41_PIN_CORES:-0,1,2,3}"
export VLLM_V41_PIN_ENGINE="${VLLM_V41_PIN_ENGINE:-4}"
export INKLING_FP8_TIERS="${INKLING_FP8_TIERS:-attn,lm_head}"
export INKLING_SPEC_K="${INKLING_SPEC_K:-7@1,5@2,3@4}"

ARGS=(
  "$MODEL_DIR"
  --served-model-name "${SERVED_NAME:-inkling-2-mini}"
  -tp 4
  --dtype half
  --kv-cache-dtype "${KV_DTYPE:-fp8}"
  --gpu-memory-utilization "${GPU_UTIL:-0.88}"
  --max-model-len "${MAX_LEN:-32768}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
  --max-num-seqs "${MAX_SEQS:-32}"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-8000}"
  -cc.mode=none
  -cc.cudagraph_mode=FULL_DECODE_ONLY
)

if [ "${PREFIX_CACHING:-1}" = "0" ]; then
  ARGS+=(--no-enable-prefix-caching)
else
  ARGS+=(--enable-prefix-caching)
fi

VLLM_VER="$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
if [[ "$VLLM_VER" == 1.5.4* ]]; then
  ARGS+=(--no-split-mtp-phase-pure)
fi

if [ "${KV_OFFLOAD_GB:-0}" != "0" ]; then
  ARGS+=(--kv-offloading-backend native --kv-offloading-size "${KV_OFFLOAD_GB}")
fi

DRAFT_DIR="${DFLY2_DIR:-${DSPARK_DIR:-}}"
if [ -n "$DRAFT_DIR" ]; then
  DRAFT_SAMPLE="${DRAFT_SAMPLE_METHOD:-probabilistic}"
  ARGS+=(
    --speculative-config
    "{\"method\":\"dflash\",\"model\":\"${DRAFT_DIR}\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"${DRAFT_SAMPLE}\"}"
  )
fi

[ -n "${EXTRA_ARGS:-}" ] && read -ra XA <<< "$EXTRA_ARGS" && ARGS+=("${XA[@]}")
exec vllm serve "${ARGS[@]}"
