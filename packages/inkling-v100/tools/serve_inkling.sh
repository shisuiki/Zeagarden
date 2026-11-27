#!/usr/bin/env bash
# Launch thinkingmachines/Inkling-2-Mini on 4x V100-SXM2-16GB with inkling_v100 (1Cat-vLLM v1.5.3).
#
# Usage:
#   MODEL_DIR=/path/to/Inkling-2-Mini DSPARK_DIR=/path/to/dspark_draft ./serve_inkling.sh
#
# Optional env vars:
#   INKLING_FP8_DENSE_PATH=<path>  Offline absmax FP8 dense weights (from tools/quantize_dense_fp8.py)
#   INKLING_FP8_TIERS=attn,lm_head Dense tiers quantized to FP8 (default: attn,lm_head)
#   INKLING_SPEC_K="7@1,5@2,3@4"   Adaptive DSpark verify width by concurrency
#   INKLING_GLA_PATH=<path>        Opt-in Post-Prologue GLA-g4 r=256 projectors (from tools/calibrate_gla_g4.py)
#   INKLING_ELR_GLOBAL_KV=0        Opt-in ELR Last-Loop KV sharing on L11/L17 (set 1 to enable; composable with GLA)
#   PREFIX_CACHING=1               Default 1 (--enable-prefix-caching auto-enables lossless RingAPC on top of RingConv 16.2x KV; set 0 for --no-enable-prefix-caching)
#   INKLING_RING_APC_STRIDE=512    Token checkpoint stride for RingAPC prefix caching
#   INKLING_RING_APC_MAX_SLOTS=3072 Max pinned host-RAM checkpoints per TP rank for RingAPC
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-${1:-}}"
if [ -z "$MODEL_DIR" ]; then
  echo "Usage: MODEL_DIR=/path/to/Inkling-2-Mini [DSPARK_DIR=/path/to/dspark_draft] $0" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_V41_PIN_CORES="${VLLM_V41_PIN_CORES:-0,1,2,3}"
export VLLM_V41_PIN_ENGINE="${VLLM_V41_PIN_ENGINE:-4}"
export INKLING_FP8_TIERS="${INKLING_FP8_TIERS:-attn,lm_head}"
export INKLING_SPEC_K="${INKLING_SPEC_K:-7@1,5@2,3@4}"

ARGS=(
  "$MODEL_DIR"
  --served-model-name "${SERVED_NAME:-inkling-2-mini}"
  -tp 4
  --dtype half
  --kv-cache-dtype fp8
  --gpu-memory-utilization "${GPU_UTIL:-0.85}"
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

if [ -n "${DSPARK_DIR:-}" ]; then
  ARGS+=(
    --speculative-config
    "{\"method\":\"dflash\",\"model\":\"${DSPARK_DIR}\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}"
  )
fi

[ -n "${EXTRA_ARGS:-}" ] && read -ra XA <<< "$EXTRA_ARGS" && ARGS+=("${XA[@]}")
exec vllm serve "${ARGS[@]}"
