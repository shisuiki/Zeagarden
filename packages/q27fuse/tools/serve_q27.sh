#!/usr/bin/env bash
# Launch Qwen3.8-27B-NVFP4 on 4x V100-SXM2-16GB with q27fuse (1Cat-vLLM v1.5.3).
#
# Usage:
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 ./serve_q27.sh [MAX_SEQS]
#
# Optional env vars:
#   SPEC=mtp4          Enable MTP4 speculative decoding + adaptive Q27_SPEC_K (default: mtp4; set SPEC=none for target-only)
#   Q27_FUSE=1         Enable q27fuse plugin (default: 1; set 0 for stock 1Cat-vLLM)
#   Q27_SPF=1          Enable exact short/agentic prefill kernels (default: 1)
#   Q27_PF_PK=0        Opt-in packed H_32 long prefill (set 1 + Q27_PF_H32_PLANE=<dir> or Q27_PF_H32_NV=1)
#   Q27_GLA_CKPT=""    Opt-in TP4-native GLA-g4 compressed KV checkpoint (.pt)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-${1:-}}"
if [ -z "$MODEL_DIR" ]; then
  echo "Usage: MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 $0 [MAX_SEQS]" >&2
  exit 1
fi
MAX_SEQS="${MAX_SEQS:-${2:-64}}"
SPEC="${SPEC:-mtp4}"

export Q27_FUSE="${Q27_FUSE:-1}"
export Q27_SPF="${Q27_SPF:-1}"

ARGS=(
  "$MODEL_DIR"
  --served-model-name "${SERVED_NAME:-qwen3.8-27b}"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-8000}"
  --tensor-parallel-size 4
  --dtype half
  --kv-cache-dtype fp8_e4m3
  --attention-backend FLASH_ATTN_V100
  --max-model-len "${MAX_LEN:-16384}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
  --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization "${GPU_UTIL:-0.90}"
  --trust-remote-code
  --seed 0
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
)

if [[ "$SPEC" =~ ^mtp[1-9]$ ]]; then
  export Q27_SPEC_K="${Q27_SPEC_K:-4@1,3@2,4@3,3@4,1@16,2@17,1@22}"
  ARGS+=(
    --enable-prefix-caching
    --mixed-prefill-step-latency-ms 0
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC#mtp}}"
  )
  [ -n "${GRAPH_SIZES:-}" ] && ARGS+=(--compilation-config "{\"cudagraph_capture_sizes\": [$GRAPH_SIZES]}")
else
  ARGS+=(
    --no-enable-prefix-caching
    --compilation-config "{\"cudagraph_capture_sizes\": [${GRAPH_SIZES:-1,2,4,8,12,16,24,32,48,64}]}"
  )
fi

[ -n "${EXTRA_ARGS:-}" ] && read -ra XA <<< "$EXTRA_ARGS" && ARGS+=("${XA[@]}")
exec vllm serve "${ARGS[@]}"
