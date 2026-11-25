#!/usr/bin/env bash
# AOT-build all 9 inkling_v100 Volta (sm_70) CUDA extensions.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="${1:-${INKLING_V100_EXT_DIR:-$SCRIPT_DIR/inkling_v100/_ext}}"

if [ -z "${CUDA_HOME:-}" ]; then
  for cand in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if [ -x "$cand/bin/nvcc" ]; then
      export CUDA_HOME="$cand"
      export PATH="$cand/bin:$PATH"
      break
    fi
  done
fi

export INKLING_V100_EXT_DIR="$EXT_DIR" CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
t0=$(date +%s)
python3 - <<'PY'
import time
import inkling_v100
from inkling_v100.ops import (
    allreduce_conv_mix_cuda,
    attn_rel_dec_cuda,
    attn_rel_gla_cuda,
    attn_rel_pf_cuda,
    ep4_route_cuda,
    fp8_dense_cuda,
    loop_blend_cuda,
    mxfp8_moe_cuda,
    nvfp4_prefill_cuda,
)

t = time.time()
print("package:", inkling_v100.__file__, flush=True)
exts = [
    ("fp8_dense", fp8_dense_cuda.load_fp8_dense_ext),
    ("nvfp4_prefill_moe", nvfp4_prefill_cuda.load_nvfp4_prefill_ext),
    ("attn_rel_pf", attn_rel_pf_cuda.load_attn_rel_pf_ext),
    ("attn_rel_dec", attn_rel_dec_cuda.load_attn_rel_dec_ext),
    ("attn_rel_gla", attn_rel_gla_cuda.load_attn_rel_gla_ext),
    ("mlp_mxfp8_moe", mxfp8_moe_cuda.load_mxfp8_moe_ext),
    ("ep4_route", ep4_route_cuda.load_ep4_route_ext),
    ("loop_blend", loop_blend_cuda.load_loop_blend_ext),
    ("all_reduce_conv_mix", allreduce_conv_mix_cuda.load_ar_conv_mix_ext),
]
for name, fn in exts:
    t1 = time.time()
    fn()
    print(f"built {name:20s} ({time.time() - t1:.1f}s)", flush=True)
print(f"ALL BUILT ({len(exts)} extensions in {time.time() - t:.1f}s)", flush=True)
PY
echo "BUILD_DONE rc=$? ($(( $(date +%s) - t0 ))s)"
