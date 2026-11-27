#!/usr/bin/env bash
# AOT-build every q27fuse CUDA extension (sm_70) into Q27_FUSE_BUILD (default: ~/.cache/q27fuse/build).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TREE="${1:-$SCRIPT_DIR}"
BUILD_PREFIX="${2:-${Q27_FUSE_BUILD:-$HOME/.cache/q27fuse/build}}"

if [ -z "${CUDA_HOME:-}" ]; then
  for cand in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if [ -x "$cand/bin/nvcc" ]; then
      export CUDA_HOME="$cand"
      export PATH="$cand/bin:$PATH"
      break
    fi
  done
fi

cd "$TREE"
export Q27_FUSE_BUILD="$BUILD_PREFIX" CUDA_VISIBLE_DEVICES="" PYTHONPATH="$TREE${PYTHONPATH:+:$PYTHONPATH}"
t0=$(date +%s)
python3 - <<'PY'
import time
import q27fuse
from q27fuse import ops, attnv, gdnv, smallrows, draftscreen, tail, head, pf2, pk_h32, shortpf, gla

t = time.time()
print("package:", q27fuse.__file__, flush=True)
exts = [
    ("gdn_glue", ops.gdn_ext),
    ("gdn_core", ops.gdn_core_ext),
    ("mlp_fp4", ops.mlp_ext),
    ("fp8_lin", ops.fp8_ext),
    ("attn_dec", ops.attn_ext),
    ("attn_mq", attnv.mq_ext),
    ("gdn_verify", gdnv.ext),
    ("gdn_decode", gdnv.dext),
    ("fp8_gdn_in", smallrows.gdn_in_ext),
    ("draft_screen", draftscreen.ext),
    ("tail_ops", tail.ext),
    ("fp8_head", head.ext),
    *pf2.exts(),
    ("pk_h32", pk_h32.ext),
    ("mlp_short", shortpf.mlp_short_ext),
    ("fp8_short", shortpf.fp8_short_ext),
    ("attn_gla_mq", gla.gla_ext),
]
for name, fn in exts:
    t1 = time.time()
    fn()
    print(f"built {name:14s} ({time.time() - t1:.1f}s)", flush=True)
print(f"ALL BUILT ({len(exts)} extensions in {time.time() - t:.1f}s)", flush=True)
PY
echo "BUILD_DONE rc=$? ($(( $(date +%s) - t0 ))s)"
