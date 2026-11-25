#!/usr/bin/env python3
"""Prepare a zero-copy DSpark draft model directory from an Inkling-2-Mini checkpoint.

Writes `config.json` and symlinks `mtp.safetensors` into `--out-dir` so vLLM's
`--speculative-config '{"method":"dflash","model":"<out-dir>",...}'` mounts the
native 3-layer DSpark drafter without duplicating weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DSPARK_CONFIG = {
    "model_type": "inkling2_dspark",
    "architectures": ["DFlashInkling2DSparkModel"],
    "torch_dtype": "float16",
    "hidden_size": 2304,
    "num_hidden_layers": 3,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9216,
    "max_position_embeddings": 1048576,
    "vocab_size": 200058,
    "unpadded_vocab_size": 200058,
    "padded_vocab_size": 201024,
    "final_logit_softcapping": 30.0,
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 1024,
    "block_size": 8,
    "markov_rank": 256,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [5, 23, 41, 48],
    "eagle_aux_hidden_state_layer_ids": [5, 23, 41, 48],
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "mask_token_id": 0,
        "is_causal": False,
        "sample_from_anchor": False,
        "target_layer_ids": [5, 23, 41, 48],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Path to thinkingmachines/Inkling-2-Mini checkpoint")
    ap.add_argument("--out-dir", required=True, help="Output directory for DSpark config + weight symlink")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    mtp_src = model_dir / "mtp.safetensors"
    if not mtp_src.is_file():
        raise FileNotFoundError(f"Expected DSpark weights at {mtp_src}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(DSPARK_CONFIG, indent=2) + "\n")
    link = out_dir / "mtp.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(mtp_src, link)
    print(f"Prepared DSpark draft directory at {out_dir} -> {mtp_src}")


if __name__ == "__main__":
    main()
