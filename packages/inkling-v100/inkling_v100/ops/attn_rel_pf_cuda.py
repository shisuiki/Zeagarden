# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 CUDA extension loader & wrapper for Inkling2 relative-bias prefill attention.

Provides a drop-in fused replacement for `rel_attention` + `gate_sink` in `attention_sm70.py`
backed by `csrc/attn_rel_pf.cu` (zero-spill HMMA.884, FP8 E4M3FN & FP16 paged KV, sheared
`rel_logits` with far-past skip, fused `gate_sink`, CUDA-graph safe).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any

import torch

_EXT: Any = None


def _ext_dir() -> pathlib.Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
    else:
        p = pathlib.Path(__file__).resolve().parents[1] / "_ext"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_attn_rel_pf_ext(verbose: bool = False) -> Any:
    """Build (if needed) and import `inkling_attn_rel_pf` from `csrc/attn_rel_pf.cu`."""
    global _EXT
    if _EXT is not None:
        return _EXT

    csrc_path = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "attn_rel_pf.cu"
    build_dir = _ext_dir() / "attn_rel_pf"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "inkling_attn_rel_pf.so"

    rebuild = os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc_path.stat().st_mtime:
        spec = importlib.util.spec_from_file_location("inkling_attn_rel_pf", str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from torch.utils.cpp_extension import load

    cuda_flags = [
        "-O3",
        "-gencode=arch=compute_70,code=sm_70",
        "-std=c++17",
        "--ptxas-options=-v",
    ]
    _EXT = load(
        name="inkling_attn_rel_pf",
        sources=[str(csrc_path)],
        build_directory=str(build_dir),
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXT


def rel_prefill_attention_cuda(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_actual_tokens: int | None = None,
    config_override: int = 0,
) -> None:
    """Volta sm_70 CUDA relative-bias GQA prefill attention with fused `gate_sink`."""
    num_tokens, _, head_dim = q.shape
    if num_tokens == 0:
        return
    assert head_dim == 128, f"attn_rel_pf_cuda requires head_dim=128, got {head_dim}"

    is_fp8_kv = key_cache.element_size() == 1
    kc_arg = key_cache.view(torch.uint8) if is_fp8_kv else key_cache
    vc_arg = value_cache.view(torch.uint8) if is_fp8_kv else value_cache
    act_tokens = num_tokens if num_actual_tokens is None else int(num_actual_tokens)

    ext = load_attn_rel_pf_ext()
    ext.attn_rel_pf_forward(
        q,
        kc_arg,
        vc_arg,
        rel_logits,
        gate,
        sinks,
        out,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        int(max_query_len),
        int(act_tokens),
        float(softmax_scale),
        int(window),
        int(config_override),
    )
