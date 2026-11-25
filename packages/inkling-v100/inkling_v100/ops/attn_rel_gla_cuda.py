# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 CUDA extension loader & runtime helpers for Post-Prologue GLA-g4 (r=256) + ELR Global KV."""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any

import torch

_EXT: Any = None
_WORKSPACES: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_LINEAR_SLOTS: dict[tuple[str, int | None], torch.Tensor] = {}
_EMPTY_F32_BY_DEV: dict[tuple[str, int | None], torch.Tensor] = {}
_GLA_SAFETENSORS: Any = None
_DEFAULT_SLOTS = 32768  # 32768 * 256 * 4 B ~= 32 MiB/GPU


def _ensure_cuda_in_path() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for cand in ("/usr/local/cuda", "/usr/local/cuda-12.9", "/usr/local/cuda-12.8", "/usr/local/cuda-12"):
        if (pathlib.Path(cand) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = cand
            path = os.environ.get("PATH", "")
            bin_dir = str(pathlib.Path(cand) / "bin")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
            return


def elr_global_kv_enabled() -> bool:
    """Return True when ELR Last-Loop KV caching is enabled on global core layers (L11, L17)."""
    raw = os.environ.get("INKLING_ELR_GLOBAL_KV", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def gla_global_path() -> str | None:
    """Return path to offline GLA-g4 safetensors if enabled, else None.

    Enable via `INKLING_GLA_PATH=/path/to/inkling2_gla_g4_r256.safetensors`
    (generated offline by `tools/calibrate_gla_g4.py`).
    """
    p = os.environ.get("INKLING_GLA_PATH", "").strip()
    if p and p != "0" and p.lower() != "none":
        return p
    raw = os.environ.get("INKLING_GLA_GLOBAL", "0").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        raise ValueError(
            "INKLING_GLA_GLOBAL=1 requires INKLING_GLA_PATH=/path/to/inkling2_gla_g4_r256.safetensors "
            "(generate offline via tools/calibrate_gla_g4.py)."
        )
    return None


def gla_global_enabled() -> bool:
    return gla_global_path() is not None


def get_gla_safetensors() -> Any:
    global _GLA_SAFETENSORS
    path = gla_global_path()
    if path is None:
        return None
    if _GLA_SAFETENSORS is None:
        from safetensors import safe_open

        _GLA_SAFETENSORS = safe_open(path, framework="pt", device="cpu")
    return _GLA_SAFETENSORS


def _ext_dir() -> pathlib.Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
    else:
        p = pathlib.Path(__file__).resolve().parents[1] / "_ext"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_attn_rel_gla_ext(verbose: bool = False) -> Any:
    """Build (if needed) and import `inkling_attn_rel_gla` from `csrc/attn_rel_gla.cu`."""
    global _EXT
    if _EXT is not None:
        return _EXT

    csrc_path = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "attn_rel_gla.cu"
    build_dir = _ext_dir() / "attn_rel_gla"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "inkling_attn_rel_gla.so"

    rebuild = os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc_path.stat().st_mtime:
        spec = importlib.util.spec_from_file_location("inkling_attn_rel_gla", str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    _ensure_cuda_in_path()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from torch.utils.cpp_extension import load

    _EXT = load(
        name="inkling_attn_rel_gla",
        sources=[str(csrc_path)],
        build_directory=str(build_dir),
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_70,code=sm_70",
            "-std=c++17",
            "--ptxas-options=-v",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXT


def preallocate_gla_workspace(
    slots: int = _DEFAULT_SLOTS,
    max_tokens: int = 8192,
    device: torch.device | str | int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-allocate persistent D=256 split-KV workspace & linear slot buffer before CUDA graph capture."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (dev.type, dev.index)

    ls = _LINEAR_SLOTS.get(key)
    if ls is None or ls.numel() < max_tokens:
        if not torch.cuda.is_current_stream_capturing():
            _LINEAR_SLOTS[key] = torch.arange(
                max(max_tokens, 8192), dtype=torch.int64, device=dev
            )

    cur = _WORKSPACES.get(key)
    needed_o = slots * 256
    needed_ml = slots * 2
    if cur is not None and cur[0].numel() >= needed_o and cur[1].numel() >= needed_ml:
        return cur
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"attn_rel_gla_cuda: split-KV workspace ({slots} slots) cannot be allocated inside "
            f"CUDA graph capture; call preallocate_gla_workspace(slots>={slots}) before capture."
        )
    alloc_slots = max(slots, _DEFAULT_SLOTS)
    o_part = torch.empty((alloc_slots * 256,), dtype=torch.float32, device=dev)
    ml_part = torch.empty((alloc_slots * 2,), dtype=torch.float32, device=dev)
    _WORKSPACES[key] = (o_part, ml_part)
    return o_part, ml_part


def get_gla_linear_slots(num_tokens: int, device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    ls = _LINEAR_SLOTS.get(key)
    if ls is None or ls.numel() < num_tokens:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"GLA linear_slots buffer too small for {num_tokens} tokens in capture")
        ls = torch.arange(max(num_tokens, 8192), dtype=torch.int64, device=device)
        _LINEAR_SLOTS[key] = ls
    return ls[:num_tokens]


def _get_empty_f32(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    t = _EMPTY_F32_BY_DEV.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return torch.empty((0,), dtype=torch.float32, device=device)
        t = torch.empty((0,), dtype=torch.float32, device=device)
        _EMPTY_F32_BY_DEV[key] = t
    return t


def plan_rel_gla_decode_splits(
    num_reqs: int,
    max_query_len: int,
) -> int:
    """Wave-aligned split-KV count for Volta sm_70 `k_rel_gla_dec` / `k_rel_gla_dec_2x8` (80 SMs)."""
    num_m_tiles = 2 if max_query_len > 5 else 1
    work = max(1, num_reqs * num_m_tiles)
    if work <= 4:
        return 32
    if work <= 8:
        return 20
    if work <= 16:
        return 10
    if work <= 32:
        return 5
    return 2


def gla_scatter_fp8_cuda(
    c_raw: torch.Tensor,
    c_lo_cache: torch.Tensor,
    c_hi_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
) -> None:
    ext = load_attn_rel_gla_ext()
    ext.gla_scatter_fp8(
        c_raw,
        c_lo_cache.view(torch.uint8),
        c_hi_cache.view(torch.uint8),
        kv_slot_mapping,
    )


def gla_pack_q_cuda(q_in: torch.Tensor, q_out: torch.Tensor) -> None:
    ext = load_attn_rel_gla_ext()
    ext.gla_pack_q(q_in, q_out)


def gla_gate_epilogue_cuda(v_out: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> None:
    ext = load_attn_rel_gla_ext()
    ext.gla_gate_epilogue(v_out, gate, out)


def rel_gla_decode_attention_cuda(
    q_lat: torch.Tensor,
    c_lo_cache: torch.Tensor,
    c_hi_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    o_lat: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    num_splits: int = 0,
) -> None:
    T_total = q_lat.shape[1]
    if T_total == 0:
        return
    if num_splits <= 0:
        num_splits = plan_rel_gla_decode_splits(int(cache_seqlens.shape[0]), max_query_len)
    ext = load_attn_rel_gla_ext()
    if num_splits > 1:
        slots = T_total * 6 * num_splits
        o_part, ml_part = preallocate_gla_workspace(slots=slots, device=q_lat.device)
    else:
        empty = _get_empty_f32(q_lat.device)
        o_part, ml_part = empty, empty
    ext.attn_rel_gla_dec_forward(
        q_lat,
        c_lo_cache.view(torch.uint8),
        c_hi_cache.view(torch.uint8),
        rel_logits,
        sinks,
        o_lat,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        o_part,
        ml_part,
        int(max_query_len),
        int(num_actual_tokens),
        float(softmax_scale),
        int(num_splits),
    )


def rel_gla_prefill_attention_cuda(
    q_lat: torch.Tensor,
    c_lo_cache: torch.Tensor,
    c_hi_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    o_lat: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
) -> None:
    if q_lat.shape[1] == 0:
        return
    ext = load_attn_rel_gla_ext()
    ext.attn_rel_gla_pf_forward(
        q_lat,
        c_lo_cache.view(torch.uint8),
        c_hi_cache.view(torch.uint8),
        rel_logits,
        sinks,
        o_lat,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        int(max_query_len),
        int(num_actual_tokens),
        float(softmax_scale),
    )
