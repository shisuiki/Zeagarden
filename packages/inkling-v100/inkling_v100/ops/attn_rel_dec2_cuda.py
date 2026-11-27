# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 short-SASS 1-launch fused `kv_conv_prologue + decode_attention` (v2).

Replaces `attention_prologue` + `rel_decode_attention` with a single CUDA launch
backed by `csrc/attn_rel_dec2.cu` supporting BOTH FP16 and OCP FP8 E4M3FN separate
paged KV caches as well as `RingConv` (`N=16, cmax=1`) and paged conv caches (`N=4`).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
from typing import Any

import torch

log = logging.getLogger(__name__)

MAX_CTAS = 80       # V100 SMs (1 CTA/SM resident for grid barrier; 2 CTAs/SM = 160 when smem <= 47 KB)
MAX_SPLITS = 40     # Default max splits per (request, KV head) (up to 80 in 2-CTA global mode)
MAX_QUERY_LEN = 8
_DEFAULT_SLOTS = 8192    # 8192 * (128 + 2) * 4 B ~= 4.06 MiB per device (C<=8 max is <=3840 slots)
_MAX_COUNTERS = 256      # >= 2 * MAX_CTAS (160) so req * H_kv + kvh never overflows in 2-CTA mode

_TWO_CTA = os.environ.get("INKLING_ATTN_DEC2_2CTA", "1").strip().lower() not in ("0", "false", "no", "off")
_NW = int(os.environ.get("INKLING_ATTN_DEC2_WARPS", "16"))
_PREFETCH = int(os.environ.get("INKLING_ATTN_DEC2_PREFETCH", "1").strip().lower() not in ("0", "false", "no", "off"))
_STOP = int(os.environ.get("INKLING_ATTN_DEC2_STOP", "0"))
assert _NW in (8, 16), f"INKLING_ATTN_DEC2_WARPS must be 8 or 16, got {_NW}"

_EXT: Any = None
_WORKSPACES: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def dec2_enabled() -> bool:
    """True unless `INKLING_USE_CUDA_ATTN_DEC2=0` (defaults to enabled)."""
    return os.environ.get("INKLING_USE_CUDA_ATTN_DEC2", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ext_dir() -> pathlib.Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
    else:
        p = pathlib.Path(__file__).resolve().parents[1] / "_ext"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_attn_rel_dec2_ext(verbose: bool = False) -> Any:
    """Build (if needed) and import `inkling_attn_rel_dec2` from `csrc/attn_rel_dec2.cu`."""
    global _EXT
    if _EXT is not None:
        return _EXT

    csrc_path = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "attn_rel_dec2.cu"
    build_dir = _ext_dir() / "attn_rel_dec2"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "inkling_attn_rel_dec2.so"

    rebuild = os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= csrc_path.stat().st_mtime:
        spec = importlib.util.spec_from_file_location("inkling_attn_rel_dec2", str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    cuda_flags = [
        "-O3",
        "-gencode=arch=compute_70,code=sm_70",
        "-std=c++17",
        "--ptxas-options=-v",
    ]
    _EXT = load(
        name="inkling_attn_rel_dec2",
        sources=[str(csrc_path)],
        build_directory=str(build_dir),
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXT


def _normalize_device(device: torch.device | str | int | None) -> torch.device:
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def preallocate_dec2_workspace(
    device: torch.device | str | int | None = None,
    max_slots: int = _DEFAULT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-allocate persistent `(o_part, ml_part, counters, fault)` workspace outside CUDA graph capture.

    Also configures 96 KB dynamic shared memory attributes for all kernel instantiations on `device`
    when called outside capture, guaranteeing zero allocations or driver attribute updates during
    CUDA graph capture or replay.
    """
    dev = _normalize_device(device)
    key = (dev.type, dev.index)
    cur = _WORKSPACES.get(key)
    needed_o = max_slots * 128
    needed_ml = max_slots * 2
    if cur is not None and cur[0].numel() >= needed_o and cur[1].numel() >= needed_ml:
        return cur
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"attn_rel_dec2_cuda: workspace ({max_slots} slots) cannot be allocated inside "
            f"CUDA graph capture; call preallocate_dec2_workspace(device, max_slots>={max_slots}) before capture."
        )
    alloc_slots = max(max_slots, _DEFAULT_SLOTS)
    o_part = torch.empty((alloc_slots * 128,), dtype=torch.float32, device=dev)
    ml_part = torch.empty((alloc_slots * 2,), dtype=torch.float32, device=dev)
    if cur is not None:
        counters, fault = cur[2], cur[3]
    else:
        counters = torch.zeros((_MAX_COUNTERS,), dtype=torch.int32, device=dev)
        fault = torch.zeros((1,), dtype=torch.int32, device=dev)
    ws = (o_part, ml_part, counters, fault)
    _WORKSPACES[key] = ws
    # Configure smem attributes eagerly outside graph capture if the extension is loadable.
    ext = load_attn_rel_dec2_ext()
    with torch.cuda.device(dev):
        ext.configure_dec2_smem()
    return ws


def dec2_fault(device: torch.device | str | int | None = None) -> bool:
    """Return True if a grid barrier ever timed out on `device` (host sync: tests/diagnostics only)."""
    dev = _normalize_device(device)
    cur = _WORKSPACES.get((dev.type, dev.index))
    if cur is None:
        return False
    return bool(cur[3].item())


def supports_dec2(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    dtype: torch.dtype,
) -> bool:
    """Static model-shape capability check for `attn_rel_dec2`."""
    return (
        dec2_enabled()
        and num_kv_heads > 0
        and num_heads == 3 * num_kv_heads
        and head_dim == 128
        and d_rel == 16
        and dtype == torch.float16
        and num_kv_heads <= MAX_CTAS
    )


def plan_dec2_splits(
    num_reqs: int,
    num_kv_heads: int,
    max_query_len: int = 1,
    is_local: bool = True,
) -> int:
    """CTAs per `(request, KV head)`; returns 0 if the batch does not fit the barrier residency budget.

    Depends only on batch dimensions (CUDA-graph safe).
    """
    if (
        not dec2_enabled()
        or num_reqs <= 0
        or num_kv_heads <= 0
        or max_query_len < 1
        or max_query_len > MAX_QUERY_LEN
    ):
        return 0
    work = num_reqs * num_kv_heads
    if _TWO_CTA and max_query_len <= (2 if _NW == 16 else 3):
        return min(40 if is_local else 80, (2 * MAX_CTAS) // work)
    return min(MAX_SPLITS, MAX_CTAS // work)


def _aligned(t: torch.Tensor, n: int) -> bool:
    return (t.data_ptr() % n) == 0


def _is_supported_kv_dtype(dtype: torch.dtype) -> bool:
    if dtype in (torch.float16, torch.uint8):
        return True
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    return fp8_dtype is not None and dtype == fp8_dtype


def usable_dec2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    conv_cache: torch.Tensor,
    positions: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    r: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    rel_t: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> bool:
    """Layout, dtype, and alignment checks for one call (CUDA-graph safe, zero device syncs).

    Accepts BOTH FP16 (`torch.float16`) and FP8 E4M3FN (`torch.uint8` / `torch.float8_e4m3fn`)
    separate paged KV caches, plus both `RingConv` (`N=16`) and paged conv (`N=4`) caches.
    """
    if not dec2_enabled():
        return False
    if (
        key_cache.dim() != 4
        or value_cache.dim() != 4
        or key_cache.shape != value_cache.shape
        or key_cache.dtype != value_cache.dtype
        or not _is_supported_kv_dtype(key_cache.dtype)
        or conv_cache.dim() != 4
    ):
        return False

    page = key_cache.shape[1]
    h_kv = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    cn = conv_cache.shape[2]
    kv_es = key_cache.element_size()

    ok = (
        head_dim == 128
        and h_kv > 0
        and page > 0
        and (page & (page - 1)) == 0
        and cn > 0
        and (cn & (cn - 1)) == 0
        and conv_cache.shape[1] == h_kv
        and conv_cache.shape[3] >= 256
        and q.dtype == torch.float16
        and q.dim() == 3
        and q.shape[1] == 3 * h_kv
        and q.shape[2] == 128
        and q.stride(2) == 1
        and q.stride(1) == 128
        and _aligned(q, 4)
        and (q.stride(0) % 2) == 0
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and k.dim() == 2
        and v.dim() == 2
        and k.shape[1] == h_kv * 128
        and v.shape[1] == h_kv * 128
        and k.stride(1) == 1
        and v.stride(1) == 1
        and _aligned(k, 4)
        and _aligned(v, 4)
        and (k.stride(0) % 2) == 0
        and (v.stride(0) % 2) == 0
        and key_cache.stride(3) == 1
        and value_cache.stride(3) == 1
        and _aligned(key_cache, 16)
        and _aligned(value_cache, 16)
        and all((s * kv_es) % 16 == 0 for s in key_cache.stride()[:3])
        and all((s * kv_es) % 16 == 0 for s in value_cache.stride()[:3])
        and conv_cache.dtype == torch.float16
        and conv_cache.stride(3) == 1
        and _aligned(conv_cache, 8)
        and all((s % 4) == 0 for s in conv_cache.stride()[:3])
        and positions.dtype == torch.int64
        and conv_slot_mapping.dtype == torch.int64
        and kv_slot_mapping.dtype == torch.int64
        and positions.dim() == 1
        and conv_slot_mapping.dim() == 1
        and kv_slot_mapping.dim() == 1
        and positions.stride(0) == 1
        and conv_slot_mapping.stride(0) == 1
        and kv_slot_mapping.stride(0) == 1
        and k_weight.dtype == torch.float16
        and v_weight.dtype == torch.float16
        and (k_dynamic is None) == (v_dynamic is None)
    )
    if not ok:
        return False

    if r is not None and not (
        r.dtype == torch.float16
        and r.dim() == 3
        and r.shape[1] == 3 * h_kv
        and r.shape[2] == 16
        and r.stride(2) == 1
        and r.stride(1) == 16
    ):
        return False
    if gate is not None and not (
        gate.dtype == torch.float16
        and gate.dim() == 2
        and gate.shape[1] == 3 * h_kv * 128
        and gate.stride(1) == 1
        and _aligned(gate, 4)
        and (gate.stride(0) % 2) == 0
    ):
        return False
    if sinks is not None and not (
        sinks.dtype == torch.float16
        and sinks.dim() == 2
        and sinks.shape[1] == 3 * h_kv
        and sinks.stride(1) == 1
    ):
        return False
    if rel_t is not None and not (
        rel_t.dtype == torch.float16
        and rel_t.dim() == 2
        and rel_t.is_contiguous()
        and rel_t.shape[1] == 16
        and _aligned(rel_t, 16)
    ):
        return False
    if out is not None and not (
        out.dtype == torch.float16
        and out.dim() == 2
        and out.is_contiguous()
        and out.shape[1] == 3 * h_kv * 128
        and _aligned(out, 8)
    ):
        return False

    if k_dynamic is None:
        return (
            k_weight.is_contiguous()
            and v_weight.is_contiguous()
            and k_weight.dim() == 2
            and v_weight.dim() == 2
            and k_weight.shape == (h_kv * 128, 4)
            and v_weight.shape == (h_kv * 128, 4)
            and _aligned(k_weight, 16)
            and _aligned(v_weight, 16)
        )

    assert v_dynamic is not None
    return (
        k_weight.dim() == 2
        and v_weight.dim() == 2
        and k_weight.shape == (4, h_kv * 128)
        and v_weight.shape == (4, h_kv * 128)
        and k_weight.stride(1) == 1
        and v_weight.stride() == k_weight.stride()
        and (k_weight.stride(0) % 4) == 0
        and _aligned(k_weight, 8)
        and _aligned(v_weight, 8)
        and k_dynamic.dtype == torch.float16
        and v_dynamic.dtype == torch.float16
        and k_dynamic.dim() == 2
        and v_dynamic.dim() == 2
        and k_dynamic.shape[1] == h_kv * 4
        and v_dynamic.shape[1] == h_kv * 4
        and k_dynamic.stride(1) == 1
        and v_dynamic.stride(1) == 1
        and (k_dynamic.stride(0) % 4) == 0
        and (v_dynamic.stride(0) % 4) == 0
        and _aligned(k_dynamic, 8)
        and _aligned(v_dynamic, 8)
    )


def rel_decode_attention2_cuda(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_t: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    conv_cache: torch.Tensor,
    positions: torch.Tensor,
    conv_block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    qk_clip: float,
    rel_clip: float,
    norm_eps: float,
    eps: float = 1e-6,
) -> None:
    """1-launch fused K/V short conv + cache writes + relative-bias gated decode/verify attention.

    Supports BOTH FP16 (`torch.float16`) and OCP FP8 E4M3FN (`torch.uint8` / `torch.float8_e4m3fn`)
    separate paged KV caches, plus both `RingConv` (`N=16, cmax=1`) and paged conv (`N=4`) caches.
    Rows `>= num_actual_tokens` in `out` are zeroed safely (CUDA-graph safe, zero per-call allocs).
    `window` is 0 for global layers.
    """
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    num_kv_heads = int(key_cache.shape[2])
    num_reqs = int(seq_lens.shape[0])

    if num_splits <= 0:
        num_splits = plan_dec2_splits(
            num_reqs=num_reqs,
            num_kv_heads=num_kv_heads,
            max_query_len=max_query_len,
            is_local=(window > 0),
        )
    assert num_splits >= 1, (
        f"attn_rel_dec2_cuda: invalid num_splits={num_splits} for "
        f"(num_reqs={num_reqs}, num_kv_heads={num_kv_heads}, max_query_len={max_query_len})"
    )

    is_fp8_kv = key_cache.element_size() == 1
    kc_arg = key_cache.view(torch.uint8) if is_fp8_kv else key_cache
    vc_arg = value_cache.view(torch.uint8) if is_fp8_kv else value_cache

    slots = num_splits * num_tokens * num_heads
    o_part, ml_part, counters, fault_flag = preallocate_dec2_workspace(
        device=q.device,
        max_slots=slots,
    )

    dynamic = k_dynamic is not None
    width = int(k_weight.shape[0] if dynamic else k_weight.shape[1])
    conv_scale = (width * (2 if dynamic else 1) + 1) ** -0.5

    max_splits = 80 if num_splits > 40 else 40
    # Use 16 warps for Q >= 3 when it fits the 1-CTA/SM barrier budget and MS == 40; otherwise 8 warps.
    if _NW == 16 and max_query_len >= 3 and max_splits == 40 and (num_splits * num_kv_heads * num_reqs <= MAX_CTAS):
        num_warps = 16
    else:
        num_warps = 8

    ext = load_attn_rel_dec2_ext()
    ext.attn_rel_dec2_forward(
        q,
        r,
        q_scale,
        r_scale,
        tau,
        rel_t,
        kc_arg,
        vc_arg,
        block_table,
        seq_lens,
        cu_seqlens_q,
        gate,
        sinks,
        out,
        o_part,
        ml_part,
        counters,
        fault_flag,
        k,
        v,
        k_weight,
        v_weight,
        k_dynamic,
        v_dynamic,
        conv_cache,
        positions,
        kv_slot_mapping,
        conv_slot_mapping,
        seq_idx,
        query_start,
        conv_block_table,
        int(num_reqs),
        int(max_query_len),
        int(num_actual_tokens),
        int(num_splits),
        int(window),
        float(softmax_scale),
        float(qk_clip),
        float(rel_clip),
        float(eps),
        float(conv_scale),
        float(norm_eps),
        int(_STOP),
        int(num_warps),
        int(max_splits),
        int(_PREFETCH),
    )
