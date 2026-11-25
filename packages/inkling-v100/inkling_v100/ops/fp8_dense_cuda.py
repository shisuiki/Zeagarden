# SPDX-License-Identifier: Apache-2.0
"""Volta sm_70 offline-FP8 QPN8 W8A16 linear wrapper for Inkling-2-Mini dense projections."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load as load_cpp_ext

log = logging.getLogger(__name__)


def _ensure_cuda_in_path() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for cand in ("/usr/local/cuda-12.9", "/usr/local/cuda-12", "/usr/local/cuda"):
        if (Path(cand) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = cand
            path = os.environ.get("PATH", "")
            bin_dir = str(Path(cand) / "bin")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
            return

_EXT: Any = None
_CFG_MAP: dict[str, int] | None = None
_DEQUANT_WS: dict[torch.device, torch.Tensor] = {}
_FP8_FILE_CACHE: dict[str, Any] = {}
_LOGGED_KEYS: set[str] = set()


def _ext_dir() -> Path:
    env_dir = os.environ.get("INKLING_V100_EXT_DIR")
    if env_dir:
        p = Path(env_dir)
    else:
        p = Path(__file__).resolve().parent.parent / "_ext"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fp8_dense_path() -> str | None:
    p = os.environ.get("INKLING_FP8_DENSE_PATH", "").strip()
    if p == "0" or p.lower() == "none" or not p:
        return None
    return p


def fp8_enabled_tiers() -> set[str]:
    if fp8_dense_path() is None:
        return set()
    raw = os.environ.get("INKLING_FP8_TIERS", "attn,lm_head").strip()
    return {t.strip() for t in raw.split(",") if t.strip()}


def get_fp8_safetensors():
    path = fp8_dense_path()
    if path is None:
        return None
    sf = _FP8_FILE_CACHE.get(path)
    if sf is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Offline FP8 dense artifact not found: {path}")
        sf = safe_open(path, framework="pt", device="cpu")
        _FP8_FILE_CACHE[path] = sf
    return sf


def load_fp8_dense_ext(verbose: bool = False, force_rebuild: bool = False):
    global _EXT, _CFG_MAP
    if _EXT is not None and not force_rebuild:
        return _EXT
    pkg_dir = Path(__file__).resolve().parent.parent
    src = pkg_dir / "csrc" / "fp8_dense.cu"
    build_dir = _ext_dir() / "fp8_dense"
    build_dir.mkdir(parents=True, exist_ok=True)
    ext_name = "inkling_fp8_dense_sm70"
    so_path = build_dir / f"{ext_name}.so"
    rebuild = force_rebuild or os.environ.get("INKLING_REBUILD_EXT") == "1"
    if not rebuild and so_path.exists() and so_path.stat().st_mtime >= src.stat().st_mtime:
        spec = importlib.util.spec_from_file_location(ext_name, str(so_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
    else:
        _ensure_cuda_in_path()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _EXT = load_cpp_ext(
            name=ext_name,
            sources=[str(src)],
            build_directory=str(build_dir),
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_70,code=sm_70",
                "--use_fast_math",
            ],
            verbose=verbose,
        )
    _CFG_MAP = {}
    for idx, desc in enumerate(_EXT.fp8_dense_configs()):
        parts = desc.split(":", 2)
        _CFG_MAP[parts[2]] = idx
    return _EXT


def preallocate_fp8_dequant_ws(device: torch.device, max_elems: int = 2176 * 2304) -> torch.Tensor:
    ws = _DEQUANT_WS.get(device)
    if ws is None or ws.numel() < max_elems:
        ws = torch.empty(max_elems, dtype=torch.float16, device=device)
        _DEQUANT_WS[device] = ws
    return ws


def prepare_qpn8_weight(
    w_fp8: torch.Tensor, s32: torch.Tensor, pad_align: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Pack [N_out, K] float8_e4m3fn + [N_out] float32 scales into 1Cat QPN8 layout.

    Pads N_out up to a multiple of `pad_align` (64 or 128) with zero rows.
    Returns `(codes [K, N_pad] u8, s16 [N_pad] fp16, s32_pad [N_pad] fp32, n_pad, n_out)`.
    """
    from vllm import _sm70_ops as sm70_ops

    assert w_fp8.dim() == 2 and w_fp8.dtype == torch.float8_e4m3fn
    n_out, k = w_fp8.shape
    assert k % 32 == 0, f"QPN8 requires K % 32 == 0, got K={k}"
    n_pad = ((n_out + pad_align - 1) // pad_align) * pad_align
    dev = w_fp8.device
    if n_pad != n_out:
        w_pad = torch.zeros((n_pad, k), dtype=torch.uint8, device=dev)
        w_pad[:n_out].copy_(w_fp8.view(torch.uint8))
        w_fp8_pad = w_pad.view(torch.float8_e4m3fn)
        s32_pad = torch.zeros((n_pad,), dtype=torch.float32, device=dev)
        s32_pad[:n_out].copy_(s32.float())
    else:
        w_fp8_pad = w_fp8.contiguous()
        s32_pad = s32.float().contiguous()

    codes, s16 = sm70_ops.fp8_qpn8_prepare_sm70(w_fp8_pad, s32_pad.view(-1, 1))[:2]
    codes = codes.contiguous()  # [K, N_pad] uint8
    s16 = s16.reshape(-1).contiguous()  # [N_pad] fp16 (= s32 * 256)
    return codes, s16, s32_pad, n_pad, n_out


def _pick_cfg(m: int, k: int, n_pad: int, is_lm_head: bool) -> int:
    assert _CFG_MAP is not None
    # Empirically tuned on V100-SXM2-16GB (tests/test_fp8_dense.py with col1 SM saturation):
    if m <= 8:
        return _CFG_MAP["1x1 wk8 col1 pr"]
    if m <= 16:
        return _CFG_MAP["2x1 wk8 col1 pr"]
    if m <= 32:
        if is_lm_head:
            return _CFG_MAP["4x1 wk4 col2 pr"]
        if k <= 1024:
            return _CFG_MAP["4x1 wk4 col1 pr"]
        return _CFG_MAP["4x1 wk8 col1 pr0"]
    if m <= 48:
        return _CFG_MAP["6x1 wk4 col2 pr"] if is_lm_head else _CFG_MAP["6x1 wk4 col1 pr"]
    if is_lm_head and n_pad % 128 == 0:
        return _CFG_MAP["8x1 wk2 col4 pr0"]
    return _CFG_MAP["4x1 wk4 col1 row2 pr"]


def fp8_qpn8_linear(
    x: torch.Tensor,
    codes: torch.Tensor,  # [K, N_pad] uint8
    s16: torch.Tensor,  # [N_pad] fp16
    s32: torch.Tensor,  # [N_pad] fp32
    n_out: int,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
    is_lm_head: bool = False,
    use_1cat_small_m: bool = False,
) -> torch.Tensor:
    """Compute `x @ W_fp8^T` -> `[M, n_out]` in FP16 or FP32 from QPN8 codes."""
    from vllm import _sm70_ops as sm70_ops

    x_c = x if x.is_contiguous() else x.contiguous()
    m, k = x_c.shape
    n_pad = s16.numel()
    if out is None:
        out = torch.empty((m, n_out), dtype=out_dtype, device=x_c.device)
    if m == 0:
        return out

    ext = load_fp8_dense_ext()
    if m <= 64:
        cfg = _pick_cfg(m, k, n_pad, is_lm_head)
        ext.fp8_dense_linear(out, x_c, codes.view(-1), s16, s32, n_pad, cfg)
        return out

    # For LM head (all M), qkvr (M<=128), and wo_ud (K<=1024, M<=256), 64-row col1 chunks
    # beat dequant+cuBLAS (no intermediate HBM write+read of FP16 weights!).
    if is_lm_head or m <= 128 or (k <= 1024 and m <= 256):
        for b0 in range(0, m, 64):
            b1 = min(b0 + 64, m)
            xb = x_c[b0:b1]
            cfg = _pick_cfg(b1 - b0, k, n_pad, is_lm_head)
            ext.fp8_dense_linear(out[b0:b1], xb, codes.view(-1), s16, s32, n_pad, cfg)
        return out

    # Large prefill (M > 64 qkvr, M > 128 wo_ud): exact QPN8 -> FP16 into shared 9.6 MiB workspace + cuBLAS!
    ws = preallocate_fp8_dequant_ws(x_c.device, k * n_pad)
    w_kn = ws[: k * n_pad].view(k, n_pad)
    sm70_ops.fp8_qpn8_dequantize_sm70_out(w_kn, codes, s16.view(1, -1))
    w_use = w_kn if n_out == n_pad else w_kn[:, :n_out]
    if out_dtype == torch.float32:
        torch.mm(x_c, w_use, out_dtype=torch.float32, out=out)
    else:
        torch.mm(x_c, w_use, out=out)
    return out
