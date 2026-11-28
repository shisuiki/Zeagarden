# SPDX-License-Identifier: Apache-2.0
"""Loop message controllers for Volta sm_70 (FP32 blend -> FP16 RMSNorm output)."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton

from .norm import rsqrt


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    stride_msg_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0.0)
    message = tl.load(Message + s * stride_msg_s + t * D + cols, mask, other=0.0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + (s * T + t) * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`[rms(dest), rms(message)]` of fp32 `(S, T, D)` inputs -> `out_dtype` `(S, T, 2D)`."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert message.stride(1) == d and message.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    if out is None:
        out = message.new_empty((num_streams, num_tokens, 2 * d), dtype=out_dtype)
    else:
        assert out.shape == (num_streams, num_tokens, 2 * d) and out.is_contiguous()
    if num_tokens:
        nw = 4 if num_tokens >= 256 else 8
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            message.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=nw,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_msg_s,
    stride_w_s,
    stride_out_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0.0)
    message = tl.load(Message + s * stride_msg_s + t * D + cols, mask, other=0.0)
    gain = tldevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0.0))
    forget = tldevice.tanh(
        tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0.0)
    )
    x = (1.0 - forget) * dest + gain * message
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0.0).to(tl.float32)
    tl.store(Out + s * stride_out_s + t * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out_dtype: torch.dtype | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm of `(1 - tanh(forget)) * dest + tanh(gain) * message` -> `out_dtype` `(S, T, D)`."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert message.stride(1) == d and message.stride(2) == 1
    assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
    assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    final_dtype = norm_weight.dtype if out_dtype is None else out_dtype
    if out is None:
        out = message.new_empty((num_streams, num_tokens, d), dtype=final_dtype)
    else:
        assert out.shape == (num_streams, num_tokens, d)
        assert out.stride(1) == d and out.stride(2) == 1
    if num_tokens:
        nw = 4 if num_tokens >= 256 else 8
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            message.stride(0),
            norm_weight.stride(0),
            out.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=nw,
            enable_fp_fusion=False,
        )
    return out
