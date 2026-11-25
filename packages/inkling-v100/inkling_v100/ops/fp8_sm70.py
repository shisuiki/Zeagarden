# SPDX-License-Identifier: Apache-2.0
"""Exact software FP8 E4M3FN <-> FP16/FP32 bit helpers for Volta sm_70 Triton.

Triton 3.6 does not allow `tl.float8e4nv` pointers on sm_70 (`ValueError: type
fp8e4nv not supported in this architecture`). All FP8 KV caches and MXFP8 weight
tensors are therefore viewed as `uint8` on the host and converted via exact
bit-level IEEE-754 RNE helpers inside Triton.
"""

from __future__ import annotations

from vllm.triton_utils import tl, triton


@triton.jit
def e4m3_u8_to_f16_scaled(b_u8):
    """Exact bit-unpack of uint8 E4M3FN -> fp16 value * 2^-8.

    For every non-NaN E4M3FN byte (0x00..0x7E, 0x80..0xFE):
      - Normal (e in 1..15): E4M3 = (-1)^s * 2^(e-7) * (1 + m/8).
        Shifted into FP16 (bias 15): e' = e, m' = m<<7 -> (-1)^s * 2^(e-15) * (1 + m/8) = exact * 2^-8.
      - Subnormal (e == 0, m in 1..7): E4M3 = (-1)^s * 2^-6 * (m/8) = (-1)^s * 2^-9 * m.
        Shifted into FP16: e' = 0, m' = m<<7 -> (-1)^s * 2^-14 * (m/8) = (-1)^s * 2^-17 * m = exact * 2^-8!
      - Zero (0x00, 0x80): exact ±0.0.
    Multiply result by 256.0 (or fold 256.0 into softmax/gemm scale) for exact value.
    """
    u = b_u8.to(tl.uint16)
    h_bits = ((u & 0x80) << 8) | ((u & 0x7F) << 7)
    return h_bits.to(tl.float16, bitcast=True)


@triton.jit
def e4m3_u8_to_f16(b_u8):
    """Exact bit-unpack of uint8 E4M3FN -> fp16 value (normals & subnormals exact)."""
    return (e4m3_u8_to_f16_scaled(b_u8).to(tl.float32) * 256.0).to(tl.float16)


@triton.jit
def f32_to_e4m3_u8(x_f32):
    """Exact IEEE-754 round-to-nearest-even FP32 -> uint8 E4M3FN (saturating to ±448.0).

    Matches PyTorch `x.to(torch.float8_e4m3fn).view(torch.uint8)` bit-for-bit across
    normals, subnormals, zero, saturation (|x| >= 448 -> 0x7E / 0xFE), and NaN (0x7F/0xFF).
    """
    bits = x_f32.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    abs_bits = bits & 0x7FFFFFFF
    is_nan = abs_bits > 0x7F800000
    # Clamp magnitude to E4M3FN max = 448.0 (0x43E00000) before rounding.
    clamped = tl.minimum(abs_bits, 0x43E00000)
    # Normal range in E4M3FN: value >= 2^-6 = 0x3C800000 (fp32 exp >= 121).
    # For fp32 exp < 121, shift significand into subnormal position by adding
    # 0.5 * 2^-6 = 2^-7? Exact approach: multiply by 2^(-120) in fp32 so
    # exp=121 becomes exp=1 (normal min), or compute exp/mantissa directly:
    exp = (clamped >> 23).to(tl.int32)
    mant = clamped & 0x007FFFFF

    # Normal case (exp >= 121): target e4 = exp - 120 in [1..15], keep top 3 mant bits (bits 22..20)
    # Round-to-nearest-even on the low 20 bits (threshold 0x80000, tie to even using bit 20).
    norm_e = exp - 120
    norm_round = ((mant >> 20) & 1) + 0x7FFFF
    norm_rounded = (norm_e << 23) + mant + norm_round
    norm_code = (norm_rounded >> 20).to(tl.uint32)
    # Guard against rounding 448+ up to NaN (0x7F): cap at 0x7E.
    norm_code = tl.minimum(norm_code, 0x7E)

    # Subnormal case (exp < 121): E4M3FN subnormal step is 2^-9.
    # Multiply |x| by 2^9 (= 512.0) and round to nearest even integer in [0..8].
    # Note: 8 * 2^-9 = 2^-6 -> code 0x08 (min normal!), which naturally matches!
    abs_x = clamped.to(tl.float32, bitcast=True)
    scaled = abs_x * 512.0
    # IEEE-754 RNE float->int via adding 2^23 magic constant (exact for 0 <= scaled <= 8.0)
    magic = 8388608.0  # 2^23
    sub_code = ((scaled + magic).to(tl.uint32, bitcast=True) - 0x4B000000)
    sub_code = tl.where(abs_bits < 0x3A800000, 0, sub_code)  # < 2^-10 rounds to 0

    code = tl.where(exp >= 121, norm_code, sub_code)
    code = tl.where(is_nan, 0x7F, code)
    return (sign | code).to(tl.uint8)
