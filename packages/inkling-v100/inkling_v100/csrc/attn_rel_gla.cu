// SPDX-License-Identifier: Apache-2.0
// attn_rel_gla.cu - Zero-spill Volta (sm_70) HMMA.884 Post-Prologue GLA-g4 (r=256, G=6, H_kv=1)
// relative-bias decode/verify & prefill attention kernels + FP8 latent scatter & gate epilogue
// for Inkling-2-Mini global layers (L5, L11, L17, L23).
//
// Latent KV layout (zero changes to vLLM FullAttentionSpec / Inkling2AttentionBackend):
//   kv_cache shape = [num_blocks, 2, page, 1, 128] uint8 (unit-scale E4M3FN)
//   c_lo_cache = kv_cache[:, 0] -> latent dims   0..127 (128 B/tok)
//   c_hi_cache = kv_cache[:, 1] -> latent dims 128..255 (128 B/tok)
//   Total: 256 B/tok (2.0x smaller than standard 2-KV-head FP8 KV, 1x HBM read for BOTH QK & PV!).
//
// Volta sm_70 decode/verify architecture (`k_rel_gla_dec`, single-warp 256D L1-resident):
//   - Mirrors `attn_rel_dec.cu`'s 94%-HBM2 design: ZERO `__syncthreads()` in the main loop,
//     ZERO SMEM staging of KV tiles (uses Volta 96 KB MaxL1 carveout so each 8 KB latent tile
//     stays 100% hot in L1 from QK -> PV!), and T_WARPS independent tile-warps per CTA.
//   - One warp owns all 256 dims (o_lo[4][8] + o_hi[4][8]) and pipelines `c_lo` / `c_hi` across
//     two register buffers `cr0[8], cr1[8]` -> HBM load of `c_hi` overlaps `QK_lo`, L1 read of
//     `c_lo` (PV) overlaps `QK_hi`, L1 read of `c_hi` (PV) overlaps `PV_lo`, and next tile's
//     `c_lo(t + T_WARPS)` overlaps `PV_hi`!
//   - When M_WARPS > 1, warps sharing tile-warp index `tw` lockstep via `bar_tw_sync` so all
//     M_WARPS warps hit the exact same 8 KB latent tile in L1 (1x HBM read across all 24 rows!).
//   - Guaranteed 0 bytes register spill across all instantiations.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int HD = 128;
constexpr int LAT_D = 256;
constexpr int HQ_GLA = 6;
constexpr int G_SUB = 3;
constexpr int TILE = 32;
constexpr int QS_GLA = LAT_D + 8;  // 264 halfs (528 B): 16B-aligned, 0-conflict uint2 Q reads
constexpr int PS = TILE + 8;       // 40 halfs (80 B): 16B-aligned, 0-conflict uint2/uint4 P reads
constexpr int TBS_DEC = 256;
constexpr int TBS_PF = 512;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ void mma_rc(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_rr(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

template <int M_WARPS, int T_WARPS>
__device__ __forceinline__ void bar_tw_sync(int tw) {
  if constexpr (M_WARPS > 1) {
    if constexpr (T_WARPS == 1) {
      __syncthreads();
    } else {
      if (tw == 0) {
        asm volatile("bar.sync 1, %0;" ::"r"(32 * M_WARPS) : "memory");
      } else {
        asm volatile("bar.sync 2, %0;" ::"r"(32 * M_WARPS) : "memory");
      }
    }
  } else {
    __syncwarp();
  }
}

// Exact bit-unpack of 4 E4M3FN bytes -> two fp16x2 (bytes 0,1 and 2,3), each = E4M3 * 2^-8.
__device__ __forceinline__ void e4m3x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned x0 = __byte_perm(w, 0u, 0x1404);
  const unsigned x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}

// Exact IEEE-754 RNE FP32 -> uint8 E4M3FN (saturating to +-448.0), bit-identical to fp8_sm70.py.
__device__ __forceinline__ uint8_t f32_to_e4m3_u8(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t sign = (bits >> 24) & 0x80u;
  const uint32_t abs_bits = bits & 0x7FFFFFFFu;
  if (abs_bits > 0x7F800000u) return static_cast<uint8_t>(sign | 0x7Fu);
  const uint32_t clamped = abs_bits < 0x43E00000u ? abs_bits : 0x43E00000u;
  const int32_t exp = static_cast<int32_t>(clamped >> 23);
  uint32_t code;
  if (exp >= 121) {
    const uint32_t mant = clamped & 0x007FFFFFu;
    const uint32_t norm_e = static_cast<uint32_t>(exp - 120);
    const uint32_t norm_round = ((mant >> 20) & 1u) + 0x7FFFFu;
    const uint32_t norm_rounded = (norm_e << 23) + mant + norm_round;
    code = norm_rounded >> 20;
    if (code > 0x7Eu) code = 0x7Eu;
  } else if (abs_bits < 0x3A800000u) {
    code = 0u;
  } else {
    const float abs_x = __uint_as_float(clamped);
    const float scaled = abs_x * 512.0f;
    code = __float_as_uint(scaled + 8388608.0f) - 0x4B000000u;
  }
  return static_cast<uint8_t>(sign | code);
}

__device__ __forceinline__ float ex2_fast(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float2 ld_h2_as_f2(const __half* p, bool aligned4) {
  if (aligned4) {
    unsigned u;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(u) : "l"(p));
    return __half22float2(*reinterpret_cast<const __half2*>(&u));
  }
  return make_float2(__half2float(p[0]), __half2float(p[1]));
}

// Conflict-free XOR swizzles for prefill 16-uint4 (128-half) rows in shared memory.
__device__ __forceinline__ int swiz_q(int r, int c) {
  return c ^ ((c >> 3) & 1) ^ (r & 7);
}

__device__ __forceinline__ int swiz_kv(int r, int c) {
  return c ^ ((c >> 3) & 1) ^ ((r & 3) | ((r >> 1) & 4));
}

// ============================================================================
// 1. Fused FP16 -> E4M3FN Latent Scatter into [num_blocks, page, 1, 128] x 2
// ============================================================================
__global__ void __launch_bounds__(64)
k_gla_scatter_fp8(
    const __half* __restrict__ c_raw,  // [T, 256] contiguous FP16
    uint8_t* __restrict__ c_lo,        // [num_blocks, page, 1, 128] uint8
    uint8_t* __restrict__ c_hi,        // [num_blocks, page, 1, 128] uint8
    const int64_t* __restrict__ kv_slot_mapping,
    int64_t c_raw_stride,
    int64_t sb,
    int64_t st,
    int page) {
  const int t = blockIdx.x;
  const int64_t slot = kv_slot_mapping[t];
  if (slot < 0) return;

  const int tid = threadIdx.x;
  const int is_hi = tid >> 5;
  const int col4 = (tid & 31) * 4;
  const __half* src = c_raw + (int64_t)t * c_raw_stride + (is_hi ? 128 : 0) + col4;
  const __half2 h01 = *reinterpret_cast<const __half2*>(src);
  const __half2 h23 = *reinterpret_cast<const __half2*>(src + 2);
  const float2 f01 = __half22float2(h01);
  const float2 f23 = __half22float2(h23);
  const uint32_t b0 = f32_to_e4m3_u8(f01.x);
  const uint32_t b1 = f32_to_e4m3_u8(f01.y);
  const uint32_t b2 = f32_to_e4m3_u8(f23.x);
  const uint32_t b3 = f32_to_e4m3_u8(f23.y);
  const uint32_t packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);

  const int64_t blk = slot / page;
  const int64_t off = slot - blk * page;
  uint8_t* dst = (is_hi ? c_hi : c_lo) + blk * sb + off * st + col4;
  *reinterpret_cast<uint32_t*>(dst) = packed;
}

// ============================================================================
// 2. Pack Q [T, 6, 128] -> [2, T, 3, 128] contiguous FP16 for single bmm
// ============================================================================
__global__ void __launch_bounds__(256)
k_gla_pack_q(
    const __half* __restrict__ q_in,
    __half* __restrict__ q_out,
    int T) {
  const int total_u4 = T * 6 * 16;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_u4) return;
  const int c = idx & 15;
  const int th = idx >> 4;
  const int head = th % 6;
  const int tok = th / 6;
  const int g = head / 3;
  const int sub = head - g * 3;
  const uint4 v = *reinterpret_cast<const uint4*>(q_in + ((size_t)tok * 6 + head) * 128 + c * 8);
  *reinterpret_cast<uint4*>(q_out + (((size_t)g * T + tok) * 3 + sub) * 128 + c * 8) = v;
}

// ============================================================================
// 3. Fused Unpack + Gate Epilogue: v_out [2, T, 3, 128] * sigmoid(gate [T, 768]) -> out [T, 768]
// ============================================================================
__global__ void __launch_bounds__(256)
k_gla_gate_epilogue(
    const __half* __restrict__ v_out,
    const __half* __restrict__ gate,
    __half* __restrict__ out,
    int T,
    int64_t stride_gate) {
  const int total_h2 = T * 6 * 64;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_h2) return;
  const int c2 = idx & 63;
  const int th = idx >> 6;
  const int head = th % 6;
  const int tok = th / 6;
  const int g = head / 3;
  const int sub = head - g * 3;
  const __half2 vh = *reinterpret_cast<const __half2*>(
      v_out + (((size_t)g * T + tok) * 3 + sub) * 128 + c2 * 2);
  const __half2 gh = *reinterpret_cast<const __half2*>(
      gate + (int64_t)tok * stride_gate + head * 128 + c2 * 2);
  const float2 vf = __half22float2(vh);
  const float2 gf = __half22float2(gh);
  const __half2 res = __floats2half2_rn(vf.x * sigmoid_f32(gf.x), vf.y * sigmoid_f32(gf.y));
  *reinterpret_cast<__half2*>(out + ((size_t)tok * 6 + head) * 128 + c2 * 2) = res;
}

// ============================================================================
// 4. Shared bias + causal mask helper for 32-token tile (global: window == 0)
// ============================================================================
__device__ __forceinline__ void apply_global_bias_mask_32(
    float* s,
    const __half* rrA,
    const __half* rrB,
    bool validA,
    bool validB,
    int posA,
    int posB,
    int rel_baseA,
    int rel_baseB,
    int tok0,
    int cb,
    int seqlen_k,
    int E,
    float sl2,
    bool rel_aligned4) {
  // Row A: entries 0, 1, 4, 5
  {
    const int k_min_ext = posA - E + 1;
    if (!validA) {
      s[0] = -INFINITY; s[1] = -INFINITY; s[4] = -INFINITY; s[5] = -INFINITY;
    } else if (tok0 + 31 < k_min_ext && tok0 + 31 <= posA && tok0 + 31 < seqlen_k) {
      s[0] *= sl2; s[1] *= sl2; s[4] *= sl2; s[5] *= sl2;
    } else if (tok0 >= k_min_ext && tok0 + 31 <= posA && tok0 + 31 < seqlen_k) {
      const float2 b01 = ld_h2_as_f2(rrA + (cb - rel_baseA), rel_aligned4);
      const float2 b45 = ld_h2_as_f2(rrA + (cb + 4 - rel_baseA), rel_aligned4);
      s[0] = fmaf(b01.x, LOG2E, s[0] * sl2);
      s[1] = fmaf(b01.y, LOG2E, s[1] * sl2);
      s[4] = fmaf(b45.x, LOG2E, s[4] * sl2);
      s[5] = fmaf(b45.y, LOG2E, s[5] * sl2);
    } else {
      const int idxA[4] = {0, 1, 4, 5};
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int e = idxA[u];
        const int col = cb + (e & 1) + (e & 4);
        const int dist = posA - col;
        if (col < seqlen_k && dist >= 0) {
          float v = s[e] * sl2;
          if (dist < E) v = fmaf(__half2float(rrA[col - rel_baseA]), LOG2E, v);
          s[e] = v;
        } else {
          s[e] = -INFINITY;
        }
      }
    }
  }
  // Row B: entries 2, 3, 6, 7
  {
    const int k_min_ext = posB - E + 1;
    if (!validB) {
      s[2] = -INFINITY; s[3] = -INFINITY; s[6] = -INFINITY; s[7] = -INFINITY;
    } else if (tok0 + 31 < k_min_ext && tok0 + 31 <= posB && tok0 + 31 < seqlen_k) {
      s[2] *= sl2; s[3] *= sl2; s[6] *= sl2; s[7] *= sl2;
    } else if (tok0 >= k_min_ext && tok0 + 31 <= posB && tok0 + 31 < seqlen_k) {
      const float2 b01 = ld_h2_as_f2(rrB + (cb - rel_baseB), rel_aligned4);
      const float2 b45 = ld_h2_as_f2(rrB + (cb + 4 - rel_baseB), rel_aligned4);
      s[2] = fmaf(b01.x, LOG2E, s[2] * sl2);
      s[3] = fmaf(b01.y, LOG2E, s[3] * sl2);
      s[6] = fmaf(b45.x, LOG2E, s[6] * sl2);
      s[7] = fmaf(b45.y, LOG2E, s[7] * sl2);
    } else {
      const int idxB[4] = {2, 3, 6, 7};
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int e = idxB[u];
        const int col = cb + (e & 1) + (e & 4);
        const int dist = posB - col;
        if (col < seqlen_k && dist >= 0) {
          float v = s[e] * sl2;
          if (dist < E) v = fmaf(__half2float(rrB[col - rel_baseB]), LOG2E, v);
          s[e] = v;
        } else {
          s[e] = -INFINITY;
        }
      }
    }
  }
}

// ============================================================================
// 5. High-Bandwidth Decode/Verify GLA-g4 Kernel (Single-Warp 256D, MaxL1, 0 Syncs)
// ============================================================================
template <int M_WARPS, int T_WARPS>
struct GlaDecSM {
  static constexpr int NT = 32 * M_WARPS * T_WARPS;
  static constexpr int R_CTA = 8 * M_WARPS;
  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS_DEC * (int)sizeof(int);
  static constexpr int RP_OFF = TB_BYTES;
  static constexpr int RP_BYTES = ((R_CTA * (int)sizeof(const __half*) + 15) & ~15);
  static constexpr int Q_OFF = RP_OFF + RP_BYTES;
  static constexpr int Q_BYTES = R_CTA * QS_GLA * (int)sizeof(__half);
  static constexpr int P_OFF = (Q_OFF + Q_BYTES + 15) & ~15;
  static constexpr int P_BYTES = M_WARPS * T_WARPS * 8 * PS * (int)sizeof(__half);
  static constexpr int MAIN_END = (P_OFF + P_BYTES + 15) & ~15;

  // Two-pass merge buffer (128 dims at a time -> half SMEM, keeps total <= 26 KB for 96 KB MaxL1!)
  static constexpr int O_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * HD * (int)sizeof(float)) : 0;
  static constexpr int ML_OFF = (MAIN_END > O_BYTES ? MAIN_END : O_BYTES);
  static constexpr int ML_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int RS_OFF = ML_OFF + ML_BYTES;
  static constexpr int RS_BYTES = (T_WARPS > 1) ? (M_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int TOTAL = ((RS_OFF + RS_BYTES + 15) & ~15);
};

template <int M_WARPS, int T_WARPS>
__device__ __forceinline__ void write_empty_gla_split(
    float2* __restrict__ ml_part, int num_splits, int s, int q_start, int first_row, int eff_nr, int tid) {
  constexpr int NT = 32 * M_WARPS * T_WARPS;
  constexpr int R_CTA = 8 * M_WARPS;
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / HQ_GLA;
      const int head = gr - tok * HQ_GLA;
      const int token = q_start + tok;
      const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
      __stcg(ml_part + slot, make_float2(-INFINITY, 0.f));
    }
  }
}

template <int M_WARPS, int T_WARPS>
__global__ void __launch_bounds__(32 * M_WARPS * T_WARPS, (M_WARPS * T_WARPS <= 4) ? 2 : 1)
k_rel_gla_dec(
    const __half* __restrict__ q_lat,   // [2, T_total, 3, 256] contiguous FP16
    const uint8_t* __restrict__ c_lo,   // [num_blocks, page, 1, 128] uint8
    const uint8_t* __restrict__ c_hi,   // [num_blocks, page, 1, 128] uint8
    const __half* __restrict__ rel_logits,
    const __half* __restrict__ sinks,
    __half* __restrict__ o_lat,         // [2, T_total, 3, 256] contiguous FP16 (when S==1)
    float* __restrict__ o_part,         // [T_total, 6, S, 256] float (when S>1)
    float2* __restrict__ ml_part,       // [T_total, 6, S] float2 (when S>1)
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cu_seqlens_q,
    int T_total,
    int E,
    int E_pad,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
    int num_splits,
    int64_t sb,
    int64_t st,
    int64_t stride_rel_t,
    int64_t stride_rel_h,
    int64_t stride_bt,
    int64_t stride_sink,
    float sl2,
    float vsc,
    bool rel_aligned4) {
  using L_ = GlaDecSM<M_WARPS, T_WARPS>;
  constexpr int NT = L_::NT;
  constexpr int R_CTA = L_::R_CTA;
  extern __shared__ __align__(16) uint8_t sm[];

  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  __half* Qs = reinterpret_cast<__half*>(sm + L_::Q_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);

  const int s = blockIdx.x;
  const int m_tile = blockIdx.y;
  const int b = blockIdx.z;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mw = warp / T_WARPS;
  const int tw = warp % T_WARPS;

  const int q_start = cu_seqlens_q[b];
  const int q_end = cu_seqlens_q[b + 1];
  const int q_len = q_end - q_start;
  const int eff_q_len = max(0, min(q_len, num_actual_tokens - q_start));
  const int eff_nr = eff_q_len * HQ_GLA;
  const int first_row = m_tile * R_CTA;
  if (first_row >= eff_nr) return;

  const int seqlen_k = cache_seqlens[b];
  const int past = seqlen_k - q_len;
  const int last_row = min(first_row + R_CTA, eff_nr) - 1;
  const int last_tok = last_row / HQ_GLA;
  const int pos_hi = past + last_tok;

  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int T_tiles = max((k_hi + TILE - 1) / TILE, 0);

  if (T_tiles <= 0) {
    if (num_splits == 1) {
      for (int i = tid; i < R_CTA * 32; i += NT) {
        const int r = i >> 5;
        const int c = i & 31;
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_GLA;
          const int head = gr - tok * HQ_GLA;
          const int g = head / G_SUB;
          const int sub = head - g * G_SUB;
          const int token = q_start + tok;
          __half* dst = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D;
          *reinterpret_cast<uint4*>(dst + c * 8) = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else {
      write_empty_gla_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, q_start, first_row, eff_nr, tid);
    }
    return;
  }

  const int Sb = max(1, min(num_splits, T_tiles));
  if (s >= Sb) {
    write_empty_gla_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, q_start, first_row, eff_nr, tid);
    return;
  }

  const int t_start = (int)((long long)T_tiles * s / Sb);
  const int t_end = (int)((long long)T_tiles * (s + 1) / Sb);

  // Cache block table, RelPtrs, and Q_lat [R_CTA, 256] in SMEM.
  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  for (int i = tid; i < min(max_blocks, TBS_DEC); i += NT) Ts[i] = tb[i];
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / HQ_GLA;
      const int head = gr - tok * HQ_GLA;
      const int token = q_start + tok;
      RelPtrs[r] = rel_logits + (size_t)token * (size_t)stride_rel_t + (size_t)head * (size_t)stride_rel_h;
    } else {
      RelPtrs[r] = rel_logits;
    }
  }
  for (int i = tid; i < R_CTA * 32; i += NT) {
    const int r = i >> 5;
    const int c = (i & 31) * 8;
    const int gr = first_row + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (gr < eff_nr) {
      const int tok = gr / HQ_GLA;
      const int head = gr - tok * HQ_GLA;
      const int g = head / G_SUB;
      const int sub = head - g * G_SUB;
      const int token = q_start + tok;
      const __half* qp = q_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + c;
      v = ldg16(qp);
    }
    *reinterpret_cast<uint4*>(Qs + r * QS_GLA + c) = v;
  }
  __syncthreads();

  // Volta HMMA.884 lane geometry (1 warp owns ALL 256 latent dims!)
  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;          // 0..7
  const int r0 = (lane & 1) + 4 * hi;    // 0, 1, 4, 5
  const int c0 = lane & 2;               // 0 or 2
  const int ktok = 8 * qp + arow;        // 0..31: QK token index
  const int vdim = 32 * qp + 16 * hi;    // 0..112 step 16: PV 16-dim block

  const int lrA = mw * 8 + r0;
  const int lrB = lrA + 2;
  const int grA = first_row + lrA;
  const int grB = first_row + lrB;
  const bool validA = grA < eff_nr;
  const bool validB = grB < eff_nr;

  int posA = -1, posB = -1, rel_baseA = 0, rel_baseB = 0;
  if (validA) {
    const int tokA = grA / HQ_GLA;
    posA = past + tokA;
    const int tile_endA = min(((tokA / 128) + 1) * 128, q_len);
    rel_baseA = ((past + tile_endA + 127) / 128) * 128 - E_pad;
  }
  if (validB) {
    const int tokB = grB / HQ_GLA;
    posB = past + tokB;
    const int tile_endB = min(((tokB / 128) + 1) * 128, q_len);
    rel_baseB = ((past + tile_endB + 127) / 128) * 128 - E_pad;
  }

  const __half* qrow_lo = Qs + (mw * 8 + arow) * QS_GLA;
  const __half* qrow_hi = qrow_lo + HD;
  __half* Pw = Ps + warp * 8 * PS;
  const int pmask = page - 1;

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TBS_DEC ? Ts[pi] : tb[pi]);
  };

  uint4 cr[8];

  auto ld_qk_half = [&](const uint8_t* base, int t_idx) {
    const int tok0 = t_idx * TILE;
    if (page_shift >= 5 && tok0 + 31 < seqlen_k) {
      const int64_t pg = get_pg(tok0 >> page_shift);
      const int sl = (tok0 & pmask) + ktok;
      const uint8_t* p = base + pg * sb + (int64_t)sl * st;
#pragma unroll
      for (int i = 0; i < 8; i++) cr[i] = ldg16(p + 16 * i);
    } else {
      const int k_idx = tok0 + ktok;
      if (k_idx < seqlen_k) {
        const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
        const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
        const int64_t pg = get_pg(pi);
        const uint8_t* p = base + pg * sb + (int64_t)sl * st;
#pragma unroll
        for (int i = 0; i < 8; i++) cr[i] = ldg16(p + 16 * i);
      } else {
#pragma unroll
        for (int i = 0; i < 8; i++) cr[i] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  auto ld_pv_half = [&](const uint8_t* base, int t_idx) {
    const int tok0 = t_idx * TILE;
    if (page_shift >= 5 && tok0 + 31 < seqlen_k) {
      const int64_t pg = get_pg(tok0 >> page_shift);
      const int s0 = (tok0 & pmask) + l3;
      const uint8_t* p = base + pg * sb + (int64_t)s0 * st + vdim;
#pragma unroll
      for (int i = 0; i < 8; i++) cr[i] = ldg16(p + (int64_t)(4 * i) * st);
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int v_idx = tok0 + l3 + 4 * i;
        if (v_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (v_idx >> page_shift) : (v_idx / page);
          const int sl = (page_shift >= 0) ? (v_idx & pmask) : (v_idx - pi * page);
          const int64_t pg = get_pg(pi);
          cr[i] = ldg16(base + pg * sb + (int64_t)sl * st + vdim);
        } else {
          cr[i] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  // Preload first tile's c_lo into cr.
  int t = t_start + tw;
  if (t < t_end) {
    ld_qk_half(c_lo, t);
  }

  float o_lo[4][8];
  float o_hi[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) {
      o_lo[j][e] = 0.f;
      o_hi[j][e] = 0.f;
    }
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;

  // Main loop: ZERO __syncthreads(), dual-acc QK (sa+sb), compact cr[8] (~165 regs -> 12 warps/SM)!
  for (; t < t_end; t += T_WARPS) {
    const int tok0 = t * TILE;
    const bool more = (t + T_WARPS) < t_end;

    float sa[8], sb_acc[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      sa[e] = 0.f;
      sb_acc[e] = 0.f;
    }

    // 1. QK_lo (32 HMMA with dual accumulators sa/sb_acc), then load c_hi
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint2 a = *reinterpret_cast<const uint2*>(qrow_lo + 4 * (4 * i + j));
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rc((i & 1) ? sb_acc : sa, a.x, a.y, b0, b1);
      }
    }
    ld_qk_half(c_hi, t);

    // 2. QK_hi (32 HMMA with dual accumulators sa/sb_acc), then issue L1-hot PV load of c_lo!
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint2 a = *reinterpret_cast<const uint2*>(qrow_hi + 4 * (4 * i + j));
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rc((i & 1) ? sb_acc : sa, a.x, a.y, b0, b1);
      }
    }
    ld_pv_half(c_lo, t);  // 100% L1 hit while softmax executes!
#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] += sb_acc[e];

    // 3. Scale, sheared relative bias, causal mask, online softmax
    const int cb = tok0 + 8 * qp + c0;
    apply_global_bias_mask_32(
        sa, RelPtrs[lrA], RelPtrs[lrB], validA, validB, posA, posB,
        rel_baseA, rel_baseB, tok0, cb, seqlen_k, E, sl2, rel_aligned4);

    float x0 = fmaxf(fmaxf(sa[0], sa[1]), fmaxf(sa[4], sa[5]));
    float x2 = fmaxf(fmaxf(sa[2], sa[3]), fmaxf(sa[6], sa[7]));
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
      x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
    }
    const float n0 = fmaxf(m0, x0);
    const float n2 = fmaxf(m2, x2);
    const float a0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - n0);
    const float a2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - n2);
    const float z0 = (n0 == -INFINITY) ? 0.f : n0;
    const float z2 = (n2 == -INFINITY) ? 0.f : n2;

#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] = ex2_fast(sa[e] - ((e & 2) ? z2 : z0));
    l0 = l0 * a0 + (sa[0] + sa[1]) + (sa[4] + sa[5]);
    l2 = l2 * a2 + (sa[2] + sa[3]) + (sa[6] + sa[7]);
    m0 = n0;
    m2 = n2;

#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float af = (e & 2) ? a2 : a0;
        o_lo[j][e] *= af;
        o_hi[j][e] *= af;
      }

    const int pc = 8 * qp + c0;
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(sa[0], sa[1]);
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(sa[4], sa[5]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(sa[2], sa[3]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(sa[6], sa[7]);
    __syncwarp();

    // 4. PV_lo (32 HMMA using cr), then load L1-hot c_hi into cr!
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rr(o_lo[j], a.x, a.y, b0, b1);
      }
    }
    ld_pv_half(c_hi, t);  // 100% L1 hit!

    // 5. PV_hi (32 HMMA using cr), then prefetch next tile's c_lo into cr!
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rr(o_hi[j], a.x, a.y, b0, b1);
      }
    }
    if (more) ld_qk_half(c_lo, t + T_WARPS);

    // Lockstep M_WARPS sharing tile-warp index tw so all M_WARPS hit the same 8 KB L1 tile!
    bar_tw_sync<M_WARPS, T_WARPS>(tw);
  }

  // Finalize per-warp row sum across the 8 lanes sharing each row.
#pragma unroll
  for (int d = 2; d <= 8; d <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, d);
    l2 += __shfl_xor_sync(0xffffffffu, l2, d);
  }

  // ---- Epilogue ----
  if constexpr (T_WARPS == 1) {
    if (num_splits == 1) {
      float rs0 = 0.f, rs2 = 0.f;
      if (validA && l0 > 0.f) {
        const int tokA = grA / HQ_GLA;
        const int headA = grA - tokA * HQ_GLA;
        const int tokenA = q_start + tokA;
        const float lse_e = (m0 + log2f(l0)) * LN2;
        const float sink_v = __half2float(sinks[(size_t)tokenA * (size_t)stride_sink + headA]);
        rs0 = (vsc / l0) * sigmoid_f32(lse_e - sink_v);
      }
      if (validB && l2 > 0.f) {
        const int tokB = grB / HQ_GLA;
        const int headB = grB - tokB * HQ_GLA;
        const int tokenB = q_start + tokB;
        const float lse_e = (m2 + log2f(l2)) * LN2;
        const float sink_v = __half2float(sinks[(size_t)tokenB * (size_t)stride_sink + headB]);
        rs2 = (vsc / l2) * sigmoid_f32(lse_e - sink_v);
      }
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int gr = sr ? grB : grA;
        if (gr < eff_nr) {
          const float rsc = sr ? rs2 : rs0;
          const int tok = gr / HQ_GLA;
          const int head = gr - tok * HQ_GLA;
          const int g = head / G_SUB;
          const int sub = head - g * G_SUB;
          const int token = q_start + tok;
          __half* ob = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            *reinterpret_cast<__half2*>(ob + dlo) =
                __floats2half2_rn(o_lo[j][2 * sr] * rsc, o_lo[j][2 * sr + 1] * rsc);
            *reinterpret_cast<__half2*>(ob + dlo + 16) =
                __floats2half2_rn(o_lo[j][4 + 2 * sr] * rsc, o_lo[j][5 + 2 * sr] * rsc);
            *reinterpret_cast<__half2*>(ob + HD + dlo) =
                __floats2half2_rn(o_hi[j][2 * sr] * rsc, o_hi[j][2 * sr + 1] * rsc);
            *reinterpret_cast<__half2*>(ob + HD + dlo + 16) =
                __floats2half2_rn(o_hi[j][4 + 2 * sr] * rsc, o_hi[j][5 + 2 * sr] * rsc);
          }
        }
      }
    } else {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int gr = sr ? grB : grA;
        if (gr < eff_nr) {
          const int tok = gr / HQ_GLA;
          const int head = gr - tok * HQ_GLA;
          const int token = q_start + tok;
          const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
          float* pob = o_part + slot * LAT_D;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            __stcg(reinterpret_cast<float2*>(pob + dlo), make_float2(o_lo[j][2 * sr], o_lo[j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + dlo + 16),
                   make_float2(o_lo[j][4 + 2 * sr], o_lo[j][5 + 2 * sr]));
            __stcg(reinterpret_cast<float2*>(pob + HD + dlo),
                   make_float2(o_hi[j][2 * sr], o_hi[j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + HD + dlo + 16),
                   make_float2(o_hi[j][4 + 2 * sr], o_hi[j][5 + 2 * sr]));
          }
          if ((lane & 14) == 0) {
            __stcg(ml_part + slot, make_float2(sr ? m2 : m0, sr ? l2 : l0));
          }
        }
      }
    }
  } else {
    // T_WARPS > 1: two-pass 128D merge in compact SMEM (keeps SMEM <= 26 KB for 96 KB L1!).
    float* Os = reinterpret_cast<float*>(sm);
    float* MLs = reinterpret_cast<float*>(sm + L_::ML_OFF);
    float* RSs = reinterpret_cast<float*>(sm + L_::RS_OFF);

    __syncthreads();
    if ((lane & 14) == 0) {
      const int ml_base = (mw * T_WARPS + tw) * 16;
      MLs[ml_base + 2 * r0] = m0;
      MLs[ml_base + 2 * r0 + 1] = l0;
      MLs[ml_base + 2 * (r0 + 2)] = m2;
      MLs[ml_base + 2 * (r0 + 2) + 1] = l2;
    }
    __syncthreads();

    float M0 = -INFINITY, M2 = -INFINITY;
#pragma unroll
    for (int k = 0; k < T_WARPS; k++) {
      M0 = fmaxf(M0, MLs[(mw * T_WARPS + k) * 16 + 2 * r0]);
      M2 = fmaxf(M2, MLs[(mw * T_WARPS + k) * 16 + 2 * (r0 + 2)]);
    }
    const float f0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - M0);
    const float f2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - M2);

    for (int r = tid; r < R_CTA; r += NT) {
      const int rm = r >> 3;
      const int lr = r & 7;
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) M = fmaxf(M, MLs[(rm * T_WARPS + k) * 16 + 2 * lr]);
      float Lt = 0.f;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) {
        const float mk = MLs[(rm * T_WARPS + k) * 16 + 2 * lr];
        if (mk != -INFINITY) Lt += MLs[(rm * T_WARPS + k) * 16 + 2 * lr + 1] * ex2_fast(mk - M);
      }
      RSs[2 * r] = M;
      RSs[2 * r + 1] = Lt;
    }

    float* Op = Os + (size_t)(mw * T_WARPS + tw) * 8 * HD;
    // Pass 0: o_lo (dims 0..127), Pass 1: o_hi (dims 128..255)
#pragma unroll
    for (int half_idx = 0; half_idx < 2; half_idx++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int sr = 0; sr < 2; sr++) {
          const int local_r = r0 + 2 * sr;
          const float f = sr ? f2 : f0;
          const int dlo = 32 * qp + 4 * j + c0;
          const float a = (half_idx == 0) ? o_lo[j][2 * sr] : o_hi[j][2 * sr];
          const float b_v = (half_idx == 0) ? o_lo[j][2 * sr + 1] : o_hi[j][2 * sr + 1];
          const float c_v = (half_idx == 0) ? o_lo[j][4 + 2 * sr] : o_hi[j][4 + 2 * sr];
          const float d_v = (half_idx == 0) ? o_lo[j][5 + 2 * sr] : o_hi[j][5 + 2 * sr];
          *reinterpret_cast<float2*>(Op + local_r * HD + dlo) = make_float2(a * f, b_v * f);
          *reinterpret_cast<float2*>(Op + local_r * HD + dlo + 16) = make_float2(c_v * f, d_v * f);
        }
      }
      __syncthreads();

      for (int i = tid; i < R_CTA * (HD / 4); i += NT) {
        const int r = i / (HD / 4);
        const int gr = first_row + r;
        if (gr >= eff_nr) continue;
        const int rm = r >> 3;
        const int lr = r & 7;
        const int d = (i % (HD / 4)) * 4;
        float4 acc = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS) * 8 * HD + lr * HD + d);
#pragma unroll
        for (int k = 1; k < T_WARPS; k++) {
          const float4 cv = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS + k) * 8 * HD + lr * HD + d);
          acc.x += cv.x; acc.y += cv.y; acc.z += cv.z; acc.w += cv.w;
        }
        const int tok = gr / HQ_GLA;
        const int head = gr - tok * HQ_GLA;
        const int token = q_start + tok;
        if (num_splits == 1) {
          const float M = RSs[2 * r];
          const float Lt = RSs[2 * r + 1];
          float rsc = 0.f;
          if (Lt > 0.f) {
            const float lse_e = (M + log2f(Lt)) * LN2;
            const float sink_v = __half2float(sinks[(size_t)token * (size_t)stride_sink + head]);
            rsc = (vsc / Lt) * sigmoid_f32(lse_e - sink_v);
          }
          const int g = head / G_SUB;
          const int sub = head - g * G_SUB;
          __half2* dst = reinterpret_cast<__half2*>(
              o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + half_idx * HD + d);
          dst[0] = __floats2half2_rn(acc.x * rsc, acc.y * rsc);
          dst[1] = __floats2half2_rn(acc.z * rsc, acc.w * rsc);
        } else {
          const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
          __stcg(reinterpret_cast<float4*>(o_part + slot * LAT_D + half_idx * HD + d), acc);
        }
      }
      if (half_idx == 0) __syncthreads();
    }
    if (num_splits > 1) {
      for (int r = tid; r < R_CTA; r += NT) {
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_GLA;
          const int head = gr - tok * HQ_GLA;
          const int token = q_start + tok;
          const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
          __stcg(ml_part + slot, make_float2(RSs[2 * r], RSs[2 * r + 1]));
        }
      }
    }
  }
}

// ============================================================================
// 5b. High-Compute 2x8 Decode/Verify GLA-g4 Kernel (Q >= 3: 16 rows/warp, D_WARPS=2, 0 e4m3 in loop!)
//     M_WARPS=2 (R_CTA=32 rows), D_WARPS=2 -> 128 threads/CTA, ~40 KB SMEM -> 2 CTAs/SM (MaxShared).
//     Unpacks E4M3FN->FP16 ONCE per 32-token tile, reuses every C LDS.128 across 2 Q strips,
//     and overlaps HBM prefetch of next tile with softmax (dw=1) and PV!
// ============================================================================
struct GlaDec2x8SM {
  static constexpr int M_WARPS = 2;
  static constexpr int R_CTA = 16 * M_WARPS;  // 32 rows
  static constexpr int NT = 64 * M_WARPS;     // 128 threads (4 warps)
  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS_DEC * (int)sizeof(int);
  static constexpr int RP_OFF = (TB_OFF + TB_BYTES + 15) & ~15;
  static constexpr int RP_BYTES = (R_CTA * (int)sizeof(const __half*) + 15) & ~15;
  static constexpr int QLO_OFF = (RP_OFF + RP_BYTES + 15) & ~15;
  static constexpr int Q_HALF_BYTES = R_CTA * HD * (int)sizeof(__half);  // 8 KB
  static constexpr int QHI_OFF = QLO_OFF + Q_HALF_BYTES;
  static constexpr int CLO_OFF = QHI_OFF + Q_HALF_BYTES;
  static constexpr int C_HALF_BYTES = TILE * HD * (int)sizeof(__half);   // 8 KB
  static constexpr int CHI_OFF = CLO_OFF + C_HALF_BYTES;
  static constexpr int S_OFF = CHI_OFF + C_HALF_BYTES;
  static constexpr int S_BYTES = M_WARPS * 2 * 8 * 32 * (int)sizeof(float);  // 4 KB
  static constexpr int P_OFF = S_OFF + S_BYTES;
  static constexpr int P_BYTES = M_WARPS * 2 * 8 * PS * (int)sizeof(__half); // 2.56 KB
  static constexpr int A_OFF = (P_OFF + P_BYTES + 15) & ~15;
  static constexpr int A_BYTES = (M_WARPS * 16 * (int)sizeof(float) + 15) & ~15;
  static constexpr int TOTAL = A_OFF + A_BYTES;
};

__global__ void __launch_bounds__(128, 2)
k_rel_gla_dec_2x8(
    const __half* __restrict__ q_lat,
    const uint8_t* __restrict__ c_lo,
    const uint8_t* __restrict__ c_hi,
    const __half* __restrict__ rel_logits,
    const __half* __restrict__ sinks,
    __half* __restrict__ o_lat,
    float* __restrict__ o_part,
    float2* __restrict__ ml_part,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cu_seqlens_q,
    int T_total,
    int E,
    int E_pad,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
    int num_splits,
    int64_t sb,
    int64_t st,
    int64_t stride_rel_t,
    int64_t stride_rel_h,
    int64_t stride_bt,
    int64_t stride_sink,
    float sl2,
    float vsc,
    bool rel_aligned4) {
  using L_ = GlaDec2x8SM;
  constexpr int NT = L_::NT;
  constexpr int HALF_NT = 64;
  constexpr int R_CTA = L_::R_CTA;
  extern __shared__ __align__(16) uint8_t sm[];

  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  uint4* Qlo_u4 = reinterpret_cast<uint4*>(sm + L_::QLO_OFF);
  uint4* Qhi_u4 = reinterpret_cast<uint4*>(sm + L_::QHI_OFF);
  uint4* Clo_u4 = reinterpret_cast<uint4*>(sm + L_::CLO_OFF);
  uint4* Chi_u4 = reinterpret_cast<uint4*>(sm + L_::CHI_OFF);
  float* Ss = reinterpret_cast<float*>(sm + L_::S_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);
  float* Alpha = reinterpret_cast<float*>(sm + L_::A_OFF);

  const int s = blockIdx.x;
  const int m_tile = blockIdx.y;
  const int b = blockIdx.z;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mw = warp >> 1;  // 0 or 1
  const int dw = warp & 1;   // 0: dims 0..127, 1: dims 128..255

  const int q_start = cu_seqlens_q[b];
  const int q_end = cu_seqlens_q[b + 1];
  const int q_len = q_end - q_start;
  const int eff_q_len = max(0, min(q_len, num_actual_tokens - q_start));
  const int eff_nr = eff_q_len * HQ_GLA;
  const int first_row = m_tile * R_CTA;
  if (first_row >= eff_nr) return;

  const int seqlen_k = cache_seqlens[b];
  const int past = seqlen_k - q_len;
  const int last_row = min(first_row + R_CTA, eff_nr) - 1;
  const int last_tok = last_row / HQ_GLA;
  const int pos_hi = past + last_tok;
  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int T_tiles = max((k_hi + TILE - 1) / TILE, 0);

  if (T_tiles <= 0) {
    if (num_splits == 1) {
      for (int i = tid; i < R_CTA * 16; i += NT) {
        const int r = i >> 4;
        const int c = i & 15;
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_GLA;
          const int head = gr - tok * HQ_GLA;
          const int g = head / G_SUB;
          const int sub = head - g * G_SUB;
          const int token = q_start + tok;
          __half* dst = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D;
          *reinterpret_cast<uint4*>(dst + c * 8) = make_uint4(0u, 0u, 0u, 0u);
          *reinterpret_cast<uint4*>(dst + 128 + c * 8) = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else {
      for (int r = tid; r < R_CTA; r += NT) {
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_GLA;
          const int head = gr - tok * HQ_GLA;
          const int token = q_start + tok;
          const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
          __stcg(ml_part + slot, make_float2(-INFINITY, 0.f));
        }
      }
    }
    return;
  }

  const int Sb = max(1, min(num_splits, T_tiles));
  if (s >= Sb) {
    for (int r = tid; r < R_CTA; r += NT) {
      const int gr = first_row + r;
      if (gr < eff_nr) {
        const int tok = gr / HQ_GLA;
        const int head = gr - tok * HQ_GLA;
        const int token = q_start + tok;
        const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
        __stcg(ml_part + slot, make_float2(-INFINITY, 0.f));
      }
    }
    return;
  }

  const int t_start = (int)((long long)T_tiles * s / Sb);
  const int t_end = (int)((long long)T_tiles * (s + 1) / Sb);

  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  for (int i = tid; i < min(max_blocks, TBS_DEC); i += NT) Ts[i] = tb[i];
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / HQ_GLA;
      const int head = gr - tok * HQ_GLA;
      const int token = q_start + tok;
      RelPtrs[r] = rel_logits + (size_t)token * (size_t)stride_rel_t + (size_t)head * (size_t)stride_rel_h;
    } else {
      RelPtrs[r] = rel_logits;
    }
  }

  const int half_tid = mw * 32 + lane;
  uint4* Q_my_u4 = (dw == 0) ? Qlo_u4 : Qhi_u4;
  uint4* C_my_u4 = (dw == 0) ? Clo_u4 : Chi_u4;
  const uint8_t* C_my_gmem = (dw == 0) ? c_lo : c_hi;

#pragma unroll
  for (int u = 0; u < 8; u++) {
    const int i = half_tid + u * HALF_NT;
    const int r = i >> 4;
    const int c = i & 15;
    const int gr = first_row + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (gr < eff_nr) {
      const int tok = gr / HQ_GLA;
      const int head = gr - tok * HQ_GLA;
      const int g = head / G_SUB;
      const int sub = head - g * G_SUB;
      const int token = q_start + tok;
      const __half* qp = q_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + dw * HD + c * 8;
      v = ldg16(qp);
    }
    Q_my_u4[r * 16 + swiz_q(r, c)] = v;
  }
  __syncthreads();

  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;
  const int r0 = (lane & 1) + 4 * hi;
  const int c0 = lane & 2;
  const int ktok = 8 * qp + arow;

  // Strip 0: local rows mw*16 + 0..7; Strip 1: local rows mw*16 + 8..15
  const int lr0A = mw * 16 + r0;
  const int lr0B = lr0A + 2;
  const int lr1A = mw * 16 + 8 + r0;
  const int lr1B = lr1A + 2;
  const int gr0A = first_row + lr0A;
  const int gr0B = first_row + lr0B;
  const int gr1A = first_row + lr1A;
  const int gr1B = first_row + lr1B;
  const bool valid0A = gr0A < eff_nr;
  const bool valid0B = gr0B < eff_nr;
  const bool valid1A = gr1A < eff_nr;
  const bool valid1B = gr1B < eff_nr;
  const bool has_s0 = (first_row + mw * 16) < eff_nr;
  const bool has_s1 = (first_row + mw * 16 + 8) < eff_nr;

  auto init_row_meta = [&](bool valid, int gr, int& pos, int& rbase) {
    if (valid) {
      const int tok = gr / HQ_GLA;
      pos = past + tok;
      const int tile_end = min(((tok / 128) + 1) * 128, q_len);
      rbase = ((past + tile_end + 127) / 128) * 128 - E_pad;
    } else {
      pos = -1;
      rbase = 0;
    }
  };
  int pos0A, pos0B, pos1A, pos1B;
  int rb0A, rb0B, rb1A, rb1B;
  init_row_meta(valid0A, gr0A, pos0A, rb0A);
  init_row_meta(valid0B, gr0B, pos0B, rb0B);
  init_row_meta(valid1A, gr1A, pos1A, rb1A);
  init_row_meta(valid1B, gr1B, pos1B, rb1B);

  const int qrow0 = mw * 16 + arow;
  const int qrow1 = mw * 16 + 8 + arow;
  const int q_swiz_base = arow & 7;  // (mw*16 + arow) & 7 == (mw*16 + 8 + arow) & 7 == arow & 7!
  const int k_swiz_base = (ktok & 3) | ((ktok >> 1) & 4);
  const int vc0 = 4 * qp + 2 * hi;
  const int vc1 = vc0 + 1;
  const int vc0_f = vc0 ^ ((vc0 >> 3) & 1);
  const int vc1_f = vc1 ^ ((vc1 >> 3) & 1);
  const int pmask = page - 1;

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TBS_DEC ? Ts[pi] : tb[pi]);
  };

  uint4 c_pref[4];
  auto prefetch_c = [&](int t_idx) {
    const int tok0 = t_idx * TILE;
    const bool single_page_ok = (page_shift >= 5) && (tok0 + 31 < seqlen_k);
    const int64_t single_pg = single_page_ok ? get_pg(tok0 >> page_shift) : 0;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = half_tid + u * HALF_NT;
      const int row = idx >> 3;
      const int col_u = idx & 7;
      if (single_page_ok) {
        const int sl = (tok0 & pmask) + row;
        c_pref[u] = ldg16(C_my_gmem + single_pg * sb + (int64_t)sl * st + 16 * col_u);
      } else {
        const int k_idx = tok0 + row;
        if (k_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
          const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
          const int64_t pg = get_pg(pi);
          c_pref[u] = ldg16(C_my_gmem + pg * sb + (int64_t)sl * st + 16 * col_u);
        } else {
          c_pref[u] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  float o0[4][8], o1[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) {
      o0[j][e] = 0.f;
      o1[j][e] = 0.f;
    }
  float m0A = -INFINITY, m0B = -INFINITY, l0A = 0.f, l0B = 0.f;
  float m1A = -INFINITY, m1B = -INFINITY, l1A = 0.f, l1B = 0.f;

  prefetch_c(t_start);

  for (int t = t_start; t < t_end; t++) {
    const int tok0 = t * TILE;
    const bool more = (t + 1) < t_end;

    // 1. Unpack prefetched FP8 C tile -> swizzled FP16 SMEM ONCE per tile
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = half_tid + u * HALF_NT;
      const int row = idx >> 3;
      const int col_u = idx & 7;
      const uint4 w = c_pref[u];
      unsigned lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
      e4m3x4(w.x, lo0, hi0);
      e4m3x4(w.y, lo1, hi1);
      C_my_u4[row * 16 + swiz_kv(row, 2 * col_u)] = make_uint4(lo0, hi0, lo1, hi1);
      e4m3x4(w.z, lo2, hi2);
      e4m3x4(w.w, lo3, hi3);
      C_my_u4[row * 16 + swiz_kv(row, 2 * col_u + 1)] = make_uint4(lo2, hi2, lo3, hi3);
    }
    __syncthreads();

    // 2. 2x8 QK: every C LDS.128 reused across BOTH strip 0 and strip 1!
    float s0[8], s1[8];
#pragma unroll
    for (int e = 0; e < 8; e++) { s0[e] = 0.f; s1[e] = 0.f; }
    if (has_s0) {
      if (has_s1) {
#pragma unroll 2
        for (int k8 = 0; k8 < 16; k8++) {
          const int c_base = k8 ^ ((k8 >> 3) & 1);
          const uint4 kv = C_my_u4[ktok * 16 + (c_base ^ k_swiz_base)];
          const uint4 qv0 = Q_my_u4[qrow0 * 16 + (c_base ^ q_swiz_base)];
          const uint4 qv1 = Q_my_u4[qrow1 * 16 + (c_base ^ q_swiz_base)];
          mma_rc(s0, qv0.x, qv0.y, kv.x, kv.y);
          mma_rc(s1, qv1.x, qv1.y, kv.x, kv.y);
          mma_rc(s0, qv0.z, qv0.w, kv.z, kv.w);
          mma_rc(s1, qv1.z, qv1.w, kv.z, kv.w);
        }
      } else {
#pragma unroll 2
        for (int k8 = 0; k8 < 16; k8++) {
          const int c_base = k8 ^ ((k8 >> 3) & 1);
          const uint4 kv = C_my_u4[ktok * 16 + (c_base ^ k_swiz_base)];
          const uint4 qv0 = Q_my_u4[qrow0 * 16 + (c_base ^ q_swiz_base)];
          mma_rc(s0, qv0.x, qv0.y, kv.x, kv.y);
          mma_rc(s0, qv0.z, qv0.w, kv.z, kv.w);
        }
      }
      if (dw == 1) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          Ss[((mw * 2 + 0) * 8 + e) * 32 + lane] = s0[e];
          if (has_s1) Ss[((mw * 2 + 1) * 8 + e) * 32 + lane] = s1[e];
        }
      }
    }
    __syncthreads();

    // 3. Softmax on dw==0 while dw==1 overlaps HBM prefetch of next tile's c_hi!
    if (dw == 1) {
      if (more) prefetch_c(t + 1);
    } else if (has_s0) {
      const int cb = tok0 + 8 * qp + c0;
      const int pc = 8 * qp + c0;
      // Strip 0
      {
#pragma unroll
        for (int e = 0; e < 8; e++) s0[e] += Ss[((mw * 2 + 0) * 8 + e) * 32 + lane];
        apply_global_bias_mask_32(
            s0, RelPtrs[lr0A], RelPtrs[lr0B], valid0A, valid0B, pos0A, pos0B,
            rb0A, rb0B, tok0, cb, seqlen_k, E, sl2, rel_aligned4);
        float xA = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[4], s0[5]));
        float xB = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s0[6], s0[7]));
#pragma unroll
        for (int d = 2; d <= 8; d <<= 1) {
          xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, d));
          xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, d));
        }
        const float nA = fmaxf(m0A, xA), nB = fmaxf(m0B, xB);
        const float aA = (m0A == -INFINITY) ? 0.f : ex2_fast(m0A - nA);
        const float aB = (m0B == -INFINITY) ? 0.f : ex2_fast(m0B - nB);
        const float zA = (nA == -INFINITY) ? 0.f : nA;
        const float zB = (nB == -INFINITY) ? 0.f : nB;
#pragma unroll
        for (int e = 0; e < 8; e++) s0[e] = ex2_fast(s0[e] - ((e & 2) ? zB : zA));
        l0A = l0A * aA + (s0[0] + s0[1]) + (s0[4] + s0[5]);
        l0B = l0B * aB + (s0[2] + s0[3]) + (s0[6] + s0[7]);
        m0A = nA; m0B = nB;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) o0[j][e] *= (e & 2) ? aB : aA;
        if ((lane & 14) == 0) {
          Alpha[mw * 16 + r0] = aA;
          Alpha[mw * 16 + r0 + 2] = aB;
        }
        __half* Pw = Ps + (mw * 2 + 0) * 8 * PS;
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s0[0], s0[1]);
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s0[4], s0[5]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s0[2], s0[3]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s0[6], s0[7]);
      }
      // Strip 1
      if (has_s1) {
#pragma unroll
        for (int e = 0; e < 8; e++) s1[e] += Ss[((mw * 2 + 1) * 8 + e) * 32 + lane];
        apply_global_bias_mask_32(
            s1, RelPtrs[lr1A], RelPtrs[lr1B], valid1A, valid1B, pos1A, pos1B,
            rb1A, rb1B, tok0, cb, seqlen_k, E, sl2, rel_aligned4);
        float xA = fmaxf(fmaxf(s1[0], s1[1]), fmaxf(s1[4], s1[5]));
        float xB = fmaxf(fmaxf(s1[2], s1[3]), fmaxf(s1[6], s1[7]));
#pragma unroll
        for (int d = 2; d <= 8; d <<= 1) {
          xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, d));
          xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, d));
        }
        const float nA = fmaxf(m1A, xA), nB = fmaxf(m1B, xB);
        const float aA = (m1A == -INFINITY) ? 0.f : ex2_fast(m1A - nA);
        const float aB = (m1B == -INFINITY) ? 0.f : ex2_fast(m1B - nB);
        const float zA = (nA == -INFINITY) ? 0.f : nA;
        const float zB = (nB == -INFINITY) ? 0.f : nB;
#pragma unroll
        for (int e = 0; e < 8; e++) s1[e] = ex2_fast(s1[e] - ((e & 2) ? zB : zA));
        l1A = l1A * aA + (s1[0] + s1[1]) + (s1[4] + s1[5]);
        l1B = l1B * aB + (s1[2] + s1[3]) + (s1[6] + s1[7]);
        m1A = nA; m1B = nB;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) o1[j][e] *= (e & 2) ? aB : aA;
        if ((lane & 14) == 0) {
          Alpha[mw * 16 + 8 + r0] = aA;
          Alpha[mw * 16 + 8 + r0 + 2] = aB;
        }
        __half* Pw = Ps + (mw * 2 + 1) * 8 * PS;
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s1[0], s1[1]);
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s1[4], s1[5]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s1[2], s1[3]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s1[6], s1[7]);
      }
    }
    __syncthreads();

    // 4. 2x8 PV on BOTH dw==0 and dw==1 (every C LDS.128 reused across BOTH strip 0 & strip 1!)
    if (has_s0) {
      if (dw == 1) {
        const float a0A = Alpha[mw * 16 + r0], a0B = Alpha[mw * 16 + r0 + 2];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) o0[j][e] *= (e & 2) ? a0B : a0A;
        if (has_s1) {
          const float a1A = Alpha[mw * 16 + 8 + r0], a1B = Alpha[mw * 16 + 8 + r0 + 2];
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) o1[j][e] *= (e & 2) ? a1B : a1A;
        }
      }
      const __half* Pw0 = Ps + ((mw * 2 + 0) * 8 + arow) * PS;
      const __half* Pw1 = Ps + ((mw * 2 + 1) * 8 + arow) * PS;
      if (has_s1) {
#pragma unroll 2
        for (int i2 = 0; i2 < 4; i2++) {
          const uint4 p0 = *reinterpret_cast<const uint4*>(Pw0 + 8 * i2);
          const uint4 p1 = *reinterpret_cast<const uint4*>(Pw1 + 8 * i2);
          const int vsw = l3 | ((i2 & 1) << 2);
          {
            const uint4* Vr_u4 = C_my_u4 + (8 * i2 + l3) * 16;
            const uint4 v0 = Vr_u4[vc0_f ^ vsw];
            const uint4 v1 = Vr_u4[vc1_f ^ vsw];
            mma_rr(o0[0], p0.x, p0.y, v0.x, v0.y); mma_rr(o1[0], p1.x, p1.y, v0.x, v0.y);
            mma_rr(o0[1], p0.x, p0.y, v0.z, v0.w); mma_rr(o1[1], p1.x, p1.y, v0.z, v0.w);
            mma_rr(o0[2], p0.x, p0.y, v1.x, v1.y); mma_rr(o1[2], p1.x, p1.y, v1.x, v1.y);
            mma_rr(o0[3], p0.x, p0.y, v1.z, v1.w); mma_rr(o1[3], p1.x, p1.y, v1.z, v1.w);
          }
          {
            const uint4* Vr_u4 = C_my_u4 + (8 * i2 + 4 + l3) * 16;
            const uint4 v0 = Vr_u4[vc0_f ^ vsw];
            const uint4 v1 = Vr_u4[vc1_f ^ vsw];
            mma_rr(o0[0], p0.z, p0.w, v0.x, v0.y); mma_rr(o1[0], p1.z, p1.w, v0.x, v0.y);
            mma_rr(o0[1], p0.z, p0.w, v0.z, v0.w); mma_rr(o1[1], p1.z, p1.w, v0.z, v0.w);
            mma_rr(o0[2], p0.z, p0.w, v1.x, v1.y); mma_rr(o1[2], p1.z, p1.w, v1.x, v1.y);
            mma_rr(o0[3], p0.z, p0.w, v1.z, v1.w); mma_rr(o1[3], p1.z, p1.w, v1.z, v1.w);
          }
        }
      } else {
#pragma unroll 2
        for (int i2 = 0; i2 < 4; i2++) {
          const uint4 p0 = *reinterpret_cast<const uint4*>(Pw0 + 8 * i2);
          const int vsw = l3 | ((i2 & 1) << 2);
          {
            const uint4* Vr_u4 = C_my_u4 + (8 * i2 + l3) * 16;
            const uint4 v0 = Vr_u4[vc0_f ^ vsw];
            const uint4 v1 = Vr_u4[vc1_f ^ vsw];
            mma_rr(o0[0], p0.x, p0.y, v0.x, v0.y);
            mma_rr(o0[1], p0.x, p0.y, v0.z, v0.w);
            mma_rr(o0[2], p0.x, p0.y, v1.x, v1.y);
            mma_rr(o0[3], p0.x, p0.y, v1.z, v1.w);
          }
          {
            const uint4* Vr_u4 = C_my_u4 + (8 * i2 + 4 + l3) * 16;
            const uint4 v0 = Vr_u4[vc0_f ^ vsw];
            const uint4 v1 = Vr_u4[vc1_f ^ vsw];
            mma_rr(o0[0], p0.z, p0.w, v0.x, v0.y);
            mma_rr(o0[1], p0.z, p0.w, v0.z, v0.w);
            mma_rr(o0[2], p0.z, p0.w, v1.x, v1.y);
            mma_rr(o0[3], p0.z, p0.w, v1.z, v1.w);
          }
        }
      }
    }
    if (dw == 0 && more) prefetch_c(t + 1);
    if (more) __syncthreads();
  }

  // Finalize denominators on dw==0
  if (dw == 0) {
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      l0A += __shfl_xor_sync(0xffffffffu, l0A, d);
      l0B += __shfl_xor_sync(0xffffffffu, l0B, d);
      l1A += __shfl_xor_sync(0xffffffffu, l1A, d);
      l1B += __shfl_xor_sync(0xffffffffu, l1B, d);
    }
  }

  auto store_strip_s1 = [&](float (&os)[4][8], int grA_s, int grB_s, int a_base) {
    const float rsA = Alpha[a_base + r0];
    const float rsB = Alpha[a_base + r0 + 2];
#pragma unroll
    for (int sr = 0; sr < 2; sr++) {
      const int gr = sr ? grB_s : grA_s;
      if (gr < eff_nr) {
        const float rsc = sr ? rsB : rsA;
        const int tok = gr / HQ_GLA;
        const int head = gr - tok * HQ_GLA;
        const int g = head / G_SUB;
        const int sub = head - g * G_SUB;
        const int token = q_start + tok;
        __half* ob = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + dw * HD;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int dlo = 32 * qp + 4 * j + c0;
          *reinterpret_cast<__half2*>(ob + dlo) =
              __floats2half2_rn(os[j][2 * sr] * rsc, os[j][2 * sr + 1] * rsc);
          *reinterpret_cast<__half2*>(ob + dlo + 16) =
              __floats2half2_rn(os[j][4 + 2 * sr] * rsc, os[j][5 + 2 * sr] * rsc);
        }
      }
    }
  };

  auto store_strip_sp = [&](float (&os)[4][8], int grA_s, int grB_s, float mA, float lA, float mB, float lB) {
#pragma unroll
    for (int sr = 0; sr < 2; sr++) {
      const int gr = sr ? grB_s : grA_s;
      if (gr < eff_nr) {
        const int tok = gr / HQ_GLA;
        const int head = gr - tok * HQ_GLA;
        const int token = q_start + tok;
        const size_t slot = ((size_t)token * HQ_GLA + head) * (size_t)num_splits + (size_t)s;
        float* pob = o_part + slot * LAT_D + dw * HD;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int dlo = 32 * qp + 4 * j + c0;
          __stcg(reinterpret_cast<float2*>(pob + dlo), make_float2(os[j][2 * sr], os[j][2 * sr + 1]));
          __stcg(reinterpret_cast<float2*>(pob + dlo + 16),
                 make_float2(os[j][4 + 2 * sr], os[j][5 + 2 * sr]));
        }
        if (dw == 0 && (lane & 14) == 0) {
          __stcg(ml_part + slot, make_float2(sr ? mB : mA, sr ? lB : lA));
        }
      }
    }
  };

  if (num_splits == 1) {
    auto comp_rs = [&](bool valid, int gr, float mv, float lv) -> float {
      if (!valid || lv <= 0.f) return 0.f;
      const int tok = gr / HQ_GLA;
      const int head = gr - tok * HQ_GLA;
      const int token = q_start + tok;
      const float lse_e = (mv + log2f(lv)) * LN2;
      const float sink_v = __half2float(sinks[(size_t)token * (size_t)stride_sink + head]);
      return (vsc / lv) * sigmoid_f32(lse_e - sink_v);
    };
    if (dw == 0 && (lane & 14) == 0) {
      Alpha[mw * 16 + r0] = comp_rs(valid0A, gr0A, m0A, l0A);
      Alpha[mw * 16 + r0 + 2] = comp_rs(valid0B, gr0B, m0B, l0B);
      Alpha[mw * 16 + 8 + r0] = comp_rs(valid1A, gr1A, m1A, l1A);
      Alpha[mw * 16 + 8 + r0 + 2] = comp_rs(valid1B, gr1B, m1B, l1B);
    }
    __syncthreads();
    store_strip_s1(o0, gr0A, gr0B, mw * 16);
    if (has_s1) store_strip_s1(o1, gr1A, gr1B, mw * 16 + 8);
  } else {
    store_strip_sp(o0, gr0A, gr0B, m0A, l0A, m0B, l0B);
    if (has_s1) store_strip_sp(o1, gr1A, gr1B, m1A, l1A, m1B, l1B);
  }
}

// Split-KV combine for D=256 + fused scalar sink scaling + padding zeroing.
__global__ void __launch_bounds__(64)
k_rel_gla_combine(
    const float* __restrict__ o_part,
    const float2* __restrict__ ml_part,
    const __half* __restrict__ sinks,
    __half* __restrict__ o_lat,
    int T_total,
    int num_actual_tokens,
    int num_splits,
    int64_t stride_sink,
    float vsc) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int g = head / G_SUB;
  const int sub = head - g * G_SUB;
  const int dw = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int d_off = dw * HD + lane * 4;
  __half* ob = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + d_off;

  if (token >= num_actual_tokens) {
    *reinterpret_cast<uint2*>(ob) = make_uint2(0u, 0u);
    return;
  }

  const size_t base = ((size_t)token * HQ_GLA + head) * (size_t)num_splits;
  const float2* mlp = ml_part + base;
  const float* op = o_part + base * LAT_D;

  float M = -INFINITY;
  float L_tot = 0.f;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

  if (num_splits <= 32) {
    const float2 ml = (lane < num_splits) ? __ldcg(mlp + lane) : make_float2(-INFINITY, 0.f);
    M = ml.x;
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));
    const float w_lane = (ml.x == -INFINITY) ? 0.f : ex2_fast(ml.x - M);
    L_tot = ml.y * w_lane;
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) L_tot += __shfl_xor_sync(0xffffffffu, L_tot, d);

    int s = 0;
    for (; s + 3 < num_splits; s += 4) {
      const float w0 = __shfl_sync(0xffffffffu, w_lane, s + 0);
      const float w1 = __shfl_sync(0xffffffffu, w_lane, s + 1);
      const float w2 = __shfl_sync(0xffffffffu, w_lane, s + 2);
      const float w3 = __shfl_sync(0xffffffffu, w_lane, s + 3);
      const float4 v0 = (w0 > 0.f) ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 0) * LAT_D + d_off))
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 v1 = (w1 > 0.f) ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 1) * LAT_D + d_off))
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 v2 = (w2 > 0.f) ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 2) * LAT_D + d_off))
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 v3 = (w3 > 0.f) ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 3) * LAT_D + d_off))
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
      acc.x = fmaf(w0, v0.x, fmaf(w1, v1.x, fmaf(w2, v2.x, fmaf(w3, v3.x, acc.x))));
      acc.y = fmaf(w0, v0.y, fmaf(w1, v1.y, fmaf(w2, v2.y, fmaf(w3, v3.y, acc.y))));
      acc.z = fmaf(w0, v0.z, fmaf(w1, v1.z, fmaf(w2, v2.z, fmaf(w3, v3.z, acc.z))));
      acc.w = fmaf(w0, v0.w, fmaf(w1, v1.w, fmaf(w2, v2.w, fmaf(w3, v3.w, acc.w))));
    }
    for (; s < num_splits; s++) {
      const float ws = __shfl_sync(0xffffffffu, w_lane, s);
      if (ws > 0.f) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (size_t)s * LAT_D + d_off));
        acc.x = fmaf(ws, v.x, acc.x);
        acc.y = fmaf(ws, v.y, acc.y);
        acc.z = fmaf(ws, v.z, acc.z);
        acc.w = fmaf(ws, v.w, acc.w);
      }
    }
  } else {
    for (int s = lane; s < num_splits; s += 32) M = fmaxf(M, __ldcg(mlp + s).x);
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));
    for (int sb = 0; sb < num_splits; sb += 32) {
      const int idx = sb + lane;
      const float2 ml = (idx < num_splits) ? __ldcg(mlp + idx) : make_float2(-INFINITY, 0.f);
      const float w_lane = (ml.x == -INFINITY) ? 0.f : ex2_fast(ml.x - M);
      L_tot += ml.y * w_lane;
      const int n_chunk = min(32, num_splits - sb);
      for (int u = 0; u < n_chunk; u++) {
        const float ws = __shfl_sync(0xffffffffu, w_lane, u);
        if (ws > 0.f) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (size_t)(sb + u) * LAT_D + d_off));
          acc.x = fmaf(ws, v.x, acc.x);
          acc.y = fmaf(ws, v.y, acc.y);
          acc.z = fmaf(ws, v.z, acc.z);
          acc.w = fmaf(ws, v.w, acc.w);
        }
      }
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) L_tot += __shfl_xor_sync(0xffffffffu, L_tot, d);
  }

  float row_scale = 0.f;
  if (L_tot > 0.f) {
    const float lse_e = (M + log2f(L_tot)) * LN2;
    const float sink_val = __half2float(sinks[(size_t)token * (size_t)stride_sink + head]);
    row_scale = (vsc / L_tot) * sigmoid_f32(lse_e - sink_val);
  }
  __half2* dst = reinterpret_cast<__half2*>(ob);
  dst[0] = __floats2half2_rn(acc.x * row_scale, acc.y * row_scale);
  dst[1] = __floats2half2_rn(acc.z * row_scale, acc.w * row_scale);
}

// ============================================================================
// 6. Prefill GLA-g4 Kernel (BM=32, BN=32, 4 row-groups x 2 D_WARPS = 256 thr)
// ============================================================================
struct GlaPfSM {
  static constexpr int M_WARPS = 4;
  static constexpr int R_CTA = 8 * M_WARPS;
  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS_PF * (int)sizeof(int);
  static constexpr int RP_OFF = (TB_OFF + TB_BYTES + 15) & ~15;
  static constexpr int RP_BYTES = (R_CTA * (int)sizeof(const __half*) + 15) & ~15;
  static constexpr int QLO_OFF = (RP_OFF + RP_BYTES + 15) & ~15;
  static constexpr int Q_HALF_BYTES = R_CTA * HD * (int)sizeof(__half);
  static constexpr int QHI_OFF = QLO_OFF + Q_HALF_BYTES;
  static constexpr int CLO_OFF = QHI_OFF + Q_HALF_BYTES;
  static constexpr int C_HALF_BYTES = TILE * HD * (int)sizeof(__half);
  static constexpr int CHI_OFF = CLO_OFF + C_HALF_BYTES;
  static constexpr int S_OFF = CHI_OFF + C_HALF_BYTES;
  static constexpr int S_BYTES = M_WARPS * 8 * 32 * (int)sizeof(float);
  static constexpr int P_OFF = S_OFF + S_BYTES;
  static constexpr int P_BYTES = M_WARPS * 8 * PS * (int)sizeof(__half);
  static constexpr int A_OFF = (P_OFF + P_BYTES + 15) & ~15;
  static constexpr int A_BYTES = (M_WARPS * 8 * (int)sizeof(float) + 15) & ~15;
  static constexpr int TOTAL = A_OFF + A_BYTES;
};

__global__ void __launch_bounds__(256, 2)
k_rel_gla_pf(
    const __half* __restrict__ q_lat,
    const uint8_t* __restrict__ c_lo,
    const uint8_t* __restrict__ c_hi,
    const __half* __restrict__ rel_logits,
    const __half* __restrict__ sinks,
    __half* __restrict__ o_lat,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cu_seqlens_q,
    int T_total,
    int E,
    int E_pad,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
    int64_t sb,
    int64_t st,
    int64_t stride_rel_t,
    int64_t stride_rel_h,
    int64_t stride_bt,
    int64_t stride_sink,
    float sl2,
    float vsc,
    bool rel_aligned4) {
  constexpr int BM = 32;
  constexpr int HALF_NT = 128;
  using L_ = GlaPfSM;
  extern __shared__ __align__(16) uint8_t sm[];

  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  uint4* Qlo_u4 = reinterpret_cast<uint4*>(sm + L_::QLO_OFF);
  uint4* Qhi_u4 = reinterpret_cast<uint4*>(sm + L_::QHI_OFF);
  uint4* Clo_u4 = reinterpret_cast<uint4*>(sm + L_::CLO_OFF);
  uint4* Chi_u4 = reinterpret_cast<uint4*>(sm + L_::CHI_OFF);
  float* Ss = reinterpret_cast<float*>(sm + L_::S_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);
  float* Alpha = reinterpret_cast<float*>(sm + L_::A_OFF);

  const int head = blockIdx.x;
  const int raw_m = blockIdx.y;
  const int b = blockIdx.z;

  const int q_start = cu_seqlens_q[b];
  const int q_end = cu_seqlens_q[b + 1];
  const int q_len = q_end - q_start;
  const int eff_q_len = max(0, min(q_len, num_actual_tokens - q_start));
  const int req_m_tiles = (eff_q_len + BM - 1) / BM;
  if (raw_m >= req_m_tiles) return;

  const int m_tile = req_m_tiles - 1 - raw_m;
  const int first_tok = m_tile * BM;
  const int last_tok = min(first_tok + BM, eff_q_len) - 1;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mw = warp >> 1;
  const int dw = warp & 1;
  const int g = head / G_SUB;
  const int sub = head - g * G_SUB;

  const int seqlen_k = cache_seqlens[b];
  const int past = seqlen_k - q_len;
  const int pos_hi = past + last_tok;
  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int t_hi = (k_hi + TILE - 1) / TILE;

  if (t_hi <= 0) {
    for (int i = tid; i < BM * 16; i += 256) {
      const int r = i >> 4;
      const int c = i & 15;
      const int tok = first_tok + r;
      if (tok < eff_q_len) {
        const int token = q_start + tok;
        __half* dst = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D;
        *reinterpret_cast<uint4*>(dst + c * 8) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(dst + 128 + c * 8) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    return;
  }

  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  for (int i = tid; i < min(max_blocks, TBS_PF); i += 256) Ts[i] = tb[i];
  for (int r = tid; r < BM; r += 256) {
    const int tok = first_tok + r;
    if (tok < eff_q_len) {
      const int token = q_start + tok;
      RelPtrs[r] = rel_logits + (size_t)token * (size_t)stride_rel_t + (size_t)head * (size_t)stride_rel_h;
    } else {
      RelPtrs[r] = rel_logits;
    }
  }

  const int half_tid = mw * 32 + lane;
  uint4* Q_my_u4 = (dw == 0) ? Qlo_u4 : Qhi_u4;
  uint4* C_my_u4 = (dw == 0) ? Clo_u4 : Chi_u4;
  const uint8_t* C_my_gmem = (dw == 0) ? c_lo : c_hi;

#pragma unroll
  for (int u = 0; u < 4; u++) {
    const int i = half_tid + u * HALF_NT;
    const int r = i >> 4;
    const int c = i & 15;
    const int tok = first_tok + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (tok < eff_q_len) {
      const int token = q_start + tok;
      const __half* qp = q_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + dw * HD + c * 8;
      v = ldg16(qp);
    }
    Q_my_u4[r * 16 + swiz_q(r, c)] = v;
  }
  __syncthreads();

  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;
  const int r0 = (lane & 1) + 4 * hi;
  const int c0 = lane & 2;
  const int ktok = 8 * qp + arow;

  const int tile_end = min(((first_tok / 128) + 1) * 128, q_len);
  const int rel_base = ((past + tile_end + 127) / 128) * 128 - E_pad;

  const int lrA = mw * 8 + r0;
  const int lrB = lrA + 2;
  const int tokA = first_tok + lrA;
  const int tokB = tokA + 2;
  const bool validA = tokA < eff_q_len;
  const bool validB = tokB < eff_q_len;
  const int posA = validA ? (past + tokA) : -1;
  const int posB = validB ? (past + tokB) : -1;

  const bool warp_has_rows = (first_tok + mw * 8) < eff_q_len;
  const int w_pos_max = past + min(first_tok + mw * 8 + 7, eff_q_len - 1);

  const int qrow_r = mw * 8 + arow;
  const int q_swiz_base = qrow_r & 7;
  const int k_swiz_base = (ktok & 3) | ((ktok >> 1) & 4);
  const int vc0 = 4 * qp + 2 * hi;
  const int vc1 = vc0 + 1;
  const int vc0_f = vc0 ^ ((vc0 >> 3) & 1);
  const int vc1_f = vc1 ^ ((vc1 >> 3) & 1);
  const int pmask = page - 1;

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TBS_PF ? Ts[pi] : tb[pi]);
  };

  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[j][e] = 0.f;
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;

  for (int t = 0; t < t_hi; t++) {
    const int tok0 = t * TILE;
    const bool warp_active = warp_has_rows && (tok0 <= w_pos_max);
    const bool single_page_ok = (page_shift >= 5) && (tok0 + 31 < seqlen_k);
    const int64_t single_pg = single_page_ok ? get_pg(tok0 >> page_shift) : 0;

#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int idx = half_tid + u * HALF_NT;
      const int row = idx >> 3;
      const int col_u = idx & 7;
      uint4 w;
      if (single_page_ok) {
        const int sl = (tok0 & pmask) + row;
        w = ldg16(C_my_gmem + single_pg * sb + (int64_t)sl * st + 16 * col_u);
      } else {
        const int k_idx = tok0 + row;
        if (k_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
          const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
          const int64_t pg = get_pg(pi);
          w = ldg16(C_my_gmem + pg * sb + (int64_t)sl * st + 16 * col_u);
        } else {
          w = make_uint4(0u, 0u, 0u, 0u);
        }
      }
      unsigned lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
      e4m3x4(w.x, lo0, hi0);
      e4m3x4(w.y, lo1, hi1);
      C_my_u4[row * 16 + swiz_kv(row, 2 * col_u)] = make_uint4(lo0, hi0, lo1, hi1);
      e4m3x4(w.z, lo2, hi2);
      e4m3x4(w.w, lo3, hi3);
      C_my_u4[row * 16 + swiz_kv(row, 2 * col_u + 1)] = make_uint4(lo2, hi2, lo3, hi3);
    }
    __syncthreads();

    float s0[8];
    if (warp_active) {
#pragma unroll
      for (int e = 0; e < 8; e++) s0[e] = 0.f;
#pragma unroll 2
      for (int k8 = 0; k8 < 16; k8++) {
        const int c_base = k8 ^ ((k8 >> 3) & 1);
        const uint4 qv = Q_my_u4[qrow_r * 16 + (c_base ^ q_swiz_base)];
        const uint4 kv = C_my_u4[ktok * 16 + (c_base ^ k_swiz_base)];
        mma_rc(s0, qv.x, qv.y, kv.x, kv.y);
        mma_rc(s0, qv.z, qv.w, kv.z, kv.w);
      }
      if (dw == 1) {
#pragma unroll
        for (int e = 0; e < 8; e++) Ss[(mw * 8 + e) * 32 + lane] = s0[e];
      }
    }
    __syncthreads();

    if (warp_active && dw == 0) {
#pragma unroll
      for (int e = 0; e < 8; e++) s0[e] += Ss[(mw * 8 + e) * 32 + lane];
      const int cb = tok0 + 8 * qp + c0;
      apply_global_bias_mask_32(
          s0, RelPtrs[lrA], RelPtrs[lrB], validA, validB, posA, posB,
          rel_base, rel_base, tok0, cb, seqlen_k, E, sl2, rel_aligned4);

      float x0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[4], s0[5]));
      float x2 = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s0[6], s0[7]));
#pragma unroll
      for (int d = 2; d <= 8; d <<= 1) {
        x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
        x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
      }
      const float n0 = fmaxf(m0, x0);
      const float n2 = fmaxf(m2, x2);
      const float a0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - n0);
      const float a2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - n2);
      const float z0 = (n0 == -INFINITY) ? 0.f : n0;
      const float z2 = (n2 == -INFINITY) ? 0.f : n2;

#pragma unroll
      for (int e = 0; e < 8; e++) s0[e] = ex2_fast(s0[e] - ((e & 2) ? z2 : z0));
      l0 = l0 * a0 + (s0[0] + s0[1]) + (s0[4] + s0[5]);
      l2 = l2 * a2 + (s0[2] + s0[3]) + (s0[6] + s0[7]);
      m0 = n0;
      m2 = n2;

#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;

      if ((lane & 14) == 0) {
        Alpha[mw * 8 + r0] = a0;
        Alpha[mw * 8 + r0 + 2] = a2;
      }
      __half* Pw = Ps + mw * 8 * PS;
      const int pc = 8 * qp + c0;
      *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s0[0], s0[1]);
      *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s0[4], s0[5]);
      *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s0[2], s0[3]);
      *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s0[6], s0[7]);
    }
    __syncthreads();

    if (warp_active) {
      if (dw == 1) {
        const float a0 = Alpha[mw * 8 + r0];
        const float a2 = Alpha[mw * 8 + r0 + 2];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;
      }
      const __half* Pw_row = Ps + (mw * 8 + arow) * PS;
#pragma unroll 2
      for (int i2 = 0; i2 < 4; i2++) {
        const uint4 p8 = *reinterpret_cast<const uint4*>(Pw_row + 8 * i2);
        const int vsw = l3 | ((i2 & 1) << 2);
        {
          const uint4* Vr_u4 = C_my_u4 + (8 * i2 + l3) * 16;
          const uint4 v0 = Vr_u4[vc0_f ^ vsw];
          const uint4 v1 = Vr_u4[vc1_f ^ vsw];
          mma_rr(o[0], p8.x, p8.y, v0.x, v0.y);
          mma_rr(o[1], p8.x, p8.y, v0.z, v0.w);
          mma_rr(o[2], p8.x, p8.y, v1.x, v1.y);
          mma_rr(o[3], p8.x, p8.y, v1.z, v1.w);
        }
        {
          const uint4* Vr_u4 = C_my_u4 + (8 * i2 + 4 + l3) * 16;
          const uint4 v0 = Vr_u4[vc0_f ^ vsw];
          const uint4 v1 = Vr_u4[vc1_f ^ vsw];
          mma_rr(o[0], p8.z, p8.w, v0.x, v0.y);
          mma_rr(o[1], p8.z, p8.w, v0.z, v0.w);
          mma_rr(o[2], p8.z, p8.w, v1.x, v1.y);
          mma_rr(o[3], p8.z, p8.w, v1.z, v1.w);
        }
      }
    }
    if (t + 1 < t_hi) __syncthreads();
  }

  if (dw == 0) {
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      l0 += __shfl_xor_sync(0xffffffffu, l0, d);
      l2 += __shfl_xor_sync(0xffffffffu, l2, d);
    }
    if ((lane & 14) == 0) {
      float rs0 = 0.f, rs2 = 0.f;
      if (validA && l0 > 0.f) {
        const int tokenA = q_start + tokA;
        const float lse_e = (m0 + log2f(l0)) * LN2;
        const float sink_v = __half2float(sinks[(size_t)tokenA * (size_t)stride_sink + head]);
        rs0 = (vsc / l0) * sigmoid_f32(lse_e - sink_v);
      }
      if (validB && l2 > 0.f) {
        const int tokenB = q_start + tokB;
        const float lse_e = (m2 + log2f(l2)) * LN2;
        const float sink_v = __half2float(sinks[(size_t)tokenB * (size_t)stride_sink + head]);
        rs2 = (vsc / l2) * sigmoid_f32(lse_e - sink_v);
      }
      Alpha[mw * 8 + r0] = rs0;
      Alpha[mw * 8 + r0 + 2] = rs2;
    }
  }
  __syncthreads();

  const float rs0 = Alpha[mw * 8 + r0];
  const float rs2 = Alpha[mw * 8 + r0 + 2];
#pragma unroll
  for (int sr = 0; sr < 2; sr++) {
    const bool valid = sr ? validB : validA;
    if (valid) {
      const float rsc = sr ? rs2 : rs0;
      const int token = q_start + (sr ? tokB : tokA);
      __half* ob = o_lat + (((size_t)g * T_total + token) * G_SUB + sub) * LAT_D + dw * HD;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int dlo = 32 * qp + 4 * j + c0;
        *reinterpret_cast<__half2*>(ob + dlo) =
            __floats2half2_rn(o[j][2 * sr] * rsc, o[j][2 * sr + 1] * rsc);
        *reinterpret_cast<__half2*>(ob + dlo + 16) =
            __floats2half2_rn(o[j][4 + 2 * sr] * rsc, o[j][5 + 2 * sr] * rsc);
      }
    }
  }
}

template <int M_WARPS, int T_WARPS>
void set_carveout_dec_l1() {
  static bool done[16] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 0 && dev < 16 && !done[dev]) {
    // 50% carveout = 64 KB SMEM + 64 KB L1 -> allows 2-3 CTAs/SM while keeping 8 KB tiles 100% L1 hot!
    cudaFuncSetAttribute(
        k_rel_gla_dec<M_WARPS, T_WARPS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        50);
    done[dev] = true;
  }
}

void set_carveout_dec_2x8() {
  static bool done[16] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 0 && dev < 16 && !done[dev]) {
    cudaFuncSetAttribute(
        k_rel_gla_dec_2x8,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    done[dev] = true;
  }
}

void set_carveout_pf() {
  static bool done[16] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 0 && dev < 16 && !done[dev]) {
    cudaFuncSetAttribute(
        k_rel_gla_pf,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    done[dev] = true;
  }
}

inline int compute_page_shift(int page) {
  if (page > 0 && (page & (page - 1)) == 0) {
    int s = 0, tmp = page;
    while (tmp > 1) { tmp >>= 1; s++; }
    return s;
  }
  return -1;
}

}  // namespace

void gla_scatter_fp8(
    torch::Tensor c_raw,
    torch::Tensor c_lo_cache,
    torch::Tensor c_hi_cache,
    torch::Tensor kv_slot_mapping) {
  const int T = (int)c_raw.size(0);
  if (T == 0) return;
  TORCH_CHECK(c_raw.dtype() == torch::kHalf && c_raw.size(1) == LAT_D && c_raw.stride(1) == 1);
  TORCH_CHECK(c_lo_cache.element_size() == 1 && c_hi_cache.element_size() == 1);
  TORCH_CHECK(c_lo_cache.sizes() == c_hi_cache.sizes() && c_lo_cache.size(2) == 1 && c_lo_cache.size(3) == HD);
  const at::cuda::CUDAGuard guard(c_raw.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  k_gla_scatter_fp8<<<T, 64, 0, stream>>>(
      reinterpret_cast<const __half*>(c_raw.data_ptr<at::Half>()),
      reinterpret_cast<uint8_t*>(c_lo_cache.data_ptr()),
      reinterpret_cast<uint8_t*>(c_hi_cache.data_ptr()),
      kv_slot_mapping.data_ptr<int64_t>(),
      c_raw.stride(0),
      c_lo_cache.stride(0),
      c_lo_cache.stride(1),
      (int)c_lo_cache.size(1));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void gla_pack_q(torch::Tensor q_in, torch::Tensor q_out) {
  const int T = (int)q_in.size(0);
  if (T == 0) return;
  TORCH_CHECK(q_in.dtype() == torch::kHalf && q_in.is_contiguous() && q_in.size(1) == HQ_GLA && q_in.size(2) == HD);
  TORCH_CHECK(q_out.dtype() == torch::kHalf && q_out.is_contiguous() && q_out.numel() == q_in.numel());
  const at::cuda::CUDAGuard guard(q_in.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int total_u4 = T * HQ_GLA * 16;
  k_gla_pack_q<<<(total_u4 + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()),
      T);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void gla_gate_epilogue(torch::Tensor v_out, torch::Tensor gate, torch::Tensor out) {
  const int T = (int)out.size(0);
  if (T == 0) return;
  TORCH_CHECK(v_out.dtype() == torch::kHalf && v_out.is_contiguous() && v_out.numel() == (int64_t)T * HQ_GLA * HD);
  TORCH_CHECK(gate.dtype() == torch::kHalf && gate.size(0) == T && gate.size(1) == HQ_GLA * HD && gate.stride(1) == 1);
  TORCH_CHECK(out.dtype() == torch::kHalf && out.is_contiguous() && out.size(1) == HQ_GLA * HD);
  const at::cuda::CUDAGuard guard(out.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int total_h2 = T * HQ_GLA * 64;
  k_gla_gate_epilogue<<<(total_h2 + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(v_out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      T,
      gate.stride(0));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void attn_rel_gla_dec_forward(
    torch::Tensor q_lat,
    torch::Tensor c_lo_cache,
    torch::Tensor c_hi_cache,
    torch::Tensor rel_logits,
    torch::Tensor sinks,
    torch::Tensor o_lat,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    torch::Tensor cu_seqlens_q,
    torch::Tensor o_part,
    torch::Tensor ml_part,
    int64_t max_query_len,
    int64_t num_actual_tokens,
    double softmax_scale,
    int64_t num_splits) {
  TORCH_CHECK(q_lat.dtype() == torch::kHalf && q_lat.is_contiguous() &&
              q_lat.dim() == 4 && q_lat.size(0) == 2 && q_lat.size(2) == G_SUB && q_lat.size(3) == LAT_D);
  const int T_total = (int)q_lat.size(1);
  if (T_total == 0) return;
  TORCH_CHECK(o_lat.dtype() == torch::kHalf && o_lat.is_contiguous() && o_lat.sizes() == q_lat.sizes());
  TORCH_CHECK(c_lo_cache.element_size() == 1 && c_hi_cache.element_size() == 1 &&
              c_lo_cache.sizes() == c_hi_cache.sizes() && c_lo_cache.size(2) == 1 && c_lo_cache.size(3) == HD &&
              c_lo_cache.stride(3) == 1 && c_hi_cache.stride(3) == 1 &&
              c_lo_cache.stride(0) == c_hi_cache.stride(0) && c_lo_cache.stride(1) == c_hi_cache.stride(1));

  const int B = (int)cache_seqlens.size(0);
  const int page = (int)c_lo_cache.size(1);
  const int page_shift = compute_page_shift(page);
  const int E_pad = (int)rel_logits.size(2);
  const int E = E_pad - 256;
  const int S = (int)num_splits;
  const int act_toks = (int)std::max<int64_t>(0, std::min<int64_t>(num_actual_tokens, T_total));

  const at::cuda::CUDAGuard guard(q_lat.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (act_toks == 0 || B == 0) {
    TORCH_CHECK(cudaMemsetAsync(o_lat.data_ptr(), 0, (size_t)o_lat.numel() * sizeof(__half), stream) == cudaSuccess);
    return;
  }

  float* o_part_ptr = nullptr;
  float2* ml_part_ptr = nullptr;
  if (S > 1) {
    const int64_t slots = (int64_t)T_total * HQ_GLA * S;
    TORCH_CHECK(o_part.numel() >= slots * LAT_D && ml_part.numel() >= slots * 2,
                "GLA split-KV workspace too small");
    o_part_ptr = o_part.data_ptr<float>();
    ml_part_ptr = reinterpret_cast<float2*>(ml_part.data_ptr<float>());
  }

  const __half* rel_ptr = reinterpret_cast<const __half*>(rel_logits.data_ptr<at::Half>());
  const bool rel_aligned4 = (((uintptr_t)rel_ptr & 3u) == 0u) && ((rel_logits.stride(0) & 1) == 0) &&
                            ((rel_logits.stride(1) & 1) == 0) && ((E_pad & 1) == 0);
  const float sl2 = (float)(softmax_scale * 256.0 * (double)LOG2E);
  const float vsc = 256.0f;
  const int max_nr = std::max(1, (int)max_query_len * HQ_GLA);
  const int max_blocks = (int)block_table.size(1);

#define LAUNCH_GLA_DEC(MW_I, TW_I)                                                                           \
  do {                                                                                                       \
    set_carveout_dec_l1<MW_I, TW_I>();                                                                       \
    constexpr int R_CTA = 8 * (MW_I);                                                                        \
    const int num_m_tiles = (max_nr + R_CTA - 1) / R_CTA;                                                    \
    dim3 grid(S, num_m_tiles, B);                                                                            \
    constexpr int smem = GlaDecSM<MW_I, TW_I>::TOTAL;                                                        \
    k_rel_gla_dec<MW_I, TW_I><<<grid, 32 * (MW_I) * (TW_I), smem, stream>>>(                                 \
        reinterpret_cast<const __half*>(q_lat.data_ptr<at::Half>()),                                         \
        reinterpret_cast<const uint8_t*>(c_lo_cache.data_ptr()),                                             \
        reinterpret_cast<const uint8_t*>(c_hi_cache.data_ptr()),                                             \
        rel_ptr, reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>()),                                \
        reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),                                               \
        o_part_ptr, ml_part_ptr, block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(),                 \
        cu_seqlens_q.data_ptr<int>(), T_total, E, E_pad, page, page_shift, max_blocks, act_toks, S,          \
        c_lo_cache.stride(0), c_lo_cache.stride(1), rel_logits.stride(0), rel_logits.stride(1),              \
        block_table.stride(0), sinks.stride(0), sl2, vsc, rel_aligned4);                                     \
  } while (0)

  if (max_nr <= 8) {
    LAUNCH_GLA_DEC(1, 4);
  } else if (max_nr <= 16) {
    LAUNCH_GLA_DEC(2, 2);
  } else {
    set_carveout_dec_2x8();
    constexpr int R_CTA = GlaDec2x8SM::R_CTA;
    const int num_m_tiles = (max_nr + R_CTA - 1) / R_CTA;
    dim3 grid(S, num_m_tiles, B);
    constexpr int smem = GlaDec2x8SM::TOTAL;
    k_rel_gla_dec_2x8<<<grid, GlaDec2x8SM::NT, smem, stream>>>(
        reinterpret_cast<const __half*>(q_lat.data_ptr<at::Half>()),
        reinterpret_cast<const uint8_t*>(c_lo_cache.data_ptr()),
        reinterpret_cast<const uint8_t*>(c_hi_cache.data_ptr()),
        rel_ptr, reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),
        o_part_ptr, ml_part_ptr, block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(),
        cu_seqlens_q.data_ptr<int>(), T_total, E, E_pad, page, page_shift, max_blocks, act_toks, S,
        c_lo_cache.stride(0), c_lo_cache.stride(1), rel_logits.stride(0), rel_logits.stride(1),
        block_table.stride(0), sinks.stride(0), sl2, vsc, rel_aligned4);
  }
#undef LAUNCH_GLA_DEC

  if (S == 1) {
    if (act_toks < T_total) {
      const size_t pad_bytes = (size_t)(T_total - act_toks) * G_SUB * LAT_D * sizeof(__half);
      __half* ob = reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>());
      TORCH_CHECK(cudaMemsetAsync(ob + (size_t)act_toks * G_SUB * LAT_D, 0, pad_bytes, stream) == cudaSuccess);
      TORCH_CHECK(cudaMemsetAsync(ob + ((size_t)T_total + act_toks) * G_SUB * LAT_D, 0, pad_bytes, stream) ==
                  cudaSuccess);
    }
  } else {
    dim3 cgrid(T_total, HQ_GLA);
    k_rel_gla_combine<<<cgrid, 64, 0, stream>>>(
        o_part_ptr, ml_part_ptr, reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),
        T_total, act_toks, S, sinks.stride(0), vsc);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void attn_rel_gla_pf_forward(
    torch::Tensor q_lat,
    torch::Tensor c_lo_cache,
    torch::Tensor c_hi_cache,
    torch::Tensor rel_logits,
    torch::Tensor sinks,
    torch::Tensor o_lat,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    torch::Tensor cu_seqlens_q,
    int64_t max_query_len,
    int64_t num_actual_tokens,
    double softmax_scale) {
  TORCH_CHECK(q_lat.dtype() == torch::kHalf && q_lat.is_contiguous() &&
              q_lat.dim() == 4 && q_lat.size(0) == 2 && q_lat.size(2) == G_SUB && q_lat.size(3) == LAT_D);
  const int T_total = (int)q_lat.size(1);
  if (T_total == 0) return;
  TORCH_CHECK(o_lat.dtype() == torch::kHalf && o_lat.is_contiguous() && o_lat.sizes() == q_lat.sizes());
  TORCH_CHECK(c_lo_cache.element_size() == 1 && c_hi_cache.element_size() == 1 &&
              c_lo_cache.sizes() == c_hi_cache.sizes() && c_lo_cache.size(2) == 1 && c_lo_cache.size(3) == HD);

  const int B = (int)cache_seqlens.size(0);
  const int page = (int)c_lo_cache.size(1);
  const int page_shift = compute_page_shift(page);
  const int E_pad = (int)rel_logits.size(2);
  const int E = E_pad - 256;
  const int act_toks = (int)std::max<int64_t>(0, std::min<int64_t>(num_actual_tokens, T_total));

  const at::cuda::CUDAGuard guard(q_lat.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (act_toks == 0 || B == 0) {
    TORCH_CHECK(cudaMemsetAsync(o_lat.data_ptr(), 0, (size_t)o_lat.numel() * sizeof(__half), stream) == cudaSuccess);
    return;
  }

  const __half* rel_ptr = reinterpret_cast<const __half*>(rel_logits.data_ptr<at::Half>());
  const bool rel_aligned4 = (((uintptr_t)rel_ptr & 3u) == 0u) && ((rel_logits.stride(0) & 1) == 0) &&
                            ((rel_logits.stride(1) & 1) == 0) && ((E_pad & 1) == 0);
  const float sl2 = (float)(softmax_scale * 256.0 * (double)LOG2E);
  const float vsc = 256.0f;
  const int max_blocks = (int)block_table.size(1);
  const int num_m_tiles = ((int)max_query_len + 31) / 32;

  set_carveout_pf();
  constexpr int smem = GlaPfSM::TOTAL;
  dim3 grid(HQ_GLA, num_m_tiles, B);
  k_rel_gla_pf<<<grid, 256, smem, stream>>>(
      reinterpret_cast<const __half*>(q_lat.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(c_lo_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(c_hi_cache.data_ptr()),
      rel_ptr, reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),
      block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(),
      T_total, E, E_pad, page, page_shift, max_blocks, act_toks,
      c_lo_cache.stride(0), c_lo_cache.stride(1), rel_logits.stride(0), rel_logits.stride(1),
      block_table.stride(0), sinks.stride(0), sl2, vsc, rel_aligned4);

  if (act_toks < T_total) {
    const size_t pad_bytes = (size_t)(T_total - act_toks) * G_SUB * LAT_D * sizeof(__half);
    __half* ob = reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>());
    TORCH_CHECK(cudaMemsetAsync(ob + (size_t)act_toks * G_SUB * LAT_D, 0, pad_bytes, stream) == cudaSuccess);
    TORCH_CHECK(cudaMemsetAsync(ob + ((size_t)T_total + act_toks) * G_SUB * LAT_D, 0, pad_bytes, stream) ==
                cudaSuccess);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gla_scatter_fp8", &gla_scatter_fp8, "Fused FP16 -> E4M3FN GLA-g4 latent scatter");
  m.def("gla_pack_q", &gla_pack_q, "Pack clipped Q [T, 6, 128] -> [2, T, 3, 128]");
  m.def("gla_gate_epilogue", &gla_gate_epilogue, "Fused unpack + sigmoid(gate) epilogue");
  m.def("attn_rel_gla_dec_forward", &attn_rel_gla_dec_forward,
        "Volta sm_70 HMMA.884 relative-bias GLA-g4 (r=256) decode/verify attention");
  m.def("attn_rel_gla_pf_forward", &attn_rel_gla_pf_forward,
        "Volta sm_70 HMMA.884 relative-bias GLA-g4 (r=256) prefill attention");
}
