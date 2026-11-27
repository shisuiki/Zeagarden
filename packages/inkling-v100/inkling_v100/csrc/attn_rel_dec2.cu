// SPDX-License-Identifier: Apache-2.0
// attn_rel_dec2.cu - Volta (sm_70) short-SASS 1-launch fused kv_conv_prologue + decode_attention
// supporting BOTH FP16 and OCP FP8 E4M3FN separate paged KV caches, plus RingConv (N=16, cmax=1)
// and paged conv caches (N=4).
//
// Architecture:
//   - Grid (S, H_kv, R), NW * 32 threads (NW in {8, 16}, MS in {40, 80}); runtime loops for short SASS.
//   - Owner CTA for each tile containing new tokens runs K/V short conv (static or dynamic taps),
//     K L2-clip / V RMSNorm, writes separate key_cache/value_cache (FP16 or exact RNE OCP E4M3FN)
//     and conv_cache, and stages the exact written values into K_s / V_s.
//   - Cached tiles load from separate key_cache / value_cache (half HBM traffic in FP8 mode: 1x uint4
//     each for K and V per thread vs 2x uint4 in FP16), with optional 3-tile-ahead page lookup + L2 prefetch.
//   - Global layers (window == 0, rows >= 12, pos0 > E) switch to a software-pipelined WMMA fp16->fp32
//     interior loop for dist >= E (reusing K_s + V_s with zero extra smem). In FP8 mode, WMMA loads
//     uint2 (8 bytes) and unpacks via e4m3x4 directly to 2^-8-scaled FP16 (zero staging multiplies),
//     folding 256.0f into softmax scale and scaling existing macc by 2^-8 before / 256.0f after WMMA.
//   - Self-resetting grid barrier per (req, kvh) (all CTAs resident: S * H_kv * R <= budget) followed
//     by I-cache-warming combine + fused sigmoid gate & sink epilogue.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>

namespace {

constexpr int HD = 128;
constexpr int GQ = 3;
constexpr int NWC = 8;              // combine warps (fixed fp32 reduction grouping for every NW)
constexpr int TK = 32;
constexpr int SPIN_MAX = (1 << 22);
constexpr float LOG2E_F = 1.4426950408889634f;
constexpr float LN2_F = 0.6931471805599453f;

struct Params {
  const __half* q;         // [T, Hq, HD] raw projection (row stride stride_q)
  const __half* r;         // [T, Hq, 16]
  const __half* q_scale;   // [Hq, HD]
  const __half* r_scale;   // [Hq, 16]
  const float* tau;        // [T] or null
  const __half* rel_t;     // [E, 16] transposed relative projection
  void* kc;                // [nb, page, Hkv, HD] (__half* when !IS_FP8, uint8_t* when IS_FP8)
  void* vc;                // [nb, page, Hkv, HD] (__half* when !IS_FP8, uint8_t* when IS_FP8)
  const int* block_table;  // [R, max_blocks]
  const int* seq_lens;     // [R]
  const int* cu_q;         // [R + 1]
  const __half* gate;      // [T, Hq * HD]
  const __half* sinks;     // [T, Hq]
  __half* out;             // [T, Hq * HD]
  float* o_part;           // [S, T, Hq, HD]
  float* ml_part;          // [S, T, Hq, 2] (running max log2, sum)
  int* counters;           // [>= R * Hkv]
  int* fault;              // [1] sticky barrier-timeout flag
  const __half* x[2];      // raw K / V projections [T, Hkv * HD]
  const __half* w[2];      // conv taps: static [Hkv * HD, 4]; dynamic [4, Hkv * HD]
  const __half* dyn[2];    // dynamic per-token tap deltas [T, Hkv * 4] or null
  __half* cache;           // conv state [cb, Hkv, N, CP]
  const int64_t* pos;      // [T]
  const int64_t* kv_slot;  // [T]
  const int64_t* cslot;    // [T]
  const int* seq_idx;      // [T]
  const int* qstart;       // [T]
  const int* cbt;          // [R, cmax]
  int64_t stride_q, stride_r, stride_bt, stride_gate, stride_sink;
  int64_t kb, ks, kh;      // key_cache strides in elements
  int64_t vb, vs, vh;      // value_cache strides in elements
  int64_t stride_x[2], stride_dyn[2], stride_w0;
  int64_t stride_c_blk, stride_c_h, stride_c_n, stride_cbt;
  int T, num_actual_tokens, Hq, Hkv, S, page_shift, window, E, maxrows, cshift, cmax, stop, pf;
  float scale_log2, q_thr, r_thr, eps, conv_scale, norm_eps;
};

__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2f(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rsqrtf_approx(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ int ld_volatile(const int* p) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
// Inline single-instruction warp shuffles (uniform per-warp control flow at all call sites).
__device__ __forceinline__ float shx(float v, int o) {
  float r;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(o));
  return r;
}
__device__ __forceinline__ float shi(float v, int l) {
  float r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(l));
  return r;
}
__device__ __forceinline__ float rnd16(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + ex2f(-x * LOG2E_F)); }

__device__ __forceinline__ void up2(uint32_t u, float& a, float& b) {
  __half2 h = *reinterpret_cast<__half2*>(&u);
  float2 f = __half22float2(h);
  a = f.x;
  b = f.y;
}
__device__ __forceinline__ float4 up4(const uint2 v) {
  float4 f;
  up2(v.x, f.x, f.y);
  up2(v.y, f.z, f.w);
  return f;
}

// Exact bit-unpack of 4 OCP E4M3FN bytes -> two fp16x2 (bytes 0,1 and 2,3), each = E4M3 * 2^-8.
// Exact across all non-NaN E4M3FN codes (normals, subnormals, +-0).
__device__ __forceinline__ void e4m3x4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t x0 = __byte_perm(w, 0u, 0x1404);
  const uint32_t x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}

// Unpack 4 E4M3FN bytes -> exact unscaled FP32 float4.
__device__ __forceinline__ float4 up4_e4m3(uint32_t w) {
  uint32_t lo, hi;
  e4m3x4(w, lo, hi);
  float4 f = up4(make_uint2(lo, hi));
  f.x *= 256.0f;
  f.y *= 256.0f;
  f.z *= 256.0f;
  f.w *= 256.0f;
  return f;
}

// Unpack 8 E4M3FN bytes -> 8 FP16 values scaled by 2^-8 (zero multiplies for WMMA staging).
__device__ __forceinline__ uint4 e4m3x8_to_h8_scaled(uint2 v) {
  uint4 h;
  e4m3x4(v.x, h.x, h.y);
  e4m3x4(v.y, h.z, h.w);
  return h;
}

__device__ __forceinline__ void ld_kv_wmma(uint4& dst, const void* base, int64_t off, int c8) {
  dst = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(base) + off + c8 * 8));
}
__device__ __forceinline__ void ld_kv_wmma(uint2& dst, const void* base, int64_t off, int c8) {
  dst = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(base) + off + c8 * 8));
}
__device__ __forceinline__ void st_kv_wmma(__half* dst, const uint4& v) {
  *reinterpret_cast<uint4*>(dst) = v;
}
__device__ __forceinline__ void st_kv_wmma(__half* dst, const uint2& v) {
  *reinterpret_cast<uint4*>(dst) = e4m3x8_to_h8_scaled(v);
}

// Exact IEEE-754 RNE FP32 -> uint8 OCP E4M3FN (saturating to +-448.0), matching ops/fp8_sm70.py bit-for-bit.
__device__ __forceinline__ uint8_t f32_to_e4m3_u8(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t sign = (bits >> 24) & 0x80u;
  const uint32_t abs_bits = bits & 0x7FFFFFFFu;
  const bool is_nan = abs_bits > 0x7F800000u;
  // Clamp magnitude to E4M3FN max = 448.0 (0x43E00000) before rounding.
  const uint32_t clamped = abs_bits < 0x43E00000u ? abs_bits : 0x43E00000u;
  const uint32_t exp = clamped >> 23;
  const uint32_t mant = clamped & 0x007FFFFFu;
  uint32_t code;
  if (is_nan) {
    code = 0x7Fu;
  } else if (exp >= 121u) {
    // Normal range: target e4 = exp - 120 in [1..15], RNE on low 20 bits, cap at 0x7E.
    const uint32_t norm_e = exp - 120u;
    const uint32_t norm_round = ((mant >> 20) & 1u) + 0x7FFFFu;
    const uint32_t norm_rounded = (norm_e << 23) + mant + norm_round;
    const uint32_t norm_code = norm_rounded >> 20;
    code = norm_code < 0x7Eu ? norm_code : 0x7Eu;
  } else if (abs_bits < 0x3A800000u) {
    // Strictly below 2^-10 (halfway to smallest subnormal 2^-9) rounds to signed zero.
    code = 0u;
  } else {
    // Subnormal range (exp < 121, |x| >= 2^-10): multiply by 2^9 (= 512.0) and RNE-round to [0..8].
    // Code 8 = 0x08 naturally equals min normal (2^-6).
    const float abs_x = __uint_as_float(clamped);
    const float rounded = __fadd_rn(__fmul_rn(abs_x, 512.0f), 8388608.0f);
    code = __float_as_uint(rounded) - 0x4B000000u;
  }
  return static_cast<uint8_t>(sign | code);
}

__device__ __forceinline__ uint32_t f32x4_to_e4m3x4(float a, float b, float c, float d) {
  return static_cast<uint32_t>(f32_to_e4m3_u8(a)) |
         (static_cast<uint32_t>(f32_to_e4m3_u8(b)) << 8) |
         (static_cast<uint32_t>(f32_to_e4m3_u8(c)) << 16) |
         (static_cast<uint32_t>(f32_to_e4m3_u8(d)) << 24);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, shx(v, o));
  return v;
}
// Triton's reduction order: a + shuffled, xor 16, 8, 4, 2, 1.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, shx(v, o));
  return v;
}
__device__ __forceinline__ float clip_factor(float ss, float thr, float eps) {
  float norm = ss > 0.f ? __fmul_rn(ss, rsqrtf_approx(ss)) : 0.f;
  return fminf(__fdiv_rn(thr, __fadd_rn(norm, eps)), 1.f);
}

// IS_FP8: false = FP16 separate KV, true = OCP E4M3FN uint8 separate KV.
// NW: warps per CTA (8 | 16). MS: max splits per (req, kvh) (40, or 80 with 2 CTAs/SM for NW=8).
template <bool IS_FP8, int NW, int MS>
__global__ void __launch_bounds__(NW * 32, (NW == 8) ? 2 : 1) attn2_kernel(const Params p) {
  constexpr int CSP = (MS + NWC - 1) / NWC;   // partials merged per warp in the combine
  constexpr int MAXW = (MS + 31) / 32;        // warps that hold the partial maxima
  extern __shared__ __align__(16) float smem[];
  const int MR = p.maxrows;
  float* K_s = smem;             // [32 quads][TK keys][4]
  float* V_s = K_s + TK * HD;    // [TK keys][HD]
  float* q_s = V_s + TK * HD;    // [MR][HD]
  float* macc = q_s + MR * HD;   // [MR][HD]
  float* r_s = macc + MR * HD;   // [MR][16]
  float* p_s = r_s + MR * 16;    // [MR][TK]
  float* m_s = p_s + MR * TK;    // [MR]
  float* l_s = m_s + MR;         // [MR]
  float* cm_s = l_s + MR;        // [MS] combine: partial max
  float* cl_s = cm_s + MS;       // [MS] combine: partial sum

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  static_assert(NWC * CSP >= MS, "combine fan-in");
  const int split = blockIdx.x, kvh = blockIdx.y, req = blockIdx.z;
  const int S = p.S;
  const int q_start = p.cu_q[req];
  const int nq = p.cu_q[req + 1] - q_start;
  const int eff_nq = max(0, min(nq, p.num_actual_tokens - q_start));
  const int seqlen = p.seq_lens[req];
  int rows = 0, active = 0, pos0 = 0, lo = 0, hi = 0, tile0 = 0, ntiles = 0;
  if (eff_nq > 0) {
    rows = min(eff_nq * GQ, MR);
    if (seqlen >= nq && seqlen > 0) {
      pos0 = seqlen - nq;  // position of the request's first new token
      hi = min(pos0 + eff_nq, seqlen);
      lo = p.window > 0 ? max(pos0 - p.window + 1, 0) : 0;
      tile0 = lo >> 5;
      ntiles = ((hi + 31) >> 5) - tile0;
      active = min(S, ntiles);
    }
  }

  if (p.stop == 1) return;
  if (split < active) {
    const int page_mask = (1 << p.page_shift) - 1;
    const int64_t cmask = ((int64_t)1 << p.cshift) - 1;
    // K/V of a tile's cached keys: thread = (key, 16-dim chunk).
    // FP16: 4 x 16 B loads (k0, k1, v0, v1). FP8: 2 x 16 B loads (k0, v0) -> half HBM traffic.
    uint4 k0, k1, v0, v1;
    // Tiles are numbered from the newest (u = 0). CTA 0 takes only that one (it also runs
    // the K/V conv of the newest tokens); CTAs 1.. share the older tiles.
    const int ustep = S > 1 ? S - 1 : 1;
    const int ulimit = (split == 0 && S > 1) ? 1 : ntiles;
    auto issue = [&](int u) {
      const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
      k0 = make_uint4(0u, 0u, 0u, 0u);
      v0 = k0;
      if constexpr (!IS_FP8) {
        k1 = k0;
        v1 = k0;
      }
      if (tid < 256 && key >= lo && key < pos0) {
        const int page = p.block_table[(int64_t)req * p.stride_bt + (key >> p.page_shift)];
        const int64_t slot = (int64_t)(key & page_mask);
        const int col16 = (tid & 7) * 16;
        if constexpr (IS_FP8) {
          const uint4* kp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(p.kc) + (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh + col16);
          const uint4* vp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(p.vc) + (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh + col16);
          k0 = __ldg(kp);
          v0 = __ldg(vp);
        } else {
          const uint4* kp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const __half*>(p.kc) + (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh + col16);
          const uint4* vp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const __half*>(p.vc) + (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh + col16);
          k0 = __ldg(kp);
          k1 = __ldg(kp + 1);
          v0 = __ldg(vp);
          v1 = __ldg(vp + 1);
        }
      }
    };
    issue(split);
    // Optional 3-tile-ahead page lookup + 2-tile-ahead L2 prefetch.
    auto page_of = [&](int u) -> int {
      if (u >= ulimit) return -1;
      const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
      if (tid < 256 && key >= lo && key < pos0) return p.block_table[(int64_t)req * p.stride_bt + (key >> p.page_shift)];
      return -1;
    };
    auto issue_pg = [&](int u, int page) {
      k0 = make_uint4(0u, 0u, 0u, 0u);
      v0 = k0;
      if constexpr (!IS_FP8) {
        k1 = k0;
        v1 = k0;
      }
      if (page >= 0) {
        const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
        const int64_t slot = (int64_t)(key & page_mask);
        const int col16 = (tid & 7) * 16;
        if constexpr (IS_FP8) {
          const uint4* kp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(p.kc) + (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh + col16);
          const uint4* vp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(p.vc) + (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh + col16);
          k0 = __ldg(kp);
          v0 = __ldg(vp);
        } else {
          const uint4* kp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const __half*>(p.kc) + (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh + col16);
          const uint4* vp = reinterpret_cast<const uint4*>(
              reinterpret_cast<const __half*>(p.vc) + (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh + col16);
          k0 = __ldg(kp);
          k1 = __ldg(kp + 1);
          v0 = __ldg(vp);
          v1 = __ldg(vp + 1);
        }
      }
    };
    auto prefetch_pg = [&](int u, int page) {
      if (page >= 0) {
        const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
        const int64_t slot = (int64_t)(key & page_mask);
        const int col16 = (tid & 7) * 16;
        if constexpr (IS_FP8) {
          const char* kp = reinterpret_cast<const char*>(
              reinterpret_cast<const uint8_t*>(p.kc) + (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh + col16);
          const char* vp = reinterpret_cast<const char*>(
              reinterpret_cast<const uint8_t*>(p.vc) + (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh + col16);
          asm volatile("prefetch.global.L2 [%0];" ::"l"(kp));
          asm volatile("prefetch.global.L2 [%0];" ::"l"(vp));
        } else {
          const char* kp = reinterpret_cast<const char*>(
              reinterpret_cast<const __half*>(p.kc) + (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh + col16);
          const char* vp = reinterpret_cast<const char*>(
              reinterpret_cast<const __half*>(p.vc) + (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh + col16);
          asm volatile("prefetch.global.L2 [%0];" ::"l"(kp));
          asm volatile("prefetch.global.L2 [%0];" ::"l"(vp));
        }
      }
    };
    int pg1 = -1, pg2 = -1, pg3 = -1;
    if (p.pf) {
      pg1 = page_of(split + ustep);
      pg2 = page_of(split + 2 * ustep);
      pg3 = page_of(split + 3 * ustep);
    }

    // ---- q / r of this warp's rows: scale, L2 clip, round (tau), stage ----
#pragma unroll 1
    for (int r = w; r < rows; r += NW) {
      const int tk = r / GQ;
      const int token = q_start + tk;
      const int head = kvh * GQ + (r - tk * GQ);
      float tau = 1.f;
      if (p.tau) tau = p.tau[token];
      {
        // q is a slice of the qkvr projection: rows are 4-byte aligned
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(p.q + (int64_t)token * p.stride_q + head * HD + lane * 4);
        float4 x = up4(make_uint2(qp[0], qp[1]));
        const float4 sc = up4(*reinterpret_cast<const uint2*>(p.q_scale + head * HD + lane * 4));
        x.x = __fmul_rn(x.x, sc.x); x.y = __fmul_rn(x.y, sc.y); x.z = __fmul_rn(x.z, sc.z); x.w = __fmul_rn(x.w, sc.w);
        float ss = __fmul_rn(x.x, x.x);
        ss = __fadd_rn(ss, __fmul_rn(x.y, x.y));
        ss = __fadd_rn(ss, __fmul_rn(x.z, x.z));
        ss = __fadd_rn(ss, __fmul_rn(x.w, x.w));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += shx(ss, o);
        const float f = clip_factor(ss, p.q_thr, p.eps);
        x.x = rnd16(__fmul_rn(x.x, f)); x.y = rnd16(__fmul_rn(x.y, f));
        x.z = rnd16(__fmul_rn(x.z, f)); x.w = rnd16(__fmul_rn(x.w, f));
        if (p.tau) {
          x.x = rnd16(__fmul_rn(x.x, tau)); x.y = rnd16(__fmul_rn(x.y, tau));
          x.z = rnd16(__fmul_rn(x.z, tau)); x.w = rnd16(__fmul_rn(x.w, tau));
        }
        *reinterpret_cast<float4*>(q_s + r * HD + lane * 4) = x;
        *reinterpret_cast<float4*>(macc + r * HD + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      {
        float x = 0.f;
        if (lane < 16) {
          x = __half2float(p.r[(int64_t)token * p.stride_r + head * 16 + lane]);
          x = __fmul_rn(x, __half2float(p.r_scale[head * 16 + lane]));
        }
        float ss = __fmul_rn(x, x);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ss = __fadd_rn(ss, shx(ss, o));
        const float f = clip_factor(ss, p.r_thr, p.eps);
        float v = rnd16(__fmul_rn(x, f));
        if (p.tau) v = rnd16(__fmul_rn(v, tau));
        if (lane < 16) r_s[r * 16 + lane] = v;
      }
      if (lane == 0) {
        m_s[r] = -INFINITY;
        l_s[r] = 0.f;
      }
    }

    if (p.stop == 2) return;
    const int interior_hi = (p.window == 0 && rows >= 12 && pos0 > p.E) ? ((pos0 - p.E) & ~31) : 0;
    const int u_wmma_start = interior_hi > 0 ? ((tile0 + ntiles) - (interior_hi >> 5)) : ntiles;
    const int u_scalar_limit = min(ulimit, u_wmma_start);
    int u = split;
#pragma unroll 1
    for (; u < u_scalar_limit; u += ustep) {
      const int base = (tile0 + ntiles - 1 - u) << 5;
      // ---- new keys of the tile: K/V short conv, K clip, V norm, cache writes.
      //      Work item i = (new key, K | V) runs on warp NW - 1 - w, lane = 4 dims. ----
      {
        const int nfirst = max(base, pos0) - pos0;
        const int nnew = min(base + TK, hi) - pos0 - nfirst;
#pragma unroll 1
        for (int i = NW - 1 - w; i < 2 * nnew; i += NW) {
          {
            const int n = nfirst + (i >> 1);
            const int s = i & 1;
            const int tok = q_start + n;
            const int kl = pos0 + n - base;
            const int64_t cpos = p.pos[tok];
            const int64_t cs = p.cslot[tok];
            const int64_t ks = p.kv_slot[tok];
            // RingConv / paged conv slot convention: -1 => padded (no read/write), -2 => read-only, >=0 => read+write.
            const bool cvalid = cs != -1;
            const bool cwrite = cs >= 0;
            const int qs = p.qstart[tok];
            const int64_t crow = (int64_t)p.seq_idx[tok] * p.stride_cbt;
            const int col = kvh * HD + lane * 4;
            const __half* x = p.x[s];
            const int64_t sx = p.stride_x[s];
            const uint32_t* xp = reinterpret_cast<const uint32_t*>(x + (int64_t)tok * sx + col);
            const uint2 curv = make_uint2(xp[0], xp[1]);
            uint2 tapv[3];
#pragma unroll
            for (int lag = 1; lag <= 3; lag++) {
              const int64_t src = cpos - lag;
              const int row = tok - lag;
              uint2 v = make_uint2(0u, 0u);
              if (cvalid && src >= 0) {
                if (row >= qs) {
                  const uint32_t* tp = reinterpret_cast<const uint32_t*>(x + (int64_t)row * sx + col);
                  v = make_uint2(tp[0], tp[1]);
                } else {
                  const int64_t lblk = min(src >> p.cshift, (int64_t)(p.cmax - 1));
                  const int64_t blk = p.cbt[crow + lblk];
                  v = *reinterpret_cast<const uint2*>(
                      p.cache + blk * p.stride_c_blk + (src & cmask) * p.stride_c_n + (int64_t)kvh * p.stride_c_h + s * HD + lane * 4);
                }
              }
              tapv[lag - 1] = v;
            }
            const float4 cur = up4(curv);
            const float4 c0 = cvalid ? cur : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 t1 = up4(tapv[0]), t2 = up4(tapv[1]), t3 = up4(tapv[2]);
            float4 y;
            if (!p.dyn[0]) {
              // static taps [C][4], oldest first: fma chain from +0
              const uint4 wv = *reinterpret_cast<const uint4*>(p.w[s] + (int64_t)col * 4);
              const float4 wa = up4(make_uint2(wv.x, wv.y)), wb = up4(make_uint2(wv.z, wv.w));
              const uint4 wv2 = *reinterpret_cast<const uint4*>(p.w[s] + (int64_t)col * 4 + 8);
              const float4 wc = up4(make_uint2(wv2.x, wv2.y)), wd = up4(make_uint2(wv2.z, wv2.w));
              y.x = __fmaf_rn(c0.x, wa.w, __fmaf_rn(t1.x, wa.z, __fmaf_rn(t2.x, wa.y, __fmaf_rn(t3.x, wa.x, 0.f))));
              y.y = __fmaf_rn(c0.y, wb.w, __fmaf_rn(t1.y, wb.z, __fmaf_rn(t2.y, wb.y, __fmaf_rn(t3.y, wb.x, 0.f))));
              y.z = __fmaf_rn(c0.z, wc.w, __fmaf_rn(t1.z, wc.z, __fmaf_rn(t2.z, wc.y, __fmaf_rn(t3.z, wc.x, 0.f))));
              y.w = __fmaf_rn(c0.w, wd.w, __fmaf_rn(t1.w, wd.z, __fmaf_rn(t2.w, wd.y, __fmaf_rn(t3.w, wd.x, 0.f))));
            } else {
              // dynamic taps: (static [4][C] + per-token per-head delta), newest first
              const float4 d = up4(*reinterpret_cast<const uint2*>(p.dyn[s] + (int64_t)tok * p.stride_dyn[s] + kvh * 4));
              const __half* ws = p.w[s] + col;
              const float4 w0 = up4(*reinterpret_cast<const uint2*>(ws));
              const float4 w1 = up4(*reinterpret_cast<const uint2*>(ws + p.stride_w0));
              const float4 w2 = up4(*reinterpret_cast<const uint2*>(ws + 2 * p.stride_w0));
              const float4 w3 = up4(*reinterpret_cast<const uint2*>(ws + 3 * p.stride_w0));
#define DYN1(c) __fmaf_rn(t3.c, __fadd_rn(w3.c, d.w), __fmaf_rn(t2.c, __fadd_rn(w2.c, d.z), \
                __fmaf_rn(t1.c, __fadd_rn(w1.c, d.y), __fmul_rn(c0.c, __fadd_rn(w0.c, d.x)))))
              y.x = DYN1(x); y.y = DYN1(y); y.z = DYN1(z); y.w = DYN1(w);
#undef DYN1
            }
            y.x = rnd16(__fmul_rn(__fadd_rn(y.x, cur.x), p.conv_scale));
            y.y = rnd16(__fmul_rn(__fadd_rn(y.y, cur.y), p.conv_scale));
            y.z = rnd16(__fmul_rn(__fadd_rn(y.z, cur.z), p.conv_scale));
            y.w = rnd16(__fmul_rn(__fadd_rn(y.w, cur.w), p.conv_scale));
            float ss;
            if (!p.dyn[0]) {
              ss = __fadd_rn(__fmul_rn(y.x, y.x), __fmul_rn(y.y, y.y));
              ss = __fadd_rn(ss, __fmul_rn(y.z, y.z));
              ss = __fadd_rn(ss, __fmul_rn(y.w, y.w));
              ss = warp_sum(ss);
            } else {
              const int l8 = lane & 7;
              float4 sq = make_float4(__fmul_rn(y.x, y.x), __fmul_rn(y.y, y.y), __fmul_rn(y.z, y.z), __fmul_rn(y.w, y.w));
              float4 pq;
#define TSUM(c) pq.c = __fadd_rn(__fadd_rn(__fadd_rn(shi(sq.c, l8), \
                shi(sq.c, l8 + 8)), shi(sq.c, l8 + 16)), \
                shi(sq.c, l8 + 24)); \
                pq.c = __fadd_rn(pq.c, shx(pq.c, 1));
              TSUM(x) TSUM(y) TSUM(z) TSUM(w)
#undef TSUM
              ss = __fadd_rn(__fadd_rn(pq.x, pq.z), __fadd_rn(pq.y, pq.w));
              ss = __fadd_rn(ss, shx(ss, 4));
              ss = __fadd_rn(ss, shx(ss, 2));
            }
            const float f = s == 0 ? clip_factor(ss, p.q_thr, p.eps)
                                   : rsqrtf_approx(__fadd_rn(__fdiv_rn(ss, (float)HD), p.norm_eps));
            const float z0 = __fmul_rn(y.x, f);
            const float z1 = __fmul_rn(y.y, f);
            const float z2 = __fmul_rn(y.z, f);
            const float z3 = __fmul_rn(y.w, f);
            if (cwrite) {
              *reinterpret_cast<uint2*>(p.cache + (cs >> p.cshift) * p.stride_c_blk + (cs & cmask) * p.stride_c_n +
                                        (int64_t)kvh * p.stride_c_h + s * HD + lane * 4) = curv;
            }
            float4 zf;
            if constexpr (IS_FP8) {
              const uint32_t zv8 = f32x4_to_e4m3x4(z0, z1, z2, z3);
              if (ks >= 0) {
                uint8_t* dst_base = reinterpret_cast<uint8_t*>(s == 0 ? p.kc : p.vc);
                const int64_t sb = s == 0 ? p.kb : p.vb;
                const int64_t ss_kv = s == 0 ? p.ks : p.vs;
                const int64_t sh = s == 0 ? p.kh : p.vh;
                *reinterpret_cast<uint32_t*>(dst_base + (ks >> p.page_shift) * sb + (ks & page_mask) * ss_kv +
                                             (int64_t)kvh * sh + lane * 4) = zv8;
              }
              zf = up4_e4m3(zv8);
            } else {
              uint2 zv;
              *reinterpret_cast<__half2*>(&zv.x) = __floats2half2_rn(z0, z1);
              *reinterpret_cast<__half2*>(&zv.y) = __floats2half2_rn(z2, z3);
              if (ks >= 0) {
                __half* dst_base = reinterpret_cast<__half*>(s == 0 ? p.kc : p.vc);
                const int64_t sb = s == 0 ? p.kb : p.vb;
                const int64_t ss_kv = s == 0 ? p.ks : p.vs;
                const int64_t sh = s == 0 ? p.kh : p.vh;
                *reinterpret_cast<uint2*>(dst_base + (ks >> p.page_shift) * sb + (ks & page_mask) * ss_kv +
                                          (int64_t)kvh * sh + lane * 4) = zv;
              }
              zf = up4(zv);
            }
            if (s == 0) *reinterpret_cast<float4*>(K_s + (lane * TK + kl) * 4) = zf;
            else *reinterpret_cast<float4*>(V_s + kl * HD + lane * 4) = zf;
          }
        }
      }
      // ---- cached keys of the tile: registers -> shared memory (unscaled fp32) ----
      {
        const int kl = tid >> 3, sub = tid & 7;
        const int key = base + kl;
        if (tid < 256 && (key < pos0 || key >= hi)) {  // new keys were staged by their conv above
          float4* kd = reinterpret_cast<float4*>(K_s + (sub * 4 * TK + kl) * 4);
          float4* vd = reinterpret_cast<float4*>(V_s + kl * HD + sub * 16);
          if constexpr (IS_FP8) {
            kd[0] = up4_e4m3(k0.x);
            kd[TK] = up4_e4m3(k0.y);
            kd[2 * TK] = up4_e4m3(k0.z);
            kd[3 * TK] = up4_e4m3(k0.w);
            vd[0] = up4_e4m3(v0.x);
            vd[1] = up4_e4m3(v0.y);
            vd[2] = up4_e4m3(v0.z);
            vd[3] = up4_e4m3(v0.w);
          } else {
            kd[0] = up4(make_uint2(k0.x, k0.y));
            kd[TK] = up4(make_uint2(k0.z, k0.w));
            kd[2 * TK] = up4(make_uint2(k1.x, k1.y));
            kd[3 * TK] = up4(make_uint2(k1.z, k1.w));
            vd[0] = up4(make_uint2(v0.x, v0.y));
            vd[1] = up4(make_uint2(v0.z, v0.w));
            vd[2] = up4(make_uint2(v1.x, v1.y));
            vd[3] = up4(make_uint2(v1.z, v1.w));
          }
        }
      }
      __syncthreads();
      if (p.stop == 3) return;
      if (p.pf) {
        if (u + ustep < u_scalar_limit) issue_pg(u + ustep, pg1);
        prefetch_pg(u + 3 * ustep, pg3);
        pg1 = pg2; pg2 = pg3; pg3 = page_of(u + 4 * ustep);
      } else if (u + ustep < u_scalar_limit) {
        issue(u + ustep);  // next tile in flight during this tile's rows
      }
      if (p.stop == 9) { __syncthreads(); continue; }  // tuning aid: tile loads + staging only
      // ---- rows of this warp: QK (lane = key), bias, softmax, P.V (lane = 4 dims) ----
#pragma unroll 1
      for (int r = w; r < rows; r += NW) {
        const int tk = r / GQ;
        const int key = base + lane;
        const int dist = pos0 + tk - key;
        const bool vis = key >= lo && key < hi && dist >= 0 && (p.window == 0 || dist < p.window);
        const bool inb = vis && dist < p.E;
        uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
        if (inb) {
          const uint4* bp = reinterpret_cast<const uint4*>(p.rel_t + (int64_t)dist * 16);
          b0 = __ldg(bp);
          b1 = __ldg(bp + 1);
        }
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        const float* qr = q_s + r * HD;
        const float* kr = K_s + lane * 4;
#pragma unroll 2
        for (int qd = 0; qd < HD / 4; qd++) {
          const float4 kq = *reinterpret_cast<const float4*>(kr + qd * (TK * 4));
          const float4 qq = *reinterpret_cast<const float4*>(qr + qd * 4);
          a0 = fmaf(kq.x, qq.x, a0);
          a1 = fmaf(kq.y, qq.y, a1);
          a2 = fmaf(kq.z, qq.z, a2);
          a3 = fmaf(kq.w, qq.w, a3);
        }
        float sc = (a0 + a1) + (a2 + a3);
        {
          const float4 r0 = *reinterpret_cast<const float4*>(r_s + r * 16);
          const float4 r1 = *reinterpret_cast<const float4*>(r_s + r * 16 + 4);
          const float4 r2 = *reinterpret_cast<const float4*>(r_s + r * 16 + 8);
          const float4 r3 = *reinterpret_cast<const float4*>(r_s + r * 16 + 12);
          const float4 p0 = up4(make_uint2(b0.x, b0.y)), p1 = up4(make_uint2(b0.z, b0.w));
          const float4 p2 = up4(make_uint2(b1.x, b1.y)), p3 = up4(make_uint2(b1.z, b1.w));
          float b = 0.f;
          b = fmaf(r0.x, p0.x, b); b = fmaf(r0.y, p0.y, b); b = fmaf(r0.z, p0.z, b); b = fmaf(r0.w, p0.w, b);
          b = fmaf(r1.x, p1.x, b); b = fmaf(r1.y, p1.y, b); b = fmaf(r1.z, p1.z, b); b = fmaf(r1.w, p1.w, b);
          b = fmaf(r2.x, p2.x, b); b = fmaf(r2.y, p2.y, b); b = fmaf(r2.z, p2.z, b); b = fmaf(r2.w, p2.w, b);
          b = fmaf(r3.x, p3.x, b); b = fmaf(r3.y, p3.y, b); b = fmaf(r3.z, p3.z, b); b = fmaf(r3.w, p3.w, b);
          b = inb ? rnd16(b) : 0.f;
          sc = vis ? fmaf(sc, p.scale_log2, b * LOG2E_F) : -INFINITY;
        }
        const float mo = m_s[r];
        const float mn = fmaxf(mo, warp_max(sc));
        const float ms = mn == -INFINITY ? 0.f : mn;
        const float alpha = ex2f(mo - ms);
        const float pr = ex2f(sc - ms);
        float lt = pr;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) lt += shx(lt, o);
        p_s[r * TK + lane] = pr;
        if (lane == 0) {
          m_s[r] = mn;
          l_s[r] = l_s[r] * alpha + lt;
        }
        __syncwarp();
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        const float* vr = V_s + lane * 4;
        const float* pp = p_s + r * TK;
#pragma unroll 2
        for (int j = 0; j < TK; j++) {
          const float4 vq = *reinterpret_cast<const float4*>(vr + j * HD);
          const float pj = pp[j];
          c0 = fmaf(pj, vq.x, c0);
          c1 = fmaf(pj, vq.y, c1);
          c2 = fmaf(pj, vq.z, c2);
          c3 = fmaf(pj, vq.w, c3);
        }
        float4* mp = reinterpret_cast<float4*>(macc + r * HD + lane * 4);
        float4 mv = *mp;
        mv.x = fmaf(mv.x, alpha, c0);
        mv.y = fmaf(mv.y, alpha, c1);
        mv.z = fmaf(mv.z, alpha, c2);
        mv.w = fmaf(mv.w, alpha, c3);
        *mp = mv;
      }
      __syncthreads();
    }

    // ---- WMMA tensor-core interior loop for global layers (dist >= E, no bias/mask/conv) ----
    if (u < ulimit) {
      using namespace nvcuda;
      constexpr int W_BM = 32;
      constexpr int W_LDK = HD + 8;  // 136 halves
      constexpr int W_LDS = TK + 4;  // 36 floats
      constexpr int W_LDP = TK + 8;  // 40 halves
      // Reuse K_s + V_s (32,768 bytes) for all WMMA staging -> zero extra shared memory.
      __half* w_sQ = reinterpret_cast<__half*>(K_s);  // 32 * 136 * 2 = 8,704 B
      // 1) Convert q_s [rows, 128] (fp32) -> w_sQ [32, 136] (fp16)
      for (int idx = tid; idx < W_BM * (HD / 4); idx += NW * 32) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        __half2 h0 = __floats2half2_rn(0.f, 0.f), h1 = h0;
        if (r < rows) {
          const float4 qv = *reinterpret_cast<const float4*>(q_s + r * HD + c4 * 4);
          h0 = __floats2half2_rn(qv.x, qv.y);
          h1 = __floats2half2_rn(qv.z, qv.w);
        }
        __half2* dst2 = reinterpret_cast<__half2*>(w_sQ + r * W_LDK + c4 * 4);
        dst2[0] = h0;
        dst2[1] = h1;
      }
      __syncthreads();

      // Load Q fragments into registers for warps 0..3 (wm = w>>1, wn = w&1)
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[8];
      if (w < 4) {
        const int q_wm = w >> 1;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          wmma::load_matrix_sync(q_frag[k], w_sQ + q_wm * 16 * W_LDK + k * 16, W_LDK);
        }
      }
      __syncthreads();

      // 2) Copy existing macc [rows, 128] -> sO_tmp [32, 128] (in K_s, 16 KB) and load into o_frag[0..1].
      // When IS_FP8=true, sV_h stores 2^-8 * V, so divide existing unscaled macc by 256.0f (exact exponent shift)
      // before entering WMMA, then multiply by 256.0f when copying sO_tmp back to macc after WMMA.
      float* sO_tmp = K_s;
      for (int idx = tid; idx < W_BM * (HD / 4); idx += NW * 32) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        float4 mv = (r < rows) ? *reinterpret_cast<const float4*>(macc + r * HD + c4 * 4)
                               : make_float4(0.f, 0.f, 0.f, 0.f);
        if constexpr (IS_FP8) {
          constexpr float INV_256 = 1.0f / 256.0f;
          mv.x *= INV_256;
          mv.y *= INV_256;
          mv.z *= INV_256;
          mv.w *= INV_256;
        }
        *reinterpret_cast<float4*>(sO_tmp + r * HD + c4 * 4) = mv;
      }
      __syncthreads();

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[2];
      if (w < 8) {
        const int o_wm = w >> 2;
        const int o_wd = w & 3;
        wmma::load_matrix_sync(o_frag[0], sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 0), HD, wmma::mem_row_major);
        wmma::load_matrix_sync(o_frag[1], sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 16), HD, wmma::mem_row_major);
      }
      __syncthreads();

      // Layout inside K_s + V_s (32,768 B total):
      // sK_h: 8,704 B | sV_h: 8,704 B | sS_w: 4,608 B | sP_h: 2,560 B | sAlpha: 128 B => 24,704 B <= 32,768 B.
      __half* sK_h = reinterpret_cast<__half*>(K_s);
      __half* sV_h = sK_h + TK * W_LDK;
      float* sS_w = reinterpret_cast<float*>(sV_h + TK * W_LDK);
      __half* sP_h = reinterpret_cast<__half*>(sS_w + W_BM * W_LDS);
      float* sAlpha = reinterpret_cast<float*>(sP_h + W_BM * W_LDP);

      const int* bt_row = p.block_table + (int64_t)req * p.stride_bt;
      const float w_scale_log2 = IS_FP8 ? (p.scale_log2 * 256.0f) : p.scale_log2;

      // Software-pipelined K/V loads: issue tile u+ustep into registers while WMMA runs on tile u.
      constexpr int NLD_KV = (TK * (HD / 8)) / (NW * 32);  // 1 for NW=16, 2 for NW=8
      using KvRegT = typename std::conditional<IS_FP8, uint2, uint4>::type;
      KvRegT pf_k[NLD_KV], pf_v[NLD_KV];
      auto load_kv_regs = [&](int uu) {
        const int base = (tile0 + ntiles - 1 - uu) << 5;
#pragma unroll
        for (int it = 0; it < NLD_KV; ++it) {
          const int idx = tid + it * (NW * 32);
          const int kl = idx >> 4;
          const int c8 = idx & 15;
          const int key = base + kl;
          const int page = bt_row[key >> p.page_shift];
          const int64_t slot = (int64_t)(key & page_mask);
          const int64_t koff = (int64_t)page * p.kb + slot * p.ks + (int64_t)kvh * p.kh;
          const int64_t voff = (int64_t)page * p.vb + slot * p.vs + (int64_t)kvh * p.vh;
          ld_kv_wmma(pf_k[it], p.kc, koff, c8);
          ld_kv_wmma(pf_v[it], p.vc, voff, c8);
        }
      };
      load_kv_regs(u);
#pragma unroll 1
      for (; u < ulimit; u += ustep) {
        // Store prefetched K/V into shared memory (FP8 unpacks via e4m3x4 directly to 2^-8-scaled FP16).
#pragma unroll
        for (int it = 0; it < NLD_KV; ++it) {
          const int idx = tid + it * (NW * 32);
          const int kl = idx >> 4;
          const int c8 = idx & 15;
          st_kv_wmma(sK_h + kl * W_LDK + c8 * 8, pf_k[it]);
          st_kv_wmma(sV_h + kl * W_LDK + c8 * 8, pf_v[it]);
        }
        __syncthreads();
        // Overlap next tile HBM loads with this tile WMMA + softmax.
        if (u + ustep < ulimit) load_kv_regs(u + ustep);

        // QK^T: warps 0..3 cover [32, 32] (4 tiles of 16x16)
        if (w < 4) {
          const int q_wm = w >> 1;
          const int q_wn = w & 1;
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
          wmma::fill_fragment(s_frag, 0.f);
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, sK_h + q_wn * 16 * W_LDK + k * 16, W_LDK);
            wmma::mma_sync(s_frag, q_frag[k], k_frag, s_frag);
          }
          wmma::store_matrix_sync(sS_w + q_wm * 16 * W_LDS + q_wn * 16, s_frag, W_LDS, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: warps 0..7 handle 4 rows each (32 cols = 1 col/lane)
        if (w < 8) {
#pragma unroll
          for (int dr = 0; dr < 4; ++dr) {
            const int r = w * 4 + dr;
            if (r < rows) {
              const float sc = sS_w[r * W_LDS + lane] * w_scale_log2;
              const float mo = m_s[r];
              const float mn = fmaxf(mo, warp_max(sc));
              const float ms = (mn == -INFINITY) ? 0.f : mn;
              const float alpha = (mo == -INFINITY) ? 0.f : ex2f(mo - ms);
              const float pr = ex2f(sc - ms);
              float lt = pr;
#pragma unroll
              for (int o = 16; o > 0; o >>= 1) lt += shx(lt, o);
              sP_h[r * W_LDP + lane] = __float2half_rn(pr);
              if (lane == 0) {
                m_s[r] = mn;
                l_s[r] = l_s[r] * alpha + lt;
                sAlpha[r] = alpha;
              }
            } else {
              sP_h[r * W_LDP + lane] = __float2half_rn(0.f);
              if (lane == 0) sAlpha[r] = 1.f;
            }
          }
        }
        __syncthreads();

        // Rescale o_frag & P @ V: warps 0..7 (o_wm = w>>2, o_wd = w&3 -> 32 dims/warp)
        if (w < 8) {
          const int o_wm = w >> 2;
          const int o_wd = w & 3;
          const int base_r = o_wm * 16;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = base_r + (lane & 1) + 2 * ((e >> 1) & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);
            const float a = sAlpha[r];
            o_frag[0].x[e] *= a;
            o_frag[1].x[e] *= a;
          }
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
            wmma::load_matrix_sync(p_frag, sP_h + o_wm * 16 * W_LDP + k * 16, W_LDP);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v0_frag, v1_frag;
            wmma::load_matrix_sync(v0_frag, sV_h + k * 16 * W_LDK + (o_wd * 32 + 0), W_LDK);
            wmma::load_matrix_sync(v1_frag, sV_h + k * 16 * W_LDK + (o_wd * 32 + 16), W_LDK);
            wmma::mma_sync(o_frag[0], p_frag, v0_frag, o_frag[0]);
            wmma::mma_sync(o_frag[1], p_frag, v1_frag, o_frag[1]);
          }
        }
        __syncthreads();
      }

      // Write final o_frag back to sO_tmp (K_s) and copy [rows, 128] -> macc (restoring 256.0f scale when IS_FP8)
      if (w < 8) {
        const int o_wm = w >> 2;
        const int o_wd = w & 3;
        wmma::store_matrix_sync(sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 0), o_frag[0], HD, wmma::mem_row_major);
        wmma::store_matrix_sync(sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 16), o_frag[1], HD, wmma::mem_row_major);
      }
      __syncthreads();
      for (int idx = tid; idx < rows * (HD / 4); idx += NW * 32) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        float4 ov = *reinterpret_cast<const float4*>(sO_tmp + r * HD + c4 * 4);
        if constexpr (IS_FP8) {
          ov.x *= 256.0f;
          ov.y *= 256.0f;
          ov.z *= 256.0f;
          ov.w *= 256.0f;
        }
        *reinterpret_cast<float4*>(macc + r * HD + c4 * 4) = ov;
      }
      __syncthreads();
    }

    if (p.stop == 4) return;
    // ---- this CTA's partial: raw unscaled accumulators + (max, sum) ----
#pragma unroll 1
    for (int r = w; r < rows; r += NW) {
      const int tk = r / GQ;
      const int64_t idx = ((int64_t)split * p.T + q_start + tk) * p.Hq + kvh * GQ + (r - tk * GQ);
      *reinterpret_cast<float4*>(p.o_part + idx * HD + lane * 4) = *reinterpret_cast<const float4*>(macc + r * HD + lane * 4);
      if (lane == 0) *reinterpret_cast<float2*>(p.ml_part + idx * 2) = make_float2(m_s[r], l_s[r]);
    }
    __threadfence();
  }

  // ---- arrival at the grid barrier of the S CTAs of this (request, KV head): counter wraps
  //      back to 0 on the S-th arrival, self-resetting for the next launch ----
  __syncthreads();
  if (p.stop >= 1 && p.stop <= 5) return;
  int* ctr = p.counters + req * p.Hkv + kvh;
  if (tid == 0) atomicInc(reinterpret_cast<unsigned int*>(ctr), (unsigned int)(S - 1));

  // ---- combine: output row r is merged by CTA S - 1 - r ----
  const int r0 = S - 1 - split;
  if (r0 >= rows) return;
  const int64_t part_stride = (int64_t)p.T * p.Hq;
  // When S > 1, pass 0 runs the combine once while waiting to warm the SM I-cache; pass 1 runs after barrier.
  const int pass_start = (S == 1) ? 1 : 0;
#pragma unroll 1
  for (int pass = pass_start; pass < 2; pass++) {
    if (pass == 1) {
      if (tid == 0) {
        int it = 0;
        while (ld_volatile(ctr) != 0 && ++it < SPIN_MAX) {
        }
        if (it >= SPIN_MAX) *p.fault = 1;
      }
      __syncthreads();
      if (p.stop == 6) return;
    }
#pragma unroll 1
    for (int r = r0; r < rows; r += S) {
      const int tk = r / GQ;
      const int token = q_start + tk;
      const int head = kvh * GQ + (r - tk * GQ);
      const int64_t row_idx = (int64_t)token * p.Hq + head;
      {
        float4 pv[CSP];
#pragma unroll
        for (int j = 0; j < CSP; j++) {
          const int sp = w + NWC * j;
          pv[j] = make_float4(0.f, 0.f, 0.f, 0.f);
          if (w < NWC && sp < active) {
            pv[j] = __ldcg(reinterpret_cast<const float4*>(p.o_part + ((int64_t)sp * part_stride + row_idx) * HD + lane * 4));
          }
        }
        float mx = -INFINITY;
        if (tid < active) {
          const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.ml_part + ((int64_t)tid * part_stride + row_idx) * 2));
          cm_s[tid] = ml.x;
          cl_s[tid] = ml.y;
          mx = ml.x;
        }
        if (w < MAXW) {  // active <= MS: the maxima sit on the first MAXW warps
          mx = warp_max(mx);
          if (lane == 0) K_s[NWC * HD + NWC + w] = mx;
        }
        __syncthreads();
        if (pass && p.stop == 7) return;
        float M = K_s[NWC * HD + NWC];
#pragma unroll
        for (int i = 1; i < MAXW; i++) M = fmaxf(M, K_s[NWC * HD + NWC + i]);
        const float Ms = M == -INFINITY ? 0.f : M;
        float4 O = make_float4(0.f, 0.f, 0.f, 0.f);
        float den = 0.f;
#pragma unroll
        for (int j = 0; j < CSP; j++) {
          const int sp = w + NWC * j;
          float wt = 0.f, lw = 0.f;
          if (w < NWC && sp < active) {
            wt = ex2f(cm_s[sp] - Ms);
            lw = cl_s[sp];
          }
          den = fmaf(lw, wt, den);
          O.x = fmaf(pv[j].x, wt, O.x);
          O.y = fmaf(pv[j].y, wt, O.y);
          O.z = fmaf(pv[j].z, wt, O.z);
          O.w = fmaf(pv[j].w, wt, O.w);
        }
        // K_s is free now: [NWC][HD] warp sums, then [NWC] denominators
        if (w < NWC) {
          *reinterpret_cast<float4*>(K_s + w * HD + lane * 4) = O;
          if (lane == 0) K_s[NWC * HD + w] = den;
        }
        __syncthreads();
        if (pass && p.stop == 8) return;
        if (w == 0) {
          den = 0.f;
          O = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
          for (int ww = 0; ww < NWC; ww++) {
            const float4 a = *reinterpret_cast<const float4*>(K_s + ww * HD + lane * 4);
            den += K_s[NWC * HD + ww];
            O.x += a.x; O.y += a.y; O.z += a.z; O.w += a.w;
          }
          const float lse2 = den > 0.f ? Ms + lg2f(den) : -INFINITY;
          const float sk = sigmoidf_(lse2 * LN2_F - __half2float(p.sinks[(int64_t)token * p.stride_sink + head]));
          // gate is a slice of the qkvr projection: rows are 4-byte aligned
          const uint32_t* gp = reinterpret_cast<const uint32_t*>(p.gate + (int64_t)token * p.stride_gate + head * HD + lane * 4);
          const float4 gt = up4(make_uint2(gp[0], gp[1]));
          float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
          if (den > 0.f) {
            y.x = (O.x / den) * sigmoidf_(gt.x) * sk;
            y.y = (O.y / den) * sigmoidf_(gt.y) * sk;
            y.z = (O.z / den) * sigmoidf_(gt.z) * sk;
            y.w = (O.w / den) * sigmoidf_(gt.w) * sk;
          }
          uint2 yv;
          *reinterpret_cast<__half2*>(&yv.x) = __floats2half2_rn(y.x, y.y);
          *reinterpret_cast<__half2*>(&yv.y) = __floats2half2_rn(y.z, y.w);
          if (pass == 1) *reinterpret_cast<uint2*>(p.out + (int64_t)token * (p.Hq * HD) + head * HD + lane * 4) = yv;
        }
      }
      __syncthreads();
    }
  }
}

template <bool IS_FP8, int NW, int MS>
static void ensure_smem_attr(int dev) {
  static uint64_t configured_mask = 0;
  const uint64_t bit = (dev >= 0 && dev < 64) ? (1ull << dev) : 0ull;
  if (bit != 0ull && (configured_mask & bit)) return;
  int max_optin = 96 * 1024;
  if (cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
    max_optin = 96 * 1024;
  }
  cudaFuncAttributes attr{};
  TORCH_CHECK(cudaFuncGetAttributes(&attr, attn2_kernel<IS_FP8, NW, MS>) == cudaSuccess,
              "cudaFuncGetAttributes failed for attn2_kernel");
  const int target = std::min(96 * 1024, std::max(0, max_optin - static_cast<int>(attr.sharedSizeBytes)));
  TORCH_CHECK(cudaFuncSetAttribute(attn2_kernel<IS_FP8, NW, MS>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   target) == cudaSuccess,
              "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed for attn2_kernel");
  if (bit != 0ull) configured_mask |= bit;
}

template <bool IS_FP8, int NW, int MS>
static void launch_attn2(dim3 grid, const Params& p, size_t smem, int dev, cudaStream_t stream) {
  ensure_smem_attr<IS_FP8, NW, MS>(dev);
  attn2_kernel<IS_FP8, NW, MS><<<grid, NW * 32, smem, stream>>>(p);
}

}  // namespace

void configure_dec2_smem() {
  int dev = 0;
  TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice failed");
  ensure_smem_attr<false, 8, 40>(dev);
  ensure_smem_attr<false, 16, 40>(dev);
  ensure_smem_attr<false, 8, 80>(dev);
  ensure_smem_attr<true, 8, 40>(dev);
  ensure_smem_attr<true, 16, 40>(dev);
  ensure_smem_attr<true, 8, 80>(dev);
}

#define CHK_HALF(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kHalf, #t " must be CUDA fp16")
#define CHK_INT(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kInt, #t " must be CUDA int32")
#define CHK_LONG(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kLong, #t " must be CUDA int64")
#define CHK_F32(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kFloat, #t " must be CUDA fp32")

void attn_rel_dec2_forward(
    torch::Tensor q, torch::Tensor r, torch::Tensor q_scale, torch::Tensor r_scale,
    c10::optional<torch::Tensor> tau, torch::Tensor rel_t,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor cu_q,
    torch::Tensor gate, torch::Tensor sinks, torch::Tensor out,
    torch::Tensor o_part, torch::Tensor ml_part, torch::Tensor counters, torch::Tensor fault,
    torch::Tensor kx, torch::Tensor vx, torch::Tensor kw, torch::Tensor vw,
    c10::optional<torch::Tensor> kdyn, c10::optional<torch::Tensor> vdyn,
    torch::Tensor cache, torch::Tensor pos, torch::Tensor kv_slot, torch::Tensor cslot,
    torch::Tensor seq_idx, torch::Tensor qstart, torch::Tensor cbt,
    int64_t num_reqs, int64_t max_query_len, int64_t num_actual_tokens,
    int64_t num_splits, int64_t window,
    double softmax_scale, double q_thr, double r_thr, double eps, double conv_scale, double norm_eps,
    int64_t stop, int64_t num_warps, int64_t max_splits, int64_t prefetch) {
  const at::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  CHK_HALF(q); CHK_HALF(r); CHK_HALF(q_scale); CHK_HALF(r_scale); CHK_HALF(rel_t);
  CHK_HALF(gate); CHK_HALF(sinks); CHK_HALF(out); CHK_HALF(kx); CHK_HALF(vx); CHK_HALF(kw); CHK_HALF(vw);
  CHK_HALF(cache);
  CHK_INT(block_table); CHK_INT(seq_lens); CHK_INT(cu_q); CHK_INT(seq_idx); CHK_INT(qstart); CHK_INT(cbt);
  CHK_INT(counters); CHK_INT(fault);
  CHK_LONG(pos); CHK_LONG(kv_slot); CHK_LONG(cslot);
  CHK_F32(o_part); CHK_F32(ml_part);

  TORCH_CHECK(q.dim() == 3 && q.size(2) == HD && q.stride(2) == 1 && q.stride(1) == HD,
              "q must be FP16 [T, Hq, 128] with stride(2)==1, stride(1)==128");
  const int T = static_cast<int>(q.size(0));
  if (T == 0) return;

  const int Hq = static_cast<int>(q.size(1));
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous() && out.size(0) == T && out.size(1) == Hq * HD,
              "out must be contiguous FP16 [T, Hq*128]");
  TORCH_CHECK(((uintptr_t)out.data_ptr() & 7u) == 0u, "out must be 8-byte aligned");

  const int act_toks = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(num_actual_tokens, T)));
  if (act_toks == 0 || num_reqs == 0) {
    TORCH_CHECK(cudaMemsetAsync(out.data_ptr(), 0, static_cast<size_t>(T) * Hq * HD * sizeof(__half), stream) == cudaSuccess);
    return;
  }

  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() && key_cache.dim() == 4 && value_cache.dim() == 4 &&
                  key_cache.sizes() == value_cache.sizes(),
              "key_cache and value_cache must be CUDA 4D tensors with matching shapes [nb, page, Hkv, 128]");
  TORCH_CHECK(key_cache.size(3) == HD && key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "key_cache/value_cache must have head_dim=128 and stride(3)==1");
  const bool is_fp8 = key_cache.element_size() == 1;
  TORCH_CHECK(is_fp8 || key_cache.scalar_type() == at::kHalf,
              "key_cache must be uint8/float8_e4m3fn or float16");
  TORCH_CHECK(key_cache.element_size() == value_cache.element_size() &&
                  (is_fp8 || value_cache.scalar_type() == at::kHalf),
              "key_cache and value_cache dtypes must match");
  const int64_t kv_es = static_cast<int64_t>(key_cache.element_size());
  TORCH_CHECK(((uintptr_t)key_cache.data_ptr() & 15u) == 0u && ((uintptr_t)value_cache.data_ptr() & 15u) == 0u,
              "key_cache and value_cache must be 16-byte aligned");
  TORCH_CHECK(((key_cache.stride(0) * kv_es) & 15) == 0 && ((key_cache.stride(1) * kv_es) & 15) == 0 &&
                  ((key_cache.stride(2) * kv_es) & 15) == 0,
              "key_cache strides must be 16-byte aligned");
  TORCH_CHECK(((value_cache.stride(0) * kv_es) & 15) == 0 && ((value_cache.stride(1) * kv_es) & 15) == 0 &&
                  ((value_cache.stride(2) * kv_es) & 15) == 0,
              "value_cache strides must be 16-byte aligned");

  const int Hkv = static_cast<int>(key_cache.size(2));
  TORCH_CHECK(Hq == Hkv * GQ, "attn_rel_dec2 supports GQA group size 3 only");

  TORCH_CHECK(((uintptr_t)q.data_ptr() & 3u) == 0u && (q.stride(0) & 1) == 0,
              "q must be 4-byte aligned with even row stride");
  TORCH_CHECK(r.dim() == 3 && r.size(0) >= T && r.size(1) == Hq && r.size(2) == 16 &&
                  r.stride(2) == 1 && r.stride(1) == 16,
              "r must be FP16 [T, Hq, 16] with stride(2)==1, stride(1)==16");
  TORCH_CHECK(q_scale.is_contiguous() && q_scale.dim() == 2 && q_scale.size(0) == Hq && q_scale.size(1) == HD &&
                  ((uintptr_t)q_scale.data_ptr() & 7u) == 0u,
              "q_scale must be contiguous 8B-aligned FP16 [Hq, 128]");
  TORCH_CHECK(r_scale.is_contiguous() && r_scale.dim() == 2 && r_scale.size(0) == Hq && r_scale.size(1) == 16,
              "r_scale must be contiguous FP16 [Hq, 16]");
  if (tau.has_value()) {
    CHK_F32(*tau);
    TORCH_CHECK(tau->dim() == 1 && tau->stride(0) == 1 && tau->size(0) >= T,
                "tau must be 1D contiguous FP32 [>= T]");
  }
  TORCH_CHECK(rel_t.is_contiguous() && rel_t.dim() == 2 && rel_t.size(1) == 16 &&
                  ((uintptr_t)rel_t.data_ptr() & 15u) == 0u,
              "rel_t must be contiguous 16B-aligned FP16 [E, 16]");
  TORCH_CHECK(gate.dim() == 2 && gate.size(0) >= T && gate.size(1) == Hq * HD && gate.stride(1) == 1 &&
                  ((uintptr_t)gate.data_ptr() & 3u) == 0u && (gate.stride(0) & 1) == 0,
              "gate must be 4B-aligned FP16 [T, Hq*128] with stride(1)==1, even stride(0)");
  TORCH_CHECK(sinks.dim() == 2 && sinks.size(0) >= T && sinks.size(1) == Hq && sinks.stride(1) == 1,
              "sinks must be FP16 [T, Hq] with stride(1)==1");
  TORCH_CHECK(kx.dim() == 2 && vx.dim() == 2 && kx.size(0) >= T && vx.size(0) >= T &&
                  kx.size(1) == Hkv * HD && vx.size(1) == Hkv * HD &&
                  kx.stride(1) == 1 && vx.stride(1) == 1 &&
                  ((uintptr_t)kx.data_ptr() & 3u) == 0u && ((uintptr_t)vx.data_ptr() & 3u) == 0u &&
                  (kx.stride(0) & 1) == 0 && (vx.stride(0) & 1) == 0,
              "kx/vx must be 4B-aligned FP16 [T, Hkv*128] with stride(1)==1, even stride(0)");

  TORCH_CHECK(cache.dim() == 4 && cache.size(1) == Hkv && cache.size(3) >= 2 * HD && cache.stride(3) == 1 &&
                  ((uintptr_t)cache.data_ptr() & 7u) == 0u &&
                  (cache.stride(0) & 3) == 0 && (cache.stride(1) & 3) == 0 && (cache.stride(2) & 3) == 0,
              "conv cache must be 8B-aligned FP16 [cb, Hkv, N, >=256] with stride(3)==1 and 4-element-aligned strides");

  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) >= num_reqs && block_table.stride(1) == 1,
              "block_table must be int32 [>=R, max_blocks] with stride(1)==1");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous() && seq_lens.size(0) >= num_reqs,
              "seq_lens must be contiguous int32 [>=R]");
  TORCH_CHECK(cu_q.dim() == 1 && cu_q.is_contiguous() && cu_q.size(0) >= num_reqs + 1,
              "cu_q must be contiguous int32 [>=R+1]");
  TORCH_CHECK(cbt.dim() == 2 && cbt.size(0) >= num_reqs && cbt.size(1) >= 1 && cbt.stride(1) == 1,
              "conv_block_table must be int32 [>=R, >=1] with stride(1)==1");
  TORCH_CHECK(pos.dim() == 1 && pos.stride(0) == 1 && pos.size(0) >= T, "pos must be contiguous int64 [>=T]");
  TORCH_CHECK(kv_slot.dim() == 1 && kv_slot.stride(0) == 1 && kv_slot.size(0) >= T, "kv_slot must be contiguous int64 [>=T]");
  TORCH_CHECK(cslot.dim() == 1 && cslot.stride(0) == 1 && cslot.size(0) >= T, "cslot must be contiguous int64 [>=T]");
  TORCH_CHECK(seq_idx.dim() == 1 && seq_idx.stride(0) == 1 && seq_idx.size(0) >= T, "seq_idx must be contiguous int32 [>=T]");
  TORCH_CHECK(qstart.dim() == 1 && qstart.stride(0) == 1 && qstart.size(0) >= T, "qstart must be contiguous int32 [>=T]");

  const bool dynamic = kdyn.has_value();
  TORCH_CHECK(dynamic == vdyn.has_value(), "k_dynamic and v_dynamic must both be present or both None");
  if (dynamic) {
    CHK_HALF(*kdyn); CHK_HALF(*vdyn);
    TORCH_CHECK(kw.dim() == 2 && vw.dim() == 2 && kw.size(0) == 4 && kw.size(1) == Hkv * HD &&
                    vw.sizes() == kw.sizes() && kw.stride(1) == 1 && vw.stride(1) == 1 &&
                    vw.stride(0) == kw.stride(0) && (kw.stride(0) & 3) == 0 &&
                    ((uintptr_t)kw.data_ptr() & 7u) == 0u && ((uintptr_t)vw.data_ptr() & 7u) == 0u,
                "dynamic conv weights must be 8B-aligned FP16 [4, Hkv*128] with matching strides");
    TORCH_CHECK(kdyn->dim() == 2 && vdyn->dim() == 2 && kdyn->size(0) >= T && vdyn->size(0) >= T &&
                    kdyn->size(1) == Hkv * 4 && vdyn->size(1) == Hkv * 4 &&
                    kdyn->stride(1) == 1 && vdyn->stride(1) == 1 &&
                    (kdyn->stride(0) & 3) == 0 && (vdyn->stride(0) & 3) == 0 &&
                    ((uintptr_t)kdyn->data_ptr() & 7u) == 0u && ((uintptr_t)vdyn->data_ptr() & 7u) == 0u,
                "k_dynamic/v_dynamic must be 8B-aligned FP16 [>=T, Hkv*4] with 4-element row strides");
  } else {
    TORCH_CHECK(kw.is_contiguous() && vw.is_contiguous() && kw.dim() == 2 && vw.dim() == 2 &&
                    kw.size(0) == Hkv * HD && kw.size(1) == 4 && vw.sizes() == kw.sizes() &&
                    ((uintptr_t)kw.data_ptr() & 15u) == 0u && ((uintptr_t)vw.data_ptr() & 15u) == 0u,
                "static conv weights must be contiguous 16B-aligned FP16 [Hkv*128, 4]");
  }

  int ps = 0;
  while ((1LL << ps) < key_cache.size(1)) ps++;
  int cs = 0;
  while ((1LL << cs) < cache.size(2)) cs++;
  TORCH_CHECK(key_cache.size(1) > 0 && (1LL << ps) == key_cache.size(1) &&
                  cache.size(2) > 0 && (1LL << cs) == cache.size(2),
              "KV and conv cache page/ring sizes must be powers of 2");

  TORCH_CHECK(max_query_len >= 1 && max_query_len <= 8, "max_query_len must be 1..8");
  TORCH_CHECK((num_warps == 8 || num_warps == 16) && (max_splits == 40 || max_splits == 80) &&
                  !(num_warps == 16 && max_splits == 80),
              "unsupported (num_warps, max_splits) variant; allowed: (8,40), (16,40), (8,80)");

  const int maxrows = static_cast<int>(max_query_len) * GQ;
  const size_t smem = static_cast<size_t>(2 * TK * HD + maxrows * (2 * HD + 16 + TK + 2) + 2 * max_splits) * sizeof(float);
  // Barrier residency: 1 CTA/SM (budget=80), or 2 CTAs/SM (budget=160) for 8-warp kernel when smem <= 47 KB.
  const bool two = (num_warps == 8) && (smem <= 47 * 1024);
  const int64_t budget = two ? 160 : 80;
  TORCH_CHECK(num_splits >= 1 && num_splits <= max_splits && num_splits * Hkv * num_reqs <= budget,
              "grid (S * H_kv * R) must fit resident SMs for grid barrier");
  TORCH_CHECK(max_splits == 40 || two || num_splits * Hkv * num_reqs <= 80,
              "max_splits=80 requires 2 CTAs/SM (NW=8 and smem <= 47 KB)");

  const int64_t slots = num_splits * T * Hq;
  TORCH_CHECK(o_part.is_contiguous() && o_part.numel() >= slots * HD && ((uintptr_t)o_part.data_ptr() & 15u) == 0u,
              "o_part workspace too small or misaligned");
  TORCH_CHECK(ml_part.is_contiguous() && ml_part.numel() >= slots * 2 && ((uintptr_t)ml_part.data_ptr() & 7u) == 0u,
              "ml_part workspace too small or misaligned");
  TORCH_CHECK(counters.is_contiguous() && counters.numel() >= num_reqs * Hkv,
              "counters workspace too small (need >= num_reqs * Hkv)");
  TORCH_CHECK(fault.is_contiguous() && fault.numel() >= 1, "fault buffer too small");

  Params p{};
  p.q = reinterpret_cast<const __half*>(q.data_ptr());
  p.r = reinterpret_cast<const __half*>(r.data_ptr());
  p.q_scale = reinterpret_cast<const __half*>(q_scale.data_ptr());
  p.r_scale = reinterpret_cast<const __half*>(r_scale.data_ptr());
  p.tau = tau.has_value() ? tau->data_ptr<float>() : nullptr;
  p.rel_t = reinterpret_cast<const __half*>(rel_t.data_ptr());
  p.kc = key_cache.data_ptr();
  p.vc = value_cache.data_ptr();
  p.block_table = block_table.data_ptr<int>();
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cu_q = cu_q.data_ptr<int>();
  p.gate = reinterpret_cast<const __half*>(gate.data_ptr());
  p.sinks = reinterpret_cast<const __half*>(sinks.data_ptr());
  p.out = reinterpret_cast<__half*>(out.data_ptr());
  p.o_part = o_part.data_ptr<float>();
  p.ml_part = ml_part.data_ptr<float>();
  p.counters = counters.data_ptr<int>();
  p.fault = fault.data_ptr<int>();
  p.x[0] = reinterpret_cast<const __half*>(kx.data_ptr());
  p.x[1] = reinterpret_cast<const __half*>(vx.data_ptr());
  p.w[0] = reinterpret_cast<const __half*>(kw.data_ptr());
  p.w[1] = reinterpret_cast<const __half*>(vw.data_ptr());
  p.dyn[0] = dynamic ? reinterpret_cast<const __half*>(kdyn->data_ptr()) : nullptr;
  p.dyn[1] = dynamic ? reinterpret_cast<const __half*>(vdyn->data_ptr()) : nullptr;
  p.cache = reinterpret_cast<__half*>(cache.data_ptr());
  p.pos = pos.data_ptr<int64_t>();
  p.kv_slot = kv_slot.data_ptr<int64_t>();
  p.cslot = cslot.data_ptr<int64_t>();
  p.seq_idx = seq_idx.data_ptr<int>();
  p.qstart = qstart.data_ptr<int>();
  p.cbt = cbt.data_ptr<int>();

  p.stride_q = q.stride(0);
  p.stride_r = r.stride(0);
  p.stride_bt = block_table.stride(0);
  p.stride_gate = gate.stride(0);
  p.stride_sink = sinks.stride(0);
  p.kb = key_cache.stride(0);
  p.ks = key_cache.stride(1);
  p.kh = key_cache.stride(2);
  p.vb = value_cache.stride(0);
  p.vs = value_cache.stride(1);
  p.vh = value_cache.stride(2);
  p.stride_x[0] = kx.stride(0);
  p.stride_x[1] = vx.stride(0);
  p.stride_dyn[0] = dynamic ? kdyn->stride(0) : 0;
  p.stride_dyn[1] = dynamic ? vdyn->stride(0) : 0;
  p.stride_w0 = kw.stride(0);
  p.stride_c_blk = cache.stride(0);
  p.stride_c_h = cache.stride(1);
  p.stride_c_n = cache.stride(2);
  p.stride_cbt = cbt.stride(0);

  p.T = T;
  p.num_actual_tokens = act_toks;
  p.Hq = Hq;
  p.Hkv = Hkv;
  p.S = static_cast<int>(num_splits);
  p.page_shift = ps;
  p.cshift = cs;
  p.cmax = static_cast<int>(cbt.size(1));
  p.window = static_cast<int>(window);
  p.E = static_cast<int>(rel_t.size(0));
  p.maxrows = maxrows;
  p.scale_log2 = static_cast<float>(softmax_scale) * LOG2E_F;
  p.q_thr = static_cast<float>(q_thr);
  p.r_thr = static_cast<float>(r_thr);
  p.eps = static_cast<float>(eps);
  p.conv_scale = static_cast<float>(conv_scale);
  p.norm_eps = static_cast<float>(norm_eps);
  p.stop = static_cast<int>(stop);
  p.pf = static_cast<int>(prefetch);

  if (act_toks < T) {
    TORCH_CHECK(cudaMemsetAsync(
                    p.out + static_cast<size_t>(act_toks) * Hq * HD,
                    0,
                    static_cast<size_t>(T - act_toks) * Hq * HD * sizeof(__half),
                    stream) == cudaSuccess);
  }

  const int dev = static_cast<int>(guard.current_device().index());
  dim3 grid(static_cast<unsigned>(p.S), static_cast<unsigned>(p.Hkv), static_cast<unsigned>(num_reqs));
  if (is_fp8) {
    if (num_warps == 8 && max_splits == 40) launch_attn2<true, 8, 40>(grid, p, smem, dev, stream);
    else if (num_warps == 16) launch_attn2<true, 16, 40>(grid, p, smem, dev, stream);
    else launch_attn2<true, 8, 80>(grid, p, smem, dev, stream);
  } else {
    if (num_warps == 8 && max_splits == 40) launch_attn2<false, 8, 40>(grid, p, smem, dev, stream);
    else if (num_warps == 16) launch_attn2<false, 16, 40>(grid, p, smem, dev, stream);
    else launch_attn2<false, 8, 80>(grid, p, smem, dev, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_rel_dec2_forward", &attn_rel_dec2_forward,
        "Volta sm_70 short-SASS 1-launch fused kv_conv_prologue + decode_attention (FP16 & FP8 E4M3FN KV)");
  m.def("configure_dec2_smem", &configure_dec2_smem,
        "Configure max dynamic shared memory attribute for all attn_rel_dec2 kernel variants on current device");
}
