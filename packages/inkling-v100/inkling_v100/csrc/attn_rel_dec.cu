// SPDX-License-Identifier: Apache-2.0
// attn_rel_dec.cu - Zero-spill Volta (sm_70) HMMA.884 relative-bias GQA decode & speculative-verify
// attention kernel for Inkling-2-Mini (HD=128, general G<=8 optimized for G=3, FP8 E4M3FN & FP16
// paged KV, local SWA-1024 & global, sheared rel_logits, fused gate_sink, split-KV).
//
// Volta sm_70 HD=128 advantage:
//   One single warp owns ALL 128 head dims for both QK (mma.m8n8k4.row.col) and PV (mma.m8n8k4.row.row)
//   on a 32-token KV tile -> ZERO warp-pair barriers and ZERO score-exchange shared memory in the main loop.
//   A CTA consists of M_WARPS row-groups (each 8 query rows) x T_WARPS tile-warps (interleaved 32-token tiles).
//   When M_WARPS > 1 (speculative verify Q=3..8), the M_WARPS warps sharing tile-warp index tw lockstep via
//   bar.sync per tile so all M_WARPS warps read the exact same 8 KB KV tile together (100% L1 hit, 1x HBM!).
//
// Split-KV:
//   S == 1: single launch, fused gate_sink epilogue (zero workspace; cudaMemsetAsync for padded rows if needed).
//   S > 1:  2-kernel split (k_rel_attn) + combine (k_rel_combine, fused gate_sink + padded row zeroing)
//           using a persistent pre-allocated workspace -> 100% deadlock-free at any grid size,
//           100% CUDA-graph safe, bitwise deterministic.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int HD = 128;
constexpr int TILE = 32;
constexpr int QS = HD + 8;     // 136 halfs (272 B): conflict-free uint2 Q loads
constexpr int PS = TILE + 8;   // 40 halfs (80 B): conflict-free uint2 P loads
constexpr int TBS = 256;       // block-table entries cached in shared memory (16K @ page=64, 32K @ page=128)
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
      // Use compile-time constant barrier IDs so ptxas reserves only 3 barriers (not all 16!).
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
// Exact across all non-NaN E4M3FN codes (normals, subnormals, +-0).
__device__ __forceinline__ void e4m3x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned x0 = __byte_perm(w, 0u, 0x1404);
  const unsigned x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}

// Fast Volta SFU exp2f (MUFU.EX2, ~22-bit precision, ex2(-inf) == +0.0f).
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
    return __half22float2(*reinterpret_cast<const __half2*>(p));
  }
  return make_float2(__half2float(p[0]), __half2float(p[1]));
}

template <int M_WARPS, int T_WARPS>
struct SM {
  static constexpr int NT = 32 * M_WARPS * T_WARPS;
  static constexpr int R_CTA = 8 * M_WARPS;
  // Main loop layout:
  //   Ts [TBS] int, RelPtrs [R_CTA] const __half*, Qs [R_CTA][QS] half, Ps [M_WARPS*T_WARPS][8][PS] half
  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS * (int)sizeof(int);
  static constexpr int RP_OFF = TB_BYTES;
  static constexpr int RP_BYTES = ((R_CTA * (int)sizeof(const __half*) + 15) & ~15);
  static constexpr int Q_OFF = RP_OFF + RP_BYTES;
  static constexpr int Q_BYTES = R_CTA * QS * (int)sizeof(__half);
  static constexpr int P_OFF = (Q_OFF + Q_BYTES + 15) & ~15;
  static constexpr int P_BYTES = M_WARPS * T_WARPS * 8 * PS * (int)sizeof(__half);
  static constexpr int MAIN_END = (P_OFF + P_BYTES + 15) & ~15;

  // Merge layout (only used when T_WARPS > 1; aliases Ts, RelPtrs, Qs, Ps after main loop):
  //   Os [M_WARPS][T_WARPS][8][HD] float
  static constexpr int O_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * HD * (int)sizeof(float)) : 0;
  static constexpr int ML_OFF = (MAIN_END > O_BYTES ? MAIN_END : O_BYTES);
  static constexpr int ML_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int RS_OFF = ML_OFF + ML_BYTES;
  static constexpr int RS_BYTES = (T_WARPS > 1) ? (M_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int TOTAL = ((RS_OFF + RS_BYTES + 15) & ~15);
};

// Zero-fill helper for unused splits (s >= Sb or T_tiles == 0)
template <int M_WARPS, int T_WARPS>
__device__ __forceinline__ void write_empty_split(
    float2* __restrict__ ml_part, int num_splits, int s, int H_q, int G, int h,
    int q_start, int first_row, int eff_nr, int tid) {
  constexpr int NT = 32 * M_WARPS * T_WARPS;
  constexpr int R_CTA = 8 * M_WARPS;
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / G;
      const int head = h * G + (gr % G);
      const int token = q_start + tok;
      const size_t slot = ((size_t)token * H_q + head) * (size_t)num_splits + (size_t)s;
      __stcg(ml_part + slot, make_float2(-INFINITY, 0.f));
    }
  }
}

template <bool IS_FP8, int M_WARPS, int T_WARPS>
__global__ void __launch_bounds__(32 * M_WARPS * T_WARPS)
k_rel_attn(
    const __half* __restrict__ q,
    const void* __restrict__ kc_raw,
    const void* __restrict__ vc_raw,
    const __half* __restrict__ rel_logits,
    const __half* __restrict__ gate,
    const __half* __restrict__ sinks,
    __half* __restrict__ out,
    float* __restrict__ o_part,
    float2* __restrict__ ml_part,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cu_seqlens_q,
    int H_q,
    int H_kv,
    int G,
    int E,
    int E_pad,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
    int num_splits,
    int64_t stride_q_t,
    int64_t stride_q_h,
    int64_t kb,
    int64_t kt,
    int64_t kh,
    int64_t vb,
    int64_t vt,
    int64_t vh,
    int64_t stride_rel_t,
    int64_t stride_rel_h,
    int64_t stride_bt,
    int64_t stride_gate,
    int64_t stride_sink,
    float sl2,
    float vsc,
    bool rel_aligned4,
    bool gate_aligned4,
    bool q_aligned16) {
  using L_ = SM<M_WARPS, T_WARPS>;
  constexpr int NT = L_::NT;
  constexpr int R_CTA = L_::R_CTA;
  extern __shared__ __align__(16) uint8_t sm[];

  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  __half* Qs = reinterpret_cast<__half*>(sm + L_::Q_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);

  const int s = blockIdx.x;
  const int h = blockIdx.y % H_kv;
  const int m_tile = blockIdx.y / H_kv;
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
  const int eff_nr = eff_q_len * G;
  const int first_row = m_tile * R_CTA;
  if (first_row >= eff_nr) return;

  const int seqlen_k = cache_seqlens[b];
  const int past = seqlen_k - q_len;
  const int last_row = min(first_row + R_CTA, eff_nr) - 1;
  const int first_tok = first_row / G;
  const int last_tok = last_row / G;
  const int pos_lo = past + first_tok;
  const int pos_hi = past + last_tok;

  const int k_lo = (window > 0) ? max(pos_lo - (window - 1), 0) : 0;
  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int t_lo = k_lo / TILE;
  const int t_hi = (k_hi + TILE - 1) / TILE;
  const int T_tiles = max(t_hi - t_lo, 0);

  if (T_tiles <= 0) {
    if (num_splits == 1) {
      for (int i = tid; i < R_CTA * (HD / 2); i += NT) {
        const int r = i / (HD / 2);
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / G;
          const int head = h * G + (gr % G);
          const int token = q_start + tok;
          const int c2 = i % (HD / 2);
          reinterpret_cast<__half2*>(out + ((size_t)token * H_q + head) * HD)[c2] =
              __floats2half2_rn(0.f, 0.f);
        }
      }
    } else {
      write_empty_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, H_q, G, h, q_start, first_row, eff_nr, tid);
    }
    return;
  }

  const int Sb = max(1, min(num_splits, T_tiles));
  if (s >= Sb) {
    write_empty_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, H_q, G, h, q_start, first_row, eff_nr, tid);
    return;
  }

  const int t_start = t_lo + (int)((long long)T_tiles * s / Sb);
  const int t_end = t_lo + (int)((long long)T_tiles * (s + 1) / Sb);

  // Cache up to TBS block-table entries + RelPtrs + stage Q rows in shared memory.
  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  for (int i = tid; i < min(max_blocks, TBS); i += NT) {
    Ts[i] = tb[i];
  }
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / G;
      const int head = h * G + (gr % G);
      const int token = q_start + tok;
      RelPtrs[r] = rel_logits + (size_t)token * (size_t)stride_rel_t + (size_t)head * (size_t)stride_rel_h;
    } else {
      RelPtrs[r] = rel_logits;
    }
  }
  for (int i = tid; i < R_CTA * (HD / 8); i += NT) {
    const int r = i / (HD / 8);
    const int c = (i % (HD / 8)) * 8;
    const int gr = first_row + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (gr < eff_nr) {
      const int tok = gr / G;
      const int head = h * G + (gr % G);
      const int token = q_start + tok;
      const __half* qp_src = q + (size_t)token * (size_t)stride_q_t + (size_t)head * (size_t)stride_q_h + c;
      if (q_aligned16) {
        v = ldg16(qp_src);
      } else {
        __half tmp[8];
#pragma unroll
        for (int u = 0; u < 8; u++) tmp[u] = qp_src[u];
        v = *reinterpret_cast<const uint4*>(tmp);
      }
    }
    *reinterpret_cast<uint4*>(Qs + r * QS + c) = v;
  }
  __syncthreads();

  // Volta mma.m8n8k4 lane geometry (1 warp owns all 128 head dims).
  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;              // 0..7: A row; B column within quadpair for QK
  const int r0 = (lane & 1) + 4 * hi;        // 0, 1, 4, 5: C rows are r0 and r0 + 2
  const int c0 = lane & 2;                   // 0 or 2: C cols are c0, c0+1, c0+4, c0+5
  const int ktok = 8 * qp + arow;            // 0..31: this thread's token in a 32-token tile (QK)
  const int vdim = 32 * qp + 16 * hi;        // 0..112 step 16: this thread's 16 V dims (PV)

  // Compact per-thread row metadata (pos & rel_base only; 64-bit rel_row pointers live in shared RelPtrs).
  const int lrA = mw * 8 + r0;
  const int lrB = lrA + 2;
  const int grA = first_row + lrA;
  const int grB = first_row + lrB;
  const bool validA = grA < eff_nr;
  const bool validB = grB < eff_nr;

  int posA = -1, posB = -1;
  int rel_baseA = 0, rel_baseB = 0;
  if (validA) {
    const int tokA = grA / G;
    posA = past + tokA;
    const int tile_endA = min(((tokA / 128) + 1) * 128, q_len);
    rel_baseA = ((past + tile_endA + 127) / 128) * 128 - E_pad;
  }
  if (validB) {
    const int tokB = grB / G;
    posB = past + tokB;
    const int tile_endB = min(((tokB / 128) + 1) * 128, q_len);
    rel_baseB = ((past + tile_endB + 127) / 128) * 128 - E_pad;
  }

  const int max_ext_dist = (window > 0 && window < E) ? window : E;

  const __half* qrow = Qs + (mw * 8 + arow) * QS;
  __half* Pw = Ps + warp * 8 * PS;

  const int pmask = page - 1;
  const int64_t kh_off = (int64_t)h * kh;
  const int64_t vh_off = (int64_t)h * vh;

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TBS ? Ts[pi] : tb[pi]);
  };

  constexpr int KR_N = IS_FP8 ? 8 : 16;
  uint4 kr[KR_N];
  uint4 vr[8];

  auto ldk = [&](int t) {
    const int tok0 = t * TILE;
    if (page_shift >= 4 && tok0 + 31 < seqlen_k) {
      const int pi0 = tok0 >> page_shift;
      const int s0 = tok0 & pmask;
      const int pi1 = (tok0 + 16) >> page_shift;
      const int s1 = (tok0 + 16) & pmask;
      const int64_t pg = (ktok < 16) ? get_pg(pi0) : get_pg(pi1);
      const int sl = (ktok < 16) ? (s0 + ktok) : (s1 + ktok - 16);
      if constexpr (IS_FP8) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
        for (int i = 0; i < 8; i++) kr[i] = ldg16(p + 16 * i);
      } else {
        const __half* p = reinterpret_cast<const __half*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
        for (int i = 0; i < 16; i++) kr[i] = ldg16(p + 8 * i);
      }
    } else {
      const int k_idx = tok0 + ktok;
      if (k_idx < seqlen_k) {
        const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
        const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
        const int64_t pg = get_pg(pi);
        if constexpr (IS_FP8) {
          const uint8_t* p = reinterpret_cast<const uint8_t*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
          for (int i = 0; i < 8; i++) kr[i] = ldg16(p + 16 * i);
        } else {
          const __half* p = reinterpret_cast<const __half*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
          for (int i = 0; i < 16; i++) kr[i] = ldg16(p + 8 * i);
        }
      } else {
#pragma unroll
        for (int i = 0; i < KR_N; i++) kr[i] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  auto ldv_fp8 = [&](int t) {
    const int tok0 = t * TILE;
    const uint8_t* vc = reinterpret_cast<const uint8_t*>(vc_raw);
    if (page_shift >= 4 && tok0 + 31 < seqlen_k) {
      const int pi0 = tok0 >> page_shift;
      const int s0 = tok0 & pmask;
      const int pi1 = (tok0 + 16) >> page_shift;
      const int s1 = (tok0 + 16) & pmask;
      const int64_t pg0 = get_pg(pi0);
      const int64_t pg1 = (pi1 == pi0) ? pg0 : get_pg(pi1);
      const uint8_t* p0 = vc + pg0 * vb + (int64_t)(s0 + l3) * vt + vh_off + vdim;
      const uint8_t* p1 = vc + pg1 * vb + (int64_t)(s1 + l3) * vt + vh_off + vdim;
#pragma unroll
      for (int i = 0; i < 4; i++) vr[i] = ldg16(p0 + (int64_t)(4 * i) * vt);
#pragma unroll
      for (int i = 0; i < 4; i++) vr[4 + i] = ldg16(p1 + (int64_t)(4 * i) * vt);
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int v_idx = tok0 + l3 + 4 * i;
        if (v_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (v_idx >> page_shift) : (v_idx / page);
          const int sl = (page_shift >= 0) ? (v_idx & pmask) : (v_idx - pi * page);
          const int64_t pg = get_pg(pi);
          vr[i] = ldg16(vc + pg * vb + (int64_t)sl * vt + vh_off + vdim);
        } else {
          vr[i] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  auto ldv_fp16_half = [&](int t, int half_idx) {
    const int tok0 = t * TILE + half_idx * 16;
    const __half* vc = reinterpret_cast<const __half*>(vc_raw);
    if (page_shift >= 4 && tok0 + 15 < seqlen_k) {
      const int pi = tok0 >> page_shift;
      const int sl = tok0 & pmask;
      const int64_t pg = get_pg(pi);
      const __half* p = vc + pg * vb + (int64_t)(sl + l3) * vt + vh_off + vdim;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const __half* qptr = p + (int64_t)(4 * i) * vt;
        vr[2 * i] = ldg16(qptr);
        vr[2 * i + 1] = ldg16(qptr + 8);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int v_idx = tok0 + l3 + 4 * i;
        if (v_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (v_idx >> page_shift) : (v_idx / page);
          const int sl = (page_shift >= 0) ? (v_idx & pmask) : (v_idx - pi * page);
          const int64_t pg = get_pg(pi);
          const __half* qptr = vc + pg * vb + (int64_t)sl * vt + vh_off + vdim;
          vr[2 * i] = ldg16(qptr);
          vr[2 * i + 1] = ldg16(qptr + 8);
        } else {
          vr[2 * i] = make_uint4(0u, 0u, 0u, 0u);
          vr[2 * i + 1] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  // Preload first tile's K and V into registers.
  int t = t_start + tw;
  if (t < t_end) {
    ldk(t);
    if constexpr (IS_FP8) {
      ldv_fp8(t);
    } else {
      ldv_fp16_half(t, 0);
    }
  }

  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[j][e] = 0.f;
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;

  // Main loop over 32-token KV tiles.
  for (; t < t_end; t += T_WARPS) {
    const int tok0 = t * TILE;
    const bool more = (t + T_WARPS) < t_end;

    // ---- QK: 32 x HMMA.884 covering all 128 head dims ----
    float sa[8];
#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] = 0.f;

    if constexpr (IS_FP8) {
      float sb[8];
#pragma unroll
      for (int e = 0; e < 8; e++) sb[e] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const unsigned w[4] = {kr[i].x, kr[i].y, kr[i].z, kr[i].w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint2 a = *reinterpret_cast<const uint2*>(qrow + 4 * (4 * i + j));
          unsigned b0, b1;
          e4m3x4(w[j], b0, b1);
          mma_rc((i & 1) ? sb : sa, a.x, a.y, b0, b1);
        }
      }
      if (more) ldk(t + T_WARPS);
#pragma unroll
      for (int e = 0; e < 8; e++) sa[e] += sb[e];
    } else {
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const uint2 a0 = *reinterpret_cast<const uint2*>(qrow + 8 * i);
        const uint2 a1 = *reinterpret_cast<const uint2*>(qrow + 8 * i + 4);
        mma_rc(sa, a0.x, a0.y, kr[i].x, kr[i].y);
        mma_rc(sa, a1.x, a1.y, kr[i].z, kr[i].w);
      }
      if (more) ldk(t + T_WARPS);
    }

    // ---- Scale, sheared relative bias, and causal + SWA masking (in-place in sa) ----
    const int cb = tok0 + 8 * qp + c0;

    // Row A: entries 0, 1 (cb, cb+1) and 4, 5 (cb+4, cb+5)
    {
      const int k_min_ext = posA - max_ext_dist + 1;
      const int k_min_vis = (window > 0) ? (posA - window + 1) : 0;
      if (!validA) {
        sa[0] = -INFINITY; sa[1] = -INFINITY; sa[4] = -INFINITY; sa[5] = -INFINITY;
      } else if (tok0 + 31 < k_min_ext && tok0 >= k_min_vis && tok0 + 31 <= posA && tok0 + 31 < seqlen_k) {
        sa[0] *= sl2; sa[1] *= sl2; sa[4] *= sl2; sa[5] *= sl2;
      } else if (tok0 >= k_min_ext && tok0 + 31 <= posA && tok0 + 31 < seqlen_k) {
        const __half* rr = RelPtrs[lrA];
        const float2 b01 = ld_h2_as_f2(rr + (cb - rel_baseA), rel_aligned4);
        const float2 b45 = ld_h2_as_f2(rr + (cb + 4 - rel_baseA), rel_aligned4);
        sa[0] = fmaf(b01.x, LOG2E, sa[0] * sl2);
        sa[1] = fmaf(b01.y, LOG2E, sa[1] * sl2);
        sa[4] = fmaf(b45.x, LOG2E, sa[4] * sl2);
        sa[5] = fmaf(b45.y, LOG2E, sa[5] * sl2);
      } else {
        const __half* rr = RelPtrs[lrA];
        const int idxA[4] = {0, 1, 4, 5};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int e = idxA[u];
          const int col = cb + (e & 1) + (e & 4);
          const int dist = posA - col;
          if (col < seqlen_k && dist >= 0 && (window == 0 || dist < window)) {
            float s_val = sa[e] * sl2;
            if (dist < E) s_val = fmaf(__half2float(rr[col - rel_baseA]), LOG2E, s_val);
            sa[e] = s_val;
          } else {
            sa[e] = -INFINITY;
          }
        }
      }
    }

    // Row B: entries 2, 3 (cb, cb+1) and 6, 7 (cb+4, cb+5)
    {
      const int k_min_ext = posB - max_ext_dist + 1;
      const int k_min_vis = (window > 0) ? (posB - window + 1) : 0;
      if (!validB) {
        sa[2] = -INFINITY; sa[3] = -INFINITY; sa[6] = -INFINITY; sa[7] = -INFINITY;
      } else if (tok0 + 31 < k_min_ext && tok0 >= k_min_vis && tok0 + 31 <= posB && tok0 + 31 < seqlen_k) {
        sa[2] *= sl2; sa[3] *= sl2; sa[6] *= sl2; sa[7] *= sl2;
      } else if (tok0 >= k_min_ext && tok0 + 31 <= posB && tok0 + 31 < seqlen_k) {
        const __half* rr = RelPtrs[lrB];
        const float2 b01 = ld_h2_as_f2(rr + (cb - rel_baseB), rel_aligned4);
        const float2 b45 = ld_h2_as_f2(rr + (cb + 4 - rel_baseB), rel_aligned4);
        sa[2] = fmaf(b01.x, LOG2E, sa[2] * sl2);
        sa[3] = fmaf(b01.y, LOG2E, sa[3] * sl2);
        sa[6] = fmaf(b45.x, LOG2E, sa[6] * sl2);
        sa[7] = fmaf(b45.y, LOG2E, sa[7] * sl2);
      } else {
        const __half* rr = RelPtrs[lrB];
        const int idxB[4] = {2, 3, 6, 7};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int e = idxB[u];
          const int col = cb + (e & 1) + (e & 4);
          const int dist = posB - col;
          if (col < seqlen_k && dist >= 0 && (window == 0 || dist < window)) {
            float s_val = sa[e] * sl2;
            if (dist < E) s_val = fmaf(__half2float(rr[col - rel_baseB]), LOG2E, s_val);
            sa[e] = s_val;
          } else {
            sa[e] = -INFINITY;
          }
        }
      }
    }

    // ---- Online Softmax (FP32, in-place in sa) ----
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
    for (int e = 0; e < 8; e++) {
      sa[e] = ex2_fast(sa[e] - ((e & 2) ? z2 : z0));
    }
    l0 = l0 * a0 + (sa[0] + sa[1]) + (sa[4] + sa[5]);
    l2 = l2 * a2 + (sa[2] + sa[3]) + (sa[6] + sa[7]);
    m0 = n0;
    m2 = n2;

#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;

    // Stage FP16 P into this warp's private shared buffer (A-fragment layout).
    const int pc = 8 * qp + c0;
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(sa[0], sa[1]);
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(sa[4], sa[5]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(sa[2], sa[3]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(sa[6], sa[7]);
    __syncwarp();

    // ---- PV: 32 x HMMA.884 accumulating all 128 head dims ----
    if constexpr (IS_FP8) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
        const unsigned w[4] = {vr[i].x, vr[i].y, vr[i].z, vr[i].w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          unsigned b0, b1;
          e4m3x4(w[j], b0, b1);
          mma_rr(o[j], a.x, a.y, b0, b1);
        }
      }
      if (more) ldv_fp8(t + T_WARPS);
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
        mma_rr(o[0], a.x, a.y, vr[2 * i].x, vr[2 * i].y);
        mma_rr(o[1], a.x, a.y, vr[2 * i].z, vr[2 * i].w);
        mma_rr(o[2], a.x, a.y, vr[2 * i + 1].x, vr[2 * i + 1].y);
        mma_rr(o[3], a.x, a.y, vr[2 * i + 1].z, vr[2 * i + 1].w);
      }
      ldv_fp16_half(t, 1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * (4 + i));
        mma_rr(o[0], a.x, a.y, vr[2 * i].x, vr[2 * i].y);
        mma_rr(o[1], a.x, a.y, vr[2 * i].z, vr[2 * i].w);
        mma_rr(o[2], a.x, a.y, vr[2 * i + 1].x, vr[2 * i + 1].y);
        mma_rr(o[3], a.x, a.y, vr[2 * i + 1].z, vr[2 * i + 1].w);
      }
      if (more) ldv_fp16_half(t + T_WARPS, 0);
    }

    // Lockstep M_WARPS warps sharing tile-warp index tw so they hit the same 8 KB KV tile in L1!
    bar_tw_sync<M_WARPS, T_WARPS>(tw);
  }

  // Finalize per-warp row sum across the 8 lanes that share a row.
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
        const int tokA = grA / G;
        const int headA = h * G + (grA % G);
        const int tokenA = q_start + tokA;
        const float lse_e = (m0 + log2f(l0)) * LN2;
        const float sink_v = __half2float(sinks[(size_t)tokenA * (size_t)stride_sink + headA]);
        rs0 = (vsc / l0) * sigmoid_f32(lse_e - sink_v);
      }
      if (validB && l2 > 0.f) {
        const int tokB = grB / G;
        const int headB = h * G + (grB % G);
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
          const int tok = gr / G;
          const int head = h * G + (gr % G);
          const int token = q_start + tok;
          const __half* gp = gate + (size_t)token * (size_t)stride_gate + (size_t)head * HD;
          __half* ob = out + ((size_t)token * H_q + head) * HD;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            const float2 g_lo = ld_h2_as_f2(gp + dlo, gate_aligned4);
            const float2 g_hi = ld_h2_as_f2(gp + dlo + 16, gate_aligned4);
            const float y0 = o[j][2 * sr] * rsc * sigmoid_f32(g_lo.x);
            const float y1 = o[j][2 * sr + 1] * rsc * sigmoid_f32(g_lo.y);
            const float y2 = o[j][4 + 2 * sr] * rsc * sigmoid_f32(g_hi.x);
            const float y3 = o[j][5 + 2 * sr] * rsc * sigmoid_f32(g_hi.y);
            *reinterpret_cast<__half2*>(ob + dlo) = __floats2half2_rn(y0, y1);
            *reinterpret_cast<__half2*>(ob + dlo + 16) = __floats2half2_rn(y2, y3);
          }
        }
      }
    } else {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int gr = sr ? grB : grA;
        if (gr < eff_nr) {
          const int tok = gr / G;
          const int head = h * G + (gr % G);
          const int token = q_start + tok;
          const size_t slot = ((size_t)token * H_q + head) * (size_t)num_splits + (size_t)s;
          float* pob = o_part + slot * HD;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            __stcg(reinterpret_cast<float2*>(pob + dlo), make_float2(o[j][2 * sr], o[j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + dlo + 16), make_float2(o[j][4 + 2 * sr], o[j][5 + 2 * sr]));
          }
          if ((lane & 14) == 0) {
            __stcg(ml_part + slot, make_float2(sr ? m2 : m0, sr ? l2 : l0));
          }
        }
      }
    }
  } else {
    // T_WARPS > 1: merge the T_WARPS tile-warps per row-group in shared memory.
    float* Os = reinterpret_cast<float*>(sm);                // [M_WARPS][T_WARPS][8][HD] (aliases offset 0)
    float* MLs = reinterpret_cast<float*>(sm + L_::ML_OFF);  // [M_WARPS][T_WARPS][8][2]
    float* RSs = reinterpret_cast<float*>(sm + L_::RS_OFF);  // [M_WARPS][8][2]

    __syncthreads();  // Ts, RelPtrs, Qs, Ps are dead (aliased by Os)
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
    float* Op = Os + (size_t)(mw * T_WARPS + tw) * 8 * HD;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int local_r = r0 + 2 * sr;
        const float f = sr ? f2 : f0;
        const int dlo = 32 * qp + 4 * j + c0;
        *reinterpret_cast<float2*>(Op + local_r * HD + dlo) =
            make_float2(o[j][2 * sr] * f, o[j][2 * sr + 1] * f);
        *reinterpret_cast<float2*>(Op + local_r * HD + dlo + 16) =
            make_float2(o[j][4 + 2 * sr] * f, o[j][5 + 2 * sr] * f);
      }
    }

    for (int r = tid; r < R_CTA; r += NT) {
      const int rm = r >> 3;
      const int lr = r & 7;
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) {
        M = fmaxf(M, MLs[(rm * T_WARPS + k) * 16 + 2 * lr]);
      }
      float Lt = 0.f;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) {
        const float mk = MLs[(rm * T_WARPS + k) * 16 + 2 * lr];
        if (mk != -INFINITY) {
          Lt += MLs[(rm * T_WARPS + k) * 16 + 2 * lr + 1] * ex2_fast(mk - M);
        }
      }
      RSs[2 * r] = M;
      RSs[2 * r + 1] = Lt;
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
        const float4 c = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS + k) * 8 * HD + lr * HD + d);
        acc.x += c.x; acc.y += c.y; acc.z += c.z; acc.w += c.w;
      }
      const int tok = gr / G;
      const int head = h * G + (gr % G);
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
        const __half* gp = gate + (size_t)token * (size_t)stride_gate + (size_t)head * HD + d;
        const float2 g01 = ld_h2_as_f2(gp, gate_aligned4);
        const float2 g23 = ld_h2_as_f2(gp + 2, gate_aligned4);
        __half2* dst = reinterpret_cast<__half2*>(out + ((size_t)token * H_q + head) * HD + d);
        dst[0] = __floats2half2_rn(acc.x * rsc * sigmoid_f32(g01.x), acc.y * rsc * sigmoid_f32(g01.y));
        dst[1] = __floats2half2_rn(acc.z * rsc * sigmoid_f32(g23.x), acc.w * rsc * sigmoid_f32(g23.y));
      } else {
        const size_t slot = ((size_t)token * H_q + head) * (size_t)num_splits + (size_t)s;
        __stcg(reinterpret_cast<float4*>(o_part + slot * HD + d), acc);
      }
    }
    if (num_splits > 1) {
      for (int r = tid; r < R_CTA; r += NT) {
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / G;
          const int head = h * G + (gr % G);
          const int token = q_start + tok;
          const size_t slot = ((size_t)token * H_q + head) * (size_t)num_splits + (size_t)s;
          __stcg(ml_part + slot, make_float2(RSs[2 * r], RSs[2 * r + 1]));
        }
      }
    }
  }
}

// Split-KV combine + fused gate_sink + CUDA-graph padding zeroing.
// Grid: (num_tokens, H_q), Block: 32 threads (1 warp per (token, head), 4 dims per lane).
__global__ void __launch_bounds__(32)
k_rel_combine(
    const float* __restrict__ o_part,
    const float2* __restrict__ ml_part,
    const __half* __restrict__ gate,
    const __half* __restrict__ sinks,
    __half* __restrict__ out,
    int num_actual_tokens,
    int num_splits,
    int64_t stride_gate,
    int64_t stride_sink,
    float vsc,
    bool gate_aligned4) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int H_q = gridDim.y;
  const int lane = threadIdx.x;
  __half* ob = out + ((size_t)token * H_q + head) * HD;

  if (token >= num_actual_tokens) {
    *reinterpret_cast<uint2*>(ob + lane * 4) = make_uint2(0u, 0u);
    return;
  }

  const size_t base = ((size_t)token * H_q + head) * (size_t)num_splits;
  const float2* mlp = ml_part + base;
  const float* op = o_part + base * HD;

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
      const float4 v0 = (w0 > 0.f)
                            ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 0) * HD + lane * 4))
                            : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 v1 = (w1 > 0.f)
                            ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 1) * HD + lane * 4))
                            : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 v2 = (w2 > 0.f)
                            ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 2) * HD + lane * 4))
                            : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 v3 = (w3 > 0.f)
                            ? __ldcg(reinterpret_cast<const float4*>(op + (size_t)(s + 3) * HD + lane * 4))
                            : make_float4(0.f, 0.f, 0.f, 0.f);
      acc.x = fmaf(w0, v0.x, fmaf(w1, v1.x, fmaf(w2, v2.x, fmaf(w3, v3.x, acc.x))));
      acc.y = fmaf(w0, v0.y, fmaf(w1, v1.y, fmaf(w2, v2.y, fmaf(w3, v3.y, acc.y))));
      acc.z = fmaf(w0, v0.z, fmaf(w1, v1.z, fmaf(w2, v2.z, fmaf(w3, v3.z, acc.z))));
      acc.w = fmaf(w0, v0.w, fmaf(w1, v1.w, fmaf(w2, v2.w, fmaf(w3, v3.w, acc.w))));
    }
    for (; s < num_splits; s++) {
      const float ws = __shfl_sync(0xffffffffu, w_lane, s);
      if (ws > 0.f) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (size_t)s * HD + lane * 4));
        acc.x = fmaf(ws, v.x, acc.x);
        acc.y = fmaf(ws, v.y, acc.y);
        acc.z = fmaf(ws, v.z, acc.z);
        acc.w = fmaf(ws, v.w, acc.w);
      }
    }
  } else {
    for (int s = lane; s < num_splits; s += 32) {
      M = fmaxf(M, __ldcg(mlp + s).x);
    }
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
          const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (size_t)(sb + u) * HD + lane * 4));
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

  const __half* gp = gate + (size_t)token * (size_t)stride_gate + (size_t)head * HD + lane * 4;
  const float2 g01 = ld_h2_as_f2(gp, gate_aligned4);
  const float2 g23 = ld_h2_as_f2(gp + 2, gate_aligned4);
  __half2* dst = reinterpret_cast<__half2*>(ob + lane * 4);
  dst[0] = __floats2half2_rn(acc.x * row_scale * sigmoid_f32(g01.x), acc.y * row_scale * sigmoid_f32(g01.y));
  dst[1] = __floats2half2_rn(acc.z * row_scale * sigmoid_f32(g23.x), acc.w * row_scale * sigmoid_f32(g23.y));
}

template <bool IS_FP8, int M_WARPS, int T_WARPS>
void dispatch_attn(
    const __half* q_ptr,
    const void* kc_ptr,
    const void* vc_ptr,
    const __half* rel_ptr,
    const __half* gate_ptr,
    const __half* sinks_ptr,
    __half* out_ptr,
    float* o_part_ptr,
    float2* ml_part_ptr,
    const int* bt_ptr,
    const int* seq_ptr,
    const int* cu_q_ptr,
    int B,
    int H_q,
    int H_kv,
    int G,
    int E,
    int E_pad,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int max_nr,
    int num_actual_tokens,
    int num_splits,
    int64_t stride_q_t,
    int64_t stride_q_h,
    int64_t kb,
    int64_t kt,
    int64_t kh,
    int64_t vb,
    int64_t vt,
    int64_t vh,
    int64_t stride_rel_t,
    int64_t stride_rel_h,
    int64_t stride_bt,
    int64_t stride_gate,
    int64_t stride_sink,
    float sl2,
    float vsc,
    bool rel_aligned4,
    bool gate_aligned4,
    bool q_aligned16,
    cudaStream_t stream) {
  constexpr int R_CTA = 8 * M_WARPS;
  const int num_m_tiles = (max_nr + R_CTA - 1) / R_CTA;
  dim3 grid(num_splits, H_kv * num_m_tiles, B);
  constexpr int smem = SM<M_WARPS, T_WARPS>::TOTAL;
  k_rel_attn<IS_FP8, M_WARPS, T_WARPS><<<grid, 32 * M_WARPS * T_WARPS, smem, stream>>>(
      q_ptr, kc_ptr, vc_ptr, rel_ptr, gate_ptr, sinks_ptr, out_ptr, o_part_ptr, ml_part_ptr,
      bt_ptr, seq_ptr, cu_q_ptr, H_q, H_kv, G, E, E_pad, window, page, page_shift, max_blocks,
      num_actual_tokens, num_splits, stride_q_t, stride_q_h, kb, kt, kh, vb, vt, vh,
      stride_rel_t, stride_rel_h, stride_bt, stride_gate, stride_sink,
      sl2, vsc, rel_aligned4, gate_aligned4, q_aligned16);
}

}  // namespace

void attn_rel_dec_forward(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor rel_logits,
    torch::Tensor gate,
    torch::Tensor sinks,
    torch::Tensor out,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    torch::Tensor cu_seqlens_q,
    torch::Tensor o_part,
    torch::Tensor ml_part,
    int64_t max_query_len,
    int64_t num_actual_tokens,
    double softmax_scale,
    int64_t window,
    int64_t num_splits,
    int64_t t_warps_override) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kHalf && q.dim() == 3 && q.size(2) == HD && q.stride(2) == 1,
              "q must be CUDA FP16 [T, H_q, 128] with stride(2)==1");
  const int num_tokens = (int)q.size(0);
  if (num_tokens == 0) return;

  const int H_q = (int)q.size(1);
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(),
              "key_cache and value_cache must have identical 4D shapes [num_blocks, page, H_kv, 128]");
  TORCH_CHECK(key_cache.size(3) == HD && key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "KV cache must have head_dim=128 and stride(3)==1");
  const bool is_fp8 = key_cache.element_size() == 1;
  TORCH_CHECK(is_fp8 || key_cache.dtype() == torch::kHalf,
              "KV cache must be uint8/float8_e4m3fn or float16");
  TORCH_CHECK(key_cache.element_size() == value_cache.element_size(),
              "key_cache and value_cache element sizes must match");

  const int page = (int)key_cache.size(1);
  const int H_kv = (int)key_cache.size(2);
  TORCH_CHECK(H_kv > 0 && H_q % H_kv == 0, "H_q must be a multiple of H_kv");
  const int G = H_q / H_kv;
  TORCH_CHECK(G >= 1 && G <= 8, "G = H_q / H_kv must be in 1..8");

  TORCH_CHECK(rel_logits.dtype() == torch::kHalf && rel_logits.dim() == 3 && rel_logits.stride(2) == 1,
              "rel_logits must be FP16 [T+128, H_q, E+256] with stride(2)==1");
  const int E_pad = (int)rel_logits.size(2);
  const int E = E_pad - 256;
  TORCH_CHECK(E >= 0, "rel_logits.size(2) must be >= 256");

  TORCH_CHECK(gate.dtype() == torch::kHalf && gate.dim() == 2 && gate.size(1) == H_q * HD && gate.stride(1) == 1,
              "gate must be FP16 [T, H_q*128] with stride(1)==1");
  TORCH_CHECK(sinks.dtype() == torch::kHalf && sinks.dim() == 2 && sinks.size(1) == H_q && sinks.stride(1) == 1,
              "sinks must be FP16 [T, H_q] with stride(1)==1");
  TORCH_CHECK(out.dtype() == torch::kHalf && out.is_contiguous() && out.dim() == 2 &&
                  out.size(0) == num_tokens && out.size(1) == H_q * HD,
              "out must be contiguous FP16 [T, H_q*128]");

  const int B = (int)cache_seqlens.size(0);
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && block_table.dim() == 2 &&
                  block_table.size(0) == B && block_table.stride(1) == 1,
              "block_table must be int32 [B, max_blocks] with stride(1)==1");
  TORCH_CHECK(cache_seqlens.dtype() == torch::kInt32 && cache_seqlens.is_contiguous() && cache_seqlens.dim() == 1,
              "cache_seqlens must be contiguous int32 [B]");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_q.is_contiguous() &&
                  cu_seqlens_q.dim() == 1 && cu_seqlens_q.size(0) == B + 1,
              "cu_seqlens_q must be contiguous int32 [B+1]");
  TORCH_CHECK(num_splits >= 1, "num_splits must be >= 1");

  const int S = (int)num_splits;
  const int act_toks = (int)std::max<int64_t>(0, std::min<int64_t>(num_actual_tokens, num_tokens));

  const at::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (act_toks == 0 || B == 0) {
    TORCH_CHECK(cudaMemsetAsync(out.data_ptr(), 0, (size_t)num_tokens * H_q * HD * sizeof(__half), stream) ==
                cudaSuccess);
    return;
  }

  float* o_part_ptr = nullptr;
  float2* ml_part_ptr = nullptr;
  if (S > 1) {
    const int64_t slots = (int64_t)num_tokens * H_q * S;
    TORCH_CHECK(o_part.is_cuda() && o_part.dtype() == torch::kFloat32 && o_part.is_contiguous() &&
                    o_part.numel() >= slots * HD,
                "o_part workspace too small");
    TORCH_CHECK(ml_part.is_cuda() && ml_part.dtype() == torch::kFloat32 && ml_part.is_contiguous() &&
                    ml_part.numel() >= slots * 2,
                "ml_part workspace too small");
    o_part_ptr = o_part.data_ptr<float>();
    ml_part_ptr = reinterpret_cast<float2*>(ml_part.data_ptr<float>());
  }

  int page_shift = -1;
  if (page > 0 && (page & (page - 1)) == 0) {
    page_shift = 0;
    int tmp = page;
    while (tmp > 1) { tmp >>= 1; page_shift++; }
  }

  const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  const void* kc_ptr = key_cache.data_ptr();
  const void* vc_ptr = value_cache.data_ptr();
  const __half* rel_ptr = reinterpret_cast<const __half*>(rel_logits.data_ptr<at::Half>());
  const __half* gate_ptr = reinterpret_cast<const __half*>(gate.data_ptr<at::Half>());
  const __half* sinks_ptr = reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>());
  __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

  const int64_t stride_q_t = q.stride(0), stride_q_h = q.stride(1);
  const int64_t kb = key_cache.stride(0), kt = key_cache.stride(1), kh = key_cache.stride(2);
  const int64_t vb = value_cache.stride(0), vt = value_cache.stride(1), vh = value_cache.stride(2);
  const int64_t stride_rel_t = rel_logits.stride(0), stride_rel_h = rel_logits.stride(1);
  const int64_t stride_bt = block_table.stride(0);
  const int64_t stride_gate = gate.stride(0);
  const int64_t stride_sink = sinks.stride(0);

  const bool rel_aligned4 = (((uintptr_t)rel_ptr & 3u) == 0u) && ((stride_rel_t & 1) == 0) &&
                            ((stride_rel_h & 1) == 0) && ((E_pad & 1) == 0);
  const bool gate_aligned4 = (((uintptr_t)gate_ptr & 3u) == 0u) && ((stride_gate & 1) == 0);
  const bool q_aligned16 = (((uintptr_t)q_ptr & 15u) == 0u) && ((stride_q_t & 7) == 0) && ((stride_q_h & 7) == 0);

  const float sl2 = (float)(softmax_scale * (is_fp8 ? 256.0 : 1.0) * (double)LOG2E);
  const float vsc = is_fp8 ? 256.0f : 1.0f;

  const int max_nr = std::max(1, (int)max_query_len * G);
  const int max_blocks = (int)block_table.size(1);

  // Select (M_WARPS, T_WARPS):
  // - NR <= 8  (Q=1,2 @ G=3): M_WARPS=1, T_WARPS=4 (1 warp per Volta scheduler; best across all B, S, ctx)
  // - NR <= 16 (Q=3,4 @ G=3): M_WARPS=2, T_WARPS=2
  // - NR > 16  (Q=5..8 @ G=3): M_WARPS=3, T_WARPS=2 (2 CTAs/SM with 3 barriers; T_WARPS=1 if >=256 CTAs)
  const int total_ctas = B * H_kv * S;
  int mw_sel = (max_nr <= 8) ? 1 : (max_nr <= 16 ? 2 : 3);
  int tw_sel;
  if (mw_sel == 1) {
    tw_sel = (t_warps_override == 2) ? 2 : 4;
  } else if (mw_sel == 2) {
    tw_sel = 2;
  } else {
    if (t_warps_override == 1 || t_warps_override == 2) {
      tw_sel = (int)t_warps_override;
    } else {
      tw_sel = (total_ctas >= 256) ? 1 : 2;
    }
  }

#define LAUNCH_CASE(FP8_B, MW_I, TW_I)                                                                       \
  dispatch_attn<FP8_B, MW_I, TW_I>(                                                                          \
      q_ptr, kc_ptr, vc_ptr, rel_ptr, gate_ptr, sinks_ptr, out_ptr, o_part_ptr, ml_part_ptr,                 \
      block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(),              \
      B, H_q, H_kv, G, E, E_pad, (int)window, page, page_shift, max_blocks, max_nr, act_toks, S,             \
      stride_q_t, stride_q_h, kb, kt, kh, vb, vt, vh, stride_rel_t, stride_rel_h, stride_bt,                 \
      stride_gate, stride_sink, sl2, vsc, rel_aligned4, gate_aligned4, q_aligned16, stream)

  if (is_fp8) {
    if (mw_sel == 1 && tw_sel == 4) { LAUNCH_CASE(true, 1, 4); }
    else if (mw_sel == 1 && tw_sel == 2) { LAUNCH_CASE(true, 1, 2); }
    else if (mw_sel == 2) { LAUNCH_CASE(true, 2, 2); }
    else if (tw_sel == 1) { LAUNCH_CASE(true, 3, 1); }
    else { LAUNCH_CASE(true, 3, 2); }
  } else {
    if (mw_sel == 1 && tw_sel == 4) { LAUNCH_CASE(false, 1, 4); }
    else if (mw_sel == 1 && tw_sel == 2) { LAUNCH_CASE(false, 1, 2); }
    else if (mw_sel == 2) { LAUNCH_CASE(false, 2, 2); }
    else if (tw_sel == 1) { LAUNCH_CASE(false, 3, 1); }
    else { LAUNCH_CASE(false, 3, 2); }
  }
#undef LAUNCH_CASE

  if (S == 1) {
    if (act_toks < num_tokens) {
      TORCH_CHECK(cudaMemsetAsync(
                      out_ptr + (size_t)act_toks * H_q * HD, 0,
                      (size_t)(num_tokens - act_toks) * H_q * HD * sizeof(__half), stream) == cudaSuccess);
    }
  } else {
    dim3 cgrid(num_tokens, H_q);
    k_rel_combine<<<cgrid, 32, 0, stream>>>(
        o_part_ptr, ml_part_ptr, gate_ptr, sinks_ptr, out_ptr, act_toks, S,
        stride_gate, stride_sink, vsc, gate_aligned4);
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_rel_dec_forward", &attn_rel_dec_forward,
        "Volta sm_70 HMMA.884 relative-bias GQA decode & verify attention (FP8/FP16 KV, fused gate_sink)");
}
