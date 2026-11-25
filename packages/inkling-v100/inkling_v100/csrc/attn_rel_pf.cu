// SPDX-License-Identifier: Apache-2.0
// attn_rel_pf.cu - Zero-spill Volta (sm_70) HMMA.884 FlashAttention prefill kernel for
// Inkling-2-Mini (HD=128, general H % H_kv == 0, FP8 E4M3FN & FP16 paged KV, local SWA-1024
// & global, sheared rel_logits with far-past skip, fused gate_sink, variable-length batches,
// chunked prefill, CUDA-graph safe).
//
// Volta sm_70 architecture:
//   - 1 CTA per (head, q_tile, req) with grid (H_q, num_m_tiles, B) and longest-first reversed
//     m_tile scheduling -> G heads sharing a KV head run concurrently (100% L2 hit on K/V) and
//     causal tail-wave imbalance is minimized.
//   - Each warp owns 8 Q rows and ALL 128 head dims of O in 32 FP32 registers (o[4][8]),
//     eliminating inter-warp reductions and keeping peak registers well below 128/thread (0 spills).
//   - Single 16 KB KVs shared-memory buffer with XOR bank-swizzle (0% bank conflicts for FP8/FP16
//     writes, QK mma.row.col reads, and PV mma.row.row reads; 0 padding bytes).
//   - Software-pipelined global loads: V[t] is prefetched into registers before QK HMMAs and
//     K[t+1] is prefetched before PV HMMAs, hiding global/L2 latency behind Tensor Core compute.
//   - Dynamic SMEM <= 44 KB with 96 KB Volta SMEM carveout -> 2 CTAs/SM (16 warps/SM) for BM=64.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int HD = 128;
constexpr int TBS = 512;  // 2 KB block-table SMEM cache (64K tokens @ PAGE=128, 8K @ PAGE=16)
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
  return 1.0f / (1.0f + ex2_fast(-x * LOG2E));
}

__device__ __forceinline__ float2 ld_h2_as_f2(const __half* p, bool aligned4) {
  if (aligned4) {
    unsigned u;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(u) : "l"(p));
    return __half22float2(*reinterpret_cast<const __half2*>(&u));
  }
  return make_float2(__half2float(p[0]), __half2float(p[1]));
}

__device__ __forceinline__ unsigned ldg_u32(const void* p) {
  unsigned u;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(u) : "l"(p));
  return u;
}

__device__ __forceinline__ float2 u32_to_f2(unsigned u) {
  return __half22float2(*reinterpret_cast<const __half2*>(&u));
}

// Conflict-free XOR swizzle for 16-chunk (128-half) rows in shared memory.
__device__ __forceinline__ int swiz_q(int r, int c) {
  return c ^ ((c >> 3) & 1) ^ (r & 7);
}

__device__ __forceinline__ int swiz_kv(int r, int c) {
  return c ^ ((c >> 3) & 1) ^ ((r & 3) | ((r >> 1) & 4));
}

template <int BM_WARPS, int BN>
struct PfSM {
  static constexpr int BM = 8 * BM_WARPS;
  static constexpr int NT = 32 * BM_WARPS;
  static constexpr int PS = BN + 8;  // 40 (BN=32) or 72 (BN=64): 16B-aligned, 0-conflict LDS.128

  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS * (int)sizeof(int);
  static constexpr int RP_OFF = (TB_OFF + TB_BYTES + 15) & ~15;
  static constexpr int RP_BYTES = ((BM * (int)sizeof(const __half*) + 15) & ~15);
  static constexpr int Q_OFF = (RP_OFF + RP_BYTES + 15) & ~15;
  static constexpr int Q_BYTES = BM * HD * (int)sizeof(__half);
  static constexpr int KV_OFF = (Q_OFF + Q_BYTES + 15) & ~15;
  static constexpr int KV_BYTES = BN * HD * (int)sizeof(__half);
  static constexpr int P_OFF = (KV_OFF + KV_BYTES + 15) & ~15;
  static constexpr int P_BYTES = BM_WARPS * 8 * PS * (int)sizeof(__half);
  static constexpr int TOTAL = (P_OFF + P_BYTES + 15) & ~15;
};

// Apply scale, sheared relative bias, and causal + SWA mask to one 32-token half-tile of scores.
__device__ __forceinline__ void apply_bias_and_mask_32(
    float* s,
    const __half* rrA,
    const __half* rrB,
    bool validA,
    bool validB,
    int posA,
    int posB,
    int rel_base,
    int tok0,
    int cb,
    int seqlen_k,
    int window,
    int E,
    int max_ext_dist,
    float sl2,
    bool rel_aligned4) {
  // Row A: entries 0, 1 (cb, cb+1) and 4, 5 (cb+4, cb+5)
  {
    const int k_min_ext = posA - max_ext_dist + 1;
    const int k_min_vis = (window > 0) ? (posA - window + 1) : 0;
    if (!validA) {
      s[0] = -INFINITY; s[1] = -INFINITY; s[4] = -INFINITY; s[5] = -INFINITY;
    } else if (tok0 + 31 < k_min_ext && tok0 >= k_min_vis && tok0 + 31 <= posA && tok0 + 31 < seqlen_k) {
      s[0] *= sl2; s[1] *= sl2; s[4] *= sl2; s[5] *= sl2;
    } else if (tok0 >= k_min_ext && tok0 + 31 <= posA && tok0 + 31 < seqlen_k) {
      const float2 b01 = ld_h2_as_f2(rrA + (cb - rel_base), rel_aligned4);
      const float2 b45 = ld_h2_as_f2(rrA + (cb + 4 - rel_base), rel_aligned4);
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
        if (col < seqlen_k && dist >= 0 && (window == 0 || dist < window)) {
          float v = s[e] * sl2;
          if (dist < E) v = fmaf(__half2float(rrA[col - rel_base]), LOG2E, v);
          s[e] = v;
        } else {
          s[e] = -INFINITY;
        }
      }
    }
  }

  // Row B: entries 2, 3 (cb, cb+1) and 6, 7 (cb+4, cb+5)
  {
    const int k_min_ext = posB - max_ext_dist + 1;
    const int k_min_vis = (window > 0) ? (posB - window + 1) : 0;
    if (!validB) {
      s[2] = -INFINITY; s[3] = -INFINITY; s[6] = -INFINITY; s[7] = -INFINITY;
    } else if (tok0 + 31 < k_min_ext && tok0 >= k_min_vis && tok0 + 31 <= posB && tok0 + 31 < seqlen_k) {
      s[2] *= sl2; s[3] *= sl2; s[6] *= sl2; s[7] *= sl2;
    } else if (tok0 >= k_min_ext && tok0 + 31 <= posB && tok0 + 31 < seqlen_k) {
      const float2 b01 = ld_h2_as_f2(rrB + (cb - rel_base), rel_aligned4);
      const float2 b45 = ld_h2_as_f2(rrB + (cb + 4 - rel_base), rel_aligned4);
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
        if (col < seqlen_k && dist >= 0 && (window == 0 || dist < window)) {
          float v = s[e] * sl2;
          if (dist < E) v = fmaf(__half2float(rrB[col - rel_base]), LOG2E, v);
          s[e] = v;
        } else {
          s[e] = -INFINITY;
        }
      }
    }
  }
}

template <bool IS_FP8, int BM_WARPS, int BN>
__global__ void __launch_bounds__(32 * BM_WARPS, 2)
k_rel_pf(
    const __half* __restrict__ q,
    const void* __restrict__ kc_raw,
    const void* __restrict__ vc_raw,
    const __half* __restrict__ rel_logits,
    const __half* __restrict__ gate,
    const __half* __restrict__ sinks,
    __half* __restrict__ out,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cu_seqlens_q,
    int H_q,
    int G,
    int E,
    int E_pad,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
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
  using L_ = PfSM<BM_WARPS, BN>;
  constexpr int BM = L_::BM;
  constexpr int NT = L_::NT;
  constexpr int PS = L_::PS;
  constexpr int TB_CAP = TBS;
  constexpr int N_LD = IS_FP8 ? ((BN * 8) / NT) : ((BN * 16) / NT);
  constexpr int PREFETCH_N = (N_LD <= 4) ? N_LD : 1;

  extern __shared__ __align__(16) uint8_t sm[];
  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  uint4* Qs_u4 = reinterpret_cast<uint4*>(sm + L_::Q_OFF);
  uint4* KVs_u4 = reinterpret_cast<uint4*>(sm + L_::KV_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);

  const int head = blockIdx.x;
  const int raw_m = blockIdx.y;
  const int b = blockIdx.z;

  const int q_start = cu_seqlens_q[b];
  const int q_end = cu_seqlens_q[b + 1];
  const int q_len = q_end - q_start;
  const int eff_q_len = max(0, min(q_len, num_actual_tokens - q_start));
  const int req_m_tiles = (eff_q_len + BM - 1) / BM;
  if (raw_m >= req_m_tiles) return;

  // Longest-first reversed m_tile scheduling for optimal causal load balancing.
  const int m_tile = req_m_tiles - 1 - raw_m;
  const int first_tok = m_tile * BM;
  const int last_tok = min(first_tok + BM, eff_q_len) - 1;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int mw = tid >> 5;
  const int kv_head = head / G;

  const int seqlen_k = cache_seqlens[b];
  const int past = seqlen_k - q_len;
  const int pos_lo = past + first_tok;
  const int pos_hi = past + last_tok;

  const int k_lo = (window > 0) ? max(pos_lo - (window - 1), 0) : 0;
  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int t_lo = k_lo / BN;
  const int t_hi = (k_hi + BN - 1) / BN;

  if (t_hi <= t_lo) {
    // Empty attention range (e.g., seqlen_k == 0 or pos < 0): zero output rows.
    for (int i = tid; i < BM * (HD / 8); i += NT) {
      const int r = i >> 4;
      const int c = (i & 15) * 8;
      const int tok = first_tok + r;
      if (tok < eff_q_len) {
        const int token = q_start + tok;
        *reinterpret_cast<uint4*>(out + ((size_t)token * H_q + head) * HD + c) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    return;
  }

  // Cache block table + RelPtrs + cooperatively load Q tile [BM, 128] into swizzled Qs.
  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  const int n_cache_blk = min(max_blocks, TBS);
  for (int i = tid; i < n_cache_blk; i += NT) {
    Ts[i] = tb[i];
  }
  for (int r = tid; r < BM; r += NT) {
    const int tok = first_tok + r;
    if (tok < eff_q_len) {
      const int token = q_start + tok;
      RelPtrs[r] = rel_logits + (size_t)token * (size_t)stride_rel_t + (size_t)head * (size_t)stride_rel_h;
    } else {
      RelPtrs[r] = rel_logits;
    }
  }

#pragma unroll
  for (int u = 0; u < 4; u++) {
    const int idx = tid + u * NT;
    const int r = idx >> 4;
    const int c = idx & 15;
    const int tok = first_tok + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (tok < eff_q_len) {
      const int token = q_start + tok;
      const __half* qp_src = q + (size_t)token * (size_t)stride_q_t + (size_t)head * (size_t)stride_q_h + 8 * c;
      if (q_aligned16) {
        v = ldg16(qp_src);
      } else {
        __half tmp[8];
#pragma unroll
        for (int z = 0; z < 8; z++) tmp[z] = qp_src[z];
        v = *reinterpret_cast<const uint4*>(tmp);
      }
    }
    Qs_u4[r * 16 + swiz_q(r, c)] = v;
  }
  __syncthreads();

  // Volta mma.m8n8k4 lane geometry.
  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;          // 0..7
  const int r0 = (lane & 1) + 4 * hi;    // 0, 1, 4, 5
  const int c0 = lane & 2;               // 0 or 2
  const int ktok = 8 * qp + arow;        // 0..31

  // Since BM in {16, 32, 64} divides 128, rel_base is CTA-uniform!
  const int tile_end = min(((first_tok / 128) + 1) * 128, q_len);
  const int rel_base = ((past + tile_end + 127) / 128) * 128 - E_pad;
  const int max_ext_dist = (window > 0 && window < E) ? window : E;

  const int lrA = mw * 8 + r0;
  const int lrB = lrA + 2;
  const int tokA = first_tok + lrA;
  const int tokB = tokA + 2;
  const bool validA = tokA < eff_q_len;
  const bool validB = tokB < eff_q_len;
  const int posA = validA ? (past + tokA) : -1;
  const int posB = validB ? (past + tokB) : -1;

  // Per-warp pos range for skipping completely masked tiles.
  const int w_pos_min = past + first_tok + mw * 8;
  const int w_pos_max = past + min(first_tok + mw * 8 + 7, eff_q_len - 1);
  const bool warp_has_rows = (first_tok + mw * 8) < eff_q_len;

  const int pmask = page - 1;
  const int qrow_r = mw * 8 + arow;
  const int q_swiz_base = qrow_r & 7;
  const int k_swiz_base = (ktok & 3) | ((ktok >> 1) & 4);
  const int vc0 = 4 * qp + 2 * hi;
  const int vc1 = vc0 + 1;
  const int vc0_f = vc0 ^ ((vc0 >> 3) & 1);
  const int vc1_f = vc1 ^ ((vc1 >> 3) & 1);

  // Fold kv_head offset into base pointers once to keep register pressure minimal.
  const uint8_t* kc_base_b = reinterpret_cast<const uint8_t*>(kc_raw) +
                             ((int64_t)kv_head * kh) * (IS_FP8 ? 1 : 2);
  const uint8_t* vc_base_b = reinterpret_cast<const uint8_t*>(vc_raw) +
                             ((int64_t)kv_head * vh) * (IS_FP8 ? 1 : 2);
  const int64_t kb_b = kb * (IS_FP8 ? 1 : 2);
  const int64_t kt_b = kt * (IS_FP8 ? 1 : 2);
  const int64_t vb_b = vb * (IS_FP8 ? 1 : 2);
  const int64_t vt_b = vt * (IS_FP8 ? 1 : 2);

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TB_CAP ? Ts[pi] : tb[pi]);
  };

  auto load_one_kv = [&](const uint8_t* base_b, int64_t sb_b, int64_t st_b,
                         int tok0, int row, int col_u, int64_t single_pg, bool single_page_ok) -> uint4 {
    if (single_page_ok) {
      const int sl = (tok0 & pmask) + row;
      return ldg16(base_b + single_pg * sb_b + (int64_t)sl * st_b + 16 * col_u);
    }
    const int k_idx = tok0 + row;
    if (k_idx < seqlen_k) {
      const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
      const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
      const int64_t pg = get_pg(pi);
      return ldg16(base_b + pg * sb_b + (int64_t)sl * st_b + 16 * col_u);
    }
    return make_uint4(0u, 0u, 0u, 0u);
  };

  auto store_one_kv = [&](uint4 w, int row, int col_u) {
    if constexpr (IS_FP8) {
      unsigned lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
      e4m3x4(w.x, lo0, hi0);
      e4m3x4(w.y, lo1, hi1);
      KVs_u4[row * 16 + swiz_kv(row, 2 * col_u)] = make_uint4(lo0, hi0, lo1, hi1);
      e4m3x4(w.z, lo2, hi2);
      e4m3x4(w.w, lo3, hi3);
      KVs_u4[row * 16 + swiz_kv(row, 2 * col_u + 1)] = make_uint4(lo2, hi2, lo3, hi3);
    } else {
      KVs_u4[row * 16 + swiz_kv(row, col_u)] = w;
    }
  };

  uint4 kv_reg[PREFETCH_N];

  auto prefetch_kv = [&](const uint8_t* base_b, int64_t sb_b, int64_t st_b, int tok0) {
    constexpr int MIN_SINGLE_SHIFT = (BN == 64) ? 6 : 5;
    const bool single_page_ok = (page_shift >= MIN_SINGLE_SHIFT) && (tok0 + BN - 1 < seqlen_k);
    const int64_t single_pg = single_page_ok ? get_pg(tok0 >> page_shift) : 0;
#pragma unroll
    for (int u = 0; u < PREFETCH_N; u++) {
      const int idx = tid + u * NT;
      const int row = IS_FP8 ? (idx >> 3) : (idx >> 4);
      const int col_u = IS_FP8 ? (idx & 7) : (idx & 15);
      kv_reg[u] = load_one_kv(base_b, sb_b, st_b, tok0, row, col_u, single_pg, single_page_ok);
    }
  };

  auto commit_or_load_kv = [&](const uint8_t* base_b, int64_t sb_b, int64_t st_b, int tok0) {
#pragma unroll
    for (int u = 0; u < PREFETCH_N; u++) {
      const int idx = tid + u * NT;
      const int row = IS_FP8 ? (idx >> 3) : (idx >> 4);
      const int col_u = IS_FP8 ? (idx & 7) : (idx & 15);
      store_one_kv(kv_reg[u], row, col_u);
    }
    if constexpr (N_LD > PREFETCH_N) {
      constexpr int MIN_SINGLE_SHIFT = (BN == 64) ? 6 : 5;
      const bool single_page_ok = (page_shift >= MIN_SINGLE_SHIFT) && (tok0 + BN - 1 < seqlen_k);
      const int64_t single_pg = single_page_ok ? get_pg(tok0 >> page_shift) : 0;
#pragma unroll
      for (int u = PREFETCH_N; u < N_LD; u++) {
        const int idx = tid + u * NT;
        const int row = IS_FP8 ? (idx >> 3) : (idx >> 4);
        const int col_u = IS_FP8 ? (idx & 7) : (idx & 15);
        uint4 w = load_one_kv(base_b, sb_b, st_b, tok0, row, col_u, single_pg, single_page_ok);
        store_one_kv(w, row, col_u);
      }
    }
  };

  // Initialize O accumulators and online softmax state.
  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[j][e] = 0.f;
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;

  // Prefetch first tile's K into registers.
  prefetch_kv(kc_base_b, kb_b, kt_b, t_lo * BN);

  // Main loop over BN-token KV tiles.
  for (int t = t_lo; t < t_hi; t++) {
    const int tok0 = t * BN;
    const bool more = (t + 1) < t_hi;
    const bool warp_active = warp_has_rows && (tok0 <= w_pos_max) &&
                             (window == 0 || (tok0 + BN - 1 >= w_pos_min - window + 1));

    // Commit K[t] -> KVs shared memory.
    commit_or_load_kv(kc_base_b, kb_b, kt_b, tok0);
    __syncthreads();

    // Issue V[t] prefetch into registers BEFORE QK HMMAs to hide memory latency!
    prefetch_kv(vc_base_b, vb_b, vt_b, tok0);

    if (warp_active) {
      // ---- QK: HMMA.884 [8, 128] @ [128, BN] ----
      float s0[8];
#pragma unroll
      for (int e = 0; e < 8; e++) s0[e] = 0.f;

      if constexpr (BN == 64) {
        float s1[8];
#pragma unroll
        for (int e = 0; e < 8; e++) s1[e] = 0.f;

#pragma unroll 2
        for (int k8 = 0; k8 < 16; k8++) {
          const int c_base = k8 ^ ((k8 >> 3) & 1);
          const uint4 qv = Qs_u4[qrow_r * 16 + (c_base ^ q_swiz_base)];
          const uint4 k0 = KVs_u4[ktok * 16 + (c_base ^ k_swiz_base)];
          const uint4 k1 = KVs_u4[(32 + ktok) * 16 + (c_base ^ k_swiz_base)];
          mma_rc(s0, qv.x, qv.y, k0.x, k0.y);
          mma_rc(s1, qv.x, qv.y, k1.x, k1.y);
          mma_rc(s0, qv.z, qv.w, k0.z, k0.w);
          mma_rc(s1, qv.z, qv.w, k1.z, k1.w);
        }

        // ---- Scale, sheared rel_logits, causal + SWA mask ----
        const __half* rrA = RelPtrs[lrA];
        const __half* rrB = RelPtrs[lrB];
        const int cb0 = tok0 + 8 * qp + c0;
        apply_bias_and_mask_32(
            s0, rrA, rrB, validA, validB, posA, posB, rel_base,
            tok0, cb0, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
        apply_bias_and_mask_32(
            s1, rrA, rrB, validA, validB, posA, posB, rel_base,
            tok0 + 32, cb0 + 32, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);

        // ---- Online Softmax across all 64 columns ----
        float x0 = fmaxf(fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[4], s0[5])),
                         fmaxf(fmaxf(s1[0], s1[1]), fmaxf(s1[4], s1[5])));
        float x2 = fmaxf(fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s0[6], s0[7])),
                         fmaxf(fmaxf(s1[2], s1[3]), fmaxf(s1[6], s1[7])));
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
          const float z = (e & 2) ? z2 : z0;
          s0[e] = ex2_fast(s0[e] - z);
          s1[e] = ex2_fast(s1[e] - z);
        }
        l0 = l0 * a0 + ((s0[0] + s0[1]) + (s0[4] + s0[5])) + ((s1[0] + s1[1]) + (s1[4] + s1[5]));
        l2 = l2 * a2 + ((s0[2] + s0[3]) + (s0[6] + s0[7])) + ((s1[2] + s1[3]) + (s1[6] + s1[7]));
        m0 = n0;
        m2 = n2;

#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;

        // Stage FP16 P into warp-private Ps buffer.
        __half* Pw = Ps + mw * 8 * PS;
        const int pc = 8 * qp + c0;
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s0[0], s0[1]);
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s0[4], s0[5]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s0[2], s0[3]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s0[6], s0[7]);
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 32) = __floats2half2_rn(s1[0], s1[1]);
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 36) = __floats2half2_rn(s1[4], s1[5]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 32) = __floats2half2_rn(s1[2], s1[3]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 36) = __floats2half2_rn(s1[6], s1[7]);
      } else {
        // BN == 32
#pragma unroll 2
        for (int k8 = 0; k8 < 16; k8++) {
          const int c_base = k8 ^ ((k8 >> 3) & 1);
          const uint4 qv = Qs_u4[qrow_r * 16 + (c_base ^ q_swiz_base)];
          const uint4 k0 = KVs_u4[ktok * 16 + (c_base ^ k_swiz_base)];
          mma_rc(s0, qv.x, qv.y, k0.x, k0.y);
          mma_rc(s0, qv.z, qv.w, k0.z, k0.w);
        }

        const __half* rrA = RelPtrs[lrA];
        const __half* rrB = RelPtrs[lrB];
        const int cb0 = tok0 + 8 * qp + c0;
        apply_bias_and_mask_32(
            s0, rrA, rrB, validA, validB, posA, posB, rel_base,
            tok0, cb0, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);

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
        for (int e = 0; e < 8; e++) {
          s0[e] = ex2_fast(s0[e] - ((e & 2) ? z2 : z0));
        }
        l0 = l0 * a0 + (s0[0] + s0[1]) + (s0[4] + s0[5]);
        l2 = l2 * a2 + (s0[2] + s0[3]) + (s0[6] + s0[7]);
        m0 = n0;
        m2 = n2;

#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;

        __half* Pw = Ps + mw * 8 * PS;
        const int pc = 8 * qp + c0;
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s0[0], s0[1]);
        *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s0[4], s0[5]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s0[2], s0[3]);
        *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s0[6], s0[7]);
      }
    }

    // All warps finished reading K[t] from KVs -> commit V[t] into KVs.
    __syncthreads();
    commit_or_load_kv(vc_base_b, vb_b, vt_b, tok0);
    __syncthreads();

    // Issue K[t+1] prefetch into registers BEFORE PV HMMAs!
    if (more) {
      prefetch_kv(kc_base_b, kb_b, kt_b, tok0 + BN);
    }

    if (warp_active) {
      // ---- PV: HMMA.884 [8, BN] @ [BN, 128] ----
      const __half* Pw_row = Ps + (mw * 8 + arow) * PS;
#pragma unroll 2
      for (int i2 = 0; i2 < BN / 8; i2++) {
        const uint4 p8 = *reinterpret_cast<const uint4*>(Pw_row + 8 * i2);
        const int vsw = l3 | ((i2 & 1) << 2);
        {
          const uint4* Vr_u4 = KVs_u4 + (8 * i2 + l3) * 16;
          const uint4 v0 = Vr_u4[vc0_f ^ vsw];
          const uint4 v1 = Vr_u4[vc1_f ^ vsw];
          mma_rr(o[0], p8.x, p8.y, v0.x, v0.y);
          mma_rr(o[1], p8.x, p8.y, v0.z, v0.w);
          mma_rr(o[2], p8.x, p8.y, v1.x, v1.y);
          mma_rr(o[3], p8.x, p8.y, v1.z, v1.w);
        }
        {
          const uint4* Vr_u4 = KVs_u4 + (8 * i2 + 4 + l3) * 16;
          const uint4 v0 = Vr_u4[vc0_f ^ vsw];
          const uint4 v1 = Vr_u4[vc1_f ^ vsw];
          mma_rr(o[0], p8.z, p8.w, v0.x, v0.y);
          mma_rr(o[1], p8.z, p8.w, v0.z, v0.w);
          mma_rr(o[2], p8.z, p8.w, v1.x, v1.y);
          mma_rr(o[3], p8.z, p8.w, v1.z, v1.w);
        }
      }
    }

    // Wait for all warps to finish reading V[t] before next iteration overwrites KVs.
    if (more) {
      __syncthreads();
    }
  }

  // Finalize per-row denominator across the 8 lanes sharing each row.
#pragma unroll
  for (int d = 2; d <= 8; d <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, d);
    l2 += __shfl_xor_sync(0xffffffffu, l2, d);
  }

  // ---- Fused gate_sink epilogue ----
  float rs0 = 0.f, rs2 = 0.f;
  if (validA && l0 > 0.f) {
    const int tokenA = q_start + tokA;
    const float lse_nat = (m0 + log2f(l0)) * LN2;
    const float sink_v = __half2float(sinks[(size_t)tokenA * (size_t)stride_sink + head]);
    rs0 = (vsc / l0) * sigmoid_f32(lse_nat - sink_v);
  }
  if (validB && l2 > 0.f) {
    const int tokenB = q_start + tokB;
    const float lse_nat = (m2 + log2f(l2)) * LN2;
    const float sink_v = __half2float(sinks[(size_t)tokenB * (size_t)stride_sink + head]);
    rs2 = (vsc / l2) * sigmoid_f32(lse_nat - sink_v);
  }

#pragma unroll
  for (int sr = 0; sr < 2; sr++) {
    const bool valid = sr ? validB : validA;
    if (valid) {
      const float rsc = sr ? rs2 : rs0;
      const int token = q_start + (sr ? tokB : tokA);
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
}

template <bool IS_FP8, int BM_WARPS, int BN>
void dispatch_pf(
    const __half* q_ptr,
    const void* kc_ptr,
    const void* vc_ptr,
    const __half* rel_ptr,
    const __half* gate_ptr,
    const __half* sinks_ptr,
    __half* out_ptr,
    const int* bt_ptr,
    const int* seq_ptr,
    const int* cu_q_ptr,
    int B,
    int H_q,
    int G,
    int E,
    int E_pad,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int max_query_len,
    int num_actual_tokens,
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
  constexpr int BM = 8 * BM_WARPS;
  constexpr int smem = PfSM<BM_WARPS, BN>::TOTAL;
  // Request 96 KB SMEM carveout on Volta so 2-3 CTAs/SM stay resident!
  static bool carveout_set[16] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 0 && dev < 16 && !carveout_set[dev]) {
    cudaFuncSetAttribute(
        k_rel_pf<IS_FP8, BM_WARPS, BN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    carveout_set[dev] = true;
  }

  const int num_m_tiles = (max_query_len + BM - 1) / BM;
  dim3 grid(H_q, num_m_tiles, B);
  k_rel_pf<IS_FP8, BM_WARPS, BN><<<grid, 32 * BM_WARPS, smem, stream>>>(
      q_ptr, kc_ptr, vc_ptr, rel_ptr, gate_ptr, sinks_ptr, out_ptr,
      bt_ptr, seq_ptr, cu_q_ptr,
      H_q, G, E, E_pad, window, page, page_shift, max_blocks, num_actual_tokens,
      stride_q_t, stride_q_h, kb, kt, kh, vb, vt, vh,
      stride_rel_t, stride_rel_h, stride_bt, stride_gate, stride_sink,
      sl2, vsc, rel_aligned4, gate_aligned4, q_aligned16);
}

// ============================================================================
// High-throughput 2x8 (16 Q rows per warp) Volta HMMA.884 kernel!
// BM_WARPS in {2, 4} -> BM in {32, 64}, BN in {32, 64}.
// Reuses every K and V LDS.128 across 2 Q-strips -> cuts SMEM read traffic in
// half (192 KB vs 320 KB per 64x64 tile), shifting QK & PV to 100% HMMA-bound!
// Also prefetches interior rel_logits in parallel BEFORE QK!
// ============================================================================
template <int BM_WARPS, int BN>
struct PfSM2x8 {
  static constexpr int BM = 16 * BM_WARPS;
  static constexpr int NW = BM_WARPS;
  static constexpr int NT = 32 * NW;
  static constexpr int PS = BN + 8;  // 40 (BN=32) or 72 (BN=64)

  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = ((BM_WARPS == 2 ? 256 : TBS) * (int)sizeof(int));
  static constexpr int TB_CAP = (BM_WARPS == 2 ? 256 : TBS);
  static constexpr int RP_OFF = (TB_OFF + TB_BYTES + 15) & ~15;
  static constexpr int RP_BYTES = ((BM * (int)sizeof(const __half*) + 15) & ~15);
  static constexpr int Q_OFF = (RP_OFF + RP_BYTES + 15) & ~15;
  static constexpr int Q_BYTES = BM * HD * (int)sizeof(__half);
  static constexpr int KV_OFF = (Q_OFF + Q_BYTES + 15) & ~15;
  static constexpr int KV_BYTES = BN * HD * (int)sizeof(__half);
  static constexpr int P_OFF = (KV_OFF + KV_BYTES + 15) & ~15;
  static constexpr int P_BYTES = BM * PS * (int)sizeof(__half);
  static constexpr int TOTAL = (P_OFF + P_BYTES + 15) & ~15;
};

template <bool IS_FP8, int BM_WARPS, int BN>
__global__ void __launch_bounds__(32 * BM_WARPS, (BM_WARPS == 2) ? 3 : ((BN == 32 && IS_FP8) ? 3 : 2))
k_rel_pf_2x8(
    const __half* __restrict__ q,
    const void* __restrict__ kc_raw,
    const void* __restrict__ vc_raw,
    const __half* __restrict__ rel_logits,
    const __half* __restrict__ gate,
    const __half* __restrict__ sinks,
    __half* __restrict__ out,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cu_seqlens_q,
    int H_q,
    int G,
    int E,
    int E_pad,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
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
  using L_ = PfSM2x8<BM_WARPS, BN>;
  constexpr int BM = L_::BM;
  constexpr int NT = L_::NT;
  constexpr int PS = L_::PS;
  constexpr int TB_CAP = L_::TB_CAP;
  constexpr int N_LD = IS_FP8 ? ((BN * 8) / NT) : ((BN * 16) / NT);
  constexpr int PREFETCH_N = (N_LD <= 4) ? N_LD : 4;

  extern __shared__ __align__(16) uint8_t sm[];
  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  uint4* Qs_u4 = reinterpret_cast<uint4*>(sm + L_::Q_OFF);
  uint4* KVs_u4 = reinterpret_cast<uint4*>(sm + L_::KV_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);

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
  const int mw = tid >> 5;
  const int kv_head = head / G;

  const int seqlen_k = cache_seqlens[b];
  const int past = seqlen_k - q_len;
  const int pos_lo = past + first_tok;
  const int pos_hi = past + last_tok;

  const int k_lo = (window > 0) ? max(pos_lo - (window - 1), 0) : 0;
  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int t_lo = k_lo / BN;
  const int t_hi = (k_hi + BN - 1) / BN;

  if (t_hi <= t_lo) {
    for (int i = tid; i < BM * (HD / 8); i += NT) {
      const int r = i >> 4;
      const int c = (i & 15) * 8;
      const int tok = first_tok + r;
      if (tok < eff_q_len) {
        const int token = q_start + tok;
        *reinterpret_cast<uint4*>(out + ((size_t)token * H_q + head) * HD + c) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    return;
  }

  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  const int n_cache_blk = min(max_blocks, TB_CAP);
  for (int i = tid; i < n_cache_blk; i += NT) {
    Ts[i] = tb[i];
  }
  if (tid < BM) {
    const int tok = first_tok + tid;
    if (tok < eff_q_len) {
      const int token = q_start + tok;
      RelPtrs[tid] = rel_logits + (size_t)token * (size_t)stride_rel_t + (size_t)head * (size_t)stride_rel_h;
    } else {
      RelPtrs[tid] = rel_logits;
    }
  }

#pragma unroll
  for (int u = 0; u < 8; u++) {
    const int idx = tid + u * NT;
    const int r = idx >> 4;
    const int c = idx & 15;
    const int tok = first_tok + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (tok < eff_q_len) {
      const int token = q_start + tok;
      const __half* qp_src = q + (size_t)token * (size_t)stride_q_t + (size_t)head * (size_t)stride_q_h + 8 * c;
      if (q_aligned16) {
        v = ldg16(qp_src);
      } else {
        __half tmp[8];
#pragma unroll
        for (int z = 0; z < 8; z++) tmp[z] = qp_src[z];
        v = *reinterpret_cast<const uint4*>(tmp);
      }
    }
    Qs_u4[r * 16 + swiz_q(r, c)] = v;
  }
  __syncthreads();

  // Volta mma.m8n8k4 lane geometry.
  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;          // 0..7
  const int r0 = (lane & 1) + 4 * hi;    // 0, 1, 4, 5
  const int c0 = lane & 2;               // 0 or 2
  const int ktok = 8 * qp + arow;        // 0..31

  const int tile_end = min(((first_tok / 128) + 1) * 128, q_len);
  const int rel_base = ((past + tile_end + 127) / 128) * 128 - E_pad;
  const int max_ext_dist = (window > 0 && window < E) ? window : E;

  const int lrA0 = mw * 16 + r0;
  const int lrB0 = lrA0 + 2;
  const int lrA1 = lrA0 + 8;
  const int lrB1 = lrA1 + 2;
  const int tokA0 = first_tok + lrA0;
  const int tokB0 = tokA0 + 2;
  const int tokA1 = tokA0 + 8;
  const int tokB1 = tokA1 + 2;
  const bool validA0 = tokA0 < eff_q_len;
  const bool validB0 = tokB0 < eff_q_len;
  const bool validA1 = tokA1 < eff_q_len;
  const bool validB1 = tokB1 < eff_q_len;
  const int posA0 = validA0 ? (past + tokA0) : -1;
  const int posB0 = validB0 ? (past + tokB0) : -1;
  const int posA1 = validA1 ? (past + tokA1) : -1;
  const int posB1 = validB1 ? (past + tokB1) : -1;

  const int w_pos_min = past + first_tok + mw * 16;
  const int w_pos_max = past + min(first_tok + mw * 16 + 15, eff_q_len - 1);
  const bool warp_has_rows = (first_tok + mw * 16) < eff_q_len;

  const int pmask = page - 1;
  const int qrow0 = mw * 16 + arow;
  const int qrow1 = qrow0 + 8;
  const int k_swiz_base = (ktok & 3) | ((ktok >> 1) & 4);
  const int vc0 = 4 * qp + 2 * hi;
  const int vc1 = vc0 + 1;
  const int vc0_f = vc0 ^ ((vc0 >> 3) & 1);
  const int vc1_f = vc1 ^ ((vc1 >> 3) & 1);

  const uint8_t* kc_base_b = reinterpret_cast<const uint8_t*>(kc_raw) +
                             ((int64_t)kv_head * kh) * (IS_FP8 ? 1 : 2);
  const uint8_t* vc_base_b = reinterpret_cast<const uint8_t*>(vc_raw) +
                             ((int64_t)kv_head * vh) * (IS_FP8 ? 1 : 2);
  const int64_t kb_b = kb * (IS_FP8 ? 1 : 2);
  const int64_t kt_b = kt * (IS_FP8 ? 1 : 2);
  const int64_t vb_b = vb * (IS_FP8 ? 1 : 2);
  const int64_t vt_b = vt * (IS_FP8 ? 1 : 2);

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TB_CAP ? Ts[pi] : tb[pi]);
  };

  auto load_one_kv = [&](const uint8_t* base_b, int64_t sb_b, int64_t st_b,
                         int tok0, int row, int col_u, int64_t single_pg, bool single_page_ok) -> uint4 {
    if (single_page_ok) {
      const int sl = (tok0 & pmask) + row;
      return ldg16(base_b + single_pg * sb_b + (int64_t)sl * st_b + 16 * col_u);
    }
    const int k_idx = tok0 + row;
    if (k_idx < seqlen_k) {
      const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
      const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
      const int64_t pg = get_pg(pi);
      return ldg16(base_b + pg * sb_b + (int64_t)sl * st_b + 16 * col_u);
    }
    return make_uint4(0u, 0u, 0u, 0u);
  };

  auto store_one_kv = [&](uint4 w, int row, int col_u) {
    if constexpr (IS_FP8) {
      unsigned lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
      e4m3x4(w.x, lo0, hi0);
      e4m3x4(w.y, lo1, hi1);
      KVs_u4[row * 16 + swiz_kv(row, 2 * col_u)] = make_uint4(lo0, hi0, lo1, hi1);
      e4m3x4(w.z, lo2, hi2);
      e4m3x4(w.w, lo3, hi3);
      KVs_u4[row * 16 + swiz_kv(row, 2 * col_u + 1)] = make_uint4(lo2, hi2, lo3, hi3);
    } else {
      KVs_u4[row * 16 + swiz_kv(row, col_u)] = w;
    }
  };

  uint4 kv_reg[PREFETCH_N];

  auto prefetch_kv = [&](const uint8_t* base_b, int64_t sb_b, int64_t st_b, int tok0) {
    constexpr int MIN_SINGLE_SHIFT = (BN == 64) ? 6 : 5;
    const bool single_page_ok = (page_shift >= MIN_SINGLE_SHIFT) && (tok0 + BN - 1 < seqlen_k);
    const int64_t single_pg = single_page_ok ? get_pg(tok0 >> page_shift) : 0;
#pragma unroll
    for (int u = 0; u < PREFETCH_N; u++) {
      const int idx = tid + u * NT;
      const int row = IS_FP8 ? (idx >> 3) : (idx >> 4);
      const int col_u = IS_FP8 ? (idx & 7) : (idx & 15);
      kv_reg[u] = load_one_kv(base_b, sb_b, st_b, tok0, row, col_u, single_pg, single_page_ok);
    }
  };

  auto commit_or_load_kv = [&](const uint8_t* base_b, int64_t sb_b, int64_t st_b, int tok0) {
#pragma unroll
    for (int u = 0; u < PREFETCH_N; u++) {
      const int idx = tid + u * NT;
      const int row = IS_FP8 ? (idx >> 3) : (idx >> 4);
      const int col_u = IS_FP8 ? (idx & 7) : (idx & 15);
      store_one_kv(kv_reg[u], row, col_u);
    }
    if constexpr (N_LD > PREFETCH_N) {
      constexpr int MIN_SINGLE_SHIFT = (BN == 64) ? 6 : 5;
      const bool single_page_ok = (page_shift >= MIN_SINGLE_SHIFT) && (tok0 + BN - 1 < seqlen_k);
      const int64_t single_pg = single_page_ok ? get_pg(tok0 >> page_shift) : 0;
#pragma unroll
      for (int u = PREFETCH_N; u < N_LD; u++) {
        const int idx = tid + u * NT;
        const int row = IS_FP8 ? (idx >> 3) : (idx >> 4);
        const int col_u = IS_FP8 ? (idx & 7) : (idx & 15);
        uint4 w = load_one_kv(base_b, sb_b, st_b, tok0, row, col_u, single_pg, single_page_ok);
        store_one_kv(w, row, col_u);
      }
    }
  };

  float o[2][4][8];
#pragma unroll
  for (int m = 0; m < 2; m++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[m][j][e] = 0.f;
  float m0[2] = {-INFINITY, -INFINITY};
  float m2[2] = {-INFINITY, -INFINITY};
  float l0[2] = {0.f, 0.f};
  float l2[2] = {0.f, 0.f};

  prefetch_kv(kc_base_b, kb_b, kt_b, t_lo * BN);

  for (int t = t_lo; t < t_hi; t++) {
    const int tok0 = t * BN;
    const bool more = (t + 1) < t_hi;
    const bool warp_active = warp_has_rows && (tok0 <= w_pos_max) &&
                             (window == 0 || (tok0 + BN - 1 >= w_pos_min - window + 1));

    commit_or_load_kv(kc_base_b, kb_b, kt_b, tok0);
    __syncthreads();

    prefetch_kv(vc_base_b, vb_b, vt_b, tok0);

    if (warp_active) {
      // Warp-uniform interior fast paths (zero warp divergence!).
      const bool warp_full = (first_tok + mw * 16 + 15) < eff_q_len;
      const bool all_far_past = warp_full && (tok0 + BN - 1 < w_pos_min - max_ext_dist + 1) &&
                                (window == 0 || tok0 >= w_pos_max - window + 1) &&
                                (tok0 + BN - 1 <= w_pos_min) && (tok0 + BN - 1 < seqlen_k);
      const bool all_in_ext = rel_aligned4 && warp_full && (tok0 >= w_pos_max - max_ext_dist + 1) &&
                              (tok0 + BN - 1 <= w_pos_min) && (tok0 + BN - 1 < seqlen_k);

      const int cb0 = tok0 + 8 * qp + c0;
      const int rel_off0 = cb0 - rel_base;

      // Prefetch interior rel_logits in parallel BEFORE QK when BN == 64 (255-reg budget).
      unsigned rb_a0_0 = 0, rb_a0_4 = 0, rb_b0_0 = 0, rb_b0_4 = 0;
      unsigned rb_a1_0 = 0, rb_a1_4 = 0, rb_b1_0 = 0, rb_b1_4 = 0;
      unsigned rb_a0_32 = 0, rb_a0_36 = 0, rb_b0_32 = 0, rb_b0_36 = 0;
      unsigned rb_a1_32 = 0, rb_a1_36 = 0, rb_b1_32 = 0, rb_b1_36 = 0;

      if constexpr (BN == 64) {
        if (all_in_ext) {
          const __half* pA0 = RelPtrs[lrA0] + rel_off0;
          const __half* pB0 = RelPtrs[lrB0] + rel_off0;
          const __half* pA1 = RelPtrs[lrA1] + rel_off0;
          const __half* pB1 = RelPtrs[lrB1] + rel_off0;
          rb_a0_0 = ldg_u32(pA0);
          rb_a0_4 = ldg_u32(pA0 + 4);
          rb_b0_0 = ldg_u32(pB0);
          rb_b0_4 = ldg_u32(pB0 + 4);
          rb_a1_0 = ldg_u32(pA1);
          rb_a1_4 = ldg_u32(pA1 + 4);
          rb_b1_0 = ldg_u32(pB1);
          rb_b1_4 = ldg_u32(pB1 + 4);
          rb_a0_32 = ldg_u32(pA0 + 32);
          rb_a0_36 = ldg_u32(pA0 + 36);
          rb_b0_32 = ldg_u32(pB0 + 32);
          rb_b0_36 = ldg_u32(pB0 + 36);
          rb_a1_32 = ldg_u32(pA1 + 32);
          rb_a1_36 = ldg_u32(pA1 + 36);
          rb_b1_32 = ldg_u32(pB1 + 32);
          rb_b1_36 = ldg_u32(pB1 + 36);
        }
      }

      if constexpr (BN == 64) {
        float s0[2][8], s1[2][8];
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
          for (int e = 0; e < 8; e++) { s0[m][e] = 0.f; s1[m][e] = 0.f; }

#pragma unroll 4
        for (int k8 = 0; k8 < 16; k8++) {
          const int c_base = k8 ^ ((k8 >> 3) & 1);
          const uint4 q0 = Qs_u4[qrow0 * 16 + (c_base ^ arow)];
          const uint4 q1 = Qs_u4[qrow1 * 16 + (c_base ^ arow)];
          const uint4 k0 = KVs_u4[ktok * 16 + (c_base ^ k_swiz_base)];
          const uint4 k1 = KVs_u4[(32 + ktok) * 16 + (c_base ^ k_swiz_base)];
          mma_rc(s0[0], q0.x, q0.y, k0.x, k0.y);
          mma_rc(s0[1], q1.x, q1.y, k0.x, k0.y);
          mma_rc(s1[0], q0.x, q0.y, k1.x, k1.y);
          mma_rc(s1[1], q1.x, q1.y, k1.x, k1.y);
          mma_rc(s0[0], q0.z, q0.w, k0.z, k0.w);
          mma_rc(s0[1], q1.z, q1.w, k0.z, k0.w);
          mma_rc(s1[0], q0.z, q0.w, k1.z, k1.w);
          mma_rc(s1[1], q1.z, q1.w, k1.z, k1.w);
        }

        if (all_far_past) {
#pragma unroll
          for (int m = 0; m < 2; m++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
              s0[m][e] *= sl2;
              s1[m][e] *= sl2;
            }
        } else if (all_in_ext) {
          auto add_rel = [&](float* sc, unsigned u01, unsigned u45, unsigned v01, unsigned v45) {
            const float2 f01 = u32_to_f2(u01);
            const float2 f45 = u32_to_f2(u45);
            const float2 g01 = u32_to_f2(v01);
            const float2 g45 = u32_to_f2(v45);
            sc[0] = fmaf(f01.x, LOG2E, sc[0] * sl2);
            sc[1] = fmaf(f01.y, LOG2E, sc[1] * sl2);
            sc[4] = fmaf(f45.x, LOG2E, sc[4] * sl2);
            sc[5] = fmaf(f45.y, LOG2E, sc[5] * sl2);
            sc[2] = fmaf(g01.x, LOG2E, sc[2] * sl2);
            sc[3] = fmaf(g01.y, LOG2E, sc[3] * sl2);
            sc[6] = fmaf(g45.x, LOG2E, sc[6] * sl2);
            sc[7] = fmaf(g45.y, LOG2E, sc[7] * sl2);
          };
          add_rel(s0[0], rb_a0_0, rb_a0_4, rb_b0_0, rb_b0_4);
          add_rel(s0[1], rb_a1_0, rb_a1_4, rb_b1_0, rb_b1_4);
          add_rel(s1[0], rb_a0_32, rb_a0_36, rb_b0_32, rb_b0_36);
          add_rel(s1[1], rb_a1_32, rb_a1_36, rb_b1_32, rb_b1_36);
        } else {
          const __half* pA0 = RelPtrs[lrA0];
          const __half* pB0 = RelPtrs[lrB0];
          const __half* pA1 = RelPtrs[lrA1];
          const __half* pB1 = RelPtrs[lrB1];
          apply_bias_and_mask_32(
              s0[0], pA0, pB0, validA0, validB0, posA0, posB0, rel_base,
              tok0, cb0, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
          apply_bias_and_mask_32(
              s1[0], pA0, pB0, validA0, validB0, posA0, posB0, rel_base,
              tok0 + 32, cb0 + 32, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
          apply_bias_and_mask_32(
              s0[1], pA1, pB1, validA1, validB1, posA1, posB1, rel_base,
              tok0, cb0, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
          apply_bias_and_mask_32(
              s1[1], pA1, pB1, validA1, validB1, posA1, posB1, rel_base,
              tok0 + 32, cb0 + 32, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
        }

        // Single unified online softmax for both strips m = 0, 1 (SFU interleaved with FP32 rescale).
#pragma unroll
        for (int m = 0; m < 2; m++) {
          float x0 = fmaxf(fmaxf(fmaxf(s0[m][0], s0[m][1]), fmaxf(s0[m][4], s0[m][5])),
                           fmaxf(fmaxf(s1[m][0], s1[m][1]), fmaxf(s1[m][4], s1[m][5])));
          float x2 = fmaxf(fmaxf(fmaxf(s0[m][2], s0[m][3]), fmaxf(s0[m][6], s0[m][7])),
                           fmaxf(fmaxf(s1[m][2], s1[m][3]), fmaxf(s1[m][6], s1[m][7])));
#pragma unroll
          for (int d = 2; d <= 8; d <<= 1) {
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
            x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
          }
          const float n0 = fmaxf(m0[m], x0);
          const float n2 = fmaxf(m2[m], x2);
          const float a0 = (m0[m] == -INFINITY) ? 0.f : ex2_fast(m0[m] - n0);
          const float a2 = (m2[m] == -INFINITY) ? 0.f : ex2_fast(m2[m] - n2);
          const float z0 = (n0 == -INFINITY) ? 0.f : n0;
          const float z2 = (n2 == -INFINITY) ? 0.f : n2;

#pragma unroll
          for (int e = 0; e < 8; e++) {
            s0[m][e] = ex2_fast(s0[m][e] - ((e & 2) ? z2 : z0));
          }
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[m][j][e] *= (e & 2) ? a2 : a0;
#pragma unroll
          for (int e = 0; e < 8; e++) {
            s1[m][e] = ex2_fast(s1[m][e] - ((e & 2) ? z2 : z0));
          }
#pragma unroll
          for (int j = 2; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[m][j][e] *= (e & 2) ? a2 : a0;

          l0[m] = l0[m] * a0 + ((s0[m][0] + s0[m][1]) + (s0[m][4] + s0[m][5])) +
                               ((s1[m][0] + s1[m][1]) + (s1[m][4] + s1[m][5]));
          l2[m] = l2[m] * a2 + ((s0[m][2] + s0[m][3]) + (s0[m][6] + s0[m][7])) +
                               ((s1[m][2] + s1[m][3]) + (s1[m][6] + s1[m][7]));
          m0[m] = n0;
          m2[m] = n2;

          __half* Pw = Ps + (mw * 16 + m * 8) * PS;
          const int pc = 8 * qp + c0;
          *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s0[m][0], s0[m][1]);
          *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s0[m][4], s0[m][5]);
          *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s0[m][2], s0[m][3]);
          *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s0[m][6], s0[m][7]);
          *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 32) = __floats2half2_rn(s1[m][0], s1[m][1]);
          *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 36) = __floats2half2_rn(s1[m][4], s1[m][5]);
          *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 32) = __floats2half2_rn(s1[m][2], s1[m][3]);
          *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 36) = __floats2half2_rn(s1[m][6], s1[m][7]);
        }
      } else {
        // BN == 32
        float s0[2][8];
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
          for (int e = 0; e < 8; e++) s0[m][e] = 0.f;

#pragma unroll 2
        for (int k8 = 0; k8 < 16; k8++) {
          const int c_base = k8 ^ ((k8 >> 3) & 1);
          const uint4 q0 = Qs_u4[qrow0 * 16 + (c_base ^ arow)];
          const uint4 q1 = Qs_u4[qrow1 * 16 + (c_base ^ arow)];
          const uint4 k0 = KVs_u4[ktok * 16 + (c_base ^ k_swiz_base)];
          mma_rc(s0[0], q0.x, q0.y, k0.x, k0.y);
          mma_rc(s0[1], q1.x, q1.y, k0.x, k0.y);
          mma_rc(s0[0], q0.z, q0.w, k0.z, k0.w);
          mma_rc(s0[1], q1.z, q1.w, k0.z, k0.w);
        }

        if (all_far_past) {
#pragma unroll
          for (int m = 0; m < 2; m++)
#pragma unroll
            for (int e = 0; e < 8; e++) s0[m][e] *= sl2;
        } else if (all_in_ext) {
          const __half* pA0 = RelPtrs[lrA0] + rel_off0;
          const __half* pB0 = RelPtrs[lrB0] + rel_off0;
          const __half* pA1 = RelPtrs[lrA1] + rel_off0;
          const __half* pB1 = RelPtrs[lrB1] + rel_off0;
          const unsigned u_a0_0 = ldg_u32(pA0);
          const unsigned u_a0_4 = ldg_u32(pA0 + 4);
          const unsigned u_b0_0 = ldg_u32(pB0);
          const unsigned u_b0_4 = ldg_u32(pB0 + 4);
          const unsigned u_a1_0 = ldg_u32(pA1);
          const unsigned u_a1_4 = ldg_u32(pA1 + 4);
          const unsigned u_b1_0 = ldg_u32(pB1);
          const unsigned u_b1_4 = ldg_u32(pB1 + 4);
          auto add_rel = [&](float* sc, unsigned u01, unsigned u45, unsigned v01, unsigned v45) {
            const float2 f01 = u32_to_f2(u01);
            const float2 f45 = u32_to_f2(u45);
            const float2 g01 = u32_to_f2(v01);
            const float2 g45 = u32_to_f2(v45);
            sc[0] = fmaf(f01.x, LOG2E, sc[0] * sl2);
            sc[1] = fmaf(f01.y, LOG2E, sc[1] * sl2);
            sc[4] = fmaf(f45.x, LOG2E, sc[4] * sl2);
            sc[5] = fmaf(f45.y, LOG2E, sc[5] * sl2);
            sc[2] = fmaf(g01.x, LOG2E, sc[2] * sl2);
            sc[3] = fmaf(g01.y, LOG2E, sc[3] * sl2);
            sc[6] = fmaf(g45.x, LOG2E, sc[6] * sl2);
            sc[7] = fmaf(g45.y, LOG2E, sc[7] * sl2);
          };
          add_rel(s0[0], u_a0_0, u_a0_4, u_b0_0, u_b0_4);
          add_rel(s0[1], u_a1_0, u_a1_4, u_b1_0, u_b1_4);
        } else {
          apply_bias_and_mask_32(
              s0[0], RelPtrs[lrA0], RelPtrs[lrB0], validA0, validB0, posA0, posB0, rel_base,
              tok0, cb0, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
          apply_bias_and_mask_32(
              s0[1], RelPtrs[lrA1], RelPtrs[lrB1], validA1, validB1, posA1, posB1, rel_base,
              tok0, cb0, seqlen_k, window, E, max_ext_dist, sl2, rel_aligned4);
        }

#pragma unroll
        for (int m = 0; m < 2; m++) {
          float x0 = fmaxf(fmaxf(s0[m][0], s0[m][1]), fmaxf(s0[m][4], s0[m][5]));
          float x2 = fmaxf(fmaxf(s0[m][2], s0[m][3]), fmaxf(s0[m][6], s0[m][7]));
#pragma unroll
          for (int d = 2; d <= 8; d <<= 1) {
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
            x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
          }
          const float n0 = fmaxf(m0[m], x0);
          const float n2 = fmaxf(m2[m], x2);
          const float a0 = (m0[m] == -INFINITY) ? 0.f : ex2_fast(m0[m] - n0);
          const float a2 = (m2[m] == -INFINITY) ? 0.f : ex2_fast(m2[m] - n2);
          const float z0 = (n0 == -INFINITY) ? 0.f : n0;
          const float z2 = (n2 == -INFINITY) ? 0.f : n2;

#pragma unroll
          for (int e = 0; e < 8; e++) {
            s0[m][e] = ex2_fast(s0[m][e] - ((e & 2) ? z2 : z0));
          }
          l0[m] = l0[m] * a0 + (s0[m][0] + s0[m][1]) + (s0[m][4] + s0[m][5]);
          l2[m] = l2[m] * a2 + (s0[m][2] + s0[m][3]) + (s0[m][6] + s0[m][7]);
          m0[m] = n0;
          m2[m] = n2;

#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[m][j][e] *= (e & 2) ? a2 : a0;

          __half* Pw = Ps + (mw * 16 + m * 8) * PS;
          const int pc = 8 * qp + c0;
          *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(s0[m][0], s0[m][1]);
          *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(s0[m][4], s0[m][5]);
          *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(s0[m][2], s0[m][3]);
          *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(s0[m][6], s0[m][7]);
        }
      }
    }

    __syncthreads();
    commit_or_load_kv(vc_base_b, vb_b, vt_b, tok0);
    __syncthreads();

    if (more) {
      prefetch_kv(kc_base_b, kb_b, kt_b, tok0 + BN);
    }

    if (warp_active) {
      const __half* Pw0 = Ps + (mw * 16 + arow) * PS;
      const __half* Pw1 = Ps + (mw * 16 + 8 + arow) * PS;
#pragma unroll 4
      for (int i2 = 0; i2 < BN / 8; i2++) {
        const uint4 p0 = *reinterpret_cast<const uint4*>(Pw0 + 8 * i2);
        const uint4 p1 = *reinterpret_cast<const uint4*>(Pw1 + 8 * i2);
        const int vsw = l3 | ((i2 & 1) << 2);
        {
          const uint4* Vr_u4 = KVs_u4 + (8 * i2 + l3) * 16;
          const uint4 v0 = Vr_u4[vc0_f ^ vsw];
          const uint4 v1 = Vr_u4[vc1_f ^ vsw];
          mma_rr(o[0][0], p0.x, p0.y, v0.x, v0.y);
          mma_rr(o[1][0], p1.x, p1.y, v0.x, v0.y);
          mma_rr(o[0][1], p0.x, p0.y, v0.z, v0.w);
          mma_rr(o[1][1], p1.x, p1.y, v0.z, v0.w);
          mma_rr(o[0][2], p0.x, p0.y, v1.x, v1.y);
          mma_rr(o[1][2], p1.x, p1.y, v1.x, v1.y);
          mma_rr(o[0][3], p0.x, p0.y, v1.z, v1.w);
          mma_rr(o[1][3], p1.x, p1.y, v1.z, v1.w);
        }
        {
          const uint4* Vr_u4 = KVs_u4 + (8 * i2 + 4 + l3) * 16;
          const uint4 v0 = Vr_u4[vc0_f ^ vsw];
          const uint4 v1 = Vr_u4[vc1_f ^ vsw];
          mma_rr(o[0][0], p0.z, p0.w, v0.x, v0.y);
          mma_rr(o[1][0], p1.z, p1.w, v0.x, v0.y);
          mma_rr(o[0][1], p0.z, p0.w, v0.z, v0.w);
          mma_rr(o[1][1], p1.z, p1.w, v0.z, v0.w);
          mma_rr(o[0][2], p0.z, p0.w, v1.x, v1.y);
          mma_rr(o[1][2], p1.z, p1.w, v1.x, v1.y);
          mma_rr(o[0][3], p0.z, p0.w, v1.z, v1.w);
          mma_rr(o[1][3], p1.z, p1.w, v1.z, v1.w);
        }
      }
    }

    if (more) {
      __syncthreads();
    }
  }

  // Finalize denominators & fused gate_sink epilogue for both strips.
#pragma unroll
  for (int m = 0; m < 2; m++) {
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      l0[m] += __shfl_xor_sync(0xffffffffu, l0[m], d);
      l2[m] += __shfl_xor_sync(0xffffffffu, l2[m], d);
    }
    const bool vA = m ? validA1 : validA0;
    const bool vB = m ? validB1 : validB0;
    const int tA = m ? tokA1 : tokA0;
    const int tB = m ? tokB1 : tokB0;
    float rs0 = 0.f, rs2 = 0.f;
    if (vA && l0[m] > 0.f) {
      const int tokenA = q_start + tA;
      const float lse_nat = (m0[m] + log2f(l0[m])) * LN2;
      const float sink_v = __half2float(sinks[(size_t)tokenA * (size_t)stride_sink + head]);
      rs0 = (vsc / l0[m]) * sigmoid_f32(lse_nat - sink_v);
    }
    if (vB && l2[m] > 0.f) {
      const int tokenB = q_start + tB;
      const float lse_nat = (m2[m] + log2f(l2[m])) * LN2;
      const float sink_v = __half2float(sinks[(size_t)tokenB * (size_t)stride_sink + head]);
      rs2 = (vsc / l2[m]) * sigmoid_f32(lse_nat - sink_v);
    }
#pragma unroll
    for (int sr = 0; sr < 2; sr++) {
      const bool valid = sr ? vB : vA;
      if (valid) {
        const float rsc = sr ? rs2 : rs0;
        const int token = q_start + (sr ? tB : tA);
        const __half* gp = gate + (size_t)token * (size_t)stride_gate + (size_t)head * HD;
        __half* ob = out + ((size_t)token * H_q + head) * HD;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int dlo = 32 * qp + 4 * j + c0;
          const float2 g_lo = ld_h2_as_f2(gp + dlo, gate_aligned4);
          const float2 g_hi = ld_h2_as_f2(gp + dlo + 16, gate_aligned4);
          const float y0 = o[m][j][2 * sr] * rsc * sigmoid_f32(g_lo.x);
          const float y1 = o[m][j][2 * sr + 1] * rsc * sigmoid_f32(g_lo.y);
          const float y2 = o[m][j][4 + 2 * sr] * rsc * sigmoid_f32(g_hi.x);
          const float y3 = o[m][j][5 + 2 * sr] * rsc * sigmoid_f32(g_hi.y);
          *reinterpret_cast<__half2*>(ob + dlo) = __floats2half2_rn(y0, y1);
          *reinterpret_cast<__half2*>(ob + dlo + 16) = __floats2half2_rn(y2, y3);
        }
      }
    }
  }
}

template <bool IS_FP8, int BM_WARPS, int BN>
void dispatch_pf_2x8(
    const __half* q_ptr,
    const void* kc_ptr,
    const void* vc_ptr,
    const __half* rel_ptr,
    const __half* gate_ptr,
    const __half* sinks_ptr,
    __half* out_ptr,
    const int* bt_ptr,
    const int* seq_ptr,
    const int* cu_q_ptr,
    int B,
    int H_q,
    int G,
    int E,
    int E_pad,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int max_query_len,
    int num_actual_tokens,
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
  constexpr int BM = 16 * BM_WARPS;
  constexpr int smem = PfSM2x8<BM_WARPS, BN>::TOTAL;
  static bool carveout_set[16] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 0 && dev < 16 && !carveout_set[dev]) {
    cudaFuncSetAttribute(
        k_rel_pf_2x8<IS_FP8, BM_WARPS, BN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    carveout_set[dev] = true;
  }

  const int num_m_tiles = (max_query_len + BM - 1) / BM;
  dim3 grid(H_q, num_m_tiles, B);
  k_rel_pf_2x8<IS_FP8, BM_WARPS, BN><<<grid, 32 * BM_WARPS, smem, stream>>>(
      q_ptr, kc_ptr, vc_ptr, rel_ptr, gate_ptr, sinks_ptr, out_ptr,
      bt_ptr, seq_ptr, cu_q_ptr,
      H_q, G, E, E_pad, window, page, page_shift, max_blocks, num_actual_tokens,
      stride_q_t, stride_q_h, kb, kt, kh, vb, vt, vh,
      stride_rel_t, stride_rel_h, stride_bt, stride_gate, stride_sink,
      sl2, vsc, rel_aligned4, gate_aligned4, q_aligned16);
}

std::string get_occupancy_report_impl() {
  std::string out;
  auto rep = [&](const char* name, auto fn, int nt, int smem) {
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    int blks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blks, fn, nt, smem);
    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, fn);
    char buf[256];
    snprintf(buf, sizeof(buf), "  %-28s nt=%3d smem=%5dB regs=%3d blks/SM=%d warps/SM=%2d\n",
             name, nt, smem, attr.numRegs, blks, blks * (nt / 32));
    out += buf;
  };
  rep("cfg1 fp8 1x8 BM16 BN32", k_rel_pf<true, 2, 32>, 64, PfSM<2, 32>::TOTAL);
  rep("cfg2 fp8 1x8 BM32 BN32", k_rel_pf<true, 4, 32>, 128, PfSM<4, 32>::TOTAL);
  rep("cfg3 fp8 1x8 BM32 BN64", k_rel_pf<true, 4, 64>, 128, PfSM<4, 64>::TOTAL);
  rep("cfg4 fp8 1x8 BM64 BN64", k_rel_pf<true, 8, 64>, 256, PfSM<8, 64>::TOTAL);
  rep("cfg5 fp8 2x8 BM64 BN64", k_rel_pf_2x8<true, 4, 64>, 128, PfSM2x8<4, 64>::TOTAL);
  rep("cfg6 fp8 2x8 BM64 BN32", k_rel_pf_2x8<true, 4, 32>, 128, PfSM2x8<4, 32>::TOTAL);
  rep("cfg5 fp16 2x8 BM64 BN64", k_rel_pf_2x8<false, 4, 64>, 128, PfSM2x8<4, 64>::TOTAL);
  rep("cfg6 fp16 2x8 BM64 BN32", k_rel_pf_2x8<false, 4, 32>, 128, PfSM2x8<4, 32>::TOTAL);
  return out;
}

}  // namespace

void attn_rel_pf_forward(
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
    int64_t max_query_len,
    int64_t num_actual_tokens,
    double softmax_scale,
    int64_t window,
    int64_t config_override) {
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

  TORCH_CHECK(rel_logits.dtype() == torch::kHalf && rel_logits.dim() == 3 && rel_logits.stride(2) == 1,
              "rel_logits must be FP16 [>=T, H_q, E+256] with stride(2)==1");
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
  TORCH_CHECK(window >= 0, "window must be >= 0");

  const int act_toks = (int)std::max<int64_t>(0, std::min<int64_t>(num_actual_tokens, num_tokens));
  const at::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (act_toks == 0 || B == 0 || max_query_len <= 0) {
    TORCH_CHECK(cudaMemsetAsync(out.data_ptr(), 0, (size_t)num_tokens * H_q * HD * sizeof(__half), stream) ==
                cudaSuccess);
    return;
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
  const int64_t kv_align_mask = is_fp8 ? 15 : 7;
  const bool kc_aligned16 = (((uintptr_t)kc_ptr & 15u) == 0u) &&
                            ((kb & kv_align_mask) == 0) && ((kt & kv_align_mask) == 0) && ((kh & kv_align_mask) == 0);
  const bool vc_aligned16 = (((uintptr_t)vc_ptr & 15u) == 0u) &&
                            ((vb & kv_align_mask) == 0) && ((vt & kv_align_mask) == 0) && ((vh & kv_align_mask) == 0);
  TORCH_CHECK(kc_aligned16 && vc_aligned16, "key_cache and value_cache must be 16-byte aligned");

  const float sl2 = (float)(softmax_scale * (is_fp8 ? 256.0 : 1.0) * (double)LOG2E);
  const float vsc = is_fp8 ? 256.0f : 1.0f;
  const int max_blocks = (int)block_table.size(1);
  const int max_q = (int)max_query_len;

  // Config selection:
  //   1: (BM=16, BN=32, 2 warps, 1x8) - tiny q_len (< 64)
  //   2: (BM=32, BN=32, 4 warps, 1x8) - small-medium FP16 prefill
  //   3: (BM=32, BN=64, 4 warps, 1x8) - small-medium FP8 prefill (q_len=128..512, 3 CTAs/SM)
  //   4: (BM=64, BN=64, 8 warps, 1x8) - 16 warps/SM fallback
  //   5: (BM=64, BN=64, 4 warps, 2x8) - primary workhorse (16 rows/warp, 100% HMMA-bound + rel prefetch)
  //   6: (BM=64, BN=32, 4 warps, 2x8) - 3 CTAs/SM (240 CTAs/wave) 16 rows/warp variant
  int cfg = (int)config_override;
  if (cfg < 1 || cfg > 6) {
    const int ctas_bm64 = H_q * B * ((max_q + 63) / 64);
    const int ctas_bm32 = H_q * B * ((max_q + 31) / 32);
    if (ctas_bm64 >= 96) {
      cfg = 5;
    } else if (ctas_bm32 >= 24) {
      cfg = is_fp8 ? 3 : 2;
    } else {
      cfg = 1;
    }
  }

#define LAUNCH_PF(FP8_B, MW_I, BN_I)                                                                          \
  dispatch_pf<FP8_B, MW_I, BN_I>(                                                                             \
      q_ptr, kc_ptr, vc_ptr, rel_ptr, gate_ptr, sinks_ptr, out_ptr,                                           \
      block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(),               \
      B, H_q, G, E, E_pad, (int)window, page, page_shift, max_blocks, max_q, act_toks,                        \
      stride_q_t, stride_q_h, kb, kt, kh, vb, vt, vh,                                                         \
      stride_rel_t, stride_rel_h, stride_bt, stride_gate, stride_sink,                                        \
      sl2, vsc, rel_aligned4, gate_aligned4, q_aligned16, stream)

#define LAUNCH_PF2(FP8_B, MW_I, BN_I)                                                                         \
  dispatch_pf_2x8<FP8_B, MW_I, BN_I>(                                                                         \
      q_ptr, kc_ptr, vc_ptr, rel_ptr, gate_ptr, sinks_ptr, out_ptr,                                           \
      block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(),               \
      B, H_q, G, E, E_pad, (int)window, page, page_shift, max_blocks, max_q, act_toks,                        \
      stride_q_t, stride_q_h, kb, kt, kh, vb, vt, vh,                                                         \
      stride_rel_t, stride_rel_h, stride_bt, stride_gate, stride_sink,                                        \
      sl2, vsc, rel_aligned4, gate_aligned4, q_aligned16, stream)

  if (is_fp8) {
    if (cfg == 1) { LAUNCH_PF(true, 2, 32); }
    else if (cfg == 2) { LAUNCH_PF(true, 4, 32); }
    else if (cfg == 3) { LAUNCH_PF(true, 4, 64); }
    else if (cfg == 4) { LAUNCH_PF(true, 8, 64); }
    else if (cfg == 5) { LAUNCH_PF2(true, 4, 64); }
    else { LAUNCH_PF2(true, 4, 32); }
  } else {
    if (cfg == 1) { LAUNCH_PF(false, 2, 32); }
    else if (cfg == 2) { LAUNCH_PF(false, 4, 32); }
    else if (cfg == 3) { LAUNCH_PF(false, 4, 64); }
    else if (cfg == 4) { LAUNCH_PF(false, 8, 64); }
    else if (cfg == 5) { LAUNCH_PF2(false, 4, 64); }
    else { LAUNCH_PF2(false, 4, 32); }
  }
#undef LAUNCH_PF
#undef LAUNCH_PF2

  if (act_toks < num_tokens) {
    TORCH_CHECK(cudaMemsetAsync(
                    out_ptr + (size_t)act_toks * H_q * HD, 0,
                    (size_t)(num_tokens - act_toks) * H_q * HD * sizeof(__half), stream) == cudaSuccess);
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_rel_pf_forward", &attn_rel_pf_forward,
        "Volta sm_70 HMMA.884 relative-bias GQA prefill attention (FP8/FP16 KV, fused gate_sink)");
  m.def("get_occupancy_report", &get_occupancy_report_impl,
        "Return Volta SM occupancy and SMEM report for all compiled kernels");
}
