// SPDX-License-Identifier: Apache-2.0
// attn_rel_pf_wmma.cu - Volta (sm_70) 3-head-shared WMMA m16n16k16 prefill attention
// kernel with on-the-fly Tensor-Core `R @ rel_proj` and separate paged FP16 / FP8 E4M3FN
// KV caches for Inkling-2 (`Zeagarden/packages/inkling-v100`).
//
// Design highlights:
//   - 1 CTA per (16 * NTT query tokens, KV head, request); each warp owns 16 consecutive
//     query tokens of 1 of the 3 query heads sharing the KV head (`3 * NTT` warps/CTA),
//     so every staged K/V tile serves `3 * NTT` warps.
//   - Longest-first reversed `blockIdx.x` scheduling (`m_tile = req_m_tiles - 1 - blockIdx.x`)
//     for optimal causal wave load balancing.
//   - Separate `key_cache` and `value_cache` tensors `[num_blocks, page, H_kv, 128]` with
//     explicit element strides `(kb, kt, kh)` and `(vb, vt, vh)`.
//   - For `IS_FP8 = true`: 128-bit (`uint4`) coalesced global loads (16 FP8 bytes/thread)
//     unpacked via exact `e4m3x4` into FP16 (`E4M3 * 2^-8`) directly into `sK` / `sV` with
//     zero extra multiplies during staging; the `2^-8` scale is folded into
//     `qk_scale = (scale * 256.0f) * LOG2E` and epilogue `vsc / inv_l` (`vsc = 256.0f`).
//   - On-the-fly relative bias `fp16(r_i . rel_proj[:, pos_q - pos_k])` computed per KV
//     tile on Tensor Cores (`16x16` R fragment x `NBF` `16x16` windows of `rel_proj`),
//     eliminating global sheared `rel_logits` materialization and bandwidth.
//   - Fused output epilogue: `o * (vsc / l) * sigmoid(gate) * sigmoid(lse - sink)` written
//     directly to FP16 `[T, H * 128]`.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int D = 128;
constexpr int G = 3;         // query heads per KV head
constexpr int D_REL = 16;    // relative projection inner dimension
constexpr int LDK = D + 8;   // smem pitch (halves) of K / V tiles (272B, 16B-aligned)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

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

// Fast Volta SFU exp2f (MUFU.EX2, ex2(-inf) == +0.0f).
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigm(float x) {
  return 1.0f / (1.0f + ex2(-x * LOG2E));
}

__device__ __forceinline__ float2 ld_h2_as_f2(const __half* p, bool aligned4) {
  if (aligned4) {
    unsigned u;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(u) : "l"(p));
    return __half22float2(*reinterpret_cast<const __half2*>(&u));
  }
  return make_float2(__half2float(p[0]), __half2float(p[1]));
}

// Volta (sm_70) WMMA m16n16k16 FP32 accumulator layout:
// maps (lane in 0..31, e in 0..7) -> (row in 0..15, col in 0..15).
__device__ __forceinline__ int acc_row(int lane, int e) {
  return (lane & 1) + 2 * ((e >> 1) & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);
}

__device__ __forceinline__ int acc_col(int lane, int e) {
  return (e & 1) + 2 * ((lane >> 1) & 1) + 4 * ((e >> 2) & 1) + 8 * ((lane >> 3) & 1);
}

// Shared-memory layout shared between host launch and device kernel.
template <int NTT, int BN>
struct WmmaSmem {
  static constexpr int NW = 3 * NTT;
  static constexpr int NT = 32 * NW;
  static constexpr int BM = 16 * NTT;
  static constexpr int NF = BN / 16;
  static constexpr int LDS = BN + 4;         // fp32 S pitch (multiple of 4 floats = 16B)
  static constexpr int LDP = BN + 8;         // half P pitch (multiple of 8 halves = 16B)
  static constexpr int NBF = (BN + 45) / 16; // 16-wide rel_proj column tiles per KV tile
  static constexpr int LDM = NBF * 16 + 4;   // fp32 M pitch (multiple of 4 floats = 16B)

  static constexpr size_t K_OFF = 0;
  static constexpr size_t K_BYTES = (size_t)BN * LDK * sizeof(__half);
  static constexpr size_t V_OFF = K_OFF + K_BYTES;
  static constexpr size_t V_BYTES = (size_t)BN * LDK * sizeof(__half);
  static constexpr size_t S_OFF = V_OFF + V_BYTES;
  static constexpr size_t S_BYTES = (size_t)NW * 16 * LDS * sizeof(float);
  static constexpr size_t P_OFF = S_OFF + S_BYTES;
  static constexpr size_t P_BYTES = (size_t)NW * 16 * LDP * sizeof(__half);
  static constexpr size_t AUX_OFF = P_OFF + P_BYTES;
  static constexpr size_t AUX_BYTES = (size_t)NW * 48 * sizeof(float);
  static constexpr size_t M_OFF = AUX_OFF + AUX_BYTES;
  static constexpr size_t M_BYTES = (size_t)NW * 16 * LDM * sizeof(float);
  static constexpr size_t R_OFF = M_OFF + M_BYTES;
  static constexpr size_t R_BYTES = (size_t)NW * 16 * 24 * sizeof(__half);
  static constexpr size_t TOTAL = R_OFF + R_BYTES;

  static_assert(K_OFF % 16 == 0, "sK must be 16B aligned");
  static_assert(V_OFF % 16 == 0, "sV must be 16B aligned");
  static_assert(S_OFF % 16 == 0, "sS must be 16B aligned");
  static_assert(P_OFF % 16 == 0, "sP must be 16B aligned");
  static_assert(AUX_OFF % 16 == 0, "sAux must be 16B aligned");
  static_assert(M_OFF % 16 == 0, "sM must be 16B aligned");
  static_assert(R_OFF % 16 == 0, "sR must be 16B aligned");
  static_assert(TOTAL <= 96 * 1024, "SMEM exceeds Volta 96 KB limit");
};

template <bool IS_FP8, int NTT, int BN, int MINB>
__global__ void __launch_bounds__(96 * NTT, MINB) prefill_attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ Rr,
    const __half* __restrict__ Proj,
    const void* __restrict__ kc_raw,
    const void* __restrict__ vc_raw,
    const int* __restrict__ BT,
    const int* __restrict__ SeqLens,
    const int* __restrict__ CuQ,
    const __half* __restrict__ Gate,
    const __half* __restrict__ Sink,
    __half* __restrict__ Out,
    int H,
    int64_t stride_q_t,
    int64_t stride_r_t,
    int64_t kb,
    int64_t kt,
    int64_t kh,
    int64_t vb,
    int64_t vt,
    int64_t vh,
    int64_t stride_bt,
    int64_t stride_gate,
    int64_t stride_sink,
    int page,
    int page_shift,
    int max_blocks,
    int num_actual_tokens,
    int E,
    int window,
    float scale,
    bool gate_aligned4) {
  using SM = WmmaSmem<NTT, BN>;
  constexpr int NT = SM::NT;
  constexpr int BM = SM::BM;
  constexpr int NF = SM::NF;
  constexpr int LDS = SM::LDS;
  constexpr int LDP = SM::LDP;
  constexpr int NBF = SM::NBF;
  constexpr int LDM = SM::LDM;

  // 16-byte global loads per KV row (K + V):
  //   IS_FP8 = true:  8 uint4 (128B) for K + 8 uint4 (128B) for V = 16 chunks/row
  //   IS_FP8 = false: 16 uint4 (256B) for K + 16 uint4 (256B) for V = 32 chunks/row
  constexpr int CHUNKS_PER_ROW = IS_FP8 ? 16 : 32;
  constexpr int TOTAL_CHUNKS = BN * CHUNKS_PER_ROW;
  constexpr int NLD = (TOTAL_CHUNKS + NT - 1) / NT;
  constexpr float VSC = IS_FP8 ? 256.0f : 1.0f;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* sK = reinterpret_cast<__half*>(smem_raw + SM::K_OFF);
  __half* sV = reinterpret_cast<__half*>(smem_raw + SM::V_OFF);
  float* sS = reinterpret_cast<float*>(smem_raw + SM::S_OFF);
  __half* sP = reinterpret_cast<__half*>(smem_raw + SM::P_OFF);
  float* sAux = reinterpret_cast<float*>(smem_raw + SM::AUX_OFF);
  float* sM = reinterpret_cast<float*>(smem_raw + SM::M_OFF);
  __half* sR = reinterpret_cast<__half*>(smem_raw + SM::R_OFF);

  const int req = (int)blockIdx.z;
  const int kvh = (int)blockIdx.y;
  const int raw_m = (int)blockIdx.x;

  const int q_start = CuQ[req];
  const int q_len = CuQ[req + 1] - q_start;
  const int eff_q_len = max(0, min(q_len, num_actual_tokens - q_start));
  const int req_m_tiles = (eff_q_len + BM - 1) / BM;
  if (raw_m >= req_m_tiles) return;

  // Longest-first reversed scheduling so causal tail tiles launch first.
  const int m_tile = req_m_tiles - 1 - raw_m;
  const int first_tok = m_tile * BM;

  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = warp % G;
  const int wt = warp / G;
  const int tok0 = first_tok + wt * 16;
  const int head = kvh * G + g;
  const int64_t qrow = (int64_t)H * D;
  const bool warp_live = tok0 < eff_q_len;

  const int seqlen_k = SeqLens[req];
  const int past = seqlen_k - q_len;

  // Key tile range of the CTA.
  const int pos_lo = past + first_tok;
  const int pos_hi = past + min(first_tok + BM, eff_q_len) - 1;
  const int k_lo = (window > 0) ? max(pos_lo - (window - 1), 0) : 0;
  const int k_hi = min(pos_hi + 1, seqlen_k);
  const int n_lo = k_lo / BN;
  const int n_hi = (k_hi > 0) ? ((k_hi + BN - 1) / BN) : 0;

  // Cleanly zero outputs when no KV tiles overlap this CTA's query range.
  if (k_hi <= k_lo || n_hi <= n_lo) {
    if (warp_live) {
      const int n_rows = min(16, eff_q_len - tok0);
      for (int row = 0; row < n_rows; ++row) {
        const int64_t base = (int64_t)(q_start + tok0 + row) * qrow + (int64_t)head * D;
        *reinterpret_cast<uint32_t*>(Out + base + lane * 2) = 0u;
        *reinterpret_cast<uint32_t*>(Out + base + (lane + 32) * 2) = 0u;
      }
    }
    return;
  }

  float* sSw = sS + warp * 16 * LDS;
  __half* sPw = sP + warp * 16 * LDP;
  float* sAlpha = sAux + warp * 48;
  float* sL = sAlpha + 16;
  float* sSink = sAlpha + 32;
  float* sMw = sM + warp * 16 * LDM;
  __half* sRw = sR + warp * 16 * 24;

  // ---- Load Q fragments (16 consecutive tokens of one head) and stage R into sRw.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa[D / 16];
  if (warp_live) {
    const int row = lane >> 1;
    const int hf = lane & 1;
    const int tok = tok0 + row;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (tok < eff_q_len) {
        v = ldg16(Q + (int64_t)(q_start + tok) * stride_q_t + (int64_t)head * D +
                  kk * 16 + hf * 8);
      }
      *reinterpret_cast<uint4*>(sPw + row * 24 + hf * 8) = v;
      __syncwarp();
      wmma::load_matrix_sync(qa[kk], sPw, 24);
      __syncwarp();
    }
    {
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (tok < eff_q_len) {
        v = ldg16(Rr + (int64_t)(q_start + tok) * stride_r_t + (int64_t)head * D_REL + hf * 8);
      }
      *reinterpret_cast<uint4*>(sRw + row * 24 + hf * 8) = v;
      __syncwarp();
    }
  }

  // Per-warp query position bounds.
  const int wpos_lo = past + tok0;
  const int wpos_hi = past + min(tok0 + 15, eff_q_len - 1);
  const float qk_scale = (IS_FP8 ? (scale * 256.0f) : scale) * LOG2E;

  // Softmax lane ownership: pair (srow = lane >> 1, shalf = lane & 1) owns row `srow`,
  // interleaved in float4 chunks (2*q + shalf) * 4.
  const int srow = lane >> 1;
  const int shalf = lane & 1;
  const int stok = tok0 + srow;
  const int spos = past + stok;
  const bool rvalid = stok < eff_q_len;
  float m_reg = -INFINITY;
  float l_reg = 0.f;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[D / 16];
#pragma unroll
  for (int f = 0; f < D / 16; ++f) wmma::fill_fragment(acc[f], 0.f);

  const int* bt = BT + (int64_t)req * stride_bt;
  const uint8_t* kc_u8 = reinterpret_cast<const uint8_t*>(kc_raw);
  const uint8_t* vc_u8 = reinterpret_cast<const uint8_t*>(vc_raw);
  const __half* kc_h = reinterpret_cast<const __half*>(kc_raw);
  const __half* vc_h = reinterpret_cast<const __half*>(vc_raw);

  uint4 buf[NLD];
  auto load_tile = [&](int n) {
#pragma unroll
    for (int it = 0; it < NLD; ++it) {
      const int idx = it * NT + tid;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (idx < TOTAL_CHUNKS) {
        const int row = idx / CHUNKS_PER_ROW;
        const int c = idx % CHUNKS_PER_ROW;
        const int key = n * BN + row;
        bool ok = key < k_hi;
        if (window > 0) ok = ok && key > pos_lo - window;
        if (ok) {
          const int blk_idx = (page_shift >= 0) ? (key >> page_shift) : (key / page);
          const int slot = (page_shift >= 0) ? (key & (page - 1)) : (key % page);
          if (blk_idx < max_blocks) {
            const int64_t pg = bt[blk_idx];
            if constexpr (IS_FP8) {
              const bool is_v = (c >= 8);
              const int c_off = (c & 7) * 16;
              const uint8_t* src = is_v
                  ? (vc_u8 + pg * vb + (int64_t)slot * vt + (int64_t)kvh * vh + c_off)
                  : (kc_u8 + pg * kb + (int64_t)slot * kt + (int64_t)kvh * kh + c_off);
              v = ldg16(src);
            } else {
              const bool is_v = (c >= 16);
              const int c_off = (c & 15) * 8;
              const __half* src = is_v
                  ? (vc_h + pg * vb + (int64_t)slot * vt + (int64_t)kvh * vh + c_off)
                  : (kc_h + pg * kb + (int64_t)slot * kt + (int64_t)kvh * kh + c_off);
              v = ldg16(src);
            }
          }
        }
      }
      buf[it] = v;
    }
  };

  auto store_tile = [&]() {
#pragma unroll
    for (int it = 0; it < NLD; ++it) {
      const int idx = it * NT + tid;
      if (idx < TOTAL_CHUNKS) {
        const int row = idx / CHUNKS_PER_ROW;
        const int c = idx % CHUNKS_PER_ROW;
        if constexpr (IS_FP8) {
          const bool is_v = (c >= 8);
          const int col = (c & 7) * 16;
          __half* dst = (is_v ? sV : sK) + row * LDK + col;
          const uint4 w = buf[it];
          uint4 h0, h1;
          e4m3x4(w.x, h0.x, h0.y);
          e4m3x4(w.y, h0.z, h0.w);
          e4m3x4(w.z, h1.x, h1.y);
          e4m3x4(w.w, h1.z, h1.w);
          *reinterpret_cast<uint4*>(dst) = h0;
          *reinterpret_cast<uint4*>(dst + 8) = h1;
        } else {
          const bool is_v = (c >= 16);
          const int col = (c & 15) * 8;
          __half* dst = (is_v ? sV : sK) + row * LDK + col;
          *reinterpret_cast<uint4*>(dst) = buf[it];
        }
      }
    }
  };

  load_tile(n_lo);
  for (int n = n_lo; n < n_hi; ++n) {
    __syncthreads();
    store_tile();
    __syncthreads();
    if (n + 1 < n_hi) load_tile(n + 1);

    const int kbase = n * BN;
    const bool live = warp_live && kbase <= wpos_hi &&
                      (window == 0 || kbase + BN - 1 > wpos_lo - window);
    if (live) {
      // ---- S = Q K^T on Tensor Cores.
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[NF];
#pragma unroll
      for (int nf = 0; nf < NF; ++nf) wmma::fill_fragment(sf[nf], 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
        for (int nf = 0; nf < NF; ++nf) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb_frag;
          wmma::load_matrix_sync(kb_frag, sK + nf * 16 * LDK + kk * 16, LDK);
          wmma::mma_sync(sf[nf], qa[kk], kb_frag, sf[nf]);
        }
      }
#pragma unroll
      for (int nf = 0; nf < NF; ++nf) {
        wmma::store_matrix_sync(sSw + nf * 16, sf[nf], LDS, wmma::mem_row_major);
      }
      __syncwarp();

      // Warp-uniform tile classification:
      //   no_bias  -> interior tile, all distances >= E (no mask, no bias)
      //   all_bias -> interior tile, all distances < E (no mask, bias on all entries)
      //   general  -> boundary tile (causal / SWA mask + bias where dist < E)
      const bool interior = (kbase + BN - 1 <= wpos_lo) && (kbase + BN <= seqlen_k) &&
                            (window == 0 || wpos_hi - kbase < window);
      const bool no_bias = interior && (wpos_lo - (kbase + BN - 1) >= E);
      const bool all_bias = interior && (wpos_hi - kbase < E);

      float s[BN / 2];
#pragma unroll
      for (int q = 0; q < BN / 8; ++q) {
        const float4 v = *reinterpret_cast<const float4*>(sSw + srow * LDS + (2 * q + shalf) * 4);
        s[4 * q + 0] = v.x * qk_scale;
        s[4 * q + 1] = v.y * qk_scale;
        s[4 * q + 2] = v.z * qk_scale;
        s[4 * q + 3] = v.w * qk_scale;
      }

      if (!no_bias) {
        // M[i][c] = r_i . rel_proj[:, dstart + c] on Tensor Cores.
        const int dmin = wpos_lo - (kbase + BN - 1);
        const int dstart = min(max(dmin, 0) & ~15, max(E - NBF * 16, 0));
        {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ra;
          wmma::load_matrix_sync(ra, sRw, 24);
#pragma unroll
          for (int nb = 0; nb < NBF; ++nb) {
            const int col_off = dstart + nb * 16;
            if (col_off + 16 <= E) {
              wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> pb;
              wmma::load_matrix_sync(pb, Proj + col_off, E);
              wmma::fragment<wmma::accumulator, 16, 16, 16, float> mf;
              wmma::fill_fragment(mf, 0.f);
              wmma::mma_sync(mf, ra, pb, mf);
              wmma::store_matrix_sync(sMw + nb * 16, mf, LDM, wmma::mem_row_major);
            }
          }
        }
        __syncwarp();

        const int base_idx = spos - kbase - dstart;  // M column corresponding to key `kbase`
        if (all_bias) {
          if (rvalid) {
#pragma unroll
            for (int q = 0; q < BN / 8; ++q) {
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                const int j = (2 * q + shalf) * 4 + u;
                const float b = __half2float(__float2half(sMw[srow * LDM + base_idx - j]));
                s[4 * q + u] += b * LOG2E;
              }
            }
          }
        } else {
#pragma unroll
          for (int q = 0; q < BN / 8; ++q) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const int j = (2 * q + shalf) * 4 + u;
              const int key = kbase + j;
              const int dist = spos - key;
              bool vis = rvalid && dist >= 0 && key < seqlen_k;
              if (window > 0) vis = vis && dist < window;
              float v = s[4 * q + u];
              if (vis && dist < E) {
                const int idx = min(max(base_idx - j, 0), NBF * 16 - 1);
                v += __half2float(__float2half(sMw[srow * LDM + idx])) * LOG2E;
              }
              s[4 * q + u] = vis ? v : -INFINITY;
            }
          }
        }
      }

      // ---- Online softmax in log2 domain.
      float mx = s[0];
#pragma unroll
      for (int j = 1; j < BN / 2; ++j) mx = fmaxf(mx, s[j]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      const float m_new = fmaxf(m_reg, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = ex2(m_reg - m_safe);

      float lsum = 0.f;
#pragma unroll
      for (int q = 0; q < BN / 8; ++q) {
        const float p0 = ex2(s[4 * q + 0] - m_safe);
        const float p1 = ex2(s[4 * q + 1] - m_safe);
        const float p2 = ex2(s[4 * q + 2] - m_safe);
        const float p3 = ex2(s[4 * q + 3] - m_safe);
        lsum += (p0 + p1) + (p2 + p3);
        const __half2 h01 = __floats2half2_rn(p0, p1);
        const __half2 h23 = __floats2half2_rn(p2, p3);
        uint2 raw;
        raw.x = *reinterpret_cast<const unsigned*>(&h01);
        raw.y = *reinterpret_cast<const unsigned*>(&h23);
        *reinterpret_cast<uint2*>(sPw + srow * LDP + (2 * q + shalf) * 4) = raw;
      }
      l_reg = l_reg * alpha + lsum;
      m_reg = m_new;
      if (shalf == 0) sAlpha[srow] = alpha;
      __syncwarp();

      // ---- Rescale O accumulator.
      float al[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) al[e] = sAlpha[acc_row(lane, e)];
#pragma unroll
      for (int f = 0; f < D / 16; ++f) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[f].x[e] *= al[e];
      }

      // ---- O += P V on Tensor Cores.
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
        wmma::load_matrix_sync(pa, sPw + j * 16, LDP);
#pragma unroll
        for (int f = 0; f < D / 16; ++f) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb_frag;
          wmma::load_matrix_sync(vb_frag, sV + j * 16 * LDK + f * 16, LDK);
          wmma::mma_sync(acc[f], pa, vb_frag, acc[f]);
        }
      }
      __syncwarp();
    }
  }

  // ---- Epilogue: o * (VSC / l) * sigmoid(gate) * sigmoid(lse - sink)
  if (warp_live) {
    const float l = l_reg + __shfl_xor_sync(0xffffffffu, l_reg, 1);
    if (shalf == 0) {
      float ss = 0.f;
      if (rvalid && l > 0.f) {
        const float sink = __half2float(Sink[(int64_t)(q_start + stok) * stride_sink + head]);
        const float lse = (m_reg + __log2f(l)) * LN2;
        ss = sigm(lse - sink);
      }
      sL[srow] = (l == 0.f) ? 1.f : l;
      sSink[srow] = ss;
    }
    __syncwarp();
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const int r = acc_row(lane, e);
      const int c = acc_col(lane, e);
      const int tok = tok0 + r;
      if (tok < eff_q_len) {
        const float inv_l = sL[r];
        const float scale_o = VSC / inv_l;
        const float ss = sSink[r];
        const int64_t base = (int64_t)(q_start + tok);
#pragma unroll
        for (int f = 0; f < D / 16; ++f) {
          const int col = f * 16 + c;
          const float2 gv = ld_h2_as_f2(
              Gate + base * stride_gate + (int64_t)head * D + col, gate_aligned4);
          float o0 = acc[f].x[e] * scale_o;
          float o1 = acc[f].x[e + 1] * scale_o;
          o0 = (o0 * sigm(gv.x)) * ss;
          o1 = (o1 * sigm(gv.y)) * ss;
          *reinterpret_cast<__half2*>(Out + base * qrow + (int64_t)head * D + col) =
              __floats2half2_rn(o0, o1);
        }
      }
    }
  }
}

template <bool IS_FP8, int NTT, int BN, int MINB>
void launch_wmma(
    const __half* q_ptr,
    const __half* r_ptr,
    const __half* proj_ptr,
    const void* kc_ptr,
    const void* vc_ptr,
    const int* bt_ptr,
    const int* seq_ptr,
    const int* cu_q_ptr,
    const __half* gate_ptr,
    const __half* sinks_ptr,
    __half* out_ptr,
    int B,
    int H_q,
    int H_kv,
    int64_t stride_q_t,
    int64_t stride_r_t,
    int64_t kb,
    int64_t kt,
    int64_t kh,
    int64_t vb,
    int64_t vt,
    int64_t vh,
    int64_t stride_bt,
    int64_t stride_gate,
    int64_t stride_sink,
    int page,
    int page_shift,
    int max_blocks,
    int max_q,
    int act_toks,
    int E,
    int window,
    float scale,
    bool gate_aligned4,
    cudaStream_t stream) {
  using SM = WmmaSmem<NTT, BN>;
  constexpr size_t smem = SM::TOTAL;
  auto kern = prefill_attn_kernel<IS_FP8, NTT, BN, MINB>;

  // Separate per-instantiation, per-device SMEM attribute cache.
  static bool attr_set[16] = {false};
  int dev = 0;
  cudaGetDevice(&dev);
  const bool cached = (dev >= 0 && dev < 16);
  if (!cached || !attr_set[dev]) {
    cudaError_t err = cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    TORCH_CHECK(err == cudaSuccess,
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", (int)smem,
                ") failed: ", cudaGetErrorString(err));
    err = cudaFuncSetAttribute(
        kern, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    TORCH_CHECK(err == cudaSuccess,
                "cudaFuncSetAttribute(PreferredSharedMemoryCarveout) failed: ",
                cudaGetErrorString(err));
    if (cached) attr_set[dev] = true;
  }

  const unsigned num_m_tiles = (unsigned)((max_q + SM::BM - 1) / SM::BM);
  dim3 grid(num_m_tiles, (unsigned)H_kv, (unsigned)B);
  kern<<<grid, SM::NT, smem, stream>>>(
      q_ptr, r_ptr, proj_ptr, kc_ptr, vc_ptr, bt_ptr, seq_ptr, cu_q_ptr,
      gate_ptr, sinks_ptr, out_ptr,
      H_q, stride_q_t, stride_r_t, kb, kt, kh, vb, vt, vh,
      stride_bt, stride_gate, stride_sink,
      page, page_shift, max_blocks, act_toks, E, window, scale, gate_aligned4);
}

}  // namespace

void attn_rel_pf_wmma_forward(
    torch::Tensor q,
    torch::Tensor r,
    torch::Tensor rel_proj,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
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
    int64_t variant) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kHalf && q.dim() == 3 &&
                  q.size(2) == D && q.stride(2) == 1 && q.stride(1) == D,
              "q must be CUDA FP16 [T, H_q, 128] with stride(1)==128, stride(2)==1");
  const int num_tokens = (int)q.size(0);
  if (num_tokens == 0) return;

  const int H_q = (int)q.size(1);
  TORCH_CHECK(r.is_cuda() && r.dtype() == torch::kHalf && r.dim() == 3 &&
                  r.size(0) == num_tokens && r.size(1) == H_q && r.size(2) == D_REL &&
                  r.stride(2) == 1 && r.stride(1) == D_REL,
              "r must be CUDA FP16 [T, H_q, 16] with stride(1)==16, stride(2)==1");

  TORCH_CHECK(rel_proj.is_cuda() && rel_proj.dtype() == torch::kHalf &&
                  rel_proj.is_contiguous() && rel_proj.dim() == 2 &&
                  rel_proj.size(0) == D_REL,
              "rel_proj must be contiguous CUDA FP16 [16, E]");
  const int E = (int)rel_proj.size(1);
  TORCH_CHECK(E >= 128 && (E & 15) == 0,
              "rel_proj extent E must be >= 128 and a multiple of 16, got ", E);

  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() &&
                  key_cache.dim() == 4 && value_cache.dim() == 4 &&
                  key_cache.sizes() == value_cache.sizes(),
              "key_cache and value_cache must be CUDA 4D tensors with identical shapes "
              "[num_blocks, page, H_kv, 128]");
  TORCH_CHECK(key_cache.size(3) == D && key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "KV cache must have head_dim=128 and stride(3)==1");
  const bool is_fp8 = key_cache.element_size() == 1;
  TORCH_CHECK(is_fp8 || (key_cache.dtype() == torch::kHalf && value_cache.dtype() == torch::kHalf),
              "KV cache must be uint8/float8_e4m3fn or float16");
  TORCH_CHECK(key_cache.element_size() == value_cache.element_size(),
              "key_cache and value_cache element sizes must match");

  const int page = (int)key_cache.size(1);
  const int H_kv = (int)key_cache.size(2);
  TORCH_CHECK(page > 0, "KV cache page size must be > 0");
  TORCH_CHECK(H_kv > 0 && H_q == G * H_kv,
              "attn_rel_pf_wmma requires H_q == 3 * H_kv, got H_q=", H_q, " H_kv=", H_kv);

  TORCH_CHECK(gate.is_cuda() && gate.dtype() == torch::kHalf && gate.dim() == 2 &&
                  gate.size(0) == num_tokens && gate.size(1) == H_q * D && gate.stride(1) == 1,
              "gate must be CUDA FP16 [T, H_q*128] with stride(1)==1");
  TORCH_CHECK(sinks.is_cuda() && sinks.dtype() == torch::kHalf && sinks.dim() == 2 &&
                  sinks.size(0) == num_tokens && sinks.size(1) == H_q && sinks.stride(1) == 1,
              "sinks must be CUDA FP16 [T, H_q] with stride(1)==1");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() &&
                  out.dim() == 2 && out.size(0) == num_tokens && out.size(1) == H_q * D,
              "out must be contiguous CUDA FP16 [T, H_q*128]");

  TORCH_CHECK(cache_seqlens.is_cuda() && cache_seqlens.dtype() == torch::kInt32 &&
                  cache_seqlens.is_contiguous() && cache_seqlens.dim() == 1,
              "cache_seqlens must be contiguous CUDA int32 [B]");
  const int B = (int)cache_seqlens.size(0);
  TORCH_CHECK(block_table.is_cuda() && block_table.dtype() == torch::kInt32 &&
                  block_table.dim() == 2 && block_table.size(0) == B && block_table.stride(1) == 1,
              "block_table must be CUDA int32 [B, max_blocks] with stride(1)==1");
  TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_q.dtype() == torch::kInt32 &&
                  cu_seqlens_q.is_contiguous() && cu_seqlens_q.dim() == 1 &&
                  cu_seqlens_q.size(0) == B + 1,
              "cu_seqlens_q must be contiguous CUDA int32 [B+1]");
  TORCH_CHECK(window >= 0, "window must be >= 0");

  const int act_toks = (int)std::max<int64_t>(0, std::min<int64_t>(num_actual_tokens, num_tokens));
  const at::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (act_toks == 0 || B == 0 || max_query_len <= 0) {
    cudaError_t err = cudaMemsetAsync(
        out.data_ptr(), 0, (size_t)num_tokens * H_q * D * sizeof(__half), stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(err));
    return;
  }

  const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  const __half* r_ptr = reinterpret_cast<const __half*>(r.data_ptr<at::Half>());
  const __half* proj_ptr = reinterpret_cast<const __half*>(rel_proj.data_ptr<at::Half>());
  const void* kc_ptr = key_cache.data_ptr();
  const void* vc_ptr = value_cache.data_ptr();
  const __half* gate_ptr = reinterpret_cast<const __half*>(gate.data_ptr<at::Half>());
  const __half* sinks_ptr = reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>());
  __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

  const int64_t stride_q_t = q.stride(0);
  const int64_t stride_r_t = r.stride(0);
  const int64_t kb = key_cache.stride(0), kt = key_cache.stride(1), kh = key_cache.stride(2);
  const int64_t vb = value_cache.stride(0), vt = value_cache.stride(1), vh = value_cache.stride(2);
  const int64_t stride_bt = block_table.stride(0);
  const int64_t stride_gate = gate.stride(0);
  const int64_t stride_sink = sinks.stride(0);

  TORCH_CHECK((((uintptr_t)q_ptr & 15u) == 0u) && (num_tokens == 1 || (stride_q_t & 7) == 0),
              "q must be 16-byte aligned with stride(0) a multiple of 8");
  TORCH_CHECK((((uintptr_t)r_ptr & 15u) == 0u) && (num_tokens == 1 || (stride_r_t & 7) == 0),
              "r must be 16-byte aligned with stride(0) a multiple of 8");
  TORCH_CHECK(((uintptr_t)proj_ptr & 15u) == 0u, "rel_proj must be 16-byte aligned");
  TORCH_CHECK(((uintptr_t)out_ptr & 3u) == 0u, "out must be 4-byte aligned");

  const int64_t kv_align_mask = is_fp8 ? 15 : 7;
  const bool kc_aligned16 = (((uintptr_t)kc_ptr & 15u) == 0u) &&
                            ((kb & kv_align_mask) == 0) &&
                            ((kt & kv_align_mask) == 0) &&
                            ((kh & kv_align_mask) == 0);
  const bool vc_aligned16 = (((uintptr_t)vc_ptr & 15u) == 0u) &&
                            ((vb & kv_align_mask) == 0) &&
                            ((vt & kv_align_mask) == 0) &&
                            ((vh & kv_align_mask) == 0);
  TORCH_CHECK(kc_aligned16 && vc_aligned16,
              "key_cache and value_cache must be 16-byte aligned");

  const bool gate_aligned4 = (((uintptr_t)gate_ptr & 3u) == 0u) && ((stride_gate & 1) == 0);

  int page_shift = -1;
  if ((page & (page - 1)) == 0) {
    page_shift = 0;
    int tmp = page;
    while (tmp > 1) {
      tmp >>= 1;
      ++page_shift;
    }
  }

  const int max_blocks = (int)block_table.size(1);
  const int max_q = (int)max_query_len;
  const float scale = (float)softmax_scale;

#define LAUNCH_WMMA(FP8_B, NTT_I, BN_I, MINB_I)                                                 \
  launch_wmma<FP8_B, NTT_I, BN_I, MINB_I>(                                                      \
      q_ptr, r_ptr, proj_ptr, kc_ptr, vc_ptr,                                                   \
      block_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(), \
      gate_ptr, sinks_ptr, out_ptr,                                                             \
      B, H_q, H_kv, stride_q_t, stride_r_t, kb, kt, kh, vb, vt, vh,                             \
      stride_bt, stride_gate, stride_sink,                                                      \
      page, page_shift, max_blocks, max_q, act_toks, E, (int)window, scale, gate_aligned4,      \
      stream)

  if (is_fp8) {
    switch (variant) {
      case 0: LAUNCH_WMMA(true, 2, 32, 1); break;
      case 1: LAUNCH_WMMA(true, 1, 32, 2); break;
      case 2: LAUNCH_WMMA(true, 1, 64, 2); break;
      default: TORCH_CHECK(false, "invalid WMMA prefill variant: ", variant, " (expected 0, 1, or 2)");
    }
  } else {
    switch (variant) {
      case 0: LAUNCH_WMMA(false, 2, 32, 1); break;
      case 1: LAUNCH_WMMA(false, 1, 32, 2); break;
      case 2: LAUNCH_WMMA(false, 1, 64, 2); break;
      default: TORCH_CHECK(false, "invalid WMMA prefill variant: ", variant, " (expected 0, 1, or 2)");
    }
  }
#undef LAUNCH_WMMA

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "attn_rel_pf_wmma_forward kernel launch failed: ", cudaGetErrorString(err));

  if (act_toks < num_tokens) {
    err = cudaMemsetAsync(
        out_ptr + (size_t)act_toks * H_q * D, 0,
        (size_t)(num_tokens - act_toks) * H_q * D * sizeof(__half), stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemsetAsync tail zero failed: ", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_rel_pf_wmma_forward", &attn_rel_pf_wmma_forward,
        "Volta sm_70 3-head-shared WMMA prefill attention with on-the-fly R @ rel_proj "
        "(FP16 & FP8 E4M3FN separate paged KV, fused gate_sink)");
}
