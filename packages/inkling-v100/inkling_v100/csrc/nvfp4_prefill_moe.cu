// SPDX-License-Identifier: Apache-2.0
// nvfp4_prefill_moe.cu - Exact native NVFP4 (E2M1 x E4M3FN x FP32 global) prefill MoE for Volta (sm_70).
//
// Targets thinkingmachines/Inkling-2-Mini EP=4 NVFP4 layers (L0..19: E_local=32, H=2304, I=1152, topk=8)
// across prefill chunks T=64..2048+ with 100% exact native NVFP4 math, ZERO lossy weight packing,
// ZERO permanent weight copies, ZERO host synchronization, and ZERO register spills.
//
// Key Volta sm_70 architecture:
//   1. Exact zero-rounding, branchless FP16 B-fragment decode directly from in-place QPN layout:
//        e2m1(c) * 2^-14    (2 ALU ops per fp16x2 pair: exact for all 16 codes incl. +/-0.5)
//        * (e4m3(s) * 2^+7) (4-op branchless unpack: (f8->f16*2^-8) * 32768.0h, exact for all
//                            254 finite E4M3FN normals & subnormals, max 448*128=57344 <= 65504,
//                            0x7F/0xFF NaN -> 0.0!)
//        = e2m1(c) * e4m3(s) * 2^-7 (100% bit-exact in FP16 for all 4,064 (E4M3FN, E2M1) pairs!)
//        with FP32 epilogue scale `g_t[e] * 128.0f` (works for ANY global scale g_t[e]!).
//   2. Register-fused w13 + exact FP32 softcapped interleaved SiTU:
//        Volta mma.m8n8k4 lands interleaved (gate, up) columns of w13_qpn in adjacent registers
//        (d[0]/d[1], d[2]/d[3], d[4]/d[5], d[6]/d[7]) of the SAME lane -> zero y13 HBM buffer!
//   3. Zero-RAW-stall HMMA.884 scheduling (k=0..3 across all accumulators before k=4..7) +
//      half-staggered weight prefetching (<= 96 regs on BM=64 -> 5 CTAs/SM, 0 bytes stack/spills!).
//   4. Zero-sync exact-fit 3-bucket M-tile dispatch (BM=64 for rows>32, BM=32 for 17..32, BM=16 for 1..16)
//      for T > 448, and single-launch BM=16/32 for T <= 448 (100% CUDA-graph safe).
//   5. Bit-exact QPN Dequantization (`nvfp4_qpn_dequant_expert` / `nvfp4_qpn_dequant_batch`) +
//      cuBLAS FP16 reference path (mode 5) for standalone use & ablation.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>

namespace {

constexpr int MAX_DEVS = 32;
constexpr int MAX_E_LOCAL = 128;
constexpr int MAX_TOPK = 32;

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t _err = (call);                                                           \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ", cudaGetErrorString(_err));          \
  } while (0)

#define CUBLAS_CHECK(call)                                                               \
  do {                                                                                   \
    cublasStatus_t _st = (call);                                                         \
    TORCH_CHECK(_st == CUBLAS_STATUS_SUCCESS, "cuBLAS error code: ", (int)_st);          \
  } while (0)

// Exact Volta HMMA.884 (row.col, FP16xFP16 -> FP32 accumulate).
__device__ __forceinline__ void mma884(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned hmul2u(unsigned a, unsigned b) {
  const __half2 ha = *reinterpret_cast<const __half2*>(&a);
  const __half2 hb = *reinterpret_cast<const __half2*>(&b);
  const __half2 r = __hmul2(ha, hb);
  return *reinterpret_cast<const unsigned*>(&r);
}

// Exact OCP FP8 E4M3FN byte -> FP32 in pure register ALU (zero constant-cache serialization).
__device__ __forceinline__ float e4m3_to_f32(unsigned u) {
  const unsigned abs_u = u & 0x7Fu;
  if (abs_u == 0x7Fu) return 0.0f;
  const unsigned sign = (u & 0x80u) << 24;
  if ((abs_u & 0x78u) == 0u) {
    const float mag = static_cast<float>(abs_u) * (1.0f / 512.0f);
    return __uint_as_float(__float_as_uint(mag) | sign);
  }
  return __uint_as_float(sign | ((abs_u << 20) + 0x3C000000u));
}

// Branchless 4-op OCP FP8 E4M3FN byte -> FP16x2 holding exact `e4m3(u) * 2^7`:
//   - Raw bit unpack `((u & 0x80) << 8) | ((u & 0x7F) << 7)` produces exact `e4m3(u) * 2^-8`
//     for ALL 254 finite codes (both normals AND subnormals!).
//   - Mask `0x7F`/`0xFF` NaN -> 0.0.
//   - One HMUL2 by `32768.0h` (`0x78007800u` = 2^+15) scales to exact `e4m3(u) * 2^+7`
//     (max 448 * 128 = 57344 <= 65504, zero overflow, zero underflow, zero branches!).
__device__ __forceinline__ unsigned e4m3_to_f16x2_mul128(unsigned u) {
  const unsigned abs_u = u & 0x7Fu;
  unsigned h = ((u & 0x80u) << 8) | (abs_u << 7);
  h = (abs_u == 0x7Fu) ? 0u : h;
  return hmul2u(h | (h << 16), 0x78007800u);
}

// Exact E2M1 4-bit code (0..15) -> FP32 in register ALU (used to seed warp LUT for dequant).
__device__ __forceinline__ float e2m1_from_u4(unsigned c) {
  const unsigned mag = c & 7u;
  unsigned bits = (mag < 2u) ? (mag ? 0x3F000000u : 0u) : (0x3F000000u + (mag << 22));
  bits |= (c & 8u) << 28;
  return __uint_as_float(bits);
}

// Exact Inkling2 softcapped SiTU in FP32 (sm_70-safe IEEE tanhf/expf/__fdiv_rn).
__device__ __forceinline__ float softcap_situ(float g, float u, float gamma, float gate_cap, float up_cap) {
  const float sig = __fdiv_rn(1.0f, 1.0f + expf(-g));
  if (gate_cap > 0.0f) g = gate_cap * tanhf(__fdiv_rn(g, gate_cap));
  if (up_cap > 0.0f) u = up_cap * tanhf(__fdiv_rn(u, up_cap));
  return ((g * sig) * u) * gamma;
}

// Zero-RAW-stall QPN k=0..7 step: runs k=0..3 across all RB accumulators FIRST,
// then unpacks b2,b3 and runs k=4..7 across all RB accumulators.
template <int RB>
__device__ __forceinline__ void mma_qpn_k8_cb1(
    unsigned wx,
    unsigned sc_h2,
    const uint4 (&a)[RB],
    float (&acc)[RB][8]) {
  const unsigned b0 = hmul2u(((wx << 12) & 0x80008000u) | ((wx << 9) & 0x0E000E00u), sc_h2);
  const unsigned b1 = hmul2u(((wx <<  8) & 0x80008000u) | ((wx << 5) & 0x0E000E00u), sc_h2);
#pragma unroll
  for (int rb = 0; rb < RB; rb++) {
    mma884(acc[rb], a[rb].x, a[rb].y, b0, b1);
  }
  const unsigned b2 = hmul2u(((wx <<  4) & 0x80008000u) | ((wx << 1) & 0x0E000E00u), sc_h2);
  const unsigned b3 = hmul2u((wx         & 0x80008000u) | ((wx >> 3) & 0x0E000E00u), sc_h2);
#pragma unroll
  for (int rb = 0; rb < RB; rb++) {
    mma884(acc[rb], a[rb].z, a[rb].w, b2, b3);
  }
}

// ---------------------------------------------------------------------------
// 1. Zero-Sync Tile Builders (Single-BM and Exact-Fit 3-Bucket Multi-BM)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 4)
k_prepare_prefill_tiles_single(
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ gids,
    const int32_t* __restrict__ goff,
    int32_t* __restrict__ inv_perm,
    int4* __restrict__ tile_desc,
    int32_t* __restrict__ tile_count,
    int max_g,
    int BM) {
  const int n_local = goff[max_g];
  if (inv_perm != nullptr) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n_local) {
      inv_perm[perm[idx]] = idx;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int nt = 0;
    for (int g = 0; g < max_g; g++) {
      const int s0 = goff[g];
      const int s1 = goff[g + 1];
      const int m = s1 - s0;
      if (m <= 0) continue;
      const int e = gids[g];
      for (int r0 = 0; r0 < m; r0 += BM) {
        const int rows = (m - r0 < BM) ? (m - r0) : BM;
        tile_desc[nt] = make_int4(e, s0 + r0, rows, 0);
        nt++;
      }
    }
    tile_count[0] = nt;
  }
}

// Exact-fit 3-bucket splitter:
//   - While rem > 32: emit BM=64 tile (rows in 33..64 -> both row-warps active!)
//   - Else if rem > 16: emit BM=32 tile (rows in 17..32 -> both row-warps active!)
//   - Else if rem > 0:  emit BM=16 tile (rows in 1..16!)
__global__ void __launch_bounds__(256, 4)
k_prepare_prefill_tiles_multi(
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ gids,
    const int32_t* __restrict__ goff,
    int32_t* __restrict__ inv_perm,
    int4* __restrict__ tiles_64,
    int4* __restrict__ tiles_32,
    int4* __restrict__ tiles_16,
    int32_t* __restrict__ counts3,
    int max_g) {
  const int n_local = goff[max_g];
  if (inv_perm != nullptr) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n_local) {
      inv_perm[perm[idx]] = idx;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int n64 = 0, n32 = 0, n16 = 0;
    for (int g = 0; g < max_g; g++) {
      const int s0 = goff[g];
      const int s1 = goff[g + 1];
      int rem = s1 - s0;
      if (rem <= 0) continue;
      const int e = gids[g];
      int cur = s0;
      while (rem > 32) {
        const int rows = (rem < 64) ? rem : 64;
        tiles_64[n64++] = make_int4(e, cur, rows, 0);
        cur += rows;
        rem -= rows;
      }
      if (rem > 16) {
        tiles_32[n32++] = make_int4(e, cur, rem, 0);
      } else if (rem > 0) {
        tiles_16[n16++] = make_int4(e, cur, rem, 0);
      }
    }
    counts3[0] = n64;
    counts3[1] = n32;
    counts3[2] = n16;
  }
}

// 2-bucket exact-fit splitter for medium batches T=16..128:
//   - While rem > 8: emit BM=16 tile (rows in 9..16)
//   - Else if rem > 0: emit BM=8 tile (rows in 1..8)
__global__ void __launch_bounds__(256, 4)
k_prepare_small_tiles_2b(
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ gids,
    const int32_t* __restrict__ goff,
    int32_t* __restrict__ inv_perm,
    int4* __restrict__ tiles_16,
    int4* __restrict__ tiles_8,
    int32_t* __restrict__ counts2,
    int max_g) {
  const int n_local = goff[max_g];
  if (inv_perm != nullptr) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n_local) {
      inv_perm[perm[idx]] = idx;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int n16 = 0, n8 = 0;
    for (int g = 0; g < max_g; g++) {
      const int s0 = goff[g];
      const int s1 = goff[g + 1];
      int rem = s1 - s0;
      if (rem <= 0) continue;
      const int e = gids[g];
      int cur = s0;
      while (rem > 8) {
        const int rows = (rem < 16) ? rem : 16;
        tiles_16[n16++] = make_int4(e, cur, rows, 0);
        cur += rows;
        rem -= rows;
      }
      if (rem > 0) {
        tiles_8[n8++] = make_int4(e, cur, rem, 0);
      }
    }
    counts2[0] = n16;
    counts2[1] = n8;
  }
}

// ---------------------------------------------------------------------------
// 2a. Zero-Syncthreads-K-Loop BM=8 Coalesced NVFP4 MoE Kernel (BN=32, WK=4)
// ---------------------------------------------------------------------------
// Eliminates skinny_nvfp4_moe_qpn's 1.24x..2.30x redundant expert weight reads for T=4..64:
//   - 1 QPN 32-column tile per CTA (grid.x = N_full / 32 = 72 -> saturates all 80 V100 SMs!)
//   - WK=4 warps split K (128 threads/CTA); each warp owns a 512 B (8x32) warp-private SMEM
//     activation tile -> 1 uint4/lane, ZERO __syncthreads() in the K-loop (only __syncwarp())!
//   - Only 8 FP32 accumulators per lane -> ~48 regs/thr, 8 CTAs/SM, 0 B spills!
__device__ __forceinline__ uint2 ldg_u2_na(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(v.x), "=r"(v.y)
               : "l"(p));
  return v;
}

template <int GATED, int DIRECT_IN, int WK>
__global__ void __launch_bounds__(32 * WK, (WK <= 4) ? 8 : 5)
k_nvfp4_hmma_bm8(
    const uint4* __restrict__ X_u4,
    const uint2* __restrict__ W_qpn_u2,
    const uint8_t* __restrict__ S_qpn,
    const float* __restrict__ g_t,
    const int32_t* __restrict__ perm,
    const float* __restrict__ weights,
    const int4* __restrict__ tile_desc,
    const int32_t* __restrict__ tile_count,
    uint4* __restrict__ Y_u4,
    int N_full,
    int K,
    int topk,
    float gate_cap,
    float up_cap,
    int scatter_out) {
  const int tile_idx = blockIdx.y;
  if (tile_idx >= tile_count[0]) return;

  const int4 td = tile_desc[tile_idx];
  const int expert = td.x;
  const int s0_tile = td.y;
  const int rows = td.z;

  __shared__ uint4 sm_a[WK][4][8];
  __shared__ float sm_red[WK][8][32];
  union OutSmem {
    __half h[8][32];
    __half2 h2[8][16];
    uint4 u4[8][4];
  };
  __shared__ OutSmem sm_out;
  __shared__ int sh_xrow[8];
  __shared__ int sh_slot[8];
  __shared__ float sh_gamma[GATED ? 8 : 1];

  const int tid = threadIdx.x;
  const int wk = tid >> 5;
  const int lane = tid & 31;
  const int w_tile = blockIdx.x;
  const int NT = N_full / 32;
  const int cgroups = K / 16;
  const int K_steps = K / 32;
  const int K8 = K / 8;

  if (tid < 8) {
    const int r_c = (tid < rows) ? tid : (rows - 1);
    const int row_sorted = s0_tile + r_c;
    const int sl = (!DIRECT_IN || GATED || scatter_out) ? perm[row_sorted] : 0;
    sh_slot[tid] = sl;
    sh_xrow[tid] = DIRECT_IN ? row_sorted : (sl / topk);
    if constexpr (GATED) {
      sh_gamma[tid] = weights[sl];
    }
  }
  __syncthreads();

  const int ks0 = (K_steps * wk) / WK;
  const int ks1 = (K_steps * (wk + 1)) / WK;
  const int my_row = sh_xrow[lane >> 2];
  const int my_jj = lane & 3;
  const int my_r_sw = (lane >> 2) ^ (my_jj << 1);
  const int rin = (lane & 3) + ((lane & 16) ? 4 : 0);

  const size_t w_base = (static_cast<size_t>(expert) * static_cast<size_t>(NT) + static_cast<size_t>(w_tile)) *
                            static_cast<size_t>(cgroups) * 32u +
                        static_cast<size_t>(lane);
  const uint2* wp = W_qpn_u2 + w_base;
  const uint8_t* sp = S_qpn + w_base;

  float acc[1][8];
#pragma unroll
  for (int e = 0; e < 8; e++) acc[0][e] = 0.0f;

  if (ks0 < ks1) {
    uint4 a_stg = __ldg(X_u4 + static_cast<size_t>(my_row) * static_cast<size_t>(K8) +
                        static_cast<size_t>(ks0) * 4u + static_cast<size_t>(my_jj));
    uint2 w_cur = ldg_u2_na(wp + static_cast<size_t>(ks0) * 64u);
    unsigned s_cur = static_cast<unsigned>(__ldg(sp + static_cast<size_t>(ks0) * 64u));
    sm_a[wk][my_jj][my_r_sw] = a_stg;
    __syncwarp();

    for (int ks = ks0; ks < ks1; ks++) {
      const bool has_next = (ks + 1 < ks1);
      if (has_next) {
        a_stg = __ldg(X_u4 + static_cast<size_t>(my_row) * static_cast<size_t>(K8) +
                      static_cast<size_t>(ks + 1) * 4u + static_cast<size_t>(my_jj));
      }
      const size_t cg1_off = static_cast<size_t>(ks) * 64u + 32u;
      const uint2 w_h1 = ldg_u2_na(wp + cg1_off);
      const unsigned s_h1 = static_cast<unsigned>(__ldg(sp + cg1_off));

      const unsigned sc0 = e4m3_to_f16x2_mul128(s_cur);
      {
        uint4 af[1] = {sm_a[wk][0][rin]};
        mma_qpn_k8_cb1<1>(w_cur.x, sc0, af, acc);
      }
      {
        uint4 af[1] = {sm_a[wk][1][rin ^ 2]};
        mma_qpn_k8_cb1<1>(w_cur.y, sc0, af, acc);
      }

      if (has_next) {
        const size_t nxt_off = static_cast<size_t>(ks + 1) * 64u;
        w_cur = ldg_u2_na(wp + nxt_off);
        s_cur = static_cast<unsigned>(__ldg(sp + nxt_off));
      }

      const unsigned sc1 = e4m3_to_f16x2_mul128(s_h1);
      {
        uint4 af[1] = {sm_a[wk][2][rin ^ 4]};
        mma_qpn_k8_cb1<1>(w_h1.x, sc1, af, acc);
      }
      {
        uint4 af[1] = {sm_a[wk][3][rin ^ 6]};
        mma_qpn_k8_cb1<1>(w_h1.y, sc1, af, acc);
      }

      if (has_next) {
        __syncwarp();
        sm_a[wk][my_jj][my_r_sw] = a_stg;
        __syncwarp();
      }
    }
  }

  // Parallel K-slice reduction through dedicated sm_red (no prior syncthreads needed!)
#pragma unroll
  for (int e = 0; e < 8; e++) {
    sm_red[wk][e][lane] = acc[0][e];
  }
  __syncthreads();

  if (wk == 0) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float sum = sm_red[0][e][lane];
#pragma unroll
      for (int w = 1; w < WK; w++) sum += sm_red[w][e][lane];
      v[e] = sum;
    }
    const float gs = __ldg(g_t + expert) * 128.0f;
    const int q = (lane >> 2) & 3;
    const int r_lo = (lane & 1) + ((lane & 16) ? 4 : 0);
    const int r_hi = r_lo + 2;
    const int c_lo = q * 8 + (lane & 2);
    const int c_hi = c_lo + 4;

    if constexpr (GATED) {
      const int ic_lo = c_lo >> 1;
      const int ic_hi = c_hi >> 1;
      const float g_lo = sh_gamma[r_lo];
      const float g_hi = sh_gamma[r_hi];
      sm_out.h[r_lo][ic_lo] = __float2half_rn(softcap_situ(v[0] * gs, v[1] * gs, g_lo, gate_cap, up_cap));
      sm_out.h[r_hi][ic_lo] = __float2half_rn(softcap_situ(v[2] * gs, v[3] * gs, g_hi, gate_cap, up_cap));
      sm_out.h[r_lo][ic_hi] = __float2half_rn(softcap_situ(v[4] * gs, v[5] * gs, g_lo, gate_cap, up_cap));
      sm_out.h[r_hi][ic_hi] = __float2half_rn(softcap_situ(v[6] * gs, v[7] * gs, g_hi, gate_cap, up_cap));
      __syncwarp();
      const int I8 = (N_full / 2) / 8;
      const int col_base = w_tile * 2;
      if (lane < rows * 2) {
        const int r = lane >> 1;
        const int c2 = lane & 1;
        if (col_base + c2 < I8) {
          Y_u4[static_cast<size_t>(s0_tile + r) * static_cast<size_t>(I8) + col_base + c2] = sm_out.u4[r][c2];
        }
      }
    } else {
      const int c2_lo = c_lo >> 1;
      const int c2_hi = c_hi >> 1;
      sm_out.h2[r_lo][c2_lo] = __floats2half2_rn(v[0] * gs, v[1] * gs);
      sm_out.h2[r_hi][c2_lo] = __floats2half2_rn(v[2] * gs, v[3] * gs);
      sm_out.h2[r_lo][c2_hi] = __floats2half2_rn(v[4] * gs, v[5] * gs);
      sm_out.h2[r_hi][c2_hi] = __floats2half2_rn(v[6] * gs, v[7] * gs);
      __syncwarp();
      const int N8 = N_full / 8;
      const int col_base = w_tile * 4;
      if (lane < rows * 4) {
        const int r = lane >> 2;
        const int c4 = lane & 3;
        if (col_base + c4 < N8) {
          const int dst_row = scatter_out ? sh_slot[r] : (s0_tile + r);
          Y_u4[static_cast<size_t>(dst_row) * static_cast<size_t>(N8) + col_base + c4] = sm_out.u4[r][c4];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Fused Large-M Volta HMMA.884 QPN NVFP4 MoE Kernel (BN=64, 128 threads/CTA)
// ---------------------------------------------------------------------------
// Half-staggered weight prefetching + branchless E4M3FN*128 keeps BM=64 at <=96 regs (5 CTAs/SM!)
// with 0 bytes stack frame and 0 bytes register spills across all specializations.
template <int BM, int GATED, int DIRECT_IN, int MIN_BLOCKS>
__global__ void __launch_bounds__(128, MIN_BLOCKS)
k_nvfp4_hmma_bn64(
    const uint4* __restrict__ X_u4,
    const uint2* __restrict__ W_qpn_u2,
    const uint8_t* __restrict__ S_qpn,
    const float* __restrict__ g_t,
    const int32_t* __restrict__ perm,
    const float* __restrict__ weights,
    const int4* __restrict__ tile_desc,
    const int32_t* __restrict__ tile_count,
    uint4* __restrict__ Y_u4,
    int N_full,
    int K,
    int topk,
    float gate_cap,
    float up_cap,
    int scatter_out) {
  static_assert(BM == 16 || BM == 32 || BM == 64, "BM must be 16, 32, or 64");
  constexpr int NROW = 2;
  constexpr int NCOL = 2;
  constexpr int RB = BM / (8 * NROW);  // 1, 2, or 4
  constexpr int A_ITERS = (BM * 4 + 127) / 128;

  const int tile_idx = blockIdx.y;
  if (tile_idx >= tile_count[0]) return;

  const int4 td = tile_desc[tile_idx];
  const int expert = td.x;
  const int s0_tile = td.y;
  const int rows = td.z;

  union SmemUnion {
    uint4 a[2][4][BM];
    __half out_h[BM][64];
    __half2 out_h2[BM][32];
    uint4 out_u4[BM][8];
  };
  __shared__ SmemUnion sm;
  __shared__ int sh_xrow[BM];
  __shared__ int sh_slot[BM];
  __shared__ float sh_gamma[GATED ? BM : 1];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gr = warp >> 1;
  const int gi = warp & 1;
  const int bx = blockIdx.x;
  const int NT = N_full / 32;
  const int w_tile = min(bx * NCOL + gi, NT - 1);
  const int cgroups = K / 16;
  const int K_steps = K / 32;
  const int K8 = K / 8;

  if (tid < BM) {
    const int r_c = (tid < rows) ? tid : (rows - 1);
    const int row_sorted = s0_tile + r_c;
    const int sl = (!DIRECT_IN || GATED || scatter_out) ? perm[row_sorted] : 0;
    sh_slot[tid] = sl;
    sh_xrow[tid] = DIRECT_IN ? row_sorted : (sl / topk);
    if constexpr (GATED) {
      sh_gamma[tid] = weights[sl];
    }
  }
  __syncthreads();

  const int warp_r0 = gr * (RB * 8);
  const bool warp_active = (BM > 16) ? true : (warp_r0 < rows);

  int a_row_reg[A_ITERS];
#pragma unroll
  for (int q = 0; q < A_ITERS; q++) {
    const int u = q * 128 + tid;
    if constexpr (BM >= 32) {
      a_row_reg[q] = sh_xrow[u >> 2];
    } else {
      a_row_reg[q] = (u < BM * 4) ? sh_xrow[u >> 2] : 0;
    }
  }

  const size_t w_base = (static_cast<size_t>(expert) * static_cast<size_t>(NT) + static_cast<size_t>(w_tile)) *
                            static_cast<size_t>(cgroups) * 32u +
                        static_cast<size_t>(lane);
  const uint2* wp = W_qpn_u2 + w_base;
  const uint8_t* sp = S_qpn + w_base;

  float acc[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; rb++) {
#pragma unroll
    for (int e = 0; e < 8; e++) acc[rb][e] = 0.0f;
  }

  uint4 a_stg[A_ITERS];
  auto load_a_global = [&](int k_step) {
    const int jj = tid & 3;
    const size_t k_off = static_cast<size_t>(k_step) * 4u + static_cast<size_t>(jj);
#pragma unroll
    for (int q = 0; q < A_ITERS; q++) {
      const int u = q * 128 + tid;
      if constexpr (BM >= 32) {
        a_stg[q] = __ldg(X_u4 + static_cast<size_t>(a_row_reg[q]) * static_cast<size_t>(K8) + k_off);
      } else {
        if (u < BM * 4) {
          a_stg[q] = __ldg(X_u4 + static_cast<size_t>(a_row_reg[q]) * static_cast<size_t>(K8) + k_off);
        }
      }
    }
  };

  auto store_a_smem = [&](int stage) {
    const int jj = tid & 3;
    const int sw = jj << 1;
#pragma unroll
    for (int q = 0; q < A_ITERS; q++) {
      const int u = q * 128 + tid;
      if constexpr (BM >= 32) {
        sm.a[stage][jj][(u >> 2) ^ sw] = a_stg[q];
      } else {
        if (u < BM * 4) {
          sm.a[stage][jj][(u >> 2) ^ sw] = a_stg[q];
        }
      }
    }
  };

  // Half-staggered weight pipeline: only 2 uint2s + 2 u8s live at once (saves 6 regs on BM=64!).
  uint2 w_cur = make_uint2(0, 0);
  unsigned s_cur = 0u;

  load_a_global(0);
  if (warp_active) {
    w_cur = __ldg(wp);
    s_cur = static_cast<unsigned>(__ldg(sp));
  }
  store_a_smem(0);
  __syncthreads();

  const int rin = (lane & 3) + ((lane & 16) ? 4 : 0);

  for (int k_step = 0; k_step < K_steps; k_step++) {
    const int stage = k_step & 1;
    const int nxt_stage = stage ^ 1;
    const bool has_next = (k_step + 1 < K_steps);

    // Prefetch next K-step's A AND second half (cg1) of current K-step's weights!
    if (has_next) {
      load_a_global(k_step + 1);
    }
    uint2 w_h1 = make_uint2(0, 0);
    unsigned s_h1 = 0u;
    if (warp_active) {
      const size_t cg1_off = static_cast<size_t>(k_step) * 64u + 32u;
      w_h1 = __ldg(wp + cg1_off);
      s_h1 = static_cast<unsigned>(__ldg(sp + cg1_off));

      // Compute half 0 (jj=0, 1) with branchless scale decode!
      const unsigned sc0 = e4m3_to_f16x2_mul128(s_cur);
      {
        uint4 af[RB];
#pragma unroll
        for (int rb = 0; rb < RB; rb++) af[rb] = sm.a[stage][0][warp_r0 + rb * 8 + rin];
        mma_qpn_k8_cb1<RB>(w_cur.x, sc0, af, acc);
      }
      {
        uint4 af[RB];
#pragma unroll
        for (int rb = 0; rb < RB; rb++) af[rb] = sm.a[stage][1][(warp_r0 + rb * 8 + rin) ^ 2];
        mma_qpn_k8_cb1<RB>(w_cur.y, sc0, af, acc);
      }

      // Prefetch next K-step's half 0 weights right before computing half 1!
      if (has_next) {
        const size_t nxt_off = static_cast<size_t>(k_step + 1) * 64u;
        w_cur = __ldg(wp + nxt_off);
        s_cur = static_cast<unsigned>(__ldg(sp + nxt_off));
      }

      // Compute half 1 (jj=2, 3)
      const unsigned sc1 = e4m3_to_f16x2_mul128(s_h1);
      {
        uint4 af[RB];
#pragma unroll
        for (int rb = 0; rb < RB; rb++) af[rb] = sm.a[stage][2][(warp_r0 + rb * 8 + rin) ^ 4];
        mma_qpn_k8_cb1<RB>(w_h1.x, sc1, af, acc);
      }
      {
        uint4 af[RB];
#pragma unroll
        for (int rb = 0; rb < RB; rb++) af[rb] = sm.a[stage][3][(warp_r0 + rb * 8 + rin) ^ 6];
        mma_qpn_k8_cb1<RB>(w_h1.y, sc1, af, acc);
      }
    }

    if (has_next) {
      store_a_smem(nxt_stage);
      __syncthreads();
    }
  }

  __syncthreads();

  if (warp_active) {
    const float gs = __ldg(g_t + expert) * 128.0f;
    const int q = (lane >> 2) & 3;
    const int r_in_8 = (lane & 1) + ((lane & 16) ? 4 : 0);
    const int c_lo = gi * 32 + q * 8 + (lane & 2);
    const int c_hi = c_lo + 4;

    if constexpr (GATED) {
      const int ic_lo = c_lo >> 1;
      const int ic_hi = c_hi >> 1;
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        const int r_lo = warp_r0 + rb * 8 + r_in_8;
        const int r_hi = r_lo + 2;
        const float g_lo = sh_gamma[r_lo];
        const float g_hi = sh_gamma[r_hi];
        sm.out_h[r_lo][ic_lo] = __float2half_rn(softcap_situ(acc[rb][0] * gs, acc[rb][1] * gs, g_lo, gate_cap, up_cap));
        sm.out_h[r_hi][ic_lo] = __float2half_rn(softcap_situ(acc[rb][2] * gs, acc[rb][3] * gs, g_hi, gate_cap, up_cap));
        sm.out_h[r_lo][ic_hi] = __float2half_rn(softcap_situ(acc[rb][4] * gs, acc[rb][5] * gs, g_lo, gate_cap, up_cap));
        sm.out_h[r_hi][ic_hi] = __float2half_rn(softcap_situ(acc[rb][6] * gs, acc[rb][7] * gs, g_hi, gate_cap, up_cap));
      }
    } else {
      const int c2_lo = c_lo >> 1;
      const int c2_hi = c_hi >> 1;
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        const int r_lo = warp_r0 + rb * 8 + r_in_8;
        const int r_hi = r_lo + 2;
        sm.out_h2[r_lo][c2_lo] = __floats2half2_rn(acc[rb][0] * gs, acc[rb][1] * gs);
        sm.out_h2[r_hi][c2_lo] = __floats2half2_rn(acc[rb][2] * gs, acc[rb][3] * gs);
        sm.out_h2[r_lo][c2_hi] = __floats2half2_rn(acc[rb][4] * gs, acc[rb][5] * gs);
        sm.out_h2[r_hi][c2_hi] = __floats2half2_rn(acc[rb][6] * gs, acc[rb][7] * gs);
      }
    }
  }
  __syncthreads();

  if constexpr (GATED) {
    const int I8 = (N_full / 2) / 8;
    const int col_base = bx * 4;
    for (int u = tid; u < rows * 4; u += 128) {
      const int r = u >> 2;
      const int c4 = u & 3;
      if (col_base + c4 < I8) {
        Y_u4[static_cast<size_t>(s0_tile + r) * static_cast<size_t>(I8) + col_base + c4] = sm.out_u4[r][c4];
      }
    }
  } else {
    const int N8 = N_full / 8;
    const int col_base = bx * 8;
    for (int u = tid; u < rows * 8; u += 128) {
      const int r = u >> 3;
      const int c8 = u & 7;
      if (col_base + c8 < N8) {
        const int dst_row = scatter_out ? sh_slot[r] : (s0_tile + r);
        Y_u4[static_cast<size_t>(dst_row) * static_cast<size_t>(N8) + col_base + c8] = sm.out_u4[r][c8];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Exact QPN NVFP4 -> FP16 [N, K] row-major dequantization kernels
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 8)
k_nvfp4_qpn_dequant_smem(
    const uint2* __restrict__ w_qpn_u2,
    const uint8_t* __restrict__ s_qpn,
    const float* __restrict__ g_t,
    const int32_t* __restrict__ expert_ids,
    int expert_idx,
    uint4* __restrict__ out_u4,
    int tiles,
    int cgroups) {
  __shared__ uint4 smem[32][8];

  const int tile = blockIdx.x;
  const int cg_blk = blockIdx.y;
  const int bidx = blockIdx.z;
  const int e = (expert_ids != nullptr) ? expert_ids[bidx] : expert_idx;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int cg = cg_blk * 4 + warp;

  const size_t expert_elems = static_cast<size_t>(tiles) * static_cast<size_t>(cgroups) * 32u;
  const size_t idx = static_cast<size_t>(e) * expert_elems +
                     (static_cast<size_t>(tile) * static_cast<size_t>(cgroups) + static_cast<size_t>(cg)) * 32u +
                     static_cast<size_t>(lane);

  const uint2 w8 = __ldg(w_qpn_u2 + idx);
  const unsigned s8 = static_cast<unsigned>(__ldg(s_qpn + idx));
  const float scale = e4m3_to_f32(s8) * __ldg(g_t + e);
  const float e2m1_lane = e2m1_from_u4(static_cast<unsigned>(lane & 15));

  auto dec_pair = [&](unsigned lo_nib, unsigned hi_nib) -> unsigned {
    const float f0 = __shfl_sync(0xFFFFFFFFu, e2m1_lane, static_cast<int>(lo_nib), 16) * scale;
    const float f1 = __shfl_sync(0xFFFFFFFFu, e2m1_lane, static_cast<int>(hi_nib), 16) * scale;
    const __half2 h2 = __floats2half2_rn(f0, f1);
    return *reinterpret_cast<const unsigned*>(&h2);
  };

  uint4 v0, v1;
  v0.x = dec_pair((w8.x >>  0) & 0xFu, (w8.x >> 16) & 0xFu);
  v0.y = dec_pair((w8.x >>  4) & 0xFu, (w8.x >> 20) & 0xFu);
  v0.z = dec_pair((w8.x >>  8) & 0xFu, (w8.x >> 24) & 0xFu);
  v0.w = dec_pair((w8.x >> 12) & 0xFu, (w8.x >> 28) & 0xFu);

  v1.x = dec_pair((w8.y >>  0) & 0xFu, (w8.y >> 16) & 0xFu);
  v1.y = dec_pair((w8.y >>  4) & 0xFu, (w8.y >> 20) & 0xFu);
  v1.z = dec_pair((w8.y >>  8) & 0xFu, (w8.y >> 24) & 0xFu);
  v1.w = dec_pair((w8.y >> 12) & 0xFu, (w8.y >> 28) & 0xFu);

  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int swiz = (col & 3) | ((col & 8) >> 1);
  smem[col][(2 * warp) ^ swiz] = v0;
  smem[col][(2 * warp + 1) ^ swiz] = v1;

  __syncthreads();

  const int c = tid & 7;
  const int r0 = tid >> 3;
  const int swiz_r = (r0 & 3) | ((r0 & 8) >> 1);
  const size_t K8 = static_cast<size_t>(cgroups) * 2u;
  const size_t out_expert_off = static_cast<size_t>(bidx) * static_cast<size_t>(tiles) * 32u * K8;
  const size_t base = out_expert_off +
                      static_cast<size_t>(tile * 32 + r0) * K8 +
                      static_cast<size_t>(cg_blk) * 8u + static_cast<size_t>(c);

  out_u4[base] = smem[r0][c ^ swiz_r];
  out_u4[base + static_cast<size_t>(16) * K8] = smem[r0 + 16][c ^ swiz_r];
}

__global__ void __launch_bounds__(256, 4)
k_nvfp4_qpn_dequant_direct(
    const uint2* __restrict__ w_qpn_u2,
    const uint8_t* __restrict__ s_qpn,
    const float* __restrict__ g_t,
    const int32_t* __restrict__ expert_ids,
    int expert_idx,
    uint4* __restrict__ out_u4,
    int tiles,
    int cgroups) {
  const int warp_global = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int total_warps = tiles * cgroups;
  if (warp_global >= total_warps) return;

  const int tile = warp_global / cgroups;
  const int cg = warp_global - tile * cgroups;
  const int bidx = blockIdx.z;
  const int e = (expert_ids != nullptr) ? expert_ids[bidx] : expert_idx;
  const int lane = threadIdx.x & 31;

  const size_t expert_elems = static_cast<size_t>(total_warps) * 32u;
  const size_t idx = static_cast<size_t>(e) * expert_elems +
                     static_cast<size_t>(warp_global) * 32u + static_cast<size_t>(lane);

  const uint2 w8 = __ldg(w_qpn_u2 + idx);
  const unsigned s8 = static_cast<unsigned>(__ldg(s_qpn + idx));
  const float scale = e4m3_to_f32(s8) * __ldg(g_t + e);
  const float e2m1_lane = e2m1_from_u4(static_cast<unsigned>(lane & 15));

  auto dec_pair = [&](unsigned lo_nib, unsigned hi_nib) -> unsigned {
    const float f0 = __shfl_sync(0xFFFFFFFFu, e2m1_lane, static_cast<int>(lo_nib), 16) * scale;
    const float f1 = __shfl_sync(0xFFFFFFFFu, e2m1_lane, static_cast<int>(hi_nib), 16) * scale;
    const __half2 h2 = __floats2half2_rn(f0, f1);
    return *reinterpret_cast<const unsigned*>(&h2);
  };

  uint4 v0, v1;
  v0.x = dec_pair((w8.x >>  0) & 0xFu, (w8.x >> 16) & 0xFu);
  v0.y = dec_pair((w8.x >>  4) & 0xFu, (w8.x >> 20) & 0xFu);
  v0.z = dec_pair((w8.x >>  8) & 0xFu, (w8.x >> 24) & 0xFu);
  v0.w = dec_pair((w8.x >> 12) & 0xFu, (w8.x >> 28) & 0xFu);

  v1.x = dec_pair((w8.y >>  0) & 0xFu, (w8.y >> 16) & 0xFu);
  v1.y = dec_pair((w8.y >>  4) & 0xFu, (w8.y >> 20) & 0xFu);
  v1.z = dec_pair((w8.y >>  8) & 0xFu, (w8.y >> 24) & 0xFu);
  v1.w = dec_pair((w8.y >> 12) & 0xFu, (w8.y >> 28) & 0xFu);

  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const size_t K8 = static_cast<size_t>(cgroups) * 2u;
  const size_t out_expert_off = static_cast<size_t>(bidx) * static_cast<size_t>(tiles) * 32u * K8;
  const size_t dst = out_expert_off + static_cast<size_t>(tile * 32 + col) * K8 + static_cast<size_t>(cg) * 2u;
  out_u4[dst] = v0;
  out_u4[dst + 1] = v1;
}

inline void launch_nvfp4_qpn_dequant(
    const uint8_t* w_qpn,
    const uint8_t* s_qpn,
    const float* g_t,
    const int32_t* expert_ids,
    int expert_idx,
    int batch_size,
    __half* out,
    int N,
    int K,
    bool use_direct,
    cudaStream_t stream) {
  const int tiles = N / 32;
  const int cgroups = K / 16;
  const uint2* w_u2 = reinterpret_cast<const uint2*>(w_qpn);
  uint4* out_u4 = reinterpret_cast<uint4*>(out);
  if (!use_direct && (cgroups % 4 == 0)) {
    dim3 grid(tiles, cgroups / 4, batch_size);
    k_nvfp4_qpn_dequant_smem<<<grid, 128, 0, stream>>>(
        w_u2, s_qpn, g_t, expert_ids, expert_idx, out_u4, tiles, cgroups);
  } else {
    const int total_warps = tiles * cgroups;
    dim3 grid((total_warps + 7) / 8, 1, batch_size);
    k_nvfp4_qpn_dequant_direct<<<grid, 256, 0, stream>>>(
        w_u2, s_qpn, g_t, expert_ids, expert_idx, out_u4, tiles, cgroups);
  }
}

// ---------------------------------------------------------------------------
// 4. Fast Routing, Gather, Interleaved SiTU, Atomic-Free Combine, Scatter
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 4)
k_route_ids_count(
    const int32_t* __restrict__ ids,
    int32_t* __restrict__ rank_buf,
    int32_t* __restrict__ counts,
    int S) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= S) return;
  const int el = ids[idx];
  rank_buf[idx] = (el >= 0) ? atomicAdd(&counts[el], 1) : -1;
}

__global__ void __launch_bounds__(64, 1)
k_route_ids_scan(
    int32_t* __restrict__ counts,
    int32_t* __restrict__ base,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int E_local,
    int max_g) {
  if (threadIdx.x != 0) return;
  int n_active = 0;
  int off = 0;
  for (int e = 0; e < E_local; e++) {
    const int c = counts[e];
    counts[e] = 0;
    base[e] = off;
    if (c > 0) {
      gids[n_active] = e;
      goff[n_active] = off;
      n_active++;
      off += c;
    }
  }
  for (int g = n_active; g < max_g; g++) gids[g] = 0;
  for (int g = n_active; g <= max_g; g++) goff[g] = off;
}

__global__ void __launch_bounds__(256, 4)
k_route_ids_scatter_perm(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ rank_buf,
    const int32_t* __restrict__ base,
    int32_t* __restrict__ perm,
    int S) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= S) return;
  const int el = ids[idx];
  if (el >= 0) {
    perm[base[el] + rank_buf[idx]] = idx;
  }
}

__global__ void __launch_bounds__(288, 4)
k_gather_tokens(
    const uint4* __restrict__ x_u4,
    const int32_t* __restrict__ perm,
    int32_t* __restrict__ inv_perm,
    uint4* __restrict__ x_sorted_u4,
    int n_local,
    int topk,
    int H8) {
  const int row = blockIdx.x;
  if (row >= n_local) return;
  const int slot = perm[row];
  if (threadIdx.x == 0 && inv_perm != nullptr) {
    inv_perm[slot] = row;
  }
  const int tok = slot / topk;
  const uint4* src = x_u4 + static_cast<size_t>(tok) * static_cast<size_t>(H8);
  uint4* dst = x_sorted_u4 + static_cast<size_t>(row) * static_cast<size_t>(H8);
  for (int c = threadIdx.x; c < H8; c += blockDim.x) {
    dst[c] = __ldg(src + c);
  }
}

__global__ void __launch_bounds__(288, 4)
k_situ_interleaved_sorted(
    const uint4* __restrict__ y13_u4,
    const float* __restrict__ weights,
    const int32_t* __restrict__ perm,
    uint2* __restrict__ inter_u2,
    int n_rows,
    int I4,
    float gate_cap,
    float up_cap) {
  const int row = blockIdx.x;
  if (row >= n_rows) return;
  const float gamma = __ldg(weights + perm[row]);
  const uint4* src = y13_u4 + static_cast<size_t>(row) * static_cast<size_t>(I4);
  uint2* dst = inter_u2 + static_cast<size_t>(row) * static_cast<size_t>(I4);

  auto eval_pair = [&](unsigned packed_gu) -> float {
    const __half2 h2 = *reinterpret_cast<const __half2*>(&packed_gu);
    const float2 gu = __half22float2(h2);
    return softcap_situ(gu.x, gu.y, gamma, gate_cap, up_cap);
  };

  for (int c = threadIdx.x; c < I4; c += blockDim.x) {
    const uint4 v = src[c];
    const float o0 = eval_pair(v.x);
    const float o1 = eval_pair(v.y);
    const float o2 = eval_pair(v.z);
    const float o3 = eval_pair(v.w);
    const __half2 h01 = __floats2half2_rn(o0, o1);
    const __half2 h23 = __floats2half2_rn(o2, o3);
    uint2 out_v;
    out_v.x = *reinterpret_cast<const unsigned*>(&h01);
    out_v.y = *reinterpret_cast<const unsigned*>(&h23);
    dst[c] = out_v;
  }
}

__global__ void __launch_bounds__(288, 4)
k_combine_from_sorted(
    const uint4* __restrict__ y2_u4,
    const int32_t* __restrict__ inv_perm,
    uint4* __restrict__ out_u4,
    int T,
    int topk,
    int H8) {
  __shared__ int sh_row[MAX_TOPK];
  const int tok = blockIdx.x;
  if (tok >= T) return;

  if (threadIdx.x < topk) {
    sh_row[threadIdx.x] = inv_perm[tok * topk + threadIdx.x];
  }
  __syncthreads();

  for (int c = threadIdx.x; c < H8; c += blockDim.x) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (int k = 0; k < topk; k++) {
      const int r = sh_row[k];
      if (r >= 0) {
        const uint4 v = __ldg(y2_u4 + static_cast<size_t>(r) * static_cast<size_t>(H8) + c);
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
        const float2 f45 = __half22float2(*reinterpret_cast<const __half2*>(&v.z));
        const float2 f67 = __half22float2(*reinterpret_cast<const __half2*>(&v.w));
        a0 += f01.x; a1 += f01.y;
        a2 += f23.x; a3 += f23.y;
        a4 += f45.x; a5 += f45.y;
        a6 += f67.x; a7 += f67.y;
      }
    }
    const __half2 h01 = __floats2half2_rn(a0, a1);
    const __half2 h23 = __floats2half2_rn(a2, a3);
    const __half2 h45 = __floats2half2_rn(a4, a5);
    const __half2 h67 = __floats2half2_rn(a6, a7);
    uint4 o;
    o.x = *reinterpret_cast<const unsigned*>(&h01);
    o.y = *reinterpret_cast<const unsigned*>(&h23);
    o.z = *reinterpret_cast<const unsigned*>(&h45);
    o.w = *reinterpret_cast<const unsigned*>(&h67);
    out_u4[static_cast<size_t>(tok) * static_cast<size_t>(H8) + c] = o;
  }
}

__global__ void __launch_bounds__(288, 4)
k_scatter_rows(
    const uint4* __restrict__ y2_u4,
    const int32_t* __restrict__ perm,
    uint4* __restrict__ down_u4,
    int n_local,
    int H8) {
  const int row = blockIdx.x;
  if (row >= n_local) return;
  const int dst_row = perm[row];
  const uint4* src = y2_u4 + static_cast<size_t>(row) * static_cast<size_t>(H8);
  uint4* dst = down_u4 + static_cast<size_t>(dst_row) * static_cast<size_t>(H8);
  for (int c = threadIdx.x; c < H8; c += blockDim.x) {
    dst[c] = src[c];
  }
}

// ---------------------------------------------------------------------------
// 5. Helpers & Launch Wrappers
// ---------------------------------------------------------------------------
struct DevState {
  bool inited = false;
  int32_t* h_meta_pinned = nullptr;
};

std::mutex g_dev_mu;
DevState g_dev_state[MAX_DEVS];

DevState& get_dev_state(int dev) {
  TORCH_CHECK(dev >= 0 && dev < MAX_DEVS, "Device index out of range: ", dev);
  std::lock_guard<std::mutex> lk(g_dev_mu);
  DevState& st = g_dev_state[dev];
  if (!st.inited) {
    CUDA_CHECK(cudaMallocHost(&st.h_meta_pinned, (2 * MAX_E_LOCAL + 2) * sizeof(int32_t)));
    st.inited = true;
  }
  return st;
}

inline int pick_row_threads(int n_vecs) {
  if (n_vecs <= 288 && (n_vecs % 32 == 0)) return n_vecs;
  return 256;
}

inline void cublas_tn_gemm_f16(
    cublasHandle_t handle,
    const __half* a_row_mk,
    const __half* b_row_nk,
    __half* c_row_mn,
    int M,
    int N,
    int K) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      N, M, K,
      &alpha,
      b_row_nk, CUDA_R_16F, K,
      a_row_mk, CUDA_R_16F, K,
      &beta,
      c_row_mn, CUDA_R_16F, N,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <int BM, int MIN_BLOCKS>
void launch_hmma_single_bn64(
    const __half* x_ptr,
    const uint8_t* w13_ptr,
    const uint8_t* s13_ptr,
    const float* g13_ptr,
    const uint8_t* w2_ptr,
    const uint8_t* s2_ptr,
    const float* g2_ptr,
    const int32_t* perm_ptr,
    const int32_t* gids_ptr,
    const int32_t* goff_ptr,
    const float* wt_ptr,
    int32_t* inv_perm_ptr,
    int4* tile_desc_ptr,
    int32_t* tile_count_ptr,
    __half* inter_ptr,
    __half* y2_ptr,
    __half* out_ptr,
    int T,
    int H,
    int I,
    int N13,
    int S,
    int max_g,
    int topk,
    float gate_cap,
    float up_cap,
    bool combine,
    cudaStream_t stream) {
  const int max_tiles = std::min(S, S / BM + max_g);
  const int prep_blocks = std::max(1, (S + 255) / 256);
  if (combine) {
    CUDA_CHECK(cudaMemsetAsync(inv_perm_ptr, 0xFF, static_cast<size_t>(S) * sizeof(int32_t), stream));
  }
  k_prepare_prefill_tiles_single<<<prep_blocks, 256, 0, stream>>>(
      perm_ptr, gids_ptr, goff_ptr, combine ? inv_perm_ptr : nullptr, tile_desc_ptr, tile_count_ptr, max_g, BM);

  const dim3 grid13((N13 + 63) / 64, max_tiles, 1);
  k_nvfp4_hmma_bn64<BM, 1, 0, MIN_BLOCKS><<<grid13, 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr),
      reinterpret_cast<const uint2*>(w13_ptr),
      s13_ptr, g13_ptr, perm_ptr, wt_ptr, tile_desc_ptr, tile_count_ptr,
      reinterpret_cast<uint4*>(inter_ptr), N13, H, topk, gate_cap, up_cap, 0);

  const dim3 grid2((H + 63) / 64, max_tiles, 1);
  __half* w2_dst = combine ? y2_ptr : out_ptr;
  k_nvfp4_hmma_bn64<BM, 0, 1, MIN_BLOCKS><<<grid2, 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(inter_ptr),
      reinterpret_cast<const uint2*>(w2_ptr),
      s2_ptr, g2_ptr, perm_ptr, wt_ptr, tile_desc_ptr, tile_count_ptr,
      reinterpret_cast<uint4*>(w2_dst), H, I, topk, 0.0f, 0.0f, combine ? 0 : 1);

  if (combine) {
    const int H8 = H / 8;
    const int comb_th = pick_row_threads(H8);
    k_combine_from_sorted<<<T, comb_th, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2_ptr), inv_perm_ptr,
        reinterpret_cast<uint4*>(out_ptr), T, topk, H8);
  }
}

template <int WK>
void launch_hmma_single_bm8(
    const __half* x_ptr,
    const uint8_t* w13_ptr,
    const uint8_t* s13_ptr,
    const float* g13_ptr,
    const uint8_t* w2_ptr,
    const uint8_t* s2_ptr,
    const float* g2_ptr,
    const int32_t* perm_ptr,
    const int32_t* gids_ptr,
    const int32_t* goff_ptr,
    const float* wt_ptr,
    int32_t* inv_perm_ptr,
    int4* tile_desc_ptr,
    int32_t* tile_count_ptr,
    __half* inter_ptr,
    __half* y2_ptr,
    __half* out_ptr,
    int T,
    int H,
    int I,
    int N13,
    int S,
    int max_g,
    int topk,
    float gate_cap,
    float up_cap,
    bool combine,
    cudaStream_t stream) {
  const int max_tiles = std::min(S, S / 8 + max_g);
  const int prep_blocks = std::max(1, (S + 255) / 256);
  if (combine) {
    CUDA_CHECK(cudaMemsetAsync(inv_perm_ptr, 0xFF, static_cast<size_t>(S) * sizeof(int32_t), stream));
  }
  k_prepare_prefill_tiles_single<<<prep_blocks, 256, 0, stream>>>(
      perm_ptr, gids_ptr, goff_ptr, combine ? inv_perm_ptr : nullptr, tile_desc_ptr, tile_count_ptr, max_g, 8);

  const dim3 grid13(N13 / 32, max_tiles, 1);
  k_nvfp4_hmma_bm8<1, 0, WK><<<grid13, 32 * WK, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr),
      reinterpret_cast<const uint2*>(w13_ptr),
      s13_ptr, g13_ptr, perm_ptr, wt_ptr, tile_desc_ptr, tile_count_ptr,
      reinterpret_cast<uint4*>(inter_ptr), N13, H, topk, gate_cap, up_cap, 0);

  const dim3 grid2(H / 32, max_tiles, 1);
  __half* w2_dst = combine ? y2_ptr : out_ptr;
  k_nvfp4_hmma_bm8<0, 1, WK><<<grid2, 32 * WK, 0, stream>>>(
      reinterpret_cast<const uint4*>(inter_ptr),
      reinterpret_cast<const uint2*>(w2_ptr),
      s2_ptr, g2_ptr, perm_ptr, wt_ptr, tile_desc_ptr, tile_count_ptr,
      reinterpret_cast<uint4*>(w2_dst), H, I, topk, 0.0f, 0.0f, combine ? 0 : 1);

  if (combine) {
    const int H8 = H / 8;
    const int comb_th = pick_row_threads(H8);
    k_combine_from_sorted<<<T, comb_th, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2_ptr), inv_perm_ptr,
        reinterpret_cast<uint4*>(out_ptr), T, topk, H8);
  }
}

void launch_hmma_small_2b(
    const __half* x_ptr,
    const uint8_t* w13_ptr,
    const uint8_t* s13_ptr,
    const float* g13_ptr,
    const uint8_t* w2_ptr,
    const uint8_t* s2_ptr,
    const float* g2_ptr,
    const int32_t* perm_ptr,
    const int32_t* gids_ptr,
    const int32_t* goff_ptr,
    const float* wt_ptr,
    int32_t* inv_perm_ptr,
    int4* tile_desc_base,
    int32_t* counts2,
    __half* inter_ptr,
    __half* y2_ptr,
    __half* out_ptr,
    int T,
    int H,
    int I,
    int N13,
    int S,
    int max_g,
    int topk,
    float gate_cap,
    float up_cap,
    bool combine,
    cudaStream_t stream) {
  const int max_16 = std::min(S, S / 9 + 1);
  const int max_8 = max_g;
  int4* tiles_16 = tile_desc_base;
  int4* tiles_8 = tiles_16 + max_16;
  const int prep_blocks = std::max(1, (S + 255) / 256);
  if (combine) {
    CUDA_CHECK(cudaMemsetAsync(inv_perm_ptr, 0xFF, static_cast<size_t>(S) * sizeof(int32_t), stream));
  }
  k_prepare_small_tiles_2b<<<prep_blocks, 256, 0, stream>>>(
      perm_ptr, gids_ptr, goff_ptr, combine ? inv_perm_ptr : nullptr, tiles_16, tiles_8, counts2, max_g);

  if (max_16 > 0) {
    k_nvfp4_hmma_bn64<16, 1, 0, 8><<<dim3((N13 + 63) / 64, max_16, 1), 128, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_ptr), reinterpret_cast<const uint2*>(w13_ptr), s13_ptr, g13_ptr,
        perm_ptr, wt_ptr, tiles_16, counts2 + 0, reinterpret_cast<uint4*>(inter_ptr),
        N13, H, topk, gate_cap, up_cap, 0);
  }
  k_nvfp4_hmma_bm8<1, 0, 4><<<dim3(N13 / 32, max_8, 1), 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr), reinterpret_cast<const uint2*>(w13_ptr), s13_ptr, g13_ptr,
      perm_ptr, wt_ptr, tiles_8, counts2 + 1, reinterpret_cast<uint4*>(inter_ptr),
      N13, H, topk, gate_cap, up_cap, 0);

  __half* w2_dst = combine ? y2_ptr : out_ptr;
  const int scat = combine ? 0 : 1;
  if (max_16 > 0) {
    k_nvfp4_hmma_bn64<16, 0, 1, 8><<<dim3((H + 63) / 64, max_16, 1), 128, 0, stream>>>(
        reinterpret_cast<const uint4*>(inter_ptr), reinterpret_cast<const uint2*>(w2_ptr), s2_ptr, g2_ptr,
        perm_ptr, wt_ptr, tiles_16, counts2 + 0, reinterpret_cast<uint4*>(w2_dst),
        H, I, topk, 0.0f, 0.0f, scat);
  }
  k_nvfp4_hmma_bm8<0, 1, 4><<<dim3(H / 32, max_8, 1), 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(inter_ptr), reinterpret_cast<const uint2*>(w2_ptr), s2_ptr, g2_ptr,
      perm_ptr, wt_ptr, tiles_8, counts2 + 1, reinterpret_cast<uint4*>(w2_dst),
      H, I, topk, 0.0f, 0.0f, scat);

  if (combine) {
    const int H8 = H / 8;
    const int comb_th = pick_row_threads(H8);
    k_combine_from_sorted<<<T, comb_th, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2_ptr), inv_perm_ptr,
        reinterpret_cast<uint4*>(out_ptr), T, topk, H8);
  }
}

void launch_hmma_exact_fit_multi(
    const __half* x_ptr,
    const uint8_t* w13_ptr,
    const uint8_t* s13_ptr,
    const float* g13_ptr,
    const uint8_t* w2_ptr,
    const uint8_t* s2_ptr,
    const float* g2_ptr,
    const int32_t* perm_ptr,
    const int32_t* gids_ptr,
    const int32_t* goff_ptr,
    const float* wt_ptr,
    int32_t* inv_perm_ptr,
    int4* tile_desc_base,
    int32_t* counts3,
    __half* inter_ptr,
    __half* y2_ptr,
    __half* out_ptr,
    int T,
    int H,
    int I,
    int N13,
    int S,
    int max_g,
    int topk,
    float gate_cap,
    float up_cap,
    bool combine,
    cudaStream_t stream) {
  const int max_64 = std::min(S, S / 33 + 1);
  const int max_32 = max_g;
  const int max_16 = max_g;
  int4* tiles_64 = tile_desc_base;
  int4* tiles_32 = tiles_64 + max_64;
  int4* tiles_16 = tiles_32 + max_32;

  const int prep_blocks = std::max(1, (S + 255) / 256);
  if (combine) {
    CUDA_CHECK(cudaMemsetAsync(inv_perm_ptr, 0xFF, static_cast<size_t>(S) * sizeof(int32_t), stream));
  }
  k_prepare_prefill_tiles_multi<<<prep_blocks, 256, 0, stream>>>(
      perm_ptr, gids_ptr, goff_ptr, combine ? inv_perm_ptr : nullptr,
      tiles_64, tiles_32, tiles_16, counts3, max_g);

  // w13 + fused SiTU across all 3 exact-fit buckets
  const int gx13 = (N13 + 63) / 64;
  if (max_64 > 0) {
    k_nvfp4_hmma_bn64<64, 1, 0, 5><<<dim3(gx13, max_64, 1), 128, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_ptr), reinterpret_cast<const uint2*>(w13_ptr), s13_ptr, g13_ptr,
        perm_ptr, wt_ptr, tiles_64, counts3 + 0, reinterpret_cast<uint4*>(inter_ptr),
        N13, H, topk, gate_cap, up_cap, 0);
  }
  k_nvfp4_hmma_bn64<32, 1, 0, 6><<<dim3(gx13, max_32, 1), 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr), reinterpret_cast<const uint2*>(w13_ptr), s13_ptr, g13_ptr,
      perm_ptr, wt_ptr, tiles_32, counts3 + 1, reinterpret_cast<uint4*>(inter_ptr),
      N13, H, topk, gate_cap, up_cap, 0);
  k_nvfp4_hmma_bn64<16, 1, 0, 8><<<dim3(gx13, max_16, 1), 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_ptr), reinterpret_cast<const uint2*>(w13_ptr), s13_ptr, g13_ptr,
      perm_ptr, wt_ptr, tiles_16, counts3 + 2, reinterpret_cast<uint4*>(inter_ptr),
      N13, H, topk, gate_cap, up_cap, 0);

  // w2 down projection across all 3 exact-fit buckets
  __half* w2_dst = combine ? y2_ptr : out_ptr;
  const int scat = combine ? 0 : 1;
  const int gx2 = (H + 63) / 64;
  if (max_64 > 0) {
    k_nvfp4_hmma_bn64<64, 0, 1, 5><<<dim3(gx2, max_64, 1), 128, 0, stream>>>(
        reinterpret_cast<const uint4*>(inter_ptr), reinterpret_cast<const uint2*>(w2_ptr), s2_ptr, g2_ptr,
        perm_ptr, wt_ptr, tiles_64, counts3 + 0, reinterpret_cast<uint4*>(w2_dst),
        H, I, topk, 0.0f, 0.0f, scat);
  }
  k_nvfp4_hmma_bn64<32, 0, 1, 6><<<dim3(gx2, max_32, 1), 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(inter_ptr), reinterpret_cast<const uint2*>(w2_ptr), s2_ptr, g2_ptr,
      perm_ptr, wt_ptr, tiles_32, counts3 + 1, reinterpret_cast<uint4*>(w2_dst),
      H, I, topk, 0.0f, 0.0f, scat);
  k_nvfp4_hmma_bn64<16, 0, 1, 8><<<dim3(gx2, max_16, 1), 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(inter_ptr), reinterpret_cast<const uint2*>(w2_ptr), s2_ptr, g2_ptr,
      perm_ptr, wt_ptr, tiles_16, counts3 + 2, reinterpret_cast<uint4*>(w2_dst),
      H, I, topk, 0.0f, 0.0f, scat);

  if (combine) {
    const int H8 = H / 8;
    const int comb_th = pick_row_threads(H8);
    k_combine_from_sorted<<<T, comb_th, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2_ptr), inv_perm_ptr,
        reinterpret_cast<uint4*>(out_ptr), T, topk, H8);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Exported Python bindings
// ---------------------------------------------------------------------------

void nvfp4_qpn_dequant_expert(
    torch::Tensor w_qpn,
    torch::Tensor s_qpn,
    torch::Tensor g_t,
    int64_t expert_idx,
    torch::Tensor out,
    bool use_direct) {
  TORCH_CHECK(w_qpn.is_cuda() && w_qpn.is_contiguous() && w_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(s_qpn.is_cuda() && s_qpn.is_contiguous() && s_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(g_t.is_cuda() && g_t.is_contiguous() && g_t.dtype() == torch::kFloat32);
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat16 && out.dim() == 2);

  const int N = static_cast<int>(out.size(0));
  const int K = static_cast<int>(out.size(1));
  const int E = static_cast<int>(g_t.numel());
  TORCH_CHECK(expert_idx >= 0 && expert_idx < E, "expert_idx out of range");
  TORCH_CHECK(N % 32 == 0 && K % 16 == 0, "Expected N % 32 == 0 and K % 16 == 0");
  TORCH_CHECK(w_qpn.numel() == static_cast<int64_t>(E) * N * (K / 2), "w_qpn numel mismatch");
  TORCH_CHECK(s_qpn.numel() == static_cast<int64_t>(E) * N * (K / 16), "s_qpn numel mismatch");

  const at::cuda::CUDAGuard guard(out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_nvfp4_qpn_dequant(
      w_qpn.data_ptr<uint8_t>(), s_qpn.data_ptr<uint8_t>(), g_t.data_ptr<float>(),
      nullptr, static_cast<int>(expert_idx), 1,
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), N, K, use_direct, stream);
  CUDA_CHECK(cudaGetLastError());
}

void nvfp4_qpn_dequant_batch(
    torch::Tensor w_qpn,
    torch::Tensor s_qpn,
    torch::Tensor g_t,
    torch::Tensor expert_ids_dev,
    int64_t n_experts,
    torch::Tensor out_buf,
    bool use_direct) {
  TORCH_CHECK(w_qpn.is_cuda() && w_qpn.is_contiguous() && w_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(s_qpn.is_cuda() && s_qpn.is_contiguous() && s_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(g_t.is_cuda() && g_t.is_contiguous() && g_t.dtype() == torch::kFloat32);
  TORCH_CHECK(expert_ids_dev.is_cuda() && expert_ids_dev.is_contiguous() && expert_ids_dev.dtype() == torch::kInt32);
  TORCH_CHECK(out_buf.is_cuda() && out_buf.is_contiguous() && out_buf.dtype() == torch::kFloat16 && out_buf.dim() == 3);
  if (n_experts <= 0) return;

  const int N = static_cast<int>(out_buf.size(1));
  const int K = static_cast<int>(out_buf.size(2));
  const int E = static_cast<int>(g_t.numel());
  TORCH_CHECK(out_buf.size(0) >= n_experts && expert_ids_dev.numel() >= n_experts);
  TORCH_CHECK(N % 32 == 0 && K % 16 == 0, "Expected N % 32 == 0 and K % 16 == 0");
  TORCH_CHECK(w_qpn.numel() == static_cast<int64_t>(E) * N * (K / 2), "w_qpn numel mismatch");
  TORCH_CHECK(s_qpn.numel() == static_cast<int64_t>(E) * N * (K / 16), "s_qpn numel mismatch");

  const at::cuda::CUDAGuard guard(out_buf.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_nvfp4_qpn_dequant(
      w_qpn.data_ptr<uint8_t>(), s_qpn.data_ptr<uint8_t>(), g_t.data_ptr<float>(),
      expert_ids_dev.data_ptr<int32_t>(), 0, static_cast<int>(n_experts),
      reinterpret_cast<__half*>(out_buf.data_ptr<at::Half>()), N, K, use_direct, stream);
  CUDA_CHECK(cudaGetLastError());
}

void nvfp4_route_from_ids(
    torch::Tensor ids,
    torch::Tensor perm,
    torch::Tensor gids,
    torch::Tensor goff,
    torch::Tensor scratch_counts,
    torch::Tensor scratch_base,
    torch::Tensor scratch_rank,
    int64_t E_local) {
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.dtype() == torch::kInt32);
  const int S = static_cast<int>(ids.numel());
  if (S == 0) return;
  TORCH_CHECK(E_local > 0 && E_local <= MAX_E_LOCAL);
  const int max_g = std::min(S, static_cast<int>(E_local));

  const at::cuda::CUDAGuard guard(ids.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (S + 255) / 256;
  k_route_ids_count<<<blocks, 256, 0, stream>>>(
      ids.data_ptr<int32_t>(), scratch_rank.data_ptr<int32_t>(), scratch_counts.data_ptr<int32_t>(), S);
  k_route_ids_scan<<<1, 64, 0, stream>>>(
      scratch_counts.data_ptr<int32_t>(), scratch_base.data_ptr<int32_t>(),
      gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(), static_cast<int>(E_local), max_g);
  k_route_ids_scatter_perm<<<blocks, 256, 0, stream>>>(
      ids.data_ptr<int32_t>(), scratch_rank.data_ptr<int32_t>(), scratch_base.data_ptr<int32_t>(),
      perm.data_ptr<int32_t>(), S);
  CUDA_CHECK(cudaGetLastError());
}

// Main prefill MoE runner.
// Modes:
//   0 (default): Optimal zero-sync HMMA.884 auto-dispatch:
//                  T <= 192: single 16x64
//                  T <= 448: single 32x64
//                  T >  448: exact-fit 3-bucket (64x64 + 32x64 + 16x64)
//   1:           Single-BM HMMA.884 16x64
//   2:           Single-BM HMMA.884 32x64
//   3:           Single-BM HMMA.884 64x64
//   4:           Forced exact-fit 3-bucket HMMA.884 (64x64 + 32x64 + 16x64)
//   5:           Dequant + FP16 cuBLAS TN GEMM two-phase (reference / ablation)
void nvfp4_prefill_moe_forward(
    torch::Tensor x,
    torch::Tensor w13_qpn,
    torch::Tensor s13_qpn,
    torch::Tensor g13_t,
    torch::Tensor w2_qpn,
    torch::Tensor s2_qpn,
    torch::Tensor g2_t,
    torch::Tensor perm,
    torch::Tensor gids,
    torch::Tensor goff,
    torch::Tensor weights,
    torch::Tensor w_buf0,
    torch::Tensor tile_desc_buf,
    torch::Tensor tile_count_buf,
    torch::Tensor inter_scratch,
    torch::Tensor y2_scratch,
    torch::Tensor inv_perm_scratch,
    torch::Tensor out,
    int64_t topk,
    double gate_cap,
    double up_cap,
    bool combine,
    int64_t mode,
    bool use_direct_dequant) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat16 && x.dim() == 2);
  TORCH_CHECK(w13_qpn.is_cuda() && w13_qpn.is_contiguous() && w13_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(s13_qpn.is_cuda() && s13_qpn.is_contiguous() && s13_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(g13_t.is_cuda() && g13_t.is_contiguous() && g13_t.dtype() == torch::kFloat32);
  TORCH_CHECK(w2_qpn.is_cuda() && w2_qpn.is_contiguous() && w2_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(s2_qpn.is_cuda() && s2_qpn.is_contiguous() && s2_qpn.dtype() == torch::kUInt8);
  TORCH_CHECK(g2_t.is_cuda() && g2_t.is_contiguous() && g2_t.dtype() == torch::kFloat32);
  TORCH_CHECK(perm.is_cuda() && perm.is_contiguous() && perm.dtype() == torch::kInt32);
  TORCH_CHECK(gids.is_cuda() && gids.is_contiguous() && gids.dtype() == torch::kInt32);
  TORCH_CHECK(goff.is_cuda() && goff.is_contiguous() && goff.dtype() == torch::kInt32);
  TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() && weights.dtype() == torch::kFloat32);
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat16);

  const int T = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  if (T == 0) return;

  const int E = static_cast<int>(g13_t.numel());
  const int N13 = static_cast<int>(w13_qpn.numel() / (static_cast<int64_t>(E) * (H / 2)));
  const int I = N13 / 2;
  const int S = T * static_cast<int>(topk);
  const int max_g = std::min(S, E);

  TORCH_CHECK(E > 0 && E <= MAX_E_LOCAL, "E out of range: ", E);
  TORCH_CHECK(topk > 0 && topk <= MAX_TOPK, "topk out of range: ", topk);
  TORCH_CHECK(H % 64 == 0 && I % 64 == 0 && N13 % 32 == 0, "Expected H%64==0, I%64==0, N13%32==0");

  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint8_t* w13_ptr = w13_qpn.data_ptr<uint8_t>();
  const uint8_t* s13_ptr = s13_qpn.data_ptr<uint8_t>();
  const float* g13_ptr = g13_t.data_ptr<float>();
  const uint8_t* w2_ptr = w2_qpn.data_ptr<uint8_t>();
  const uint8_t* s2_ptr = s2_qpn.data_ptr<uint8_t>();
  const float* g2_ptr = g2_t.data_ptr<float>();
  const int32_t* perm_ptr = perm.data_ptr<int32_t>();
  const int32_t* gids_ptr = gids.data_ptr<int32_t>();
  const int32_t* goff_ptr = goff.data_ptr<int32_t>();
  const float* wt_ptr = weights.data_ptr<float>();

  if ((mode >= 0 && mode <= 4) || mode == 6 || mode == 7 || mode == 8) {
    TORCH_CHECK(tile_count_buf.numel() >= 4, "tile_count_buf too small");
    TORCH_CHECK(inter_scratch.numel() >= static_cast<int64_t>(S) * I, "inter_scratch too small");
    if (combine) {
      TORCH_CHECK(y2_scratch.numel() >= static_cast<int64_t>(S) * H, "y2_scratch too small");
      TORCH_CHECK(inv_perm_scratch.numel() >= S, "inv_perm_scratch too small");
    }

    __half* x_h = reinterpret_cast<__half*>(x.data_ptr<at::Half>());
    __half* inter_h = reinterpret_cast<__half*>(inter_scratch.data_ptr<at::Half>());
    __half* y2_h = combine ? reinterpret_cast<__half*>(y2_scratch.data_ptr<at::Half>()) : nullptr;
    __half* out_h = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    int32_t* inv_perm_ptr = combine ? inv_perm_scratch.data_ptr<int32_t>() : nullptr;
    int4* tile_desc_ptr = reinterpret_cast<int4*>(tile_desc_buf.data_ptr<int32_t>());
    int32_t* tile_count_ptr = tile_count_buf.data_ptr<int32_t>();

    int eff_mode = static_cast<int>(mode);
    if (eff_mode == 0) {
      eff_mode = (T <= 4) ? 8 : ((T <= 256) ? 6 : ((T <= 512) ? 2 : 4));
    }

    if (eff_mode == 6) {
      launch_hmma_single_bm8<4>(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    } else if (eff_mode == 8) {
      launch_hmma_single_bm8<6>(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    } else if (eff_mode == 7) {
      const int need_tiles = std::min(S, S / 9 + 1) + max_g;
      TORCH_CHECK(tile_desc_buf.numel() >= static_cast<int64_t>(need_tiles) * 4, "tile_desc_buf too small");
      launch_hmma_small_2b(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    } else if (eff_mode == 4) {
      const int need_tiles = std::min(S, S / 33 + 1) + 2 * max_g;
      TORCH_CHECK(tile_desc_buf.numel() >= static_cast<int64_t>(need_tiles) * 4, "tile_desc_buf too small");
      launch_hmma_exact_fit_multi(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    } else if (eff_mode == 1) {
      launch_hmma_single_bn64<16, 8>(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    } else if (eff_mode == 2) {
      launch_hmma_single_bn64<32, 6>(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    } else {
      launch_hmma_single_bn64<64, 5>(
          x_h, w13_ptr, s13_ptr, g13_ptr, w2_ptr, s2_ptr, g2_ptr,
          perm_ptr, gids_ptr, goff_ptr, wt_ptr, inv_perm_ptr, tile_desc_ptr, tile_count_ptr,
          inter_h, y2_h, out_h, T, H, I, N13, S, max_g, static_cast<int>(topk),
          static_cast<float>(gate_cap), static_cast<float>(up_cap), combine, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // Mode 5: Reference dequant + FP16 cuBLAS TN GEMM two-phase path.
  TORCH_CHECK(w_buf0.is_cuda() && w_buf0.is_contiguous() && w_buf0.dtype() == torch::kFloat16);
  TORCH_CHECK(w_buf0.numel() >= static_cast<int64_t>(N13) * H, "w_buf0 too small");

  const int dev = x.device().index();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  DevState& st = get_dev_state(dev);

  int32_t* h_gids = st.h_meta_pinned;
  int32_t* h_goff = st.h_meta_pinned + MAX_E_LOCAL;
  CUDA_CHECK(cudaMemcpyAsync(h_gids, gids_ptr, max_g * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_goff, goff_ptr, (max_g + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int act_e[MAX_E_LOCAL];
  int act_off[MAX_E_LOCAL];
  int act_m[MAX_E_LOCAL];
  int n_act = 0;
  const int n_local = h_goff[max_g];
  for (int g = 0; g < max_g; g++) {
    const int m = h_goff[g + 1] - h_goff[g];
    if (m > 0) {
      act_e[n_act] = h_gids[g];
      act_off[n_act] = h_goff[g];
      act_m[n_act] = m;
      n_act++;
    }
  }
  if (n_act == 0 || n_local == 0) {
    if (combine) {
      CUDA_CHECK(cudaMemsetAsync(out.data_ptr<at::Half>(), 0, static_cast<size_t>(T) * H * sizeof(at::Half), stream));
    }
    return;
  }

  auto opts_f16 = x.options();
  auto opts_i32 = perm.options();
  torch::Tensor x_sorted = torch::empty({n_local, H}, opts_f16);
  torch::Tensor y13_sorted = torch::empty({n_local, N13}, opts_f16);
  torch::Tensor inter_sorted = torch::empty({n_local, I}, opts_f16);
  torch::Tensor y2_sorted = torch::empty({n_local, H}, opts_f16);
  torch::Tensor inv_perm;
  int32_t* inv_perm_ptr = nullptr;
  if (combine) {
    inv_perm = torch::empty({S}, opts_i32);
    inv_perm_ptr = inv_perm.data_ptr<int32_t>();
    CUDA_CHECK(cudaMemsetAsync(inv_perm_ptr, 0xFF, static_cast<size_t>(S) * sizeof(int32_t), stream));
  }

  __half* x_sorted_h = reinterpret_cast<__half*>(x_sorted.data_ptr<at::Half>());
  __half* y13_h = reinterpret_cast<__half*>(y13_sorted.data_ptr<at::Half>());
  __half* inter_h = reinterpret_cast<__half*>(inter_sorted.data_ptr<at::Half>());
  __half* y2_h = reinterpret_cast<__half*>(y2_sorted.data_ptr<at::Half>());
  __half* wb0 = reinterpret_cast<__half*>(w_buf0.data_ptr<at::Half>());

  const int H8 = H / 8;
  const int I4 = I / 4;
  const int gather_th = pick_row_threads(H8);
  const int situ_th = pick_row_threads(I4);

  k_gather_tokens<<<n_local, gather_th, 0, stream>>>(
      reinterpret_cast<const uint4*>(x.data_ptr<at::Half>()),
      perm_ptr, inv_perm_ptr, reinterpret_cast<uint4*>(x_sorted_h), n_local, static_cast<int>(topk), H8);

  for (int g = 0; g < n_act; g++) {
    launch_nvfp4_qpn_dequant(
        w13_ptr, s13_ptr, g13_ptr, nullptr, act_e[g], 1, wb0, N13, H, use_direct_dequant, stream);
    cublas_tn_gemm_f16(
        handle, x_sorted_h + static_cast<size_t>(act_off[g]) * H, wb0,
        y13_h + static_cast<size_t>(act_off[g]) * N13, act_m[g], N13, H);
  }

  k_situ_interleaved_sorted<<<n_local, situ_th, 0, stream>>>(
      reinterpret_cast<const uint4*>(y13_h), wt_ptr, perm_ptr, reinterpret_cast<uint2*>(inter_h),
      n_local, I4, static_cast<float>(gate_cap), static_cast<float>(up_cap));

  for (int g = 0; g < n_act; g++) {
    launch_nvfp4_qpn_dequant(
        w2_ptr, s2_ptr, g2_ptr, nullptr, act_e[g], 1, wb0, H, I, use_direct_dequant, stream);
    cublas_tn_gemm_f16(
        handle, inter_h + static_cast<size_t>(act_off[g]) * I, wb0,
        y2_h + static_cast<size_t>(act_off[g]) * H, act_m[g], H, I);
  }

  if (combine) {
    k_combine_from_sorted<<<T, gather_th, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2_h), inv_perm_ptr,
        reinterpret_cast<uint4*>(out.data_ptr<at::Half>()), T, static_cast<int>(topk), H8);
  } else {
    k_scatter_rows<<<n_local, gather_th, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2_h), perm_ptr,
        reinterpret_cast<uint4*>(out.data_ptr<at::Half>()), n_local, H8);
  }
  CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_qpn_dequant_expert", &nvfp4_qpn_dequant_expert,
        "Bit-exact Volta sm_70 QPN NVFP4 -> FP16 [N, K] dequantization (single expert)");
  m.def("nvfp4_qpn_dequant_batch", &nvfp4_qpn_dequant_batch,
        "Bit-exact Volta sm_70 QPN NVFP4 -> FP16 [B, N, K] batched dequantization");
  m.def("nvfp4_route_from_ids", &nvfp4_route_from_ids,
        "Fast CUDA EP=4 routing (ids -> perm, gids, goff) for prefill MoE");
  m.def("nvfp4_prefill_moe_forward", &nvfp4_prefill_moe_forward,
        "Exact native NVFP4 prefill MoE: fused large-M Volta HMMA.884 + SiTU + combine (zero sync)");
}
