// SPDX-License-Identifier: Apache-2.0
// mlp_mxfp8_moe.cu - Exact native MXFP8 (UE8M0 x E4M3FN) grouped MoE kernels for Volta (sm_70).
//
// Targets Inkling-2-Mini EP=4 MXFP8 MoE layers (layers 20..23: E_local=32, H=2304, I=1152, topk=8)
// with zero host synchronization, 100% CUDA-graph safety, zero register spills, and exact native
// MXFP8 math (no lossy weight packing):
//
// 1. Exact QPN8 Weight Layout (in-place byte-equal permutation):
//    - W_qpn: [E, N/32, K/32, 2, 32, 16] uint8 (viewed as [E, N/32, K/32, 32, 32], CH8 = 1024 B
//      per (32-col tile, 32-k chunk)). Half 0 (+0, 512 B) holds k=0..15, half 1 (+512, 512 B) holds
//      k=16..31. Lane l holds col = 8*((l>>2)&3) + (l&3) + 4*(l>>4), bytes ordered
//      0,2,4,6,1,3,5,7, 8,10,12,14,9,11,13,15 so 1 __byte_perm + f8x2 unpacks exact E4M3 * 2^-8.
//    - S_qpn: [E, N/32, K/32, 32] uint8 (1 coalesced byte per (tile, chunk_k32, lane)).
//    - For w13 (GATED=1, CB=2): checkpoint interleaved (gate_0, up_0, ...) rows are prepacked as
//      gate tiles 0..I/32-1 followed by up tiles I/32..2*I/32-1, fusing w13 + exact FP32 softcapped
//      SiTU (gate_cap * tanh(g/gate_cap) * sigmoid(g) * up_cap * tanh(u/up_cap) * gamma) into
//      inter_act [slots, I] fp16.
//
// 2. Exact UE8M0 Scale Handling:
//    - Fast path (WIDE_SCALE=0, expert UE8M0 span <= 22 — bit-exact for all 254 E4M3FN codes
//      including subnormals, and up to span 29 within FP16 normal exponent range):
//      s_reb = s + shift_e in [105..134] (255 -> 255), decoded to exact FP16x2 power of two
//      ((s_reb - 104) << 10) duplicated across both halves (0 if s_reb==255), applied via hmul2
//      to unpacked p0..p3 before mma.m8n8k4, with g_scale[e] = 2^(-shift_e) in FP32 epilogue.
//    - General wide-span path (WIDE_SCALE=1, arbitrary UE8M0 span 0..254):
//      Unpacks exact E4M3 = f8x2 * 256.0h, runs chunk mma884 into tmp[8], shuffles the 4 column
//      scales across the warp via __shfl_sync, and accumulates exact FP32 UE8M0 2^(s-127) per chunk.
//
// 3. Zero-Sync EP=4 Grouped Dispatch:
//    - Consumes exact (perm, gids, goff) from ep4_route_to_skinny_groups(ids, E_local).
//    - CTA (bx, by) reads s0 = goff[by], s1 = goff[by+1]; exits immediately if s0 >= s1.
//    - Loops r0 = 0..m_e-1 in tiles of R8 to handle arbitrary m_e (decode T=1..32 and prefill T=128..4096+).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int CH8 = 1024;  // bytes per (32-column tile, 32-k chunk)
constexpr int MAX_DEVS = 32;

__device__ __forceinline__ void mma884(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16na(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

template <int CG>
__device__ __forceinline__ uint4 ldw16(const void* p) {
  if constexpr (CG) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
  } else {
    return ldg16na(p);
  }
}

template <int CG>
__device__ __forceinline__ unsigned lds1(const void* p) {
  unsigned v;
  if constexpr (CG) {
    asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(v) : "l"(p));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];" : "=r"(v) : "l"(p));
  }
  return v;
}

__device__ __forceinline__ uint4 lda16(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ void bar_sync_n(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// Bytes 0 and 2 of p (finite E4M3FN) -> fp16x2 = exact E4M3 * 2^-8 (normals & subnormals exact).
__device__ __forceinline__ unsigned f8x2(unsigned p) {
  return ((p << 8) & 0x80008000u) | ((p << 7) & 0x3F803F80u);
}

__device__ __forceinline__ unsigned hmul2u(unsigned a, unsigned b) {
  const __half2 ha = *reinterpret_cast<const __half2*>(&a);
  const __half2 hb = *reinterpret_cast<const __half2*>(&b);
  const __half2 r = __hmul2(ha, hb);
  return *reinterpret_cast<const unsigned*>(&r);
}

// Rebased UE8M0 byte s_reb in [105..134] -> fp16x2 power of two 2^(s_reb - 119); s_reb == 255 -> 0.
__device__ __forceinline__ unsigned ue8m0_reb_to_h2(unsigned s_reb) {
  const unsigned h = (s_reb - 104u) << 10;
  const unsigned h2 = h | (h << 16);
  return (s_reb == 255u) ? 0u : h2;
}

// Exact raw UE8M0 byte s in [0..255] -> FP32 2^(s - 127), exact for all 0..254 (0 is FP32 subnormal 2^-127), 255 -> 0.
__device__ __forceinline__ float ue8m0_to_f32(unsigned s) {
  if (s == 255u) return 0.0f;
  if (s == 0u) return __uint_as_float(0x00400000u);  // 2^-127
  return __uint_as_float(s << 23);
}

// Exact softcapped SiTU matching Inkling2 _situ_interleaved_kernel (sm_70-safe tanhf/expf).
__device__ __forceinline__ float softcap_situ(float g, float u, float gamma, float gate_cap, float up_cap) {
  const float sig = 1.0f / (1.0f + expf(-g));
  if (gate_cap > 0.0f) g = gate_cap * tanhf(g / gate_cap);
  if (up_cap > 0.0f) u = up_cap * tanhf(u / up_cap);
  return ((g * sig) * u) * gamma;
}

// Fast-path 32-k chunk: applies rebased UE8M0 fp16x2 power-of-two scale directly to unpacked E4M3*2^-8.
template <int RB, int CB, class AGet>
__device__ __forceinline__ void chunk8_mxfp8(const uint4 (&w)[CB][2], const unsigned (&sv)[CB], AGet aget,
                                             float (&acc)[RB][CB][8]) {
  unsigned sc[CB];
#pragma unroll
  for (int cb = 0; cb < CB; cb++) sc[cb] = ue8m0_reb_to_h2(sv[cb]);
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    uint4 a[RB];
#pragma unroll
    for (int rb = 0; rb < RB; rb++) a[rb] = aget(jj, rb);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const uint4& g = w[cb][jj >> 1];
      const unsigned qx = (jj & 1) ? g.z : g.x;
      const unsigned qy = (jj & 1) ? g.w : g.y;
      const unsigned p0 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0400)), sc[cb]);
      const unsigned p1 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0501)), sc[cb]);
      const unsigned p2 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0602)), sc[cb]);
      const unsigned p3 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0703)), sc[cb]);
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        mma884(acc[rb][cb], a[rb].x, a[rb].y, p0, p1);
        mma884(acc[rb][cb], a[rb].z, a[rb].w, p2, p3);
      }
    }
  }
}

// Exact general-path 32-k chunk for wide UE8M0 spans (up to full 0..254):
// Unpacks exact E4M3 (f8x2 * 256.0h), runs chunk mma884 into tmp[8], then scales the 4 output columns
// owned by each lane in acc[rb][cb][0..7] with exact FP32 2^(s - 127) shuffled from the B-fragment lanes.
template <int RB, int CB, class AGet>
__device__ __forceinline__ void chunk8_wide(const uint4 (&w)[CB][2], const unsigned (&sv)[CB], AGet aget,
                                            float (&acc)[RB][CB][8]) {
  const int lane = threadIdx.x & 31;
  const int lb_base = lane & 0x0E;
  float s0[CB], s1[CB], s4[CB], s5[CB];
#pragma unroll
  for (int cb = 0; cb < CB; cb++) {
    const float sf = ue8m0_to_f32(sv[cb]);
    s0[cb] = __shfl_sync(0xFFFFFFFFu, sf, lb_base);
    s1[cb] = __shfl_sync(0xFFFFFFFFu, sf, lb_base | 1);
    s4[cb] = __shfl_sync(0xFFFFFFFFu, sf, lb_base | 16);
    s5[cb] = __shfl_sync(0xFFFFFFFFu, sf, lb_base | 17);
  }
#pragma unroll
  for (int cb = 0; cb < CB; cb++) {
#pragma unroll
    for (int rb = 0; rb < RB; rb++) {
      float tmp[8];
#pragma unroll
      for (int e = 0; e < 8; e++) tmp[e] = 0.0f;
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const uint4 a = aget(jj, rb);
        const uint4& g = w[cb][jj >> 1];
        const unsigned qx = (jj & 1) ? g.z : g.x;
        const unsigned qy = (jj & 1) ? g.w : g.y;
        const unsigned p0 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0400)), 0x5C005C00u);
        const unsigned p1 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0501)), 0x5C005C00u);
        const unsigned p2 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0602)), 0x5C005C00u);
        const unsigned p3 = hmul2u(f8x2(__byte_perm(qx, qy, 0x0703)), 0x5C005C00u);
        mma884(tmp, a.x, a.y, p0, p1);
        mma884(tmp, a.z, a.w, p2, p3);
      }
      acc[rb][cb][0] += tmp[0] * s0[cb];
      acc[rb][cb][2] += tmp[2] * s0[cb];
      acc[rb][cb][1] += tmp[1] * s1[cb];
      acc[rb][cb][3] += tmp[3] * s1[cb];
      acc[rb][cb][4] += tmp[4] * s4[cb];
      acc[rb][cb][6] += tmp[6] * s4[cb];
      acc[rb][cb][5] += tmp[5] * s5[cb];
      acc[rb][cb][7] += tmp[7] * s5[cb];
    }
  }
}

template <int RB, int CB, int WK, int NCOL, int NROW, int PR = 1>
struct KM8 {
  static constexpr int NCW = WK * NCOL * NROW;
  static constexpr int RBT = RB * NROW;
  static constexpr int R8 = RBT * 8;
  static constexpr int NR = RB * CB * 8;
  static constexpr int TS = NROW * NR * 32;
  static constexpr int AB = RBT * 512;
  static constexpr int REDB = NCOL * TS * 4 * (PR ? WK : 1);
  static constexpr int RING = WK * 2 * AB;
  static constexpr int SMEM = REDB > RING ? REDB : RING;
  static constexpr int NTH = 32 * NCW;
};

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int PR, int CG, int GATED, int DIRECT_IN, int WIDE_SCALE>
__global__ void __launch_bounds__(KM8<RB, CB, WK, NCOL, NROW, PR>::NTH, MB)
k_mxfp8_moe(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ g_scale,
            const __half* __restrict__ X, const int32_t* __restrict__ perm, const int32_t* __restrict__ gids,
            const int32_t* __restrict__ goff, const float* __restrict__ weights, __half* __restrict__ Y,
            int N_out, int K, int topk, float gate_cap, float up_cap, int max_groups, int zero) {
  using P = KM8<RB, CB, WK, NCOL, NROW, PR>;
  static_assert(WK <= 14 && (RB * NROW) % (NCOL * NROW) == 0, "named barriers 1..WK, even staging pieces");
  static_assert(!GATED || CB == 2, "GATED requires CB == 2 (gate + up)");
  static_assert(GATED || CB == 1, "non-GATED uses CB == 1");

  constexpr int NCW = P::NCW;
  constexpr int R8 = P::R8;
  constexpr int NR = P::NR;
  constexpr int AB = P::AB;
  constexpr int G = NCOL * NROW;
  constexpr int NPC = P::RBT / G;
  constexpr int SLAB = NCOL * P::TS;

  extern __shared__ __align__(16) uint8_t sm[];
  __shared__ int sh_slot[R8];
  __shared__ int sh_xrow[R8];
  __shared__ float sh_gamma[GATED ? R8 : 1];

  float* red = reinterpret_cast<float*>(sm);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int NC = K / 32;
  const int NT_OUT = N_out / 32;
  const int NT_TOTAL = GATED ? (2 * NT_OUT) : NT_OUT;
  const int rin = (lane & 3) + 4 * (lane >> 4);
  const int wk = warp % WK;
  const int gi = (warp / WK) % NCOL;
  const int gr = warp / (WK * NCOL);
  const int bx = blockIdx.x;
  const int t = bx * NCOL + gi;
  const int s0_k = NC * wk / WK;
  const int n_k = NC * (wk + 1) / WK - s0_k;
  const int gidx = gr * NCOL + gi;
  const unsigned zmask = (unsigned)zero;

  uint8_t* abuf = sm + (size_t)wk * 2 * AB;
  const size_t cbs_w = GATED ? (size_t)NT_OUT * NC * CH8 : (size_t)NC * CH8;
  const size_t cbs_s = GATED ? (size_t)NT_OUT * NC * 32 : (size_t)NC * 32;

  auto slice_bar = [&]() {
    if constexpr (G > 1) {
      bar_sync_n(1 + wk, G * 32);
    } else {
      __syncwarp();
    }
  };

  auto rv = [&](int idx) {
    if constexpr (PR) {
      float v = red[idx];
#pragma unroll
      for (int w = 1; w < WK; w++) v += red[w * SLAB + idx];
      return v;
    } else {
      return red[idx];
    }
  };

  auto ridx = [&](int c, int row, int col) {
    const int grr = (NROW == 1) ? 0 : (row / (RB * 8));
    const int lrow = (NROW == 1) ? row : (row % (RB * 8));
    const int rb = lrow >> 3;
    const int r8 = lrow & 7;
    const int cb = col >> 5;
    const int cc = col & 31;
    const int q = cc >> 3;
    const int cq = cc & 7;
    const int l = (r8 & 1) | (cq & 2) | (q << 2) | (((r8 >> 2) & 1) << 4);
    const int e_idx = (r8 & 2) | (cq & 4) | (cq & 1);
    return (c * NROW + grr) * NR * 32 + ((rb * CB + cb) * 8 + e_idx) * 32 + l;
  };

  for (int by = blockIdx.y; by < max_groups; by += gridDim.y) {
    const int s0 = goff[by];
    const int s1 = goff[by + 1];
    if (s0 >= s1) continue;
    const int expert = gids[by];
    const int m_e = s1 - s0;
    const float gs = WIDE_SCALE ? 1.0f : g_scale[expert];

    const uint8_t* wp = W + (((size_t)expert * NT_TOTAL + t) * NC + s0_k) * CH8 + lane * 16;
    const uint8_t* sp = S + (((size_t)expert * NT_TOTAL + t) * NC + s0_k) * 32 + lane;

    for (int r0 = 0; r0 < m_e; r0 += R8) {
      if (r0 > 0) bar_sync_n(15, NCW * 32);  // ensure prior epilogue finished reading red & sh_*
      const int rows = min(R8, m_e - r0);
      if (threadIdx.x < R8) {
        const int r_clamped = min((int)threadIdx.x, rows - 1);
        const int sl = perm[s0 + r0 + r_clamped];
        sh_slot[threadIdx.x] = sl;
        sh_xrow[threadIdx.x] = DIRECT_IN ? sl : (sl / topk);
        if constexpr (GATED) {
          sh_gamma[threadIdx.x] = weights[sl];
        }
      }
      bar_sync_n(15, NCW * 32);

      // Cache this thread's NPC activation row offsets for the tile (invariant across K-chunks).
      int xrow_q[NPC];
#pragma unroll
      for (int q = 0; q < NPC; q++) {
        const int u = (q * G + gidx) * 32 + lane;
        xrow_q[q] = sh_xrow[u >> 2];
      }

      float acc[RB][CB][8];
#pragma unroll
      for (int i = 0; i < RB; i++)
#pragma unroll
        for (int j = 0; j < CB; j++)
#pragma unroll
          for (int e_i = 0; e_i < 8; e_i++) acc[i][j][e_i] = 0.0f;

      uint4 wA[CB][2], wB[CB][2], stg[NPC];
      unsigned svA[CB], svB[CB];
      unsigned xr = 0;

      auto ldw = [&](uint4 (&w)[CB][2], unsigned (&sv)[CB], int i, unsigned dep) {
        const uint8_t* pw = wp + (size_t)i * CH8 + dep;
        const uint8_t* ps = sp + (size_t)i * 32 + dep;
#pragma unroll
        for (int cb = 0; cb < CB; cb++) {
          w[cb][0] = ldw16<CG>(pw + cb * cbs_w);
          w[cb][1] = ldw16<CG>(pw + cb * cbs_w + 512);
          sv[cb] = lds1<CG>(ps + cb * cbs_s);
        }
      };

      auto ldx = [&](int i, unsigned dep) {
        const int ch = s0_k + i;
#pragma unroll
        for (int q = 0; q < NPC; q++) {
          const int u = (q * G + gidx) * 32 + lane;
          const int jj = u & 3;
          stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)xrow_q[q] * K + ch * 32 + jj * 8) + dep);
        }
      };

      auto stx = [&](int i) {
        uint8_t* pb = abuf + (i & 1) * AB;
#pragma unroll
        for (int q = 0; q < NPC; q++) {
          const int u = (q * G + gidx) * 32 + lane;
          *reinterpret_cast<uint4*>(pb + ((u & 3) * R8 + (u >> 2)) * 16) = stg[q];
        }
      };

      auto touch = [&](const uint4 (&w)[CB][2], const unsigned (&sv)[CB]) {
        unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
        for (int cb = 0; cb < CB; cb++) x ^= w[cb][0].x ^ w[cb][1].x ^ sv[cb];
#pragma unroll
        for (int q = 0; q < NPC; q++) x ^= stg[q].x;
        xr ^= x;
        return x & zmask;
      };

      auto step = [&](const uint4 (&wc)[CB][2], const unsigned (&sc_c)[CB], uint4 (&wn)[CB][2],
                      unsigned (&sc_n)[CB], int i, unsigned& dep) {
        const int inx = min(i + 1, n_k - 1);
        ldw(wn, sc_n, inx, dep);
        ldx(inx, dep);
        const uint8_t* ap = abuf + (i & 1) * AB + (gr * RB * 8 + rin) * 16;
        auto aget = [&](int jj, int rb) {
          return *reinterpret_cast<const uint4*>(ap + (jj * R8 + rb * 8) * 16);
        };
        if constexpr (WIDE_SCALE) {
          chunk8_wide<RB, CB>(wc, sc_c, aget, acc);
        } else {
          chunk8_mxfp8<RB, CB>(wc, sc_c, aget, acc);
        }
        dep = touch(wn, sc_n);
        stx(i + 1);
        slice_bar();
      };

      if (n_k > 0) {
        ldw(wA, svA, 0, 0u);
        ldx(0, 0u);
        unsigned dep = touch(wA, svA);
        stx(0);
        slice_bar();
        int i = 0;
        for (; i + 2 <= n_k; i += 2) {
          step(wA, svA, wB, svB, i, dep);
          step(wB, svB, wA, svA, i + 1, dep);
        }
        if (i < n_k) step(wA, svA, wB, svB, i, dep);
      }

      bar_sync_n(15, NCW * 32);
      if (xr == 0x9E3779B9u && zero == 1) Y[0] = __float2half(0.0f);

      if constexpr (PR) {
        float* rt = red + wk * SLAB + (gi * NROW + gr) * NR * 32;
#pragma unroll
        for (int i = 0; i < RB; i++)
#pragma unroll
          for (int j = 0; j < CB; j++)
#pragma unroll
            for (int e_i = 0; e_i < 8; e_i++) rt[((i * CB + j) * 8 + e_i) * 32 + lane] = acc[i][j][e_i];
        bar_sync_n(15, NCW * 32);
      } else {
        float* rt = red + (gi * NROW + gr) * NR * 32;
        for (int w = 0; w < WK; w++) {
          if (wk == w) {
#pragma unroll
            for (int i = 0; i < RB; i++)
#pragma unroll
              for (int j = 0; j < CB; j++)
#pragma unroll
                for (int e_i = 0; e_i < 8; e_i++) {
                  float& r = rt[((i * CB + j) * 8 + e_i) * 32 + lane];
                  r = w ? (r + acc[i][j][e_i]) : acc[i][j][e_i];
                }
          }
          bar_sync_n(15, NCW * 32);
        }
      }

      const int total_pairs = NCOL * rows * 16;
      if constexpr (GATED) {
        for (int idx = threadIdx.x; idx < total_pairs; idx += NCW * 32) {
          const int c = (NCOL == 1) ? 0 : (idx / (rows * 16));
          const int rr = (NCOL == 1) ? idx : (idx % (rows * 16));
          const int row = rr >> 4;
          const int col = (rr & 15) * 2;
          const int gcol = (bx * NCOL + c) * 32 + col;
          const float g0 = rv(ridx(c, row, col)) * gs;
          const float g1 = rv(ridx(c, row, col + 1)) * gs;
          const float u0 = rv(ridx(c, row, 32 + col)) * gs;
          const float u1 = rv(ridx(c, row, 33 + col)) * gs;
          const int out_slot = sh_slot[row];
          const float gamma = sh_gamma[row];
          const float v0 = softcap_situ(g0, u0, gamma, gate_cap, up_cap);
          const float v1 = softcap_situ(g1, u1, gamma, gate_cap, up_cap);
          *reinterpret_cast<__half2*>(Y + (size_t)out_slot * N_out + gcol) = __floats2half2_rn(v0, v1);
        }
      } else {
        for (int idx = threadIdx.x; idx < total_pairs; idx += NCW * 32) {
          const int c = (NCOL == 1) ? 0 : (idx / (rows * 16));
          const int rr = (NCOL == 1) ? idx : (idx % (rows * 16));
          const int row = rr >> 4;
          const int col = (rr & 15) * 2;
          const int gcol = (bx * NCOL + c) * 32 + col;
          const float v0 = rv(ridx(c, row, col)) * gs;
          const float v1 = rv(ridx(c, row, col + 1)) * gs;
          const int out_slot = sh_slot[row];
          *reinterpret_cast<__half2*>(Y + (size_t)out_slot * N_out + gcol) = __floats2half2_rn(v0, v1);
        }
      }
    }
  }
}

// Per-device kernel attribute setup (called once outside CUDA graph capture).
template <class Fn>
void configure_smem(Fn fn, int nth, int smem) {
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
              "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed");
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess,
              "cudaFuncSetAttribute(PreferredSharedMemoryCarveout=100) failed");
  int occ = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, nth, smem) == cudaSuccess,
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed");
  const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * smem + 98303) / 98304));
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess,
              "cudaFuncSetAttribute(PreferredSharedMemoryCarveout) failed");
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int PR, int CG, int GATED, int DIRECT_IN, int WIDE_SCALE>
void init_one_kernel() {
  using P = KM8<RB, CB, WK, NCOL, NROW, PR>;
  configure_smem(k_mxfp8_moe<RB, CB, WK, NCOL, NROW, MB, PR, CG, GATED, DIRECT_IN, WIDE_SCALE>, P::NTH, P::SMEM);
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int PR, int CG, int GATED, int DIRECT_IN, int WIDE_SCALE>
void launch_moe(const uint8_t* W, const uint8_t* S, const float* g_scale, const __half* X,
                const int32_t* perm, const int32_t* gids, const int32_t* goff, const float* weights,
                __half* Y, int N_out, int K, int topk, float gate_cap, float up_cap, int max_groups,
                cudaStream_t st) {
  using P = KM8<RB, CB, WK, NCOL, NROW, PR>;
  auto fn = k_mxfp8_moe<RB, CB, WK, NCOL, NROW, MB, PR, CG, GATED, DIRECT_IN, WIDE_SCALE>;
  TORCH_CHECK(N_out % (32 * NCOL) == 0, "N_out must be a multiple of ", 32 * NCOL, ", got ", N_out);
  const int grid_x = N_out / (32 * NCOL);
  const int grid_y = std::min(max_groups, 65535);
  dim3 grid(grid_x, grid_y, 1);
  fn<<<grid, P::NTH, P::SMEM, st>>>(W, S, g_scale, X, perm, gids, goff, weights, Y, N_out, K, topk, gate_cap, up_cap,
                                    max_groups, 0);
}

using LaunchFn = void (*)(const uint8_t*, const uint8_t*, const float*, const __half*, const int32_t*, const int32_t*,
                          const int32_t*, const float*, __half*, int, int, int, float, float, int, cudaStream_t);
using InitFn = void (*)();

struct MoeCfg {
  int r8;
  int cols;
  const char* name;
  LaunchFn launch;
  InitFn init;
};

// Fast-path w13_situ (GATED=1, CB=2, DIRECT_IN=0, WIDE_SCALE=0)
const MoeCfg W13_CFGS[] = {
    {8, 32, "w13 r8 wk6 col1 pr", launch_moe<1, 2, 6, 1, 1, 2, 1, 0, 1, 0, 0>,
     init_one_kernel<1, 2, 6, 1, 1, 2, 1, 0, 1, 0, 0>},
    {8, 32, "w13 r8 wk4 col1 pr", launch_moe<1, 2, 4, 1, 1, 4, 1, 0, 1, 0, 0>,
     init_one_kernel<1, 2, 4, 1, 1, 4, 1, 0, 1, 0, 0>},
    {16, 32, "w13 r16 wk4 col1 pr", launch_moe<2, 2, 4, 1, 1, 2, 1, 0, 1, 0, 0>,
     init_one_kernel<2, 2, 4, 1, 1, 2, 1, 0, 1, 0, 0>},
    {32, 32, "w13 r32 wk4 col1 pr", launch_moe<4, 2, 4, 1, 1, 2, 1, 0, 1, 0, 0>,
     init_one_kernel<4, 2, 4, 1, 1, 2, 1, 0, 1, 0, 0>},
    {48, 32, "w13 r48 wk4 col1 row2 pr", launch_moe<3, 2, 4, 1, 2, 1, 1, 0, 1, 0, 0>,
     init_one_kernel<3, 2, 4, 1, 2, 1, 1, 0, 1, 0, 0>},
};
constexpr int NW13 = sizeof(W13_CFGS) / sizeof(W13_CFGS[0]);

// Fast-path w2 (GATED=0, CB=1, DIRECT_IN=1, WIDE_SCALE=0)
const MoeCfg W2_CFGS[] = {
    {8, 32, "w2 r8 wk6 col1 pr", launch_moe<1, 1, 6, 1, 1, 3, 1, 0, 0, 1, 0>,
     init_one_kernel<1, 1, 6, 1, 1, 3, 1, 0, 0, 1, 0>},
    {8, 32, "w2 r8 wk4 col1 pr", launch_moe<1, 1, 4, 1, 1, 4, 1, 0, 0, 1, 0>,
     init_one_kernel<1, 1, 4, 1, 1, 4, 1, 0, 0, 1, 0>},
    {16, 64, "w2 r16 wk4 col2 pr", launch_moe<2, 1, 4, 2, 1, 2, 1, 0, 0, 1, 0>,
     init_one_kernel<2, 1, 4, 2, 1, 2, 1, 0, 0, 1, 0>},
    {32, 64, "w2 r32 wk4 col2 pr", launch_moe<4, 1, 4, 2, 1, 2, 1, 0, 0, 1, 0>,
     init_one_kernel<4, 1, 4, 2, 1, 2, 1, 0, 0, 1, 0>},
    {48, 64, "w2 r48 wk4 col2 pr", launch_moe<6, 1, 4, 2, 1, 1, 1, 0, 0, 1, 0>,
     init_one_kernel<6, 1, 4, 2, 1, 1, 1, 0, 0, 1, 0>},
    {16, 32, "w2 r16 wk4 col1 pr", launch_moe<2, 1, 4, 1, 1, 4, 1, 0, 0, 1, 0>,
     init_one_kernel<2, 1, 4, 1, 1, 4, 1, 0, 0, 1, 0>},
};
constexpr int NW2 = sizeof(W2_CFGS) / sizeof(W2_CFGS[0]);

// Exact wide-span fallbacks (WIDE_SCALE=1, arbitrary UE8M0 span 0..254)
const MoeCfg W13_WIDE_CFG = {
    8, 32, "w13 wide r8 wk4 col1 pr", launch_moe<1, 2, 4, 1, 1, 2, 1, 0, 1, 0, 1>,
    init_one_kernel<1, 2, 4, 1, 1, 2, 1, 0, 1, 0, 1>,
};
const MoeCfg W2_WIDE_CFG = {
    8, 32, "w2 wide r8 wk4 col1 pr", launch_moe<1, 1, 4, 1, 1, 4, 1, 0, 0, 1, 1>,
    init_one_kernel<1, 1, 4, 1, 1, 4, 1, 0, 0, 1, 1>,
};

std::mutex g_init_mu;
bool g_dev_inited[MAX_DEVS] = {};

void ensure_device_init(int dev) {
  TORCH_CHECK(dev >= 0 && dev < MAX_DEVS, "device index out of range: ", dev);
  if (g_dev_inited[dev]) return;
  std::lock_guard<std::mutex> lk(g_init_mu);
  if (g_dev_inited[dev]) return;
  for (int i = 0; i < NW13; i++) W13_CFGS[i].init();
  for (int i = 0; i < NW2; i++) W2_CFGS[i].init();
  W13_WIDE_CFG.init();
  W2_WIDE_CFG.init();
  g_dev_inited[dev] = true;
}

int pick_default_w13_cfg(int num_tokens) {
  if (num_tokens <= 48) return 0;   // r8 wk6 col1
  if (num_tokens <= 192) return 2;  // r16 wk4 col1
  if (num_tokens <= 384) return 3;  // r32 wk4 col1
  return 4;                         // r48 wk4 col1 row2
}

int pick_default_w2_cfg(int num_tokens, int N_out) {
  if (num_tokens <= 48) return 0;   // r8 wk6 col1
  if (N_out % 64 != 0) return 5;    // r16 wk4 col1 fallback
  if (num_tokens <= 192) return 2;  // r16 wk4 col2
  if (num_tokens <= 384) return 3;  // r32 wk4 col2
  return 4;                         // r48 wk4 col2
}

}  // namespace

void init_mxfp8_moe() {
  int dev = 0;
  TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  ensure_device_init(dev);
}

void mxfp8_moe_w13_situ(torch::Tensor x, torch::Tensor w13_qpn, torch::Tensor s13_qpn, torch::Tensor g13_scale,
                        torch::Tensor perm, torch::Tensor gids, torch::Tensor goff, torch::Tensor weights,
                        torch::Tensor out, int64_t topk, double gate_cap, double up_cap, bool wide_scale,
                        int64_t cfg, int64_t max_groups) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(w13_qpn.is_cuda() && w13_qpn.dtype() == torch::kUInt8 && w13_qpn.is_contiguous());
  TORCH_CHECK(s13_qpn.is_cuda() && s13_qpn.dtype() == torch::kUInt8 && s13_qpn.is_contiguous());
  TORCH_CHECK(g13_scale.is_cuda() && g13_scale.dtype() == torch::kFloat32 && g13_scale.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt32 && perm.is_contiguous() && perm.dim() == 1);
  TORCH_CHECK(gids.is_cuda() && gids.dtype() == torch::kInt32 && gids.is_contiguous() && gids.dim() == 1);
  TORCH_CHECK(goff.is_cuda() && goff.dtype() == torch::kInt32 && goff.is_contiguous() && goff.dim() == 1);
  TORCH_CHECK(weights.is_cuda() && weights.dtype() == torch::kFloat32 && weights.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.is_contiguous());

  const int num_tokens = (int)x.size(0);
  const int K = (int)x.size(1);
  const int E = (int)w13_qpn.size(0);
  const int I = (int)out.size(1);
  const int S = (int)out.size(0);
  if (S == 0 || num_tokens == 0) return;

  TORCH_CHECK(K > 0 && K % 32 == 0, "K must be a positive multiple of 32");
  TORCH_CHECK(I > 0 && I % 32 == 0, "I must be a positive multiple of 32");
  TORCH_CHECK(topk > 0 && S == num_tokens * topk, "out.size(0) must equal num_tokens * topk");
  TORCH_CHECK(w13_qpn.numel() == (int64_t)E * 2 * I * K, "w13_qpn numel mismatch");
  TORCH_CHECK(s13_qpn.numel() == (int64_t)E * 2 * I * (K / 32), "s13_qpn numel mismatch");
  TORCH_CHECK(g13_scale.numel() == E, "g13_scale numel must equal E");
  TORCH_CHECK(perm.size(0) == S && gids.size(0) == S && goff.size(0) == S + 1 && weights.numel() == S,
              "routing tensor shapes mismatch");

  const at::cuda::CUDAGuard guard(x.device());
  ensure_device_init(x.device().index());

  const int ng = (max_groups > 0) ? std::min((int)max_groups, S) : std::min(S, E);
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();

  const MoeCfg* chosen = nullptr;
  if (wide_scale) {
    chosen = &W13_WIDE_CFG;
  } else {
    const int idx = (cfg < 0) ? pick_default_w13_cfg(num_tokens) : (int)cfg;
    TORCH_CHECK(idx >= 0 && idx < NW13, "invalid w13 cfg: ", idx);
    chosen = &W13_CFGS[idx];
  }
  chosen->launch(w13_qpn.data_ptr<uint8_t>(), s13_qpn.data_ptr<uint8_t>(), g13_scale.data_ptr<float>(),
                 reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), perm.data_ptr<int32_t>(),
                 gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(), weights.data_ptr<float>(),
                 reinterpret_cast<__half*>(out.data_ptr<at::Half>()), I, K, (int)topk, (float)gate_cap,
                 (float)up_cap, ng, st);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void mxfp8_moe_w2(torch::Tensor x, torch::Tensor w2_qpn, torch::Tensor s2_qpn, torch::Tensor g2_scale,
                  torch::Tensor perm, torch::Tensor gids, torch::Tensor goff, torch::Tensor out,
                  bool wide_scale, int64_t cfg, int64_t max_groups, int64_t num_tokens_hint) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(w2_qpn.is_cuda() && w2_qpn.dtype() == torch::kUInt8 && w2_qpn.is_contiguous());
  TORCH_CHECK(s2_qpn.is_cuda() && s2_qpn.dtype() == torch::kUInt8 && s2_qpn.is_contiguous());
  TORCH_CHECK(g2_scale.is_cuda() && g2_scale.dtype() == torch::kFloat32 && g2_scale.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt32 && perm.is_contiguous() && perm.dim() == 1);
  TORCH_CHECK(gids.is_cuda() && gids.dtype() == torch::kInt32 && gids.is_contiguous() && gids.dim() == 1);
  TORCH_CHECK(goff.is_cuda() && goff.dtype() == torch::kInt32 && goff.is_contiguous() && goff.dim() == 1);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.is_contiguous());

  const int S = (int)x.size(0);
  const int K = (int)x.size(1);
  const int E = (int)w2_qpn.size(0);
  const int N = (int)out.size(1);
  if (S == 0) return;

  TORCH_CHECK(K > 0 && K % 32 == 0, "K must be a positive multiple of 32");
  TORCH_CHECK(N > 0 && N % 32 == 0, "N must be a positive multiple of 32");
  TORCH_CHECK(out.size(0) == S, "out.size(0) must equal S");
  TORCH_CHECK(w2_qpn.numel() == (int64_t)E * N * K, "w2_qpn numel mismatch");
  TORCH_CHECK(s2_qpn.numel() == (int64_t)E * N * (K / 32), "s2_qpn numel mismatch");
  TORCH_CHECK(g2_scale.numel() == E, "g2_scale numel must equal E");
  TORCH_CHECK(perm.size(0) == S && gids.size(0) == S && goff.size(0) == S + 1, "routing tensor shapes mismatch");

  const at::cuda::CUDAGuard guard(x.device());
  ensure_device_init(x.device().index());

  const int ng = (max_groups > 0) ? std::min((int)max_groups, S) : std::min(S, E);
  const int tok_hint = (num_tokens_hint > 0) ? (int)num_tokens_hint : std::max(1, S / 8);
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();

  const MoeCfg* chosen = nullptr;
  if (wide_scale) {
    chosen = &W2_WIDE_CFG;
  } else {
    const int idx = (cfg < 0) ? pick_default_w2_cfg(tok_hint, N) : (int)cfg;
    TORCH_CHECK(idx >= 0 && idx < NW2, "invalid w2 cfg: ", idx);
    chosen = &W2_CFGS[idx];
  }
  chosen->launch(w2_qpn.data_ptr<uint8_t>(), s2_qpn.data_ptr<uint8_t>(), g2_scale.data_ptr<float>(),
                 reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), perm.data_ptr<int32_t>(),
                 gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(), nullptr,
                 reinterpret_cast<__half*>(out.data_ptr<at::Half>()), N, K, 1, 0.0f, 0.0f, ng, st);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> w13_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NW13; i++)
    v.push_back(std::to_string(i) + ":" + std::to_string(W13_CFGS[i].r8) + ":" + std::to_string(W13_CFGS[i].cols) +
                ":" + W13_CFGS[i].name);
  return v;
}

std::vector<std::string> w2_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NW2; i++)
    v.push_back(std::to_string(i) + ":" + std::to_string(W2_CFGS[i].r8) + ":" + std::to_string(W2_CFGS[i].cols) +
                ":" + W2_CFGS[i].name);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init_mxfp8_moe, "Pre-configure SMEM carveouts on current CUDA device (call outside graph capture)");
  m.def("mxfp8_moe_w13_situ", &mxfp8_moe_w13_situ, "Zero-sync grouped MXFP8 w13 + exact FP32 softcapped SiTU (sm_70)");
  m.def("mxfp8_moe_w2", &mxfp8_moe_w2, "Zero-sync grouped MXFP8 w2 down projection (sm_70)");
  m.def("w13_configs", &w13_configs, "List w13 fast-path kernel configs");
  m.def("w2_configs", &w2_configs, "List w2 fast-path kernel configs");
}
