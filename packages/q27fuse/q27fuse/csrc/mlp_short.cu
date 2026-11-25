// mlp_short.cu - Exact single-pass 2D-tiled NVFP4 (W4A16) HMMA.884 MLP for Volta (sm_70, Tesla V100),
// targeting short / agentic prefills (M = 65..768+, works for any M >= 1).
//
// Zero extra weight copies, zero DRAM dequantization, zero k_to_frag, zero k_silu_pair:
//   - Reads mlp_fp4.cu's existing sole-copy prepacked weights q27_w ([N/32, K/32, 576] uint8):
//     per 32-column decode tile p and 32-k chunk c, 512 B of bit-rotated E2M1 codes in mma.m8n8k4
//     B-fragment order + 64 B of E4M3 scales (32 lanes x 2 k-halves).
//   - Finite-code weight contract (matching mlp_fp4.cu): quantized checkpoint scales in q27_w contain
//     only finite positive E4M3FN values (bytes 0x00..0x7E; no E4M3FN NaN 0x7F/0xFF). scale2() masks
//     (e & 0x7Fu) and reproduces exact fp16(E4M3) for every finite positive E4M3FN byte.
//   - Activations X [M, K] row-major fp16 are cooperatively staged per 32-k chunk into a double-buffered
//     conflict-free fragment-order SMEM buffer sA[2][4][BM + 2] (uint4): 8.25 KB at BM=64, 16.25 KB at BM=128.
//   - Each warp (wm, wn) owns RB*8 M-rows and CB*32 N-cols (CB decode tiles). Weights are double-buffered
//     in registers across K-chunks, decoded in-register with exact dq<0..3> + scale2 + hmul2u, and reused
//     across all RB M-blocks (and across WM M-warps via L1/L2).
//   - M-inner, N-outer CTA launch order (bm = blockIdx.x % gridM, bn = blockIdx.x / gridM) keeps all CTAs
//     sharing the same weight column-slab resident simultaneously so weights hit L2 across M-tiles.
//   - Epilogue reuses the A-buffer SMEM (512 B/warp for EPI=0, 256 B/warp for EPI=1) with zero bank conflicts
//     and 100% coalesced row-major STG.E.128 / STG.E.64 stores:
//       * EPI=1 (gate+up): prepack(..., gated=True) puts gate cols [0..15] in quadpairs q=0,1 ((lane&8)==0)
//         and matching up cols [16..31] in quadpairs q=2,3 ((lane&8)!=0) at the exact same accumulator index e.
//         Four __shfl_xor_sync(..., 8) instructions pair gate & up in registers across all 32 lanes (4 __expf
//         per lane, 100% warp efficiency, zero SMEM needed for SiLU), writing row-major mid [M, N/2] fp16.
//       * EPI=0 (down / plain linear): scales acc * epi and writes row-major out [M, N] fp16.
//   - Call init() once per device outside CUDA graph capture to pre-configure SMEM carveouts thread-safely.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace {
constexpr int CHUNK = 576;
constexpr int MAX_DEVS = 32;

inline bool is_aligned(const torch::Tensor& t, size_t align) {
  return t.numel() == 0 || (reinterpret_cast<uintptr_t>(t.data_ptr()) & (align - 1u)) == 0u;
}

inline bool tensors_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  const size_t ab = (size_t)a.numel() * (size_t)a.element_size();
  const size_t bb = (size_t)b.numel() * (size_t)b.element_size();
  if (ab == 0 || bb == 0) return false;
  const uintptr_t a0 = reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t b0 = reinterpret_cast<uintptr_t>(b.data_ptr());
  return a0 < (b0 + bb) && b0 < (a0 + ab);
}

__device__ __forceinline__ void mma884(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// WLD 0: ld.global.nc (caches in L1 + L2, allowing WM M-warps in the same CTA to reuse weights out of L1).
// WLD 1: ld.global.nc.L1::no_allocate (caches in L2 only, preserving L1 for activations).
template <int WLD>
__device__ __forceinline__ uint4 ldw16(const void* p) {
  uint4 v;
  if constexpr (WLD == 0) {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
  }
  return v;
}

template <int WLD>
__device__ __forceinline__ unsigned ldw2(const void* p) {
  unsigned short v;
  if constexpr (WLD == 0) {
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(p));
  }
  return (unsigned)v;
}

__device__ __forceinline__ uint4 lda16(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ unsigned rotl(unsigned w, int r) {
  return r ? __funnelshift_l(w, w, r) : w;
}
__host__ __device__ constexpr int RA(int p) { return p * 3; }    // magnitude rotation of code pair p
__host__ __device__ constexpr int RB_(int p) { return 3 - p; }   // sign rotation of code pair p

template <int P>
__device__ __forceinline__ unsigned dq(unsigned w) {             // fp16x2 = E2M1 * 2^-14 for code pair P
  return (rotl(w, RA(P)) & 0x0E000E00u) | (rotl(w, RB_(P)) & 0x80008000u);
}

__device__ __forceinline__ unsigned hmul2u(unsigned a, unsigned b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<unsigned*>(&r);
}

__device__ __forceinline__ unsigned scale2(unsigned e) {         // E4M3 byte (positive) -> fp16x2 (s, s), exact
  const unsigned h = (e & 0x7Fu) * 0x00800080u;                  // E4M3 * 2^-8 in both halves
  return hmul2u(h, 0x5C005C00u);                                 // x 256
}

// Compute one 32-k chunk (8 HMMA.884 K-steps = 4 jj x 2 halves), reusing decoded weights across all RB M-blocks.
// ap points to sA[stage][0][warp_row0 + rin]; jj-stride is STRIDE = BM + 2 uint4s (0 bank conflicts on LDS.128).
template <int RB, int CB, int STRIDE>
__device__ __forceinline__ void chunk_math_w4(const uint4 (&w)[CB], const unsigned (&sv)[CB],
                                              const uint4* __restrict__ ap, float (&acc)[RB][CB][8]) {
#pragma unroll
  for (int hk = 0; hk < 2; hk++) {
    unsigned s2[CB];
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      s2[cb] = scale2(hk ? (sv[cb] >> 8) : sv[cb]);
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int jj = hk * 2 + j;
      unsigned b0[CB], b1[CB], b2[CB], b3[CB];
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        const unsigned wd = (jj == 0) ? w[cb].x : (jj == 1) ? w[cb].y : (jj == 2) ? w[cb].z : w[cb].w;
        b0[cb] = hmul2u(dq<0>(wd), s2[cb]);
        b1[cb] = hmul2u(dq<1>(wd), s2[cb]);
        b2[cb] = hmul2u(dq<2>(wd), s2[cb]);
        b3[cb] = hmul2u(dq<3>(wd), s2[cb]);
      }
      const uint4* apj = ap + jj * STRIDE;
      uint4 a_cur = apj[0];
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        uint4 a_nxt;
        if (rb + 1 < RB) a_nxt = apj[(rb + 1) * 8];
#pragma unroll
        for (int cb = 0; cb < CB; cb++) {
          mma884(acc[rb][cb], a_cur.x, a_cur.y, b0[cb], b1[cb]);
          mma884(acc[rb][cb], a_cur.z, a_cur.w, b2[cb], b3[cb]);
        }
        if (rb + 1 < RB) a_cur = a_nxt;
      }
    }
  }
}

template <int WM, int WN, int RB, int CB, int EPI>
struct SP4 {
  static constexpr int NW = WM * WN;
  static constexpr int NTH = NW * 32;
  static constexpr int BM = WM * RB * 8;
  static constexpr int BN = WN * CB * 32;
  static constexpr int STRIDE = BM + 2;                          // uint4 stride per jj: ((BM+2)*4)%32 == 8 -> 0 bank conflicts
  static constexpr int STAGE_U4 = 4 * STRIDE;                    // uint4s per A stage
  static constexpr int SMEM_A = 2 * STAGE_U4 * 16;               // bytes for double-buffered A
  static constexpr int SMEM_EPI = NW * (EPI ? 256 : 512);        // per-warp epilogue scratch (reuses A buffer)
  static constexpr int SMEM = SMEM_A > SMEM_EPI ? SMEM_A : SMEM_EPI;
  static constexpr int NPC = (BM * 4) / NTH;                     // 16-B A-load units per thread per 32-k chunk
  static_assert((BM * 4) % NTH == 0 && NPC >= 1, "NTH must evenly divide BM * 4");
  static_assert(BM % 8 == 0, "BM must be a multiple of 8");
  static_assert(CB == 1 || CB == 2, "CB must be 1 or 2");
};

// 2D-tiled single-pass NVFP4 W4A16 HMMA.884 GEMM (+ optional fused SiLU-and-mul epilogue for EPI=1).
// W: [N/32][K/32][576] u8 (mlp_fp4.cu prepack layout). X: [M][K] row-major fp16.
// Y: EPI=0 -> [M][N] row-major fp16; EPI=1 -> [M][N/2] row-major fp16.
// zero: runtime 0 (opaque offset forcing scoreboard separation between double-buffered K-chunks).
template <int WM, int WN, int RB, int CB, int EPI, int MB, int WLD = 0>
__global__ void __launch_bounds__(SP4<WM, WN, RB, CB, EPI>::NTH, MB)
k_mlp_short(const uint8_t* __restrict__ W, const __half* __restrict__ X, __half* __restrict__ Y,
            int M, int N, int K, float epi, int gridM, int zero) {
  using P = SP4<WM, WN, RB, CB, EPI>;
  constexpr int NTH = P::NTH, BM = P::BM, STRIDE = P::STRIDE, STAGE_U4 = P::STAGE_U4, NPC = P::NPC;
  extern __shared__ __align__(16) uint8_t sm[];
  uint4* sA = reinterpret_cast<uint4*>(sm);                      // [2][4][BM + 2] uint4

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wn = warp % WN, wm = warp / WN;

  // M-inner, N-outer CTA swizzle: all gridM CTAs sharing the same weight column-slab execute concurrently in L2.
  const int bm = blockIdx.x % gridM;
  const int bn = blockIdx.x / gridM;

  const int NC = K / 32;
  const int NT = N / 32;
  const int row0_cta = bm * BM;
  const int row0_warp = wm * RB * 8;
  const int p0 = (bn * WN + wn) * CB;
  const int rin = (lane & 3) + 4 * (lane >> 4);                  // A-fragment row inside an 8-row block
  const unsigned zmask = (unsigned)zero;

  const uint8_t* wp0 = W + (size_t)min(p0, NT - 1) * NC * CHUNK;
  const size_t cbs = (CB > 1 && (p0 + 1) < NT) ? (size_t)NC * CHUNK : 0;

  float acc[RB][CB][8];
#pragma unroll
  for (int rb = 0; rb < RB; rb++)
#pragma unroll
    for (int cb = 0; cb < CB; cb++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[rb][cb][e] = 0.f;

  uint4 wA[CB], wB[CB], stg[NPC];
  unsigned scA[CB], scB[CB];
  unsigned xr = 0;

  auto ldw = [&](uint4 (&w)[CB], unsigned (&sv)[CB], int ch, unsigned dep) {
    const size_t off = (size_t)ch * CHUNK + dep;
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const uint8_t* pw = wp0 + cb * cbs + off;
      w[cb] = ldw16<WLD>(pw + lane * 16);
      sv[cb] = ldw2<WLD>(pw + 512 + lane * 2);
    }
  };

  auto ldx = [&](int ch, unsigned dep) {
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = q * NTH + tid;                               // 16-B unit = row * 4 + jj
      const int row = u >> 2, jj = u & 3;
      const int grow = row0_cta + row;
      if (grow < M) {
        stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)grow * K + ch * 32 + jj * 8) + dep);
      } else {
        stg[q] = make_uint4(0, 0, 0, 0);
      }
    }
  };

  auto stx = [&](int stage) {
    uint4* dst = sA + stage * STAGE_U4;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = q * NTH + tid;
      const int row = u >> 2, jj = u & 3;
      dst[jj * STRIDE + row] = stg[q];
    }
  };

  auto touch = [&](const uint4 (&w)[CB], const unsigned (&sv)[CB]) {
    unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) x ^= w[cb].x ^ sv[cb];
#pragma unroll
    for (int q = 0; q < NPC; q++) x ^= stg[q].x;
    xr ^= x;
    return x & zmask;
  };

  auto step = [&](const uint4 (&wc)[CB], const unsigned (&svc)[CB],
                  uint4 (&wn_buf)[CB], unsigned (&svn_buf)[CB],
                  int ch, int stage, unsigned& dep) {
    ldw(wn_buf, svn_buf, ch + 1, dep);
    ldx(ch + 1, dep);
    const uint4* ap = sA + stage * STAGE_U4 + row0_warp + rin;
    chunk_math_w4<RB, CB, STRIDE>(wc, svc, ap, acc);
    dep = touch(wn_buf, svn_buf);
    stx(stage ^ 1);
    __syncthreads();
  };

  // Prologue: load & stage chunk 0
  ldw(wA, scA, 0, 0u);
  ldx(0, 0u);
  unsigned dep = touch(wA, scA);
  stx(0);
  __syncthreads();

  // Main K-loop (2x unrolled for compile-time constant SMEM buffer indices and register double-buffering)
  int ch = 0;
  for (; ch + 2 < NC; ch += 2) {
    step(wA, scA, wB, scB, ch, 0, dep);
    step(wB, scB, wA, scA, ch + 1, 1, dep);
  }
  if (ch + 1 < NC) {
    step(wA, scA, wB, scB, ch, 0, dep);
    const uint4* ap = sA + STAGE_U4 + row0_warp + rin;
    chunk_math_w4<RB, CB, STRIDE>(wB, scB, ap, acc);
  } else if (ch < NC) {
    const uint4* ap = sA + row0_warp + rin;
    chunk_math_w4<RB, CB, STRIDE>(wA, scA, ap, acc);
  }

  // Ensure all warps have finished reading sA before aliasing sm[] for the per-warp epilogue.
  __syncthreads();
  if (xr == 0x9E3779B9u && zero == 1) Y[0] = __float2half(0.f);  // never true; keeps forced waits live

  // Per-warp zero-bank-conflict epilogue (reuses sm[]; only __syncwarp() needed per (rb, cb))
  if constexpr (EPI == 0) {
    // 512 B/warp = [8 rows][16 half2], XOR-swizzled on bit 1 of col2 when r8 >= 4 (0 bank conflicts on STS.32 & LDS.128)
    __half2* sm_ep = reinterpret_cast<__half2*>(sm) + warp * 128;
    const int r8_lo = (lane & 1) | ((lane >> 4) << 2);           // 0, 1, 4, 5
    const int r8_hi = r8_lo + 2;                                 // 2, 3, 6, 7
    const int swiz = (lane >> 4) << 1;                           // 0 for r8 < 4, 2 for r8 >= 4
    const int q = (lane >> 2) & 3;
    const int col2_0 = ((q << 2) | ((lane & 2) >> 1)) ^ swiz;
    const int col2_4 = ((q << 2) | 2 | ((lane & 2) >> 1)) ^ swiz;
    const int rd_r8 = lane >> 2;                                 // 0..7
    const int rd_u = lane & 3;                                   // 0..3 -> 8 halfs = 16 B

#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const int p = p0 + cb;
      if (p < NT) {
#pragma unroll
        for (int rb = 0; rb < RB; rb++) {
          const int m_base = row0_cta + row0_warp + rb * 8;
          sm_ep[r8_lo * 16 + col2_0] = __floats2half2_rn(acc[rb][cb][0] * epi, acc[rb][cb][1] * epi);
          sm_ep[r8_hi * 16 + col2_0] = __floats2half2_rn(acc[rb][cb][2] * epi, acc[rb][cb][3] * epi);
          sm_ep[r8_lo * 16 + col2_4] = __floats2half2_rn(acc[rb][cb][4] * epi, acc[rb][cb][5] * epi);
          sm_ep[r8_hi * 16 + col2_4] = __floats2half2_rn(acc[rb][cb][6] * epi, acc[rb][cb][7] * epi);
          __syncwarp();
          const int grow = m_base + rd_r8;
          if (grow < M) {
            uint4 v = *reinterpret_cast<const uint4*>(sm_ep + rd_r8 * 16 + rd_u * 4);
            if (rd_r8 >= 4) v = make_uint4(v.z, v.w, v.x, v.y);
            *reinterpret_cast<uint4*>(Y + (size_t)grow * N + p * 32 + rd_u * 8) = v;
          }
          __syncwarp();
        }
      }
    }
  } else {
    // EPI == 1: fused SiLU(gate) * up -> row-major [M, N/2] fp16.
    // Prepacked tile p holds gate cols [0..15] in quadpairs q=0,1 ((lane&8)==0) and matching up cols [16..31]
    // in quadpairs q=2,3 ((lane&8)!=0) at the exact same accumulator index e.
    // Gate lanes (is_up==0) compute e=0..3; up lanes (is_up==1) compute e=4..7 (all 32 lanes do 4 __expf each).
    // 256 B/warp = [8 slots][8 half2] with slot = ((r8 & 4) >> 1) | (r8 & 1) | ((r8 & 2) << 1) (0 bank conflicts).
    __half2* sm_ep = reinterpret_cast<__half2*>(sm) + warp * 64;
    const bool is_up = (lane & 8) != 0;
    const int q_g = (lane >> 2) & 1;
    const int col2 = (q_g << 2) | ((lane & 8) >> 2) | ((lane & 2) >> 1);
    const int slot_lo = ((lane >> 4) << 1) | (lane & 1);         // 0..3 for r8 in {0, 1, 4, 5}
    const int slot_hi = slot_lo + 4;                             // 4..7 for r8 in {2, 3, 6, 7}
    const int rd_slot = lane >> 2;                               // 0..7
    const int rd_u = lane & 3;                                   // 0..3 -> 4 halfs = 8 B
    const int rd_r8 = (rd_slot & 1) | ((rd_slot & 4) >> 1) | ((rd_slot & 2) << 1);
    const int I = N >> 1;

    auto silu_mul = [](float g, float u) -> float {
      return g / (1.f + __expf(-g)) * u;
    };

#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const int p = p0 + cb;
      if (p < NT) {
#pragma unroll
        for (int rb = 0; rb < RB; rb++) {
          const int m_base = row0_cta + row0_warp + rb * 8;
          // Unconditional warp-wide shuffle pairing gate (lane&8==0) and up (lane&8!=0)
          const float sh0 = __shfl_xor_sync(0xFFFFFFFFu, is_up ? acc[rb][cb][0] : acc[rb][cb][4], 8);
          const float sh1 = __shfl_xor_sync(0xFFFFFFFFu, is_up ? acc[rb][cb][1] : acc[rb][cb][5], 8);
          const float sh2 = __shfl_xor_sync(0xFFFFFFFFu, is_up ? acc[rb][cb][2] : acc[rb][cb][6], 8);
          const float sh3 = __shfl_xor_sync(0xFFFFFFFFu, is_up ? acc[rb][cb][3] : acc[rb][cb][7], 8);
          const float g0 = (is_up ? sh0 : acc[rb][cb][0]) * epi;
          const float g1 = (is_up ? sh1 : acc[rb][cb][1]) * epi;
          const float g2 = (is_up ? sh2 : acc[rb][cb][2]) * epi;
          const float g3 = (is_up ? sh3 : acc[rb][cb][3]) * epi;
          const float u0 = (is_up ? acc[rb][cb][4] : sh0) * epi;
          const float u1 = (is_up ? acc[rb][cb][5] : sh1) * epi;
          const float u2 = (is_up ? acc[rb][cb][6] : sh2) * epi;
          const float u3 = (is_up ? acc[rb][cb][7] : sh3) * epi;

          sm_ep[slot_lo * 8 + col2] = __floats2half2_rn(silu_mul(g0, u0), silu_mul(g1, u1));
          sm_ep[slot_hi * 8 + col2] = __floats2half2_rn(silu_mul(g2, u2), silu_mul(g3, u3));
          __syncwarp();
          const int grow = m_base + rd_r8;
          if (grow < M) {
            *reinterpret_cast<uint2*>(Y + (size_t)grow * I + p * 16 + rd_u * 4) =
                *reinterpret_cast<const uint2*>(sm_ep + rd_slot * 8 + rd_u * 2);
          }
          __syncwarp();
        }
      }
    }
  }
}

template <int WM, int WN, int RB, int CB, int EPI, int MB, int WLD>
void init_w4_dev(int dev, cudaStream_t st) {
  using P = SP4<WM, WN, RB, CB, EPI>;
  auto fn = k_mlp_short<WM, WN, RB, CB, EPI, MB, WLD>;
  static std::once_flag flags[MAX_DEVS];
  TORCH_CHECK(dev >= 0 && dev < MAX_DEVS, "mlp_short: device index out of range [0, 32)");
  std::call_once(flags[dev], [&]() {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    TORCH_CHECK(cudaStreamIsCapturing(st, &cap) == cudaSuccess);
    TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                "mlp_short: call init() before CUDA graph capture (cannot run cudaFuncSetAttribute during capture)");
    // Keep the smallest SMEM carveout that preserves full occupancy so the remainder stays L1 cache.
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    int occ = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, P::NTH, P::SMEM) == cudaSuccess);
    const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * P::SMEM + 98303) / 98304));
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess);
  });
}

template <int WM, int WN, int RB, int CB, int EPI, int MB, int WLD>
void launch_w4(const uint8_t* W, const __half* X, __half* Y, int M, int N, int K, float epi, int dev, cudaStream_t st) {
  using P = SP4<WM, WN, RB, CB, EPI>;
  auto fn = k_mlp_short<WM, WN, RB, CB, EPI, MB, WLD>;
  init_w4_dev<WM, WN, RB, CB, EPI, MB, WLD>(dev, st);
  const int gridM = (M + P::BM - 1) / P::BM;
  const int gridN = (N + P::BN - 1) / P::BN;
  const int grid = gridM * gridN;
  fn<<<grid, P::NTH, P::SMEM, st>>>(W, X, Y, M, N, K, epi, gridM, 0);
}

using InitFn = void (*)(int, cudaStream_t);
using LaunchW4 = void (*)(const uint8_t*, const __half*, __half*, int, int, int, float, int, cudaStream_t);
struct W4Cfg {
  const char* name;
  int bm;
  int bn;
  InitFn init_lin;
  InitFn init_gated;
  LaunchW4 fn_lin;
  LaunchW4 fn_gated;
};

#define W4_CFG(NAME, WM, WN, RB, CB, MB, WLD) \
  {NAME, (WM) * (RB) * 8, (WN) * (CB) * 32, \
   init_w4_dev<WM, WN, RB, CB, 0, MB, WLD>, init_w4_dev<WM, WN, RB, CB, 1, MB, WLD>, \
   launch_w4<WM, WN, RB, CB, 0, MB, WLD>, launch_w4<WM, WN, RB, CB, 1, MB, WLD>}

const W4Cfg CFGS[] = {
    // Primary tile sweep (WLD=0: ld.global.nc L1+L2 weight caching for intra-CTA WM reuse)
    W4_CFG("64x64_w2x2_r4c1_l1",   2, 2, 4, 1, 4, 0),  // cfg 0: 128 thr, high SM wave count (M=65..256)
    W4_CFG("64x128_w2x2_r4c2_l1",  2, 2, 4, 2, 3, 0),  // cfg 1: 128 thr, 2x A reuse
    W4_CFG("128x64_w4x2_r4c1_l1",  4, 2, 4, 1, 2, 0),  // cfg 2: 256 thr, high W reuse (M=128..768)
    W4_CFG("128x128_w4x2_r4c2_l1", 4, 2, 4, 2, 1, 0),  // cfg 3: 256 thr, max arithmetic intensity (M=384..768)
    W4_CFG("64x64_w4x2_r2c1_l1",   4, 2, 2, 1, 3, 0),  // cfg 4: 256 thr, low-reg high-occupancy
    // Low-register CB=1 alternatives for 64x128 and 128x128 (~80 regs/thr, zero spill risk)
    W4_CFG("64x128_w2x4_r4c1_l1",  2, 4, 4, 1, 2, 0),  // cfg 5: 256 thr, low-reg 64x128
    W4_CFG("128x128_w4x4_r4c1_l1", 4, 4, 4, 1, 1, 0),  // cfg 6: 512 thr, low-reg 128x128
    // L2-only weight load A/B variants (WLD=1: ld.global.nc.L1::no_allocate)
    W4_CFG("64x64_w2x2_r4c1_na",   2, 2, 4, 1, 4, 1),  // cfg 7
    W4_CFG("128x64_w4x2_r4c1_na",  4, 2, 4, 1, 2, 1),  // cfg 8
    W4_CFG("128x128_w4x2_r4c2_na", 4, 2, 4, 2, 1, 1),  // cfg 9
    W4_CFG("64x128_w2x2_r4c2_na",  2, 2, 4, 2, 3, 1),  // cfg 10
};
#undef W4_CFG
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);
}  // namespace

// Pre-initialize CUDA function attributes on the current (or specified) device outside CUDA graph capture.
void init(int64_t device_index = -1) {
  int dev = (int)device_index;
  if (dev < 0) {
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  }
  const at::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, (c10::DeviceIndex)dev));
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  for (int i = 0; i < NCFG; i++) {
    CFGS[i].init_lin(dev, st);
    CFGS[i].init_gated(dev, st);
  }
}

void gemm_w4_into(torch::Tensor x, torch::Tensor w, torch::Tensor out, double epi, int64_t N, int64_t K,
                  int64_t epi_mode, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous(), "gemm_w4: x");
  TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kUInt8 && w.is_contiguous(), "gemm_w4: w");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.is_contiguous(), "gemm_w4: out");
  TORCH_CHECK(w.device() == x.device() && out.device() == x.device(), "gemm_w4: all tensors must be on x.device()");
  const int M = (int)x.size(0);
  TORCH_CHECK(x.size(1) == K && K >= 32 && K % 32 == 0, "gemm_w4: K must be positive multiple of 32");
  TORCH_CHECK(N >= 32 && N % 32 == 0, "gemm_w4: N must be positive multiple of 32");
  TORCH_CHECK(epi_mode == 0 || epi_mode == 1, "gemm_w4: epi_mode must be 0 or 1");
  const int64_t out_n = epi_mode ? (N / 2) : N;
  TORCH_CHECK(out.size(0) == M && out.size(1) == out_n, "gemm_w4: out shape mismatch");
  TORCH_CHECK(w.numel() == (N / 32) * (K / 32) * CHUNK, "gemm_w4: w numel mismatch");
  TORCH_CHECK(cfg >= 0 && cfg < NCFG, "gemm_w4: invalid cfg");
  TORCH_CHECK(is_aligned(x, 16) && is_aligned(w, 16) && is_aligned(out, 16),
              "gemm_w4: x, w, and out pointers must be 16-byte aligned");
  TORCH_CHECK(!tensors_overlap(out, x) && !tensors_overlap(out, w),
              "gemm_w4: out must not alias x or w");
  if (M == 0) return;

  int dev = (int)x.device().index();
  const at::cuda::CUDAGuard guard(x.device());
  if (dev < 0) TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  LaunchW4 fn = epi_mode ? CFGS[cfg].fn_gated : CFGS[cfg].fn_lin;
  fn(w.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
     reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, (int)N, (int)K, (float)epi, dev, st);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

torch::Tensor gemm_w4(torch::Tensor x, torch::Tensor w, double epi, int64_t N, int64_t K,
                      int64_t epi_mode, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && (epi_mode == 0 || epi_mode == 1), "gemm_w4: x/epi_mode");
  const int64_t out_n = epi_mode ? (N / 2) : N;
  auto out = torch::empty({x.size(0), out_n}, x.options());
  gemm_w4_into(x, w, out, epi, N, K, epi_mode, cfg);
  return out;
}

void mlp_short_into(torch::Tensor x, torch::Tensor wgu, torch::Tensor wd, double epi_gu, double epi_d,
                    int64_t ir, torch::Tensor mid, torch::Tensor out, int64_t cfg_gu, int64_t cfg_d) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous(), "mlp_short: x");
  TORCH_CHECK(wgu.is_cuda() && wgu.dtype() == torch::kUInt8 && wgu.is_contiguous(), "mlp_short: wgu");
  TORCH_CHECK(wd.is_cuda() && wd.dtype() == torch::kUInt8 && wd.is_contiguous(), "mlp_short: wd");
  TORCH_CHECK(mid.is_cuda() && mid.dtype() == torch::kHalf && mid.dim() == 2 && mid.is_contiguous(), "mlp_short: mid");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.is_contiguous(), "mlp_short: out");
  TORCH_CHECK(wgu.device() == x.device() && wd.device() == x.device() &&
              mid.device() == x.device() && out.device() == x.device(),
              "mlp_short: all tensors must be on x.device()");
  const int64_t M = x.size(0), H = x.size(1), I = ir;
  TORCH_CHECK(H >= 32 && H % 32 == 0 && I >= 32 && I % 32 == 0, "mlp_short: H and ir must be multiples of 32");
  TORCH_CHECK(mid.size(0) == M && mid.size(1) == I, "mlp_short: mid shape mismatch");
  TORCH_CHECK(out.size(0) == M && out.size(1) == H, "mlp_short: out shape mismatch");
  TORCH_CHECK(is_aligned(x, 16) && is_aligned(wgu, 16) && is_aligned(wd, 16) &&
              is_aligned(mid, 16) && is_aligned(out, 16),
              "mlp_short: x, wgu, wd, mid, and out pointers must be 16-byte aligned");
  TORCH_CHECK(!tensors_overlap(mid, x) && !tensors_overlap(mid, wgu) && !tensors_overlap(mid, wd) &&
              !tensors_overlap(out, x) && !tensors_overlap(out, wgu) && !tensors_overlap(out, wd) &&
              !tensors_overlap(out, mid),
              "mlp_short: mid and out must not alias each other or any input tensor");
  if (M == 0) return;
  gemm_w4_into(x, wgu, mid, epi_gu, 2 * I, H, 1, cfg_gu);
  gemm_w4_into(mid, wd, out, epi_d, H, I, 0, cfg_d);
}

torch::Tensor mlp_short(torch::Tensor x, torch::Tensor wgu, torch::Tensor wd, double epi_gu, double epi_d,
                        int64_t ir, int64_t cfg_gu, int64_t cfg_d) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "mlp_short: x");
  auto mid = torch::empty({x.size(0), ir}, x.options());
  auto out = torch::empty({x.size(0), x.size(1)}, x.options());
  mlp_short_into(x, wgu, wd, epi_gu, epi_d, ir, mid, out, cfg_gu, cfg_d);
  return out;
}

std::vector<std::string> mlp_short_configs() {
  std::vector<std::string> v;
  v.reserve(NCFG);
  for (int i = 0; i < NCFG; i++) {
    v.push_back(std::to_string(i) + ":" + CFGS[i].name);
  }
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, pybind11::arg("device_index") = -1,
        "Pre-initialize CUDA function attributes on device (call outside CUDA graph capture)");
  m.def("mlp_short", &mlp_short,
        "Single-pass 2D-tiled NVFP4 W4A16 HMMA.884 MLP (fused gate+up SiLU-and-mul + down)");
  m.def("mlp_short_into", &mlp_short_into,
        "Zero-allocation single-pass 2D-tiled NVFP4 W4A16 MLP into caller-provided mid and out");
  m.def("gemm_w4", &gemm_w4,
        "Single-pass 2D-tiled NVFP4 W4A16 GEMM (epi_mode=0 linear, epi_mode=1 fused SiLU-and-mul)");
  m.def("gemm_w4_into", &gemm_w4_into,
        "Zero-allocation single-pass 2D-tiled NVFP4 W4A16 GEMM into caller-provided out");
  m.def("mlp_short_configs", &mlp_short_configs,
        "List available tile configuration strings for mlp_short / gemm_w4");
}
