// mlp_fp4.cu - NVFP4 (W4A16) MLP for Qwen3.8-27B decode on V100, 1..64 rows: gate+up with fused SiLU-and-mul, then
// down. Kernels from ../../../ladder/mlpladder.cu (k_w5 MODE 3 and MODE 4, ladder/REPORT.md §7):
//   - weights prepacked once (prepack): per 32-column tile and 32-k chunk, 512 B of codes in mma B-fragment order
//     (bit-rotated so one funnel shift + one LOP3 per pair gives fp16x2 = E2M1 * 2^-14 exactly) + 64 B of E4M3 scales;
//     gate+up tiles hold 16 gate columns then the same 16 up columns, so a CTA owns matching gate/up pairs;
//   - MODE 3: weights and activation fragments double-buffered in registers, one forced wait per chunk, next loads
//     ordered after it by an always-zero opaque address offset (ptxas would otherwise share scoreboards and stall);
//   - MODE 4: weights as MODE 3, activations staged per k-slice in a shared-memory double buffer by the slice's own
//     warps (ROWIN: straight from the row-major input), fragments read with LDS;
//   - input: row-major [M][5120] (the output of 1Cat's fused all-reduce + Gemma norm); 1..16 rows are staged by
//     gate+up (ROWIN), 17..64 rows are first copied into fragment order [K/8][R8][8] (k_to_frag);
//   - EPI 1 (gate+up): silu(gate) * up, written in mma-fragment order [Ir/8][R8][8] for the down kernel; R8 = padded
//     rows of the M bucket (16, 32, 48, 64); EPI 0 (down): row-major [M][N] fp16.
// y = x W^T with W = E2M1 * E4M3 / g; the MMAs see E2M1 * E4M3 * 2^-14, the epilogue multiplies by epi = 2^14 / g.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace {
constexpr int CHUNK = 576;

__device__ __forceinline__ void mma884(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldg16na(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ldg2na(const void* p) {
  unsigned short v;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
// smallrows: weight loads with a template-chosen cache operator. CG 0 = ld.global.nc.L1::no_allocate (read-only
// path); CG 1 = ld.global.cg (L2 only, normal path): .nc streams leave the dirty L2 lines of a preceding store kernel
// (the GDN state stores) to be written back piecemeal under later streams, .cg reads displace them (notes/kgap.md
// rounds 6-7; notes/smallrows.md for the server A/B)
template <int CG>
__device__ __forceinline__ uint4 ldw16(const void* p) {
  if constexpr (CG) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
  } else {
    return ldg16na(p);
  }
}
template <int CG>
__device__ __forceinline__ unsigned ldw2(const void* p) {
  if constexpr (CG) {
    unsigned short v;
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
  } else {
    return ldg2na(p);
  }
}
__device__ __forceinline__ uint4 lda16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ unsigned rotl(unsigned w, int r) { return r ? __funnelshift_l(w, w, r) : w; }
__host__ __device__ constexpr int RA(int p) { return p * 3; }      // magnitude rotation of code pair p
__host__ __device__ constexpr int RB_(int p) { return 3 - p; }     // sign rotation of code pair p
template <int P>
__device__ __forceinline__ unsigned dq(unsigned w) {               // fp16x2 = E2M1 * 2^-14 for code pair P
  return (rotl(w, RA(P)) & 0x0E000E00u) | (rotl(w, RB_(P)) & 0x80008000u);
}
__device__ __forceinline__ unsigned hmul2u(unsigned a, unsigned b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ unsigned scale2(unsigned e) {           // E4M3 byte (positive) -> fp16x2 (s, s), exact
  const unsigned h = (e & 0x7Fu) * 0x00800080u;                    // E4M3 * 2^-8 in both halves
  return hmul2u(h, 0x5C005C00u);                                   // x 256
}
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

template <int RB, int CB, class AGet>
__device__ __forceinline__ void chunk_math(const uint4 (&w)[CB], const unsigned (&sv)[CB], AGet aget, float (&acc)[RB][CB][8]) {
  unsigned sc[CB][2];
#pragma unroll
  for (int cb = 0; cb < CB; cb++) { sc[cb][0] = scale2(sv[cb]); sc[cb][1] = scale2(sv[cb] >> 8); }
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    uint4 a[RB];
#pragma unroll
    for (int rb = 0; rb < RB; rb++) a[rb] = aget(jj, rb);
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        const unsigned wd = jj == 0 ? w[cb].x : jj == 1 ? w[cb].y : jj == 2 ? w[cb].z : w[cb].w;
        const unsigned s2 = jj >> 1 ? sc[cb][1] : sc[cb][0];
        const unsigned b0 = hmul2u(h ? dq<2>(wd) : dq<0>(wd), s2), b1 = hmul2u(h ? dq<3>(wd) : dq<1>(wd), s2);
#pragma unroll
        for (int rb = 0; rb < RB; rb++) mma884(acc[rb][cb], h ? a[rb].z : a[rb].x, h ? a[rb].w : a[rb].y, b0, b1);
      }
  }
}

// PR (smallrows, from kgap): the k-slice partials are reduced in parallel (every warp writes its own slab, one barrier,
// the epilogue sums the WK slabs in slice order = the same additions as the serial reduction: bitwise unchanged)
template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int PR = 0>
struct KP {
  static constexpr int NCW = WK * NCOL * NROW, RBT = RB * NROW, R8 = RBT * 8;
  static constexpr int NR = RB * CB * 8, TS = NROW * NR * 32;      // floats per warp tile / per column tile
  static constexpr int AB = RBT * 512;                               // bytes of one slice's activation chunk
  static constexpr int REDB = NCOL * TS * 4 * (PR ? WK : 1);
  static constexpr int RING = MODE == 4 ? WK * 2 * AB : 0;
  static constexpr int SMEM = REDB > RING ? REDB : RING;            // the reduction reuses the ring after the loop
  static constexpr int NTH = 32 * NCW;
};

// One CTA: NCOL column tiles (32 * CB columns each) x NROW row groups x WK k-slices. W: prepacked, 32-column tiles
// [N/32][K/32][576]. X: ROWIN ? row-major [M][K] : fragment order [K/8][R8][8]. Y: EPI 1 fragment order [N/16][R8][8]
// (N/2 outputs), EPI 0 row-major [M][N]. zero: 0 at run time (an opaque offset for load ordering).
template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int ROWIN, int EPI, int MB, int PR = 0, int CG = 0>
__global__ void __launch_bounds__(KP<RB, CB, WK, NCOL, NROW, MODE, PR>::NTH, MB)
k_mlp(const uint8_t* __restrict__ W, const __half* __restrict__ X, __half* __restrict__ Y, int M, int N, int K,
      float epi, int zero) {
  using P = KP<RB, CB, WK, NCOL, NROW, MODE, PR>;
  static_assert(MODE == 3 || MODE == 4, "MODE 3 or 4");
  static_assert(MODE == 4 || !ROWIN, "row-major input needs the staged path");
  static_assert(MODE != 4 || (WK <= 14 && (RB * NROW) % (NCOL * NROW) == 0), "MODE 4: barriers 1..WK, even pieces");
  constexpr int NCW = P::NCW, RBT = P::RBT, R8 = P::R8, NR = P::NR, TS = P::TS, AB = P::AB;
  extern __shared__ __align__(16) uint8_t sm[];
  float* red = reinterpret_cast<float*>(sm);                       // [NCOL][NROW][NR * 32]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int NC = K / 32;
  const int rin = (lane & 3) + 4 * (lane >> 4);                    // A-fragment row inside a block of 8
  const int wk = warp % WK, gi = (warp / WK) % NCOL, gr = warp / (WK * NCOL);
  const int t = blockIdx.x * NCOL + gi;                            // column tile (32 * CB columns)
  const int s0 = NC * wk / WK, n = NC * (wk + 1) / WK - s0;        // this warp's k-chunks
  const unsigned zmask = (unsigned)zero;
  const uint8_t* wp = W + ((size_t)t * CB * NC + s0) * CHUNK;      // column block cb at + cb * NC * CHUNK
  float acc[RB][CB][8];
#pragma unroll
  for (int i = 0; i < RB; i++)
#pragma unroll
    for (int j = 0; j < CB; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;
  unsigned xr = 0;
  auto ldw = [&](uint4 (&w)[CB], unsigned (&sv)[CB], int i, unsigned dep) {
    const uint8_t* pw = wp + (size_t)i * CHUNK + dep;
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      w[cb] = ldw16<CG>(pw + (size_t)cb * NC * CHUNK + lane * 16);
      sv[cb] = ldw2<CG>(pw + (size_t)cb * NC * CHUNK + 512 + lane * 2);
    }
  };
  if constexpr (MODE == 3) {
    const __half* xb0 = X + ((size_t)s0 * 4 * R8 + gr * RB * 8 + rin) * 8;
    constexpr int XCH = 4 * R8 * 8;                                // halfs per chunk (fragment order)
    uint4 wA[CB], wB[CB], aA[4][RB], aB[4][RB];
    unsigned sA[CB], sB[CB];
    auto ldall = [&](uint4 (&w)[CB], unsigned (&sv)[CB], uint4 (&a)[4][RB], int i, unsigned dep) {
      ldw(w, sv, i, dep);
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(xb0 + (size_t)i * XCH) + dep;
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int rb = 0; rb < RB; rb++) a[jj][rb] = lda16(pa + (jj * R8 + rb * 8) * 16);
    };
    auto touch = [&](const uint4 (&w)[CB], const unsigned (&sv)[CB], const uint4 (&a)[4][RB]) {
      unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) x ^= w[cb].x ^ sv[cb];
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int rb = 0; rb < RB; rb++) x ^= a[jj][rb].x;
      xr ^= x;
      return x & zmask;
    };
    if (n > 0) {
      ldall(wA, sA, aA, 0, 0u);
      unsigned dep = touch(wA, sA, aA);
      int i = 0;
      for (; i + 2 <= n; i += 2) {
        ldall(wB, sB, aB, i + 1, dep);
        chunk_math<RB, CB>(wA, sA, [&](int jj, int rb) { return aA[jj][rb]; }, acc);
        dep = touch(wB, sB, aB);
        ldall(wA, sA, aA, min(i + 2, n - 1), dep);
        chunk_math<RB, CB>(wB, sB, [&](int jj, int rb) { return aB[jj][rb]; }, acc);
        dep = touch(wA, sA, aA);
      }
      if (i < n) chunk_math<RB, CB>(wA, sA, [&](int jj, int rb) { return aA[jj][rb]; }, acc);
    }
  } else {
    constexpr int G = NCOL * NROW, NPC = RBT / G;                  // warps per slice, 512-B pieces per warp
    const int gidx = gr * NCOL + gi;
    uint8_t* abuf = sm + (size_t)wk * 2 * AB;                      // this slice's [2][AB]
    uint4 wA[CB], wB[CB], stg[NPC];
    unsigned sA[CB], sB[CB];
    auto slice_bar = [&]() { if (G > 1) bar_sync_n(1 + wk, G * 32); else __syncwarp(); };
    auto ldx = [&](int i, unsigned dep) {
      const int ch = s0 + i;
#pragma unroll
      for (int q = 0; q < NPC; q++) {
        const int u = (q * G + gidx) * 32 + lane;                  // 16-B unit of the chunk
        if constexpr (ROWIN) {
          // row-major: u = row * 4 + jj, so a warp-load reads 8 whole 64-B row segments; rows >= M repeat row M - 1
          // (their results are discarded)
          const int row = min(u >> 2, M - 1), jj = u & 3;
          stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)row * K + ch * 32 + jj * 8) + dep);
        } else {                                                   // fragment order: u = jj * R8 + row
          stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)ch * 4 * R8 * 8) + u * 16 + dep);
        }
      }
    };
    auto stx = [&](int i) {
      uint8_t* pb = abuf + (i & 1) * AB;
#pragma unroll
      for (int q = 0; q < NPC; q++) {
        const int u = (q * G + gidx) * 32 + lane;
        const int f = ROWIN ? (u & 3) * R8 + (u >> 2) : u;         // fragment-order unit jj * R8 + row
        *reinterpret_cast<uint4*>(pb + f * 16) = stg[q];
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
    auto step = [&](const uint4 (&wc)[CB], const unsigned (&sc)[CB], uint4 (&wn)[CB], unsigned (&sn)[CB], int i, unsigned& dep) {
      const int inx = min(i + 1, n - 1);
      ldw(wn, sn, inx, dep);
      ldx(inx, dep);
      const uint8_t* ap = abuf + (i & 1) * AB + (gr * RB * 8 + rin) * 16;
      chunk_math<RB, CB>(wc, sc, [&](int jj, int rb) { return *reinterpret_cast<const uint4*>(ap + (jj * R8 + rb * 8) * 16); }, acc);
      dep = touch(wn, sn);
      stx(i + 1);
      slice_bar();
    };
    if (n > 0) {
      ldw(wA, sA, 0, 0u);
      ldx(0, 0u);
      unsigned dep = touch(wA, sA);
      stx(0);
      slice_bar();
      int i = 0;
      for (; i + 2 <= n; i += 2) {
        step(wA, sA, wB, sB, i, dep);
        step(wB, sB, wA, sA, i + 1, dep);
      }
      if (i < n) step(wA, sA, wB, sB, i, dep);
    }
    bar_sync_n(15, NCW * 32);                                     // every ring read is done before red (aliased) is written
  }
  if (xr == 0x9E3779B9u && zero == 1) Y[0] = __float2half(0.f);  // never true; keeps the forced waits live
  // reduce the WK slices of each warp tile through shared memory (PR: in parallel, same addition order)
  constexpr int SLAB = NCOL * TS;                                  // floats of one k-slice's partials (PR)
  if constexpr (PR) {
    float* rt = red + wk * SLAB + (gi * NROW + gr) * NR * 32;
#pragma unroll
    for (int i = 0; i < RB; i++)
#pragma unroll
      for (int j = 0; j < CB; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) rt[((i * CB + j) * 8 + e) * 32 + lane] = acc[i][j][e];
    bar_sync_n(15, NCW * 32);
  } else {
    float* rt = red + (gi * NROW + gr) * NR * 32;
    for (int w = 0; w < WK; w++) {
      if (wk == w)
#pragma unroll
        for (int i = 0; i < RB; i++)
#pragma unroll
          for (int j = 0; j < CB; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
              float& r = rt[((i * CB + j) * 8 + e) * 32 + lane];
              r = w ? r + acc[i][j][e] : acc[i][j][e];
            }
      bar_sync_n(15, NCW * 32);
    }
  }
  auto rv = [&](int idx) {                                         // the reduced value at red index idx
    if constexpr (PR) {
      float v = red[idx];
#pragma unroll
      for (int w = 1; w < WK; w++) v += red[w * SLAB + idx];
      return v;
    } else {
      return red[idx];
    }
  };
  auto ridx = [&](int c, int row, int col) {                      // red index of (row, column) in column tile c
    const int grr = row / (RB * 8), lrow = row % (RB * 8), rb = lrow >> 3, r8 = lrow & 7;
    const int cb = col >> 5, cc = col & 31, q = cc >> 3, cq = cc & 7;
    const int l = (r8 & 1) | (cq & 2) | (q << 2) | (((r8 >> 2) & 1) << 4);
    const int e = (r8 & 2) | (cq & 4) | (cq & 1);
    return (c * NROW + grr) * NR * 32 + ((rb * CB + cb) * 8 + e) * 32 + l;
  };
  constexpr int TC = 32 * CB;
  if constexpr (EPI == 0) {
    const int rows = min(M, R8);
    for (int idx = threadIdx.x; idx < NCOL * rows * (TC / 2); idx += NCW * 32) {
      const int c = idx / (rows * (TC / 2)), rr = idx % (rows * (TC / 2));
      const int row = rr / (TC / 2), col = (rr % (TC / 2)) * 2;
      const float v0 = rv(ridx(c, row, col)), v1 = rv(ridx(c, row, col + 1));
      *reinterpret_cast<__half2*>(Y + (size_t)row * N + (blockIdx.x * NCOL + c) * TC + col) = __floats2half2_rn(v0 * epi, v1 * epi);
    }
  } else {
    // silu(gate) * up; physical 32-column block pcb holds gate outputs [16 pcb, 16 pcb + 16) in columns 0..15 and the
    // matching up columns in 16..31. One thread writes 8 consecutive outputs of one row (one 16-B fragment unit).
    constexpr int TCO = TC / 2;
    for (int idx = threadIdx.x; idx < NCOL * (TCO / 8) * R8; idx += NCW * 32) {
      const int c = idx / ((TCO / 8) * R8), rem = idx % ((TCO / 8) * R8), og = rem / R8, row = rem % R8;
      const int o0 = og * 8, gcol0 = (o0 >> 4) * 32 + (o0 & 15);
      __align__(16) __half v[8];
#pragma unroll
      for (int e8 = 0; e8 < 8; e8++) {
        const float g = rv(ridx(c, row, gcol0 + e8)) * epi, u = rv(ridx(c, row, gcol0 + 16 + e8)) * epi;
        v[e8] = __float2half_rn(g / (1.f + __expf(-g)) * u);
      }
      const int j = (blockIdx.x * NCOL + c) * TCO + o0;            // output column (multiple of 8)
      *reinterpret_cast<uint4*>(Y + ((size_t)(j >> 3) * R8 + row) * 8) = *reinterpret_cast<const uint4*>(v);
    }
  }
}

// checkpoint NVFP4 -> kernel order. codes [N][K/2] (low nibble = even k), scales [N][K/16] E4M3. One thread per
// (32-column tile p, 32-k chunk c, lane l). gated: tile p holds gate rows [16p, 16p+16) then up rows N/2 + [16p, 16p+16).
__global__ void k_prepack(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, uint8_t* __restrict__ out,
                          int N, int K, int gated) {
  const int NC = K / 32;
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)(N / 32) * NC * 32) return;
  const int l = gid & 31, c = (gid >> 5) % NC, p = (int)((gid >> 5) / NC);
  const int q = (l >> 2) & 3, cB = (l & 3) + 4 * (l >> 4), lc = 8 * q + cB;
  const int nrow = gated ? (lc < 16 ? 16 * p + lc : N / 2 + 16 * p + lc - 16) : 32 * p + lc;
  const uint8_t* cr = codes + (size_t)nrow * (K / 2) + 16 * c;    // 32 codes of this chunk = 16 bytes
  uint32_t wv[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t w = 0;
#pragma unroll
    for (int pp = 0; pp < 4; pp++) {
      const int b = cr[4 * i + pp], lo = b & 15, hi = b >> 4;    // k = 32c + 8i + 2pp (+1)
      const uint32_t T = ((uint32_t)(lo & 7) << 9) | ((uint32_t)(lo >> 3) << 15) | ((uint32_t)(hi & 7) << 25) |
                         ((uint32_t)(hi >> 3) << 31);
      const uint32_t m = T & 0x0E000E00u, s = T & 0x80008000u;
      const int ra = RA(pp), rb = RB_(pp);
      w |= (ra ? (m >> ra) | (m << (32 - ra)) : m) | (rb ? (s >> rb) | (s << (32 - rb)) : s);
    }
    wv[i] = w;
  }
  uint8_t* o = out + ((size_t)p * NC + c) * CHUNK;
  *reinterpret_cast<uint4*>(o + l * 16) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
  o[512 + 2 * l] = scales[(size_t)nrow * (K / 16) + 2 * c];
  o[512 + 2 * l + 1] = scales[(size_t)nrow * (K / 16) + 2 * c + 1];
}

// prepacked weights -> fp16 [N][K] (physical column order), exactly E2M1 * E4M3 * 2^-14 (the GEMM's alpha carries
// 2^14 / g). One warp per (32-column tile p, 4 chunks = 128 k): each lane decodes its column's 128 values into shared
// memory, then the warp writes whole 256-byte row segments (two rows per store instruction).
__global__ void __launch_bounds__(128) k_dequant(const uint8_t* __restrict__ W, __half* __restrict__ out, int N, int K) {
  __shared__ uint4 sh[4][32][17];                                  // per warp: 32 rows x 16 x 16 B (+1 pad: no conflicts)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int NC = K / 32, NCG = NC / 4;
  const long long unit = (long long)blockIdx.x * 4 + warp;
  if (unit >= (long long)(N / 32) * NCG) return;
  const int p = (int)(unit / NCG), cg = (int)(unit % NCG);
#pragma unroll
  for (int cc = 0; cc < 4; cc++) {
    const uint8_t* src = W + ((size_t)p * NC + cg * 4 + cc) * CHUNK;
    const uint4 w = *reinterpret_cast<const uint4*>(src + lane * 16);
    const unsigned sv = *reinterpret_cast<const unsigned short*>(src + 512 + 2 * lane);
    const unsigned s0 = scale2(sv), s1 = scale2(sv >> 8);
    const unsigned wd[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned s2 = i < 2 ? s0 : s1;
      sh[warp][lane][cc * 4 + i] = make_uint4(hmul2u(dq<0>(wd[i]), s2), hmul2u(dq<1>(wd[i]), s2),
                                              hmul2u(dq<2>(wd[i]), s2), hmul2u(dq<3>(wd[i]), s2));
    }
  }
  __syncwarp();
#pragma unroll
  for (int it = 0; it < 16; it++) {
    const int idx = it * 32 + lane, r = idx >> 4, u = idx & 15;     // smem row r = the lane that decoded column n
    const int n = 32 * p + 8 * ((r >> 2) & 3) + (r & 3) + 4 * (r >> 4);
    *reinterpret_cast<uint4*>(out + (size_t)n * K + cg * 128 + u * 8) = sh[warp][r][u];
  }
}

// row-major x [M][K] -> fragment order [K/8][R8][8] (rows M..R8-1 zero): one 16-byte unit per (row, k/8)
__global__ void k_to_frag(const uint4* __restrict__ x, uint4* __restrict__ xf, int M, int K8, int R8) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= K8 * R8) return;
  const int kb = i / R8, r = i % R8;
  xf[i] = r < M ? x[(size_t)r * K8 + kb] : make_uint4(0, 0, 0, 0);
}

// silu(gate) * up from the paired gate+up GEMM output [M][2I] (physical order: tile p = 16 gate then 16 up columns)
// into row-major [M][I]; 8 outputs per thread
__global__ void k_silu_pair(const __half* __restrict__ gu, __half* __restrict__ out, int M, int I) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)M * (I / 8)) return;
  const int row = (int)(idx / (I / 8)), j0 = (int)(idx % (I / 8)) * 8;
  const __half* g = gu + (size_t)row * 2 * I + (j0 >> 4) * 32 + (j0 & 15);
  const uint4 gv = *reinterpret_cast<const uint4*>(g), uv = *reinterpret_cast<const uint4*>(g + 16);
  const __half* gh = reinterpret_cast<const __half*>(&gv);
  const __half* uh = reinterpret_cast<const __half*>(&uv);
  __align__(16) __half v[8];
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float a = __half2float(gh[e]), b = __half2float(uh[e]);
    v[e] = __float2half_rn(a / (1.f + __expf(-a)) * b);
  }
  *reinterpret_cast<uint4*>(out + (size_t)row * I + j0) = *reinterpret_cast<const uint4*>(v);
}

// C[M][Nn] = alpha * X[M][Kk] W[Nn][Kk]^T, fp16 in/out, FP32 accumulation (cuBLAS, column-major view)
void gemm(const __half* W, const __half* X, __half* C, int Nn, int M, int Kk, float alpha) {
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
  const float beta = 0.f;
  TORCH_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, Nn, M, Kk, &alpha, W, CUDA_R_16F, Kk, X, CUDA_R_16F, Kk, &beta,
                           C, CUDA_R_16F, Nn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
              "q27 mlp: cublasGemmEx failed");
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int ROWIN, int EPI, int MB, int PR = 0, int CG = 0>
void launch(const uint8_t* W, const __half* X, __half* Y, int M, int N, int K, float epi, cudaStream_t st) {
  using P = KP<RB, CB, WK, NCOL, NROW, MODE, PR>;
  auto fn = k_mlp<RB, CB, WK, NCOL, NROW, MODE, ROWIN, EPI, MB, PR, CG>;
  static bool init = false;
  if (!init) {
    // the smallest shared-memory carveout that keeps the occupancy MB (the rest stays L1)
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    int occ = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, P::NTH, P::SMEM) == cudaSuccess);
    const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * P::SMEM + 98303) / 98304));
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess);
    init = true;
  }
  const int grid = N / (32 * CB) / NCOL;
  fn<<<grid, P::NTH, P::SMEM, st>>>(W, X, Y, M, N, K, epi, 0);
}

// variants per M bucket (ladder/results/w5/confirm5.csv, confirm6.csv; µs per launch in the bench):
//   gate+up (MODE 4, row-major input, SiLU epilogue): R8 16: 2 col tiles x 4 slices (37.7); 32: 4 slices, 2 col
//   blocks (50.1); 64: 2 col tiles x 2 slices (94.8). down: 16: MODE 3, 8 slices (23.1); 32: MODE 4 2 col tiles x 4
//   slices (28.5); 64: MODE 3, 4 slices (47.3).
//   R8 8 (1..8 rows, 2026-11-21, sweep of 5 gate+up x 5 down shapes, results/sweep_mlp.log): gate+up MODE 0 on the
//   fragment copy (34.1-34.5) or MODE 4 staging row-major input (34.8); down MODE 3, 8 slices (18.0-18.3, MODE 0 18.6-19.0).
//   (MODE 0 and the fragment-order input from the fused norm went with that norm: ../../attic/mlp_fp4_fraginput.cu.)
//   At 10..48 rows the shapes below stay best; one column tile per CTA (160 CTAs) for down loses its weight reuse
//   (32 rows 34.7 against 28.9, 48 rows 52.3 against 39.2).
//   gate+up at 33..64 rows (2026-11-21, a temporary variant switch, whole MLP us, fragment input): 136 CTAs x 4 warps leaves the
//   SMs holding one CTA idle for the last ~15% (272 column tiles on 80 SMs; cta_time per-SM timeline), but every
//   other shape loses more: 136 x 8 (WK 4) 0.991-1.022x at 40/48 rows and 1.29-1.32x at 56/64, 136 x 8 (2 row
//   groups) 1.08-1.11x, 272 x 2 1.05-1.16x, 272 x 4 1.27-1.52x, 272 x 8 1.02-1.55x, 136 x 16 1.03-1.47x.
//   R8 48 (33..48 rows, 2026-11-20; test_mlp.py, whole MLP at 40 rows against the 64-row tile, row-major/fragment
//   input): gate+up 2 col tiles x 2 slices as R8 64, down MODE 4 2 col tiles x 4 slices: 127.7/112.1 µs against
//   164.0/144.5 (QPN2 155.7). Of 12 shapes tried, a MODE 3 down spills at 6 row blocks and one column tile per warp
//   pair with 2 row groups loses its weight reuse (~1.5x slower).

// smallrows: kernel variant bits (set_variant; smallrows.py sets them from Q27_SR_MLP, default 7 = adopted):
//   1 = gate+up .cg loads at 1..8 rows, 2 = down .cg at 1..8, 4 = parallel k-slice reduction at 1..16 (both kernels;
//   all three bitwise equal to the p14 kernels). 0 = the p14 kernels. (.cg at 9..16 rows lost 2.4% of the C4 step in
//   the server; .cg at 17..64 rows lost 1-7 us per kernel on one GPU and was not taken further.)
int MLPV = 7;
template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int ROWIN, int EPI, int MB, int CGBIT, int PRBIT>
void launch_v(const uint8_t* W, const __half* X, __half* Y, int M, int N, int K, float epi, cudaStream_t st, int v) {
  constexpr int PRV = PRBIT ? 1 : 0;
  const bool pr = PRBIT && (v & PRBIT) != 0;
  if constexpr (CGBIT != 0) {
    if (v & CGBIT) {
      if (pr) launch<RB, CB, WK, NCOL, NROW, MODE, ROWIN, EPI, MB, PRV, 1>(W, X, Y, M, N, K, epi, st);
      else launch<RB, CB, WK, NCOL, NROW, MODE, ROWIN, EPI, MB, 0, 1>(W, X, Y, M, N, K, epi, st);
      return;
    }
  }
  if (pr) launch<RB, CB, WK, NCOL, NROW, MODE, ROWIN, EPI, MB, PRV, 0>(W, X, Y, M, N, K, epi, st);
  else launch<RB, CB, WK, NCOL, NROW, MODE, ROWIN, EPI, MB, 0, 0>(W, X, Y, M, N, K, epi, st);
}
}  // namespace
void set_variant(int64_t v) { MLPV = (int)v; }
int64_t get_variant() { return MLPV; }

torch::Tensor mlp(torch::Tensor x, torch::Tensor wgu, torch::Tensor wd, double epi_gu, double epi_d, int64_t ir,
                  torch::Tensor ws);
int CHUNKED_MAX = 256;   // rows up to which prefill runs the decode kernels in 64-row chunks (set_chunked_max)
void set_chunked_max(int64_t m) { CHUNKED_MAX = (int)m; }

torch::Tensor prepack(torch::Tensor codes, torch::Tensor scales, bool gated) {
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.dim() == 2 && codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 && scales.dim() == 2 && scales.is_contiguous());
  const int N = codes.size(0), K = codes.size(1) * 2;
  TORCH_CHECK(N % 32 == 0 && K % 32 == 0 && scales.size(0) == N && scales.size(1) == K / 16);
  TORCH_CHECK(!gated || N % 32 == 0);
  const at::cuda::CUDAGuard guard(codes.device());
  auto out = torch::empty({(long long)N / 32 * (K / 32) * CHUNK}, codes.options());
  const long long total = (long long)(N / 32) * (K / 32) * 32;
  k_prepack<<<(unsigned)((total + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), N, K, gated ? 1 : 0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  return out;
}

// x [M, H] fp16 row-major; wgu: prepack(gate_up [2 Ir, H/2], gated);
// wd: prepack(down [H, Ir/2]); ws: fp16 workspace of >= max(2 Ir H, H Ir) elements, used only for M > 64 (prefill:
// exact dequantisation + cuBLAS). Returns [M, H] fp16 (before the TP all-reduce).
// MLP chunk plan for M > 64 rows: chunks of at most 64 rows, sized to minimise the summed per-bucket cost of the
// decode kernels on row-major input, as measured for chunks inside a CUDA graph sequence (chunk_bench, whole MLP, µs:
// 1-16 rows 64, 17-32 87, 33-48 129, 49-64 166). 65..95 rows gain (80 = 32 + 48: 216 against 229 for 64 + 16); from
// 96 rows the plan is 64-row chunks again. Returns the chunk sizes in order.
std::vector<int> chunk_plan(int M) {
  auto cost = [](int r) { return r <= 16 ? 64 : r <= 32 ? 87 : r <= 48 ? 129 : 166; };
  std::vector<int> best(M + 1, 0), take(M + 1, 0);
  for (int m = 1; m <= M; m++) {
    best[m] = INT32_MAX;
    for (int c : {16, 32, 48, 64}) {
      const int s = std::min(c, m), v = cost(s) + best[m - s];
      if (v < best[m]) { best[m] = v; take[m] = s; }
    }
  }
  std::vector<int> plan;
  for (int m = M; m > 0; m -= take[m]) plan.push_back(take[m]);
  return plan;
}

// out [M][H] fp16, contiguous (a row range of a larger output is fine)
void mlp_into(torch::Tensor x, torch::Tensor wgu, torch::Tensor wd, double epi_gu, double epi_d, int64_t ir,
              torch::Tensor ws, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  const int M = x.size(0), H = x.size(1), I = (int)ir;
  TORCH_CHECK(M >= 1, "q27 mlp: at least one row");
  TORCH_CHECK(H % 32 == 0 && I % 32 == 0);
  TORCH_CHECK(wgu.numel() == (long long)(2 * I) / 32 * (H / 32) * CHUNK && wd.numel() == (long long)H / 32 * (I / 32) * CHUNK);
  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(out.dtype() == torch::kHalf && out.is_contiguous() && out.size(0) == M && out.size(1) == H, "q27 mlp: out");
  if (M > 64 && M <= CHUNKED_MAX) {
    // small prefill / mixed batches and MTP verify batches: the decode kernels on chunks of the row-major input
    // (no dequantisation), each written straight into its rows of out
    int r0 = 0;
    for (int c : chunk_plan(M)) {
      mlp_into(x.narrow(0, r0, c), wgu, wd, epi_gu, epi_d, ir, ws, out.narrow(0, r0, c));
      r0 += c;
    }
    return;
  }
  if (M > 64) {
    TORCH_CHECK(ws.numel() >= std::max((long long)2 * I * H, (long long)H * I) && ws.dtype() == torch::kHalf);
    __half* Wt = reinterpret_cast<__half*>(ws.data_ptr<at::Half>());
    auto gu = torch::empty({M, 2 * I}, x.options());
    auto mid = torch::empty({M, I}, x.options());
    TORCH_CHECK((H / 32) % 4 == 0 && (I / 32) % 4 == 0, "q27 mlp prefill: K must be a multiple of 128");
    const long long t1 = (long long)(2 * I / 32) * (H / 128), t2 = (long long)(H / 32) * (I / 128);   // warps
    k_dequant<<<(unsigned)((t1 + 3) / 4), 128, 0, st>>>(wgu.data_ptr<uint8_t>(), Wt, 2 * I, H);
    gemm(Wt, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), reinterpret_cast<__half*>(gu.data_ptr<at::Half>()),
         2 * I, M, H, (float)epi_gu);
    const long long ts = (long long)M * (I / 8);
    k_silu_pair<<<(unsigned)((ts + 255) / 256), 256, 0, st>>>(reinterpret_cast<const __half*>(gu.data_ptr<at::Half>()),
                                                            reinterpret_cast<__half*>(mid.data_ptr<at::Half>()), M, I);
    k_dequant<<<(unsigned)((t2 + 3) / 4), 128, 0, st>>>(wd.data_ptr<uint8_t>(), Wt, H, I);
    gemm(Wt, reinterpret_cast<const __half*>(mid.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
         H, M, I, (float)epi_d);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    return;
  }
  const int R8 = M <= 8 ? 8 : M <= 16 ? 16 : M <= 32 ? 32 : M <= 48 ? 48 : 64;   // row bucket (padded rows)
  auto mid = torch::empty({(long long)I / 8 * R8 * 8}, x.options());  // silu(gate) * up, fragment order [I/8][R8][8]
  const uint8_t* Wg = wgu.data_ptr<uint8_t>();
  const uint8_t* Wd = wd.data_ptr<uint8_t>();
  const __half* X = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  __half* Mid = reinterpret_cast<__half*>(mid.data_ptr<at::Half>());
  __half* Y = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  const float eg = (float)epi_gu, ed = (float)epi_d;
  if (R8 == 8) {
    // 1..8 rows (point 12, 2026-11-21): 8-row tiles, gate+up staging the rows, down MODE 3. Until then these rows ran
    // the 16-row tile: 59.7-61.8 us per MLP on one GPU against 52.1-53.2 here; in the server down drops 24.1 -> 20.0
    // us per layer, gate+up stays 39.3 -> 39.8, and the step moves <= 0.8%.
    launch_v<1, 1, 8, 1, 1, 4, 1, 1, 1, 1, 4>(Wg, X, Mid, M, 2 * I, H, eg, st, MLPV);  // smallrows: variant bits
    launch_v<1, 1, 8, 1, 1, 3, 0, 0, 1, 2, 4>(Wd, Mid, Y, M, H, I, ed, st, MLPV);
  } else if (R8 == 16) {
    // 9..16 rows: gate+up stages the rows (at 16 rows or fewer the fragment copy costs more than staging saves)
    launch_v<2, 1, 4, 2, 1, 4, 1, 1, 2, 0, 4>(Wg, X, Mid, M, 2 * I, H, eg, st, MLPV);  // smallrows: PR bit only
    launch_v<2, 1, 8, 1, 1, 3, 0, 0, 1, 0, 4>(Wd, Mid, Y, M, H, I, ed, st, MLPV);
  } else {
    // 17..64 rows: one copy into fragment order, then the fragment kernels. frag_bench.py (attic) 2026-11-21, us per
    // MLP staged -> copy + fragment path: 24 rows 84.2 -> 83.5, 32: 85.0 -> 83.7, 40: 128.4 -> 110.8, 48: 127.0 ->
    // 112.2, 64: 161.3 -> 142.3; at 16 rows or fewer the copy costs more than staging saves (1.005-1.023x).
    auto xf = torch::empty({(long long)H / 8 * R8 * 8}, x.options());
    const int units = H / 8 * R8;
    k_to_frag<<<(units + 255) / 256, 256, 0, st>>>(reinterpret_cast<const uint4*>(x.data_ptr<at::Half>()),
                                                 reinterpret_cast<uint4*>(xf.data_ptr<at::Half>()), M, H / 8, R8);
    const __half* XF = reinterpret_cast<const __half*>(xf.data_ptr<at::Half>());
    if (R8 == 32) {
      launch<4, 2, 4, 1, 1, 4, 0, 1, 1>(Wg, XF, Mid, M, 2 * I, H, eg, st);
      launch<4, 1, 4, 2, 1, 4, 0, 0, 2>(Wd, Mid, Y, M, H, I, ed, st);
    } else if (R8 == 48) {
      launch<6, 1, 2, 2, 1, 4, 0, 1, 3>(Wg, XF, Mid, M, 2 * I, H, eg, st);
      launch<6, 1, 4, 2, 1, 4, 0, 0, 2>(Wd, Mid, Y, M, H, I, ed, st);
    } else {
      launch<8, 1, 2, 2, 1, 4, 0, 1, 3>(Wg, XF, Mid, M, 2 * I, H, eg, st);
      launch<8, 1, 4, 1, 1, 3, 0, 0, 1>(Wd, Mid, Y, M, H, I, ed, st);
    }
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

torch::Tensor mlp(torch::Tensor x, torch::Tensor wgu, torch::Tensor wd, double epi_gu, double epi_d, int64_t ir,
                  torch::Tensor ws) {
  TORCH_CHECK(x.dim() == 2);
  auto out = torch::empty({x.size(0), x.size(1)}, x.options());
  mlp_into(x, wgu, wd, epi_gu, epi_d, ir, ws, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack", &prepack, "NVFP4 checkpoint layout -> q27 MLP kernel layout");
  m.def("mlp", &mlp, "NVFP4 MLP (gate+up, SiLU-and-mul, down): decode kernels up to 64 rows (chunked up to 256), dequant + cuBLAS above");
  m.def("set_variant", &set_variant, "smallrows: kernel variant bits (see MLPV)");
  m.def("get_variant", &get_variant, "smallrows: kernel variant bits");
  m.def("chunk_plan", &chunk_plan, "row chunks the decode kernels take for 65..256 rows");
  m.def("set_chunked_max", &set_chunked_max, "rows up to which prefill uses 64-row chunks of the decode kernels (test hook)");
}
