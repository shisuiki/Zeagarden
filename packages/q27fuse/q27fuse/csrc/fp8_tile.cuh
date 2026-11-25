// fp8_tile.cuh - one CTA of the FP8 (E4M3, per-channel scale) W8A16 linear on 1Cat's QPN8 layout (see fp8_lin.cu for
// the layout and the MODE 4 structure). Used by fp8_lin.cu (k_fp8: one CTA per column block; the attic's fused GDN
// layer kernel ran it in its GEMM CTAs). bx = this CTA's column block (NCOL tiles of 32 CB columns);
// sm = dynamic shared memory of KP8<...>::SMEM bytes.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace q27fp8 {
constexpr int CH8 = 1024;                                          // bytes per (32-column tile, 32-k chunk)

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
__device__ __forceinline__ uint4 lda16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
// smallrows: weight loads, CG 0 = ld.global.nc.L1::no_allocate (read-only path), CG 1 = ld.global.cg (fp8_gdn_in.cu at
// 33..64 rows; why: mlp_fp4.cu ldw16)
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
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ unsigned f8x2(unsigned p) {             // bytes 0 and 2 of p (E4M3) -> fp16x2 = E4M3 * 2^-8
  return ((p << 8) & 0x80008000u) | ((p << 7) & 0x3F803F80u);
}

// one 32-k chunk: w[cb][g] = this lane's 16 B of k-group g (k 0..15, 16..31); aget(jj, rb) = A fragment (8 halves of
// k 8 jj .. 8 jj + 7) of row block rb
template <int RB, int CB, class AGet>
__device__ __forceinline__ void chunk8(const uint4 (&w)[CB][2], AGet aget, float (&acc)[RB][CB][8]) {
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    uint4 a[RB];
#pragma unroll
    for (int rb = 0; rb < RB; rb++) a[rb] = aget(jj, rb);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const uint4& g = w[cb][jj >> 1];
      const unsigned qx = (jj & 1) ? g.z : g.x, qy = (jj & 1) ? g.w : g.y;
      const unsigned p0 = f8x2(__byte_perm(qx, qy, 0x0400)), p1 = f8x2(__byte_perm(qx, qy, 0x0501));
      const unsigned p2 = f8x2(__byte_perm(qx, qy, 0x0602)), p3 = f8x2(__byte_perm(qx, qy, 0x0703));
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        mma884(acc[rb][cb], a[rb].x, a[rb].y, p0, p1);
        mma884(acc[rb][cb], a[rb].z, a[rb].w, p2, p3);
      }
    }
  }
}

// PR (smallrows, from kgap): parallel k-slice reduction, bitwise identical to the serial one (WK slabs summed in order)
template <int RB, int CB, int WK, int NCOL, int NROW, int MODE = 4, int PR = 0>
struct KP8 {
  static constexpr int NCW = WK * NCOL * NROW, RBT = RB * NROW, R8 = RBT * 8;
  static constexpr int NR = RB * CB * 8, TS = NROW * NR * 32;      // floats per warp tile / per column tile
  static constexpr int AB = RBT * 512;                               // bytes of one slice's activation chunk
  static constexpr int REDB = NCOL * TS * 4 * (PR ? WK : 1), RING = WK * 2 * AB;
  static constexpr int SMEM = REDB > RING ? REDB : RING;            // the reduction reuses the ring after the loop
  static constexpr int NTH = 32 * NCW;
};

// One CTA: NCOL column tiles (32 CB columns each) x NROW row groups (8 RB rows each) x WK k-slices; rows padded to
// R8 = 8 RB NROW (rows >= M repeat row M - 1, results discarded). X row-major [M][K]; Y row-major [M][N].
// smallrows: with Y2 != nullptr the output is split by column: columns < split go to Y [M][split], the rest to
// Y2 [M][N - split] (the GDN in-projection's qkv | z; split must be a multiple of the CTA's columns).
// smallrows GATED: the FP8 MLP gate_up (QPN8 codes of the merged [gate | up] weight, N = 2 I): CB = 2 column blocks =
// gate tile t and up tile t + N / 64; epilogue silu(gate) * up (fp32) -> Y [M][N / 2] fp16.
template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int PR = 0, int CG = 0, int GATED = 0>
__device__ __forceinline__ void fp8_tile(const uint8_t* __restrict__ W, const __half* __restrict__ X,
                                         const __half* __restrict__ S, __half* __restrict__ Y, int M, int N, int K,
                                         int zero, int bx, uint8_t* sm, __half* __restrict__ Y2 = nullptr,
                                         int split = 0x7FFFFFFF) {
  using P = KP8<RB, CB, WK, NCOL, NROW, MODE, PR>;
  static_assert(MODE == 4, "MODE 4: staged activations (direct A-fragment loads and L1-allocated weights lost)");
  static_assert(WK <= 14 && (RB * NROW) % (NCOL * NROW) == 0, "named barriers 1..WK, whole staging pieces");
  constexpr int NCW = P::NCW, R8 = P::R8, NR = P::NR, AB = P::AB, G = NCOL * NROW, NPC = P::RBT / G;
  float* red = reinterpret_cast<float*>(sm);                       // [NCOL][NROW][NR * 32]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int NC = K / 32;
  const int rin = (lane & 3) + 4 * (lane >> 4);                    // A-fragment row inside a block of 8
  const int wk = warp % WK, gi = (warp / WK) % NCOL, gr = warp / (WK * NCOL);
  const int t = bx * NCOL + gi;                            // column tile (32 CB columns)
  const int s0 = NC * wk / WK, n = NC * (wk + 1) / WK - s0;        // this warp's k-chunks
  const unsigned zmask = (unsigned)zero;
  static_assert(!GATED || CB == 2, "gated: one gate and one up column block");
  const uint8_t* wp = W + ((size_t)t * (GATED ? 1 : CB) * NC + s0) * CH8 + lane * 16;
  const size_t cbs = GATED ? (size_t)(N / 64) * NC * CH8 : (size_t)NC * CH8;   // bytes between column blocks
  float acc[RB][CB][8];
#pragma unroll
  for (int i = 0; i < RB; i++)
#pragma unroll
    for (int j = 0; j < CB; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;
  unsigned xr = 0;
  auto ldw = [&](uint4 (&w)[CB][2], int i, unsigned dep) {
    const uint8_t* pw = wp + (size_t)i * CH8 + dep;
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      w[cb][0] = ldw16<CG>(pw + cb * cbs);
      w[cb][1] = ldw16<CG>(pw + cb * cbs + 512);
    }
  };
  const int gidx = gr * NCOL + gi;
  uint8_t* abuf = sm + (size_t)wk * 2 * AB;                        // this slice's [2][AB]
  uint4 wA[CB][2], wB[CB][2], stg[NPC];
  auto slice_bar = [&]() { if (G > 1) bar_sync_n(1 + wk, G * 32); else __syncwarp(); };
  auto ldx = [&](int i, unsigned dep) {
    const int ch = s0 + i;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = (q * G + gidx) * 32 + lane;                    // 16-B unit u = row * 4 + jj of the chunk
      const int row = min(u >> 2, M - 1), jj = u & 3;
      stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)row * K + ch * 32 + jj * 8) + dep);
    }
  };
  auto stx = [&](int i) {
    uint8_t* pb = abuf + (i & 1) * AB;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = (q * G + gidx) * 32 + lane;
      *reinterpret_cast<uint4*>(pb + ((u & 3) * R8 + (u >> 2)) * 16) = stg[q];   // fragment order jj * R8 + row
    }
  };
  auto touch = [&](const uint4 (&w)[CB][2]) {
    unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) x ^= w[cb][0].x ^ w[cb][1].x;
#pragma unroll
    for (int q = 0; q < NPC; q++) x ^= stg[q].x;
    xr ^= x;
    return x & zmask;
  };
  auto step = [&](const uint4 (&wc)[CB][2], uint4 (&wn)[CB][2], int i, unsigned& dep) {
    const int inx = min(i + 1, n - 1);
    ldw(wn, inx, dep);
    ldx(inx, dep);
    const uint8_t* ap = abuf + (i & 1) * AB + (gr * RB * 8 + rin) * 16;
    chunk8<RB, CB>(wc, [&](int jj, int rb) { return *reinterpret_cast<const uint4*>(ap + (jj * R8 + rb * 8) * 16); }, acc);
    dep = touch(wn);
    stx(i + 1);
    slice_bar();
  };
  if (n > 0) {
    ldw(wA, 0, 0u);
    ldx(0, 0u);
    unsigned dep = touch(wA);
    stx(0);
    slice_bar();
    int i = 0;
    for (; i + 2 <= n; i += 2) {
      step(wA, wB, i, dep);
      step(wB, wA, i + 1, dep);
    }
    if (i < n) step(wA, wB, i, dep);
  }
  bar_sync_n(15, NCW * 32);                                       // every ring read is done before red (aliased) is written
  if (xr == 0x9E3779B9u && zero == 1) Y[0] = __float2half(0.f);  // never true; keeps the forced waits live
  // reduce the WK slices of each warp tile through shared memory (PR: in parallel, same addition order)
  constexpr int SLAB = NCOL * P::TS;
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
  auto ridx = [&](int c, int row, int col) {                      // red index of (row, column) in column tile c
    const int grr = row / (RB * 8), lrow = row % (RB * 8), rb = lrow >> 3, r8 = lrow & 7;
    const int cb = col >> 5, cc = col & 31, q = cc >> 3, cq = cc & 7;
    const int l = (r8 & 1) | (cq & 2) | (q << 2) | (((r8 >> 2) & 1) << 4);
    const int e = (r8 & 2) | (cq & 4) | (cq & 1);
    return (c * NROW + grr) * NR * 32 + ((rb * CB + cb) * 8 + e) * 32 + l;
  };
  constexpr int TC = 32 * CB;
  const int rows = min(M, R8);
  if constexpr (GATED) {
    const int NO = N / 2;
    for (int idx = threadIdx.x; idx < NCOL * rows * 16; idx += NCW * 32) {
      const int c = idx / (rows * 16), rr = idx % (rows * 16), row = rr / 16, col = (rr % 16) * 2;
      const int gcol = (bx * NCOL + c) * 32 + col;
      const __half2 sg = *reinterpret_cast<const __half2*>(S + gcol), su = *reinterpret_cast<const __half2*>(S + NO + gcol);
      const float g0 = rv(ridx(c, row, col)) * __low2float(sg), g1 = rv(ridx(c, row, col + 1)) * __high2float(sg);
      const float u0 = rv(ridx(c, row, 32 + col)) * __low2float(su), u1 = rv(ridx(c, row, 33 + col)) * __high2float(su);
      *reinterpret_cast<__half2*>(Y + (size_t)row * NO + gcol) =
          __floats2half2_rn(g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < NCOL * rows * (TC / 2); idx += NCW * 32) {
    const int c = idx / (rows * (TC / 2)), rr = idx % (rows * (TC / 2));
    const int row = rr / (TC / 2), col = (rr % (TC / 2)) * 2;
    const int gcol = (bx * NCOL + c) * TC + col;
    const __half2 s = *reinterpret_cast<const __half2*>(S + gcol);
    const float v0 = rv(ridx(c, row, col)) * __low2float(s), v1 = rv(ridx(c, row, col + 1)) * __high2float(s);
    __half* yp = Y2 == nullptr ? Y + (size_t)row * N + gcol
                 : gcol < split ? Y + (size_t)row * split + gcol : Y2 + (size_t)row * (N - split) + (gcol - split);
    *reinterpret_cast<__half2*>(yp) = __floats2half2_rn(v0, v1);
  }
}
}  // namespace q27fp8
