// fp8_head.cu (track head) - the target LM head (TP shard: N = 62,080 vocabulary rows x K = 5,120, channel-FP8 E4M3 with
// one scale per vocabulary row) on our FP8 family (fp8_tile.cuh's MODE 4 structure: K split across the CTA's warps,
// weights double-buffered in registers, activations staged per k-slice in shared memory, parallel slice reduction).
// ONE copy of the head per rank: the QPN8 layout (fp8_lin.cu header; built by 1Cat's fp8_qpn8_prepare_sm70 from the
// checkpoint rows) + fp16 scales = fp16(scale * 256) [N] + f32 scales [N]. Also read by draft_screen.cu's rerank (gather).
// Math (WS 1, the default): every weight is decoded to exactly fp16(code * scale) - the E4M3 bits placed in an fp16
// (value * 2^-8), one HMUL2 by fp16(scale * 256) - which is the weight TurboMind's FP8 head GEMM uses (draftscreen
// checked fp16(code * scale) == 1Cat's _hot_rows for every element); FP32 accumulation on the tensor cores; fp16 logits.
// WS 0 (bench only): exact E4M3 * 2^-8 operands, the f32 scale applied to the f32 sum in the epilogue.
// Epilogues: EPI 0 = logits Y [M][ldy] fp16; EPI 1 = top-1: per row the pair torch's logits.max(-1) gives on these fp16
// logits (NaN above every number, ties and repeated NaNs -> lowest index, -0 == +0), as (value f32, global index f32)
// at pairs + 2 row, through a 32-bit atomic max per row (ordered fp16 value << 16 | 0xFFFF - shard column; CTA b uses key
// slot b % 32, so the CTAs do not all hit the same L2 lines) and a one-CTA finalize kernel after the grid (k_head_fin),
// which writes the pairs and resets the keys (i32 [32 slots][64 rows], zero at rest) - logits never stored.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

#include "fp8_tile.cuh"

namespace {
using namespace q27fp8;
constexpr int KSLOTS = 32, KROWS = 64;     // top-1 key slots x rows per launch (keys i32 [KSLOTS * KROWS], zero at rest)
__device__ int g_dbg = 0;                  // bench-only timing experiments (Q27_HEAD_DBG; wrong results when != 0)

__device__ __forceinline__ uint32_t ord16(__half h) {
  uint32_t u = __half_as_ushort(h);
  if ((u & 0x7FFFu) > 0x7C00u) return 0xFFFFu;    // every NaN above +inf
  if (u == 0x8000u) u = 0u;                       // -0 == +0
  return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}
__device__ __forceinline__ float unord16(uint32_t o) {
  const uint32_t b = o >= 0x8000u ? (o ^ 0x8000u) : (~o & 0xFFFFu);
  return __half2float(__ushort_as_half((unsigned short)b));
}

// one 32-k chunk with the per-weight fp16 scale (WS 1): fp16(code * 2^-8) * fp16(scale * 256) per weight; CB column
// blocks share every A fragment; the k 0-3 MMAs of all row blocks are issued before the k 4-7 ones (independent chains)
template <int RB, int CB, int WS, class AGet>
__device__ __forceinline__ void chunk8h(const uint4 (&w)[CB][2], AGet aget, float (&acc)[RB][CB][8],
                                        const unsigned (&s2)[CB]) {
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    uint4 a[RB];
#pragma unroll
    for (int rb = 0; rb < RB; rb++) a[rb] = aget(jj, rb);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const uint4& g = w[cb][jj >> 1];
      const unsigned qx = (jj & 1) ? g.z : g.x, qy = (jj & 1) ? g.w : g.y;
      unsigned p0 = f8x2(__byte_perm(qx, qy, 0x0400)), p1 = f8x2(__byte_perm(qx, qy, 0x0501));
      unsigned p2 = f8x2(__byte_perm(qx, qy, 0x0602)), p3 = f8x2(__byte_perm(qx, qy, 0x0703));
      if (WS) {
        const __half2 s = *reinterpret_cast<const __half2*>(&s2[cb]);
        __half2 h0 = __hmul2(*reinterpret_cast<__half2*>(&p0), s), h1 = __hmul2(*reinterpret_cast<__half2*>(&p1), s);
        __half2 h2 = __hmul2(*reinterpret_cast<__half2*>(&p2), s), h3 = __hmul2(*reinterpret_cast<__half2*>(&p3), s);
        p0 = *reinterpret_cast<unsigned*>(&h0); p1 = *reinterpret_cast<unsigned*>(&h1);
        p2 = *reinterpret_cast<unsigned*>(&h2); p3 = *reinterpret_cast<unsigned*>(&h3);
      }
#pragma unroll
      for (int rb = 0; rb < RB; rb++) mma884(acc[rb][cb], a[rb].x, a[rb].y, p0, p1);
#pragma unroll
      for (int rb = 0; rb < RB; rb++) mma884(acc[rb][cb], a[rb].z, a[rb].w, p2, p3);
    }
  }
}

// One CTA: NCOL column tiles (32 vocabulary rows each) x NROW row groups (8 RB rows each) x WK k-slices; rows padded to
// R8 = 8 RB NROW (rows >= M repeat row M - 1, results discarded). X [M][ldx] fp16 (ldx % 8 == 0).
template <int RB, int CB, int WK, int NCOL, int NROW, int EPI, int WS, int PR>
__device__ __forceinline__ void head_tile(const uint8_t* __restrict__ W, const __half* __restrict__ X, int ldx,
                                          const __half* __restrict__ S16, const float* __restrict__ S32,
                                          __half* __restrict__ Y, int ldy, int M, int N, int K, int nvalid, int vstart,
                                          unsigned* __restrict__ keys, int* __restrict__ counter,
                                          float* __restrict__ pairs, int zero, int bx, uint8_t* sm) {
  using P = KP8<RB, CB, WK, NCOL, NROW, 4, PR>;
  static_assert(WK <= 14 && (RB * NROW) % (NCOL * NROW) == 0, "named barriers 1..WK, whole staging pieces");
  constexpr int NCW = P::NCW, R8 = P::R8, NR = P::NR, AB = P::AB, G = NCOL * NROW, NPC = P::RBT / G;
  float* red = reinterpret_cast<float*>(sm);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int NC = K / 32;
  const int rin = (lane & 3) + 4 * (lane >> 4);
  const int wk = warp % WK, gi = (warp / WK) % NCOL, gr = warp / (WK * NCOL);
  const int t = bx * NCOL + gi;                                     // column tile of 32 CB columns
  const int s0 = NC * wk / WK, n = NC * (wk + 1) / WK - s0;
  const unsigned zmask = (unsigned)zero;
  const uint8_t* wp = W + ((size_t)t * CB * NC + s0) * CH8 + lane * 16;
  const size_t cbs = (size_t)NC * CH8;                              // bytes between the CB 32-column blocks
  unsigned s2[CB];
#pragma unroll
  for (int cb = 0; cb < CB; cb++) {
    s2[cb] = 0;
    if (WS) {
      const int col = (t * CB + cb) * 32 + 8 * ((lane >> 2) & 3) + (lane & 3) + 4 * (lane >> 4);
      const __half s = S16[col];
      const __half2 ss = __halves2half2(s, s);
      s2[cb] = *reinterpret_cast<const unsigned*>(&ss);
    }
  }
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
      w[cb][0] = ldg16na(pw + cb * cbs);
      w[cb][1] = ldg16na(pw + cb * cbs + 512);
    }
  };
  const int gidx = gr * NCOL + gi;
  uint8_t* abuf = sm + (size_t)wk * 2 * AB;
  uint4 wA[CB][2], wB[CB][2], stg[NPC];
  auto slice_bar = [&]() { if (G > 1) bar_sync_n(1 + wk, G * 32); else __syncwarp(); };
  auto ldx_ = [&](int i, unsigned dep) {
    const int ch = s0 + i;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = (q * G + gidx) * 32 + lane;
      const int row = min(u >> 2, M - 1), jj = u & 3;
      stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)row * ldx + ch * 32 + jj * 8) + dep);
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
    ldx_(inx, dep);
    const uint8_t* ap = abuf + (i & 1) * AB + (gr * RB * 8 + rin) * 16;
    chunk8h<RB, CB, WS>(wc, [&](int jj, int rb) { return *reinterpret_cast<const uint4*>(ap + (jj * R8 + rb * 8) * 16); },
                    acc, s2);
    dep = touch(wn);
    stx(i + 1);
    slice_bar();
  };
  if (n > 0) {
    ldw(wA, 0, 0u);
    ldx_(0, 0u);
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
  bar_sync_n(15, NCW * 32);
  if (xr == 0x9E3779B9u && zero == 1) pairs[0] = 0.f;            // never true; keeps the forced waits live
  constexpr int SLAB = NCOL * P::TS;
  if constexpr (PR) {                                            // WK slabs, summed in order by the epilogue
    float* rt = red + wk * SLAB + (gi * NROW + gr) * NR * 32;
#pragma unroll
    for (int i = 0; i < RB; i++)
#pragma unroll
      for (int j = 0; j < CB; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) rt[((i * CB + j) * 8 + e) * 32 + lane] = acc[i][j][e];
    bar_sync_n(15, NCW * 32);
  } else {                                                       // one slab, the slices added in order (same sums)
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
    float v = red[idx];
    if constexpr (PR) {
#pragma unroll
      for (int w = 1; w < WK; w++) v += red[w * SLAB + idx];
    }
    return v;
  };
  auto ridx = [&](int c, int row, int col) {
    const int grr = row / (RB * 8), lrow = row % (RB * 8), rb = lrow >> 3, r8 = lrow & 7;
    const int cb = col >> 5, cc = col & 31, q = cc >> 3, cq = cc & 7;
    const int l = (r8 & 1) | (cq & 2) | (q << 2) | (((r8 >> 2) & 1) << 4);
    const int e = (r8 & 2) | (cq & 4) | (cq & 1);
    return (c * NROW + grr) * NR * 32 + ((rb * CB + cb) * 8 + e) * 32 + l;
  };
  const int rows = min(M, R8);
  constexpr int TC = 32 * CB;
  auto value = [&](int c, int row, int col) {                     // fp32 logit before the fp16 rounding
    const float v = rv(ridx(c, row, col));
    return WS ? v : v * S32[(bx * NCOL + c) * TC + col] * 256.f;
  };
  if constexpr (EPI == 0) {
    for (int idx = threadIdx.x; idx < NCOL * rows * (TC / 2); idx += NCW * 32) {
      const int c = idx / (rows * (TC / 2)), rr = idx % (rows * (TC / 2)), row = rr / (TC / 2), col = (rr % (TC / 2)) * 2;
      const int gcol = (bx * NCOL + c) * TC + col;
      *reinterpret_cast<__half2*>(Y + (size_t)row * ldy + gcol) =
          __floats2half2_rn(value(c, row, col), value(c, row, col + 1));
    }
  } else {
    // 32-bit keys (ordered fp16 value << 16 | 0xFFFF - shard column; the shard has < 65536 columns) per (row, column
    // pair): the 16 CB column pairs of one (tile, row) are 16 CB consecutive threads of one warp, so a shuffle max over
    // them, then a shared atomic max per row, then one global RED max per row and CTA (all native 32-bit atomics)
    __shared__ unsigned skey[R8];
    for (int i = threadIdx.x; i < R8; i += NCW * 32) skey[i] = 0u;
    __syncthreads();
    constexpr int GP = TC / 2;                                   // column pairs per (tile, row): 16 CB lanes
    constexpr int RPP = NCW * 32 / GP;                           // rows per pass
    constexpr int U = R8 / RPP < 1 ? 1 : (R8 / RPP > 4 ? 4 : R8 / RPP);   // rows per thread per pass (chains)
    static_assert(GP <= 32 && 32 % GP == 0, "a (tile, row) group lies inside one warp");
    // Each thread owns one column pair p of every (tile c, row): the column part of the index math is per thread, the
    // row part is constant-divisor arithmetic (the epilogue runs with only this CTA's warps on the SM at 33-64 rows, so
    // its instruction count is exposed: a generic (c, row, column) item loop cost 30-45 us per call there).
    const int dbg = g_dbg;
    const int p = threadIdx.x % GP, r0 = threadIdx.x / GP, col = 2 * p;
#pragma unroll
    for (int c = 0; c < NCOL; c++) {
      const int gcol = (bx * NCOL + c) * TC + col;
      const bool ok0 = gcol < nvalid, ok1 = gcol + 1 < nvalid;
      for (int rb0 = 0; rb0 < ((dbg & 4) ? 0 : rows); rb0 += U * RPP) {
        unsigned key[U];
        int rowu[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
          const int row = rb0 + u * RPP + r0;
          rowu[u] = row;
          key[u] = 0u;
          if (row < rows) {
            const unsigned a = ord16(__float2half_rn(value(c, row, col)));
            const unsigned b = ord16(__float2half_rn(value(c, row, col + 1)));
            const unsigned k0 = ok0 ? (a << 16) | (0xFFFFu - (unsigned)gcol) : 0u;
            const unsigned k1 = ok1 ? (b << 16) | (0xFFFFu - (unsigned)(gcol + 1)) : 0u;
            key[u] = k1 > k0 ? k1 : k0;
          }
        }
#pragma unroll
        for (int o = GP / 2; o; o >>= 1)
#pragma unroll
          for (int u = 0; u < U; u++) if (!(dbg & 1)) {
            const unsigned t2 = __shfl_xor_sync(0xffffffffu, key[u], o);
            key[u] = t2 > key[u] ? t2 : key[u];
          }
#pragma unroll
        for (int u = 0; u < U; u++)
          if (p == 0 && key[u]) atomicMax(skey + rowu[u], key[u]);
      }
    }
    __syncthreads();
    // the CTA's row maxima go to key slot bx % KSLOTS ([KSLOTS][KROWS] keys): spreads the same-address atomics of the
    // ~500-2000 CTAs over 32 x 2 L2 lines instead of 2 (one line took every CTA's 64 row atomics)
    unsigned* ks = keys + (size_t)(bx % KSLOTS) * KROWS;
    for (int r = threadIdx.x; r < rows; r += NCW * 32)
      if (skey[r] && !(dbg & 2)) atomicMax(ks + r, skey[r]);   // fire-and-forget (RED): k_head_fin reads them after the grid
  }
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int EPI, int WS, int PR>
__global__ void __launch_bounds__(KP8<RB, CB, WK, NCOL, NROW, 4, PR>::NTH, MB)
k_head(const uint8_t* __restrict__ W, const __half* __restrict__ X, int ldx, const __half* __restrict__ S16,
       const float* __restrict__ S32, __half* __restrict__ Y, int ldy, int M, int N, int K, int nvalid, int vstart,
       unsigned* __restrict__ keys, int* __restrict__ counter, float* __restrict__ pairs, int zero) {
  extern __shared__ __align__(16) uint8_t sm[];
  head_tile<RB, CB, WK, NCOL, NROW, EPI, WS, PR>(W, X, ldx, S16, S32, Y, ldy, M, N, K, nvalid, vstart, keys, counter, pairs,
                                         zero, blockIdx.x, sm);
}

// top-1 finalize (one CTA after the head grid, same stream): per row the max of its KSLOTS keys -> (value, global index)
// pair, keys reset to zero. A last-CTA hand-off inside the grid (fence + counter round trip per CTA) cost 30-45 us at
// 40-64 rows: every CTA of a wave reaches it at the same moment, so the SMs idled through the L2 round trips.
__global__ void __launch_bounds__(64) k_head_fin(unsigned* __restrict__ keys, float* __restrict__ pairs, int M, int vstart) {
  const int row = threadIdx.x;
  if (row >= M) return;
  unsigned k = 0u, v[KSLOTS];
#pragma unroll
  for (int s = 0; s < KSLOTS; s++) v[s] = __ldcg(keys + s * KROWS + row);
#pragma unroll
  for (int s = 0; s < KSLOTS; s++) {
    k = v[s] > k ? v[s] : k;
    __stcg(keys + s * KROWS + row, 0u);
  }
  pairs[2 * row] = unord16(k >> 16);
  pairs[2 * row + 1] = (float)(vstart + (int)(0xFFFFu - (k & 0xFFFFu)));
}

struct Args {
  const uint8_t* W; const __half* X; int ldx; const __half* S16; const float* S32; __half* Y; int ldy;
  int M, N, K, nvalid, vstart; unsigned* keys; int* counter; float* pairs; cudaStream_t st;
};

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int EPI, int WS, int PR>
void launch(const Args& a) {
  using P = KP8<RB, CB, WK, NCOL, NROW, 4, PR>;
  auto fn = k_head<RB, CB, WK, NCOL, NROW, MB, EPI, WS, PR>;
  static bool init = false;
  if (!init) {
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    int occ = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, P::NTH, P::SMEM) == cudaSuccess);
    const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * P::SMEM + 98303) / 98304));
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess);
    init = true;
  }
  TORCH_CHECK(a.N % (32 * CB * NCOL) == 0, "fp8_head: N must be a multiple of the CTA's columns");
  fn<<<a.N / (32 * CB * NCOL), P::NTH, P::SMEM, a.st>>>(a.W, a.X, a.ldx, a.S16, a.S32, a.Y, a.ldy, a.M, a.N, a.K, a.nvalid,
                                                    a.vstart, a.keys, a.counter, a.pairs, 0);
  if (EPI == 1) k_head_fin<<<1, 64, 0, a.st>>>(a.keys, a.pairs, a.M, a.vstart);
  static bool dset = false;
  if (!dset) {
    const char* e = getenv("Q27_HEAD_DBG");
    const int d = e ? atoi(e) : 0;
    if (d) TORCH_CHECK(cudaMemcpyToSymbol(g_dbg, &d, sizeof(int)) == cudaSuccess);
    dset = true;
  }
}

using LaunchFn = void (*)(const Args&);
struct Cfg { int r8; const char* name; LaunchFn fn[2][2]; };   // fn[EPI][WS]
#define HCP(R8, NAME, RB, CB, WK, NCOL, NROW, MB, PR) \
  {R8, NAME, {{launch<RB, CB, WK, NCOL, NROW, MB, 0, 0, PR>, launch<RB, CB, WK, NCOL, NROW, MB, 0, 1, PR>}, \
              {launch<RB, CB, WK, NCOL, NROW, MB, 1, 0, PR>, launch<RB, CB, WK, NCOL, NROW, MB, 1, 1, PR>}}}
#define HC(R8, NAME, RB, CB, WK, NCOL, NROW, MB) HCP(R8, NAME, RB, CB, WK, NCOL, NROW, MB, 1)
// N = 62,080 = 1940 tiles of 32: every CB * NCOL in {1, 2, 4, 5} divides it (CTA columns 32 CB NCOL); shapes whose
// parallel-reduction slabs (NCOL x NROW x 8 RB x 32 CB x WK floats) reach 96 KB are not instantiable (+ static smem);
// "pr0" = one reduction slab (serial slice sums, same addition order): less shared memory, more CTAs per SM
const Cfg CFGS[] = {
    HC(8, "h 1x1 wk8 col1", 1, 1, 8, 1, 1, 2),
    HC(8, "h 1x1 wk4 col1", 1, 1, 4, 1, 1, 4),
    HC(16, "h 2x1 wk8 col2", 2, 1, 8, 2, 1, 1),
    HC(16, "h 2x1 wk4 col2", 2, 1, 4, 2, 1, 2),
    HC(32, "h 4x1 wk4 col4", 4, 1, 4, 4, 1, 1),
    HC(32, "h 4x2 wk4 col2", 4, 2, 4, 2, 1, 1),
    HCP(32, "h 4x1 wk4 col4 pr0", 4, 1, 4, 4, 1, 1, 0),
    HCP(32, "h 4x1 wk4 col2 pr0", 4, 1, 4, 2, 1, 2, 0),
    HCP(32, "h 4x1 wk2 col4 pr0", 4, 1, 2, 4, 1, 2, 0),
    HC(48, "h 6x1 wk4 col2", 6, 1, 4, 2, 1, 1),
    HCP(48, "h 6x1 wk4 col2 pr0", 6, 1, 4, 2, 1, 2, 0),
    HCP(48, "h 6x1 wk2 col2 pr0", 6, 1, 2, 2, 1, 2, 0),
    HC(64, "h 8x1 wk4 col2", 8, 1, 4, 2, 1, 1),
    HCP(64, "h 8x1 wk4 col2 pr0", 8, 1, 4, 2, 1, 1, 0),
    HCP(64, "h 8x1 wk2 col4 pr0", 8, 1, 2, 4, 1, 1, 0),
    HCP(64, "h 8x1 wk2 col2 pr0", 8, 1, 2, 2, 1, 2, 0),
    HCP(64, "h 8x1 wk1 col4 pr0", 8, 1, 1, 4, 1, 2, 0),
    HCP(64, "h 4x1 wk4 col2 row2 pr0", 4, 1, 4, 2, 2, 1, 0),
    HCP(64, "h 4x1 wk2 col4 row2 pr0", 4, 1, 2, 4, 2, 1, 0),
    HCP(64, "h 4x2 wk4 col1 row2 pr0", 4, 2, 4, 1, 2, 1, 0),
};
#undef HC
#undef HCP
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);

void check_common(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& s16, const torch::Tensor& s32,
                  int64_t N, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "fp8_head: x must be 16-B aligned");
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous());
  TORCH_CHECK(s16.dtype() == torch::kHalf && s16.is_contiguous() && s16.numel() == N);
  TORCH_CHECK(s32.dtype() == torch::kFloat && s32.is_contiguous() && s32.numel() == N);
  const int64_t K = x.size(1);
  TORCH_CHECK(codes.numel() == N * K && K % 32 == 0);
  TORCH_CHECK(cfg >= 0 && cfg < NCFG && x.size(0) >= 1 && x.size(0) <= CFGS[cfg].r8, "fp8_head: config rows < M");
}
}  // namespace

// out [M][N] (row stride ldo % 2 == 0) fp16 logits of x [M][K] against the QPN8 codes of the head; ws: 1 per-weight
// fp16 scale (TurboMind's weights), 0 f32 scale in the epilogue
void head_logits(torch::Tensor out, torch::Tensor x, torch::Tensor codes, torch::Tensor s16, torch::Tensor s32, int64_t cfg,
                 int64_t ws) {
  const int64_t N = s16.numel();
  check_common(x, codes, s16, s32, N, cfg);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.stride(1) == 1 && out.size(0) == x.size(0)
              && out.size(1) == N && out.stride(0) % 2 == 0);
  const at::cuda::CUDAGuard g(x.device());
  Args a{codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), (int)x.stride(0),
         reinterpret_cast<const __half*>(s16.data_ptr<at::Half>()), s32.data_ptr<float>(),
         reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)out.stride(0), (int)x.size(0), (int)N, (int)x.size(1),
         (int)N, 0, nullptr, nullptr, nullptr, at::cuda::getCurrentCUDAStream()};
  CFGS[cfg].fn[0][ws ? 1 : 0](a);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// top-1 pairs (value f32, vstart + index f32) per row of x at pairs_ptr (device address; e.g. the registered
// all-reduce staging buffer) - columns >= nvalid excluded; keys i32 [32 x 64] zero at rest (counter: unused)
void head_top1(int64_t pairs_ptr, torch::Tensor x, torch::Tensor codes, torch::Tensor s16, torch::Tensor s32,
               torch::Tensor keys, torch::Tensor counter, int64_t vstart, int64_t nvalid, int64_t cfg, int64_t ws) {
  const int64_t N = s16.numel();
  check_common(x, codes, s16, s32, N, cfg);
  TORCH_CHECK(keys.dtype() == torch::kInt && keys.numel() >= KSLOTS * KROWS && x.size(0) <= KROWS
              && counter.dtype() == torch::kInt && counter.numel() >= 1, "fp8_head: top-1 keys [32 * 64] i32, <= 64 rows");
  TORCH_CHECK(pairs_ptr != 0 && pairs_ptr % 8 == 0 && nvalid >= 1 && nvalid <= N && N <= 65535, "fp8_head: top-1 keys");
  const at::cuda::CUDAGuard g(x.device());
  Args a{codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), (int)x.stride(0),
         reinterpret_cast<const __half*>(s16.data_ptr<at::Half>()), s32.data_ptr<float>(), nullptr, 0, (int)x.size(0),
         (int)N, (int)x.size(1), (int)nvalid, (int)vstart,
         reinterpret_cast<unsigned*>(keys.data_ptr<int>()), counter.data_ptr<int>(),
         reinterpret_cast<float*>(pairs_ptr), at::cuda::getCurrentCUDAStream()};
  CFGS[cfg].fn[1][ws ? 1 : 0](a);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> head_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NCFG; i++) v.push_back(std::to_string(CFGS[i].r8) + ":" + CFGS[i].name);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_logits", &head_logits, "FP8 LM head: fp16 logits (<= the config's rows)");
  m.def("head_top1", &head_top1, "FP8 LM head with the top-1 epilogue: (max, index) pairs per row");
  m.def("head_configs", &head_configs, "'rows:name' per configuration");
}
