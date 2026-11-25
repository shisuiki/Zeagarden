// q27fuse/csrc/draft_screen.cu - the MTP draft head on a rotated 1-bit screen (track "draftscreen"; see ../draftscreen.py
// for the build of the codes and the hook).
//
// Per rank, for its vocabulary shard of V rows (V % 32 == 0, H = 5120 = 5 Hadamard blocks of 1024):
//   rotated query   q' = blockdiag_5(H_1024 D_b) x / 32  (D_b = fixed random signs; FP32 butterflies, rounded to FP16)
//   screen score    s_i = f_i * <sign(Q w_i), q'>
//                   k_screen_mma: q' in FP16, sign codes -> +-1 FP16 MMA operands, m8n8k4 sums (FP16 per 512-k group,
//                   FP32 across groups); k_screen_pc (1-2 rows): q' quantised to 4 bits (u = round((q' - lo) / delta)),
//                   <sign, q'> ~ delta (sum u - 2 sum_k b_k u_k) + lo (5120 - 2 popc(b)), sum_k b_k u_k by popcounts of
//                   the code words against the four bit planes of u (the drafthead track's rbq4 arithmetic)
//   per CTA         top-NC (score, id) keys of its rows, key = ordered FP16 score << 16 | (65535 - id) (1Cat's key)
//   k_merge         top-KEEP keys of all CTAs -> exact rerank of those KEEP rows (FP32 dot of the FP16 input with the
//                   effective FP16 head row, rounded to FP16; on the tensor cores) -> best score, lowest id among ties
//                   -> pair (score, start + id), handed to 1Cat's cross-device top-1 unchanged.
// Work split: one CTA per slice of V/8/G 8-row units (per 8-query column tile for the MMA kernel). Code layout per unit:
//   [10 groups of 512 k][32 lanes] uint4; lane l = (quad pair q = (l >> 2) & 3, row j = (l & 3) + 4 (l >> 4)) holds
//   row j's 32 k-steps {16 i + 4 q} of the group; word t bit p (m = p & 15, hi = p >> 4): coordinate
//   k = 512 g + 16 (8 t + m / 2) + 4 q + 2 (m & 1) + hi; bit set = negative. The four quad pairs see disjoint k and are
//   summed by two shuffles at the end of a unit (the work quantum is 8 rows).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int HD = 5120;
constexpr int NG = 10;      // 512-k groups per row
constexpr int QS = 5136;    // smem row stride of the rotated query in halves (10272 B; /4 = 2568 = 8 mod 32: no conflicts)
constexpr int MAXQ = 8;     // query columns per CTA (one MMA n-tile)
constexpr int TMAX = 32;    // rows per CTA <= 32 * TMAX
constexpr int NPW = 160;    // plane words (uint4: 4 bit planes) per query row
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ void mma16(uint32_t (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma32(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// accumulator layout of m8n8k4 f32 (checked by test_draftscreen.py layout): row / column of register r of lane l
__device__ __forceinline__ int f32_row(int l, int r) { return (l & 1) + 2 * ((r >> 1) & 1) + 4 * ((l >> 4) & 1); }
__device__ __forceinline__ int f32_col(int l, int r) { return 2 * ((l >> 1) & 1) + (r & 1) + 4 * (r >> 2); }

__device__ __forceinline__ uint32_t ord16(__half h) {
  // 1Cat's _score_id_keys order for FP16 scores: larger is better, both zeros equal, NaN first
  const uint32_t b = __half_as_ushort(h);
  uint32_t o = (b & 0x8000u) ? (~b & 0xFFFFu) : (b ^ 0x8000u);
  if ((b & 0x7FFFu) == 0) o = 0x8000u;
  if ((b & 0x7C00u) == 0x7C00u && (b & 0x03FFu)) o = 0xFFFFu;
  return o;
}

__device__ __forceinline__ uint32_t word(const uint4& c, int t) { return t == 0 ? c.x : t == 1 ? c.y : t == 2 ? c.z : c.w; }

// (t & 0x80008000) | one, one = 0x3C003C00 held in a register: one LOP3 (the C form gets two, as LOP3 takes one immediate)
__device__ __forceinline__ uint32_t pm1(uint32_t t, uint32_t one) {
  uint32_t d;
  asm("lop3.b32 %0, %1, 0x80008000, %2, 0xEA;\n" : "=r"(d) : "r"(t), "r"(one));
  return d;
}

// one warp: one 1024-block of one query row, element e = 32 lane + i (bits 0-4 in registers, 5-9 across lanes);
// dbits: the block's sign bits (word lane covers e = 32 lane .. 32 lane + 31). Writes FP16 q' and returns the block's
// min / max of the FP16 values (lane 0's values valid after the shuffles).
__device__ __forceinline__ void fwht_task(const __half* __restrict__ xr, const uint32_t* __restrict__ dbits,
                                          __half* __restrict__ out, int lane, float& mn, float& mx) {
  uint4 raw[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) raw[j] = *reinterpret_cast<const uint4*>(xr + 32 * lane + 8 * j);
  const uint32_t d = dbits[lane];
  float v[32];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw[j]);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 f = __half22float2(h2[t]);
      v[8 * j + 2 * t] = f.x;
      v[8 * j + 2 * t + 1] = f.y;
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = ((d >> i) & 1u) ? -v[i] : v[i];
#pragma unroll
  for (int s = 1; s < 32; s <<= 1)
#pragma unroll
    for (int i = 0; i < 32; ++i)
      if (!(i & s)) {
        const float a = v[i], b = v[i | s];
        v[i] = a + b;
        v[i | s] = a - b;
      }
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const bool hi = lane & s;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float p = __shfl_xor_sync(FULL, v[i], s);
      v[i] = hi ? (p - v[i]) : (v[i] + p);
    }
  }
  float lo = INFINITY, hi = -INFINITY;
  uint4 o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half2* h2 = reinterpret_cast<__half2*>(&o[j]);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      h2[t] = __floats2half2_rn(v[8 * j + 2 * t] * (1.f / 32.f), v[8 * j + 2 * t + 1] * (1.f / 32.f));
      const float2 f = __half22float2(h2[t]);
      lo = fminf(lo, fminf(f.x, f.y));
      hi = fmaxf(hi, fmaxf(f.x, f.y));
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(out + 32 * lane + 8 * j) = o[j];
#pragma unroll
  for (int ofs = 16; ofs; ofs >>= 1) {
    lo = fminf(lo, __shfl_xor_sync(FULL, lo, ofs));
    hi = fmaxf(hi, __shfl_xor_sync(FULL, hi, ofs));
  }
  mn = lo;
  mx = hi;
}

template <int U>
__device__ __forceinline__ void preload(const uint4* __restrict__ codes, int u, int lane, uint4 (&cur)[U], uint4 (&nxt)[U]) {
#pragma unroll
  for (int k = 0; k < U; ++k) {
    const uint4* p = codes + (size_t)(u + k) * NG * 32 + lane;
    cur[k] = ld_stream(p);
    nxt[k] = ld_stream(p + 32);
  }
}

// ------------------------------------------------------------------------------------------------ MMA main loop
// U consecutive units starting at u (groups 0 and 1 preloaded): screen scores of their 8 U rows for the CTA's query
// columns -> sk (FP16 order keys)
template <int U>
__device__ __forceinline__ void units_mma(const uint4* __restrict__ codes, int u, int ub, const __half* __restrict__ bq,
                                          const float* __restrict__ rowf, uint16_t* __restrict__ sk, int rw, int nq,
                                          int lane, uint32_t one, uint4 (&cur)[U], uint4 (&nxt)[U]) {
  const uint4* cp[U];
#pragma unroll
  for (int k = 0; k < U; ++k) cp[k] = codes + (size_t)(u + k) * NG * 32 + lane;
  float acc[U][8];
#pragma unroll
  for (int k = 0; k < U; ++k)
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[k][c] = 0.f;
#pragma unroll 1
  for (int g = 0; g < NG; ++g) {
    uint4 nn[U];
    const int gn = min(g + 2, NG - 1);
#pragma unroll
    for (int k = 0; k < U; ++k) nn[k] = ld_stream(cp[k] + gn * 32);
    const __half* bg = bq + 512 * g;
    uint32_t h[U][4];
#pragma unroll
    for (int k = 0; k < U; ++k) h[k][0] = h[k][1] = h[k][2] = h[k][3] = 0u;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const uint2 b = *reinterpret_cast<const uint2*>(bg + 16 * i);   // rows >= nq of qs are zero
      const int m = 2 * (i & 7);
#pragma unroll
      for (int k = 0; k < U; ++k) {
        const uint32_t w = word(cur[k], i >> 3);
        mma16(h[k], pm1(w << (15 - m), one), pm1(w << (14 - m), one), b.x, b.y);
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[k][c]));
        acc[k][2 * c] += f.x;
        acc[k][2 * c + 1] += f.y;
      }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      cur[k] = nxt[k];
      nxt[k] = nn[k];
    }
  }
#pragma unroll
  for (int k = 0; k < U; ++k)
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      acc[k][c] += __shfl_xor_sync(FULL, acc[k][c], 4);
      acc[k][c] += __shfl_xor_sync(FULL, acc[k][c], 8);
    }
  if ((lane & 12) == 0) {
    const int row = (lane & 3) + 4 * (lane >> 4);   // f16 accumulator: the lane's A row, register half c = column c
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int base = 8 * (u + k);
      const float f = rowf[base + row];
#pragma unroll
      for (int c = 0; c < 8; ++c)
        if (c < nq) sk[c * rw + base - 8 * ub + row] = (uint16_t)ord16(__float2half_rn(acc[k][c] * f));
    }
  }
}

// --------------------------------------------------------------------------------------------- popcount main loop
// planes: [nq][NPW] uint4 (bit planes 0..3 of u for the 32 coordinates of code word (g, t, q), index (4 g + t) 4 + q);
// qa: [nq][4] floats lo, delta, sum u
template <int U, int NQ>
__device__ __forceinline__ void units_pc(const uint4* __restrict__ codes, int u, int ub, const uint4* __restrict__ planes,
                                         const float* __restrict__ qa, const float* __restrict__ rowf,
                                         const float* __restrict__ rowpb, uint16_t* __restrict__ sk, int rw, int lane,
                                         uint4 (&cur)[U], uint4 (&nxt)[U]) {
  const int q = (lane >> 2) & 3;
  const uint4* cp[U];
#pragma unroll
  for (int k = 0; k < U; ++k) cp[k] = codes + (size_t)(u + k) * NG * 32 + lane;
  int acc[U][NQ];
#pragma unroll
  for (int k = 0; k < U; ++k)
#pragma unroll
    for (int r = 0; r < NQ; ++r) acc[k][r] = 0;
#pragma unroll 1
  for (int g = 0; g < NG; ++g) {
    uint4 nn[U];
    const int gn = min(g + 2, NG - 1);
#pragma unroll
    for (int k = 0; k < U; ++k) nn[k] = ld_stream(cp[k] + gn * 32);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int r = 0; r < NQ; ++r) {
        const uint4 P = planes[r * NPW + (4 * g + t) * 4 + q];
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const uint32_t w = word(cur[k], t);
          acc[k][r] += __popc(w & P.x) + (__popc(w & P.y) << 1) + (__popc(w & P.z) << 2) + (__popc(w & P.w) << 3);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      cur[k] = nxt[k];
      nxt[k] = nn[k];
    }
  }
#pragma unroll
  for (int k = 0; k < U; ++k)
#pragma unroll
    for (int r = 0; r < NQ; ++r) {
      acc[k][r] += __shfl_xor_sync(FULL, acc[k][r], 4);
      acc[k][r] += __shfl_xor_sync(FULL, acc[k][r], 8);
    }
  if ((lane & 12) == 0) {
    const int row = (lane & 3) + 4 * (lane >> 4);
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int base = 8 * (u + k);
      const float f = rowf[base + row], pb = rowpb[base + row];
#pragma unroll
      for (int r = 0; r < NQ; ++r) {
        const float s = qa[4 * r + 1] * (qa[4 * r + 2] - 2.f * (float)acc[k][r]) + qa[4 * r] * ((float)HD - 2.f * pb);
        sk[r * rw + base - 8 * ub + row] = (uint16_t)ord16(__float2half_rn(s * f));
      }
    }
  }
}

// ------------------------------------------------------- popcount kernel: balanced warp split, leftover units by group
// A CTA's nu units (97 for this model: V = 62080 per rank, 80 CTAs) go nu / W to every warp; the nu % W leftover units
// at the end of the CTA's range (only when nu % W * NG <= W) are split by 512-k group, one (unit, group) task per warp,
// and their per-group integer partial sums are added afterwards (integers: the same keys bitwise). Without it one warp
// of 12 streams 9 units while the others stream 8 (k_screen_pc R = 1: 66.6 -> 63.6 us, R = 2: 76.9 -> 74.0 us; the MMA
// kernel did not gain from the same split: -1.3 / +0.8 / +0.4 / -1.2 us at R = 3 / 4 / 6 / 8, not used there).
template <int NQ>
__device__ __forceinline__ void group_pc(const uint4 c, const uint4* __restrict__ planes, int g, int q, int (&a)[NQ]) {
#pragma unroll
  for (int r = 0; r < NQ; ++r) a[r] = 0;
#pragma unroll
  for (int t = 0; t < 4; ++t)
#pragma unroll
    for (int r = 0; r < NQ; ++r) {
      const uint4 P = planes[r * NPW + (4 * g + t) * 4 + q];
      const uint32_t w = word(c, t);
      a[r] += __popc(w & P.x) + (__popc(w & P.y) << 1) + (__popc(w & P.z) << 2) + (__popc(w & P.w) << 3);
    }
}

template <int NQ>
__device__ __forceinline__ void unit_scores_pc(int (&acc)[NQ], int u, int ub, const float* __restrict__ qa,
                                               const float* __restrict__ rowf, const float* __restrict__ rowpb,
                                               uint16_t* __restrict__ sk, int rw, int lane) {
#pragma unroll
  for (int r = 0; r < NQ; ++r) {
    acc[r] += __shfl_xor_sync(FULL, acc[r], 4);
    acc[r] += __shfl_xor_sync(FULL, acc[r], 8);
  }
  if ((lane & 12) == 0) {
    const int row = (lane & 3) + 4 * (lane >> 4);
    const int base = 8 * u;
    const float f = rowf[base + row], pb = rowpb[base + row];
#pragma unroll
    for (int r = 0; r < NQ; ++r) {
      const float s = qa[4 * r + 1] * (qa[4 * r + 2] - 2.f * (float)acc[r]) + qa[4 * r] * ((float)HD - 2.f * pb);
      sk[r * rw + base - 8 * ub + row] = (uint16_t)ord16(__float2half_rn(s * f));
    }
  }
}

// per-CTA selection: the NC best keys of query column c among the CTA's rows (repeated warp max; keys are unique)
template <int NC>
__device__ __forceinline__ void select_top(const uint16_t* __restrict__ s, int rows, int id0, uint32_t* __restrict__ out,
                                           int lane) {
  uint32_t kk[TMAX];
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    const int idx = lane + 32 * t;
    kk[t] = idx < rows ? (((uint32_t)s[idx] << 16) | (65535u - (uint32_t)(id0 + idx))) : 0u;
  }
#pragma unroll 1
  for (int j = 0; j < NC; ++j) {
    uint32_t m = 0u;
#pragma unroll
    for (int t = 0; t < TMAX; ++t) m = max(m, kk[t]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(FULL, m, o));
#pragma unroll
    for (int t = 0; t < TMAX; ++t) kk[t] = kk[t] == m ? 0u : kk[t];
    if (lane == 0) out[j] = m;
  }
}

// grid (G, ceil(R / 8)), block W warps; smem = 8 * QS halves + nq * rw uint16
template <int W, int NC>
__global__ void __launch_bounds__(W * 32, 1)
k_screen_mma(const __half* __restrict__ x, int ldx, int R, const uint32_t* __restrict__ dbits,
             const uint4* __restrict__ codes, const float* __restrict__ rowf, uint32_t* __restrict__ cand, int V,
             int dbg) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* qs = reinterpret_cast<__half*>(smem);
  const int q0 = blockIdx.y * MAXQ;
  const int nq = min(MAXQ, R - q0);
  const int units = V >> 3;
  const int ub = (int)((long)blockIdx.x * units / gridDim.x), ue = (int)((long)(blockIdx.x + 1) * units / gridDim.x);
  const int rows = (ue - ub) * 8, rw = (rows + 31) & ~31;
  uint16_t* sk = reinterpret_cast<uint16_t*>(smem + (size_t)MAXQ * QS * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nu = ue - ub;
  const int wb = ub + warp * nu / W, we = ub + (warp + 1) * nu / W;
  uint4 c0[2], c1[2];
  if (we - wb >= 2) preload<2>(codes, wb, lane, c0, c1);   // the first loads fly while the query is rotated

  for (int t = warp; t < ((dbg & 2) ? 0 : nq * 5); t += W) {
    const int r = t / 5, b = t - r * 5;
    float mn, mx;
    fwht_task(x + (size_t)(q0 + r) * ldx + b * 1024, dbits + b * 32, qs + r * QS + b * 1024, lane, mn, mx);
  }
  for (int i = threadIdx.x; i < (MAXQ - nq) * (HD / 8); i += W * 32)   // zero B columns >= nq
    *reinterpret_cast<uint4*>(qs + (nq + i / (HD / 8)) * QS + 8 * (i % (HD / 8))) = make_uint4(0u, 0u, 0u, 0u);
  __syncthreads();
  if (!(dbg & 1)) {
    const int q = (lane >> 2) & 3, n = (lane & 3) + 4 * (lane >> 4);
    const __half* bq = qs + n * QS + 4 * q;
    uint32_t one;
    asm volatile("mov.b32 %0, 0x3C003C00;\n" : "=r"(one));
    int u = wb;
    for (; u + 2 <= we; u += 2) {
      if (u != wb) preload<2>(codes, u, lane, c0, c1);
      units_mma<2>(codes, u, ub, bq, rowf, sk, rw, nq, lane, one, c0, c1);
    }
    if (u < we) {
      uint4 d0[1], d1[1];
      preload<1>(codes, u, lane, d0, d1);
      units_mma<1>(codes, u, ub, bq, rowf, sk, rw, nq, lane, one, d0, d1);
    }
  }
  __syncthreads();
  for (int c = warp; c < nq; c += W)
    select_top<NC>(sk + c * rw, rows, 8 * ub, cand + ((size_t)(q0 + c) * gridDim.x + blockIdx.x) * NC, lane);
}

// grid (G), block W warps, NQ (1 or 2) query rows; smem = NQ * QS halves + NQ * NPW uint4 + NQ * rw uint16 + scalars
template <int W, int NC, int NQ>
__global__ void __launch_bounds__(W * 32, 1)
k_screen_pc(const __half* __restrict__ x, int ldx, const uint32_t* __restrict__ dbits, const uint4* __restrict__ codes,
            const float* __restrict__ rowf, const float* __restrict__ rowpb, uint32_t* __restrict__ cand, int V, int dbg) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* qs = reinterpret_cast<__half*>(smem);
  uint4* planes = reinterpret_cast<uint4*>(smem + (size_t)NQ * QS * 2);
  __shared__ float mm[NQ][5][2];
  __shared__ float qa[NQ][4];
  __shared__ int su[NQ];
  const int units = V >> 3;
  const int ub = (int)((long)blockIdx.x * units / gridDim.x), ue = (int)((long)(blockIdx.x + 1) * units / gridDim.x);
  const int rows = (ue - ub) * 8, rw = (rows + 31) & ~31;
  uint16_t* sk = reinterpret_cast<uint16_t*>(planes + NQ * NPW);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nu = ue - ub;
  const bool bal = !(dbg & 32) && (nu % W) * NG <= W;   // dbg & 32: the plain split (A/B in benches)
  const int nb = nu / W;
  const int wb = bal ? ub + warp * nb : ub + warp * nu / W, we = bal ? wb + nb : ub + (warp + 1) * nu / W;
  uint4 c0[2], c1[2];
  if (we - wb >= 2) preload<2>(codes, wb, lane, c0, c1);
  if (threadIdx.x < NQ) su[threadIdx.x] = 0;

  for (int t = warp; t < ((dbg & 2) ? 0 : NQ * 5); t += W) {
    const int r = t / 5, b = t - r * 5;
    float mn, mx;
    fwht_task(x + (size_t)r * ldx + b * 1024, dbits + b * 32, qs + r * QS + b * 1024, lane, mn, mx);
    if (lane == 0) {
      mm[r][b][0] = mn;
      mm[r][b][1] = mx;
    }
  }
  __syncthreads();
  if (threadIdx.x < NQ) {
    const int r = threadIdx.x;
    float lo = mm[r][0][0], hi = mm[r][0][1];
#pragma unroll
    for (int b = 1; b < 5; ++b) {
      lo = fminf(lo, mm[r][b][0]);
      hi = fmaxf(hi, mm[r][b][1]);
    }
    qa[r][0] = lo;
    qa[r][1] = fmaxf(hi - lo, 1e-20f) / 15.f;
  }
  __syncthreads();
  // bit planes: word (g, t, q) of row r, lane p = bit p: coordinate 512 g + 16 (8 t + m / 2) + 4 q + 2 (m & 1) + hi
  {
    const int m = lane & 15, hb = lane >> 4;
    for (int w0 = warp * 4; w0 < ((dbg & 2) ? 0 : NQ * NPW); w0 += W * 4) {
      uint4 pw[4];
      int sum[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int wi = w0 + e;
        const int r = wi / NPW, ww = wi - r * NPW;
        const int g = ww >> 4, t = (ww >> 2) & 3, q = ww & 3;
        const int k = 512 * g + 16 * (8 * t + (m >> 1)) + 4 * q + 2 * (m & 1) + hb;
        const float v = wi < NQ * NPW ? __half2float(qs[r * QS + k]) : 0.f;
        const int uu = wi < NQ * NPW ? min(15, max(0, __float2int_rn((v - qa[r][0]) / qa[r][1]))) : 0;
        pw[e].x = __ballot_sync(FULL, uu & 1);
        pw[e].y = __ballot_sync(FULL, uu & 2);
        pw[e].z = __ballot_sync(FULL, uu & 4);
        pw[e].w = __ballot_sync(FULL, uu & 8);
        sum[e] = __popc(pw[e].x) + 2 * __popc(pw[e].y) + 4 * __popc(pw[e].z) + 8 * __popc(pw[e].w);
      }
      if (lane == 0) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int wi = w0 + e;
          if (wi < NQ * NPW) {
            planes[wi] = pw[e];
            atomicAdd(&su[wi / NPW], sum[e]);
          }
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < NQ) qa[threadIdx.x][2] = (float)su[threadIdx.x];
  __syncthreads();
  if (!(dbg & 1)) {
    int u = wb;
    for (; u + 2 <= we; u += 2) {
      if (u != wb) preload<2>(codes, u, lane, c0, c1);
      units_pc<2, NQ>(codes, u, ub, planes, &qa[0][0], rowf, rowpb, sk, rw, lane, c0, c1);
    }
    if (u < we) {
      uint4 d0[1], d1[1];
      preload<1>(codes, u, lane, d0, d1);
      units_pc<1, NQ>(codes, u, ub, planes, &qa[0][0], rowf, rowpb, sk, rw, lane, d0, d1);
    }
  }
  if (bal) {
    const int nl = nu - W * nb, ul = ub + W * nb;
    int* part = reinterpret_cast<int*>(smem);          // [task][lane][NQ] over the query rows (not read after the planes)
    if (warp < nl * NG && !(dbg & 1)) {
      const int g = warp % NG;
      const uint4 c = ld_stream(codes + ((size_t)(ul + warp / NG) * NG + g) * 32 + lane);
      int a[NQ];
      group_pc<NQ>(c, planes, g, (lane >> 2) & 3, a);
#pragma unroll
      for (int r = 0; r < NQ; ++r) part[(warp * 32 + lane) * NQ + r] = a[r];
    }
    __syncthreads();
    if (warp < nl && !(dbg & 1)) {
      int acc[NQ];
#pragma unroll
      for (int r = 0; r < NQ; ++r) acc[r] = 0;
#pragma unroll 1
      for (int g = 0; g < NG; ++g)
#pragma unroll
        for (int r = 0; r < NQ; ++r) acc[r] += part[((warp * NG + g) * 32 + lane) * NQ + r];
      unit_scores_pc<NQ>(acc, ul + warp, ub, &qa[0][0], rowf, rowpb, sk, rw, lane);
    }
  }
  __syncthreads();
  for (int c = warp; c < NQ; c += W)
    select_top<NC>(sk + c * rw, rows, 8 * ub, cand + ((size_t)c * gridDim.x + blockIdx.x) * NC, lane);
}

// --------------------------------------------------------------------------------------- 9+ rows: 16 queries per CTA
// k_rotate: q' of every query row into global memory (one warp per (row, 1024-block)), FP16 natural order.
__global__ void __launch_bounds__(32) k_rotate(const __half* __restrict__ x, int ldx, const uint32_t* __restrict__ dbits,
                                               __half* __restrict__ qg) {
  const int r = blockIdx.x / 5, b = blockIdx.x % 5;
  float mn, mx;
  fwht_task(x + (size_t)r * ldx + b * 1024, dbits + b * 32, qg + (size_t)r * HD + b * 1024, threadIdx.x, mn, mx);
}

// k_screen_mma16: grid (G, ceil(R / 16)), W warps. Loop over the 5 Hadamard blocks: q' of the block for the CTA's 16
// queries in smem (2 MMA n-tiles), every warp runs its units over the block's two 512-k groups (one A build feeds both
// n-tiles), the per-unit FP32 sums live in smem between blocks. Codes are read once for 16 queries.
constexpr int QB = 1040;      // smem row stride of one block of q' (halves): 2080 B, /4 = 520 = 8 mod 32
constexpr int MAXU = 100;     // units per CTA
template <int W, int NC>
__global__ void __launch_bounds__(W * 32, 1)
k_screen_mma16(const __half* __restrict__ qg, int R, const uint4* __restrict__ codes, const float* __restrict__ rowf,
               uint32_t* __restrict__ cand, int V) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* qb = reinterpret_cast<__half*>(smem);                              // [16][QB]
  float* accs = reinterpret_cast<float*>(smem + 16 * QB * 2);                 // [MAXU][8 rows][16 cols]
  uint16_t* sk = reinterpret_cast<uint16_t*>(smem);                           // [16][rw], aliases qb after the loop
  const int q0 = blockIdx.y * 16;
  const int nq = min(16, R - q0);
  const int units = V >> 3;
  const int ub = (int)((long)blockIdx.x * units / gridDim.x), ue = (int)((long)(blockIdx.x + 1) * units / gridDim.x);
  const int rows = (ue - ub) * 8, rw = (rows + 31) & ~31;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nu = ue - ub;
  const int wb = ub + warp * nu / W, we = ub + (warp + 1) * nu / W;
  const int q = (lane >> 2) & 3, n = (lane & 3) + 4 * (lane >> 4);
  uint32_t one;
  asm volatile("mov.b32 %0, 0x3C003C00;\n" : "=r"(one));
  for (int i = threadIdx.x; i < nu * 128; i += W * 32) accs[i] = 0.f;
#pragma unroll 1
  for (int b = 0; b < 5; ++b) {
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 128; i += W * 32) {
      const int r = i >> 7, c = (i & 127) * 8;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (r < nq) v = *reinterpret_cast<const uint4*>(qg + (size_t)(q0 + r) * HD + b * 1024 + c);
      *reinterpret_cast<uint4*>(qb + r * QB + c) = v;
    }
    __syncthreads();
    const __half* bq0 = qb + n * QB + 4 * q;
    const __half* bq1 = qb + (8 + n) * QB + 4 * q;
    uint4 c[2][2], nx[2][2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int gg = 0; gg < 2; ++gg)
        c[k][gg] = ld_stream(codes + ((size_t)min(wb + k, we - 1) * NG + 2 * b + gg) * 32 + lane);
#pragma unroll 1
    for (int u = wb; u < we; u += 2) {
      const bool two = u + 1 < we;
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int gg = 0; gg < 2; ++gg)
          nx[k][gg] = ld_stream(codes + ((size_t)min(u + 2 + k, we - 1) * NG + 2 * b + gg) * 32 + lane);
      float acc[2][16];
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[k][e] = 0.f;
#pragma unroll
      for (int gg = 0; gg < 2; ++gg) {
        uint32_t h[2][2][4];
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
          for (int t = 0; t < 2; ++t) h[k][t][0] = h[k][t][1] = h[k][t][2] = h[k][t][3] = 0u;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const uint2 b0 = *reinterpret_cast<const uint2*>(bq0 + 512 * gg + 16 * i);
          const uint2 b1 = *reinterpret_cast<const uint2*>(bq1 + 512 * gg + 16 * i);
          const int m = 2 * (i & 7);
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const uint32_t w = word(c[k][gg], i >> 3);
            const uint32_t a0 = pm1(w << (15 - m), one), a1 = pm1(w << (14 - m), one);
            mma16(h[k][0], a0, a1, b0.x, b0.y);
            mma16(h[k][1], a0, a1, b1.x, b1.y);
          }
        }
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
          for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[k][t][e]));
              acc[k][8 * t + 2 * e] += f.x;
              acc[k][8 * t + 2 * e + 1] += f.y;
            }
      }
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          acc[k][e] += __shfl_xor_sync(FULL, acc[k][e], 4);
          acc[k][e] += __shfl_xor_sync(FULL, acc[k][e], 8);
        }
      if ((lane & 12) == 0) {
        const int row = (lane & 3) + 4 * (lane >> 4);
#pragma unroll
        for (int k = 0; k < 2; ++k)
          if (k == 0 || two) {
            float* a = accs + ((u + k - ub) * 8 + row) * 16;
#pragma unroll
            for (int e = 0; e < 16; ++e) a[e] += acc[k][e];
          }
      }
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int gg = 0; gg < 2; ++gg) c[k][gg] = nx[k][gg];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nu * 128; i += W * 32) {
    const int col = i & 15, rr = i >> 4;            // rr = local row
    if (col < nq) sk[col * rw + rr] = (uint16_t)ord16(__float2half_rn(accs[i] * rowf[8 * ub + rr]));
  }
  __syncthreads();
  for (int c = warp; c < nq; c += W)
    select_top<NC>(sk + c * rw, rows, 8 * ub, cand + ((size_t)(q0 + c) * gridDim.x + blockIdx.x) * NC, lane);
}

// ---------------------------------------------------------------------------------------------------------- merge
// Selection (every merge CTA of a row): the KEEP best keys of the G sorted per-CTA lists.
//   threshold = KEEP-th largest list head (KEEP heads are >= it, so the KEEP best keys are among the keys >= it);
//   survivors = keys >= threshold; rank by counting among survivors (keys are unique: the id is in the low half).
template <int NC, int KEEP, int NT>
__device__ __forceinline__ void select_keep(const uint32_t* __restrict__ cr, int G, uint32_t* keys, uint32_t* surv,
                                            int* top, int& ns_s, uint32_t& thr_s) {
  const int tid = threadIdx.x, n = G * NC;
  for (int i = tid; i < n; i += NT) keys[i] = cr[i];
  if (tid == 0) {
    ns_s = 0;
    thr_s = 0u;
  }
  if (tid < KEEP) top[tid] = 0;
  __syncthreads();
  if (G >= KEEP && tid < G) {
    const uint32_t h = keys[tid * NC];
    int cnt = 0;
    for (int c = 0; c < G; ++c) cnt += keys[c * NC] > h;
    if (cnt == KEEP - 1) thr_s = h;
  }
  __syncthreads();
  const uint32_t thr = thr_s;
  for (int i = tid; i < n; i += NT) {
    const uint32_t k = keys[i];
    if (k >= thr && k != 0u) surv[atomicAdd(&ns_s, 1)] = k;
  }
  __syncthreads();
  const int ns = ns_s;
  for (int i = tid; i < ns; i += NT) {
    const uint32_t k = surv[i];
    int cnt = 0;
    for (int j = 0; j < ns; ++j) cnt += surv[j] > k;
    if (cnt < KEEP) top[cnt] = 65535 - (int)(k & 0xFFFFu);
  }
  __syncthreads();
}

constexpr int MAXN = 2048;

// ids only (for 1Cat's Triton rerank): one CTA of 256 per row
template <int NC, int KEEP>
__global__ void __launch_bounds__(256)
k_merge_ids(const uint32_t* __restrict__ cand, int G, int* __restrict__ ids_out) {
  __shared__ uint32_t keys[MAXN], surv[MAXN];
  __shared__ int ns_s;
  __shared__ uint32_t thr_s;
  __shared__ int top[KEEP];
  select_keep<NC, KEEP, 256>(cand + (size_t)blockIdx.x * G * NC, G, keys, surv, top, ns_s, thr_s);
  if (threadIdx.x < KEEP) ids_out[blockIdx.x * KEEP + threadIdx.x] = top[threadIdx.x];
}

__device__ __forceinline__ float unord16(uint32_t o) {
  if (o == 0xFFFFu) return __int_as_float(0x7fc00000);
  const uint32_t b = o >= 0x8000u ? (o ^ 0x8000u) : (~o & 0xFFFFu);
  return __half2float(__ushort_as_half((unsigned short)b));
}

// Rerank: grid (R, KEEP / 8), 256 threads: CTA y reranks candidates 8 y .. 8 y + 7 (one MMA row tile), warp w = k-part w
// (640 k), each quad pair every 4th 8-wide k chunk, B = the input row in column 0. FP16 rows (dense) or FP8 E4M3 rows x
// FP32 row scale (decoded to exactly fp16(code * scale): the E4M3 bits placed in an FP16 (value * 2^-8), one FP16
// multiply by scale * 256). The 8 parts are added in a fixed order and rounded to FP16; the CTA's best goes into a
// 64-bit atomic max per row (ordered FP16 score, then lowest id), and the row's last CTA writes the pair and resets the
// workspace (ws: [rows] u64 keys, [rows] i32 counters; zero at rest).
// [head] RR 3: the FP8 rows read out of the unified head's QPN8 layout (fp8_head.cu / fp8_lin.cu header): row id is
// column id & 31 of 32-column tile id >> 5; its 16 k of k-group g of 32-k chunk c are the 16 B of lane p (col(p) =
// id & 31) at ((tile * 160 + c) * 1024 + 512 g + 16 p), bytes k 0 2 4 6 1 3 5 7 per 8 (same values, same arithmetic as
// RR 2: the scores are bitwise RR 2's)
template <int NC, int KEEP, int RR>  // RR: 1 FP16 rows, 2 FP8 rows, 3 FP8 rows in the QPN8 layout [head]
__global__ void __launch_bounds__(256)
k_merge(const uint32_t* __restrict__ cand, int G, const __half* __restrict__ x, int ldx, const void* __restrict__ wrows,
        const float* __restrict__ wscale, int start, float* __restrict__ pairs, unsigned long long* __restrict__ wsk,
        int* __restrict__ wsc) {
  static_assert(KEEP == 32 || KEEP == 16, "KEEP");
  __shared__ uint32_t keys[MAXN], surv[MAXN];
  __shared__ int ns_s;
  __shared__ uint32_t thr_s;
  __shared__ int top[KEEP];
  __shared__ float part[8][8];
  const int r = blockIdx.x, y = blockIdx.y;
  select_keep<NC, KEEP, 256>(cand + (size_t)r * G * NC, G, keys, surv, top, ns_s, thr_s);
  const int tid = threadIdx.x, lane = tid & 31, kp = tid >> 5;
  const int q = (lane >> 2) & 3, j = (lane & 3) + 4 * (lane >> 4);
  const int id = top[8 * y + j];
  const __half* xr = x + (size_t)r * ldx;
  float acc[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) acc[c] = 0.f;
  const bool bcol = (lane & 3) == 0 && (lane >> 4) == 0;  // B column 0 holders
  if (RR == 1) {
    const __half* wr = reinterpret_cast<const __half*>(wrows) + (size_t)id * HD;
    uint4 a[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) a[i] = *reinterpret_cast<const uint4*>(wr + kp * 640 + 32 * i + 8 * q);
#pragma unroll
    for (int i = 0; i < 20; ++i) {
      const int k = kp * 640 + 32 * i + 8 * q;
      uint4 b = make_uint4(0u, 0u, 0u, 0u);
      if (bcol) b = *reinterpret_cast<const uint4*>(xr + k);
      mma32(acc, a[i].x, a[i].y, b.x, b.y);
      mma32(acc, a[i].z, a[i].w, b.z, b.w);
    }
  } else if (RR == 3) {                                   // [head] QPN8 layout
    const int cc = id & 31, p = 4 * (cc >> 3) + (cc & 3) + 16 * ((cc >> 2) & 1);
    const uint8_t* wt = reinterpret_cast<const uint8_t*>(wrows) + (size_t)(id >> 5) * (HD / 32) * 1024 + 16 * p
                        + 512 * (q >> 1) + 8 * (q & 1);
    const __half s256 = __float2half_rn(wscale[id] * 256.f);
    const __half2 s2 = __halves2half2(s256, s256);
    uint2 c8[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) c8[i] = *reinterpret_cast<const uint2*>(wt + (size_t)(kp * 20 + i) * 1024);
#pragma unroll
    for (int i = 0; i < 20; ++i) {
      const int k = kp * 640 + 32 * i + 8 * q;
      uint32_t a[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        // pair (k + 2t, k + 2t + 1) = bytes t (x) and t (y) of the k 0 2 4 6 | 1 3 5 7 order
        const uint32_t tw = __byte_perm(c8[i].x, c8[i].y, (uint32_t)(t | ((4 + t) << 4)));
        const uint32_t pr = __byte_perm(tw, 0u, 0x7170);   // bytes 0, 1 -> low bytes of the halves
        const uint32_t hb = ((pr & 0x00800080u) << 8) | ((pr & 0x007F007Fu) << 7);
        const __half2 hv = __hmul2(*reinterpret_cast<const __half2*>(&hb), s2);
        a[t] = *reinterpret_cast<const uint32_t*>(&hv);
      }
      uint4 b = make_uint4(0u, 0u, 0u, 0u);
      if (bcol) b = *reinterpret_cast<const uint4*>(xr + k);
      mma32(acc, a[0], a[1], b.x, b.y);
      mma32(acc, a[2], a[3], b.z, b.w);
    }
  } else {
    const uint8_t* wr = reinterpret_cast<const uint8_t*>(wrows) + (size_t)id * HD;
    const __half s256 = __float2half_rn(wscale[id] * 256.f);
    const __half2 s2 = __halves2half2(s256, s256);
    uint2 c8[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) c8[i] = *reinterpret_cast<const uint2*>(wr + kp * 640 + 32 * i + 8 * q);
#pragma unroll
    for (int i = 0; i < 20; ++i) {
      const int k = kp * 640 + 32 * i + 8 * q;
      uint32_t a[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint32_t w = t < 2 ? c8[i].x : c8[i].y;
        const uint32_t pr = __byte_perm(w, 0u, (t & 1) ? 0x7372 : 0x7170);   // bytes 2t, 2t+1 -> low bytes of halves
        const uint32_t hb = ((pr & 0x00800080u) << 8) | ((pr & 0x007F007Fu) << 7);
        const __half2 hv = __hmul2(*reinterpret_cast<const __half2*>(&hb), s2);
        a[t] = *reinterpret_cast<const uint32_t*>(&hv);
      }
      uint4 b = make_uint4(0u, 0u, 0u, 0u);
      if (bcol) b = *reinterpret_cast<const uint4*>(xr + k);
      mma32(acc, a[0], a[1], b.x, b.y);
      mma32(acc, a[2], a[3], b.z, b.w);
    }
  }
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    acc[c] += __shfl_xor_sync(FULL, acc[c], 4);
    acc[c] += __shfl_xor_sync(FULL, acc[c], 8);
  }
  // column 0 of the f32 accumulator: lanes with (lane >> 1) & 1 == 0, registers 0 and 2 (rows f32_row(lane, 0 / 2))
  if ((lane & 12) == 0 && ((lane >> 1) & 1) == 0) {
    part[kp][f32_row(lane, 0)] = acc[0];
    part[kp][f32_row(lane, 2)] = acc[2];
  }
  __syncthreads();
  if (kp == 0) {
    unsigned long long key = 0ull;
    if (lane < 8) {
      float sm = part[0][lane];
#pragma unroll
      for (int p = 1; p < 8; ++p) sm += part[p][lane];
      const uint32_t o = ord16(__float2half_rn(sm));
      key = ((unsigned long long)o << 32) | (unsigned long long)(0xFFFFFFFFu - (uint32_t)top[8 * y + lane]);
    }
#pragma unroll
    for (int ofs = 4; ofs; ofs >>= 1) {
      const unsigned long long o2 = __shfl_xor_sync(FULL, key, ofs);
      key = o2 > key ? o2 : key;
    }
    if (lane == 0) {
      atomicMax(wsk + r, key);
      __threadfence();
      if (atomicAdd(wsc + r, 1) == (int)gridDim.y - 1) {
        __threadfence();
        const unsigned long long best = atomicExch(wsk + r, 0ull);
        wsc[r] = 0;
        pairs[2 * r] = unord16((uint32_t)(best >> 32));
        pairs[2 * r + 1] = (float)((int)(0xFFFFFFFFu - (uint32_t)(best & 0xFFFFFFFFull)) + start);
      }
    }
  }
}

// layout probe for the accumulator fragments: out [2 types][2 runs][32 lanes][8 regs] (run 0: A[r][0] = r + 1 + 10 q,
// B[0][c] = 1 -> row; run 1: A[r][0] = 1, B[0][c] = c + 1 + 10 q -> column)
__global__ void k_layout(float* out) {
  const int l = threadIdx.x, q = (l >> 2) & 3, j = (l & 3) + 4 * (l >> 4);
  for (int run = 0; run < 2; ++run) {
    const __half a0 = __float2half(run == 0 ? (float)(j + 1 + 10 * q) : 1.f);
    const __half b0 = __float2half(run == 0 ? 1.f : (float)(j + 1 + 10 * q));
    const __half z = __float2half(0.f);
    __half2 A0 = __halves2half2(a0, z), A1 = __halves2half2(z, z), B0 = __halves2half2(b0, z), B1 = A1;
    uint32_t ua0 = *reinterpret_cast<uint32_t*>(&A0), ua1 = *reinterpret_cast<uint32_t*>(&A1);
    uint32_t ub0 = *reinterpret_cast<uint32_t*>(&B0), ub1 = *reinterpret_cast<uint32_t*>(&B1);
    float c32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    mma32(c32, ua0, ua1, ub0, ub1);
    uint32_t c16[4] = {0, 0, 0, 0};
    mma16(c16, ua0, ua1, ub0, ub1);
    for (int r = 0; r < 8; ++r) {
      out[((0 * 2 + run) * 32 + l) * 8 + r] = c32[r];
      out[((1 * 2 + run) * 32 + l) * 8 + r] = __half2float(reinterpret_cast<__half*>(c16)[r]);
    }
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------- host
template <typename K>
static void big_smem(K kern) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kern));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 - (int)a.sharedSizeBytes));
}

// x [R, 5120] fp16 (unit column stride, row stride % 8 == 0), dbits int32 [160], codes int32 [V/8 * 10 * 32 * 4],
// rowf / rowpb [V] f32 -> cand int32 [R, G, nc]. pc: rows R <= 2 by the popcount kernel.
// qg: FP16 workspace [>= R * 5120] for 9+ rows (the 16-query kernel); empty -> 8-query tiles
void screen(torch::Tensor x, torch::Tensor dbits, torch::Tensor codes, torch::Tensor rowf, torch::Tensor rowpb,
            torch::Tensor cand, int64_t G, int64_t nc, int64_t warps, bool pc, int64_t dbg, torch::Tensor qg) {
  TORCH_CHECK(x.dtype() == torch::kHalf && x.size(1) == HD && x.stride(1) == 1 && x.stride(0) % 8 == 0, "x");
  const int R = x.size(0), V = rowf.size(0);
  TORCH_CHECK(V % 32 == 0 && codes.numel() == (int64_t)V / 8 * NG * 32 * 4, "codes/rowf");
  TORCH_CHECK(cand.numel() >= R * G * nc, "cand");
  const int rows_max = ((V / 8 + G - 1) / G) * 8, rw = (rows_max + 31) & ~31;
  TORCH_CHECK(rows_max <= 32 * TMAX, "draft_screen: too many rows per CTA");
  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  const uint32_t* dp = reinterpret_cast<const uint32_t*>(dbits.data_ptr());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr());
  uint32_t* kp = reinterpret_cast<uint32_t*>(cand.data_ptr());
  if (pc && R <= 4) {
#define PC_CASE(WW, NN, QQ)                                                                                        \
  if (warps == WW && nc == NN && R == QQ) {                                                                       \
    static bool once = false;                                                                                     \
    if (!once) { big_smem(k_screen_pc<WW, NN, QQ>); once = true; }                                                \
    const int sm = QQ * QS * 2 + QQ * NPW * 16 + QQ * rw * 2;                                                     \
    k_screen_pc<WW, NN, QQ><<<G, WW * 32, sm, st>>>(xp, x.stride(0), dp, cp, rowf.data_ptr<float>(),              \
                                                     rowpb.data_ptr<float>(), kp, V, (int)dbg);                   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                               \
    return;                                                                                                        \
  }
    PC_CASE(12, 8, 1) PC_CASE(12, 8, 2) PC_CASE(12, 8, 3) PC_CASE(12, 8, 4)
    PC_CASE(12, 16, 1) PC_CASE(12, 16, 2) PC_CASE(12, 16, 3) PC_CASE(12, 16, 4)
    PC_CASE(16, 8, 1) PC_CASE(16, 8, 2) PC_CASE(16, 8, 3) PC_CASE(16, 8, 4)
    TORCH_CHECK(false, "draft_screen: unsupported popcount warps/nc");
  }
  // 16-query kernel when it takes fewer passes over the codes at its cost (measured per pass: 8-query tile ~100 us,
  // 16-query tile ~170 us at 1 to 2 tiles): 9-16 and 25-32 rows; 17-24 and 33-40 rows stay on 8-query tiles
  const bool use16 = R > MAXQ && (R + 15) / 16 * 17 < (R + 7) / 8 * 10;
  if (use16 && qg.numel() >= (int64_t)R * HD) {
    TORCH_CHECK(V / 8 <= (int64_t)G * MAXU, "draft_screen: too many units per CTA");
    __half* qp = reinterpret_cast<__half*>(qg.data_ptr());
    k_rotate<<<R * 5, 32, 0, st>>>(xp, x.stride(0), dp, qp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int sm16 = 16 * QB * 2 + MAXU * 128 * 4;
    TORCH_CHECK(16 * rw * 2 <= 16 * QB * 2, "draft_screen: score keys do not fit");
    dim3 g16(G, (R + 15) / 16);
#define M16_CASE(WW, NN)                                                                                           \
    if (warps == WW && nc == NN) {                                                                                \
      static bool once = false;                                                                                   \
      if (!once) { big_smem(k_screen_mma16<WW, NN>); once = true; }                                               \
      k_screen_mma16<WW, NN><<<g16, WW * 32, sm16, st>>>(qp, R, cp, rowf.data_ptr<float>(), kp, V);               \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
      return;                                                                                                      \
    }
    M16_CASE(12, 8) M16_CASE(12, 16) M16_CASE(16, 8) M16_CASE(16, 16)
    TORCH_CHECK(false, "draft_screen: unsupported warps/nc");
  }
  const int nq = std::min(R, MAXQ);
  const int sm = MAXQ * QS * 2 + nq * rw * 2;
  dim3 grid(G, (R + MAXQ - 1) / MAXQ);
#define MMA_CASE(WW, NN)                                                                                           \
  if (warps == WW && nc == NN) {                                                                                  \
    static bool once = false;                                                                                     \
    if (!once) { big_smem(k_screen_mma<WW, NN>); once = true; }                                                    \
    k_screen_mma<WW, NN><<<grid, WW * 32, sm, st>>>(xp, x.stride(0), R, dp, cp, rowf.data_ptr<float>(), kp, V,    \
                                                    (int)dbg);                                                     \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                               \
    return;                                                                                                        \
  }
  MMA_CASE(12, 8) MMA_CASE(12, 16) MMA_CASE(16, 8) MMA_CASE(16, 16)
  TORCH_CHECK(false, "draft_screen: unsupported warps/nc");
}

// cand [R, G, nc] -> rr 0: ids [R, keep] int32; rr 1/2/3: pairs [R, 2] f32 (rerank from FP16 rows / FP8 rows + scales /
// [head] FP8 rows in the unified head's QPN8 layout + scales;
// ws int64 [>= 2 R] zero at rest: keys then counters)
void merge(torch::Tensor cand, int64_t G, int64_t nc, int64_t keep, torch::Tensor x, int64_t rr, torch::Tensor wrows,
           torch::Tensor wscale, int64_t start, torch::Tensor pairs, torch::Tensor ids, torch::Tensor ws) {
  const int R = x.size(0);
  TORCH_CHECK(G * nc <= MAXN && G >= keep, "merge: candidate count");
  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  const uint32_t* cp = reinterpret_cast<const uint32_t*>(cand.data_ptr());
  if (rr == 0) {
#define DI_CASE(NN, KK) \
    if (nc == NN && keep == KK) { k_merge_ids<NN, KK><<<R, 256, 0, st>>>(cp, (int)G, ids.data_ptr<int>()); \
      C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    DI_CASE(8, 16) DI_CASE(8, 32) DI_CASE(16, 16) DI_CASE(16, 32)
    TORCH_CHECK(false, "draft_screen merge: unsupported nc/keep");
  }
  TORCH_CHECK(ws.numel() >= 2 * R && ws.dtype() == torch::kLong, "merge: workspace");
  unsigned long long* wk = reinterpret_cast<unsigned long long*>(ws.data_ptr());
  int* wc = reinterpret_cast<int*>(wk + R);
  const void* wp = wrows.data_ptr();
  const float* sp = rr >= 2 ? wscale.data_ptr<float>() : nullptr;   // [head] rr 3: QPN8 layout
  float* pp = pairs.data_ptr<float>();
  dim3 grid(R, keep / 8);
#define DM_CASE(NN, KK, RRR) \
  if (nc == NN && keep == KK && rr == RRR) { \
    k_merge<NN, KK, RRR><<<grid, 256, 0, st>>>(cp, (int)G, xp, x.stride(0), wp, sp, (int)start, pp, wk, wc); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
#define DM_CASES(NN, KK) DM_CASE(NN, KK, 1) DM_CASE(NN, KK, 2) DM_CASE(NN, KK, 3)
  DM_CASES(8, 16) DM_CASES(8, 32) DM_CASES(16, 16) DM_CASES(16, 32)
  TORCH_CHECK(false, "draft_screen merge: unsupported nc/keep/rr");
}

torch::Tensor layout_probe() {
  auto out = torch::zeros({2, 2, 32, 8}, torch::dtype(torch::kFloat).device(torch::kCUDA));
  k_layout<<<1, 32>>>(out.data_ptr<float>());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("screen", &screen);
  m.def("merge", &merge);
  m.def("layout_probe", &layout_probe);
}
