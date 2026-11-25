// attn_mq.cu - split-KV decode attention with Q query rows per request (track attnv): MTP verify rows (Q = K + 1),
// the drafter's first pass, and single-query batches (Q = 1) incl. B = 1. Same contract as 1Cat's
// sm70_grouped_long_fwd (grouped_e4m3_2cta_partial + parallel_combine): q [B Q][6 Hkv][256] fp16 (rows of a request
// contiguous), paged E4M3 KV [blocks][page][Hkv][256] uint8, per-REQUEST block table [B][W], per-ROW lengths [B Q]
// (0 = padding row; row i of a request sees its own length, so the causal mask among the new tokens is a per-row
// length), FP32 QK and PV accumulation, P in FP16, exact E4M3 decode, FP16 output.
//
// Work unit: a set of NW warps on a 32-token tile of one (request, KV head). The R = 6 Q query rows (padded to RG
// groups of 8) all use the same pass over the tile; warp w owns head dims [D w, D w + D), D = 256 / NW, for both
// products, so the O accumulators (R x 256 FP32) are spread over the set (16 RG regs/thread at NW 4, 32 RG at NW 2).
//   QK (mma m8n8k4 row.col): B = K^T, one column per thread = that thread's token (its D K bytes loaded straight from
//      the paged cache, no staging); A = Q rows from shared memory; one K decode feeds all RG row groups.
//   exchange: each warp's partial scores (its D dims) -> shared X; barrier; the softmax of row group g is done by warp
//      g mod NW (sums the NW partials in warp order, mask = per-row length, online max/sum in FP32), which writes P
//      (FP16) and the per-row rescale factor to shared memory; barrier.
//   PV (mma m8n8k4 row.row): B = V rows (each thread DH = D / 8 contiguous bytes of one token), A = P from shared.
// Pipeline: the next tile's K is loaded into the registers QK has just consumed, its V into those PV has; page bases
// are cached per thread (one table read per page).
// Splits: grid (S, Hkv, B); each CTA (one set) takes a contiguous tile range; with S > 1 every split writes an FP32
// partial (relative to its row max) + (max, sum), waits at a per-(request, head) counter, then reduces its 1/S slice
// of the R x 256 outputs. The host caps the grid at one resident wave whenever S > 1, so the wait cannot deadlock;
// the last split out resets the counter.
// Debug timing (set_stop(n)): 1 returns after Q staging, 2 after the main loop, 3 after the partial is written,
// 4 after the barrier, 5 after the combine weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <map>
#include <tuple>

namespace {
constexpr int HD = 256, NQ = 6, TILE = 32, QS = HD + 8, PS = TILE + 8, SMAX = 128, QMAX = 8, TBS = 64;

__device__ __forceinline__ void mma_rc(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_rr(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldg8(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ldg4(const void* p) {
  unsigned v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
template <int VW>
__device__ __forceinline__ void ldvw(unsigned* w, const uint8_t* p) {
  if constexpr (VW == 4) { const uint4 v = ldg16(p); w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w; }
  else if constexpr (VW == 2) { const uint2 v = ldg8(p); w[0] = v.x; w[1] = v.y; }
  else { w[0] = ldg4(p); }
}
// four E4M3 bytes -> two fp16x2 (bytes 0,1 and 2,3), each = E4M3 * 2^-8 exactly
__device__ __forceinline__ void e4m3x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned x0 = __byte_perm(w, 0u, 0x1404), x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}
__device__ __forceinline__ float ex2(float x) { return exp2f(x); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// RS (row split): the NG sets of a CTA take the SAME tiles, set g owning row groups [g RG, g RG + RG) of the request's
// rows (the other sets' reads of a tile come from L1 / L2); otherwise the sets take interleaved tiles with all rows and
// merge in shared memory.
template <int NW, int RG, int NG, int PD, bool RS>
struct MQ {
  static constexpr int NT = 32 * NW * NG, D = HD / NW, KU = D / 16, DQ = D / 4, DH = D / 8, NB = D / 32, VW = DH / 4;
  static constexpr int R = 8 * RG, RGW = (RG + NW - 1) / NW;
  static constexpr int RT = RS ? R * NG : R;                       // rows of the CTA
  static constexpr bool MERGE = NG > 1 && !RS;
  // main loop shared memory (bytes): Q shared by the NG sets, then per set X / P / A, then ML
  static constexpr int Q_ = 0, QB = RT * QS * 2;                   // Q rows [RT][QS] fp16
  static constexpr int XB = NW * RG * 32 * 8 * 4;                  // partial scores [NW][RG][32 lanes][8] fp32
  static constexpr int PB = R * PS * 2;                            // P [R][PS] fp16
  static constexpr int AB = R * 4;                                 // per-row rescale factor [R] fp32
  static constexpr int SETB = XB + PB + AB, SET_ = QB;             // set g at SET_ + g SETB: X, P, A
  static constexpr int OB = MERGE ? R * HD * 4 : 0;                // set merge [R][256] fp32 (aliases Q / sets)
  static constexpr int ML_ = (QB + NG * SETB > OB ? QB + NG * SETB : OB), MLB = NG * R * 2 * 4;   // [NG][R][2] = [RT][2] (RS)
  static constexpr int RS_ = ML_ + MLB, RSB = R * 2 * 4;           // CTA row (max, sum) [R][2] (NG > 1)
  static constexpr int TB_ = RS_ + RSB, TBB = TBS * 4;             // the request's first TBS block-table entries
  static constexpr int MAIN = TB_ + TBB + 16;
};
// combine shared memory (floats, aliases the main layout from 0; the block-table copy at TB_ stays): Acc [G][E][4]
// (G = ceil(Sb / 8) blocks of splits), Mk [Sb][nrs], Lk [Sb][nrs], Rs [2][nrs] with E <= ceil(n4 / Sb) output float4s
// of the split's slice and nrs <= ceil(E / 64) + 1 rows it touches
__host__ __device__ inline int comb_emax(int Sb, int nr) { return (nr * HD / 4 + Sb - 1) / Sb; }
__host__ __device__ inline int comb_rows(int Sb, int nr) { return (comb_emax(Sb, nr) + HD / 4 - 1) / (HD / 4) + 1; }
__host__ __device__ inline int comb_floats(int Sb, int nr) {
  return 4 * comb_emax(Sb, nr) * ((Sb + 7) / 8) + (2 * Sb + 2) * comb_rows(Sb, nr);
}

template <int NG>
__device__ __forceinline__ void bar_set(int sg, int nthreads) {
  if (NG == 1) __syncthreads();
  else asm volatile("bar.sync %0, %1;" ::"r"(1 + sg), "r"(nthreads) : "memory");
}

// ---- combine of one slice: split s of Sb (all partials written and visible) reduces output float4s [lo, lo + E) of
// the NR x 256 outputs of (request b, KV head h). Item (e, g) = output float4 e of the slice over the splits of block g
// (<= 8 splits); its loads are issued before the split statistics are read, block sums go to Acc[g][e] and are added
// in block order (deterministic). sm: comb_floats(Sb, NR) floats of shared memory.
template <int NT>
__device__ __forceinline__ void combine_slice(uint8_t* sm, const float* __restrict__ po, const float* __restrict__ pml,
                                              __half* __restrict__ out, int bh, int SG, int NR, int Sb, int s, int b,
                                              int h, int Hkv, int Q, float vsc, int stop) {
  const int tid = threadIdx.x;
  auto orow = [&](int r) -> size_t { return (((size_t)b * Q + r / NQ) * Hkv + h) * NQ + r % NQ; };
  const int n4 = NR * HD / 4, lo = (int)((long long)n4 * s / Sb), E = (int)((long long)n4 * (s + 1) / Sb) - lo;
  const int G = (Sb + 7) / 8;                                // blocks of <= 8 splits per output float4
  const int rlo = lo / (HD / 4), nrs = (lo + E - 1) / (HD / 4) - rlo + 1;   // rows the slice touches
  float* Acc = reinterpret_cast<float*>(sm);                 // [G][E][4]
  float* Mk = Acc + 4 * comb_emax(Sb, NR) * G;               // [Sb][nrs] split row maxima
  float* Lk = Mk + Sb * nrs;                                 // [Sb][nrs] split row sums
  float* Rs = Lk + Sb * nrs;                                 // [2][nrs] row max, v_scale / row sum
  // first item's partial loads in flight together with the statistics loads
  const int EG = E * G;
  float4 c[8];
  {
    const int idx = tid;
    if (idx < EG) {
      const int e = idx % E, g = idx / E, el = lo + e, row = el / (HD / 4), dd = (el % (HD / 4)) * 4;
      const int k0 = 8 * g, nk = min(8, Sb - k0);
      const float* src = po + ((size_t)bh * SG * NR + row) * HD + dd;
#pragma unroll
      for (int u = 0; u < 8; u++)
        if (u < nk) c[u] = __ldcg(reinterpret_cast<const float4*>(src + (size_t)(k0 + u) * NR * HD));
    }
  }
  const float* pm = pml + (size_t)bh * SG * NR * 2;
#pragma unroll 4
  for (int i = tid; i < Sb * nrs; i += NT) {
    const int k = i / nrs, r = rlo + i % nrs;
    Mk[i] = __ldcg(pm + 2 * (k * NR + r)); Lk[i] = __ldcg(pm + 2 * (k * NR + r) + 1);
  }
  __syncthreads();
  for (int rb = 0; rb < nrs; rb += NT / 8) {                 // 8 threads per row; every thread runs every pass
    const int r = rb + (tid >> 3), j = tid & 7;
    float M = -INFINITY;
    if (r < nrs) for (int k = j; k < Sb; k += 8) M = fmaxf(M, Mk[k * nrs + r]);
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));
    const float Mz = M == -INFINITY ? 0.f : M;
    float Lt = 0.f;
    if (r < nrs) for (int k = j; k < Sb; k += 8) Lt += Lk[k * nrs + r] * ex2(Mk[k * nrs + r] - Mz);
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) Lt += __shfl_xor_sync(0xffffffffu, Lt, d);
    if (j == 0 && r < nrs) { Rs[r] = Mz; Rs[nrs + r] = Lt > 0.f ? vsc / Lt : 0.f; }
  }
  __syncthreads();
  if (stop == 5) return;                                     // (debug: weights)
  for (int idx = tid; idx < EG; idx += NT) {
    const int e = idx % E, g = idx / E, el = lo + e, row = el / (HD / 4);
    const int k0 = 8 * g, nk = min(8, Sb - k0);
    if (idx != tid) {                                        // later items (large slices): load now
      const float* src = po + ((size_t)bh * SG * NR + row) * HD + (el % (HD / 4)) * 4;
#pragma unroll
      for (int u = 0; u < 8; u++)
        if (u < nk) c[u] = __ldcg(reinterpret_cast<const float4*>(src + (size_t)(k0 + u) * NR * HD));
    }
    const float M = Rs[row - rlo], inv = Rs[nrs + row - rlo];
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (u < nk) {
        const float wk = ex2(Mk[(k0 + u) * nrs + row - rlo] - M) * inv;
        a.x += wk * c[u].x; a.y += wk * c[u].y; a.z += wk * c[u].z; a.w += wk * c[u].w;
      }
    *reinterpret_cast<float4*>(Acc + 4 * (g * E + e)) = a;
  }
  __syncthreads();
  for (int i = tid; i < E; i += NT) {
    float4 a = *reinterpret_cast<const float4*>(Acc + 4 * i);
    for (int g = 1; g < G; g++) {
      const float4 u = *reinterpret_cast<const float4*>(Acc + 4 * (g * E + i));
      a.x += u.x; a.y += u.y; a.z += u.z; a.w += u.w;
    }
    const int eli = lo + i, rw = eli / (HD / 4), d = (eli % (HD / 4)) * 4;
    __half2* dst = reinterpret_cast<__half2*>(out + orow(rw) * HD + d);
    dst[0] = __floats2half2_rn(a.x, a.y);
    dst[1] = __floats2half2_rn(a.z, a.w);
  }
}

#ifdef Q27_MINB   // bench-only build switch (test_attn.py MQ_MINB): minimum resident CTAs per SM for the registers
#define Q27_LB(n) __launch_bounds__(n, Q27_MINB)
#else
#define Q27_LB(n) __launch_bounds__(n)
#endif
template <int NW, int RG, int NG, int PD, bool RS>
__global__ void Q27_LB(32 * NW * NG)
k_mq(const __half* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
     const int* __restrict__ table, int tstride, int tw, const int* __restrict__ lens, __half* __restrict__ out,
     float* __restrict__ po, float* __restrict__ pml, int* __restrict__ cnt, int Q, int S, int min_tiles, int page,
     long long kb, long long kt, long long kh, long long vb, long long vt, long long vh, float sl2, float vsc,
     int pf, int sep, int stop) {
  using C = MQ<NW, RG, NG, PD, RS>;
  constexpr int TS = RS ? 1 : NG;                            // tile stride of a set
  constexpr int NT = C::NT, D = C::D, KU = C::KU, DQ = C::DQ, DH = C::DH, NB = C::NB, VW = C::VW, RGW = C::RGW;
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, sg = warp / NW, wi = warp % NW;
  __half* Qs = reinterpret_cast<__half*>(sm + C::Q_);
  float* Xs = reinterpret_cast<float*>(sm + C::SET_ + sg * C::SETB);
  __half* Ps = reinterpret_cast<__half*>(sm + C::SET_ + sg * C::SETB + C::XB);
  float* As = reinterpret_cast<float*>(sm + C::SET_ + sg * C::SETB + C::XB + C::PB);
  float* MLa = reinterpret_cast<float*>(sm + C::ML_);              // [NG][R][2]
  float* MLs = MLa + sg * C::R * 2;
  int* Ts = reinterpret_cast<int*>(sm + C::TB_);

  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z, Hkv = gridDim.y;
  const int bh = b * Hkv + h, NR = NQ * Q;
  const int* lr = lens + (size_t)b * Q;
  // attention row r of this (request, KV head) = q / out row (b Q + r / 6, head 6 h + r % 6)
  auto orow = [&](int r) -> size_t { return (((size_t)b * Q + r / NQ) * Hkv + h) * NQ + r % NQ; };
  // all loads of the Q rows first (registers), stored to shared memory once the first tiles' loads are in flight
  constexpr int QN = C::RT * HD / 8, QPT = (QN + NT - 1) / NT;
  uint4 qv[QPT];
#pragma unroll
  for (int u = 0; u < QPT; u++) {
    const int i = tid + u * NT, r = i / (HD / 8), c = (i % (HD / 8)) * 8;
    qv[u] = make_uint4(0, 0, 0, 0);
    if (i < QN && r < NR) qv[u] = *reinterpret_cast<const uint4*>(q + orow(r) * HD + c);
  }
  // the request's first TBS block-table entries, loaded with the Q rows (so the first tile's address does not wait on a
  // dependent table load), stored to shared memory below
  const int* tb = table + (size_t)b * tstride;
  const int tbv = tid < min(tw, TBS) ? tb[tid] : 0;
  int Lmax = 0;
#pragma unroll
  for (int i = 0; i < QMAX; i++)
    if (i < Q) Lmax = max(Lmax, lr[i]);
  if (Lmax <= 0) {                                           // padding request: zeros
    if (s == 0)
      for (int i = tid; i < NR * HD / 2; i += NT)
        reinterpret_cast<__half2*>(out + orow(i / (HD / 2)) * HD)[i % (HD / 2)] = __floats2half2_rn(0.f, 0.f);
    return;
  }
  const int T = (Lmax + TILE - 1) / TILE;
  const int Sb = max(1, min(S, T / (TS * min_tiles)));
  if (s >= Sb) return;
  const int t0 = (int)((long long)T * s / Sb), t1 = (int)((long long)T * (s + 1) / Sb);

  const int qp = (lane >> 2) & 3, hi = (lane >> 4) & 1, l3 = lane & 3;
  const int arow = l3 + 4 * hi;                      // A row; B column (QK token inside the quadpair)
  const int r0 = (lane & 1) + 4 * hi, c0 = lane & 2;     // C rows r0, r0 + 2; columns c0, c0+1, c0+4, c0+5
  const int dim0 = D * wi;
  const int roff = RS ? sg * C::R : 0;                       // first row of this set
  const int ktok = 8 * qp + arow;                    // this thread's token in a tile (QK)
  const int vdim = dim0 + DQ * qp + DH * hi;         // this thread's DH V bytes (PV)
  if (tid < TBS) Ts[tid] = tbv;
  __syncthreads();
  uint4 kr0[KU], kr1[KU];                                    // K / V of two tiles in flight (PD 2); PD 1 uses kr0 / vr0
  unsigned vr0[8][VW], vr1[8][VW];
  int pcur = -1;
  const uint8_t *kpb = kc, *vpb = vc;
  auto setpage = [&](int t) {
    const int pi = t * TILE / page;
    if (pi != pcur) {
      pcur = pi;
      const long long pg = pi < TBS ? Ts[pi] : tb[pi];
      kpb = kc + pg * kb + (long long)h * kh + (long long)ktok * kt + dim0;
      vpb = vc + pg * vb + (long long)h * vh + (long long)l3 * vt + vdim;
    }
  };
  auto ldk = [&](int t, uint4 (&kr)[KU]) {
    setpage(t);
    const uint8_t* p = kpb + (long long)(t * TILE % page) * kt;
#pragma unroll
    for (int i = 0; i < KU; i++) kr[i] = ldg16(p + 16 * i);
  };
  auto ldv = [&](int t, unsigned (&vr)[8][VW]) {
    setpage(t);
    const uint8_t* p = vpb + (long long)(t * TILE % page) * vt;
#pragma unroll
    for (int i = 0; i < 8; i++) ldvw<VW>(vr[i], p + (long long)(4 * i) * vt);
  };
  // L2 prefetch of the pf tiles after the ones loaded into registers (no registers held): each thread its token's
  // D K bytes and D V bytes; separate page-base cache
  int pfcur = -1;
  const uint8_t *kpf = kc, *vpf = vc;
  auto pfx = [&](int t) {
    if (t >= t1) return;
    const int pi = t * TILE / page;
    if (pi != pfcur) {
      pfcur = pi;
      const long long pg = pi < TBS ? Ts[pi] : tb[pi];
      kpf = kc + pg * kb + (long long)h * kh + (long long)ktok * kt + dim0;
      vpf = vc + pg * vb + (long long)h * vh + (long long)ktok * vt + dim0;
    }
    const long long o = (long long)(t * TILE % page);
#pragma unroll
    for (int c = 0; c < D; c += 32) { pf_l2(kpf + o * kt + c); pf_l2(vpf + o * vt + c); }
  };
  int t = RS ? t0 : t0 + sg;                                 // interleaved tiles (all sets the same ones with RS)
  if (t < t1) { ldk(t, kr0); ldv(t, vr0); }
  if (PD == 2 && t + TS < t1) { ldk(t + TS, kr1); ldv(t + TS, vr1); }
  for (int j = 0; j < pf; j++) pfx(t + (PD + j) * TS);

#pragma unroll
  for (int u = 0; u < QPT; u++) {
    const int i = tid + u * NT, r = i / (HD / 8), c = (i % (HD / 8)) * 8;
    if (i < QN) *reinterpret_cast<uint4*>(Qs + r * QS + c) = qv[u];
  }
  // lengths of the rows whose softmax this warp owns (rows 8 g + r0 and 8 g + r0 + 2 of row groups g = wi + u NW)
  int lenA[RGW], lenB[RGW];
#pragma unroll
  for (int u = 0; u < RGW; u++) {
    const int ra = roff + 8 * (wi + u * NW) + r0, rb = ra + 2;
    lenA[u] = ra < NR ? lr[ra / NQ] : 0;
    lenB[u] = rb < NR ? lr[rb / NQ] : 0;
  }
  __syncthreads();
  if (stop == 1) return;                                     // (debug timing: launch + Q staging)
  const __half* qa = Qs + (roff + arow) * QS + dim0;

  float o[RG][NB][8];
#pragma unroll
  for (int g = 0; g < RG; g++)
#pragma unroll
    for (int j = 0; j < NB; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[g][j][e] = 0.f;
  float m0[RGW], m2[RGW], l0[RGW], l2[RGW];
#pragma unroll
  for (int u = 0; u < RGW; u++) { m0[u] = -INFINITY; m2[u] = -INFINITY; l0[u] = 0.f; l2[u] = 0.f; }

  auto tile = [&](const int t, uint4 (&kr)[KU], unsigned (&vr)[8][VW]) {
    const int tok0 = t * TILE, tn = t + PD * TS;             // tn: the tile whose loads go into these registers next
    const bool more = tn < t1;
    // ---- QK: partial scores over this warp's D dims for all row groups (one K decode feeds RG MMAs)
    float sc[RG][8], sx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      sx[e] = 0.f;
#pragma unroll
      for (int g = 0; g < RG; g++) sc[g][e] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < KU; i++) {
      const unsigned w[4] = {kr[i].x, kr[i].y, kr[i].z, kr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
#pragma unroll
        for (int g = 0; g < RG; g++) {
          const uint2 a = *reinterpret_cast<const uint2*>(qa + g * 8 * QS + 4 * (4 * i + j));
          mma_rc((RG == 1 && (i & 1)) ? sx : sc[g], a.x, a.y, b0, b1);   // RG 1: two chains
        }
      }
    }
    if (more) {
      ldk(tn, kr);
      if (pf) pfx(tn + pf * TS);
    }
    if (RG == 1) {
#pragma unroll
      for (int e = 0; e < 8; e++) sc[0][e] += sx[e];
    }
#pragma unroll
    for (int g = 0; g < RG; g++) {
      float* xw = Xs + ((wi * RG + g) * 32 + lane) * 8;
      *reinterpret_cast<float4*>(xw) = make_float4(sc[g][0], sc[g][1], sc[g][2], sc[g][3]);
      *reinterpret_cast<float4*>(xw + 4) = make_float4(sc[g][4], sc[g][5], sc[g][6], sc[g][7]);
    }
    bar_set<NG>(sg, 32 * NW);
    // ---- softmax of the owned row groups (g warp-uniform: whole warps, full-mask shuffles are safe)
#pragma unroll
    for (int u = 0; u < RGW; u++) {
      const int g = wi + u * NW;
      if (g < RG) {
        float sv[8];
        {
          const float* x = Xs + (g * 32 + lane) * 8;
          const float4 a = *reinterpret_cast<const float4*>(x), c = *reinterpret_cast<const float4*>(x + 4);
          sv[0] = a.x; sv[1] = a.y; sv[2] = a.z; sv[3] = a.w; sv[4] = c.x; sv[5] = c.y; sv[6] = c.z; sv[7] = c.w;
        }
#pragma unroll
        for (int w2 = 1; w2 < NW; w2++) {                    // warp order: the same sum for every row
          const float* x = Xs + ((w2 * RG + g) * 32 + lane) * 8;
          const float4 a = *reinterpret_cast<const float4*>(x), c = *reinterpret_cast<const float4*>(x + 4);
          sv[0] += a.x; sv[1] += a.y; sv[2] += a.z; sv[3] += a.w; sv[4] += c.x; sv[5] += c.y; sv[6] += c.z; sv[7] += c.w;
        }
        const int cb = tok0 + 8 * qp + c0;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int col = cb + (e & 1) + (e & 4);
          sv[e] = col < ((e & 2) ? lenB[u] : lenA[u]) ? sv[e] * sl2 : -INFINITY;
        }
        float x0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), x2 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
#pragma unroll
        for (int d = 2; d <= 8; d <<= 1) {
          x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
          x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
        }
        const float n0 = fmaxf(m0[u], x0), n2 = fmaxf(m2[u], x2);
        const float a0 = m0[u] == -INFINITY ? 0.f : ex2(m0[u] - n0), a2 = m2[u] == -INFINITY ? 0.f : ex2(m2[u] - n2);
        const float z0 = n0 == -INFINITY ? 0.f : n0, z2 = n2 == -INFINITY ? 0.f : n2;   // fully masked rows: p = 0
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; e++) p[e] = ex2(sv[e] - ((e & 2) ? z2 : z0));
        l0[u] = l0[u] * a0 + (p[0] + p[1]) + (p[4] + p[5]);
        l2[u] = l2[u] * a2 + (p[2] + p[3]) + (p[6] + p[7]);
        m0[u] = n0; m2[u] = n2;
        __half* Pg = Ps + g * 8 * PS;
        const int pc = 8 * qp + c0;
        *reinterpret_cast<__half2*>(Pg + r0 * PS + pc) = __floats2half2_rn(p[0], p[1]);
        *reinterpret_cast<__half2*>(Pg + r0 * PS + pc + 4) = __floats2half2_rn(p[4], p[5]);
        *reinterpret_cast<__half2*>(Pg + (r0 + 2) * PS + pc) = __floats2half2_rn(p[2], p[3]);
        *reinterpret_cast<__half2*>(Pg + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(p[6], p[7]);
        if ((lane & 14) == 0) { As[g * 8 + r0] = a0; As[g * 8 + r0 + 2] = a2; }
      }
    }
    bar_set<NG>(sg, 32 * NW);
    // ---- PV: rescale, then accumulate this warp's D dims for all row groups (one V decode feeds RG MMAs)
#pragma unroll
    for (int g = 0; g < RG; g++) {
      const float a0 = As[g * 8 + r0], a2 = As[g * 8 + r0 + 2];
#pragma unroll
      for (int j = 0; j < NB; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) o[g][j][e] *= (e & 2) ? a2 : a0;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint2 a[RG];
#pragma unroll
      for (int g = 0; g < RG; g++) a[g] = *reinterpret_cast<const uint2*>(Ps + (g * 8 + arow) * PS + 4 * i);
#pragma unroll
      for (int j = 0; j < NB; j++) {
        unsigned b0, b1;
        e4m3x4(vr[i][j], b0, b1);
#pragma unroll
        for (int g = 0; g < RG; g++) mma_rr(o[g][j], a[g].x, a[g].y, b0, b1);
      }
    }
    if (more) ldv(tn, vr);
  };
  for (; t < t1; t += PD * TS) {
    tile(t, kr0, vr0);
    if (PD == 2 && t + TS < t1) tile(t + TS, kr1, vr1);
  }
  if (stop == 2) { if (m0[0] == 12345.f) out[0] = __float2half(o[0][0][0] + o[RG - 1][NB - 1][7]); return; }
  // owners: row sums over the 8 lanes that share a row, then (max, sum) per row -> shared (per set)
#pragma unroll
  for (int u = 0; u < RGW; u++) {
    const int g = wi + u * NW;
    if (g < RG) {
#pragma unroll
      for (int d = 2; d <= 8; d <<= 1) {
        l0[u] += __shfl_xor_sync(0xffffffffu, l0[u], d);
        l2[u] += __shfl_xor_sync(0xffffffffu, l2[u], d);
      }
      if ((lane & 14) == 0) {                                // per set [R][2]; RS: the set's rows of [RT][2]
        MLs[(g * 8 + r0) * 2] = m0[u]; MLs[(g * 8 + r0) * 2 + 1] = l0[u];
        MLs[(g * 8 + r0 + 2) * 2] = m2[u]; MLs[(g * 8 + r0 + 2) * 2 + 1] = l2[u];
      }
    }
  }
  __syncthreads();
  const int SG = gridDim.x;                                  // partial slots per (request, head)
  float* pob = po + ((size_t)bh * SG + s) * NR * HD;
  float* pmb = pml + ((size_t)bh * SG + s) * NR * 2;
  if constexpr (!C::MERGE) {
    if (Sb == 1) {
#pragma unroll
      for (int g = 0; g < RG; g++)
#pragma unroll
        for (int sr = 0; sr < 2; sr++) {
          const int row = roff + g * 8 + r0 + 2 * sr;
          if (row < NR) {
            const float l = MLa[row * 2 + 1], f = l > 0.f ? vsc / l : 0.f;
            __half* ob = out + orow(row) * HD;
#pragma unroll
            for (int j = 0; j < NB; j++) {
              const int dlo = dim0 + DQ * qp + 4 * j + c0;
              *reinterpret_cast<__half2*>(ob + dlo) = __floats2half2_rn(o[g][j][2 * sr] * f, o[g][j][2 * sr + 1] * f);
              *reinterpret_cast<__half2*>(ob + dlo + DH) = __floats2half2_rn(o[g][j][4 + 2 * sr] * f, o[g][j][5 + 2 * sr] * f);
            }
          }
        }
      return;
    }
    // partial straight from the registers: FP32 numerators relative to the split's row max, (max, sum) per row
#pragma unroll
    for (int g = 0; g < RG; g++)
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int row = roff + g * 8 + r0 + 2 * sr;
        if (row < NR) {
#pragma unroll
          for (int j = 0; j < NB; j++) {
            const int dlo = dim0 + DQ * qp + 4 * j + c0;
            __stcg(reinterpret_cast<float2*>(pob + row * HD + dlo), make_float2(o[g][j][2 * sr], o[g][j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + row * HD + dlo + DH), make_float2(o[g][j][4 + 2 * sr], o[g][j][5 + 2 * sr]));
          }
        }
      }
    for (int i = tid; i < NR * 2; i += NT) __stcg(pmb + i, MLa[i]);
  } else {
    // merge the NG sets: CTA row max M, each set's accumulators scaled by 2^(m_set - M), summed set by set into Os
    float* Os = reinterpret_cast<float*>(sm);                // [NR][256] (Q / X / P are dead)
    float* RSs = reinterpret_cast<float*>(sm + C::RS_);      // [R][2] CTA row (max, sum)
    float fr[RG][2];
#pragma unroll
    for (int g = 0; g < RG; g++)
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int row = g * 8 + r0 + 2 * sr;
        float M = -INFINITY;
#pragma unroll
        for (int k = 0; k < NG; k++) M = fmaxf(M, MLa[(k * C::R + row) * 2]);
        const float mk = MLs[row * 2];
        fr[g][sr] = mk == -INFINITY ? 0.f : ex2(mk - M);
      }
#pragma unroll
    for (int k = 0; k < NG; k++) {
      if (sg == k) {
#pragma unroll
        for (int g = 0; g < RG; g++)
#pragma unroll
          for (int sr = 0; sr < 2; sr++) {
            const int row = g * 8 + r0 + 2 * sr;
            if (row < NR) {
              const float f = fr[g][sr];
#pragma unroll
              for (int j = 0; j < NB; j++) {
                const int dlo = dim0 + DQ * qp + 4 * j + c0;
                float2* d0 = reinterpret_cast<float2*>(Os + row * HD + dlo);
                float2* d1 = reinterpret_cast<float2*>(Os + row * HD + dlo + DH);
                float2 v0 = make_float2(o[g][j][2 * sr] * f, o[g][j][2 * sr + 1] * f);
                float2 v1 = make_float2(o[g][j][4 + 2 * sr] * f, o[g][j][5 + 2 * sr] * f);
                if (k > 0) { const float2 u0 = *d0, u1 = *d1; v0.x += u0.x; v0.y += u0.y; v1.x += u1.x; v1.y += u1.y; }
                *d0 = v0; *d1 = v1;
              }
            }
          }
      }
      __syncthreads();
    }
    for (int r = tid; r < NR; r += NT) {
      float M = -INFINITY, Lt = 0.f;
#pragma unroll
      for (int k = 0; k < NG; k++) M = fmaxf(M, MLa[(k * C::R + r) * 2]);
#pragma unroll
      for (int k = 0; k < NG; k++) {
        const float mk = MLa[(k * C::R + r) * 2];
        if (mk != -INFINITY) Lt += MLa[(k * C::R + r) * 2 + 1] * ex2(mk - M);
      }
      RSs[2 * r] = M; RSs[2 * r + 1] = Lt;
    }
    __syncthreads();
    if (Sb == 1) {
      for (int i = tid; i < NR * HD / 4; i += NT) {
        const int row = i / (HD / 4), d = (i % (HD / 4)) * 4;
        const float l = RSs[2 * row + 1], f = l > 0.f ? vsc / l : 0.f;
        const float4 a = *reinterpret_cast<const float4*>(Os + row * HD + d);
        __half2* dst = reinterpret_cast<__half2*>(out + orow(row) * HD + d);
        dst[0] = __floats2half2_rn(a.x * f, a.y * f);
        dst[1] = __floats2half2_rn(a.z * f, a.w * f);
      }
      return;
    }
    for (int i = tid; i < NR * HD / 4; i += NT)
      __stcg(reinterpret_cast<float4*>(pob) + i, *reinterpret_cast<const float4*>(Os + 4 * i));
    for (int i = tid; i < NR * 2; i += NT) __stcg(pmb + i, RSs[i]);
  }
  if (sep) return;                                           // combine in k_comb (separate launch)
  __threadfence();
  __syncthreads();                                           // shared memory read by every thread above; partial out
  if (stop == 3) return;                                     // (debug: partial written)
  if (tid == 0) {
    const int arr = atomicAdd(cnt + bh, 1);
    long long spins = 0;
    while (*reinterpret_cast<volatile int*>(cnt + bh) < Sb) {
      __nanosleep(64);
      if (++spins == (1ll << 22)) {                          // ~0.5-1 s: report once (diagnostic)
        printf("q27 attn_mq: split barrier stalled: b %d h %d s %d Sb %d grid %d arrival %d cnt %d Lmax %d Q %d\n", b, h, s, Sb,
               (int)gridDim.x, arr, *reinterpret_cast<volatile int*>(cnt + bh), Lmax, Q);
      }
    }
    __threadfence();
  }
  __syncthreads();
  if (stop == 4) return;                                     // (debug: barrier passed)
  combine_slice<NT>(sm, po, pml, out, bh, (int)gridDim.x, NR, Sb, s, b, h, Hkv, Q, vsc, stop);
  if (tid == 0 && atomicAdd(cnt + bh, 1) == 2 * Sb - 1) atomicExch(cnt + bh, 0);   // last one out resets
}

// separate combine (sep): grid (S, Hkv, B) x NT threads; recomputes each request's split count exactly as k_mq did
// (ts_mt = tile stride x min tiles of the producing configuration), slice s < Sb reduces its outputs.
template <int NT>
__global__ void __launch_bounds__(NT)
k_comb(const float* __restrict__ po, const float* __restrict__ pml, const int* __restrict__ lens, __half* __restrict__ out,
       int Q, int S, int ts_mt, float vsc) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z, Hkv = gridDim.y, bh = b * Hkv + h;
  const int* lr = lens + (size_t)b * Q;
  int Lmax = 0;
  for (int i = 0; i < Q; i++) Lmax = max(Lmax, lr[i]);
  if (Lmax <= 0) return;
  const int T = (Lmax + TILE - 1) / TILE;
  const int Sb = max(1, min(S, T / ts_mt));
  if (Sb == 1 || s >= Sb) return;
  combine_slice<NT>(sm, po, pml, out, bh, (int)gridDim.x, NQ * Q, Sb, s, b, h, Hkv, Q, vsc, 0);
}

int STOP = 0;

// dynamic shared memory of a launch with up to S splits and nr real rows
template <int NW, int RG, int NG, int PD, bool RS>
int smem_bytes(int S, int nr) {
  int b = MQ<NW, RG, NG, PD, RS>::MAIN;
  for (int sb = 2; sb <= S; sb++) b = std::max(b, 4 * comb_floats(sb, nr) + 16);   // splits are cut at run time
  return b;
}
// resident CTAs per wave for this kernel at this shared memory size, planned with the carveout pinned to the
// shared-memory maximum (the in-kernel split wait, sep 0, needs the launch to get exactly this occupancy: one
// resident wave). The split count is planned this way in both modes (workspace sizing stays one rule); launch() then
// sets the carveout per mode: pinned for sep 0, the driver's default for sep 1 (nothing depends on residency, and no
// L1 / shared reconfiguration is forced around this kernel inside a graph of other kernels).
template <int NW, int RG, int NG, int PD, bool RS>
int capacity(int smem) {
  static std::map<int, int> caps;
  auto it = caps.find(smem);
  if (it != caps.end()) return it->second;
  int occ = 0, dev = 0, sms = 0;
  auto f = k_mq<NW, RG, NG, PD, RS>;
  TORCH_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) == cudaSuccess);
  TORCH_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared) == cudaSuccess);
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 32 * NW * NG, smem) == cudaSuccess);
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return caps[smem] = occ * sms;
}
// splits a launch really uses: S cut to SMAX and to one resident wave of B Hkv S CTAs
template <int NW, int RG, int NG, int PD, bool RS>
int plan_splits(long long bh, int S, int nr) {
  S = std::max(1, std::min(S, SMAX));
  if (S > 1) {
    const int cap = capacity<NW, RG, NG, PD, RS>(smem_bytes<NW, RG, NG, PD, RS>(S, nr));
    if (bh * S > cap) S = std::max<long long>(1, cap / bh);
  }
  return S;
}

struct Args {
  const torch::Tensor *q, *k, *v, *table, *lens, *out, *po, *pml, *cnt;
  int Q, S, min_tiles, pf, sep;
  float sl2, vsc;
  cudaStream_t st;
};

template <int NW, int RG, int NG, int PD, bool RS>
void launch(const Args& a) {
  const int B = a.table->size(0), Hkv = a.k->size(2), nr = NQ * a.Q;
  const int S = plan_splits<NW, RG, NG, PD, RS>((long long)B * Hkv, a.S, nr);
  const int smem = smem_bytes<NW, RG, NG, PD, RS>(S, nr);
  capacity<NW, RG, NG, PD, RS>(smem);                                // sets the function attributes once
  // carveout for this launch (capacity() pins it while planning; graph replays run no host code, so this costs one
  // driver call per eager / captured launch only)
  TORCH_CHECK(cudaFuncSetAttribute(k_mq<NW, RG, NG, PD, RS>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   a.sep ? (int)cudaSharedmemCarveoutDefault : (int)cudaSharedmemCarveoutMaxShared) == cudaSuccess);
  const torch::Tensor &k = *a.k, &v = *a.v;
  dim3 grid(S, Hkv, B);
  k_mq<NW, RG, NG, PD, RS><<<grid, 32 * NW * NG, smem, a.st>>>(
      reinterpret_cast<const __half*>(a.q->data_ptr<at::Half>()), k.data_ptr<uint8_t>(), v.data_ptr<uint8_t>(),
      a.table->data_ptr<int>(), (int)a.table->stride(0), (int)a.table->size(1), a.lens->data_ptr<int>(),
      reinterpret_cast<__half*>(a.out->data_ptr<at::Half>()), a.po->data_ptr<float>(), a.pml->data_ptr<float>(),
      a.cnt->data_ptr<int>(), a.Q, S, a.min_tiles, (int)k.size(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
      v.stride(1), v.stride(2), a.sl2, a.vsc, a.pf, a.sep, STOP);
  if (a.sep && S > 1) {
    constexpr int CT = 256;
    int cs = 0;
    for (int sb = 2; sb <= S; sb++) cs = std::max(cs, 4 * comb_floats(sb, nr) + 16);
    static bool attr = false;
    if (!attr) {
      TORCH_CHECK(cudaFuncSetAttribute(k_comb<CT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) == cudaSuccess);
      attr = true;
    }
    k_comb<CT><<<grid, CT, cs, a.st>>>(a.po->data_ptr<float>(), a.pml->data_ptr<float>(), a.lens->data_ptr<int>(),
                                        reinterpret_cast<__half*>(a.out->data_ptr<at::Half>()), a.Q, S,
                                        (RS ? 1 : NG) * a.min_tiles, a.vsc);
  }
}

// instantiated (nw, row groups per set, ng, prefetch depth, row split): nw 2 up to 3 row groups (spills above), ng 2
// up to 4 row groups, ng 4 for 1; none above 512 threads or that the 512-thread register cap (128) makes spill.
// Row split (rs 1): the set count is the row-group count / RG, e.g. Q 5 (4 row groups) = rg 1 x ng 4 or rg 2 x ng 2.
#define MQ_CASES(X) \
  X(2, 1, 1, 1, 0) X(2, 2, 1, 1, 0) X(2, 3, 1, 1, 0) \
  X(4, 1, 1, 1, 0) X(4, 2, 1, 1, 0) X(4, 3, 1, 1, 0) X(4, 4, 1, 1, 0) X(4, 5, 1, 1, 0) X(4, 6, 1, 1, 0) \
  X(8, 1, 1, 1, 0) X(8, 2, 1, 1, 0) X(8, 3, 1, 1, 0) X(8, 4, 1, 1, 0) X(8, 5, 1, 1, 0) X(8, 6, 1, 1, 0) \
  X(4, 1, 2, 1, 0) X(4, 2, 2, 1, 0) X(4, 3, 2, 1, 0) X(4, 4, 2, 1, 0) X(8, 1, 2, 1, 0) X(8, 2, 2, 1, 0) X(8, 3, 2, 1, 0) \
  X(4, 1, 1, 2, 0) X(4, 2, 1, 2, 0) X(4, 3, 1, 2, 0) X(4, 4, 1, 2, 0) \
  X(8, 1, 1, 2, 0) X(8, 2, 1, 2, 0) X(8, 3, 1, 2, 0) X(8, 4, 1, 2, 0) X(8, 5, 1, 2, 0) X(8, 6, 1, 2, 0) \
  X(4, 1, 2, 2, 0) X(4, 2, 2, 2, 0) X(8, 1, 2, 2, 0) X(8, 2, 2, 2, 0) \
  X(4, 1, 2, 1, 1) X(4, 1, 3, 1, 1) X(4, 1, 4, 1, 1) X(4, 2, 2, 1, 1) X(4, 1, 2, 2, 1) X(4, 1, 3, 2, 1) X(4, 1, 4, 2, 1) \
  X(8, 1, 2, 1, 1) X(8, 2, 2, 1, 1) X(8, 1, 2, 2, 1) X(2, 1, 2, 1, 1) X(2, 1, 3, 1, 1) X(2, 1, 4, 1, 1) X(2, 2, 2, 1, 1)

int splits_for(int nw, int rg, int ng, int pd, int rs, long long bh, int S, int nr) {
#define X(a, b, c, d, e) if (nw == a && rg == b && ng == c && pd == d && rs == e) return plan_splits<a, b, c, d, e>(bh, S, nr);
  MQ_CASES(X)
#undef X
  return -1;
}
bool launch_for(int nw, int rg, int ng, int pd, int rs, const Args& a) {
#define X(a_, b_, c_, d_, e_) if (nw == a_ && rg == b_ && ng == c_ && pd == d_ && rs == e_) { launch<a_, b_, c_, d_, e_>(a); return true; }
  MQ_CASES(X)
#undef X
  return false;
}
}  // namespace

void set_stop(int64_t s) { STOP = (int)s; }
int64_t row_groups(int64_t Q) { return (NQ * Q + 7) / 8; }
// splits a launch uses for B Hkv sequences (1 = no partials; -1 = configuration not instantiated)
// ng: sets per CTA (1..4); pd: tiles in flight per set (1, 2); rs: row split (the ng sets share each tile, each owns
// ceil(row groups / ng) of them)
int rg_per_set(int64_t Q, int64_t ng, int64_t rs) { return rs ? (int)((row_groups(Q) + ng - 1) / ng) : (int)row_groups(Q); }
int64_t mq_splits(int64_t bh, int64_t S, int64_t nw, int64_t Q, int64_t ng, int64_t pd, int64_t rs) {
  return splits_for((int)nw, rg_per_set(Q, ng, rs), (int)ng, (int)pd, (int)rs, bh, (int)S, NQ * (int)Q);
}
// partial workspace (floats) a launch needs: po, pml
std::tuple<int64_t, int64_t> mq_workspace(int64_t bh, int64_t S, int64_t nw, int64_t Q, int64_t ng, int64_t pd,
                                          int64_t rs) {
  const int64_t s = mq_splits(bh, S, nw, Q, ng, pd, rs);
  if (s <= 1) return {0, 0};
  return {bh * s * NQ * Q * HD, bh * s * NQ * Q * 2};
}

// q [B Q][6 Hkv][256] fp16; k, v [blocks][page][Hkv][256] uint8 E4M3 (any strides, last dim dense; page % 32 == 0);
// table [B][W] int32 (per request); lens [B Q] int32 (per row: tokens that row attends to, 0 = padding); out like q.
// po / pml >= mq_workspace floats; cnt [>= B Hkv] int32 zeroed once (the kernel leaves it zeroed).
// nw: warps per set (2, 4, 8; head dims split nw ways); ng: sets per CTA (tiles interleaved, merged in shared memory);
// S: splits per request (cut at run time for short sequences and to one resident wave); min_tiles: minimum 32-token
// tiles per set; pd: tiles in flight in registers per set (1, 2); pf: further tiles per set prefetched into L2 (0 = off).
void attn_mq(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor table, torch::Tensor lens,
             torch::Tensor out, torch::Tensor po, torch::Tensor pml, torch::Tensor cnt, double scale, double k_scale,
             double v_scale, int64_t Q, int64_t nw, int64_t S, int64_t min_tiles, int64_t ng, int64_t pd, int64_t pf,
             int64_t rs, int64_t sep) {
  TORCH_CHECK(q.dtype() == torch::kHalf && q.is_contiguous() && q.dim() == 3 && q.size(2) == HD);
  TORCH_CHECK(k.dtype() == torch::kUInt8 && v.dtype() == torch::kUInt8 && k.dim() == 4 && v.dim() == 4);
  TORCH_CHECK(k.size(3) == HD && k.stride(3) == 1 && v.stride(3) == 1 && k.sizes() == v.sizes());
  TORCH_CHECK(q.size(1) == NQ * k.size(2), "6 query heads per KV head");
  TORCH_CHECK(k.size(1) % TILE == 0, "page size must be a multiple of 32");
  TORCH_CHECK(Q >= 1 && Q <= QMAX, "1..8 query rows per request");
  TORCH_CHECK(table.dtype() == torch::kInt && table.dim() == 2 && table.stride(1) == 1 && table.size(0) * Q == q.size(0));
  TORCH_CHECK(lens.dtype() == torch::kInt && lens.is_contiguous() && lens.size(0) == q.size(0));
  TORCH_CHECK(out.sizes() == q.sizes() && out.is_contiguous() && out.dtype() == torch::kHalf);
  TORCH_CHECK(min_tiles >= 1);
  const int64_t bh = table.size(0) * k.size(2);
  TORCH_CHECK(mq_splits(bh, S, nw, Q, ng, pd, rs) >= 1, "q27 attn_mq: (nw, row groups, ng, pd, rs) not instantiated");
  const auto ws = mq_workspace(bh, S, nw, Q, ng, pd, rs);
  TORCH_CHECK(po.numel() >= std::get<0>(ws) && pml.numel() >= std::get<1>(ws), "q27 attn_mq: workspace too small");
  TORCH_CHECK(cnt.numel() >= bh && cnt.dtype() == torch::kInt);
  if (q.size(0) == 0) return;
  const at::cuda::CUDAGuard g(q.device());
  TORCH_CHECK(pf >= 0 && pf <= 8);
  Args a{&q, &k, &v, &table, &lens, &out, &po, &pml, &cnt, (int)Q, (int)S, (int)min_tiles, (int)pf, (int)sep,
         (float)(scale * k_scale * 256.0 * 1.4426950408889634), (float)(v_scale * 256.0),
         at::cuda::getCurrentCUDAStream()};
  launch_for((int)nw, rg_per_set(Q, ng, rs), (int)ng, (int)pd, (int)rs, a);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_mq", &attn_mq, "split-KV E4M3 decode attention, Q rows per request, 6 query heads per KV head");
  m.def("mq_splits", &mq_splits, "splits a launch uses (bh, S, nw, Q, ng, pd, rs); -1 = not instantiated");
  m.def("mq_workspace", &mq_workspace, "partial workspace floats (po, pml) a launch needs (bh, S, nw, Q, ng, pd, rs)");
  m.def("set_stop", &set_stop, "debug timing: kernels return after phase n (0 = off)");
  m.attr("SMAX") = (int64_t)SMAX;
}
