// attn_dec.cu - split-KV decode attention for the Qwen3.8-27B full-attention layers on V100: one query token per
// sequence, 6 query heads per KV head, head dim 256, FP8 E4M3 paged KV cache (1Cat's layout and numerics contract:
// FP32 scores and softmax, P in FP16, FP32 accumulation; replaces grouped_e4m3_2cta_partial + parallel_combine).
//
// Work unit: a pair of warps on a 32-token tile. Both warps take all 32 tokens; warp wg owns head dims
// [128 wg, 128 wg + 128) for both products:
//   QK (mma m8n8k4 row.col): A = Q rows (6 heads, padded to 8) from shared memory; B = K^T, one column per thread =
//      that thread's token, so each thread streams its own token's 128 K bytes with 16-byte loads (no staging). The
//      two warps add their partial scores through a 2 KB shared exchange (pair barrier).
//   softmax: online, FP32, per row; P (FP16) goes through a 640-byte per-warp buffer into the A-fragment layout.
//   PV (mma m8n8k4 row.row): B = V rows, so each thread loads 16 contiguous bytes of one token (a warp-load covers
//      4 tokens x 128 B); quadpair qp, half h own dims 128 wg + 32 qp + 16 h + [0, 16), four 8-column MMA blocks.
// E4M3 -> FP16 exactly (x 2^-8, folded into the scales): one byte_perm + shift + LOP3 per pair.
// Splits: grid (S, kv heads, B); a CTA = NG pairs on interleaved tiles of its token range (each pair loads the next
// tile's K into the registers QK has just consumed and its V into those PV has, so loads overlap compute with no
// extra registers; page bases are cached per pair), merged in shared memory into one partial (FP32, relative to the
// CTA's row max). With S > 1 every split of a sequence waits at a per-sequence counter, then reduces its own 1/S slice
// of the 6 x 256 outputs (each thread issues its <= 8 partial loads at once). The host keeps the grid within one
// resident wave whenever S > 1, so the wait cannot deadlock; the last split out resets the counter.
// Debug timing (set_stop(n), test_attn.py; the server never sets it): 1 returns after Q staging, 2 after the main loop,
// 3 after the partial is written, 4 after the barrier, 5 after the combine weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int HD = 256, NQ = 6, TILE = 32, QS = HD + 8, PS = TILE + 8, SMAX = 64;

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
// four E4M3 bytes -> two fp16x2 (bytes 0,1 and 2,3), each = E4M3 * 2^-8 exactly
__device__ __forceinline__ void e4m3x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned x0 = __byte_perm(w, 0u, 0x1404), x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}
__device__ __forceinline__ void bar_pair(int id) { asm volatile("bar.sync %0, 64;" ::"r"(id) : "memory"); }
__device__ __forceinline__ float ex2(float x) { return exp2f(x); }

template <int NG>
struct SM {   // shared memory layout (bytes)
  static constexpr int Q = 0, QB = 8 * QS * 2;                     // Q rows [8][QS] fp16
  static constexpr int X = QB, XB = 2 * NG * 2 * 32 * 8 * 4;      // score exchange [2 parity][NG][2 warps][32][8]
  static constexpr int P = X + XB, PB = NG * 2 * 8 * PS * 2;      // P per warp [NG * 2][8][PS] fp16
  static constexpr int O = QB, OB = NG * NQ * HD * 4;             // merge [NG][6][256] fp32 (aliases X and P)
  static constexpr int ML = (P + PB > O + OB ? P + PB : O + OB);  // [NG][8][2] fp32
  static constexpr int W = ML + NG * 8 * 2 * 4;                    // CTA row stats [2][6] + combine [2][SMAX][6] fp32
  static constexpr int WB = (2 + 2 * SMAX) * NQ * 4;
  static constexpr int TOTAL = W + WB + 64;
};

template <int NG>
__global__ void __launch_bounds__(64 * NG)
k_attn(const __half* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
       const int* __restrict__ table, int tstride, const int* __restrict__ lens, __half* __restrict__ out,
       float* __restrict__ po, float* __restrict__ pml, int* __restrict__ cnt, int S, int min_tiles, int page,
       long long kb, long long kt, long long kh, long long vb, long long vt, long long vh, float sl2, float vsc,
       int stop) {
  using L_ = SM<NG>;
  extern __shared__ __align__(16) uint8_t sm[];
  __half* Qs = reinterpret_cast<__half*>(sm + L_::Q);
  float* Xs = reinterpret_cast<float*>(sm + L_::X);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P);
  float* Os = reinterpret_cast<float*>(sm + L_::O);
  float* MLs = reinterpret_cast<float*>(sm + L_::ML);
  float* Ws = reinterpret_cast<float*>(sm + L_::W);
  int* misc = reinterpret_cast<int*>(sm + L_::W + L_::WB);

  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z, Hkv = gridDim.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, pair = warp >> 1, wg = warp & 1;
  const int bh = b * Hkv + h;
  const int L = lens[b];
  __half* ob = out + ((size_t)b * Hkv * NQ + h * NQ) * HD;
  if (L <= 0) {
    if (s == 0)
      for (int i = tid; i < NQ * HD / 2; i += 64 * NG) reinterpret_cast<__half2*>(ob)[i] = __floats2half2_rn(0.f, 0.f);
    return;
  }
  const int T = (L + TILE - 1) / TILE;
  const int Sb = max(1, min(S, T / (NG * min_tiles)));
  if (s >= Sb) return;
  const int t0 = (int)((long long)T * s / Sb), t1 = (int)((long long)T * (s + 1) / Sb);

  const int qp = (lane >> 2) & 3, hi = (lane >> 4) & 1, l3 = lane & 3;
  const int arow = l3 + 4 * hi;                    // A row; B column (QK token inside the quadpair)
  const int r0 = (lane & 1) + 4 * hi, c0 = lane & 2;   // C rows r0, r0 + 2; columns c0, c0+1, c0+4, c0+5
  const int dim0 = 128 * wg;
  const int ktok = 8 * qp + arow;                  // this thread's token in a tile (QK)
  const int vdim = dim0 + 32 * qp + 16 * hi;       // this thread's 16 V bytes (PV)
  const int* tb = table + (size_t)b * tstride;
  // K / V of tile t into kr / vr. The loop reloads kr as soon as QK has consumed it and vr once PV has, so the next
  // tile's loads overlap this tile's softmax and PV (K) and the next QK (V) with no extra registers.
  uint4 kr[8], vr[8];
  // page bases of the current page, cached per thread: the table is read once per 1568-token page, not per tile (a
  // dependent table load before every tile's address would stall the warp ahead of its prefetch)
  int pcur = -1;
  const uint8_t *kpb = kc, *vpb = vc;
  auto setpage = [&](int t) {
    const int pi = t * TILE / page;
    if (pi != pcur) {
      pcur = pi;
      const long long pg = tb[pi];
      kpb = kc + pg * kb + (long long)h * kh + (long long)ktok * kt + dim0;
      vpb = vc + pg * vb + (long long)h * vh + (long long)l3 * vt + vdim;
    }
  };
  auto ldk = [&](int t) {
    setpage(t);
    const uint8_t* p = kpb + (long long)(t * TILE % page) * kt;
#pragma unroll
    for (int i = 0; i < 8; i++) kr[i] = ldg16(p + 16 * i);
  };
  auto ldv = [&](int t) {
    setpage(t);
    const uint8_t* p = vpb + (long long)(t * TILE % page) * vt;
#pragma unroll
    for (int i = 0; i < 8; i++) vr[i] = ldg16(p + (long long)(4 * i) * vt);
  };
  int t = t0 + pair;
  if (t < t1) { ldk(t); ldv(t); }

  // Q rows 0..5 of this KV head (rows 6, 7 zero)
  for (int i = tid; i < 8 * HD / 8; i += 64 * NG) {
    const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < NQ) v = *reinterpret_cast<const uint4*>(q + ((size_t)b * Hkv * NQ + h * NQ + r) * HD + c);
    *reinterpret_cast<uint4*>(Qs + r * QS + c) = v;
  }
  __syncthreads();
  if (stop == 1) return;                                     // (debug timing: launch + Q staging)
  const __half* qrow = Qs + arow * QS + dim0;
  __half* Pw = Ps + (pair * 2 + wg) * 8 * PS;

  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[j][e] = 0.f;
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;
  int it = 0;
  for (; t < t1; t += NG, it++) {
    const int tok0 = t * TILE;
    const bool more = t + NG < t1;
    float sa[8], sb[8];                            // two accumulation chains (even / odd 16-byte chunks)
#pragma unroll
    for (int e = 0; e < 8; e++) { sa[e] = 0.f; sb[e] = 0.f; }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const unsigned w[4] = {kr[i].x, kr[i].y, kr[i].z, kr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint2 a = *reinterpret_cast<const uint2*>(qrow + 4 * (4 * i + j));
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rc((i & 1) ? sb : sa, a.x, a.y, b0, b1);
      }
    }
    if (more) ldk(t + NG);
#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] += sb[e];
    // the pair's two dim halves -> full scores (double-buffered by iteration parity)
    float* xw = Xs + ((((it & 1) * NG + pair) * 2 + wg) * 32 + lane) * 8;
    float* xr = Xs + ((((it & 1) * NG + pair) * 2 + (wg ^ 1)) * 32 + lane) * 8;
    *reinterpret_cast<float4*>(xw) = make_float4(sa[0], sa[1], sa[2], sa[3]);
    *reinterpret_cast<float4*>(xw + 4) = make_float4(sa[4], sa[5], sa[6], sa[7]);
    if (NG == 1) __syncthreads(); else bar_pair(1 + pair);
    {
      const float4 u = *reinterpret_cast<const float4*>(xr), v = *reinterpret_cast<const float4*>(xr + 4);
      // a + b in warp order so both warps hold bit-identical scores
      if (wg == 0) { sa[0] += u.x; sa[1] += u.y; sa[2] += u.z; sa[3] += u.w; sa[4] += v.x; sa[5] += v.y; sa[6] += v.z; sa[7] += v.w; }
      else { sa[0] = u.x + sa[0]; sa[1] = u.y + sa[1]; sa[2] = u.z + sa[2]; sa[3] = u.w + sa[3];
             sa[4] = v.x + sa[4]; sa[5] = v.y + sa[5]; sa[6] = v.z + sa[6]; sa[7] = v.w + sa[7]; }
    }
    // scale (log2 domain) and mask tokens >= L
    const int cb = tok0 + 8 * qp + c0;
    float sv[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int col = cb + (e & 1) + (e & 4);
      sv[e] = col < L ? sa[e] * sl2 : -INFINITY;
    }
    float x0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), x2 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
      x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
    }
    const float n0 = fmaxf(m0, x0), n2 = fmaxf(m2, x2);
    const float a0 = m0 == -INFINITY ? 0.f : ex2(m0 - n0), a2 = m2 == -INFINITY ? 0.f : ex2(m2 - n2);
    float p[8];
#pragma unroll
    for (int e = 0; e < 8; e++) p[e] = ex2(sv[e] - ((e & 2) ? n2 : n0));   // -inf -> 0
    l0 = l0 * a0 + (p[0] + p[1]) + (p[4] + p[5]);
    l2 = l2 * a2 + (p[2] + p[3]) + (p[6] + p[7]);
    m0 = n0; m2 = n2;
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;
    // P -> this warp's buffer (A-fragment source)
    const int pc = 8 * qp + c0;
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(p[0], p[1]);
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(p[4], p[5]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(p[2], p[3]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(p[6], p[7]);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
      const unsigned w[4] = {vr[i].x, vr[i].y, vr[i].z, vr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rr(o[j], a.x, a.y, b0, b1);
      }
    }
    if (more) ldv(t + NG);
    __syncwarp();
  }
  if (stop == 2) { if (m0 == 12345.f) ob[0] = __float2half(o[0][0] + o[3][7]); return; }   // (debug: main loop)
  // row sums over the 8 lanes that share a row
#pragma unroll
  for (int d = 2; d <= 8; d <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, d);
    l2 += __shfl_xor_sync(0xffffffffu, l2, d);
  }
  const int SG = gridDim.x;                                  // partial slots per (b, h)
  float* pob = po + ((size_t)bh * SG + s) * NQ * HD;
  float* pmb = pml + ((size_t)bh * SG + s) * NQ * 2;
  if (NG == 1) {
    // one pair: its registers are the CTA result (rows r0, r0 + 2; dims dlo.. and dlo + 16..)
    if (Sb == 1) {
      const float i0 = vsc / l0, i2 = vsc / l2;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int sr = 0; sr < 2; sr++) {
          const int row = r0 + 2 * sr;
          if (row < NQ) {
            const float f = sr ? i2 : i0;
            const int dlo = 32 * qp + 4 * j + c0 + dim0;
            *reinterpret_cast<__half2*>(ob + row * HD + dlo) = __floats2half2_rn(o[j][2 * sr] * f, o[j][2 * sr + 1] * f);
            *reinterpret_cast<__half2*>(ob + row * HD + dlo + 16) = __floats2half2_rn(o[j][4 + 2 * sr] * f, o[j][5 + 2 * sr] * f);
          }
        }
      return;
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int row = r0 + 2 * sr;
        if (row < NQ) {
          const int dlo = 32 * qp + 4 * j + c0 + dim0;
          __stcg(reinterpret_cast<float2*>(pob + row * HD + dlo), make_float2(o[j][2 * sr], o[j][2 * sr + 1]));
          __stcg(reinterpret_cast<float2*>(pob + row * HD + dlo + 16), make_float2(o[j][4 + 2 * sr], o[j][5 + 2 * sr]));
        }
      }
    if (wg == 0 && (lane & 14) == 0) {
      __stcg(pmb + 2 * r0, m0); __stcg(pmb + 2 * r0 + 1, l0);
      if (r0 + 2 < NQ) { __stcg(pmb + 2 * (r0 + 2), m2); __stcg(pmb + 2 * (r0 + 2) + 1, l2); }
    }
  } else {
    // merge the NG pairs: stats, then scaled accumulators into Os[pair][6][256]
    __syncthreads();                                         // X / P (aliased by Os) are dead
    if (wg == 0 && (lane & 14) == 0) {                       // lanes 0, 1, 16, 17: rows r0 and r0 + 2
      MLs[(pair * 8 + r0) * 2] = m0; MLs[(pair * 8 + r0) * 2 + 1] = l0;
      MLs[(pair * 8 + r0 + 2) * 2] = m2; MLs[(pair * 8 + r0 + 2) * 2 + 1] = l2;
    }
    __syncthreads();
    float M0 = -INFINITY, M2 = -INFINITY;
#pragma unroll
    for (int g = 0; g < NG; g++) { M0 = fmaxf(M0, MLs[(g * 8 + r0) * 2]); M2 = fmaxf(M2, MLs[(g * 8 + r0 + 2) * 2]); }
    const float f0 = m0 == -INFINITY ? 0.f : ex2(m0 - M0), f2 = m2 == -INFINITY ? 0.f : ex2(m2 - M2);
    float* Op = Os + pair * NQ * HD;
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int row = r0 + 2 * sr;
        if (row < NQ) {
          const float f = sr ? f2 : f0;
          const int dlo = 32 * qp + 4 * j + c0 + dim0;
          *reinterpret_cast<float2*>(Op + row * HD + dlo) = make_float2(o[j][2 * sr] * f, o[j][2 * sr + 1] * f);
          *reinterpret_cast<float2*>(Op + row * HD + dlo + 16) = make_float2(o[j][4 + 2 * sr] * f, o[j][5 + 2 * sr] * f);
        }
      }
    if (tid < NQ) {                                          // CTA row max and row sum
      float M = -INFINITY, Lt = 0.f;
      for (int g = 0; g < NG; g++) M = fmaxf(M, MLs[(g * 8 + tid) * 2]);
      for (int g = 0; g < NG; g++) {
        const float mg = MLs[(g * 8 + tid) * 2];
        if (mg != -INFINITY) Lt += MLs[(g * 8 + tid) * 2 + 1] * ex2(mg - M);
      }
      Ws[tid] = M; Ws[NQ + tid] = Lt;
    }
    __syncthreads();
    for (int i = tid; i < NQ * HD / 4; i += 64 * NG) {
      const int row = i / (HD / 4), d = (i % (HD / 4)) * 4;
      float4 a = *reinterpret_cast<const float4*>(Os + row * HD + d);
#pragma unroll
      for (int g = 1; g < NG; g++) {
        const float4 c = *reinterpret_cast<const float4*>(Os + (g * NQ + row) * HD + d);
        a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
      }
      if (Sb == 1) {
        const float k = vsc / Ws[NQ + row];
        __half2* dst = reinterpret_cast<__half2*>(ob + row * HD + d);
        dst[0] = __floats2half2_rn(a.x * k, a.y * k);
        dst[1] = __floats2half2_rn(a.z * k, a.w * k);
      } else {
        __stcg(reinterpret_cast<float4*>(pob + row * HD + d), a);
      }
    }
    if (Sb == 1) return;
    if (tid < NQ) { __stcg(pmb + 2 * tid, Ws[tid]); __stcg(pmb + 2 * tid + 1, Ws[NQ + tid]); }
  }
  // ---- combine (Sb > 1): every split waits for the others, then reduces slice s of the 6 x 256 outputs. The grid is
  // one resident wave whenever Sb > 1, so the wait cannot deadlock. Item = (output float4 e, block g of <= 8 splits).
  constexpr int NTH = 64 * NG;
  const int n4 = NQ * HD / 4, lo = n4 * s / Sb, E = n4 * (s + 1) / Sb - lo;
  const int G = (Sb + 7) / 8;                                // blocks of <= 8 splits per output float4
  float* Acc = Os;                                           // [E][4] fp32 (Os is dead)
  float* Mk = Ws + 2 * NQ;                                   // [SMAX][6] split row maxima
  float* Lk = Mk + SMAX * NQ;                                // [SMAX][6] split row sums
  float* Rs = Ws;                                            // [2][6] row max, v_scale / row sum
  __syncthreads();                                           // Os / X / P (aliased by Acc) are done with
  for (int i = tid; i < E * 4; i += NTH) Acc[i] = 0.f;
  __threadfence();
  __syncthreads();
  if (stop == 3) return;                                     // (debug: partial written)
  if (tid == 0) {
    atomicAdd(cnt + bh, 1);
    while (*reinterpret_cast<volatile int*>(cnt + bh) < Sb) __nanosleep(100);
    __threadfence();
  }
  __syncthreads();
  if (stop == 4) return;                                     // (debug: barrier passed)
  // split statistics -> row max M_r and v_scale / L_r
  const float* pm = pml + (size_t)bh * SG * NQ * 2;
  for (int i = tid; i < Sb * NQ; i += NTH) { Mk[i] = __ldcg(pm + 2 * i); Lk[i] = __ldcg(pm + 2 * i + 1); }
  __syncthreads();
  if (tid < 64) {                                            // two whole warps (shuffles need every lane): 8 per row
    const int r = tid >> 3, j = tid & 7;
    float M = -INFINITY;
    if (r < NQ) for (int k = j; k < Sb; k += 8) M = fmaxf(M, Mk[k * NQ + r]);
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));
    float Lt = 0.f;
    if (r < NQ) for (int k = j; k < Sb; k += 8) Lt += Lk[k * NQ + r] * ex2(Mk[k * NQ + r] - M);
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) Lt += __shfl_xor_sync(0xffffffffu, Lt, d);
    if (j == 0 && r < NQ) { Rs[r] = M; Rs[NQ + r] = vsc / Lt; }
  }
  __syncthreads();
  if (stop == 5) return;                                     // (debug: weights)
  // items (output float4 e, block g of <= 8 splits): each issues its <= 8 loads before accumulating
  for (int idx = tid; idx < E * G; idx += NTH) {
    const int e = idx % E, g = idx / E, el = lo + e, row = el / (HD / 4), dd = (el % (HD / 4)) * 4;
    const int k0 = 8 * g, nk = min(8, Sb - k0);
    const float* src = po + ((size_t)bh * SG * NQ + row) * HD + dd;
    float4 c[8];
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (u < nk) c[u] = __ldcg(reinterpret_cast<const float4*>(src + (size_t)(k0 + u) * NQ * HD));
    const float M = Rs[row], inv = Rs[NQ + row];
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; u++)
      if (u < nk) {
        const float wk = ex2(Mk[(k0 + u) * NQ + row] - M) * inv;
        a.x += wk * c[u].x; a.y += wk * c[u].y; a.z += wk * c[u].z; a.w += wk * c[u].w;
      }
    if (G == 1) { Acc[4 * e] = a.x; Acc[4 * e + 1] = a.y; Acc[4 * e + 2] = a.z; Acc[4 * e + 3] = a.w; }
    else { atomicAdd(Acc + 4 * e, a.x); atomicAdd(Acc + 4 * e + 1, a.y); atomicAdd(Acc + 4 * e + 2, a.z); atomicAdd(Acc + 4 * e + 3, a.w); }
  }
  __syncthreads();
  for (int i = tid; i < E; i += NTH) {
    const int eli = lo + i, rw = eli / (HD / 4), d = (eli % (HD / 4)) * 4;
    __half2* dst = reinterpret_cast<__half2*>(ob + rw * HD + d);
    dst[0] = __floats2half2_rn(Acc[4 * i], Acc[4 * i + 1]);
    dst[1] = __floats2half2_rn(Acc[4 * i + 2], Acc[4 * i + 3]);
  }
  if (tid == 0 && atomicAdd(cnt + bh, 1) == 2 * Sb - 1) atomicExch(cnt + bh, 0);   // last one out resets
}

int STOP = 0;   // debug timing (set_stop): 1 after Q staging, 2 after the main loop, 3 after the partial

template <int NG>
int capacity() {
  static int cap = 0;
  if (!cap) {
    int occ = 0, dev = 0, sms = 0;
    TORCH_CHECK(cudaFuncSetAttribute(k_attn<NG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM<NG>::TOTAL) == cudaSuccess);
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_attn<NG>, 64 * NG, SM<NG>::TOTAL) == cudaSuccess);
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cap = occ * sms;
  }
  return cap;
}

template <int NG>
void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& table,
            const torch::Tensor& lens, torch::Tensor& out, torch::Tensor& po, torch::Tensor& pml, torch::Tensor& cnt,
            int S, int min_tiles, float sl2, float vsc, cudaStream_t st) {
  const int B = q.size(0), Hkv = k.size(2);
  S = std::min(S, SMAX);
  const int cap = capacity<NG>();
  if ((long long)B * Hkv * S > cap) S = std::max(1, cap / (B * Hkv));   // a spinning combine needs one resident wave
  dim3 grid(S, Hkv, B);
  k_attn<NG><<<grid, 64 * NG, SM<NG>::TOTAL, st>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), k.data_ptr<uint8_t>(), v.data_ptr<uint8_t>(),
      table.data_ptr<int>(), (int)table.stride(0), lens.data_ptr<int>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      po.data_ptr<float>(), pml.data_ptr<float>(), cnt.data_ptr<int>(), S, min_tiles, (int)k.size(1), k.stride(0),
      k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), sl2, vsc, STOP);
}
}  // namespace

void set_stop(int64_t s) { STOP = (int)s; }
int64_t attn_capacity(int64_t ng) { return ng == 1 ? capacity<1>() : ng == 2 ? capacity<2>() : capacity<4>(); }
// partial slots a launch uses: B Hkv S, with S cut to one resident wave (none when S ends up 1)
int64_t attn_slots(int64_t bh, int64_t S, int64_t ng) {
  long long s = std::min<long long>(S, SMAX);
  const long long cap = attn_capacity(ng);
  if (bh * s > cap) s = std::max<long long>(1, cap / bh);
  return s > 1 ? bh * s : 0;
}

// q [B][6 Hkv][256] fp16; k, v [blocks][page][Hkv][256] uint8 E4M3 (any strides, last dim dense; page % 32 == 0);
// table [B][W] int32; lens [B] int32 (tokens incl. the current one); out like q. po [>= attn_slots 6 256] fp32,
// pml [>= attn_slots 12] fp32, cnt [>= B Hkv] int32 zeroed once (the kernel leaves it zeroed).
// ng: pairs of warps per CTA (1, 2, 4); S: splits per sequence (reduced at run time for short sequences, and to
// one resident wave); min_tiles: minimum 32-token tiles per pair.
void attn_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor table, torch::Tensor lens,
                 torch::Tensor out, torch::Tensor po, torch::Tensor pml, torch::Tensor cnt, double scale,
                 double k_scale, double v_scale, int64_t ng, int64_t S, int64_t min_tiles) {
  TORCH_CHECK(q.dtype() == torch::kHalf && q.is_contiguous() && q.dim() == 3 && q.size(2) == HD);
  TORCH_CHECK(k.dtype() == torch::kUInt8 && v.dtype() == torch::kUInt8 && k.dim() == 4 && v.dim() == 4);
  TORCH_CHECK(k.size(3) == HD && k.stride(3) == 1 && v.stride(3) == 1 && k.sizes() == v.sizes());
  TORCH_CHECK(q.size(1) == NQ * k.size(2), "6 query heads per KV head");
  TORCH_CHECK(k.size(1) % TILE == 0, "page size must be a multiple of 32");
  TORCH_CHECK(table.dtype() == torch::kInt && table.dim() == 2 && table.stride(1) == 1 && table.size(0) == q.size(0));
  TORCH_CHECK(lens.dtype() == torch::kInt && lens.is_contiguous() && lens.size(0) == q.size(0));
  TORCH_CHECK(out.sizes() == q.sizes() && out.is_contiguous() && out.dtype() == torch::kHalf);
  // po / pml: at least min(cap, B Hkv S) partial slots (the launch keeps B Hkv S within one resident wave when S > 1)
  TORCH_CHECK(po.numel() >= attn_slots(q.size(0) * k.size(2), S, ng) * NQ * HD &&
              pml.numel() >= attn_slots(q.size(0) * k.size(2), S, ng) * NQ * 2, "q27 attn: workspace too small");
  TORCH_CHECK(cnt.numel() >= q.size(0) * k.size(2) && cnt.dtype() == torch::kInt);
  const at::cuda::CUDAGuard g(q.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const float sl2 = (float)(scale * k_scale * 256.0 * 1.4426950408889634), vsc = (float)(v_scale * 256.0);
  if (ng == 1) launch<1>(q, k, v, table, lens, out, po, pml, cnt, (int)S, (int)min_tiles, sl2, vsc, st);
  else if (ng == 2) launch<2>(q, k, v, table, lens, out, po, pml, cnt, (int)S, (int)min_tiles, sl2, vsc, st);
  else if (ng == 4) launch<4>(q, k, v, table, lens, out, po, pml, cnt, (int)S, (int)min_tiles, sl2, vsc, st);
  else TORCH_CHECK(false, "ng must be 1, 2 or 4");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_decode", &attn_decode, "split-KV E4M3 decode attention, 6 query heads per KV head, in-kernel combine");
  m.def("attn_capacity", &attn_capacity, "resident CTAs per wave for a CTA of ng pairs");
  m.def("attn_slots", &attn_slots, "partial slots used by a launch (B*Hkv, S, ng)");
  m.def("set_stop", &set_stop, "debug timing: kernels return after phase n (0 = off)");
  m.attr("SMAX") = (int64_t)SMAX;
}
