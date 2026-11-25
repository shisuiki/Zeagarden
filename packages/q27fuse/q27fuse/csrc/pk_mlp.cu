// pk_mlp.cu (track prefill) - packed prefill MLP for Qwen3.8-27B NVFP4 on V100: three useful products per two
// tensor-core products (3-in-2 FP4 chain, ../../../../tmp/REPORT.md §7/§14, ../../../REPORT.md §10.4).
//
// Format ("ours", pf_format.py): W[n,k] = alpha'[n] * 2^rel[t(n), k/16] * E2M1(c[n,k]); rel in 0..3 shared by the 3 rows
// of a chain triple per 16-k block; activations per-token INT8 (integers held as fp16). The chain rows
//   P1 = (2 c1 + 331 c2) 2^rel,  P2 = (2 c2 + 331 c3) 2^rel     (c = 2 E2M1 in -12..12; FP16-exact, |P| <= 15,984)
// give n1 = 2 S1 + 331 S2, n2 = 2 S2 + 331 S3 with S = sum_k xq c 2^rel (integers; products xq P < 2^21, HMMA-exact):
// FP32 accumulation is exact while |n| < 2^24, and S1 is recovered modulo 331^2 = 109,561 (window |S1| < 54,780) by unwrap331, once
// after the K loop (PT = K) or every PT terms. y = S * sx[row] * ae[slot], ae = alpha' / 2.
//
// Pieces (one forward of one layer):
//   k_quant   per-token INT8 of x [M][K] fp16 -> xq (fp16 integers), sx
//   k_conv    decode-layout NVFP4 (mlp_fp4.cu prepack: 32-column tiles x 32-k chunks of 576 B) + optional 1-bit
//             selection plane -> packed scratch [64 * groups][K] fp16 (96 decode rows = 3 tiles = 1 group = 64 rows)
//   k_pk      HMMA.884 GEMM on (xq, packed scratch), chain decode in the epilogue; EPI 0: fp16 out [M][Nout];
//             EPI 1: SiLU(gate) * up -> h fp16 [M][I] (gate/up slots of a group: 48 gate then 48 up)
//
// [prefill2] FOLD mode (k_pk template FP > 0, gemm cfg 30..33): the exact periodic decode with THREE registers per column
// pair instead of five, so the 32x96 warp tile (128x192 CTA) fits: every FP K-tiles the two FP32 accumulators are reduced
// by multiples of Q = 32 * 331^2 (exact; the residues the unwrap reads are unchanged) and the multiples q1, q2 go into
// one int32 E = S1 + 2^20 q1 + 2^26 q2; every FU K-tiles S1 (E's low 20 bits) is advanced by the centred step mod 331^2.
// Exact while the growth of |n| inside a reduction period is < 2^24 - Q / 2 (15.0M), |dS1| per unwrap period < 54,780,
// |S1| < 2^19 and |n| < 31.5 Q (110M) over the full K. Epilogue: n = acc + q Q in int32, S2 = (n1 - 2 S1) / 331,
// S3 = (n2 - 2 S2) / 331.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cublas_v2.h>   // [prefill3]

namespace {
constexpr int CHUNK = 576;

__device__ __forceinline__ unsigned rotl(unsigned w, int r) { return r ? __funnelshift_l(w, w, r) : w; }
__host__ __device__ constexpr int RA(int p) { return p * 3; }
__host__ __device__ constexpr int RB_(int p) { return 3 - p; }
template <int P>
__device__ __forceinline__ unsigned dq(unsigned w) {               // fp16x2 = E2M1 * 2^-14 for code pair P (mlp_fp4.cu)
  return (rotl(w, RA(P)) & 0x0E000E00u) | (rotl(w, RB_(P)) & 0x80008000u);
}
__device__ __forceinline__ unsigned hmul2u(unsigned a, unsigned b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ unsigned hfma2u(unsigned a, unsigned b, unsigned c) {
  __half2 r = __hfma2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b), *reinterpret_cast<__half2*>(&c));
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ldg4(const void* p) {
  unsigned v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ldg2(const void* p) {
  unsigned short v;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float e4m3f(unsigned e) {               // E4M3 byte (positive) -> float, exact
  return __half2float(__ushort_as_half((unsigned short)((e & 0x7Fu) << 7))) * 256.f;
}

// ------------------------------------------------------------------------------------------- per-token INT8 rows
// one warp per row; xq = rint(x * 127 / amax) (exact integers in fp16), sx = amax / 127; the row is read once and held
// in registers (VPL 16-B vectors per lane, K = 256 * VPL)
template <int VPL>
__global__ void __launch_bounds__(256) k_quant(const __half* __restrict__ X, __half* __restrict__ Xq, float* __restrict__ sx,
                                               int M, int K, float qm,      // [prefill3] qm = integer range (127: INT8)
                                               const unsigned* __restrict__ U, unsigned* __restrict__ R) {   // [prefill3] Job 4
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * K);
  uint4* qr = reinterpret_cast<uint4*>(Xq + (size_t)row * K);
  uint4 u[VPL];
#pragma unroll
  for (int v = 0; v < VPL; v++) u[v] = __ldcs(xr + v * 32 + lane);
  float am = 0.f;
#pragma unroll
  for (int v = 0; v < VPL; v++) {
    const __half2* h = reinterpret_cast<const __half2*>(&u[v]);
#pragma unroll
    for (int e = 0; e < 4; e++) { const float2 f = __half22float2(h[e]); am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y))); }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  const float inv = am > 0.f ? qm / am : 0.f;
  unsigned ref = 0u;                                                // [prefill3] Job 4: sum_k xq[k] U[k] mod 2^32
#pragma unroll
  for (int v = 0; v < VPL; v++) {
    __half2* h = reinterpret_cast<__half2*>(&u[v]);
    float qv[8];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float2 f = __half22float2(h[e]);
      qv[2 * e] = rintf(f.x * inv); qv[2 * e + 1] = rintf(f.y * inv);
      h[e] = __floats2half2_rn(qv[2 * e], qv[2 * e + 1]);
    }
    qr[v * 32 + lane] = u[v];
    if (U != nullptr) {
      const uint4* up = reinterpret_cast<const uint4*>(U + (size_t)(v * 32 + lane) * 8);
      const uint4 a = __ldg(up), b = __ldg(up + 1);
      const unsigned uw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int e = 0; e < 8; e++) ref += (unsigned)__float2int_rn(qv[e]) * uw[e];
    }
  }
  if (U != nullptr) {
#pragma unroll
    for (int o = 16; o; o >>= 1) ref += __shfl_xor_sync(0xffffffffu, ref, o);
    if (lane == 0) R[row] = ref;
  }
  if (lane == 0) sx[row] = am / qm;
}

// [prefill3] Job 4: reference checksum of given integer rows (fp16 integers) - for rows quantised elsewhere (pf_arn)
template <int VPL>
__global__ void __launch_bounds__(256) k_rowdot(const __half* __restrict__ Xq, const unsigned* __restrict__ U,
                                                unsigned* __restrict__ R, int M, int K) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const uint4* xr = reinterpret_cast<const uint4*>(Xq + (size_t)row * K);
  unsigned ref = 0u;
#pragma unroll
  for (int v = 0; v < VPL; v++) {
    const uint4 x = __ldcs(xr + v * 32 + lane);
    const __half2* h = reinterpret_cast<const __half2*>(&x);
    const uint4* up = reinterpret_cast<const uint4*>(U + (size_t)(v * 32 + lane) * 8);
    const uint4 a = __ldg(up), b = __ldg(up + 1);
    const unsigned uw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float2 f = __half22float2(h[e]);
      ref += (unsigned)__float2int_rn(f.x) * uw[2 * e] + (unsigned)__float2int_rn(f.y) * uw[2 * e + 1];
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ref += __shfl_xor_sync(0xffffffffu, ref, o);
  if (lane == 0) R[row] = ref;
}

// [prefill3] Job 4: compare the GEMM's checksums with the references; zero the GEMM side for the next launch.
// flag[0] |= 1 on any mismatch, flag[1] += mismatching rows
__global__ void __launch_bounds__(1024) k_chk(unsigned* __restrict__ C, const unsigned* __restrict__ R, int M,
                                              int* __restrict__ flag) {
  int bad = 0;
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < M; r += blockDim.x * gridDim.x) {
    bad += C[r] != R[r];
    C[r] = 0u;
  }
  bad = __syncthreads_count(bad);
  if (threadIdx.x == 0 && bad) { atomicOr(flag, 1); atomicAdd(flag + 1, bad); }
}

// ------------------------------------------------------------------------------------------- conversion pass
// grid (padded groups, ceil(NC / CPB)), 96 threads: warp i decodes tile 3G+i of the decode layout (lane -> its column,
// 32 k per chunk) into shared memory as E2M1 * 2^-14 fp16 (the selection-bit rounding), then all threads write the
// group's 64 packed rows for the CPB chunks (64 * CPB contiguous bytes per row). PL: PLANE 1: per (group, chunk) 400 B =
// u32 sel[3 tiles][32 lanes] (bit j = pair j's low half (k even), bit 16+j its high half; pair j = 4 * word + pair-in-
// word, i.e. k = 2j, 2j+1 of the lane's 32) + u32 rel[4] (triple t's 2-bit rel of block b at bit 4(t%8)+2b of word t/8);
// PLANE 0: only the 16-B rel record, the decoder rounds to nearest. IA: per (group, slot) f32 = 1 / (g alpha').
// Codes above E2M1 6 (the encoder's clipped blocks) saturate at 6; the encoder keeps |y16| below 30 * 2^-14.
template <int PLANE, int GATED, int CPB>
__global__ void __launch_bounds__(96) k_conv(const uint8_t* __restrict__ W, const uint8_t* __restrict__ PL,
                                             const float* __restrict__ IA, __half* __restrict__ out, int ntile,
                                             int ngroups, int K) {
  constexpr int RS = CPB * 16 + 4;                                  // u32 per slot row (padded)
  constexpr int RECB = PLANE ? 400 : 16, RELOFF = PLANE ? 384 : 0;
  __shared__ __align__(16) uint32_t dec[96 * RS];
  __shared__ uint32_t relw[CPB][4];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
  const int G = blockIdx.x, NC = K / 32, c0 = blockIdx.y * CPB;
  const int nch = min(CPB, NC - c0);
  const int p = 3 * G + wi;
  const bool real = G < ngroups && p < ntile;
  const int lc = 8 * ((lane >> 2) & 3) + (lane & 3) + 4 * (lane >> 4);
  const int slot = GATED ? (lc < 16 ? 16 * wi + lc : 48 + 16 * wi + lc - 16) : 32 * wi + lc;
  const int t = slot / 3;
  const float ia = real ? IA[G * 96 + slot] : 0.f;
  if (threadIdx.x < nch * 4) {
    const int cc = threadIdx.x >> 2, q = threadIdx.x & 3;
    relw[cc][q] = G < ngroups ? *reinterpret_cast<const uint32_t*>(PL + ((size_t)G * NC + c0 + cc) * RECB + RELOFF + 4 * q) : 0u;
  }
  uint4 w[CPB];
  unsigned sv[CPB], sel[CPB];
#pragma unroll
  for (int cc = 0; cc < CPB; cc++) {                                 // all loads of the CTA's chunks in flight at once
    w[cc] = make_uint4(0, 0, 0, 0); sv[cc] = 0; sel[cc] = 0;
    if (real && cc < nch) {
      const uint8_t* src = W + ((size_t)p * NC + c0 + cc) * CHUNK;
      w[cc] = ldg16(src + lane * 16);
      sv[cc] = ldg2(src + 512 + 2 * lane);
      if (PLANE) sel[cc] = ldg4(PL + ((size_t)G * NC + c0 + cc) * RECB + (wi * 32 + lane) * 4);
    }
  }
  __syncthreads();
#pragma unroll
  for (int cc = 0; cc < CPB; cc++) {
    if (cc >= nch) break;
    const unsigned rw = relw[cc][t >> 3] >> (4 * (t & 7));
    const int rel0 = rw & 3, rel1 = (rw >> 2) & 3;
    const float s0 = e4m3f(sv[cc] & 0xFFu), s1 = e4m3f(sv[cc] >> 8);
    const __half h0 = __float2half_rn(__fmul_rn(__fmul_rn(s0, ia), __int_as_float((127 - rel0) << 23)));
    const __half h1 = __float2half_rn(__fmul_rn(__fmul_rn(s1, ia), __int_as_float((127 - rel1) << 23)));
    const unsigned R0 = (unsigned)__half_as_ushort(h0) * 0x00010001u, R1 = (unsigned)__half_as_ushort(h1) * 0x00010001u;
    const unsigned wd[4] = {w[cc].x, w[cc].y, w[cc].z, w[cc].w};
    uint32_t o[16];
#pragma unroll
    for (int i4 = 0; i4 < 4; i4++) {
      const unsigned r2 = i4 < 2 ? R0 : R1;
      const unsigned v[4] = {dq<0>(wd[i4]), dq<1>(wd[i4]), dq<2>(wd[i4]), dq<3>(wd[i4])};
#pragma unroll
      for (int pp = 0; pp < 4; pp++) {
        const unsigned y = hmul2u(v[pp], r2);
        const unsigned bias = PLANE ? ((sel[cc] >> (4 * i4 + pp)) & 0x00010001u) * 0x1FFu : 0x01000100u;
        const unsigned c = (y + bias) & 0xFE00FE00u;
        const unsigned sm = ((c >> 12) & 0x00010001u) * 0x7FFFu;    // E2M1 > 6 (exponent field 4): saturate at 6
        o[4 * i4 + pp] = hmul2u((c & ~sm) | (sm & 0x0E000E00u), 0x74007400u);   // x 2^14: the E2M1 value, exact
      }
    }
    uint4* d = reinterpret_cast<uint4*>(dec + slot * RS + cc * 16);
    d[0] = make_uint4(o[0], o[1], o[2], o[3]); d[1] = make_uint4(o[4], o[5], o[6], o[7]);
    d[2] = make_uint4(o[8], o[9], o[10], o[11]); d[3] = make_uint4(o[12], o[13], o[14], o[15]);
  }
  __syncthreads();
  // pack: item = (packed row pr, 16-B vector v of its 64 * CPB bytes)
  const int VPR = 4 * nch;
  for (int item = threadIdx.x; item < 64 * VPR; item += 96) {
    const int pr = item / VPR, v = item % VPR;
    const int tt = pr >> 1, a = 3 * tt + (pr & 1);
    const int j0 = 4 * v, cc = j0 >> 4, bb = (j0 & 15) >> 3;
    const int rel = (relw[cc][tt >> 3] >> (4 * (tt & 7) + 2 * bb)) & 3;
    const unsigned K1 = ((unsigned)(17 + rel) << 10) * 0x00010001u;                 // 4 * 2^rel: 2 c 2^rel from the E2M1 value
    const unsigned K2 = (((unsigned)(24 + rel) << 10) | 0x12Cu) * 0x00010001u;        // 662 * 2^rel = 331 * 2 * 2^rel
    const uint4 A = *reinterpret_cast<const uint4*>(dec + a * RS + j0);
    const uint4 B = *reinterpret_cast<const uint4*>(dec + (a + 1) * RS + j0);
    uint4 P;
    P.x = hfma2u(B.x, K2, hmul2u(A.x, K1)); P.y = hfma2u(B.y, K2, hmul2u(A.y, K1));
    P.z = hfma2u(B.z, K2, hmul2u(A.z, K1)); P.w = hfma2u(B.w, K2, hmul2u(A.w, K1));
    *reinterpret_cast<uint4*>(out + (size_t)(64 * G + pr) * K + c0 * 32 + 2 * j0) = P;
  }
}

// ------------------------------------------------------------------------------------------- packed GEMM
__device__ __forceinline__ void mma884(float* d, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}
// FP4 chain (2, 331) unwrap (../../../bench/w4a8.cu): the S1 congruent to (n1, n2) within m/2 of the previous S1
__device__ __forceinline__ float unwrap331(float n1, float n2, float S1) {
  const float m = 109561.f, im = 1.f / 109561.f, MAG = 12582912.f;
  const float a = n1 + n1;
  const float q1 = fmaf(a, im, MAG) - MAG, ar = fmaf(-m, q1, a);
  const float q2 = fmaf(n2, 1.f / 331.f, MAG) - MAG, n2r = fmaf(-331.f, q2, n2);
  const float wf = fmaf(-331.f, n2r, ar);
  const int wi = __float_as_int(wf + MAG) - 0x4B400000;
  const float kf = __int_as_float(((-wi) & 3) + 0x4B400000) - MAG;
  const float s1 = fmaf(kf, m, wf) * 0.25f;
  const float d = s1 - S1, qd = fmaf(d, im, MAG) - MAG;
  return S1 + fmaf(-m, qd, d);
}

// [prefill2] FOLD helpers (see the header)
namespace pkf {
constexpr float MAG = 12582912.f;
__device__ __forceinline__ float step331(float n1, float n2, float S1) {   // S1_now - S1_prev, centred mod 331^2
  const float m = 109561.f, im = 1.f / 109561.f;
  const float a = n1 + n1;
  const float ar = fmaf(-m, fmaf(a, im, MAG) - MAG, a);
  const float n2r = fmaf(-331.f, fmaf(n2, 1.f / 331.f, MAG) - MAG, n2);
  const float wf = fmaf(-331.f, n2r, ar);
  const int kfi = (-__float_as_int(wf + MAG)) & 3;
  const float kf = __int_as_float(kfi + 0x4B400000) - MAG;
  const float dd = fmaf(fmaf(kf, m, wf), 0.25f, -S1);
  return fmaf(-m, fmaf(dd, im, MAG) - MAG, dd);
}
// QM: Q = QM * 331^2 (32: residual <= 1.75M, |n| < 110M; 64: residual <= 3.5M, |n| < 221M)
template <int QM>
__device__ __forceinline__ void reduce(float& a, float& b, int& E) {
  constexpr float Q = (float)(QM * 109561), IQ = 1.f / (float)(QM * 109561);
  const float t1 = fmaf(a, IQ, MAG), t2 = fmaf(b, IQ, MAG);
  a = fmaf(-Q, t1 - MAG, a);
  b = fmaf(-Q, t2 - MAG, b);
  E += (int)(((unsigned)__float_as_int(t1) << 20) + ((unsigned)__float_as_int(t2) << 26));
}
__device__ __forceinline__ void unwrap(float a, float b, int& E) {
  const int S1i = (E << 12) >> 12;
  const float d = step331(a, b, __int_as_float(S1i + 0x4B400000) - MAG);
  E += __float_as_int(d + MAG) - 0x4B400000;
}
template <int QM>
__device__ __forceinline__ void final3(float a, float b, int E, float& S1, float& S2, float& S3) {
  reduce<QM>(a, b, E);
  unwrap(a, b, E);
  const int s1 = (E << 12) >> 12, r = (E - s1) >> 20, q1 = (r << 26) >> 26, q2 = (r - q1) >> 6;
  const int n1 = __float2int_rn(a) + q1 * (QM * 109561), n2 = __float2int_rn(b) + q2 * (QM * 109561);
  const int s2 = (n1 - 2 * s1) / 331, s3 = (n2 - 2 * s2) / 331;
  S1 = (float)s1; S2 = (float)s2; S3 = (float)s3;
}
}  // namespace pkf

// the same with a zero reference (a period that restarted from zero): S1 in (-m/2, m/2)
__device__ __forceinline__ float unwrap331_0(float n1, float n2) {
  const float m = 109561.f, im = 1.f / 109561.f, MAG = 12582912.f;
  const float a = n1 + n1;
  const float q1 = fmaf(a, im, MAG) - MAG, ar = fmaf(-m, q1, a);
  const float q2 = fmaf(n2, 1.f / 331.f, MAG) - MAG, n2r = fmaf(-331.f, q2, n2);
  const float wf = fmaf(-331.f, n2r, ar);
  const int wi = __float_as_int(wf + MAG) - 0x4B400000;
  const float kf = __int_as_float(((-wi) & 3) + 0x4B400000) - MAG;
  const float s1 = fmaf(kf, m, wf) * 0.25f;
  return fmaf(-m, fmaf(s1, im, MAG) - MAG, s1);
}

// RB: 8-row blocks per warp, CB: 32-column blocks per warp; WM x WN warps; K tile 64; PT: unwrap period (>= K: once).
// A: xq [M][K] fp16 integers, W: packed [Ncols][K]; sx [M]; ae [slots = 3 Ncols / 2] (alpha' / 2, slot order).
// EPI 0: out [M][Nout] fp16 (slot = output channel). EPI 1: tile -> h [M][Nout] = silu(gate) * up, Nout = I, group G of
// the tile = slots 96G..96G+95 = 48 gate (channels 48G + i) then 48 up.
template <int RB, int CB, int WM, int WN, int PT, int EPI, int SPR, int RST = 0, int RSTMODE = 0, int FP = 0, int FU = 0, int QM = 32, int CHK = 0>
__global__ void __launch_bounds__(WM * WN * 32, 1)
k_pk(const __half* __restrict__ A, const __half* __restrict__ W, __half* __restrict__ out, const float* __restrict__ sx,
     const float* __restrict__ ae, int M, int Ncols, int K, int Nout, int zmask,
     const unsigned* __restrict__ chkv, unsigned* __restrict__ chkc) {     // [prefill3] Job 4 checksum (CHK)
  constexpr int KTT = 64, BM = WM * RB * 8, BN = WN * CB * 32, NT = WM * WN * 32, NB = RB * CB, LD = KTT + 8, NK8 = KTT / 8;
  constexpr bool W8 = PT < 1 << 20;
  extern __shared__ __align__(16) __half smem[];
  auto sA = [&](int b, int r, int k) { return smem + ((size_t)b * (BM + BN) + r) * LD + k; };
  auto sB = [&](int b, int r, int k) { return smem + ((size_t)b * (BM + BN) + BM + r) * LD + k; };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp % WM, wn = warp / WM;
  const int npm = (M + BM - 1) / BM, npn = Ncols / BN, pid = blockIdx.x, GM = 8;
  const int gid = pid / (GM * npn), first = gid * GM, gsz = min(npm - first, GM);
  const int pm = first + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
  const int m0 = pm * BM, n0 = pn * BN;
  constexpr int CA = BM * 4 / NT, CBn = BN * 4 / NT;
  static_assert(CA >= 1 && CBn >= 1 && BM * 4 % NT == 0 && BN * 4 % NT == 0, "tile vs threads");
  auto cmap = [&](int c, int p, int rows, int& r, int& kc) { r = p * (rows / 2) + (c >> 3); kc = (c & 7) * 8; };
  uint4 ra[CA], rb[CBn];
  auto ldg = [&](int kt0, int p) {
#pragma unroll
    for (int i = 0; i < CA; i++) { int r, kc; cmap(tid + i * NT, p, BM, r, kc);
      ra[i] = __ldg(reinterpret_cast<const uint4*>(A + (size_t)min(m0 + r, M - 1) * K + kt0 + kc)); }
#pragma unroll
    for (int i = 0; i < CBn; i++) { int r, kc; cmap(tid + i * NT, p, BN, r, kc);
      rb[i] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(n0 + r) * K + kt0 + kc)); }
  };
  auto sts = [&](int b, int p) {
#pragma unroll
    for (int i = 0; i < CA; i++) { int r, kc; cmap(tid + i * NT, p, BM, r, kc); *reinterpret_cast<uint4*>(sA(b, r, kc)) = ra[i]; }
#pragma unroll
    for (int i = 0; i < CBn; i++) { int r, kc; cmap(tid + i * NT, p, BN, r, kc); *reinterpret_cast<uint4*>(sB(b, r, kc)) = rb[i]; }
  };
  const int qp = (lane >> 2) & 3, arow = (lane & 3) + ((lane >> 4) << 2);
  const int brow = wn * CB * 32 + qp * 4 + (lane & 3) + ((lane >> 4) << 4);
  float acc[NB][8];
#pragma unroll
  for (int i = 0; i < NB; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
  float S1f[W8 ? NB * 4 : 1];
#pragma unroll
  for (int i = 0; i < (W8 ? NB * 4 : 1); i++) S1f[i] = 0.f;
  // RST: exact periodic decode - every RST K tiles the period's lane sums (|dS1| < 54,780, |2^11 dn| < 2^24) are
  // decoded into running totals T (exact integers in fp32) and the accumulators restart from 0.
  static_assert(RSTMODE != 3 || (NB * 4) % (RST ? RST : 1) == 0, "pair groups");
  float Tt[RST ? NB * 4 : 1][3];
#pragma unroll
  for (int i = 0; i < (RST ? NB * 4 : 1); i++) { Tt[i][0] = 0.f; Tt[i][1] = 0.f; Tt[i][2] = 0.f; }
  constexpr bool FOLD = FP > 0;                                    // [prefill2] FOLD mode
  static_assert(!FOLD || (RST == 0 && !W8 && FU % FP == 0), "fold: no other periodic decode; FU a multiple of FP");
  int E[FOLD ? NB * 4 : 1];
#pragma unroll
  for (int i = 0; i < (FOLD ? NB * 4 : 1); i++) E[i] = 0;
  // (staggering the warps' periods was measured slower: every K tile ends at a CTA barrier, so one flushing warp per
  // tile delays every tile; all warps flush in the same tile instead)
  const int rphase = 0;
  auto flush = [&]() {
#pragma unroll
    for (int b = 0; b < NB; b++)
#pragma unroll
      for (int p = 0; p < 4; p++) {
        if constexpr (RSTMODE == 0) {
          const float n1 = acc[b][2 * p], n2 = acc[b][2 * p + 1];
          const float d1 = unwrap331_0(n1, n2);
          const float d2 = rintf((n1 - 2.f * d1) * (1.f / 331.f)), d3 = rintf((n2 - 2.f * d2) * (1.f / 331.f));
          Tt[b * 4 + p][0] += d1; Tt[b * 4 + p][1] += d2; Tt[b * 4 + p][2] += d3;
        } else if constexpr (RSTMODE == 2) {                        // timing only: cheap fold, totals live (wrong sums)
          Tt[b * 4 + p][0] += acc[b][2 * p]; Tt[b * 4 + p][1] += acc[b][2 * p + 1];
        }
        acc[b][2 * p] = 0.f; acc[b][2 * p + 1] = 0.f;
      }
  };
  uint4 fa[2][RB], fb[2][CB];
  auto ldf = [&](int s, int b, int k8) {
#pragma unroll
    for (int c = 0; c < CB; c++) fb[s][c] = *reinterpret_cast<const uint4*>(sB(b, brow + c * 32, k8 * 8));
#pragma unroll
    for (int i = 0; i < RB; i++) fa[s][i] = *reinterpret_cast<const uint4*>(sA(b, wm * RB * 8 + i * 8 + arow, k8 * 8));
  };
  auto mma_k8_spread = [&](int s, int nx, int b, int k8n) {
    constexpr int NM = 2 * RB * CB, NL = RB + CB;
#pragma unroll
    for (int m = 0; m < NM; m++) {
      const int hf = m / (RB * CB), idx = m % (RB * CB), i = idx / CB, c = idx % CB, cc = (i & 1) ? CB - 1 - c : c;
      unsigned a[2] = {hf ? fa[s][i].z : fa[s][i].x, hf ? fa[s][i].w : fa[s][i].y};
      unsigned bb[2] = {hf ? fb[s][cc].z : fb[s][cc].x, hf ? fb[s][cc].w : fb[s][cc].y};
      mma884(acc[i * CB + cc], a, bb);
#pragma unroll
      for (int l = 0; l < NL; l++)
        if (m == (l + 1) * NM / (NL + 1) - 1) {
          const int md = m - NM / (NL + 1);
          int dep = 0;
          if (md >= 0) {
            const int id2 = md % (RB * CB), i2 = id2 / CB, c2 = id2 % CB, cc2 = (i2 & 1) ? CB - 1 - c2 : c2;
            dep = __float_as_int(acc[i2 * CB + cc2][0]) & zmask;
          }
          if (l < CB) fb[nx][l] = *reinterpret_cast<const uint4*>(sB(b, brow + l * 32, k8n * 8) + dep);
          else fa[nx][l - CB] = *reinterpret_cast<const uint4*>(sA(b, wm * RB * 8 + (l - CB) * 8 + arow, k8n * 8) + dep);
        }
    }
  };
  auto mma_k8 = [&](int s) {
#pragma unroll
    for (int i = 0; i < RB; i++)
#pragma unroll
      for (int c = 0; c < CB; c++) { int cc = (i & 1) ? CB - 1 - c : c;
        unsigned a[2] = {fa[s][i].x, fa[s][i].y}, b[2] = {fb[s][cc].x, fb[s][cc].y}; mma884(acc[i * CB + cc], a, b); }
#pragma unroll
    for (int i = 0; i < RB; i++)
#pragma unroll
      for (int c = 0; c < CB; c++) { int cc = (i & 1) ? CB - 1 - c : c;
        unsigned a[2] = {fa[s][i].z, fa[s][i].w}, b[2] = {fb[s][cc].z, fb[s][cc].w}; mma884(acc[i * CB + cc], a, b); }
  };

  const int ntiles = K / KTT;
#pragma unroll
  for (int p = 0; p < 2; p++) { ldg(0, p); sts(0, p); }
  __syncthreads();
  ldf(0, 0, 0);
  for (int t = 0; t < ntiles; t++) {
    const int buf = t & 1;
    const bool more = t + 1 < ntiles;
#pragma unroll
    for (int k8 = 0; k8 < NK8; k8++) {
      const int cur = k8 & 1, nx = cur ^ 1;
      if (more && k8 % 4 == 0) ldg((t + 1) * KTT, k8 / 4);
      if (more && k8 % 4 == 3) sts(buf ^ 1, k8 / 4);
      if (SPR == 2 && k8 == NK8 - 1 && more) { __syncthreads(); mma_k8_spread(cur, nx, buf ^ 1, 0); continue; }
      if (k8 < NK8 - 1) ldf(nx, buf, k8 + 1);
      else if (more) { __syncthreads(); ldf(nx, buf ^ 1, 0); }
      mma_k8(cur);
    }
    if constexpr (RST > 0 && RSTMODE == 3) {
      // pair-staggered periods: at the end of tile t the pair group (t + 1) % RST of every warp is decoded and restarted,
      // so each pair's period is RST tiles (the first one shorter) and every tile carries the same small share of the
      // decode work in every warp (no barrier imbalance, the decode interleaves with the other pairs' MMAs)
      if (t + 1 < ntiles) {
        const int gsel = (t + 1) % RST;
#pragma unroll
        for (int gg = 0; gg < RST; gg++)
          if (gg == gsel) {
#pragma unroll
            for (int q = 0; q < NB * 4 / RST; q++) {
              const int idx = gg * (NB * 4 / RST) + q, b = idx / 4, p = idx % 4;
              const float n1 = acc[b][2 * p], n2 = acc[b][2 * p + 1];
              const float d1 = unwrap331_0(n1, n2);
              const float d2 = rintf((n1 - 2.f * d1) * (1.f / 331.f)), d3 = rintf((n2 - 2.f * d2) * (1.f / 331.f));
              Tt[idx][0] += d1; Tt[idx][1] += d2; Tt[idx][2] += d3;
              acc[b][2 * p] = 0.f; acc[b][2 * p + 1] = 0.f;
            }
          }
      }
    } else if constexpr (RST > 0) {
      if ((t + 1 + rphase) % RST == 0 && t + 1 < ntiles) flush();
    }
    if constexpr (FOLD) {                                           // [prefill2]
      if ((t + 1) % FP == 0 && t + 1 < ntiles) {
#pragma unroll
        for (int b = 0; b < NB; b++)
#pragma unroll
          for (int p = 0; p < 4; p++) pkf::reduce<QM>(acc[b][2 * p], acc[b][2 * p + 1], E[b * 4 + p]);
        if ((t + 1) % FU == 0) {
#pragma unroll
          for (int b = 0; b < NB; b++)
#pragma unroll
            for (int p = 0; p < 4; p++) pkf::unwrap(acc[b][2 * p], acc[b][2 * p + 1], E[b * 4 + p]);
        }
      }
    }
    if constexpr (W8) {
      if (((t + 1) * KTT) % PT == 0) {
#pragma unroll
        for (int b = 0; b < NB; b++)
#pragma unroll
          for (int p = 0; p < 4; p++) S1f[b * 4 + p] = unwrap331(acc[b][2 * p], acc[b][2 * p + 1], S1f[b * 4 + p]);
      }
    }
  }

  // epilogue: accumulator (i, j): row = (lane&1) + 4*(lane>=16) + (j&2), qp-col c = (lane&2) + (j&4) + (j&1);
  // qp-col c -> warp col: qp*4 + (c&3) + 16*(c>>2). The decoded triples go through shared memory as fp16.
  constexpr int BNO = BN * 3 / 2, LDO = BNO + 8;
  const int n0o = n0 / 2 * 3;
  __syncthreads();
  // [prefill3] Job 4: exact output checksum (Freivalds over Z / 2^32): per row C = sum over this tile's slots of v[slot] S,
  // v = per-slot random words (chkv), staged in shared memory past the output tile; lanes sharing rows reduce, one atomic
  // per (row, warp) into chkc[row]. The reference sum_k xq[row, k] u[k], u = sum_slot v w, comes from the quant pass.
  unsigned* chs = reinterpret_cast<unsigned*>(smem + (size_t)BM * LDO);
  unsigned chk[CHK ? RB * 2 : 1];
  if constexpr (CHK) {
    for (int i = tid; i < BNO; i += NT) chs[i] = chkv[n0o + i];
#pragma unroll
    for (int i = 0; i < RB * 2; i++) chk[i] = 0u;
    __syncthreads();
  }
#pragma unroll
  for (int ib = 0; ib < NB; ib++)
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const int i = ib / CB, cb = ib % CB, j = 2 * p;
      const int rl = wm * RB * 8 + i * 8 + (lane & 1) + ((lane >> 4) << 2) + (j & 2);
      const int c = (lane & 2) + (j & 4);
      const int lcol = wn * CB * 32 + cb * 32 + qp * 4 + (c & 3) + 16 * (c >> 2);
      const int row = m0 + rl;
      const float s = row < M ? sx[row] : 0.f;
      const float n1 = acc[ib][j], n2 = acc[ib][j + 1];
      float S1v, S2v, S3v;
      if constexpr (FOLD) pkf::final3<QM>(n1, n2, E[ib * 4 + p], S1v, S2v, S3v);   // [prefill2]
      else {
        S1v = unwrap331(n1, n2, W8 ? S1f[ib * 4 + p] : 0.f);
        S2v = rintf((n1 - 2.f * S1v) * (1.f / 331.f)); S3v = rintf((n2 - 2.f * S2v) * (1.f / 331.f));
      }
      if constexpr (RST > 0) { S1v += Tt[ib * 4 + p][0]; S2v += Tt[ib * 4 + p][1]; S3v += Tt[ib * 4 + p][2]; }
      const int ol = (lcol >> 1) * 3, o = n0o + ol;
      if constexpr (CHK) {                                            // [prefill3] Job 4
        chk[i * 2 + ((j & 2) >> 1)] += chs[ol] * (unsigned)__float2int_rn(S1v) + chs[ol + 1] * (unsigned)__float2int_rn(S2v)
                                       + chs[ol + 2] * (unsigned)__float2int_rn(S3v);
      }
      if constexpr (EPI == 2) {                                     // debug: the exact lane sums S as fp32 [M][Nout]
        if (row < M) { float* of = reinterpret_cast<float*>(out) + (size_t)row * Nout + o; of[0] = S1v; of[1] = S2v; of[2] = S3v; }
        continue;
      }
      __half* q = smem + rl * LDO + ol;
      q[0] = __float2half_rn(S1v * s * ae[o]); q[1] = __float2half_rn(S2v * s * ae[o + 1]); q[2] = __float2half_rn(S3v * s * ae[o + 2]);
    }
  if constexpr (CHK) {                                              // [prefill3] Job 4: rows depend on lane bits 0 and 4
#pragma unroll
    for (int r = 0; r < RB * 2; r++) {
      unsigned v = chk[r];
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      chk[r] = v;
    }
    if ((lane & 0xE) == 0) {
#pragma unroll
      for (int r = 0; r < RB * 2; r++) {
        const int row = m0 + wm * RB * 8 + (r >> 1) * 8 + (lane & 1) + ((lane >> 4) << 2) + ((r & 1) << 1);
        if (row < M) atomicAdd(chkc + row, chk[r]);
      }
    }
  }
  __syncthreads();
  if constexpr (EPI == 2) return;
  if constexpr (EPI == 0) {
    constexpr int CPR = BNO / 8;
    for (int c = tid; c < BM * CPR; c += NT) {
      const int r = c / CPR, qv = c % CPR, col = n0o + qv * 8;
      if (m0 + r < M && col < Nout)
        *reinterpret_cast<uint4*>(out + (size_t)(m0 + r) * Nout + col) = *reinterpret_cast<const uint4*>(smem + r * LDO + qv * 8);
    }
  } else {
    constexpr int GPT = BN / 64, VPR = GPT * 6;                    // groups per tile, 16-B h vectors per row
    const int h0 = n0 / 64 * 48;                                    // first h channel of the tile
    for (int c = tid; c < BM * VPR; c += NT) {
      const int r = c / VPR, v = c % VPR, gl = v / 6, i0 = (v % 6) * 8;
      if (m0 + r >= M || h0 + gl * 48 + i0 >= Nout) continue;
      const uint4 gv = *reinterpret_cast<const uint4*>(smem + r * LDO + 96 * gl + i0);
      const uint4 uv = *reinterpret_cast<const uint4*>(smem + r * LDO + 96 * gl + 48 + i0);
      const __half* gh = reinterpret_cast<const __half*>(&gv);
      const __half* uh = reinterpret_cast<const __half*>(&uv);
      __align__(16) __half hv[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float a = __half2float(gh[e]), b = __half2float(uh[e]);
        hv[e] = __float2half_rn(a / (1.f + __expf(-a)) * b);
      }
      *reinterpret_cast<uint4*>(out + (size_t)(m0 + r) * Nout + h0 + gl * 48 + i0) = *reinterpret_cast<const uint4*>(hv);
    }
  }
}

template <int RB, int CB, int WM, int WN, int PT, int EPI, int SPR, int RST = 0, int RSTMODE = 0, int FP = 0, int FU = 0, int QM = 32, int CHK = 0>
void launch_pk(const __half* A, const __half* W, __half* out, const float* sx, const float* ae, int M, int Ncols, int K,
               int Nout, cudaStream_t st, const unsigned* chkv = nullptr, unsigned* chkc = nullptr) {
  constexpr int BM = WM * RB * 8, BN = WN * CB * 32, BNO = BN * 3 / 2;
  constexpr size_t SMm = (size_t)2 * (BM + BN) * (64 + 8) * 2, SMe = (size_t)BM * (BNO + 8) * 2 + (CHK ? BNO * 4 : 0);
  constexpr size_t SM = SMm > SMe ? SMm : SMe;
  static_assert(SM <= 96 * 1024, "shared memory");
  static_assert(EPI == 0 || BN % 64 == 0, "gated tiles hold whole groups");
  auto kern = k_pk<RB, CB, WM, WN, PT, EPI, SPR, RST, RSTMODE, FP, FU, QM, CHK>;
  static bool once = false;
  if (!once) { TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM) == cudaSuccess); once = true; }
  TORCH_CHECK(Ncols % BN == 0 && K % 64 == 0, "pk: Ncols ", Ncols, " not a multiple of ", BN, " or K ", K);
  kern<<<(Ncols / BN) * ((M + BM - 1) / BM), WM * WN * 32, SM, st>>>(A, W, out, sx, ae, M, Ncols, K, Nout, 0, chkv, chkc);
}

// >>> [prefill3] today's prefill pieces of mlp_fp4.cu (verbatim copies: k_dequant, k_silu_pair, the cuBLAS call), so the
// packed path can run one matrix packed and the other on today's maths (and the quality emulation can use NVFP4 weights)
namespace pk3 {
__device__ __forceinline__ unsigned scale2(unsigned e) {           // E4M3 byte (positive) -> fp16x2 (s, s), exact
  const unsigned h = (e & 0x7Fu) * 0x00800080u;
  return hmul2u(h, 0x5C005C00u);
}
__global__ void __launch_bounds__(128) k_dequant(const uint8_t* __restrict__ W, __half* __restrict__ out, int N, int K) {
  __shared__ uint4 sh[4][32][17];
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
    const int idx = it * 32 + lane, r = idx >> 4, u = idx & 15;
    const int n = 32 * p + 8 * ((r >> 2) & 3) + (r & 3) + 4 * (r >> 4);
    *reinterpret_cast<uint4*>(out + (size_t)n * K + cg * 128 + u * 8) = sh[warp][r][u];
  }
}
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
}  // namespace pk3
// <<< [prefill3]
}  // namespace

// ------------------------------------------------------------------------------------------- host entry points
void quant_rows(torch::Tensor x, torch::Tensor xq, torch::Tensor sx, double qmax, c10::optional<torch::Tensor> u,
                c10::optional<torch::Tensor> r) {   // [prefill3] qmax (127 = INT8); u, r: Job 4 reference checksum
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous() && x.dim() == 2 && x.size(1) % 8 == 0);
  const at::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1);
  auto X = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto Q = reinterpret_cast<__half*>(xq.data_ptr<at::Half>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const float qm = (float)qmax;
  const unsigned* U = u.has_value() ? reinterpret_cast<const unsigned*>(u->data_ptr<int>()) : nullptr;
  unsigned* Rr = r.has_value() ? reinterpret_cast<unsigned*>(r->data_ptr<int>()) : nullptr;
  TORCH_CHECK((U == nullptr) == (Rr == nullptr) && (U == nullptr || u->numel() == K), "quant_rows: u / r");
  if (K == 5120) k_quant<20><<<(M + 7) / 8, 256, 0, st>>>(X, Q, sx.data_ptr<float>(), M, K, qm, U, Rr);
  else if (K == 4352) k_quant<17><<<(M + 7) / 8, 256, 0, st>>>(X, Q, sx.data_ptr<float>(), M, K, qm, U, Rr);
  else TORCH_CHECK(false, "quant_rows: K 5120 or 4352");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// w: decode layout (mlp_fp4 prepack) [ntile][K/32][576] u8; pl: plane records [groups][K/32][400 or 16] u8;
// ia [groups * 96] f32; out: packed [64 * gpad][K] fp16 (gpad = groups rounded up to the GEMM tile)
void convert(torch::Tensor w, torch::Tensor pl, torch::Tensor ia, torch::Tensor out, int64_t ntile, int64_t ngroups,
             int64_t K, bool plane, bool gated, int64_t cpb) {
  const at::cuda::CUDAGuard guard(w.device());
  const int gpad = out.size(0) / 64, NC = K / 32;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto W = w.data_ptr<uint8_t>(); auto P = pl.data_ptr<uint8_t>(); auto I = ia.data_ptr<float>();
  auto O = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
#define CONV(PL_, GA_, CP_) k_conv<PL_, GA_, CP_><<<dim3(gpad, (NC + CP_ - 1) / CP_), 96, 0, st>>>(W, P, I, O, ntile, ngroups, K)
#define CONV_C(PL_, GA_) do { if (cpb == 2) CONV(PL_, GA_, 2); else if (cpb == 4) CONV(PL_, GA_, 4); else CONV(PL_, GA_, 6); } while (0)
  if (plane) { if (gated) CONV_C(1, 1); else CONV_C(1, 0); }
  else { if (gated) CONV_C(0, 1); else CONV_C(0, 0); }
#undef CONV_C
#undef CONV
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// packed GEMM: xq [M][K], wp [Ncols][K], sx [M], ae [3 Ncols / 2], out [M][Nout]; epi 0 plain, 1 SiLU pairs, 2 debug:
// out is fp32 [M][3 Ncols / 2] holding the exact lane sums S (no scaling).
// cfg: 0 = 128x192 tile (warp 32x96), decode once; 1 = same, unwrap every 1024; 2 = 64x192 (8 warps of 16x96), once;
//      3 = 128x128 (warp 32x64), once; 4..8 exact periodic decode (see the switch)
void gemm(torch::Tensor xq, torch::Tensor wp, torch::Tensor sx, torch::Tensor ae, torch::Tensor out, int64_t epi, int64_t cfg,
          c10::optional<torch::Tensor> chkv, c10::optional<torch::Tensor> chkc) {    // [prefill3] Job 4 checksum
  const at::cuda::CUDAGuard guard(xq.device());
  const int M = xq.size(0), K = xq.size(1), Ncols = wp.size(0), Nout = out.size(1);
  TORCH_CHECK(epi != 2 || (out.dtype() == torch::kFloat && Nout == Ncols / 2 * 3), "pk gemm debug output");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto A = reinterpret_cast<const __half*>(xq.data_ptr<at::Half>());
  auto W = reinterpret_cast<const __half*>(wp.data_ptr<at::Half>());
  auto O = reinterpret_cast<__half*>(out.data_ptr());
  auto S = sx.data_ptr<float>(); auto E = ae.data_ptr<float>();
  constexpr int ONCE = 1 << 30;
  const bool CK = chkv.has_value();
  TORCH_CHECK(CK == chkc.has_value() && (!CK || chkv->numel() >= Ncols / 2 * 3), "pk gemm: checksum arrays");
  const unsigned* CV = CK ? reinterpret_cast<const unsigned*>(chkv->data_ptr<int>()) : nullptr;
  unsigned* CC = CK ? reinterpret_cast<unsigned*>(chkc->data_ptr<int>()) : nullptr;
  // [prefill3] Job 4: the checksum variants (CHK = 1) of the configurations the server uses: cfg 4 and FOLD 30..35
#define PKRC(R_) do { if (epi == 1) launch_pk<4, 2, 4, 2, ONCE, 1, 2, R_, 0, 0, 0, 32, 1>(A, W, O, S, E, M, Ncols, K, Nout, st, CV, CC); \
                      else launch_pk<4, 2, 4, 2, ONCE, 0, 2, R_, 0, 0, 0, 32, 1>(A, W, O, S, E, M, Ncols, K, Nout, st, CV, CC); } while (0)
#define PKFC(FP_, FU_, QM_) do { if (epi == 1) launch_pk<4, 3, 4, 2, ONCE, 1, 2, 0, 0, FP_, FU_, QM_, 1>(A, W, O, S, E, M, Ncols, K, Nout, st, CV, CC); \
                                 else launch_pk<4, 3, 4, 2, ONCE, 0, 2, 0, 0, FP_, FU_, QM_, 1>(A, W, O, S, E, M, Ncols, K, Nout, st, CV, CC); } while (0)
  if (CK) {
    TORCH_CHECK(epi != 2, "pk gemm: no checksum in the debug epilogue");
    switch (cfg) {
      case 4: PKRC(8); break;
      case 30: PKFC(8, 8, 32); break;
      case 31: PKFC(8, 16, 32); break;
      case 32: PKFC(8, 32, 32); break;        // (unwrap /2048: not exact on early gate+up layers - the guard's test case)
      case 33: PKFC(4, 16, 32); break;
      case 34: PKFC(4, 8, 32); break;
      case 35: PKFC(4, 8, 64); break;
      default: TORCH_CHECK(false, "pk gemm: no checksum variant of cfg ", cfg);
    }
#undef PKRC
#undef PKFC
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    return;
  }
#define PKR(RB_, CB_, WM_, WN_, PT_, R_) do { if (epi == 1) launch_pk<RB_, CB_, WM_, WN_, PT_, 1, 2, R_>(A, W, O, S, E, M, Ncols, K, Nout, st); \
                                         else if (epi == 2) launch_pk<RB_, CB_, WM_, WN_, PT_, 2, 2, R_>(A, W, O, S, E, M, Ncols, K, Nout, st); \
                                         else launch_pk<RB_, CB_, WM_, WN_, PT_, 0, 2, R_>(A, W, O, S, E, M, Ncols, K, Nout, st); } while (0)
#define PK(RB_, CB_, WM_, WN_, PT_) PKR(RB_, CB_, WM_, WN_, PT_, 0)
#define PKM(RB_, CB_, WM_, WN_, R_) do { if (epi == 1) launch_pk<RB_, CB_, WM_, WN_, ONCE, 1, 2, R_, 3>(A, W, O, S, E, M, Ncols, K, Nout, st); \
                                         else if (epi == 2) launch_pk<RB_, CB_, WM_, WN_, ONCE, 2, 2, R_, 3>(A, W, O, S, E, M, Ncols, K, Nout, st); \
                                         else launch_pk<RB_, CB_, WM_, WN_, ONCE, 0, 2, R_, 3>(A, W, O, S, E, M, Ncols, K, Nout, st); } while (0)
#define PKF(FP_, FU_, QM_) do { if (epi == 1) launch_pk<4, 3, 4, 2, ONCE, 1, 2, 0, 0, FP_, FU_, QM_>(A, W, O, S, E, M, Ncols, K, Nout, st); \
                             else if (epi == 2) launch_pk<4, 3, 4, 2, ONCE, 2, 2, 0, 0, FP_, FU_, QM_>(A, W, O, S, E, M, Ncols, K, Nout, st); \
                             else launch_pk<4, 3, 4, 2, ONCE, 0, 2, 0, 0, FP_, FU_, QM_>(A, W, O, S, E, M, Ncols, K, Nout, st); } while (0)
#define PKA(RB_, CB_, WM_, WN_, R_, MD_) launch_pk<RB_, CB_, WM_, WN_, ONCE, 0, 2, R_, MD_>(A, W, O, S, E, M, Ncols, K, Nout, st)
  switch (cfg) {
    case 0: PK(4, 3, 4, 2, ONCE); break;
    case 1: PK(4, 3, 4, 2, 1024); break;
    case 2: PK(2, 3, 4, 2, ONCE); break;
    case 3: PK(4, 2, 4, 2, ONCE); break;
    // exact periodic decode (reset every RST x 64 terms)
    case 4: PKR(4, 2, 4, 2, ONCE, 8); break;     // 128x128, warp 32x64, every 512
    case 5: PKR(2, 3, 4, 2, ONCE, 8); break;     // 64x192, warp 16x96, every 512
    case 6: PKR(4, 2, 4, 2, ONCE, 4); break;     // 128x128, every 256
    case 7: PKR(2, 3, 4, 2, ONCE, 4); break;     // 64x192, every 256
    case 8: PKR(4, 3, 4, 2, ONCE, 8); break;     // 128x192, warp 32x96, every 512 (register test)
    case 9: PKR(4, 2, 4, 2, ONCE, 16); break;    // 128x128, every 1024
    // pair-staggered exact periodic decode (RSTMODE 3)
    case 10: PKM(4, 2, 4, 2, 8); break;          // 128x128, every 512
    case 11: PKM(4, 2, 4, 2, 4); break;          // 128x128, every 256
    case 12: PKM(2, 3, 4, 2, 8); break;          // 64x192, every 512
    case 13: PKM(4, 3, 4, 2, 8); break;          // 128x192 (register test)
    // [prefill2] FOLD mode, 128x192 (warp 32x96): reduce every FP x 64 terms, unwrap every FU x 64
    case 30: PKF(8, 8, 32); break;                   // reduce + unwrap every 512
    case 31: PKF(8, 16, 32); break;                  // reduce every 512, unwrap every 1024
    case 32: PKF(8, 32, 32); break;                  // reduce every 512, unwrap every 2048
    case 33: PKF(4, 16, 32); break;                  // reduce every 256, unwrap every 1024
    case 34: PKF(4, 8, 32); break;                   // reduce every 256, unwrap every 512
    case 35: PKF(4, 8, 64); break;               // reduce every 256, unwrap every 512, Q = 64 * 331^2 (full-K |n| < 221M)
    // timing-only ablations (wrong outputs; plain epilogue): flush = zero only / fold without decode
    case 20: PKA(4, 2, 4, 2, 8, 1); break;
    case 21: PKA(4, 2, 4, 2, 8, 2); break;
    case 22: PKA(2, 3, 4, 2, 8, 1); break;
    default: TORCH_CHECK(false, "pk gemm cfg");
  }
#undef PK
#undef PKR
#undef PKA
#undef PKM
#undef PKF
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}


// >>> [prefill3] Job 4 host entry points
void rowdot(torch::Tensor xq, torch::Tensor u, torch::Tensor r) {
  const at::cuda::CUDAGuard guard(xq.device());
  const int M = xq.size(0), K = xq.size(1);
  auto X = reinterpret_cast<const __half*>(xq.data_ptr<at::Half>());
  auto U = reinterpret_cast<const unsigned*>(u.data_ptr<int>());
  auto Rr = reinterpret_cast<unsigned*>(r.data_ptr<int>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (K == 5120) k_rowdot<20><<<(M + 7) / 8, 256, 0, st>>>(X, U, Rr, M, K);
  else if (K == 4352) k_rowdot<17><<<(M + 7) / 8, 256, 0, st>>>(X, U, Rr, M, K);
  else TORCH_CHECK(false, "rowdot: K 5120 or 4352");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
void check(torch::Tensor c, torch::Tensor r, int64_t M, torch::Tensor flag) {
  const at::cuda::CUDAGuard guard(c.device());
  const int nb = std::max(1, std::min(8, (int)((M + 1023) / 1024)));
  k_chk<<<nb, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned*>(c.data_ptr<int>()),
                                                          reinterpret_cast<const unsigned*>(r.data_ptr<int>()), (int)M,
                                                          flag.data_ptr<int>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
// <<< [prefill3] Job 4
// >>> [prefill3] host entry points of today's pieces
// prepacked NVFP4 (mlp_fp4.cu layout) -> fp16 [N][K] physical column order, E2M1 * E4M3 * 2^-14 (alpha carries 2^14 / g)
void dequant_nv(torch::Tensor w, torch::Tensor out, int64_t N, int64_t K) {
  const at::cuda::CUDAGuard guard(w.device());
  TORCH_CHECK(out.dtype() == torch::kHalf && out.numel() >= N * K && (K / 32) % 4 == 0);
  const long long t = (long long)(N / 32) * (K / 128);
  pk3::k_dequant<<<(unsigned)((t + 3) / 4), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      w.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)N, (int)K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
// silu(gate) * up from the physical-order gate+up output [M][2I] -> [M][I]
void silu_pair(torch::Tensor gu, torch::Tensor out) {
  const at::cuda::CUDAGuard guard(gu.device());
  const int M = gu.size(0), I = out.size(1);
  const long long ts = (long long)M * (I / 8);
  pk3::k_silu_pair<<<(unsigned)((ts + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(gu.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, I);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
// C[M][N] = alpha X[M][K] W[N][K]^T (fp16, FP32 accumulation; the same cublasGemmEx call as mlp_fp4.cu)
void gemm_nt(torch::Tensor w, torch::Tensor x, torch::Tensor c, int64_t N, int64_t K, double alpha) {
  const at::cuda::CUDAGuard guard(x.device());
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
  const int M = x.size(0);
  const float a = (float)alpha, beta = 0.f;
  TORCH_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, M, (int)K, &a, w.data_ptr(), CUDA_R_16F, (int)K, x.data_ptr(),
                           CUDA_R_16F, (int)K, &beta, c.data_ptr(), CUDA_R_16F, (int)N, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS, "pk gemm_nt: cublasGemmEx failed");
}
// <<< [prefill3]

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_rows", &quant_rows, "per-token INT8 rows (fp16 integers) + scales", pybind11::arg("x"), pybind11::arg("xq"),
        pybind11::arg("sx"), pybind11::arg("qmax") = 127.0, pybind11::arg("u") = pybind11::none(),
        pybind11::arg("r") = pybind11::none());   // [prefill3] qmax, Job 4 reference checksum
  m.def("rowdot", &rowdot, "[prefill3] Job 4: reference checksums sum_k xq[row, k] u[k] mod 2^32");
  m.def("check", &check, "[prefill3] Job 4: compare checksums (zeroes c), flag[0] |= mismatch, flag[1] += rows");
  m.def("convert", &convert, "decode-layout NVFP4 (+ selection plane) -> packed chain rows");
  m.def("gemm", &gemm, "packed chain GEMM (decode in the epilogue)", pybind11::arg("xq"), pybind11::arg("wp"), pybind11::arg("sx"),
        pybind11::arg("ae"), pybind11::arg("out"), pybind11::arg("epi"), pybind11::arg("cfg"),
        pybind11::arg("chkv") = pybind11::none(), pybind11::arg("chkc") = pybind11::none());   // [prefill3] Job 4
  m.def("dequant_nv", &dequant_nv, "[prefill3] today's NVFP4 dequantisation (mlp_fp4.cu k_dequant)");
  m.def("silu_pair", &silu_pair, "[prefill3] today's SiLU-and-mul on the physical gate+up order");
  m.def("gemm_nt", &gemm_nt, "[prefill3] today's cuBLAS call");
}
