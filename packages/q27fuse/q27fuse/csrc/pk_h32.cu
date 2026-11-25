// pk_h32.cu (track prefill4) - online H_32 (nosign) + INT4 (1, 256) FOLD + INT7/INT8 packed prefill MLP for V100 (sm_70).
//
// Three useful products per two tensor-core products via the (1, 256) INT4 chain:
//   P1 = c1 + 256 c2,   P2 = c2 + 256 c3    (c in [-7, 7]; |P| <= 1,799 < 2^11, FP16-exact)
// giving n1 = S1 + 256 S2, n2 = S2 + 256 S3 with S = sum_k xq[k] c[k] (integers; |xq P| < 2^18, HMMA-exact).
// Both activations and weights are rotated online by blockwise orthogonal H_32 (no sign vector, 1/sqrt(32) normalized):
//   y = (x H_32) (W H_32)^T = x (H_32 H_32^T) W^T = x W^T.
//
// Pieces:
//   k_quant_h32  online H_32 butterfly (warp-local, 4 lanes per 32-block, zero SMEM) + per-token INT7/INT8 + Freivalds ref
//   k_conv_h32   decode-layout NVFP4 (mlp_fp4 prepack) -> in-register H_32 butterfly -> INT4 [-7, 7] (+ optional 1-bit
//                floor/ceil selection plane, NO rel bytes) -> (1, 256) packed scratch [64 * gpad][K] fp16
//   k_pk_i4      HMMA.884 128x192 / 128x128 GEMM with INT4 (1, 256) 3-register FOLD (Q = 2^21, F1 = 19, F2 = 25),
//                DM = 1 (burst) and DM = 2 (interleaved), EPI = 0/1/2, CHK = 0/1 (Freivalds guard over Z / 2^32)
//   k_rowdot / k_chk / k_dequant / k_silu_pair / gemm_nt / prepack (standalone comparison & guard helpers)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>

namespace {
constexpr int CHUNK = 576;
constexpr float INV_SQRT32 = 0.17677669529663687f;   // exact f32 1 / sqrt(32) = 0x3e3504f3

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
__device__ __forceinline__ float e4m3f(unsigned e) {               // E4M3FN byte (positive) -> float, exact
  return __half2float(__ushort_as_half((unsigned short)((e & 0x7Fu) << 7))) * 256.f;
}

// ------------------------------------------------------------------------------------------- online H_32 + per-token INT7/INT8
// One warp per row, 256 threads/CTA (8 warps). Each lane holds VPL uint4 vectors (8 consecutive fp16 elements each),
// so 4 consecutive lanes (lane & 3 == 0..3) hold one exact 32-element H_32 block. After the 5-stage butterfly and
// 1/sqrt(32) scaling, rotated values are rounded to __half2 and packed back into u[VPL] (identical register footprint
// to k_quant<VPL>, zero SMEM, zero spills); amax is tracked from the __half2-rounded values to match _qdq(_rot32(x).half()).
template <int VPL>
__global__ void __launch_bounds__(256) k_quant_h32(const __half* __restrict__ X, __half* __restrict__ Xq,
                                                   float* __restrict__ sx, int M, int K, float qm,
                                                   const unsigned* __restrict__ U, unsigned* __restrict__ R) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * K);
  uint4* qr = reinterpret_cast<uint4*>(Xq + (size_t)row * K);
  uint4 u[VPL];
#pragma unroll
  for (int v = 0; v < VPL; v++) u[v] = __ldcs(xr + v * 32 + lane);

  const bool hi1 = (lane & 1) != 0;
  const bool hi2 = (lane & 2) != 0;
  float am = 0.f;
#pragma unroll
  for (int v = 0; v < VPL; v++) {
    __half2* h = reinterpret_cast<__half2*>(&u[v]);
    float f[8];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float2 p = __half22float2(h[e]);
      f[2 * e] = p.x;
      f[2 * e + 1] = p.y;
    }
    // Stage 1 (stride 1): (0,1), (2,3), (4,5), (6,7)
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const float a = f[e], b = f[e + 1];
      f[e] = a + b;
      f[e + 1] = a - b;
    }
    // Stage 2 (stride 2): (0,2), (1,3), (4,6), (5,7)
#pragma unroll
    for (int base = 0; base < 8; base += 4) {
      const float a0 = f[base], b0 = f[base + 2];
      f[base] = a0 + b0;
      f[base + 2] = a0 - b0;
      const float a1 = f[base + 1], b1 = f[base + 3];
      f[base + 1] = a1 + b1;
      f[base + 3] = a1 - b1;
    }
    // Stage 3 (stride 4): (0,4), (1,5), (2,6), (3,7)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float a = f[e], b = f[e + 4];
      f[e] = a + b;
      f[e + 4] = a - b;
    }
    // Stage 4 (stride 8): xor lane bit 0
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const float peer = __shfl_xor_sync(0xffffffffu, f[e], 1);
      f[e] = hi1 ? (peer - f[e]) : (f[e] + peer);
    }
    // Stage 5 (stride 16): xor lane bit 1
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const float peer = __shfl_xor_sync(0xffffffffu, f[e], 2);
      f[e] = hi2 ? (peer - f[e]) : (f[e] + peer);
    }
    // Scale by 1/sqrt(32), clamp to finite FP16 range before __half2 packing so H_32 never creates inf -> inv=0
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float r0 = fminf(65504.f, fmaxf(-65504.f, f[2 * e] * INV_SQRT32));
      const float r1 = fminf(65504.f, fmaxf(-65504.f, f[2 * e + 1] * INV_SQRT32));
      const __half2 hr = __floats2half2_rn(r0, r1);
      h[e] = hr;
      const float2 hf = __half22float2(hr);
      am = fmaxf(am, fmaxf(fabsf(hf.x), fabsf(hf.y)));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  const float inv = am > 0.f ? __fdiv_rn(qm, am) : 0.f;
  unsigned ref = 0u;
#pragma unroll
  for (int v = 0; v < VPL; v++) {
    __half2* h = reinterpret_cast<__half2*>(&u[v]);
    float qv[8];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float2 f = __half22float2(h[e]);
      qv[2 * e] = fminf(qm, fmaxf(-qm, rintf(f.x * inv)));
      qv[2 * e + 1] = fminf(qm, fmaxf(-qm, rintf(f.y * inv)));
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
  if (lane == 0) sx[row] = __fdiv_rn(am, qm);
}

// Reference checksum of given integer rows (fp16 integers) mod 2^32
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

// Compare GEMM checksums with references; zero C for the next launch.
// flag[0] |= 1 on any mismatch, flag[1] += mismatching rows.
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

// Single-CTA (1024 threads) zero-sync Freivalds check: exact sum of mismatching rows across any M, zeroes C,
// overwrites flag_cur[0..1] directly (no separate flag.zero_() launch needed), and updates optional flag_tot[0..1].
__global__ void __launch_bounds__(1024) k_chk_single(unsigned* __restrict__ C, const unsigned* __restrict__ R, int M,
                                                     int* __restrict__ flag_cur, int* __restrict__ flag_tot) {
  __shared__ int s_bad[32];
  int bad = 0;
  for (int r = threadIdx.x; r < M; r += 1024) {
    bad += (C[r] != R[r]) ? 1 : 0;
    C[r] = 0u;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) bad += __shfl_xor_sync(0xffffffffu, bad, o);
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  if (lane == 0) s_bad[wp] = bad;
  __syncthreads();
  if (wp == 0) {
    int tot = s_bad[lane];
#pragma unroll
    for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
    if (lane == 0) {
      flag_cur[0] = (tot > 0) ? 1 : 0;
      flag_cur[1] = tot;
      if (tot > 0 && flag_tot != nullptr) {
        flag_tot[0] += 1;
        flag_tot[1] += tot;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------- H_32 conversion pass
// Grid: dim3(gpad, ceil(NC / CPB)), 96 threads (3 warps = 1 group of 96 decode rows = 3 tiles of 32 cols).
// Each lane decodes 32 consecutive K weights of its column (slot), multiplies by e4m3 * ia_scaled (with 2^14 / (g * alpha * sqrt(32))
// pre-folded into ia_scaled), runs an exact in-register 5-stage H_32 butterfly (zero warp shuffles), quantizes to INT4 [-7, 7]
// (PLANE=0: rintf; PLANE=1: floorf + bit from SEL[G, chunk, wi, lane]), writes to dec[slot], then packs 3->2 rows with (1, 256):
//   P1 = A + 256 * B,  P2 = B + 256 * C   (0x5C00 = exact FP16 256.0).
template <int PLANE, int GATED, int CPB>
__global__ void __launch_bounds__(96) k_conv_h32(const uint8_t* __restrict__ W, const uint8_t* __restrict__ SEL,
                                                 const float* __restrict__ IA, __half* __restrict__ out,
                                                 int ntile, int ngroups, int K) {
  constexpr int RS = CPB * 16 + 4;                                  // u32 per slot row (16B aligned, conflict-free)
  constexpr unsigned K256 = 0x5C005C00u;                            // exact FP16 (256.0, 256.0)
  __shared__ __align__(16) uint32_t dec[96 * RS];
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
  const int G = blockIdx.x, NC = K / 32, c0 = blockIdx.y * CPB;
  const int nch = min(CPB, NC - c0);
  const int p = 3 * G + wi;
  const bool real = G < ngroups && p < ntile;
  const int lc = 8 * ((lane >> 2) & 3) + (lane & 3) + 4 * (lane >> 4);
  const int slot = GATED ? (lc < 16 ? 16 * wi + lc : 48 + 16 * wi + lc - 16) : 32 * wi + lc;
  const float ia = real ? IA[G * 96 + slot] : 0.f;

  uint4 w[CPB];
  unsigned sv[CPB];
  [[maybe_unused]] unsigned sel[PLANE ? CPB : 1];
#pragma unroll
  for (int cc = 0; cc < CPB; cc++) {
    w[cc] = make_uint4(0, 0, 0, 0);
    sv[cc] = 0u;
    if constexpr (PLANE) sel[cc] = 0u;
    if (real && cc < nch) {
      const uint8_t* src = W + ((size_t)p * NC + c0 + cc) * CHUNK;
      w[cc] = ldg16(src + lane * 16);
      sv[cc] = ldg2(src + 512 + 2 * lane);
      if constexpr (PLANE) sel[cc] = ldg4(SEL + ((size_t)G * NC + c0 + cc) * 384 + (wi * 32 + lane) * 4);
    }
  }
#pragma unroll
  for (int cc = 0; cc < CPB; cc++) {
    if (cc >= nch) break;
    const float sc0 = __fmul_rn(e4m3f(sv[cc] & 0xFFu), ia);
    const float sc1 = __fmul_rn(e4m3f(sv[cc] >> 8), ia);
    const unsigned wd[4] = {w[cc].x, w[cc].y, w[cc].z, w[cc].w};
    float f[32];
#pragma unroll
    for (int i4 = 0; i4 < 4; i4++) {
      const float sc = i4 < 2 ? sc0 : sc1;
      const unsigned v0 = dq<0>(wd[i4]);
      const unsigned v1 = dq<1>(wd[i4]);
      const unsigned v2 = dq<2>(wd[i4]);
      const unsigned v3 = dq<3>(wd[i4]);
      const float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&v0));
      const float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&v1));
      const float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&v2));
      const float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&v3));
      // __fmul_rn prevents nvcc FMA contraction into Stage-1 butterfly (bit-exact with PyTorch conv_h32_ref)
      f[8 * i4 + 0] = __fmul_rn(p0.x, sc); f[8 * i4 + 1] = __fmul_rn(p0.y, sc);
      f[8 * i4 + 2] = __fmul_rn(p1.x, sc); f[8 * i4 + 3] = __fmul_rn(p1.y, sc);
      f[8 * i4 + 4] = __fmul_rn(p2.x, sc); f[8 * i4 + 5] = __fmul_rn(p2.y, sc);
      f[8 * i4 + 6] = __fmul_rn(p3.x, sc); f[8 * i4 + 7] = __fmul_rn(p3.y, sc);
    }
    // Exact in-register 5-stage H_32 butterfly (strides 1, 2, 4, 8, 16)
    // Stage 1: stride 1
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
      const float a = f[i], b = f[i + 1];
      f[i] = a + b; f[i + 1] = a - b;
    }
    // Stage 2: stride 2
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
      const float a0 = f[i], b0 = f[i + 2]; f[i] = a0 + b0; f[i + 2] = a0 - b0;
      const float a1 = f[i + 1], b1 = f[i + 3]; f[i + 1] = a1 + b1; f[i + 3] = a1 - b1;
    }
    // Stage 3: stride 4
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float a = f[i + j], b = f[i + j + 4];
        f[i + j] = a + b; f[i + j + 4] = a - b;
      }
    }
    // Stage 4: stride 8
#pragma unroll
    for (int i = 0; i < 32; i += 16) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float a = f[i + j], b = f[i + j + 8];
        f[i + j] = a + b; f[i + j + 8] = a - b;
      }
    }
    // Stage 5: stride 16
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const float a = f[j], b = f[j + 16];
      f[j] = a + b; f[j + 16] = a - b;
    }
    // Quantize to INT4 [-7, 7] as __half2 and write directly to dec in 4 x uint4 (keeps peak regs low)
    uint4* d = reinterpret_cast<uint4*>(dec + slot * RS + cc * 16);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      unsigned pk[4];
#pragma unroll
      for (int m = 0; m < 4; m++) {
        const int k0 = q * 8 + m * 2;
        float c0v, c1v;
        if constexpr (PLANE) {
          const unsigned b0 = (sel[cc] >> k0) & 1u;
          const unsigned b1 = (sel[cc] >> (k0 + 1)) & 1u;
          c0v = fminf(7.f, fmaxf(-7.f, floorf(f[k0]) + (float)b0));
          c1v = fminf(7.f, fmaxf(-7.f, floorf(f[k0 + 1]) + (float)b1));
        } else {
          c0v = fminf(7.f, fmaxf(-7.f, rintf(f[k0])));
          c1v = fminf(7.f, fmaxf(-7.f, rintf(f[k0 + 1])));
        }
        const __half2 h2 = __floats2half2_rn(c0v, c1v);
        pk[m] = *reinterpret_cast<const unsigned*>(&h2);
      }
      d[q] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  }
  __syncthreads();
  // Pack 3 slot rows -> 2 chain rows: P1 = A + 256*B, P2 = B + 256*C
  const int VPR = 4 * nch;
  for (int item = threadIdx.x; item < 64 * VPR; item += 96) {
    const int pr = item / VPR, v = item % VPR;
    const int tt = pr >> 1, a = 3 * tt + (pr & 1);
    const int j0 = 4 * v;
    const uint4 A = *reinterpret_cast<const uint4*>(dec + a * RS + j0);
    const uint4 B = *reinterpret_cast<const uint4*>(dec + (a + 1) * RS + j0);
    uint4 P;
    P.x = hfma2u(B.x, K256, A.x);
    P.y = hfma2u(B.y, K256, A.y);
    P.z = hfma2u(B.z, K256, A.z);
    P.w = hfma2u(B.w, K256, A.w);
    *reinterpret_cast<uint4*>(out + (size_t)(64 * G + pr) * K + c0 * 32 + 2 * j0) = P;
  }
}

// ------------------------------------------------------------------------------------------- INT4 (1, 256) FOLD GEMM
__device__ __forceinline__ void mma884(float* d, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

// INT4 (1, 256) FOLD helpers (jp_fold.cuh FoldK<0>):
//   Q = 2^21 = 2,097,152 (exact multiple of 2^16 -> leaves unwrap residue unchanged)
//   E = s1 + 2^19 * q1 + 2^25 * q2 (mod 2^32): |s1| < 2^18 (262,144), |q1| < 32 (|n1| < 66M), |q2| < 64 (|n2| < 133M).
//   Explicit __fadd_rn / __fsub_rn / fmaf guard MAG = 12582912.f rounding against --use_fast_math optimizations.
namespace i4f {
constexpr float Q = 2097152.f;
constexpr float IQ = 1.f / 2097152.f;
constexpr float MAG = 12582912.f;                                   // 0x4B400000 = 3 * 2^22

__device__ __forceinline__ void reduce(float& a, float& b, uint32_t& E) {
  const float t1 = fmaf(a, IQ, MAG), t2 = fmaf(b, IQ, MAG);
  a = fmaf(-Q, __fsub_rn(t1, MAG), a);
  b = fmaf(-Q, __fsub_rn(t2, MAG), b);
  E += (((uint32_t)__float_as_int(t1) << 19) + ((uint32_t)__float_as_int(t2) << 25));
}

__device__ __forceinline__ void unwrap(float a, float b, uint32_t& E) {
  const uint32_t b1 = (uint32_t)__float_as_int(__fadd_rn(a, MAG));
  const uint32_t b2 = (uint32_t)__float_as_int(__fadd_rn(b, MAG));
  const uint32_t x = b1 - (b2 << 8) + 255u * 0x4B400000u;           // n1 - 256 n2 (mod 2^32) == S1 (mod 2^16)
  const int d = (int)((x - E) << 16) >> 16;                         // centred step in [-32768, 32767]
  E += (uint32_t)d;
}

__device__ __forceinline__ void fold(float& a, float& b, uint32_t& E) {
  reduce(a, b, E);
  unwrap(a, b, E);
}

__device__ __forceinline__ void final3(float a, float b, uint32_t E, int& s1, int& s2, int& s3) {
  reduce(a, b, E);
  unwrap(a, b, E);
  s1 = (int)(E << 13) >> 13;                                        // low 19-bit signed: [-2^18, 2^18 - 1]
  const int r = (int)(E - (uint32_t)s1) >> 19;                      // q1 + 64 * q2 (exact 13-bit signed)
  const int q1 = (int)((uint32_t)r << 26) >> 26;                    // low 6-bit signed: [-32, 31]
  const int q2 = (r - q1) >> 6;                                     // high 7-bit signed: [-63, 63]
  const int n1 = __float2int_rn(a) + q1 * 2097152;
  const int n2 = __float2int_rn(b) + q2 * 2097152;
  s2 = (n1 - s1) >> 8;
  s3 = (n2 - s2) >> 8;
}
}  // namespace i4f

// RB: 8-row blocks/warp, CB: 32-col blocks/warp, WM x WN warps, K tile 64.
// FP: reduce period (K-tiles of 64), FU: unwrap period (multiple of FP).
// DM = 1: burst fold at end of period tile; DM = 2: interleaved fold inside mma_k8 (requires FP == FU).
// EPI = 0: fp16 out [M][Nout]; EPI = 1: fused SiLU(gate) * up -> [M][I]; EPI = 2: debug fp32 exact lane sums S [M][3*Ncols/2].
// CHK = 1: Freivalds output checksum over Z / 2^32 accumulated into chkc[row].
template <int RB, int CB, int WM, int WN, int EPI, int SPR, int FP, int FU, int DM, int CHK>
__global__ void __launch_bounds__(WM * WN * 32, 1)
k_pk_i4(const __half* __restrict__ A, const __half* __restrict__ W, __half* __restrict__ out,
        const float* __restrict__ sx, const float* __restrict__ ae, int M, int Ncols, int K, int Nout, int zmask,
        const unsigned* __restrict__ chkv, unsigned* __restrict__ chkc) {
  static_assert(FP > 0 && FU > 0 && FU % FP == 0, "fold: FU must be a multiple of FP");
  static_assert(DM == 1 || (DM == 2 && FP == FU), "DM 2 interleaved fold requires FP == FU");
  static_assert(!CHK || EPI != 2, "no checksum in debug epilogue");
  constexpr int KTT = 64, BM = WM * RB * 8, BN = WN * CB * 32, NT = WM * WN * 32, NB = RB * CB, LD = KTT + 8, NK8 = KTT / 8;
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
  uint32_t E[NB * 4];
#pragma unroll
  for (int i = 0; i < NB * 4; i++) E[i] = 0u;
  uint4 fa[2][RB], fb[2][CB];
  auto ldf = [&](int s, int b, int k8) {
#pragma unroll
    for (int c = 0; c < CB; c++) fb[s][c] = *reinterpret_cast<const uint4*>(sB(b, brow + c * 32, k8 * 8));
#pragma unroll
    for (int i = 0; i < RB; i++) fa[s][i] = *reinterpret_cast<const uint4*>(sA(b, wm * RB * 8 + i * 8 + arow, k8 * 8));
  };
  [[maybe_unused]] auto foldb = [&](int ib) {
#pragma unroll
    for (int p = 0; p < 4; p++) i4f::fold(acc[ib][2 * p], acc[ib][2 * p + 1], E[ib * 4 + p]);
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
  // DM == 2 interleaved fold: fold each block right before its first-half MMA at k8 == 0 of a period boundary
  [[maybe_unused]] auto mma_k8_fold = [&](int s) {
#pragma unroll
    for (int i = 0; i < RB; i++)
#pragma unroll
      for (int c = 0; c < CB; c++) { int cc = (i & 1) ? CB - 1 - c : c;
        foldb(i * CB + cc);
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
    [[maybe_unused]] const bool fd = (DM == 2) && (t > 0) && (t % FP == 0);
#pragma unroll
    for (int k8 = 0; k8 < NK8; k8++) {
      const int cur = k8 & 1, nx = cur ^ 1;
      if (more && k8 % 4 == 0) ldg((t + 1) * KTT, k8 / 4);
      if (more && k8 % 4 == 3) sts(buf ^ 1, k8 / 4);
      if (SPR == 2 && k8 == NK8 - 1 && more) { __syncthreads(); mma_k8_spread(cur, nx, buf ^ 1, 0); continue; }
      if (k8 < NK8 - 1) ldf(nx, buf, k8 + 1);
      else if (more) { __syncthreads(); ldf(nx, buf ^ 1, 0); }
      if constexpr (DM == 2) {
        if (k8 == 0 && fd) mma_k8_fold(cur);
        else mma_k8(cur);
      } else {
        mma_k8(cur);
      }
    }
    if constexpr (DM == 1) {
      if ((t + 1) % FP == 0 && more) {
#pragma unroll
        for (int b = 0; b < NB; b++)
#pragma unroll
          for (int p = 0; p < 4; p++) i4f::reduce(acc[b][2 * p], acc[b][2 * p + 1], E[b * 4 + p]);
        if ((t + 1) % FU == 0) {
#pragma unroll
          for (int b = 0; b < NB; b++)
#pragma unroll
            for (int p = 0; p < 4; p++) i4f::unwrap(acc[b][2 * p], acc[b][2 * p + 1], E[b * 4 + p]);
        }
      }
    }
  }

  // Epilogue: decode (S1, S2, S3) via i4f::final3, accumulate optional Freivalds checksum, stage scaled fp16 through SMEM.
  constexpr int BNO = BN * 3 / 2, LDO = BNO + 8;
  const int n0o = n0 / 2 * 3;
  __syncthreads();
  [[maybe_unused]] unsigned* chs = reinterpret_cast<unsigned*>(smem + (size_t)BM * LDO);
  [[maybe_unused]] unsigned chk[CHK ? RB * 2 : 1];
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
      int s1, s2, s3;
      i4f::final3(acc[ib][j], acc[ib][j + 1], E[ib * 4 + p], s1, s2, s3);
      const int ol = (lcol >> 1) * 3, o = n0o + ol;
      if constexpr (CHK) {
        chk[i * 2 + ((j & 2) >> 1)] += chs[ol] * (unsigned)s1 + chs[ol + 1] * (unsigned)s2 + chs[ol + 2] * (unsigned)s3;
      }
      if constexpr (EPI == 2) {
        if (row < M) {
          float* of = reinterpret_cast<float*>(out) + (size_t)row * Nout + o;
          of[0] = (float)s1; of[1] = (float)s2; of[2] = (float)s3;
        }
        continue;
      }
      __half* q = smem + rl * LDO + ol;
      q[0] = __float2half_rn((float)s1 * s * ae[o]);
      q[1] = __float2half_rn((float)s2 * s * ae[o + 1]);
      q[2] = __float2half_rn((float)s3 * s * ae[o + 2]);
    }
  if constexpr (CHK) {
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
    constexpr int GPT = BN / 64, VPR = GPT * 6;
    const int h0 = n0 / 64 * 48;
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

template <int RB, int CB, int WM, int WN, int EPI, int SPR, int FP, int FU, int DM, int CHK>
void launch_pk_i4(const __half* A, const __half* W, __half* out, const float* sx, const float* ae, int M, int Ncols,
                  int K, int Nout, cudaStream_t st, const unsigned* chkv = nullptr, unsigned* chkc = nullptr) {
  constexpr int BM = WM * RB * 8, BN = WN * CB * 32, BNO = BN * 3 / 2;
  constexpr size_t SMm = (size_t)2 * (BM + BN) * (64 + 8) * 2;
  constexpr size_t SMe = (size_t)BM * (BNO + 8) * 2 + (CHK ? (size_t)BNO * 4 : 0);
  constexpr size_t SM = SMm > SMe ? SMm : SMe;
  static_assert(SM <= 96 * 1024, "shared memory");
  static_assert(EPI == 0 || BN % 64 == 0, "gated tiles hold whole groups");
  auto kern = k_pk_i4<RB, CB, WM, WN, EPI, SPR, FP, FU, DM, CHK>;
  static unsigned set_dev_mask = 0u;
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 32 && !(set_dev_mask & (1u << dev))) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM) == cudaSuccess);
    set_dev_mask |= (1u << dev);
  }
  TORCH_CHECK(Ncols % BN == 0 && K % 64 == 0, "gemm_i4: Ncols ", Ncols, " not a multiple of ", BN, " or K ", K);
  kern<<<(Ncols / BN) * ((M + BM - 1) / BM), WM * WN * 32, SM, st>>>(A, W, out, sx, ae, M, Ncols, K, Nout, 0, chkv, chkc);
}

// ------------------------------------------------------------------------------------------- today's reference pieces + prepack
// Verbatim from pk_mlp.cu / mlp_fp4.cu so test_h32.py and bench_h32.py can compare directly in one extension.
namespace ref {
__device__ __forceinline__ unsigned scale2(unsigned e) {
  const unsigned h = (e & 0x7Fu) * 0x00800080u;
  return hmul2u(h, 0x5C005C00u);
}

__global__ void k_prepack(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, uint8_t* __restrict__ out,
                          int N, int K, int gated) {
  const int NC = K / 32;
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)(N / 32) * NC * 32) return;
  const int l = gid & 31, c = (gid >> 5) % NC, p = (int)((gid >> 5) / NC);
  const int q = (l >> 2) & 3, cB = (l & 3) + 4 * (l >> 4), lc = 8 * q + cB;
  const int nrow = gated ? (lc < 16 ? 16 * p + lc : N / 2 + 16 * p + lc - 16) : 32 * p + lc;
  const uint8_t* cr = codes + (size_t)nrow * (K / 2) + 16 * c;
  uint32_t wv[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t w = 0;
#pragma unroll
    for (int pp = 0; pp < 4; pp++) {
      const int b = cr[4 * i + pp], lo = b & 15, hi = b >> 4;
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

// ------------------------------------------------------------------------------------------- zero-sync device-gated fallbacks
// Bounded-grid (160 CTAs) exact NVFP4 dequantisation gated on *flag != 0: ~1.5-2 us early exit when flag == 0,
// persistent grid-stride k_dequant when flag != 0.
__global__ void __launch_bounds__(128) k_dequant_if(const int* __restrict__ flag, const uint8_t* __restrict__ W,
                                                    __half* __restrict__ out, int N, int K) {
  if (flag != nullptr && __ldg(flag) == 0) return;
  __shared__ uint4 sh[4][32][17];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int NC = K / 32, NCG = NC / 4;
  const long long total_units = (long long)(N / 32) * NCG;
  const long long stride = (long long)gridDim.x * 4;
  for (long long unit = (long long)blockIdx.x * 4 + warp; unit < total_units; unit += stride) {
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
    __syncwarp();
  }
}

// Bounded-grid (160 CTAs) exact FP16 HMMA.884 GEMM (FP32 accum, 128x128 tile, KTT=64, SPR=2) gated on *flag != 0:
// ~1.5-2 us early exit when flag == 0; persistent grid-stride over 128x128 tiles when flag != 0.
// EPI == 0 (down): out[M][N] = fp16(alpha * X @ W^T).
// EPI == 1 (gate+up): W is k_dequant physical order (32-row blocks = 16 gate + 16 up); fused SiLU(gate) * up -> out[M][N/2].
template <int EPI>
__global__ void __launch_bounds__(256, 1)
k_gemm_exact_if(const int* __restrict__ flag, const __half* __restrict__ A, const __half* __restrict__ W,
                __half* __restrict__ out, float alpha, int M, int N, int K, int Nout, int zmask) {
  if (flag != nullptr && __ldg(flag) == 0) return;
  constexpr int RB = 4, CB = 2, WM = 4, WN = 2, KTT = 64;
  constexpr int BM = WM * RB * 8, BN = WN * CB * 32, NT = WM * WN * 32, NB = RB * CB, LD = KTT + 8, NK8 = KTT / 8;
  constexpr int LDO = BN + 8;
  constexpr int CA = BM * 4 / NT, CBn = BN * 4 / NT;
  static_assert(BM == 128 && BN == 128 && NT == 256 && CA == 2 && CBn == 2, "exact fallback geometry");
  extern __shared__ __align__(16) __half smem[];
  auto sA = [&](int b, int r, int k) { return smem + ((size_t)b * (BM + BN) + r) * LD + k; };
  auto sB = [&](int b, int r, int k) { return smem + ((size_t)b * (BM + BN) + BM + r) * LD + k; };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp % WM, wn = warp / WM;
  const int qp = (lane >> 2) & 3, arow = (lane & 3) + ((lane >> 4) << 2);
  const int brow = wn * CB * 32 + qp * 4 + (lane & 3) + ((lane >> 4) << 4);
  const int npm = (M + BM - 1) / BM, npn = N / BN, total_tiles = npm * npn, ntiles = K / KTT;
  constexpr int GM = 8;
  auto cmap = [&](int c, int p, int rows, int& r, int& kc) { r = p * (rows / 2) + (c >> 3); kc = (c & 7) * 8; };
  uint4 ra[CA], rb[CBn];
  uint4 fa[2][RB], fb[2][CB];
  float acc[NB][8];

  auto ldg = [&](int m0, int n0, int kt0, int p) {
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

  for (int pid = blockIdx.x; pid < total_tiles; pid += gridDim.x) {
    const int gid = pid / (GM * npn), first = gid * GM, gsz = min(npm - first, GM);
    const int pm = first + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    const int m0 = pm * BM, n0 = pn * BN;
#pragma unroll
    for (int i = 0; i < NB; i++)
#pragma unroll
      for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
    for (int p = 0; p < 2; p++) { ldg(m0, n0, 0, p); sts(0, p); }
    __syncthreads();
    ldf(0, 0, 0);
    for (int t = 0; t < ntiles; t++) {
      const int buf = t & 1;
      const bool more = t + 1 < ntiles;
#pragma unroll
      for (int k8 = 0; k8 < NK8; k8++) {
        const int cur = k8 & 1, nx = cur ^ 1;
        if (more && k8 % 4 == 0) ldg(m0, n0, (t + 1) * KTT, k8 / 4);
        if (more && k8 % 4 == 3) sts(buf ^ 1, k8 / 4);
        if (k8 == NK8 - 1 && more) { __syncthreads(); mma_k8_spread(cur, nx, buf ^ 1, 0); continue; }
        if (k8 < NK8 - 1) ldf(nx, buf, k8 + 1);
        mma_k8(cur);
      }
    }
    __syncthreads();
#pragma unroll
    for (int ib = 0; ib < NB; ib++) {
      const int i = ib / CB, cb = ib % CB;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int rl = wm * RB * 8 + i * 8 + (lane & 1) + ((lane >> 4) << 2) + (j & 2);
        const int c = (lane & 2) + (j & 4) + (j & 1);
        const int lcol = wn * CB * 32 + cb * 32 + qp * 4 + (c & 3) + 16 * (c >> 2);
        smem[rl * LDO + lcol] = __float2half_rn(acc[ib][j] * alpha);
      }
    }
    __syncthreads();
    if constexpr (EPI == 0) {
      constexpr int CPR = BN / 8;
      for (int c = tid; c < BM * CPR; c += NT) {
        const int r = c / CPR, qv = c % CPR;
        if (m0 + r < M)
          *reinterpret_cast<uint4*>(out + (size_t)(m0 + r) * Nout + n0 + qv * 8) =
              *reinterpret_cast<const uint4*>(smem + r * LDO + qv * 8);
      }
    } else {
      constexpr int VPR = (BN / 2) / 8;                            // 4 x 32-col blocks = 64 h channels = 8 uint4/row
      const int h0 = n0 >> 1;
      for (int c = tid; c < BM * VPR; c += NT) {
        const int r = c / VPR, v = c % VPR;
        if (m0 + r >= M) continue;
        const int blk = v >> 1, i0 = (v & 1) * 8;
        const uint4 gv = *reinterpret_cast<const uint4*>(smem + r * LDO + blk * 32 + i0);
        const uint4 uv = *reinterpret_cast<const uint4*>(smem + r * LDO + blk * 32 + 16 + i0);
        const __half* gh = reinterpret_cast<const __half*>(&gv);
        const __half* uh = reinterpret_cast<const __half*>(&uv);
        __align__(16) __half hv[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float a = __half2float(gh[e]), b = __half2float(uh[e]);
          hv[e] = __float2half_rn(a / (1.f + __expf(-a)) * b);
        }
        *reinterpret_cast<uint4*>(out + (size_t)(m0 + r) * Nout + h0 + blk * 16 + i0) = *reinterpret_cast<const uint4*>(hv);
      }
    }
    __syncthreads();                                               // wait for all epilogue reads before next pid reuses smem
  }
}

template <int EPI>
void launch_gemm_exact_if(const int* flag, const __half* A, const __half* W, __half* out, float alpha,
                          int M, int N, int K, int Nout, cudaStream_t st) {
  constexpr size_t SMm = (size_t)2 * (128 + 128) * (64 + 8) * 2;
  constexpr size_t SMe = (size_t)128 * (128 + 8) * 2;
  constexpr size_t SM = SMm > SMe ? SMm : SMe;
  static_assert(SM <= 96 * 1024, "exact fallback shared memory");
  auto kern = k_gemm_exact_if<EPI>;
  static unsigned set_dev_mask = 0u;
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 32 && !(set_dev_mask & (1u << dev))) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM) == cudaSuccess);
    set_dev_mask |= (1u << dev);
  }
  kern<<<160, 256, SM, st>>>(flag, A, W, out, alpha, M, N, K, Nout, 0);
}
}  // namespace ref
}  // namespace

// ------------------------------------------------------------------------------------------- host entry points
void quant_rows_h32(torch::Tensor x, torch::Tensor xq, torch::Tensor sx, double qmax, c10::optional<torch::Tensor> u,
                    c10::optional<torch::Tensor> r) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous() && x.dim() == 2 && x.size(1) % 32 == 0);
  TORCH_CHECK(xq.is_cuda() && xq.dtype() == torch::kHalf && xq.is_contiguous() && xq.sizes() == x.sizes());
  TORCH_CHECK(sx.is_cuda() && sx.dtype() == torch::kFloat && sx.is_contiguous() && sx.numel() >= x.size(0));
  const at::cuda::CUDAGuard guard(x.device());
  const int M = (int)x.size(0), K = (int)x.size(1);
  if (M == 0) return;
  auto X = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto Q = reinterpret_cast<__half*>(xq.data_ptr<at::Half>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const float qm = (float)qmax;
  TORCH_CHECK(qm >= 1.f && qm <= 127.f && qmax == std::floor(qmax), "quant_rows_h32: qmax must be integer in [1, 127]");
  const unsigned* U = u.has_value() ? reinterpret_cast<const unsigned*>(u->data_ptr<int>()) : nullptr;
  unsigned* Rr = r.has_value() ? reinterpret_cast<unsigned*>(r->data_ptr<int>()) : nullptr;
  TORCH_CHECK((U == nullptr) == (Rr == nullptr) &&
              (U == nullptr || (u->is_cuda() && u->dtype() == torch::kInt32 && u->is_contiguous() && u->numel() == K &&
                                r->is_cuda() && r->dtype() == torch::kInt32 && r->is_contiguous() && r->numel() >= M)),
              "quant_rows_h32: u / r");
  if (K == 5120) k_quant_h32<20><<<(M + 7) / 8, 256, 0, st>>>(X, Q, sx.data_ptr<float>(), M, K, qm, U, Rr);
  else if (K == 4352) k_quant_h32<17><<<(M + 7) / 8, 256, 0, st>>>(X, Q, sx.data_ptr<float>(), M, K, qm, U, Rr);
  else TORCH_CHECK(false, "quant_rows_h32: K must be 5120 or 4352");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void rowdot(torch::Tensor xq, torch::Tensor u, torch::Tensor r) {
  TORCH_CHECK(xq.is_cuda() && xq.dtype() == torch::kHalf && xq.is_contiguous() && xq.dim() == 2);
  const at::cuda::CUDAGuard guard(xq.device());
  const int M = (int)xq.size(0), K = (int)xq.size(1);
  if (M == 0) return;
  TORCH_CHECK(u.is_cuda() && u.dtype() == torch::kInt32 && u.is_contiguous() && u.numel() == K);
  TORCH_CHECK(r.is_cuda() && r.dtype() == torch::kInt32 && r.is_contiguous() && r.numel() >= M);
  auto X = reinterpret_cast<const __half*>(xq.data_ptr<at::Half>());
  auto U = reinterpret_cast<const unsigned*>(u.data_ptr<int>());
  auto Rr = reinterpret_cast<unsigned*>(r.data_ptr<int>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (K == 5120) k_rowdot<20><<<(M + 7) / 8, 256, 0, st>>>(X, U, Rr, M, K);
  else if (K == 4352) k_rowdot<17><<<(M + 7) / 8, 256, 0, st>>>(X, U, Rr, M, K);
  else TORCH_CHECK(false, "rowdot: K must be 5120 or 4352");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void check(torch::Tensor c, torch::Tensor r, int64_t M, torch::Tensor flag) {
  if (M <= 0) return;
  const at::cuda::CUDAGuard guard(c.device());
  const int nb = std::max(1, std::min(8, (int)((M + 1023) / 1024)));
  k_chk<<<nb, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned*>(c.data_ptr<int>()),
                                                          reinterpret_cast<const unsigned*>(r.data_ptr<int>()), (int)M,
                                                          flag.data_ptr<int>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void check_guard(torch::Tensor c, torch::Tensor r, int64_t M, torch::Tensor flag_cur,
                 c10::optional<torch::Tensor> flag_tot) {
  TORCH_CHECK(M >= 0, "check_guard: M < 0");
  TORCH_CHECK(c.is_cuda() && c.dtype() == torch::kInt32 && c.is_contiguous() && c.numel() >= M);
  TORCH_CHECK(r.is_cuda() && r.dtype() == torch::kInt32 && r.is_contiguous() && r.numel() >= M);
  TORCH_CHECK(flag_cur.is_cuda() && flag_cur.dtype() == torch::kInt32 && flag_cur.is_contiguous() && flag_cur.numel() >= 2);
  int* ft = nullptr;
  if (flag_tot.has_value()) {
    TORCH_CHECK(flag_tot->is_cuda() && flag_tot->dtype() == torch::kInt32 && flag_tot->is_contiguous() && flag_tot->numel() >= 2);
    ft = flag_tot->data_ptr<int>();
  }
  const at::cuda::CUDAGuard guard(c.device());
  k_chk_single<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned*>(c.data_ptr<int>()), reinterpret_cast<const unsigned*>(r.data_ptr<int>()), (int)M,
      flag_cur.data_ptr<int>(), ft);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void convert_h32(torch::Tensor w, c10::optional<torch::Tensor> sel, torch::Tensor ia, torch::Tensor out,
                 int64_t ntile, int64_t ngroups, int64_t K, bool plane, bool gated, int64_t cpb) {
  TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kUInt8 && w.is_contiguous());
  TORCH_CHECK(ia.is_cuda() && ia.dtype() == torch::kFloat && ia.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() && out.dim() == 2);
  TORCH_CHECK(out.size(0) % 64 == 0 && out.size(1) == K && K % 32 == 0);
  const at::cuda::CUDAGuard guard(w.device());
  const int gpad = (int)(out.size(0) / 64), NC = (int)(K / 32);
  TORCH_CHECK(ia.numel() >= (int64_t)gpad * 96, "convert_h32: ia too small");
  const uint8_t* S = nullptr;
  if (plane) {
    TORCH_CHECK(sel.has_value() && sel->is_cuda() && sel->dtype() == torch::kUInt8 && sel->is_contiguous() &&
                sel->numel() >= (int64_t)gpad * NC * 384, "convert_h32: sel tensor");
    S = sel->data_ptr<uint8_t>();
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto W = w.data_ptr<uint8_t>();
  auto I = ia.data_ptr<float>();
  auto O = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
#define CONV(PL_, GA_, CP_) \
  k_conv_h32<PL_, GA_, CP_><<<dim3(gpad, (NC + CP_ - 1) / CP_), 96, 0, st>>>(W, S, I, O, (int)ntile, (int)ngroups, (int)K)
#define CONV_C(PL_, GA_)                                                                        \
  do {                                                                                          \
    if (cpb == 2) CONV(PL_, GA_, 2);                                                            \
    else if (cpb == 4) CONV(PL_, GA_, 4);                                                       \
    else if (cpb == 6) CONV(PL_, GA_, 6);                                                       \
    else TORCH_CHECK(false, "convert_h32: cpb must be 2, 4, or 6");                             \
  } while (0)
  if (plane) { if (gated) CONV_C(1, 1); else CONV_C(1, 0); }
  else { if (gated) CONV_C(0, 1); else CONV_C(0, 0); }
#undef CONV_C
#undef CONV
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void gemm_i4(torch::Tensor xq, torch::Tensor wp, torch::Tensor sx, torch::Tensor ae, torch::Tensor out,
             int64_t epi, int64_t cfg, c10::optional<torch::Tensor> chkv, c10::optional<torch::Tensor> chkc) {
  TORCH_CHECK(xq.is_cuda() && xq.dtype() == torch::kHalf && xq.is_contiguous() && xq.dim() == 2);
  TORCH_CHECK(wp.is_cuda() && wp.dtype() == torch::kHalf && wp.is_contiguous() && wp.dim() == 2 && wp.size(1) == xq.size(1));
  TORCH_CHECK(sx.is_cuda() && sx.dtype() == torch::kFloat && sx.is_contiguous() && sx.numel() >= xq.size(0));
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 && out.size(0) == xq.size(0));
  const at::cuda::CUDAGuard guard(xq.device());
  const int M = (int)xq.size(0), K = (int)xq.size(1), Ncols = (int)wp.size(0), Nout = (int)out.size(1);
  if (M == 0) return;
  TORCH_CHECK(ae.is_cuda() && ae.dtype() == torch::kFloat && ae.is_contiguous() && ae.numel() >= (int64_t)Ncols / 2 * 3);
  TORCH_CHECK(epi == 0 || epi == 1 || epi == 2, "gemm_i4: epi must be 0, 1, or 2");
  TORCH_CHECK(epi != 2 || (out.dtype() == torch::kFloat && Nout == Ncols / 2 * 3), "gemm_i4: debug output");
  TORCH_CHECK(epi == 2 || (out.dtype() == torch::kHalf && Nout % 8 == 0), "gemm_i4: fp16 output (Nout multiple of 8)");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto A = reinterpret_cast<const __half*>(xq.data_ptr<at::Half>());
  auto W = reinterpret_cast<const __half*>(wp.data_ptr<at::Half>());
  auto O = reinterpret_cast<__half*>(out.data_ptr());
  auto S = sx.data_ptr<float>();
  auto E = ae.data_ptr<float>();
  const bool CK = chkv.has_value();
  TORCH_CHECK(CK == chkc.has_value() &&
              (!CK || (chkv->is_cuda() && chkv->dtype() == torch::kInt32 && chkv->is_contiguous() &&
                       chkv->numel() >= (int64_t)Ncols / 2 * 3 &&
                       chkc->is_cuda() && chkc->dtype() == torch::kInt32 && chkc->is_contiguous() &&
                       chkc->numel() >= M)),
              "gemm_i4: checksum arrays");
  const unsigned* CV = CK ? reinterpret_cast<const unsigned*>(chkv->data_ptr<int>()) : nullptr;
  unsigned* CC = CK ? reinterpret_cast<unsigned*>(chkc->data_ptr<int>()) : nullptr;

#define DISPATCH_C(RB_, CB_, FP_, FU_, DM_)                                                                               \
  do {                                                                                                                    \
    if (epi == 1) launch_pk_i4<RB_, CB_, 4, 2, 1, 2, FP_, FU_, DM_, 1>(A, W, O, S, E, M, Ncols, K, Nout, st, CV, CC);     \
    else launch_pk_i4<RB_, CB_, 4, 2, 0, 2, FP_, FU_, DM_, 1>(A, W, O, S, E, M, Ncols, K, Nout, st, CV, CC);              \
  } while (0)
#define DISPATCH_N(RB_, CB_, FP_, FU_, DM_)                                                                               \
  do {                                                                                                                    \
    if (epi == 1) launch_pk_i4<RB_, CB_, 4, 2, 1, 2, FP_, FU_, DM_, 0>(A, W, O, S, E, M, Ncols, K, Nout, st);             \
    else if (epi == 2) launch_pk_i4<RB_, CB_, 4, 2, 2, 2, FP_, FU_, DM_, 0>(A, W, O, S, E, M, Ncols, K, Nout, st);        \
    else launch_pk_i4<RB_, CB_, 4, 2, 0, 2, FP_, FU_, DM_, 0>(A, W, O, S, E, M, Ncols, K, Nout, st);                      \
  } while (0)

  if (CK) {
    TORCH_CHECK(epi != 2, "gemm_i4: no checksum in debug epilogue");
    switch (cfg) {
      case 4:  DISPATCH_C(4, 2, 16, 16, 1); break;   // 128x128 r1024u1024 burst (small M)
      case 5:  DISPATCH_C(4, 2,  8,  8, 1); break;   // 128x128 r512u512 burst (small M, INT8)
      case 40: DISPATCH_C(4, 3, 16, 16, 1); break;   // 128x192 r1024u1024 burst
      case 41: DISPATCH_C(4, 3, 16, 16, 2); break;   // 128x192 r1024u1024 interleaved
      case 42: DISPATCH_C(4, 3,  8,  8, 1); break;   // 128x192 r512u512 burst
      case 43: DISPATCH_C(4, 3,  8, 16, 1); break;   // 128x192 r512u1024 burst
      case 44: DISPATCH_C(4, 3, 32, 32, 1); break;   // 128x192 r2048u2048 burst (unsafe guard test)
      case 45: DISPATCH_C(4, 3, 16, 32, 1); break;   // 128x192 r1024u2048 burst
      default: TORCH_CHECK(false, "gemm_i4: unknown cfg ", cfg);
    }
  } else {
    switch (cfg) {
      case 4:  DISPATCH_N(4, 2, 16, 16, 1); break;
      case 5:  DISPATCH_N(4, 2,  8,  8, 1); break;
      case 40: DISPATCH_N(4, 3, 16, 16, 1); break;
      case 41: DISPATCH_N(4, 3, 16, 16, 2); break;
      case 42: DISPATCH_N(4, 3,  8,  8, 1); break;
      case 43: DISPATCH_N(4, 3,  8, 16, 1); break;
      case 44: DISPATCH_N(4, 3, 32, 32, 1); break;
      case 45: DISPATCH_N(4, 3, 16, 32, 1); break;
      default: TORCH_CHECK(false, "gemm_i4: unknown cfg ", cfg);
    }
  }
#undef DISPATCH_C
#undef DISPATCH_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

torch::Tensor prepack(torch::Tensor codes, torch::Tensor scales, bool gated) {
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.dim() == 2 && codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 && scales.dim() == 2 && scales.is_contiguous());
  const int N = (int)codes.size(0), K = (int)codes.size(1) * 2;
  TORCH_CHECK(N % 32 == 0 && K % 32 == 0 && scales.size(0) == N && scales.size(1) == K / 16);
  const at::cuda::CUDAGuard guard(codes.device());
  auto out = torch::empty({(long long)(N / 32) * (K / 32) * CHUNK}, codes.options());
  const long long total = (long long)(N / 32) * (K / 32) * 32;
  ref::k_prepack<<<(unsigned)((total + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), N, K, gated ? 1 : 0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
  return out;
}

void dequant_nv(torch::Tensor w, torch::Tensor out, int64_t N, int64_t K) {
  const at::cuda::CUDAGuard guard(w.device());
  TORCH_CHECK(out.dtype() == torch::kHalf && out.numel() >= N * K && (K / 32) % 4 == 0);
  const long long t = (long long)(N / 32) * (K / 128);
  ref::k_dequant<<<(unsigned)((t + 3) / 4), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      w.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)N, (int)K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void silu_pair(torch::Tensor gu, torch::Tensor out) {
  const at::cuda::CUDAGuard guard(gu.device());
  const int M = (int)gu.size(0), I = (int)out.size(1);
  if (M == 0) return;
  const long long ts = (long long)M * (I / 8);
  ref::k_silu_pair<<<(unsigned)((ts + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(gu.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, I);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void gemm_nt(torch::Tensor w, torch::Tensor x, torch::Tensor c, int64_t N, int64_t K, double alpha) {
  const at::cuda::CUDAGuard guard(x.device());
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
  const int M = (int)x.size(0);
  if (M == 0) return;
  const float a = (float)alpha, beta = 0.f;
  TORCH_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, M, (int)K, &a, w.data_ptr(), CUDA_R_16F, (int)K, x.data_ptr(),
                           CUDA_R_16F, (int)K, &beta, c.data_ptr(), CUDA_R_16F, (int)N, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS, "gemm_nt: cublasGemmEx failed");
}

void dequant_nv_if(c10::optional<torch::Tensor> flag, torch::Tensor w, torch::Tensor out, int64_t N, int64_t K) {
  TORCH_CHECK(w.is_cuda() && w.dtype() == torch::kUInt8 && w.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() && out.numel() >= N * K);
  TORCH_CHECK(N > 0 && K > 0 && N % 32 == 0 && K % 128 == 0, "dequant_nv_if: positive N % 32 == 0 and K % 128 == 0 required");
  const int* fp = nullptr;
  if (flag.has_value()) {
    TORCH_CHECK(flag->is_cuda() && flag->dtype() == torch::kInt32 && flag->numel() >= 1);
    fp = flag->data_ptr<int>();
  }
  const at::cuda::CUDAGuard guard(w.device());
  ref::k_dequant_if<<<160, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      fp, w.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)N, (int)K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void gemm_exact_if(c10::optional<torch::Tensor> flag, torch::Tensor x, torch::Tensor w_fp16, torch::Tensor out,
                   double alpha, int64_t N, int64_t K, int64_t epi) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous() && x.dim() == 2 && x.size(1) == K);
  TORCH_CHECK(w_fp16.is_cuda() && w_fp16.dtype() == torch::kHalf && w_fp16.is_contiguous() && w_fp16.numel() >= N * K);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() && out.dim() == 2 && out.size(0) == x.size(0));
  TORCH_CHECK(N > 0 && K > 0 && N % 128 == 0 && K % 64 == 0, "gemm_exact_if: positive N % 128 == 0 and K % 64 == 0 required");
  TORCH_CHECK(epi == 0 || epi == 1, "gemm_exact_if: epi must be 0 or 1");
  const int M = (int)x.size(0), Nout = (int)out.size(1);
  TORCH_CHECK(Nout == (epi == 1 ? N / 2 : N), "gemm_exact_if: out cols mismatch");
  if (M == 0) return;
  const int* fp = nullptr;
  if (flag.has_value()) {
    TORCH_CHECK(flag->is_cuda() && flag->dtype() == torch::kInt32 && flag->numel() >= 1);
    fp = flag->data_ptr<int>();
  }
  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto A = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto W = reinterpret_cast<const __half*>(w_fp16.data_ptr<at::Half>());
  auto O = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  if (epi == 1) ref::launch_gemm_exact_if<1>(fp, A, W, O, (float)alpha, M, (int)N, (int)K, Nout, st);
  else ref::launch_gemm_exact_if<0>(fp, A, W, O, (float)alpha, M, (int)N, (int)K, Nout, st);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_rows_h32", &quant_rows_h32, "online H_32 (nosign) + per-token INT7/INT8 quantization + optional Freivalds ref",
        pybind11::arg("x"), pybind11::arg("xq"), pybind11::arg("sx"), pybind11::arg("qmax") = 127.0,
        pybind11::arg("u") = pybind11::none(), pybind11::arg("r") = pybind11::none());
  m.def("rowdot", &rowdot, "reference checksums sum_k xq[row, k] u[k] mod 2^32");
  m.def("check", &check, "compare checksums (zeroes c), flag[0] |= mismatch, flag[1] += rows");
  m.def("check_guard", &check_guard, "single-CTA zero-sync checksum check: overwrites flag_cur[0..1], updates optional flag_tot[0..1]",
        pybind11::arg("c"), pybind11::arg("r"), pybind11::arg("M"), pybind11::arg("flag_cur"),
        pybind11::arg("flag_tot") = pybind11::none());
  m.def("convert_h32", &convert_h32, "decode-layout NVFP4 -> online H_32 INT4 [-7,7] -> (1, 256) packed scratch",
        pybind11::arg("w"), pybind11::arg("sel"), pybind11::arg("ia"), pybind11::arg("out"),
        pybind11::arg("ntile"), pybind11::arg("ngroups"), pybind11::arg("K"),
        pybind11::arg("plane"), pybind11::arg("gated"), pybind11::arg("cpb") = 4);
  m.def("gemm_i4", &gemm_i4, "INT4 (1, 256) FOLD packed GEMM with optional Freivalds guard",
        pybind11::arg("xq"), pybind11::arg("wp"), pybind11::arg("sx"), pybind11::arg("ae"), pybind11::arg("out"),
        pybind11::arg("epi"), pybind11::arg("cfg"),
        pybind11::arg("chkv") = pybind11::none(), pybind11::arg("chkc") = pybind11::none());
  m.def("prepack", &prepack, "NVFP4 checkpoint layout -> decode 32-col tile layout (mlp_fp4.cu compatible)");
  m.def("dequant_nv", &dequant_nv, "NVFP4 decode layout -> fp16 [N][K] physical order");
  m.def("dequant_nv_if", &dequant_nv_if, "bounded-grid (160 CTAs) device-gated NVFP4 dequant when flag[0] != 0",
        pybind11::arg("flag"), pybind11::arg("w"), pybind11::arg("out"), pybind11::arg("N"), pybind11::arg("K"));
  m.def("gemm_exact_if", &gemm_exact_if, "bounded-grid (160 CTAs) device-gated exact FP16 HMMA GEMM (+ optional SiLU*up) when flag[0] != 0",
        pybind11::arg("flag"), pybind11::arg("x"), pybind11::arg("w_fp16"), pybind11::arg("out"),
        pybind11::arg("alpha"), pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("epi"));
  m.def("silu_pair", &silu_pair, "SiLU(gate) * up on physical gate+up order");
  m.def("gemm_nt", &gemm_nt, "cuBLAS FP16 GEMM (FP32 accum): C = alpha * X @ W^T");
}
