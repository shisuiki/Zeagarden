// pf_arn.cu (track prefill2) - the prefill (> 256 rows) TP4 all-reduce + residual add + Gemma RMSNorm boundary as ONE
// kernel over NVLink-mapped (cudaIpc) buffers: reduce-scatter by row quarters, the norm on the owner's quarter only,
// all-gather of the normalised rows. Replaces 1Cat's fallback above its 64-row fused op (pynccl all-reduce, 572 us at
// 4,096 rows, + the separate add + norm pass over all rows, 312 us).
//
// Rows: quarter q = rows [q MQ, min(M, (q+1) MQ)), MQ = ceil(M / 4), owned by rank q; blocks of RB rows inside a quarter.
// Roles (CTAs of one launch, all co-resident; dependencies through epoch flags):
//   pushers  copy the local partial's rows of quarter q (q != rank) into rank q's receive slot of this rank, block by
//            block (peers interleaved), then publish flag1[q][src = rank][block] = epoch (stores fenced at system scope);
//   reducers for each block of the own quarter: wait for the three peers' flags, sum the four fp16 partials in FP32 in rank
//            order 0..3 (the own partial read in place), round to fp16 (= an fp16 all-reduce result), add the FP32
//            residual, write the new residual (OWN QUARTER ONLY: the residual stays row-sharded between these calls),
//            Gemma RMSNorm (y = v * rsqrt(mean(v^2) + eps) * (1 + w)), write the fp16 rows to the output and to the own
//            norm buffer, publish flag2[peer][src = rank][block] = epoch on every peer;
//   pullers  for every peer p and block of quarter p: wait for flag2, copy the peer's normalised rows into the output.
// Buffers per rank (one cudaMalloc, exported): recv [3 slots][MAXQ][H] fp16 (slot of src p at rank q: (p - q - 1) mod 4),
// norm [MAXQ][H] fp16, flag1 [4][NBMAX] u32, flag2 [4][NBMAX] u32. A single set suffices: rank q rewrites its norm buffer
// only after every peer's next partial arrived (they finished pulling), and a peer pushes the next partial into q's slot
// only after it pulled q's rows of this call (q finished reading the slot before publishing them).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#define CK(x) TORCH_CHECK((x) == cudaSuccess, #x " failed: ", cudaGetErrorString(cudaGetLastError()))

namespace {
constexpr int H = 5120, NV = H / 8, NRK = 4;
constexpr float INV_SQRT32 = 0.17677669529663687f;   // [prefill4] exact f32 1 / sqrt(32) = 0x3e3504f3
constexpr int MAXM = 4096 + 512, MAXQ = MAXM / 4, RBMIN = 4, NBMAX = MAXQ / RBMIN;
constexpr size_t RECV_HALVES = (size_t)3 * MAXQ * H, NORM_HALVES = (size_t)MAXQ * H;
constexpr size_t FLAG_OFF = (RECV_HALVES + NORM_HALVES) * 2, FLAG_WORDS = (size_t)2 * NRK * NBMAX;
constexpr size_t AM_OFF = FLAG_OFF + FLAG_WORDS * 4;               // [prefill3] Job 5: row amax of the norm buffer
constexpr size_t BUF_BYTES = AM_OFF + (size_t)MAXQ * 4;

struct Comm {
  char* base[NRK];
  int rank;
};
Comm g_comm;
bool g_init = false;

__device__ __forceinline__ __half* recv_of(const Comm& c, int q) { return reinterpret_cast<__half*>(c.base[q]); }
__device__ __forceinline__ __half* norm_of(const Comm& c, int q) { return reinterpret_cast<__half*>(c.base[q]) + RECV_HALVES; }
__device__ __forceinline__ unsigned* flag1_of(const Comm& c, int q) { return reinterpret_cast<unsigned*>(c.base[q] + FLAG_OFF); }
__device__ __forceinline__ unsigned* flag2_of(const Comm& c, int q) { return flag1_of(c, q) + NRK * NBMAX; }
__device__ __forceinline__ float* am_of(const Comm& c, int q) { return reinterpret_cast<float*>(c.base[q] + AM_OFF); }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void wait_flag(const unsigned* p, unsigned epoch) {
  while (ld_acq(p) != epoch) __nanosleep(64);
}
__device__ __forceinline__ uint4 ldcg4(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }

struct Args {
  Comm c;
  const __half* x;        // [M][H] local partial
  const float* res;       // [M][H] residual (own quarter read)
  float* res_out;         // [M][H] (own quarter written)
  const __half* w;        // [H] Gemma weight (y uses 1 + w)
  __half* out;            // [M][H] normalised
  int M, MQ, RB, NB;      // rows, rows per quarter, rows per block, blocks per quarter
  int GP, GR, GA;         // CTAs per role
  unsigned epoch;
  float eps;
  __half* xq;             // [prefill3] Job 5: per-token integer rows of out (fp16 integers) or nullptr
  float* sx;              // [M] scales (amax / qm)
  float qm;               // integer range (127: INT8), the same arithmetic as pk_mlp.cu k_quant
  const unsigned* u;      // [H] or nullptr: + the packed GEMM's reference checksum R[row] = sum_k xq u[k] mod 2^32
  unsigned* R;
};

__device__ __forceinline__ int qrows(const Args& a, int q) { return max(0, min(a.M, (q + 1) * a.MQ) - q * a.MQ); }

// [prefill4] Exact 5-stage H_32 butterfly + finite FP16 clamp across 4 consecutive lanes (lane & 3 == 0..3; 8 elems/lane
// = 32 elems / 4 lanes, NV = 640 = 20 full warps = 160 H_32 blocks). Bit-identical to pk_h32.cu k_quant_h32.
__device__ __forceinline__ void h32_rotate_vec8(const uint4& in, __half2 (&hr)[4], int lane) {
  const __half2* h = reinterpret_cast<const __half2*>(&in);
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
  const bool hi1 = (lane & 1) != 0;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float peer = __shfl_xor_sync(0xffffffffu, f[e], 1);
    f[e] = hi1 ? (peer - f[e]) : (f[e] + peer);
  }
  // Stage 5 (stride 16): xor lane bit 1
  const bool hi2 = (lane & 2) != 0;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float peer = __shfl_xor_sync(0xffffffffu, f[e], 2);
    f[e] = hi2 ? (peer - f[e]) : (f[e] + peer);
  }
  // Scale by 1/sqrt(32), clamp to finite FP16 range before __half2 packing (matches k_quant_h32)
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const float r0 = fminf(65504.f, fmaxf(-65504.f, f[2 * e] * INV_SQRT32));
    const float r1 = fminf(65504.f, fmaxf(-65504.f, f[2 * e + 1] * INV_SQRT32));
    hr[e] = __floats2half2_rn(r0, r1);
  }
}

// pushers: units u = 3 j + i (block j, peer i), rows of quarter q = (rank + 1 + i) % 4
__device__ __forceinline__ void copy_rows(uint4* __restrict__ dst, const uint4* __restrict__ src, int n, bool cg) {
  constexpr int U = 4;
  int t = threadIdx.x;
  for (; t + (U - 1) * NV < n; t += U * NV) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = cg ? ldcg4(src + t + u * NV) : __ldcs(src + t + u * NV);
#pragma unroll
    for (int u = 0; u < U; u++) dst[t + u * NV] = v[u];
  }
  for (; t < n; t += NV) dst[t] = cg ? ldcg4(src + t) : __ldcs(src + t);
}
__device__ void pusher(const Args& a, int cta) {
  const int r = a.c.rank;
  const int nu = 3 * a.NB;
  for (int u = cta; u < nu; u += a.GP) {
    const int j = u / 3, i = u % 3, q = (r + 1 + i) & 3;
    const int nr = min(a.RB, qrows(a, q) - j * a.RB);
    if (nr > 0) {
      const int slot = (r - q - 1 + 8) & 3;                         // my slot at rank q
      const uint4* src = reinterpret_cast<const uint4*>(a.x + (size_t)(q * a.MQ + j * a.RB) * H);
      uint4* dst = reinterpret_cast<uint4*>(recv_of(a.c, q) + ((size_t)slot * MAXQ + j * a.RB) * H);
      copy_rows(dst, src, nr * NV, false);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      st_rel(flag1_of(a.c, q) + r * NBMAX + j, a.epoch);
    }
  }
}

// reducers: blocks j of the own quarter; one row at a time, thread t = 16-byte unit t of the row (NV = 640 threads)
template <int H32>
__device__ void reducer(const Args& a, int cta, float* red) {
  const int r = a.c.rank, t = threadIdx.x, lane = t & 31, wp = t >> 5;
  const int nq = qrows(a, r);
  const __half* rv = recv_of(a.c, r);
  __half* nb = norm_of(a.c, r);
  const unsigned* f1 = flag1_of(a.c, r);
  float wg[8];
  {
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(a.w) + t);
    const __half2* h = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
    for (int e = 0; e < 4; e++) { const float2 f = __half22float2(h[e]); wg[2 * e] = 1.f + f.x; wg[2 * e + 1] = 1.f + f.y; }
  }
  unsigned uw[8];
  if (a.u != nullptr) {
    const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * t), u1 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * t + 1);
    uw[0] = u0.x; uw[1] = u0.y; uw[2] = u0.z; uw[3] = u0.w; uw[4] = u1.x; uw[5] = u1.y; uw[6] = u1.z; uw[7] = u1.w;
  }
  for (int j = cta; j < a.NB; j += a.GR) {
    const int nr = min(a.RB, nq - j * a.RB);
    if (nr <= 0) break;
    if (t < 3) wait_flag(f1 + ((r + 1 + t) & 3) * NBMAX + j, a.epoch);
    __syncthreads();
    for (int rr = 0; rr < nr; rr++) {
      const int qrow = j * a.RB + rr, row = r * a.MQ + qrow;
      // partials in rank order 0..3
      uint4 pv[4];
#pragma unroll
      for (int p = 0; p < 4; p++) {
        if (p == r) pv[p] = ldcg4(a.x + (size_t)row * H + 8 * t);
        else pv[p] = ldcg4(rv + ((size_t)((p - r - 1 + 8) & 3) * MAXQ + qrow) * H + 8 * t);
      }
      float s[8];
#pragma unroll
      for (int e = 0; e < 8; e++) s[e] = 0.f;
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const __half2* h = reinterpret_cast<const __half2*>(&pv[p]);
#pragma unroll
        for (int e = 0; e < 4; e++) { const float2 f = __half22float2(h[e]); s[2 * e] += f.x; s[2 * e + 1] += f.y; }
      }
      const float4* rp = reinterpret_cast<const float4*>(a.res + (size_t)row * H + 8 * t);
      const float4 r0 = __ldcs(rp), r1 = __ldcs(rp + 1);
      const float rs[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
      float v[8], ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        v[e] = __half2float(__float2half_rn(s[e])) + rs[e];        // the fp16 all-reduce result + FP32 residual
        ss += v[e] * v[e];
      }
      float4* ro = reinterpret_cast<float4*>(a.res_out + (size_t)row * H + 8 * t);
      __stcs(ro, make_float4(v[0], v[1], v[2], v[3]));
      __stcs(ro + 1, make_float4(v[4], v[5], v[6], v[7]));
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) red[wp] = ss;
      __syncthreads();
      float tot = 0.f;
#pragma unroll
      for (int k = 0; k < NV / 32; k++) tot += red[k];
      const float inv = rsqrtf(tot / (float)H + a.eps);
      __align__(16) __half o[8];
#pragma unroll
      for (int e = 0; e < 8; e++) o[e] = __float2half_rn(v[e] * inv * wg[e]);
      const uint4 ov = *reinterpret_cast<const uint4*>(o);
      *reinterpret_cast<uint4*>(a.out + (size_t)row * H + 8 * t) = ov;
      *reinterpret_cast<uint4*>(nb + (size_t)qrow * H + 8 * t) = ov;
      if (a.xq != nullptr) {                                        // [prefill3/4] Job 5: k_quant / k_quant_h32 arithmetic
        __align__(16) __half2 hbuf[4];
        const __half2* h2 = nullptr;
        if constexpr (H32) {
          h32_rotate_vec8(ov, hbuf, lane);
          h2 = hbuf;
        } else {
          h2 = reinterpret_cast<const __half2*>(&ov);
        }
        float am = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) { const float2 f = __half22float2(h2[e]); am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y))); }
#pragma unroll
        for (int o2 = 16; o2; o2 >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o2));
        __syncthreads();                                            // everyone has read red[] (the sum of squares)
        if (lane == 0) red[wp] = am;
        __syncthreads();
        am = 0.f;
#pragma unroll
        for (int k = 0; k < NV / 32; k++) am = fmaxf(am, red[k]);
        const float qinv = am > 0.f ? (H32 ? __fdiv_rn(a.qm, am) : (a.qm / am)) : 0.f;
        __align__(16) __half2 q2[4];
        float qv[8];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float2 f = __half22float2(h2[e]);
          if constexpr (H32) {
            qv[2 * e] = fminf(a.qm, fmaxf(-a.qm, rintf(f.x * qinv)));
            qv[2 * e + 1] = fminf(a.qm, fmaxf(-a.qm, rintf(f.y * qinv)));
          } else {
            qv[2 * e] = rintf(f.x * qinv);
            qv[2 * e + 1] = rintf(f.y * qinv);
          }
          q2[e] = __floats2half2_rn(qv[2 * e], qv[2 * e + 1]);
        }
        *reinterpret_cast<uint4*>(a.xq + (size_t)row * H + 8 * t) = *reinterpret_cast<const uint4*>(q2);
        if (t == 0) { a.sx[row] = H32 ? __fdiv_rn(am, a.qm) : (am / a.qm); am_of(a.c, r)[qrow] = am; }
        if (a.u != nullptr) {                                       // [prefill3/4] Job 4 reference checksum of the row
          unsigned ref = 0u;
#pragma unroll
          for (int e = 0; e < 8; e++) ref += (unsigned)__float2int_rn(qv[e]) * uw[e];
#pragma unroll
          for (int o2 = 16; o2; o2 >>= 1) ref += __shfl_xor_sync(0xffffffffu, ref, o2);
          __syncthreads();                                          // red[] (the amax) read by everyone
          if (lane == 0) reinterpret_cast<unsigned*>(red)[wp] = ref;
          __syncthreads();
          if (t == 0) {
            unsigned tot = 0u;
#pragma unroll
            for (int k = 0; k < NV / 32; k++) tot += reinterpret_cast<unsigned*>(red)[k];
            a.R[row] = tot;
          }
        }
      }
      __syncthreads();                                              // red[] reuse
    }
    if (t < 3) {
      __threadfence_system();
      st_rel(flag2_of(a.c, (r + 1 + t) & 3) + r * NBMAX + j, a.epoch);
    }
  }
}

// pullers: units u = 3 j + i (block j of peer p = (rank + 1 + i) % 4's quarter)
template <int H32>
__device__ void puller(const Args& a, int cta) {
  const int r = a.c.rank, lane = threadIdx.x & 31;
  const unsigned* f2 = flag2_of(a.c, r);
  const int nu = 3 * a.NB;
  unsigned uw[8];
  if (a.u != nullptr) {
    const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * threadIdx.x);
    const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * threadIdx.x + 1);
    uw[0] = u0.x; uw[1] = u0.y; uw[2] = u0.z; uw[3] = u0.w; uw[4] = u1.x; uw[5] = u1.y; uw[6] = u1.z; uw[7] = u1.w;
  }
  for (int u = cta; u < nu; u += a.GA) {
    const int j = u / 3, i = u % 3, p = (r + 1 + i) & 3;
    const int nr = min(a.RB, qrows(a, p) - j * a.RB);
    if (nr <= 0) continue;
    if (threadIdx.x == 0) wait_flag(f2 + p * NBMAX + j, a.epoch);
    __syncthreads();
    const uint4* src = reinterpret_cast<const uint4*>(norm_of(a.c, p) + (size_t)(j * a.RB) * H);
    uint4* dst = reinterpret_cast<uint4*>(a.out + (size_t)(p * a.MQ + j * a.RB) * H);
    if (a.xq == nullptr) {
      copy_rows(dst, src, nr * NV, true);
    } else {                                                        // [prefill3/4] Job 5: copy + owner's row amax -> xq
      __shared__ float amr[64];
      __shared__ unsigned rref[64];
      if (threadIdx.x < nr) {
        rref[threadIdx.x] = 0u;
        const float am = __ldcg(am_of(a.c, p) + j * a.RB + threadIdx.x);
        amr[threadIdx.x] = am;
        a.sx[p * a.MQ + j * a.RB + threadIdx.x] = H32 ? __fdiv_rn(am, a.qm) : (am / a.qm);
      }
      __syncthreads();
      uint4* dq = reinterpret_cast<uint4*>(a.xq + (size_t)(p * a.MQ + j * a.RB) * H);
      constexpr int U = 4;                                          // thread t = 16-byte unit t of a row (NV = 640)
      for (int r0 = 0; r0 < nr; r0 += U) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) if (r0 + u < nr) v[u] = ldcg4(src + (size_t)(r0 + u) * NV + threadIdx.x);
#pragma unroll
        for (int u = 0; u < U; u++) {
          if (r0 + u >= nr) break;
          const size_t off = (size_t)(r0 + u) * NV + threadIdx.x;
          dst[off] = v[u];
          const float am = amr[r0 + u];
          const float inv = am > 0.f ? (H32 ? __fdiv_rn(a.qm, am) : (a.qm / am)) : 0.f;
          __align__(16) __half2 hbuf[4];
          const __half2* h2 = nullptr;
          if constexpr (H32) {
            h32_rotate_vec8(v[u], hbuf, lane);
            h2 = hbuf;
          } else {
            h2 = reinterpret_cast<const __half2*>(&v[u]);
          }
          __align__(16) __half2 q2[4];
          float qv[8];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 f = __half22float2(h2[e]);
            if constexpr (H32) {
              qv[2 * e] = fminf(a.qm, fmaxf(-a.qm, rintf(f.x * inv)));
              qv[2 * e + 1] = fminf(a.qm, fmaxf(-a.qm, rintf(f.y * inv)));
            } else {
              qv[2 * e] = rintf(f.x * inv);
              qv[2 * e + 1] = rintf(f.y * inv);
            }
            q2[e] = __floats2half2_rn(qv[2 * e], qv[2 * e + 1]);
          }
          dq[off] = *reinterpret_cast<const uint4*>(q2);
          if (a.u != nullptr) {                                     // [prefill3/4] Job 4 reference checksum
            unsigned ref = 0u;
#pragma unroll
            for (int e = 0; e < 8; e++) ref += (unsigned)__float2int_rn(qv[e]) * uw[e];
#pragma unroll
            for (int o2 = 16; o2; o2 >>= 1) ref += __shfl_xor_sync(0xffffffffu, ref, o2);
            if (lane == 0) atomicAdd(&rref[r0 + u], ref);
          }
        }
      }
      if (a.u != nullptr) {
        __syncthreads();
        if (threadIdx.x < nr) a.R[p * a.MQ + j * a.RB + threadIdx.x] = rref[threadIdx.x];
      }
    }
    __syncthreads();
  }
}

template <int H32>
__global__ void __launch_bounds__(NV) k_arn(Args a) {              // 640 threads per CTA, every role
  __shared__ float red[32];
  const int b = blockIdx.x;
  if (b < a.GP) pusher(a, b);
  else if (b < a.GP + a.GR) reducer<H32>(a, b - a.GP, red);
  else puller<H32>(a, b - a.GP - a.GR);
}
}  // namespace

// ------------------------------------------------------------------------------------------------ host
int64_t buf_bytes() { return (int64_t)BUF_BYTES; }
int64_t max_rows() { return MAXM; }
int64_t alloc_buf() {
  void* p;
  CK(cudaMalloc(&p, BUF_BYTES));
  CK(cudaMemset(p, 0, BUF_BYTES));
  CK(cudaDeviceSynchronize());
  return (int64_t)p;
}
py::bytes ipc_handle(int64_t ptr) {
  cudaIpcMemHandle_t h;
  CK(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(ptr)));
  return py::bytes(reinterpret_cast<const char*>(&h), sizeof(h));
}
int64_t ipc_open(py::bytes b) {
  std::string s = b;
  TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t h;
  memcpy(&h, s.data(), sizeof(h));
  void* p;
  CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)p;
}
void init(std::vector<int64_t> bases, int64_t rank) {
  TORCH_CHECK(bases.size() == NRK);
  for (int q = 0; q < NRK; q++) g_comm.base[q] = reinterpret_cast<char*>(bases[q]);
  g_comm.rank = (int)rank;
  g_init = true;
}

// normalized [M][H] fp16 (new), residual_out [M][H] fp32 (new; own quarter rows valid)
std::vector<torch::Tensor> arn(torch::Tensor x, torch::Tensor res, torch::Tensor w, double eps, int64_t epoch,
                               int64_t rb, int64_t gp, int64_t gr, int64_t ga, double qmax,   // [prefill3] qmax > 0: + xq, sx
                               c10::optional<torch::Tensor> u, c10::optional<torch::Tensor> R, bool h32) {   // [prefill4] + h32
  TORCH_CHECK(g_init, "pf_arn: not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.size(1) == H && x.is_contiguous());
  TORCH_CHECK(res.dtype() == torch::kFloat && res.sizes() == x.sizes() && res.is_contiguous());
  TORCH_CHECK(w.dtype() == torch::kHalf && w.numel() == H && w.is_contiguous());
  const int M = x.size(0);
  TORCH_CHECK(M >= 4 && M <= MAXM, "pf_arn: rows");
  TORCH_CHECK(rb >= RBMIN && rb % RBMIN == 0);
  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  auto res_out = torch::empty_like(res);
  Args a;
  a.c = g_comm; a.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>()); a.res = res.data_ptr<float>();
  a.res_out = res_out.data_ptr<float>(); a.w = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  a.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  a.M = M; a.MQ = (M + 3) / 4; a.RB = (int)rb; a.NB = (a.MQ + a.RB - 1) / a.RB;
  TORCH_CHECK(a.NB <= NBMAX);
  a.GP = (int)gp; a.GR = (int)gr; a.GA = (int)ga; a.epoch = (unsigned)epoch; a.eps = (float)eps;
  torch::Tensor xq, sx;                                             // [prefill3] Job 5
  a.xq = nullptr; a.sx = nullptr; a.qm = (float)qmax; a.u = nullptr; a.R = nullptr;
  if (u.has_value()) {
    TORCH_CHECK(qmax > 0 && R.has_value() && u->numel() == H && R->numel() >= M, "pf_arn: checksum arrays");
    a.u = reinterpret_cast<const unsigned*>(u->data_ptr<int>()); a.R = reinterpret_cast<unsigned*>(R->data_ptr<int>());
  }
  if (qmax > 0) {
    TORCH_CHECK(rb <= 64, "pf_arn: fused quantisation needs <= 64 rows per block");
    if (h32) TORCH_CHECK(a.qm >= 1.f && a.qm <= 127.f && qmax == std::floor(qmax), "pf_arn: h32 qmax must be integer in [1, 127]");
    xq = torch::empty_like(x);
    sx = torch::empty({M}, x.options().dtype(torch::kFloat));
    a.xq = reinterpret_cast<__half*>(xq.data_ptr<at::Half>()); a.sx = sx.data_ptr<float>();
  }
  if (h32 && qmax > 0) k_arn<1><<<a.GP + a.GR + a.GA, NV, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  else k_arn<0><<<a.GP + a.GR + a.GA, NV, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  CK(cudaGetLastError());
  if (qmax > 0) return {out, res_out, xq, sx};
  return {out, res_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buf_bytes", &buf_bytes); m.def("max_rows", &max_rows); m.def("alloc_buf", &alloc_buf);
  m.def("ipc_handle", &ipc_handle); m.def("ipc_open", &ipc_open); m.def("init", &init);
  m.def("arn", &arn, "prefill TP4 all-reduce + residual add + Gemma RMSNorm (row-sharded residual)", pybind11::arg("x"),
        pybind11::arg("res"), pybind11::arg("w"), pybind11::arg("eps"), pybind11::arg("epoch"), pybind11::arg("rb"),
        pybind11::arg("gp"), pybind11::arg("gr"), pybind11::arg("ga"), pybind11::arg("qmax") = 0.0,
        pybind11::arg("u") = pybind11::none(), pybind11::arg("R") = pybind11::none(),
        pybind11::arg("h32") = false);   // [prefill4]
}
