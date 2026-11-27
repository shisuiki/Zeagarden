// arn.cu - Unified CUDA-graph-safe TP4 All-Reduce + Gemma RMSNorm (and bare All-Reduce) for M = 4..4608 rows (H = 5120)
// on 4x Tesla V100-SXM2 (sm_70). Replaces both legacy pf_arn.cu and short_arn.cu with ONE IPC pool (45.0 MiB/rank),
// ONE device-side monotonic u64 epoch, zero __nanosleep tight-spin, and 4 specialized modes sharing the same buffer:
//   - MODE 0 (ar):        bare FP16 all-reduce (M = 65..1024)
//   - MODE 1 (arn_full):  all-reduce + COMPLETE FP32 residual add + Gemma RMSNorm (M = 65..1024 or non-sharded callers)
//   - MODE 2 (arn_shard): all-reduce + ROW-SHARDED FP32 residual add + Gemma RMSNorm (M = 1025..4608, 2.2x vs 1Cat)
//   - MODE 3 (arn_quant): MODE 2 + fused per-token INT7/INT8 (optional H_32 Walsh-Hadamard) quant + Freivalds checksum
//
// 1-slot race-freedom proof (all M >= 4 -> every quarter q in 0..3 has qrows(q) >= 1 and block j = 0 active):
//   - recv[q] hazard (ep read vs ep+1 write): rank q reducer reads recv[q] -> bar.sync -> fence.acq_rel.sys -> writes
//     flag2[p][q][j] = ep. Peer p only exits ep (and enters ep+1 to write recv[q]) after observing ALL flag2[p][q][*] == ep.
//   - ag[q]/am[q] hazard (ep read vs ep+1 write): peer p finishes reading ALL ag[q]/am[q] of ep before exiting ep, entering
//     ep+1, and writing flag1[q][p][j] = ep+1. Rank q reducer waits for ALL 3 peers' flag1[q][p][j] == ep+1 before writing
//     ag[q]/am[q] of ep+1. All 4 modes share the identical flag1/flag2/epoch protocol and mix arbitrarily across calls.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) TORCH_CHECK((x) == cudaSuccess, #x " failed: ", cudaGetErrorString(cudaGetLastError()))

namespace {

constexpr int H = 5120;
constexpr int NRK = 4;
constexpr int NV = H / 8;                                               // 640 threads/CTA = 16-B vectors/row
constexpr float INV_SQRT32 = 0.17677669529663687f;                      // exact f32 1 / sqrt(32)
constexpr int MAXM = 4096 + 512;                                        // 4608 rows
constexpr int MAXQ = MAXM / NRK;                                        // 1152 max rows per quarter
constexpr int NBMAX = 256;                                              // max blocks per quarter (RB>=1 @<=1024, RB>=8 @<=4608)
constexpr size_t RECV_HALVES = (size_t)(NRK - 1) * MAXQ * H;            // 3 * 1152 * 5120 halves (33.75 MiB)
constexpr size_t AG_HALVES = (size_t)MAXQ * H;                          // 1 * 1152 * 5120 halves (11.25 MiB)
constexpr size_t DATA_BYTES = (RECV_HALVES + AG_HALVES) * sizeof(__half);  // 47,185,920 B (45.00 MiB, 4 KiB aligned)
constexpr size_t FLAG_U64S = (size_t)2 * NRK * NBMAX;                   // 2 * 4 * 256 u64s (16 KiB)
constexpr size_t AM_OFF = DATA_BYTES + FLAG_U64S * sizeof(uint64_t);    // 4 KiB aligned
constexpr size_t BUF_BYTES = AM_OFF + (size_t)MAXQ * sizeof(float);     // 47,206,912 B (~45.02 MiB/rank)
constexpr unsigned long long TRAP_NS = 5000000000ull;                   // 5 s deadlock guard

struct CfgSpec {
  int rb, gp, gr, ga;
};

// 9 hardware-tuned configs (all total <= 80 CTAs -> guaranteed co-resident on 80-SM V100)
constexpr CfgSpec kCfgs[9] = {
    {1, 16, 24, 40},   // cfg 0: short RB=1 ultra-low-latency (M <= 128)
    {2, 16, 24, 40},   // cfg 1: short RB=2 puller-heavy (M = 64..256)
    {2, 24, 24, 32},   // cfg 2: short RB=2 balanced AR (M = 64..256)
    {4, 16, 24, 40},   // cfg 3: short RB=4 puller-heavy (M = 65..256 ARN / 192..512)
    {4, 24, 20, 36},   // cfg 4: short RB=4 pusher-heavy (M = 192..768)
    {8, 16, 24, 40},   // cfg 5: medium RB=8 bulk (M = 257..1024)
    {16, 12, 40, 12},  // cfg 6: long RB=16 (M = 768..1535)
    {24, 12, 48, 12},  // cfg 7: long RB=24 (M = 1536..3071)
    {32, 12, 40, 12},  // cfg 8: long RB=32 (M = 3072..4608)
};

struct Comm {
  char* base[NRK];
  int rank;
};

struct DevState {
  uint64_t epoch;
  uint32_t done;
};

struct Args {
  Comm c;
  DevState* dev_state;
  const __half* x;
  const float* res;
  float* res_out;
  const __half* w;
  __half* out;
  float eps;
  int M, MQ, RB, NB;
  int GP, GR, GA;
  // Optional fused quantize / Freivalds checksum (MODE 3)
  __half* xq;
  float* sx;
  float qm;
  const unsigned* u;
  unsigned* R;
};

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void fence_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ unsigned atom_add_acq_rel_gpu_u32(uint32_t* p, uint32_t v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__ uint64_t ld_rlx_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_rlx_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t ld_acq_gpu_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_rel_gpu_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ void st_rlx_gpu_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void wait_flag1_u64(const uint64_t* p, uint64_t ep) {
  if (ld_rlx_sys_u64(p) != ep) {
    const unsigned long long t0 = gtime();
    do {
      if (gtime() - t0 > TRAP_NS) __trap();
    } while (ld_rlx_sys_u64(p) != ep);
  }
  fence_sys();
}

__device__ __forceinline__ void wait_flag3_u64(const uint64_t* p0, const uint64_t* p1, const uint64_t* p2, uint64_t ep) {
  if (ld_rlx_sys_u64(p0) != ep || ld_rlx_sys_u64(p1) != ep || ld_rlx_sys_u64(p2) != ep) {
    const unsigned long long t0 = gtime();
    do {
      if (gtime() - t0 > TRAP_NS) __trap();
    } while (ld_rlx_sys_u64(p0) != ep || ld_rlx_sys_u64(p1) != ep || ld_rlx_sys_u64(p2) != ep);
  }
  fence_sys();
}

__device__ __forceinline__ char* base_of(const Comm& c, int q) {
  return q == 0 ? c.base[0] : q == 1 ? c.base[1] : q == 2 ? c.base[2] : c.base[3];
}
__device__ __forceinline__ __half* recv_of(const Comm& c, int q) { return reinterpret_cast<__half*>(base_of(c, q)); }
__device__ __forceinline__ __half* ag_of(const Comm& c, int q) { return recv_of(c, q) + RECV_HALVES; }
__device__ __forceinline__ uint64_t* flag1_of(const Comm& c, int q) { return reinterpret_cast<uint64_t*>(base_of(c, q) + DATA_BYTES); }
__device__ __forceinline__ uint64_t* flag2_of(const Comm& c, int q) { return flag1_of(c, q) + NRK * NBMAX; }
__device__ __forceinline__ float* am_of(const Comm& c, int q) { return reinterpret_cast<float*>(base_of(c, q) + AM_OFF); }

__device__ __forceinline__ int slot_idx(int src, int dst) { return (src - dst + NRK - 1) & (NRK - 1); }

__device__ __forceinline__ int qrows(const Args& a, int q) {
  const int lo = q * a.MQ;
  const int hi = a.M < (q + 1) * a.MQ ? a.M : (q + 1) * a.MQ;
  return hi > lo ? hi - lo : 0;
}

__device__ __forceinline__ uint4 ldcg4(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }

__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
  const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
  uint4 u;
  __half2* h = reinterpret_cast<__half2*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  return u;
}

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

// Exact 5-stage H_32 butterfly + finite FP16 clamp across 4 consecutive lanes (bit-identical to pk_h32.cu)
__device__ __forceinline__ void h32_rotate_vec8(const uint4& in, __half2 (&hr)[4], int lane) {
  float f[8];
  unpack8(in, f);
#pragma unroll
  for (int e = 0; e < 8; e += 2) { const float a = f[e], b = f[e + 1]; f[e] = a + b; f[e + 1] = a - b; }
#pragma unroll
  for (int base = 0; base < 8; base += 4) {
    const float a0 = f[base], b0 = f[base + 2]; f[base] = a0 + b0; f[base + 2] = a0 - b0;
    const float a1 = f[base + 1], b1 = f[base + 3]; f[base + 1] = a1 + b1; f[base + 3] = a1 - b1;
  }
#pragma unroll
  for (int e = 0; e < 4; e++) { const float a = f[e], b = f[e + 4]; f[e] = a + b; f[e + 4] = a - b; }
  const bool hi1 = (lane & 1) != 0;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float peer = __shfl_xor_sync(0xffffffffu, f[e], 1);
    f[e] = hi1 ? (peer - f[e]) : (f[e] + peer);
  }
  const bool hi2 = (lane & 2) != 0;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float peer = __shfl_xor_sync(0xffffffffu, f[e], 2);
    f[e] = hi2 ? (peer - f[e]) : (f[e] + peer);
  }
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const float r0 = fminf(65504.f, fmaxf(-65504.f, f[2 * e] * INV_SQRT32));
    const float r1 = fminf(65504.f, fmaxf(-65504.f, f[2 * e + 1] * INV_SQRT32));
    hr[e] = __floats2half2_rn(r0, r1);
  }
}

__device__ __forceinline__ uint4 gemma_norm_row(const Args& a, int row, int col, int lane, int warp, int rbuf,
                                                uint4 av, float4 q0, float4 q1, uint4 wv, float (&red)[2][20]) {
  float f[8];
  unpack8(av, f);
  f[0] = __fadd_rn(f[0], q0.x); f[1] = __fadd_rn(f[1], q0.y);
  f[2] = __fadd_rn(f[2], q0.z); f[3] = __fadd_rn(f[3], q0.w);
  f[4] = __fadd_rn(f[4], q1.x); f[5] = __fadd_rn(f[5], q1.y);
  f[6] = __fadd_rn(f[6], q1.z); f[7] = __fadd_rn(f[7], q1.w);
  float4* ro = reinterpret_cast<float4*>(a.res_out + (size_t)row * H + col);
  __stcs(ro, make_float4(f[0], f[1], f[2], f[3]));
  __stcs(ro + 1, make_float4(f[4], f[5], f[6], f[7]));

  float ss = __fmul_rn(f[0], f[0]);
#pragma unroll
  for (int i = 1; i < 8; i++) ss = __fmaf_rn(f[i], f[i], ss);
#pragma unroll
  for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  if (lane == 0) red[rbuf][warp] = ss;
  __syncthreads();

  float tot = red[rbuf][0];
#pragma unroll
  for (int k = 1; k < 20; k++) tot = __fadd_rn(tot, red[rbuf][k]);
  const float inv = rsqrtf(tot * (1.f / (float)H) + a.eps);

  float wf[8], o[8];
  unpack8(wv, wf);
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = __fmul_rn(__fadd_rn(wf[i], 1.f), __fmul_rn(f[i], inv));
  const uint4 ov = pack8(o);
  *reinterpret_cast<uint4*>(a.out + (size_t)row * H + col) = ov;
  return ov;
}

// ------------------------------------------------------------------------------------------------ Role 1: Pusher
__device__ __forceinline__ void role_pusher(const Args& a, int cta, uint64_t ep) {
  const int r = a.c.rank;
  const int nu = 3 * a.NB;
  for (int u = cta; u < nu; u += a.GP) {
    const int j = u / 3, i = u % 3, q = (r + 1 + i) & (NRK - 1);
    const int rem = qrows(a, q) - j * a.RB;
    const int nr = rem < a.RB ? rem : a.RB;
    if (nr <= 0) continue;
    const int sl = slot_idx(r, q);
    const uint4* src = reinterpret_cast<const uint4*>(a.x + (size_t)(q * a.MQ + j * a.RB) * H);
    uint4* dst = reinterpret_cast<uint4*>(recv_of(a.c, q) + ((size_t)sl * MAXQ + j * a.RB) * H);
    copy_rows(dst, src, nr * NV, false);
    __syncthreads();
    if (threadIdx.x == 0) {
      fence_sys();
      st_rlx_sys_u64(flag1_of(a.c, q) + r * NBMAX + j, ep);
    }
  }
}

// ------------------------------------------------------------------------------------------------ Role 2: Reducer
// MODE: 0=bare AR, 1=full-res ARN, 2=sharded-res ARN, 3=sharded-res ARN + fused H_32 quant
template <int MODE>
__device__ __forceinline__ void role_reducer(const Args& a, int cta, uint64_t ep, float (&red)[2][20]) {
  constexpr bool DO_NORM = (MODE >= 1);
  constexpr bool SHARDED = (MODE >= 2);
  constexpr bool DO_QUANT = (MODE == 3);

  const int r = a.c.rank, t = threadIdx.x, lane = t & 31, warp = t >> 5, col = t * 8;
  const int nq = qrows(a, r);
  const __half* const rv = recv_of(a.c, r);
  __half* const ag = ag_of(a.c, r);
  const uint64_t* const f1 = flag1_of(a.c, r);
  const uint4 wv = DO_NORM ? __ldg(reinterpret_cast<const uint4*>(a.w) + t) : make_uint4(0, 0, 0, 0);

  unsigned uw[8];
  if constexpr (DO_QUANT) {
    if (a.u != nullptr) {
      const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * t);
      const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * t + 1);
      uw[0] = u0.x; uw[1] = u0.y; uw[2] = u0.z; uw[3] = u0.w;
      uw[4] = u1.x; uw[5] = u1.y; uw[6] = u1.z; uw[7] = u1.w;
    }
  }

  int rbuf = 0;
  for (int j = cta; j < a.NB; j += a.GR) {
    const int rem = nq - j * a.RB;
    const int nr = rem < a.RB ? rem : a.RB;
    if (nr <= 0) break;

    if (t == 0) {
      wait_flag3_u64(f1 + ((r + 1) & 3) * NBMAX + j,
                     f1 + ((r + 2) & 3) * NBMAX + j,
                     f1 + ((r + 3) & 3) * NBMAX + j, ep);
    }
    __syncthreads();

    if constexpr (!SHARDED) {
      // MODE 0 (AR) / MODE 1 (full-res ARN): reduce -> publish ag + flag2 immediately -> optional own-quarter norm
      for (int rr = 0; rr < nr; rr++) {
        const int qrow = j * a.RB + rr;
        const int row = r * a.MQ + qrow;
        uint4 pv[NRK];
#pragma unroll
        for (int p = 0; p < NRK; p++) {
          pv[p] = (p == r) ? ldcg4(a.x + (size_t)row * H + col)
                           : ldcg4(rv + ((size_t)slot_idx(p, r) * MAXQ + qrow) * H + col);
        }
        float s[8];
#pragma unroll
        for (int p = 0; p < NRK; p++) {
          float xf[8];
          unpack8(pv[p], xf);
#pragma unroll
          for (int e = 0; e < 8; e++) s[e] = (p == 0) ? xf[e] : __fadd_rn(s[e], xf[e]);
        }
        const uint4 ar_v = pack8(s);
        *reinterpret_cast<uint4*>(ag + (size_t)qrow * H + col) = ar_v;
        if constexpr (!DO_NORM) {
          *reinterpret_cast<uint4*>(a.out + (size_t)row * H + col) = ar_v;
        }
      }
      __syncthreads();
      if (t == 0) {
        fence_sys();
#pragma unroll
        for (int p = 1; p < NRK; p++) {
          st_rlx_sys_u64(flag2_of(a.c, (r + p) & (NRK - 1)) + r * NBMAX + j, ep);
        }
      }
      if constexpr (DO_NORM) {
        for (int rr = 0; rr < nr; rr++, rbuf ^= 1) {
          const int qrow = j * a.RB + rr;
          const int row = r * a.MQ + qrow;
          const uint4 av = *reinterpret_cast<const uint4*>(ag + (size_t)qrow * H + col);
          const float4* rp = reinterpret_cast<const float4*>(a.res + (size_t)row * H + col);
          gemma_norm_row(a, row, col, lane, warp, rbuf, av, rp[0], rp[1], wv, red);
        }
      }
    } else {
      // MODE 2/3/4 (sharded-res ARN): reduce + own-quarter residual/norm (+ optional quant) -> publish normalized ag + flag2
      for (int rr = 0; rr < nr; rr++, rbuf ^= 1) {
        const int qrow = j * a.RB + rr;
        const int row = r * a.MQ + qrow;
        uint4 pv[NRK];
#pragma unroll
        for (int p = 0; p < NRK; p++) {
          pv[p] = (p == r) ? ldcg4(a.x + (size_t)row * H + col)
                           : ldcg4(rv + ((size_t)slot_idx(p, r) * MAXQ + qrow) * H + col);
        }
        float s[8];
#pragma unroll
        for (int p = 0; p < NRK; p++) {
          float xf[8];
          unpack8(pv[p], xf);
#pragma unroll
          for (int e = 0; e < 8; e++) s[e] = (p == 0) ? xf[e] : __fadd_rn(s[e], xf[e]);
        }
        const uint4 ar_v = pack8(s);
        const float4* rp = reinterpret_cast<const float4*>(a.res + (size_t)row * H + col);
        const uint4 ov = gemma_norm_row(a, row, col, lane, warp, rbuf, ar_v, __ldcs(rp), __ldcs(rp + 1), wv, red);
        *reinterpret_cast<uint4*>(ag + (size_t)qrow * H + col) = ov;

        if constexpr (DO_QUANT) {
          __align__(16) __half2 hbuf[4];
          h32_rotate_vec8(ov, hbuf, lane);
          float am = 0.f;
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 f = __half22float2(hbuf[e]);
            am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
          }
#pragma unroll
          for (int o2 = 16; o2; o2 >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o2));
          const int qbuf = rbuf ^ 1;
          __syncthreads();
          if (lane == 0) red[qbuf][warp] = am;
          __syncthreads();
          am = 0.f;
#pragma unroll
          for (int k = 0; k < 20; k++) am = fmaxf(am, red[qbuf][k]);
          const float qinv = am > 0.f ? __fdiv_rn(a.qm, am) : 0.f;
          __align__(16) __half2 q2[4];
          float qv[8];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 f = __half22float2(hbuf[e]);
            qv[2 * e] = fminf(a.qm, fmaxf(-a.qm, rintf(f.x * qinv)));
            qv[2 * e + 1] = fminf(a.qm, fmaxf(-a.qm, rintf(f.y * qinv)));
            q2[e] = __floats2half2_rn(qv[2 * e], qv[2 * e + 1]);
          }
          *reinterpret_cast<uint4*>(a.xq + (size_t)row * H + col) = *reinterpret_cast<const uint4*>(q2);
          if (t == 0) {
            a.sx[row] = __fdiv_rn(am, a.qm);
            am_of(a.c, r)[qrow] = am;
          }
          if (a.u != nullptr) {
            unsigned ref = 0u;
#pragma unroll
            for (int e = 0; e < 8; e++) ref += (unsigned)__float2int_rn(qv[e]) * uw[e];
#pragma unroll
            for (int o2 = 16; o2; o2 >>= 1) ref += __shfl_xor_sync(0xffffffffu, ref, o2);
            __syncthreads();
            if (lane == 0) reinterpret_cast<unsigned*>(red[qbuf])[warp] = ref;
            __syncthreads();
            if (t == 0) {
              unsigned tot = 0u;
#pragma unroll
              for (int k = 0; k < 20; k++) tot += reinterpret_cast<unsigned*>(red[qbuf])[k];
              a.R[row] = tot;
            }
          }
          __syncthreads();
        }
      }
      __syncthreads();
      if (t == 0) {
        fence_sys();
#pragma unroll
        for (int p = 1; p < NRK; p++) {
          st_rlx_sys_u64(flag2_of(a.c, (r + p) & (NRK - 1)) + r * NBMAX + j, ep);
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ Role 3: Puller
template <int MODE>
__device__ __forceinline__ void role_puller(const Args& a, int cta, uint64_t ep, float (&red)[2][20]) {
  constexpr bool FULL_NORM = (MODE == 1);
  constexpr bool DO_QUANT = (MODE == 3);

  const int r = a.c.rank, t = threadIdx.x, lane = t & 31, warp = t >> 5, col = t * 8;
  const uint64_t* const f2 = flag2_of(a.c, r);
  const int nu = 3 * a.NB;
  const uint4 wv = FULL_NORM ? __ldg(reinterpret_cast<const uint4*>(a.w) + t) : make_uint4(0, 0, 0, 0);

  unsigned uw[8];
  if constexpr (DO_QUANT) {
    if (a.u != nullptr) {
      const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * t);
      const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(a.u) + 2 * t + 1);
      uw[0] = u0.x; uw[1] = u0.y; uw[2] = u0.z; uw[3] = u0.w;
      uw[4] = u1.x; uw[5] = u1.y; uw[6] = u1.z; uw[7] = u1.w;
    }
  }

  int rbuf = 0;
  for (int u = cta; u < nu; u += a.GA) {
    const int j = u / 3, i = u % 3, p = (r + 1 + i) & (NRK - 1);
    const int rem = qrows(a, p) - j * a.RB;
    const int nr = rem < a.RB ? rem : a.RB;
    if (nr <= 0) continue;

    if (t == 0) wait_flag1_u64(f2 + p * NBMAX + j, ep);
    __syncthreads();

    const __half* const src = ag_of(a.c, p) + (size_t)(j * a.RB) * H;
    if constexpr (MODE == 0 || MODE == 2) {
      uint4* dst = reinterpret_cast<uint4*>(a.out + (size_t)(p * a.MQ + j * a.RB) * H);
      copy_rows(dst, reinterpret_cast<const uint4*>(src), nr * NV, true);
    } else if constexpr (FULL_NORM) {
      for (int rr = 0; rr < nr; rr++, rbuf ^= 1) {
        const int row = p * a.MQ + j * a.RB + rr;
        const uint4 av = ldcg4(src + (size_t)rr * H + col);
        const float4* rp = reinterpret_cast<const float4*>(a.res + (size_t)row * H + col);
        gemma_norm_row(a, row, col, lane, warp, rbuf, av, rp[0], rp[1], wv, red);
      }
    } else {
      // MODE 3: pull normalized rows + owner's row amax -> H_32 xq (+ optional Freivalds checksum)
      __shared__ float amr[64];
      __shared__ unsigned rref[64];
      if (t < nr) {
        rref[t] = 0u;
        const float am = __ldcg(am_of(a.c, p) + j * a.RB + t);
        amr[t] = am;
        a.sx[p * a.MQ + j * a.RB + t] = __fdiv_rn(am, a.qm);
      }
      __syncthreads();
      const uint4* src4 = reinterpret_cast<const uint4*>(src);
      uint4* dst = reinterpret_cast<uint4*>(a.out + (size_t)(p * a.MQ + j * a.RB) * H);
      uint4* dq = reinterpret_cast<uint4*>(a.xq + (size_t)(p * a.MQ + j * a.RB) * H);
      constexpr int U = 4;
      for (int r0 = 0; r0 < nr; r0 += U) {
        uint4 v[U];
#pragma unroll
        for (int uu = 0; uu < U; uu++) if (r0 + uu < nr) v[uu] = ldcg4(src4 + (size_t)(r0 + uu) * NV + t);
#pragma unroll
        for (int uu = 0; uu < U; uu++) {
          if (r0 + uu >= nr) break;
          const size_t off = (size_t)(r0 + uu) * NV + t;
          dst[off] = v[uu];
          const float am = amr[r0 + uu];
          const float inv = am > 0.f ? __fdiv_rn(a.qm, am) : 0.f;
          __align__(16) __half2 hbuf[4];
          h32_rotate_vec8(v[uu], hbuf, lane);
          __align__(16) __half2 q2[4];
          float qv[8];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 f = __half22float2(hbuf[e]);
            qv[2 * e] = fminf(a.qm, fmaxf(-a.qm, rintf(f.x * inv)));
            qv[2 * e + 1] = fminf(a.qm, fmaxf(-a.qm, rintf(f.y * inv)));
            q2[e] = __floats2half2_rn(qv[2 * e], qv[2 * e + 1]);
          }
          dq[off] = *reinterpret_cast<const uint4*>(q2);
          if (a.u != nullptr) {
            unsigned ref = 0u;
#pragma unroll
            for (int e = 0; e < 8; e++) ref += (unsigned)__float2int_rn(qv[e]) * uw[e];
#pragma unroll
            for (int o2 = 16; o2; o2 >>= 1) ref += __shfl_xor_sync(0xffffffffu, ref, o2);
            if (lane == 0) atomicAdd(&rref[r0 + uu], ref);
          }
        }
      }
      if (a.u != nullptr) {
        __syncthreads();
        if (t < nr) a.R[p * a.MQ + j * a.RB + t] = rref[t];
      }
      __syncthreads();
    }
  }
}

// ------------------------------------------------------------------------------------------------ unified kernel
template <int MODE>
__global__ void __launch_bounds__(NV, 1) k_arn_unified(Args a) {
  __shared__ uint64_t s_ep;
  __shared__ float red[2][20];
  if (threadIdx.x == 0) s_ep = ld_acq_gpu_u64(&a.dev_state->epoch);
  __syncthreads();
  const uint64_t ep = s_ep;

  const int b = blockIdx.x;
  if (b < a.GP) role_pusher(a, b, ep);
  else if (b < a.GP + a.GR) role_reducer<MODE>(a, b - a.GP, ep, red);
  else role_puller<MODE>(a, b - a.GP - a.GR, ep, red);

  __syncthreads();
  if (threadIdx.x == 0) {
    if (atom_add_acq_rel_gpu_u32(&a.dev_state->done, 1u) == (unsigned)(gridDim.x - 1)) {
      st_rlx_gpu_u32(&a.dev_state->done, 0u);
      st_rel_gpu_u64(&a.dev_state->epoch, ep + 1ull);
    }
  }
}

// ------------------------------------------------------------------------------------------------ host state
Comm g_comm{};
DevState* g_dev_state = nullptr;
bool g_init = false;
int g_num_sms = 80;
std::vector<std::string> g_info;

inline bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

template <class F>
int probe_kernel(const char* name, F fn) {
  cudaFuncAttributes attr{};
  CK(cudaFuncGetAttributes(&attr, fn));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, NV, 0));
  TORCH_CHECK(occ > 0, "q27_arn: kernel occupancy is 0 for ", name);
  const int max_co = occ * g_num_sms;
  g_info.push_back(std::string(name) + " regs=" + std::to_string(attr.numRegs) +
                   " lmem=" + std::to_string(attr.localSizeBytes) +
                   " smem=" + std::to_string(attr.sharedSizeBytes) +
                   " occ/SM=" + std::to_string(occ) +
                   " co_ctas=" + std::to_string(max_co));
  return max_co;
}

void probe_all_kernels() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev));
  TORCH_CHECK(g_num_sms > 0, "q27_arn: invalid SM count ", g_num_sms);
  g_info.clear();
  int min_co = probe_kernel("k_ar", k_arn_unified<0>);
  min_co = std::min(min_co, probe_kernel("k_arn_full", k_arn_unified<1>));
  min_co = std::min(min_co, probe_kernel("k_arn_shard", k_arn_unified<2>));
  min_co = std::min(min_co, probe_kernel("k_arn_h32", k_arn_unified<3>));
  for (int i = 0; i < 9; i++) {
    const int tot = kCfgs[i].gp + kCfgs[i].gr + kCfgs[i].ga;
    TORCH_CHECK(tot <= min_co, "q27_arn: cfg ", i, " needs ", tot, " CTAs > max co-resident ", min_co);
  }
}

void fill_geometry(Args& a, int M, int cfg) {
  const CfgSpec sp = kCfgs[cfg];
  a.M = M;
  a.MQ = (M + NRK - 1) / NRK;
  a.RB = sp.rb;
  a.NB = (a.MQ + a.RB - 1) / a.RB;
  TORCH_CHECK(a.NB >= 1 && a.NB <= NBMAX, "q27_arn: NB out of range: ", a.NB, " for M=", M, " cfg=", cfg);
  const int nu = 3 * a.NB;
  a.GP = std::max(1, std::min(sp.gp, nu));
  a.GR = std::max(1, std::min(sp.gr, a.NB));
  a.GA = std::max(1, std::min(sp.ga, nu));
}

}  // namespace

// ------------------------------------------------------------------------------------------------ host API
int64_t buf_bytes() { return (int64_t)BUF_BYTES; }
int64_t max_rows() { return MAXM; }
std::vector<std::string> info() { return g_info; }

int64_t alloc_buf() {
  void* p = nullptr;
  CK(cudaMalloc(&p, BUF_BYTES));
  CK(cudaMemset(p, 0, BUF_BYTES));
  CK(cudaDeviceSynchronize());
  return reinterpret_cast<int64_t>(p);
}

py::bytes ipc_handle(int64_t ptr) {
  cudaIpcMemHandle_t h;
  CK(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(ptr)));
  return py::bytes(reinterpret_cast<const char*>(&h), sizeof(h));
}

int64_t ipc_open(py::bytes b) {
  std::string s = b;
  TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "q27_arn: invalid IPC handle size");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, s.data(), sizeof(h));
  void* p = nullptr;
  CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

void init(std::vector<int64_t> bases, int64_t rank) {
  TORCH_CHECK(!g_init, "q27_arn: already initialised");
  TORCH_CHECK(bases.size() == NRK && rank >= 0 && rank < NRK, "q27_arn: expected 4 rank bases and rank in 0..3");
  for (int q = 0; q < NRK; q++) {
    TORCH_CHECK(bases[q] != 0 && (static_cast<uintptr_t>(bases[q]) & 15u) == 0, "q27_arn: bad IPC ptr for rank ", q);
    for (int p = 0; p < q; p++) {
      TORCH_CHECK(bases[q] != bases[p], "q27_arn: duplicate IPC ptr for ranks ", p, " and ", q);
    }
    g_comm.base[q] = reinterpret_cast<char*>(bases[q]);
  }
  g_comm.rank = (int)rank;
  if (g_dev_state == nullptr) CK(cudaMalloc(&g_dev_state, sizeof(DevState)));
  const DevState init_st{1ull, 0u};
  CK(cudaMemcpy(g_dev_state, &init_st, sizeof(DevState), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  probe_all_kernels();
  g_init = true;
}

std::vector<torch::Tensor> arn(torch::Tensor x, torch::Tensor res, torch::Tensor w, double eps,
                               int64_t cfg, bool shard_res, double qmax,
                               c10::optional<torch::Tensor> u, c10::optional<torch::Tensor> R) {
  TORCH_CHECK(g_init, "q27_arn: not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.size(1) == H && x.is_contiguous());
  TORCH_CHECK(res.is_cuda() && res.device() == x.device() && res.dtype() == torch::kFloat &&
              res.sizes() == x.sizes() && res.is_contiguous());
  TORCH_CHECK(w.is_cuda() && w.device() == x.device() && w.dtype() == torch::kHalf &&
              w.numel() == H && w.is_contiguous());
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 4 && M <= MAXM, "q27_arn: M must be in 4..", MAXM, ", got ", M);
  TORCH_CHECK(cfg >= 0 && cfg <= 8, "q27_arn: cfg must be in 0..8, got ", cfg);

  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  auto res_out = torch::empty_like(res);

  Args a{};
  a.c = g_comm;
  a.dev_state = g_dev_state;
  a.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  a.res = res.data_ptr<float>();
  a.res_out = res_out.data_ptr<float>();
  a.w = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  a.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  a.eps = (float)eps;
  TORCH_CHECK(is_aligned16(a.x) && is_aligned16(a.res) && is_aligned16(a.w) &&
              is_aligned16(a.out) && is_aligned16(a.res_out), "q27_arn: tensors must be 16-B aligned");
  fill_geometry(a, M, (int)cfg);

  torch::Tensor xq, sx;
  a.xq = nullptr; a.sx = nullptr; a.qm = (float)qmax; a.u = nullptr; a.R = nullptr;
  if (qmax > 0.0) {
    TORCH_CHECK(shard_res && a.RB <= 64, "q27_arn: fused H_32 quant requires shard_res=true and RB <= 64");
    TORCH_CHECK(a.qm >= 1.f && a.qm <= 127.f && qmax == std::floor(qmax), "q27_arn: bad h32 qmax");
    if (u.has_value()) {
      TORCH_CHECK(R.has_value() && u->numel() == H && R->numel() >= M, "q27_arn: bad checksum arrays");
      a.u = reinterpret_cast<const unsigned*>(u->data_ptr<int>());
      a.R = reinterpret_cast<unsigned*>(R->data_ptr<int>());
    }
    xq = torch::empty_like(x);
    sx = torch::empty({M}, x.options().dtype(torch::kFloat));
    a.xq = reinterpret_cast<__half*>(xq.data_ptr<at::Half>());
    a.sx = sx.data_ptr<float>();
  }

  const int grid = a.GP + a.GR + a.GA;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (qmax > 0.0) {
    k_arn_unified<3><<<grid, NV, 0, stream>>>(a);
  } else if (shard_res) {
    k_arn_unified<2><<<grid, NV, 0, stream>>>(a);
  } else {
    k_arn_unified<1><<<grid, NV, 0, stream>>>(a);
  }
  CK(cudaGetLastError());
  if (qmax > 0.0) return {out, res_out, xq, sx};
  return {out, res_out};
}

torch::Tensor ar(torch::Tensor x, int64_t cfg) {
  TORCH_CHECK(g_init, "q27_arn: not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.size(1) == H && x.is_contiguous());
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 4 && M <= MAXM, "q27_arn: M must be in 4..", MAXM, ", got ", M);
  TORCH_CHECK(cfg >= 0 && cfg <= 8, "q27_arn: cfg must be in 0..8, got ", cfg);

  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  Args a{};
  a.c = g_comm;
  a.dev_state = g_dev_state;
  a.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  a.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  TORCH_CHECK(is_aligned16(a.x) && is_aligned16(a.out), "q27_arn: tensors must be 16-B aligned");
  fill_geometry(a, M, (int)cfg);

  const int grid = a.GP + a.GR + a.GA;
  k_arn_unified<0><<<grid, NV, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  CK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buf_bytes", &buf_bytes);
  m.def("max_rows", &max_rows);
  m.def("alloc_buf", &alloc_buf);
  m.def("ipc_handle", &ipc_handle);
  m.def("ipc_open", &ipc_open);
  m.def("init", &init);
  m.def("info", &info);
  m.def("arn", &arn, "Unified TP4 all-reduce + Gemma RMSNorm (full or sharded residual, optional fused H_32 quant)",
        py::arg("x"), py::arg("res"), py::arg("w"), py::arg("eps"),
        py::arg("cfg") = 3, py::arg("shard_res") = false, py::arg("qmax") = 0.0,
        py::arg("u") = py::none(), py::arg("R") = py::none());
  m.def("ar", &ar, "Unified TP4 bare all-reduce (fp16)", py::arg("x"), py::arg("cfg") = 3);
}
