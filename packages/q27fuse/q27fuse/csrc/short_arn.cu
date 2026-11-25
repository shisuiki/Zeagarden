// short_arn.cu v2 (track shortpf) - CUDA-graph-safe, ultra-fast TP4 short-prefill All-Reduce + COMPLETE FP32
// Residual Add + Gemma RMSNorm (short_arn) and bare All-Reduce (short_ar) for M = 0..1024 rows (H = 5120) on
// 4x Tesla V100-SXM2 (sm_70).
//
// Architecture (block-flagged 3-role row-quarter RS + AG with 2-slot ping-pong & device-side monotonic u64 epoch):
//   - Rows are partitioned into 4 row quarters: MQ = ceil(M / 4), quarter q = [q*MQ, min(M, (q+1)*MQ)), split into
//     blocks of RB rows (NB = ceil(MQ / RB) <= NBMAX = 256).
//   - Three co-resident CTA roles (640 threads/CTA = H / 8, total GP + GR + GA <= 80 <= occ * num_sms):
//       pushers  : bulk unrolled uint4 copies of own partial rows for quarter q != rank into peer q's recv[slot],
//                  then 1 thread (t == 0) executes fence.acq_rel.sys + st.relaxed.sys u64 flag1[q][rank][j] = ep.
//       reducers : for own quarter's blocks j, 1 thread (t == 0) tight-spins (no __nanosleep, 5 s trap) on all 3
//                  peers' flag1 via ld.relaxed.sys u64 followed by ONE same-thread fence.acq_rel.sys + __syncthreads();
//                  sums 4 partials in strict rank order 0..3 in FP32 (__fadd_rn), rounds to FP16 (__floats2half2_rn),
//                  writes ar_half to local ag[slot], publishes flag2 = ep on all 3 peers via 1 fence.acq_rel.sys
//                  BEFORE running own-quarter FP32 residual add + exact IEEE Gemma RMSNorm (overlapping norm with
//                  peers' NVLink pulls!).
//       pullers  : for peer p != rank block j, t == 0 tight-spins on flag2 via ld.relaxed.sys u64 + fence.acq_rel.sys
//                  + __syncthreads(), reads ar_half via __ldcg from peer p's ag[slot]; short_ar copies to out, while
//                  short_arn fuses FP32 residual add (writing COMPLETE res_out on every rank!) + exact IEEE Gemma
//                  RMSNorm directly into out.
//   - Zero sentinel hacks: flag-based synchronization preserves arbitrary FP16 bit patterns (including NaN 0x7f7f)
//     and FP32 subnormals bit-identically.
//   - Device-side monotonic uint64_t epoch (DevState { uint64_t epoch; uint32_t done; } in local device memory):
//       Every CTA reads epoch via ld.acquire.gpu at entry; the strictly last CTA (atomicAdd(&done, 1) == gridDim.x - 1)
//       resets done = 0 (st.relaxed.gpu) and stores epoch = ep + 1 (st.release.gpu). Zero host epoch argument ->
//       100% CUDA-graph-safe for arbitrary odd/even captured call counts, replays, eager interleaves, and changing M/cfg.
//   - 2-slot ping-pong IPC buffer (slot = ep & 1, ~20.0 MiB/rank, zeroed via cudaMemset(p, 0, BUF_BYTES)):
//       Per slot s in {0, 1}: recv[3][256][5120] fp16 (7.5 MiB) + ag[256][5120] fp16 (2.5 MiB)
//                             + flag1[4][256] u64 (8 KiB) + flag2[4][256] u64 (8 KiB) = 10,502,144 B (4 KiB aligned).
//       Single-VA-per-physical-buffer-per-GPU invariant: enforced pairwise-distinct + 16-B aligned in init()
//       (optional -DSHORT_ARN_ALIAS_FENCE emits fence.proxy.alias).
//
// Formal PTX 12.9 Memory-Ordering & Race-Freedom Proof (sm_70, PTX 6.0+):
//   (1) Data -> Flag -> Data within call ep:
//       - Pusher: T_i writes recv[slot] --(po)--> bar.sync --(sw.cta)--> T_0: fence.acq_rel.sys --(po)--> st.relaxed.sys [flag1], ep.
//       - Reducer: T_0: ld.relaxed.sys [flag1] == ep (all 3 peers) --(po)--> T_0: fence.acq_rel.sys --(po)--> bar.sync
//                  --(sw.cta)--> T_j: __ldcg(recv[slot]).
//       - Reducer -> Puller follows the identical T_0 same-thread release-fence -> st.relaxed.sys [flag2] --(sw.sys)-->
//         T_0 same-thread ld.relaxed.sys [flag2] -> acquire-fence -> bar.sync -> T_j: __ldcg(ag[slot]) chain.
//       - Using __ldcg (L1-bypass) for recv and ag reads guarantees no stale L1 lines from call ep - 2.
//   (2) Ping-pong slot reuse safety (call ep vs call ep + 2, even when M_k varies arbitrarily in 1..1024):
//       - For EVERY call with M >= 1, quarter q = 0 has qrows(0) >= 1 and block j = 0 has nr(0, 0) >= 1.
//       - Thus every call ep + 1 enforces a full 4-rank .sys release-acquire barrier through quarter 0:
//         ranks 1, 2, 3 publish flag1[q=0] = ep + 1 (after finishing call ep), rank 0 waits for all 3 and publishes
//         flag2[q=0] = ep + 1, and ranks 1, 2, 3 wait for flag2[q=0] = ep + 1 before exiting call ep + 1.
//       - Therefore no rank can enter call ep + 2 (reusing slot = ep & 1) until ALL 4 ranks have completely finished
//         call ep, with a transitive .sys happens-before chain between every pair of ranks.
//   (3) Co-residency deadlock freedom:
//       - All 3 roles run in one launch with GP + GR + GA <= occ(kernel) * num_sms (TORCH_CHECK(occ > 0) in init()).
//       - Blocks with nr <= 0 (possible when M < 4 or tail blocks of quarter 3) are skipped identically on all 3 roles.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) TORCH_CHECK((x) == cudaSuccess, #x " failed: ", cudaGetErrorString(cudaGetLastError()))

namespace {

constexpr int H = 5120;
constexpr int NRK = 4;
constexpr int NV = H / 8;                                               // 640 uint4 vectors per row (== threads/CTA)
constexpr int MAXM = 1024;
constexpr int MAXQ = MAXM / NRK;                                        // 256 max rows per quarter
constexpr int NBMAX = MAXQ;                                             // 256 max blocks per quarter (RB >= 1)
constexpr size_t RECV_HALVES = (size_t)(NRK - 1) * MAXQ * H;            // 3 * 256 * 5120 = 3,932,160 halves (7.5 MiB)
constexpr size_t AG_HALVES = (size_t)MAXQ * H;                          // 1 * 256 * 5120 = 1,310,720 halves (2.5 MiB)
constexpr size_t DATA_BYTES = (RECV_HALVES + AG_HALVES) * sizeof(__half);  // 10,485,760 B (10.0 MiB, 4 KiB aligned)
constexpr size_t FLAG_U64S = (size_t)2 * NRK * NBMAX;                   // 2 * 4 * 256 = 2,048 u64s (16 KiB)
constexpr size_t SLOT_BYTES = DATA_BYTES + FLAG_U64S * sizeof(uint64_t);   // 10,502,144 B (4 KiB aligned)
constexpr size_t BUF_BYTES = 2 * SLOT_BYTES;                            // 21,004,288 B (~20.03 MiB)
constexpr unsigned long long TRAP_NS = 5000000000ull;                   // 5 s deadlock guard

struct CfgSpec {
  int rb, gp, gr, ga;
};

// 6 short-M-tuned configs (total <= 80 CTAs -> guaranteed co-resident with occ/SM >= 1 on 80-SM V100)
constexpr CfgSpec kCfgs[6] = {
    {1, 16, 24, 40},  // cfg 0: RB=1 ultra-low-latency streaming (M = 1..128)
    {2, 16, 24, 40},  // cfg 1: RB=2 puller-heavy balanced (M = 64..256)
    {2, 24, 24, 32},  // cfg 2: RB=2 pusher/reducer balanced (M = 64..256, short_ar)
    {4, 16, 24, 40},  // cfg 3: RB=4 puller-heavy (M = 192..512)
    {4, 24, 20, 36},  // cfg 4: RB=4 pusher-heavy (M = 192..768)
    {8, 16, 24, 40},  // cfg 5: RB=8 bulk (M = 512..1024)
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
};

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void fence_sys() {
#if defined(SHORT_ARN_ALIAS_FENCE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("fence.proxy.alias;" ::: "memory");
#endif
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

// Tight-spin on 1 u64 flag via ld.relaxed.sys (zero MEMBAR in spin loop), then ONE same-thread fence.acq_rel.sys
__device__ __forceinline__ void wait_flag1_u64(const uint64_t* p, uint64_t ep) {
  if (ld_rlx_sys_u64(p) != ep) {
    const unsigned long long t0 = gtime();
    do {
      if (gtime() - t0 > TRAP_NS) __trap();
    } while (ld_rlx_sys_u64(p) != ep);
  }
  fence_sys();
}

// Tight-spin on all 3 peer flags by a single thread, then ONE same-thread fence.acq_rel.sys covering all 3
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

__device__ __forceinline__ char* slot_base(const Comm& c, int q, int slot) {
  return base_of(c, q) + (size_t)slot * SLOT_BYTES;
}

__device__ __forceinline__ __half* recv_of(const Comm& c, int q, int slot) {
  return reinterpret_cast<__half*>(slot_base(c, q, slot));
}

__device__ __forceinline__ __half* ag_of(const Comm& c, int q, int slot) {
  return reinterpret_cast<__half*>(slot_base(c, q, slot)) + RECV_HALVES;
}

__device__ __forceinline__ uint64_t* flag1_of(const Comm& c, int q, int slot) {
  return reinterpret_cast<uint64_t*>(slot_base(c, q, slot) + DATA_BYTES);
}

__device__ __forceinline__ uint64_t* flag2_of(const Comm& c, int q, int slot) {
  return flag1_of(c, q, slot) + NRK * NBMAX;
}

__device__ __forceinline__ int slot_idx(int src, int dst) {
  return (src - dst + NRK - 1) & (NRK - 1);
}

__device__ __forceinline__ int qrows(const Args& a, int q) {
  const int lo = q * a.MQ;
  const int hi = a.M < (q + 1) * a.MQ ? a.M : (q + 1) * a.MQ;
  return hi > lo ? hi - lo : 0;
}

__device__ __forceinline__ uint4 ldcg4(const void* p) {
  return __ldcg(reinterpret_cast<const uint4*>(p));
}

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

// Bulk unrolled 16-B vectorized copy (4x unrolled for NVLink burst saturation)
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

// Exact FP32 residual add (complete residual on every rank) + exact IEEE Gemma RMSNorm.
// Double-buffered red[2][20] uses only ONE __syncthreads() per row.
__device__ __forceinline__ void gemma_norm_row(const Args& a, int row, int col, int lane, int warp, int rbuf,
                                               uint4 av, float4 q0, float4 q1, uint4 wv, float (&red)[2][20]) {
  float f[8];
  unpack8(av, f);
  f[0] = __fadd_rn(f[0], q0.x);
  f[1] = __fadd_rn(f[1], q0.y);
  f[2] = __fadd_rn(f[2], q0.z);
  f[3] = __fadd_rn(f[3], q0.w);
  f[4] = __fadd_rn(f[4], q1.x);
  f[5] = __fadd_rn(f[5], q1.y);
  f[6] = __fadd_rn(f[6], q1.z);
  f[7] = __fadd_rn(f[7], q1.w);
  float4* ro = reinterpret_cast<float4*>(a.res_out + (size_t)row * H + col);
  ro[0] = make_float4(f[0], f[1], f[2], f[3]);
  ro[1] = make_float4(f[4], f[5], f[6], f[7]);

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
  const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(tot, (float)H), a.eps)));

  float wf[8], o[8];
  unpack8(wv, wf);
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = __fmul_rn(__fadd_rn(wf[i], 1.f), __fmul_rn(f[i], inv));
  *reinterpret_cast<uint4*>(a.out + (size_t)row * H + col) = pack8(o);
}

// ------------------------------------------------------------------------------------------------ 3 CTA roles

// Role 1 (pushers): units u = 3*j + i (block j of peer quarter q = (rank + 1 + i) & 3)
__device__ __forceinline__ void role_pusher(const Args& a, int cta, uint64_t ep, int slot) {
  const int r = a.c.rank;
  const int nu = 3 * a.NB;
  for (int u = cta; u < nu; u += a.GP) {
    const int j = u / 3, i = u % 3, q = (r + 1 + i) & (NRK - 1);
    const int rem = qrows(a, q) - j * a.RB;
    const int nr = rem < a.RB ? rem : a.RB;
    if (nr <= 0) continue;
    const int sl = slot_idx(r, q);
    const uint4* src = reinterpret_cast<const uint4*>(a.x + (size_t)(q * a.MQ + j * a.RB) * H);
    uint4* dst = reinterpret_cast<uint4*>(recv_of(a.c, q, slot) + ((size_t)sl * MAXQ + j * a.RB) * H);
    copy_rows(dst, src, nr * NV, false);
    __syncthreads();
    if (threadIdx.x == 0) {
      fence_sys();
      st_rlx_sys_u64(flag1_of(a.c, q, slot) + r * NBMAX + j, ep);
    }
  }
}

// Role 2 (reducers): blocks j of own quarter r.
// Publishes ag[slot] + flag2 BEFORE running own-quarter residual+norm so peers' pullers overlap immediately!
template <bool DO_NORM>
__device__ __forceinline__ void role_reducer(const Args& a, int cta, uint64_t ep, int slot, float (&red)[2][20]) {
  const int r = a.c.rank, t = threadIdx.x, lane = t & 31, warp = t >> 5, col = t * 8;
  const int nq = qrows(a, r);
  const __half* const rv = recv_of(a.c, r, slot);
  __half* const ag = ag_of(a.c, r, slot);
  const uint64_t* const f1 = flag1_of(a.c, r, slot);
  const uint4 wv = DO_NORM ? __ldg(reinterpret_cast<const uint4*>(a.w) + t) : make_uint4(0, 0, 0, 0);

  int rbuf = 0;
  for (int j = cta; j < a.NB; j += a.GR) {
    const int rem = nq - j * a.RB;
    const int nr = rem < a.RB ? rem : a.RB;
    if (nr <= 0) break;

    if (t == 0) {
      const uint64_t* const p0 = f1 + ((r + 1) & 3) * NBMAX + j;
      const uint64_t* const p1 = f1 + ((r + 2) & 3) * NBMAX + j;
      const uint64_t* const p2 = f1 + ((r + 3) & 3) * NBMAX + j;
      wait_flag3_u64(p0, p1, p2, ep);
    }
    __syncthreads();

    // Phase R1: strict rank-order 0..3 FP32 sum -> FP16 -> write ag[slot] (and out for bare AR)
    for (int rr = 0; rr < nr; rr++) {
      const int qrow = j * a.RB + rr;
      const int row = r * a.MQ + qrow;
      uint4 pv[NRK];
#pragma unroll
      for (int p = 0; p < NRK; p++) {
        if (p == r) pv[p] = ldcg4(a.x + (size_t)row * H + col);
        else pv[p] = ldcg4(rv + ((size_t)slot_idx(p, r) * MAXQ + qrow) * H + col);
      }
      float s[8];
#pragma unroll
      for (int p = 0; p < NRK; p++) {
        float x[8];
        unpack8(pv[p], x);
#pragma unroll
        for (int e = 0; e < 8; e++) s[e] = (p == 0) ? x[e] : __fadd_rn(s[e], x[e]);
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
        st_rlx_sys_u64(flag2_of(a.c, (r + p) & (NRK - 1), slot) + r * NBMAX + j, ep);
      }
    }

    // Phase R2 (short_arn only): own-quarter FP32 residual add + exact Gemma RMSNorm (overlapped with peers' pulls)
    if constexpr (DO_NORM) {
      for (int rr = 0; rr < nr; rr++, rbuf ^= 1) {
        const int qrow = j * a.RB + rr;
        const int row = r * a.MQ + qrow;
        const uint4 av = *reinterpret_cast<const uint4*>(ag + (size_t)qrow * H + col);
        const float4* rp = reinterpret_cast<const float4*>(a.res + (size_t)row * H + col);
        const float4 q0 = rp[0], q1 = rp[1];
        gemma_norm_row(a, row, col, lane, warp, rbuf, av, q0, q1, wv, red);
      }
    }
  }
}

// Role 3 (pullers): units u = 3*j + i (block j of peer quarter p = (rank + 1 + i) & 3).
// short_ar copies ar_half to out; short_arn fuses FP32 residual add (complete res_out) + exact Gemma RMSNorm.
template <bool DO_NORM>
__device__ __forceinline__ void role_puller(const Args& a, int cta, uint64_t ep, int slot, float (&red)[2][20]) {
  const int r = a.c.rank, t = threadIdx.x, lane = t & 31, warp = t >> 5, col = t * 8;
  const uint64_t* const f2 = flag2_of(a.c, r, slot);
  const int nu = 3 * a.NB;
  const uint4 wv = DO_NORM ? __ldg(reinterpret_cast<const uint4*>(a.w) + t) : make_uint4(0, 0, 0, 0);

  int rbuf = 0;
  for (int u = cta; u < nu; u += a.GA) {
    const int j = u / 3, i = u % 3, p = (r + 1 + i) & (NRK - 1);
    const int rem = qrows(a, p) - j * a.RB;
    const int nr = rem < a.RB ? rem : a.RB;
    if (nr <= 0) continue;

    if (t == 0) wait_flag1_u64(f2 + p * NBMAX + j, ep);
    __syncthreads();

    const __half* const src = ag_of(a.c, p, slot) + (size_t)(j * a.RB) * H;
    if constexpr (!DO_NORM) {
      uint4* dst = reinterpret_cast<uint4*>(a.out + (size_t)(p * a.MQ + j * a.RB) * H);
      copy_rows(dst, reinterpret_cast<const uint4*>(src), nr * NV, true);
    } else {
      for (int rr = 0; rr < nr; rr++, rbuf ^= 1) {
        const int row = p * a.MQ + j * a.RB + rr;
        const uint4 av = ldcg4(src + (size_t)rr * H + col);
        const float4* rp = reinterpret_cast<const float4*>(a.res + (size_t)row * H + col);
        const float4 q0 = rp[0], q1 = rp[1];
        gemma_norm_row(a, row, col, lane, warp, rbuf, av, q0, q1, wv, red);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ kernels
template <bool DO_NORM>
__global__ void __launch_bounds__(NV, 1) k_short_comm(Args a) {
  __shared__ uint64_t s_ep;
  __shared__ float red[2][20];
  if (threadIdx.x == 0) s_ep = ld_acq_gpu_u64(&a.dev_state->epoch);
  __syncthreads();
  const uint64_t ep = s_ep;
  const int slot = (int)(ep & 1ull);

  const int b = blockIdx.x;
  if (b < a.GP) role_pusher(a, b, ep, slot);
  else if (b < a.GP + a.GR) role_reducer<DO_NORM>(a, b - a.GP, ep, slot, red);
  else role_puller<DO_NORM>(a, b - a.GP - a.GR, ep, slot, red);

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
int g_max_co_arn = 80;
int g_max_co_ar = 80;
std::vector<std::string> g_info;

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <class F>
int probe_kernel(const char* name, F fn) {
  cudaFuncAttributes attr{};
  CK(cudaFuncGetAttributes(&attr, fn));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, NV, 0));
  TORCH_CHECK(occ > 0, "short_arn: kernel occupancy is 0 for ", name);
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
  TORCH_CHECK(g_num_sms > 0, "short_arn: invalid SM count ", g_num_sms);
  g_info.clear();
  g_max_co_arn = probe_kernel("k_short_arn", k_short_comm<true>);
  g_max_co_ar = probe_kernel("k_short_ar", k_short_comm<false>);
  for (int i = 0; i < 6; i++) {
    const int tot = kCfgs[i].gp + kCfgs[i].gr + kCfgs[i].ga;
    TORCH_CHECK(tot <= g_max_co_arn && tot <= g_max_co_ar,
                "short_arn: cfg ", i, " needs ", tot, " CTAs but max co-resident is ",
                std::min(g_max_co_arn, g_max_co_ar));
  }
}

void fill_launch_geometry(Args& a, int M, int cfg) {
  const CfgSpec sp = kCfgs[cfg];
  a.M = M;
  a.MQ = (M + NRK - 1) / NRK;
  a.RB = sp.rb;
  a.NB = (a.MQ + a.RB - 1) / a.RB;
  TORCH_CHECK(a.NB >= 1 && a.NB <= NBMAX, "short_arn: NB out of range: ", a.NB);
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
  TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "short_arn: invalid IPC handle size");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, s.data(), sizeof(h));
  void* p = nullptr;
  CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

void init(std::vector<int64_t> bases, int64_t rank) {
  TORCH_CHECK(!g_init, "short_arn: already initialised (re-init not supported)");
  TORCH_CHECK(bases.size() == NRK, "short_arn: expected 4 rank bases");
  TORCH_CHECK(rank >= 0 && rank < NRK, "short_arn: rank must be in 0..3");
  for (int q = 0; q < NRK; q++) {
    TORCH_CHECK(bases[q] != 0, "short_arn: null IPC base pointer for rank ", q);
    TORCH_CHECK((static_cast<uintptr_t>(bases[q]) & 15u) == 0,
                "short_arn: IPC base pointer must be 16-byte aligned for rank ", q);
    for (int p = 0; p < q; p++) {
      TORCH_CHECK(bases[q] != bases[p], "short_arn: duplicate IPC base pointer for ranks ", p, " and ", q);
    }
    g_comm.base[q] = reinterpret_cast<char*>(bases[q]);
  }
  g_comm.rank = (int)rank;

  if (g_dev_state == nullptr) {
    CK(cudaMalloc(&g_dev_state, sizeof(DevState)));
  }
  const DevState init_st{1ull, 0u};
  CK(cudaMemcpy(g_dev_state, &init_st, sizeof(DevState), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());

  probe_all_kernels();
  g_init = true;
}

// Fused TP4 All-Reduce + COMPLETE FP32 Residual Add + Gemma RMSNorm
// Returns {out [M, 5120] fp16, res_out [M, 5120] fp32}
std::vector<torch::Tensor> short_arn(torch::Tensor x, torch::Tensor res, torch::Tensor w, double eps, int64_t cfg) {
  TORCH_CHECK(g_init, "short_arn: not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.size(1) == H && x.is_contiguous(),
              "short_arn: x must be contiguous CUDA fp16 [M, 5120]");
  TORCH_CHECK(res.is_cuda() && res.device() == x.device() && res.dtype() == torch::kFloat &&
                  res.sizes() == x.sizes() && res.is_contiguous(),
              "short_arn: res must be contiguous CUDA fp32 [M, 5120] on same device");
  TORCH_CHECK(w.is_cuda() && w.device() == x.device() && w.dtype() == torch::kHalf && w.numel() == H &&
                  w.is_contiguous(),
              "short_arn: w must be contiguous CUDA fp16 [5120] on same device");
  TORCH_CHECK(is_aligned16(w.data_ptr<at::Half>()), "short_arn: w must be 16-byte aligned");
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 0 && M <= MAXM, "short_arn: M must be in 0..", MAXM, ", got ", M);
  TORCH_CHECK(cfg >= 0 && cfg <= 5, "short_arn: cfg must be in 0..5, got ", cfg);

  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  auto res_out = torch::empty_like(res);
  if (M == 0) return {out, res_out};

  const __half* x_ptr = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  const float* res_ptr = res.data_ptr<float>();
  float* res_out_ptr = res_out.data_ptr<float>();
  const __half* w_ptr = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  TORCH_CHECK(is_aligned16(x_ptr), "short_arn: x must be 16-byte aligned");
  TORCH_CHECK(is_aligned16(res_ptr), "short_arn: res must be 16-byte aligned");
  TORCH_CHECK(is_aligned16(out_ptr), "short_arn: out must be 16-byte aligned");
  TORCH_CHECK(is_aligned16(res_out_ptr), "short_arn: res_out must be 16-byte aligned");

  Args a{};
  a.c = g_comm;
  a.dev_state = g_dev_state;
  a.x = x_ptr;
  a.res = res_ptr;
  a.res_out = res_out_ptr;
  a.w = w_ptr;
  a.out = out_ptr;
  a.eps = (float)eps;
  fill_launch_geometry(a, M, (int)cfg);

  const int grid = a.GP + a.GR + a.GA;
  k_short_comm<true><<<grid, NV, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  CK(cudaGetLastError());
  return {out, res_out};
}

// Bare TP4 All-Reduce (exact FP32 rank-order 0..3 sum -> FP16)
// Returns out [M, 5120] fp16
torch::Tensor short_ar(torch::Tensor x, int64_t cfg) {
  TORCH_CHECK(g_init, "short_ar: not initialised");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.size(1) == H && x.is_contiguous(),
              "short_ar: x must be contiguous CUDA fp16 [M, 5120]");
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 0 && M <= MAXM, "short_ar: M must be in 0..", MAXM, ", got ", M);
  TORCH_CHECK(cfg >= 0 && cfg <= 5, "short_ar: cfg must be in 0..5, got ", cfg);

  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  if (M == 0) return out;

  const __half* x_ptr = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  TORCH_CHECK(is_aligned16(x_ptr), "short_ar: x must be 16-byte aligned");
  TORCH_CHECK(is_aligned16(out_ptr), "short_ar: out must be 16-byte aligned");

  Args a{};
  a.c = g_comm;
  a.dev_state = g_dev_state;
  a.x = x_ptr;
  a.out = out_ptr;
  fill_launch_geometry(a, M, (int)cfg);

  const int grid = a.GP + a.GR + a.GA;
  k_short_comm<false><<<grid, NV, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  CK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buf_bytes", &buf_bytes, "IPC buffer size in bytes per rank (~20.0 MiB, 2-slot ping-pong)");
  m.def("max_rows", &max_rows, "maximum supported rows M (1024)");
  m.def("alloc_buf", &alloc_buf, "allocate and zero-initialize the per-rank IPC buffer");
  m.def("ipc_handle", &ipc_handle, "export cudaIpcMemHandle for local buffer");
  m.def("ipc_open", &ipc_open, "open peer cudaIpcMemHandle");
  m.def("init", &init, "bind all 4 rank IPC pointers, init DevState epoch=1, and probe kernel occupancy");
  m.def("info", &info, "kernel register, SMEM, and occupancy diagnostics");
  m.def("short_arn", &short_arn, "TP4 short-prefill all-reduce + complete FP32 residual add + Gemma RMSNorm",
        py::arg("x"), py::arg("res"), py::arg("w"), py::arg("eps"), py::arg("cfg") = 0);
  m.def("short_ar", &short_ar, "TP4 short-prefill bare all-reduce (fp16)",
        py::arg("x"), py::arg("cfg") = 0);
}
