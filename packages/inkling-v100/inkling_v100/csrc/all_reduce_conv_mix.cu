// SPDX-License-Identifier: Apache-2.0
// all_reduce_conv_mix.cu - Fused 1-shot Volta (sm_70) TP4 NVLink P2P AllReduce + optional MoE
// topk=8 combine + W=4 paged short_conv + FP32 AttnRes mix (Q in {1,3}) + optional FP16 RMSNorm
// for Inkling-2-Mini (H = 2304, T = 1..64, CUDA-graph safe, zero spills).
//
// Architecture (1-shot direct NVLink P2P push, 1 CTA of 288 threads per token t in [0, T)):
//   - H = 2304 = 288 * 8: every thread owns 8 contiguous elements (1 uint4 of FP16 / 2 float4 FP32).
//     Because stream_width = 1152 = 144 * 8, all 8 elements of each thread belong to ONE conv head
//     (ch = tid / 144 in {0, 1}), giving 100% coalesced uint4 conv-cache reads & writes!
//   - 2-slot ping-pong IPC buffer (~1.74 MiB/rank):
//       Per slot s in {0, 1}: recv[3][MAX_T][2304] fp16 + flag[4][MAX_T] u64.
//   - Device-side monotonic uint64_t epoch (DevState { uint64_t epoch; uint32_t done; }):
//       Zero host epoch argument -> 100% CUDA-graph safe across arbitrary odd/even chains and
//       varying T in [1..64].
//   - Formal race-freedom (PTX 6.0+ / sm_70):
//       (1) Within call ep, token t: Pusher writes recv[slot] -> __syncthreads() -> tid==0:
//           fence.acq_rel.sys -> st.relaxed.sys [flag[peer][slot][rank][t]], ep.
//           Receiver tid==0 spins on ld.relaxed.sys [flag[rank][slot][p][t]] == ep for all 3 peers
//           -> ONE fence.acq_rel.sys -> __syncthreads() -> __ldcg(recv[slot]).
//       (2) Ping-pong slot reuse across varying T_k >= 1: token t=0 executes on EVERY call ep+1,
//           forming a full 4-rank transitive .sys happens-before barrier before any rank can enter
//           call ep+2 and reuse slot = ep & 1.
//       (3) Intra-batch Q > 1 (e.g., DSpark verify Q=8): each CTA writes reduced cur_h to local
//           ar_tmp[t] and signals ar_ready[t] = ep via fence.acq_rel.gpu + st.relaxed.gpu. Since
//           grid = T <= 64 <= occ * 80 SMs, all T CTAs are co-resident and deadlock-free; for Q=1
//           decode (t == qstart), the intra-batch branch is never taken.

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

constexpr int H = 2304;
constexpr int NRK = 4;
constexpr int NV = H / 8;                                                  // 288 threads/CTA (9 warps)
constexpr int NWARPS = NV / 32;                                            // 9 warps
constexpr int MAX_T = 64;
constexpr int W_CONV = 4;
constexpr int TOPK_C = 8;
constexpr int MAX_NSTACK = 5;                                              // max stack depth (0..4)
constexpr size_t RECV_HALVES = (size_t)(NRK - 1) * MAX_T * H;              // 3 * 64 * 2304 = 442,368 halves
constexpr size_t DATA_BYTES = RECV_HALVES * sizeof(__half);                // 884,736 B (4 KiB aligned!)
constexpr size_t FLAG_U64S = (size_t)NRK * MAX_T;                          // 4 * 64 = 256 u64s (2 KiB)
constexpr size_t SLOT_BYTES = DATA_BYTES + FLAG_U64S * sizeof(uint64_t);   // 886,784 B (256 B aligned)
constexpr size_t BUF_BYTES = 2 * SLOT_BYTES;                               // 1,773,568 B (~1.69 MiB)
constexpr unsigned long long TRAP_NS = 5000000000ull;                      // 5 s deadlock guard

struct Comm {
  char* base[NRK];
  int rank;
};

struct DevState {
  uint64_t epoch;
  uint32_t done;
};

struct LocalScratch {
  __half ar_tmp[MAX_T * H];
  uint64_t ar_ready[MAX_T];
};

struct Args {
  Comm c;
  DevState* dev_state;
  LocalScratch* scratch;

  // Input: either x [T, H] fp16 OR (down [T*8, H] fp16, src2dst [T*8] int32)
  const __half* x;
  const __half* down;
  const int32_t* src2dst;

  // Short-conv inputs
  const __half* conv_weight;      // [H, 4] fp16 contiguous
  __half* cache;                  // [num_blocks, H_kv, N, head_size] fp16
  int64_t stride_c_blk;
  int64_t stride_c_h;
  int64_t stride_c_n;
  int cache_N;
  int cache_H;
  int stream_offset;
  int stream_width;
  float conv_scale;               // 5^-0.5

  const int64_t* positions;       // [T]
  const int32_t* block_table;     // [num_reqs, max_blocks]
  int64_t stride_bt;
  int max_blocks;
  const int32_t* seq_idx;         // [T]
  const int64_t* slot_mapping;    // [T]
  const int32_t* query_start;     // [T]

  // Residual/addend + conv output (all FP32 [T, H])
  const float* addend;
  const float* residual;
  float* conv_out;

  // AttnRes inputs
  const float* stack;             // [N_STACK, T, H] fp32 (stride_s between stack planes)
  int64_t stride_s;
  int n_stack;                    // 0..4
  const float* last;              // optional [T, H] fp32 (added to conv_out when HAS_LAST)
  const float* queries;           // [Q, H] fp32
  const __half* norm_w;           // [Q, H] fp16 (when APPLY_NORM)
  float eps;
  float norm_eps;

  // Output: [Q, T, H] fp16 (APPLY_NORM) or fp32 (!APPLY_NORM)
  void* out;
  int T;
  int rehearse;
};

// ----------------------------------------------------------------------------- PTX helpers
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ float shfl_bfly_f32(float v, int o) {
  float r;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(o));
  return r;
}

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void fence_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void fence_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
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

__device__ __forceinline__ void wait_flag3_u64(
    const uint64_t* p0, const uint64_t* p1, const uint64_t* p2, uint64_t ep) {
  if (ld_rlx_sys_u64(p0) != ep || ld_rlx_sys_u64(p1) != ep || ld_rlx_sys_u64(p2) != ep) {
    const unsigned long long t0 = gtime();
    do {
      if (gtime() - t0 > TRAP_NS) __trap();
    } while (ld_rlx_sys_u64(p0) != ep || ld_rlx_sys_u64(p1) != ep || ld_rlx_sys_u64(p2) != ep);
  }
  fence_sys();
}

__device__ __forceinline__ void wait_local_u64(const uint64_t* p, uint64_t ep) {
  if (ld_acq_gpu_u64(p) != ep) {
    const unsigned long long t0 = gtime();
    do {
      if (gtime() - t0 > TRAP_NS) __trap();
    } while (ld_acq_gpu_u64(p) != ep);
  }
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

__device__ __forceinline__ uint64_t* flag_of(const Comm& c, int q, int slot) {
  return reinterpret_cast<uint64_t*>(slot_base(c, q, slot) + DATA_BYTES);
}

__device__ __forceinline__ int slot_idx(int src, int dst) {
  return (src - dst + NRK - 1) & (NRK - 1);
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

__device__ __forceinline__ void load_f32_8(const float* p, float (&f)[8]) {
  const float4* fp = reinterpret_cast<const float4*>(p);
  const float4 a = fp[0];
  const float4 b = fp[1];
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
  f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

__device__ __forceinline__ void store_f32_8(float* p, const float (&f)[8]) {
  float4* fp = reinterpret_cast<float4*>(p);
  fp[0] = make_float4(f[0], f[1], f[2], f[3]);
  fp[1] = make_float4(f[4], f[5], f[6], f[7]);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = __fadd_rn(v, shfl_bfly_f32(v, o));
  return v;
}

template <int M>
__device__ __forceinline__ void block_sum_m(
    float (&vals)[M], int lane, int warp, int rbuf, float (&red)[2][4][NWARPS]) {
#pragma unroll
  for (int m = 0; m < M; m++) {
    const float wv = warp_sum(vals[m]);
    if (lane == 0) red[rbuf][m][warp] = wv;
  }
  __syncthreads();
#pragma unroll
  for (int m = 0; m < M; m++) {
    float tot = red[rbuf][m][0];
#pragma unroll
    for (int k = 1; k < NWARPS; k++) tot = __fadd_rn(tot, red[rbuf][m][k]);
    vals[m] = tot;
  }
}

// ----------------------------------------------------------------------------- Fused Kernel
template <bool HAS_COMBINE, bool HAS_ADDEND, bool HAS_RES, bool HAS_LAST, bool APPLY_NORM, int Q_C>
__global__ void __launch_bounds__(NV, 2)
k_ar_conv_mix(Args a) {
  __shared__ uint64_t s_ep;
  __shared__ float red[2][4][NWARPS];
  __shared__ float s_inv_rms[MAX_NSTACK];
  __shared__ float s_logits[3][MAX_NSTACK];
  __shared__ float s_probs[3][MAX_NSTACK];
  __shared__ int64_t s_slot_m;
  __shared__ int64_t s_pos;
  __shared__ int s_req;
  __shared__ int s_qstart;
  __shared__ int s_blk[W_CONV];
  __shared__ int s_row_mode[W_CONV];  // 0=zero, 1=cur_reg, 2=intra_ar_tmp, 3=cache
  __shared__ int s_need_rehearse;

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col = tid * 8;
  const int r = a.c.rank;

  if (tid == 0) {
    s_ep = ld_acq_gpu_u64(&a.dev_state->epoch);
    s_slot_m = a.slot_mapping[t];
    s_pos = a.positions[t];
    s_req = a.seq_idx[t];
    s_qstart = a.query_start[t];
  }
  __syncthreads();
  const uint64_t ep = s_ep;
  const int slot = (int)(ep & 1ull);

  // ---------------- Step 1: Local partial (or MoE topk=8 combine) + 1-shot NVLink push
  uint4 my_v;
  if constexpr (HAS_COMBINE) {
    float cacc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int base_s = t * TOPK_C;
#pragma unroll
    for (int k = 0; k < TOPK_C; k++) {
      const int row = a.src2dst[base_s + k];
      if (row >= 0) {
        const uint4 dv = __ldg(reinterpret_cast<const uint4*>(a.down + (size_t)row * H + col));
        float df[8];
        unpack8(dv, df);
#pragma unroll
        for (int e = 0; e < 8; e++) cacc[e] = __fadd_rn(cacc[e], df[e]);
      }
    }
    my_v = pack8(cacc);
  } else {
    my_v = __ldcs(reinterpret_cast<const uint4*>(a.x + (size_t)t * H + col));
  }

  // Push my_v to all 3 peers' recv[slot][peer_slot][t] BEFORE anything else
#pragma unroll
  for (int i = 1; i < NRK; i++) {
    const int q = (r + i) & (NRK - 1);
    const int sl = slot_idx(r, q);
    __half* dst = recv_of(a.c, q, slot) + ((size_t)sl * MAX_T + t) * H + col;
    *reinterpret_cast<uint4*>(dst) = my_v;
  }
  __syncthreads();
  const uint64_t* const f_base = flag_of(a.c, r, slot);
  const uint64_t* const f_p1 = f_base + (size_t)((r + 1) & 3) * MAX_T + t;
  const uint64_t* const f_p2 = f_base + (size_t)((r + 2) & 3) * MAX_T + t;
  const uint64_t* const f_p3 = f_base + (size_t)((r + 3) & 3) * MAX_T + t;
  if (tid == 0) {
    fence_sys();
#pragma unroll
    for (int i = 1; i < NRK; i++) {
      const int q = (r + i) & (NRK - 1);
      st_rlx_sys_u64(flag_of(a.c, q, slot) + (size_t)r * MAX_T + t, ep);
    }
    // Precompute conv tap block table lookups while peers' pushes are in flight!
    const bool valid = s_slot_m >= 0;
    const int32_t* bt_row = a.block_table + (int64_t)s_req * a.stride_bt;
    for (int k = 0; k < W_CONV; k++) {
      const int lag = (W_CONV - 1) - k;
      const int64_t src = s_pos - lag;
      const int row = t - lag;
      if (!valid || src < 0) {
        s_row_mode[k] = 0;
        s_blk[k] = 0;
      } else if (lag == 0) {
        s_row_mode[k] = 1;  // cur_h in registers
        s_blk[k] = 0;
      } else if (row >= s_qstart) {
        s_row_mode[k] = 2;  // intra-batch ar_tmp[row]
        s_blk[k] = row;
      } else {
        s_row_mode[k] = 3;  // conv cache
        int lblk = (int)(src / a.cache_N);
        if (lblk >= a.max_blocks) lblk = a.max_blocks - 1;
        s_blk[k] = bt_row[lblk];
      }
    }
    // Check if peers have already arrived; only run the I-cache warmup pass when peers are in flight!
    const bool ready_now =
        (ld_rlx_sys_u64(f_p1) == ep && ld_rlx_sys_u64(f_p2) == ep && ld_rlx_sys_u64(f_p3) == ep);
    s_need_rehearse = (a.rehearse && !ready_now) ? 1 : 0;
  }
  __syncthreads();

  // Pull independent conv/residual/stack/query weights & cache lines toward L2 during NVLink flight!
  const int ch = col / a.stream_width;  // 0 or 1 (since stream_width == 1152 == 144*8)
  const int c_in_h = a.stream_offset + (col - ch * a.stream_width);
  const int64_t slot_m = s_slot_m;
  const int64_t pos = s_pos;
  const uint4* w_ptr = reinterpret_cast<const uint4*>(a.conv_weight + (size_t)col * W_CONV);
  prefetch_l2(w_ptr);
  prefetch_l2(w_ptr + 2);
  if constexpr (HAS_ADDEND) prefetch_l2(a.addend + (size_t)t * H + col);
  if constexpr (HAS_RES) prefetch_l2(a.residual + (size_t)t * H + col);
  if constexpr (HAS_LAST) prefetch_l2(a.last + (size_t)t * H + col);
  for (int n = 0; n < a.n_stack; n++) {
    prefetch_l2(a.stack + (int64_t)n * a.stride_s + (size_t)t * H + col);
  }
#pragma unroll
  for (int q = 0; q < Q_C; q++) {
    prefetch_l2(a.queries + (size_t)q * H + col);
    if constexpr (APPLY_NORM) prefetch_l2(a.norm_w + (size_t)q * H + col);
  }
#pragma unroll
  for (int k = 0; k < W_CONV - 1; k++) {
    if (s_row_mode[k] == 3) {
      const int lag = (W_CONV - 1) - k;
      const int64_t src_pos = pos - lag;
      const int64_t blk = (int64_t)s_blk[k];
      const int64_t off_n = src_pos % a.cache_N;
      prefetch_l2(a.cache + blk * a.stride_c_blk + (int64_t)ch * a.stride_c_h + off_n * a.stride_c_n + c_in_h);
    }
  }

  // Pass 0 (optional, ONLY when peers are late): warm SM I-cache on local partial with all stores off.
  // Pass 1 (always): wait for peers, reduce in exact rank order 0..3, run conv + AttnRes + norm, store.
  const __half* const rv = recv_of(a.c, r, slot);
#pragma unroll 1
  for (int pass = s_need_rehearse ? 0 : 1; pass < 2; pass++) {
    // ---------------- Step 2: Rank-order 0..3 FP32 AllReduce -> FP16 cur_h
    uint4 cur_h = my_v;
    float cur_f[8];
    if (pass) {
      if (tid == 0) {
        wait_flag3_u64(f_p1, f_p2, f_p3, ep);
      }
      __syncthreads();
#pragma unroll
      for (int p = 0; p < NRK; p++) {
        const uint4 pv = (p == r) ? my_v : ldcg4(rv + ((size_t)slot_idx(p, r) * MAX_T + t) * H + col);
        float pf[8];
        unpack8(pv, pf);
#pragma unroll
        for (int e = 0; e < 8; e++) cur_f[e] = (p == 0) ? pf[e] : __fadd_rn(cur_f[e], pf[e]);
      }
      cur_h = pack8(cur_f);
    }
    // Exact FP16-rounded cur_f matching `_conv` (`cur = x.to(dtype); ... + cur.to(fp32)`):
    unpack8(cur_h, cur_f);

    // If intra-batch Q > 1 might be present (t > s_qstart or t + 1 < T), publish cur_h to ar_tmp[t].
    if (pass && a.T > 1) {
      *reinterpret_cast<uint4*>(a.scratch->ar_tmp + (size_t)t * H + col) = cur_h;
      __syncthreads();
      if (tid == 0) {
        fence_gpu();
        st_rel_gpu_u64(&a.scratch->ar_ready[t], ep);
      }
    }

    // ---------------- Step 3: Fused W=4 Short Conv + Cache Write + Residual/Addend -> FP32 conv_out
    if (pass && slot_m >= 0) {
      const int64_t blk = slot_m / a.cache_N;
      const int64_t off_n = slot_m % a.cache_N;
      __half* cdst = a.cache + blk * a.stride_c_blk + (int64_t)ch * a.stride_c_h + off_n * a.stride_c_n + c_in_h;
      *reinterpret_cast<uint4*>(cdst) = cur_h;
    }

    // Load static conv weights [col..col+7, 0..3] packed as 4 uint4 (16 regs instead of 32 floats)
    const uint4 wv0 = __ldg(w_ptr + 0);
    const uint4 wv1 = __ldg(w_ptr + 1);
    const uint4 wv2 = __ldg(w_ptr + 2);
    const uint4 wv3 = __ldg(w_ptr + 3);
    const __half* wh0 = reinterpret_cast<const __half*>(&wv0);
    const __half* wh1 = reinterpret_cast<const __half*>(&wv1);
    const __half* wh2 = reinterpret_cast<const __half*>(&wv2);
    const __half* wh3 = reinterpret_cast<const __half*>(&wv3);

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < W_CONV; k++) {
      const int mode = s_row_mode[k];
      if (mode == 0) continue;
      float xt[8];
      if (mode == 1 || (!pass && mode == 2)) {
#pragma unroll
        for (int e = 0; e < 8; e++) xt[e] = cur_f[e];
      } else if (mode == 2) {
        const int src_row = s_blk[k];
        if (tid == 0) wait_local_u64(&a.scratch->ar_ready[src_row], ep);
        __syncthreads();
        const uint4 tv = ldcg4(a.scratch->ar_tmp + (size_t)src_row * H + col);
        unpack8(tv, xt);
      } else {
        const int lag = (W_CONV - 1) - k;
        const int64_t src_pos = pos - lag;
        const int64_t blk = (int64_t)s_blk[k];
        const int64_t off_n = src_pos % a.cache_N;
        const __half* csrc =
            a.cache + blk * a.stride_c_blk + (int64_t)ch * a.stride_c_h + off_n * a.stride_c_n + c_in_h;
        const uint4 tv = __ldg(reinterpret_cast<const uint4*>(csrc));
        unpack8(tv, xt);
      }
      acc[0] = __fmaf_rn(xt[0], __half2float(wh0[k]), acc[0]);
      acc[1] = __fmaf_rn(xt[1], __half2float(wh0[4 + k]), acc[1]);
      acc[2] = __fmaf_rn(xt[2], __half2float(wh1[k]), acc[2]);
      acc[3] = __fmaf_rn(xt[3], __half2float(wh1[4 + k]), acc[3]);
      acc[4] = __fmaf_rn(xt[4], __half2float(wh2[k]), acc[4]);
      acc[5] = __fmaf_rn(xt[5], __half2float(wh2[4 + k]), acc[5]);
      acc[6] = __fmaf_rn(xt[6], __half2float(wh3[k]), acc[6]);
      acc[7] = __fmaf_rn(xt[7], __half2float(wh3[4 + k]), acc[7]);
    }

    float y[8];
#pragma unroll
    for (int e = 0; e < 8; e++) y[e] = __fmul_rn(__fadd_rn(acc[e], cur_f[e]), a.conv_scale);

    if constexpr (HAS_ADDEND) {
      float ad[8];
      load_f32_8(a.addend + (size_t)t * H + col, ad);
      if constexpr (HAS_RES) {
        float rs[8];
        load_f32_8(a.residual + (size_t)t * H + col, rs);
#pragma unroll
        for (int e = 0; e < 8; e++) ad[e] = __fadd_rn(rs[e], ad[e]);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) y[e] = __fadd_rn(ad[e], y[e]);
    }
    if (pass) store_f32_8(a.conv_out + (size_t)t * H + col, y);

    // ---------------- Step 4: Fused AttnRes Mix + Optional FP16 RMSNorm (y still in registers!)
    float last_src[8];
    if constexpr (HAS_LAST) {
      float lf[8];
      load_f32_8(a.last + (size_t)t * H + col, lf);
#pragma unroll
      for (int e = 0; e < 8; e++) last_src[e] = __fadd_rn(lf[e], y[e]);
    } else {
#pragma unroll
      for (int e = 0; e < 8; e++) last_src[e] = y[e];
    }

    const int n_stack = a.n_stack;
    const int n_tot = n_stack + 1;
    const float inv_H = 1.0f / (float)H;
    int rbuf = 0;

    // Pass 1: compute rstd[n] (shared across Q_C) and dot(q, x_n) in ONE barrier per n!
    for (int n = 0; n < n_tot; n++) {
      float xn[8];
      if (n < n_stack) {
        load_f32_8(a.stack + (int64_t)n * a.stride_s + (size_t)t * H + col, xn);
      } else {
#pragma unroll
        for (int e = 0; e < 8; e++) xn[e] = last_src[e];
      }
      float sums[Q_C + 1];
      float ss = __fmul_rn(xn[0], xn[0]);
#pragma unroll
      for (int e = 1; e < 8; e++) ss = __fmaf_rn(xn[e], xn[e], ss);
      sums[0] = ss;
#pragma unroll
      for (int q = 0; q < Q_C; q++) {
        float qf[8];
        load_f32_8(a.queries + (size_t)q * H + col, qf);
        float dt = __fmul_rn(qf[0], xn[0]);
#pragma unroll
        for (int e = 1; e < 8; e++) dt = __fmaf_rn(qf[e], xn[e], dt);
        sums[1 + q] = dt;
      }
      block_sum_m<Q_C + 1>(sums, lane, warp, rbuf, red);
      rbuf ^= 1;
      if (tid == 0) {
        const float rstd = rsqrtf(__fmaf_rn(sums[0], inv_H, a.eps));
        s_inv_rms[n] = rstd;
#pragma unroll
        for (int q = 0; q < Q_C; q++) {
          s_logits[q][n] = __fmul_rn(sums[1 + q], rstd);
        }
      }
    }
    if (tid == 0) {
#pragma unroll
      for (int q = 0; q < Q_C; q++) {
        float mx = s_logits[q][0];
        for (int n = 1; n < n_tot; n++) mx = fmaxf(mx, s_logits[q][n]);
        float psum = 0.0f;
        for (int n = 0; n < n_tot; n++) {
          const float pv = expf(s_logits[q][n] - mx);
          s_probs[q][n] = pv;
          psum = __fadd_rn(psum, pv);
        }
        const float inv_sum = __fdiv_rn(1.0f, psum);
        for (int n = 0; n < n_tot; n++) {
          s_probs[q][n] = __fmul_rn(__fmul_rn(s_probs[q][n], s_inv_rms[n]), inv_sum);
        }
      }
    }
    __syncthreads();

    // Pass 2: weighted mix + optional RMSNorm per query q in 0..Q_C-1
#pragma unroll
    for (int q = 0; q < Q_C; q++) {
      float mix[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int n = 0; n < n_tot; n++) {
        float xn[8];
        if (n < n_stack) {
          load_f32_8(a.stack + (int64_t)n * a.stride_s + (size_t)t * H + col, xn);
        } else {
#pragma unroll
          for (int e = 0; e < 8; e++) xn[e] = last_src[e];
        }
        const float pn = s_probs[q][n];
#pragma unroll
        for (int e = 0; e < 8; e++) mix[e] = __fmaf_rn(pn, xn[e], mix[e]);
      }

      const size_t out_off = ((size_t)q * a.T + t) * H + col;
      if constexpr (APPLY_NORM) {
        float s1[1];
        float ss = __fmul_rn(mix[0], mix[0]);
#pragma unroll
        for (int e = 1; e < 8; e++) ss = __fmaf_rn(mix[e], mix[e], ss);
        s1[0] = ss;
        block_sum_m<1>(s1, lane, warp, rbuf, red);
        rbuf ^= 1;
        const float out_rstd = rsqrtf(__fadd_rn(__fdiv_rn(s1[0], (float)H), a.norm_eps));
        const uint4 wv = __ldg(reinterpret_cast<const uint4*>(a.norm_w + (size_t)q * H + col));
        float wf[8];
        unpack8(wv, wf);
#pragma unroll
        for (int e = 0; e < 8; e++) mix[e] = __fmul_rn(__fmul_rn(mix[e], out_rstd), wf[e]);
        if (pass) *reinterpret_cast<uint4*>(static_cast<__half*>(a.out) + out_off) = pack8(mix);
      } else {
        if (pass) store_f32_8(static_cast<float*>(a.out) + out_off, mix);
      }
    }
    if (!pass) __syncthreads();
  }

  // ---------------- Step 5: Advance device-side monotonic epoch on strictly last CTA
  __syncthreads();
  if (tid == 0) {
    if (atom_add_acq_rel_gpu_u32(&a.dev_state->done, 1u) == (unsigned)(gridDim.x - 1)) {
      st_rlx_gpu_u32(&a.dev_state->done, 0u);
      st_rel_gpu_u64(&a.dev_state->epoch, ep + 1ull);
    }
  }
}

// ----------------------------------------------------------------------------- Host state
Comm g_comm{};
DevState* g_dev_state = nullptr;
LocalScratch* g_scratch = nullptr;
bool g_init = false;
int g_num_sms = 80;
std::vector<std::string> g_info;

template <class F>
int probe_kernel(const char* name, F fn) {
  cudaFuncAttributes attr{};
  CK(cudaFuncGetAttributes(&attr, fn));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, NV, 0));
  TORCH_CHECK(occ > 0, "all_reduce_conv_mix: occupancy is 0 for ", name);
  const int max_co = occ * g_num_sms;
  TORCH_CHECK(max_co >= MAX_T, "all_reduce_conv_mix: max co-resident CTAs ", max_co,
              " < MAX_T ", MAX_T, " for ", name);
  g_info.push_back(std::string(name) + " regs=" + std::to_string(attr.numRegs) +
                   " lmem=" + std::to_string(attr.localSizeBytes) +
                   " smem=" + std::to_string(attr.sharedSizeBytes) +
                   " occ/SM=" + std::to_string(occ) +
                   " co_ctas=" + std::to_string(max_co));
  return max_co;
}

void probe_all() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev));
  g_info.clear();
  probe_kernel("attn_last0_norm1", k_ar_conv_mix<false, false, false, false, true, 1>);
  probe_kernel("attn_last1_norm1", k_ar_conv_mix<false, false, false, true, true, 1>);
  probe_kernel("attn_last0_norm0", k_ar_conv_mix<false, false, false, false, false, 1>);
  probe_kernel("attn_last1_norm0", k_ar_conv_mix<false, false, false, true, false, 1>);
  probe_kernel("moe_res0_q1_norm1", k_ar_conv_mix<true, true, false, false, true, 1>);
  probe_kernel("moe_res1_q1_norm1", k_ar_conv_mix<true, true, true, false, true, 1>);
  probe_kernel("moe_res0_q3_norm1", k_ar_conv_mix<true, true, false, false, true, 3>);
  probe_kernel("moe_res1_q3_norm1", k_ar_conv_mix<true, true, true, false, true, 3>);
  probe_kernel("moe_res0_q1_norm0", k_ar_conv_mix<true, true, false, false, false, 1>);
  probe_kernel("moe_res1_q1_norm0", k_ar_conv_mix<true, true, true, false, false, 1>);
  probe_kernel("moe_res0_q3_norm0", k_ar_conv_mix<true, true, false, false, false, 3>);
  probe_kernel("moe_res1_q3_norm0", k_ar_conv_mix<true, true, true, false, false, 3>);
  probe_kernel("moe_direct_res0_q1_norm1", k_ar_conv_mix<false, true, false, false, true, 1>);
  probe_kernel("moe_direct_res1_q1_norm1", k_ar_conv_mix<false, true, true, false, true, 1>);
  probe_kernel("moe_direct_res0_q3_norm1", k_ar_conv_mix<false, true, false, false, true, 3>);
  probe_kernel("moe_direct_res1_q3_norm1", k_ar_conv_mix<false, true, true, false, true, 3>);
  probe_kernel("moe_direct_res0_q1_norm0", k_ar_conv_mix<false, true, false, false, false, 1>);
  probe_kernel("moe_direct_res1_q1_norm0", k_ar_conv_mix<false, true, true, false, false, 1>);
  probe_kernel("moe_direct_res0_q3_norm0", k_ar_conv_mix<false, true, false, false, false, 3>);
  probe_kernel("moe_direct_res1_q3_norm0", k_ar_conv_mix<false, true, true, false, false, 3>);
}

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

}  // namespace

int64_t buf_bytes() { return (int64_t)BUF_BYTES; }
int64_t max_tokens() { return MAX_T; }
bool is_initialized() { return g_init; }
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
  TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "invalid IPC handle size");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, s.data(), sizeof(h));
  void* p = nullptr;
  CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

void init(std::vector<int64_t> bases, int64_t rank) {
  TORCH_CHECK(!g_init, "all_reduce_conv_mix already initialized");
  TORCH_CHECK(bases.size() == NRK && rank >= 0 && rank < NRK);
  for (int q = 0; q < NRK; q++) {
    TORCH_CHECK(bases[q] != 0 && (static_cast<uintptr_t>(bases[q]) & 15u) == 0);
    for (int p = 0; p < q; p++) TORCH_CHECK(bases[q] != bases[p]);
    g_comm.base[q] = reinterpret_cast<char*>(bases[q]);
  }
  g_comm.rank = (int)rank;

  if (g_dev_state == nullptr) CK(cudaMalloc(&g_dev_state, sizeof(DevState)));
  const DevState init_st{1ull, 0u};
  CK(cudaMemcpy(g_dev_state, &init_st, sizeof(DevState), cudaMemcpyHostToDevice));

  if (g_scratch == nullptr) {
    CK(cudaMalloc(&g_scratch, sizeof(LocalScratch)));
    CK(cudaMemset(g_scratch, 0, sizeof(LocalScratch)));
  }
  CK(cudaDeviceSynchronize());
  probe_all();
  g_init = true;
}

torch::Tensor ar_conv_mix_forward(
    c10::optional<torch::Tensor> x_opt,
    c10::optional<torch::Tensor> down_opt,
    c10::optional<torch::Tensor> src2dst_opt,
    torch::Tensor conv_weight,
    torch::Tensor cache,
    torch::Tensor positions,
    torch::Tensor block_table,
    torch::Tensor seq_idx,
    torch::Tensor slot_mapping,
    torch::Tensor query_start,
    int64_t stream_offset,
    int64_t stream_width,
    c10::optional<torch::Tensor> addend_opt,
    c10::optional<torch::Tensor> residual_opt,
    torch::Tensor conv_out,
    torch::Tensor stack,
    c10::optional<torch::Tensor> last_opt,
    torch::Tensor queries,
    c10::optional<torch::Tensor> norm_w_opt,
    double eps,
    double norm_eps,
    int64_t rehearse) {
  TORCH_CHECK(g_init, "all_reduce_conv_mix not initialized");
  const int T = (int)conv_out.size(0);
  const int Q = (int)queries.size(0);
  const int n_stack = (int)stack.size(0);
  TORCH_CHECK(T >= 1 && T <= MAX_T, "T out of range: ", T);
  TORCH_CHECK(conv_out.size(1) == H && conv_out.dtype() == torch::kFloat && conv_out.is_contiguous());
  TORCH_CHECK(Q == 1 || Q == 3, "Q must be 1 or 3, got ", Q);
  TORCH_CHECK(n_stack >= 0 && n_stack < MAX_NSTACK, "n_stack out of range: ", n_stack);
  TORCH_CHECK(stream_width % 8 == 0 && stream_offset % 8 == 0 && H == cache.size(1) * stream_width);
  TORCH_CHECK(conv_weight.dtype() == torch::kHalf && conv_weight.is_contiguous() &&
              conv_weight.size(0) == H && conv_weight.size(1) == W_CONV);
  TORCH_CHECK(cache.dtype() == torch::kHalf && cache.stride(3) == 1);
  TORCH_CHECK(queries.dtype() == torch::kFloat && queries.is_contiguous());
  TORCH_CHECK(stack.dtype() == torch::kFloat && stack.stride(1) == H && stack.stride(2) == 1);

  const bool has_combine = down_opt.has_value() && down_opt.value().defined();
  const bool has_addend = addend_opt.has_value() && addend_opt.value().defined();
  const bool has_res = residual_opt.has_value() && residual_opt.value().defined();
  const bool has_last = last_opt.has_value() && last_opt.value().defined();
  const bool apply_norm = norm_w_opt.has_value() && norm_w_opt.value().defined();

  const at::cuda::CUDAGuard guard(conv_out.device());
  auto out = torch::empty(
      {Q, T, H},
      conv_out.options().dtype(apply_norm ? torch::kHalf : torch::kFloat));

  Args a{};
  a.c = g_comm;
  a.dev_state = g_dev_state;
  a.scratch = g_scratch;
  a.T = T;
  a.rehearse = (rehearse != 0 && T <= 16) ? 1 : 0;

  if (has_combine) {
    TORCH_CHECK(src2dst_opt.has_value() && src2dst_opt.value().defined());
    auto& down = down_opt.value();
    auto& s2d = src2dst_opt.value();
    TORCH_CHECK(down.dtype() == torch::kHalf && down.is_contiguous() && down.size(1) == H);
    TORCH_CHECK(s2d.dtype() == torch::kInt32 && s2d.is_contiguous() && s2d.numel() == T * TOPK_C);
    a.down = reinterpret_cast<const __half*>(down.data_ptr<at::Half>());
    a.src2dst = s2d.data_ptr<int32_t>();
    TORCH_CHECK(is_aligned16(a.down));
  } else {
    TORCH_CHECK(x_opt.has_value() && x_opt.value().defined());
    auto& x = x_opt.value();
    TORCH_CHECK(x.dtype() == torch::kHalf && x.is_contiguous() && x.size(0) == T && x.size(1) == H);
    a.x = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    TORCH_CHECK(is_aligned16(a.x));
  }

  a.conv_weight = reinterpret_cast<const __half*>(conv_weight.data_ptr<at::Half>());
  a.cache = reinterpret_cast<__half*>(cache.data_ptr<at::Half>());
  a.stride_c_blk = cache.stride(0);
  a.stride_c_h = cache.stride(1);
  a.stride_c_n = cache.stride(2);
  a.cache_N = (int)cache.size(2);
  a.cache_H = (int)cache.size(1);
  a.stream_offset = (int)stream_offset;
  a.stream_width = (int)stream_width;
  a.conv_scale = 0.4472135954999579f;  // 5^-0.5

  a.positions = positions.data_ptr<int64_t>();
  a.block_table = block_table.data_ptr<int32_t>();
  a.stride_bt = block_table.stride(0);
  a.max_blocks = (int)block_table.size(1);
  a.seq_idx = seq_idx.data_ptr<int32_t>();
  a.slot_mapping = slot_mapping.data_ptr<int64_t>();
  a.query_start = query_start.data_ptr<int32_t>();

  if (has_addend) {
    a.addend = addend_opt.value().data_ptr<float>();
    TORCH_CHECK(addend_opt.value().dtype() == torch::kFloat && addend_opt.value().is_contiguous());
  }
  if (has_res) {
    a.residual = residual_opt.value().data_ptr<float>();
    TORCH_CHECK(residual_opt.value().dtype() == torch::kFloat && residual_opt.value().is_contiguous());
  }
  a.conv_out = conv_out.data_ptr<float>();

  a.stack = stack.data_ptr<float>();
  a.stride_s = stack.stride(0);
  a.n_stack = n_stack;
  if (has_last) {
    a.last = last_opt.value().data_ptr<float>();
    TORCH_CHECK(last_opt.value().dtype() == torch::kFloat && last_opt.value().is_contiguous());
  }
  a.queries = queries.data_ptr<float>();
  if (apply_norm) {
    auto& nw = norm_w_opt.value();
    TORCH_CHECK(nw.dtype() == torch::kHalf && nw.is_contiguous() && nw.size(0) == Q && nw.size(1) == H);
    a.norm_w = reinterpret_cast<const __half*>(nw.data_ptr<at::Half>());
  }
  a.eps = (float)eps;
  a.norm_eps = (float)norm_eps;
  a.out = apply_norm ? static_cast<void*>(out.data_ptr<at::Half>())
                     : static_cast<void*>(out.data_ptr<float>());

  const cudaStream_t st = at::cuda::getCurrentCUDAStream();

  if (!has_combine && !has_addend) {
    TORCH_CHECK(!has_res && Q == 1,
                "Post-attn path expects has_addend=false, has_res=false, Q=1");
    if (has_last) {
      if (apply_norm) k_ar_conv_mix<false, false, false, true, true, 1><<<T, NV, 0, st>>>(a);
      else k_ar_conv_mix<false, false, false, true, false, 1><<<T, NV, 0, st>>>(a);
    } else {
      if (apply_norm) k_ar_conv_mix<false, false, false, false, true, 1><<<T, NV, 0, st>>>(a);
      else k_ar_conv_mix<false, false, false, false, false, 1><<<T, NV, 0, st>>>(a);
    }
  } else if (!has_combine) {
    TORCH_CHECK(has_addend && !has_last, "Direct post-MoE path expects has_addend=true, has_last=false");
    if (Q == 1) {
      if (has_res) {
        if (apply_norm) k_ar_conv_mix<false, true, true, false, true, 1><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<false, true, true, false, false, 1><<<T, NV, 0, st>>>(a);
      } else {
        if (apply_norm) k_ar_conv_mix<false, true, false, false, true, 1><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<false, true, false, false, false, 1><<<T, NV, 0, st>>>(a);
      }
    } else {
      if (has_res) {
        if (apply_norm) k_ar_conv_mix<false, true, true, false, true, 3><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<false, true, true, false, false, 3><<<T, NV, 0, st>>>(a);
      } else {
        if (apply_norm) k_ar_conv_mix<false, true, false, false, true, 3><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<false, true, false, false, false, 3><<<T, NV, 0, st>>>(a);
      }
    }
  } else {
    TORCH_CHECK(has_addend && !has_last, "Post-MoE path expects has_addend=true, has_last=false");
    if (Q == 1) {
      if (has_res) {
        if (apply_norm) k_ar_conv_mix<true, true, true, false, true, 1><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<true, true, true, false, false, 1><<<T, NV, 0, st>>>(a);
      } else {
        if (apply_norm) k_ar_conv_mix<true, true, false, false, true, 1><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<true, true, false, false, false, 1><<<T, NV, 0, st>>>(a);
      }
    } else {
      if (has_res) {
        if (apply_norm) k_ar_conv_mix<true, true, true, false, true, 3><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<true, true, true, false, false, 3><<<T, NV, 0, st>>>(a);
      } else {
        if (apply_norm) k_ar_conv_mix<true, true, false, false, true, 3><<<T, NV, 0, st>>>(a);
        else k_ar_conv_mix<true, true, false, false, false, 3><<<T, NV, 0, st>>>(a);
      }
    }
  }
  CK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buf_bytes", &buf_bytes);
  m.def("max_tokens", &max_tokens);
  m.def("is_initialized", &is_initialized);
  m.def("info", &info);
  m.def("alloc_buf", &alloc_buf);
  m.def("ipc_handle", &ipc_handle);
  m.def("ipc_open", &ipc_open);
  m.def("init", &init);
  m.def("ar_conv_mix_forward", &ar_conv_mix_forward);
}
