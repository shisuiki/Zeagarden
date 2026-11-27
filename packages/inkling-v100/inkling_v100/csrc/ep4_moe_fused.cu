// SPDX-License-Identifier: Apache-2.0
// ep4_moe_fused.cu - 1-launch persistent Volta (sm_70) EP=4 NVFP4 MoE kernel for M <= 8.
//
// Fuses the entire Inkling-2-Mini NVFP4 MoE visit (router GEMV + top-8 selection & EP=4
// local grouping + W13 with register-fused softcapped SiTU + W2 + slot-order token sum)
// into ONE persistent 80-CTA x 1024-thread launch (1 CTA/SM on V100) separated by two
// self-resetting grid barriers (zero host sync, zero memset, 100% CUDA-graph safe):
//
//   Phase R    Router GEMV: FP16 x [M, 2304] @ FP16 gate_w [128, 2304]^T -> FP32 logits [M, 128].
//              CTA c computes rows c and c + 80 (9 warps/row, 288 threads x 8 cols = 2304).
//   --         grid_barrier(counters + 0, 80)
//   Select     Replicated in every CTA: exact `warp_select_top8_fast<true>` (1 warp/token,
//              bit-exact tie-breaking & `_sum8` log-sigmoid softmax renormalization * route_scale)
//              + `is_padding` handling + EP=4 local filter [expert_lo, expert_hi) -> compact
//              local table in SMEM (D_local in 0..32, n_local_routes in 0..64).
//              If D_local == 0, CTAs 0..71 zero `out` and all 80 CTAs exit immediately.
//   Phase W13  Jobs = D_local x 36 tile pairs (each pair = 2 QPN 32-col tiles = 32 intermediate
//              outputs), distributed round-robin across 80 CTAs; 32 warps = 2 tiles x 16 K-splits
//              (9 K=16 groups/warp). Reads in-place QPN NVFP4 weights & E4M3FN scales via exact
//              branchless `e4m3_to_f16x2_mul128` + `mma_qpn_k8_cb1<1>`, scales by `g13_t[e] * 128`,
//              applies exact FP32 `softcap_situ(g, u, w_of[lr], gate_cap, up_cap)`, and writes
//              FP16 `act [n_local_routes, 1152]` via coalesced uint4 stores.
//   --         grid_barrier(counters + 1, 80)
//   Phase W2   CTAs 0..71 each own one 32-column output tile (CTAs 72..79 return immediately).
//              When `n_local_routes <= 20` (<= 46,080 B <= 46 KB), stages `act` into dynamic SMEM
//              once via `__ldcg`; otherwise reads `act` via `__ldcg`. Processes 4 distinct local
//              experts in parallel (kS2 = 8 K-splits x 9 K=16 groups/warp), scales by
//              `g2_t[e] * 128`, applies per-route FP16 rounding matching `nvfp4_prefill_moe.cu`,
//              then sums per-token routes in exact slot order (0..7) in FP32 -> FP16 `out [M, 2304]`.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

namespace {

constexpr int kMaxDevs = 32;
constexpr int kE = 128;
constexpr int kELocal = 32;
constexpr int kTopK = 8;
constexpr int kH = 2304;
constexpr int kI = 1152;
constexpr int kMaxM = 8;
constexpr int kMaxR = kMaxM * kTopK;       // 64 max routes (also max local routes)
constexpr int kThreads = 1024;
constexpr int kWarps = 32;
constexpr int kGrid = 80;                  // 1 CTA per SM on Tesla V100 (80 SMs)
constexpr int kRW = 9;                     // router warps per expert row (288 * 8 = 2304)

constexpr int kTiles13 = kH / 32;          // 72 QPN 32-col tiles (interleaved gate/up -> 16 inter/tile)
constexpr int kPairs13 = kTiles13 / 2;     // 36 tile pairs per active local expert
constexpr int kCG13 = kH / 16;             // 144 K=16 groups
constexpr int kS13 = 16;                   // 16 K-splits per tile (2 tiles x 16 = 32 warps)
constexpr int kNG13 = kCG13 / kS13;        // 9 K=16 groups per warp
constexpr int kChunk13 = 3;                // 3 divides kNG13 = 9

constexpr int kTiles2 = kH / 32;           // 72 output 32-col tiles of W2
constexpr int kCG2 = kI / 16;              // 72 K=16 groups
constexpr int kS2 = 8;                     // 8 K-splits per expert in W2
constexpr int kNG2 = kCG2 / kS2;           // 9 K=16 groups per warp
constexpr int kPerIter2 = kWarps / kS2;    // 4 distinct local experts processed in parallel
constexpr int kChunk2 = 3;                 // 3 divides kNG2 = 9

constexpr size_t kMaxStage = 46 * 1024;    // 47,104 B dynamic SMEM (static ~41.7 KB + 46 KB <= 96 KB)
constexpr int kMaxStageRoutes = static_cast<int>(kMaxStage / (static_cast<size_t>(kI) * sizeof(__half)));  // 20

static_assert(kH % 32 == 0 && kI % 32 == 0, "H and I must be multiples of 32");
static_assert(kRW * 32 * 8 == kH, "Router thread tiling must cover kH exactly");
static_assert(kCG13 % kS13 == 0 && kNG13 % kChunk13 == 0, "W13 split/chunk divisibility");
static_assert(kCG2 % kS2 == 0 && kNG2 % kChunk2 == 0, "W2 split/chunk divisibility");
static_assert(kMaxStageRoutes == 20, "46 KB dynamic SMEM holds exactly 20 routes of I=1152 FP16");

#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    cudaError_t _err = (call);                                                         \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ", cudaGetErrorString(_err));        \
  } while (0)

// ---- Low-level Volta (sm_70) primitives ----

__device__ __forceinline__ void mma884(
    float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned hmul2u(unsigned a, unsigned b) {
  const __half2 ha = *reinterpret_cast<const __half2*>(&a);
  const __half2 hb = *reinterpret_cast<const __half2*>(&b);
  const __half2 r = __hmul2(ha, hb);
  return *reinterpret_cast<const unsigned*>(&r);
}

__device__ __forceinline__ uint2 ldg_u2_na(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(v.x), "=r"(v.y)
               : "l"(p));
  return v;
}

// Branchless 4-op OCP FP8 E4M3FN byte -> FP16x2 holding exact `e4m3(u) * 2^7`.
// Exact for all 254 finite normals & subnormals (max 448 * 128 = 57344 <= 65504); NaN -> 0.
__device__ __forceinline__ unsigned e4m3_to_f16x2_mul128(unsigned u) {
  const unsigned abs_u = u & 0x7Fu;
  unsigned h = ((u & 0x80u) << 8) | (abs_u << 7);
  h = (abs_u == 0x7Fu) ? 0u : h;
  return hmul2u(h | (h << 16), 0x78007800u);
}

// Exact Inkling2 softcapped SiTU in FP32 (sm_70-safe IEEE tanhf/expf/__fdiv_rn).
__device__ __forceinline__ float softcap_situ(
    float g, float u, float gamma, float gate_cap, float up_cap) {
  const float sig = __fdiv_rn(1.0f, 1.0f + expf(-g));
  if (gate_cap > 0.0f) g = gate_cap * tanhf(__fdiv_rn(g, gate_cap));
  if (up_cap > 0.0f) u = up_cap * tanhf(__fdiv_rn(u, up_cap));
  return ((g * sig) * u) * gamma;
}

// Zero-RAW-stall QPN k=0..7 step matching `nvfp4_prefill_moe.cu` (`mma_qpn_k8_cb1<1>`).
template <int RB>
__device__ __forceinline__ void mma_qpn_k8_cb1(
    unsigned wx,
    unsigned sc_h2,
    const uint4 (&a)[RB],
    float (&acc)[RB][8]) {
  const unsigned b0 = hmul2u(((wx << 12) & 0x80008000u) | ((wx << 9) & 0x0E000E00u), sc_h2);
  const unsigned b1 = hmul2u(((wx <<  8) & 0x80008000u) | ((wx << 5) & 0x0E000E00u), sc_h2);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    mma884(acc[rb], a[rb].x, a[rb].y, b0, b1);
  }
  const unsigned b2 = hmul2u(((wx <<  4) & 0x80008000u) | ((wx << 1) & 0x0E000E00u), sc_h2);
  const unsigned b3 = hmul2u((wx         & 0x80008000u) | ((wx >> 3) & 0x0E000E00u), sc_h2);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    mma884(acc[rb], a[rb].z, a[rb].w, b2, b3);
  }
}

// One warp: 32 output columns (`w_tile`) of local `expert` over K=16 groups `[g0, g0 + ng)`
// against up to 8 activation rows. `row_u4` is this lane's activation row (`a_row = (lane & 3)
// + ((lane & 16) ? 4 : 0)`) or nullptr when `a_row >= nrows`.
template <bool kUseLdcg, int kChunk>
__device__ __forceinline__ void gemv_qpn_rows(
    float (&acc)[1][8],
    const uint4* __restrict__ row_u4,
    const uint2* __restrict__ W_qpn_u2,
    const uint8_t* __restrict__ S_qpn,
    int expert,
    int w_tile,
    int nt,
    int cgroups,
    int g0,
    int ng,
    int lane) {
  const size_t w_base =
      (static_cast<size_t>(expert) * static_cast<size_t>(nt) + static_cast<size_t>(w_tile)) *
          static_cast<size_t>(cgroups) * 32u +
      static_cast<size_t>(lane);
  const uint2* wp = W_qpn_u2 + w_base;
  const uint8_t* sp = S_qpn + w_base;

#pragma unroll
  for (int i = 0; i < 8; ++i) acc[0][i] = 0.0f;

  for (int c0 = 0; c0 < ng; c0 += kChunk) {
    uint2 wv[kChunk];
    unsigned sv[kChunk];
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const size_t off = static_cast<size_t>(g0 + c0 + j) * 32u;
      wv[j] = ldg_u2_na(wp + off);
      sv[j] = static_cast<unsigned>(__ldg(sp + off));
    }
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const int cg = g0 + c0 + j;
      const unsigned sc = e4m3_to_f16x2_mul128(sv[j]);
      uint4 a0 = make_uint4(0u, 0u, 0u, 0u);
      uint4 a1 = make_uint4(0u, 0u, 0u, 0u);
      if (row_u4 != nullptr) {
        const uint4* ap = row_u4 + static_cast<size_t>(cg) * 2u;
        if constexpr (kUseLdcg) {
          a0 = __ldcg(ap);
          a1 = __ldcg(ap + 1);
        } else {
          a0 = ap[0];
          a1 = ap[1];
        }
      }
      {
        uint4 af[1] = {a0};
        mma_qpn_k8_cb1<1>(wv[j].x, sc, af, acc);
      }
      {
        uint4 af[1] = {a1};
        mma_qpn_k8_cb1<1>(wv[j].y, sc, af, acc);
      }
    }
  }
}

// Exact router selection key and log-sigmoid matching `ep4_route.cu` bit-for-bit.
__device__ __forceinline__ uint64_t make_sel_key(float logit, float bias, int e, int E) {
  const float sig = __fdiv_rn(1.0f, 1.0f + expf(-logit));
  const float sel = sig + bias;
  unsigned bits = __float_as_uint(sel);
  bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(E - e);
}

__device__ __forceinline__ float log_sigmoid_exact(float x) {
  return fminf(x, 0.0f) - log1pf(expf(-fabsf(x)));
}

// Bit-exact copy of `ep4_route.cu`'s `warp_select_top8_fast<USE_LDCG>`.
template <bool USE_LDCG>
__device__ __forceinline__ void warp_select_top8_fast(
    const float* __restrict__ row_logits,
    const float* __restrict__ bias,
    int E,
    float route_scale,
    int lo,
    int hi,
    bool is_pad,
    int& out_local,
    float& out_weight) {
  const int lane = threadIdx.x & 31;
  uint64_t my_key[4];
  float my_logit[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int e = lane + j * 32;
    if (e < E) {
      const float lg = USE_LDCG ? __ldcg(row_logits + e) : row_logits[e];
      my_logit[j] = lg;
      my_key[j] = make_sel_key(lg, bias[e], e, E);
    } else {
      my_logit[j] = -INFINITY;
      my_key[j] = 0ull;
    }
  }

  int my_sel_id = 0;
  float my_sel_logit = 0.0f;

#pragma unroll
  for (int k = 0; k < 8; ++k) {
    uint64_t best_k = 0ull;
    float best_l = -INFINITY;
    int best_j = -1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (my_key[j] > best_k) {
        best_k = my_key[j];
        best_l = my_logit[j];
        best_j = j;
      }
    }
    uint64_t g_k = best_k;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      const unsigned lo_w = __shfl_xor_sync(0xFFFFFFFFu, static_cast<unsigned>(g_k), off);
      const unsigned hi_w = __shfl_xor_sync(0xFFFFFFFFu, static_cast<unsigned>(g_k >> 32), off);
      const uint64_t ok = (static_cast<uint64_t>(hi_w) << 32) | lo_w;
      if (ok > g_k) g_k = ok;
    }
    const bool is_winner = (best_j >= 0 && best_k == g_k);
    const int win_lane = __ffs(__ballot_sync(0xFFFFFFFFu, is_winner)) - 1;
    const float g_l = __shfl_sync(0xFFFFFFFFu, best_l, win_lane);
    if (is_winner) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j == best_j) my_key[j] = 0ull;
      }
    }
    if (lane == k) {
      my_sel_id = E - static_cast<int>(g_k & 0xFFFFull);
      my_sel_logit = g_l;
    }
  }

  // Parallel softmax renormalization across lanes 0..7 with bit-exact `_sum8` tree.
  const float lp = (lane < 8) ? log_sigmoid_exact(my_sel_logit) : -INFINITY;
  float max_lp = lp;
  max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 4));
  max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 2));
  max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 1));

  const float sh = (lane < 8) ? expf(lp - max_lp) : 0.0f;
  // Exact `_sum8`: ((sh[0]+sh[2])+(sh[1]+sh[3])) + ((sh[4]+sh[6])+(sh[5]+sh[7]))
  float tot = sh + __shfl_xor_sync(0xFFFFFFFFu, sh, 2);
  tot = tot + __shfl_xor_sync(0xFFFFFFFFu, tot, 1);
  tot = tot + __shfl_xor_sync(0xFFFFFFFFu, tot, 4);
  const float lse = max_lp + logf(tot);

  out_weight = (lane < 8) ? (expf(lp - lse) * route_scale) : 0.0f;
  out_local = (lane < 8 && !is_pad && my_sel_id >= lo && my_sel_id < hi)
                  ? (my_sel_id - lo)
                  : -1;
}

// Self-resetting grid-wide barrier over G co-resident CTAs (CUDA-graph safe, zero host reset).
__device__ __forceinline__ void grid_barrier(unsigned* ctr, int G) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned old = atomicInc(ctr, 2u * static_cast<unsigned>(G) - 1u);
    const bool ph1 = old >= static_cast<unsigned>(G);
    volatile unsigned* vc = ctr;
    while (true) {
      const unsigned v = *vc;
      if (ph1 ? (v < static_cast<unsigned>(G)) : (v >= static_cast<unsigned>(G))) break;
    }
    __threadfence();
  }
  __syncthreads();
}

// Compact EP=4 local routing table in shared memory (~1.7 KB).
struct Ep4Tables {
  uint32_t exp_tok_mask[kELocal];       // 128 B: bit m set iff token m picked local expert e
  float    exp_tok_w[kELocal * kMaxM];  // 1024 B: routing weight for (local expert e, token m)
  float    w_of[kMaxR];                 // 256 B: routing weight for local route lr
  int8_t   sel_el[kMaxR];               // 64 B: [m][slot] local expert id or -1
  int8_t   tok_slot_lr[kMaxR];          // 64 B: [m][slot] local route index lr or -1
  uint8_t  dmap[kELocal];               // 32 B: local expert -> distinct active index d
  uint8_t  dexp[kELocal];               // 32 B: distinct active index d -> local expert (ascending)
  uint8_t  nrows[kELocal];              // 32 B: number of tokens (1..8) routed to active expert d
  uint8_t  d_route_off[kELocal];        // 32 B: starting local route index of active expert d
  uint8_t  tok_of[kMaxR];               // 64 B: token index (0..M-1) of local route lr
  int      D_local;                     // 4 B: number of distinct active local experts (0..32)
  int      n_local_routes;              // 4 B: total active local routes (0..64)
};

union Out13Smem {
  __half h[2][8][16];
  uint4  u4[2][8][2];
};

// Non-overlapping phase scratch (8,192 B total).
union PhaseScratch {
  float     red[2][kRW][kMaxM];         // 576 B (Phase R only)
  Out13Smem out13;                      // 512 B (Phase W13 epilogue only)
  float     routed[kMaxR * 32];         // 8,192 B (Phase W2 only)
};

// Static SMEM breakdown:
//   part:         32 * 256 * 4 = 32,768 B
//   sh_scratch:    8,192 B
//   tb:            1,736 B
//   Total static: 42,696 B (~41.7 KB) + 47,104 B (46 KB) dynamic = 89,800 B (~87.7 KB <= 96 KB)
__global__ void __launch_bounds__(kThreads, 1) ep4_moe_fused_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gate_w,
    const float* __restrict__ gate_b,
    const bool* __restrict__ is_padding,
    const uint2* __restrict__ W13_qpn_u2,
    const uint8_t* __restrict__ S13_qpn,
    const float* __restrict__ g13_t,
    const uint2* __restrict__ W2_qpn_u2,
    const uint8_t* __restrict__ S2_qpn,
    const float* __restrict__ g2_t,
    float* __restrict__ logits,
    __half* __restrict__ act,
    unsigned* __restrict__ counters,
    __half* __restrict__ out,
    int M,
    float route_scale,
    float gate_cap,
    float up_cap,
    int expert_lo,
    int expert_hi) {
  extern __shared__ __align__(16) unsigned char stage_raw[];
  __shared__ __align__(16) float part[kWarps * 256];
  __shared__ __align__(16) PhaseScratch sh_scratch;
  __shared__ __align__(16) Ep4Tables tb;

  const int c = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // ---------------- Phase R: Router GEMV (FP16 x @ FP16 gate_w^T -> FP32 logits) ----------------
  {
    const int rr = warp / kRW;
    const int e = c + rr * kGrid;
    if (rr < 2 && e < kE) {
      const int lw = warp - rr * kRW;
      const int t = lw * 32 + lane;  // 0..287 (288 * 8 = 2304 == kH)
      const uint4 wv = __ldg(reinterpret_cast<const uint4*>(gate_w + static_cast<size_t>(e) * kH) + t);
      const __half2* wh = reinterpret_cast<const __half2*>(&wv);
      const float2 fw0 = __half22float2(wh[0]);
      const float2 fw1 = __half22float2(wh[1]);
      const float2 fw2 = __half22float2(wh[2]);
      const float2 fw3 = __half22float2(wh[3]);
      const uint4* x_u4_base = reinterpret_cast<const uint4*>(x);

#pragma unroll
      for (int m = 0; m < kMaxM; ++m) {
        if (m >= M) break;
        const uint4 xv = __ldg(x_u4_base + static_cast<size_t>(m) * (kH / 8) + t);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const float2 fx0 = __half22float2(xh[0]);
        const float2 fx1 = __half22float2(xh[1]);
        const float2 fx2 = __half22float2(xh[2]);
        const float2 fx3 = __half22float2(xh[3]);
        float acc = fw0.x * fx0.x;
        acc = __fmaf_rn(fw0.y, fx0.y, acc);
        acc = __fmaf_rn(fw1.x, fx1.x, acc);
        acc = __fmaf_rn(fw1.y, fx1.y, acc);
        acc = __fmaf_rn(fw2.x, fx2.x, acc);
        acc = __fmaf_rn(fw2.y, fx2.y, acc);
        acc = __fmaf_rn(fw3.x, fx3.x, acc);
        acc = __fmaf_rn(fw3.y, fx3.y, acc);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) sh_scratch.red[rr][lw][m] = acc;
      }
    }
    __syncthreads();

    if (tid < 2 * kMaxM) {
      const int r2 = tid / kMaxM;
      const int m = tid - r2 * kMaxM;
      const int e2 = c + r2 * kGrid;
      if (e2 < kE && m < M) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < kRW; ++w) v += sh_scratch.red[r2][w][m];
        logits[static_cast<size_t>(m) * kE + e2] = v;
      }
    }
  }
  grid_barrier(counters + 0, kGrid);

  // ---------------- Select + Local EP4 Table (replicated in every CTA) ----------------
  {
    if (tid < kELocal) tb.exp_tok_mask[tid] = 0u;
    if (tid < kMaxR) tb.sel_el[tid] = -1;
    __syncthreads();

    if (warp < M) {
      const bool pad = (is_padding != nullptr) ? is_padding[warp] : false;
      int el = -1;
      float wt = 0.0f;
      warp_select_top8_fast<true>(
          logits + static_cast<size_t>(warp) * kE,
          gate_b,
          kE,
          route_scale,
          expert_lo,
          expert_hi,
          pad,
          el,
          wt);
      if (lane < kTopK) {
        tb.sel_el[warp * kTopK + lane] = static_cast<int8_t>(el);
        if (el >= 0) {
          atomicOr(&tb.exp_tok_mask[el], 1u << warp);
          tb.exp_tok_w[el * kMaxM + warp] = wt;
        }
      }
    }
    __syncthreads();

    if (warp == 0) {
      // Lane = local expert index in 0..31.
      const uint32_t mk = tb.exp_tok_mask[lane];
      const int cnt = __popc(mk);
      int off_end = cnt;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, off_end, d);
        if (lane >= d) off_end += n;
      }
      const int off_start = off_end - cnt;
      const uint32_t active_mask = __ballot_sync(0xffffffffu, cnt > 0);
      const int D_local_w = __popc(active_mask);
      const int n_routes_w = __shfl_sync(0xffffffffu, off_end, 31);
      if (lane == 0) {
        tb.D_local = D_local_w;
        tb.n_local_routes = n_routes_w;
      }
      if (cnt > 0) {
        const int d = __popc(active_mask & ((1u << lane) - 1u));
        tb.dmap[lane] = static_cast<uint8_t>(d);
        tb.dexp[d] = static_cast<uint8_t>(lane);
        tb.nrows[d] = static_cast<uint8_t>(cnt);
        tb.d_route_off[d] = static_cast<uint8_t>(off_start);
        int dst = off_start;
        uint32_t m_bits = mk;
        while (m_bits) {
          const int m = __ffs(m_bits) - 1;
          tb.tok_of[dst] = static_cast<uint8_t>(m);
          tb.w_of[dst] = tb.exp_tok_w[lane * kMaxM + m];
          dst++;
          m_bits &= (m_bits - 1u);
        }
      }
    }
    __syncthreads();

    if (tid < M * kTopK) {
      const int m = tid >> 3;
      const int el = static_cast<int>(tb.sel_el[tid]);
      int lr = -1;
      if (el >= 0) {
        const int d = static_cast<int>(tb.dmap[el]);
        const int pos = __popc(tb.exp_tok_mask[el] & ((1u << m) - 1u));
        lr = static_cast<int>(tb.d_route_off[d]) + pos;
      }
      tb.tok_slot_lr[tid] = static_cast<int8_t>(lr);
    }
    __syncthreads();
  }

  const int D_local = tb.D_local;
  const int n_local_routes = tb.n_local_routes;

  // Fast exit when no token in this batch routes to any local expert (or all tokens padded).
  // All 80 CTAs see the exact same deterministic D_local == 0, so skipping barrier 1 is safe.
  if (D_local == 0) {
    if (c < kTiles2 && tid < M * 32) {
      const int m = tid >> 5;
      const int col = tid & 31;
      out[static_cast<size_t>(m) * kH + static_cast<size_t>(c) * 32 + col] = __float2half_rn(0.0f);
    }
    return;
  }

  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  uint4* act_u4 = reinterpret_cast<uint4*>(act);

  // ---------------- Phase W13 + Softcapped Interleaved SiTU ----------------
  {
    const uint4* x_u4_base = reinterpret_cast<const uint4*>(x);
    const int tile_slot = warp >> 4;  // 0 or 1 (2 tiles per CTA)
    const int split = warp & 15;      // 0..15 (16 K-splits per tile -> 9 K=16 groups/warp)
    const int njobs = D_local * kPairs13;

    for (int job = c; job < njobs; job += kGrid) {
      const int d = job / kPairs13;
      const int tp = job - d * kPairs13;
      const int expert = static_cast<int>(tb.dexp[d]);
      const int nrows = static_cast<int>(tb.nrows[d]);
      const int route_off = static_cast<int>(tb.d_route_off[d]);
      const int w_tile = tp * 2 + tile_slot;

      const uint4* rp = nullptr;
      if (a_row < nrows) {
        const int tok = static_cast<int>(tb.tok_of[route_off + a_row]);
        rp = x_u4_base + static_cast<size_t>(tok) * (kH / 8);
      }

      float acc[1][8];
      gemv_qpn_rows<false, kChunk13>(
          acc, rp, W13_qpn_u2, S13_qpn, expert, w_tile, kTiles13, kCG13,
          split * kNG13, kNG13, lane);

#pragma unroll
      for (int e = 0; e < 8; ++e) {
        part[warp * 256 + e * 32 + lane] = acc[0][e];
      }
      __syncthreads();

      // Warps 0 and 1 reduce the 16 K-splits for tile_slot 0 and tile_slot 1 respectively.
      if (warp < 2) {
        const int ts = warp;
        const int wt_out = tp * 2 + ts;
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float sum = 0.0f;
#pragma unroll
          for (int s = 0; s < kS13; ++s) {
            sum += part[(ts * kS13 + s) * 256 + e * 32 + lane];
          }
          v[e] = sum;
        }
        const float gs13 = __ldg(g13_t + expert) * 128.0f;
        const int q = (lane >> 2) & 3;
        const int r_lo = (lane & 1) + ((lane & 16) ? 4 : 0);
        const int r_hi = r_lo + 2;
        const int c_lo = q * 8 + (lane & 2);
        const int c_hi = c_lo + 4;
        const int ic_lo = c_lo >> 1;
        const int ic_hi = c_hi >> 1;

        if (r_lo < nrows) {
          const float g_lo = tb.w_of[route_off + r_lo];
          sh_scratch.out13.h[ts][r_lo][ic_lo] =
              __float2half_rn(softcap_situ(v[0] * gs13, v[1] * gs13, g_lo, gate_cap, up_cap));
          sh_scratch.out13.h[ts][r_lo][ic_hi] =
              __float2half_rn(softcap_situ(v[4] * gs13, v[5] * gs13, g_lo, gate_cap, up_cap));
        }
        if (r_hi < nrows) {
          const float g_hi = tb.w_of[route_off + r_hi];
          sh_scratch.out13.h[ts][r_hi][ic_lo] =
              __float2half_rn(softcap_situ(v[2] * gs13, v[3] * gs13, g_hi, gate_cap, up_cap));
          sh_scratch.out13.h[ts][r_hi][ic_hi] =
              __float2half_rn(softcap_situ(v[6] * gs13, v[7] * gs13, g_hi, gate_cap, up_cap));
        }
        __syncwarp();

        if (lane < nrows * 2) {
          const int r = lane >> 1;
          const int c2 = lane & 1;
          const int lr = route_off + r;
          act_u4[static_cast<size_t>(lr) * (kI / 8) + wt_out * 2 + c2] =
              sh_scratch.out13.u4[ts][r][c2];
        }
      }
      __syncthreads();
    }
  }
  grid_barrier(counters + 1, kGrid);

  // ---------------- Phase W2 + Slot-Order Token Sum -> out [M, 2304] FP16 ----------------
  {
    const int tile = c;
    if (tile >= kTiles2) return;

    const bool staged = (n_local_routes <= kMaxStageRoutes);
    uint4* stage_u4 = reinterpret_cast<uint4*>(stage_raw);
    if (staged) {
      const int total_u4 = n_local_routes * (kI / 8);
      for (int i = tid; i < total_u4; i += kThreads) {
        stage_u4[i] = __ldcg(act_u4 + i);
      }
      __syncthreads();
    }

    const int di = warp / kS2;       // 0..3 (4 distinct local experts in parallel)
    const int split = warp - di * kS2;  // 0..7 (8 K-splits x 9 K=16 groups/warp = 72)

    for (int d0 = 0; d0 < D_local; d0 += kPerIter2) {
      const int d = d0 + di;
      if (d < D_local) {
        const int expert = static_cast<int>(tb.dexp[d]);
        const int nrows = static_cast<int>(tb.nrows[d]);
        const int route_off = static_cast<int>(tb.d_route_off[d]);
        float acc[1][8];
        if (staged) {
          const uint4* rp = (a_row < nrows)
                                ? (stage_u4 + static_cast<size_t>(route_off + a_row) * (kI / 8))
                                : nullptr;
          gemv_qpn_rows<false, kChunk2>(
              acc, rp, W2_qpn_u2, S2_qpn, expert, tile, kTiles2, kCG2,
              split * kNG2, kNG2, lane);
        } else {
          const uint4* rp = (a_row < nrows)
                                ? (act_u4 + static_cast<size_t>(route_off + a_row) * (kI / 8))
                                : nullptr;
          gemv_qpn_rows<true, kChunk2>(
              acc, rp, W2_qpn_u2, S2_qpn, expert, tile, kTiles2, kCG2,
              split * kNG2, kNG2, lane);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          part[warp * 256 + e * 32 + lane] = acc[0][e];
        }
      }
      __syncthreads();

      // Warps 0..3 each reduce the 8 K-splits of one active expert and store FP16-rounded routes.
      if (warp < kPerIter2) {
        const int dd = d0 + warp;
        if (dd < D_local) {
          const int expert_dd = static_cast<int>(tb.dexp[dd]);
          const int nrows_dd = static_cast<int>(tb.nrows[dd]);
          const int roff_dd = static_cast<int>(tb.d_route_off[dd]);
          const float gs2 = __ldg(g2_t + expert_dd) * 128.0f;

          float v[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < kS2; ++p) {
              sum += part[(warp * kS2 + p) * 256 + e * 32 + lane];
            }
            v[e] = sum;
          }

          const int q = (lane >> 2) & 3;
          const int r_lo = (lane & 1) + ((lane & 16) ? 4 : 0);
          const int r_hi = r_lo + 2;
          const int c_lo = q * 8 + (lane & 2);
          const int c_hi = c_lo + 4;

          if (r_lo < nrows_dd) {
            const int lr = roff_dd + r_lo;
            sh_scratch.routed[lr * 32 + c_lo + 0] = __half2float(__float2half_rn(v[0] * gs2));
            sh_scratch.routed[lr * 32 + c_lo + 1] = __half2float(__float2half_rn(v[1] * gs2));
            sh_scratch.routed[lr * 32 + c_hi + 0] = __half2float(__float2half_rn(v[4] * gs2));
            sh_scratch.routed[lr * 32 + c_hi + 1] = __half2float(__float2half_rn(v[5] * gs2));
          }
          if (r_hi < nrows_dd) {
            const int lr = roff_dd + r_hi;
            sh_scratch.routed[lr * 32 + c_lo + 0] = __half2float(__float2half_rn(v[2] * gs2));
            sh_scratch.routed[lr * 32 + c_lo + 1] = __half2float(__float2half_rn(v[3] * gs2));
            sh_scratch.routed[lr * 32 + c_hi + 0] = __half2float(__float2half_rn(v[6] * gs2));
            sh_scratch.routed[lr * 32 + c_hi + 1] = __half2float(__float2half_rn(v[7] * gs2));
          }
        }
      }
      __syncthreads();
    }

    // Direct per-token FP32 sum in exact slot order (0..7) -> FP16 out [M, 2304].
    if (tid < M * 32) {
      const int m = tid >> 5;
      const int col = tid & 31;
      float sum = 0.0f;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        const int lr = static_cast<int>(tb.tok_slot_lr[m * kTopK + k]);
        if (lr >= 0) {
          sum += sh_scratch.routed[lr * 32 + col];
        }
      }
      out[static_cast<size_t>(m) * kH + static_cast<size_t>(tile) * 32 + col] = __float2half_rn(sum);
    }
  }
}

// Per-device one-time kernel attribute & occupancy validation (safe outside CUDA graph capture).
std::mutex g_init_mu;
bool g_dev_ready[kMaxDevs] = {};

void ensure_device_ready(int dev) {
  TORCH_CHECK(dev >= 0 && dev < kMaxDevs, "ep4_moe_fused: invalid CUDA device ", dev);
  if (g_dev_ready[dev]) return;
  std::lock_guard<std::mutex> lk(g_init_mu);
  if (g_dev_ready[dev]) return;

  const at::cuda::CUDAGuard guard(dev);
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  TORCH_CHECK(
      prop.multiProcessorCount >= kGrid,
      "ep4_moe_fused requires >= ", kGrid, " SMs for persistent grid barriers, got ",
      prop.multiProcessorCount);

  CUDA_CHECK(cudaFuncSetAttribute(
      ep4_moe_fused_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kMaxStage)));

  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, ep4_moe_fused_kernel, kThreads, kMaxStage));
  TORCH_CHECK(
      occ >= 1,
      "ep4_moe_fused_kernel occupancy is ", occ,
      " (< 1 CTA/SM with ", kThreads, " threads and ", kMaxStage,
      " B dynamic SMEM); persistent grid barrier requires >= 1 CTA/SM.");
  TORCH_CHECK(
      occ * prop.multiProcessorCount >= kGrid,
      "ep4_moe_fused_kernel max active blocks ", occ * prop.multiProcessorCount,
      " < required grid ", kGrid);

  g_dev_ready[dev] = true;
}

}  // namespace

void init_ep4_moe_fused(int64_t device_index) {
  ensure_device_ready(static_cast<int>(device_index));
}

std::vector<int64_t> ep4_moe_fused_kernel_info(int64_t device_index) {
  const int dev = static_cast<int>(device_index);
  ensure_device_ready(dev);
  const at::cuda::CUDAGuard guard(dev);
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, ep4_moe_fused_kernel));
  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, ep4_moe_fused_kernel, kThreads, kMaxStage));
  return {
      static_cast<int64_t>(attr.numRegs),
      static_cast<int64_t>(attr.sharedSizeBytes),
      static_cast<int64_t>(attr.localSizeBytes),
      static_cast<int64_t>(attr.maxDynamicSharedSizeBytes),
      static_cast<int64_t>(occ),
      static_cast<int64_t>(kMaxStage),
      static_cast<int64_t>(kMaxStageRoutes),
  };
}

void ep4_nvfp4_moe_fused_forward(
    torch::Tensor x,
    torch::Tensor gate_w,
    torch::Tensor gate_b,
    c10::optional<torch::Tensor> is_padding,
    torch::Tensor w13_qpn,
    torch::Tensor s13_qpn,
    torch::Tensor g13_t,
    torch::Tensor w2_qpn,
    torch::Tensor s2_qpn,
    torch::Tensor g2_t,
    torch::Tensor logits,
    torch::Tensor act,
    torch::Tensor counters,
    torch::Tensor out,
    double route_scale,
    double gate_cap,
    double up_cap,
    int64_t expert_lo,
    int64_t expert_hi) {
  const int64_t M = x.size(0);
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
          x.dim() == 2 && M >= 1 && M <= kMaxM && x.size(1) == kH,
      "ep4_moe_fused: x must be CUDA contiguous FP16 [1..8, 2304]");
  const auto dev = x.device();
  TORCH_CHECK(
      gate_w.is_cuda() && gate_w.device() == dev && gate_w.scalar_type() == torch::kFloat16 &&
          gate_w.is_contiguous() && gate_w.dim() == 2 &&
          gate_w.size(0) == kE && gate_w.size(1) == kH,
      "ep4_moe_fused: gate_w must be CUDA contiguous FP16 [128, 2304]");
  TORCH_CHECK(
      gate_b.is_cuda() && gate_b.device() == dev && gate_b.scalar_type() == torch::kFloat32 &&
          gate_b.is_contiguous() && gate_b.numel() == kE,
      "ep4_moe_fused: gate_b must be CUDA contiguous FP32 [128]");

  const bool* pad_ptr = nullptr;
  if (is_padding.has_value() && is_padding.value().defined() && is_padding.value().numel() > 0) {
    const auto& pad = is_padding.value();
    TORCH_CHECK(
        pad.is_cuda() && pad.device() == dev && pad.scalar_type() == torch::kBool &&
            pad.is_contiguous() && pad.numel() >= M,
        "ep4_moe_fused: is_padding must be CUDA contiguous bool [>=M]");
    pad_ptr = pad.data_ptr<bool>();
  }

  TORCH_CHECK(
      w13_qpn.is_cuda() && w13_qpn.device() == dev && w13_qpn.scalar_type() == torch::kUInt8 &&
          w13_qpn.is_contiguous() && w13_qpn.dim() == 3 &&
          w13_qpn.size(0) == kELocal && w13_qpn.size(1) == kH && w13_qpn.size(2) == kH / 2,
      "ep4_moe_fused: w13_qpn must be CUDA contiguous uint8 [32, 2304, 1152]");
  TORCH_CHECK(
      s13_qpn.is_cuda() && s13_qpn.device() == dev && s13_qpn.scalar_type() == torch::kUInt8 &&
          s13_qpn.is_contiguous() && s13_qpn.dim() == 3 &&
          s13_qpn.size(0) == kELocal && s13_qpn.size(1) == kH && s13_qpn.size(2) == kCG13,
      "ep4_moe_fused: s13_qpn must be CUDA contiguous uint8 [32, 2304, 144]");
  TORCH_CHECK(
      g13_t.is_cuda() && g13_t.device() == dev && g13_t.scalar_type() == torch::kFloat32 &&
          g13_t.is_contiguous() && g13_t.numel() == kELocal,
      "ep4_moe_fused: g13_t must be CUDA contiguous FP32 [32]");

  TORCH_CHECK(
      w2_qpn.is_cuda() && w2_qpn.device() == dev && w2_qpn.scalar_type() == torch::kUInt8 &&
          w2_qpn.is_contiguous() && w2_qpn.dim() == 3 &&
          w2_qpn.size(0) == kELocal && w2_qpn.size(1) == kH && w2_qpn.size(2) == kI / 2,
      "ep4_moe_fused: w2_qpn must be CUDA contiguous uint8 [32, 2304, 576]");
  TORCH_CHECK(
      s2_qpn.is_cuda() && s2_qpn.device() == dev && s2_qpn.scalar_type() == torch::kUInt8 &&
          s2_qpn.is_contiguous() && s2_qpn.dim() == 3 &&
          s2_qpn.size(0) == kELocal && s2_qpn.size(1) == kH && s2_qpn.size(2) == kCG2,
      "ep4_moe_fused: s2_qpn must be CUDA contiguous uint8 [32, 2304, 72]");
  TORCH_CHECK(
      g2_t.is_cuda() && g2_t.device() == dev && g2_t.scalar_type() == torch::kFloat32 &&
          g2_t.is_contiguous() && g2_t.numel() == kELocal,
      "ep4_moe_fused: g2_t must be CUDA contiguous FP32 [32]");

  TORCH_CHECK(
      logits.is_cuda() && logits.device() == dev && logits.scalar_type() == torch::kFloat32 &&
          logits.is_contiguous() && logits.numel() >= kMaxM * kE,
      "ep4_moe_fused: logits scratch too small");
  TORCH_CHECK(
      act.is_cuda() && act.device() == dev && act.scalar_type() == torch::kFloat16 &&
          act.is_contiguous() && act.numel() >= kMaxR * kI,
      "ep4_moe_fused: act scratch too small");
  TORCH_CHECK(
      counters.is_cuda() && counters.device() == dev && counters.scalar_type() == torch::kInt32 &&
          counters.is_contiguous() && counters.numel() >= 2,
      "ep4_moe_fused: counters scratch must be int32 [>=2]");
  TORCH_CHECK(
      out.is_cuda() && out.device() == dev && out.scalar_type() == torch::kFloat16 &&
          out.is_contiguous() && out.dim() == 2 && out.size(0) == M && out.size(1) == kH,
      "ep4_moe_fused: out must be CUDA contiguous FP16 [M, 2304]");
  TORCH_CHECK(
      expert_lo >= 0 && expert_hi <= kE && (expert_hi - expert_lo) == kELocal,
      "ep4_moe_fused: expected expert_hi - expert_lo == 32 within [0, 128], got [",
      expert_lo, ", ", expert_hi, ")");

  ensure_device_ready(dev.index());

  const at::cuda::CUDAGuard guard(dev);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ep4_moe_fused_kernel<<<kGrid, kThreads, kMaxStage, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(gate_w.data_ptr<at::Half>()),
      gate_b.data_ptr<float>(),
      pad_ptr,
      reinterpret_cast<const uint2*>(w13_qpn.data_ptr<uint8_t>()),
      s13_qpn.data_ptr<uint8_t>(),
      g13_t.data_ptr<float>(),
      reinterpret_cast<const uint2*>(w2_qpn.data_ptr<uint8_t>()),
      s2_qpn.data_ptr<uint8_t>(),
      g2_t.data_ptr<float>(),
      logits.data_ptr<float>(),
      reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
      reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      static_cast<int>(M),
      static_cast<float>(route_scale),
      static_cast<float>(gate_cap),
      static_cast<float>(up_cap),
      static_cast<int>(expert_lo),
      static_cast<int>(expert_hi));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_ep4_moe_fused", &init_ep4_moe_fused,
        "Configure dynamic SMEM and verify >=1 CTA/SM occupancy for ep4_moe_fused_kernel");
  m.def("ep4_moe_fused_kernel_info", &ep4_moe_fused_kernel_info,
        "Return [numRegs, staticSmem, localBytes, maxDynSmem, occupancy, reqDynSmem, maxStageRoutes]");
  m.def("ep4_nvfp4_moe_fused_forward", &ep4_nvfp4_moe_fused_forward,
        "1-launch persistent Volta sm_70 EP=4 NVFP4 MoE (router + ep4_select + W13+SiTU + W2+sum) for M<=8");
}
