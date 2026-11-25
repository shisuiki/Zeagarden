// SPDX-License-Identifier: Apache-2.0
// ep4_route.cu - Fused Volta (sm_70) Inkling2 MoE gate GEMV + top-k router + zero-sync EP=4 grouping.
//
// Paths:
//   1. `ep4_gemv_select_and_route` (small T <= 32, E == 128, topk == 8, E_local <= 32, H % 8 == 0):
//      Single-kernel fused FP16xFP16->FP32 gate GEMV across 64..128 CTAs (exact uint4 loads, L2-cached x)
//      + atomic last-CTA arrival -> inline zero-smem 1-warp ballot/popc router (T=1) or
//      6/8-warp bitmask router (2 <= T <= 32). Zero extra launch, zero memset, 100% CUDA-graph safe.
//   2. `ep4_select_and_route` (precomputed FP32 logits):
//      - T == 1, E <= 128, topk == 8, E_local <= 32: 1-warp (32-thread) zero-smem ballot/popc kernel.
//      - 2 <= T <= 32, E <= 128, topk == 8, E_local <= 32: 1-CTA (128/256/512-thread) bitmask kernel
//        (1 single routing wave for all T <= 16!).
//      - General T <= 16: 1-CTA (256-thread) general kernel.
//      - Large T (prefill up to 8192+): 3 tiny zero-sync kernels.
//
// Exact Inkling2 math:
//   - Selection key: (order_preserving_u32(sigmoid(logit) + bias) << 16) | (E - e)
//     so ties deterministically prefer the lower expert index.
//   - Routing weights: exact `_sum8` log-sigmoid softmax renormalization * route_scale.
//   - Outputs (perm, gids, goff) match `skinny_moe_qpn_sm70` and `mlp_mxfp8_moe.cu` identically.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int MAX_E_LOCAL = 64;
constexpr int MAX_TOPK = 16;
constexpr int FUSED_MAX_T = 16;
constexpr int FUSED_MAX_S = FUSED_MAX_T * MAX_TOPK;  // 256
constexpr int GEMV_MAX_T = 32;
constexpr int GEMV_E = 128;

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

// Fast warp-level top-8 over E <= 128 experts with:
//   - Only key (64-bit) shuffled during 5-step max reduction; winner logit fetched in 1 ballot+shfl!
//   - Parallel lanes 0..7 softmax renormalization with bit-exact `_sum8` tree.
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
  for (int j = 0; j < 4; j++) {
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
  for (int k = 0; k < 8; k++) {
    uint64_t best_k = 0ull;
    float best_l = -INFINITY;
    int best_j = -1;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (my_key[j] > best_k) {
        best_k = my_key[j];
        best_l = my_logit[j];
        best_j = j;
      }
    }
    uint64_t g_k = best_k;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      const unsigned lo_w = __shfl_xor_sync(0xFFFFFFFFu, (unsigned)g_k, off);
      const unsigned hi_w = __shfl_xor_sync(0xFFFFFFFFu, (unsigned)(g_k >> 32), off);
      const uint64_t ok = (static_cast<uint64_t>(hi_w) << 32) | lo_w;
      if (ok > g_k) g_k = ok;
    }
    const bool is_winner = (best_j >= 0 && best_k == g_k);
    const int win_lane = __ffs(__ballot_sync(0xFFFFFFFFu, is_winner)) - 1;
    const float g_l = __shfl_sync(0xFFFFFFFFu, best_l, win_lane);
    if (is_winner) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
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

// Single-warp (32-thread) T==1, topk==8, E<=128, E_local<=32 router + EP4 grouping.
// Zero shared memory, zero __syncthreads(), zero atomics.
template <bool USE_LDCG>
__device__ __forceinline__ void route_t1_1warp_ep4(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int E,
    float route_scale,
    int lo,
    int hi) {
  const int lane = threadIdx.x & 31;
  const bool pad = (is_padding != nullptr) ? is_padding[0] : false;
  int el = -1;
  float wt = 0.0f;
  warp_select_top8_fast<USE_LDCG>(logits, bias, E, route_scale, lo, hi, pad, el, wt);

  if (lane < 8) {
    ids[lane] = el;
    weights[lane] = wt;
    src2dst[lane] = (el >= 0) ? lane : -1;
  }

  uint32_t bit = (lane < 8 && el >= 0) ? (1u << el) : 0u;
  bit |= __shfl_xor_sync(0xFFFFFFFFu, bit, 4);
  bit |= __shfl_xor_sync(0xFFFFFFFFu, bit, 2);
  bit |= __shfl_xor_sync(0xFFFFFFFFu, bit, 1);
  const uint32_t active_mask = __shfl_sync(0xFFFFFFFFu, bit, 0);
  const int n_active = __popc(active_mask);

  if (lane < 8 && el >= 0) {
    const int rank = __popc(active_mask & ((1u << el) - 1u));
    gids[rank] = el;
    goff[rank] = rank;
    perm[rank] = lane;
  }
  if (lane >= n_active && lane < 8) {
    perm[lane] = 0;
    gids[lane] = 0;
  }
  if (lane >= n_active && lane <= 8) {
    goff[lane] = n_active;
  }
}

// 1-CTA bitmask router for 2 <= T <= 32, E <= 128, topk == 8, E_local <= 32.
// NUM_WARPS in {4, 6, 8, 16} -> 1 single wave for T <= NUM_WARPS!
template <bool USE_LDCG, int NUM_WARPS>
__device__ __forceinline__ void route_small_t_bitmask_ep4(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int T,
    int E,
    float route_scale,
    int lo,
    int hi,
    uint32_t sh_mask[32][8],
    int* sh_meta) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int S = T * 8;
  const int n_words = (S + 31) >> 5;  // 1..8

  for (int i = tid; i < 256; i += NUM_WARPS * 32) {
    sh_mask[i >> 3][i & 7] = 0u;
  }
  __syncthreads();

  for (int t = warp; t < T; t += NUM_WARPS) {
    const bool pad = (is_padding != nullptr) ? is_padding[t] : false;
    int el = -1;
    float wt = 0.0f;
    warp_select_top8_fast<USE_LDCG>(
        logits + (size_t)t * E, bias, E, route_scale, lo, hi, pad, el, wt);
    if (lane < 8) {
      const int s = t * 8 + lane;
      ids[s] = el;
      weights[s] = wt;
      src2dst[s] = (el >= 0) ? s : -1;
      if (el >= 0) {
        atomicOr(&sh_mask[el][s >> 5], 1u << (s & 31));
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    const int E_local = hi - lo;
    uint32_t mw[8];
    int c = 0;
    if (lane < E_local) {
#pragma unroll
      for (int w = 0; w < 8; w++) {
        uint32_t m = (w < n_words) ? sh_mask[lane][w] : 0u;
        mw[w] = m;
        c += __popc(m);
      }
    }
    int off_end = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int n = __shfl_up_sync(0xFFFFFFFFu, off_end, d);
      if (lane >= d) off_end += n;
    }
    const int off_start = off_end - c;
    const uint32_t nz_mask = __ballot_sync(0xFFFFFFFFu, c > 0);
    const int n_active = __popc(nz_mask);
    const int n_local = __shfl_sync(0xFFFFFFFFu, off_end, 31);
    if (lane == 0) {
      sh_meta[0] = n_active;
      sh_meta[1] = n_local;
    }
    if (c > 0) {
      const int slot = __popc(nz_mask & ((1u << lane) - 1u));
      gids[slot] = lane;
      goff[slot] = off_start;
      int dst = off_start;
#pragma unroll
      for (int w = 0; w < 8; w++) {
        if (w >= n_words) break;
        uint32_t m = mw[w];
        const int base_s = w << 5;
        while (m) {
          const int b = __ffs(m) - 1;
          perm[dst++] = base_s + b;
          m &= (m - 1u);
        }
      }
    }
  }
  __syncthreads();

  const int n_active = sh_meta[0];
  const int n_local = sh_meta[1];
  for (int idx = n_local + tid; idx < S; idx += NUM_WARPS * 32) perm[idx] = 0;
  for (int idx = n_active + tid; idx < S; idx += NUM_WARPS * 32) gids[idx] = 0;
  for (int idx = n_active + tid; idx <= S; idx += NUM_WARPS * 32) goff[idx] = n_local;
}

// ---- Fused Small-M Gate GEMV + Last-CTA Inline Router ----
// Variant A (T <= 4): Block = 192 (6 warps: warps 0..2 -> r0, warps 3..5 -> r1), Grid = (64, 1).
//   Exact 288 / 96 = 3 uint4s/thread for H=2304; 6 warps cover T <= 4 in 1 routing wave!
// Variant B (5 <= T <= 32): Block = 256 (8 warps: warps 0..3 -> r0, warps 4..7 -> r1),
//   T_TILE = 4 -> Grid = (64, ceil(T/4)) = 128..512 CTAs across all 80 SMs, low regs,
//   and 8 warps route T=8 in 1 SINGLE WAVE and T=16 in 2 WAVES!
template <int T_TILE, int H_C, int WARPS_PER_ROW>
__global__ void __launch_bounds__(WARPS_PER_ROW * 64, 4)
k_gate_gemv_route_fused(
    const __half* __restrict__ x,
    const __half* __restrict__ gate_weight,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    float* __restrict__ scratch_logits,
    int32_t* __restrict__ scratch_done,
    int T,
    int H,
    float route_scale,
    int lo,
    int hi) {
  constexpr int THREADS_PER_ROW = WARPS_PER_ROW * 32;  // 96 or 128
  constexpr int NUM_WARPS = WARPS_PER_ROW * 2;         // 6 or 8

  __shared__ float sh_red[2][WARPS_PER_ROW][T_TILE];
  __shared__ int sh_is_last;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int half_id = (tid >= THREADS_PER_ROW) ? 1 : 0;
  const int h_tid = tid - half_id * THREADS_PER_ROW;
  const int lw = warp - half_id * WARPS_PER_ROW;

  const int h_dim = (H_C > 0) ? H_C : H;
  const int n_vec = h_dim >> 3;  // 288 for H=2304
  const int e = blockIdx.x * 2 + half_id;
  const int t0 = blockIdx.y * T_TILE;

  const uint4* w_ptr = reinterpret_cast<const uint4*>(gate_weight + (size_t)e * h_dim);
  const uint4* x_base = reinterpret_cast<const uint4*>(x);

  float acc[T_TILE];
#pragma unroll
  for (int ti = 0; ti < T_TILE; ti++) acc[ti] = 0.0f;

  auto process_vec = [&](int v) {
    const uint4 wv = w_ptr[v];
    const __half2* wh = reinterpret_cast<const __half2*>(&wv);
    const float2 fw0 = __half22float2(wh[0]);
    const float2 fw1 = __half22float2(wh[1]);
    const float2 fw2 = __half22float2(wh[2]);
    const float2 fw3 = __half22float2(wh[3]);
#pragma unroll
    for (int ti = 0; ti < T_TILE; ti++) {
      const int tc = (t0 + ti < T) ? (t0 + ti) : (T - 1);
      const uint4 xv = x_base[(size_t)tc * n_vec + v];
      const __half2* xh = reinterpret_cast<const __half2*>(&xv);
      const float2 fx0 = __half22float2(xh[0]);
      const float2 fx1 = __half22float2(xh[1]);
      const float2 fx2 = __half22float2(xh[2]);
      const float2 fx3 = __half22float2(xh[3]);
      float s = acc[ti];
      s = __fmaf_rn(fx0.x, fw0.x, s);
      s = __fmaf_rn(fx0.y, fw0.y, s);
      s = __fmaf_rn(fx1.x, fw1.x, s);
      s = __fmaf_rn(fx1.y, fw1.y, s);
      s = __fmaf_rn(fx2.x, fw2.x, s);
      s = __fmaf_rn(fx2.y, fw2.y, s);
      s = __fmaf_rn(fx3.x, fw3.x, s);
      s = __fmaf_rn(fx3.y, fw3.y, s);
      acc[ti] = s;
    }
  };

  if constexpr (H_C == 2304 && WARPS_PER_ROW == 3) {
#pragma unroll
    for (int it = 0; it < 3; it++) {
      process_vec(h_tid + it * 96);
    }
  } else if constexpr (H_C == 2304 && WARPS_PER_ROW == 4) {
    process_vec(h_tid);
    process_vec(h_tid + 128);
    if (h_tid < 32) {
      process_vec(h_tid + 256);
    }
  } else {
    for (int v = h_tid; v < n_vec; v += THREADS_PER_ROW) {
      process_vec(v);
    }
  }

#pragma unroll
  for (int ti = 0; ti < T_TILE; ti++) {
    float s = acc[ti];
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    }
    if (lane == 0) sh_red[half_id][lw][ti] = s;
  }
  __syncthreads();

  if (tid < 2 * T_TILE) {
    const int row_local = tid / T_TILE;
    const int ti = tid - row_local * T_TILE;
    const int t = t0 + ti;
    if (t < T) {
      float val = sh_red[row_local][0][ti] + sh_red[row_local][1][ti] + sh_red[row_local][2][ti];
      if constexpr (WARPS_PER_ROW == 4) {
        val += sh_red[row_local][3][ti];
      }
      scratch_logits[t * GEMV_E + (blockIdx.x * 2 + row_local)] = val;
    }
    __threadfence();
  }
  __syncthreads();

  if (tid == 0) {
    const int target = gridDim.x * gridDim.y;
    const int prev = atomicAdd(scratch_done, 1);
    if (prev == target - 1) {
      *scratch_done = 0;
      sh_is_last = 1;
    } else {
      sh_is_last = 0;
    }
  }
  __syncthreads();
  if (!sh_is_last) return;

  if constexpr (T_TILE == 1 && WARPS_PER_ROW == 3) {
    if (warp == 0) {
      route_t1_1warp_ep4<true>(
          scratch_logits, bias, is_padding,
          ids, weights, src2dst, perm, gids, goff,
          GEMV_E, route_scale, lo, hi);
    }
  } else {
    __shared__ uint32_t sh_mask[32][8];
    __shared__ int sh_meta[2];
    route_small_t_bitmask_ep4<true, NUM_WARPS>(
        scratch_logits, bias, is_padding,
        ids, weights, src2dst, perm, gids, goff,
        T, GEMV_E, route_scale, lo, hi, sh_mask, sh_meta);
  }
}

// ---- Standalone fast routing kernels (for precomputed logits) ----

__global__ void __launch_bounds__(32, 1)
k_select_route_t1_1warp(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int E,
    float route_scale,
    int lo,
    int hi) {
  route_t1_1warp_ep4<false>(
      logits, bias, is_padding,
      ids, weights, src2dst, perm, gids, goff,
      E, route_scale, lo, hi);
}

template <int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1)
k_select_route_small_bitmask(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int T,
    int E,
    float route_scale,
    int lo,
    int hi) {
  __shared__ uint32_t sh_mask[32][8];
  __shared__ int sh_meta[2];
  route_small_t_bitmask_ep4<false, NUM_WARPS>(
      logits, bias, is_padding,
      ids, weights, src2dst, perm, gids, goff,
      T, E, route_scale, lo, hi, sh_mask, sh_meta);
}

// ---- General fallback warp selector (arbitrary topk <= 16, E <= 256, E_local <= 64) ----

template <int MAX_PER_LANE, int TOPK_C>
__device__ __forceinline__ void warp_select_token(
    const float* __restrict__ row_logits,
    const float* __restrict__ bias,
    int E,
    int topk,
    float route_scale,
    int lo,
    int hi,
    bool is_pad,
    int* out_local,
    float* out_weight) {
  constexpr int K_ARR = (TOPK_C > 0) ? TOPK_C : MAX_TOPK;
  const int lane = threadIdx.x & 31;
  uint64_t my_key[MAX_PER_LANE];
  float my_logit[MAX_PER_LANE];
#pragma unroll
  for (int j = 0; j < MAX_PER_LANE; j++) {
    const int e = lane + j * 32;
    if (e < E) {
      const float lg = row_logits[e];
      my_logit[j] = lg;
      my_key[j] = make_sel_key(lg, bias[e], e, E);
    } else {
      my_logit[j] = -INFINITY;
      my_key[j] = 0ull;
    }
  }

  float sel_logits[K_ARR];
  int sel_ids[K_ARR];

  const int k_lim = (TOPK_C > 0) ? TOPK_C : topk;
#pragma unroll
  for (int k = 0; k < K_ARR; k++) {
    if constexpr (TOPK_C == 0) {
      if (k >= k_lim) break;
    }
    uint64_t best_k = 0ull;
    float best_l = -INFINITY;
    int best_j = -1;
#pragma unroll
    for (int j = 0; j < MAX_PER_LANE; j++) {
      if (my_key[j] > best_k) {
        best_k = my_key[j];
        best_l = my_logit[j];
        best_j = j;
      }
    }
    uint64_t g_k = best_k;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      const unsigned lo_w = __shfl_xor_sync(0xFFFFFFFFu, (unsigned)g_k, off);
      const unsigned hi_w = __shfl_xor_sync(0xFFFFFFFFu, (unsigned)(g_k >> 32), off);
      const uint64_t ok = (static_cast<uint64_t>(hi_w) << 32) | lo_w;
      if (ok > g_k) g_k = ok;
    }
    const bool is_winner = (best_j >= 0 && best_k == g_k);
    const int win_lane = __ffs(__ballot_sync(0xFFFFFFFFu, is_winner)) - 1;
    const float g_l = __shfl_sync(0xFFFFFFFFu, best_l, win_lane);
    if (is_winner) {
#pragma unroll
      for (int j = 0; j < MAX_PER_LANE; j++) {
        if (j == best_j) my_key[j] = 0ull;
      }
    }
    if (lane == 0) {
      sel_ids[k] = E - static_cast<int>(g_k & 0xFFFFull);
      sel_logits[k] = g_l;
    }
  }

  if (lane == 0) {
    float lp[K_ARR];
    float max_lp = -INFINITY;
#pragma unroll
    for (int k = 0; k < K_ARR; k++) {
      if constexpr (TOPK_C == 0) {
        if (k >= k_lim) break;
      }
      lp[k] = log_sigmoid_exact(sel_logits[k]);
      if (lp[k] > max_lp) max_lp = lp[k];
    }
    float sh[K_ARR];
#pragma unroll
    for (int k = 0; k < K_ARR; k++) {
      if constexpr (TOPK_C == 0) {
        if (k >= k_lim) break;
      }
      sh[k] = expf(lp[k] - max_lp);
    }
    float total = 0.0f;
    if constexpr (TOPK_C == 8) {
      total = ((sh[0] + sh[2]) + (sh[1] + sh[3])) + ((sh[4] + sh[6]) + (sh[5] + sh[7]));
    } else if (k_lim == 8) {
      total = ((sh[0] + sh[2]) + (sh[1] + sh[3])) + ((sh[4] + sh[6]) + (sh[5] + sh[7]));
    } else {
      for (int k = 0; k < k_lim; k++) total += sh[k];
    }
    const float lse = max_lp + logf(total);
#pragma unroll
    for (int k = 0; k < K_ARR; k++) {
      if constexpr (TOPK_C == 0) {
        if (k >= k_lim) break;
      }
      const int id = sel_ids[k];
      out_local[k] = (!is_pad && id >= lo && id < hi) ? (id - lo) : -1;
      out_weight[k] = expf(lp[k] - lse) * route_scale;
    }
  }
}

template <int TOPK_C>
__global__ void __launch_bounds__(256, 2)
k_select_and_route_fused(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int T,
    int E,
    int topk,
    float route_scale,
    int lo,
    int hi) {
  constexpr int K_ARR = (TOPK_C > 0) ? TOPK_C : MAX_TOPK;
  __shared__ int8_t sh_local[FUSED_MAX_S];
  __shared__ int sh_count[MAX_E_LOCAL];
  __shared__ int sh_base[MAX_E_LOCAL];
  __shared__ int sh_n_active;
  __shared__ int sh_n_local;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int E_local = hi - lo;
  const int k_lim = (TOPK_C > 0) ? TOPK_C : topk;
  const int S = T * k_lim;

  if (tid < E_local) sh_count[tid] = 0;
  __syncthreads();

  for (int t = warp; t < T; t += 8) {
    const bool pad = (is_padding != nullptr) ? is_padding[t] : false;
    int loc[K_ARR];
    float wt[K_ARR];
    warp_select_token<4, TOPK_C>(logits + (size_t)t * E, bias, E, k_lim, route_scale, lo, hi, pad, loc, wt);
    if (lane == 0) {
      const int row0 = t * k_lim;
#pragma unroll
      for (int k = 0; k < K_ARR; k++) {
        if constexpr (TOPK_C == 0) {
          if (k >= k_lim) break;
        }
        const int s = row0 + k;
        const int el = loc[k];
        ids[s] = el;
        weights[s] = wt[k];
        src2dst[s] = (el >= 0) ? s : -1;
        sh_local[s] = static_cast<int8_t>(el);
        if (el >= 0) atomicAdd(&sh_count[el], 1);
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    int n_active = 0;
    int off = 0;
    for (int e = 0; e < E_local; e++) {
      const int c = sh_count[e];
      sh_base[e] = off;
      if (c > 0) {
        gids[n_active] = e;
        goff[n_active] = off;
        n_active++;
        off += c;
      }
    }
    sh_n_active = n_active;
    sh_n_local = off;
  }
  __syncthreads();

  const int n_active = sh_n_active;
  const int n_local = sh_n_local;

  if (tid < E_local) {
    if (sh_count[tid] > 0) {
      int dst = sh_base[tid];
      const int8_t target = static_cast<int8_t>(tid);
      for (int s = 0; s < S; s++) {
        if (sh_local[s] == target) {
          perm[dst++] = s;
        }
      }
    }
  }
  for (int idx = n_local + tid; idx < S; idx += 256) perm[idx] = 0;
  for (int idx = n_active + tid; idx < S; idx += 256) gids[idx] = 0;
  for (int idx = n_active + tid; idx <= S; idx += 256) goff[idx] = n_local;
}

// ---- General multi-CTA path for T > 32 (up to 8192+) ----

__global__ void __launch_bounds__(256, 4)
k_select_count_multi_fast8(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ rank_buf,
    int32_t* __restrict__ counts,
    int T,
    int E,
    float route_scale,
    int lo,
    int hi) {
  const int warp_global = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp_global >= T) return;
  const bool pad = (is_padding != nullptr) ? is_padding[warp_global] : false;
  int el = -1;
  float wt = 0.0f;
  warp_select_top8_fast<false>(
      logits + (size_t)warp_global * E, bias, E, route_scale, lo, hi, pad, el, wt);
  if (lane < 8) {
    const int s = warp_global * 8 + lane;
    ids[s] = el;
    weights[s] = wt;
    src2dst[s] = (el >= 0) ? s : -1;
    rank_buf[s] = (el >= 0) ? atomicAdd(&counts[el], 1) : -1;
  }
}

template <int MAX_PER_LANE, int TOPK_C>
__global__ void __launch_bounds__(256, 2)
k_select_count_multi(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ is_padding,
    int32_t* __restrict__ ids,
    float* __restrict__ weights,
    int32_t* __restrict__ src2dst,
    int32_t* __restrict__ rank_buf,
    int32_t* __restrict__ counts,
    int T,
    int E,
    int topk,
    float route_scale,
    int lo,
    int hi) {
  constexpr int K_ARR = (TOPK_C > 0) ? TOPK_C : MAX_TOPK;
  const int warp_global = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp_global >= T) return;
  const bool pad = (is_padding != nullptr) ? is_padding[warp_global] : false;
  const int k_lim = (TOPK_C > 0) ? TOPK_C : topk;
  int loc[K_ARR];
  float wt[K_ARR];
  warp_select_token<MAX_PER_LANE, TOPK_C>(
      logits + (size_t)warp_global * E, bias, E, k_lim, route_scale, lo, hi, pad, loc, wt);
  if (lane == 0) {
    const int row0 = warp_global * k_lim;
#pragma unroll
    for (int k = 0; k < K_ARR; k++) {
      if constexpr (TOPK_C == 0) {
        if (k >= k_lim) break;
      }
      const int s = row0 + k;
      const int el = loc[k];
      ids[s] = el;
      weights[s] = wt[k];
      src2dst[s] = (el >= 0) ? s : -1;
      rank_buf[s] = (el >= 0) ? atomicAdd(&counts[el], 1) : -1;
    }
  }
}

__global__ void __launch_bounds__(64, 1)
k_scan_groups(
    int32_t* __restrict__ counts,
    int32_t* __restrict__ base,
    int32_t* __restrict__ meta,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int E_local) {
  if (threadIdx.x != 0) return;
  int n_active = 0;
  int off = 0;
  for (int e = 0; e < E_local; e++) {
    const int c = counts[e];
    counts[e] = 0;
    base[e] = off;
    if (c > 0) {
      gids[n_active] = e;
      goff[n_active] = off;
      n_active++;
      off += c;
    }
  }
  meta[0] = n_active;
  meta[1] = off;
}

__global__ void __launch_bounds__(256, 4)
k_scatter_perm_tail(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ rank_buf,
    const int32_t* __restrict__ base,
    const int32_t* __restrict__ meta,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gids,
    int32_t* __restrict__ goff,
    int S) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  const int n_active = meta[0];
  const int n_local = meta[1];
  if (idx < S) {
    const int el = ids[idx];
    if (el >= 0) {
      perm[base[el] + rank_buf[idx]] = idx;
    }
    if (idx >= n_local) perm[idx] = 0;
    if (idx >= n_active) gids[idx] = 0;
  }
  if (idx >= n_active && idx <= S) {
    goff[idx] = n_local;
  }
}

}  // namespace

void ep4_gemv_select_and_route(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor bias,
    c10::optional<torch::Tensor> is_padding,
    torch::Tensor ids,
    torch::Tensor weights,
    torch::Tensor src2dst,
    torch::Tensor perm,
    torch::Tensor gids,
    torch::Tensor goff,
    torch::Tensor scratch_logits,
    torch::Tensor scratch_done,
    int64_t topk,
    double route_scale,
    int64_t lo,
    int64_t hi) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16 && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(gate_weight.is_cuda() && gate_weight.dtype() == torch::kFloat16 &&
              gate_weight.dim() == 2 && gate_weight.is_contiguous());
  TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.is_contiguous());
  const int T = (int)x.size(0);
  const int H = (int)x.size(1);
  const int E = (int)gate_weight.size(0);
  const int E_local = (int)(hi - lo);
  if (T == 0) return;

  TORCH_CHECK(E == GEMV_E, "ep4_gemv_select_and_route requires E == 128, got ", E);
  TORCH_CHECK(gate_weight.size(1) == H && (H % 8) == 0, "H mismatch or not multiple of 8");
  TORCH_CHECK(topk == 8, "ep4_gemv_select_and_route requires topk == 8, got ", topk);
  TORCH_CHECK(E_local > 0 && E_local <= 32, "ep4_gemv_select_and_route requires E_local in [1..32]");
  TORCH_CHECK(T <= GEMV_MAX_T, "ep4_gemv_select_and_route requires T <= 32, got ", T);
  TORCH_CHECK(scratch_logits.numel() >= T * E && scratch_logits.dtype() == torch::kFloat32);
  TORCH_CHECK(scratch_done.numel() >= 1 && scratch_done.dtype() == torch::kInt32);
  TORCH_CHECK((((uintptr_t)x.data_ptr()) & 15) == 0 && (((uintptr_t)gate_weight.data_ptr()) & 15) == 0);

  const at::cuda::CUDAGuard guard(x.device());
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const bool* pad_ptr = nullptr;
  if (is_padding.has_value() && is_padding.value().defined() && is_padding.value().numel() > 0) {
    TORCH_CHECK(is_padding.value().is_cuda() && is_padding.value().dtype() == torch::kBool &&
                is_padding.value().is_contiguous());
    pad_ptr = is_padding.value().data_ptr<bool>();
  }

  auto launch = [&](auto t_tile_tag, auto wpr_tag) {
    constexpr int TT = decltype(t_tile_tag)::value;
    constexpr int WPR = decltype(wpr_tag)::value;
    const dim3 grid(64, (T + TT - 1) / TT, 1);
    const int threads = WPR * 64;
    if (H == 2304) {
      k_gate_gemv_route_fused<TT, 2304, WPR><<<grid, threads, 0, st>>>(
          reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(gate_weight.data_ptr<at::Half>()),
          bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
          scratch_logits.data_ptr<float>(), scratch_done.data_ptr<int32_t>(),
          T, H, (float)route_scale, (int)lo, (int)hi);
    } else {
      k_gate_gemv_route_fused<TT, 0, WPR><<<grid, threads, 0, st>>>(
          reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(gate_weight.data_ptr<at::Half>()),
          bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
          scratch_logits.data_ptr<float>(), scratch_done.data_ptr<int32_t>(),
          T, H, (float)route_scale, (int)lo, (int)hi);
    }
  };

  if (T == 1) {
    launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 3>{});
  } else if (T == 2) {
    launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 3>{});
  } else if (T <= 4) {
    launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 3>{});
  } else {
    // 8 warps/CTA (WPR=4) -> T=8 routes in 1 single wave, T=16 in 2 waves, grid=(64, ceil(T/4))!
    launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void ep4_select_and_route(
    torch::Tensor logits,
    torch::Tensor bias,
    c10::optional<torch::Tensor> is_padding,
    torch::Tensor ids,
    torch::Tensor weights,
    torch::Tensor src2dst,
    torch::Tensor perm,
    torch::Tensor gids,
    torch::Tensor goff,
    torch::Tensor scratch_counts,
    torch::Tensor scratch_base,
    torch::Tensor scratch_meta,
    torch::Tensor scratch_rank,
    int64_t topk,
    double route_scale,
    int64_t lo,
    int64_t hi) {
  TORCH_CHECK(logits.is_cuda() && logits.dtype() == torch::kFloat32 && logits.dim() == 2 && logits.is_contiguous());
  TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.is_contiguous());
  const int T = (int)logits.size(0);
  const int E = (int)logits.size(1);
  const int E_local = (int)(hi - lo);
  const int S = T * (int)topk;
  if (T == 0) return;

  TORCH_CHECK(E > 0 && E <= 256, "E must be in [1..256], got ", E);
  TORCH_CHECK(E_local > 0 && E_local <= MAX_E_LOCAL, "E_local must be in [1..64], got ", E_local);
  TORCH_CHECK(topk > 0 && topk <= MAX_TOPK, "topk must be in [1..16], got ", topk);

  const at::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const bool* pad_ptr = nullptr;
  if (is_padding.has_value() && is_padding.value().defined() && is_padding.value().numel() > 0) {
    TORCH_CHECK(is_padding.value().is_cuda() && is_padding.value().dtype() == torch::kBool &&
                is_padding.value().is_contiguous());
    pad_ptr = is_padding.value().data_ptr<bool>();
  }

  if (T == 1 && E <= 128 && topk == 8 && E_local <= 32) {
    k_select_route_t1_1warp<<<1, 32, 0, st>>>(
        logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
        ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
        perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
        E, (float)route_scale, (int)lo, (int)hi);
  } else if (T <= 4 && E <= 128 && topk == 8 && E_local <= 32) {
    k_select_route_small_bitmask<4><<<1, 128, 0, st>>>(
        logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
        ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
        perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
        T, E, (float)route_scale, (int)lo, (int)hi);
  } else if (T <= 8 && E <= 128 && topk == 8 && E_local <= 32) {
    k_select_route_small_bitmask<8><<<1, 256, 0, st>>>(
        logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
        ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
        perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
        T, E, (float)route_scale, (int)lo, (int)hi);
  } else if (T <= 32 && E <= 128 && topk == 8 && E_local <= 32) {
    k_select_route_small_bitmask<16><<<1, 512, 0, st>>>(
        logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
        ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
        perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
        T, E, (float)route_scale, (int)lo, (int)hi);
  } else if (T <= FUSED_MAX_T && E <= 128 && S <= FUSED_MAX_S) {
    if (topk == 8) {
      k_select_and_route_fused<8><<<1, 256, 0, st>>>(
          logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
          T, E, 8, (float)route_scale, (int)lo, (int)hi);
    } else {
      k_select_and_route_fused<0><<<1, 256, 0, st>>>(
          logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          perm.data_ptr<int32_t>(), gids.data_ptr<int32_t>(), goff.data_ptr<int32_t>(),
          T, E, (int)topk, (float)route_scale, (int)lo, (int)hi);
    }
  } else {
    TORCH_CHECK(scratch_rank.numel() >= S, "scratch_rank too small");
    const int blocks_t = (T + 7) / 8;
    if (E <= 128 && topk == 8) {
      k_select_count_multi_fast8<<<blocks_t, 256, 0, st>>>(
          logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          scratch_rank.data_ptr<int32_t>(), scratch_counts.data_ptr<int32_t>(),
          T, E, (float)route_scale, (int)lo, (int)hi);
    } else if (E <= 128) {
      k_select_count_multi<4, 0><<<blocks_t, 256, 0, st>>>(
          logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          scratch_rank.data_ptr<int32_t>(), scratch_counts.data_ptr<int32_t>(),
          T, E, (int)topk, (float)route_scale, (int)lo, (int)hi);
    } else {
      k_select_count_multi<8, 0><<<blocks_t, 256, 0, st>>>(
          logits.data_ptr<float>(), bias.data_ptr<float>(), pad_ptr,
          ids.data_ptr<int32_t>(), weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
          scratch_rank.data_ptr<int32_t>(), scratch_counts.data_ptr<int32_t>(),
          T, E, (int)topk, (float)route_scale, (int)lo, (int)hi);
    }
    k_scan_groups<<<1, 64, 0, st>>>(
        scratch_counts.data_ptr<int32_t>(),
        scratch_base.data_ptr<int32_t>(),
        scratch_meta.data_ptr<int32_t>(),
        gids.data_ptr<int32_t>(),
        goff.data_ptr<int32_t>(),
        E_local);
    const int blocks_s = (S + 1 + 255) / 256;
    k_scatter_perm_tail<<<blocks_s, 256, 0, st>>>(
        ids.data_ptr<int32_t>(),
        scratch_rank.data_ptr<int32_t>(),
        scratch_base.data_ptr<int32_t>(),
        scratch_meta.data_ptr<int32_t>(),
        perm.data_ptr<int32_t>(),
        gids.data_ptr<int32_t>(),
        goff.data_ptr<int32_t>(),
        S);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ep4_gemv_select_and_route", &ep4_gemv_select_and_route,
        "Single-kernel fused Inkling2 gate GEMV + top-8 router + zero-sync EP=4 grouping (sm_70)");
  m.def("ep4_select_and_route", &ep4_select_and_route,
        "Fused Inkling2 MoE top-k router + zero-sync EP=4 sentinel grouping (sm_70)");
}
