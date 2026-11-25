// SPDX-License-Identifier: Apache-2.0
// loop_blend.cu - Fused 2-kernel Volta (sm_70) Inkling2 loop message controller blend().
//
// Natively supports gate_rank R in {64, 128} (Inkling-2-Mini uses R=64, H=2304, S in {1, 3})
// for small-M decode/verify (T <= 32, D % 32 == 0):
//   Kernel 1 (`k_blend_down`):
//     Exploits dot-product linearity to fuse `pair_rms_norm` directly into `down` GEMV:
//       hidden[s, t, r] = half(dest_rstd * dot(dest, down_0[r]) + msg_rstd * dot(msg, down_1[r]))
//     100% coalesced contiguous float4 (dest/msg) + uint2 (down) loads (576/96 = 6 exact iters
//     for D=2304), R_PER_CTA in {1, 2, 4} for optimal SM occupancy & 2x-4x L2 reuse!
//   Kernel 2 (`k_blend_up_norm`):
//     72 CTAs/stream (32 cols/CTA), T_TILE in {1, 2, 4} (so T=8..32 split across z-tiles for
//     multi-CTA parallel RMSNorm finish!):
//       - R=64:  8 lanes/col (128 B exact cache line), all 32 cols in 1 iteration, 3-step shuffle!
//       - R=128: 16 lanes/col (256 B coalesced), 32 cols in 2 iterations, 4-step shuffle!
//     Fuses `tanh(gain/forget)` + `(1 - forget) * dest + gain * message` + block `sum_sq`
//     atomicAdd + atomic last-CTA arrival -> warp-parallel inline RMSNorm finish via `__ldcg`!
//     Zero 3rd kernel launch, zero memset (self-resetting counters), 100% CUDA-graph safe!

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BLEND_MAX_T = 32;
constexpr int BLEND_MAX_S = 4;

// ---- Kernel 1: Fused pair_rms_norm + down GEMV -> hidden [S, T, R] FP16 ----
// Grid: (R / R_PER_CTA, S, num_t_tiles), Block: 192 (6 warps):
//   Warps 0..2 (tid 0..95)  -> dest half [0..D-1] & down[s, r, 0..D-1]
//   Warps 3..5 (tid 96..191)-> msg  half [0..D-1] & down[s, r, D..2D-1]
// Contiguous float4 (x) + uint2 (down) loads -> ZERO stride-2 cache-line waste!
template <int T_TILE, int D_C, int R_C, int R_PER_CTA>
__global__ void __launch_bounds__(192, 4)
k_blend_down(
    const float* __restrict__ dest,
    const float* __restrict__ message,
    const __half* __restrict__ down,
    __half* __restrict__ hidden,
    int64_t stride_dest_s,
    int T,
    int D) {
  // [hw 0..1][lw 0..2][0..R_PER_CTA-1 = dots, R_PER_CTA = ssq][ti]
  __shared__ float sh_red[2][3][R_PER_CTA + 1][T_TILE];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hw = (tid >= 96) ? 1 : 0;   // 0 = dest, 1 = message
  const int h_tid = tid - hw * 96;      // 0..95
  const int lw = warp - hw * 3;         // 0..2

  const int d_dim = (D_C > 0) ? D_C : D;
  const int n_vec4 = d_dim >> 2;        // 576 for D=2304 (576 / 96 = 6 exact!)
  const int r0 = blockIdx.x * R_PER_CTA;
  const int s = blockIdx.y;
  const int t0 = blockIdx.z * T_TILE;

  const size_t down_row_stride4 = ((size_t)2 * d_dim) >> 2;
  const size_t down_s_off4 = (((size_t)s * R_C * 2 + hw) * d_dim) >> 2;
  const uint2* w_base = reinterpret_cast<const uint2*>(down) + down_s_off4 + (size_t)r0 * down_row_stride4;

  const float4* x_base = reinterpret_cast<const float4*>(
      (hw == 0)
          ? (dest + (int64_t)s * stride_dest_s)
          : (message + (int64_t)s * T * d_dim));

  float dots[R_PER_CTA][T_TILE];
  float ssq[T_TILE];
#pragma unroll
  for (int ti = 0; ti < T_TILE; ti++) {
    ssq[ti] = 0.0f;
#pragma unroll
    for (int ri = 0; ri < R_PER_CTA; ri++) {
      dots[ri][ti] = 0.0f;
    }
  }

  auto process_v4 = [&](int v) {
    uint2 wu[R_PER_CTA];
#pragma unroll
    for (int ri = 0; ri < R_PER_CTA; ri++) {
      wu[ri] = w_base[(size_t)ri * down_row_stride4 + v];
    }
#pragma unroll
    for (int ti = 0; ti < T_TILE; ti++) {
      const int tc = (t0 + ti < T) ? (t0 + ti) : (T - 1);
      const float4 xv = x_base[(int64_t)tc * n_vec4 + v];
      float ss = ssq[ti];
      ss = __fmaf_rn(xv.x, xv.x, ss);
      ss = __fmaf_rn(xv.y, xv.y, ss);
      ss = __fmaf_rn(xv.z, xv.z, ss);
      ss = __fmaf_rn(xv.w, xv.w, ss);
      ssq[ti] = ss;

#pragma unroll
      for (int ri = 0; ri < R_PER_CTA; ri++) {
        const __half2* hh = reinterpret_cast<const __half2*>(&wu[ri]);
        const float2 w01 = __half22float2(hh[0]);
        const float2 w23 = __half22float2(hh[1]);
        float d = dots[ri][ti];
        d = __fmaf_rn(xv.x, w01.x, d);
        d = __fmaf_rn(xv.y, w01.y, d);
        d = __fmaf_rn(xv.z, w23.x, d);
        d = __fmaf_rn(xv.w, w23.y, d);
        dots[ri][ti] = d;
      }
    }
  };

  if constexpr (D_C == 2304) {
    // 576 / 96 = 6 exact contiguous float4/uint2 iterations!
#pragma unroll
    for (int it = 0; it < 6; it++) {
      process_v4(h_tid + it * 96);
    }
  } else {
    for (int v = h_tid; v < n_vec4; v += 96) {
      process_v4(v);
    }
  }

  // Warp-reduce dots and ssq.
#pragma unroll
  for (int ti = 0; ti < T_TILE; ti++) {
    float ss = ssq[ti];
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    }
    if (lane == 0) sh_red[hw][lw][R_PER_CTA][ti] = ss;

#pragma unroll
    for (int ri = 0; ri < R_PER_CTA; ri++) {
      float d = dots[ri][ti];
#pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
        d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
      }
      if (lane == 0) sh_red[hw][lw][ri][ti] = d;
    }
  }
  __syncthreads();

  if (tid < T_TILE) {
    const int t = t0 + tid;
    if (t < T) {
      const float d_ssq = sh_red[0][0][R_PER_CTA][tid] + sh_red[0][1][R_PER_CTA][tid] + sh_red[0][2][R_PER_CTA][tid];
      const float m_ssq = sh_red[1][0][R_PER_CTA][tid] + sh_red[1][1][R_PER_CTA][tid] + sh_red[1][2][R_PER_CTA][tid];
      const float d_rstd = rsqrtf(__fdiv_rn(d_ssq, (float)d_dim) + 1e-6f);
      const float m_rstd = rsqrtf(__fdiv_rn(m_ssq, (float)d_dim) + 1e-6f);
      const size_t out_off = ((size_t)s * T + t) * R_C;
#pragma unroll
      for (int ri = 0; ri < R_PER_CTA; ri++) {
        const float dd = sh_red[0][0][ri][tid] + sh_red[0][1][ri][tid] + sh_red[0][2][ri][tid];
        const float md = sh_red[1][0][ri][tid] + sh_red[1][1][ri][tid] + sh_red[1][2][ri][tid];
        hidden[out_off + r0 + ri] = __float2half_rn(__fmaf_rn(dd, d_rstd, md * m_rstd));
      }
    }
  }
}

// ---- Kernel 2: Fused up GEMV + tanh + blend + warp-parallel last-CTA RMSNorm ----
// Grid: (D / 32 = 72, S, ceil(T / T_TILE)), Block: 256 (8 warps):
//   - T_TILE in {1, 2, 4}: for T=8..32, multiple z-tiles finish RMSNorm in parallel across SMs!
//   - Inside Step 4 (last CTA), 8 warps split evenly across T_TILE tokens (2 warps/token at T_TILE=4
//     -> 576 / 64 = 9 exact contiguous float4s per thread, zero inner token loop!).
template <int T_TILE, int R_C>
__global__ void __launch_bounds__(256, 4)
k_blend_up_norm(
    const __half* __restrict__ hidden,
    const __half* __restrict__ up,
    const float* __restrict__ dest,
    const float* __restrict__ message,
    const __half* __restrict__ norm_weight,
    float* __restrict__ scratch_x,
    float* __restrict__ scratch_sum_sq,
    int32_t* __restrict__ scratch_done,
    __half* __restrict__ out,
    int64_t stride_dest_s,
    int64_t stride_w_s,
    int T,
    int D,
    float norm_eps) {
  constexpr int VECS_PER_ROW = R_C / 8;               // 8 (R=64) or 16 (R=128)
  constexpr int LANES_PER_COL = VECS_PER_ROW;         // 8 or 16
  constexpr int COLS_PER_ITER = 256 / LANES_PER_COL;  // 32 (R=64) or 16 (R=128)
  constexpr int N_ITERS = 32 / COLS_PER_ITER;         // 1 (R=64) or 2 (R=128)
  constexpr int LANE_SHIFT = (LANES_PER_COL == 8) ? 3 : 4;
  constexpr int LANE_MASK = LANES_PER_COL - 1;

  __shared__ uint4 sh_h[T_TILE][VECS_PER_ROW];
  __shared__ float sh_g[T_TILE][33];
  __shared__ float sh_f[T_TILE][33];
  __shared__ float sh_rstd[T_TILE];
  __shared__ int sh_is_last;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int grp = tid >> LANE_SHIFT;
  const int gl = tid & LANE_MASK;

  const int d_base = blockIdx.x * 32;
  const int s = blockIdx.y;
  const int tz = blockIdx.z;
  const int t0 = tz * T_TILE;
  const int t_cnt = ((T - t0) < T_TILE) ? (T - t0) : T_TILE;

  // Step 1: Load hidden[s, t0..t0+t_cnt-1, 0..R_C-1] into shared memory.
  if (tid < T_TILE * VECS_PER_ROW) {
    const int ti = tid >> LANE_SHIFT;
    const int vi = tid & LANE_MASK;
    const int tc = (t0 + ti < T) ? (t0 + ti) : (T - 1);
    sh_h[ti][vi] = reinterpret_cast<const uint4*>(hidden + ((size_t)s * T + tc) * R_C)[vi];
  }
  __syncthreads();

  // Step 2: Stream up weights (gain rows d_base..d_base+31, forget rows D+d_base..D+d_base+31).
  const size_t up_s_off = (size_t)s * (size_t)(2 * D) * R_C;

#pragma unroll
  for (int it = 0; it < N_ITERS; it++) {
    const int col = it * COLS_PER_ITER + grp;
    const int d = d_base + col;

    // Gain row `d`
    {
      const uint4 wg = reinterpret_cast<const uint4*>(up + up_s_off + (size_t)d * R_C)[gl];
      const __half2* wh = reinterpret_cast<const __half2*>(&wg);
      const float2 fw0 = __half22float2(wh[0]);
      const float2 fw1 = __half22float2(wh[1]);
      const float2 fw2 = __half22float2(wh[2]);
      const float2 fw3 = __half22float2(wh[3]);
      float acc[T_TILE];
#pragma unroll
      for (int ti = 0; ti < T_TILE; ti++) {
        const uint4 hv = sh_h[ti][gl];
        const __half2* hh = reinterpret_cast<const __half2*>(&hv);
        const float2 fh0 = __half22float2(hh[0]);
        const float2 fh1 = __half22float2(hh[1]);
        const float2 fh2 = __half22float2(hh[2]);
        const float2 fh3 = __half22float2(hh[3]);
        float v = fh0.x * fw0.x;
        v = __fmaf_rn(fh0.y, fw0.y, v);
        v = __fmaf_rn(fh1.x, fw1.x, v);
        v = __fmaf_rn(fh1.y, fw1.y, v);
        v = __fmaf_rn(fh2.x, fw2.x, v);
        v = __fmaf_rn(fh2.y, fw2.y, v);
        v = __fmaf_rn(fh3.x, fw3.x, v);
        v = __fmaf_rn(fh3.y, fw3.y, v);
        acc[ti] = v;
      }
#pragma unroll
      for (int ti = 0; ti < T_TILE; ti++) {
        float v = acc[ti];
        if constexpr (LANES_PER_COL == 16) {
          v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
        }
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
        if (gl == 0) sh_g[ti][col] = v;
      }
    }

    // Forget row `D + d`
    {
      const uint4 wf = reinterpret_cast<const uint4*>(up + up_s_off + (size_t)(D + d) * R_C)[gl];
      const __half2* wh = reinterpret_cast<const __half2*>(&wf);
      const float2 fw0 = __half22float2(wh[0]);
      const float2 fw1 = __half22float2(wh[1]);
      const float2 fw2 = __half22float2(wh[2]);
      const float2 fw3 = __half22float2(wh[3]);
      float acc[T_TILE];
#pragma unroll
      for (int ti = 0; ti < T_TILE; ti++) {
        const uint4 hv = sh_h[ti][gl];
        const __half2* hh = reinterpret_cast<const __half2*>(&hv);
        const float2 fh0 = __half22float2(hh[0]);
        const float2 fh1 = __half22float2(hh[1]);
        const float2 fh2 = __half22float2(hh[2]);
        const float2 fh3 = __half22float2(hh[3]);
        float v = fh0.x * fw0.x;
        v = __fmaf_rn(fh0.y, fw0.y, v);
        v = __fmaf_rn(fh1.x, fw1.x, v);
        v = __fmaf_rn(fh1.y, fw1.y, v);
        v = __fmaf_rn(fh2.x, fw2.x, v);
        v = __fmaf_rn(fh2.y, fw2.y, v);
        v = __fmaf_rn(fh3.x, fw3.x, v);
        v = __fmaf_rn(fh3.y, fw3.y, v);
        acc[ti] = v;
      }
#pragma unroll
      for (int ti = 0; ti < T_TILE; ti++) {
        float v = acc[ti];
        if constexpr (LANES_PER_COL == 16) {
          v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
        }
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
        if (gl == 0) sh_f[ti][col] = v;
      }
    }
  }
  __syncthreads();

  // Step 3: Coalesced tanh + blend + scratch_x write + warp sum_sq atomicAdd.
  if (warp < t_cnt) {
    const int t = t0 + warp;
    const int d = d_base + lane;
    const float g = tanhf(sh_g[warp][lane]);
    const float f = tanhf(sh_f[warp][lane]);
    const float dv = dest[(int64_t)s * stride_dest_s + (int64_t)t * D + d];
    const float mv = message[((int64_t)s * T + t) * D + d];
    const float xv = (1.0f - f) * dv + g * mv;
    scratch_x[((size_t)s * T + t) * D + d] = xv;
    float ss = xv * xv;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    }
    if (lane == 0) {
      atomicAdd(&scratch_sum_sq[s * T + t], ss);
    }
    __threadfence();
  }
  __syncthreads();

  if (tid == 0) {
    const int done_idx = s * gridDim.z + tz;
    const int prev = atomicAdd(&scratch_done[done_idx], 1);
    if (prev == gridDim.x - 1) {
      scratch_done[done_idx] = 0;
      sh_is_last = 1;
    } else {
      sh_is_last = 0;
    }
  }
  __syncthreads();
  if (!sh_is_last) return;

  // Step 4 (Unique last CTA per (s, tz)): Warp-parallel RMSNorm across all t_cnt tokens!
  if (tid < t_cnt) {
    const int t = t0 + tid;
    const float ss = __ldcg(&scratch_sum_sq[s * T + t]);
    scratch_sum_sq[s * T + t] = 0.0f;
    sh_rstd[tid] = rsqrtf(__fdiv_rn(ss, (float)D) + norm_eps);
  }
  __syncthreads();

  // Split 8 warps evenly across T_TILE tokens:
  //   T_TILE=1 -> 256 threads/token, T_TILE=2 -> 128 threads/token, T_TILE=4 -> 64 threads/token!
  constexpr int THREADS_PER_TOK = 256 / T_TILE;
  const int ti = tid / THREADS_PER_TOK;
  const int t_tid = tid - ti * THREADS_PER_TOK;
  if (ti < t_cnt) {
    const int t = t0 + ti;
    const float rstd = sh_rstd[ti];
    const int n_vec4 = D >> 2;
    const size_t row_off4 = (((size_t)s * T + t) * D) >> 2;
    const uint2* nw_ptr = reinterpret_cast<const uint2*>(norm_weight + (int64_t)s * stride_w_s);
    const float4* x_ptr = reinterpret_cast<const float4*>(scratch_x) + row_off4;
    uint2* out_ptr = reinterpret_cast<uint2*>(out) + row_off4;
    for (int idx4 = t_tid; idx4 < n_vec4; idx4 += THREADS_PER_TOK) {
      const float4 xv = __ldcg(x_ptr + idx4);
      const uint2 nw_u = nw_ptr[idx4];
      const __half2* nwh = reinterpret_cast<const __half2*>(&nw_u);
      const float2 w01 = __half22float2(nwh[0]);
      const float2 w23 = __half22float2(nwh[1]);
      uint2 out_u;
      reinterpret_cast<__half2*>(&out_u)[0] = __floats2half2_rn(xv.x * rstd * w01.x, xv.y * rstd * w01.y);
      reinterpret_cast<__half2*>(&out_u)[1] = __floats2half2_rn(xv.z * rstd * w23.x, xv.w * rstd * w23.y);
      out_ptr[idx4] = out_u;
    }
  }
}

template <int T_TILE, int R_C>
void dispatch_loop_blend_typed(
    const float* dest_ptr,
    const float* msg_ptr,
    const __half* down_ptr,
    const __half* up_ptr,
    const __half* nw_ptr,
    __half* out_ptr,
    __half* sh_hidden,
    float* sh_x,
    float* sh_ss,
    int32_t* sh_done,
    int64_t stride_dest_s,
    int64_t stride_w_s,
    int S,
    int T,
    int D,
    float norm_eps,
    cudaStream_t st) {
  const int n_tiles = (T + T_TILE - 1) / T_TILE;
  if (D == 2304) {
    // Choose R_PER_CTA to balance SM occupancy (>= 48..64 CTAs on V100's 80 SMs) vs L2 reuse!
    //   - S * n_tiles == 1 && R == 64 && T_TILE == 1 -> R_PER_CTA=1 (64 CTAs)
    //   - S * n_tiles >= 2 && T_TILE >= 4            -> R_PER_CTA=4 (4x dest/msg register reuse!)
    //   - Otherwise                                  -> R_PER_CTA=2 (2x dest/msg register reuse!)
    if constexpr (R_C == 64) {
      if (S * n_tiles == 1 && T_TILE <= 2) {
        const dim3 grid_down(64, S, n_tiles);
        k_blend_down<T_TILE, 2304, 64, 1><<<grid_down, 192, 0, st>>>(
            dest_ptr, msg_ptr, down_ptr, sh_hidden, stride_dest_s, T, D);
      } else if (S * n_tiles >= 3 && T_TILE == 4) {
        const dim3 grid_down(16, S, n_tiles);
        k_blend_down<T_TILE, 2304, 64, 4><<<grid_down, 192, 0, st>>>(
            dest_ptr, msg_ptr, down_ptr, sh_hidden, stride_dest_s, T, D);
      } else {
        const dim3 grid_down(32, S, n_tiles);
        k_blend_down<T_TILE, 2304, 64, 2><<<grid_down, 192, 0, st>>>(
            dest_ptr, msg_ptr, down_ptr, sh_hidden, stride_dest_s, T, D);
      }
    } else {
      if (S * n_tiles >= 2 && T_TILE == 4) {
        const dim3 grid_down(32, S, n_tiles);
        k_blend_down<T_TILE, 2304, 128, 4><<<grid_down, 192, 0, st>>>(
            dest_ptr, msg_ptr, down_ptr, sh_hidden, stride_dest_s, T, D);
      } else {
        const dim3 grid_down(64, S, n_tiles);
        k_blend_down<T_TILE, 2304, 128, 2><<<grid_down, 192, 0, st>>>(
            dest_ptr, msg_ptr, down_ptr, sh_hidden, stride_dest_s, T, D);
      }
    }
  } else {
    const dim3 grid_down(R_C, S, n_tiles);
    k_blend_down<T_TILE, 0, R_C, 1><<<grid_down, 192, 0, st>>>(
        dest_ptr, msg_ptr, down_ptr, sh_hidden, stride_dest_s, T, D);
  }

  const dim3 grid_up(D / 32, S, n_tiles);
  k_blend_up_norm<T_TILE, R_C><<<grid_up, 256, 0, st>>>(
      sh_hidden, up_ptr, dest_ptr, msg_ptr, nw_ptr,
      sh_x, sh_ss, sh_done, out_ptr,
      stride_dest_s, stride_w_s, T, D, norm_eps);
}

}  // namespace

void loop_blend_sm70(
    torch::Tensor dest,
    torch::Tensor message,
    torch::Tensor down,
    torch::Tensor up,
    torch::Tensor norm_weight,
    torch::Tensor out,
    torch::Tensor scratch_hidden,
    torch::Tensor scratch_x,
    torch::Tensor scratch_sum_sq,
    torch::Tensor scratch_done,
    double norm_eps) {
  TORCH_CHECK(dest.is_cuda() && dest.dtype() == torch::kFloat32 && dest.dim() == 3);
  TORCH_CHECK(message.is_cuda() && message.dtype() == torch::kFloat32 && message.dim() == 3 &&
              message.is_contiguous());
  TORCH_CHECK(down.is_cuda() && down.dtype() == torch::kFloat16 && down.dim() == 3 &&
              down.is_contiguous());
  TORCH_CHECK(up.is_cuda() && up.dtype() == torch::kFloat16 && up.dim() == 3 &&
              up.is_contiguous());
  TORCH_CHECK(norm_weight.is_cuda() && norm_weight.dtype() == torch::kFloat16 &&
              norm_weight.dim() == 2);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat16 && out.dim() == 3 &&
              out.is_contiguous());

  const int S = (int)message.size(0);
  const int T = (int)message.size(1);
  const int D = (int)message.size(2);
  const int R = (int)down.size(1);
  if (S == 0 || T == 0) return;

  TORCH_CHECK(S > 0 && S <= BLEND_MAX_S, "S must be in [1..4], got ", S);
  TORCH_CHECK(T > 0 && T <= BLEND_MAX_T, "T must be in [1..32], got ", T);
  TORCH_CHECK(R == 64 || R == 128, "R must be 64 or 128, got ", R);
  TORCH_CHECK((D % 32) == 0, "D must be a multiple of 32, got ", D);
  TORCH_CHECK(dest.size(0) == S && dest.size(1) == T && dest.size(2) == D);
  TORCH_CHECK(dest.stride(1) == D && dest.stride(2) == 1);
  TORCH_CHECK((dest.stride(0) % 4) == 0, "dest stride(0) must be multiple of 4");
  TORCH_CHECK(down.size(0) == S && down.size(2) == 2 * D);
  TORCH_CHECK(up.size(0) == S && up.size(1) == 2 * D && up.size(2) == R);
  TORCH_CHECK(norm_weight.size(0) == S && norm_weight.size(1) == D && norm_weight.stride(1) == 1);
  TORCH_CHECK((norm_weight.stride(0) % 4) == 0, "norm_weight stride(0) must be multiple of 4");
  TORCH_CHECK(out.size(0) == S && out.size(1) == T && out.size(2) == D);

  TORCH_CHECK((((uintptr_t)dest.data_ptr()) & 15) == 0);
  TORCH_CHECK((((uintptr_t)message.data_ptr()) & 15) == 0);
  TORCH_CHECK((((uintptr_t)down.data_ptr()) & 15) == 0);
  TORCH_CHECK((((uintptr_t)up.data_ptr()) & 15) == 0);
  TORCH_CHECK((((uintptr_t)norm_weight.data_ptr()) & 7) == 0);

  TORCH_CHECK(scratch_hidden.numel() >= (int64_t)S * T * R);
  TORCH_CHECK(scratch_x.numel() >= (int64_t)S * T * D);
  TORCH_CHECK(scratch_sum_sq.numel() >= (int64_t)S * T);
  TORCH_CHECK(scratch_done.numel() >= (int64_t)S * 8);

  const at::cuda::CUDAGuard guard(message.device());
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();

  const float* dest_ptr = dest.data_ptr<float>();
  const float* msg_ptr = message.data_ptr<float>();
  const __half* down_ptr = reinterpret_cast<const __half*>(down.data_ptr<at::Half>());
  const __half* up_ptr = reinterpret_cast<const __half*>(up.data_ptr<at::Half>());
  const __half* nw_ptr = reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>());
  __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  __half* sh_hidden = reinterpret_cast<__half*>(scratch_hidden.data_ptr<at::Half>());
  float* sh_x = scratch_x.data_ptr<float>();
  float* sh_ss = scratch_sum_sq.data_ptr<float>();
  int32_t* sh_done = scratch_done.data_ptr<int32_t>();
  const int64_t stride_dest_s = dest.stride(0);
  const int64_t stride_w_s = norm_weight.stride(0);

  auto run_r = [&](auto r_tag) {
    constexpr int RC = decltype(r_tag)::value;
    if (T == 1) {
      dispatch_loop_blend_typed<1, RC>(
          dest_ptr, msg_ptr, down_ptr, up_ptr, nw_ptr, out_ptr,
          sh_hidden, sh_x, sh_ss, sh_done, stride_dest_s, stride_w_s,
          S, T, D, (float)norm_eps, st);
    } else if (T == 2) {
      dispatch_loop_blend_typed<2, RC>(
          dest_ptr, msg_ptr, down_ptr, up_ptr, nw_ptr, out_ptr,
          sh_hidden, sh_x, sh_ss, sh_done, stride_dest_s, stride_w_s,
          S, T, D, (float)norm_eps, st);
    } else {
      dispatch_loop_blend_typed<4, RC>(
          dest_ptr, msg_ptr, down_ptr, up_ptr, nw_ptr, out_ptr,
          sh_hidden, sh_x, sh_ss, sh_done, stride_dest_s, stride_w_s,
          S, T, D, (float)norm_eps, st);
    }
  };

  if (R == 64) {
    run_r(std::integral_constant<int, 64>{});
  } else {
    run_r(std::integral_constant<int, 128>{});
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("loop_blend_sm70", &loop_blend_sm70,
        "Fused 2-kernel Volta (sm_70) Inkling2 loop message controller blend (R=64, 128)");
}
