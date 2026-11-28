// SPDX-License-Identifier: Apache-2.0
// dfly2_ops.cu - Zero-spill Volta (sm_70) fused CUDA kernels for Inkling-2-Mini DFly2-KV
// (block_size=8, gamma=7, 0 B draft KV cache, TP4: H=2304, Hq_local=6, Hkv_local=2, D=128).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int BLK = 8;
constexpr int H_DIM = 2304;
constexpr int HQ_L = 6;
constexpr int HKV_L = 2;
constexpr int G_RATIO = 3;
constexpr int HD = 128;
constexpr int D_REL = 16;
constexpr int E_REL = 1024;
constexpr int LAT_D = 256;
constexpr int HALF_INTRA = 15;   // intra_rel_bias: [6, 31], center = 15
constexpr int MAX_BLK_SLOTS = 16;
constexpr int TILE = 32;
constexpr int QS = HD + 8;
constexpr int QS_GLA = LAT_D + 8;
constexpr int PS = TILE + 8;
constexpr int TBS = 256;

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float Q_CLIP = 22.627416997969522f;
constexpr float R_CLIP = 8.0f;
constexpr float LOG_FLOOR = 8192.0f;
constexpr float LOG_ALPHA = 0.1f;

__device__ __forceinline__ void mma_rc(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_rr(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

template <int M_WARPS, int T_WARPS>
__device__ __forceinline__ void bar_tw_sync(int tw) {
  if constexpr (M_WARPS > 1) {
    if constexpr (T_WARPS == 1) {
      __syncthreads();
    } else {
      if (tw == 0) {
        asm volatile("bar.sync 1, %0;" ::"r"(32 * M_WARPS) : "memory");
      } else {
        asm volatile("bar.sync 2, %0;" ::"r"(32 * M_WARPS) : "memory");
      }
    }
  } else {
    __syncwarp();
  }
}

__device__ __forceinline__ void e4m3x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned x0 = __byte_perm(w, 0u, 0x1404);
  const unsigned x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}

__device__ __forceinline__ float ex2_fast(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
  return v;
}

__device__ __forceinline__ float block_sum_256(float v) {
  __shared__ float sm[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  float tot = (lane < 8) ? sm[lane] : 0.f;
  if (warp == 0) {
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, d);
    if (lane == 0) sm[0] = tot;
  }
  __syncthreads();
  return sm[0];
}

__device__ __forceinline__ float compute_tau(int pos) {
  if ((float)pos <= LOG_FLOOR) return 1.0f;
  return 1.0f + LOG_ALPHA * logf((float)pos / LOG_FLOOR);
}

// ============================================================================
// 1. Zero-Sync Input Extraction
// ============================================================================
template <bool POS64, bool LS64, bool NP64>
__global__ void __launch_bounds__(256)
k_extract_inputs(
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ idx_mapping,
    const void* __restrict__ positions,
    const __half* __restrict__ last_hidden_states,
    const int32_t* __restrict__ num_sampled,
    const int32_t* __restrict__ num_rejected,
    const void* __restrict__ last_sampled,
    const void* __restrict__ next_prefill_tokens,
    int32_t* __restrict__ anchor_tok,
    int32_t* __restrict__ ctx_len,
    int32_t* __restrict__ anchor_pos,
    __half* __restrict__ prev_hidden,
    int32_t* __restrict__ req_slots,
    int num_reqs,
    int b_pad,
    int max_model_len) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  if (b >= b_pad) return;

  __shared__ int s_row;
  if (tid == 0) {
    if (b < num_reqs) {
      const int rslot = idx_mapping[b];
      const int q_start = query_start_loc[b];
      const int q_end = query_start_loc[b + 1];
      const int n_rej = num_rejected[b];
      const int n_samp = num_sampled[b];
      int row = q_end - 1 - n_rej;
      if (row < q_start) row = q_start;
      if (row >= q_end) row = q_end - 1;
      s_row = row;

      int cl = seq_lens[b] - n_rej;
      if (cl < 0) cl = 0;
      if (cl > max_model_len) cl = max_model_len;
      const int prev_pos = POS64
          ? (int)reinterpret_cast<const int64_t*>(positions)[row]
          : reinterpret_cast<const int32_t*>(positions)[row];
      int apos = prev_pos + 1;
      if (apos < 0) apos = 0;
      if (apos >= max_model_len) apos = max_model_len - 1;
      ctx_len[b] = cl;
      anchor_pos[b] = apos;
      int tok;
      if (n_samp > 0) {
        tok = LS64 ? (int)reinterpret_cast<const int64_t*>(last_sampled)[rslot]
                   : reinterpret_cast<const int32_t*>(last_sampled)[rslot];
      } else {
        tok = NP64 ? (int)reinterpret_cast<const int64_t*>(next_prefill_tokens)[rslot]
                   : reinterpret_cast<const int32_t*>(next_prefill_tokens)[rslot];
      }
      anchor_tok[b] = tok;
      req_slots[b] = rslot;
    } else {
      s_row = -1;
      ctx_len[b] = 0;
      anchor_pos[b] = 0;
      anchor_tok[b] = 0;
      req_slots[b] = -1;
    }
  }
  __syncthreads();

  const int row = s_row;
  __half* dst = prev_hidden + (size_t)b * H_DIM;
  if (row >= 0) {
    const __half* src = last_hidden_states + (size_t)row * H_DIM;
    for (int i = tid; i < H_DIM / 8; i += 256) {
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
    }
  } else {
    for (int i = tid; i < H_DIM / 8; i += 256) {
      reinterpret_cast<uint4*>(dst)[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

// ============================================================================
// 2. Stage A: MTPFusion Prep (TP4-sharded) + Block Init & Layer 0 Input LN
// ============================================================================
// Reference: cat([enorm(e0), hnorm(prev_hidden)], dim=-1) -> [B, 4608].
// TP4 shard r in 0..3 owns cols [r*1152 .. (r+1)*1152]:
//   r==0 -> enorm(e0)[0:1152],     r==1 -> enorm(e0)[1152:2304]
//   r==2 -> hnorm(prev)[0:1152],   r==3 -> hnorm(prev)[1152:2304]
__global__ void __launch_bounds__(256)
k_stage_a_prep(
    const __half* __restrict__ e0,
    const __half* __restrict__ prev_hidden,
    const __half* __restrict__ enorm_w,
    const __half* __restrict__ hnorm_w,
    __half* __restrict__ eh_in_local,
    int tp_rank,
    float eps) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const bool use_h = (tp_rank >= 2);
  const int col_off = (tp_rank & 1) * (H_DIM / 2);
  const __half* src = (use_h ? prev_hidden : e0) + (size_t)b * H_DIM;
  const __half* w = use_h ? hnorm_w : enorm_w;

  float ss = 0.f;
  for (int i = tid; i < H_DIM; i += 256) {
    const float v = __half2float(src[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum_256(ss) * (1.0f / H_DIM) + eps);

  __half* dst = eh_in_local + (size_t)b * (H_DIM / 2);
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const int d = col_off + i;
    dst[i] = __float2half_rn(__half2float(src[d]) * inv * __half2float(w[d]));
  }
}

__global__ void __launch_bounds__(256)
k_stage_a_fuse_norm(
    const __half* __restrict__ e0,
    const __half* __restrict__ eh_red,
    const __half* __restrict__ mtp_gate,
    const __half* __restrict__ mask_emb,
    const __half* __restrict__ ln0_w,
    float* __restrict__ res_f32,
    __half* __restrict__ x_norm,
    float eps) {
  const int tok = blockIdx.x;
  const int b = tok / BLK;
  const int slot = tok - b * BLK;
  const int tid = threadIdx.x;

  float* rdst = res_f32 + (size_t)tok * H_DIM;
  float ss = 0.f;
  if (slot == 0) {
    const __half* ep = e0 + (size_t)b * H_DIM;
    const __half* ehp = eh_red + (size_t)b * H_DIM;
    for (int i = tid; i < H_DIM; i += 256) {
      const float v = __half2float(ep[i]) + __half2float(ehp[i]) * __half2float(mtp_gate[i]);
      rdst[i] = v;
      ss += v * v;
    }
  } else {
    for (int i = tid; i < H_DIM; i += 256) {
      const float v = __half2float(mask_emb[i]);
      rdst[i] = v;
      ss += v * v;
    }
  }
  const float inv = rsqrtf(block_sum_256(ss) * (1.0f / H_DIM) + eps);
  __half* xdst = x_norm + (size_t)tok * H_DIM;
  for (int i = tid; i < H_DIM; i += 256) {
    xdst[i] = __float2half_rn(rdst[i] * inv * __half2float(ln0_w[i]));
  }
}

// ============================================================================
// 3. Standard Inkling2 Residual Add (FP32) + Standard RMSNorm (w * inv * x)
// ============================================================================
__global__ void __launch_bounds__(256)
k_rmsnorm_res(
    const __half* __restrict__ delta,
    float* __restrict__ res_f32,
    const __half* __restrict__ norm_w,
    __half* __restrict__ x_norm,
    float eps) {
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  const __half* dp = delta + (size_t)tok * H_DIM;
  float* rp = res_f32 + (size_t)tok * H_DIM;

  float ss = 0.f;
  for (int i = tid; i < H_DIM; i += 256) {
    const float v = rp[i] + __half2float(dp[i]);
    rp[i] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum_256(ss) * (1.0f / H_DIM) + eps);
  __half* xp = x_norm + (size_t)tok * H_DIM;
  for (int i = tid; i < H_DIM; i += 256) {
    xp[i] = __float2half_rn(rp[i] * inv * __half2float(norm_w[i]));
  }
}

// ============================================================================
// 4. Stage B: Fused Q/R/K L2-Clip + V Unit-RMSNorm (+ optional tau on L1)
// ============================================================================
// qrgkv row layout (stride = qrgkv_stride):
//   [q: 0..767 | r: 768..863 | gate: 864..1631 | k: 1632..1887 | v: 1888..2143]
__global__ void __launch_bounds__(256)
k_stage_b_qkrv_prep(
    const __half* __restrict__ qrgkv,
    const int32_t* __restrict__ anchor_pos,
    const __half* __restrict__ q_scale,   // [6, 128]
    const __half* __restrict__ r_scale,   // [6, 16]
    __half* __restrict__ q_out,           // [N8, 6, 128]
    __half* __restrict__ r_out,           // [N8, 6, 16]
    __half* __restrict__ k_blk_out,       // [B, 8, 2, 128]
    __half* __restrict__ v_blk_out,       // [B, 8, 2, 128]
    int qrgkv_stride,
    bool apply_tau,
    float eps) {
  const int tok = blockIdx.x;
  const int b = tok / BLK;
  const int slot = tok - b * BLK;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const __half* row = qrgkv + (size_t)tok * qrgkv_stride;
  const float tau = apply_tau ? compute_tau(anchor_pos[b] + slot) : 1.0f;

  if (warp < HQ_L) {
    const int h = warp;
    const __half* qp = row + h * HD;
    const __half* qsp = q_scale + h * HD;
    float qv[4];
    float sq = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int d = lane * 4 + u;
      qv[u] = __half2float(qp[d]) * __half2float(qsp[d]);
      sq += qv[u] * qv[u];
    }
    const float qnorm = sqrtf(warp_sum(sq) + eps);
    const float qfac = ((qnorm > Q_CLIP) ? (Q_CLIP / qnorm) : 1.0f) * tau;
    __half* qd = q_out + ((size_t)tok * HQ_L + h) * HD;
#pragma unroll
    for (int u = 0; u < 4; u++) qd[lane * 4 + u] = __float2half_rn(qv[u] * qfac);

    float rv = 0.f;
    if (lane < D_REL) {
      rv = __half2float(row[768 + h * D_REL + lane]) * __half2float(r_scale[h * D_REL + lane]);
    }
    const float rnorm = sqrtf(warp_sum(rv * rv) + eps);
    const float rfac = ((rnorm > R_CLIP) ? (R_CLIP / rnorm) : 1.0f) * tau;
    if (lane < D_REL) {
      r_out[((size_t)tok * HQ_L + h) * D_REL + lane] = __float2half_rn(rv * rfac);
    }
  } else {
    const int hkv = warp - HQ_L;
    const __half* kp = row + 1632 + hkv * HD;
    float kv[4];
    float sk = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      kv[u] = __half2float(kp[lane * 4 + u]);
      sk += kv[u] * kv[u];
    }
    const float knorm = sqrtf(warp_sum(sk) + eps);
    const float kfac = (knorm > Q_CLIP) ? (Q_CLIP / knorm) : 1.0f;
    __half* kd = k_blk_out + (((size_t)b * BLK + slot) * HKV_L + hkv) * HD;
#pragma unroll
    for (int u = 0; u < 4; u++) kd[lane * 4 + u] = __float2half_rn(kv[u] * kfac);

    const __half* vp = row + 1888 + hkv * HD;
    float vv[4];
    float sv = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      vv[u] = __half2float(vp[lane * 4 + u]);
      sv += vv[u] * vv[u];
    }
    const float vinv = rsqrtf(warp_sum(sv) * (1.0f / HD) + eps);
    __half* vd = v_blk_out + (((size_t)b * BLK + slot) * HKV_L + hkv) * HD;
#pragma unroll
    for (int u = 0; u < 4; u++) vd[lane * 4 + u] = __float2half_rn(vv[u] * vinv);
  }
}

// ============================================================================
// 5. Shared-Target-KV Context Attention (Volta HMMA.884, FP16 & FP8 E4M3FN)
// ============================================================================
template <int M_WARPS, int T_WARPS>
struct CtxSM {
  static constexpr int NT = 32 * M_WARPS * T_WARPS;
  static constexpr int R_CTA = 8 * M_WARPS;
  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS * (int)sizeof(int);
  static constexpr int RP_OFF = TB_BYTES;
  static constexpr int RP_BYTES = ((R_CTA * (int)sizeof(const __half*) + 15) & ~15);
  static constexpr int Q_OFF = RP_OFF + RP_BYTES;
  static constexpr int Q_BYTES = R_CTA * QS * (int)sizeof(__half);
  static constexpr int P_OFF = (Q_OFF + Q_BYTES + 15) & ~15;
  static constexpr int P_BYTES = M_WARPS * T_WARPS * 8 * PS * (int)sizeof(__half);
  static constexpr int MAIN_END = (P_OFF + P_BYTES + 15) & ~15;
  static constexpr int O_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * HD * (int)sizeof(float)) : 0;
  static constexpr int ML_OFF = (MAIN_END > O_BYTES ? MAIN_END : O_BYTES);
  static constexpr int ML_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int RS_OFF = ML_OFF + ML_BYTES;
  static constexpr int RS_BYTES = (T_WARPS > 1) ? (M_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int TOTAL = ((RS_OFF + RS_BYTES + 15) & ~15);
};

template <int M_WARPS, int T_WARPS>
__device__ __forceinline__ void write_empty_ctx_split(
    float2* __restrict__ ml_part, int num_splits, int s, int h, int b, int Q, int eff_nr, int tid) {
  constexpr int NT = 32 * M_WARPS * T_WARPS;
  constexpr int R_CTA = 8 * M_WARPS;
  for (int r = tid; r < R_CTA; r += NT) {
    if (r < eff_nr) {
      const int tok = r / G_RATIO;
      const int head = h * G_RATIO + (r % G_RATIO);
      const int token = b * Q + tok;
      const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
      __stcg(ml_part + slot, make_float2(-INFINITY, 0.f));
    }
  }
}

__device__ __forceinline__ void apply_dfly2_bias_mask_row(
    float& s0, float& s1, float& s4, float& s5,
    const __half* __restrict__ rr,
    bool valid,
    int pos,
    int tok0,
    int cb,
    int seqlen_k,
    int window,
    float sl2) {
  if (!valid) {
    s0 = -INFINITY; s1 = -INFINITY; s4 = -INFINITY; s5 = -INFINITY;
    return;
  }
  const int k_min_ext = pos - E_REL + 1;
  const int k_min_vis = (window > 0) ? (pos - window + 1) : 0;
  if (tok0 + 31 < k_min_ext && tok0 >= k_min_vis && tok0 + 31 < seqlen_k) {
    s0 *= sl2; s1 *= sl2; s4 *= sl2; s5 *= sl2;
  } else if (tok0 >= k_min_ext && tok0 >= k_min_vis && tok0 + 31 < seqlen_k && tok0 + 31 <= pos) {
    const int d0 = pos - cb;
    s0 = fmaf(__half2float(rr[d0]), LOG2E, s0 * sl2);
    s1 = fmaf(__half2float(rr[d0 - 1]), LOG2E, s1 * sl2);
    s4 = fmaf(__half2float(rr[d0 - 4]), LOG2E, s4 * sl2);
    s5 = fmaf(__half2float(rr[d0 - 5]), LOG2E, s5 * sl2);
  } else {
    float* sp[4] = {&s0, &s1, &s4, &s5};
    const int offs[4] = {0, 1, 4, 5};
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int col = cb + offs[u];
      const int dist = pos - col;
      if (col < seqlen_k && dist >= 0 && (window == 0 || dist < window)) {
        float v = (*sp[u]) * sl2;
        if (dist < E_REL) v = fmaf(__half2float(rr[dist]), LOG2E, v);
        *sp[u] = v;
      } else {
        *sp[u] = -INFINITY;
      }
    }
  }
}

template <bool IS_FP8, int M_WARPS, int T_WARPS>
__global__ void __launch_bounds__(32 * M_WARPS * T_WARPS)
k_dfly2_ctx_attn(
    const __half* __restrict__ q,          // [B*Q, 6, 128]
    const void* __restrict__ kc_raw,       // [NB, page, 2, 128]
    const void* __restrict__ vc_raw,       // [NB, page, 2, 128]
    const __half* __restrict__ rel_all,    // [B*Q, 6, 1024]
    __half* __restrict__ out,              // [B*Q, 6, 128] (when S==1)
    float* __restrict__ lse_out,           // [B*Q, 6] optional (when S==1)
    float* __restrict__ o_part,            // [B*Q, 6, S, 128] (when S>1)
    float2* __restrict__ ml_part,          // [B*Q, 6, S] (when S>1)
    const int* __restrict__ block_table,
    const int* __restrict__ ctx_len_arr,
    const int* __restrict__ anchor_pos_arr,
    int Q,
    int slot_fixed,
    int window,
    int page,
    int page_shift,
    int max_blocks,
    int num_splits,
    int64_t kb,
    int64_t kt,
    int64_t kh,
    int64_t vb,
    int64_t vt,
    int64_t vh,
    int64_t stride_bt,
    float sl2,
    float vsc) {
  using L_ = CtxSM<M_WARPS, T_WARPS>;
  constexpr int NT = L_::NT;
  constexpr int R_CTA = L_::R_CTA;
  extern __shared__ __align__(16) uint8_t sm[];

  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  __half* Qs = reinterpret_cast<__half*>(sm + L_::Q_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);

  const int s = blockIdx.x;
  const int h = blockIdx.y;  // 0..1
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mw = warp / T_WARPS;
  const int tw = warp % T_WARPS;

  const int eff_nr = Q * G_RATIO;  // 24 (Stage B) or 3 (Stage C)
  const int seqlen_k = ctx_len_arr[b];
  const int anc = anchor_pos_arr[b];
  const int slot_lo = (Q == 1) ? slot_fixed : 0;
  const int pos_lo = anc + slot_lo;

  const int k_lo = (window > 0) ? max(pos_lo - (window - 1), 0) : 0;
  const int k_hi = seqlen_k;
  const int t_lo = k_lo / TILE;
  const int t_hi = (k_hi > k_lo) ? ((k_hi + TILE - 1) / TILE) : t_lo;
  const int T_tiles = max(t_hi - t_lo, 0);

  if (T_tiles <= 0) {
    if (num_splits == 1) {
      for (int i = tid; i < eff_nr * (HD / 2); i += NT) {
        const int r = i / (HD / 2);
        const int tok = r / G_RATIO;
        const int head = h * G_RATIO + (r % G_RATIO);
        const int token = b * Q + tok;
        const int c2 = i % (HD / 2);
        reinterpret_cast<__half2*>(out + ((size_t)token * HQ_L + head) * HD)[c2] =
            __floats2half2_rn(0.f, 0.f);
      }
      if (lse_out != nullptr && tid < eff_nr) {
        const int tok = tid / G_RATIO;
        const int head = h * G_RATIO + (tid % G_RATIO);
        lse_out[(size_t)(b * Q + tok) * HQ_L + head] = -INFINITY;
      }
    } else {
      write_empty_ctx_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, h, b, Q, eff_nr, tid);
    }
    return;
  }

  const int Sb = max(1, min(num_splits, T_tiles));
  if (s >= Sb) {
    write_empty_ctx_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, h, b, Q, eff_nr, tid);
    return;
  }

  const int t_start = t_lo + (int)((long long)T_tiles * s / Sb);
  const int t_end = t_lo + (int)((long long)T_tiles * (s + 1) / Sb);

  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  for (int i = tid; i < min(max_blocks, TBS); i += NT) Ts[i] = tb[i];
  for (int r = tid; r < R_CTA; r += NT) {
    if (r < eff_nr) {
      const int tok = r / G_RATIO;
      const int head = h * G_RATIO + (r % G_RATIO);
      const int token = b * Q + tok;
      RelPtrs[r] = rel_all + ((size_t)token * HQ_L + head) * E_REL;
    } else {
      RelPtrs[r] = rel_all;
    }
  }
  for (int i = tid; i < R_CTA * (HD / 8); i += NT) {
    const int r = i / (HD / 8);
    const int c = (i % (HD / 8)) * 8;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < eff_nr) {
      const int tok = r / G_RATIO;
      const int head = h * G_RATIO + (r % G_RATIO);
      const int token = b * Q + tok;
      v = ldg16(q + ((size_t)token * HQ_L + head) * HD + c);
    }
    *reinterpret_cast<uint4*>(Qs + r * QS + c) = v;
  }
  __syncthreads();

  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;
  const int r0 = (lane & 1) + 4 * hi;
  const int c0 = lane & 2;
  const int ktok = 8 * qp + arow;
  const int vdim = 32 * qp + 16 * hi;

  const int lrA = mw * 8 + r0;
  const int lrB = lrA + 2;
  const bool validA = lrA < eff_nr;
  const bool validB = lrB < eff_nr;
  const int posA = validA ? (anc + ((Q == 1) ? slot_fixed : (lrA / G_RATIO))) : -1;
  const int posB = validB ? (anc + ((Q == 1) ? slot_fixed : (lrB / G_RATIO))) : -1;

  const __half* qrow = Qs + (mw * 8 + arow) * QS;
  __half* Pw = Ps + warp * 8 * PS;
  const int pmask = page - 1;
  const int64_t kh_off = (int64_t)h * kh;
  const int64_t vh_off = (int64_t)h * vh;

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TBS ? Ts[pi] : tb[pi]);
  };

  constexpr int KR_N = IS_FP8 ? 8 : 16;
  uint4 kr[KR_N];
  uint4 vr[8];

  auto ldk = [&](int t) {
    const int tok0 = t * TILE;
    if (page_shift >= 4 && tok0 + 31 < seqlen_k) {
      const int pi0 = tok0 >> page_shift;
      const int s0 = tok0 & pmask;
      const int pi1 = (tok0 + 16) >> page_shift;
      const int s1 = (tok0 + 16) & pmask;
      const int64_t pg = (ktok < 16) ? get_pg(pi0) : get_pg(pi1);
      const int sl = (ktok < 16) ? (s0 + ktok) : (s1 + ktok - 16);
      if constexpr (IS_FP8) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
        for (int i = 0; i < 8; i++) kr[i] = ldg16(p + 16 * i);
      } else {
        const __half* p = reinterpret_cast<const __half*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
        for (int i = 0; i < 16; i++) kr[i] = ldg16(p + 8 * i);
      }
    } else {
      const int k_idx = tok0 + ktok;
      if (k_idx < seqlen_k) {
        const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
        const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
        const int64_t pg = get_pg(pi);
        if constexpr (IS_FP8) {
          const uint8_t* p = reinterpret_cast<const uint8_t*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
          for (int i = 0; i < 8; i++) kr[i] = ldg16(p + 16 * i);
        } else {
          const __half* p = reinterpret_cast<const __half*>(kc_raw) + pg * kb + (int64_t)sl * kt + kh_off;
#pragma unroll
          for (int i = 0; i < 16; i++) kr[i] = ldg16(p + 8 * i);
        }
      } else {
#pragma unroll
        for (int i = 0; i < KR_N; i++) kr[i] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  auto ldv_fp8 = [&](int t) {
    const int tok0 = t * TILE;
    const uint8_t* vc = reinterpret_cast<const uint8_t*>(vc_raw);
    if (page_shift >= 4 && tok0 + 31 < seqlen_k) {
      const int pi0 = tok0 >> page_shift;
      const int s0 = tok0 & pmask;
      const int pi1 = (tok0 + 16) >> page_shift;
      const int s1 = (tok0 + 16) & pmask;
      const int64_t pg0 = get_pg(pi0);
      const int64_t pg1 = (pi1 == pi0) ? pg0 : get_pg(pi1);
      const uint8_t* p0 = vc + pg0 * vb + (int64_t)(s0 + l3) * vt + vh_off + vdim;
      const uint8_t* p1 = vc + pg1 * vb + (int64_t)(s1 + l3) * vt + vh_off + vdim;
#pragma unroll
      for (int i = 0; i < 4; i++) vr[i] = ldg16(p0 + (int64_t)(4 * i) * vt);
#pragma unroll
      for (int i = 0; i < 4; i++) vr[4 + i] = ldg16(p1 + (int64_t)(4 * i) * vt);
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int v_idx = tok0 + l3 + 4 * i;
        if (v_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (v_idx >> page_shift) : (v_idx / page);
          const int sl = (page_shift >= 0) ? (v_idx & pmask) : (v_idx - pi * page);
          const int64_t pg = get_pg(pi);
          vr[i] = ldg16(vc + pg * vb + (int64_t)sl * vt + vh_off + vdim);
        } else {
          vr[i] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  auto ldv_fp16_half = [&](int t, int half_idx) {
    const int tok0 = t * TILE + half_idx * 16;
    const __half* vc = reinterpret_cast<const __half*>(vc_raw);
    if (page_shift >= 4 && tok0 + 15 < seqlen_k) {
      const int pi = tok0 >> page_shift;
      const int sl = tok0 & pmask;
      const int64_t pg = get_pg(pi);
      const __half* p = vc + pg * vb + (int64_t)(sl + l3) * vt + vh_off + vdim;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const __half* qptr = p + (int64_t)(4 * i) * vt;
        vr[2 * i] = ldg16(qptr);
        vr[2 * i + 1] = ldg16(qptr + 8);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int v_idx = tok0 + l3 + 4 * i;
        if (v_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (v_idx >> page_shift) : (v_idx / page);
          const int sl = (page_shift >= 0) ? (v_idx & pmask) : (v_idx - pi * page);
          const int64_t pg = get_pg(pi);
          const __half* qptr = vc + pg * vb + (int64_t)sl * vt + vh_off + vdim;
          vr[2 * i] = ldg16(qptr);
          vr[2 * i + 1] = ldg16(qptr + 8);
        } else {
          vr[2 * i] = make_uint4(0u, 0u, 0u, 0u);
          vr[2 * i + 1] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  int t = t_start + tw;
  if (t < t_end) {
    ldk(t);
    if constexpr (IS_FP8) {
      ldv_fp8(t);
    } else {
      ldv_fp16_half(t, 0);
    }
  }

  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[j][e] = 0.f;
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;

  for (; t < t_end; t += T_WARPS) {
    const int tok0 = t * TILE;
    const bool more = (t + T_WARPS) < t_end;

    float sa[8];
#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] = 0.f;

    if constexpr (IS_FP8) {
      float sb_acc[8];
#pragma unroll
      for (int e = 0; e < 8; e++) sb_acc[e] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const unsigned w[4] = {kr[i].x, kr[i].y, kr[i].z, kr[i].w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint2 a = *reinterpret_cast<const uint2*>(qrow + 4 * (4 * i + j));
          unsigned b0, b1;
          e4m3x4(w[j], b0, b1);
          mma_rc((i & 1) ? sb_acc : sa, a.x, a.y, b0, b1);
        }
      }
      if (more) ldk(t + T_WARPS);
#pragma unroll
      for (int e = 0; e < 8; e++) sa[e] += sb_acc[e];
    } else {
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const uint2 a0 = *reinterpret_cast<const uint2*>(qrow + 8 * i);
        const uint2 a1 = *reinterpret_cast<const uint2*>(qrow + 8 * i + 4);
        mma_rc(sa, a0.x, a0.y, kr[i].x, kr[i].y);
        mma_rc(sa, a1.x, a1.y, kr[i].z, kr[i].w);
      }
      if (more) ldk(t + T_WARPS);
    }

    const int cb = tok0 + 8 * qp + c0;
    apply_dfly2_bias_mask_row(sa[0], sa[1], sa[4], sa[5], RelPtrs[lrA], validA, posA, tok0, cb, seqlen_k, window, sl2);
    apply_dfly2_bias_mask_row(sa[2], sa[3], sa[6], sa[7], RelPtrs[lrB], validB, posB, tok0, cb, seqlen_k, window, sl2);

    float x0 = fmaxf(fmaxf(sa[0], sa[1]), fmaxf(sa[4], sa[5]));
    float x2 = fmaxf(fmaxf(sa[2], sa[3]), fmaxf(sa[6], sa[7]));
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
      x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
    }
    const float n0 = fmaxf(m0, x0);
    const float n2 = fmaxf(m2, x2);
    const float a0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - n0);
    const float a2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - n2);
    const float z0 = (n0 == -INFINITY) ? 0.f : n0;
    const float z2 = (n2 == -INFINITY) ? 0.f : n2;

#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] = ex2_fast(sa[e] - ((e & 2) ? z2 : z0));
    l0 = l0 * a0 + (sa[0] + sa[1]) + (sa[4] + sa[5]);
    l2 = l2 * a2 + (sa[2] + sa[3]) + (sa[6] + sa[7]);
    m0 = n0;
    m2 = n2;

#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[j][e] *= (e & 2) ? a2 : a0;

    const int pc = 8 * qp + c0;
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(sa[0], sa[1]);
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(sa[4], sa[5]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(sa[2], sa[3]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(sa[6], sa[7]);
    __syncwarp();

    if constexpr (IS_FP8) {
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
      if (more) ldv_fp8(t + T_WARPS);
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
        mma_rr(o[0], a.x, a.y, vr[2 * i].x, vr[2 * i].y);
        mma_rr(o[1], a.x, a.y, vr[2 * i].z, vr[2 * i].w);
        mma_rr(o[2], a.x, a.y, vr[2 * i + 1].x, vr[2 * i + 1].y);
        mma_rr(o[3], a.x, a.y, vr[2 * i + 1].z, vr[2 * i + 1].w);
      }
      ldv_fp16_half(t, 1);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * (4 + i));
        mma_rr(o[0], a.x, a.y, vr[2 * i].x, vr[2 * i].y);
        mma_rr(o[1], a.x, a.y, vr[2 * i].z, vr[2 * i].w);
        mma_rr(o[2], a.x, a.y, vr[2 * i + 1].x, vr[2 * i + 1].y);
        mma_rr(o[3], a.x, a.y, vr[2 * i + 1].z, vr[2 * i + 1].w);
      }
      if (more) ldv_fp16_half(t + T_WARPS, 0);
    }
    bar_tw_sync<M_WARPS, T_WARPS>(tw);
  }

#pragma unroll
  for (int d = 2; d <= 8; d <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, d);
    l2 += __shfl_xor_sync(0xffffffffu, l2, d);
  }

  if constexpr (T_WARPS == 1) {
    if (num_splits == 1) {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int r = sr ? lrB : lrA;
        if (r < eff_nr) {
          const float m_v = sr ? m2 : m0;
          const float l_v = sr ? l2 : l0;
          const float rsc = (l_v > 0.f) ? (vsc / l_v) : 0.f;
          const int tok = r / G_RATIO;
          const int head = h * G_RATIO + (r % G_RATIO);
          const int token = b * Q + tok;
          __half* ob = out + ((size_t)token * HQ_L + head) * HD;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            *reinterpret_cast<__half2*>(ob + dlo) =
                __floats2half2_rn(o[j][2 * sr] * rsc, o[j][2 * sr + 1] * rsc);
            *reinterpret_cast<__half2*>(ob + dlo + 16) =
                __floats2half2_rn(o[j][4 + 2 * sr] * rsc, o[j][5 + 2 * sr] * rsc);
          }
          if (lse_out != nullptr && (lane & 14) == 0) {
            lse_out[(size_t)token * HQ_L + head] = (l_v > 0.f) ? ((m_v + log2f(l_v)) * LN2) : -INFINITY;
          }
        }
      }
    } else {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int r = sr ? lrB : lrA;
        if (r < eff_nr) {
          const int tok = r / G_RATIO;
          const int head = h * G_RATIO + (r % G_RATIO);
          const int token = b * Q + tok;
          const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
          float* pob = o_part + slot * HD;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            __stcg(reinterpret_cast<float2*>(pob + dlo), make_float2(o[j][2 * sr], o[j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + dlo + 16), make_float2(o[j][4 + 2 * sr], o[j][5 + 2 * sr]));
          }
          if ((lane & 14) == 0) {
            __stcg(ml_part + slot, make_float2(sr ? m2 : m0, sr ? l2 : l0));
          }
        }
      }
    }
  } else {
    float* Os = reinterpret_cast<float*>(sm);
    float* MLs = reinterpret_cast<float*>(sm + L_::ML_OFF);
    float* RSs = reinterpret_cast<float*>(sm + L_::RS_OFF);
    __syncthreads();
    if ((lane & 14) == 0) {
      const int ml_base = (mw * T_WARPS + tw) * 16;
      MLs[ml_base + 2 * r0] = m0;
      MLs[ml_base + 2 * r0 + 1] = l0;
      MLs[ml_base + 2 * (r0 + 2)] = m2;
      MLs[ml_base + 2 * (r0 + 2) + 1] = l2;
    }
    __syncthreads();

    float M0 = -INFINITY, M2 = -INFINITY;
#pragma unroll
    for (int k = 0; k < T_WARPS; k++) {
      M0 = fmaxf(M0, MLs[(mw * T_WARPS + k) * 16 + 2 * r0]);
      M2 = fmaxf(M2, MLs[(mw * T_WARPS + k) * 16 + 2 * (r0 + 2)]);
    }
    const float f0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - M0);
    const float f2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - M2);
    float* Op = Os + (size_t)(mw * T_WARPS + tw) * 8 * HD;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int local_r = r0 + 2 * sr;
        const float f = sr ? f2 : f0;
        const int dlo = 32 * qp + 4 * j + c0;
        *reinterpret_cast<float2*>(Op + local_r * HD + dlo) =
            make_float2(o[j][2 * sr] * f, o[j][2 * sr + 1] * f);
        *reinterpret_cast<float2*>(Op + local_r * HD + dlo + 16) =
            make_float2(o[j][4 + 2 * sr] * f, o[j][5 + 2 * sr] * f);
      }
    }

    for (int r = tid; r < R_CTA; r += NT) {
      const int rm = r >> 3;
      const int lr = r & 7;
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) M = fmaxf(M, MLs[(rm * T_WARPS + k) * 16 + 2 * lr]);
      float Lt = 0.f;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) {
        const float mk = MLs[(rm * T_WARPS + k) * 16 + 2 * lr];
        if (mk != -INFINITY) Lt += MLs[(rm * T_WARPS + k) * 16 + 2 * lr + 1] * ex2_fast(mk - M);
      }
      RSs[2 * r] = M;
      RSs[2 * r + 1] = Lt;
    }
    __syncthreads();

    for (int i = tid; i < R_CTA * (HD / 4); i += NT) {
      const int r = i / (HD / 4);
      if (r >= eff_nr) continue;
      const int rm = r >> 3;
      const int lr = r & 7;
      const int d = (i % (HD / 4)) * 4;
      float4 acc = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS) * 8 * HD + lr * HD + d);
#pragma unroll
      for (int k = 1; k < T_WARPS; k++) {
        const float4 c = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS + k) * 8 * HD + lr * HD + d);
        acc.x += c.x; acc.y += c.y; acc.z += c.z; acc.w += c.w;
      }
      const int tok = r / G_RATIO;
      const int head = h * G_RATIO + (r % G_RATIO);
      const int token = b * Q + tok;
      if (num_splits == 1) {
        const float Lt = RSs[2 * r + 1];
        const float rsc = (Lt > 0.f) ? (vsc / Lt) : 0.f;
        __half2* dst = reinterpret_cast<__half2*>(out + ((size_t)token * HQ_L + head) * HD + d);
        dst[0] = __floats2half2_rn(acc.x * rsc, acc.y * rsc);
        dst[1] = __floats2half2_rn(acc.z * rsc, acc.w * rsc);
      } else {
        const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
        __stcg(reinterpret_cast<float4*>(o_part + slot * HD + d), acc);
      }
    }
    if (num_splits == 1 && lse_out != nullptr) {
      for (int r = tid; r < eff_nr; r += NT) {
        const int tok = r / G_RATIO;
        const int head = h * G_RATIO + (r % G_RATIO);
        const int token = b * Q + tok;
        const float M = RSs[2 * r];
        const float Lt = RSs[2 * r + 1];
        lse_out[(size_t)token * HQ_L + head] = (Lt > 0.f) ? ((M + log2f(Lt)) * LN2) : -INFINITY;
      }
    } else if (num_splits > 1) {
      for (int r = tid; r < eff_nr; r += NT) {
        const int tok = r / G_RATIO;
        const int head = h * G_RATIO + (r % G_RATIO);
        const int token = b * Q + tok;
        const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
        __stcg(ml_part + slot, make_float2(RSs[2 * r], RSs[2 * r + 1]));
      }
    }
  }
}

__global__ void __launch_bounds__(32)
k_dfly2_ctx_combine(
    const float* __restrict__ o_part,
    const float2* __restrict__ ml_part,
    __half* __restrict__ out,
    float* __restrict__ lse_out,
    int num_splits,
    float vsc) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const size_t base_slot = ((size_t)token * HQ_L + head) * (size_t)num_splits;
  const float2* mlp = ml_part + base_slot;
  const float* op = o_part + base_slot * HD;

  float M = -INFINITY;
  for (int s = lane; s < num_splits; s += 32) M = fmaxf(M, __ldcg(mlp + s).x);
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float L_tot = 0.f;
  for (int sb = 0; sb < num_splits; sb += 32) {
    const int idx = sb + lane;
    const float2 ml = (idx < num_splits) ? __ldcg(mlp + idx) : make_float2(-INFINITY, 0.f);
    const float w_lane = (ml.x == -INFINITY) ? 0.f : ex2_fast(ml.x - M);
    L_tot += ml.y * w_lane;
    const int n_chunk = min(32, num_splits - sb);
    for (int u = 0; u < n_chunk; u++) {
      const float ws = __shfl_sync(0xffffffffu, w_lane, u);
      if (ws > 0.f) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (size_t)(sb + u) * HD + lane * 4));
        acc.x = fmaf(ws, v.x, acc.x);
        acc.y = fmaf(ws, v.y, acc.y);
        acc.z = fmaf(ws, v.z, acc.z);
        acc.w = fmaf(ws, v.w, acc.w);
      }
    }
  }
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) L_tot += __shfl_xor_sync(0xffffffffu, L_tot, d);

  const float rsc = (L_tot > 0.f) ? (vsc / L_tot) : 0.f;
  __half2* dst = reinterpret_cast<__half2*>(out + ((size_t)token * HQ_L + head) * HD + lane * 4);
  dst[0] = __floats2half2_rn(acc.x * rsc, acc.y * rsc);
  dst[1] = __floats2half2_rn(acc.z * rsc, acc.w * rsc);
  if (lse_out != nullptr && lane == 0) {
    lse_out[(size_t)token * HQ_L + head] = (L_tot > 0.f) ? ((M + log2f(L_tot)) * LN2) : -INFINITY;
  }
}

// ============================================================================
// 6. Zero-Retrain GLA-g4 (r=256 FP8) Shared Context Attention for L23
// ============================================================================
template <int M_WARPS, int T_WARPS>
struct GlaCtxSM {
  static constexpr int NT = 32 * M_WARPS * T_WARPS;
  static constexpr int R_CTA = 8 * M_WARPS;
  static constexpr int TB_OFF = 0;
  static constexpr int TB_BYTES = TBS * (int)sizeof(int);
  static constexpr int RP_OFF = TB_BYTES;
  static constexpr int RP_BYTES = ((R_CTA * (int)sizeof(const __half*) + 15) & ~15);
  static constexpr int Q_OFF = RP_OFF + RP_BYTES;
  static constexpr int Q_BYTES = R_CTA * QS_GLA * (int)sizeof(__half);
  static constexpr int P_OFF = (Q_OFF + Q_BYTES + 15) & ~15;
  static constexpr int P_BYTES = M_WARPS * T_WARPS * 8 * PS * (int)sizeof(__half);
  static constexpr int MAIN_END = (P_OFF + P_BYTES + 15) & ~15;
  static constexpr int O_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * HD * (int)sizeof(float)) : 0;
  static constexpr int ML_OFF = (MAIN_END > O_BYTES ? MAIN_END : O_BYTES);
  static constexpr int ML_BYTES = (T_WARPS > 1) ? (M_WARPS * T_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int RS_OFF = ML_OFF + ML_BYTES;
  static constexpr int RS_BYTES = (T_WARPS > 1) ? (M_WARPS * 8 * 2 * (int)sizeof(float)) : 0;
  static constexpr int TOTAL = ((RS_OFF + RS_BYTES + 15) & ~15);
};

template <int M_WARPS, int T_WARPS>
__device__ __forceinline__ void write_empty_gla_ctx_split(
    float2* __restrict__ ml_part, int num_splits, int s, int b, int first_row, int eff_nr, int tid) {
  constexpr int NT = 32 * M_WARPS * T_WARPS;
  constexpr int R_CTA = 8 * M_WARPS;
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / HQ_L;
      const int head = gr - tok * HQ_L;
      const int token = b * BLK + tok;
      const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
      __stcg(ml_part + slot, make_float2(-INFINITY, 0.f));
    }
  }
}

template <int M_WARPS, int T_WARPS>
__global__ void __launch_bounds__(32 * M_WARPS * T_WARPS, (M_WARPS * T_WARPS <= 4) ? 2 : 1)
k_dfly2_gla_ctx_attn(
    const __half* __restrict__ q_lat,      // [2, B*8, 3, 256]
    const uint8_t* __restrict__ c_lo,      // [NB, page, 1, 128]
    const uint8_t* __restrict__ c_hi,      // [NB, page, 1, 128]
    const __half* __restrict__ rel_all,    // [B*8, 6, 1024]
    __half* __restrict__ o_lat,            // [2, B*8, 3, 256] (when S==1)
    float* __restrict__ lse_out,           // [B*8, 6] (when S==1)
    float* __restrict__ o_part,            // [B*8, 6, S, 256] (when S>1)
    float2* __restrict__ ml_part,          // [B*8, 6, S] (when S>1)
    const int* __restrict__ block_table,
    const int* __restrict__ ctx_len_arr,
    const int* __restrict__ anchor_pos_arr,
    int T_total,
    int page,
    int page_shift,
    int max_blocks,
    int num_splits,
    int64_t sb,
    int64_t st,
    int64_t stride_bt,
    float sl2,
    float vsc) {
  using L_ = GlaCtxSM<M_WARPS, T_WARPS>;
  constexpr int NT = L_::NT;
  constexpr int R_CTA = L_::R_CTA;
  extern __shared__ __align__(16) uint8_t sm[];

  int* Ts = reinterpret_cast<int*>(sm + L_::TB_OFF);
  const __half** RelPtrs = reinterpret_cast<const __half**>(sm + L_::RP_OFF);
  __half* Qs = reinterpret_cast<__half*>(sm + L_::Q_OFF);
  __half* Ps = reinterpret_cast<__half*>(sm + L_::P_OFF);

  const int s = blockIdx.x;
  const int m_tile = blockIdx.y;  // 0..1 (each covers 24 of 48 rows)
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mw = warp / T_WARPS;
  const int tw = warp % T_WARPS;

  const int eff_nr = BLK * HQ_L;  // 48
  const int first_row = m_tile * R_CTA;
  if (first_row >= eff_nr) return;

  const int seqlen_k = ctx_len_arr[b];
  const int anc = anchor_pos_arr[b];
  const int T_tiles = max((seqlen_k + TILE - 1) / TILE, 0);

  if (T_tiles <= 0) {
    if (num_splits == 1) {
      for (int i = tid; i < R_CTA * 32; i += NT) {
        const int r = i >> 5;
        const int c = i & 31;
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_L;
          const int head = gr - tok * HQ_L;
          const int g = head / G_RATIO;
          const int sub = head - g * G_RATIO;
          const int token = b * BLK + tok;
          __half* dst = o_lat + (((size_t)g * T_total + token) * G_RATIO + sub) * LAT_D;
          *reinterpret_cast<uint4*>(dst + c * 8) = make_uint4(0u, 0u, 0u, 0u);
        }
      }
      if (lse_out != nullptr && tid < R_CTA && first_row + tid < eff_nr) {
        const int gr = first_row + tid;
        const int tok = gr / HQ_L;
        const int head = gr - tok * HQ_L;
        lse_out[(size_t)(b * BLK + tok) * HQ_L + head] = -INFINITY;
      }
    } else {
      write_empty_gla_ctx_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, b, first_row, eff_nr, tid);
    }
    return;
  }

  const int Sb = max(1, min(num_splits, T_tiles));
  if (s >= Sb) {
    write_empty_gla_ctx_split<M_WARPS, T_WARPS>(ml_part, num_splits, s, b, first_row, eff_nr, tid);
    return;
  }

  const int t_start = (int)((long long)T_tiles * s / Sb);
  const int t_end = (int)((long long)T_tiles * (s + 1) / Sb);

  const int* tb = block_table + (size_t)b * (size_t)stride_bt;
  for (int i = tid; i < min(max_blocks, TBS); i += NT) Ts[i] = tb[i];
  for (int r = tid; r < R_CTA; r += NT) {
    const int gr = first_row + r;
    if (gr < eff_nr) {
      const int tok = gr / HQ_L;
      const int head = gr - tok * HQ_L;
      RelPtrs[r] = rel_all + ((size_t)(b * BLK + tok) * HQ_L + head) * E_REL;
    } else {
      RelPtrs[r] = rel_all;
    }
  }
  for (int i = tid; i < R_CTA * 32; i += NT) {
    const int r = i >> 5;
    const int c = (i & 31) * 8;
    const int gr = first_row + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (gr < eff_nr) {
      const int tok = gr / HQ_L;
      const int head = gr - tok * HQ_L;
      const int g = head / G_RATIO;
      const int sub = head - g * G_RATIO;
      const int token = b * BLK + tok;
      v = ldg16(q_lat + (((size_t)g * T_total + token) * G_RATIO + sub) * LAT_D + c);
    }
    *reinterpret_cast<uint4*>(Qs + r * QS_GLA + c) = v;
  }
  __syncthreads();

  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;
  const int r0 = (lane & 1) + 4 * hi;
  const int c0 = lane & 2;
  const int ktok = 8 * qp + arow;
  const int vdim = 32 * qp + 16 * hi;

  const int lrA = mw * 8 + r0;
  const int lrB = lrA + 2;
  const int grA = first_row + lrA;
  const int grB = first_row + lrB;
  const bool validA = grA < eff_nr;
  const bool validB = grB < eff_nr;
  const int posA = validA ? (anc + (grA / HQ_L)) : -1;
  const int posB = validB ? (anc + (grB / HQ_L)) : -1;

  const __half* qrow_lo = Qs + (mw * 8 + arow) * QS_GLA;
  const __half* qrow_hi = qrow_lo + HD;
  __half* Pw = Ps + warp * 8 * PS;
  const int pmask = page - 1;

  auto get_pg = [&](int pi) -> int64_t {
    return (int64_t)(pi < TBS ? Ts[pi] : tb[pi]);
  };

  uint4 cr[8];
  auto ld_qk_half = [&](const uint8_t* base, int t_idx) {
    const int tok0 = t_idx * TILE;
    if (page_shift >= 5 && tok0 + 31 < seqlen_k) {
      const int64_t pg = get_pg(tok0 >> page_shift);
      const int sl = (tok0 & pmask) + ktok;
      const uint8_t* p = base + pg * sb + (int64_t)sl * st;
#pragma unroll
      for (int i = 0; i < 8; i++) cr[i] = ldg16(p + 16 * i);
    } else {
      const int k_idx = tok0 + ktok;
      if (k_idx < seqlen_k) {
        const int pi = (page_shift >= 0) ? (k_idx >> page_shift) : (k_idx / page);
        const int sl = (page_shift >= 0) ? (k_idx & pmask) : (k_idx - pi * page);
        const int64_t pg = get_pg(pi);
        const uint8_t* p = base + pg * sb + (int64_t)sl * st;
#pragma unroll
        for (int i = 0; i < 8; i++) cr[i] = ldg16(p + 16 * i);
      } else {
#pragma unroll
        for (int i = 0; i < 8; i++) cr[i] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto ld_pv_half = [&](const uint8_t* base, int t_idx) {
    const int tok0 = t_idx * TILE;
    if (page_shift >= 5 && tok0 + 31 < seqlen_k) {
      const int64_t pg = get_pg(tok0 >> page_shift);
      const int s0 = (tok0 & pmask) + l3;
      const uint8_t* p = base + pg * sb + (int64_t)s0 * st + vdim;
#pragma unroll
      for (int i = 0; i < 8; i++) cr[i] = ldg16(p + (int64_t)(4 * i) * st);
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int v_idx = tok0 + l3 + 4 * i;
        if (v_idx < seqlen_k) {
          const int pi = (page_shift >= 0) ? (v_idx >> page_shift) : (v_idx / page);
          const int sl = (page_shift >= 0) ? (v_idx & pmask) : (v_idx - pi * page);
          const int64_t pg = get_pg(pi);
          cr[i] = ldg16(base + pg * sb + (int64_t)sl * st + vdim);
        } else {
          cr[i] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  int t = t_start + tw;
  if (t < t_end) ld_qk_half(c_lo, t);

  float o_lo[4][8], o_hi[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) { o_lo[j][e] = 0.f; o_hi[j][e] = 0.f; }
  float m0 = -INFINITY, m2 = -INFINITY, l0 = 0.f, l2 = 0.f;

  for (; t < t_end; t += T_WARPS) {
    const int tok0 = t * TILE;
    const bool more = (t + T_WARPS) < t_end;
    float sa[8], sb_acc[8];
#pragma unroll
    for (int e = 0; e < 8; e++) { sa[e] = 0.f; sb_acc[e] = 0.f; }

#pragma unroll
    for (int i = 0; i < 8; i++) {
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint2 a = *reinterpret_cast<const uint2*>(qrow_lo + 4 * (4 * i + j));
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rc((i & 1) ? sb_acc : sa, a.x, a.y, b0, b1);
      }
    }
    ld_qk_half(c_hi, t);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint2 a = *reinterpret_cast<const uint2*>(qrow_hi + 4 * (4 * i + j));
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rc((i & 1) ? sb_acc : sa, a.x, a.y, b0, b1);
      }
    }
    ld_pv_half(c_lo, t);
#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] += sb_acc[e];

    const int cb = tok0 + 8 * qp + c0;
    apply_dfly2_bias_mask_row(sa[0], sa[1], sa[4], sa[5], RelPtrs[lrA], validA, posA, tok0, cb, seqlen_k, 0, sl2);
    apply_dfly2_bias_mask_row(sa[2], sa[3], sa[6], sa[7], RelPtrs[lrB], validB, posB, tok0, cb, seqlen_k, 0, sl2);

    float x0 = fmaxf(fmaxf(sa[0], sa[1]), fmaxf(sa[4], sa[5]));
    float x2 = fmaxf(fmaxf(sa[2], sa[3]), fmaxf(sa[6], sa[7]));
#pragma unroll
    for (int d = 2; d <= 8; d <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
      x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
    }
    const float n0 = fmaxf(m0, x0);
    const float n2 = fmaxf(m2, x2);
    const float a0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - n0);
    const float a2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - n2);
    const float z0 = (n0 == -INFINITY) ? 0.f : n0;
    const float z2 = (n2 == -INFINITY) ? 0.f : n2;

#pragma unroll
    for (int e = 0; e < 8; e++) sa[e] = ex2_fast(sa[e] - ((e & 2) ? z2 : z0));
    l0 = l0 * a0 + (sa[0] + sa[1]) + (sa[4] + sa[5]);
    l2 = l2 * a2 + (sa[2] + sa[3]) + (sa[6] + sa[7]);
    m0 = n0;
    m2 = n2;

#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float af = (e & 2) ? a2 : a0;
        o_lo[j][e] *= af;
        o_hi[j][e] *= af;
      }

    const int pc = 8 * qp + c0;
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc) = __floats2half2_rn(sa[0], sa[1]);
    *reinterpret_cast<__half2*>(Pw + r0 * PS + pc + 4) = __floats2half2_rn(sa[4], sa[5]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc) = __floats2half2_rn(sa[2], sa[3]);
    *reinterpret_cast<__half2*>(Pw + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(sa[6], sa[7]);
    __syncwarp();

#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rr(o_lo[j], a.x, a.y, b0, b1);
      }
    }
    ld_pv_half(c_hi, t);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint2 a = *reinterpret_cast<const uint2*>(Pw + arow * PS + 4 * i);
      const unsigned w[4] = {cr[i].x, cr[i].y, cr[i].z, cr[i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned b0, b1;
        e4m3x4(w[j], b0, b1);
        mma_rr(o_hi[j], a.x, a.y, b0, b1);
      }
    }
    if (more) ld_qk_half(c_lo, t + T_WARPS);
    bar_tw_sync<M_WARPS, T_WARPS>(tw);
  }

#pragma unroll
  for (int d = 2; d <= 8; d <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, d);
    l2 += __shfl_xor_sync(0xffffffffu, l2, d);
  }

  if constexpr (T_WARPS == 1) {
    if (num_splits == 1) {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int gr = sr ? grB : grA;
        if (gr < eff_nr) {
          const float m_v = sr ? m2 : m0;
          const float l_v = sr ? l2 : l0;
          const float rsc = (l_v > 0.f) ? (vsc / l_v) : 0.f;
          const int tok = gr / HQ_L;
          const int head = gr - tok * HQ_L;
          const int g = head / G_RATIO;
          const int sub = head - g * G_RATIO;
          const int token = b * BLK + tok;
          __half* ob = o_lat + (((size_t)g * T_total + token) * G_RATIO + sub) * LAT_D;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            *reinterpret_cast<__half2*>(ob + dlo) =
                __floats2half2_rn(o_lo[j][2 * sr] * rsc, o_lo[j][2 * sr + 1] * rsc);
            *reinterpret_cast<__half2*>(ob + dlo + 16) =
                __floats2half2_rn(o_lo[j][4 + 2 * sr] * rsc, o_lo[j][5 + 2 * sr] * rsc);
            *reinterpret_cast<__half2*>(ob + HD + dlo) =
                __floats2half2_rn(o_hi[j][2 * sr] * rsc, o_hi[j][2 * sr + 1] * rsc);
            *reinterpret_cast<__half2*>(ob + HD + dlo + 16) =
                __floats2half2_rn(o_hi[j][4 + 2 * sr] * rsc, o_hi[j][5 + 2 * sr] * rsc);
          }
          if (lse_out != nullptr && (lane & 14) == 0) {
            lse_out[(size_t)token * HQ_L + head] = (l_v > 0.f) ? ((m_v + log2f(l_v)) * LN2) : -INFINITY;
          }
        }
      }
    } else {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int gr = sr ? grB : grA;
        if (gr < eff_nr) {
          const int tok = gr / HQ_L;
          const int head = gr - tok * HQ_L;
          const int token = b * BLK + tok;
          const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
          float* pob = o_part + slot * LAT_D;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int dlo = 32 * qp + 4 * j + c0;
            __stcg(reinterpret_cast<float2*>(pob + dlo), make_float2(o_lo[j][2 * sr], o_lo[j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + dlo + 16), make_float2(o_lo[j][4 + 2 * sr], o_lo[j][5 + 2 * sr]));
            __stcg(reinterpret_cast<float2*>(pob + HD + dlo), make_float2(o_hi[j][2 * sr], o_hi[j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + HD + dlo + 16), make_float2(o_hi[j][4 + 2 * sr], o_hi[j][5 + 2 * sr]));
          }
          if ((lane & 14) == 0) {
            __stcg(ml_part + slot, make_float2(sr ? m2 : m0, sr ? l2 : l0));
          }
        }
      }
    }
  } else {
    float* Os = reinterpret_cast<float*>(sm);
    float* MLs = reinterpret_cast<float*>(sm + L_::ML_OFF);
    float* RSs = reinterpret_cast<float*>(sm + L_::RS_OFF);
    __syncthreads();
    if ((lane & 14) == 0) {
      const int ml_base = (mw * T_WARPS + tw) * 16;
      MLs[ml_base + 2 * r0] = m0;
      MLs[ml_base + 2 * r0 + 1] = l0;
      MLs[ml_base + 2 * (r0 + 2)] = m2;
      MLs[ml_base + 2 * (r0 + 2) + 1] = l2;
    }
    __syncthreads();

    float M0 = -INFINITY, M2 = -INFINITY;
#pragma unroll
    for (int k = 0; k < T_WARPS; k++) {
      M0 = fmaxf(M0, MLs[(mw * T_WARPS + k) * 16 + 2 * r0]);
      M2 = fmaxf(M2, MLs[(mw * T_WARPS + k) * 16 + 2 * (r0 + 2)]);
    }
    const float f0 = (m0 == -INFINITY) ? 0.f : ex2_fast(m0 - M0);
    const float f2 = (m2 == -INFINITY) ? 0.f : ex2_fast(m2 - M2);

    for (int r = tid; r < R_CTA; r += NT) {
      const int rm = r >> 3;
      const int lr = r & 7;
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) M = fmaxf(M, MLs[(rm * T_WARPS + k) * 16 + 2 * lr]);
      float Lt = 0.f;
#pragma unroll
      for (int k = 0; k < T_WARPS; k++) {
        const float mk = MLs[(rm * T_WARPS + k) * 16 + 2 * lr];
        if (mk != -INFINITY) Lt += MLs[(rm * T_WARPS + k) * 16 + 2 * lr + 1] * ex2_fast(mk - M);
      }
      RSs[2 * r] = M;
      RSs[2 * r + 1] = Lt;
    }

    float* Op = Os + (size_t)(mw * T_WARPS + tw) * 8 * HD;
#pragma unroll
    for (int half_idx = 0; half_idx < 2; half_idx++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int sr = 0; sr < 2; sr++) {
          const int local_r = r0 + 2 * sr;
          const float f = sr ? f2 : f0;
          const int dlo = 32 * qp + 4 * j + c0;
          const float a = (half_idx == 0) ? o_lo[j][2 * sr] : o_hi[j][2 * sr];
          const float b_v = (half_idx == 0) ? o_lo[j][2 * sr + 1] : o_hi[j][2 * sr + 1];
          const float c_v = (half_idx == 0) ? o_lo[j][4 + 2 * sr] : o_hi[j][4 + 2 * sr];
          const float d_v = (half_idx == 0) ? o_lo[j][5 + 2 * sr] : o_hi[j][5 + 2 * sr];
          *reinterpret_cast<float2*>(Op + local_r * HD + dlo) = make_float2(a * f, b_v * f);
          *reinterpret_cast<float2*>(Op + local_r * HD + dlo + 16) = make_float2(c_v * f, d_v * f);
        }
      }
      __syncthreads();
      for (int i = tid; i < R_CTA * (HD / 4); i += NT) {
        const int r = i / (HD / 4);
        const int gr = first_row + r;
        if (gr >= eff_nr) continue;
        const int rm = r >> 3;
        const int lr = r & 7;
        const int d = (i % (HD / 4)) * 4;
        float4 acc = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS) * 8 * HD + lr * HD + d);
#pragma unroll
        for (int k = 1; k < T_WARPS; k++) {
          const float4 c = *reinterpret_cast<const float4*>(Os + (size_t)(rm * T_WARPS + k) * 8 * HD + lr * HD + d);
          acc.x += c.x; acc.y += c.y; acc.z += c.z; acc.w += c.w;
        }
        const int tok = gr / HQ_L;
        const int head = gr - tok * HQ_L;
        const int token = b * BLK + tok;
        if (num_splits == 1) {
          const float Lt = RSs[2 * r + 1];
          const float rsc = (Lt > 0.f) ? (vsc / Lt) : 0.f;
          const int g = head / G_RATIO;
          const int sub = head - g * G_RATIO;
          __half2* dst = reinterpret_cast<__half2*>(
              o_lat + (((size_t)g * T_total + token) * G_RATIO + sub) * LAT_D + half_idx * HD + d);
          dst[0] = __floats2half2_rn(acc.x * rsc, acc.y * rsc);
          dst[1] = __floats2half2_rn(acc.z * rsc, acc.w * rsc);
        } else {
          const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
          __stcg(reinterpret_cast<float4*>(o_part + slot * LAT_D + half_idx * HD + d), acc);
        }
      }
      if (half_idx == 0) __syncthreads();
    }

    if (num_splits == 1 && lse_out != nullptr) {
      for (int r = tid; r < R_CTA; r += NT) {
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_L;
          const int head = gr - tok * HQ_L;
          const float M = RSs[2 * r];
          const float Lt = RSs[2 * r + 1];
          lse_out[(size_t)(b * BLK + tok) * HQ_L + head] = (Lt > 0.f) ? ((M + log2f(Lt)) * LN2) : -INFINITY;
        }
      }
    } else if (num_splits > 1) {
      for (int r = tid; r < R_CTA; r += NT) {
        const int gr = first_row + r;
        if (gr < eff_nr) {
          const int tok = gr / HQ_L;
          const int head = gr - tok * HQ_L;
          const int token = b * BLK + tok;
          const size_t slot = ((size_t)token * HQ_L + head) * (size_t)num_splits + (size_t)s;
          __stcg(ml_part + slot, make_float2(RSs[2 * r], RSs[2 * r + 1]));
        }
      }
    }
  }
}

__global__ void __launch_bounds__(64)
k_dfly2_gla_ctx_combine(
    const float* __restrict__ o_part,
    const float2* __restrict__ ml_part,
    __half* __restrict__ o_lat,
    float* __restrict__ lse_out,
    int T_total,
    int num_splits,
    float vsc) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;  // 0..63 -> 4 dims each of 256
  const int lane = tid & 31;
  const size_t base_slot = ((size_t)token * HQ_L + head) * (size_t)num_splits;
  const float2* mlp = ml_part + base_slot;
  const float* op = o_part + base_slot * LAT_D;

  float M = -INFINITY;
  for (int s = lane; s < num_splits; s += 32) M = fmaxf(M, __ldcg(mlp + s).x);
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float L_tot = 0.f;
  for (int sb = 0; sb < num_splits; sb += 32) {
    const int idx = sb + lane;
    const float2 ml = (idx < num_splits) ? __ldcg(mlp + idx) : make_float2(-INFINITY, 0.f);
    const float w_lane = (ml.x == -INFINITY) ? 0.f : ex2_fast(ml.x - M);
    L_tot += ml.y * w_lane;
    const int n_chunk = min(32, num_splits - sb);
    for (int u = 0; u < n_chunk; u++) {
      const float ws = __shfl_sync(0xffffffffu, w_lane, u);
      if (ws > 0.f) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (size_t)(sb + u) * LAT_D + tid * 4));
        acc.x = fmaf(ws, v.x, acc.x);
        acc.y = fmaf(ws, v.y, acc.y);
        acc.z = fmaf(ws, v.z, acc.z);
        acc.w = fmaf(ws, v.w, acc.w);
      }
    }
  }
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) L_tot += __shfl_xor_sync(0xffffffffu, L_tot, d);

  const float rsc = (L_tot > 0.f) ? (vsc / L_tot) : 0.f;
  const int g = head / G_RATIO;
  const int sub = head - g * G_RATIO;
  __half2* dst = reinterpret_cast<__half2*>(
      o_lat + (((size_t)g * T_total + token) * G_RATIO + sub) * LAT_D + tid * 4);
  dst[0] = __floats2half2_rn(acc.x * rsc, acc.y * rsc);
  dst[1] = __floats2half2_rn(acc.z * rsc, acc.w * rsc);
  if (lse_out != nullptr && tid == 0) {
    lse_out[(size_t)token * HQ_L + head] = (L_tot > 0.f) ? ((M + log2f(L_tot)) * LN2) : -INFINITY;
  }
}

// ============================================================================
// 7. Stage B: 8x8 Bidirectional Intra-Block + LSE Merge + Gate * Sink
// ============================================================================
// Grid: (B, 6), Block: 32 threads (1 warp per (b, head); each lane owns 4 head dims of D=128).
__global__ void __launch_bounds__(32)
k_stage_b_merge_gate_sink(
    const __half* __restrict__ ao_ctx,         // [B*8, 6, 128]
    const float* __restrict__ lse_ctx,         // [B*8, 6] natural log
    const __half* __restrict__ q_b,            // [B*8, 6, 128]
    const __half* __restrict__ k_blk,          // [B, 8, 2, 128]
    const __half* __restrict__ v_blk,          // [B, 8, 2, 128]
    const __half* __restrict__ rel_all,        // [B*8, 6, 1024]
    const __half* __restrict__ intra_rel_bias, // [6, 31]
    const __half* __restrict__ qrgkv,          // [B*8, qrgkv_stride] (gate at +864)
    const __half* __restrict__ sinks,          // [B*8, 6]
    __half* __restrict__ attn_out,             // [B*8, 768]
    int qrgkv_stride,
    float scale) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int hkv = h / G_RATIO;
  const int lane = threadIdx.x;
  const int d0 = lane * 4;

  // Load 8 k_blk and 8 v_blk rows (4 dims per lane) into registers.
  float kb[BLK][4], vb[BLK][4];
#pragma unroll
  for (int k = 0; k < BLK; k++) {
    const __half* kp = k_blk + (((size_t)b * BLK + k) * HKV_L + hkv) * HD + d0;
    const __half* vp = v_blk + (((size_t)b * BLK + k) * HKV_L + hkv) * HD + d0;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      kb[k][u] = __half2float(kp[u]);
      vb[k][u] = __half2float(vp[u]);
    }
  }

  // Process each of the 8 query slots in the block.
  for (int qi = 0; qi < BLK; qi++) {
    const int tok = b * BLK + qi;
    const __half* qp = q_b + ((size_t)tok * HQ_L + h) * HD + d0;
    float qv[4];
#pragma unroll
    for (int u = 0; u < 4; u++) qv[u] = __half2float(qp[u]);

    float sc[BLK];
    float mx = -INFINITY;
    const __half* rr = rel_all + ((size_t)tok * HQ_L + h) * E_REL;
#pragma unroll
    for (int ki = 0; ki < BLK; ki++) {
      float dot = qv[0] * kb[ki][0] + qv[1] * kb[ki][1] + qv[2] * kb[ki][2] + qv[3] * kb[ki][3];
      dot = warp_sum(dot) * scale;
      const int rel_s = qi - ki;
      if (rel_s >= 0) dot += __half2float(rr[rel_s]);
      dot += __half2float(intra_rel_bias[h * 31 + (rel_s + HALF_INTRA)]);
      sc[ki] = dot;
      mx = fmaxf(mx, dot);
    }

    float lsum = 0.f;
    float pb[BLK];
#pragma unroll
    for (int ki = 0; ki < BLK; ki++) {
      pb[ki] = expf(sc[ki] - mx);
      lsum += pb[ki];
    }
    const float lse_blk = mx + logf(lsum);
    const float inv_l = 1.0f / lsum;

    float ao_b[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ki = 0; ki < BLK; ki++) {
      const float w = pb[ki] * inv_l;
      ao_b[0] = fmaf(w, vb[ki][0], ao_b[0]);
      ao_b[1] = fmaf(w, vb[ki][1], ao_b[1]);
      ao_b[2] = fmaf(w, vb[ki][2], ao_b[2]);
      ao_b[3] = fmaf(w, vb[ki][3], ao_b[3]);
    }

    // Exact online-softmax LSE merge with (ao_ctx, lse_ctx)
    const float lc = lse_ctx[(size_t)tok * HQ_L + h];
    float lse_tot, w_ctx, w_blk;
    if (lc == -INFINITY) {
      lse_tot = lse_blk;
      w_ctx = 0.f;
      w_blk = 1.f;
    } else {
      const float m_ab = fmaxf(lc, lse_blk);
      lse_tot = m_ab + logf(expf(lc - m_ab) + expf(lse_blk - m_ab));
      w_ctx = expf(lc - lse_tot);
      w_blk = expf(lse_blk - lse_tot);
    }

    const float sink_v = __half2float(sinks[(size_t)tok * HQ_L + h]);
    const float ss = sigmoid_f32(lse_tot - sink_v);
    const __half* cp = ao_ctx + ((size_t)tok * HQ_L + h) * HD + d0;
    const __half* gp = qrgkv + (size_t)tok * qrgkv_stride + 864 + h * HD + d0;
    __half* dst = attn_out + (size_t)tok * (HQ_L * HD) + h * HD + d0;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const float ao = w_ctx * __half2float(cp[u]) + w_blk * ao_b[u];
      const float y = ao * sigmoid_f32(__half2float(gp[u])) * ss;
      dst[u] = __float2half_rn(y);
    }
  }
}

// ============================================================================
// 8. Stage C: AttentiveCorrection Kernels (Standard RMSNorm, L2-Clip, Combine)
// ============================================================================
__global__ void __launch_bounds__(256)
k_stage_c_prep(
    const __half* __restrict__ h_bb,         // [B*8, 2304]
    const __half* __restrict__ e_prev,       // [B, 2304]
    const __half* __restrict__ hidden_norm,  // [2304]
    const __half* __restrict__ embed_norm,   // [2304]
    __half* __restrict__ inp_c,              // [B, 4608]
    int slot,
    float eps) {
  const int b = blockIdx.x;
  const int which = blockIdx.y;  // 0 = hidden_norm, 1 = embed_norm
  const int tid = threadIdx.x;
  const __half* src = (which == 0) ? (h_bb + (size_t)(b * BLK + slot) * H_DIM) : (e_prev + (size_t)b * H_DIM);
  const __half* w = (which == 0) ? hidden_norm : embed_norm;
  __half* dst = inp_c + (size_t)b * (2 * H_DIM) + (size_t)which * H_DIM;

  float ss = 0.f;
  for (int i = tid; i < H_DIM; i += 256) {
    const float v = __half2float(src[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum_256(ss) * (1.0f / H_DIM) + eps);
  for (int i = tid; i < H_DIM; i += 256) {
    dst[i] = __float2half_rn(__half2float(src[i]) * inv * __half2float(w[i]));
  }
}

// Slot 0: kv0 [B, 512] = [k_blk(256), v_blk(256)] -> blk_k_c[b, 0], blk_v_c[b, 0]
__global__ void __launch_bounds__(64)
k_stage_c_slot0_kv(
    const __half* __restrict__ kv0,
    __half* __restrict__ blk_k_c,  // [B, 8, 2, 128]
    __half* __restrict__ blk_v_c,  // [B, 8, 2, 128]
    int kv0_stride,
    float eps) {
  const int b = blockIdx.x;
  const int hkv = threadIdx.x >> 5;  // 0..1
  const int lane = threadIdx.x & 31;
  const __half* row = kv0 + (size_t)b * kv0_stride;

  const __half* kp = row + hkv * HD;
  float kv[4];
  float sk = 0.f;
#pragma unroll
  for (int u = 0; u < 4; u++) {
    kv[u] = __half2float(kp[lane * 4 + u]);
    sk += kv[u] * kv[u];
  }
  const float knorm = sqrtf(warp_sum(sk) + eps);
  const float kfac = (knorm > Q_CLIP) ? (Q_CLIP / knorm) : 1.0f;
  __half* kd = blk_k_c + ((size_t)b * BLK * HKV_L + hkv) * HD;
#pragma unroll
  for (int u = 0; u < 4; u++) kd[lane * 4 + u] = __float2half_rn(kv[u] * kfac);

  const __half* vp = row + 256 + hkv * HD;
  float vv[4];
  float sv = 0.f;
#pragma unroll
  for (int u = 0; u < 4; u++) {
    vv[u] = __half2float(vp[lane * 4 + u]);
    sv += vv[u] * vv[u];
  }
  const float vinv = rsqrtf(warp_sum(sv) * (1.0f / HD) + eps);
  __half* vd = blk_v_c + ((size_t)b * BLK * HKV_L + hkv) * HD;
#pragma unroll
  for (int u = 0; u < 4; u++) vd[lane * 4 + u] = __float2half_rn(vv[u] * vinv);
}

// mega_c row layout (stride = mega_stride):
//   [gate: 0..575 | up: 576..1151 | q: 1152..1919 | r: 1920..2015 | k_blk: 2016..2271 | v_blk: 2272..2527]
__global__ void __launch_bounds__(256)
k_stage_c_qkrv_silu(
    const __half* __restrict__ mega_c,
    const __half* __restrict__ q_scale,   // [6, 128]
    const __half* __restrict__ r_scale,   // [6, 16]
    __half* __restrict__ mid_c,           // [B, 576]
    __half* __restrict__ q_c,             // [B, 6, 128]
    __half* __restrict__ r_c,             // [B, 6, 16]
    __half* __restrict__ blk_k_c,         // [B, 8, 2, 128]
    __half* __restrict__ blk_v_c,         // [B, 8, 2, 128]
    int mega_stride,
    int slot,
    float eps) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const __half* row = mega_c + (size_t)b * mega_stride;

  // All 256 threads compute SwiGLU for 576 elements first!
  __half* mp = mid_c + (size_t)b * 576;
  for (int i = tid; i < 576; i += 256) {
    const float g = __half2float(row[i]);
    const float u = __half2float(row[576 + i]);
    mp[i] = __float2half_rn(silu_f32(g) * u);
  }

  // Warps 0..5: Q & R clip (SWA local -> tau = 1.0)
  if (warp < HQ_L) {
    const int h = warp;
    const __half* qp = row + 1152 + h * HD;
    const __half* qsp = q_scale + h * HD;
    float qv[4];
    float sq = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int d = lane * 4 + u;
      qv[u] = __half2float(qp[d]) * __half2float(qsp[d]);
      sq += qv[u] * qv[u];
    }
    const float qnorm = sqrtf(warp_sum(sq) + eps);
    const float qfac = (qnorm > Q_CLIP) ? (Q_CLIP / qnorm) : 1.0f;
    __half* qd = q_c + ((size_t)b * HQ_L + h) * HD;
#pragma unroll
    for (int u = 0; u < 4; u++) qd[lane * 4 + u] = __float2half_rn(qv[u] * qfac);

    float rv = 0.f;
    if (lane < D_REL) {
      rv = __half2float(row[1920 + h * D_REL + lane]) * __half2float(r_scale[h * D_REL + lane]);
    }
    const float rnorm = sqrtf(warp_sum(rv * rv) + eps);
    const float rfac = (rnorm > R_CLIP) ? (R_CLIP / rnorm) : 1.0f;
    if (lane < D_REL) {
      r_c[((size_t)b * HQ_L + h) * D_REL + lane] = __float2half_rn(rv * rfac);
    }
  } else {
    // Warps 6..7: K_blk & V_blk into slot
    const int hkv = warp - HQ_L;
    const __half* kp = row + 2016 + hkv * HD;
    float kv[4];
    float sk = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      kv[u] = __half2float(kp[lane * 4 + u]);
      sk += kv[u] * kv[u];
    }
    const float knorm = sqrtf(warp_sum(sk) + eps);
    const float kfac = (knorm > Q_CLIP) ? (Q_CLIP / knorm) : 1.0f;
    __half* kd = blk_k_c + (((size_t)b * BLK + slot) * HKV_L + hkv) * HD;
#pragma unroll
    for (int u = 0; u < 4; u++) kd[lane * 4 + u] = __float2half_rn(kv[u] * kfac);

    const __half* vp = row + 2272 + hkv * HD;
    float vv[4];
    float sv = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      vv[u] = __half2float(vp[lane * 4 + u]);
      sv += vv[u] * vv[u];
    }
    const float vinv = rsqrtf(warp_sum(sv) * (1.0f / HD) + eps);
    __half* vd = blk_v_c + (((size_t)b * BLK + slot) * HKV_L + hkv) * HD;
#pragma unroll
    for (int u = 0; u < 4; u++) vd[lane * 4 + u] = __float2half_rn(vv[u] * vinv);
  }
}

// Grid: (B, 6), Block: 32 threads (1 warp per (b, head), 4 head dims per lane).
__global__ void __launch_bounds__(32)
k_stage_c_combine(
    const __half* __restrict__ o_ctx_c,         // [B, 6, 128]
    const __half* __restrict__ q_c,             // [B, 6, 128]
    const __half* __restrict__ blk_k_c,         // [B, 8, 2, 128]
    const __half* __restrict__ blk_v_c,         // [B, 8, 2, 128]
    const __half* __restrict__ gate_raw,        // [B, 6]
    const __half* __restrict__ block_gate_bias, // [6, 16]
    const __half* __restrict__ block_rel_bias,  // [6, 16]
    const __half* __restrict__ block_sink,      // [6]
    __half* __restrict__ attn_out_c,            // [B, 768]
    int slot,
    float scale) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int hkv = h / G_RATIO;
  const int lane = threadIdx.x;
  const int d0 = lane * 4;

  const __half* qp = q_c + ((size_t)b * HQ_L + h) * HD + d0;
  float qv[4];
#pragma unroll
  for (int u = 0; u < 4; u++) qv[u] = __half2float(qp[u]);

  float sc[BLK];
  float mx = -INFINITY;
  for (int t = 0; t <= slot; t++) {
    const __half* kp = blk_k_c + (((size_t)b * BLK + t) * HKV_L + hkv) * HD + d0;
    float dot = qv[0] * __half2float(kp[0]) + qv[1] * __half2float(kp[1])
              + qv[2] * __half2float(kp[2]) + qv[3] * __half2float(kp[3]);
    dot = warp_sum(dot) * scale + __half2float(block_rel_bias[h * MAX_BLK_SLOTS + (slot - t)]);
    sc[t] = dot;
    mx = fmaxf(mx, dot);
  }
  const float sink_v = __half2float(block_sink[h]);
  const float m_all = fmaxf(mx, sink_v);
  float denom = expf(sink_v - m_all);
  for (int t = 0; t <= slot; t++) denom += expf(sc[t] - m_all);
  const float inv_d = 1.0f / denom;

  float read[4] = {0.f, 0.f, 0.f, 0.f};
  for (int t = 0; t <= slot; t++) {
    const float p = expf(sc[t] - m_all) * inv_d;
    const __half* vp = blk_v_c + (((size_t)b * BLK + t) * HKV_L + hkv) * HD + d0;
#pragma unroll
    for (int u = 0; u < 4; u++) read[u] = fmaf(p, __half2float(vp[u]), read[u]);
  }

  const float g = tanhf(__half2float(gate_raw[(size_t)b * HQ_L + h]) +
                        __half2float(block_gate_bias[h * MAX_BLK_SLOTS + slot]));
  const __half* cp = o_ctx_c + ((size_t)b * HQ_L + h) * HD + d0;
  __half* dst = attn_out_c + (size_t)b * (HQ_L * HD) + h * HD + d0;
#pragma unroll
  for (int u = 0; u < 4; u++) {
    dst[u] = __float2half_rn(__half2float(cp[u]) + read[u] * g);
  }
}

__global__ void __launch_bounds__(256)
k_add_slot_residual(
    const __half* __restrict__ h_bb,
    const __half* __restrict__ delta,
    __half* __restrict__ h_corr,
    int slot) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const __half* hp = h_bb + (size_t)(b * BLK + slot) * H_DIM;
  const __half* dp = delta + (size_t)b * H_DIM;
  __half* dst = h_corr + (size_t)b * H_DIM;
  for (int i = tid; i < H_DIM; i += 256) {
    dst[i] = __float2half_rn(__half2float(hp[i]) + __half2float(dp[i]));
  }
}

// Helper for launching dfly2_ctx_attn
template <bool IS_FP8, int M_WARPS, int T_WARPS>
void dispatch_ctx(
    const __half* q_ptr, const void* kc_ptr, const void* vc_ptr, const __half* rel_ptr,
    __half* out_ptr, float* lse_ptr, float* o_part_ptr, float2* ml_part_ptr,
    const int* bt_ptr, const int* cl_ptr, const int* ap_ptr,
    int B, int Q, int slot_fixed, int window, int page, int page_shift, int max_blocks, int S,
    int64_t kb, int64_t kt, int64_t kh, int64_t vb, int64_t vt, int64_t vh, int64_t stride_bt,
    float sl2, float vsc, cudaStream_t stream) {
  dim3 grid(S, HKV_L, B);
  constexpr int smem = CtxSM<M_WARPS, T_WARPS>::TOTAL;
  k_dfly2_ctx_attn<IS_FP8, M_WARPS, T_WARPS><<<grid, 32 * M_WARPS * T_WARPS, smem, stream>>>(
      q_ptr, kc_ptr, vc_ptr, rel_ptr, out_ptr, lse_ptr, o_part_ptr, ml_part_ptr,
      bt_ptr, cl_ptr, ap_ptr, Q, slot_fixed, window, page, page_shift, max_blocks, S,
      kb, kt, kh, vb, vt, vh, stride_bt, sl2, vsc);
}

}  // namespace

// ============================================================================
// C++ / PyBind Entry Points
// ============================================================================
void extract_inputs(
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor idx_mapping,
    torch::Tensor positions,
    torch::Tensor last_hidden_states,
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens,
    torch::Tensor anchor_tok,
    torch::Tensor ctx_len,
    torch::Tensor anchor_pos,
    torch::Tensor prev_hidden,
    torch::Tensor req_slots,
    int64_t num_reqs,
    int64_t b_pad,
    int64_t max_model_len) {
  if (b_pad <= 0) return;
  const at::cuda::CUDAGuard guard(anchor_tok.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool pos64 = (positions.scalar_type() == torch::kInt64);
  const bool ls64 = (last_sampled.scalar_type() == torch::kInt64);
  const bool np64 = (next_prefill_tokens.scalar_type() == torch::kInt64);
  const int mask = (pos64 ? 4 : 0) | (ls64 ? 2 : 0) | (np64 ? 1 : 0);
#define LAUNCH_EXT(P, L, N) \
  k_extract_inputs<P, L, N><<<(int)b_pad, 256, 0, stream>>>( \
      query_start_loc.data_ptr<int32_t>(), \
      seq_lens.data_ptr<int32_t>(), \
      idx_mapping.data_ptr<int32_t>(), \
      positions.data_ptr(), \
      reinterpret_cast<const __half*>(last_hidden_states.data_ptr<at::Half>()), \
      num_sampled.data_ptr<int32_t>(), \
      num_rejected.data_ptr<int32_t>(), \
      last_sampled.data_ptr(), \
      next_prefill_tokens.data_ptr(), \
      anchor_tok.data_ptr<int32_t>(), \
      ctx_len.data_ptr<int32_t>(), \
      anchor_pos.data_ptr<int32_t>(), \
      reinterpret_cast<__half*>(prev_hidden.data_ptr<at::Half>()), \
      req_slots.data_ptr<int32_t>(), \
      (int)num_reqs, (int)b_pad, (int)max_model_len)
  switch (mask) {
    case 0: LAUNCH_EXT(false, false, false); break;
    case 1: LAUNCH_EXT(false, false, true);  break;
    case 2: LAUNCH_EXT(false, true,  false); break;
    case 3: LAUNCH_EXT(false, true,  true);  break;
    case 4: LAUNCH_EXT(true,  false, false); break;
    case 5: LAUNCH_EXT(true,  false, true);  break;
    case 6: LAUNCH_EXT(true,  true,  false); break;
    default: LAUNCH_EXT(true, true,  true);  break;
  }
#undef LAUNCH_EXT
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_a_prep(
    torch::Tensor e0, torch::Tensor prev_hidden,
    torch::Tensor enorm_w, torch::Tensor hnorm_w,
    torch::Tensor eh_in_local, int64_t tp_rank, double eps) {
  const int B = (int)e0.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(e0.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_stage_a_prep<<<B, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(e0.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(enorm_w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(hnorm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(eh_in_local.data_ptr<at::Half>()),
      (int)tp_rank, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_a_fuse_norm(
    torch::Tensor e0, torch::Tensor eh_red, torch::Tensor mtp_gate,
    torch::Tensor mask_emb, torch::Tensor ln0_w,
    torch::Tensor res_f32, torch::Tensor x_norm, double eps) {
  const int N8 = (int)res_f32.size(0);
  if (N8 <= 0) return;
  const at::cuda::CUDAGuard guard(e0.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_stage_a_fuse_norm<<<N8, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(e0.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(eh_red.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(mtp_gate.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(mask_emb.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(ln0_w.data_ptr<at::Half>()),
      res_f32.data_ptr<float>(),
      reinterpret_cast<__half*>(x_norm.data_ptr<at::Half>()),
      (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void rmsnorm_res(
    torch::Tensor delta, torch::Tensor res_f32,
    torch::Tensor norm_w, torch::Tensor x_norm, double eps) {
  const int N8 = (int)delta.size(0);
  if (N8 <= 0) return;
  const at::cuda::CUDAGuard guard(delta.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_rmsnorm_res<<<N8, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(delta.data_ptr<at::Half>()),
      res_f32.data_ptr<float>(),
      reinterpret_cast<const __half*>(norm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(x_norm.data_ptr<at::Half>()),
      (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_b_qkrv_prep(
    torch::Tensor qrgkv, torch::Tensor anchor_pos,
    torch::Tensor q_scale, torch::Tensor r_scale,
    torch::Tensor q_out, torch::Tensor r_out,
    torch::Tensor k_blk_out, torch::Tensor v_blk_out,
    bool apply_tau, double eps) {
  const int N8 = (int)qrgkv.size(0);
  if (N8 <= 0) return;
  const at::cuda::CUDAGuard guard(qrgkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_stage_b_qkrv_prep<<<N8, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(qrgkv.data_ptr<at::Half>()),
      anchor_pos.data_ptr<int32_t>(),
      reinterpret_cast<const __half*>(q_scale.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(r_scale.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(r_out.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(k_blk_out.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(v_blk_out.data_ptr<at::Half>()),
      (int)qrgkv.stride(0), apply_tau, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void dfly2_ctx_attn_forward(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor rel_all,
    torch::Tensor out,
    c10::optional<torch::Tensor> lse_out,
    torch::Tensor block_table,
    torch::Tensor ctx_len,
    torch::Tensor anchor_pos,
    torch::Tensor o_part,
    torch::Tensor ml_part,
    int64_t Q,
    int64_t slot_fixed,
    double softmax_scale,
    int64_t window,
    int64_t num_splits) {
  const int B = (int)ctx_len.size(0);
  if (B <= 0) return;
  const bool is_fp8 = key_cache.element_size() == 1;
  const int page = (int)key_cache.size(1);
  int page_shift = -1;
  if (page > 0 && (page & (page - 1)) == 0) {
    page_shift = 0;
    int tmp = page;
    while (tmp > 1) { tmp >>= 1; page_shift++; }
  }
  const int S = std::max(1, (int)num_splits);
  const int max_blocks = (int)block_table.size(1);
  const float sl2 = (float)(softmax_scale * (is_fp8 ? 256.0 : 1.0) * (double)LOG2E);
  const float vsc = is_fp8 ? 256.0f : 1.0f;

  float* lse_ptr = lse_out.has_value() ? lse_out->data_ptr<float>() : nullptr;
  float* op_ptr = (S > 1) ? o_part.data_ptr<float>() : nullptr;
  float2* mlp_ptr = (S > 1) ? reinterpret_cast<float2*>(ml_part.data_ptr<float>()) : nullptr;

  const at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const __half* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  const void* kcp = key_cache.data_ptr();
  const void* vcp = value_cache.data_ptr();
  const __half* relp = reinterpret_cast<const __half*>(rel_all.data_ptr<at::Half>());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  const int* btp = block_table.data_ptr<int32_t>();
  const int* clp = ctx_len.data_ptr<int32_t>();
  const int* app = anchor_pos.data_ptr<int32_t>();

  const int64_t kb = key_cache.stride(0), kt = key_cache.stride(1), kh = key_cache.stride(2);
  const int64_t vb = value_cache.stride(0), vt = value_cache.stride(1), vh = value_cache.stride(2);
  const int64_t sbt = block_table.stride(0);

  if (Q == 1) {
    if (is_fp8) {
      dispatch_ctx<true, 1, 4>(qp, kcp, vcp, relp, outp, lse_ptr, op_ptr, mlp_ptr, btp, clp, app,
                               B, 1, (int)slot_fixed, (int)window, page, page_shift, max_blocks, S,
                               kb, kt, kh, vb, vt, vh, sbt, sl2, vsc, stream);
    } else {
      dispatch_ctx<false, 1, 4>(qp, kcp, vcp, relp, outp, lse_ptr, op_ptr, mlp_ptr, btp, clp, app,
                                B, 1, (int)slot_fixed, (int)window, page, page_shift, max_blocks, S,
                                kb, kt, kh, vb, vt, vh, sbt, sl2, vsc, stream);
    }
  } else {
    TORCH_CHECK(Q == BLK, "dfly2_ctx_attn expects Q==1 or Q==8, got ", Q);
    const bool tw1 = (B * HKV_L * S >= 256);
    if (is_fp8) {
      if (tw1) {
        dispatch_ctx<true, 3, 1>(qp, kcp, vcp, relp, outp, lse_ptr, op_ptr, mlp_ptr, btp, clp, app,
                                 B, BLK, 0, (int)window, page, page_shift, max_blocks, S,
                                 kb, kt, kh, vb, vt, vh, sbt, sl2, vsc, stream);
      } else {
        dispatch_ctx<true, 3, 2>(qp, kcp, vcp, relp, outp, lse_ptr, op_ptr, mlp_ptr, btp, clp, app,
                                 B, BLK, 0, (int)window, page, page_shift, max_blocks, S,
                                 kb, kt, kh, vb, vt, vh, sbt, sl2, vsc, stream);
      }
    } else {
      if (tw1) {
        dispatch_ctx<false, 3, 1>(qp, kcp, vcp, relp, outp, lse_ptr, op_ptr, mlp_ptr, btp, clp, app,
                                  B, BLK, 0, (int)window, page, page_shift, max_blocks, S,
                                  kb, kt, kh, vb, vt, vh, sbt, sl2, vsc, stream);
      } else {
        dispatch_ctx<false, 3, 2>(qp, kcp, vcp, relp, outp, lse_ptr, op_ptr, mlp_ptr, btp, clp, app,
                                  B, BLK, 0, (int)window, page, page_shift, max_blocks, S,
                                  kb, kt, kh, vb, vt, vh, sbt, sl2, vsc, stream);
      }
    }
  }

  if (S > 1) {
    dim3 cgrid(B * (int)Q, HQ_L);
    k_dfly2_ctx_combine<<<cgrid, 32, 0, stream>>>(op_ptr, mlp_ptr, outp, lse_ptr, S, vsc);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void dfly2_gla_ctx_attn_forward(
    torch::Tensor q_lat,
    torch::Tensor c_lo,
    torch::Tensor c_hi,
    torch::Tensor rel_all,
    torch::Tensor o_lat,
    torch::Tensor lse_out,
    torch::Tensor block_table,
    torch::Tensor ctx_len,
    torch::Tensor anchor_pos,
    torch::Tensor o_part,
    torch::Tensor ml_part,
    double softmax_scale,
    int64_t num_splits) {
  const int B = (int)ctx_len.size(0);
  if (B <= 0) return;
  const int T_total = B * BLK;
  const int page = (int)c_lo.size(1);
  int page_shift = -1;
  if (page > 0 && (page & (page - 1)) == 0) {
    page_shift = 0;
    int tmp = page;
    while (tmp > 1) { tmp >>= 1; page_shift++; }
  }
  const int S = std::max(1, (int)num_splits);
  const int max_blocks = (int)block_table.size(1);
  const float sl2 = (float)(softmax_scale * 256.0 * (double)LOG2E);
  const float vsc = 256.0f;

  float* op_ptr = (S > 1) ? o_part.data_ptr<float>() : nullptr;
  float2* mlp_ptr = (S > 1) ? reinterpret_cast<float2*>(ml_part.data_ptr<float>()) : nullptr;

  const at::cuda::CUDAGuard guard(q_lat.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // 48 rows per req -> 2 m_tiles of 24 rows (M_WARPS = 3)
  dim3 grid(S, 2, B);
  const bool tw1 = (B * 2 * S >= 256);
  if (tw1) {
    constexpr int smem = GlaCtxSM<3, 1>::TOTAL;
    k_dfly2_gla_ctx_attn<3, 1><<<grid, 32 * 3, smem, stream>>>(
        reinterpret_cast<const __half*>(q_lat.data_ptr<at::Half>()),
        c_lo.data_ptr<uint8_t>(), c_hi.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(rel_all.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),
        lse_out.data_ptr<float>(), op_ptr, mlp_ptr,
        block_table.data_ptr<int32_t>(), ctx_len.data_ptr<int32_t>(), anchor_pos.data_ptr<int32_t>(),
        T_total, page, page_shift, max_blocks, S, c_lo.stride(0), c_lo.stride(1), block_table.stride(0),
        sl2, vsc);
  } else {
    constexpr int smem = GlaCtxSM<3, 2>::TOTAL;
    k_dfly2_gla_ctx_attn<3, 2><<<grid, 32 * 6, smem, stream>>>(
        reinterpret_cast<const __half*>(q_lat.data_ptr<at::Half>()),
        c_lo.data_ptr<uint8_t>(), c_hi.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(rel_all.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),
        lse_out.data_ptr<float>(), op_ptr, mlp_ptr,
        block_table.data_ptr<int32_t>(), ctx_len.data_ptr<int32_t>(), anchor_pos.data_ptr<int32_t>(),
        T_total, page, page_shift, max_blocks, S, c_lo.stride(0), c_lo.stride(1), block_table.stride(0),
        sl2, vsc);
  }
  if (S > 1) {
    dim3 cgrid(T_total, HQ_L);
    k_dfly2_gla_ctx_combine<<<cgrid, 64, 0, stream>>>(
        op_ptr, mlp_ptr, reinterpret_cast<__half*>(o_lat.data_ptr<at::Half>()),
        lse_out.data_ptr<float>(), T_total, S, vsc);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_b_merge_gate_sink(
    torch::Tensor ao_ctx, torch::Tensor lse_ctx,
    torch::Tensor q_b, torch::Tensor k_blk, torch::Tensor v_blk,
    torch::Tensor rel_all, torch::Tensor intra_rel_bias,
    torch::Tensor qrgkv, torch::Tensor sinks,
    torch::Tensor attn_out, double scale) {
  const int B = (int)k_blk.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(ao_ctx.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(B, HQ_L);
  k_stage_b_merge_gate_sink<<<grid, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(ao_ctx.data_ptr<at::Half>()),
      lse_ctx.data_ptr<float>(),
      reinterpret_cast<const __half*>(q_b.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k_blk.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(v_blk.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(rel_all.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(intra_rel_bias.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(qrgkv.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(sinks.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(attn_out.data_ptr<at::Half>()),
      (int)qrgkv.stride(0), (float)scale);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_c_prep(
    torch::Tensor h_bb, torch::Tensor e_prev,
    torch::Tensor hidden_norm, torch::Tensor embed_norm,
    torch::Tensor inp_c, int64_t slot, double eps) {
  const int B = (int)e_prev.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(h_bb.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(B, 2);
  k_stage_c_prep<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(h_bb.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(e_prev.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(hidden_norm.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(embed_norm.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(inp_c.data_ptr<at::Half>()),
      (int)slot, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_c_slot0_kv(
    torch::Tensor kv0, torch::Tensor blk_k_c, torch::Tensor blk_v_c, double eps) {
  const int B = (int)kv0.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(kv0.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_stage_c_slot0_kv<<<B, 64, 0, stream>>>(
      reinterpret_cast<const __half*>(kv0.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_k_c.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_v_c.data_ptr<at::Half>()),
      (int)kv0.stride(0), (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_c_qkrv_silu(
    torch::Tensor mega_c, torch::Tensor q_scale, torch::Tensor r_scale,
    torch::Tensor mid_c, torch::Tensor q_c, torch::Tensor r_c,
    torch::Tensor blk_k_c, torch::Tensor blk_v_c, int64_t slot, double eps) {
  const int B = (int)mega_c.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(mega_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_stage_c_qkrv_silu<<<B, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(mega_c.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(q_scale.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(r_scale.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(mid_c.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(q_c.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(r_c.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_k_c.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_v_c.data_ptr<at::Half>()),
      (int)mega_c.stride(0), (int)slot, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void stage_c_combine(
    torch::Tensor o_ctx_c, torch::Tensor q_c, torch::Tensor blk_k_c, torch::Tensor blk_v_c,
    torch::Tensor gate_raw, torch::Tensor block_gate_bias, torch::Tensor block_rel_bias,
    torch::Tensor block_sink, torch::Tensor attn_out_c, int64_t slot, double scale) {
  const int B = (int)o_ctx_c.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(o_ctx_c.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(B, HQ_L);
  k_stage_c_combine<<<grid, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(o_ctx_c.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(q_c.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(blk_k_c.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(blk_v_c.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(gate_raw.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(block_gate_bias.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(block_rel_bias.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(block_sink.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(attn_out_c.data_ptr<at::Half>()),
      (int)slot, (float)scale);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

void add_slot_residual(
    torch::Tensor h_bb, torch::Tensor delta, torch::Tensor h_corr, int64_t slot) {
  const int B = (int)delta.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(delta.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  k_add_slot_residual<<<B, 256, 0, stream>>>(
      reinterpret_cast<const __half*>(h_bb.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(delta.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(h_corr.data_ptr<at::Half>()),
      (int)slot);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_inputs", &extract_inputs);
  m.def("stage_a_prep", &stage_a_prep);
  m.def("stage_a_fuse_norm", &stage_a_fuse_norm);
  m.def("rmsnorm_res", &rmsnorm_res);
  m.def("stage_b_qkrv_prep", &stage_b_qkrv_prep);
  m.def("dfly2_ctx_attn_forward", &dfly2_ctx_attn_forward);
  m.def("dfly2_gla_ctx_attn_forward", &dfly2_gla_ctx_attn_forward);
  m.def("stage_b_merge_gate_sink", &stage_b_merge_gate_sink);
  m.def("stage_c_prep", &stage_c_prep);
  m.def("stage_c_slot0_kv", &stage_c_slot0_kv);
  m.def("stage_c_qkrv_silu", &stage_c_qkrv_silu);
  m.def("stage_c_combine", &stage_c_combine);
  m.def("add_slot_residual", &add_slot_residual);
}
