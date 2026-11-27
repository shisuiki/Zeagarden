// dfly2_ops.cu - Fused Volta sm_70 kernels for Qwen3.8-27B DFly2-KV drafter (TP4, H=5120, Hq_local=6, Hkv_local=1,
// D=256, rotary_dim=64, rope_theta=1e7, block_size=6, zero-centered Gemma RMSNorm everywhere).
//
// Zero-allocation, CUDA-graph safe, supports BOTH Exact E4M3 target KV and GLA-g4 (r_lat in {192, 256}) compressed
// target KV without retraining.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int H_DIM = 5120;
constexpr int H_HALF = 2560;
constexpr int NQ = 6;
constexpr int HD = 256;
constexpr int R_DIM = 64;
constexpr int R_HALF = 32;
constexpr int BLK = 6;
constexpr int GLA_ROPE_DIM = 62;
constexpr int GLA_NOPE_DIM = 194;
constexpr int HC_MLP_I = 1280;   // 5120 / 4 TP4 shard

// Exact FP32 inv_freq = 1.0 / (1e7 ** (2*i / 64)) for i = 0..31 (matches Qwen3.5/3.8 rope_theta = 1e7)
__device__ __constant__ float D_INV_FREQ[32] = {
    1.00000000e+00f, 6.04296390e-01f, 3.65174127e-01f, 2.20673407e-01f,
    1.33352143e-01f, 8.05842188e-02f, 4.86967525e-02f, 2.94272718e-02f,
    1.77827941e-02f, 1.07460783e-02f, 6.49381632e-03f, 3.92418976e-03f,
    2.37137371e-03f, 1.43301257e-03f, 8.65964323e-04f, 5.23299115e-04f,
    3.16227766e-04f, 1.91095297e-04f, 1.15478198e-04f, 6.97830585e-05f,
    4.21696503e-05f, 2.54829675e-05f, 1.53992653e-05f, 9.30572041e-06f,
    5.62341325e-06f, 3.39820833e-06f, 2.05352503e-06f, 1.24093776e-06f,
    7.49894209e-07f, 4.53158364e-07f, 2.73841963e-07f, 1.65481710e-07f,
};

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
  return v;
}

__device__ __forceinline__ float bsum256(float v, float* sm) {
  v = wsum(v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  float tot = 0.f;
  if (warp == 0) {
    tot = (lane < 8) ? sm[lane] : 0.f;
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, d);
    if (lane == 0) sm[0] = tot;
  }
  __syncthreads();
  return sm[0];
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// =================================================================================================
// 1. dfly2_extract_inputs: gather anchor_tok, ctx_len, anchor_pos, prev_hidden, req_slots on GPU
// =================================================================================================
template <bool POS64, bool LS64, bool NP64>
__global__ void __launch_bounds__(256)
k_extract_inputs(const int* __restrict__ qsl, const int* __restrict__ seq_lens, const int* __restrict__ idx_map,
                 const void* __restrict__ positions, int pos_stride, const __half* __restrict__ last_hidden,
                 const int* __restrict__ num_sampled, const int* __restrict__ num_rejected,
                 const void* __restrict__ last_sampled, const void* __restrict__ next_prefill,
                 int* __restrict__ out_anchor_tok, int* __restrict__ out_ctx_len, int* __restrict__ out_ctx_len_6,
                 int* __restrict__ out_anchor_pos, __half* __restrict__ out_prev_hidden,
                 int* __restrict__ out_req_slots, int num_reqs, int B_pad, int max_model_len) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  if (b >= B_pad) return;

  __shared__ int s_last_idx;
  if (tid == 0) {
    if (b < num_reqs) {
      const int rslot = idx_map[b];
      const int q0 = qsl[b];
      const int q1 = qsl[b + 1];
      const int rej = num_rejected[b];
      const int last_idx = max(q0, q1 - 1 - rej);
      const int clen = max(0, seq_lens[b] - rej);
      int pos_last = POS64 ? (int)reinterpret_cast<const int64_t*>(positions)[(int64_t)last_idx * pos_stride]
                           : reinterpret_cast<const int32_t*>(positions)[(int64_t)last_idx * pos_stride];
      int apos = pos_last + 1;
      if (apos < 0) apos = 0;
      if (apos >= max_model_len) apos = max_model_len - 1;
      int tok;
      if (num_sampled[b] > 0) {
        tok = LS64 ? (int)reinterpret_cast<const int64_t*>(last_sampled)[rslot]
                   : reinterpret_cast<const int32_t*>(last_sampled)[rslot];
      } else {
        tok = NP64 ? (int)reinterpret_cast<const int64_t*>(next_prefill)[rslot]
                   : reinterpret_cast<const int32_t*>(next_prefill)[rslot];
      }
      out_anchor_tok[b] = tok;
      out_ctx_len[b] = clen;
#pragma unroll
      for (int s = 0; s < BLK; s++) out_ctx_len_6[b * BLK + s] = clen;
      out_anchor_pos[b] = apos;
      out_req_slots[b] = rslot;
      s_last_idx = last_idx;
    } else {
      out_anchor_tok[b] = 0;
      out_ctx_len[b] = 0;
#pragma unroll
      for (int s = 0; s < BLK; s++) out_ctx_len_6[b * BLK + s] = 0;
      out_anchor_pos[b] = 0;
      out_req_slots[b] = -1;
      s_last_idx = -1;
    }
  }
  __syncthreads();

  const int lidx = s_last_idx;
  uint4* dst = reinterpret_cast<uint4*>(out_prev_hidden + (int64_t)b * H_DIM);
  constexpr int N_VEC = H_DIM / 8;   // 640 uint4s
  if (lidx >= 0) {
    const uint4* src = reinterpret_cast<const uint4*>(last_hidden + (int64_t)lidx * H_DIM);
    for (int i = tid; i < N_VEC; i += 256) dst[i] = src[i];
  } else {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = tid; i < N_VEC; i += 256) dst[i] = z;
  }
}

// =================================================================================================
// 2. dfly2_stage_a_prep: rank-local slice of [enorm(e0), hnorm(prev_hidden)] -> [B, 2560]
//    Matches reference_model.py::Q27MTPFusion: cat([enorm(anchor_embeds), hnorm(prev_hidden)], -1)
// =================================================================================================
__global__ void __launch_bounds__(256)
k_stage_a_prep(const __half* __restrict__ prev_hidden, const __half* __restrict__ e0,
               const __half* __restrict__ hnorm_w, const __half* __restrict__ enorm_w,
               __half* __restrict__ eh_in_local, int tp_rank, float eps) {
  __shared__ float sm[8];
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const bool use_e = (tp_rank < 2);
  const int col_off = (tp_rank & 1) * H_HALF;
  const __half* x = (use_e ? e0 : prev_hidden) + (int64_t)b * H_DIM;
  const __half* w = use_e ? enorm_w : hnorm_w;

  float ss = 0.f;
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float2 v = __half22float2(reinterpret_cast<const __half2*>(x)[i]);
    ss = fmaf(v.x, v.x, fmaf(v.y, v.y, ss));
  }
  const float inv = rsqrtf(bsum256(ss, sm) * (1.0f / (float)H_DIM) + eps);

  __half2* dst = reinterpret_cast<__half2*>(eh_in_local + (int64_t)b * H_HALF);
  const __half2* x2 = reinterpret_cast<const __half2*>(x + col_off);
  const __half2* w2 = reinterpret_cast<const __half2*>(w + col_off);
  for (int i = tid; i < H_HALF / 2; i += 256) {
    const float2 xf = __half22float2(x2[i]);
    const float2 wf = __half22float2(w2[i]);
    dst[i] = __floats2half2_rn(xf.x * inv * (1.0f + wf.x), xf.y * inv * (1.0f + wf.y));
  }
}

// =================================================================================================
// 3. dfly2_stage_a_fuse_norm: build h_block [B*6, 5120] + residual_f32 [B*6, 5120] + ln0(h_block)
//    Slot 0 = anchor_emb + eh_out * gate_w (exact Q27MTPFusion, raw gate, NO sigmoid!)
//    Slots 1..5 = mask_embedding.
// =================================================================================================
__global__ void __launch_bounds__(256)
k_stage_a_fuse_norm(const __half* __restrict__ anchor_emb, const __half* __restrict__ eh_out,
                    const __half* __restrict__ gate_w, const __half* __restrict__ mask_emb,
                    const __half* __restrict__ ln0_w, __half* __restrict__ h_block,
                    float* __restrict__ res_f32, __half* __restrict__ x_norm, float eps) {
  __shared__ float sm[8];
  const int row = blockIdx.x;   // b * 6 + s
  const int b = row / BLK;
  const int s = row % BLK;
  const int tid = threadIdx.x;

  float* rdst = res_f32 + (int64_t)row * H_DIM;
  __half2* hdst = reinterpret_cast<__half2*>(h_block + (int64_t)row * H_DIM);
  __half2* ndst = reinterpret_cast<__half2*>(x_norm + (int64_t)row * H_DIM);

  float ss = 0.f;
  if (s == 0) {
    const __half2* ae = reinterpret_cast<const __half2*>(anchor_emb + (int64_t)b * H_DIM);
    const __half2* eh = reinterpret_cast<const __half2*>(eh_out + (int64_t)b * H_DIM);
    const __half2* gw = reinterpret_cast<const __half2*>(gate_w);
    for (int i = tid; i < H_DIM / 2; i += 256) {
      const float2 a = __half22float2(ae[i]);
      const float2 e = __half22float2(eh[i]);
      const float2 g = __half22float2(gw[i]);
      const float v0 = fmaf(e.x, g.x, a.x);
      const float v1 = fmaf(e.y, g.y, a.y);
      const __half2 hv = __floats2half2_rn(v0, v1);
      hdst[i] = hv;
      const float2 hvf = __half22float2(hv);
      rdst[2 * i] = hvf.x;
      rdst[2 * i + 1] = hvf.y;
      ss = fmaf(hvf.x, hvf.x, fmaf(hvf.y, hvf.y, ss));
    }
  } else {
    const __half2* mk = reinterpret_cast<const __half2*>(mask_emb);
    for (int i = tid; i < H_DIM / 2; i += 256) {
      const __half2 hv = mk[i];
      hdst[i] = hv;
      const float2 hvf = __half22float2(hv);
      rdst[2 * i] = hvf.x;
      rdst[2 * i + 1] = hvf.y;
      ss = fmaf(hvf.x, hvf.x, fmaf(hvf.y, hvf.y, ss));
    }
  }

  const float inv = rsqrtf(bsum256(ss, sm) * (1.0f / (float)H_DIM) + eps);
  const __half2* w2 = reinterpret_cast<const __half2*>(ln0_w);
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float x0 = rdst[2 * i], x1 = rdst[2 * i + 1];
    const float2 wf = __half22float2(w2[i]);
    ndst[i] = __floats2half2_rn(x0 * inv * (1.0f + wf.x), x1 * inv * (1.0f + wf.y));
  }
}

// =================================================================================================
// 4. dfly2_stage_b_qk_prep: split qkv [B*6, 3584] -> q_out [B*6, 6, 256], k_blk [B, 6, 256], v_blk [B, 6, 256]
//    Applies zero-centered Gemma q_norm/k_norm + partial NeoX RoPE (0:64, theta=1e7) at pos = anchor_pos[b] + s.
// =================================================================================================
__global__ void __launch_bounds__(224)
k_stage_b_qk_prep(const __half* __restrict__ qkv, const int* __restrict__ anchor_pos,
                  const __half* __restrict__ q_norm_w, const __half* __restrict__ k_norm_w,
                  __half* __restrict__ q_out, __half* __restrict__ k_blk, __half* __restrict__ v_blk,
                  float eps) {
  // 7 warps per token row (0..5 = Q heads, 6 = K + V head)
  const int row = blockIdx.x;   // b * 6 + s
  const int b = row / BLK;
  const int s = row % BLK;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float pos = (float)(anchor_pos[b] + s);

  const __half* row_ptr = qkv + (int64_t)row * 3584;
  if (warp < NQ) {
    // Q head `warp` sits at [warp * 512 .. warp * 512 + 255] (interleaved [q, gate])
    const __half* src = row_ptr + warp * 512;
    __half* dst = q_out + ((int64_t)row * NQ + warp) * HD;

    // Each lane owns 8 dims: d = lane + 32 * k (k = 0..7) so pair (lane, lane+32) is in the SAME lane!
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = __half2float(src[d]);
      ss = fmaf(v[k], v[k], ss);
    }
    const float inv = rsqrtf(wsum(ss) * (1.0f / (float)HD) + eps);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = v[k] * inv * (1.0f + __half2float(q_norm_w[d]));
    }
    // Partial NeoX RoPE on k = 0 (dims 0..31) and k = 1 (dims 32..63)
    float sn, cs;
    sincosf(pos * D_INV_FREQ[lane], &sn, &cs);
    const float r0 = v[0] * cs - v[1] * sn;
    const float r1 = v[1] * cs + v[0] * sn;
    v[0] = r0;
    v[1] = r1;
#pragma unroll
    for (int k = 0; k < 8; k++) dst[k * 32 + lane] = __float2half_rn(v[k]);
  } else {
    // Warp 6: K at [3072..3327], V at [3328..3583]
    const __half* ksrc = row_ptr + 3072;
    const __half* vsrc = row_ptr + 3328;
    __half* kdst = k_blk + (int64_t)row * HD;
    __half* vdst = v_blk + (int64_t)row * HD;

    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = __half2float(ksrc[d]);
      ss = fmaf(v[k], v[k], ss);
    }
    const float inv = rsqrtf(wsum(ss) * (1.0f / (float)HD) + eps);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = v[k] * inv * (1.0f + __half2float(k_norm_w[d]));
    }
    float sn, cs;
    sincosf(pos * D_INV_FREQ[lane], &sn, &cs);
    const float r0 = v[0] * cs - v[1] * sn;
    const float r1 = v[1] * cs + v[0] * sn;
    v[0] = r0;
    v[1] = r1;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      kdst[d] = __float2half_rn(v[k]);
      vdst[d] = vsrc[d];
    }
  }
}

// =================================================================================================
// 5. dfly2_gla_q_absorb: convert post-RoPE draft q [N, 6, 256] -> q_cat [N, 6, R_LAT + 64]
//    Un-rotates dropped pair 31 (dims 31, 63) so q_nope = [unrot_31, unrot_63, q[64..255]] matches
//    target GLA-g4's k_up_sc_cm [194, R_LAT] exactly, and copies kept RoPE dims [0..30, 32..62, 0, 0].
// =================================================================================================
template <int R_LAT>
__global__ void __launch_bounds__(192)
k_gla_q_absorb(const __half* __restrict__ q_in, const int* __restrict__ anchor_pos, int stride_blk, int slot_fixed,
               const __half* __restrict__ w_k_up_sc, __half* __restrict__ q_cat) {
  constexpr int ROW_B = R_LAT + 64;
  constexpr int QN_STRIDE = 200;
  __shared__ __align__(16) __half q_nope_sm[NQ * QN_STRIDE];

  const int n = blockIdx.x;
  const int by = blockIdx.y;   // 0 .. R_LAT/64 - 1
  const int h = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const __half* qh = q_in + ((int64_t)n * NQ + h) * HD;
  __half* qn = q_nope_sm + h * QN_STRIDE;

  // Compute exact position to un-rotate pair 31
  const int b = (stride_blk > 0) ? (n / stride_blk) : n;
  const int s = (stride_blk > 0) ? (n % stride_blk) : slot_fixed;
  const float pos = (float)(anchor_pos[b] + s);

  if (lane == 0) {
    float sn, cs;
    sincosf(pos * D_INV_FREQ[31], &sn, &cs);
    const float y1 = __half2float(qh[31]);
    const float y2 = __half2float(qh[63]);
    qn[0] = __float2half_rn(y1 * cs + y2 * sn);
    qn[1] = __float2half_rn(y2 * cs - y1 * sn);
  }
  for (int i = lane; i < 192; i += 32) qn[2 + i] = qh[64 + i];
  __syncwarp(0xffffffffu);

  __half* dst = q_cat + ((int64_t)n * NQ + h) * ROW_B;
  if (by == 0) {
    if (lane < 31) {
      dst[R_LAT + lane] = qh[lane];
      dst[R_LAT + 31 + lane] = qh[32 + lane];
    } else {
      dst[R_LAT + 62] = __float2half_rn(0.f);
      dst[R_LAT + 63] = __float2half_rn(0.f);
    }
  }

  const int col = by * 64 + 2 * lane;
  float a0 = 0.f, a1 = 0.f;
  const __half* wcol = w_k_up_sc + col;
#pragma unroll 4
  for (int k = 0; k < GLA_NOPE_DIM; k++) {
    const float qv = __half2float(qn[k]);
    const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(wcol + (int64_t)k * R_LAT));
    a0 = fmaf(qv, wf.x, a0);
    a1 = fmaf(qv, wf.y, a1);
  }
  *reinterpret_cast<__half2*>(dst + col) = __floats2half2_rn(a0, a1);
}

// =================================================================================================
// 6. dfly2_stage_b_merge: exact 6x6 ephemeral block online-softmax merge + sigmoid(gate)
//    Supports Exact mode (R_LAT == 0: reads o_ctx [B*6, 6, 256]) and GLA mode (R_LAT in {192, 256}:
//    fuses o_lat [B*6, 6, R_LAT] @ w_v_up_cm [R_LAT, 256] in-kernel!).
// =================================================================================================
template <int R_LAT>
__global__ void __launch_bounds__(192)
k_stage_b_merge(const __half* __restrict__ o_in, const float* __restrict__ lse_ctx,
                const __half* __restrict__ q_in, const __half* __restrict__ k_blk,
                const __half* __restrict__ v_blk, const __half* __restrict__ qkv,
                const __half* __restrict__ w_v_up_cm, __half* __restrict__ out, float scale) {
  // Grid: (B, NQ), 6 warps per CTA (warp qi owns query slot qi in 0..5)
  __shared__ __align__(16) __half sk[BLK * HD];
  __shared__ __align__(16) __half sv[BLK * HD];
  __shared__ __align__(16) __half solat[R_LAT > 0 ? BLK * R_LAT : 1];

  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int qi = tid >> 5;
  const int lane = tid & 31;

  // Cooperatively stage k_blk[b, 0..5, 0..255] and v_blk[b, 0..5, 0..255]
  const uint4* kg = reinterpret_cast<const uint4*>(k_blk + (int64_t)b * BLK * HD);
  const uint4* vg = reinterpret_cast<const uint4*>(v_blk + (int64_t)b * BLK * HD);
  uint4* ks = reinterpret_cast<uint4*>(sk);
  uint4* vs = reinterpret_cast<uint4*>(sv);
  for (int i = tid; i < BLK * (HD / 8); i += 192) {
    ks[i] = kg[i];
    vs[i] = vg[i];
  }
  const int row = b * BLK + qi;
  if constexpr (R_LAT > 0) {
    const __half* ol_src = o_in + ((int64_t)row * NQ + h) * R_LAT;
    __half* ol_dst = solat + qi * R_LAT;
    for (int d = lane; d < R_LAT; d += 32) ol_dst[d] = ol_src[d];
  }
  __syncthreads();

  // Load query q[row, h, :] (8 floats per lane)
  const __half* qh = q_in + ((int64_t)row * NQ + h) * HD;
  float qv[8];
#pragma unroll
  for (int k = 0; k < 8; k++) qv[k] = __half2float(qh[k * 32 + lane]);

  // 6 dot products against k_blk[0..5]
  float sc[BLK];
  float m_blk = -INFINITY;
#pragma unroll
  for (int j = 0; j < BLK; j++) {
    float dot = 0.f;
    const __half* kj = sk + j * HD;
#pragma unroll
    for (int k = 0; k < 8; k++) dot = fmaf(qv[k], __half2float(kj[k * 32 + lane]), dot);
    sc[j] = wsum(dot) * scale;
    m_blk = fmaxf(m_blk, sc[j]);
  }

  float l_blk = 0.f;
#pragma unroll
  for (int j = 0; j < BLK; j++) l_blk += expf(sc[j] - m_blk);
  const float lse_blk = m_blk + logf(l_blk);

  const float lse_c = lse_ctx[(int64_t)row * NQ + h];
  const float m_all = fmaxf(lse_c, lse_blk);
  const float w_c = (lse_c > -1e30f) ? expf(lse_c - m_all) : 0.f;
  float w_j[BLK];
  float Z = w_c;
#pragma unroll
  for (int j = 0; j < BLK; j++) {
    w_j[j] = expf(sc[j] - m_all);
    Z += w_j[j];
  }
  const float inv_Z = (Z > 0.f) ? (1.0f / Z) : 0.f;
  const float ac = w_c * inv_Z;
#pragma unroll
  for (int j = 0; j < BLK; j++) w_j[j] *= inv_Z;

  // Compute merged output + sigmoid(gate) -> out[row, h * 256 + d]
  const __half* gptr = qkv + (int64_t)row * 3584 + h * 512 + 256;
  __half* dst = out + (int64_t)row * (NQ * HD) + h * HD;

#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int d = k * 32 + lane;
    float oc;
    if constexpr (R_LAT == 0) {
      oc = __half2float(o_in[((int64_t)row * NQ + h) * HD + d]);
    } else {
      const __half* ol = solat + qi * R_LAT;
      float acc = 0.f;
#pragma unroll 4
      for (int r = 0; r < R_LAT; r++) acc = fmaf(__half2float(ol[r]), __half2float(w_v_up_cm[(int64_t)r * HD + d]), acc);
      oc = acc;
    }
    float val = ac * oc;
#pragma unroll
    for (int j = 0; j < BLK; j++) val = fmaf(w_j[j], __half2float(sv[j * HD + d]), val);
    val *= sigmoid_f(__half2float(gptr[d]));
    dst[d] = __float2half_rn(val);
  }
}

// =================================================================================================
// 7. dfly2_stage_c_prep: inp [B, 10240] = [hidden_norm(h_block[b*6 + slot]), embed_norm(e_prev[b])]
// =================================================================================================
__global__ void __launch_bounds__(256)
k_stage_c_prep(const __half* __restrict__ h_block, const __half* __restrict__ e_prev,
               const __half* __restrict__ h_norm_w, const __half* __restrict__ e_norm_w,
               __half* __restrict__ inp, int slot, float eps) {
  __shared__ float sm[8];
  const int b = blockIdx.x;
  const int is_e = blockIdx.y;   // 0 = hidden, 1 = embed
  const int tid = threadIdx.x;

  const __half* src = is_e ? (e_prev + (int64_t)b * H_DIM) : (h_block + (int64_t)(b * BLK + slot) * H_DIM);
  const __half* w = is_e ? e_norm_w : h_norm_w;
  __half2* dst = reinterpret_cast<__half2*>(inp + (int64_t)b * (2 * H_DIM) + (is_e ? H_DIM : 0));

  float ss = 0.f;
  const __half2* s2 = reinterpret_cast<const __half2*>(src);
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float2 v = __half22float2(s2[i]);
    ss = fmaf(v.x, v.x, fmaf(v.y, v.y, ss));
  }
  const float inv = rsqrtf(bsum256(ss, sm) * (1.0f / (float)H_DIM) + eps);
  const __half2* w2 = reinterpret_cast<const __half2*>(w);
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float2 xf = __half22float2(s2[i]);
    const float2 wf = __half22float2(w2[i]);
    dst[i] = __floats2half2_rn(xf.x * inv * (1.0f + wf.x), xf.y * inv * (1.0f + wf.y));
  }
}

// =================================================================================================
// 8. dfly2_stage_c_slot0_kv: k_blk_norm + RoPE on slot 0 kv [B, 512] -> blk_k/v_cache[b, 0, :]
// =================================================================================================
__global__ void __launch_bounds__(32)
k_stage_c_slot0_kv(const __half* __restrict__ kv0, const int* __restrict__ anchor_pos,
                   const __half* __restrict__ k_blk_norm_w, __half* __restrict__ blk_k,
                   __half* __restrict__ blk_v, float eps) {
  const int b = blockIdx.x;
  const int lane = threadIdx.x;
  const float pos = (float)anchor_pos[b];
  const __half* ksrc = kv0 + (int64_t)b * 512;
  const __half* vsrc = ksrc + HD;
  __half* kdst = blk_k + (int64_t)(b * BLK + 0) * HD;
  __half* vdst = blk_v + (int64_t)(b * BLK + 0) * HD;

  float v[8];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int d = k * 32 + lane;
    v[k] = __half2float(ksrc[d]);
    ss = fmaf(v[k], v[k], ss);
  }
  const float inv = rsqrtf(wsum(ss) * (1.0f / (float)HD) + eps);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int d = k * 32 + lane;
    v[k] = v[k] * inv * (1.0f + __half2float(k_blk_norm_w[d]));
  }
  float sn, cs;
  sincosf(pos * D_INV_FREQ[lane], &sn, &cs);
  const float r0 = v[0] * cs - v[1] * sn;
  const float r1 = v[1] * cs + v[0] * sn;
  v[0] = r0;
  v[1] = r1;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int d = k * 32 + lane;
    kdst[d] = __float2half_rn(v[k]);
    vdst[d] = vsrc[d];
  }
}

// =================================================================================================
// 9. dfly2_stage_c_qk_silu: split mega [B, 4608] -> silu(gate)*up -> out_in[:, 0:1280],
//    q_norm+RoPE -> q_out [B, 6, 256], k_blk_norm+RoPE -> blk_k/v[b, slot, :]
// =================================================================================================
__global__ void __launch_bounds__(256)
k_stage_c_qk_silu(const __half* __restrict__ mega, const int* __restrict__ anchor_pos,
                  const __half* __restrict__ q_norm_w, const __half* __restrict__ k_blk_norm_w,
                  __half* __restrict__ q_out, __half* __restrict__ blk_k, __half* __restrict__ blk_v,
                  __half* __restrict__ out_in, int slot, float eps) {
  // 8 warps per request b:
  //   warps 0..5: Q heads (offset 2560 + h * 256)
  //   warp 6:     K/V block (offset 4096 / 4352)
  //   warp 7:     silu(gate) * up (1280 elements -> 40 elements/lane)
  const int b = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float pos = (float)(anchor_pos[b] + slot);
  const __half* mrow = mega + (int64_t)b * 4608;

  if (warp < NQ) {
    const __half* src = mrow + 2560 + warp * HD;
    __half* dst = q_out + ((int64_t)b * NQ + warp) * HD;
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = __half2float(src[d]);
      ss = fmaf(v[k], v[k], ss);
    }
    const float inv = rsqrtf(wsum(ss) * (1.0f / (float)HD) + eps);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = v[k] * inv * (1.0f + __half2float(q_norm_w[d]));
    }
    float sn, cs;
    sincosf(pos * D_INV_FREQ[lane], &sn, &cs);
    const float r0 = v[0] * cs - v[1] * sn;
    const float r1 = v[1] * cs + v[0] * sn;
    v[0] = r0;
    v[1] = r1;
#pragma unroll
    for (int k = 0; k < 8; k++) dst[k * 32 + lane] = __float2half_rn(v[k]);
  } else if (warp == 6) {
    const __half* ksrc = mrow + 4096;
    const __half* vsrc = mrow + 4352;
    __half* kdst = blk_k + (int64_t)(b * BLK + slot) * HD;
    __half* vdst = blk_v + (int64_t)(b * BLK + slot) * HD;
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = __half2float(ksrc[d]);
      ss = fmaf(v[k], v[k], ss);
    }
    const float inv = rsqrtf(wsum(ss) * (1.0f / (float)HD) + eps);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      v[k] = v[k] * inv * (1.0f + __half2float(k_blk_norm_w[d]));
    }
    float sn, cs;
    sincosf(pos * D_INV_FREQ[lane], &sn, &cs);
    const float r0 = v[0] * cs - v[1] * sn;
    const float r1 = v[1] * cs + v[0] * sn;
    v[0] = r0;
    v[1] = r1;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int d = k * 32 + lane;
      kdst[d] = __float2half_rn(v[k]);
      vdst[d] = vsrc[d];
    }
  } else {
    const __half2* g2 = reinterpret_cast<const __half2*>(mrow);
    const __half2* u2 = reinterpret_cast<const __half2*>(mrow + HC_MLP_I);
    __half2* dst2 = reinterpret_cast<__half2*>(out_in + (int64_t)b * HC_MLP_I);
    for (int i = lane; i < HC_MLP_I / 2; i += 32) {
      const float2 gf = __half22float2(g2[i]);
      const float2 uf = __half22float2(u2[i]);
      const float y0 = (gf.x / (1.0f + expf(-gf.x))) * uf.x;
      const float y1 = (gf.y / (1.0f + expf(-gf.y))) * uf.y;
      dst2[i] = __floats2half2_rn(y0, y1);
    }
  }
}

// =================================================================================================
// 10. dfly2_stage_c_combine: causal 1 x (slot+1) block attention with block_rel_bias + block_sink,
//     gated by tanh(gate_raw + block_gate_bias[:, slot]), added to ctx_out (Exact or GLA v_up),
//     written straight into out_in[:, 1280:2816].
// =================================================================================================
template <int R_LAT>
__global__ void __launch_bounds__(192)
k_stage_c_combine(const __half* __restrict__ o_in, const __half* __restrict__ q_in,
                  const __half* __restrict__ blk_k, const __half* __restrict__ blk_v,
                  const __half* __restrict__ gate_raw, const __half* __restrict__ block_gate_bias,
                  const __half* __restrict__ block_rel_bias, const __half* __restrict__ block_sink,
                  const __half* __restrict__ w_v_up_cm, __half* __restrict__ out_in,
                  int slot, int bias_stride, float scale) {
  // Grid: (B,), 6 warps (one per local Q head h = 0..5)
  __shared__ __align__(16) __half sk[BLK * HD];
  __shared__ __align__(16) __half sv[BLK * HD];
  __shared__ __align__(16) __half solat[R_LAT > 0 ? NQ * R_LAT : 1];

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int h = tid >> 5;
  const int lane = tid & 31;
  const int n_vis = slot + 1;

  // Stage visible blk_k/v[b, 0..slot, 0..255]
  const uint4* kg = reinterpret_cast<const uint4*>(blk_k + (int64_t)b * BLK * HD);
  const uint4* vg = reinterpret_cast<const uint4*>(blk_v + (int64_t)b * BLK * HD);
  uint4* ks = reinterpret_cast<uint4*>(sk);
  uint4* vs = reinterpret_cast<uint4*>(sv);
  for (int i = tid; i < n_vis * (HD / 8); i += 192) {
    ks[i] = kg[i];
    vs[i] = vg[i];
  }
  if constexpr (R_LAT > 0) {
    const __half* ol_src = o_in + ((int64_t)b * NQ + h) * R_LAT;
    __half* ol_dst = solat + h * R_LAT;
    for (int d = lane; d < R_LAT; d += 32) ol_dst[d] = ol_src[d];
  }
  __syncthreads();

  const __half* qh = q_in + ((int64_t)b * NQ + h) * HD;
  float qv[8];
#pragma unroll
  for (int k = 0; k < 8; k++) qv[k] = __half2float(qh[k * 32 + lane]);

  const float sink = __half2float(block_sink[h]);
  float sc[BLK];
  float m_all = sink;
  for (int j = 0; j < n_vis; j++) {
    float dot = 0.f;
    const __half* kj = sk + j * HD;
#pragma unroll
    for (int k = 0; k < 8; k++) dot = fmaf(qv[k], __half2float(kj[k * 32 + lane]), dot);
    const float rel = __half2float(block_rel_bias[h * bias_stride + (slot - j)]);
    sc[j] = wsum(dot) * scale + rel;
    m_all = fmaxf(m_all, sc[j]);
  }
  float Z = expf(sink - m_all);
  float w_j[BLK];
  for (int j = 0; j < n_vis; j++) {
    w_j[j] = expf(sc[j] - m_all);
    Z += w_j[j];
  }
  const float inv_Z = 1.0f / Z;
  for (int j = 0; j < n_vis; j++) w_j[j] *= inv_Z;

  const float g_val = tanhf(__half2float(gate_raw[(int64_t)b * NQ + h]) +
                            __half2float(block_gate_bias[h * bias_stride + slot]));
  __half* dst = out_in + (int64_t)b * (NQ * HD) + h * HD;

#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int d = k * 32 + lane;
    float oc;
    if constexpr (R_LAT == 0) {
      oc = __half2float(o_in[((int64_t)b * NQ + h) * HD + d]);
    } else {
      const __half* ol = solat + h * R_LAT;
      float acc = 0.f;
#pragma unroll 4
      for (int r = 0; r < R_LAT; r++) acc = fmaf(__half2float(ol[r]), __half2float(w_v_up_cm[(int64_t)r * HD + d]), acc);
      oc = acc;
    }
    float br = 0.f;
    for (int j = 0; j < n_vis; j++) br = fmaf(w_j[j], __half2float(sv[j * HD + d]), br);
    dst[d] = __float2half_rn(oc + g_val * br);
  }
}

// =================================================================================================
// 11. dfly2_add_slot_residual: h_out [B, 5120] = h_block[b*6 + slot, :] + delta [B, 5120]
// =================================================================================================
__global__ void __launch_bounds__(256)
k_add_slot_residual(const __half* __restrict__ h_block, const __half* __restrict__ delta,
                    __half* __restrict__ out, int slot) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const __half2* hs = reinterpret_cast<const __half2*>(h_block + (int64_t)(b * BLK + slot) * H_DIM);
  const __half2* ds = reinterpret_cast<const __half2*>(delta + (int64_t)b * H_DIM);
  __half2* os = reinterpret_cast<__half2*>(out + (int64_t)b * H_DIM);
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float2 a = __half22float2(hs[i]);
    const float2 c = __half22float2(ds[i]);
    os[i] = __floats2half2_rn(a.x + c.x, a.y + c.y);
  }
}

// =================================================================================================
// 12. dfly2_gemma_rmsnorm_res: residual += x; out = gemma_rmsnorm(residual, w) (FP32 residual)
// =================================================================================================
__global__ void __launch_bounds__(256)
k_gemma_rmsnorm_res(const __half* __restrict__ x, float* __restrict__ res, const __half* __restrict__ w,
                    __half* __restrict__ out, float eps) {
  __shared__ float sm[8];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __half2* x2 = reinterpret_cast<const __half2*>(x + (int64_t)row * H_DIM);
  float* rr = res + (int64_t)row * H_DIM;
  __half2* o2 = reinterpret_cast<__half2*>(out + (int64_t)row * H_DIM);
  const __half2* w2 = reinterpret_cast<const __half2*>(w);

  float ss = 0.f;
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float2 xf = __half22float2(x2[i]);
    const float r0 = rr[2 * i] + xf.x;
    const float r1 = rr[2 * i + 1] + xf.y;
    rr[2 * i] = r0;
    rr[2 * i + 1] = r1;
    ss = fmaf(r0, r0, fmaf(r1, r1, ss));
  }
  const float inv = rsqrtf(bsum256(ss, sm) * (1.0f / (float)H_DIM) + eps);
  for (int i = tid; i < H_DIM / 2; i += 256) {
    const float2 wf = __half22float2(w2[i]);
    o2[i] = __floats2half2_rn(rr[2 * i] * inv * (1.0f + wf.x), rr[2 * i + 1] * inv * (1.0f + wf.y));
  }
}

}  // namespace

// =================================================================================================
// Python bindings
// =================================================================================================

void dfly2_extract_inputs(torch::Tensor qsl, torch::Tensor seq_lens, torch::Tensor idx_map, torch::Tensor positions,
                          torch::Tensor last_hidden, torch::Tensor num_sampled, torch::Tensor num_rejected,
                          torch::Tensor last_sampled, torch::Tensor next_prefill, torch::Tensor out_anchor_tok,
                          torch::Tensor out_ctx_len, torch::Tensor out_ctx_len_6, torch::Tensor out_anchor_pos,
                          torch::Tensor out_prev_hidden, torch::Tensor out_req_slots, int64_t num_reqs,
                          int64_t B_pad, int64_t max_model_len) {
  if (B_pad <= 0) return;
  const at::cuda::CUDAGuard guard(last_hidden.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int pos_stride = (positions.dim() == 2) ? (int)positions.stride(1) : (int)positions.stride(0);
  const void* pos_ptr = positions.data_ptr();
  const bool pos64 = (positions.dtype() == torch::kLong);
  const bool ls64 = (last_sampled.dtype() == torch::kLong);
  const bool np64 = (next_prefill.dtype() == torch::kLong);
  const void* np_ptr = next_prefill.data_ptr();   // row 0 of [num_lookahead, max_reqs]

#define LAUNCH_EX(P64, L64, N64) \
  k_extract_inputs<P64, L64, N64><<<(unsigned)B_pad, 256, 0, st>>>( \
      qsl.data_ptr<int>(), seq_lens.data_ptr<int>(), idx_map.data_ptr<int>(), pos_ptr, pos_stride, \
      reinterpret_cast<const __half*>(last_hidden.data_ptr<at::Half>()), num_sampled.data_ptr<int>(), \
      num_rejected.data_ptr<int>(), last_sampled.data_ptr(), np_ptr, out_anchor_tok.data_ptr<int>(), \
      out_ctx_len.data_ptr<int>(), out_ctx_len_6.data_ptr<int>(), out_anchor_pos.data_ptr<int>(), \
      reinterpret_cast<__half*>(out_prev_hidden.data_ptr<at::Half>()), out_req_slots.data_ptr<int>(), \
      (int)num_reqs, (int)B_pad, (int)max_model_len)

  const int key = (pos64 ? 4 : 0) | (ls64 ? 2 : 0) | (np64 ? 1 : 0);
  switch (key) {
    case 0: LAUNCH_EX(false, false, false); break;
    case 1: LAUNCH_EX(false, false, true); break;
    case 2: LAUNCH_EX(false, true, false); break;
    case 3: LAUNCH_EX(false, true, true); break;
    case 4: LAUNCH_EX(true, false, false); break;
    case 5: LAUNCH_EX(true, false, true); break;
    case 6: LAUNCH_EX(true, true, false); break;
    case 7: LAUNCH_EX(true, true, true); break;
  }
#undef LAUNCH_EX
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_extract_inputs failed");
}

void dfly2_stage_a_prep(torch::Tensor prev_hidden, torch::Tensor e0, torch::Tensor hnorm_w, torch::Tensor enorm_w,
                        torch::Tensor eh_in_local, int64_t tp_rank, double eps) {
  const int B = (int)prev_hidden.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(prev_hidden.device());
  k_stage_a_prep<<<(unsigned)B, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(e0.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(hnorm_w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(enorm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(eh_in_local.data_ptr<at::Half>()), (int)tp_rank, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_a_prep failed");
}

void dfly2_stage_a_fuse_norm(torch::Tensor prev_hidden, torch::Tensor eh_out, torch::Tensor gate_w,
                             torch::Tensor mask_emb, torch::Tensor ln0_w, torch::Tensor h_block,
                             torch::Tensor res_f32, torch::Tensor x_norm, double eps) {
  const int N = (int)h_block.size(0);
  if (N <= 0) return;
  const at::cuda::CUDAGuard guard(prev_hidden.device());
  k_stage_a_fuse_norm<<<(unsigned)N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(eh_out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(gate_w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(mask_emb.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(ln0_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(h_block.data_ptr<at::Half>()),
      res_f32.data_ptr<float>(),
      reinterpret_cast<__half*>(x_norm.data_ptr<at::Half>()), (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_a_fuse_norm failed");
}

void dfly2_stage_b_qk_prep(torch::Tensor qkv, torch::Tensor anchor_pos, torch::Tensor q_norm_w,
                           torch::Tensor k_norm_w, torch::Tensor q_out, torch::Tensor k_blk,
                           torch::Tensor v_blk, double eps) {
  const int N = (int)qkv.size(0);
  if (N <= 0) return;
  const at::cuda::CUDAGuard guard(qkv.device());
  k_stage_b_qk_prep<<<(unsigned)N, 224, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), anchor_pos.data_ptr<int>(),
      reinterpret_cast<const __half*>(q_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(k_blk.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(v_blk.data_ptr<at::Half>()), (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_b_qk_prep failed");
}

void dfly2_gla_q_absorb(torch::Tensor q_in, torch::Tensor anchor_pos, int64_t stride_blk, int64_t slot_fixed,
                        torch::Tensor w_k_up_sc, torch::Tensor q_cat) {
  const int N = (int)q_in.size(0);
  if (N <= 0) return;
  const int r_lat = (int)w_k_up_sc.size(1);
  TORCH_CHECK(r_lat == 192 || r_lat == 256, "dfly2_gla_q_absorb: r_lat must be 192 or 256");
  const at::cuda::CUDAGuard guard(q_in.device());
  dim3 grid((unsigned)N, (unsigned)(r_lat / 64));
  auto st = at::cuda::getCurrentCUDAStream();
  if (r_lat == 192) {
    k_gla_q_absorb<192><<<grid, 192, 0, st>>>(
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()), anchor_pos.data_ptr<int>(),
        (int)stride_blk, (int)slot_fixed, reinterpret_cast<const __half*>(w_k_up_sc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(q_cat.data_ptr<at::Half>()));
  } else {
    k_gla_q_absorb<256><<<grid, 192, 0, st>>>(
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()), anchor_pos.data_ptr<int>(),
        (int)stride_blk, (int)slot_fixed, reinterpret_cast<const __half*>(w_k_up_sc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(q_cat.data_ptr<at::Half>()));
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_gla_q_absorb failed");
}

void dfly2_stage_b_merge(torch::Tensor o_in, torch::Tensor lse_ctx, torch::Tensor q_in, torch::Tensor k_blk,
                         torch::Tensor v_blk, torch::Tensor qkv, c10::optional<torch::Tensor> w_v_up_cm,
                         torch::Tensor out, double scale) {
  const int B = (int)k_blk.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(q_in.device());
  dim3 grid((unsigned)B, (unsigned)NQ);
  auto st = at::cuda::getCurrentCUDAStream();
  const bool gla = w_v_up_cm.has_value() && w_v_up_cm->defined() && w_v_up_cm->numel() > 0;
  const int r_lat = gla ? (int)w_v_up_cm->size(0) : 0;
  const __half* wv = gla ? reinterpret_cast<const __half*>(w_v_up_cm->data_ptr<at::Half>()) : nullptr;
  if (r_lat == 0) {
    k_stage_b_merge<0><<<grid, 192, 0, st>>>(
        reinterpret_cast<const __half*>(o_in.data_ptr<at::Half>()), lse_ctx.data_ptr<float>(),
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_blk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_blk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), nullptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (float)scale);
  } else if (r_lat == 192) {
    k_stage_b_merge<192><<<grid, 192, 0, st>>>(
        reinterpret_cast<const __half*>(o_in.data_ptr<at::Half>()), lse_ctx.data_ptr<float>(),
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_blk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_blk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), wv,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (float)scale);
  } else {
    TORCH_CHECK(r_lat == 256, "dfly2_stage_b_merge: r_lat must be 0, 192, or 256");
    k_stage_b_merge<256><<<grid, 192, 0, st>>>(
        reinterpret_cast<const __half*>(o_in.data_ptr<at::Half>()), lse_ctx.data_ptr<float>(),
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_blk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_blk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()), wv,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (float)scale);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_b_merge failed");
}

void dfly2_stage_c_prep(torch::Tensor h_block, torch::Tensor e_prev, torch::Tensor h_norm_w, torch::Tensor e_norm_w,
                        torch::Tensor inp, int64_t slot, double eps) {
  const int B = (int)e_prev.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(e_prev.device());
  dim3 grid((unsigned)B, 2u);
  k_stage_c_prep<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(h_block.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(e_prev.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(h_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(e_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(inp.data_ptr<at::Half>()), (int)slot, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_c_prep failed");
}

void dfly2_stage_c_slot0_kv(torch::Tensor kv0, torch::Tensor anchor_pos, torch::Tensor k_blk_norm_w,
                            torch::Tensor blk_k, torch::Tensor blk_v, double eps) {
  const int B = (int)kv0.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(kv0.device());
  k_stage_c_slot0_kv<<<(unsigned)B, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(kv0.data_ptr<at::Half>()), anchor_pos.data_ptr<int>(),
      reinterpret_cast<const __half*>(k_blk_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_k.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_v.data_ptr<at::Half>()), (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_c_slot0_kv failed");
}

void dfly2_stage_c_qk_silu(torch::Tensor mega, torch::Tensor anchor_pos, torch::Tensor q_norm_w,
                           torch::Tensor k_blk_norm_w, torch::Tensor q_out, torch::Tensor blk_k,
                           torch::Tensor blk_v, torch::Tensor out_in, int64_t slot, double eps) {
  const int B = (int)mega.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(mega.device());
  k_stage_c_qk_silu<<<(unsigned)B, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(mega.data_ptr<at::Half>()), anchor_pos.data_ptr<int>(),
      reinterpret_cast<const __half*>(q_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k_blk_norm_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_k.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(blk_v.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out_in.data_ptr<at::Half>()), (int)slot, (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_c_qk_silu failed");
}

void dfly2_stage_c_combine(torch::Tensor o_in, torch::Tensor q_in, torch::Tensor blk_k, torch::Tensor blk_v,
                           torch::Tensor gate_raw, torch::Tensor block_gate_bias, torch::Tensor block_rel_bias,
                           torch::Tensor block_sink, c10::optional<torch::Tensor> w_v_up_cm,
                           torch::Tensor out_in, int64_t slot, double scale) {
  const int B = (int)q_in.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(q_in.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int bias_stride = (int)block_gate_bias.size(1);
  const bool gla = w_v_up_cm.has_value() && w_v_up_cm->defined() && w_v_up_cm->numel() > 0;
  const int r_lat = gla ? (int)w_v_up_cm->size(0) : 0;
  const __half* wv = gla ? reinterpret_cast<const __half*>(w_v_up_cm->data_ptr<at::Half>()) : nullptr;
  if (r_lat == 0) {
    k_stage_c_combine<0><<<(unsigned)B, 192, 0, st>>>(
        reinterpret_cast<const __half*>(o_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(blk_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(blk_v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gate_raw.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_gate_bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_rel_bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_sink.data_ptr<at::Half>()), nullptr,
        reinterpret_cast<__half*>(out_in.data_ptr<at::Half>()), (int)slot, bias_stride, (float)scale);
  } else if (r_lat == 192) {
    k_stage_c_combine<192><<<(unsigned)B, 192, 0, st>>>(
        reinterpret_cast<const __half*>(o_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(blk_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(blk_v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gate_raw.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_gate_bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_rel_bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_sink.data_ptr<at::Half>()), wv,
        reinterpret_cast<__half*>(out_in.data_ptr<at::Half>()), (int)slot, bias_stride, (float)scale);
  } else {
    TORCH_CHECK(r_lat == 256, "dfly2_stage_c_combine: r_lat must be 0, 192, or 256");
    k_stage_c_combine<256><<<(unsigned)B, 192, 0, st>>>(
        reinterpret_cast<const __half*>(o_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(blk_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(blk_v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gate_raw.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_gate_bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_rel_bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(block_sink.data_ptr<at::Half>()), wv,
        reinterpret_cast<__half*>(out_in.data_ptr<at::Half>()), (int)slot, bias_stride, (float)scale);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_stage_c_combine failed");
}

void dfly2_add_slot_residual(torch::Tensor h_block, torch::Tensor delta, torch::Tensor out, int64_t slot) {
  const int B = (int)delta.size(0);
  if (B <= 0) return;
  const at::cuda::CUDAGuard guard(delta.device());
  k_add_slot_residual<<<(unsigned)B, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(h_block.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(delta.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)slot);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_add_slot_residual failed");
}

void dfly2_gemma_rmsnorm_res(torch::Tensor x, torch::Tensor res, torch::Tensor w, torch::Tensor out, double eps) {
  const int N = (int)x.size(0);
  if (N <= 0) return;
  const at::cuda::CUDAGuard guard(x.device());
  k_gemma_rmsnorm_res<<<(unsigned)N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), res.data_ptr<float>(),
      reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (float)eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dfly2_gemma_rmsnorm_res failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_inputs", &dfly2_extract_inputs);
  m.def("stage_a_prep", &dfly2_stage_a_prep);
  m.def("stage_a_fuse_norm", &dfly2_stage_a_fuse_norm);
  m.def("stage_b_qk_prep", &dfly2_stage_b_qk_prep);
  m.def("gla_q_absorb", &dfly2_gla_q_absorb);
  m.def("stage_b_merge", &dfly2_stage_b_merge);
  m.def("stage_c_prep", &dfly2_stage_c_prep);
  m.def("stage_c_slot0_kv", &dfly2_stage_c_slot0_kv);
  m.def("stage_c_qk_silu", &dfly2_stage_c_qk_silu);
  m.def("stage_c_combine", &dfly2_stage_c_combine);
  m.def("add_slot_residual", &dfly2_add_slot_residual);
  m.def("gemma_rmsnorm_res", &dfly2_gemma_rmsnorm_res);
}
