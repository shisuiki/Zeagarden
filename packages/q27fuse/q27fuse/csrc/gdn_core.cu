// gdn_core.cu - the whole Gated DeltaNet decode core of a Qwen3.8-27B layer in one launch (TP4 shard on V100: 4 key
// heads, 12 value heads, head dim 128, conv width 4), for pure-decode batches (one token per sequence).
// It replaces four kernels between the qkvz input projection and the output projection:
//   k_gdn_in (gdn_glue.cu)            b/a projection, fp16 weight [2 Hv][5120], fp32 sums (same order, same bits)
//   causal_conv1d_update (1Cat triton) depthwise conv of width 4 over [conv state (3 taps), x], fp16 products summed
//                                      in fp32, silu, fp16 output; conv state shifted by one token
//   gdn_decode state update            gdn_state.cu's math (= 1Cat's gdn_decode_mixed_qkv_global_state_kernel to 1e-8),
//                                      FP32 state [slots][Hv][V 128][K 128]
//   k_gdn_gated_norm (gdn_glue.cu)     RMSNormGated per head of 128 with silu(z), same order, same bits
// Work split: CTA (value head hv, token t, slice) owns NCOL x WPC state columns (WPC warps, lane l holds rows
// 4 l .. 4 l + 3 of each column: one 16-byte access per column). Every CTA issues its state loads first, then the b/a
// operands (each warp takes "virtual warps" of gdn_in's 640-thread reduction, so the sums are bitwise the same), then
// the conv operands; q/k of its key head are convolved redundantly by every warp that needs them.
// Two read-then-shift hazards cross CTAs, both resolved by "last one out" counters (no CTA ever waits):
//   - the q/k conv state of key head hq is read by all (Hv / Hk) x NS CTAs of token t; the last of them to finish its
//     conv writes the shifted state.
//   - the gated norm needs all 128 outputs of a head: every slice writes its fp16 outputs to `core`, the last slice of
//     (t, hv) to finish normalises the whole head. With one slice per head the CTA does it after a barrier.
// Counters live in a zeroed int32 workspace; the last CTA resets its counter, so the workspace stays zero between
// launches. Padded rows (t >= Tdec or state index outside [0, slots)) touch no state and write y = 0.
// (gdn_fused, the same core behind FP8 qkvz projection CTAs in one launch, was bitwise but not faster: REPORT.md point 10;
// it is in ../../attic/gdn_core_fused.cu.)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int D = 128, KH = 5120, NV = KH / 8, NVW = NV / 32;    // gdn_in: 640 threads x 8 halves = 20 warps

struct Args {
  const __half* x;      // [n][5120] layer input
  const __half* qkvz;   // [n][ldq]: q (Hk D) | k (Hk D) | v (Hv D) | z (Hv D)
  const __half* wba;    // [2 Hv][5120]: b rows, then a rows
  const __half* cw;     // [dim][4] conv weight
  __half* cst;          // conv state, per slot [3][dim]
  const float* A_log;   // [Hv]
  const __half* dtb;    // [Hv]
  float* ssm;           // per slot [Hv][128][128]
  const int32_t* idx;   // [>= Tdec] state slot per token
  const __half* nw;     // [128] norm weight
  __half* core;         // [n][Hv][128] scratch (unnormalised outputs)
  __half* y;            // [n][Hv * 128]
  int* cnt;             // [n Hk] q/k conv counters, then [n Hv] norm counters
  long ldq, cst_stride, ssm_stride;
  int slots, Tdec, n, Hk, Hv, dim;
  float scale, eps;
  int dbg;              // debug (test_gdn_core.py phases): bit 0 skip b/a, bit 1 skip the q/k conv counter, bit 2 skip
                        // the norm phase
};

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void h8(const uint4& u, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ void h4(const uint2& u, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&u);
  const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
__device__ __forceinline__ uint2 ld8(const __half* p) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ uint2 ldcg8(const __half* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ __half hx(const uint2& u, int i) { return reinterpret_cast<const __half*>(&u)[i]; }

// 1Cat's triton conv for one channel and one token: acc = sum_j fp16(x_j * w_j) in fp32, silu, fp16
__device__ __forceinline__ float conv1(__half s0, __half s1, __half s2, __half x, __half w0, __half w1, __half w2,
                                       __half w3) {
  float acc = __half2float(__hmul(s0, w0));
  acc += __half2float(__hmul(s1, w1));
  acc += __half2float(__hmul(s2, w2));
  acc += __half2float(__hmul(x, w3));
  acc = acc / (1.f + __expf(-acc));
  return __half2float(__float2half_rn(acc));
}

template <int NCOL, int WPC>
__device__ __forceinline__ void core_cta(const Args& p, const int hv, const int t, const int sl) {
  constexpr int NS = D / (NCOL * WPC), NVPW = (NVW + WPC - 1) / WPC;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
  const int c0 = (sl * WPC + warp) * NCOL;
  const int Hv = p.Hv, Hk = p.Hk, rep = Hv / Hk, hq = hv / rep, dim = p.dim;
  const int slot = t < p.Tdec ? p.idx[t] : -1;
  __half* yh = p.y + ((size_t)t * Hv + hv) * D;
  if (slot < 0 || slot >= p.slots) {
    if (lane < NCOL) yh[c0 + lane] = __float2half(0.f);
    return;
  }
  // 1. state columns, all in flight before anything else
  float* sp = p.ssm + (size_t)slot * p.ssm_stride + ((size_t)hv * D + c0) * D + 4 * lane;
  float s[NCOL][4];
#pragma unroll
  for (int c = 0; c < NCOL; c++) {
    const float4 v = *reinterpret_cast<const float4*>(sp + c * D);
    s[c][0] = v.x; s[c][1] = v.y; s[c][2] = v.z; s[c][3] = v.w;
  }
  // 2. b/a operands: virtual warp vw of gdn_in's reduction (threads 32 vw .. 32 vw + 31, 8 halves each)
  const uint4* xr = reinterpret_cast<const uint4*>(p.x + (size_t)t * KH);
  const uint4* wb = reinterpret_cast<const uint4*>(p.wba + (size_t)hv * KH);
  const uint4* wa = reinterpret_cast<const uint4*>(p.wba + (size_t)(Hv + hv) * KH);
  uint4 xu[NVPW], bu[NVPW], au[NVPW];
#pragma unroll
  for (int j = 0; j < NVPW; j++) {
    const int vw = warp + j * WPC;
    if (vw < NVW && !(p.dbg & 1)) {
      const int e = vw * 32 + lane;
      xu[j] = xr[e];
      bu[j] = __ldg(wb + e);
      au[j] = __ldg(wa + e);
    }
  }
  // 3. conv operands: q and k channels 4 lane .. 4 lane + 3 of key head hq; v channel c0 + lane of head hv
  const __half* qr = p.qkvz + (size_t)t * p.ldq;
  __half* cs = p.cst + (size_t)slot * p.cst_stride;
  const int chq = hq * D + 4 * lane, chk = Hk * D + hq * D + 4 * lane, chv = 2 * Hk * D + hv * D + c0 + lane;
  uint2 xq = ld8(qr + chq), xk = ld8(qr + chk), zu = make_uint2(0, 0);
  uint2 sq[3], sk[3];
#pragma unroll
  for (int r = 0; r < 3; r++) { sq[r] = ld8(cs + (size_t)r * dim + chq); sk[r] = ld8(cs + (size_t)r * dim + chk); }
  uint4 wq[2], wk[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    wq[i] = __ldg(reinterpret_cast<const uint4*>(p.cw + (size_t)chq * 4) + i);
    wk[i] = __ldg(reinterpret_cast<const uint4*>(p.cw + (size_t)chk * 4) + i);
  }
  __half xv = __float2half(0.f), sv0 = xv, sv1 = xv, sv2 = xv;
  uint2 wv = make_uint2(0, 0);
  if (lane < NCOL) {
    xv = qr[chv];
    sv0 = cs[chv]; sv1 = cs[(size_t)dim + chv]; sv2 = cs[2 * (size_t)dim + chv];
    wv = __ldg(reinterpret_cast<const uint2*>(p.cw + (size_t)chv * 4));
  }
  uint2 nu = make_uint2(0, 0);                          // gated-norm operands (used by warp 0 of the finishing CTA)
  if (warp == 0) {
    zu = ld8(qr + dim + hv * D + 4 * lane);
    nu = __ldg(reinterpret_cast<const uint2*>(p.nw + 4 * lane));
  }

  // b/a: the same products, warp sums and cross-warp order as k_gdn_in<1, *>
  __shared__ float red[2][NVW];
  __shared__ int last_sh[2];
#pragma unroll
  for (int j = 0; j < NVPW; j++) {
    const int vw = warp + j * WPC;
    if (vw < NVW && !(p.dbg & 1)) {                     // warp-uniform
      float xf[8], wf[8];
      h8(xu[j], xf);
      h8(bu[j], wf);
      float sb = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) sb = fmaf(xf[e], wf[e], sb);
      h8(au[j], wf);
      float sa = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) sa = fmaf(xf[e], wf[e], sa);
      sb = wsum(sb);
      sa = wsum(sa);
      if (lane == 0) { red[0][vw] = sb; red[1][vw] = sa; }
    }
  }

  // conv: q, k (4 channels each per lane, every warp), v (lane < NCOL)
  float q[4], k[4];
  {
    const __half* wqh = reinterpret_cast<const __half*>(wq);
    const __half* wkh = reinterpret_cast<const __half*>(wk);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      q[r] = conv1(hx(sq[0], r), hx(sq[1], r), hx(sq[2], r), hx(xq, r), wqh[4 * r], wqh[4 * r + 1], wqh[4 * r + 2],
                   wqh[4 * r + 3]);
      k[r] = conv1(hx(sk[0], r), hx(sk[1], r), hx(sk[2], r), hx(xk, r), wkh[4 * r], wkh[4 * r + 1], wkh[4 * r + 2],
                   wkh[4 * r + 3]);
    }
  }
  const float vconv = conv1(sv0, sv1, sv2, xv, hx(wv, 0), hx(wv, 1), hx(wv, 2), hx(wv, 3));
  __syncthreads();                                      // red[] complete; every warp has used its q/k conv state
  // q/k conv state counter: it orders reads (already consumed) before the last reader's writes, so no fence; the
  // result is only needed at the end, so the atomic's round trip overlaps the update
  int old_qk = -1;
  if (tid == 0 && !(p.dbg & 2)) old_qk = atomicAdd(p.cnt + t * Hk + hq, 1);
  float bsum = 0.f, asum = 0.f;
#pragma unroll
  for (int i = 0; i < NVW; i++) { bsum += red[0][i]; asum += red[1][i]; }
  const float bval = __half2float(__float2half_rn(bsum)), aval = __half2float(__float2half_rn(asum));
  if (lane < NCOL) {                                    // this column's own v channel: shift its conv state
    cs[chv] = sv1; cs[(size_t)dim + chv] = sv2; cs[2 * (size_t)dim + chv] = xv;
  }

  // recurrent update (gdn_state.cu)
  {
    const float qi = rsqrtf(wsum(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) + 1e-6f);
    const float ki = rsqrtf(wsum(k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3]) + 1e-6f);
#pragma unroll
    for (int r = 0; r < 4; r++) { q[r] *= qi; k[r] *= ki; }
  }
  float vv[NCOL];
#pragma unroll
  for (int c = 0; c < NCOL; c++) vv[c] = __shfl_sync(0xffffffffu, vconv, c);
  const float xa = aval + __half2float(p.dtb[hv]);
  const float sp_x = xa <= 20.f ? log1pf(__expf(xa)) : xa;
  const float g = __expf(-__expf(p.A_log[hv]) * sp_x);
  const float beta = 1.f / (1.f + __expf(-bval));
  float myo = 0.f;
#pragma unroll
  for (int c = 0; c < NCOL; c++) {
    const float kv = wsum(s[c][0] * k[0] + s[c][1] * k[1] + s[c][2] * k[2] + s[c][3] * k[3]);
    const float d = (vv[c] - g * kv) * beta;
    float acc = 0.f;
#pragma unroll
    for (int r = 0; r < 4; r++) {
      s[c][r] = fmaf(k[r], d, g * s[c][r]);
      acc = fmaf(s[c][r], q[r], acc);
    }
    const float oc = wsum(acc);
    if (lane == c) myo = oc;
  }
  __half* ch = p.core + ((size_t)t * Hv + hv) * D;
  if (lane < NCOL) ch[c0 + lane] = __float2half(myo * p.scale);
#pragma unroll
  for (int c = 0; c < NCOL; c++)
    *reinterpret_cast<float4*>(sp + c * D) = make_float4(s[c][0], s[c][1], s[c][2], s[c][3]);

  if (tid == 0) {
    const int lq = old_qk == rep * NS - 1;
    if (lq) p.cnt[t * Hk + hq] = 0;
    last_sh[0] = lq;
  }
  if (NS > 1 && !(p.dbg & 4)) __threadfence();          // core outputs visible before the norm counter
  __syncthreads();
  if (last_sh[0] && warp == 0) {                        // last reader of this key head's q/k conv state shifts it
#pragma unroll
    for (int r = 0; r < 2; r++) {
      *reinterpret_cast<uint2*>(cs + (size_t)r * dim + chq) = sq[r + 1];
      *reinterpret_cast<uint2*>(cs + (size_t)r * dim + chk) = sk[r + 1];
    }
    *reinterpret_cast<uint2*>(cs + 2 * (size_t)dim + chq) = xq;
    *reinterpret_cast<uint2*>(cs + 2 * (size_t)dim + chk) = xk;
  }

  // gated norm by the CTA that completes the head
  if (p.dbg & 4) return;
  if (NS > 1) {
    if (tid == 0) {
      int* cn = p.cnt + p.n * Hk + t * Hv + hv;
      const int last = atomicAdd(cn, 1) == NS - 1;
      if (last) { *cn = 0; __threadfence(); }
      last_sh[1] = last;
    }
    __syncthreads();
    if (!last_sh[1]) return;
  }
  if (warp == 0) {
    float xf[4], zf[4], wf[4];
    h4(ldcg8(ch + 4 * lane), xf);
    h4(zu, zf);
    h4(nu, wf);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) ss = fmaf(xf[i], xf[i], ss);
    ss = wsum(ss);
    const float inv = rsqrtf(ss * (1.f / D) + p.eps);
    __half2 o2[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      float v[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int kk = 2 * i + e;
        const float gz = zf[kk] / (1.f + __expf(-zf[kk]));
        v[e] = ((xf[kk] * inv) * wf[kk]) * gz;
      }
      o2[i] = __floats2half2_rn(v[0], v[1]);
    }
    *reinterpret_cast<uint2*>(yh + 4 * lane) = *reinterpret_cast<const uint2*>(o2);
  }
}

template <int NCOL, int WPC>
__global__ void __launch_bounds__(32 * WPC) k_gdn_core(const Args p) {
  core_cta<NCOL, WPC>(p, blockIdx.x, blockIdx.y, blockIdx.z);
}

template <int NCOL, int WPC>
void launch(const Args& a, cudaStream_t st) {
  const dim3 grid(a.Hv, a.n, D / (NCOL * WPC));
  k_gdn_core<NCOL, WPC><<<grid, 32 * WPC, 0, st>>>(a);
}

}  // namespace

// cfg: (columns per warp, warps per CTA) = 0 (8,4) 1 (4,4) 2 (2,4) 3 (16,4) 4 (16,8) 5 (8,8) 6 (4,8) 7 (8,16)
void gdn_core(torch::Tensor x, torch::Tensor qkvz, torch::Tensor wba, torch::Tensor cw, torch::Tensor cst,
              torch::Tensor A_log, torch::Tensor dtb, torch::Tensor ssm, torch::Tensor idx, int64_t Tdec,
              torch::Tensor nw, double eps, torch::Tensor core, torch::Tensor y, torch::Tensor cnt, int64_t cfg,
              int64_t dbg) {
  const int n = x.size(0), Hv = ssm.size(1), dim = cw.size(0);
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous() && x.size(1) == KH, "x [n][5120] fp16");
  TORCH_CHECK(qkvz.scalar_type() == at::kHalf && qkvz.stride(1) == 1 && qkvz.size(0) == n &&
                  qkvz.size(1) >= dim + Hv * D && qkvz.stride(0) % 4 == 0, "qkvz");
  TORCH_CHECK(wba.scalar_type() == at::kHalf && wba.is_contiguous() && wba.size(0) == 2 * Hv && wba.size(1) == KH, "wba");
  TORCH_CHECK(cw.scalar_type() == at::kHalf && cw.is_contiguous() && cw.dim() == 2 && cw.size(1) == 4, "conv weight [dim][4]");
  TORCH_CHECK(cst.scalar_type() == at::kHalf && cst.dim() == 3 && cst.size(1) >= 3 && cst.size(2) == dim &&
                  cst.stride(2) == 1 && cst.stride(1) == dim && cst.stride(0) % 4 == 0, "conv state [slots][>=3][dim] (SD; MTP slots hold 3 + K rows, rows 0..2 are the decode window)");
  TORCH_CHECK(ssm.scalar_type() == at::kFloat && ssm.dim() == 4 && ssm.size(2) == D && ssm.size(3) == D &&
                  ssm.stride(1) == D * D && ssm.stride(2) == D && ssm.stride(3) == 1 && ssm.stride(0) % 4 == 0, "ssm state");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dtb.scalar_type() == at::kHalf && idx.scalar_type() == at::kInt, "params");
  TORCH_CHECK(nw.scalar_type() == at::kHalf && nw.numel() == D && nw.is_contiguous(), "norm weight");
  TORCH_CHECK(core.scalar_type() == at::kHalf && core.is_contiguous() && core.numel() >= (int64_t)n * Hv * D, "core");
  TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous() && y.numel() == (int64_t)n * Hv * D, "y");
  const int Hk = (dim - Hv * D) / (2 * D);
  TORCH_CHECK(Hk > 0 && Hv % Hk == 0 && dim == (2 * Hk + Hv) * D, "head counts");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= (int64_t)n * (Hk + Hv), "counter workspace");
  TORCH_CHECK(Tdec <= n && idx.numel() >= Tdec, "Tdec");
  if (n == 0) return;
  Args a;
  a.x = reinterpret_cast<const __half*>(x.data_ptr());
  a.qkvz = reinterpret_cast<const __half*>(qkvz.data_ptr());
  a.wba = reinterpret_cast<const __half*>(wba.data_ptr());
  a.cw = reinterpret_cast<const __half*>(cw.data_ptr());
  a.cst = reinterpret_cast<__half*>(cst.data_ptr());
  a.A_log = A_log.data_ptr<float>();
  a.dtb = reinterpret_cast<const __half*>(dtb.data_ptr());
  a.ssm = ssm.data_ptr<float>();
  a.idx = idx.data_ptr<int32_t>();
  a.nw = reinterpret_cast<const __half*>(nw.data_ptr());
  a.core = reinterpret_cast<__half*>(core.data_ptr());
  a.y = reinterpret_cast<__half*>(y.data_ptr());
  a.cnt = cnt.data_ptr<int>();
  a.ldq = qkvz.stride(0); a.cst_stride = cst.stride(0); a.ssm_stride = ssm.stride(0);
  a.slots = std::min(cst.size(0), ssm.size(0)); a.Tdec = Tdec; a.n = n; a.Hk = Hk; a.Hv = Hv; a.dim = dim;
  a.scale = 1.f / sqrtf((float)D); a.eps = (float)eps; a.dbg = (int)dbg;
  const at::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  switch (cfg) {
    case 0: launch<8, 4>(a, st); break;
    case 1: launch<4, 4>(a, st); break;
    case 2: launch<2, 4>(a, st); break;
    case 3: launch<16, 4>(a, st); break;
    case 4: launch<16, 8>(a, st); break;
    case 5: launch<8, 8>(a, st); break;
    case 6: launch<4, 8>(a, st); break;
    case 7: launch<8, 16>(a, st); break;
    default: TORCH_CHECK(false, "gdn_core: unknown config ", cfg);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_core", &gdn_core, "fused GDN decode core: b/a projection, conv update, state update, gated norm");
}
