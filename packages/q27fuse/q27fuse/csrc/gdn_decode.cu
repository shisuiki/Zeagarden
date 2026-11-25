// gdn_decode.cu - gdn_core.cu's fused GDN decode core (one token per sequence: b/a projection, conv update, state
// update, gated norm in one launch) with the replay commit of q27fuse.gdnv's records folded in: before its token, a
// request replays the accepted tokens of its last verification (records written by gdn_verify.cu: k fp16 + 1/||k||,
// v fp16, decay, beta; side rmap ^ 1, rn tokens) into the state it already holds in registers, so a draft-free MTP step
// reads and writes every state once and launches nothing else. With nothing to replay (rn = 0, the steady state of
// K = 0 steps) the kernel is gdn_core.cu's, bit for bit. Everything below except the replay block and the extra
// arguments is a copy of gdn_core.cu (prune tree copied 2026-11-21, md5 36e75f0bb84a619c1fd74203faba0bd0).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#ifndef GDND_REPLAY
#define GDND_REPLAY 2   // 2: rn / rmap loaded where the replay starts. Bench variants only (scripts/bench_dec*.py):
#endif                  // 0 = without the replay code (gdn_core.cu), 1 = rn / rmap loaded next to idx
#ifndef GDND_UNR
#define GDND_UNR 8      // replay loop fully unrolled up to 8 tokens (1Cat's TMAX); 0 = rolled loop (bench only: a loop
#endif                  // back-edge in the kernel costs the 16-warp config ~22 us per layer at 48 rows even at rn = 0)

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
  const int* rn;        // [>= Tdec] tokens to replay per row (gdnv records), nullptr: none
  const int* rmap;      // [>= Tdec] record side map (replay side rmap ^ 1)
  const uint8_t* rbuf;  // records [2 max_reqs][tmax][REC]
  int tmax;
  const int* ccs;       // [ckpt_fix] [>= Tdec] commit checkpoint slot: the committed state (after the replay, before
                        // this step's token) is also stored there (-1 / nullptr: none)
};
constexpr int REC = 4224, R_K = 0, R_V = 4 * 128 * 2, R_G = R_V + 12 * 128 * 2, R_B = R_G + 12 * 4, R_KI = R_B + 12 * 4;

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
#if GDND_REPLAY == 1                                    // (slower: 114.6 vs 111.1 us at 48 rows, cfg 0)
  const int rn_raw = (p.rn && t < p.Tdec) ? p.rn[t] : 0, rside = (p.rn && t < p.Tdec) ? p.rmap[t] ^ 1 : 0;
#endif
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
  // gate parameters, loaded with the operands (gdn_core.cu reads them at the update; with the replay code in between
  // the compiler sank those loads behind the barrier, exposing one memory latency per CTA: +20 us per layer at 48 rows)
  const float alog = __ldg(p.A_log + hv);
  const __half dtbh = __ldg(p.dtb + hv);
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

  // replay: the accepted tokens of this request's last verification (gdnv records), into the state registers, just
  // before the state is first used. Unrolled (GDND_UNR): with a rolled loop here the kernel was ~22 us per layer slower
  // in gdn_core.cu's 16-warp config at 48 rows even with nothing to replay (132 vs 110 us; the loop header makes the
  // compiler settle every outstanding load first, which a 1-CTA-per-SM config cannot hide; scripts/bench_dec2.py:
  // unrolled 110.9-111.3 vs gdn_core.cu 108.7, rolled 132.4)
#if GDND_REPLAY
#if GDND_REPLAY == 2
  const int rn_raw = p.rn ? p.rn[t] : 0, rside = p.rn ? p.rmap[t] ^ 1 : 0;
#endif
  const int rn = min(max(rn_raw, 0), min(p.tmax, 8));
  if (rn > 0) {
    const uint8_t* rb = p.rbuf + (size_t)rside * p.tmax * REC;
#if GDND_UNR
#pragma unroll
    for (int j = 0; j < GDND_UNR; j++) {
      if (j >= rn) break;
#else
#pragma unroll 1
    for (int j = 0; j < rn; j++) {
#endif
      const uint8_t* r = rb + (size_t)j * REC;
      const uint2 ku = *reinterpret_cast<const uint2*>(r + R_K + (hq * D + 4 * lane) * 2);
      const float ki = *reinterpret_cast<const float*>(r + R_KI + hq * 4);
      const float g = *reinterpret_cast<const float*>(r + R_G + hv * 4);
      const float be = *reinterpret_cast<const float*>(r + R_B + hv * 4);
      const float vme = lane < NCOL ? __half2float(reinterpret_cast<const __half*>(r + R_V)[hv * D + c0 + lane]) : 0.f;
      float k4[4];
      h4(ku, k4);
#pragma unroll
      for (int e = 0; e < 4; e++) k4[e] *= ki;
#pragma unroll
      for (int c = 0; c < NCOL; c++) {
        const float vv = __shfl_sync(0xffffffffu, vme, c);
        const float kv = wsum(s[c][0] * k4[0] + s[c][1] * k4[1] + s[c][2] * k4[2] + s[c][3] * k4[3]);
        const float d = (vv - g * kv) * be;
#pragma unroll
        for (int e = 0; e < 4; e++) s[c][e] = fmaf(k4[e], d, g * s[c][e]);
      }
    }
    if (p.ccs) {                        // [ckpt_fix] deferred prefix-cache checkpoint: this commit ends exactly on a
      const int cc = p.ccs[t];          // Mamba block boundary
      if (cc >= 0 && cc < p.slots && cc != slot) {
        float* cp = p.ssm + (size_t)cc * p.ssm_stride + ((size_t)hv * D + c0) * D + 4 * lane;
#pragma unroll
        for (int c = 0; c < NCOL; c++)
          *reinterpret_cast<float4*>(cp + c * D) = make_float4(s[c][0], s[c][1], s[c][2], s[c][3]);
      }
    }
  }
#endif
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
  const float xa = aval + __half2float(dtbh);
  const float sp_x = xa <= 20.f ? log1pf(__expf(xa)) : xa;
  const float g = __expf(-__expf(alog) * sp_x);
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
void gdn_decode(torch::Tensor x, torch::Tensor qkvz, torch::Tensor wba, torch::Tensor cw, torch::Tensor cst,
                torch::Tensor A_log, torch::Tensor dtb, torch::Tensor ssm, torch::Tensor idx, int64_t Tdec,
                torch::Tensor nw, double eps, torch::Tensor core, torch::Tensor y, torch::Tensor cnt, int64_t cfg,
                int64_t dbg, c10::optional<torch::Tensor> rn, c10::optional<torch::Tensor> rmap, torch::Tensor rbuf,
                int64_t tmax, c10::optional<torch::Tensor> ccs) {
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
  const bool rp = rn.has_value() && rn->defined() && rn->numel() > 0;
  TORCH_CHECK(!rp || (rmap.has_value() && rn->scalar_type() == at::kInt && rmap->scalar_type() == at::kInt &&
                      rbuf.scalar_type() == at::kByte && tmax >= 1 && tmax <= 8 && Hv == 12 && Hk == 4 &&
                      rbuf.numel() % ((int64_t)tmax * REC) == 0), "replay tables (records: 4 key heads, 12 value heads)");
  a.rn = rp ? rn->data_ptr<int>() : nullptr; a.rmap = rp ? rmap->data_ptr<int>() : nullptr;
  a.rbuf = rp ? rbuf.data_ptr<uint8_t>() : nullptr; a.tmax = (int)tmax;
  const bool cp = rp && ccs.has_value() && ccs->defined() && ccs->numel() > 0;                      // [ckpt_fix]
  TORCH_CHECK(!cp || (ccs->scalar_type() == at::kInt && ccs->is_cuda()), "ccs");                     // [ckpt_fix]
  a.ccs = cp ? ccs->data_ptr<int>() : nullptr;                                                      // [ckpt_fix]
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
    default: TORCH_CHECK(false, "gdn_decode: unknown config ", cfg);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_decode", &gdn_decode, "gdn_core.cu's fused decode core with the gdnv replay commit folded in");
}
