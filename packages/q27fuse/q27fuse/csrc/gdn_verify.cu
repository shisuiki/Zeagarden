// gdn_verify.cu - the Gated DeltaNet core of a Qwen3.8-27B layer under MTP (TP4 shard on V100: 4 key heads, 12 value
// heads, head dim 128, conv width 4) in one launch per layer, with OUR replay records ("deferred commit").
//
// Under MTP a request keeps one committed FP32 state slot; a verify round of T = K + 1 tokens does not store its
// per-token states. Instead every round writes a small record per token (k, v, decay, beta), and the next step that
// touches the request first replays the accepted prefix of those records into the committed state ("commit").
// Bookkeeping (which side to write/replay, how many tokens, the prefix-cache checkpoint) stays 1Cat's
// (spec_replay.SpecReplayState tables, indexed per request row b):
//   rn[b]    tokens to replay (the previous verification's accepted count, 0 if the last step did not verify)
//   rmap[b]  2 req + side: records of THIS round go to side rmap, the replay reads side rmap ^ 1 (1Cat's convention;
//            plain rows get rmap = 2 req + (side ^ 1), so their replay reads what the last verification wrote)
//   ckt[b], cks[b]  in-round token whose state is stored into prefix-cache slot cks (-1: none)
//
// Record format (ours; 1Cat's buffer is never allocated): per (request side, token) REC = 4224 bytes:
//   [0, 1024)     k    fp16 [4][128]  conv outputs of the key heads (fp16 by construction: exact)
//   [1024, 4096)  v    fp16 [12][128] conv outputs of the value heads
//   [4096, 4144)  g    f32 [12]       decay factor exp(-exp(A_log) softplus(a + dt_bias))
//   [4144, 4192)  beta f32 [12]       sigmoid(b)
//   [4192, 4208)  kinv f32 [4]        1 / ||k|| as used by the round (replay uses float(k) * kinv: bitwise the same k)
//
// Modes (template MODE bits):
//   VERIFY     conv over the (3+K)-row window at offset acc-1 with the T+2-row roll-back, gates from the projection's
//              a/b, replay commit (state stored once), T-token recurrence in registers with outputs from the
//              intermediate states, records of this round, checkpoint store, gated RMSNorm * silu(z) -> y
//   VERIFY_BA  the same, b/a projection in the kernel (rows > 32, where 1Cat's fused in-proj does not apply)
//   DECODE     draft-free decode (T = 1, window offset 0, b/a in the kernel, replay folded in, final state stored once)
//   VONLY      1Cat's mixed-batch route: convolved qkv in, raw scaled outputs out, replay + records (no conv, no norm)
//   COMMIT     replay only (plain decode rows of 1Cat's mixed-batch route); rn = 0 -> nothing loaded
//
// Work split: CTA (value head hv, request b, slice sl) owns NCOL x WPC state columns (S[hv][v][:], 512 B each; lane l
// holds k-rows 4l .. 4l+3: one 16-byte access per column). Each state column evolves independently given the per-token
// vectors, so a column is loaded once, takes rn replay updates, is stored (commit), then takes T speculative updates in
// registers while emitting T outputs: one read + one write of the state per step whatever K is.
// Every global operand of a CTA (state, records, conv rows, weights, a/b, z) is issued before any of it is used; the
// q/k conv of the key head (warp w takes tokens w, w + WPC, ...; lane l channels 4l .. 4l+3 as gdn_core.cu) and the token
// norms go to shared memory once per CTA; the v channels of the CTA's columns are convolved by the thread that owns
// them (and rolled back by it).
// Cross-CTA hazards ("last one out" counters in a zeroed int32 workspace, self-resetting, no CTA ever waits):
//   - the q/k window of (request, key head) is read by REP x NS CTAs; the last to finish its conv reads writes it back;
//   - the gated norm of (request, head) needs all 128 outputs: with NS > 1 slices the last slice normalises.
// Arithmetic per value is gdn_core.cu's (conv: fp16 products, fp32 sum, silu, fp16; L2 norm via the same lane grouping;
// state update and output dot products; gated norm), so DECODE with rn = 0 reproduces gdn_core.cu bit for bit.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int D = 128, HK = 4, HV = 12, REP = HV / HK, DIM = (2 * HK + HV) * D, KH = 5120, NV = KH / 8, NVW = NV / 32;
constexpr int QKC = 2 * D;                              // q + k channels of one key head
constexpr int R_K = 0, R_V = HK * D * 2, R_G = R_V + HV * D * 2, R_B = R_G + HV * 4, R_KI = R_B + HV * 4;
constexpr int REC = 4224;
static_assert(R_KI + HK * 4 <= REC && REC % 128 == 0, "record layout");
constexpr int RMAX = 8;                                 // replay capacity per request side (1Cat's TMAX)
#ifndef GDNV_TOKEN_UNROLL
#define GDNV_TOKEN_UNROLL 0   // 1 = token / replay loops fully unrolled: 1.7 us slower at B=1 T=5 (instruction cache,
#endif                        // results/r1/gdnv/t5_unroll.log)

enum { M_CONV = 1, M_BA = 2, M_NORM = 4, M_REC = 8, M_FINAL = 16, M_TOK = 32 };
constexpr int VERIFY = M_CONV | M_NORM | M_REC | M_TOK, VERIFY_BA = VERIFY | M_BA,
              DECODE = M_CONV | M_BA | M_NORM | M_FINAL | M_TOK, VONLY = M_REC | M_TOK, COMMIT = 0;

struct Args {
  const __half* qkv; long ldq;                     // [n][>= DIM]: raw projection (M_CONV) or convolved qkv
  const __half* z; long ldz;                       // [n][HV D] (M_NORM)
  const __half* av; const __half* bv; long ldab;   // [n][HV] a, b (no M_BA)
  const __half* x; const __half* wba;              // [n][5120], [2 HV][5120] b rows then a rows (M_BA)
  const __half* cw;                                // [DIM][4]
  __half* cst; long cst_stride;                    // conv window per slot [3 + K][DIM]
  const float* A_log; const __half* dtb;
  float* ssm; long ssm_stride; int slots;          // per slot [HV][128][128]
  const int* idx; int idx_stride;                  // [B] state slot
  const int* cu;                                   // [B + 1] rows of each request (nullptr: b T .. b T + T)
  const int* acc;                                  // [B] conv window offset + 1 (nullptr: offset 0)
  const int* rn; const int* rmap;                  // [B] replay count, record side (nullptr: no replay)
  const int* ckt; const int* cks;                  // [B] checkpoint token, slot (nullptr: none)
  const int* ccs;                                  // [ckpt_fix] [B] commit checkpoint slot: the committed state (after
                                                   // the replay) is also stored there (-1 / nullptr: none)
  uint8_t* rbuf; int tmax;                         // records [2 max_reqs][tmax][REC]
  const __half* nw; float eps;                     // norm weight [128]
  __half* core;                                    // [n][HV][D]: norm scratch (NS > 1) or the raw output (no M_NORM)
  __half* y;                                       // [n][HV D] (M_NORM)
  int* cnt;                                        // zeroed: [B HK] conv counters, then [B HV] norm counters
  int B, T, n;
  float scale;
  int dbg;                                         // bench only (test_gdn_verify.py phases): bit 0 skip the replay
                                                   // updates, bit 1 the token updates, bit 2 the norm hand-off
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
__device__ __forceinline__ uint2 ldcg8(const __half* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ __half hx(const uint2& u, int i) { return reinterpret_cast<const __half*>(&u)[i]; }

// 1Cat's triton conv for one channel and one token: acc = sum_j fp16(x_j * w_j) in fp32, silu, fp16 (gdn_core.cu)
__device__ __forceinline__ float conv1(__half s0, __half s1, __half s2, __half x, __half w0, __half w1, __half w2,
                                       __half w3) {
  float acc = __half2float(__hmul(s0, w0));
  acc += __half2float(__hmul(s1, w1));
  acc += __half2float(__hmul(s2, w2));
  acc += __half2float(__hmul(x, w3));
  acc = acc / (1.f + __expf(-acc));
  return __half2float(__float2half_rn(acc));
}
__device__ __forceinline__ void gates(float aval, float bval, float A_log, float dtb, float& g, float& beta) {
  const float xa = aval + dtb;
  const float sp_x = xa <= 20.f ? log1pf(__expf(xa)) : xa;
  g = __expf(-__expf(A_log) * sp_x);
  beta = 1.f / (1.f + __expf(-bval));
}
// q/k channel ch (0..255) of key head hq -> channel in the qkv row
__device__ __forceinline__ int qk_ch(int hq, int ch) { return ch < D ? hq * D + ch : HK * D + hq * D + (ch - D); }

template <int NC, int TM>
struct Smem {                        // every member 16-byte aligned (uint4 / float4 accesses)
  alignas(16) float qn[TM][D];       // q_t * 1/||q_t||, k_t * 1/||k_t|| (the values the recurrence uses)
  alignas(16) float kn[TM][D];
  alignas(16) float vc[TM][NC];      // conv outputs of the CTA's v channels
  alignas(16) float g[TM];
  alignas(16) float beta[TM];
  alignas(16) __half raw[TM + 2][QKC];   // q/k window rows to roll back: h1, h2, x_0 .. x_{L-1}
  alignas(16) __half rk[RMAX][D];    // replay records: k of hq
  alignas(16) __half rv[RMAX][NC];   //                 v of the CTA's columns
  alignas(16) float rs[RMAX][3];     //                 g, beta, kinv
  alignas(16) float red[TM][2][NVW]; // b/a partial sums (M_BA)
  alignas(16) __half co[TM][D];      // the head's outputs for the norm (NS == 1)
  int last[2];
};

template <int NCOL, int WPC, int TM, int MODE>
__global__ void __launch_bounds__(32 * WPC) k_gdnv(const Args p) {
  constexpr int NT = 32 * WPC, NC = NCOL * WPC, NS = D / NC;
  constexpr int NVPW = (NVW + WPC - 1) / WPC, NZ = (TM + WPC - 1) / WPC;   // NZ: tokens per warp
  constexpr int RCH = 16 + NC / 8, NRC = (RMAX * RCH + NT - 1) / NT;      // record chunks (uint4) per CTA / thread
  constexpr bool TOK = MODE & M_TOK, CONV = MODE & M_CONV, BA = MODE & M_BA, NORM = MODE & M_NORM,
                 RECW = MODE & M_REC, FINAL = MODE & M_FINAL;
  __shared__ Smem<NC, TM> sm;
  const int hv = blockIdx.x, b = blockIdx.y, sl = blockIdx.z, hq = hv / REP;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  static_assert(NCOL == 8, "4 lanes per state column, 8 columns per warp");
  const int cb = sl * NC, c0 = cb + warp * NCOL;       // first column of the CTA / of the warp
  // lane l owns column c0 + l/4; within the column, sub-lane qd = l % 4 holds the 16-byte chunks qd, qd+4, ..., qd+28
  // (elements 16 i + 4 qd .. + 3, i < 8): a column's dot products reduce over 4 lanes (2 shuffles) instead of 32
  const int qd = lane & 3, ccol = c0 + (lane >> 2), cl = ccol - cb;

  int row0 = 0, L = 0;
  if (TOK) {
    if (p.cu) { row0 = p.cu[b]; L = p.cu[b + 1] - row0; } else { row0 = b * p.T; L = p.T; }
    L = max(0, min(L, TM));
  }
  // rows after the last request (padding of a captured batch): zero outputs
  if (TOK && b == p.B - 1) {
    const int tail = p.cu ? p.cu[p.B] : p.B * p.T;
    for (int i = tid; i < (p.n - tail) * NC; i += NT) {
      const size_t o = ((size_t)(tail + i / NC) * HV + hv) * D + cb + i % NC;
      if (NORM) p.y[o] = __float2half(0.f); else p.core[o] = __float2half(0.f);
    }
  }
  const int slot = p.idx[(size_t)b * p.idx_stride];
  int rn = 0, rsrc = 0, wdst = 0;
  if (p.rn) {
    rn = min(max(p.rn[b], 0), min(p.tmax, RMAX));
    wdst = p.rmap[b];
    rsrc = wdst ^ 1;
  }
  if (slot < 0 || slot >= p.slots || (!TOK && rn == 0) || (TOK && L == 0)) {
    for (int i = tid; i < L * NC; i += NT) {             // invalid request: zero outputs
      const size_t o = ((size_t)(row0 + i / NC) * HV + hv) * D + cb + i % NC;
      if (NORM) p.y[o] = __float2half(0.f); else p.core[o] = __float2half(0.f);
    }
    return;
  }
  int ck = -1, cslot = -1;
  if (TOK && p.ckt) {
    ck = p.ckt[b];
    cslot = p.cks[b];
    if (cslot < 0 || cslot >= p.slots || ck < 0 || ck >= L) ck = -1;
  }
  const int offs = (CONV && p.acc) ? max(p.acc[b] - 1, 0) : 0;
  uint8_t* wrec = RECW ? p.rbuf + (size_t)wdst * p.tmax * REC : nullptr;
  __half* cs = CONV ? p.cst + (size_t)slot * p.cst_stride : nullptr;
  const __half* xr = p.qkv + (size_t)row0 * p.ldq;

  // ------------------------------------------------------------------ 1. every global load, issued before any use
  float* sp = p.ssm + (size_t)slot * p.ssm_stride + ((size_t)hv * D + ccol) * D + 4 * qd;
  float s[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float4 v = *reinterpret_cast<const float4*>(sp + 16 * i);
    s[i][0] = v.x; s[i][1] = v.y; s[i][2] = v.z; s[i][3] = v.w;
  }
  // records of the previous round (side rmap ^ 1): k of hq and v of the CTA's columns as uint4 chunks, scalars
  const uint8_t* rbase = p.rbuf + (size_t)rsrc * p.tmax * REC;
  uint4 rr[NRC];
#pragma unroll
  for (int m = 0; m < NRC; m++) {
    const int i = tid + m * NT;
    if (i < rn * RCH) {
      const int j = i / RCH, e = i % RCH;
      const uint8_t* r = rbase + (size_t)j * REC;
      rr[m] = e < 16 ? reinterpret_cast<const uint4*>(r + R_K + hq * D * 2)[e]
                     : reinterpret_cast<const uint4*>(r + R_V + (hv * D + cb) * 2)[e - 16];
    }
  }
  float rsc = 0.f;
  if (tid < 3 * rn) {
    const int j = tid / 3, f = tid % 3;
    rsc = *reinterpret_cast<const float*>(rbase + (size_t)j * REC +
                                          (f == 0 ? R_G + hv * 4 : f == 1 ? R_B + hv * 4 : R_KI + hq * 4));
  }
  // b/a operands: virtual warps of gdn_glue's 640-thread reduction (same sums, same bits as gdn_core.cu)
  uint4 bu[NVPW], au[NVPW], xu[BA ? TM : 1][NVPW];
  if (BA) {
    const uint4* wb = reinterpret_cast<const uint4*>(p.wba + (size_t)hv * KH);
    const uint4* wa = reinterpret_cast<const uint4*>(p.wba + (size_t)(HV + hv) * KH);
#pragma unroll
    for (int j = 0; j < NVPW; j++) {
      const int vw = warp + j * WPC;
      if (vw < NVW) {
        const int e = vw * 32 + lane;
        bu[j] = __ldg(wb + e);
        au[j] = __ldg(wa + e);
#pragma unroll
        for (int t = 0; t < (BA ? TM : 1); t++)
          if (t < L) xu[t][j] = reinterpret_cast<const uint4*>(p.x + (size_t)(row0 + t) * KH)[e];
      }
    }
  }
  // q/k operands: warp w takes tokens t = w + i WPC; lane l: q and k channels 4l .. 4l+3 of key head hq.
  // seq[r] = window row offs + r (r < 3), else x_{r-3}; token t convolves seq[t .. t+3]
  const int chq = hq * D + 4 * lane, chk = HK * D + hq * D + 4 * lane;
  uint2 sq[NZ][4], sk[NZ][4];
#pragma unroll
  for (int i = 0; i < NZ; i++) {
    const int t = warp + i * WPC;
    if (t < L) {
      if (CONV) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int q = t + r;
          const __half* src = q < 3 ? cs + (size_t)(offs + q) * DIM : xr + (size_t)(q - 3) * p.ldq;
          sq[i][r] = *reinterpret_cast<const uint2*>(src + chq);
          sk[i][r] = *reinterpret_cast<const uint2*>(src + chk);
        }
      } else {
        sq[i][3] = *reinterpret_cast<const uint2*>(xr + (size_t)t * p.ldq + chq);
        sk[i][3] = *reinterpret_cast<const uint2*>(xr + (size_t)t * p.ldq + chk);
      }
    }
  }
  uint4 wq[2], wk[2];
  if (CONV) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      wq[i] = __ldg(reinterpret_cast<const uint4*>(p.cw + (size_t)chq * 4) + i);
      wk[i] = __ldg(reinterpret_cast<const uint4*>(p.cw + (size_t)chk * 4) + i);
    }
  }
  // v operands: thread c < NC owns channel 2 HK D + hv D + cb + c
  // (the v channels and the gates go to the highest threads: the q/k conv occupies warps 0 .. TM-1)
  constexpr int VB = NT - NC, GB = NT - NC - 32 > 0 ? NT - NC - 32 : 0;
  const int vt = tid - VB, gt = tid - GB;
  const int gv = 2 * HK * D + hv * D + cb + vt;
  __half vh[3], vx[TM];
  uint2 vwt = make_uint2(0, 0);
  if (vt >= 0) {
    if (CONV) {
#pragma unroll
      for (int r = 0; r < 3; r++) vh[r] = cs[(size_t)(offs + r) * DIM + gv];
      vwt = __ldg(reinterpret_cast<const uint2*>(p.cw + (size_t)gv * 4));
    }
#pragma unroll
    for (int t = 0; t < TM; t++)
      if (t < L) vx[t] = xr[(size_t)t * p.ldq + gv];
  }
  __half ga = __float2half(0.f), gb = ga;
  if (TOK && !BA && gt >= 0 && gt < L) {
    const size_t o = (size_t)(row0 + gt) * p.ldab + hv;
    ga = p.av[o];
    gb = p.bv[o];
  }
  const float alog = p.A_log[hv], dtbv = __half2float(p.dtb[hv]);
  uint2 zu[NZ], nu = make_uint2(0, 0);
  if (NORM) {
#pragma unroll
    for (int i = 0; i < NZ; i++) {
      const int t = warp + i * WPC;
      if (t < L) zu[i] = *reinterpret_cast<const uint2*>(p.z + (size_t)(row0 + t) * p.ldz + hv * D + 4 * lane);
    }
    nu = __ldg(reinterpret_cast<const uint2*>(p.nw + 4 * lane));
  }

  // ------------------------------------------------------------------ 2. into shared memory
#pragma unroll
  for (int m = 0; m < NRC; m++) {
    const int i = tid + m * NT;
    if (i < rn * RCH) {
      const int j = i / RCH, e = i % RCH;
      if (e < 16) reinterpret_cast<uint4*>(sm.rk[j])[e] = rr[m];
      else reinterpret_cast<uint4*>(sm.rv[j])[e - 16] = rr[m];
    }
  }
  if (tid < 3 * rn) sm.rs[tid / 3][tid % 3] = rsc;
  if (BA) {
#pragma unroll
    for (int t = 0; t < (BA ? TM : 1); t++) {
      if (t >= L) break;
#pragma unroll
      for (int j = 0; j < NVPW; j++) {
        const int vw = warp + j * WPC;
        if (vw < NVW) {
          float xf[8], wf[8];
          h8(xu[t][j], xf);
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
          if (lane == 0) { sm.red[t][0][vw] = sb; sm.red[t][1][vw] = sa; }
        }
      }
    }
  }
  if (TOK) {
    // q/k conv (1Cat's triton per channel), L2 norms (gdn_core.cu's lane grouping), rows to roll back, k records
#pragma unroll
    for (int i = 0; i < NZ; i++) {
      const int t = warp + i * WPC;
      if (t < L) {
        float q[4], k[4];
        if (CONV) {
          const __half* wqh = reinterpret_cast<const __half*>(wq);
          const __half* wkh = reinterpret_cast<const __half*>(wk);
#pragma unroll
          for (int r = 0; r < 4; r++) {
            q[r] = conv1(hx(sq[i][0], r), hx(sq[i][1], r), hx(sq[i][2], r), hx(sq[i][3], r), wqh[4 * r],
                         wqh[4 * r + 1], wqh[4 * r + 2], wqh[4 * r + 3]);
            k[r] = conv1(hx(sk[i][0], r), hx(sk[i][1], r), hx(sk[i][2], r), hx(sk[i][3], r), wkh[4 * r],
                         wkh[4 * r + 1], wkh[4 * r + 2], wkh[4 * r + 3]);
          }
          *reinterpret_cast<uint2*>(&sm.raw[2 + t][4 * lane]) = sq[i][3];
          *reinterpret_cast<uint2*>(&sm.raw[2 + t][D + 4 * lane]) = sk[i][3];
          if (t == 0) {
#pragma unroll
            for (int r = 0; r < 2; r++) {
              *reinterpret_cast<uint2*>(&sm.raw[r][4 * lane]) = sq[i][r + 1];
              *reinterpret_cast<uint2*>(&sm.raw[r][D + 4 * lane]) = sk[i][r + 1];
            }
          }
        } else {
          h4(sq[i][3], q);
          h4(sk[i][3], k);
        }
        const float qi = rsqrtf(wsum(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) + 1e-6f);
        const float ki = rsqrtf(wsum(k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3]) + 1e-6f);
        if (RECW && hv % REP == 0 && sl == 0) {
          __half2 k2[2] = {__floats2half2_rn(k[0], k[1]), __floats2half2_rn(k[2], k[3])};
          *reinterpret_cast<uint2*>(wrec + (size_t)t * REC + R_K + (hq * D + 4 * lane) * 2) =
              *reinterpret_cast<const uint2*>(k2);
          if (lane == 0) *reinterpret_cast<float*>(wrec + (size_t)t * REC + R_KI + hq * 4) = ki;
        }
#pragma unroll
        for (int r = 0; r < 4; r++) { q[r] *= qi; k[r] *= ki; }
        *reinterpret_cast<float4*>(&sm.qn[t][4 * lane]) = make_float4(q[0], q[1], q[2], q[3]);
        *reinterpret_cast<float4*>(&sm.kn[t][4 * lane]) = make_float4(k[0], k[1], k[2], k[3]);
      }
    }
    // v channels: the owning thread convolves, rolls its window back and records v
    if (vt >= 0) {
      float vo[TM];
      if (CONV) {
        __half h0 = vh[0], h1 = vh[1], h2 = vh[2];
        cs[gv] = h1;
        cs[(size_t)DIM + gv] = h2;
#pragma unroll
        for (int t = 0; t < TM; t++) {
          if (t < L) {
            vo[t] = conv1(h0, h1, h2, vx[t], hx(vwt, 0), hx(vwt, 1), hx(vwt, 2), hx(vwt, 3));
            cs[(size_t)(2 + t) * DIM + gv] = vx[t];
            h0 = h1; h1 = h2; h2 = vx[t];
          }
        }
      } else {
#pragma unroll
        for (int t = 0; t < TM; t++)
          if (t < L) vo[t] = __half2float(vx[t]);
      }
#pragma unroll
      for (int t = 0; t < TM; t++) {
        if (t < L) {
          sm.vc[t][vt] = vo[t];
          if (RECW)
            reinterpret_cast<__half*>(wrec + (size_t)t * REC + R_V)[hv * D + cb + vt] = __float2half_rn(vo[t]);
        }
      }
    }
    if (!BA && gt >= 0 && gt < L) {
      float g, be;
      gates(__half2float(ga), __half2float(gb), alog, dtbv, g, be);
      sm.g[gt] = g;
      sm.beta[gt] = be;
      if (RECW && sl == 0) {
        *reinterpret_cast<float*>(wrec + (size_t)gt * REC + R_G + hv * 4) = g;
        *reinterpret_cast<float*>(wrec + (size_t)gt * REC + R_B + hv * 4) = be;
      }
    }
  }
  __syncthreads();                                      // conv outputs, norms, records, b/a partial sums
  int old_qk = -1;
  // q/k window counter: orders this CTA's window reads (consumed above) before the last reader's roll-back; no fence,
  // the result is only needed at the end
  if (TOK && CONV && tid == 0) old_qk = atomicAdd(p.cnt + b * HK + hq, 1);
  if (BA) {
    if (tid < L) {
      float bsum = 0.f, asum = 0.f;
#pragma unroll
      for (int i = 0; i < NVW; i++) { bsum += sm.red[tid][0][i]; asum += sm.red[tid][1][i]; }
      float g, be;
      gates(__half2float(__float2half_rn(asum)), __half2float(__float2half_rn(bsum)), alog, dtbv, g, be);
      sm.g[tid] = g;
      sm.beta[tid] = be;
      if (RECW && sl == 0) {
        *reinterpret_cast<float*>(wrec + (size_t)tid * REC + R_G + hv * 4) = g;
        *reinterpret_cast<float*>(wrec + (size_t)tid * REC + R_B + hv * 4) = be;
      }
    }
    __syncthreads();                                    // gates
  }

  // ------------------------------------------------------------------ 3. replay: commit the previous accepted tokens
#if !GDNV_TOKEN_UNROLL
#pragma unroll 1
#endif
  for (int j = 0; j < rn; j++) {
    if (p.dbg & 1) break;
    const float g = sm.rs[j][0], be = sm.rs[j][1], ki = sm.rs[j][2];
    const float vv = __half2float(sm.rv[j][cl]);
    float k[8][4], a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      h4(*reinterpret_cast<const uint2*>(&sm.rk[j][16 * i + 4 * qd]), k[i]);
#pragma unroll
      for (int r = 0; r < 4; r++) {
        k[i][r] *= ki;
        a4[r] = fmaf(s[i][r], k[i][r], a4[r]);
      }
    }
    float kv = (a4[0] + a4[1]) + (a4[2] + a4[3]);
    kv += __shfl_xor_sync(0xffffffffu, kv, 1);
    kv += __shfl_xor_sync(0xffffffffu, kv, 2);
    const float d = (vv - g * kv) * be;
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int r = 0; r < 4; r++) s[i][r] = fmaf(k[i][r], d, g * s[i][r]);
  }
  if (!FINAL && rn > 0) {                               // the committed state, stored once (a checkpoint into the
#pragma unroll                                         // running slot below overwrites it, as 1Cat's op does)
    for (int i = 0; i < 8; i++)
      *reinterpret_cast<float4*>(sp + 16 * i) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
  }
  if (p.ccs && rn > 0) {                                // [ckpt_fix] deferred prefix-cache checkpoint: this commit
    const int cc = p.ccs[b];                            // ends exactly on a Mamba block boundary
    if (cc >= 0 && cc < p.slots && cc != slot) {
      float* cp = p.ssm + (size_t)cc * p.ssm_stride + ((size_t)hv * D + ccol) * D + 4 * qd;
#pragma unroll
      for (int i = 0; i < 8; i++)
        *reinterpret_cast<float4*>(cp + 16 * i) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
    }
  }
  if (!TOK) return;

  // ------------------------------------------------------------------ 4. this round: updates in registers, outputs
  __half* const cog = p.core;
#if GDNV_TOKEN_UNROLL
#pragma unroll
#else
#pragma unroll 1
#endif
  for (int t = 0; t < TM; t++) {
    if (t >= L || (p.dbg & 2)) break;
    const float g = sm.g[t], beta = sm.beta[t], vv = sm.vc[t][cl];
    float k[8][4], a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 kv4 = *reinterpret_cast<const float4*>(&sm.kn[t][16 * i + 4 * qd]);
      k[i][0] = kv4.x; k[i][1] = kv4.y; k[i][2] = kv4.z; k[i][3] = kv4.w;
#pragma unroll
      for (int r = 0; r < 4; r++) a4[r] = fmaf(s[i][r], k[i][r], a4[r]);
    }
    float kv = (a4[0] + a4[1]) + (a4[2] + a4[3]);
    kv += __shfl_xor_sync(0xffffffffu, kv, 1);
    kv += __shfl_xor_sync(0xffffffffu, kv, 2);
    const float d = (vv - g * kv) * beta;
    float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 q4 = *reinterpret_cast<const float4*>(&sm.qn[t][16 * i + 4 * qd]);
      const float q[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
      for (int r = 0; r < 4; r++) {
        s[i][r] = fmaf(k[i][r], d, g * s[i][r]);
        o4[r] = fmaf(s[i][r], q[r], o4[r]);
      }
    }
    float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
    o += __shfl_xor_sync(0xffffffffu, o, 1);
    o += __shfl_xor_sync(0xffffffffu, o, 2);
    if (qd == 0) {
      const __half oh = __float2half(o * p.scale);
      if (NORM && NS == 1) sm.co[t][ccol] = oh;
      else cog[((size_t)(row0 + t) * HV + hv) * D + ccol] = oh;
    }
    if (t == ck) {                                      // prefix-cache checkpoint: the state after in-round token ck
      float* cp = p.ssm + (size_t)cslot * p.ssm_stride + ((size_t)hv * D + ccol) * D + 4 * qd;
#pragma unroll
      for (int i = 0; i < 8; i++)
        *reinterpret_cast<float4*>(cp + 16 * i) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
    }
  }

  // ------------------------------------------------------------------ 5. hand-offs: q/k window, norm
  constexpr bool HAND = NORM && NS > 1;
  if (HAND && qd == 0) __threadfence();                 // this thread's core outputs visible before the norm counter
  __syncthreads();                                      // (also: every warp's core outputs in smem when NS == 1)
  if (tid == 0) {
    int lq = 0, ln = 1;
    if (CONV) {
      lq = old_qk == REP * NS - 1;
      if (lq) p.cnt[b * HK + hq] = 0;
    }
    if (HAND && !(p.dbg & 4)) {
      int* cn = p.cnt + p.B * HK + b * HV + hv;
      ln = atomicAdd(cn, 1) == NS - 1;
      if (ln) { *cn = 0; __threadfence(); }
    }
    sm.last[0] = lq;
    sm.last[1] = ln;
  }
  if (FINAL) {                                          // decode: the new state, overlapping the counter round trips
#pragma unroll
    for (int i = 0; i < 8; i++)
      *reinterpret_cast<float4*>(sp + 16 * i) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
  }
  __syncthreads();
  if (CONV && sm.last[0]) {                             // the last reader of the q/k window rolls it back
    for (int i = tid; i < (L + 2) * (QKC / 4); i += NT) {
      const int r = i / (QKC / 4), c4 = (i % (QKC / 4)) * 4;
      *reinterpret_cast<uint2*>(cs + (size_t)r * DIM + qk_ch(hq, c4)) = *reinterpret_cast<const uint2*>(&sm.raw[r][c4]);
    }
  }
  if (!NORM || !sm.last[1] || (p.dbg & 4)) return;

  // ------------------------------------------------------------------ 6. gated RMSNorm * silu(z), by the CTA that
  // completes the head
#pragma unroll
  for (int i = 0; i < NZ; i++) {
    const int t = warp + i * WPC;
    if (t >= L) break;
    float xf[4], zf[4], wf[4];
    if (NS == 1) h4(*reinterpret_cast<const uint2*>(&sm.co[t][4 * lane]), xf);
    else h4(ldcg8(cog + ((size_t)(row0 + t) * HV + hv) * D + 4 * lane), xf);
    h4(zu[i], zf);
    h4(nu, wf);
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 4; e++) ss = fmaf(xf[e], xf[e], ss);
    ss = wsum(ss);
    const float inv = rsqrtf(ss * (1.f / D) + p.eps);
    __half2 o2[2];
#pragma unroll
    for (int e2 = 0; e2 < 2; e2++) {
      float v[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int kk = 2 * e2 + e;
        const float gz = zf[kk] / (1.f + __expf(-zf[kk]));
        v[e] = ((xf[kk] * inv) * wf[kk]) * gz;
      }
      o2[e2] = __floats2half2_rn(v[0], v[1]);
    }
    *reinterpret_cast<uint2*>(p.y + ((size_t)(row0 + t) * HV + hv) * D + 4 * lane) = *reinterpret_cast<const uint2*>(o2);
  }
}

template <int NCOL, int WPC, int TM, int MODE>
void launch(const Args& a, cudaStream_t st) {
  const dim3 grid(HV, a.B, D / (NCOL * WPC));
  k_gdnv<NCOL, WPC, TM, MODE><<<grid, 32 * WPC, 0, st>>>(a);
}

// configurations: (columns per warp, warps per CTA)
#define GDNV_CFGS(X, TM, MODE) \
  case 0: X(8, 1, TM, MODE); break; case 1: X(8, 2, TM, MODE); break; case 2: X(8, 4, TM, MODE); break; \
  case 3: X(8, 8, TM, MODE); break; case 4: X(8, 16, TM, MODE); break;
#define GDNV_L(NC_, W_, TM_, M_) launch<NC_, W_, TM_, M_>(a, st)

template <int TM, int MODE>
bool dispatch_cfg(int cfg, const Args& a, cudaStream_t st) {
  switch (cfg) {
    GDNV_CFGS(GDNV_L, TM, MODE)
    default: return false;
  }
  return true;
}

const char* CFG_NAMES[] = {"8x1", "8x2", "8x4", "8x8", "8x16"};   // columns per warp x warps per CTA
constexpr int NCFG = 5;

bool dispatch(int mode, int tm, int cfg, const Args& a, cudaStream_t st) {
  switch (mode) {
    case 0:   // VERIFY
      switch (tm) {
        case 2: return dispatch_cfg<2, VERIFY>(cfg, a, st);
        case 3: return dispatch_cfg<3, VERIFY>(cfg, a, st);
        case 4: return dispatch_cfg<4, VERIFY>(cfg, a, st);
        case 5: return dispatch_cfg<5, VERIFY>(cfg, a, st);
        case 8: return dispatch_cfg<8, VERIFY>(cfg, a, st);
      }
      return false;
    case 1:   // VERIFY_BA (rows > 32: one CTA per head or two slices)
#define GDNV_BA(TM_) \
  if (tm == TM_) { if (cfg == 3) { launch<8, 8, TM_, VERIFY_BA>(a, st); return true; } \
                   if (cfg == 4) { launch<8, 16, TM_, VERIFY_BA>(a, st); return true; } return false; }
      GDNV_BA(2) GDNV_BA(4) GDNV_BA(5)
#undef GDNV_BA
      return false;
    case 2:   // DECODE
      if (tm != 1) return false;                        // b/a operands: at least 4 warps
      if (cfg == 2) { launch<8, 4, 1, DECODE>(a, st); return true; }
      if (cfg == 3) { launch<8, 8, 1, DECODE>(a, st); return true; }
      if (cfg == 4) { launch<8, 16, 1, DECODE>(a, st); return true; }
      return false;
    case 3:   // VONLY (1Cat's mixed-batch route)
      if (tm != 8) return false;
      launch<8, 16, 8, VONLY>(a, st);
      return true;
    case 4:   // COMMIT
      launch<8, 16, 1, COMMIT>(a, st);
      return true;
  }
  return false;
}

const __half* hp(const torch::Tensor& t) { return reinterpret_cast<const __half*>(t.data_ptr()); }
const int* ip(const c10::optional<torch::Tensor>& t) {
  if (!t.has_value() || !t->defined() || t->numel() == 0) return nullptr;
  TORCH_CHECK(t->scalar_type() == at::kInt && t->is_cuda(), "int32 tensor expected");
  return t->data_ptr<int>();
}
}  // namespace

// mode: 0 VERIFY, 1 VERIFY_BA, 2 DECODE, 3 VONLY, 4 COMMIT; tm: per-request token capacity (template); B requests;
// T rows per request when cu is absent. Tensors not used by a mode may be empty.
void gdnv(int64_t mode, int64_t tm, int64_t cfg, torch::Tensor qkv, torch::Tensor z, torch::Tensor a, torch::Tensor b,
          torch::Tensor x, torch::Tensor wba, torch::Tensor cw, torch::Tensor cst, torch::Tensor A_log, torch::Tensor dtb,
          torch::Tensor ssm, torch::Tensor idx, c10::optional<torch::Tensor> cu, c10::optional<torch::Tensor> acc,
          c10::optional<torch::Tensor> rn, c10::optional<torch::Tensor> rmap, c10::optional<torch::Tensor> ckt,
          c10::optional<torch::Tensor> cks, torch::Tensor rbuf, int64_t tmax, torch::Tensor nw, double eps,
          torch::Tensor core, torch::Tensor y, torch::Tensor cnt, int64_t B, int64_t T, int64_t n, int64_t dbg,
          c10::optional<torch::Tensor> ccs) {
  TORCH_CHECK(ssm.scalar_type() == at::kFloat && ssm.dim() == 4 && ssm.size(1) == HV && ssm.size(2) == D &&
                  ssm.size(3) == D && ssm.stride(1) == D * D && ssm.stride(2) == D && ssm.stride(3) == 1 &&
                  ssm.stride(0) % 4 == 0, "ssm state [slots][12][128][128] fp32");
  TORCH_CHECK(idx.scalar_type() == at::kInt && idx.is_cuda() && idx.numel() >= B, "idx");
  TORCH_CHECK(rbuf.scalar_type() == at::kByte && tmax >= 1 && tmax <= RMAX, "rbuf / tmax");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= B * (HK + HV), "counter workspace");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && A_log.numel() == HV && dtb.scalar_type() == at::kHalf, "params");
  if (B <= 0) return;
  const bool tok = mode != 4, conv = mode <= 2, norm = mode <= 2, ba = mode == 1 || mode == 2;
  Args p{};
  if (tok) {
    TORCH_CHECK(qkv.scalar_type() == at::kHalf && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(1) >= DIM &&
                    qkv.size(0) >= n, "qkv");
    p.qkv = hp(qkv); p.ldq = qkv.stride(0);
  }
  if (conv) {
    TORCH_CHECK(cw.scalar_type() == at::kHalf && cw.is_contiguous() && cw.numel() == DIM * 4, "conv weight [2560][4]");
    TORCH_CHECK(cst.scalar_type() == at::kHalf && cst.dim() == 3 && cst.size(2) == DIM && cst.stride(2) == 1 &&
                    cst.stride(1) == DIM && cst.size(1) >= tm + 2 && cst.stride(0) % 4 == 0,
                "conv state [slots][3+K][2560] (SD)");
    p.cw = hp(cw); p.cst = reinterpret_cast<__half*>(cst.data_ptr()); p.cst_stride = cst.stride(0);
  }
  if (norm) {
    TORCH_CHECK(z.scalar_type() == at::kHalf && z.dim() == 2 && z.stride(1) == 1 && z.size(1) >= HV * D, "z");
    TORCH_CHECK(nw.scalar_type() == at::kHalf && nw.numel() == D && nw.is_contiguous(), "norm weight");
    TORCH_CHECK(y.scalar_type() == at::kHalf && y.is_contiguous() && y.numel() >= n * HV * D, "y");
    p.z = hp(z); p.ldz = z.stride(0); p.nw = hp(nw); p.y = reinterpret_cast<__half*>(y.data_ptr());
  }
  if (tok) {
    TORCH_CHECK(core.scalar_type() == at::kHalf && core.is_contiguous() && core.numel() >= n * HV * D, "core");
    p.core = reinterpret_cast<__half*>(core.data_ptr());
  }
  if (ba) {
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous() && x.size(1) == KH && x.size(0) >= n, "x");
    TORCH_CHECK(wba.scalar_type() == at::kHalf && wba.is_contiguous() && wba.size(0) == 2 * HV && wba.size(1) == KH,
                "wba");
    p.x = hp(x); p.wba = hp(wba);
  } else if (tok) {
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf && a.stride(1) == 1 &&
                    b.stride(1) == 1 && a.stride(0) == b.stride(0) && a.size(1) == HV, "a / b");
    p.av = hp(a); p.bv = hp(b); p.ldab = a.stride(0);
  }
  p.A_log = A_log.data_ptr<float>(); p.dtb = hp(dtb);
  p.ssm = ssm.data_ptr<float>(); p.ssm_stride = ssm.stride(0);
  p.slots = conv ? (int)std::min(ssm.size(0), cst.size(0)) : (int)ssm.size(0);
  p.idx = idx.data_ptr<int>(); p.idx_stride = idx.dim() == 2 ? (int)idx.stride(0) : (int)idx.stride(0);
  p.cu = ip(cu); p.acc = ip(acc); p.rn = ip(rn); p.rmap = ip(rmap); p.ckt = ip(ckt); p.cks = ip(cks);
  p.ccs = ip(ccs);                                                                                   // [ckpt_fix]
  TORCH_CHECK((p.rn == nullptr) == (p.rmap == nullptr) && (p.ckt == nullptr) == (p.cks == nullptr), "replay tables");
  p.rbuf = rbuf.data_ptr<uint8_t>(); p.tmax = (int)tmax;
  p.eps = (float)eps; p.cnt = cnt.data_ptr<int>();
  p.B = (int)B; p.T = (int)T; p.n = (int)n; p.scale = 1.f / sqrtf((float)D); p.dbg = (int)dbg;
  const at::cuda::CUDAGuard guard(ssm.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(dispatch((int)mode, (int)tm, (int)cfg, p, st), "gdnv: no kernel for mode ", mode, " tm ", tm, " cfg ", cfg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<std::string> gdnv_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NCFG; i++) v.push_back(CFG_NAMES[i]);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdnv", &gdnv, "fused GDN verify / decode core with replay records (gdn_verify.cu)");
  m.def("configs", &gdnv_configs);
  m.attr("REC") = REC;
}
