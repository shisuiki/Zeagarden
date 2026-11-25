// SPDX-License-Identifier: Apache-2.0
// fp8_dense.cu - Single-pass Volta (sm_70) OCP FP8 E4M3FN (per-channel scale) W8A16 GEMM
// for Inkling-2-Mini dense projections (qkvr, wo_ud, lm_head) on 1Cat's in-place QPN8 layout.
//
// Features:
//   - Zero permanent FP16 weight copies: reads 1Cat's QPN8 layout ([K, N] u8 + [N] fp16 s16 = scale*256
//     + [N] fp32 s32) directly in registers with branchless byte_perm + shift E4M3FN -> FP16*2^-8 decode.
//   - Single-pass M = 1..64 via Volta HMMA.884 (FP32 accumulation) with staged SMEM A-fragments
//     and parallel/serial K-slice reduction (zero register spills, 0 B local stack).
//   - Two output epilogues:
//       OUT_F32 = 0: FP16 [M, N_out] (used by qkvr, wo_ud, DSpark base_logits)
//       OUT_F32 = 1: FP32 [M, N_out] (used by Inkling2 target lm_head before softcap=30.0!)
//   - Supports N_out <= N_pad cleanly (e.g. qkvr N_out=2152 padded to N_pad=2176,
//     q_side N_out=1640 padded to 1664, lm_head N_out=50256 padded to 50304).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr int CH8 = 1024;  // bytes per (32-column tile, 32-k chunk) in QPN8 layout

__device__ __forceinline__ void mma884(
    float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16na(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 lda16(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ void bar_sync_n(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// Bytes 0 and 2 of p (OCP FP8 E4M3FN) -> fp16x2 = exact E4M3FN * 2^-8.
__device__ __forceinline__ unsigned f8x2(unsigned p) {
  return ((p << 8) & 0x80008000u) | ((p << 7) & 0x3F803F80u);
}

template <int RB, int CB, int WK, int NCOL, int NROW, int PR>
struct KP8 {
  static constexpr int NCW = WK * NCOL * NROW;
  static constexpr int RBT = RB * NROW;
  static constexpr int R8 = RBT * 8;
  static constexpr int NR = RB * CB * 8;
  static constexpr int TS = NROW * NR * 32;
  static constexpr int AB = RBT * 512;
  static constexpr int REDB = NCOL * TS * 4 * (PR ? WK : 1);
  static constexpr int RING = WK * 2 * AB;
  static constexpr int SMEM = REDB > RING ? REDB : RING;
  static constexpr int NTH = 32 * NCW;
};

template <int RB, int CB, class AGet>
__device__ __forceinline__ void chunk8_episcale(
    const uint4 (&w)[CB][2], AGet aget, float (&acc)[RB][CB][8]) {
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    uint4 a[RB];
#pragma unroll
    for (int rb = 0; rb < RB; rb++) a[rb] = aget(jj, rb);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const uint4& g = w[cb][jj >> 1];
      const unsigned qx = (jj & 1) ? g.z : g.x;
      const unsigned qy = (jj & 1) ? g.w : g.y;
      const unsigned p0 = f8x2(__byte_perm(qx, qy, 0x0400));
      const unsigned p1 = f8x2(__byte_perm(qx, qy, 0x0501));
      const unsigned p2 = f8x2(__byte_perm(qx, qy, 0x0602));
      const unsigned p3 = f8x2(__byte_perm(qx, qy, 0x0703));
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        mma884(acc[rb][cb], a[rb].x, a[rb].y, p0, p1);
      }
#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        mma884(acc[rb][cb], a[rb].z, a[rb].w, p2, p3);
      }
    }
  }
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int PR, int OUT_F32>
__global__ void __launch_bounds__(KP8<RB, CB, WK, NCOL, NROW, PR>::NTH, MB)
k_fp8_dense(
    const uint8_t* __restrict__ W,
    const __half* __restrict__ X,
    const __half* __restrict__ S16,
    const float* __restrict__ S32,
    void* __restrict__ Y_raw,
    int M,
    int N_pad,
    int N_out,
    int K,
    int zero) {
  extern __shared__ __align__(16) uint8_t sm[];
  using P = KP8<RB, CB, WK, NCOL, NROW, PR>;
  static_assert(WK <= 14 && (RB * NROW) % (NCOL * NROW) == 0,
                "named barriers 1..WK, whole staging pieces");
  constexpr int NCW = P::NCW;
  constexpr int R8 = P::R8;
  constexpr int NR = P::NR;
  constexpr int AB = P::AB;
  constexpr int G = NCOL * NROW;
  constexpr int NPC = P::RBT / G;

  float* red = reinterpret_cast<float*>(sm);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int bx = blockIdx.x;
  const int NC = K / 32;
  const int rin = (lane & 3) + 4 * (lane >> 4);
  const int wk = warp % WK;
  const int gi = (warp / WK) % NCOL;
  const int gr = warp / (WK * NCOL);
  const int t = bx * NCOL + gi;
  const int s0 = NC * wk / WK;
  const int n = NC * (wk + 1) / WK - s0;
  const unsigned zmask = (unsigned)zero;

  const uint8_t* wp = W + ((size_t)t * CB * NC + s0) * CH8 + lane * 16;
  const size_t cbs = (size_t)NC * CH8;

  float acc[RB][CB][8];
#pragma unroll
  for (int i = 0; i < RB; i++)
#pragma unroll
    for (int j = 0; j < CB; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;

  unsigned xr = 0;
  auto ldw = [&](uint4 (&w)[CB][2], int i, unsigned dep) {
    const uint8_t* pw = wp + (size_t)i * CH8 + dep;
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      w[cb][0] = ldg16na(pw + cb * cbs);
      w[cb][1] = ldg16na(pw + cb * cbs + 512);
    }
  };

  const int gidx = gr * NCOL + gi;
  uint8_t* abuf = sm + (size_t)wk * 2 * AB;
  uint4 wA[CB][2], wB[CB][2], stg[NPC];
  auto slice_bar = [&]() {
    if (G > 1)
      bar_sync_n(1 + wk, G * 32);
    else
      __syncwarp();
  };
  auto ldx = [&](int i, unsigned dep) {
    const int ch = s0 + i;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = (q * G + gidx) * 32 + lane;
      const int row = min(u >> 2, M - 1);
      const int jj = u & 3;
      stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)row * K + ch * 32 + jj * 8) + dep);
    }
  };
  auto stx = [&](int i) {
    uint8_t* pb = abuf + (i & 1) * AB;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = (q * G + gidx) * 32 + lane;
      *reinterpret_cast<uint4*>(pb + ((u & 3) * R8 + (u >> 2)) * 16) = stg[q];
    }
  };
  auto touch = [&](const uint4 (&w)[CB][2]) {
    unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
    for (int cb = 0; cb < CB; cb++) x ^= w[cb][0].x ^ w[cb][1].x;
#pragma unroll
    for (int q = 0; q < NPC; q++) x ^= stg[q].x;
    xr ^= x;
    return x & zmask;
  };
  auto step = [&](const uint4 (&wc)[CB][2], uint4 (&wn)[CB][2], int i, unsigned& dep) {
    const int inx = min(i + 1, n - 1);
    ldw(wn, inx, dep);
    ldx(inx, dep);
    const uint8_t* ap = abuf + (i & 1) * AB + (gr * RB * 8 + rin) * 16;
    chunk8_episcale<RB, CB>(
        wc,
        [&](int jj, int rb) {
          return *reinterpret_cast<const uint4*>(ap + (jj * R8 + rb * 8) * 16);
        },
        acc);
    dep = touch(wn);
    stx(i + 1);
    slice_bar();
  };

  if (n > 0) {
    ldw(wA, 0, 0u);
    ldx(0, 0u);
    unsigned dep = touch(wA);
    stx(0);
    slice_bar();
    int i = 0;
    for (; i + 2 <= n; i += 2) {
      step(wA, wB, i, dep);
      step(wB, wA, i + 1, dep);
    }
    if (i < n) step(wA, wB, i, dep);
  }
  bar_sync_n(15, NCW * 32);
  if (xr == 0x9E3779B9u && zero == 1) {
    if constexpr (OUT_F32)
      reinterpret_cast<float*>(Y_raw)[0] = 0.f;
    else
      reinterpret_cast<__half*>(Y_raw)[0] = __float2half(0.f);
  }

  constexpr int SLAB = NCOL * P::TS;
  if constexpr (PR) {
    float* rt = red + wk * SLAB + (gi * NROW + gr) * NR * 32;
#pragma unroll
    for (int i = 0; i < RB; i++)
#pragma unroll
      for (int j = 0; j < CB; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) rt[((i * CB + j) * 8 + e) * 32 + lane] = acc[i][j][e];
    bar_sync_n(15, NCW * 32);
  } else {
    float* rt = red + (gi * NROW + gr) * NR * 32;
    for (int w = 0; w < WK; w++) {
      if (wk == w)
#pragma unroll
        for (int i = 0; i < RB; i++)
#pragma unroll
          for (int j = 0; j < CB; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
              float& r = rt[((i * CB + j) * 8 + e) * 32 + lane];
              r = w ? r + acc[i][j][e] : acc[i][j][e];
            }
      bar_sync_n(15, NCW * 32);
    }
  }

  auto rv = [&](int idx) {
    float v = red[idx];
    if constexpr (PR) {
#pragma unroll
      for (int w = 1; w < WK; w++) v += red[w * SLAB + idx];
    }
    return v;
  };
  auto ridx = [&](int c, int row, int col) {
    const int grr = row / (RB * 8);
    const int lrow = row % (RB * 8);
    const int rb = lrow >> 3;
    const int r8 = lrow & 7;
    const int cb = col >> 5;
    const int cc = col & 31;
    const int q = cc >> 3;
    const int cq = cc & 7;
    const int l = (r8 & 1) | (cq & 2) | (q << 2) | (((r8 >> 2) & 1) << 4);
    const int e = (r8 & 2) | (cq & 4) | (cq & 1);
    return (c * NROW + grr) * NR * 32 + ((rb * CB + cb) * 8 + e) * 32 + l;
  };

  constexpr int TC = 32 * CB;
  const int rows = min(M, R8);
  for (int idx = threadIdx.x; idx < NCOL * rows * (TC / 2); idx += NCW * 32) {
    const int c = idx / (rows * (TC / 2));
    const int rr = idx % (rows * (TC / 2));
    const int row = rr / (TC / 2);
    const int col = (rr % (TC / 2)) * 2;
    const int gcol = (bx * NCOL + c) * TC + col;
    if (gcol >= N_out) continue;

    if constexpr (OUT_F32) {
      // Exact FP32 scale * 256 (undoes E4M3FN * 2^-8 without FP16 scale quantization!)
      const float s0 = S32[gcol] * 256.f;
      const float s1 = S32[gcol + 1] * 256.f;
      const float v0 = rv(ridx(c, row, col)) * s0;
      const float v1 = rv(ridx(c, row, col + 1)) * s1;
      float* yp = reinterpret_cast<float*>(Y_raw) + (size_t)row * N_out + gcol;
      if (gcol + 1 < N_out) {
        *reinterpret_cast<float2*>(yp) = make_float2(v0, v1);
      } else {
        yp[0] = v0;
      }
    } else {
      const __half2 s = *reinterpret_cast<const __half2*>(S16 + gcol);
      const float v0 = rv(ridx(c, row, col)) * __low2float(s);
      const float v1 = rv(ridx(c, row, col + 1)) * __high2float(s);
      __half* yp = reinterpret_cast<__half*>(Y_raw) + (size_t)row * N_out + gcol;
      if (gcol + 1 < N_out) {
        *reinterpret_cast<__half2*>(yp) = __floats2half2_rn(v0, v1);
      } else {
        yp[0] = __float2half_rn(v0);
      }
    }
  }
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int PR, int OUT_F32>
void launch_one(
    const uint8_t* W,
    const __half* X,
    const __half* S16,
    const float* S32,
    void* Y,
    int M,
    int N_pad,
    int N_out,
    int K,
    cudaStream_t st) {
  using P = KP8<RB, CB, WK, NCOL, NROW, PR>;
  auto fn = k_fp8_dense<RB, CB, WK, NCOL, NROW, MB, PR, OUT_F32>;
  static bool init = false;
  if (!init) {
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    int occ = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, P::NTH, P::SMEM) == cudaSuccess);
    const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * P::SMEM + 98303) / 98304));
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess);
    init = true;
  }
  TORCH_CHECK(N_pad % (32 * CB * NCOL) == 0, "fp8_dense: N_pad must be a multiple of CTA columns");
  fn<<<N_pad / (32 * CB * NCOL), P::NTH, P::SMEM, st>>>(W, X, S16, S32, Y, M, N_pad, N_out, K, 0);
}

using LaunchFn = void (*)(
    const uint8_t*, const __half*, const __half*, const float*, void*,
    int, int, int, int, cudaStream_t);

struct Cfg {
  int r8;
  int cols;
  const char* name;
  LaunchFn fn_f16;
  LaunchFn fn_f32;
};

#define CF(R8, COLS, NAME, RB, CB, WK, NCOL, NROW, MB, PR) \
  {R8, COLS, NAME,                                         \
   launch_one<RB, CB, WK, NCOL, NROW, MB, PR, 0>,          \
   launch_one<RB, CB, WK, NCOL, NROW, MB, PR, 1>}

const Cfg CFGS[] = {
    CF(8,  32, "1x1 wk8 col1 pr",      1, 1, 8, 1, 1, 2, 1),
    CF(8,  32, "1x1 wk4 col1 pr",      1, 1, 4, 1, 1, 4, 1),
    CF(16, 32, "2x1 wk8 col1 pr",      2, 1, 8, 1, 1, 2, 1),
    CF(16, 64, "2x1 wk8 col2 pr",      2, 1, 8, 2, 1, 1, 1),
    CF(16, 64, "2x1 wk4 col2 pr",      2, 1, 4, 2, 1, 2, 1),
    CF(32, 32, "4x1 wk4 col1 pr",      4, 1, 4, 1, 1, 2, 1),
    CF(32, 32, "4x1 wk8 col1 pr0",     4, 1, 8, 1, 1, 1, 0),
    CF(32, 64, "4x1 wk4 col2 pr",      4, 1, 4, 2, 1, 2, 1),
    CF(32, 64, "4x1 wk8 col2 pr0",     4, 1, 8, 2, 1, 1, 0),
    CF(32, 128,"4x1 wk4 col4 pr0",     4, 1, 4, 4, 1, 1, 0),
    CF(48, 32, "6x1 wk4 col1 pr",      6, 1, 4, 1, 1, 1, 1),
    CF(48, 64, "6x1 wk4 col2 pr",      6, 1, 4, 2, 1, 1, 1),
    CF(64, 32, "4x1 wk4 col1 row2 pr", 4, 1, 4, 1, 2, 1, 1),
    CF(64, 32, "4x1 wk3 col1 row2 pr", 4, 1, 3, 1, 2, 1, 1),
    CF(64, 32, "4x1 wk2 col1 row2 pr", 4, 1, 2, 1, 2, 2, 1),
    CF(64, 32, "8x1 wk4 col1 pr0",     8, 1, 4, 1, 1, 1, 0),
    CF(64, 32, "8x1 wk2 col1 pr0",     8, 1, 2, 1, 1, 2, 0),
    CF(64, 64, "4x1 wk3 col2 row2 pr", 4, 1, 3, 2, 2, 1, 1),
    CF(64, 64, "4x1 wk2 col2 row2 pr", 4, 1, 2, 2, 2, 2, 1),
    CF(64, 64, "8x1 wk2 col2 pr0",     8, 1, 2, 2, 1, 2, 0),
    CF(64, 128,"8x1 wk2 col4 pr0",     8, 1, 2, 4, 1, 1, 0),
};
#undef CF
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);

}  // namespace

void fp8_dense_linear(
    torch::Tensor out,
    torch::Tensor x,
    torch::Tensor codes,
    torch::Tensor s16,
    torch::Tensor s32,
    int64_t n_pad,
    int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous());
  const bool out_f32 = (out.dtype() == torch::kFloat);
  TORCH_CHECK(out_f32 || out.dtype() == torch::kHalf, "fp8_dense: out must be fp16 or fp32");
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N_out = (int)out.size(1);
  const int N_pad = (int)n_pad;
  TORCH_CHECK(out.size(0) == M && N_out <= N_pad && N_out % 2 == 0);
  TORCH_CHECK(codes.numel() == (int64_t)N_pad * K && K % 32 == 0);
  TORCH_CHECK(s16.dtype() == torch::kHalf && s16.is_contiguous() && s16.numel() == N_pad);
  TORCH_CHECK(s32.dtype() == torch::kFloat && s32.is_contiguous() && s32.numel() == N_pad);
  TORCH_CHECK(cfg >= 0 && cfg < NCFG && M >= 1 && M <= CFGS[cfg].r8, "fp8_dense: invalid cfg or M > r8");

  const at::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (out_f32) {
    CFGS[cfg].fn_f32(
        codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(s16.data_ptr<at::Half>()),
        s32.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N_pad, N_out, K, st);
  } else {
    CFGS[cfg].fn_f16(
        codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(s16.data_ptr<at::Half>()),
        s32.data_ptr<float>(),
        out.data_ptr<at::Half>(),
        M, N_pad, N_out, K, st);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> fp8_dense_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NCFG; i++) {
    v.push_back(std::to_string(CFGS[i].r8) + ":" + std::to_string(CFGS[i].cols) + ":" + CFGS[i].name);
  }
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_dense_linear", &fp8_dense_linear, "Volta sm_70 single-pass QPN8 FP8 W8A16 linear (fp16 or fp32 out)");
  m.def("fp8_dense_configs", &fp8_dense_configs, "Available kernel configs 'r8:cols:name'");
}
