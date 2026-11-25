// fp8_gdn_in.cu (track smallrows) - the GDN layer's input projections in one launch for MTP verify / any decode batch of
// 1..64 rows: qkvz = x @ W_qkvz^T (FP8 E4M3 on 1Cat's QPN8 codes, read in place: the k_fp8 tile of fp8_tile.cuh)
// written split into qkv [M][2560] and z [M][1536] (the contiguous outputs 1Cat's verify chain takes), plus
// b, a = x @ W_ba^T (fp16 weight [2 heads][5120], fp32 accumulation) by NBA extra CTAs of the same grid.
// It replaces 1Cat's apply_gdn_ba_verify (fp8_qpn8_dispatch_ba_masked/split: GEMV at <= 8 rows, m32 two-phase at
// 17..32 rows: 57 us at 32 rows) and, above 32 rows, 1Cat's split route (QPN8 or fp8_lin.cu qkvz + cuBLAS b/a +
// the split-copy kernel); by default from 17 rows (smallrows.py GDNIN_MIN; at 5-16 rows it lost in the server).
// See notes/smallrows.md.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

#include "fp8_tile.cuh"

namespace {
using namespace q27fp8;
constexpr int NBA = 8, NPB = 3, RPB = 8;                           // b/a CTAs, outputs per CTA, rows per pass

__device__ __forceinline__ void h8f(const uint4& u, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// one b/a CTA: outputs n0 .. n0 + NPB - 1 of [b | a] (2 heads), all M rows in passes of RPB rows
template <int NTH>
__device__ __forceinline__ void ba_block(const __half* __restrict__ X, const __half* __restrict__ Wba,
                                         __half* __restrict__ B, __half* __restrict__ A, int M, int K, int heads,
                                         int cta, uint8_t* sm) {
  constexpr int NW = NTH / 32;
  float* red = reinterpret_cast<float*>(sm);                       // [NW][NPB * RPB]
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5, n0 = cta * NPB, KV = K / 8;
  for (int r0 = 0; r0 < M; r0 += RPB) {
    float acc[NPB][RPB];
#pragma unroll
    for (int j = 0; j < NPB; j++)
#pragma unroll
      for (int r = 0; r < RPB; r++) acc[j][r] = 0.f;
    for (int v = t; v < KV; v += NTH) {
      float wf[NPB][8];
#pragma unroll
      for (int j = 0; j < NPB; j++) h8f(__ldg(reinterpret_cast<const uint4*>(Wba + (size_t)(n0 + j) * K) + v), wf[j]);
#pragma unroll
      for (int r = 0; r < RPB; r++) {
        float xf[8];
        h8f(__ldg(reinterpret_cast<const uint4*>(X + (size_t)min(r0 + r, M - 1) * K) + v), xf);
#pragma unroll
        for (int j = 0; j < NPB; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc[j][r] = fmaf(xf[e], wf[j][e], acc[j][r]);
      }
    }
#pragma unroll
    for (int j = 0; j < NPB; j++)
#pragma unroll
      for (int r = 0; r < RPB; r++) {
        float v = acc[j][r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[wp * NPB * RPB + j * RPB + r] = v;
      }
    __syncthreads();
    if (t < NPB * RPB) {
      const int j = t / RPB, r = t % RPB, n = n0 + j, row = r0 + r;
      if (row < M) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NW; w++) v += red[w * NPB * RPB + t];
        if (n < heads) B[(size_t)row * heads + n] = __float2half_rn(v);
        else A[(size_t)row * heads + (n - heads)] = __float2half_rn(v);
      }
    }
    __syncthreads();
  }
}

// grid = nba + N / (32 CB NCOL): CTAs 0..nba-1 compute b/a, the rest are FP8 tiles
template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int PR, int CG>
__global__ void __launch_bounds__(KP8<RB, CB, WK, NCOL, NROW, 4, PR>::NTH, MB)
k_gdn_in8(const uint8_t* __restrict__ W, const __half* __restrict__ X, const __half* __restrict__ S,
          __half* __restrict__ Yqkv, __half* __restrict__ Yz, int split, int M, int N, int K,
          const __half* __restrict__ Wba, __half* __restrict__ B, __half* __restrict__ A, int heads, int nba, int zero) {
  extern __shared__ __align__(16) uint8_t sm[];
  if ((int)blockIdx.x < nba) {
    ba_block<KP8<RB, CB, WK, NCOL, NROW, 4, PR>::NTH>(X, Wba, B, A, M, K, heads, blockIdx.x, sm);
  } else {
    fp8_tile<RB, CB, WK, NCOL, NROW, 4, PR, CG>(W, X, S, Yqkv, M, N, K, zero, blockIdx.x - nba, sm, Yz, split);
  }
}

// CG = 1: ld.global.cg weight loads (server A/B 2026-11-22: better at 64 rows, worse at 32; see GDNIN_CFG in smallrows.py)
template <int RB, int CB, int WK, int NCOL, int NROW, int MB, int CG>
void launch(const uint8_t* W, const __half* X, const __half* S, __half* Yq, __half* Yz, int split, int M, int N, int K,
             const __half* Wba, __half* B, __half* A, int heads, int nba, cudaStream_t st) {
  using P = KP8<RB, CB, WK, NCOL, NROW, 4, 1>;
  auto fn = k_gdn_in8<RB, CB, WK, NCOL, NROW, MB, 1, CG>;
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
  static_assert(P::SMEM >= (P::NTH / 32) * NPB * RPB * 4, "b/a reduction fits the tile's shared memory");
  TORCH_CHECK(N % (32 * CB * NCOL) == 0 && split % (32 * CB * NCOL) == 0, "N and split: whole CTAs");
  fn<<<nba + N / (32 * CB) / NCOL, P::NTH, P::SMEM, st>>>(W, X, S, Yq, Yz, split, M, N, K, Wba, B, A, heads, nba, 0);
}
using LaunchFn = void (*)(const uint8_t*, const __half*, const __half*, __half*, __half*, int, int, int, int,
                          const __half*, __half*, __half*, int, int, cudaStream_t);
struct Cfg { int r8; const char* name; LaunchFn fn; int cols; };
// the parallel-reduction shapes of fp8_lin.cu (same names) that fit the qkvz projection (N 4096, K 5120); "cg" = with
// .cg weight loads
const Cfg CFGS[] = {
    {64, "s 4x1 wk3 col2 row2 pr", launch<4, 1, 3, 2, 2, 1, 0>, 64},
    {64, "s 4x1 wk3 col2 row2 pr cg", launch<4, 1, 3, 2, 2, 1, 1>, 64},
    {48, "s 6x1 wk4 col2 pr", launch<6, 1, 4, 2, 1, 1, 0>, 64},
    {48, "s 6x1 wk4 col2 pr cg", launch<6, 1, 4, 2, 1, 1, 1>, 64},
    {32, "s 4x1 wk4 col2 pr", launch<4, 1, 4, 2, 1, 2, 0>, 64},
    {32, "s 4x1 wk4 col2 pr cg", launch<4, 1, 4, 2, 1, 2, 1>, 64},
    {16, "s 2x1 wk8 col2 pr", launch<2, 1, 8, 2, 1, 1, 0>, 64},
    {8, "s 1x1 wk8 col1 pr", launch<1, 1, 8, 1, 1, 2, 0>, 32},
};
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);
}  // namespace

// qkv [M][split], z [M][N - split] = x [M][K] @ QPN8 (codes [K*N] u8, scales [N] fp16 = scale * 256)^T; with_ba: also
// b, a [M][heads] = x @ wba [2 heads][K]^T (fp16). M <= the config's row bucket.
void gdn_in8(torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor wba, torch::Tensor qkv,
             torch::Tensor z, torch::Tensor b, torch::Tensor a, int64_t cfg, bool with_ba) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(codes.dtype() == torch::kUInt8 && codes.is_contiguous() && scales.dtype() == torch::kHalf && scales.is_contiguous());
  TORCH_CHECK(qkv.dtype() == torch::kHalf && qkv.is_contiguous() && z.dtype() == torch::kHalf && z.is_contiguous());
  const int M = x.size(0), K = x.size(1), split = qkv.size(1), N = split + z.size(1);
  TORCH_CHECK(qkv.size(0) == M && z.size(0) == M && codes.numel() == (int64_t)N * K && scales.numel() == N && K % 32 == 0);
  TORCH_CHECK(cfg >= 0 && cfg < NCFG && M >= 1 && M <= CFGS[cfg].r8, "config's row bucket does not cover M");
  int heads = 0;
  if (with_ba) {
    TORCH_CHECK(wba.dtype() == torch::kHalf && wba.is_contiguous() && wba.dim() == 2 && wba.size(1) == K);
    heads = wba.size(0) / 2;
    TORCH_CHECK(2 * heads == NBA * NPB && b.is_contiguous() && a.is_contiguous() && b.size(0) == M && a.size(0) == M &&
                b.size(1) == heads && a.size(1) == heads && b.dtype() == torch::kHalf && a.dtype() == torch::kHalf);
  }
  const at::cuda::CUDAGuard g(x.device());
  CFGS[cfg].fn(codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
               reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), reinterpret_cast<__half*>(qkv.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(z.data_ptr<at::Half>()), split, M, N, K,
               with_ba ? reinterpret_cast<const __half*>(wba.data_ptr<at::Half>()) : nullptr,
               with_ba ? reinterpret_cast<__half*>(b.data_ptr<at::Half>()) : nullptr,
               with_ba ? reinterpret_cast<__half*>(a.data_ptr<at::Half>()) : nullptr, heads, with_ba ? NBA : 0,
               at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> gdn_in8_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NCFG; i++) v.push_back(std::to_string(CFGS[i].r8) + ":" + std::to_string(CFGS[i].cols) + ":" + CFGS[i].name);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_in8", &gdn_in8, "GDN qkv | z (FP8 QPN8) + b/a (fp16) projections in one launch");
  m.def("gdn_in8_configs", &gdn_in8_configs, "'rows:columns per CTA:name'");
}
