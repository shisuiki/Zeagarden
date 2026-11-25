// fp8_lin.cu - FP8 (E4M3, one scale per output channel) W8A16 linear for Qwen3.8-27B decode on V100, 9..64 rows, one
// pass over the weights, reading 1Cat's QPN8 weight layout in place (no second copy). 1Cat's own QPN8 kernels stop at
// 32 rows, so 33..64 rows read every weight twice there.
//   - QPN8 layout (1Cat csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu, fp8_qpn8_prepack_sm70_kernel): per 32-column tile and
//     16-k group, 512 B: lane l holds output column 8 ((l >> 2) & 3) + (l & 3) + 4 (l >> 4) (the mma B-fragment column,
//     the same map as mlp_fp4.cu) x 16 consecutive k, bytes ordered k 0 2 4 6 1 3 5 7 per 8, so one byte_perm + two
//     shifts give fp16x2 (k, k + 1) = E4M3 * 2^-8 exactly. Tiles are k-group-minor: a 32-k chunk is 1024 contiguous B.
//   - scales: QPN8's channel scales, fp16 = scale * 256 (undoes the 2^-8), applied in the epilogue.
//   - kernel structure from mlp_fp4.cu MODE 4 (ladder/REPORT.md §7): K split across a CTA's warps, weights double-
//     buffered in registers with one forced wait per chunk, activations staged per k-slice from the row-major input in
//     a shared-memory double buffer by the slice's own warps and read as mma fragments; slices reduced through shared
//     memory; output row-major [M][N] fp16.
//   - tried and dropped (bench_fp8.py, 2026-11-20): A fragments loaded straight from the row-major input as 1Cat's M32
//     kernel does (232 registers at 64 rows, 1.9x slower), L1-allocated weight loads for two row groups (no change).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

#include "fp8_tile.cuh"

namespace {
using namespace q27fp8;

template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int MB, int PR = 0>   // smallrows: PR
__global__ void __launch_bounds__(KP8<RB, CB, WK, NCOL, NROW, MODE, PR>::NTH, MB)
k_fp8(const uint8_t* __restrict__ W, const __half* __restrict__ X, const __half* __restrict__ S, __half* __restrict__ Y,
      int M, int N, int K, int zero) {
  extern __shared__ __align__(16) uint8_t sm[];
  fp8_tile<RB, CB, WK, NCOL, NROW, MODE, PR>(W, X, S, Y, M, N, K, zero, blockIdx.x, sm);
}

template <int RB, int CB, int WK, int NCOL, int NROW, int MODE, int MB, int PR = 0>   // smallrows: PR
void launch(const uint8_t* W, const __half* X, const __half* S, __half* Y, int M, int N, int K, cudaStream_t st) {
  using P = KP8<RB, CB, WK, NCOL, NROW, MODE, PR>;
  auto fn = k_fp8<RB, CB, WK, NCOL, NROW, MODE, MB, PR>;
  static bool init = false;
  if (!init) {
    // the smallest shared-memory carveout that keeps the occupancy MB (the rest stays L1)
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    int occ = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, P::NTH, P::SMEM) == cudaSuccess);
    const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * P::SMEM + 98303) / 98304));
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess);
    init = true;
  }
  TORCH_CHECK(N % (32 * CB * NCOL) == 0, "N must be a multiple of the CTA's columns");
  fn<<<N / (32 * CB) / NCOL, P::NTH, P::SMEM, st>>>(W, X, S, Y, M, N, K, 0);
}

// smallrows: the FP8 MLP gate_up with the SiLU-and-mul epilogue (fp8_tile GATED): N = 2 I, Y [M][I]
template <int RB, int WK, int NCOL, int NROW, int MB>
__global__ void __launch_bounds__(KP8<RB, 2, WK, NCOL, NROW, 4, 1>::NTH, MB)
k_fp8g(const uint8_t* __restrict__ W, const __half* __restrict__ X, const __half* __restrict__ S, __half* __restrict__ Y,
       int M, int N, int K, int zero) {
  extern __shared__ __align__(16) uint8_t sm[];
  fp8_tile<RB, 2, WK, NCOL, NROW, 4, 1, 0, 1>(W, X, S, Y, M, N, K, zero, blockIdx.x, sm);
}
template <int RB, int WK, int NCOL, int NROW, int MB>
void launchg(const uint8_t* W, const __half* X, const __half* S, __half* Y, int M, int N, int K, cudaStream_t st) {
  using P = KP8<RB, 2, WK, NCOL, NROW, 4, 1>;
  auto fn = k_fp8g<RB, WK, NCOL, NROW, MB>;
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
  TORCH_CHECK(N % (64 * NCOL) == 0, "gated: N / 2 must be a multiple of the CTA's columns");
  fn<<<N / 64 / NCOL, P::NTH, P::SMEM, st>>>(W, X, S, Y, M, N, K, 0);
}

using LaunchFn = void (*)(const uint8_t*, const __half*, const __half*, __half*, int, int, int, cudaStream_t);
struct Cfg { int r8; const char* name; LaunchFn fn; int cols; };
// candidate configurations (bench_fp8.py picks one per shape and row bucket): <RB, CB, WK, NCOL, NROW, MB>
// (ops.py FP8_CFG picks per row bucket; bench_fp8.py compares them)
const Cfg CFGS[] = {
    {64, "s 4x1 wk3 col2 row2", launch<4, 1, 3, 2, 2, 4, 1>, 64},
    {64, "s 4x1 wk2 col2 row2", launch<4, 1, 2, 2, 2, 4, 2>, 64},
    {64, "s 8x1 wk4 col2", launch<8, 1, 4, 2, 1, 4, 1>, 64},
    {32, "s 4x1 wk4 col2", launch<4, 1, 4, 2, 1, 4, 2>, 64},
    {32, "s 4x1 wk8 col2", launch<4, 1, 8, 2, 1, 4, 1>, 64},
    {16, "s 2x1 wk8 col2", launch<2, 1, 8, 2, 1, 4, 1>, 64},
    {16, "s 2x1 wk4 col2", launch<2, 1, 4, 2, 1, 4, 2>, 64},
    // one 32-column tile per CTA (the GEMM CTA shapes of the fused GDN layer kernel, ../../attic/gdn_core_fused.cu)
    {8, "s 1x1 wk8 col1", launch<1, 1, 8, 1, 1, 4, 2>, 32},
    {8, "s 1x1 wk4 col1", launch<1, 1, 4, 1, 1, 4, 4>, 32},
    {16, "s 2x1 wk8 col1", launch<2, 1, 8, 1, 1, 4, 2>, 32},
    // smallrows: the shapes of smallrows.FP8_CFG (parallel k-slice reduction: bitwise equal to the serial twins above;
    // 16-row shapes for the cutover of 1Cat's paired m16 kernel at 9..16 rows, a 48-row shape for 33..48 rows)
    {64, "s 4x1 wk3 col2 row2 pr", launch<4, 1, 3, 2, 2, 4, 1, 1>, 64},
    {48, "s 6x1 wk4 col2 pr", launch<6, 1, 4, 2, 1, 4, 1, 1>, 64},
    {32, "s 4x1 wk4 col2 pr", launch<4, 1, 4, 2, 1, 4, 2, 1>, 64},
    {16, "s 2x1 wk8 col2 pr", launch<2, 1, 8, 2, 1, 4, 1, 1>, 64},
    {16, "s 2x1 wk4 col2 pr", launch<2, 1, 4, 2, 1, 4, 2, 1>, 64},
};
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);
// gated gate_up shapes (cols = output columns per CTA)
const Cfg GCFGS[] = {
    {16, "g 2x2 wk4 col1 pr", launchg<2, 4, 1, 1, 2>, 32},
    {32, "g 4x2 wk4 col1 pr", launchg<4, 4, 1, 1, 2>, 32},
    {48, "g 3x2 wk4 col1 row2 pr", launchg<3, 4, 1, 2, 1>, 32},
    {64, "g 4x2 wk3 col1 row2 pr", launchg<4, 3, 1, 2, 1>, 32},
};
constexpr int NGCFG = sizeof(GCFGS) / sizeof(GCFGS[0]);
}  // namespace

// out [M][N] fp16 = x [M][K] fp16 @ (QPN8 codes [N*K] u8, channel scales [N] fp16 = scale * 256)^T, M <= 64, one
// pass. cfg: index into CFGS (its row bucket must cover M).
void fp8_linear(torch::Tensor out, torch::Tensor x, torch::Tensor codes, torch::Tensor scales, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf && scales.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() && out.dim() == 2);
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(out.size(0) == M && codes.numel() == (int64_t)N * K && scales.numel() == N && K % 32 == 0);
  TORCH_CHECK(cfg >= 0 && cfg < NCFG && M >= 1 && M <= CFGS[cfg].r8, "config's row bucket does not cover M");
  const at::cuda::CUDAGuard g(x.device());
  CFGS[cfg].fn(codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
               reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
               M, N, K, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> fp8_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NCFG; i++) v.push_back(std::to_string(CFGS[i].r8) + ":" + std::to_string(CFGS[i].cols) + ":" + CFGS[i].name);
  return v;
}

// smallrows: out [M][N/2] = silu(x @ Wg^T) * (x @ Wu^T) for the QPN8 codes of a merged [gate | up] weight (N = 2 I)
void fp8_gated(torch::Tensor out, torch::Tensor x, torch::Tensor codes, torch::Tensor scales, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf && scales.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() && out.dim() == 2);
  const int M = x.size(0), K = x.size(1), N = 2 * out.size(1);
  TORCH_CHECK(out.size(0) == M && codes.numel() == (int64_t)N * K && scales.numel() == N && K % 32 == 0);
  TORCH_CHECK(cfg >= 0 && cfg < NGCFG && M >= 1 && M <= GCFGS[cfg].r8, "config's row bucket does not cover M");
  const at::cuda::CUDAGuard g(x.device());
  GCFGS[cfg].fn(codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M, N, K, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
std::vector<std::string> fp8_gated_configs() {
  std::vector<std::string> v;
  for (int i = 0; i < NGCFG; i++) v.push_back(std::to_string(GCFGS[i].r8) + ":" + std::to_string(GCFGS[i].cols) + ":" + GCFGS[i].name);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_linear", &fp8_linear, "FP8 W8A16 linear on 1Cat's QPN8 layout, up to 64 rows in one pass");
  m.def("fp8_gated", &fp8_gated, "smallrows: FP8 gate_up with SiLU-and-mul on 1Cat's QPN8 layout");
  m.def("fp8_gated_configs", &fp8_gated_configs, "smallrows: gated configurations");
  m.def("fp8_configs", &fp8_configs, "kernel configurations: 'rows:columns per CTA:name'");
}
