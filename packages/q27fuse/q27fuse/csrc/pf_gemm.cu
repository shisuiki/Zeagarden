// pf_gemm.cu (track prefill2) - prefill GEMMs through cublasGemmEx with an explicit algorithm, for the FP8 projections'
// dense route (1Cat dequantises the QPN8 codes to fp16 W[K][N] = fp16(E4M3 * s16 / 256) and runs a cuBLAS GEMM that
// picks volta_fp16_s884gemm_fp16_64x128 / 256x128 "nn" kernels; mlp_fp4.cu's cublasGemmEx "tn" call on the NVFP4 MLP
// lands on cutlass_70_tensorop_f16_s884gemm_relu_f16_128x128).
//   gemm(W, x, out, layout, algo): out[M][N] = x[M][K] W, fp16 in/out, FP32 accumulation (CUBLAS_COMPUTE_32F);
//     layout 0: W row-major [K][N] ("nn" in cuBLAS' column-major view), 1: W row-major [N][K] ("tn");
//     algo -1 = CUBLAS_GEMM_DEFAULT_TENSOR_OP, 0..15 = CUBLAS_GEMM_ALGO<i>_TENSOR_OP.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

int64_t gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t layout, int64_t algo) {
  TORCH_CHECK(w.dtype() == torch::kHalf && x.dtype() == torch::kHalf && out.dtype() == torch::kHalf);
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.is_contiguous());
  const int M = x.size(0), K = x.size(1);
  const int N = layout == 0 ? w.size(1) : w.size(0);
  TORCH_CHECK((layout == 0 ? w.size(0) : w.size(1)) == K && out.size(0) == M && out.size(1) == N);
  const at::cuda::CUDAGuard guard(x.device());
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.f, beta = 0.f;
  const cublasGemmAlgo_t a = algo < 0 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0_TENSOR_OP + algo);
  const cublasOperation_t opw = layout == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int ldw = layout == 0 ? N : K;
  const cublasStatus_t s = cublasGemmEx(h, opw, CUBLAS_OP_N, N, M, K, &alpha, w.data_ptr(), CUDA_R_16F, ldw, x.data_ptr(),
                                        CUDA_R_16F, K, &beta, out.data_ptr(), CUDA_R_16F, N, CUBLAS_COMPUTE_32F, a);
  return (int64_t)s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm, "cublasGemmEx fp16 with an explicit algorithm"); }
