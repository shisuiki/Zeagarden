// gdn_glue.cu - fused Gated DeltaNet decode glue for Qwen3.8-27B on V100 (sm_70), TP4 shard: 12 value heads x 128.
//
// gdn_in: the b/a projection (fp16 weight [2*heads, 5120], no bias) for M rows, written straight into contiguous
//   b [M, heads] and a [M, heads], plus the zero-fill of the recurrent core output [M, heads, 128]. It replaces
//   cuBLAS dot+reduce (M < 16) or cutlass+splitK (M >= 16), the inductor tail-copy combo and the zeros kernel.
//   Grid (2*heads/NPB, ceil(M/8)): 640 threads, NPB output columns and 8 rows per block; fp32 accumulation.
// gdn_gated_norm: RMSNormGated (norm_before_gate, silu gate, per-head group of 128, plain weight):
//   y = fp16(((x * rsqrt(mean(x^2) + eps)) * w) * (z * sigmoid(z))), x = core output, z read in place from the
//   qkvz projection (column offset zoff, row stride ldz). One warp per (row, head). It replaces the inductor
//   sum-of-squares and apply kernels and the z tail copy.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr int K = 5120, VEC = 8, NT_IN = K / VEC, D = 128;

__device__ __forceinline__ void h8(const uint4& u, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// One thread per 16-byte weight vector (640 threads cover K = 5120); every row's x load is independent, so a block
// keeps RPB x 10 KB in flight. NPB output columns per block: 1 for small M (more blocks), 3 above 8 rows (fewer L2
// re-reads of x).
template <int NPB, int RPB>
__global__ void __launch_bounds__(NT_IN) k_gdn_in(const half* __restrict__ x, const half* __restrict__ w,
                                                  half* __restrict__ b, half* __restrict__ a, half* __restrict__ core,
                                                  int M, int heads) {
  const int n0 = blockIdx.x * NPB, r0 = blockIdx.y * RPB, t = threadIdx.x, lane = t & 31, wp = t >> 5;
  const int rows = min(RPB, M - r0);
  {  // zero-fill this block's share of the row group's core output
    const int total = rows * heads * D / VEC;
    const int chunk = (total + gridDim.x - 1) / gridDim.x;
    uint4* c = reinterpret_cast<uint4*>(core + (size_t)r0 * heads * D);
    const int beg = blockIdx.x * chunk, end = min(total, beg + chunk);
    for (int i = beg + t; i < end; i += NT_IN) c[i] = make_uint4(0, 0, 0, 0);
  }
  uint4 xu[RPB];
#pragma unroll
  for (int r = 0; r < RPB; r++)
    xu[r] = r < rows ? reinterpret_cast<const uint4*>(x + (size_t)(r0 + r) * K)[t] : make_uint4(0, 0, 0, 0);
  float acc[NPB][RPB];
#pragma unroll
  for (int j = 0; j < NPB; j++) {
    float wf[VEC];
    h8(__ldg(reinterpret_cast<const uint4*>(w + (size_t)(n0 + j) * K) + t), wf);
#pragma unroll
    for (int r = 0; r < RPB; r++) {
      float xf[VEC];
      h8(xu[r], xf);
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < VEC; e++) s = fmaf(xf[e], wf[e], s);
      acc[j][r] = s;
    }
  }
  __shared__ float red[NT_IN / 32][NPB * RPB];
#pragma unroll
  for (int j = 0; j < NPB; j++)
#pragma unroll
    for (int r = 0; r < RPB; r++) {
      float v = acc[j][r];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[wp][j * RPB + r] = v;
    }
  __syncthreads();
  if (t < NPB * RPB) {
    const int j = t / RPB, r = t - j * RPB, n = n0 + j;
    if (r < rows) {
      float v = 0.f;
#pragma unroll
      for (int i = 0; i < NT_IN / 32; i++) v += red[i][t];
      const size_t row = r0 + r;
      if (n < heads)
        b[row * heads + n] = __float2half_rn(v);
      else
        a[row * heads + (n - heads)] = __float2half_rn(v);
    }
  }
}

constexpr int WPB = 8;  // warps per block in the gated norm
__global__ void __launch_bounds__(WPB * 32) k_gdn_gated_norm(const half* __restrict__ x, const half* __restrict__ z,
                                                             const half* __restrict__ w, half* __restrict__ y,
                                                             int M, int heads, long ldz, int zoff, float eps) {
  const int gw = blockIdx.x * WPB + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (gw >= M * heads) return;
  const int row = gw / heads, h = gw - row * heads;
  const uint2 xv = *reinterpret_cast<const uint2*>(x + (size_t)gw * D + lane * 4);
  const uint2 zv = *reinterpret_cast<const uint2*>(z + (size_t)row * ldz + zoff + h * D + lane * 4);
  const uint2 wv = __ldg(reinterpret_cast<const uint2*>(w + lane * 4));
  const __half2* xh = reinterpret_cast<const __half2*>(&xv);
  const __half2* zh = reinterpret_cast<const __half2*>(&zv);
  const __half2* wh = reinterpret_cast<const __half2*>(&wv);
  float xf[4], zf[4], wf[4];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    float2 t = __half22float2(xh[i]);
    xf[2 * i] = t.x;
    xf[2 * i + 1] = t.y;
    t = __half22float2(zh[i]);
    zf[2 * i] = t.x;
    zf[2 * i + 1] = t.y;
    t = __half22float2(wh[i]);
    wf[2 * i] = t.x;
    wf[2 * i + 1] = t.y;
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) ss = fmaf(xf[i], xf[i], ss);
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss * (1.f / D) + eps);
  __half2 o2[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    float v[2];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      const int k = 2 * i + e;
      const float g = zf[k] / (1.f + __expf(-zf[k]));
      v[e] = ((xf[k] * inv) * wf[k]) * g;
    }
    o2[i] = __floats2half2_rn(v[0], v[1]);
  }
  *reinterpret_cast<uint2*>(y + (size_t)gw * D + lane * 4) = *reinterpret_cast<const uint2*>(o2);
}
}  // namespace

void gdn_in(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor a, torch::Tensor core) {
  const int M = x.size(0), heads = b.size(1);
  TORCH_CHECK(x.is_contiguous() && x.size(1) == K && x.scalar_type() == at::kHalf, "x [M,5120] fp16 contiguous");
  TORCH_CHECK(w.is_contiguous() && w.size(0) == 2 * heads && w.size(1) == K && w.scalar_type() == at::kHalf, "w");
  TORCH_CHECK(b.is_contiguous() && a.is_contiguous() && core.is_contiguous() && core.size(0) == M &&
                  core.size(1) == heads && core.size(2) == D,
              "outputs");
  if (M == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  const half* wp = reinterpret_cast<const half*>(w.data_ptr());
  half *bp = reinterpret_cast<half*>(b.data_ptr()), *ap = reinterpret_cast<half*>(a.data_ptr()),
       *cp = reinterpret_cast<half*>(core.data_ptr());
  const int N = 2 * heads;
#define Q27_IN(NPB, RPB) \
  k_gdn_in<NPB, RPB><<<dim3(N / NPB, (M + RPB - 1) / RPB), NT_IN, 0, st>>>(xp, wp, bp, ap, cp, M, heads)
  if (M == 1)
    Q27_IN(1, 1);
  else if (M == 2)
    Q27_IN(1, 2);
  else if (M <= 4)
    Q27_IN(1, 4);
  else if (M <= 8 || N % 3 != 0)
    Q27_IN(1, 8);
  else
    Q27_IN(3, 8);
#undef Q27_IN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_gated_norm(torch::Tensor x, torch::Tensor z, torch::Tensor w, torch::Tensor y, double eps, int64_t zoff) {
  const int M = x.size(0), heads = x.size(1);
  TORCH_CHECK(x.is_contiguous() && x.size(2) == D && x.scalar_type() == at::kHalf, "x [M,heads,128] fp16");
  TORCH_CHECK(z.stride(1) == 1 && z.size(0) == M && z.size(1) >= zoff + heads * D && z.scalar_type() == at::kHalf &&
                  (z.stride(0) % 4) == 0 && (zoff % 4) == 0,
              "z");
  TORCH_CHECK(w.is_contiguous() && w.numel() == D && w.scalar_type() == at::kHalf, "w [128] fp16");
  TORCH_CHECK(y.is_contiguous() && y.numel() == (int64_t)M * heads * D, "y");
  if (M == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int warps = M * heads;
  k_gdn_gated_norm<<<(warps + WPB - 1) / WPB, WPB * 32, 0, st>>>(
      reinterpret_cast<const half*>(x.data_ptr()), reinterpret_cast<const half*>(z.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()), reinterpret_cast<half*>(y.data_ptr()), M, heads, z.stride(0),
      (int)zoff, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_in", &gdn_in);
  m.def("gdn_gated_norm", &gdn_gated_norm);
}
