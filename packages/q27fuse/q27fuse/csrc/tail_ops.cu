// tail_ops.cu - small kernels for the MTP step outside the verify layers (track "tail", q27fuse/tail.py).
//
// row_argmax_pair: per row of the local LM-head logits [rows, V] (fp16), the (value, index) pair torch's
//   logits.max(dim=-1) returns (NaN wins; ties and repeated NaNs -> lowest index; +0 == -0), written as the fp32 pair
//   (value, index + vocab_start) that LogitsProcessor.get_top_tokens stacks for the cross-device top-1. One launch
//   replaces reduce_kernel + add + 2 casts + stack. Grid (S, rows): each CTA reduces V/S of a row to a 64-bit key
//   (orderable value << 32 | ~index) and the last CTA of the row (counter in a zeroed int32 workspace, reset by it)
//   reduces the S keys and re-reads the winning element, so the value is the element itself. The pair goes to `out`,
//   a raw device pointer (the custom all-reduce staging buffer, or a tensor).
// greedy_verify_fused: _greedy_verify_kernel + the index/fill/cast kernels in front of it + _get_num_sampled_and_
//   rejected_kernel in one launch (one warp per request; same arithmetic: sampled[r, i] = target[start + i] for every
//   logit of the request, zeros after; leading draft matches counted; chunked-prefill rows give 0 / 0).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ unsigned long long key_of(unsigned short h, unsigned idx) {
  unsigned o;
  if ((h & 0x7c00u) == 0x7c00u && (h & 0x03ffu)) {
    o = 0x10000u;                                  // NaN: above every number (torch: NaN is the max)
  } else {
    if (h == 0x8000u) h = 0;                         // -0 == +0
    o = (h & 0x8000u) ? (~h & 0xffffu) : (h | 0x8000u);
  }
  return ((unsigned long long)o << 32) | (unsigned long long)(0xffffffffu - idx);
}

__device__ __forceinline__ unsigned long long umax64(unsigned long long a, unsigned long long b) {
  return a > b ? a : b;
}

template <int THREADS>
__device__ __forceinline__ unsigned long long block_max(unsigned long long k, unsigned long long* sh) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) k = umax64(k, __shfl_xor_sync(0xffffffffu, k, o));
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) sh[w] = k;
  __syncthreads();
  if (w == 0) {
    k = l < THREADS / 32 ? sh[l] : 0ull;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) k = umax64(k, __shfl_xor_sync(0xffffffffu, k, o));
  }
  return k;                                        // valid in warp 0
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS) k_row_argmax_pair(const __half* __restrict__ logits, long ld, int V,
                                                              long vocab_start, float* __restrict__ out,
                                                              unsigned long long* __restrict__ partial,
                                                              int* __restrict__ counter) {
  __shared__ unsigned long long sh[THREADS / 32];
  __shared__ bool last;
  const int S = gridDim.x, s = blockIdx.x, row = blockIdx.y;
  const unsigned short* x = reinterpret_cast<const unsigned short*>(logits + (long)row * ld);
  const int nvec = V >> 3;                         // 8 halves per 16-byte vector (V % 8 == 0 checked on the host)
  const int per = (nvec + S - 1) / S;
  const int v0 = s * per, v1 = min(nvec, v0 + per);
  unsigned long long best = 0ull;
  for (int v = v0 + threadIdx.x; v < v1; v += THREADS) {
    const uint4 q = __ldcg(reinterpret_cast<const uint4*>(x) + v);
    const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      best = umax64(best, key_of((unsigned short)(w[j] & 0xffffu), (unsigned)(v * 8 + 2 * j)));
      best = umax64(best, key_of((unsigned short)(w[j] >> 16), (unsigned)(v * 8 + 2 * j + 1)));
    }
  }
  best = block_max<THREADS>(best, sh);
  if (threadIdx.x == 0) {
    partial[(long)row * S + s] = best;
    __threadfence();
    last = atomicAdd(counter + row, 1) == S - 1;
  }
  __syncthreads();
  if (!last) return;
  if (threadIdx.x < 32) {
    __threadfence();
    unsigned long long k = 0ull;
    for (int i = threadIdx.x; i < S; i += 32) k = umax64(k, __ldcg(partial + (long)row * S + i));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) k = umax64(k, __shfl_xor_sync(0xffffffffu, k, o));
    if (threadIdx.x == 0) {
      const unsigned idx = 0xffffffffu - (unsigned)(k & 0xffffffffull);
      const float val = __half2float(logits[(long)row * ld + idx]);
      out[2 * row] = val;
      out[2 * row + 1] = (float)(vocab_start + (long)idx);
      counter[row] = 0;                            // ready for the next launch
    }
  }
}

__global__ void k_greedy_verify_fused(const long* __restrict__ target, const int* __restrict__ input_ids,
                                      const long* __restrict__ logits_indices, const int* __restrict__ cu,
                                      const int* __restrict__ seq_lens, const int* __restrict__ prefill_len,
                                      const int* __restrict__ idx_mapping, long* __restrict__ sampled, int cols,
                                      int* __restrict__ num_sampled, int* __restrict__ num_rejected) {
  const int r = blockIdx.x, l = threadIdx.x;
  const int start = cu[r], n = cu[r + 1] - start;
  // sampled row: target ids for the request's logits, zeros after (new_zeros in greedy_verify)
  for (int i = l; i < cols; i += 32) sampled[(long)r * cols + i] = i < n ? target[start + i] : 0;
  // leading matches: target[start + i] == draft at logit i + 1 (input_ids[logits_indices[start + i + 1]])
  bool ok = true;
  if (l < n - 1) ok = target[start + l] == (long)input_ids[logits_indices[start + l + 1]];
  unsigned bad = __ballot_sync(0xffffffffu, !ok && l < n - 1);
  int count = bad ? __ffs(bad) - 1 : max(n - 1, 0);
  if (n - 1 > 32) count = -1;                      // not reached: rows <= K + 1 <= 9 (checked on the host)
  if (l == 0) {
    const int req = idx_mapping[r];
    const bool chunked = seq_lens[r] < prefill_len[req];
    const int ns = chunked ? 0 : count + 1;
    num_sampled[r] = ns;
    num_rejected[r] = chunked ? 0 : n - ns;
  }
}

}  // namespace

void row_argmax_pair(torch::Tensor logits, int64_t vocab_start, int64_t out_ptr, torch::Tensor partial,
                     torch::Tensor counter, int64_t splits) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kHalf && logits.dim() == 2, "logits: cuda fp16 2-D");
  TORCH_CHECK(logits.stride(1) == 1 && logits.size(1) % 8 == 0 && logits.stride(0) % 8 == 0, "logits layout");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0, "logits alignment");
  const int rows = logits.size(0), V = logits.size(1), S = (int)splits;
  TORCH_CHECK(rows >= 1 && S >= 1 && S <= 64 && partial.numel() >= (long)rows * S && counter.numel() >= rows, "ws");
  constexpr int T = 256;
  k_row_argmax_pair<T><<<dim3(S, rows), T, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(logits.data_ptr()), logits.stride(0), V, vocab_start,
      reinterpret_cast<float*>(out_ptr), reinterpret_cast<unsigned long long*>(partial.data_ptr()),
      counter.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void greedy_verify_fused(torch::Tensor target, torch::Tensor input_ids, torch::Tensor logits_indices,
                         torch::Tensor cu, torch::Tensor seq_lens, torch::Tensor prefill_len,
                         torch::Tensor idx_mapping, torch::Tensor sampled, torch::Tensor num_sampled,
                         torch::Tensor num_rejected) {
  const int reqs = cu.numel() - 1;
  TORCH_CHECK(target.scalar_type() == at::kLong && input_ids.scalar_type() == at::kInt &&
              logits_indices.scalar_type() == at::kLong && cu.scalar_type() == at::kInt &&
              seq_lens.scalar_type() == at::kInt && prefill_len.scalar_type() == at::kInt &&
              idx_mapping.scalar_type() == at::kInt && sampled.scalar_type() == at::kLong &&
              num_sampled.scalar_type() == at::kInt && num_rejected.scalar_type() == at::kInt, "dtypes");
  TORCH_CHECK(sampled.dim() == 2 && sampled.size(0) == reqs && sampled.is_contiguous() && sampled.size(1) <= 32,
              "sampled");
  if (reqs <= 0) return;
  k_greedy_verify_fused<<<reqs, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      target.data_ptr<long>(), input_ids.data_ptr<int>(), logits_indices.data_ptr<long>(), cu.data_ptr<int>(),
      seq_lens.data_ptr<int>(), prefill_len.data_ptr<int>(), idx_mapping.data_ptr<int>(), sampled.data_ptr<long>(),
      (int)sampled.size(1), num_sampled.data_ptr<int>(), num_rejected.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor from_ptr(int64_t ptr, int64_t numel, torch::Tensor like) {
  // a float32 tensor view of raw device memory (the custom all-reduce staging buffer); not owned
  return torch::from_blob(reinterpret_cast<void*>(ptr), {numel}, [](void*) {},
                          torch::TensorOptions().dtype(at::kFloat).device(like.device()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_argmax_pair", &row_argmax_pair);
  m.def("greedy_verify_fused", &greedy_verify_fused);
  m.def("from_ptr", &from_ptr);
}
