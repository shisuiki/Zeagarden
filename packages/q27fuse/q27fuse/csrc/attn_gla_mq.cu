// attn_gla_mq.cu - Custom Volta sm_70 absorbed multi-query GLA-g4 FP8-KV attention extension for Qwen3.8-27B
// (4x V100-SXM2-16GB, 1Cat-vLLM, CUDA 12.9).
//
// Packed KV cache layout (R_LAT in {192, 256}, ROPE_DIM = 62, ROW_B = R_LAT + 64, HALF_B = ROW_B / 2):
//   R_LAT = 192: ROW_B = 256, HALF_B = 128
//     k_cache [num_blocks, page_size, Hkv, 128] uint8: bytes 0..127   = c[0..127]       (E4M3 / kv_scale)
//     v_cache [num_blocks, page_size, Hkv, 128] uint8: bytes 0..63    = c[128..191]     (E4M3 / kv_scale)
//                                                      bytes 64..125  = k_rope[0..61]   (E4M3 / kv_scale)
//                                                      bytes 126..127 = inv_rms         (__half LE, unscaled)
//   R_LAT = 256: ROW_B = 320, HALF_B = 160
//     k_cache [num_blocks, page_size, Hkv, 160] uint8: bytes 0..159   = c[0..159]       (E4M3 / kv_scale)
//     v_cache [num_blocks, page_size, Hkv, 160] uint8: bytes 0..95    = c[160..255]     (E4M3 / kv_scale)
//                                                      bytes 96..157  = k_rope[0..61]   (E4M3 / kv_scale)
//                                                      bytes 158..159 = inv_rms         (__half LE, unscaled)
//
// Ops exported:
//   1. gla_cache_write(c, k_rope, inv_rms, k_cache, v_cache, slot_mapping, kv_scale)
//   2. gla_gather_dequant(k_cache, v_cache, block_table, seq_lens, cu_seq_lens, out_c, out_k_rope, out_inv_rms, kv_scale)
//   3. attn_gla_mq(q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, scale, kv_scale, Q, S, min_tiles, sep)
//   4. gla_splits(bh, S, R_LAT, Q, sep), gla_workspace(bh, S, R_LAT, Q, sep), gla_smem_and_occ(R_LAT, Q, dev_idx), init(dev_idx)
//   5. gla_prep_route_a(qckr, positions, q_norm_w1p, kw_rope_perm, w_k_up_raw_cm, w_k_up_sc_cm, cos_sin_cache, slot_mapping, k_cache, v_cache, q_cat, eps, kv_scale)
//   6. gla_prep_prefill(qckr, positions, q_norm_w1p, kw_rope_perm, w_k_up_raw_cm, cos_sin_cache, slot_mapping, k_cache, v_cache, q_full, k_rot_out, inv_rms_out, eps, kv_scale)
//   7. gla_vup_gate(o_lat, w_v_up_cm, qckr, out)
//   8. gla_gate_only(attn_tmp, qckr, out)
//
// Core kernel design (k_gla<R_LAT, RG>, NW = 4 warps = 128 threads/CTA):
//   - Single-read shared-latent staging: each 32-token tile of k_cache + v_cache is loaded from HBM ONCE via
//     coalesced uint4 loads into prefetch registers (overlapping tile t+1 HBM loads with tile t QK+Softmax+PV) and
//     unpacked into SMEM Tbuf[32][TS_W] (TS_W = ROW_B/4 + 7: 0-way QK bank conflicts, <=2-way on 3/32 banks in PV).
//   - inv_rms (__half) is extracted directly from the last v_cache uint4 in register into SMEM InvR[32], and bytes
//     ROW_B-2..ROW_B-1 are zero-masked in BOTH Tbuf and Qs so garbage/NaN in q_cat[..., -2:] can never leak.
//   - QK (mma.sync.aligned.m8n8k4.row.col): all 4 warps split ROW_B evenly (64 dims/warp for 192, 80 for 256).
//     Each warp keeps its owned row groups' partial scores in registers and exchanges only the 3 non-owner warps via
//     3-slot SMEM Xs (saving 25% SMEM so R_LAT=256 RG=4 [Q=5 MTP4] fits in 46.6 KiB <= 48 KiB -> 2 CTAs/SM!).
//   - Online FP32 softmax multiplies summed scores by InvR[tok] * sl2, masks col >= seq_len, writes FP16 P & FP32 alpha.
//   - PV (mma.sync.aligned.m8n8k4.row.row): reads c directly from SMEM Tbuf (ZERO second HBM read!).
//     All 4 warps split R_LAT evenly (48 dims/warp for 192, 64 for 256):
//       - R_LAT=256: 2 m8n8 blocks/quadpair x 8 K-steps.
//       - R_LAT=192: j=0 (32 cols/warp) x 8 K-steps + j=1 (16 cols/warp) split-K across quadpairs (qp 0,1 tokens 0..15,
//         qp 2,3 tokens 16..31 -> 4 K-steps, exact 0.75x PV compute, single __shfl_xor_sync(..., 8) after the loop!).
//   - Split-KV: sep=1 runs deterministic k_comb<R_LAT, 256>; sep=0 runs fused single-wave barrier + combine_slice.
//   - CUDA-graph safe: all cudaFuncSetAttribute & occupancy queries are cached once per device via std::call_once.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <tuple>

namespace {

constexpr int NQ = 6;
constexpr int HD = 256;
constexpr int TILE = 32;
constexpr int ROPE_DIM = 62;
constexpr int NOPE_DIM = HD - ROPE_DIM;   // 194
constexpr int NW = 4;
constexpr int NT_SET = 32 * NW;   // 128 threads per warp-set; CTA has NT_SET * NG threads (128 or 256)
constexpr int SMAX = 128;
constexpr int QMAX = 8;
constexpr int TBS = 64;
constexpr int MAX_DEVS = 32;
constexpr double LOG2E = 1.44269504088896340736;

// ---- Volta sm_70 HMMA.884 intrinsics (exact lane/quadpair layout inherited from attn_mq_ref.cu) ----
__device__ __forceinline__ void mma_rc(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_rr(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned ldg4(const void* p) {
  unsigned v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 ldg16_safe(const uint8_t* p, bool aligned16) {
  if (aligned16) return ldg16(p);
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint8_t* q = p + 4 * i;
    w[i] = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ uint32_t ldg4_safe(const uint8_t* p, bool aligned4) {
  if (aligned4) return ldg4(p);
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ void st4_safe(uint8_t* p, uint32_t v, bool aligned4) {
  if (aligned4) {
    *reinterpret_cast<uint32_t*>(p) = v;
  } else {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
  }
}

__device__ __forceinline__ void st_h2_safe(__half* p, __half2 v, bool aligned4) {
  if (aligned4) {
    *reinterpret_cast<__half2*>(p) = v;
  } else {
    p[0] = __low2half(v);
    p[1] = __high2half(v);
  }
}

// Four E4M3FN bytes -> two fp16x2 (bytes 0,1 and 2,3), each = exact E4M3_val * 2^-8.
// Finite E4M3FN contract (matching attn_mq_ref.cu): assumes finite E4M3 codes 0x00..0x7E and 0x80..0xFE
// (0x7F/0xFF NaN codes decode to +-1.875 * 2^8 rather than FP16 NaN). Finite inputs written by
// gla_cache_write are clamped to [-448.0, 448.0] and never emit 0x7F/0xFF.
__device__ __forceinline__ void e4m3x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned x0 = __byte_perm(w, 0u, 0x1404), x1 = __byte_perm(w, 0u, 0x3424);
  lo = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
  hi = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
}

// Explicit Volta sm_70 (PTX 6.0+) GPU-scope acquire/release primitives for sep == 0 split barrier
__device__ __forceinline__ int atom_add_acq_rel_gpu(int* p, int v) {
  int out;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(out) : "l"(p), "r"(v) : "memory");
  return out;
}

__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int out;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(out) : "l"(p) : "memory");
  return out;
}

__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Exact IEEE-754 round-to-nearest-even FP32 -> E4M3FN matching
// (x.float() * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn) bit-for-bit
// across normals, subnormals, ties-to-even, +-0, saturation at +-448, and NaN.
__device__ __forceinline__ uint8_t f32_to_e4m3fn(float x) {
  const unsigned u = __float_as_uint(x);
  const unsigned sign = (u >> 24) & 0x80u;
  const unsigned abs_u = u & 0x7FFFFFFFu;
  if (abs_u > 0x7F800000u) return (uint8_t)(sign | 0x7Fu);    // NaN
  if (abs_u >= 0x43E00000u) return (uint8_t)(sign | 0x7Eu);   // |x| >= 448.0 -> clamp to +-448.0
  if (abs_u >= 0x3C800000u) {                                 // normal: |x| >= 2^-6
    const unsigned rounded = abs_u + 0x0007FFFFu + ((abs_u >> 20) & 1u);
    return (uint8_t)(sign | ((rounded - 0x3C000000u) >> 20));
  }
  // subnormal or zero: |x| < 2^-6; adding 2^14 (ULP = 2^-9) with IEEE RNE rounds exactly to nearest multiple of 2^-9
  const float t = __fadd_rn(__uint_as_float(abs_u), 16384.0f);
  return (uint8_t)(sign | (__float_as_uint(t) - 0x46800000u));
}

__device__ __forceinline__ uint32_t pack4_e4m3_h(const __half* p, float inv_scale) {
  const uint8_t b0 = f32_to_e4m3fn(__fmul_rn(__half2float(p[0]), inv_scale));
  const uint8_t b1 = f32_to_e4m3fn(__fmul_rn(__half2float(p[1]), inv_scale));
  const uint8_t b2 = f32_to_e4m3fn(__fmul_rn(__half2float(p[2]), inv_scale));
  const uint8_t b3 = f32_to_e4m3fn(__fmul_rn(__half2float(p[3]), inv_scale));
  return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
}

__device__ __forceinline__ void unpack4_e4m3_to_h2(uint32_t w, float vsc, __half2& h0, __half2& h1) {
  unsigned lo, hi;
  e4m3x4(w, lo, hi);
  const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  h0 = __floats2half2_rn(__fmul_rn(f0.x, vsc), __fmul_rn(f0.y, vsc));
  h1 = __floats2half2_rn(__fmul_rn(f1.x, vsc), __fmul_rn(f1.y, vsc));
}

__device__ __forceinline__ float ex2(float x) { return exp2f(x); }

// =================================================================================================
// Op 1: gla_cache_write
// =================================================================================================
template <int R_LAT, bool SLOT64, bool INV_FP16>
__global__ void __launch_bounds__(128)
k_gla_cache_write(const __half* __restrict__ c_ptr, const __half* __restrict__ kr_ptr, const void* __restrict__ inv_ptr,
                  uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, const void* __restrict__ slot_ptr,
                  int64_t N, int Hkv, int page_size, int64_t max_slots,
                  int64_t kb, int64_t kt, int64_t kh, int64_t vb, int64_t vt, int64_t vh,
                  float inv_scale, bool k_aligned4, bool v_aligned4) {
  constexpr int ROW_B = R_LAT + 64;
  constexpr int HALF_B = ROW_B / 2;
  constexpr int HALF_W = HALF_B / 4;   // 32 for 192, 40 for 256

  const int64_t warp_id = (int64_t)blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp_id >= N * (int64_t)Hkv) return;

  const int64_t n = warp_id / Hkv;
  const int h = (int)(warp_id % Hkv);
  const int64_t slot = SLOT64 ? reinterpret_cast<const int64_t*>(slot_ptr)[n]
                              : (int64_t)reinterpret_cast<const int32_t*>(slot_ptr)[n];
  if (slot < 0 || slot >= max_slots) return;

  const int64_t blk = slot / page_size;
  const int64_t off = slot % page_size;

  const __half* c_row = c_ptr + warp_id * R_LAT;
  const __half* kr_row = kr_ptr + warp_id * ROPE_DIM;
  uint8_t* k_dst = kc + blk * kb + off * kt + (int64_t)h * kh;
  uint8_t* v_dst = vc + blk * vb + off * vt + (int64_t)h * vh;

  // k_cache: c[0 .. HALF_B - 1]
  for (int w = lane; w < HALF_W; w += 32) {
    const uint32_t word = pack4_e4m3_h(c_row + 4 * w, inv_scale);
    st4_safe(k_dst + 4 * w, word, k_aligned4);
  }

  // v_cache: [c[HALF_B .. R_LAT - 1] | k_rope[0 .. 61] | __half(inv_rms)]
  for (int w = lane; w < HALF_W; w += 32) {
    const int pos = HALF_B + 4 * w;
    uint32_t word;
    if (pos < R_LAT) {
      word = pack4_e4m3_h(c_row + pos, inv_scale);
    } else if (w < HALF_W - 1) {
      word = pack4_e4m3_h(kr_row + (pos - R_LAT), inv_scale);
    } else {
      const uint8_t b0 = f32_to_e4m3fn(__fmul_rn(__half2float(kr_row[60]), inv_scale));
      const uint8_t b1 = f32_to_e4m3fn(__fmul_rn(__half2float(kr_row[61]), inv_scale));
      uint16_t inv_u16;
      if constexpr (INV_FP16) {
        inv_u16 = reinterpret_cast<const uint16_t*>(inv_ptr)[warp_id];
      } else {
        const float ir = reinterpret_cast<const float*>(inv_ptr)[warp_id];
        inv_u16 = __half_as_ushort(__float2half_rn(ir));
      }
      word = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)inv_u16 << 16);
    }
    st4_safe(v_dst + 4 * w, word, v_aligned4);
  }
}

// =================================================================================================
// Op 2: gla_gather_dequant
// =================================================================================================
template <int R_LAT>
__global__ void __launch_bounds__(128)
k_gla_gather_dequant(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                     const int* __restrict__ block_table, int tb_stride, int max_blocks,
                     const int* __restrict__ seq_lens, const int* __restrict__ cu_seq_lens,
                     __half* __restrict__ out_c, __half* __restrict__ out_kr, float* __restrict__ out_inv,
                     int B, int64_t total_kv, int Hkv, int page_size, int num_blocks,
                     int64_t kb, int64_t kt, int64_t kh, int64_t vb, int64_t vt, int64_t vh,
                     float vsc, bool k_aligned4, bool v_aligned4, bool oc_aligned4, bool okr_aligned4) {
  constexpr int ROW_B = R_LAT + 64;
  constexpr int HALF_B = ROW_B / 2;
  constexpr int HALF_W = HALF_B / 4;

  const int64_t warp_id = (int64_t)blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp_id >= total_kv * (int64_t)Hkv) return;

  const int64_t gtok = warp_id / Hkv;
  const int h = (int)(warp_id % Hkv);
  if (gtok >= (int64_t)cu_seq_lens[B]) return;

  // Upper-bound binary search: find b in [0, B-1] with cu_seq_lens[b] <= gtok < cu_seq_lens[b + 1]
  int lo = 0, hi = B;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (cu_seq_lens[mid] <= (int)gtok) lo = mid;
    else hi = mid;
  }
  const int b = lo;
  const int tok = (int)gtok - cu_seq_lens[b];
  if (tok < 0 || tok >= seq_lens[b]) return;

  const int blk_idx = tok / page_size;
  if (blk_idx >= max_blocks) return;
  const int blk = block_table[(int64_t)b * tb_stride + blk_idx];
  if (blk < 0 || blk >= num_blocks) return;
  const int off = tok % page_size;

  const uint8_t* k_src = kc + (int64_t)blk * kb + (int64_t)off * kt + (int64_t)h * kh;
  const uint8_t* v_src = vc + (int64_t)blk * vb + (int64_t)off * vt + (int64_t)h * vh;

  __half* c_dst = out_c + warp_id * R_LAT;
  __half* kr_dst = out_kr + warp_id * ROPE_DIM;

  // k_cache -> out_c[0 .. HALF_B - 1]
  for (int w = lane; w < HALF_W; w += 32) {
    const uint32_t word = ldg4_safe(k_src + 4 * w, k_aligned4);
    __half2 h0, h1;
    unpack4_e4m3_to_h2(word, vsc, h0, h1);
    st_h2_safe(c_dst + 4 * w, h0, oc_aligned4);
    st_h2_safe(c_dst + 4 * w + 2, h1, oc_aligned4);
  }

  // v_cache -> out_c[HALF_B .. R_LAT - 1], out_k_rope[0 .. 61], out_inv_rms
  for (int w = lane; w < HALF_W; w += 32) {
    const uint32_t word = ldg4_safe(v_src + 4 * w, v_aligned4);
    const int pos = HALF_B + 4 * w;
    if (pos < R_LAT) {
      __half2 h0, h1;
      unpack4_e4m3_to_h2(word, vsc, h0, h1);
      st_h2_safe(c_dst + pos, h0, oc_aligned4);
      st_h2_safe(c_dst + pos + 2, h1, oc_aligned4);
    } else if (w < HALF_W - 1) {
      __half2 h0, h1;
      unpack4_e4m3_to_h2(word, vsc, h0, h1);
      const int kro = pos - R_LAT;
      st_h2_safe(kr_dst + kro, h0, okr_aligned4);
      st_h2_safe(kr_dst + kro + 2, h1, okr_aligned4);
    } else {
      __half2 h0, h1;
      unpack4_e4m3_to_h2(word & 0x0000FFFFu, vsc, h0, h1);
      st_h2_safe(kr_dst + 60, h0, okr_aligned4);
      const uint16_t raw_h = (uint16_t)(word >> 16);
      out_inv[warp_id] = __half2float(__ushort_as_half(raw_h));
    }
  }
}

// =================================================================================================
// Op 3: attn_gla_mq (Single-Read Shared-Latent Volta sm_70 HMMA.884 Split-KV Kernel + Combine)
// =================================================================================================
template <int NG>
__device__ __forceinline__ void bar_set(int sg) {
  if constexpr (NG == 1) {
    __syncthreads();
  } else {
    asm volatile("bar.sync %0, 128;" ::"r"(1 + sg) : "memory");
  }
}

template <int R_LAT, int RG_SET, int NG>
struct GLA {
  static_assert(R_LAT == 192 || R_LAT == 256, "R_LAT must be 192 or 256");
  static_assert(RG_SET >= 1 && RG_SET <= 3, "RG_SET must be in 1..3");
  static_assert(NG == 1 || NG == 2, "NG must be 1 or 2");

  static constexpr int NT = NT_SET * NG;               // 128 (NG=1) or 256 (NG=2)
  static constexpr int RG_CTA = RG_SET * NG;           // total row groups in CTA (1, 2, 4, 6)
  static constexpr int ROW_B = R_LAT + 64;             // 256 (R=192) or 320 (R=256)
  static constexpr int HALF_B = ROW_B / 2;             // 128 or 160
  static constexpr int NV_HALF = HALF_B / 16;          // 8 or 10 uint4s per token half
  // uint4s per thread per tile: NG=1 -> 4 (192) / 5 (256); NG=2 -> 2 (192) / 3 (256)
  static constexpr int NVEC = ((TILE * ROW_B) / 16 + NT - 1) / NT;
  static constexpr int TS_W = ROW_B / 4 + 7;           // 71 (mod 32 = 7) or 87 (mod 32 = 23 = -9): 0-way QK, ~0-way PV
  static constexpr int TS_B = TS_W * 4;                // 284 or 348 bytes per tile row
  static constexpr int QS = ROW_B + 4;                 // 260 or 324 halves (8-byte aligned, 0-way QK bank conflicts)
  static constexpr int PS = TILE + 8;                  // 40 halves (0-way PV bank conflicts)
  static constexpr int R_SET = 8 * RG_SET;
  static constexpr int R = 8 * RG_CTA;
  static constexpr int RGW = (RG_SET + NW - 1) / NW;   // always 1 for RG_SET <= 3
  static constexpr int DK_B = ROW_B / NW;              // 64 or 80 QK bytes/dims per warp
  static constexpr int DK_W = DK_B / 4;                // 16 or 20 QK u32 steps per warp
  static constexpr int DV = R_LAT / NW;                // 48 or 64 PV output dims per warp

  // Dynamic SMEM layout (all offsets >= 16-byte aligned; Tbuf/InvR/Ts shared by all NG warp-sets):
  //   Qs   [R][QS]                   fp16  (0-way QK reads)
  //   Tbuf [32][TS_W]                u32   (single shared-latent tile; aliased as ML [R][2] fp32 after tile loop)
  //   InvR [32]                      fp32  (per-token inv_rms extracted from Tbuf tail)
  //   Xs   [RG_CTA][NW - 1][32][8]   fp32  (3-slot QK partial exchange; owners keep own_sc in registers!)
  //   Ps   [R][PS]                   fp16  (softmax probabilities)
  //   As   [R]                       fp32  (online softmax rescale alphas)
  //   Ts   [TBS]                     int32 (first 64 block-table entries cached in SMEM)
  static constexpr int Q_ = 0;
  static constexpr int QB = R * QS * 2;
  static constexpr int T_ = Q_ + QB;
  static constexpr int TB = TILE * TS_B;
  static constexpr int IR_ = T_ + TB;
  static constexpr int IRB = TILE * 4;
  static constexpr int X_ = IR_ + IRB;
  static constexpr int XB = RG_CTA * (NW - 1) * 32 * 8 * 4;
  static constexpr int P_ = X_ + XB;
  static constexpr int PB = R * PS * 2;
  static constexpr int A_ = P_ + PB;
  static constexpr int AB = R * 4;
  static constexpr int B_ = A_ + AB;
  static constexpr int BB = TBS * 4;
  static constexpr int MAIN = B_ + BB;

  static_assert(TB >= R * 2 * 4, "Tbuf must be large enough to alias ML after the tile loop");
  static_assert(MAIN <= 96 * 1024, "MAIN SMEM exceeds Volta 96 KiB limit");
};

template <int R_LAT>
__host__ __device__ inline int comb_emax(int Sb, int nr) {
  constexpr int D4 = R_LAT / 4;
  return (nr * D4 + Sb - 1) / Sb;
}

template <int R_LAT>
__host__ __device__ inline int comb_rows(int Sb, int nr) {
  constexpr int D4 = R_LAT / 4;
  return (comb_emax<R_LAT>(Sb, nr) + D4 - 1) / D4 + 1;
}

template <int R_LAT>
__host__ __device__ inline int comb_floats(int Sb, int nr) {
  return 4 * comb_emax<R_LAT>(Sb, nr) * ((Sb + 7) / 8) + (2 * Sb + 2) * comb_rows<R_LAT>(Sb, nr);
}

// Deterministic split-KV reduction slice (exact same algorithm as attn_mq_ref.cu, generalized to R_LAT)
template <int R_LAT, int CT>
__device__ __forceinline__ void combine_slice(uint8_t* sm, const float* __restrict__ po, const float* __restrict__ pml,
                                              __half* __restrict__ out, int bh, int SG, int NR, int Sb, int s, int b,
                                              int h, int Hkv, int Q, float vsc, bool out_aligned4) {
  constexpr int D4 = R_LAT / 4;
  const int tid = threadIdx.x;
  auto orow = [&](int r) -> size_t { return (((size_t)b * Q + r / NQ) * Hkv + h) * NQ + r % NQ; };

  const int n4 = NR * D4;
  const int lo = (int)((long long)n4 * s / Sb);
  const int E = (int)((long long)n4 * (s + 1) / Sb) - lo;
  const int G = (Sb + 7) / 8;
  const int rlo = lo / D4;
  const int nrs = (lo + E - 1) / D4 - rlo + 1;

  float* Acc = reinterpret_cast<float*>(sm);
  float* Mk = Acc + 4 * comb_emax<R_LAT>(Sb, NR) * G;
  float* Lk = Mk + Sb * nrs;
  float* Rs = Lk + Sb * nrs;

  // Issue first item's partial loads together with split statistics loads
  const int EG = E * G;
  float4 c[8];
  {
    const int idx = tid;
    if (idx < EG) {
      const int e = idx % E, g = idx / E, el = lo + e, row = el / D4, dd = (el % D4) * 4;
      const int k0 = 8 * g, nk = min(8, Sb - k0);
      const float* src = po + ((size_t)bh * SG * NR + row) * R_LAT + dd;
#pragma unroll
      for (int u = 0; u < 8; u++)
        if (u < nk) c[u] = __ldcg(reinterpret_cast<const float4*>(src + (size_t)(k0 + u) * NR * R_LAT));
    }
  }
  const float* pm = pml + (size_t)bh * SG * NR * 2;
#pragma unroll 4
  for (int i = tid; i < Sb * nrs; i += CT) {
    const int k = i / nrs, r = rlo + i % nrs;
    Mk[i] = __ldcg(pm + 2 * (k * NR + r));
    Lk[i] = __ldcg(pm + 2 * (k * NR + r) + 1);
  }
  __syncthreads();

  for (int rb = 0; rb < nrs; rb += CT / 8) {
    const int r = rb + (tid >> 3), j = tid & 7;
    float M = -INFINITY;
    if (r < nrs) {
      for (int k = j; k < Sb; k += 8) M = fmaxf(M, Mk[k * nrs + r]);
    }
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, d));
    const float Mz = (M == -INFINITY) ? 0.f : M;
    float Lt = 0.f;
    if (r < nrs) {
      for (int k = j; k < Sb; k += 8) Lt += Lk[k * nrs + r] * ex2(Mk[k * nrs + r] - Mz);
    }
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) Lt += __shfl_xor_sync(0xffffffffu, Lt, d);
    if (j == 0 && r < nrs) {
      Rs[r] = Mz;
      Rs[nrs + r] = (Lt > 0.f) ? (vsc / Lt) : 0.f;
    }
  }
  __syncthreads();

  for (int idx = tid; idx < EG; idx += CT) {
    const int e = idx % E, g = idx / E, el = lo + e, row = el / D4;
    const int k0 = 8 * g, nk = min(8, Sb - k0);
    if (idx != tid) {
      const float* src = po + ((size_t)bh * SG * NR + row) * R_LAT + (el % D4) * 4;
#pragma unroll
      for (int u = 0; u < 8; u++)
        if (u < nk) c[u] = __ldcg(reinterpret_cast<const float4*>(src + (size_t)(k0 + u) * NR * R_LAT));
    }
    const float M = Rs[row - rlo], inv = Rs[nrs + row - rlo];
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if (u < nk) {
        const float wk = ex2(Mk[(k0 + u) * nrs + row - rlo] - M) * inv;
        a.x += wk * c[u].x;
        a.y += wk * c[u].y;
        a.z += wk * c[u].z;
        a.w += wk * c[u].w;
      }
    }
    *reinterpret_cast<float4*>(Acc + 4 * (g * E + e)) = a;
  }
  __syncthreads();

  for (int i = tid; i < E; i += CT) {
    float4 a = *reinterpret_cast<const float4*>(Acc + 4 * i);
    for (int g = 1; g < G; g++) {
      const float4 u = *reinterpret_cast<const float4*>(Acc + 4 * (g * E + i));
      a.x += u.x;
      a.y += u.y;
      a.z += u.z;
      a.w += u.w;
    }
    const int eli = lo + i, rw = eli / D4, d = (eli % D4) * 4;
    __half* dst = out + orow(rw) * R_LAT + d;
    st_h2_safe(dst, __floats2half2_rn(a.x, a.y), out_aligned4);
    st_h2_safe(dst + 2, __floats2half2_rn(a.z, a.w), out_aligned4);
  }
}

#ifdef Q27_MINB
#define Q27_GLA_LB(nt) __launch_bounds__(nt, Q27_MINB)
#else
#define Q27_GLA_LB(nt) __launch_bounds__(nt)
#endif

template <int R_LAT, int RG_SET, int NG, bool FUSED_COMB>
__global__ void Q27_GLA_LB(NT_SET * NG)
k_gla(const __half* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
      const int* __restrict__ table, int tstride, int tw, const int* __restrict__ lens, __half* __restrict__ out,
      float* __restrict__ po, float* __restrict__ pml, int* __restrict__ cnt, int Q, int S, int min_tiles, int page,
      int num_blocks, int64_t kb, int64_t kt, int64_t kh, int64_t vb, int64_t vt, int64_t vh, float sl2, float vsc,
      bool fast_page, bool kv_aligned16, bool q_aligned16, bool out_aligned4) {
  using C = GLA<R_LAT, RG_SET, NG>;
  constexpr int NT = C::NT;
  constexpr int ROW_B = C::ROW_B;
  constexpr int HALF_B = C::HALF_B;
  constexpr int NV_HALF = C::NV_HALF;
  constexpr int NVEC = C::NVEC;
  constexpr int TS_W = C::TS_W;
  constexpr int QS = C::QS;
  constexpr int PS = C::PS;
  constexpr int RGW = C::RGW;
  constexpr int DK_B = C::DK_B;
  constexpr int DK_W = C::DK_W;
  constexpr int DV = C::DV;

  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int sg = (NG == 1) ? 0 : (tid >> 7);
  const int wi = (tid >> 5) & 3;
  const int roff = sg * C::R_SET;

  __half* Qs = reinterpret_cast<__half*>(sm + C::Q_);
  uint32_t* Tbuf = reinterpret_cast<uint32_t*>(sm + C::T_);
  float* InvR = reinterpret_cast<float*>(sm + C::IR_);
  float* Xs = reinterpret_cast<float*>(sm + C::X_) + sg * RG_SET * (NW - 1) * 32 * 8;
  __half* Ps = reinterpret_cast<__half*>(sm + C::P_) + roff * PS;
  float* As = reinterpret_cast<float*>(sm + C::A_) + roff;
  int* Ts = reinterpret_cast<int*>(sm + C::B_);

  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z, Hkv = gridDim.y;
  const int bh = b * Hkv + h, NR = NQ * Q;
  const int* lr = lens + (size_t)b * Q;
  auto orow = [&](int r) -> size_t { return (((size_t)b * Q + r / NQ) * Hkv + h) * NQ + r % NQ; };

  // Compute max sequence length across the Q query rows of request b
  int Lmax = 0;
#pragma unroll
  for (int i = 0; i < QMAX; i++) {
    if (i < Q) Lmax = max(Lmax, lr[i]);
  }
  if (Lmax <= 0) {
    if (s == 0) {
      const __half2 z = __floats2half2_rn(0.f, 0.f);
      for (int i = tid; i < NR * (R_LAT / 2); i += NT) {
        const int row = i / (R_LAT / 2), d2 = (i % (R_LAT / 2)) * 2;
        st_h2_safe(out + orow(row) * R_LAT + d2, z, out_aligned4);
      }
    }
    return;
  }

  const int T = (Lmax + TILE - 1) / TILE;
  const int Sb = max(1, min(S, T / min_tiles));
  if (s >= Sb) return;
  const int t0 = (int)((long long)T * s / Sb), t1 = (int)((long long)T * (s + 1) / Sb);

  // Cache the request's first TBS block-table entries in SMEM
  const int* tb = table + (size_t)b * tstride;
  if (tid < TBS) Ts[tid] = (tid < tw) ? tb[tid] : -1;
  __syncthreads();

  // ---- Tile loader: cooperative coalesced uint4 HBM reads -> prefetch registers rbuf[NVEC] ----
  int pcur = -1;
  bool page_valid = false;
  const uint8_t* kpb = kc;
  const uint8_t* vpb = vc;

  auto load_one = [&](int tok0, int poff, bool is_v, int tok, int ch) -> uint4 {
    const int gtok = tok0 + tok;
    if (gtok >= Lmax) return make_uint4(0u, 0u, 0u, 0u);
    const uint8_t* p;
    if (fast_page) {
      if (!page_valid) return make_uint4(0u, 0u, 0u, 0u);
      p = (is_v ? vpb : kpb) + (int64_t)(poff + tok) * (is_v ? vt : kt) + 16 * ch;
    } else {
      const int pi = gtok / page;
      if (pi < 0 || pi >= tw) return make_uint4(0u, 0u, 0u, 0u);
      const int64_t pg = (pi < TBS) ? (int64_t)Ts[pi] : (int64_t)tb[pi];
      if (pg < 0 || pg >= num_blocks) return make_uint4(0u, 0u, 0u, 0u);
      const int off = gtok % page;
      p = (is_v ? vc : kc) + pg * (is_v ? vb : kb) + (int64_t)h * (is_v ? vh : kh) + (int64_t)off * (is_v ? vt : kt) +
          16 * ch;
    }
    return ldg16_safe(p, kv_aligned16);
  };

  auto load_tile_regs = [&](int t, uint4 (&rbuf)[NVEC]) {
    const int tok0 = t * TILE;
    int poff = 0;
    if (fast_page) {
      const int pi = tok0 / page;
      if (pi != pcur) {
        pcur = pi;
        page_valid = false;
        if (pi >= 0 && pi < tw) {
          const int64_t pg = (pi < TBS) ? (int64_t)Ts[pi] : (int64_t)tb[pi];
          if (pg >= 0 && pg < num_blocks) {
            kpb = kc + pg * kb + (int64_t)h * kh;
            vpb = vc + pg * vb + (int64_t)h * vh;
            page_valid = true;
          }
        }
      }
      poff = tok0 % page;
    }
    if constexpr (NG == 1 && R_LAT == 192) {
      // NT=128, NV_HALF=8, NVEC=4: u=0,1 k_cache; u=2,3 v_cache (each thread touches 2 tokens)
      const int tok_lo = tid >> 3, ch = tid & 7;
      rbuf[0] = load_one(tok0, poff, false, tok_lo, ch);
      rbuf[1] = load_one(tok0, poff, false, tok_lo + 16, ch);
      rbuf[2] = load_one(tok0, poff, true, tok_lo, ch);
      rbuf[3] = load_one(tok0, poff, true, tok_lo + 16, ch);
    } else if constexpr (NG == 1 && R_LAT == 256) {
      // NT=128, NV_HALF=10, NVEC=5 (640 uint4s / 128 threads = 5 exact)
      const int i0 = tid, i1 = tid + 128;
      const int tok0_ = i0 / 10, ch0 = i0 % 10;
      const int tok1_ = i1 / 10, ch1 = i1 % 10;
      const bool v4 = (tid >= 64);
      const int i4 = 256 + (tid & 63);
      const int tok4 = i4 / 10, ch4 = i4 % 10;
      rbuf[0] = load_one(tok0, poff, false, tok0_, ch0);
      rbuf[1] = load_one(tok0, poff, false, tok1_, ch1);
      rbuf[2] = load_one(tok0, poff, true, tok0_, ch0);
      rbuf[3] = load_one(tok0, poff, true, tok1_, ch1);
      rbuf[4] = load_one(tok0, poff, v4, tok4, ch4);
    } else if constexpr (NG == 2 && R_LAT == 192) {
      // NT=256, NV_HALF=8, NVEC=2: 512 uint4s / 256 threads = 2 exact (1 token per thread, 0 divergence!)
      const int tok = tid >> 3, ch = tid & 7;
      rbuf[0] = load_one(tok0, poff, false, tok, ch);
      rbuf[1] = load_one(tok0, poff, true, tok, ch);
    } else {
      // NG == 2 && R_LAT == 256: NT=256, NV_HALF=10, NVEC=3 (256 k + 256 v + 64 k [warps 0..1] + 64 v [warps 2..3])
      const int tok_ = tid / 10, ch_ = tid % 10;
      rbuf[0] = load_one(tok0, poff, false, tok_, ch_);
      rbuf[1] = load_one(tok0, poff, true, tok_, ch_);
      if (tid < 128) {
        const bool v2 = (tid >= 64);
        const int i2 = 256 + (tid & 63);
        rbuf[2] = load_one(tok0, poff, v2, i2 / 10, i2 % 10);
      }
    }
  };

  auto store_one = [&](bool is_v, int tok, int ch, uint4 v) {
    if (is_v && ch == NV_HALF - 1) {
      // Last 2 bytes of v_cache row are raw __half inv_rms; extract to InvR[tok] and zero-mask in Tbuf
      InvR[tok] = __half2float(__ushort_as_half((uint16_t)(v.w >> 16)));
      v.w &= 0x0000FFFFu;
    }
    const int woff = (is_v ? (HALF_B / 4) : 0) + 4 * ch;
    uint32_t* dst = Tbuf + tok * TS_W + woff;
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
  };

  auto store_tile_smem = [&](const uint4 (&rbuf)[NVEC]) {
    if constexpr (NG == 1 && R_LAT == 192) {
      const int tok_lo = tid >> 3, ch = tid & 7;
      store_one(false, tok_lo, ch, rbuf[0]);
      store_one(false, tok_lo + 16, ch, rbuf[1]);
      store_one(true, tok_lo, ch, rbuf[2]);
      store_one(true, tok_lo + 16, ch, rbuf[3]);
    } else if constexpr (NG == 1 && R_LAT == 256) {
      const int i0 = tid, i1 = tid + 128;
      const int tok0_ = i0 / 10, ch0 = i0 % 10;
      const int tok1_ = i1 / 10, ch1 = i1 % 10;
      const bool v4 = (tid >= 64);
      const int i4 = 256 + (tid & 63);
      const int tok4 = i4 / 10, ch4 = i4 % 10;
      store_one(false, tok0_, ch0, rbuf[0]);
      store_one(false, tok1_, ch1, rbuf[1]);
      store_one(true, tok0_, ch0, rbuf[2]);
      store_one(true, tok1_, ch1, rbuf[3]);
      store_one(v4, tok4, ch4, rbuf[4]);
    } else if constexpr (NG == 2 && R_LAT == 192) {
      const int tok = tid >> 3, ch = tid & 7;
      store_one(false, tok, ch, rbuf[0]);
      store_one(true, tok, ch, rbuf[1]);
    } else {
      const int tok_ = tid / 10, ch_ = tid % 10;
      store_one(false, tok_, ch_, rbuf[0]);
      store_one(true, tok_, ch_, rbuf[1]);
      if (tid < 128) {
        const bool v2 = (tid >= 64);
        const int i2 = 256 + (tid & 63);
        store_one(v2, i2 / 10, i2 % 10, rbuf[2]);
      }
    }
  };

  // Issue first tile's HBM loads immediately so they overlap with Q staging
  uint4 rbuf[NVEC];
  load_tile_regs(t0, rbuf);

  // Stage Q rows [0..R-1] into SMEM Qs (zero-pad rows >= NR and zero-mask the 2 unused tail halves [ROW_B-2..ROW_B-1])
  constexpr int Q_CH = ROW_B / 8;   // 32 (R=192) or 40 (R=256) uint4s per row
  constexpr int QN = C::R * Q_CH;
  for (int i = tid; i < QN; i += NT) {
    const int r = i / Q_CH, c = (i % Q_CH) * 8;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < NR) {
      v = ldg16_safe(reinterpret_cast<const uint8_t*>(q + orow(r) * ROW_B + c), q_aligned16);
      if (c == ROW_B - 8) v.w = 0u;   // mask q_cat[..., ROW_B-2..ROW_B-1] to exact +0.0
    }
    uint2* qd = reinterpret_cast<uint2*>(Qs + r * QS + c);
    qd[0] = make_uint2(v.x, v.y);
    qd[1] = make_uint2(v.z, v.w);
  }

  // Volta HMMA.884 quadpair lane indices
  const int qp = (lane >> 2) & 3;
  const int hi = (lane >> 4) & 1;
  const int l3 = lane & 3;
  const int arow = l3 + 4 * hi;               // A row (0..7); B col for QK inside quadpair
  const int r0 = (lane & 1) + 4 * hi;         // C rows r0, r0 + 2
  const int c0 = lane & 2;                    // C cols c0, c0+1, c0+4, c0+5
  const int ktok = 8 * qp + arow;             // this thread's token in the 32-token tile (QK)
  const int pc = 8 * qp + c0;                 // base local token for C cols in Softmax
  const __half* qa = Qs + (roff + arow) * QS + wi * DK_B;

  // Per-row lengths for the row groups owned by this warp within set sg (local g = wi + u * NW)
  int lenA[RGW], lenB[RGW];
#pragma unroll
  for (int u = 0; u < RGW; u++) {
    const int ra = roff + 8 * (wi + u * NW) + r0, rb = ra + 2;
    lenA[u] = (ra < NR) ? lr[ra / NQ] : 0;
    lenB[u] = (rb < NR) ? lr[rb / NQ] : 0;
  }

  // Output accumulators in latent space + online softmax state (for this set's RG_SET row groups)
  float o[RG_SET][2][8];
#pragma unroll
  for (int g = 0; g < RG_SET; g++)
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[g][j][e] = 0.f;

  float m0[RGW], m2[RGW], l0[RGW], l2[RGW];
#pragma unroll
  for (int u = 0; u < RGW; u++) {
    m0[u] = -INFINITY;
    m2[u] = -INFINITY;
    l0[u] = 0.f;
    l2[u] = 0.f;
  }

  const int kw0 = wi * DK_W;
  const int col_w0 = (DV * wi) / 4 + 2 * qp + hi;
  const int col_w1 = (R_LAT == 192) ? ((DV * wi) / 4 + 8 + 2 * (qp & 1) + hi) : (col_w0 + 8);
  const int half_tok = (qp >> 1) * 16;   // R_LAT == 192 split-K half offset: 0 for qp=0,1; 16 for qp=2,3

  // ===================================== MAIN TILE LOOP =====================================
  for (int t = t0; t < t1; t++) {
    // 1. Unpack current tile from prefetch regs into SMEM Tbuf + InvR, then CTA-wide sync (also syncs Qs on t == t0)
    store_tile_smem(rbuf);
    __syncthreads();

    // 2. Prefetch next tile from HBM into rbuf while QK + Softmax + PV run on SMEM Tbuf
    if (t + 1 < t1) load_tile_regs(t + 1, rbuf);

    // 3. QK (mma_rc): warp wi accumulates partial scores over DK_B dims for this set's RG_SET row groups
    float sc[RG_SET][8], sx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      if constexpr (RG_SET == 1) sx[e] = 0.f;
#pragma unroll
      for (int g = 0; g < RG_SET; g++) sc[g][e] = 0.f;
    }

    const uint32_t* tk = Tbuf + ktok * TS_W + kw0;
#pragma unroll
    for (int k = 0; k < DK_W; k++) {
      unsigned b0, b1;
      e4m3x4(tk[k], b0, b1);
#pragma unroll
      for (int g = 0; g < RG_SET; g++) {
        const uint2 a = *reinterpret_cast<const uint2*>(qa + g * 8 * QS + 4 * k);
        if constexpr (RG_SET == 1) {
          mma_rc((k & 1) ? sx : sc[0], a.x, a.y, b0, b1);
        } else {
          mma_rc(sc[g], a.x, a.y, b0, b1);
        }
      }
    }
    if constexpr (RG_SET == 1) {
#pragma unroll
      for (int e = 0; e < 8; e++) sc[0][e] += sx[e];
    }

    // 4. Exchange partial QK scores via set-local 3-slot Xs (owners retain own partial in own_sc, compile-time indexed)
    float own_sc[RGW][8] = {};
#pragma unroll
    for (int g = 0; g < RG_SET; g++) {
      const int owner = g % NW;
      if (wi == owner) {
#pragma unroll
        for (int e = 0; e < 8; e++) own_sc[g / NW][e] = sc[g][e];
      } else {
        const int slot = (wi < owner) ? wi : (wi - 1);
        float* xw = Xs + ((g * (NW - 1) + slot) * 32 + lane) * 8;
        *reinterpret_cast<float4*>(xw) = make_float4(sc[g][0], sc[g][1], sc[g][2], sc[g][3]);
        *reinterpret_cast<float4*>(xw + 4) = make_float4(sc[g][4], sc[g][5], sc[g][6], sc[g][7]);
      }
    }
    bar_set<NG>(sg);

    // 5. Online FP32 softmax on warp-owned row groups within set sg (incorporating per-token inv_rms * sl2)
    const int tok0 = t * TILE;
    const float ir[4] = {InvR[pc + 0] * sl2, InvR[pc + 1] * sl2, InvR[pc + 4] * sl2, InvR[pc + 5] * sl2};
#pragma unroll
    for (int u = 0; u < RGW; u++) {
      const int g = wi + u * NW;
      if (g < RG_SET) {
        float sv[8];
#pragma unroll
        for (int e = 0; e < 8; e++) sv[e] = own_sc[u][e];
#pragma unroll
        for (int slot = 0; slot < NW - 1; slot++) {
          const float* x = Xs + ((g * (NW - 1) + slot) * 32 + lane) * 8;
          const float4 a = *reinterpret_cast<const float4*>(x);
          const float4 c = *reinterpret_cast<const float4*>(x + 4);
          sv[0] += a.x;
          sv[1] += a.y;
          sv[2] += a.z;
          sv[3] += a.w;
          sv[4] += c.x;
          sv[5] += c.y;
          sv[6] += c.z;
          sv[7] += c.w;
        }
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int col = tok0 + pc + (e & 1) + (e & 4);
          const float sc_e = ir[((e & 4) >> 1) | (e & 1)];
          sv[e] = (col < ((e & 2) ? lenB[u] : lenA[u])) ? (sv[e] * sc_e) : -INFINITY;
        }
        float x0 = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5]));
        float x2 = fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7]));
#pragma unroll
        for (int d = 2; d <= 8; d <<= 1) {
          x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, d));
          x2 = fmaxf(x2, __shfl_xor_sync(0xffffffffu, x2, d));
        }
        const float n0 = fmaxf(m0[u], x0), n2 = fmaxf(m2[u], x2);
        const float a0 = (m0[u] == -INFINITY) ? 0.f : ex2(m0[u] - n0);
        const float a2 = (m2[u] == -INFINITY) ? 0.f : ex2(m2[u] - n2);
        const float z0 = (n0 == -INFINITY) ? 0.f : n0;
        const float z2 = (n2 == -INFINITY) ? 0.f : n2;
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; e++) p[e] = ex2(sv[e] - ((e & 2) ? z2 : z0));
        l0[u] = l0[u] * a0 + (p[0] + p[1]) + (p[4] + p[5]);
        l2[u] = l2[u] * a2 + (p[2] + p[3]) + (p[6] + p[7]);
        m0[u] = n0;
        m2[u] = n2;

        __half* Pg = Ps + g * 8 * PS;
        *reinterpret_cast<__half2*>(Pg + r0 * PS + pc) = __floats2half2_rn(p[0], p[1]);
        *reinterpret_cast<__half2*>(Pg + r0 * PS + pc + 4) = __floats2half2_rn(p[4], p[5]);
        *reinterpret_cast<__half2*>(Pg + (r0 + 2) * PS + pc) = __floats2half2_rn(p[2], p[3]);
        *reinterpret_cast<__half2*>(Pg + (r0 + 2) * PS + pc + 4) = __floats2half2_rn(p[6], p[7]);
        if ((lane & 14) == 0) {
          As[g * 8 + r0] = a0;
          As[g * 8 + r0 + 2] = a2;
        }
      }
    }
    bar_set<NG>(sg);

    // 6. PV (mma_rr): rescale o, then accumulate P @ c reading c straight from shared SMEM Tbuf (ZERO HBM reads!)
#pragma unroll
    for (int g = 0; g < RG_SET; g++) {
      const float a0 = As[g * 8 + r0], a2 = As[g * 8 + r0 + 2];
#pragma unroll
      for (int j = 0; j < 2; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) o[g][j][e] *= (e & 2) ? a2 : a0;
    }

    if constexpr (R_LAT == 256) {
      // All 4 quadpairs own 2 m8n8 blocks each (64 dims/warp = 256 dims total)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint32_t* tbase = Tbuf + (4 * i + l3) * TS_W;
        unsigned b00, b01, b10, b11;
        e4m3x4(tbase[col_w0], b00, b01);
        e4m3x4(tbase[col_w1], b10, b11);
#pragma unroll
        for (int g = 0; g < RG_SET; g++) {
          const uint2 a = *reinterpret_cast<const uint2*>(Ps + (g * 8 + arow) * PS + 4 * i);
          mma_rr(o[g][0], a.x, a.y, b00, b01);
          mma_rr(o[g][1], a.x, a.y, b10, b11);
        }
      }
    } else {
      // R_LAT == 192 (48 dims/warp): j=0 covers 32 cols/warp (8 K-steps), j=1 covers 16 cols/warp split-K (4 K-steps)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        unsigned b0, b1;
        e4m3x4(Tbuf[(4 * i + l3) * TS_W + col_w0], b0, b1);
#pragma unroll
        for (int g = 0; g < RG_SET; g++) {
          const uint2 a = *reinterpret_cast<const uint2*>(Ps + (g * 8 + arow) * PS + 4 * i);
          mma_rr(o[g][0], a.x, a.y, b0, b1);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        unsigned b0, b1;
        e4m3x4(Tbuf[(half_tok + 4 * i + l3) * TS_W + col_w1], b0, b1);
#pragma unroll
        for (int g = 0; g < RG_SET; g++) {
          const uint2 a = *reinterpret_cast<const uint2*>(Ps + (g * 8 + arow) * PS + half_tok + 4 * i);
          mma_rr(o[g][1], a.x, a.y, b0, b1);
        }
      }
    }

    // CTA-wide sync: ensure all sets finish reading Tbuf before next iteration overwrites it (or epilogue aliases ML)
    __syncthreads();
  }

  // R_LAT == 192: reduce split-K halves of j=1 across quadpairs (qp=0 <-> qp=2, qp=1 <-> qp=3 => lane xor 8)
  if constexpr (R_LAT == 192) {
#pragma unroll
    for (int g = 0; g < RG_SET; g++)
#pragma unroll
      for (int e = 0; e < 8; e++) o[g][1][e] += __shfl_xor_sync(0xffffffffu, o[g][1][e], 8);
  }

  // Reduce row sums l across the 8 lanes sharing each row and store (m, l) into ML (aliases dead Tbuf)
  float* MLa = reinterpret_cast<float*>(sm + C::T_);
#pragma unroll
  for (int u = 0; u < RGW; u++) {
    const int g = wi + u * NW;
    if (g < RG_SET) {
#pragma unroll
      for (int d = 2; d <= 8; d <<= 1) {
        l0[u] += __shfl_xor_sync(0xffffffffu, l0[u], d);
        l2[u] += __shfl_xor_sync(0xffffffffu, l2[u], d);
      }
      if ((lane & 14) == 0) {
        const int rbase = roff + g * 8 + r0;
        MLa[rbase * 2] = m0[u];
        MLa[rbase * 2 + 1] = l0[u];
        MLa[(rbase + 2) * 2] = m2[u];
        MLa[(rbase + 2) * 2 + 1] = l2[u];
      }
    }
  }
  __syncthreads();

  // ---- Single-split direct output write ----
  if (Sb == 1) {
#pragma unroll
    for (int g = 0; g < RG_SET; g++) {
#pragma unroll
      for (int sr = 0; sr < 2; sr++) {
        const int row = roff + g * 8 + r0 + 2 * sr;
        if (row < NR) {
          const float l = MLa[row * 2 + 1];
          const float f = (l > 0.f) ? (vsc / l) : 0.f;
          __half* ob = out + orow(row) * R_LAT;
#pragma unroll
          for (int j = 0; j < 2; j++) {
            if (R_LAT == 256 || j == 0 || qp < 2) {
              const int dlo = DV * wi + 32 * j + 8 * qp + c0;
              st_h2_safe(ob + dlo, __floats2half2_rn(o[g][j][2 * sr] * f, o[g][j][2 * sr + 1] * f), out_aligned4);
              st_h2_safe(ob + dlo + 4, __floats2half2_rn(o[g][j][4 + 2 * sr] * f, o[g][j][5 + 2 * sr] * f),
                         out_aligned4);
            }
          }
        }
      }
    }
    return;
  }

  // ---- Multi-split partial write ----
  const int SG = gridDim.x;
  float* pob = po + ((size_t)bh * SG + s) * NR * R_LAT;
  float* pmb = pml + ((size_t)bh * SG + s) * NR * 2;
#pragma unroll
  for (int g = 0; g < RG_SET; g++) {
#pragma unroll
    for (int sr = 0; sr < 2; sr++) {
      const int row = roff + g * 8 + r0 + 2 * sr;
      if (row < NR) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          if (R_LAT == 256 || j == 0 || qp < 2) {
            const int dlo = DV * wi + 32 * j + 8 * qp + c0;
            __stcg(reinterpret_cast<float2*>(pob + row * R_LAT + dlo),
                   make_float2(o[g][j][2 * sr], o[g][j][2 * sr + 1]));
            __stcg(reinterpret_cast<float2*>(pob + row * R_LAT + dlo + 4),
                   make_float2(o[g][j][4 + 2 * sr], o[g][j][5 + 2 * sr]));
          }
        }
      }
    }
  }
  for (int i = tid; i < NR * 2; i += NT) __stcg(pmb + i, MLa[i]);

  if constexpr (!FUSED_COMB) {
    return;   // combine in separate k_comb launch (sep == 1)
  } else {
    // ---- Fused single-wave split barrier + combine (sep == 0) ----
    // NOTE: sep == 0 requires single-stream serialization per device (reuses cnt[] and relies on 1-wave residency);
    // sep == 1 (default) uses separate deterministic k_comb and is safe for concurrent streams.
    __syncthreads();
    if (tid == 0) {
      const int arr = atom_add_acq_rel_gpu(cnt + bh, 1);
      int cur = arr + 1;
#ifdef Q27_GLA_DEBUG_STALL
      long long spins = 0;
#endif
      while (cur < Sb) {
        __nanosleep(64);
        cur = ld_acquire_gpu(cnt + bh);
#ifdef Q27_GLA_DEBUG_STALL
        if (++spins == (1ll << 22)) {
          printf("q27 attn_gla_mq: split barrier stalled: b %d h %d s %d Sb %d grid %d arrival %d cnt %d Lmax %d Q %d\n",
                 b, h, s, Sb, (int)gridDim.x, arr, cur, Lmax, Q);
        }
#endif
      }
    }
    __syncthreads();

    combine_slice<R_LAT, NT>(sm, po, pml, out, bh, SG, NR, Sb, s, b, h, Hkv, Q, vsc, out_aligned4);
    __syncthreads();
    if (tid == 0 && atom_add_acq_rel_gpu(cnt + bh, 1) == 2 * Sb - 1) st_release_gpu(cnt + bh, 0);
  }
}

// Separate deterministic combine kernel (sep == 1)
template <int R_LAT, int CT>
__global__ void __launch_bounds__(CT)
k_comb(const float* __restrict__ po, const float* __restrict__ pml, const int* __restrict__ lens,
       __half* __restrict__ out, int Q, int S, int min_tiles, float vsc, bool out_aligned4) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z, Hkv = gridDim.y, bh = b * Hkv + h;
  const int* lr = lens + (size_t)b * Q;
  int Lmax = 0;
#pragma unroll
  for (int i = 0; i < QMAX; i++) {
    if (i < Q) Lmax = max(Lmax, lr[i]);
  }
  if (Lmax <= 0) return;
  const int T = (Lmax + TILE - 1) / TILE;
  const int Sb = max(1, min(S, T / min_tiles));
  if (Sb == 1 || s >= Sb) return;
  combine_slice<R_LAT, CT>(sm, po, pml, out, bh, (int)gridDim.x, NQ * Q, Sb, s, b, h, Hkv, Q, vsc, out_aligned4);
}

// =================================================================================================
// Per-device one-time attribute & occupancy initialization (ZERO driver calls during graph capture)
// Config index per Q (RG_total = ceil(6*Q/8)):
//   cfg 0: Q=1         (RG_total=1) -> RG_SET=1, NG=1 (128 threads, 4-5 CTAs/SM = 16-20 warps/SM)
//   cfg 1: Q=2         (RG_total=2) -> RG_SET=1, NG=2 (256 threads, 3 CTAs/SM   = 24 warps/SM)
//   cfg 2: Q=3,4,5     (RG_total=3,4)-> RG_SET=2, NG=2 (256 threads, 2 CTAs/SM  = 16 warps/SM)
//   cfg 3: Q=6,7,8     (RG_total=5,6)-> RG_SET=3, NG=2 (256 threads, 1 CTA/SM   = 8 warps/SM)
// =================================================================================================
struct DevCaps {
  int sms = 0;
  int cap[2][4] = {};   // [r_idx: 0=192, 1=256][cfg: 0..3] = occ_fus * sms
  int occ[2][4] = {};   // [r_idx: 0=192, 1=256][cfg: 0..3] = occ_sep
};

std::once_flag g_dev_once[MAX_DEVS];
std::atomic<bool> g_dev_ready[MAX_DEVS] = {};
DevCaps g_dev_caps[MAX_DEVS];

inline int rg_for_q(int64_t Q) { return (int)((NQ * Q + 7) / 8); }

inline int cfg_for_q(int64_t Q) {
  const int rg = rg_for_q(Q);
  if (rg == 1) return 0;
  if (rg == 2) return 1;
  if (rg <= 4) return 2;
  return 3;
}

template <int R_LAT, int RG_SET, int NG>
void init_one(int dev, int r_idx, int cfg_idx) {
  using C = GLA<R_LAT, RG_SET, NG>;
  auto f_sep = k_gla<R_LAT, RG_SET, NG, false>;
  auto f_fus = k_gla<R_LAT, RG_SET, NG, true>;
  for (auto f : {f_sep, f_fus}) {
    TORCH_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) == cudaSuccess,
                "attn_gla_mq: cudaFuncSetAttribute MaxDynamicSharedMemorySize failed");
    TORCH_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared) == cudaSuccess,
                "attn_gla_mq: cudaFuncSetAttribute Carveout failed");
  }
  int occ_fus = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_fus, f_fus, C::NT, C::MAIN) == cudaSuccess,
              "attn_gla_mq: cudaOccupancyMaxActiveBlocksPerMultiprocessor failed");
  TORCH_CHECK(occ_fus >= 1, "attn_gla_mq: kernel occupancy < 1");
  int occ_sep = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sep, f_sep, C::NT, C::MAIN) == cudaSuccess,
              "attn_gla_mq: cudaOccupancyMaxActiveBlocksPerMultiprocessor failed");
  TORCH_CHECK(occ_sep >= 1, "attn_gla_mq: sep kernel occupancy < 1");
  // cap[] is used to bound sep==0 to one resident wave, so store occ_fus for safety
  g_dev_caps[dev].occ[r_idx][cfg_idx] = occ_sep;
  g_dev_caps[dev].cap[r_idx][cfg_idx] = occ_fus * g_dev_caps[dev].sms;
}

void ensure_init(int dev, cudaStream_t st = nullptr) {
  TORCH_CHECK(dev >= 0 && dev < MAX_DEVS, "attn_gla_mq: invalid CUDA device index ", dev);
  if (g_dev_ready[dev].load(std::memory_order_acquire)) return;
  if (st != nullptr) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
      TORCH_CHECK(false, "attn_gla_mq: device ", dev,
                  " not initialized; call init(dev_idx) before CUDA graph capture");
    }
  }
  std::call_once(g_dev_once[dev], [dev]() {
    const at::cuda::CUDAGuard guard(dev);
    int sms = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess,
                "attn_gla_mq: cudaDeviceGetAttribute failed");
    g_dev_caps[dev].sms = sms;
    init_one<192, 1, 1>(dev, 0, 0);
    init_one<192, 1, 2>(dev, 0, 1);
    init_one<192, 2, 2>(dev, 0, 2);
    init_one<192, 3, 2>(dev, 0, 3);
    init_one<256, 1, 1>(dev, 1, 0);
    init_one<256, 1, 2>(dev, 1, 1);
    init_one<256, 2, 2>(dev, 1, 2);
    init_one<256, 3, 2>(dev, 1, 3);
    TORCH_CHECK(cudaFuncSetAttribute(k_comb<192, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) ==
                    cudaSuccess,
                "attn_gla_mq: k_comb<192> attribute failed");
    TORCH_CHECK(cudaFuncSetAttribute(k_comb<256, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) ==
                    cudaSuccess,
                "attn_gla_mq: k_comb<256> attribute failed");
    g_dev_ready[dev].store(true, std::memory_order_release);
  });
}

inline int plan_splits_dev(int dev, int64_t bh, int64_t S, int64_t R_LAT, int64_t Q, int64_t sep = 1,
                           cudaStream_t st = nullptr) {
  TORCH_CHECK(R_LAT == 192 || R_LAT == 256, "attn_gla_mq: R_LAT must be 192 or 256, got ", R_LAT);
  TORCH_CHECK(Q >= 1 && Q <= QMAX, "attn_gla_mq: Q must be in 1..8, got ", Q);
  TORCH_CHECK(bh >= 0, "attn_gla_mq: bh must be >= 0");
  TORCH_CHECK(sep == 0 || sep == 1, "attn_gla_mq: sep must be 0 or 1");
  ensure_init(dev, st);
  int s = (int)std::max<int64_t>(1, std::min<int64_t>(S, SMAX));
  // 1-resident-wave cap is required ONLY for sep == 0 (in-kernel barrier); sep == 1 allows up to SMAX splits
  // (bounded only by k_comb's 96 KiB SMEM, which holds for all S <= 128, Q <= 8).
  if (sep == 0 && s > 1 && bh > 0) {
    const int r_idx = (R_LAT == 192) ? 0 : 1;
    const int cfg = cfg_for_q(Q);
    const int cap = g_dev_caps[dev].cap[r_idx][cfg];
    if (bh * (int64_t)s > (int64_t)cap) {
      s = (int)std::max<int64_t>(1, (int64_t)cap / bh);
    }
  }
  return s;
}

template <int R_LAT, int RG_SET, int NG>
void launch_gla(const torch::Tensor& q_cat, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
                const torch::Tensor& block_table, const torch::Tensor& seq_lens, const torch::Tensor& out_lat,
                const torch::Tensor& po, const torch::Tensor& pml, const torch::Tensor& cnt, int Q, int S_eff,
                int min_tiles, int sep, float sl2, float vsc, cudaStream_t st) {
  using C = GLA<R_LAT, RG_SET, NG>;
  const int B = (int)block_table.size(0);
  const int num_blocks = (int)k_cache.size(0);
  const int Hkv = (int)k_cache.size(2);
  const int page = (int)k_cache.size(1);
  const int nr = NQ * Q;

  const auto* q_ptr = reinterpret_cast<const __half*>(q_cat.data_ptr<at::Half>());
  const auto* kc_ptr = k_cache.data_ptr<uint8_t>();
  const auto* vc_ptr = v_cache.data_ptr<uint8_t>();
  auto* out_ptr = reinterpret_cast<__half*>(out_lat.data_ptr<at::Half>());

  const int64_t kb = k_cache.stride(0), kt = k_cache.stride(1), kh = k_cache.stride(2);
  const int64_t vb = v_cache.stride(0), vt = v_cache.stride(1), vh = v_cache.stride(2);
  const bool fast_page = (page >= TILE) && (page % TILE == 0);
  const bool kv_aligned16 = ((((uintptr_t)kc_ptr | (uintptr_t)vc_ptr) & 15u) == 0u) &&
                            (((kb | kt | kh | vb | vt | vh) & 15ll) == 0ll);
  const bool q_aligned16 = (((uintptr_t)q_ptr & 15u) == 0u);
  const bool out_aligned4 = (((uintptr_t)out_ptr & 3u) == 0u);

  constexpr int smem = C::MAIN;
  if (sep == 0 && S_eff > 1) {
    int cmax = 0;
    for (int sb = 2; sb <= S_eff; sb++) cmax = std::max(cmax, 4 * comb_floats<R_LAT>(sb, nr) + 16);
    TORCH_CHECK(cmax <= smem, "attn_gla_mq: combine SMEM exceeds MAIN SMEM (", cmax, " > ", smem, ")");
  }

  float* po_ptr = (S_eff > 1) ? po.data_ptr<float>() : nullptr;
  float* pml_ptr = (S_eff > 1) ? pml.data_ptr<float>() : nullptr;
  int* cnt_ptr = (S_eff > 1 && sep == 0) ? cnt.data_ptr<int>() : nullptr;

  dim3 grid(S_eff, Hkv, B);
  if (sep) {
    k_gla<R_LAT, RG_SET, NG, false><<<grid, C::NT, smem, st>>>(
        q_ptr, kc_ptr, vc_ptr, block_table.data_ptr<int>(), (int)block_table.stride(0), (int)block_table.size(1),
        seq_lens.data_ptr<int>(), out_ptr, po_ptr, pml_ptr, cnt_ptr, Q, S_eff, min_tiles, page, num_blocks, kb, kt, kh,
        vb, vt, vh, sl2, vsc, fast_page, kv_aligned16, q_aligned16, out_aligned4);
  } else {
    k_gla<R_LAT, RG_SET, NG, true><<<grid, C::NT, smem, st>>>(
        q_ptr, kc_ptr, vc_ptr, block_table.data_ptr<int>(), (int)block_table.stride(0), (int)block_table.size(1),
        seq_lens.data_ptr<int>(), out_ptr, po_ptr, pml_ptr, cnt_ptr, Q, S_eff, min_tiles, page, num_blocks, kb, kt, kh,
        vb, vt, vh, sl2, vsc, fast_page, kv_aligned16, q_aligned16, out_aligned4);
  }

  if (sep && S_eff > 1) {
    constexpr int CT = 256;
    int cs = 0;
    for (int sb = 2; sb <= S_eff; sb++) cs = std::max(cs, 4 * comb_floats<R_LAT>(sb, nr) + 16);
    TORCH_CHECK(cs <= 96 * 1024, "attn_gla_mq: k_comb SMEM exceeds 96 KiB (", cs, ")");
    k_comb<R_LAT, CT><<<grid, CT, cs, st>>>(po_ptr, pml_ptr, seq_lens.data_ptr<int>(), out_ptr, Q, S_eff, min_tiles,
                                            vsc, out_aligned4);
  }
}

template <int R_LAT>
void dispatch_cfg(int cfg, const torch::Tensor& q_cat, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
                  const torch::Tensor& block_table, const torch::Tensor& seq_lens, const torch::Tensor& out_lat,
                  const torch::Tensor& po, const torch::Tensor& pml, const torch::Tensor& cnt, int Q, int S_eff,
                  int min_tiles, int sep, float sl2, float vsc, cudaStream_t st) {
  switch (cfg) {
    case 0:
      launch_gla<R_LAT, 1, 1>(q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, Q, S_eff,
                              min_tiles, sep, sl2, vsc, st);
      return;
    case 1:
      launch_gla<R_LAT, 1, 2>(q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, Q, S_eff,
                              min_tiles, sep, sl2, vsc, st);
      return;
    case 2:
      launch_gla<R_LAT, 2, 2>(q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, Q, S_eff,
                              min_tiles, sep, sl2, vsc, st);
      return;
    case 3:
      launch_gla<R_LAT, 3, 2>(q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, Q, S_eff,
                              min_tiles, sep, sl2, vsc, st);
      return;
    default:
      TORCH_CHECK(false, "attn_gla_mq: unsupported cfg = ", cfg);
  }
}

bool ranges_overlap(const void* a, size_t na, const void* b, size_t nb) {
  if (na == 0 || nb == 0) return false;
  const uintptr_t a0 = reinterpret_cast<uintptr_t>(a);
  const uintptr_t b0 = reinterpret_cast<uintptr_t>(b);
  return (a0 < b0 + nb) && (b0 < a0 + na);
}

struct ByteRange {
  uintptr_t lo = 0;
  uintptr_t hi = 0;
};

ByteRange tensor_byte_range(const torch::Tensor& t) {
  if (t.numel() == 0) return {0, 0};
  const uintptr_t base = reinterpret_cast<uintptr_t>(t.data_ptr());
  const int64_t elem = (int64_t)t.element_size();
  int64_t min_off = 0, max_off = 0;
  for (int64_t d = 0; d < t.dim(); d++) {
    const int64_t span = (t.size(d) - 1) * t.stride(d) * elem;
    if (span < 0) min_off += span;
    else max_off += span;
  }
  const uintptr_t lo = (min_off < 0) ? (base - (uintptr_t)(-min_off)) : (base + (uintptr_t)min_off);
  const uintptr_t hi = base + (uintptr_t)(max_off + elem);
  return {lo, hi};
}

bool byte_ranges_overlap(ByteRange a, ByteRange b) {
  if (a.lo == a.hi || b.lo == b.hi) return false;
  return (a.lo < b.hi) && (b.lo < a.hi);
}

// Verify k_cache and v_cache ([num_blocks, page_size, 1, HALF_B] uint8, stride(3)==1) have strictly
// non-overlapping writable rows within each cache AND are byte-disjoint from each other, while
// preserving support for disjoint interleaved layouts like [num_blocks, 2, page_size, 1, HALF_B] and
// [num_blocks, page_size, 2, 1, HALF_B].
void validate_kv_caches_disjoint(const torch::Tensor& k_cache, const torch::Tensor& v_cache, const char* op) {
  const int64_t nb = k_cache.size(0);
  const int64_t ps = k_cache.size(1);
  const int64_t hb = k_cache.size(3);
  const int64_t kb = k_cache.stride(0), kt = k_cache.stride(1);
  const int64_t vb = v_cache.stride(0), vt = v_cache.stride(1);

  if (nb == 0) return;

  TORCH_CHECK(ps == 1 || kt >= hb, op, ": k_cache token rows overlap or have invalid stride(1)=", kt);
  TORCH_CHECK(ps == 1 || vt >= hb, op, ": v_cache token rows overlap or have invalid stride(1)=", vt);
  const int64_t k_blk_span = (ps - 1) * kt + hb;
  const int64_t v_blk_span = (ps - 1) * vt + hb;
  TORCH_CHECK(nb == 1 || kb >= k_blk_span, op, ": k_cache blocks overlap or have invalid stride(0)=", kb);
  TORCH_CHECK(nb == 1 || vb >= v_blk_span, op, ": v_cache blocks overlap or have invalid stride(0)=", vb);

  const ByteRange kr = tensor_byte_range(k_cache);
  const ByteRange vr = tensor_byte_range(v_cache);
  if (!byte_ranges_overlap(kr, vr)) return;

  // Allow disjoint interleaved block-strided or page-strided views of the same backing buffer
  const uintptr_t k0 = kr.lo, v0 = vr.lo;
  TORCH_CHECK(k0 != v0, op, ": k_cache and v_cache alias the same starting address");
  const bool lo_is_k = (k0 < v0);
  const uintptr_t diff = lo_is_k ? (v0 - k0) : (k0 - v0);
  const uintptr_t lo_blk = (uintptr_t)(lo_is_k ? k_blk_span : v_blk_span);
  const uintptr_t hi_blk = (uintptr_t)(lo_is_k ? v_blk_span : k_blk_span);

  // Case 1: block-interleaved (e.g. [nb, 2, ps, 1, hb])
  const bool block_interleaved = (nb > 1 && kb == vb && diff >= lo_blk && diff + hi_blk <= (uintptr_t)kb);
  // Case 2: token-interleaved (e.g. [nb, ps, 2, 1, hb])
  const bool token_interleaved =
      (ps > 1 && kt == vt && (nb == 1 || kb == vb) && diff >= (uintptr_t)hb && diff + (uintptr_t)hb <= (uintptr_t)kt &&
       (nb == 1 || diff + hi_blk <= (uintptr_t)kb));

  TORCH_CHECK(block_interleaved || token_interleaved, op, ": k_cache and v_cache memory regions overlap");
}

// =================================================================================================
// Fused Attention Preparation & Output Epilogue Helpers + Kernels (Follow-up #1)
// =================================================================================================
__device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
  __half2 h;
  memcpy(&h, &u, sizeof(h));
  return h;
}

__device__ __forceinline__ uint32_t h2_to_u32(__half2 h) {
  uint32_t u;
  memcpy(&u, &h, sizeof(u));
  return u;
}

__device__ __forceinline__ uint4 ldg16_h_safe(const __half* p, bool aligned16, bool aligned4) {
  if (aligned16) return ldg16(p);
  if (aligned4) {
    const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
    return make_uint4(ldg4(q), ldg4(q + 1), ldg4(q + 2), ldg4(q + 3));
  }
  const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
  const uint32_t w0 = (uint32_t)h[0] | ((uint32_t)h[1] << 16);
  const uint32_t w1 = (uint32_t)h[2] | ((uint32_t)h[3] << 16);
  const uint32_t w2 = (uint32_t)h[4] | ((uint32_t)h[5] << 16);
  const uint32_t w3 = (uint32_t)h[6] | ((uint32_t)h[7] << 16);
  return make_uint4(w0, w1, w2, w3);
}

__device__ __forceinline__ uint4 ld16_h_coherent_safe(const __half* p, bool aligned16, bool aligned4) {
  if (aligned16) return *reinterpret_cast<const uint4*>(p);
  if (aligned4) {
    const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
    return make_uint4(q[0], q[1], q[2], q[3]);
  }
  const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
  const uint32_t w0 = (uint32_t)h[0] | ((uint32_t)h[1] << 16);
  const uint32_t w1 = (uint32_t)h[2] | ((uint32_t)h[3] << 16);
  const uint32_t w2 = (uint32_t)h[4] | ((uint32_t)h[5] << 16);
  const uint32_t w3 = (uint32_t)h[6] | ((uint32_t)h[7] << 16);
  return make_uint4(w0, w1, w2, w3);
}

__device__ __forceinline__ void st16_h_safe(__half* p, uint4 v, bool aligned16, bool aligned4) {
  if (aligned16) {
    *reinterpret_cast<uint4*>(p) = v;
  } else if (aligned4) {
    uint32_t* q = reinterpret_cast<uint32_t*>(p);
    q[0] = v.x;
    q[1] = v.y;
    q[2] = v.z;
    q[3] = v.w;
  } else {
    uint16_t* h = reinterpret_cast<uint16_t*>(p);
    h[0] = (uint16_t)(v.x & 0xFFFFu);
    h[1] = (uint16_t)(v.x >> 16);
    h[2] = (uint16_t)(v.y & 0xFFFFu);
    h[3] = (uint16_t)(v.y >> 16);
    h[4] = (uint16_t)(v.z & 0xFFFFu);
    h[5] = (uint16_t)(v.z >> 16);
    h[6] = (uint16_t)(v.w & 0xFFFFu);
    h[7] = (uint16_t)(v.w >> 16);
  }
}

__device__ __forceinline__ __half2 ldg_h2_safe(const __half* p, bool aligned4) {
  if (aligned4) return u32_to_h2(ldg4(p));
  return __halves2half2(p[0], p[1]);
}

// Load M-RoPE (cos, sin) for frequency f in 0..30 (matching sm70_qwen38_qk_rope.py):
//   - pos_planes == 3 uses interleaved plane assignment p = 1 if f%3==1 else (2 if f%3==2 && f<30 else 0).
//   - Position bounds contract: negative pos wraps once (pos += cache_rows), then clamps safely to
//     [0, cache_rows - 1] so CUDA-graph dummy/padding positions can never cause out-of-bounds access.
template <bool POS64>
__device__ __forceinline__ void load_mrope_cs(const __half* __restrict__ cs_cache, int64_t cache_rows,
                                              const void* __restrict__ pos_ptr, int pos_planes,
                                              int64_t pos_row_stride, int64_t pos_col_stride, int64_t n, int f,
                                              float& c, float& s) {
  const int p = (pos_planes == 3) ? ((f % 3 == 1) ? 1 : ((f % 3 == 2 && f < 30) ? 2 : 0)) : 0;
  const int64_t pidx = (int64_t)p * pos_row_stride + n * pos_col_stride;
  int64_t pos = POS64 ? reinterpret_cast<const int64_t*>(pos_ptr)[pidx]
                      : (int64_t)reinterpret_cast<const int32_t*>(pos_ptr)[pidx];
  if (pos < 0) pos += cache_rows;
  if (pos < 0) pos = 0;
  else if (pos >= cache_rows) pos = cache_rows - 1;
  c = __half2float(cs_cache[pos * 64 + f]);
  s = __half2float(cs_cache[pos * 64 + 32 + f]);
}

template <int R_LAT, bool SLOT64, bool POS64, bool WRITE_EXTRA>
__device__ __forceinline__ void kv_prep_and_cache_write(
    const __half* __restrict__ qckr, int64_t stride_qckr, const void* __restrict__ pos_ptr, int pos_planes,
    int64_t pos_row_stride, int64_t pos_col_stride, const float* __restrict__ kw_rope_perm,
    const __half* __restrict__ w_k_up_raw, const __half* __restrict__ cs_cache, int64_t cache_rows,
    const void* __restrict__ slot_ptr, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    __half* __restrict__ k_rot_out, float* __restrict__ inv_rms_out, int64_t n, int page_size, int64_t max_slots,
    int64_t kb, int64_t kt, int64_t vb, int64_t vt, float eps_f, float inv_scale, bool k_aligned4, bool v_aligned4,
    __half* c_sm, __half* kr_sm, float* wsum, float* inv_rms_sm) {
  constexpr int ROW_B = R_LAT + 64;
  constexpr int HALF_B = ROW_B / 2;
  constexpr int HALF_W = HALF_B / 4;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  const __half* c_src = qckr + n * stride_qckr + 3072;
  const __half* kr_src = c_src + R_LAT;
  for (int i = tid; i < R_LAT; i += 192) c_sm[i] = c_src[i];
  if (tid < ROPE_DIM) kr_sm[tid] = kr_src[tid];
  __syncthreads();

  // Exact pre-norm RMS: k_nope_raw = c @ W_k_up_raw^T (w_k_up_raw is [R_LAT, 194]) + raw k_rope
  float sq = 0.f;
  for (int row = tid; row < NOPE_DIM; row += 192) {
    float dot = 0.f;
    const __half* w_ptr = w_k_up_raw + row;
#pragma unroll 4
    for (int j = 0; j < R_LAT; j++) {
      dot = fmaf(__half2float(c_sm[j]), __half2float(w_ptr[(int64_t)j * NOPE_DIM]), dot);
    }
    sq = fmaf(dot, dot, sq);
  }
  if (warp_id == 0) {
    const float k0 = __half2float(kr_sm[lane]);
    sq = fmaf(k0, k0, sq);
    if (lane < 30) {
      const float k1 = __half2float(kr_sm[lane + 32]);
      sq = fmaf(k1, k1, sq);
    }
  }
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, d);
  if (lane == 0) wsum[warp_id] = sq;
  __syncthreads();

  if (tid == 0) {
    const float total_sq = wsum[0] + wsum[1] + wsum[2] + wsum[3] + wsum[4] + wsum[5];
    *inv_rms_sm = rsqrtf(total_sq * (1.0f / 256.0f) + eps_f);
  }
  if (tid < 31) {
    const float a = __half2float(__float2half_rn(__half2float(kr_sm[tid]) * kw_rope_perm[tid]));
    const float b = __half2float(__float2half_rn(__half2float(kr_sm[tid + 31]) * kw_rope_perm[tid + 31]));
    float c, s;
    load_mrope_cs<POS64>(cs_cache, cache_rows, pos_ptr, pos_planes, pos_row_stride, pos_col_stride, n, tid, c, s);
    kr_sm[tid] = __float2half_rn(a * c - b * s);
    kr_sm[tid + 31] = __float2half_rn(b * c + a * s);
  }
  __syncthreads();

  if constexpr (WRITE_EXTRA) {
    if (tid < ROPE_DIM) k_rot_out[n * ROPE_DIM + tid] = kr_sm[tid];
    if (tid == 0) inv_rms_out[n] = *inv_rms_sm;
  }

  const int64_t slot = SLOT64 ? reinterpret_cast<const int64_t*>(slot_ptr)[n]
                              : (int64_t)reinterpret_cast<const int32_t*>(slot_ptr)[n];
  if (slot >= 0 && slot < max_slots) {
    const int64_t blk = slot / page_size;
    const int64_t off = slot % page_size;
    if (warp_id == 0) {
      uint8_t* k_dst = kc + blk * kb + off * kt;
      for (int w = lane; w < HALF_W; w += 32) {
        const uint32_t word = pack4_e4m3_h(c_sm + 4 * w, inv_scale);
        st4_safe(k_dst + 4 * w, word, k_aligned4);
      }
    } else if (warp_id == 1) {
      uint8_t* v_dst = vc + blk * vb + off * vt;
      const uint16_t inv_u16 = __half_as_ushort(__float2half_rn(*inv_rms_sm));
      for (int w = lane; w < HALF_W; w += 32) {
        const int pos = HALF_B + 4 * w;
        uint32_t word;
        if (pos < R_LAT) {
          word = pack4_e4m3_h(c_sm + pos, inv_scale);
        } else if (w < HALF_W - 1) {
          word = pack4_e4m3_h(kr_sm + (pos - R_LAT), inv_scale);
        } else {
          const uint8_t b0 = f32_to_e4m3fn(__fmul_rn(__half2float(kr_sm[60]), inv_scale));
          const uint8_t b1 = f32_to_e4m3fn(__fmul_rn(__half2float(kr_sm[61]), inv_scale));
          word = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)inv_u16 << 16);
        }
        st4_safe(v_dst + 4 * w, word, v_aligned4);
      }
    }
  }
}

// =================================================================================================
// Op 5: gla_prep_route_a
// =================================================================================================
template <int R_LAT, bool SLOT64, bool POS64>
__global__ void __launch_bounds__(192)
k_gla_prep_route_a(const __half* __restrict__ qckr, int64_t stride_qckr, const void* __restrict__ pos_ptr,
                   int pos_planes, int64_t pos_row_stride, int64_t pos_col_stride,
                   const float* __restrict__ q_norm_w1p, const float* __restrict__ kw_rope_perm,
                   const __half* __restrict__ w_k_up_raw, const __half* __restrict__ w_k_up_sc,
                   const __half* __restrict__ cs_cache, int64_t cache_rows, const void* __restrict__ slot_ptr,
                   uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, __half* __restrict__ q_cat,
                   int page_size, int64_t max_slots, int64_t kb, int64_t kt, int64_t vb, int64_t vt,
                   float eps_f, float inv_scale, bool qckr_al16, bool qckr_al4, bool w_sc_al4, bool qcat_al4,
                   bool k_aligned4, bool v_aligned4) {
  constexpr int ROW_B = R_LAT + 64;
  constexpr int N_QBLK = R_LAT / 64;
  constexpr int QN_STRIDE = 200;   // 16B-aligned padded stride for [6][194]

  // Shared memory (trivial POD bytes, 16B aligned):
  //   Q-block:  q_nope[6][200] (2400 B) + q_rot[6][64] (768 B) = 3168 B
  //   KV-block: c[R_LAT] (<=512 B) + kr[64] (128 B) + wsum[6] (24 B) + inv_rms (4 B) = 668 B
  __shared__ __align__(16) uint8_t sm_raw[NQ * QN_STRIDE * 2 + NQ * 64 * 2];

  const int64_t n = (int64_t)blockIdx.x;
  const int by = (int)blockIdx.y;

  if (by < N_QBLK) {
    __half* q_nope_base = reinterpret_cast<__half*>(sm_raw);
    __half* q_rot_base = reinterpret_cast<__half*>(sm_raw + NQ * QN_STRIDE * 2);

    const int h = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __half* q_nope_h = q_nope_base + h * QN_STRIDE;
    __half* q_rot_h = q_rot_base + h * 64;

    const int base = lane * 8;
    const uint4 qv = ldg16_h_safe(qckr + n * stride_qckr + h * 512 + base, qckr_al16, qckr_al4);

    auto sq_h2 = [](uint32_t u) -> float {
      const float2 f = __half22float2(u32_to_h2(u));
      return fmaf(f.x, f.x, f.y * f.y);
    };
    float ss = sq_h2(qv.x) + sq_h2(qv.y) + sq_h2(qv.z) + sq_h2(qv.w);
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float inv = rsqrtf(ss * (1.0f / 256.0f) + eps_f);

    auto norm_h2 = [&](uint32_t u, int idx) -> __half2 {
      const float2 f = __half22float2(u32_to_h2(u));
      return __floats2half2_rn(f.x * inv * q_norm_w1p[idx], f.y * inv * q_norm_w1p[idx + 1]);
    };
    const __half2 n0 = norm_h2(qv.x, base + 0);
    const __half2 n1 = norm_h2(qv.y, base + 2);
    const __half2 n2 = norm_h2(qv.z, base + 4);
    const __half2 n3 = norm_h2(qv.w, base + 6);

    if (lane < 8) {
      __half2* dst = reinterpret_cast<__half2*>(q_rot_h + base);
      dst[0] = n0;
      dst[1] = n1;
      dst[2] = n2;
      dst[3] = n3;
      if (lane == 3) q_nope_h[0] = __high2half(n3);   // idx 31 -> q_nope[0]
      if (lane == 7) q_nope_h[1] = __high2half(n3);   // idx 63 -> q_nope[1]
    } else {
      __half2* dst = reinterpret_cast<__half2*>(q_nope_h + (base - 62));
      dst[0] = n0;
      dst[1] = n1;
      dst[2] = n2;
      dst[3] = n3;
    }
    __syncwarp(0xffffffffu);

    __half* q_cat_h = q_cat + (n * NQ + h) * ROW_B;

    if (by == 0) {
      if (lane < 31) {
        const float a = __half2float(q_rot_h[lane]);
        const float b = __half2float(q_rot_h[lane + 32]);
        float c, s;
        load_mrope_cs<POS64>(cs_cache, cache_rows, pos_ptr, pos_planes, pos_row_stride, pos_col_stride, n, lane, c, s);
        q_cat_h[R_LAT + lane] = __float2half_rn(a * c - b * s);
        q_cat_h[R_LAT + 31 + lane] = __float2half_rn(b * c + a * s);
      } else {
        q_cat_h[R_LAT + 62] = __float2half_rn(0.f);
        q_cat_h[R_LAT + 63] = __float2half_rn(0.f);
      }
    }

    const int col = by * 64 + 2 * lane;
    float acc0 = 0.f, acc1 = 0.f;
    const __half* w_col = w_k_up_sc + col;
#pragma unroll 4
    for (int k = 0; k < NOPE_DIM; k++) {
      const float qv_k = __half2float(q_nope_h[k]);
      const float2 wf = __half22float2(ldg_h2_safe(w_col + (int64_t)k * R_LAT, w_sc_al4));
      acc0 = fmaf(qv_k, wf.x, acc0);
      acc1 = fmaf(qv_k, wf.y, acc1);
    }
    st_h2_safe(q_cat_h + col, __floats2half2_rn(acc0, acc1), qcat_al4);
  } else {
    __half* c_sm = reinterpret_cast<__half*>(sm_raw);
    __half* kr_sm = reinterpret_cast<__half*>(sm_raw + 512);
    float* wsum = reinterpret_cast<float*>(sm_raw + 640);
    float* inv_rms_sm = reinterpret_cast<float*>(sm_raw + 664);
    kv_prep_and_cache_write<R_LAT, SLOT64, POS64, false>(
        qckr, stride_qckr, pos_ptr, pos_planes, pos_row_stride, pos_col_stride, kw_rope_perm, w_k_up_raw, cs_cache,
        cache_rows, slot_ptr, kc, vc, nullptr, nullptr, n, page_size, max_slots, kb, kt, vb, vt, eps_f, inv_scale,
        k_aligned4, v_aligned4, c_sm, kr_sm, wsum, inv_rms_sm);
  }
}

// =================================================================================================
// Op 6: gla_prep_prefill
// =================================================================================================
template <int R_LAT, bool SLOT64, bool POS64>
__global__ void __launch_bounds__(192)
k_gla_prep_prefill(const __half* __restrict__ qckr, int64_t stride_qckr, const void* __restrict__ pos_ptr,
                   int pos_planes, int64_t pos_row_stride, int64_t pos_col_stride,
                   const float* __restrict__ q_norm_w1p, const float* __restrict__ kw_rope_perm,
                   const __half* __restrict__ w_k_up_raw, const __half* __restrict__ cs_cache, int64_t cache_rows,
                   const void* __restrict__ slot_ptr, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                   __half* __restrict__ q_full, __half* __restrict__ k_rot_out, float* __restrict__ inv_rms_out,
                   int page_size, int64_t max_slots, int64_t kb, int64_t kt, int64_t vb, int64_t vt,
                   float eps_f, float inv_scale, bool qckr_al16, bool qckr_al4, bool qfull_al16, bool qfull_al4,
                   bool k_aligned4, bool v_aligned4) {
  // Shared memory: Q-block needs q_rot[6][64] = 768 B; KV-block needs 668 B
  __shared__ __align__(16) uint8_t sm_raw[NQ * 64 * 2];

  const int64_t n = (int64_t)blockIdx.x;
  if (blockIdx.y == 0) {
    __half* q_rot_base = reinterpret_cast<__half*>(sm_raw);
    const int h = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __half* q_rot_h = q_rot_base + h * 64;

    const int base = lane * 8;
    const uint4 qv = ldg16_h_safe(qckr + n * stride_qckr + h * 512 + base, qckr_al16, qckr_al4);

    auto sq_h2 = [](uint32_t u) -> float {
      const float2 f = __half22float2(u32_to_h2(u));
      return fmaf(f.x, f.x, f.y * f.y);
    };
    float ss = sq_h2(qv.x) + sq_h2(qv.y) + sq_h2(qv.z) + sq_h2(qv.w);
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float inv = rsqrtf(ss * (1.0f / 256.0f) + eps_f);

    auto norm_h2 = [&](uint32_t u, int idx) -> __half2 {
      const float2 f = __half22float2(u32_to_h2(u));
      return __floats2half2_rn(f.x * inv * q_norm_w1p[idx], f.y * inv * q_norm_w1p[idx + 1]);
    };
    const __half2 n0 = norm_h2(qv.x, base + 0);
    const __half2 n1 = norm_h2(qv.y, base + 2);
    const __half2 n2 = norm_h2(qv.z, base + 4);
    const __half2 n3 = norm_h2(qv.w, base + 6);

    __half* q_full_h = q_full + (n * NQ + h) * HD;
    if (lane < 8) {
      __half2* dst = reinterpret_cast<__half2*>(q_rot_h + base);
      dst[0] = n0;
      dst[1] = n1;
      dst[2] = n2;
      dst[3] = n3;
      if (lane == 3) q_full_h[62] = __high2half(n3);   // idx 31 -> q_nope[0]
      if (lane == 7) q_full_h[63] = __high2half(n3);   // idx 63 -> q_nope[1]
    } else {
      const uint4 ov = make_uint4(h2_to_u32(n0), h2_to_u32(n1), h2_to_u32(n2), h2_to_u32(n3));
      st16_h_safe(q_full_h + base, ov, qfull_al16, qfull_al4);
    }
    __syncwarp(0xffffffffu);

    if (lane < 31) {
      const float a = __half2float(q_rot_h[lane]);
      const float b = __half2float(q_rot_h[lane + 32]);
      float c, s;
      load_mrope_cs<POS64>(cs_cache, cache_rows, pos_ptr, pos_planes, pos_row_stride, pos_col_stride, n, lane, c, s);
      q_full_h[lane] = __float2half_rn(a * c - b * s);
      q_full_h[31 + lane] = __float2half_rn(b * c + a * s);
    }
  } else {
    __half* c_sm = reinterpret_cast<__half*>(sm_raw);
    __half* kr_sm = reinterpret_cast<__half*>(sm_raw + 512);
    float* wsum = reinterpret_cast<float*>(sm_raw + 640);
    float* inv_rms_sm = reinterpret_cast<float*>(sm_raw + 664);
    kv_prep_and_cache_write<R_LAT, SLOT64, POS64, true>(
        qckr, stride_qckr, pos_ptr, pos_planes, pos_row_stride, pos_col_stride, kw_rope_perm, w_k_up_raw, cs_cache,
        cache_rows, slot_ptr, kc, vc, k_rot_out, inv_rms_out, n, page_size, max_slots, kb, kt, vb, vt, eps_f, inv_scale,
        k_aligned4, v_aligned4, c_sm, kr_sm, wsum, inv_rms_sm);
  }
}

// =================================================================================================
// Op 7: gla_vup_gate
// =================================================================================================
template <int R_LAT>
__global__ void __launch_bounds__(192)
k_gla_vup_gate(const __half* __restrict__ o_lat, const __half* __restrict__ w_v_up_cm,
               const __half* __restrict__ qckr, int64_t stride_qckr, __half* __restrict__ out,
               bool olat_al16, bool olat_al4, bool wv_al4, bool qckr_al4, bool out_al4) {
  constexpr int O_VEC = R_LAT / 8;   // 24 (192) or 32 (256)
  __shared__ __align__(16) uint8_t sm_raw[NQ * R_LAT * 2];
  __half* o_sm = reinterpret_cast<__half*>(sm_raw);

  const int64_t n = (int64_t)blockIdx.x;
  const int h = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __half* o_sm_h = o_sm + h * R_LAT;

  if (lane < O_VEC) {
    const uint4 v = ldg16_h_safe(o_lat + (n * NQ + h) * R_LAT + lane * 8, olat_al16, olat_al4);
    *reinterpret_cast<uint4*>(o_sm_h + lane * 8) = v;
  }
  __syncwarp(0xffffffffu);

  const int col = (int)blockIdx.y * 64 + 2 * lane;
  float acc0 = 0.f, acc1 = 0.f;
  const __half* w_col = w_v_up_cm + col;
#pragma unroll 4
  for (int j = 0; j < R_LAT; j++) {
    const float ov = __half2float(o_sm_h[j]);
    const float2 wf = __half22float2(ldg_h2_safe(w_col + (int64_t)j * HD, wv_al4));
    acc0 = fmaf(ov, wf.x, acc0);
    acc1 = fmaf(ov, wf.y, acc1);
  }
  const float2 gf = __half22float2(ldg_h2_safe(qckr + n * stride_qckr + h * 512 + 256 + col, qckr_al4));
  const float s0 = 1.0f / (1.0f + expf(-gf.x));
  const float s1 = 1.0f / (1.0f + expf(-gf.y));
  st_h2_safe(out + (n * NQ + h) * HD + col, __floats2half2_rn(acc0 * s0, acc1 * s1), out_al4);
}

// =================================================================================================
// Op 8: gla_gate_only (supports out == attn_tmp in-place!)
// =================================================================================================
__global__ void __launch_bounds__(192)
k_gla_gate_only(const __half* attn_tmp, const __half* __restrict__ qckr, int64_t stride_qckr, __half* out,
                bool a_al16, bool a_al4, bool qckr_al16, bool qckr_al4, bool o_al16, bool o_al4) {
  const int64_t n = (int64_t)blockIdx.x;
  const int h = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int off = lane * 8;

  const uint4 av = ld16_h_coherent_safe(attn_tmp + (n * NQ + h) * HD + off, a_al16, a_al4);
  const uint4 gv = ldg16_h_safe(qckr + n * stride_qckr + h * 512 + 256 + off, qckr_al16, qckr_al4);

  auto gate_h2 = [](uint32_t au, uint32_t gu) -> uint32_t {
    const float2 af = __half22float2(u32_to_h2(au));
    const float2 gf = __half22float2(u32_to_h2(gu));
    const float s0 = 1.0f / (1.0f + expf(-gf.x));
    const float s1 = 1.0f / (1.0f + expf(-gf.y));
    return h2_to_u32(__floats2half2_rn(af.x * s0, af.y * s1));
  };

  uint4 ov;
  ov.x = gate_h2(av.x, gv.x);
  ov.y = gate_h2(av.y, gv.y);
  ov.z = gate_h2(av.z, gv.z);
  ov.w = gate_h2(av.w, gv.w);
  st16_h_safe(out + (n * NQ + h) * HD + off, ov, o_al16, o_al4);
}

}   // namespace

// =================================================================================================
// Public Python / PyTorch API
// =================================================================================================

void init(int64_t dev_idx) {
  int dev = (int)dev_idx;
  if (dev < 0) TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "attn_gla_mq: cudaGetDevice failed");
  ensure_init(dev, at::cuda::getCurrentCUDAStream(dev));
}

int64_t gla_splits(int64_t bh, int64_t S, int64_t R_LAT, int64_t Q, int64_t sep) {
  int dev = 0;
  TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "attn_gla_mq: cudaGetDevice failed");
  return (int64_t)plan_splits_dev(dev, bh, S, R_LAT, Q, sep, at::cuda::getCurrentCUDAStream(dev));
}

std::tuple<int64_t, int64_t> gla_workspace(int64_t bh, int64_t S, int64_t R_LAT, int64_t Q, int64_t sep) {
  const int64_t s = gla_splits(bh, S, R_LAT, Q, sep);
  if (s <= 1 || bh <= 0) return {0, 0};
  return {bh * s * NQ * Q * R_LAT, bh * s * NQ * Q * 2};
}

std::tuple<int64_t, int64_t> gla_smem_and_occ(int64_t R_LAT, int64_t Q, int64_t dev_idx) {
  TORCH_CHECK(R_LAT == 192 || R_LAT == 256, "attn_gla_mq: R_LAT must be 192 or 256");
  TORCH_CHECK(Q >= 1 && Q <= QMAX, "attn_gla_mq: Q must be in 1..8");
  int dev = (int)dev_idx;
  if (dev < 0) TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  ensure_init(dev, at::cuda::getCurrentCUDAStream(dev));
  const int r_idx = (R_LAT == 192) ? 0 : 1;
  const int cfg = cfg_for_q(Q);
  static constexpr int smems[2][4] = {
      {GLA<192, 1, 1>::MAIN, GLA<192, 1, 2>::MAIN, GLA<192, 2, 2>::MAIN, GLA<192, 3, 2>::MAIN},
      {GLA<256, 1, 1>::MAIN, GLA<256, 1, 2>::MAIN, GLA<256, 2, 2>::MAIN, GLA<256, 3, 2>::MAIN},
  };
  return {smems[r_idx][cfg], g_dev_caps[dev].occ[r_idx][cfg]};
}

void gla_cache_write(torch::Tensor c, torch::Tensor k_rope, torch::Tensor inv_rms, torch::Tensor k_cache,
                     torch::Tensor v_cache, torch::Tensor slot_mapping, double kv_scale) {
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "gla_cache_write: caches must be CUDA tensors");
  const auto dev = k_cache.device();
  TORCH_CHECK(v_cache.device() == dev && c.device() == dev && k_rope.device() == dev && inv_rms.device() == dev &&
                  slot_mapping.device() == dev,
              "gla_cache_write: all tensors must be on the same CUDA device");

  TORCH_CHECK(k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8,
              "gla_cache_write: caches must be uint8");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
              "gla_cache_write: k_cache and v_cache must be matching 4D tensors");
  TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "gla_cache_write: caches must have dense stride 1 on dim 3");

  const int64_t num_blocks = k_cache.size(0);
  const int64_t page_size = k_cache.size(1);
  const int64_t Hkv = k_cache.size(2);
  const int64_t HALF_B = k_cache.size(3);
  TORCH_CHECK(HALF_B == 128 || HALF_B == 160, "gla_cache_write: cache head_size must be 128 or 160, got ", HALF_B);
  TORCH_CHECK(page_size >= 1 && Hkv >= 1, "gla_cache_write: page_size and Hkv must be >= 1");
  const int64_t R_LAT = HALF_B * 2 - 64;

  TORCH_CHECK(c.dtype() == torch::kHalf && c.is_contiguous(), "gla_cache_write: c must be contiguous fp16");
  TORCH_CHECK(k_rope.dtype() == torch::kHalf && k_rope.is_contiguous(),
              "gla_cache_write: k_rope must be contiguous fp16");
  TORCH_CHECK((inv_rms.dtype() == torch::kFloat || inv_rms.dtype() == torch::kHalf) && inv_rms.is_contiguous(),
              "gla_cache_write: inv_rms must be contiguous fp32 or fp16");
  TORCH_CHECK((slot_mapping.dtype() == torch::kLong || slot_mapping.dtype() == torch::kInt) &&
                  slot_mapping.is_contiguous() && slot_mapping.dim() == 1,
              "gla_cache_write: slot_mapping must be 1D contiguous int64 or int32");
  TORCH_CHECK(std::isfinite(kv_scale) && kv_scale > 0.0, "gla_cache_write: kv_scale must be finite and > 0");
  const float inv_scale = (float)(1.0 / kv_scale);
  TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f,
              "gla_cache_write: 1.0/kv_scale invalid in fp32 (kv_scale=", kv_scale, ")");

  const int64_t N = slot_mapping.size(0);
  if (c.dim() == 3) {
    TORCH_CHECK(c.size(0) == N && c.size(1) == Hkv && c.size(2) == R_LAT,
                "gla_cache_write: c shape mismatch, expected [", N, ", ", Hkv, ", ", R_LAT, "]");
  } else {
    TORCH_CHECK(c.dim() == 2 && Hkv == 1 && c.size(0) == N && c.size(1) == R_LAT,
                "gla_cache_write: c must be [N, Hkv, R_LAT] (or [N, R_LAT] when Hkv==1)");
  }
  if (k_rope.dim() == 3) {
    TORCH_CHECK(k_rope.size(0) == N && k_rope.size(1) == Hkv && k_rope.size(2) == ROPE_DIM,
                "gla_cache_write: k_rope shape mismatch");
  } else {
    TORCH_CHECK(k_rope.dim() == 2 && Hkv == 1 && k_rope.size(0) == N && k_rope.size(1) == ROPE_DIM,
                "gla_cache_write: k_rope must be [N, Hkv, 62] (or [N, 62] when Hkv==1)");
  }
  if (inv_rms.dim() == 2) {
    TORCH_CHECK(inv_rms.size(0) == N && inv_rms.size(1) == Hkv, "gla_cache_write: inv_rms shape mismatch");
  } else {
    TORCH_CHECK(inv_rms.dim() == 1 && Hkv == 1 && inv_rms.size(0) == N,
                "gla_cache_write: inv_rms must be [N, Hkv] (or [N] when Hkv==1)");
  }

  if (N == 0) return;

  const at::cuda::CUDAGuard guard(dev);
  const auto st = at::cuda::getCurrentCUDAStream();
  const int64_t kb = k_cache.stride(0), kt = k_cache.stride(1), kh = k_cache.stride(2);
  const int64_t vb = v_cache.stride(0), vt = v_cache.stride(1), vh = v_cache.stride(2);
  auto* kc_ptr = k_cache.data_ptr<uint8_t>();
  auto* vc_ptr = v_cache.data_ptr<uint8_t>();
  const bool k_al4 = (((uintptr_t)kc_ptr & 3u) == 0u) && (((kb | kt | kh) & 3ll) == 0ll);
  const bool v_al4 = (((uintptr_t)vc_ptr & 3u) == 0u) && (((vb | vt | vh) & 3ll) == 0ll);
  const int64_t max_slots = num_blocks * page_size;
  const int64_t total_warps = N * Hkv;
  const int grid = (int)((total_warps + 3) / 4);

  const auto* c_ptr = reinterpret_cast<const __half*>(c.data_ptr<at::Half>());
  const auto* kr_ptr = reinterpret_cast<const __half*>(k_rope.data_ptr<at::Half>());
  const void* inv_ptr = inv_rms.data_ptr();
  const void* slot_ptr = slot_mapping.data_ptr();
  const bool slot64 = (slot_mapping.dtype() == torch::kLong);
  const bool inv_fp16 = (inv_rms.dtype() == torch::kHalf);

#define LAUNCH_CW(RL, S64, IF16)                                                                                      \
  k_gla_cache_write<RL, S64, IF16><<<grid, 128, 0, st>>>(c_ptr, kr_ptr, inv_ptr, kc_ptr, vc_ptr, slot_ptr, N,         \
                                                         (int)Hkv, (int)page_size, max_slots, kb, kt, kh, vb, vt, vh, \
                                                         inv_scale, k_al4, v_al4)
  if (R_LAT == 192) {
    if (slot64 && !inv_fp16) LAUNCH_CW(192, true, false);
    else if (slot64 && inv_fp16) LAUNCH_CW(192, true, true);
    else if (!slot64 && !inv_fp16) LAUNCH_CW(192, false, false);
    else LAUNCH_CW(192, false, true);
  } else {
    if (slot64 && !inv_fp16) LAUNCH_CW(256, true, false);
    else if (slot64 && inv_fp16) LAUNCH_CW(256, true, true);
    else if (!slot64 && !inv_fp16) LAUNCH_CW(256, false, false);
    else LAUNCH_CW(256, false, true);
  }
#undef LAUNCH_CW
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gla_cache_write: kernel launch failed");
}

void gla_gather_dequant(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor block_table, torch::Tensor seq_lens,
                        torch::Tensor cu_seq_lens, torch::Tensor out_c, torch::Tensor out_k_rope,
                        torch::Tensor out_inv_rms, double kv_scale) {
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "gla_gather_dequant: caches must be CUDA tensors");
  const auto dev = k_cache.device();
  TORCH_CHECK(v_cache.device() == dev && block_table.device() == dev && seq_lens.device() == dev &&
                  cu_seq_lens.device() == dev && out_c.device() == dev && out_k_rope.device() == dev &&
                  out_inv_rms.device() == dev,
              "gla_gather_dequant: all tensors must be on the same CUDA device");

  TORCH_CHECK(k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8,
              "gla_gather_dequant: caches must be uint8");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
              "gla_gather_dequant: k_cache and v_cache must be matching 4D tensors");
  TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "gla_gather_dequant: caches must have dense stride 1 on dim 3");

  const int64_t num_blocks = k_cache.size(0);
  const int64_t page_size = k_cache.size(1);
  const int64_t Hkv = k_cache.size(2);
  const int64_t HALF_B = k_cache.size(3);
  TORCH_CHECK(HALF_B == 128 || HALF_B == 160, "gla_gather_dequant: cache head_size must be 128 or 160");
  TORCH_CHECK(page_size >= 1 && Hkv >= 1, "gla_gather_dequant: page_size and Hkv must be >= 1");
  const int64_t R_LAT = HALF_B * 2 - 64;

  TORCH_CHECK(block_table.dtype() == torch::kInt && block_table.dim() == 2 && block_table.stride(1) == 1,
              "gla_gather_dequant: block_table must be 2D int32 with stride(1)==1");
  const int64_t B = block_table.size(0);
  TORCH_CHECK(seq_lens.dtype() == torch::kInt && seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
                  seq_lens.size(0) == B,
              "gla_gather_dequant: seq_lens must be [B] contiguous int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt && cu_seq_lens.dim() == 1 && cu_seq_lens.is_contiguous() &&
                  cu_seq_lens.size(0) == B + 1,
              "gla_gather_dequant: cu_seq_lens must be [B+1] contiguous int32");

  TORCH_CHECK(out_c.dtype() == torch::kHalf && out_c.is_contiguous(),
              "gla_gather_dequant: out_c must be contiguous fp16");
  TORCH_CHECK(out_k_rope.dtype() == torch::kHalf && out_k_rope.is_contiguous(),
              "gla_gather_dequant: out_k_rope must be contiguous fp16");
  TORCH_CHECK(out_inv_rms.dtype() == torch::kFloat && out_inv_rms.is_contiguous(),
              "gla_gather_dequant: out_inv_rms must be contiguous fp32");
  TORCH_CHECK(std::isfinite(kv_scale) && kv_scale > 0.0, "gla_gather_dequant: kv_scale must be finite and > 0");
  const float vsc = (float)(kv_scale * 256.0);
  TORCH_CHECK(std::isfinite(vsc) && vsc > 0.f,
              "gla_gather_dequant: kv_scale*256 invalid in fp32 (kv_scale=", kv_scale, ")");

  const int64_t total_kv = out_c.size(0);
  if (out_c.dim() == 3) {
    TORCH_CHECK(out_c.size(1) == Hkv && out_c.size(2) == R_LAT, "gla_gather_dequant: out_c shape mismatch");
  } else {
    TORCH_CHECK(out_c.dim() == 2 && Hkv == 1 && out_c.size(1) == R_LAT,
                "gla_gather_dequant: out_c must be [total_kv, Hkv, R_LAT]");
  }
  if (out_k_rope.dim() == 3) {
    TORCH_CHECK(out_k_rope.size(0) == total_kv && out_k_rope.size(1) == Hkv && out_k_rope.size(2) == ROPE_DIM,
                "gla_gather_dequant: out_k_rope shape mismatch");
  } else {
    TORCH_CHECK(out_k_rope.dim() == 2 && Hkv == 1 && out_k_rope.size(0) == total_kv && out_k_rope.size(1) == ROPE_DIM,
                "gla_gather_dequant: out_k_rope must be [total_kv, Hkv, 62]");
  }
  if (out_inv_rms.dim() == 2) {
    TORCH_CHECK(out_inv_rms.size(0) == total_kv && out_inv_rms.size(1) == Hkv,
                "gla_gather_dequant: out_inv_rms shape mismatch");
  } else {
    TORCH_CHECK(out_inv_rms.dim() == 1 && Hkv == 1 && out_inv_rms.size(0) == total_kv,
                "gla_gather_dequant: out_inv_rms must be [total_kv, Hkv]");
  }

  if (B == 0 || total_kv == 0) return;

  const at::cuda::CUDAGuard guard(dev);
  const auto st = at::cuda::getCurrentCUDAStream();
  const int64_t kb = k_cache.stride(0), kt = k_cache.stride(1), kh = k_cache.stride(2);
  const int64_t vb = v_cache.stride(0), vt = v_cache.stride(1), vh = v_cache.stride(2);
  const auto* kc_ptr = k_cache.data_ptr<uint8_t>();
  const auto* vc_ptr = v_cache.data_ptr<uint8_t>();
  auto* oc_ptr = reinterpret_cast<__half*>(out_c.data_ptr<at::Half>());
  auto* okr_ptr = reinterpret_cast<__half*>(out_k_rope.data_ptr<at::Half>());
  const bool k_al4 = (((uintptr_t)kc_ptr & 3u) == 0u) && (((kb | kt | kh) & 3ll) == 0ll);
  const bool v_al4 = (((uintptr_t)vc_ptr & 3u) == 0u) && (((vb | vt | vh) & 3ll) == 0ll);
  const bool oc_al4 = (((uintptr_t)oc_ptr & 3u) == 0u);
  const bool okr_al4 = (((uintptr_t)okr_ptr & 3u) == 0u);
  const int grid = (int)((total_kv * Hkv + 3) / 4);

  if (R_LAT == 192) {
    k_gla_gather_dequant<192><<<grid, 128, 0, st>>>(
        kc_ptr, vc_ptr, block_table.data_ptr<int>(), (int)block_table.stride(0), (int)block_table.size(1),
        seq_lens.data_ptr<int>(), cu_seq_lens.data_ptr<int>(), oc_ptr, okr_ptr, out_inv_rms.data_ptr<float>(), (int)B,
        total_kv, (int)Hkv, (int)page_size, (int)num_blocks, kb, kt, kh, vb, vt, vh, vsc, k_al4, v_al4, oc_al4,
        okr_al4);
  } else {
    k_gla_gather_dequant<256><<<grid, 128, 0, st>>>(
        kc_ptr, vc_ptr, block_table.data_ptr<int>(), (int)block_table.stride(0), (int)block_table.size(1),
        seq_lens.data_ptr<int>(), cu_seq_lens.data_ptr<int>(), oc_ptr, okr_ptr, out_inv_rms.data_ptr<float>(), (int)B,
        total_kv, (int)Hkv, (int)page_size, (int)num_blocks, kb, kt, kh, vb, vt, vh, vsc, k_al4, v_al4, oc_al4,
        okr_al4);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gla_gather_dequant: kernel launch failed");
}

void attn_gla_mq(torch::Tensor q_cat, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor block_table,
                 torch::Tensor seq_lens, torch::Tensor out_lat, torch::Tensor po, torch::Tensor pml, torch::Tensor cnt,
                 double scale, double kv_scale, int64_t Q, int64_t S, int64_t min_tiles, int64_t sep) {
  TORCH_CHECK(q_cat.is_cuda(), "attn_gla_mq: q_cat must be a CUDA tensor");
  const auto dev = q_cat.device();
  TORCH_CHECK(k_cache.device() == dev && v_cache.device() == dev && block_table.device() == dev &&
                  seq_lens.device() == dev && out_lat.device() == dev,
              "attn_gla_mq: all input/output tensors must be on the same CUDA device");

  TORCH_CHECK(k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8 && k_cache.dim() == 4 &&
                  v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
              "attn_gla_mq: k_cache and v_cache must be matching 4D uint8 tensors");
  TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "attn_gla_mq: caches must have dense stride 1 on dim 3");
  const int64_t page_size = k_cache.size(1);
  const int64_t Hkv = k_cache.size(2);
  const int64_t HALF_B = k_cache.size(3);
  TORCH_CHECK(HALF_B == 128 || HALF_B == 160, "attn_gla_mq: cache head_size must be 128 or 160, got ", HALF_B);
  TORCH_CHECK(page_size >= 1 && Hkv >= 1 && Hkv <= 65535, "attn_gla_mq: invalid page_size or Hkv");
  const int64_t ROW_B = HALF_B * 2;
  const int64_t R_LAT = ROW_B - 64;

  TORCH_CHECK(Q >= 1 && Q <= QMAX, "attn_gla_mq: Q must be in 1..8, got ", Q);
  TORCH_CHECK(q_cat.dtype() == torch::kHalf && q_cat.is_contiguous() && q_cat.dim() == 3 && q_cat.size(2) == ROW_B,
              "attn_gla_mq: q_cat must be [B*Q, 6*Hkv, ROW_B] contiguous fp16");
  TORCH_CHECK(q_cat.size(1) == NQ * Hkv, "attn_gla_mq: expected 6 query heads per KV head");
  TORCH_CHECK(block_table.dtype() == torch::kInt && block_table.dim() == 2 && block_table.stride(1) == 1,
              "attn_gla_mq: block_table must be [B, max_blocks] int32 with stride(1)==1");
  const int64_t B = block_table.size(0);
  TORCH_CHECK(B <= 65535 && B * Q == q_cat.size(0), "attn_gla_mq: block_table.size(0) * Q must equal q_cat.size(0)");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt && seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
                  seq_lens.size(0) == q_cat.size(0),
              "attn_gla_mq: seq_lens must be [B*Q] contiguous int32");
  TORCH_CHECK(out_lat.dtype() == torch::kHalf && out_lat.is_contiguous() && out_lat.dim() == 3 &&
                  out_lat.size(0) == q_cat.size(0) && out_lat.size(1) == q_cat.size(1) && out_lat.size(2) == R_LAT,
              "attn_gla_mq: out_lat must be [B*Q, 6*Hkv, R_LAT] contiguous fp16");
  TORCH_CHECK(min_tiles >= 1, "attn_gla_mq: min_tiles must be >= 1");
  TORCH_CHECK(sep == 0 || sep == 1, "attn_gla_mq: sep must be 0 or 1");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(kv_scale) && kv_scale > 0.0,
              "attn_gla_mq: scale and kv_scale must be finite with kv_scale > 0");
  const float sl2 = (float)(scale * kv_scale * 256.0 * LOG2E);
  const float vsc = (float)(kv_scale * 256.0);
  TORCH_CHECK(std::isfinite(sl2) && std::isfinite(vsc) && vsc > 0.f,
              "attn_gla_mq: scale/kv_scale intermediates invalid in fp32");

  if (B == 0) return;

  TORCH_CHECK(!ranges_overlap(q_cat.data_ptr(), (size_t)q_cat.numel() * sizeof(at::Half), out_lat.data_ptr(),
                              (size_t)out_lat.numel() * sizeof(at::Half)),
              "attn_gla_mq: out_lat must not alias q_cat");

  const at::cuda::CUDAGuard guard(dev);
  int dev_idx = dev.index();
  if (dev_idx < 0) TORCH_CHECK(cudaGetDevice(&dev_idx) == cudaSuccess, "attn_gla_mq: cudaGetDevice failed");
  const auto st = at::cuda::getCurrentCUDAStream();
  ensure_init(dev_idx, st);

  const int64_t bh = B * Hkv;
  const int S_eff = plan_splits_dev(dev_idx, bh, S, R_LAT, Q, sep, st);
  if (S_eff > 1) {
    const int64_t need_po = bh * (int64_t)S_eff * NQ * Q * R_LAT;
    const int64_t need_pml = bh * (int64_t)S_eff * NQ * Q * 2;
    TORCH_CHECK(po.device() == dev && po.dtype() == torch::kFloat && po.is_contiguous() && po.numel() >= need_po,
                "attn_gla_mq: po workspace invalid or too small (need ", need_po, " floats)");
    TORCH_CHECK(((uintptr_t)po.data_ptr<float>() & 15u) == 0u, "attn_gla_mq: po pointer must be 16-byte aligned");
    TORCH_CHECK(pml.device() == dev && pml.dtype() == torch::kFloat && pml.is_contiguous() && pml.numel() >= need_pml,
                "attn_gla_mq: pml workspace invalid or too small (need ", need_pml, " floats)");
    if (sep == 0) {
      TORCH_CHECK(cnt.device() == dev && cnt.dtype() == torch::kInt && cnt.is_contiguous() && cnt.numel() >= bh,
                  "attn_gla_mq: cnt buffer invalid or too small for sep==0 (need ", bh, " int32s)");
    }
  }

  const int cfg = cfg_for_q(Q);

  if (R_LAT == 192) {
    dispatch_cfg<192>(cfg, q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, (int)Q, S_eff,
                      (int)min_tiles, (int)sep, sl2, vsc, st);
  } else {
    dispatch_cfg<256>(cfg, q_cat, k_cache, v_cache, block_table, seq_lens, out_lat, po, pml, cnt, (int)Q, S_eff,
                      (int)min_tiles, (int)sep, sl2, vsc, st);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "attn_gla_mq: kernel launch failed");
}

static bool is_n_6_256_fp16(const torch::Tensor& t, int64_t N) {
  if (t.dtype() != torch::kHalf || !t.is_contiguous()) return false;
  if (t.dim() == 3) return t.size(0) == N && t.size(1) == NQ && t.size(2) == HD;
  if (t.dim() == 2) return t.size(0) == N && t.size(1) == NQ * HD;
  return false;
}

// Precision & caller contracts for fused ops (Follow-up #1):
//   - Precision: internal FP32 accumulation is used for k_nope_raw RMS, M-RoPE rotation, q_abs / v_up GEMVs,
//     and sigmoid gating before rounding outputs to FP16 (intentionally higher precision than intermediate FP16
//     PyTorch ops, avoiding intermediate FP16 rounding / overflow).
//   - slot_mapping: valid in-range slots (0 <= slot < num_blocks * page_size) across n in 0..N-1 MUST be unique
//     (matching gla_cache_write and vLLM reshape_and_cache_flash); negative (<0) and out-of-range (>=max_slots)
//     padding slots may repeat freely and are safely skipped without writing cache memory.
//   - positions: negative indices wrap once (+cache_rows), then clamp safely to [0, cache_rows - 1].

void gla_prep_route_a(torch::Tensor qckr, torch::Tensor positions, torch::Tensor q_norm_w1p,
                      torch::Tensor kw_rope_perm, torch::Tensor w_k_up_raw_cm, torch::Tensor w_k_up_sc_cm,
                      torch::Tensor cos_sin_cache, torch::Tensor slot_mapping, torch::Tensor k_cache,
                      torch::Tensor v_cache, torch::Tensor q_cat, double eps, double kv_scale) {
  TORCH_CHECK(qckr.is_cuda(), "gla_prep_route_a: qckr must be a CUDA tensor");
  const auto dev = qckr.device();
  TORCH_CHECK(positions.device() == dev && q_norm_w1p.device() == dev && kw_rope_perm.device() == dev &&
                  w_k_up_raw_cm.device() == dev && w_k_up_sc_cm.device() == dev && cos_sin_cache.device() == dev &&
                  slot_mapping.device() == dev && k_cache.device() == dev && v_cache.device() == dev &&
                  q_cat.device() == dev,
              "gla_prep_route_a: all tensors must be on the same CUDA device");

  TORCH_CHECK(k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8 && k_cache.dim() == 4 &&
                  v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
              "gla_prep_route_a: k_cache and v_cache must be matching 4D uint8 tensors");
  TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "gla_prep_route_a: caches must have dense stride 1 on dim 3");
  const int64_t num_blocks = k_cache.size(0);
  const int64_t page_size = k_cache.size(1);
  const int64_t Hkv = k_cache.size(2);
  const int64_t HALF_B = k_cache.size(3);
  TORCH_CHECK(HALF_B == 128 || HALF_B == 160, "gla_prep_route_a: cache head_size must be 128 or 160, got ", HALF_B);
  TORCH_CHECK(page_size >= 1 && page_size <= INT_MAX && Hkv == 1,
              "gla_prep_route_a: require page_size in [1, INT_MAX] and Hkv == 1");
  validate_kv_caches_disjoint(k_cache, v_cache, "gla_prep_route_a");
  const int64_t ROW_B = HALF_B * 2;
  const int64_t R_LAT = ROW_B - 64;

  TORCH_CHECK(qckr.dtype() == torch::kHalf && qckr.dim() == 2 && qckr.stride(1) == 1,
              "gla_prep_route_a: qckr must be 2D fp16 with stride(1)==1");
  const int64_t N = qckr.size(0);
  const int64_t stride_qckr = qckr.stride(0);
  TORCH_CHECK(N <= INT_MAX, "gla_prep_route_a: N exceeds INT_MAX");
  TORCH_CHECK(qckr.size(1) >= 3072 + R_LAT + ROPE_DIM && (N <= 1 || stride_qckr >= qckr.size(1)),
              "gla_prep_route_a: qckr.size(1) must be >= ", 3072 + R_LAT + ROPE_DIM, " with non-overlapping rows");

  TORCH_CHECK(positions.dtype() == torch::kLong || positions.dtype() == torch::kInt,
              "gla_prep_route_a: positions must be int64 or int32");
  int pos_planes = 1;
  int64_t pos_row_stride = 0, pos_col_stride = 0;
  if (positions.dim() == 1) {
    TORCH_CHECK(positions.size(0) == N, "gla_prep_route_a: 1D positions must have length N");
    pos_planes = 1;
    pos_row_stride = 0;
    pos_col_stride = positions.stride(0);
  } else if (positions.dim() == 2) {
    TORCH_CHECK(positions.size(0) == 3 && positions.size(1) == N,
                "gla_prep_route_a: 2D positions must have shape [3, N]");
    pos_planes = 3;
    pos_row_stride = positions.stride(0);
    pos_col_stride = positions.stride(1);
  } else {
    TORCH_CHECK(false, "gla_prep_route_a: positions must be 1D [N] or 2D [3, N]");
  }

  TORCH_CHECK(q_norm_w1p.dtype() == torch::kFloat && q_norm_w1p.is_contiguous() && q_norm_w1p.dim() == 1 &&
                  q_norm_w1p.size(0) == HD,
              "gla_prep_route_a: q_norm_w1p must be [256] contiguous fp32");
  TORCH_CHECK(kw_rope_perm.dtype() == torch::kFloat && kw_rope_perm.is_contiguous() && kw_rope_perm.dim() == 1 &&
                  kw_rope_perm.size(0) == ROPE_DIM,
              "gla_prep_route_a: kw_rope_perm must be [62] contiguous fp32");
  TORCH_CHECK(w_k_up_raw_cm.dtype() == torch::kHalf && w_k_up_raw_cm.is_contiguous() && w_k_up_raw_cm.dim() == 2 &&
                  w_k_up_raw_cm.size(0) == R_LAT && w_k_up_raw_cm.size(1) == NOPE_DIM,
              "gla_prep_route_a: w_k_up_raw_cm must be [R_LAT, 194] contiguous fp16");
  TORCH_CHECK(w_k_up_sc_cm.dtype() == torch::kHalf && w_k_up_sc_cm.is_contiguous() && w_k_up_sc_cm.dim() == 2 &&
                  w_k_up_sc_cm.size(0) == NOPE_DIM && w_k_up_sc_cm.size(1) == R_LAT,
              "gla_prep_route_a: w_k_up_sc_cm must be [194, R_LAT] contiguous fp16");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kHalf && cos_sin_cache.is_contiguous() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(0) >= 1 && cos_sin_cache.size(1) == 64,
              "gla_prep_route_a: cos_sin_cache must be [max_pos>=1, 64] contiguous fp16");
  TORCH_CHECK((slot_mapping.dtype() == torch::kLong || slot_mapping.dtype() == torch::kInt) &&
                  slot_mapping.is_contiguous() && slot_mapping.dim() == 1 && slot_mapping.size(0) == N,
              "gla_prep_route_a: slot_mapping must be [N] contiguous int64 or int32");
  TORCH_CHECK(q_cat.dtype() == torch::kHalf && q_cat.is_contiguous() && q_cat.dim() == 3 && q_cat.size(0) == N &&
                  q_cat.size(1) == NQ && q_cat.size(2) == ROW_B,
              "gla_prep_route_a: q_cat must be [N, 6, ROW_B] contiguous fp16");

  TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "gla_prep_route_a: eps must be finite and >= 0");
  const float eps_f = (float)eps;
  TORCH_CHECK(std::isfinite(eps_f) && eps_f >= 0.f, "gla_prep_route_a: eps invalid in fp32");
  TORCH_CHECK(std::isfinite(kv_scale) && kv_scale > 0.0, "gla_prep_route_a: kv_scale must be finite and > 0");
  const float inv_scale = (float)(1.0 / kv_scale);
  TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f, "gla_prep_route_a: 1.0/kv_scale invalid in fp32");

  if (N == 0) return;

  // Check every writable region (q_cat, k_cache, v_cache) against all inputs and against each other
  const ByteRange in_ranges[8] = {
      tensor_byte_range(qckr),          tensor_byte_range(positions),     tensor_byte_range(q_norm_w1p),
      tensor_byte_range(kw_rope_perm),  tensor_byte_range(w_k_up_raw_cm), tensor_byte_range(w_k_up_sc_cm),
      tensor_byte_range(cos_sin_cache), tensor_byte_range(slot_mapping),
  };
  const ByteRange qc_r = tensor_byte_range(q_cat);
  const ByteRange kc_r = tensor_byte_range(k_cache);
  const ByteRange vc_r = tensor_byte_range(v_cache);
  TORCH_CHECK(!byte_ranges_overlap(qc_r, kc_r) && !byte_ranges_overlap(qc_r, vc_r),
              "gla_prep_route_a: q_cat must not alias k_cache or v_cache");
  for (int i = 0; i < 8; i++) {
    TORCH_CHECK(!byte_ranges_overlap(qc_r, in_ranges[i]) && !byte_ranges_overlap(kc_r, in_ranges[i]) &&
                    !byte_ranges_overlap(vc_r, in_ranges[i]),
                "gla_prep_route_a: writable outputs (q_cat/k_cache/v_cache) must not alias input tensors");
  }

  const at::cuda::CUDAGuard guard(dev);
  const auto st = at::cuda::getCurrentCUDAStream();

  const auto* qckr_ptr = reinterpret_cast<const __half*>(qckr.data_ptr<at::Half>());
  const auto* w_raw_ptr = reinterpret_cast<const __half*>(w_k_up_raw_cm.data_ptr<at::Half>());
  const auto* w_sc_ptr = reinterpret_cast<const __half*>(w_k_up_sc_cm.data_ptr<at::Half>());
  const auto* cs_ptr = reinterpret_cast<const __half*>(cos_sin_cache.data_ptr<at::Half>());
  auto* qcat_ptr = reinterpret_cast<__half*>(q_cat.data_ptr<at::Half>());
  auto* kc_ptr = k_cache.data_ptr<uint8_t>();
  auto* vc_ptr = v_cache.data_ptr<uint8_t>();

  const int64_t kb = k_cache.stride(0), kt = k_cache.stride(1);
  const int64_t vb = v_cache.stride(0), vt = v_cache.stride(1);
  const int64_t max_slots = num_blocks * page_size;
  const int64_t cache_rows = cos_sin_cache.size(0);

  const bool qckr_al16 = (((uintptr_t)qckr_ptr & 15u) == 0u) && (N <= 1 || (stride_qckr & 7ll) == 0ll);
  const bool qckr_al4 = (((uintptr_t)qckr_ptr & 3u) == 0u) && (N <= 1 || (stride_qckr & 1ll) == 0ll);
  const bool w_sc_al4 = (((uintptr_t)w_sc_ptr & 3u) == 0u);
  const bool qcat_al4 = (((uintptr_t)qcat_ptr & 3u) == 0u);
  const bool k_al4 = (((uintptr_t)kc_ptr & 3u) == 0u) && (((kb | kt) & 3ll) == 0ll);
  const bool v_al4 = (((uintptr_t)vc_ptr & 3u) == 0u) && (((vb | vt) & 3ll) == 0ll);

  const bool slot64 = (slot_mapping.dtype() == torch::kLong);
  const bool pos64 = (positions.dtype() == torch::kLong);
  const dim3 grid((unsigned int)N, (unsigned int)(R_LAT / 64 + 1));

#define LAUNCH_PRA(RL, S64, P64)                                                                                   \
  k_gla_prep_route_a<RL, S64, P64><<<grid, 192, 0, st>>>(                                                          \
      qckr_ptr, stride_qckr, positions.data_ptr(), pos_planes, pos_row_stride, pos_col_stride,                     \
      q_norm_w1p.data_ptr<float>(), kw_rope_perm.data_ptr<float>(), w_raw_ptr, w_sc_ptr, cs_ptr, cache_rows,       \
      slot_mapping.data_ptr(), kc_ptr, vc_ptr, qcat_ptr, (int)page_size, max_slots, kb, kt, vb, vt, eps_f,         \
      inv_scale, qckr_al16, qckr_al4, w_sc_al4, qcat_al4, k_al4, v_al4)

  if (R_LAT == 192) {
    if (slot64 && pos64) LAUNCH_PRA(192, true, true);
    else if (slot64 && !pos64) LAUNCH_PRA(192, true, false);
    else if (!slot64 && pos64) LAUNCH_PRA(192, false, true);
    else LAUNCH_PRA(192, false, false);
  } else {
    if (slot64 && pos64) LAUNCH_PRA(256, true, true);
    else if (slot64 && !pos64) LAUNCH_PRA(256, true, false);
    else if (!slot64 && pos64) LAUNCH_PRA(256, false, true);
    else LAUNCH_PRA(256, false, false);
  }
#undef LAUNCH_PRA
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gla_prep_route_a: kernel launch failed");
}

void gla_prep_prefill(torch::Tensor qckr, torch::Tensor positions, torch::Tensor q_norm_w1p,
                      torch::Tensor kw_rope_perm, torch::Tensor w_k_up_raw_cm, torch::Tensor cos_sin_cache,
                      torch::Tensor slot_mapping, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor q_full,
                      torch::Tensor k_rot_out, torch::Tensor inv_rms_out, double eps, double kv_scale) {
  TORCH_CHECK(qckr.is_cuda(), "gla_prep_prefill: qckr must be a CUDA tensor");
  const auto dev = qckr.device();
  TORCH_CHECK(positions.device() == dev && q_norm_w1p.device() == dev && kw_rope_perm.device() == dev &&
                  w_k_up_raw_cm.device() == dev && cos_sin_cache.device() == dev && slot_mapping.device() == dev &&
                  k_cache.device() == dev && v_cache.device() == dev && q_full.device() == dev &&
                  k_rot_out.device() == dev && inv_rms_out.device() == dev,
              "gla_prep_prefill: all tensors must be on the same CUDA device");

  TORCH_CHECK(k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8 && k_cache.dim() == 4 &&
                  v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
              "gla_prep_prefill: k_cache and v_cache must be matching 4D uint8 tensors");
  TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "gla_prep_prefill: caches must have dense stride 1 on dim 3");
  const int64_t num_blocks = k_cache.size(0);
  const int64_t page_size = k_cache.size(1);
  const int64_t Hkv = k_cache.size(2);
  const int64_t HALF_B = k_cache.size(3);
  TORCH_CHECK(HALF_B == 128 || HALF_B == 160, "gla_prep_prefill: cache head_size must be 128 or 160, got ", HALF_B);
  TORCH_CHECK(page_size >= 1 && page_size <= INT_MAX && Hkv == 1,
              "gla_prep_prefill: require page_size in [1, INT_MAX] and Hkv == 1");
  validate_kv_caches_disjoint(k_cache, v_cache, "gla_prep_prefill");
  const int64_t R_LAT = HALF_B * 2 - 64;

  TORCH_CHECK(qckr.dtype() == torch::kHalf && qckr.dim() == 2 && qckr.stride(1) == 1,
              "gla_prep_prefill: qckr must be 2D fp16 with stride(1)==1");
  const int64_t N = qckr.size(0);
  const int64_t stride_qckr = qckr.stride(0);
  TORCH_CHECK(N <= INT_MAX, "gla_prep_prefill: N exceeds INT_MAX");
  TORCH_CHECK(qckr.size(1) >= 3072 + R_LAT + ROPE_DIM && (N <= 1 || stride_qckr >= qckr.size(1)),
              "gla_prep_prefill: qckr.size(1) must be >= ", 3072 + R_LAT + ROPE_DIM, " with non-overlapping rows");

  TORCH_CHECK(positions.dtype() == torch::kLong || positions.dtype() == torch::kInt,
              "gla_prep_prefill: positions must be int64 or int32");
  int pos_planes = 1;
  int64_t pos_row_stride = 0, pos_col_stride = 0;
  if (positions.dim() == 1) {
    TORCH_CHECK(positions.size(0) == N, "gla_prep_prefill: 1D positions must have length N");
    pos_planes = 1;
    pos_row_stride = 0;
    pos_col_stride = positions.stride(0);
  } else if (positions.dim() == 2) {
    TORCH_CHECK(positions.size(0) == 3 && positions.size(1) == N,
                "gla_prep_prefill: 2D positions must have shape [3, N]");
    pos_planes = 3;
    pos_row_stride = positions.stride(0);
    pos_col_stride = positions.stride(1);
  } else {
    TORCH_CHECK(false, "gla_prep_prefill: positions must be 1D [N] or 2D [3, N]");
  }

  TORCH_CHECK(q_norm_w1p.dtype() == torch::kFloat && q_norm_w1p.is_contiguous() && q_norm_w1p.dim() == 1 &&
                  q_norm_w1p.size(0) == HD,
              "gla_prep_prefill: q_norm_w1p must be [256] contiguous fp32");
  TORCH_CHECK(kw_rope_perm.dtype() == torch::kFloat && kw_rope_perm.is_contiguous() && kw_rope_perm.dim() == 1 &&
                  kw_rope_perm.size(0) == ROPE_DIM,
              "gla_prep_prefill: kw_rope_perm must be [62] contiguous fp32");
  TORCH_CHECK(w_k_up_raw_cm.dtype() == torch::kHalf && w_k_up_raw_cm.is_contiguous() && w_k_up_raw_cm.dim() == 2 &&
                  w_k_up_raw_cm.size(0) == R_LAT && w_k_up_raw_cm.size(1) == NOPE_DIM,
              "gla_prep_prefill: w_k_up_raw_cm must be [R_LAT, 194] contiguous fp16");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kHalf && cos_sin_cache.is_contiguous() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(0) >= 1 && cos_sin_cache.size(1) == 64,
              "gla_prep_prefill: cos_sin_cache must be [max_pos>=1, 64] contiguous fp16");
  TORCH_CHECK((slot_mapping.dtype() == torch::kLong || slot_mapping.dtype() == torch::kInt) &&
                  slot_mapping.is_contiguous() && slot_mapping.dim() == 1 && slot_mapping.size(0) == N,
              "gla_prep_prefill: slot_mapping must be [N] contiguous int64 or int32");
  TORCH_CHECK(is_n_6_256_fp16(q_full, N), "gla_prep_prefill: q_full must be [N, 6, 256] or [N, 1536] contiguous fp16");
  TORCH_CHECK(k_rot_out.dtype() == torch::kHalf && k_rot_out.is_contiguous() &&
                  ((k_rot_out.dim() == 2 && k_rot_out.size(0) == N && k_rot_out.size(1) == ROPE_DIM) ||
                   (k_rot_out.dim() == 3 && k_rot_out.size(0) == N && k_rot_out.size(1) == 1 &&
                    k_rot_out.size(2) == ROPE_DIM)),
              "gla_prep_prefill: k_rot_out must be [N, 62] (or [N, 1, 62]) contiguous fp16");
  TORCH_CHECK(inv_rms_out.dtype() == torch::kFloat && inv_rms_out.is_contiguous() &&
                  ((inv_rms_out.dim() == 1 && inv_rms_out.size(0) == N) ||
                   (inv_rms_out.dim() == 2 && inv_rms_out.size(0) == N && inv_rms_out.size(1) == 1)),
              "gla_prep_prefill: inv_rms_out must be [N] (or [N, 1]) contiguous fp32");

  TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "gla_prep_prefill: eps must be finite and >= 0");
  const float eps_f = (float)eps;
  TORCH_CHECK(std::isfinite(eps_f) && eps_f >= 0.f, "gla_prep_prefill: eps invalid in fp32");
  TORCH_CHECK(std::isfinite(kv_scale) && kv_scale > 0.0, "gla_prep_prefill: kv_scale must be finite and > 0");
  const float inv_scale = (float)(1.0 / kv_scale);
  TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0.f, "gla_prep_prefill: 1.0/kv_scale invalid in fp32");

  if (N == 0) return;

  // Check every writable region (q_full, k_rot_out, inv_rms_out, k_cache, v_cache) against all inputs and each other
  const ByteRange in_ranges[7] = {
      tensor_byte_range(qckr),          tensor_byte_range(positions),    tensor_byte_range(q_norm_w1p),
      tensor_byte_range(kw_rope_perm),  tensor_byte_range(w_k_up_raw_cm), tensor_byte_range(cos_sin_cache),
      tensor_byte_range(slot_mapping),
  };
  const ByteRange qf_r = tensor_byte_range(q_full);
  const ByteRange kr_r = tensor_byte_range(k_rot_out);
  const ByteRange ir_r = tensor_byte_range(inv_rms_out);
  const ByteRange kc_r = tensor_byte_range(k_cache);
  const ByteRange vc_r = tensor_byte_range(v_cache);
  const ByteRange out_ranges[5] = {qf_r, kr_r, ir_r, kc_r, vc_r};

  // Pairwise disjointness among outputs (k_cache vs v_cache already validated by validate_kv_caches_disjoint)
  TORCH_CHECK(!byte_ranges_overlap(qf_r, kr_r) && !byte_ranges_overlap(qf_r, ir_r) &&
                  !byte_ranges_overlap(kr_r, ir_r) && !byte_ranges_overlap(qf_r, kc_r) &&
                  !byte_ranges_overlap(qf_r, vc_r) && !byte_ranges_overlap(kr_r, kc_r) &&
                  !byte_ranges_overlap(kr_r, vc_r) && !byte_ranges_overlap(ir_r, kc_r) &&
                  !byte_ranges_overlap(ir_r, vc_r),
              "gla_prep_prefill: output tensors (including caches) must not alias each other");
  for (int o = 0; o < 5; o++) {
    for (int i = 0; i < 7; i++) {
      TORCH_CHECK(!byte_ranges_overlap(out_ranges[o], in_ranges[i]),
                  "gla_prep_prefill: writable outputs (including caches) must not alias input tensors");
    }
  }

  const at::cuda::CUDAGuard guard(dev);
  const auto st = at::cuda::getCurrentCUDAStream();

  const auto* qckr_ptr = reinterpret_cast<const __half*>(qckr.data_ptr<at::Half>());
  const auto* w_raw_ptr = reinterpret_cast<const __half*>(w_k_up_raw_cm.data_ptr<at::Half>());
  const auto* cs_ptr = reinterpret_cast<const __half*>(cos_sin_cache.data_ptr<at::Half>());
  auto* qf_ptr = reinterpret_cast<__half*>(q_full.data_ptr<at::Half>());
  auto* kr_ptr = reinterpret_cast<__half*>(k_rot_out.data_ptr<at::Half>());
  auto* ir_ptr = inv_rms_out.data_ptr<float>();
  auto* kc_ptr = k_cache.data_ptr<uint8_t>();
  auto* vc_ptr = v_cache.data_ptr<uint8_t>();

  const int64_t kb = k_cache.stride(0), kt = k_cache.stride(1);
  const int64_t vb = v_cache.stride(0), vt = v_cache.stride(1);
  const int64_t max_slots = num_blocks * page_size;
  const int64_t cache_rows = cos_sin_cache.size(0);

  const bool qckr_al16 = (((uintptr_t)qckr_ptr & 15u) == 0u) && (N <= 1 || (stride_qckr & 7ll) == 0ll);
  const bool qckr_al4 = (((uintptr_t)qckr_ptr & 3u) == 0u) && (N <= 1 || (stride_qckr & 1ll) == 0ll);
  const bool qf_al16 = (((uintptr_t)qf_ptr & 15u) == 0u);
  const bool qf_al4 = (((uintptr_t)qf_ptr & 3u) == 0u);
  const bool k_al4 = (((uintptr_t)kc_ptr & 3u) == 0u) && (((kb | kt) & 3ll) == 0ll);
  const bool v_al4 = (((uintptr_t)vc_ptr & 3u) == 0u) && (((vb | vt) & 3ll) == 0ll);

  const bool slot64 = (slot_mapping.dtype() == torch::kLong);
  const bool pos64 = (positions.dtype() == torch::kLong);
  const dim3 grid((unsigned int)N, 2u);

#define LAUNCH_PPF(RL, S64, P64)                                                                                   \
  k_gla_prep_prefill<RL, S64, P64><<<grid, 192, 0, st>>>(                                                          \
      qckr_ptr, stride_qckr, positions.data_ptr(), pos_planes, pos_row_stride, pos_col_stride,                     \
      q_norm_w1p.data_ptr<float>(), kw_rope_perm.data_ptr<float>(), w_raw_ptr, cs_ptr, cache_rows,                 \
      slot_mapping.data_ptr(), kc_ptr, vc_ptr, qf_ptr, kr_ptr, ir_ptr, (int)page_size, max_slots, kb, kt, vb, vt,  \
      eps_f, inv_scale, qckr_al16, qckr_al4, qf_al16, qf_al4, k_al4, v_al4)

  if (R_LAT == 192) {
    if (slot64 && pos64) LAUNCH_PPF(192, true, true);
    else if (slot64 && !pos64) LAUNCH_PPF(192, true, false);
    else if (!slot64 && pos64) LAUNCH_PPF(192, false, true);
    else LAUNCH_PPF(192, false, false);
  } else {
    if (slot64 && pos64) LAUNCH_PPF(256, true, true);
    else if (slot64 && !pos64) LAUNCH_PPF(256, true, false);
    else if (!slot64 && pos64) LAUNCH_PPF(256, false, true);
    else LAUNCH_PPF(256, false, false);
  }
#undef LAUNCH_PPF
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gla_prep_prefill: kernel launch failed");
}

void gla_vup_gate(torch::Tensor o_lat, torch::Tensor w_v_up_cm, torch::Tensor qckr, torch::Tensor out) {
  TORCH_CHECK(o_lat.is_cuda(), "gla_vup_gate: o_lat must be a CUDA tensor");
  const auto dev = o_lat.device();
  TORCH_CHECK(w_v_up_cm.device() == dev && qckr.device() == dev && out.device() == dev,
              "gla_vup_gate: all tensors must be on the same CUDA device");

  TORCH_CHECK(o_lat.dtype() == torch::kHalf && o_lat.is_contiguous() && o_lat.dim() == 3 && o_lat.size(1) == NQ &&
                  (o_lat.size(2) == 192 || o_lat.size(2) == 256),
              "gla_vup_gate: o_lat must be [N, 6, 192|256] contiguous fp16");
  const int64_t N = o_lat.size(0);
  const int64_t R_LAT = o_lat.size(2);
  TORCH_CHECK(N <= INT_MAX, "gla_vup_gate: N exceeds INT_MAX");

  TORCH_CHECK(w_v_up_cm.dtype() == torch::kHalf && w_v_up_cm.is_contiguous() && w_v_up_cm.dim() == 2 &&
                  w_v_up_cm.size(0) == R_LAT && w_v_up_cm.size(1) == HD,
              "gla_vup_gate: w_v_up_cm must be [R_LAT, 256] contiguous fp16");
  TORCH_CHECK(qckr.dtype() == torch::kHalf && qckr.dim() == 2 && qckr.stride(1) == 1 && qckr.size(0) == N &&
                  qckr.size(1) >= 3072 && (N <= 1 || qckr.stride(0) >= qckr.size(1)),
              "gla_vup_gate: qckr must be [N, >=3072] fp16 with stride(1)==1");
  TORCH_CHECK(is_n_6_256_fp16(out, N), "gla_vup_gate: out must be [N, 6, 256] or [N, 1536] contiguous fp16");

  if (N == 0) return;

  const ByteRange out_r = tensor_byte_range(out);
  TORCH_CHECK(!byte_ranges_overlap(out_r, tensor_byte_range(o_lat)) &&
                  !byte_ranges_overlap(out_r, tensor_byte_range(w_v_up_cm)) &&
                  !byte_ranges_overlap(out_r, tensor_byte_range(qckr)),
              "gla_vup_gate: out must not alias input tensors");

  const at::cuda::CUDAGuard guard(dev);
  const auto st = at::cuda::getCurrentCUDAStream();

  const auto* olat_ptr = reinterpret_cast<const __half*>(o_lat.data_ptr<at::Half>());
  const auto* wv_ptr = reinterpret_cast<const __half*>(w_v_up_cm.data_ptr<at::Half>());
  const auto* qckr_ptr = reinterpret_cast<const __half*>(qckr.data_ptr<at::Half>());
  auto* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  const int64_t stride_qckr = qckr.stride(0);

  const bool olat_al16 = (((uintptr_t)olat_ptr & 15u) == 0u);
  const bool olat_al4 = (((uintptr_t)olat_ptr & 3u) == 0u);
  const bool wv_al4 = (((uintptr_t)wv_ptr & 3u) == 0u);
  const bool qckr_al4 = (((uintptr_t)qckr_ptr & 3u) == 0u) && (N <= 1 || (stride_qckr & 1ll) == 0ll);
  const bool out_al4 = (((uintptr_t)out_ptr & 3u) == 0u);
  const dim3 grid((unsigned int)N, 4u);

  if (R_LAT == 192) {
    k_gla_vup_gate<192><<<grid, 192, 0, st>>>(olat_ptr, wv_ptr, qckr_ptr, stride_qckr, out_ptr, olat_al16, olat_al4,
                                              wv_al4, qckr_al4, out_al4);
  } else {
    k_gla_vup_gate<256><<<grid, 192, 0, st>>>(olat_ptr, wv_ptr, qckr_ptr, stride_qckr, out_ptr, olat_al16, olat_al4,
                                              wv_al4, qckr_al4, out_al4);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gla_vup_gate: kernel launch failed");
}

void gla_gate_only(torch::Tensor attn_tmp, torch::Tensor qckr, torch::Tensor out) {
  TORCH_CHECK(attn_tmp.is_cuda(), "gla_gate_only: attn_tmp must be a CUDA tensor");
  const auto dev = attn_tmp.device();
  TORCH_CHECK(qckr.device() == dev && out.device() == dev,
              "gla_gate_only: all tensors must be on the same CUDA device");

  TORCH_CHECK(attn_tmp.dtype() == torch::kHalf && attn_tmp.is_contiguous() &&
                  ((attn_tmp.dim() == 3 && attn_tmp.size(1) == NQ && attn_tmp.size(2) == HD) ||
                   (attn_tmp.dim() == 2 && attn_tmp.size(1) == NQ * HD)),
              "gla_gate_only: attn_tmp must be [N, 6, 256] or [N, 1536] contiguous fp16");
  const int64_t N = attn_tmp.size(0);
  TORCH_CHECK(N <= INT_MAX, "gla_gate_only: N exceeds INT_MAX");

  TORCH_CHECK(qckr.dtype() == torch::kHalf && qckr.dim() == 2 && qckr.stride(1) == 1 && qckr.size(0) == N &&
                  qckr.size(1) >= 3072 && (N <= 1 || qckr.stride(0) >= qckr.size(1)),
              "gla_gate_only: qckr must be [N, >=3072] fp16 with stride(1)==1");
  TORCH_CHECK(is_n_6_256_fp16(out, N), "gla_gate_only: out must be [N, 6, 256] or [N, 1536] contiguous fp16");

  if (N == 0) return;

  const ByteRange out_r = tensor_byte_range(out);
  TORCH_CHECK(!byte_ranges_overlap(out_r, tensor_byte_range(qckr)),
              "gla_gate_only: out must not alias qckr");
  TORCH_CHECK(out.data_ptr() == attn_tmp.data_ptr() || !byte_ranges_overlap(out_r, tensor_byte_range(attn_tmp)),
              "gla_gate_only: partial overlap between attn_tmp and out is forbidden");

  const at::cuda::CUDAGuard guard(dev);
  const auto st = at::cuda::getCurrentCUDAStream();

  const auto* a_ptr = reinterpret_cast<const __half*>(attn_tmp.data_ptr<at::Half>());
  const auto* qckr_ptr = reinterpret_cast<const __half*>(qckr.data_ptr<at::Half>());
  auto* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  const int64_t stride_qckr = qckr.stride(0);

  const bool a_al16 = (((uintptr_t)a_ptr & 15u) == 0u);
  const bool a_al4 = (((uintptr_t)a_ptr & 3u) == 0u);
  const bool qckr_al16 = (((uintptr_t)qckr_ptr & 15u) == 0u) && (N <= 1 || (stride_qckr & 7ll) == 0ll);
  const bool qckr_al4 = (((uintptr_t)qckr_ptr & 3u) == 0u) && (N <= 1 || (stride_qckr & 1ll) == 0ll);
  const bool o_al16 = (((uintptr_t)out_ptr & 15u) == 0u);
  const bool o_al4 = (((uintptr_t)out_ptr & 3u) == 0u);

  k_gla_gate_only<<<(unsigned int)N, 192, 0, st>>>(a_ptr, qckr_ptr, stride_qckr, out_ptr, a_al16, a_al4, qckr_al16,
                                                   qckr_al4, o_al16, o_al4);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gla_gate_only: kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, py::arg("dev_idx") = -1,
        "Pre-initialize kernel attributes and occupancy cache for a CUDA device (call before CUDA graph capture)");
  m.def("gla_cache_write", &gla_cache_write,
        "Quantize and write [c | k_rope | inv_rms] into packed E4M3 paged k_cache/v_cache");
  m.def("gla_gather_dequant", &gla_gather_dequant,
        "Gather and dequantize cached GLA tokens across a batch into contiguous fp16/fp32 tensors");
  m.def("attn_gla_mq", &attn_gla_mq,
        "Single-read shared-latent Volta sm_70 HMMA.884 split-KV absorbed multi-query GLA attention");
  m.def("gla_splits", &gla_splits, py::arg("bh"), py::arg("S"), py::arg("R_LAT"), py::arg("Q"), py::arg("sep") = 1,
        "Effective split count for (bh, S, R_LAT, Q, sep=1)");
  m.def("gla_workspace", &gla_workspace, py::arg("bh"), py::arg("S"), py::arg("R_LAT"), py::arg("Q"),
        py::arg("sep") = 1, "Partial workspace float counts (po, pml) for (bh, S, R_LAT, Q, sep=1)");
  m.def("gla_smem_and_occ", &gla_smem_and_occ, py::arg("R_LAT"), py::arg("Q"), py::arg("dev_idx") = -1,
        "Return (smem_bytes, active_blocks_per_sm) for (R_LAT, Q)");
  m.def("gla_prep_route_a", &gla_prep_route_a,
        "Fused Route-A attention prep (FP32 internal RMS/RoPE/GEMV; requires unique valid slots; wraps+clamps OOB positions)");
  m.def("gla_prep_prefill", &gla_prep_prefill,
        "Fused prefill attention prep (FP32 internal RMS/RoPE; requires unique valid slots; wraps+clamps OOB positions)");
  m.def("gla_vup_gate", &gla_vup_gate,
        "Fused Route-A output epilogue: o_lat @ W_v_up^T * sigmoid(gate) (FP32 internal accumulation & sigmoid)");
  m.def("gla_gate_only", &gla_gate_only,
        "Fused prefill output epilogue: attn_tmp * sigmoid(gate) in FP32 (exact in-place out==attn_tmp safe)");
  m.attr("SMAX") = (int64_t)SMAX;
  m.attr("QMAX") = (int64_t)QMAX;
  m.attr("ROPE_DIM") = (int64_t)ROPE_DIM;
}
