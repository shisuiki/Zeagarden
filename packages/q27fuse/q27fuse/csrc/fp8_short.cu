// fp8_short.cu - Exact single-pass 2D-tiled FP8 QPN8 (W8A16) HMMA.884 linear & gated MLP for Volta (sm_70, Tesla V100),
// targeting short / agentic prefills (M = 65..512+, works for any M >= 1).
//
// Zero extra weight copies, zero fp8_qpn8_dequantize_sm70_kernel DRAM round-trip:
//   - Reads 1Cat's existing QPN8 FP8 layout directly:
//       * codes: contiguous uint8 (numel = N * K), k-group-minor 32-column tiles x 32-k chunks (CH8 = 1024 B):
//         half 0 (+0, 512 B) holds k=0..15, half 1 (+512, 512 B) holds k=16..31; each lane l holds output
//         column 8*((l>>2)&3) + (l&3) + 4*(l>>4) (exact mma.m8n8k4 B-fragment map) with bytes ordered
//         k 0 2 4 6 1 3 5 7 per 8 so __byte_perm + f8x2 decodes exact fp16x2 = E4M3 * 2^-8.
//       * scales: contiguous fp16 (numel = N), equal to channel_scale * 256 (undoes the 2^-8 in epilogue).
//   - Finite-code weight contract (matching fp8_tile.cuh / 1Cat QPN8): quantized checkpoint weights contain
//     only finite E4M3FN codes (no E4M3FN NaN bytes 0x7F/0xFF). f8x2() reproduces exact E4M3 * 2^-8 for
//     every finite E4M3FN byte (normal and subnormal).
//   - Activations X [M, K] row-major fp16 are cooperatively staged per 32-k chunk into a double-buffered
//     conflict-free fragment-order SMEM buffer sA[2][4][BM + 2] (uint4): 8.25 KB at BM=64, 16.25 KB at BM=128.
//   - Two register-tight pipelining modes (both <= ~116 regs/thread, 0 spills):
//       * HALF_PIPE = 0 (CB = 1): full 32-k chunk weight double-buffering (16 weight regs, 32-HMMA overlap).
//       * HALF_PIPE = 1 (CB = 2): 16-k half-chunk weight double-buffering (16 weight regs instead of 32,
//         still 32-HMMA overlap per half-chunk since 4 * RB * CB = 32 HMMAs!).
//   - Fused GATED = 1 mode (fp8_short_gated, for FP8 MLP L56-63 gate_up 5120 -> 4352):
//       * QPN8 stores merged [gate (0..I-1) | up (I..2I-1)] as NT_OUT gate tiles followed by NT_OUT up tiles.
//       * With CB = 2, cb=0 loads gate tile p and cb=1 loads matching up tile p + NT_OUT (stride NT_OUT*NC*CH8).
//       * Every lane owns exact matching gate (acc[rb][0][e]) and up (acc[rb][1][e]) in registers (0 shuffles!),
//         applies per-channel gate/up scales and silu(g)*u in FP32, and writes row-major [M, I] directly with STG.E.128.
//   - M-inner, N-outer CTA launch order (bm = blockIdx.x % gridM, bn = blockIdx.x / gridM) keeps all CTAs
//     sharing the same weight column-slab resident simultaneously so weights hit L2 across M-tiles.
//   - Call init() once per device outside CUDA graph capture to pre-configure SMEM carveouts thread-safely.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace {
constexpr int CH8 = 1024;                                        // bytes per (32-col tile, 32-k chunk)
constexpr int MAX_DEVS = 32;

inline bool is_aligned(const torch::Tensor& t, size_t align) {
  return t.numel() == 0 || (reinterpret_cast<uintptr_t>(t.data_ptr()) & (align - 1u)) == 0u;
}

inline bool tensors_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  const size_t ab = (size_t)a.numel() * (size_t)a.element_size();
  const size_t bb = (size_t)b.numel() * (size_t)b.element_size();
  if (ab == 0 || bb == 0) return false;
  const uintptr_t a0 = reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t b0 = reinterpret_cast<uintptr_t>(b.data_ptr());
  return a0 < (b0 + bb) && b0 < (a0 + ab);
}

__device__ __forceinline__ void mma884(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// WLD 0: ld.global.nc (caches in L1 + L2, allowing WM M-warps in the same CTA to reuse weights out of L1).
// WLD 1: ld.global.nc.L1::no_allocate (caches in L2 only, preserving L1 for activations).
template <int WLD>
__device__ __forceinline__ uint4 ldw16(const void* p) {
  uint4 v;
  if constexpr (WLD == 0) {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
  }
  return v;
}

__device__ __forceinline__ uint4 lda16(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ unsigned f8x2(unsigned p) {           // bytes 0 and 2 (E4M3) -> fp16x2 = E4M3 * 2^-8, exact
  return ((p << 8) & 0x80008000u) | ((p << 7) & 0x3F803F80u);
}

// Compute one 16-k half-chunk (HALF = 0 -> jj = 0,1; HALF = 1 -> jj = 2,3 = 4 HMMA.884 K-steps per (rb, cb)).
template <int RB, int CB, int HALF, int STRIDE>
__device__ __forceinline__ void chunk8_half(const uint4 (&w)[CB], const uint4* __restrict__ ap,
                                            float (&acc)[RB][CB][8]) {
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int jj = HALF * 2 + j;
    unsigned b0[CB], b1[CB], b2[CB], b3[CB];
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const unsigned qx = j ? w[cb].z : w[cb].x;
      const unsigned qy = j ? w[cb].w : w[cb].y;
      b0[cb] = f8x2(__byte_perm(qx, qy, 0x0400));
      b1[cb] = f8x2(__byte_perm(qx, qy, 0x0501));
      b2[cb] = f8x2(__byte_perm(qx, qy, 0x0602));
      b3[cb] = f8x2(__byte_perm(qx, qy, 0x0703));
    }
    const uint4* apj = ap + jj * STRIDE;
    uint4 a_cur = apj[0];
#pragma unroll
    for (int rb = 0; rb < RB; rb++) {
      uint4 a_nxt;
      if (rb + 1 < RB) a_nxt = apj[(rb + 1) * 8];
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        mma884(acc[rb][cb], a_cur.x, a_cur.y, b0[cb], b1[cb]);
        mma884(acc[rb][cb], a_cur.z, a_cur.w, b2[cb], b3[cb]);
      }
      if (rb + 1 < RB) a_cur = a_nxt;
    }
  }
}

template <int WM, int WN, int RB, int CB>
struct SP8 {
  static constexpr int NW = WM * WN;
  static constexpr int NTH = NW * 32;
  static constexpr int BM = WM * RB * 8;
  static constexpr int BN = WN * CB * 32;
  static constexpr int STRIDE = BM + 2;                          // uint4 stride per jj -> 0 bank conflicts on STS & LDS
  static constexpr int STAGE_U4 = 4 * STRIDE;                    // uint4s per A stage
  static constexpr int SMEM_A = 2 * STAGE_U4 * 16;               // bytes for double-buffered A
  static constexpr int SMEM_EPI = NW * 512;                      // per-warp epilogue scratch (reuses A buffer)
  static constexpr int SMEM = SMEM_A > SMEM_EPI ? SMEM_A : SMEM_EPI;
  static constexpr int NPC = (BM * 4) / NTH;                     // 16-B A-load units per thread per 32-k chunk
  static_assert((BM * 4) % NTH == 0 && NPC >= 1, "NTH must evenly divide BM * 4");
  static_assert(BM % 8 == 0, "BM must be a multiple of 8");
  static_assert(CB == 1 || CB == 2, "CB must be 1 or 2");
};

// 2D-tiled single-pass FP8 QPN8 W8A16 HMMA.884 linear (GATED=0) or fused SiLU-and-mul gate_up (GATED=1).
// W: QPN8 codes (N*K u8). X: [M][K] row-major fp16. S: [N] fp16 (= scale * 256).
// Y: GATED=0 -> [M][N] row-major fp16; GATED=1 -> [M][N/2] row-major fp16.
template <int WM, int WN, int RB, int CB, int MB, int WLD = 0, int HALF_PIPE = 0, int GATED = 0>
__global__ void __launch_bounds__(SP8<WM, WN, RB, CB>::NTH, MB)
k_fp8_short(const uint8_t* __restrict__ W, const __half* __restrict__ X, const __half* __restrict__ S,
            __half* __restrict__ Y, int M, int N, int K, int gridM, int zero) {
  static_assert(!GATED || CB == 2, "GATED requires CB == 2 (gate tile p + up tile p + NT/2)");
  using P = SP8<WM, WN, RB, CB>;
  constexpr int NTH = P::NTH, BM = P::BM, STRIDE = P::STRIDE, STAGE_U4 = P::STAGE_U4, NPC = P::NPC;
  extern __shared__ __align__(16) uint8_t sm[];
  uint4* sA = reinterpret_cast<uint4*>(sm);                      // [2][4][BM + 2] uint4

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wn = warp % WN, wm = warp / WN;

  // M-inner, N-outer CTA swizzle: CTAs sharing the same weight column-slab execute concurrently and hit L2.
  const int bm = blockIdx.x % gridM;
  const int bn = blockIdx.x / gridM;

  const int NC = K / 32;
  const int NT = N / 32;
  const int NT_OUT = GATED ? (NT >> 1) : NT;
  const int row0_cta = bm * BM;
  const int row0_warp = wm * RB * 8;
  const int p0 = GATED ? (bn * WN + wn) : ((bn * WN + wn) * CB);
  const int rin = (lane & 3) + 4 * (lane >> 4);                  // A-fragment row inside an 8-row block
  const unsigned zmask = (unsigned)zero;

  const uint8_t* wp0 = W + (size_t)min(p0, NT_OUT - 1) * NC * CH8 + lane * 16;
  const size_t cbs = GATED
                         ? ((size_t)NT_OUT * NC * CH8)
                         : ((CB > 1 && (p0 + 1) < NT) ? (size_t)NC * CH8 : 0);

  float acc[RB][CB][8];
#pragma unroll
  for (int rb = 0; rb < RB; rb++)
#pragma unroll
    for (int cb = 0; cb < CB; cb++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[rb][cb][e] = 0.f;

  uint4 stg[NPC];
  unsigned xr = 0;

  auto ldx = [&](int ch, unsigned dep) {
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = q * NTH + tid;                               // 16-B unit = row * 4 + jj
      const int row = u >> 2, jj = u & 3;
      const int grow = row0_cta + row;
      if (grow < M) {
        stg[q] = lda16(reinterpret_cast<const uint8_t*>(X + (size_t)grow * K + ch * 32 + jj * 8) + dep);
      } else {
        stg[q] = make_uint4(0, 0, 0, 0);
      }
    }
  };

  auto stx = [&](int stage) {
    uint4* dst = sA + stage * STAGE_U4;
#pragma unroll
    for (int q = 0; q < NPC; q++) {
      const int u = q * NTH + tid;
      const int row = u >> 2, jj = u & 3;
      dst[jj * STRIDE + row] = stg[q];
    }
  };

  if constexpr (HALF_PIPE == 0) {
    // Full 32-k chunk weight double-buffering (best for CB = 1: 16 weight regs, full 32-HMMA hiding window)
    uint4 wA0[CB], wA1[CB], wB0[CB], wB1[CB];

    auto ldw = [&](uint4 (&w0)[CB], uint4 (&w1)[CB], int ch, unsigned dep) {
      const size_t off = (size_t)ch * CH8 + dep;
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        const uint8_t* pw = wp0 + cb * cbs + off;
        w0[cb] = ldw16<WLD>(pw);
        w1[cb] = ldw16<WLD>(pw + 512);
      }
    };

    auto touch = [&](const uint4 (&w0)[CB], const uint4 (&w1)[CB]) {
      unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) x ^= w0[cb].x ^ w1[cb].x;
#pragma unroll
      for (int q = 0; q < NPC; q++) x ^= stg[q].x;
      xr ^= x;
      return x & zmask;
    };

    auto step = [&](const uint4 (&wc0)[CB], const uint4 (&wc1)[CB],
                    uint4 (&wn0)[CB], uint4 (&wn1)[CB], int ch, int stage, unsigned& dep) {
      ldw(wn0, wn1, ch + 1, dep);
      ldx(ch + 1, dep);
      const uint4* ap = sA + stage * STAGE_U4 + row0_warp + rin;
      chunk8_half<RB, CB, 0, STRIDE>(wc0, ap, acc);
      chunk8_half<RB, CB, 1, STRIDE>(wc1, ap, acc);
      dep = touch(wn0, wn1);
      stx(stage ^ 1);
      __syncthreads();
    };

    ldw(wA0, wA1, 0, 0u);
    ldx(0, 0u);
    unsigned dep = touch(wA0, wA1);
    stx(0);
    __syncthreads();

    int ch = 0;
    for (; ch + 2 < NC; ch += 2) {
      step(wA0, wA1, wB0, wB1, ch, 0, dep);
      step(wB0, wB1, wA0, wA1, ch + 1, 1, dep);
    }
    if (ch + 1 < NC) {
      step(wA0, wA1, wB0, wB1, ch, 0, dep);
      const uint4* ap = sA + STAGE_U4 + row0_warp + rin;
      chunk8_half<RB, CB, 0, STRIDE>(wB0, ap, acc);
      chunk8_half<RB, CB, 1, STRIDE>(wB1, ap, acc);
    } else if (ch < NC) {
      const uint4* ap = sA + row0_warp + rin;
      chunk8_half<RB, CB, 0, STRIDE>(wA0, ap, acc);
      chunk8_half<RB, CB, 1, STRIDE>(wA1, ap, acc);
    }
  } else {
    // Half-chunk (16-k) weight double-buffering (best for CB = 2: cuts weight regs 32 -> 16, zero spills,
    // still overlaps each half-chunk load with 4 * RB * CB = 32 HMMAs!)
    uint4 w0[CB], w1[CB];

    auto ldw_h = [&](uint4 (&w)[CB], int ch, int half, unsigned dep) {
      const size_t off = (size_t)ch * CH8 + (half ? 512 : 0) + dep;
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        w[cb] = ldw16<WLD>(wp0 + cb * cbs + off);
      }
    };

    auto touch_w = [&](const uint4 (&w)[CB]) {
      unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) x ^= w[cb].x;
      xr ^= x;
      return x & zmask;
    };

    auto touch_wx = [&](const uint4 (&w)[CB]) {
      unsigned x = __float_as_uint(acc[RB - 1][CB - 1][7]);
#pragma unroll
      for (int cb = 0; cb < CB; cb++) x ^= w[cb].x;
#pragma unroll
      for (int q = 0; q < NPC; q++) x ^= stg[q].x;
      xr ^= x;
      return x & zmask;
    };

    auto step_h = [&](int ch, int stage, unsigned& dep) {
      ldw_h(w1, ch, 1, dep);
      const uint4* ap = sA + stage * STAGE_U4 + row0_warp + rin;
      chunk8_half<RB, CB, 0, STRIDE>(w0, ap, acc);
      dep = touch_w(w1);
      ldw_h(w0, ch + 1, 0, dep);
      ldx(ch + 1, dep);
      chunk8_half<RB, CB, 1, STRIDE>(w1, ap, acc);
      dep = touch_wx(w0);
      stx(stage ^ 1);
      __syncthreads();
    };

    auto tail_h = [&](int ch, int stage, unsigned& dep) {
      ldw_h(w1, ch, 1, dep);
      const uint4* ap = sA + stage * STAGE_U4 + row0_warp + rin;
      chunk8_half<RB, CB, 0, STRIDE>(w0, ap, acc);
      dep = touch_w(w1);
      chunk8_half<RB, CB, 1, STRIDE>(w1, ap, acc);
    };

    ldw_h(w0, 0, 0, 0u);
    ldx(0, 0u);
    unsigned dep = touch_wx(w0);
    stx(0);
    __syncthreads();

    int ch = 0;
    for (; ch + 2 < NC; ch += 2) {
      step_h(ch, 0, dep);
      step_h(ch + 1, 1, dep);
    }
    if (ch + 1 < NC) {
      step_h(ch, 0, dep);
      tail_h(ch + 1, 1, dep);
    } else if (ch < NC) {
      tail_h(ch, 0, dep);
    }
  }

  // Ensure all warps have finished reading sA before aliasing sm[] for the per-warp epilogue.
  __syncthreads();
  if (xr == 0x9E3779B9u && zero == 1) Y[0] = __float2half(0.f);  // never true; keeps forced waits live

  // Per-warp zero-bank-conflict epilogue (512 B/warp = [8 rows][16 half2], XOR-swizzled on bit 1 of col2 when r8 >= 4)
  __half2* sm_ep = reinterpret_cast<__half2*>(sm) + warp * 128;
  const int r8_lo = (lane & 1) | ((lane >> 4) << 2);             // 0, 1, 4, 5
  const int r8_hi = r8_lo + 2;                                   // 2, 3, 6, 7
  const int swiz = (lane >> 4) << 1;                             // 0 for r8 < 4, 2 for r8 >= 4
  const int q = (lane >> 2) & 3;
  const int col2_0 = ((q << 2) | ((lane & 2) >> 1)) ^ swiz;
  const int col2_4 = ((q << 2) | 2 | ((lane & 2) >> 1)) ^ swiz;
  const int rd_r8 = lane >> 2;                                   // 0..7
  const int rd_u = lane & 3;                                     // 0..3 -> 8 halfs = 16 B

  if constexpr (GATED) {
    // Fused SiLU(gate) * up -> row-major [M, N/2] fp16.
    // cb=0 holds gate tile p0, cb=1 holds exact matching up tile p0 + NT_OUT in the SAME lane and register index e!
    if (p0 < NT_OUT) {
      const int I = N >> 1;
      const int col_lo = p0 * 32 + q * 8 + (lane & 2);
      const __half2 sg_lo = __ldg(reinterpret_cast<const __half2*>(S + col_lo));
      const __half2 sg_hi = __ldg(reinterpret_cast<const __half2*>(S + col_lo + 4));
      const __half2 su_lo = __ldg(reinterpret_cast<const __half2*>(S + I + col_lo));
      const __half2 su_hi = __ldg(reinterpret_cast<const __half2*>(S + I + col_lo + 4));
      const float gl0 = __low2float(sg_lo), gl1 = __high2float(sg_lo);
      const float gh0 = __low2float(sg_hi), gh1 = __high2float(sg_hi);
      const float ul0 = __low2float(su_lo), ul1 = __high2float(su_lo);
      const float uh0 = __low2float(su_hi), uh1 = __high2float(su_hi);

      auto silu_mul = [](float g, float u) -> float {
        return g / (1.f + __expf(-g)) * u;
      };

#pragma unroll
      for (int rb = 0; rb < RB; rb++) {
        const int m_base = row0_cta + row0_warp + rb * 8;
        sm_ep[r8_lo * 16 + col2_0] = __floats2half2_rn(
            silu_mul(acc[rb][0][0] * gl0, acc[rb][1][0] * ul0),
            silu_mul(acc[rb][0][1] * gl1, acc[rb][1][1] * ul1));
        sm_ep[r8_hi * 16 + col2_0] = __floats2half2_rn(
            silu_mul(acc[rb][0][2] * gl0, acc[rb][1][2] * ul0),
            silu_mul(acc[rb][0][3] * gl1, acc[rb][1][3] * ul1));
        sm_ep[r8_lo * 16 + col2_4] = __floats2half2_rn(
            silu_mul(acc[rb][0][4] * gh0, acc[rb][1][4] * uh0),
            silu_mul(acc[rb][0][5] * gh1, acc[rb][1][5] * uh1));
        sm_ep[r8_hi * 16 + col2_4] = __floats2half2_rn(
            silu_mul(acc[rb][0][6] * gh0, acc[rb][1][6] * uh0),
            silu_mul(acc[rb][0][7] * gh1, acc[rb][1][7] * uh1));
        __syncwarp();
        const int grow = m_base + rd_r8;
        if (grow < M) {
          uint4 v = *reinterpret_cast<const uint4*>(sm_ep + rd_r8 * 16 + rd_u * 4);
          if (rd_r8 >= 4) v = make_uint4(v.z, v.w, v.x, v.y);
          *reinterpret_cast<uint4*>(Y + (size_t)grow * I + p0 * 32 + rd_u * 8) = v;
        }
        __syncwarp();
      }
    }
  } else {
#pragma unroll
    for (int cb = 0; cb < CB; cb++) {
      const int p = p0 + cb;
      if (p < NT) {
        // Load the 4 channel scales owned by this lane once and reuse across all RB row blocks.
        const int col_lo = p * 32 + q * 8 + (lane & 2);
        const __half2 s_lo = __ldg(reinterpret_cast<const __half2*>(S + col_lo));
        const __half2 s_hi = __ldg(reinterpret_cast<const __half2*>(S + col_lo + 4));
        const float sl0 = __low2float(s_lo), sl1 = __high2float(s_lo);
        const float sh0 = __low2float(s_hi), sh1 = __high2float(s_hi);
#pragma unroll
        for (int rb = 0; rb < RB; rb++) {
          const int m_base = row0_cta + row0_warp + rb * 8;
          sm_ep[r8_lo * 16 + col2_0] = __floats2half2_rn(acc[rb][cb][0] * sl0, acc[rb][cb][1] * sl1);
          sm_ep[r8_hi * 16 + col2_0] = __floats2half2_rn(acc[rb][cb][2] * sl0, acc[rb][cb][3] * sl1);
          sm_ep[r8_lo * 16 + col2_4] = __floats2half2_rn(acc[rb][cb][4] * sh0, acc[rb][cb][5] * sh1);
          sm_ep[r8_hi * 16 + col2_4] = __floats2half2_rn(acc[rb][cb][6] * sh0, acc[rb][cb][7] * sh1);
          __syncwarp();
          const int grow = m_base + rd_r8;
          if (grow < M) {
            uint4 v = *reinterpret_cast<const uint4*>(sm_ep + rd_r8 * 16 + rd_u * 4);
            if (rd_r8 >= 4) v = make_uint4(v.z, v.w, v.x, v.y);
            *reinterpret_cast<uint4*>(Y + (size_t)grow * N + p * 32 + rd_u * 8) = v;
          }
          __syncwarp();
        }
      }
    }
  }
}

template <int WM, int WN, int RB, int CB, int MB, int WLD, int HALF_PIPE, int GATED>
void init_fp8_dev(int dev, cudaStream_t st) {
  using P = SP8<WM, WN, RB, CB>;
  auto fn = k_fp8_short<WM, WN, RB, CB, MB, WLD, HALF_PIPE, GATED>;
  static std::once_flag flags[MAX_DEVS];
  TORCH_CHECK(dev >= 0 && dev < MAX_DEVS, "fp8_short: device index out of range [0, 32)");
  std::call_once(flags[dev], [&]() {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    TORCH_CHECK(cudaStreamIsCapturing(st, &cap) == cudaSuccess);
    TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                "fp8_short: call init() before CUDA graph capture (cannot run cudaFuncSetAttribute during capture)");
    // Minimal SMEM carveout preserving full occupancy so the rest of the 128 KB unified SRAM stays L1 cache.
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    int occ = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, P::NTH, P::SMEM) == cudaSuccess);
    const int pct = std::min(100, (int)((100LL * std::max(occ, 1) * P::SMEM + 98303) / 98304));
    TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, pct) == cudaSuccess);
  });
}

template <int WM, int WN, int RB, int CB, int MB, int WLD, int HALF_PIPE, int GATED = 0>
void launch_fp8(const uint8_t* W, const __half* X, const __half* S, __half* Y,
                int M, int N, int K, int dev, cudaStream_t st) {
  using P = SP8<WM, WN, RB, CB>;
  auto fn = k_fp8_short<WM, WN, RB, CB, MB, WLD, HALF_PIPE, GATED>;
  init_fp8_dev<WM, WN, RB, CB, MB, WLD, HALF_PIPE, GATED>(dev, st);
  const int gridM = (M + P::BM - 1) / P::BM;
  const int out_n = GATED ? (N / 2) : N;
  const int cta_out_n = GATED ? (WN * 32) : P::BN;
  const int gridN = (out_n + cta_out_n - 1) / cta_out_n;
  const int grid = gridM * gridN;
  fn<<<grid, P::NTH, P::SMEM, st>>>(W, X, S, Y, M, N, K, gridM, 0);
}

using InitFn = void (*)(int, cudaStream_t);
using LaunchFP8 = void (*)(const uint8_t*, const __half*, const __half*, __half*, int, int, int, int, cudaStream_t);
struct FP8Cfg {
  const char* name;
  int bm;
  int bn;
  InitFn init_fn;
  LaunchFP8 fn;
};

#define FP8_CFG(NAME, WM, WN, RB, CB, MB, WLD, HP) \
  {NAME, (WM) * (RB) * 8, (WN) * (CB) * 32, \
   init_fp8_dev<WM, WN, RB, CB, MB, WLD, HP, 0>, launch_fp8<WM, WN, RB, CB, MB, WLD, HP, 0>}

#define FP8_GCFG(NAME, WM, WN, RB, MB, WLD) \
  {NAME, (WM) * (RB) * 8, (WN) * 32, \
   init_fp8_dev<WM, WN, RB, 2, MB, WLD, 1, 1>, launch_fp8<WM, WN, RB, 2, MB, WLD, 1, 1>}

const FP8Cfg CFGS[] = {
    // Primary 4 tile shapes (WLD=0: ld.global.nc L1+L2 weight caching for intra-CTA WM reuse)
    FP8_CFG("64x64_w2x2_r4c1_l1",   2, 2, 4, 1, 4, 0, 0),  // cfg 0: 128 thr, high SM waves (M=65..256)
    FP8_CFG("64x128_w2x2_r4c2_l1",  2, 2, 4, 2, 3, 0, 1),  // cfg 1: 128 thr, 2x A reuse, half-chunk pipe
    FP8_CFG("128x64_w4x2_r4c1_l1",  4, 2, 4, 1, 2, 0, 0),  // cfg 2: 256 thr, 1x weight read at M<=128 (N=5120/8704)
    FP8_CFG("128x128_w4x2_r4c2_l1", 4, 2, 4, 2, 1, 0, 1),  // cfg 3: 256 thr, max arithmetic intensity (M=256..512)
    // Low-register CB=1 alternatives & high-wave shapes (~80 regs/thr, zero spills)
    FP8_CFG("64x128_w2x4_r4c1_l1",  2, 4, 4, 1, 2, 0, 0),  // cfg 4: 256 thr, low-reg 64x128
    FP8_CFG("128x128_w4x4_r4c1_l1", 4, 4, 4, 1, 1, 0, 0),  // cfg 5: 512 thr, low-reg 128x128
    FP8_CFG("64x64_w4x2_r2c1_l1",   4, 2, 2, 1, 3, 0, 0),  // cfg 6: 256 thr, high-occ 64x64
    FP8_CFG("128x32_w4x1_r4c1_l1",  4, 1, 4, 1, 4, 0, 0),  // cfg 7: 128 thr, >=112 CTAs & 1x W read at M<=128 (N=3584/4096)
    // L2-only weight load A/B variants (WLD=1: ld.global.nc.L1::no_allocate)
    FP8_CFG("64x64_w2x2_r4c1_na",   2, 2, 4, 1, 4, 1, 0),  // cfg 8
    FP8_CFG("128x64_w4x2_r4c1_na",  4, 2, 4, 1, 2, 1, 0),  // cfg 9
    FP8_CFG("128x128_w4x2_r4c2_na", 4, 2, 4, 2, 1, 1, 1),  // cfg 10
    FP8_CFG("64x128_w2x4_r4c1_na",  2, 4, 4, 1, 2, 1, 0),  // cfg 11
};
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);

// Fused gated gate_up configs (CB=2: cb=0 gate tile p, cb=1 up tile p + NT_OUT; HALF_PIPE=1; bn = output cols/CTA)
const FP8Cfg GCFGS[] = {
    FP8_GCFG("64x64out_w2x2_r4c2_l1",  2, 2, 4, 3, 0),     // gcfg 0: 128 thr, 64 out cols (128 wt cols)
    FP8_GCFG("128x64out_w4x2_r4c2_l1", 4, 2, 4, 1, 0),     // gcfg 1: 256 thr, 64 out cols (128 wt cols)
    FP8_GCFG("64x32out_w2x1_r4c2_l1",  2, 1, 4, 3, 0),     // gcfg 2:  64 thr, 32 out cols (136 N-tiles on I=4352)
    FP8_GCFG("128x32out_w4x1_r4c2_l1", 4, 1, 4, 1, 0),     // gcfg 3: 128 thr, 32 out cols (136 N-tiles, 1x W read M<=128)
    FP8_GCFG("64x64out_w4x2_r2c2_l1",  4, 2, 2, 2, 0),     // gcfg 4: 256 thr, low-reg (~84 regs/thr, 2 CTAs/SM)
    FP8_GCFG("128x64out_w4x2_r4c2_na", 4, 2, 4, 1, 1),     // gcfg 5: L2-only A/B
    FP8_GCFG("64x64out_w2x2_r4c2_na",  2, 2, 4, 3, 1),     // gcfg 6: L2-only A/B
};
constexpr int NGCFG = sizeof(GCFGS) / sizeof(GCFGS[0]);
#undef FP8_CFG
#undef FP8_GCFG
}  // namespace

// Pre-initialize CUDA function attributes on the current (or specified) device outside CUDA graph capture.
void init(int64_t device_index = -1) {
  int dev = (int)device_index;
  if (dev < 0) {
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  }
  const at::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, (c10::DeviceIndex)dev));
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  for (int i = 0; i < NCFG; i++) CFGS[i].init_fn(dev, st);
  for (int i = 0; i < NGCFG; i++) GCFGS[i].init_fn(dev, st);
}

// out [M][N] fp16 = x [M][K] fp16 @ (QPN8 codes [N*K] u8, scales [N] fp16 = scale * 256)^T
void fp8_short(torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor out, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous(), "fp8_short: x");
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous(), "fp8_short: codes");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf && scales.is_contiguous(), "fp8_short: scales");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.is_contiguous(), "fp8_short: out");
  TORCH_CHECK(codes.device() == x.device() && scales.device() == x.device() && out.device() == x.device(),
              "fp8_short: all tensors must be on x.device()");
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
  TORCH_CHECK(out.size(0) == M, "fp8_short: M mismatch");
  TORCH_CHECK(K >= 32 && K % 32 == 0, "fp8_short: K must be positive multiple of 32");
  TORCH_CHECK(N >= 32 && N % 32 == 0, "fp8_short: N must be positive multiple of 32");
  TORCH_CHECK(codes.numel() == (int64_t)N * K, "fp8_short: codes numel mismatch");
  TORCH_CHECK(scales.numel() == N, "fp8_short: scales numel mismatch");
  TORCH_CHECK(cfg >= 0 && cfg < NCFG, "fp8_short: invalid cfg");
  TORCH_CHECK(is_aligned(x, 16) && is_aligned(codes, 16) && is_aligned(out, 16) && is_aligned(scales, 4),
              "fp8_short: x, codes, out must be 16-byte aligned and scales 4-byte aligned");
  TORCH_CHECK(!tensors_overlap(out, x) && !tensors_overlap(out, codes) && !tensors_overlap(out, scales),
              "fp8_short: out must not alias x, codes, or scales");
  if (M == 0) return;

  int dev = (int)x.device().index();
  const at::cuda::CUDAGuard guard(x.device());
  if (dev < 0) TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  CFGS[cfg].fn(codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
               reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, N, K, dev, st);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// out [M][I] fp16 = silu(x @ W_gate^T) * (x @ W_up^T) on 1Cat's merged QPN8 [gate | up] layout (N = 2 * I)
void fp8_short_gated(torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor out, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous(), "fp8_short_gated: x");
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous(), "fp8_short_gated: codes");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf && scales.is_contiguous(), "fp8_short_gated: scales");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.dim() == 2 && out.is_contiguous(), "fp8_short_gated: out");
  TORCH_CHECK(codes.device() == x.device() && scales.device() == x.device() && out.device() == x.device(),
              "fp8_short_gated: all tensors must be on x.device()");
  const int64_t M = x.size(0), K = x.size(1), I = out.size(1);
  TORCH_CHECK(out.size(0) == M, "fp8_short_gated: M mismatch");
  TORCH_CHECK(K >= 32 && K % 32 == 0, "fp8_short_gated: K must be positive multiple of 32");
  TORCH_CHECK(I >= 32 && I % 32 == 0, "fp8_short_gated: I (out.size(1)) must be positive multiple of 32");
  const int64_t N = 2 * I;
  TORCH_CHECK(codes.numel() == N * K, "fp8_short_gated: codes numel mismatch");
  TORCH_CHECK(scales.numel() == N, "fp8_short_gated: scales numel mismatch");
  TORCH_CHECK(cfg >= 0 && cfg < NGCFG, "fp8_short_gated: invalid cfg");
  TORCH_CHECK(is_aligned(x, 16) && is_aligned(codes, 16) && is_aligned(out, 16) && is_aligned(scales, 4),
              "fp8_short_gated: x, codes, out must be 16-byte aligned and scales 4-byte aligned");
  TORCH_CHECK(!tensors_overlap(out, x) && !tensors_overlap(out, codes) && !tensors_overlap(out, scales),
              "fp8_short_gated: out must not alias x, codes, or scales");
  if (M == 0) return;

  int dev = (int)x.device().index();
  const at::cuda::CUDAGuard guard(x.device());
  if (dev < 0) TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  GCFGS[cfg].fn(codes.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)M, (int)N, (int)K, dev, st);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> fp8_short_configs() {
  std::vector<std::string> v;
  v.reserve(NCFG);
  for (int i = 0; i < NCFG; i++) v.push_back(std::to_string(i) + ":" + CFGS[i].name);
  return v;
}

std::vector<std::string> fp8_short_gated_configs() {
  std::vector<std::string> v;
  v.reserve(NGCFG);
  for (int i = 0; i < NGCFG; i++) v.push_back(std::to_string(i) + ":" + GCFGS[i].name);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, pybind11::arg("device_index") = -1,
        "Pre-initialize CUDA function attributes on device (call outside CUDA graph capture)");
  m.def("fp8_short", &fp8_short,
        "Single-pass 2D-tiled FP8 QPN8 W8A16 HMMA.884 linear (out = x @ W^T, zero DRAM dequant)");
  m.def("fp8_short_gated", &fp8_short_gated,
        "Single-pass 2D-tiled FP8 QPN8 W8A16 fused gate_up SiLU-and-mul (out [M, N/2] = silu(x Wg^T) * (x Wu^T))");
  m.def("fp8_short_configs", &fp8_short_configs,
        "List available tile configuration strings for fp8_short");
  m.def("fp8_short_gated_configs", &fp8_short_gated_configs,
        "List available tile configuration strings for fp8_short_gated");
}
