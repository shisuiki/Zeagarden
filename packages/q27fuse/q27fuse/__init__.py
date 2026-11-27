"""q27fuse - vLLM plugin for serving Qwen3.8-27B-NVFP4 on 4x Tesla V100-SXM2 (sm_70, 1Cat-vLLM v1.5.3 / v1.5.4).

Master switch:
  Q27_FUSE unset or 0 -> plugin is inert; any non-zero value enables the exact serving stack:
    - Decode & MTP verify: single-copy NVFP4 MLP (mlp_fp4.cu), single-pass QPN8 FP8 projections
      (fp8_lin.cu, fp8_gdn_in.cu), single-copy FP8 LM head (fp8_head.cu), split-KV verify attention
      (attn_mq.cu, attn_dec.cu), fused GDN decode/verify cores (gdn_core.cu, gdn_verify.cu, gdn_decode.cu),
      rotated 1-bit MTP draft screen (draft_screen.cu), step-tail fusions (tail_ops.cu), and batch-adaptive
      speculative length (Q27_SPEC_K).
    - Exact prefill (default ON): unified CUDA-graph-safe TP4 all-reduce + Gemma norm (arn.cu, M = 65..4608,
      Q27_ARN=1), cuBLAS algo 6 FP8 prefill GEMMs (pf_gemm.cu), and short/agentic W4/W8 GEMMs
      (mlp_short.cu, fp8_short.cu; Q27_SPF=0 opts out).

Opt-in lossy acceleration modes (default OFF):
  - Packed long prefill (Q27_PF_PK=1, optional Q27_PF_H32_PLANE=<dir> for 1-bit BF16 plane):
      Online H_32 Walsh-Hadamard INT4-(1,256) packed NVFP4 MLP (pk_h32.cu) with zero-sync Freivalds guard.
  - TP4-native GLA-g4 KV compression (Q27_GLA_CKPT=<path>):
      256 B/tok (2.0x) or 320 B/tok (1.6x) FP8 compressed KV with spliced QPN8 qkv_proj and single-read
      shared-latent HMMA.884 attention (attn_gla_mq.cu).
"""
import logging
import os

log = logging.getLogger("q27fuse")


def enabled() -> bool:
    return (os.environ.get("Q27_FUSE", "") or "0").strip() != "0"


def register():
    if not enabled():
        if os.environ.get("Q27_SPEC_K"):
            log.warning("q27fuse: Q27_SPEC_K is set but Q27_FUSE is unset/0; ignored")
        return
    log.info(
        "q27fuse: enabled (Q27_FUSE=%s, Q27_SPEC_K=%s, Q27_SPF=%s, Q27_PF_PK=%s, Q27_GLA=%s, Q27_DFLY2=%s)",
        os.environ.get("Q27_FUSE"),
        os.environ.get("Q27_SPEC_K", ""),
        os.environ.get("Q27_SPF", "1"),
        os.environ.get("Q27_PF_PK", "0"),
        bool(os.environ.get("Q27_GLA_CKPT", "").strip()),
        bool(os.environ.get("Q27_DFLY2_CKPT", "").strip()),
    )
    from . import draftscreen, gdnv, head, patch, spec_k, tail, tail_mtp

    has_dfly2 = bool(os.environ.get("Q27_DFLY2_CKPT", "").strip())
    patch.apply()
    spec_k.apply()
    if not has_dfly2:
        draftscreen.apply()
    tail.apply()
    if not has_dfly2:
        tail_mtp.apply()
    head.apply()  # wraps tail's get_top_tokens
    if gdnv.CKPT_FIX and patch._on("Q27_GDNV"):
        gdnv.install_ckpt_fix()

    if (os.environ.get("Q27_SPF", "1") or "1").strip() != "0":
        from . import shortpf

        shortpf.apply()

    if os.environ.get("Q27_GLA_CKPT", "").strip():
        from . import gla

        gla.apply()

    if has_dfly2:
        from . import dfly2

        dfly2.apply()
