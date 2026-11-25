"""q27fuse.patch - everything the plugin changes in 1Cat-vLLM for Qwen3.8-27B-NVFP4 at TP4 (one configuration, on when
the plugin is on; see __init__.py).

Target-only decoder layer (apply): both norms of every layer are 1Cat's fused all-reduce + residual add + Gemma
        RMSNorm (vllm.sm70_tp4_all_reduce_gemma_rms_norm, the row-push kernel 1Cat's speculative route compiles in; it
        falls back by itself above 64 rows) with an FP32 residual; the mixer output projection and the MLP down_proj
        stop reducing (the last layer's down_proj keeps its all-reduce for the final norm); the GDN mixer output is
        taken as a value; the NVFP4 MLP is one opaque op (q27_mlp). arnorm_bench.py (attic), us per call against
        1Cat's all-reduce + our fused norm: 1 row 7.08 vs 7.43, 8: 6.76 vs 9.11, 16: 7.58 vs 11.34, 32: 9.64 vs 15.23,
        48: 12.30 vs 21.91, 64: 15.87 vs 26.52. Under MTP 1Cat's own decoder forward runs (its verifier route), and the
        layer patch leaves it alone.
NVFP4 MLP (apply_mlp): the MLP layers run csrc/mlp_fp4.cu; their weights are packed from the checkpoint tensors at load
        and are the only copy (1Cat's TurboMind/QPN2 preparation is skipped and the checkpoint tensors freed). Under
        1Cat's decoder forward (MTP) Qwen2MoeMLP.forward reaches the same kernels (q27_mlp_local), with the all-reduce
        left in the traced graph.
FP8 projections (apply_fp8): GDN in_proj_qkvz / out_proj, attention qkv_proj / o_proj and the FP8 MLP down_proj run
        csrc/fp8_lin.cu on 1Cat's own QPN8-prepared codes (no second copy) for 17..32 and 41..64 rows, one pass over the
        weights; 1Cat's QPN8 kernels otherwise.
Attention prep (apply_qkprep): the full-attention layers use 1Cat's fused preparation op (sm70_qwen38_qk_norm_rope_cache:
        q/k RMSNorm, interleaved MRoPE, output-gate split and the FP8 E4M3 KV-cache write in one Triton kernel), which
        1Cat enables by itself only with a speculative verifier config (five kernels otherwise, ~39 us per layer).
GDN (apply_gdn): everything between a GDN layer's qkvz projection and its output projection is one graph-splitting op,
        q27_gdn_core: pure-decode batches run csrc/gdn_core.cu (b/a projection, conv update, state update and gated
        norm in one launch); other batches run gdn_glue.cu's b/a projection, 1Cat's recurrent core and gdn_glue.cu's
        gated norm. Under MTP only draft-free decode steps take it (after 1Cat's replay commit).
Decode attention (apply_attn): full-attention decode (1Cat's shipped E4M3 long-context operator: split-KV partial
        kernel + combine kernel) runs csrc/attn_dec.cu for batches of 2 or more: one kernel, K and V streamed straight
        into mma fragments, 6 query heads per KV head, the combine done by the splits themselves.
"""
import logging
import re
import torch

log = logging.getLogger("q27fuse")


def _on(knob: str) -> bool:      # [merge] per-feature opt-out knobs of the merged tree (unset = on, 0 = off)
    import os
    return (os.environ.get(knob, "") or "1").strip() != "0"


def apply():
    from vllm.model_executor.models import qwen3_5
    from vllm.distributed import get_tensor_model_parallel_world_size
    from . import ops

    ops.register_op()
    apply_mlp()
    apply_fp8()
    apply_qkprep()
    apply_attn()
    if _on("Q27_ATTNV"):         # [merge] opt-out knob Q27_ATTNV=0 (p14's attention routing)
        from . import attnv      # [attnv] verify rows / drafter first pass / B = 1 decode -> csrc/attn_mq.cu
        attnv.apply()            # [attnv]
    # q27_gdn_core reads each batch's GDN metadata, so it must split the compiled graph like 1Cat's GDN core ops (the
    # plugin loads before the engine config is built, so the default splitting list picks it up)
    from vllm.config.compilation import CompilationConfig
    if "vllm::q27_gdn_core" not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append("vllm::q27_gdn_core")
    apply_gdn()
    # >>> q27 gdnv: MTP GDN verify/decode core with our own replay records (gdnv.py, csrc/gdn_verify.cu); wraps the
    # GDN layer patched by apply_gdn
    if _on("Q27_GDNV"):          # [merge] opt-out knob Q27_GDNV=0 (1Cat's GDN verify op and replay records)
        from . import gdnv
        gdnv.apply()
    # <<< q27 gdnv
    from . import smallrows      # smallrows: MLP variants, GDN in-projection, FP8 9..64-row table + gated gate_up
    smallrows.apply()
    from . import pf2            # [prefill2] prefill paths above the graph range (pf2.py; opt-out knobs Q27_PF_*)
    pf2.apply()                  # [prefill2]
    L = qwen3_5.Qwen3_5DecoderLayer
    if getattr(L, "_q27_patched", False):
        return
    orig_init = L.__init__

    def __init__(self, vllm_config, layer_type, prefix=""):
        orig_init(self, vllm_config, layer_type, prefix)
        cfg = vllm_config.model_config.hf_text_config
        # 1Cat's speculative route (sm70_dflash2_direct_attention_output: MTP / DFlash2 at TP 2/4) keeps its own forward
        self._q27_on = (cfg.model_type == "qwen3_5_text" and cfg.hidden_size == 5120 and not self.layer_scale
                        and vllm_config.model_config.dtype == torch.float16
                        and not getattr(self, "sm70_dflash2_direct_attention_output", False))
        tp = get_tensor_model_parallel_world_size()
        if self._q27_on:
            if tp != 4:
                raise RuntimeError(f"q27fuse supports TP4 only (tensor parallel size {tp}); unset Q27_FUSE")
            from vllm.distributed import get_tp_group
            self._q27_group = get_tp_group().unique_name
            proj = self.linear_attn.out_proj if layer_type == "linear_attention" else self.self_attn.o_proj
            proj.reduce_results = False
            if self.layer_idx != cfg.num_hidden_layers - 1:
                self.mlp.down_proj.reduce_results = False
            else:   # [prefill3] the model's final norm reads the WHOLE residual after this layer's post-attention boundary,
                from . import pf2   # so pf2's prefill boundary must not leave it row-sharded there (target-only route)
                pf2.FULL_RESIDUAL.append(self.post_attention_layernorm)
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 \
            import CompressedTensorsW4A4Fp4
        self._q27_mlp = False
        if isinstance(getattr(self.mlp.gate_up_proj, "scheme", None), CompressedTensorsW4A4Fp4) and isinstance(
                getattr(self.mlp.down_proj, "scheme", None), CompressedTensorsW4A4Fp4):
            ops._MLPS[self.layer_idx] = self.mlp
            from . import pf3                                        # [prefill3] Job 5: the norm feeding this MLP
            pf3.POSTNORMS[self.layer_idx] = self.post_attention_layernorm
            # with 1Cat's own layer forward (MTP's verifier route) the sole-copy weights are reached through
            # Qwen2MoeMLP.forward (apply_mlp)
            self.mlp._q27_idx = self.layer_idx
            self._q27_mlp = self._q27_on
        if self.layer_idx == 0:
            log.warning("q27fuse: Qwen3.8 decoder layer %s: %s (tp=%d)", prefix,
                        "fused all-reduce + norm route" if self._q27_on else "1Cat's forward", tp)

    def forward(self, hidden_states, residual, positions=None, **kwargs):
        if not self._q27_on:
            return L._q27_orig_forward(self, hidden_states, residual, positions=positions, **kwargs)
        if residual is None:
            residual = hidden_states.float()               # the fused op takes an FP32 residual
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm.weight, self.input_layernorm.variance_epsilon,
                self._q27_group)
        if self.layer_type == "linear_attention":
            # take the GDN output as a value: passing a preallocated output buffer costs one extra copy kernel per
            # layer (the buffer crosses the piecewise split at the recurrent core, so inductor must write it)
            out = self.linear_attn(hidden_states=hidden_states, output=None)
        else:
            out = torch.empty_like(hidden_states)
            self.self_attn(hidden_states=hidden_states, output=out, positions=positions)
        hidden_states, residual = torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm(
            out, residual, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon,
            self._q27_group)
        if self._q27_mlp:          # all-reduces only in the last layer (down_proj.reduce_results)
            return torch.ops.vllm.q27_mlp(hidden_states, self.layer_idx), residual
        return self.mlp(hidden_states), residual

    L._q27_orig_forward = L.forward
    L.__init__ = __init__
    L.forward = forward
    L._q27_patched = True


def apply_gdn():
    """Replace Qwen3_5GatedDeltaNet.forward_cuda on eligible layers with the q27_gdn_core route."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as G
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.models import qwen3_5
    A = qwen3_5.Qwen3_5GatedDeltaNet  # overrides forward_cuda of QwenGatedDeltaNetAttention
    if getattr(A, "_q27_patched", False):
        return
    orig_init, orig_fc = A.__init__, A.forward_cuda

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config
        vc = get_current_vllm_config()
        pol = self.gdn_policy.projection
        heads = self.num_v_heads // self.tp_size
        self._q27_heads = heads
        self._q27_qkv = (self.key_dim * 2 + self.value_dim) // self.tp_size
        why = None
        sc = vc.speculative_config
        # MTP with 1Cat's replayed GDN verification: one committed state slot per request, as in target-only decode
        # (the conv slot has 3 + K rows, of which a 1-token step uses 0..2). Draft-free decode steps can then run the
        # fused core (after 1Cat's replay commit, ops.gdn_core_impl); verify, prefill and mixed steps keep 1Cat's route.
        self._q27_spec = bool(sc is not None and getattr(sc, "method", None) == "mtp"
                              and getattr(self, "_spec_replay", False))
        if sc is not None and not self._q27_spec:
            why = "speculative config"
        elif sc is None and getattr(self, "_sm70_spec_cache_stride", 1) != 1:
            why = "spec cache stride"
        elif self.gqa_interleaved_layout or self.disable_tp_for_ba_proj:
            why = "layout"
        elif not (self.head_k_dim == self.head_v_dim == 128 and self.hidden_size == 5120):
            why = "shape"
        elif (not getattr(self, "use_split_input_projections", False) or not isinstance(self.in_proj_ba, LinearBase)
              or getattr(self.in_proj_ba, "bias", None) is not None):
            why = "ba projection"
        elif getattr(self, "enable_sm70_gdn_qpn8_ba_split", False) or getattr(self, "enable_sm70_gdn_rmsnorm_onepass",
                                                                             False):
            why = "1Cat GDN projection knobs set"
        elif not (self.norm.group_size is None and self.norm.norm_before_gate
                  and self.norm.activation in ("silu", "swish")):
            why = "norm"
        elif any(getattr(pol, f, False) for f in ("input_core", "input_projection", "output_projection",
                                                   "context_core", "full_forward")):
            why = "projection policy"
        elif getattr(self, "sm70_qwen38_fp16_fused_input", False):
            why = "fp16 fused input"
        self._q27_gdn = why is None
        if why and not getattr(A, "_q27_warned", False):
            A._q27_warned = True
            log.warning("q27fuse: GDN route disabled for %s (%s)", self.prefix, why)

    def forward_cuda(self, hidden_states, output):
        if (not self._q27_gdn or self.in_proj_ba.weight.dtype != torch.float16
                or getattr(self, "input_projection_override", None) is not None
                or getattr(self, "output_projection_override", None) is not None
                or (not self._q27_spec and getattr(self, "enable_sm70_dflash2_fused_gdn_norm", False))):
            return orig_fc(self, hidden_states, output)
        n = hidden_states.size(0)
        if self._q27_spec:
            # MTP: runs eagerly inside 1Cat's qwen_gdn_full_forward op, so the route is chosen per step (and per
            # captured FULL graph): only pure draft-free decode batches take the fused core
            from vllm.forward_context import get_forward_context
            amd = get_forward_context().attn_metadata
            md = amd.get(self.prefix) if isinstance(amd, dict) else None
            if not (md is not None and getattr(md, "num_spec_decodes", 0) == 0 and md.spec_sequence_masks is None
                    and md.num_prefills == 0 and 0 < md.num_decodes <= n):
                return orig_fc(self, hidden_states, output)
            if not getattr(A, "_q27_spec_logged", False):
                A._q27_spec_logged = True
                log.warning("q27fuse: MTP draft-free decode steps use the fused GDN core (%s, %d rows)", self.prefix, n)
        layer_name = G._encode_layer_name(self.prefix)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        y = torch.empty((n, self._q27_heads * 128), dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops.vllm.q27_gdn_core(hidden_states, mixed_qkvz, y, layer_name)
        proj, _ = self.out_proj(y)
        if output is not None:
            output[:n] = proj
        return proj

    A.__init__ = __init__
    A.forward_cuda = forward_cuda
    A._q27_patched = True


def apply_mlp():
    """Pack every NVFP4 MLP projection for mlp_fp4.cu from its checkpoint tensors in place of 1Cat's own weight
    processing (which would repack them for TurboMind/QPN2); our layout is the only copy."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w4a4_nvfp4 as S
    from vllm.model_executor.models import qwen2_moe
    from torch.nn import Parameter
    from . import ops
    if not getattr(qwen2_moe.Qwen2MoeMLP, "_q27_patched", False):
        orig_mlp_forward = qwen2_moe.Qwen2MoeMLP.forward

        def mlp_forward(self, x):
            # a layer whose forward is 1Cat's (MTP) still runs our kernels; the all-reduce stays outside the op, in the
            # traced graph
            idx = getattr(self, "_q27_idx", None)
            if idx is None or not getattr(self.gate_up_proj, "q27_sole", False):
                return orig_mlp_forward(self, x)
            out = torch.ops.vllm.q27_mlp_local(x, idx)
            if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
                from vllm.distributed import tensor_model_parallel_all_reduce
                out = tensor_model_parallel_all_reduce(out)
            return out

        qwen2_moe.Qwen2MoeMLP.forward = mlp_forward
        qwen2_moe.Qwen2MoeMLP._q27_patched = True
    cls = S.CompressedTensorsW4A4Fp4
    if getattr(cls, "_q27_patched", False):
        return
    orig = cls.process_weights_after_loading
    pat = re.compile(r".*\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)$")
    counter = {"n": 0}

    def process_weights_after_loading(self, layer):
        mm = pat.match(getattr(layer, "prefix", "") or "")
        w = getattr(layer, "weight_packed", None)
        if mm and w is not None and w.is_cuda and w.dtype == torch.uint8 and w.dim() == 2:
            ops.mlp_prepack(layer, w.data, layer.weight_scale.data, layer.weight_global_scale.data,
                            mm.group(2) == "gate_up_proj")
            counter["n"] += 1
            if counter["n"] == 1:
                log.warning("q27fuse: packing NVFP4 MLP weights for mlp_fp4.cu (%s: %s, scale %s); sole copy, 1Cat's "
                            "TurboMind/QPN2 preparation skipped", layer.prefix, tuple(w.shape),
                            tuple(layer.weight_scale.shape))
            dev = w.device
            layer.weight_packed = Parameter(torch.empty(0, dtype=torch.uint8, device=dev), requires_grad=False)
            layer.weight_scale = Parameter(torch.empty(0, dtype=torch.float8_e4m3fn, device=dev), requires_grad=False)
            layer.q27_sole = True
            return None
        return orig(self, layer)

    cls.process_weights_after_loading = process_weights_after_loading
    cls._q27_patched = True


def apply_fp8():
    """Route the channel-FP8 projections that 1Cat serves with QPN8 (no batch TurboMind copy) through q27_fp8_linear."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a16_fp8 as S
    cls = S.CompressedTensorsW8A16Fp8
    if getattr(cls, "_q27_fp8_patched", False):
        return
    pat = re.compile(r".*\.(in_proj_qkvz|qkv_proj|out_proj|o_proj|down_proj)$")
    orig_pw, orig_aw = cls.process_weights_after_loading, cls.apply_weights
    counter = {"n": 0}

    def process_weights_after_loading(self, layer):
        r = orig_pw(self, layer)
        w, sc = getattr(layer, "weight", None), getattr(layer, "weight_scale_inv", None)
        if (pat.match(getattr(layer, "prefix", "") or "") and getattr(layer, "sm70_fp8_qpn8", False)
                and getattr(layer, "sm70_fp8_turbomind", False) and not getattr(layer, "sm70_fp8_batch_tm", False)
                and w is not None and sc is not None and w.dtype == torch.uint8 and w.dim() == 2
                and sc.dtype == torch.float16 and sc.numel() == w.shape[1] and w.shape[1] % 64 == 0
                and w.shape[0] % 32 == 0):
            layer.q27_fp8 = True
            counter["n"] += 1
            if counter["n"] == 1:
                log.warning("q27fuse: FP8 projections on fp8_lin.cu over 1Cat's QPN8 codes (%s: codes %s, scales %s)",
                            layer.prefix, tuple(w.shape), tuple(sc.shape))
        return r

    def apply_weights(self, layer, x, bias=None, output=None):
        if getattr(layer, "q27_fp8", False) and bias is None and x.dtype == torch.float16:
            n = layer.output_size_per_partition
            out_shape = (*x.shape[:-1], n)
            x2 = x.reshape(-1, x.shape[-1])
            if x2.stride(-1) != 1:
                x2 = x2.contiguous()
            out2 = (output.reshape(-1, n) if output is not None
                    else torch.empty((x2.shape[0], n), device=x.device, dtype=x.dtype))
            torch.ops.vllm.q27_fp8_linear(out2, x2, layer.weight, layer.weight_scale_inv,
                                          int(layer.sm70_fp8_qpn8_split_k), int(layer.sm70_fp8_qpn8_nacc),
                                          bool(layer.sm70_fp8_qpn8_prefetch), self.native_ops.arguments)
            return out2.reshape(out_shape)
        return orig_aw(self, layer, x, bias, output)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    cls._q27_fp8_patched = True


def apply_qkprep():
    """Enable 1Cat's fused q/k norm + MRoPE + gate + FP8 KV-cache op (Qwen3NextAttention.sm70_qk_rope_prep) without a
    speculative config. Same eligibility test as 1Cat's constructor, minus the verifier-config term."""
    from vllm.model_executor.models import qwen3_next
    from vllm.platforms import current_platform
    from vllm.config import get_current_vllm_config
    cls = qwen3_next.Qwen3NextAttention
    if getattr(cls, "_q27_qkprep_patched", False):
        return
    orig_init = cls.__init__

    def __init__(self, config, model_config=None, cache_config=None, quant_config=None, prefix=""):
        orig_init(self, config, model_config, cache_config, quant_config, prefix)
        if self.sm70_qk_rope_prep:
            return
        rope = self.rotary_emb
        ok = (current_platform.is_cuda()
              and current_platform.is_device_capability(70, device_id=torch.accelerator.current_device_index())
              and model_config is not None and model_config.dtype == torch.float16
              and config.model_type == "qwen3_5_text" and self.attn_output_gate and self.num_kv_heads > 0
              and self.num_heads == 6 * self.num_kv_heads and self.head_dim == 256
              and getattr(rope, "rotary_dim", None) == 64 and getattr(rope, "mrope_interleaved", False)
              and getattr(rope, "mrope_section", None) == [11, 11, 10]
              and rope.cos_sin_cache.dtype == torch.float16 and not self.attn.calculate_kv_scales
              and self.attn.attn_backend.get_name() == "FLASH_ATTN_V100"
              and self.attn.kv_sharing_target_layer_name is None
              and get_current_vllm_config().speculative_config is None)
        if ok:
            from vllm.model_executor.layers.attention import sm70_qwen38_qk_rope  # noqa: F401 (registers the op)
            self.sm70_qk_rope_prep = True
            if self.layer_idx == 3:
                log.warning("q27fuse: fused attention prep (1Cat sm70_qwen38_qk_norm_rope_cache) on %s", prefix)
        elif self.layer_idx == 3:
            log.warning("q27fuse: %s is not eligible for the fused attention prep; unfused prep kept", prefix)

    cls.__init__ = __init__
    cls._q27_qkprep_patched = True


def apply_attn():
    """Route vllm/v1/attention/ops/sm70_e4m3_decode.py's call of run_six_head_groups (1Cat's E4M3 decode operator,
    looked up as a module global at each call) to attn_dec.cu when the shapes qualify; 1Cat's operator otherwise."""
    from vllm.v1.attention.ops import sm70_e4m3_decode as D
    from . import ops
    if getattr(D, "_q27_attn_patched", False):
        return
    orig = D.run_six_head_groups
    ops.attn_ext()
    seen = {"n": 0}

    def run_six_head_groups(operator, q, k, v, out, table, seq_lens, partial, lse, scale, k_scale, v_scale,
                            *rest, **kw):
        B = q.shape[0]
        if (not rest and B >= ops.ATTN_MIN_B and q.dtype == torch.float16 and q.is_contiguous() and q.dim() == 3
                and q.shape[2] == 256 and k.dim() == 4 and k.dtype == torch.uint8 and v.dtype == torch.uint8
                and k.shape == v.shape and k.shape[3] == 256 and k.stride(3) == 1 and v.stride(3) == 1
                and q.shape[1] == 6 * k.shape[2] and k.shape[1] % 32 == 0 and table.dtype == torch.int32
                and table.dim() == 2 and table.stride(1) == 1 and seq_lens.dtype == torch.int32
                and seq_lens.is_contiguous() and out.is_contiguous() and out.shape == q.shape):
            ng, S, mt = ops.attn_cfg(B)
            po, pml, cnt = ops.attn_workspace(q.device)
            ops.attn_ext().attn_decode(q, k, v, table, seq_lens, out, po, pml, cnt, float(scale), float(k_scale),
                                       float(v_scale), ng, S, mt)
            if seen["n"] == 0:
                seen["n"] = 1
                log.warning("q27fuse: decode attention on attn_dec.cu (B=%d, page %d, kv heads %d, cfg %s)",
                            B, k.shape[1], k.shape[2], (ng, S, mt))
            return out
        return orig(operator, q, k, v, out, table, seq_lens, partial, lse, scale, k_scale, v_scale, *rest, **kw)

    D.run_six_head_groups = run_six_head_groups
    D._q27_attn_patched = True
