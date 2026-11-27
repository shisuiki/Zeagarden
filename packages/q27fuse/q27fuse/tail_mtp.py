"""q27fuse.tail_mtp - MTP drafter glue (track "tail", piece "mtp"; on by default with the plugin, Q27_TAIL_MTP=0 turns it
off). It changes the drafter's traced graph; the compile-cache key is tagged accordingly (below), so a cache written
without it (or by an older version of it) is never loaded with it, and vice versa.

1Cat's drafter input projection mtp.fc is ColumnParallelLinear(10240 -> 5120, gather_output=True): every rank computes
1280 output columns (a 40-CTA FP8 GEMV over 13 MB) and an NCCL all-gather (16-19 us traced, plus a reshape copy in pass
1) assembles the row; the MTP decoder layer then starts from residual=None, so its input norm is a separate kernel and,
because the residual stays FP16 until the post-attention norm, the o_proj all-reduce and that norm stay unfused too
(mtp-map 2.2). Here:
  fc     mtp.fc is RowParallelLinear over the same weight (each rank: all 5120 outputs over its 2560-input slice; 13 MB),
         the partial sums are all-reduced with the custom all-reduce, and the layer gets an FP32 zero residual: the
         compile pass then fuses that all-reduce with the input norm and the o_proj all-reduce with the post-attention
         norm into 1Cat's row-push all-reduce + Gemma RMSNorm (as in every target layer).
  emb    the embedding all-reduce of every draft pass (VocabParallelEmbedding: masked lookup + all-reduce) is followed by
         pre_fc_norm_embedding; called with an FP32 zero residual, the same compile pass fuses that all-reduce with the
         norm (one row-push launch instead of 1Cat's one-stage all-reduce + a separate norm kernel). The reduced row is
         exact either way (one rank holds the row, the others add zeros); the norm's rounding may differ.
Drafts change at the rounding level (fc summation order and per-slice FP8 channel scales, the fused norms); the target
is untouched.

Compile cache: vLLM's AOT artifact key for the drafter is hash(vLLM env vars, vllm config, version + qualname + first
line of the OUTER model's forward, Qwen3_5MTP.forward); the source check on load (_verify_source_unchanged) only covers
the files the stored artifact itself traced. Neither sees this patch, so without a tag a cache written by the unpatched
drafter is loaded with the patched weights (1Cat's QPN8 dispatch then fails on the fc shape, tail.md section 6) and a
cache written with the patch would be loaded without it. _tag_cache_key() mixes TAG (and any other string registered
with add_drafter_tag) into that key for the drafter only; the target's key is unchanged.
"""
import hashlib
import logging
import os

log = logging.getLogger("q27fuse")

TAG = "q27fuse.tail_mtp/fc-rowpar+emb-ar-norm/v2"
DRAFTER_TAGS = []     # every switch that changes the drafter's traced graph appends a string (add_drafter_tag)


def enabled():
    return os.environ.get("Q27_TAIL_MTP", "1").strip() not in ("", "0")


def add_drafter_tag(tag: str):
    """register a string that is mixed into the drafter's compile-cache key (call before the model is built); other
    plugin parts that change the drafter's traced graph (e.g. an op swap inside its forward) should use this too"""
    from vllm.model_executor.models import qwen3_5_mtp as M
    if tag not in DRAFTER_TAGS:
        DRAFTER_TAGS.append(tag)
    _tag_cache_key(M)


def _tag_cache_key(M):
    from vllm.compilation import decorators as D
    orig = D._model_hash_key
    if getattr(orig, "_q27_tail_mtp", False):
        return

    def _model_hash_key(fn):
        h = orig(fn)
        owner = getattr(fn, "__self__", None)
        if DRAFTER_TAGS and isinstance(owner, (M.Qwen3_5MTP, M.Qwen3_5MultiTokenPredictor)):
            h = hashlib.sha256((h + "|" + "|".join(DRAFTER_TAGS)).encode()).hexdigest()
        return h

    _model_hash_key._q27_tail_mtp = True
    D._model_hash_key = _model_hash_key


def apply():
    if not enabled():
        return
    import torch
    from vllm.model_executor.models import qwen3_5_mtp as M
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.distributed import tensor_model_parallel_all_reduce
    P = M.Qwen3_5MultiTokenPredictor
    if getattr(P, "_q27_tail_mtp", False):
        return
    add_drafter_tag(TAG)
    orig_init = P.__init__
    orig_fwd = P.forward
    force_mtp = os.environ.get("Q27_TAIL_MTP_FORCE", "0").strip() == "1"

    def __init__(self, *, vllm_config, prefix="", **kw):
        orig_init(self, vllm_config=vllm_config, prefix=prefix, **kw)
        if getattr(self, "_fc_input_sharded", False) and not force_mtp:
            self._q27_tail_mtp_skip = True
            log.warning("q27fuse tail_mtp: skipped on %s (1Cat v1.5.4 native _fc_input_sharded active)", prefix)
            return
        self._q27_tail_mtp_skip = False
        quant_config = vllm_config.quant_config
        fc_quant = None if (quant_config and quant_config.get_name() == "modelopt_fp4") else quant_config
        old = self.fc
        self.fc = RowParallelLinear(self.config.hidden_size * 2, self.config.hidden_size, bias=False,
                                    input_is_parallel=False, reduce_results=False, return_bias=False,
                                    quant_config=fc_quant, prefix=f"{prefix}.fc")
        del old
        M._mark_sm70_mtp_dense_f16_modules(self, M.layer_policy(vllm_config))
        log.warning("q27fuse tail: %s.fc row-parallel (partial sums + all-reduce fused with the input norm), "
                    "embedding all-reduce fused with pre_fc_norm_embedding; compile-cache key tagged %s", prefix, TAG)

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0):
        if getattr(self, "_q27_tail_mtp_skip", False):
            return orig_fwd(self, input_ids, positions, hidden_states, intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds, spec_step_idx=spec_step_idx)
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)          # masked lookup + all-reduce (fused below)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        zero = torch.zeros(inputs_embeds.shape, dtype=torch.float32, device=inputs_embeds.device)
        inputs_embeds, _ = self.pre_fc_norm_embedding(inputs_embeds, zero)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = tensor_model_parallel_all_reduce(self.fc(hidden_states))
        residual = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions, hidden_states=hidden_states, residual=residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    P.__init__ = __init__
    P.forward = forward
    P._q27_tail_mtp = True
