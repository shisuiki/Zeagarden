# SPDX-License-Identifier: Apache-2.0
"""Inkling2 model configs (ported for 1Cat-vLLM v1.5.4 / Volta sm_70)."""

from __future__ import annotations

from typing import Any, ClassVar

from transformers.configuration_utils import PreTrainedConfig

INKLING2_ARCHITECTURE = "Inkling2ForConditionalGeneration"
INKLING2_LOOP_ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
INKLING2_ARCHITECTURES = frozenset({INKLING2_ARCHITECTURE, INKLING2_LOOP_ARCHITECTURE})


class Inkling2ModelConfig(PreTrainedConfig):
    model_type = "inkling2_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 2304,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 1024,
        sconv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        model_max_length: int = 1048576,
        log_scaling_n_floor: int | None = 8192,
        log_scaling_alpha: float = 0.1,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        final_logit_softcapping: float | None = 30.0,
        n_routed_experts: int = 128,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 1152,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        loop_config: dict[str, int | bool] | None = None,
        loop_num_passes: int | None = None,
        loop_share_cache: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if pre_norm_pre_affine or final_norm_pre_affine:
            raise ValueError("Inkling2 pre-affine norms are not supported")
        self.padded_vocab_size = vocab_size
        self.vocab_size = unpadded_vocab_size or vocab_size
        self.unpadded_vocab_size = self.vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads or num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads or num_key_value_heads
        self.swa_head_dim = swa_head_dim or head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.local_layer_ids = list(local_layer_ids or [])
        self.sliding_window_size = sliding_window_size
        self.sconv_kernel_size = sconv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.final_logit_softcapping = final_logit_softcapping
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        if loop_config is not None:
            int_fields = {"core_start", "core_end", "num_passes", "gate_rank"}
            if (
                set(loop_config) != int_fields | {"is_varloop"}
                or any(type(loop_config[name]) is not int for name in int_fields)
                or type(loop_config["is_varloop"]) is not bool
            ):
                raise ValueError(
                    f"loop_config requires integer fields {sorted(int_fields)} "
                    "and a boolean is_varloop"
                )
        self.loop_config = loop_config
        self.loop_num_passes = loop_num_passes
        self.loop_share_cache = loop_share_cache
        if loop_config is not None:
            self.check_loop_options()
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def loop_core_layers(self) -> range:
        loop = self.loop_config
        return range(0) if loop is None else range(loop["core_start"], loop["core_end"])

    def check_loop_options(self) -> None:
        loop = self.loop_config
        varloop = loop is not None and loop["is_varloop"]
        if (self.loop_num_passes is not None or self.loop_share_cache) and not varloop:
            raise ValueError(
                "loop_num_passes and loop_share_cache need a checkpoint with "
                "loop_config.is_varloop=true"
            )
        if loop is not None and self.num_loop_passes < (1 if varloop else 2):
            raise ValueError(
                f"This loop checkpoint needs at least {1 if varloop else 2} passes, "
                f"got {self.num_loop_passes}"
            )

    @property
    def num_loop_passes(self) -> int:
        if self.loop_config is None:
            return 1
        if self.loop_num_passes is not None:
            return self.loop_num_passes
        return int(self.loop_config["num_passes"])

    @property
    def num_layer_visits(self) -> int:
        extra_passes = self.num_loop_passes - 1
        return self.num_hidden_layers + extra_passes * len(self.loop_core_layers)

    def layer_visit_ids(self, layer_id: int) -> list[int]:
        core, passes = self.loop_core_layers, self.num_loop_passes
        if not core or layer_id < core.start:
            return [layer_id]
        if layer_id >= core.stop:
            return [layer_id + (passes - 1) * len(core)]
        return [layer_id + p * len(core) for p in range(passes)]


class Inkling2MMConfig(PreTrainedConfig):
    model_type = "inkling2"
    architecture = INKLING2_ARCHITECTURE
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "text_config": Inkling2ModelConfig,
    }

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | Inkling2ModelConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config
            if isinstance(text_config, Inkling2ModelConfig)
            else Inkling2ModelConfig(**(text_config or {}))
        )
        if (self.text_config.loop_config is not None) != (
            self.model_type == "inkling2_loop"
        ):
            raise ValueError(
                "Looped Inkling2 checkpoints use model_type inkling2_loop with a "
                "loop_config; model_type inkling2 has none"
            )
        self.vision_config = vision_config
        # Keep LM head logits in FP32 before final_logit_softcapping=30.0.
        kwargs.setdefault("head_dtype", "float32")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        try:
            from .conv_state import mark_inkling_config_loaded

            mark_inkling_config_loaded()
        except Exception:
            pass

    def get_text_config(self, *args: Any, **kwargs: Any) -> Inkling2ModelConfig:
        return self.text_config


class Inkling2LoopMMConfig(Inkling2MMConfig):
    model_type = "inkling2_loop"
    architecture = INKLING2_LOOP_ARCHITECTURE
    has_no_defaults_at_init = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Inkling2DSparkConfig(PreTrainedConfig):
    """The DSpark draft of an Inkling2 checkpoint, from its `mtp_config`."""

    model_type = "inkling2_dspark"

    def __init__(
        self,
        *,
        block_size: int = 8,
        target_layer_ids: list[int] | None = None,
        vocab_size: int = 200058,
        padded_vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 2304,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 9216,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 1024,
        markov_rank: int = 256,
        markov_head_type: str = "rnn_hidden",
        final_logit_softcapping: float = 30.0,
        dflash_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.padded_vocab_size = padded_vocab_size
        self.vocab_size = unpadded_vocab_size or vocab_size
        self.unpadded_vocab_size = self.vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.local_layer_ids = list(local_layer_ids if local_layer_ids is not None else [0, 2])
        self.sliding_window_size = sliding_window_size
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.final_logit_softcapping = final_logit_softcapping
        self.block_size = block_size
        self.draft_block_size = block_size
        self.dspark_block_size = block_size - 1
        self.sample_from_anchor = False
        self.is_causal = False
        self.mask_token_id = 0
        self.target_layer_ids = list(target_layer_ids or [5, 23, 41, 48])
        self.eagle_aux_hidden_state_layer_ids = list(self.target_layer_ids)
        self.dflash_config = dict(
            dflash_config
            or {
                "block_size": block_size,
                "mask_token_id": 0,
                "is_causal": False,
                "sample_from_anchor": False,
                "target_layer_ids": list(self.target_layer_ids),
            }
        )
        kwargs.setdefault("architectures", ["DFlashInkling2DSparkModel"])
        kwargs.setdefault("torch_dtype", "float16")
        super().__init__(tie_word_embeddings=False, **kwargs)

    @classmethod
    def from_target(cls, config: Inkling2MMConfig) -> Inkling2DSparkConfig:
        mtp_config = getattr(config, "mtp_config", None)
        if not mtp_config:
            raise ValueError("The Inkling2 checkpoint has no mtp_config")
        text = config.text_config
        draft = {
            k: v
            for k, v in mtp_config.items()
            if k not in ("torch_dtype", "dtype", "architectures")
        }
        return cls(
            **draft,
            vocab_size=text.vocab_size,
            padded_vocab_size=text.padded_vocab_size,
            final_logit_softcapping=text.final_logit_softcapping or 30.0,
            torch_dtype="float16",
        )


class Inkling2DFly2Config(PreTrainedConfig):
    """Zero-KV-cache DFly2-KV 7-token speculative drafter config for Inkling-2-Mini."""

    model_type = "inkling2_dfly2_kv"

    def __init__(
        self,
        *,
        hidden_size: int = 2304,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        d_rel: int = 16,
        rel_extent: int = 1024,
        sliding_window_size: int = 1024,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        log_scaling_n_floor: int = 8192,
        log_scaling_alpha: float = 0.1,
        rms_norm_eps: float = 1e-6,
        final_logit_softcapping: float = 30.0,
        vocab_size: int = 200058,
        padded_vocab_size: int = 201024,
        target_kv_layers: list[int] | None = None,
        local_layer_ids: list[int] | None = None,
        block_size: int = 8,
        mtp_fusion: bool = True,
        enable_hidden_correction: bool = True,
        hidden_correction_intermediate_size: int = 2304,
        attentive_correction: bool = True,
        attentive_correction_block_gated: bool = True,
        attentive_correction_is_local: bool = True,
        causal_draft_attn: bool = False,
        max_position_embeddings: int = 1048576,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.sliding_window_size = sliding_window_size
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.final_logit_softcapping = final_logit_softcapping
        self.padded_vocab_size = padded_vocab_size
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = vocab_size
        self.target_kv_layers = list(target_kv_layers if target_kv_layers is not None else [20, 23, 22])
        self.local_layer_ids = list(local_layer_ids if local_layer_ids is not None else [0, 2])
        self.block_size = block_size
        self.draft_block_size = block_size
        self.mtp_fusion = mtp_fusion
        self.enable_hidden_correction = enable_hidden_correction
        self.hidden_correction_intermediate_size = hidden_correction_intermediate_size
        self.attentive_correction = attentive_correction
        self.attentive_correction_block_gated = attentive_correction_block_gated
        self.attentive_correction_is_local = attentive_correction_is_local
        self.causal_draft_attn = causal_draft_attn
        self.max_position_embeddings = max_position_embeddings
        self.is_causal = False
        self.mask_token_id = 0
        # Zero-cost sentinel aux layer (final output reference only) so vLLM V2
        # `method="dflash"` unpacks `(out, [out])` with zero extra tensor copies.
        self.target_layer_ids = [48]
        self.eagle_aux_hidden_state_layer_ids = [48]
        self.dflash_config = {
            "block_size": block_size,
            "mask_token_id": 0,
            "is_causal": False,
            "sample_from_anchor": False,
            "target_layer_ids": [48],
        }
        kwargs.setdefault("architectures", ["I2MDFly2DraftModel"])
        kwargs.setdefault("torch_dtype", "float16")
        super().__init__(tie_word_embeddings=False, **kwargs)
