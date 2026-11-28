# SPDX-License-Identifier: Apache-2.0
"""Inkling2 DSpark 7-token speculative draft model (`mtp.safetensors`) for Volta sm_70.

Runs natively on 1Cat-vLLM v1.5.4 with:
- Pure FP16 weights + FP32 internal accumulations (zero bf16).
- Native `FLASH_ATTN_V100` non-causal paged draft attention with mixed SWA-1024
  (layers 0, 2) and full (layer 1) KV cache groups and FULL CUDA graph support.
- Shared target `embed_tokens`, `embed_norm`, and `lm_head` (zero weight duplication).
- Volta TP-local Markov argmax eliminating 7x full-vocab all-gathers per step.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .configs import Inkling2DSparkConfig
from .ops.dspark_markov import markov_sample as _shard_markov_sample
from .ops.dspark_sm70 import (
    add_softcap,
    markov_gate,
    mix_rmsnorm,
    qk_norm_rope_,
    rope_,
    silu_and_mul,
    tp_local_markov_argmax,
)
from .ops.fp8_dense_cuda import fp8_qpn8_linear, get_fp8_safetensors


def _extract_dspark_config(hf_config) -> Inkling2DSparkConfig:
    if isinstance(hf_config, Inkling2DSparkConfig):
        return hf_config
    inner = getattr(hf_config, "model", None)
    if isinstance(inner, Inkling2DSparkConfig):
        return inner
    # EAGLEConfig copies all attributes from inner.to_dict() onto itself.
    fields = (
        "block_size",
        "target_layer_ids",
        "vocab_size",
        "padded_vocab_size",
        "unpadded_vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "local_layer_ids",
        "sliding_window_size",
        "markov_rank",
        "markov_head_type",
        "final_logit_softcapping",
        "dflash_config",
    )
    kwargs = {k: getattr(hf_config, k) for k in fields if hasattr(hf_config, k)}
    return Inkling2DSparkConfig(**kwargs)


class Inkling2DraftSelfAttention(nn.Module):
    """Non-causal block attention over pre-inserted target-feature context K/V and query block."""

    def __init__(
        self,
        config: Inkling2DSparkConfig,
        is_local: bool,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= config.num_key_value_heads
        hidden = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.is_local = is_local
        self.sliding_window = config.sliding_window_size if is_local else None

        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.qkv_proj",
        )
        self.kv_proj_ctx = MergedColumnParallelLinear(
            hidden,
            [config.num_key_value_heads * self.head_dim] * 2,
            bias=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.kv_proj_ctx",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=False,
            reduce_results=True,
            params_dtype=torch.float16,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=self.eps, dtype=torch.float16)
        self.k_norm = RMSNorm(self.head_dim, eps=self.eps, dtype=torch.float16)
        self.k_norm_ctx = RMSNorm(self.head_dim, eps=self.eps, dtype=torch.float16)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
        )
        # Declare non-causal DFlash contract for 1Cat's FLASH_ATTN_V100 backend.
        self.attn.is_dflash_draft_attn = True
        self.attn.dflash_expected_causal = False
        self.attn.dflash_expected_sliding_window = self.sliding_window
        self.attn.dflash_rope_is_neox_style = True

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        qk_norm_rope_(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            self.num_heads,
            self.num_kv_heads,
            self.eps,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_out = self.attn(q, k, v)
        out, _ = self.o_proj(attn_out)
        return out

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        kv, _ = self.kv_proj_ctx(context)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        k = self.k_norm_ctx(k.reshape(-1, self.head_dim))
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        rope_(k, positions, cos_sin_cache)
        if slot_mapping is not None and self.attn.kv_cache.numel() > 0:
            self.attn.impl.do_kv_cache_update(
                self.attn,
                k,
                v.view(-1, self.num_kv_heads, self.head_dim),
                self.attn.kv_cache,
                slot_mapping,
            )


class Inkling2DraftMLP(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=True,
            params_dtype=torch.float16,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        hidden = silu_and_mul(gate_up)
        out, _ = self.down_proj(hidden)
        return out


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2DSparkConfig,
        is_local: bool,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden, eps=eps, dtype=torch.float16)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps, dtype=torch.float16)
        self.dflare_norm = RMSNorm(hidden, eps=eps, dtype=torch.float16)
        self.self_attn = Inkling2DraftSelfAttention(
            config, is_local, cache_config, f"{prefix}.self_attn"
        )
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(positions, x, cos_sin_cache)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (`rnn_hidden`) correction of the draft's base logits."""

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size,
            rank,
            _weight=torch.empty(config.padded_vocab_size, rank, dtype=torch.float16),
            _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(
            2 * rank + 2 * hidden, 3 * rank, bias=True, params_dtype=torch.float16
        )
        self.hidden_out = ReplicatedLinear(
            rank, hidden, bias=True, params_dtype=torch.float16
        )
        self.dflare_norm = RMSNorm(hidden, eps=config.rms_norm_eps, dtype=torch.float16)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None
        self.rnn_w_step_t: torch.Tensor | None = None
        self.rnn_w_ctx_t: torch.Tensor | None = None
        self.w_state: torch.Tensor | None = None

    @torch.no_grad()
    def fold(
        self,
        lm_head_weight: torch.Tensor,
        *,
        fp8_w8: torch.Tensor | None = None,
        fp8_s32: torch.Tensor | None = None,
    ) -> None:
        """Fold `hidden_out` into the TP-local LM head rows in FP64 -> FP16."""
        projection = self.hidden_out.weight.double()
        bias = self.hidden_out.bias.double()
        dev = projection.device
        num_rows = fp8_w8.shape[0] if fp8_w8 is not None else lm_head_weight.shape[0]
        if self.fold_weight is None or self.fold_bias is None:
            self.fold_weight = torch.empty(
                (num_rows, self.markov_rank), dtype=torch.float16, device=dev
            )
            self.fold_bias = torch.empty((num_rows,), dtype=torch.float16, device=dev)
        for start in range(0, num_rows, 4096):
            rows = slice(start, start + 4096)
            if fp8_w8 is not None and fp8_s32 is not None:
                w_c = fp8_w8[rows].to(dev).float()
                s_c = fp8_s32[rows].to(dev).float()
                head = (w_c * s_c[:, None]).double()
            else:
                head = lm_head_weight[rows].double()
            self.fold_weight[rows].copy_((head @ projection).to(torch.float16))
            self.fold_bias[rows].copy_((head @ bias).to(torch.float16))
        # Cache contiguous slices of rnn_proj.weight for cuBLAS addmm & shard-local Markov kernel.
        w = self.rnn_proj.weight
        w_state = w[:, : 2 * self.markov_rank].contiguous()
        w_step_t = w_state.t().contiguous()
        w_ctx_t = w[:, 2 * self.markov_rank :].t().contiguous()
        if self.rnn_w_step_t is None:
            self.w_state = w_state
            self.rnn_w_step_t = w_step_t
            self.rnn_w_ctx_t = w_ctx_t
        else:
            assert self.w_state is not None
            self.w_state.copy_(w_state)
            self.rnn_w_step_t.copy_(w_step_t)
            assert self.rnn_w_ctx_t is not None
            self.rnn_w_ctx_t.copy_(w_ctx_t)

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """FP32 `rnn_proj` bias plus `[hidden, f_rnn]` term -> `(n_spec, num_reqs, 3 * r)`."""
        assert self.rnn_bias is not None and self.rnn_w_ctx_t is not None
        num_reqs, width = f_rnn.shape
        hidden_3d = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden_3d, f_rnn.unsqueeze(0).expand_as(hidden_3d)], dim=-1)
        out = torch.addmm(
            self.rnn_bias,
            inputs.flatten(0, 1),
            self.rnn_w_ctx_t,
            out_dtype=torch.float32,
        )
        return out.view(hidden_3d.shape[0], num_reqs, -1)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; returns `(new_state, tp_local_bias)` in FP16."""
        assert (
            self.fold_weight is not None
            and self.fold_bias is not None
            and self.rnn_w_step_t is not None
        )
        inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
        raw = torch.addmm(context, inputs, self.rnn_w_step_t, out_dtype=torch.float32)
        state, latent = markov_gate(raw, state)
        bias = torch.addmm(self.fold_bias, latent, self.fold_weight.t())
        return state, bias


class Inkling2DSparkDecoder(nn.Module):
    def __init__(
        self,
        config: Inkling2DSparkConfig,
        max_model_len: int,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(
                config, i in local_ids, cache_config, f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(hidden, eps=config.rms_norm_eps, dtype=torch.float16)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = nn.Parameter(
            torch.zeros(
                config.num_hidden_layers + 1,
                len(config.target_layer_ids),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(
            torch.zeros(hidden, dtype=torch.float16), requires_grad=False
        )
        self.register_buffer(
            "fuse_alpha",
            torch.empty_like(self.fuse_weight, dtype=torch.float32),
            persistent=False,
        )
        # FP32 [cos | sin] table read directly by dspark_sm70 RoPE kernels.
        self.cos_sin_cache = get_rope(
            config.head_dim,
            max_position=max_model_len,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            dtype=torch.float32,
        ).cos_sin_cache
        # Shared with target model via load_dflash_model + attach_target.
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: RMSNorm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.variance_epsilon)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalized target embedding of each block's anchor (cast to FP16), then masks."""
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.config.draft_block_size
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        anchor = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous()))
        x[:, 0].copy_(anchor.to(x.dtype))
        return x.view(-1, self.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            assert input_ids is not None
            x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin_cache)
        assert residual is not None
        x, _ = self.norm(x, residual)
        return x


class Inkling2DSparkModel(nn.Module):
    """Native Inkling2 DSpark 7-token drafter registered as `DFlashInkling2DSparkModel`."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx.", 0),
        (".v_proj_ctx.", ".kv_proj_ctx.", 1),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        raw_cfg = vllm_config.speculative_config.draft_model_config.hf_config
        config = _extract_dspark_config(raw_cfg)
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        if num_spec != config.draft_block_size - 1:
            raise ValueError(
                "Inkling2 DSpark requires num_speculative_tokens = "
                f"{config.draft_block_size - 1}, got {num_spec}"
            )
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(
                f"max_model_len={vllm_config.model_config.max_model_len} exceeds "
                f"DSpark max_position_embeddings={config.max_position_embeddings}"
            )
        self.config = config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.vocab_shard_size = config.padded_vocab_size // self.tp_size
        self.model = Inkling2DSparkDecoder(
            config,
            vllm_config.model_config.max_model_len,
            vllm_config.cache_config,
            f"{prefix}model",
        )
        self.lm_head: nn.Module | None = None

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Bind target `embed_norm` and precompute folded FP16 Markov LM head weights."""
        assert self.lm_head is not None and self.lm_head.weight.dtype == torch.float16
        self.model.embed_norm = target_model.model.embed_norm
        # Keep cos_sin_cache on GPU for CUDA-graph safety.
        self.model.cos_sin_cache = self.model.cos_sin_cache.to(
            device=self.lm_head.weight.device, dtype=torch.float32
        )
        head = self.model.markov_head
        head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)
        self.refresh_derived_weights()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        assert self.lm_head is not None
        head = self.model.markov_head
        assert head.rnn_bias is not None
        if getattr(self.lm_head, "_fp8_ready", False):
            sf = get_fp8_safetensors()
            assert sf is not None
            w_full = sf.get_tensor("model.llm.unembed.weight")
            s_full = sf.get_tensor("model.llm.unembed.weight_scale")
            n_sh = w_full.shape[0] // self.tp_size
            w_sh = w_full[self.tp_rank * n_sh : (self.tp_rank + 1) * n_sh]
            s_sh = s_full[self.tp_rank * n_sh : (self.tp_rank + 1) * n_sh]
            head.fold(self.lm_head.weight, fp8_w8=w_sh, fp8_s32=s_sh)
        else:
            head.fold(self.lm_head.weight)
        head.rnn_bias.copy_(head.rnn_proj.bias.float())
        self.model.fuse_alpha.copy_(
            torch.softmax(self.model.fuse_weight.float(), dim=-1)
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.model.layers)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_block(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            slot = None if context_slot_mappings is None else context_slot_mappings[i]
            layer.self_attn.store_context_kv(
                context, context_positions, self.model.cos_sin_cache, slot
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.mix(features, -1, self.model.markov_head.dflare_norm)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_base_logits(
        self, hidden_states: torch.Tensor, *, out_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        assert self.lm_head is not None
        if getattr(self.lm_head, "_fp8_ready", False):
            return fp8_qpn8_linear(
                hidden_states,
                self.lm_head._fp8_codes,
                self.lm_head._fp8_s16,
                self.lm_head._fp8_s32,
                self.lm_head._fp8_n_out,
                out_dtype=out_dtype,
                is_lm_head=True,
            )
        if out_dtype == torch.float32 and hidden_states.dtype == torch.float16:
            return torch.mm(hidden_states, self.lm_head.weight.t(), out_dtype=torch.float32)
        return F.linear(hidden_states, self.lm_head.weight)

    def markov_sample(
        self,
        hidden_states: torch.Tensor,
        f_rnn: torch.Tensor,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        anchor_stride: int,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dense: torch.Tensor | None,
        num_spec: int,
        use_fp64: bool,
    ) -> None:
        """Shard-local recurrent Markov head sampling (both greedy and probabilistic)."""
        head = self.model.markov_head
        assert head.fold_weight is not None and head.fold_bias is not None and head.w_state is not None
        shard = head.fold_weight.shape[0]
        id_base = self.tp_rank * shard
        num_valid = min(max(self.config.vocab_size - id_base, 0), shard)
        _shard_markov_sample(
            self.compute_base_logits(hidden_states, out_dtype=torch.float32),
            head.context(hidden_states, f_rnn),
            head.w_state,
            head.markov_w1.weight,
            head.fold_weight,
            head.fold_bias,
            tokens,
            anchors,
            anchor_stride,
            idx_map,
            sample_pos,
            temperature,
            seeds,
            dense,
            num_spec,
            num_valid,
            id_base,
            self.tp_rank,
            self.tp_size,
            float(self.config.final_logit_softcapping),
            use_fp64,
            tensor_model_parallel_all_reduce,
        )

    def gather_dense_logits(self, dense: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return dense
        return tensor_model_parallel_all_gather(dense)

    def markov_step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.markov_head.step(state, prev_tokens, context)

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        logits = add_softcap(base_logits, bias, self.config.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size]

    def markov_argmax(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return tp_local_markov_argmax(
            base_logits,
            bias,
            tp_rank=self.tp_rank,
            vocab_shard_size=self.vocab_shard_size,
            org_vocab_size=self.config.vocab_size,
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        loaded_shards: dict[str, set] = {}
        expected_shards = {
            ".qkv_proj.": 3,
            ".kv_proj_ctx.": 2,
            ".gate_up_proj.": 2,
        }
        for raw_name, weight in weights:
            name, shard = f"model.{raw_name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            if name not in params:
                raise ValueError(f"Unexpected DSpark tensor: {raw_name} -> {name}")
            param = params[name]
            if weight.dtype != param.dtype:
                weight = weight.to(param.dtype)
            if shard is None:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
            else:
                param.weight_loader(param, weight, shard)
                loaded_shards.setdefault(name, set()).add(shard)
        for name, seen in loaded_shards.items():
            for key, count in expected_shards.items():
                if key in name and len(seen) == count:
                    loaded.add(name)
                    break
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing DSpark checkpoint tensors: {sorted(missing)}")
        return loaded
