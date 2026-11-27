# SPDX-License-Identifier: Apache-2.0
"""Inkling2 / Inkling2Loop model for Volta sm_70 on 1Cat-vLLM v1.5.3.

Design choices specific to Volta (sm_70):
- Residual `stack`, `bank`, `prelude_partial`, and `attn_out` stay strictly in FP32
  across all 48 logical layer visits, eliminating BF16->FP16 residual overflow.
- Casting to FP16 occurs ONLY at sublayer RMSNorm outputs (inputs to Tensor Core GEMMs).
- LM head outputs FP32 logits (via 1Cat's FP32 LM head / cuBLAS FP32 out) before
  `final_logit_softcapping = 30.0`.
- Text-only Phase 0 cleanly ignores `model.visual.*` weights when vision is disabled.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.sequence import IntermediateTensors

from .attention import Inkling2Attention
from .configs import Inkling2MMConfig, Inkling2ModelConfig
from .conv_state import (
    ATTN_STREAM,
    MLP_STREAM,
    Inkling2ShortConv,
    InklingConvState,
    conv_metadata,
)
from .moe import Inkling2MoE, QuantFormat
from .ops.allreduce_conv_mix_cuda import (
    all_reduce_conv_mix_cuda,
    init_tp4_ar_conv_mix,
    is_ar_conv_mix_ready,
)
from .ops.attention_sm70 import compute_log_scaling_tau
from .ops.attn_res import attn_res
from .ops.fp8_dense_cuda import (
    fp8_enabled_tiers,
    fp8_qpn8_linear,
    get_fp8_safetensors,
    load_fp8_dense_ext,
    prepare_qpn8_weight,
)
from .ops.loop import blend_rms_norm, pair_rms_norm
from .ops.loop_blend_cuda import (
    load_loop_blend_ext,
    loop_blend_cuda,
    preallocate_loop_blend_scratch,
)
from .ops.norm import rms_norm
from .quant_config import Inkling2QuantConfig

LOOP_STREAMS = ("attn", "k", "v", "mlp")


class Inkling2RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.float16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size, dtype=torch.float16), requires_grad=False
        )
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded.to(param.dtype))

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """FP32 pseudo-queries pre-scaled by `hidden_size**-0.5` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden), dtype=torch.float32
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.float().view(param.shape)).mul_(self.scale)


class Inkling2MessageController(nn.Module):
    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)
        for p in (self.down_weight, self.gain_weight, self.forget_weight):
            p.weight_loader = lambda param, loaded: param.data.copy_(loaded.to(param.dtype))


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
        num_passes: int = 1,
    ) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        self.conv_states = tuple(
            InklingConvState(
                num_kv_heads=num_kv_heads,
                head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                hidden_size=config.hidden_size,
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_state" + (f".pass{p}" if p else ""),
                replicate_hidden=True,
            )
            for p in range(num_passes)
        )
        self.conv_state = self.conv_states[0]
        self.attn = Inkling2Attention(
            config,
            self.is_local,
            self.conv_state,
            prefix=f"{prefix}.attn",
            pass_conv_states=self.conv_states[1:],
        )
        eps = config.rms_norm_eps
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(
            config.hidden_size, eps, out_dtype=torch.float16, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2RMSNorm(
            config.hidden_size, eps, out_dtype=torch.float16
        )
        expert_format: QuantFormat = (
            "mxfp8"
            if quant_config is not None and quant_config.is_mxfp8_layer(layer_id)
            else "nvfp4"
        )
        self.mlp = Inkling2MoE(config, expert_format, config.layer_visit_ids(layer_id))
        self.attn_sconv, self.mlp_sconv = (
            Inkling2ShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_local and len(inputs) == 1:
            return self.attn(positions, inputs[0], out=out, pass_idx=pass_idx)
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            out,
            pass_idx,
        )


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.float16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

        self._use_cuda_loop_blend = os.environ.get("INKLING_USE_CUDA_LOOP_BLEND", "1") != "0"

    def process_weights_after_loading(self) -> None:
        for i, controller in enumerate(self.controllers.values()):
            assert controller.down_weight.data_ptr() == self.controller_down[i].data_ptr()
            assert (
                controller.gain_weight.data_ptr()
                == self.controller_gain_forget[i].data_ptr()
            )
        if self._use_cuda_loop_blend:
            load_loop_blend_ext()
            preallocate_loop_blend_scratch(
                self.controller_down.device, self.controller_down.shape[2] // 2
            )

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]
        if self._use_cuda_loop_blend:
            return loop_blend_cuda(dest, message, down, up, norm_weight, norm_eps)
        num_tokens = dest.shape[1]
        if num_tokens <= 1024:
            pair = pair_rms_norm(dest, message, out_dtype=torch.float16)
            hidden = torch.bmm(pair, down.transpose(1, 2))
            del pair
            gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
            del hidden
            return blend_rms_norm(
                dest, message, gain_forget, norm_weight, norm_eps, out_dtype=torch.float16
            )
        out = message.new_empty(message.shape, dtype=torch.float16)
        for t0 in range(0, num_tokens, 1024):
            t1 = min(t0 + 1024, num_tokens)
            d_c = dest[:, t0:t1]
            m_c = message[:, t0:t1].contiguous()
            pair = pair_rms_norm(d_c, m_c, out_dtype=torch.float16)
            hidden = torch.bmm(pair, down.transpose(1, 2))
            del pair
            gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
            del hidden
            out[:, t0:t1] = blend_rms_norm(
                d_c, m_c, gain_forget, norm_weight, norm_eps, out_dtype=torch.float16
            )
        return out


class Inkling2Model(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size, params_dtype=torch.float16
        )
        # Keep embed_norm output in FP32 to seed the FP32 residual stack!
        self.embed_norm = Inkling2RMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=torch.float32
        )
        self.layers = nn.ModuleList(
            self._make_layer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=torch.float16
        )
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        return Inkling2DecoderLayer(config, layer_id, quant_config, prefix)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _log_scaling_tau(self, positions: torch.Tensor) -> torch.Tensor | None:
        if self.config.log_scaling_n_floor is None:
            return None
        return compute_log_scaling_tau(
            positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        )

    def _mix(
        self,
        stack: torch.Tensor,
        last: torch.Tensor,
        query: torch.Tensor,
        norm: Inkling2RMSNorm | None,
        addend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return attn_res(
            stack,
            last,
            addend,
            query,
            None if norm is None else norm.weight,
            0.0 if norm is None else norm.eps,
            self.normalize_values,
        )

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens, hidden = inputs_embeds.shape
        num_layers = len(self.layers)
        max_blocks = -(-num_layers // self.block_size)
        stack = torch.empty(
            (max_blocks + 1, num_tokens, hidden),
            dtype=torch.float32,
            device=inputs_embeds.device,
        )
        stack[0].copy_(inputs_embeds.float())
        tau = self._log_scaling_tau(positions)
        num_blocks = 0
        aux_hidden_states = []
        for i, layer in enumerate(self.layers):
            partial = stack[num_blocks]
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(partial.half())
            if num_blocks == 0:
                inputs = [
                    layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
                ]
            else:
                inputs = self._mix(
                    stack[:num_blocks], partial, self.attn_res.attn_queries[i - 1], layer.attn_norm
                )
            attn_partial = layer.attention(positions, inputs, tau)
            attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
            attn_out = torch.empty(
                (num_tokens, hidden), dtype=torch.float32, device=inputs_embeds.device
            )
            layer.attn_sconv(attn_reduced, positions, out=attn_out)
            first = i % self.block_size == 0
            mlp_query = self.attn_res.mlp_queries[i]
            if first:
                mlp_in = self._mix(stack[: num_blocks + 1], attn_out, mlp_query, layer.mlp_norm)
            else:
                mlp_in = self._mix(stack[:num_blocks], partial, mlp_query, layer.mlp_norm, attn_out)
            residual = None if first else partial
            if residual is None:
                num_blocks += 1
            moe_out = layer.mlp(mlp_in[0])
            layer.mlp_sconv(
                moe_out, positions, addend=attn_out, residual=residual, out=stack[num_blocks]
            )
        out = self._mix(stack[:num_blocks], stack[num_blocks], self.attn_res.output_query, self.norm)[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopModel(Inkling2Model):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(-1, 2, self.hidden_size)
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)
        init_tp4_ar_conv_mix()

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: Inkling2RMSNorm | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        messages = None
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs = layer.blend(
                slice(0, 3), dests, messages[:3], norm_weight, norm.eps
            )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs

        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        cm = conv_metadata(owner)
        use_ar_mix = (
            cm is not None
            and cm.num_tail_writes == 0
            and cm.num_ckpt_writes == 0
            and is_ar_conv_mix_ready(partial.shape[0])
        )

        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)  # FP32
        mlp_query = self.attn_res.mlp_queries[i]
        mlp_norm = None if messages is not None else layer.mlp_norm
        if use_ar_mix:
            assert cm is not None
            off_a, w_a = owner.stream_ranges[ATTN_STREAM]
            mlp_in = all_reduce_conv_mix_cuda(
                x=attn_partial,
                conv_weight=layer.attn_sconv.conv_weight,
                cache=owner.kv_cache,
                positions=positions,
                block_table=cm.block_table,
                seq_idx=cm.seq_idx,
                slot_mapping=cm.slot_mapping,
                query_start=cm.query_start,
                stream_offset=off_a,
                stream_width=w_a,
                addend=None,
                residual=None,
                conv_out=attn_out,
                stack=stack[: num_blocks + (1 if first else 0)],
                last=None if first else partial,
                queries=mlp_query,
                norm_weight=None if mlp_norm is None else mlp_norm.weight,
                norm_eps=0.0 if mlp_norm is None else mlp_norm.eps,
            )
        else:
            attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
            layer.attn_sconv(attn_reduced, positions, out=attn_out, owner=owner)
            if first:
                mlp_in = self._mix(stack[: num_blocks + 1], attn_out, mlp_query, mlp_norm)
            else:
                mlp_in = self._mix(
                    stack[:num_blocks], partial, mlp_query, mlp_norm, attn_out
                )
        if messages is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:], layer.mlp_norm.weight, layer.mlp_norm.eps
            )
            messages = None

        residual = None if first else partial
        if residual is None:
            num_blocks += 1

        if use_ar_mix:
            assert cm is not None
            off_m, w_m = owner.stream_ranges[MLP_STREAM]
            if layer.mlp.can_use_fused_m8(mlp_in[0]):
                moe_partial = layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx)
                outputs = all_reduce_conv_mix_cuda(
                    x=moe_partial,
                    conv_weight=layer.mlp_sconv.conv_weight,
                    cache=owner.kv_cache,
                    positions=positions,
                    block_table=cm.block_table,
                    seq_idx=cm.seq_idx,
                    slot_mapping=cm.slot_mapping,
                    query_start=cm.query_start,
                    stream_offset=off_m,
                    stream_width=w_m,
                    addend=attn_out,
                    residual=residual,
                    conv_out=stack[num_blocks],
                    stack=stack[:num_blocks],
                    last=None,
                    queries=next_query,
                    norm_weight=None if next_norm is None else next_norm.weight,
                    norm_eps=0.0 if next_norm is None else next_norm.eps,
                )
            else:
                down, src2dst = layer.mlp(
                    mlp_in[0], pass_idx=pass_idx, return_unreduced_pair=True
                )
                outputs = all_reduce_conv_mix_cuda(
                    combine=(down, src2dst),
                    conv_weight=layer.mlp_sconv.conv_weight,
                    cache=owner.kv_cache,
                    positions=positions,
                    block_table=cm.block_table,
                    seq_idx=cm.seq_idx,
                    slot_mapping=cm.slot_mapping,
                    query_start=cm.query_start,
                    stream_offset=off_m,
                    stream_width=w_m,
                    addend=attn_out,
                    residual=residual,
                    conv_out=stack[num_blocks],
                    stack=stack[:num_blocks],
                    last=None,
                    queries=next_query,
                    norm_weight=None if next_norm is None else next_norm.weight,
                    norm_eps=0.0 if next_norm is None else next_norm.eps,
                )
        else:
            moe_out = layer.mlp(mlp_in[0], pass_idx=pass_idx)
            layer.mlp_sconv(
                moe_out,
                positions,
                addend=attn_out,
                residual=residual,
                out=stack[num_blocks],
                owner=owner,
            )
            outputs = self._mix(
                stack[:num_blocks], stack[num_blocks], next_query, next_norm
            )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = torch.empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden),
            dtype=torch.float32,
            device=inputs_embeds.device,
        )
        stack[0].copy_(inputs_embeds.float())
        tau = self._log_scaling_tau(positions)
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]

        aux_hidden_states = []
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks], prelude_partial, None, query, None, 0.0, True
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].half())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
            )
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            if len(visits) in self.aux_hidden_state_layers:
                aux_hidden_states.append(out)
            return out, aux_hidden_states
        return out


class Inkling2FP32LogitsProcessor(LogitsProcessor):
    """Ensure FP16 hidden_states @ FP16 lm_head.weight -> FP32 logits before softcap(30.0)."""

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if getattr(lm_head, "_fp8_ready", False):
            logits = fp8_qpn8_linear(
                hidden_states,
                lm_head._fp8_codes,
                lm_head._fp8_s16,
                lm_head._fp8_s32,
                lm_head._fp8_n_out,
                out_dtype=torch.float32,
                is_lm_head=True,
            )
            if embedding_bias is not None:
                logits = logits + embedding_bias.float()
            logits = self._gather_logits(logits)
            if logits is not None:
                logits = logits[..., : self.org_vocab_size]
            return logits
        if hidden_states.dtype == torch.float16 and lm_head.weight.dtype == torch.float16:
            logits = torch.mm(hidden_states, lm_head.weight.t(), out_dtype=torch.float32)
            if embedding_bias is not None:
                logits = logits + embedding_bias.float()
            logits = self._gather_logits(logits)
            if logits is not None:
                logits = logits[..., : self.org_vocab_size]
            return logits
        return super()._get_logits(hidden_states, lm_head, embedding_bias)


class Inkling2ForConditionalGeneration(nn.Module, SupportsEagle3):
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2QuantConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        self.config = config
        self.text_config = text_config = config.text_config
        model_cls: type[Inkling2Model] = (
            Inkling2LoopModel if text_config.loop_config is not None else Inkling2Model
        )
        self.model = model_cls(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size,
            text_config.hidden_size,
            params_dtype=torch.float16,
        )
        self.logits_processor = Inkling2FP32LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return (5, 23, 41, 48)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def prepare_loaded_linear_weights(self) -> None:
        """Called by 1Cat-vLLM's `process_weights_after_loading` on target_device."""
        self.postprocess_weights()

    def postprocess_weights(self) -> None:
        """Run load-time weight transforms on Inkling2 submodules (idempotent)."""
        for _, module in self.named_modules():
            if module is self:
                continue
            fn = getattr(module, "process_weights_after_loading", None)
            if fn is not None and isinstance(
                module,
                (
                    Inkling2LoopModel,
                    Inkling2LoopDecoderLayer,
                    Inkling2Attention,
                ),
            ):
                fn()
            elif fn is not None and module.__class__.__name__ == "Inkling2Sm70Experts":
                fn()
        if "lm_head" in fp8_enabled_tiers() and not getattr(self.lm_head, "_fp8_ready", False):
            dev = self.lm_head.weight.device
            sf = get_fp8_safetensors()
            assert sf is not None
            load_fp8_dense_ext()
            r = get_tensor_model_parallel_rank()
            tp = get_tensor_model_parallel_world_size()
            w_full = sf.get_tensor("model.llm.unembed.weight")
            s_full = sf.get_tensor("model.llm.unembed.weight_scale")
            n_sh = w_full.shape[0] // tp
            w_sh = w_full[r * n_sh : (r + 1) * n_sh].to(dev)
            s_sh = s_full[r * n_sh : (r + 1) * n_sh].to(dev)
            codes, s16, s32, _, n_out = prepare_qpn8_weight(w_sh, s_sh, pad_align=128)
            self.lm_head._fp8_codes = codes
            self.lm_head._fp8_s16 = s16
            self.lm_head._fp8_s32 = s32
            self.lm_head._fp8_n_out = n_out
            self.lm_head._fp8_ready = True
            self.lm_head.weight.data = torch.empty(0, dtype=torch.float16, device=dev)
        if fp8_enabled_tiers():
            torch.cuda.empty_cache()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        for name, weight in weights:
            if name.startswith("model.visual."):
                continue  # Text-only Phase 0-2 skips vision tower weights cleanly.
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(
                    r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                )
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            # Cast BF16 checkpoint weights -> FP16 (or FP32 for queries) on sm_70.
            if param.dtype != weight.dtype and weight.dtype == torch.bfloat16:
                weight = weight.to(param.dtype)
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        return loaded


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """Architecture entry point for `inkling2_loop` checkpoints (Inkling-2-Mini)."""
