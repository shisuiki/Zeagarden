# SPDX-License-Identifier: Apache-2.0
"""Inkling2 EP=4 Mixture-of-Experts module for Volta sm_70 (exact native NVFP4 + MXFP8)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    nvfp4_skinny_scale_reason,
    qpn_prepack,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from .ops.ep4_moe_fused_cuda import (
    ep4_nvfp4_moe_fused_cuda,
    is_ep4_moe_fused_eligible,
    is_ep4_moe_fused_enabled,
    load_ep4_moe_fused_ext,
    preallocate_ep4_moe_fused_scratch,
)
from .ops.ep4_route_cuda import (
    load_ep4_route_ext,
    preallocate_ep4_route_scratch,
    select_and_route_ep4_cuda,
)
from .ops.moe_sm70 import (
    combine_sm70,
    run_mxfp8_experts_sm70,
    run_nvfp4_experts_sm70,
    select_experts_sm70,
)
from .ops.mxfp8_moe_cuda import (
    load_mxfp8_moe_ext,
    mxfp8_qpn8_prepack,
    run_mxfp8_experts_cuda,
)
from .ops.nvfp4_prefill_cuda import (
    load_nvfp4_prefill_ext,
    preallocate_nvfp4_prefill_scratch,
    run_nvfp4_experts_prefill_cuda,
)

QuantFormat = Literal["nvfp4", "mxfp8"]


class Inkling2Gate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.float16), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(
            self.weight, {"weight_loader": lambda p, w: p.data.copy_(w.to(p.dtype))}
        )
        set_weight_attrs(self.bias, {"weight_loader": default_weight_loader})


class Inkling2Sm70Experts(nn.Module):
    """EP=4 routed experts on Volta sm_70.

    - NVFP4 layers (0..19): repacked in place to QPN fragment order at load time (zero extra
      weight copies) and executed by 1Cat-vLLM's compiled `skinny_moe_qpn_sm70`.
    - MXFP8 layers (20..23): stored verbatim as `uint8` E4M3 codes + `uint8` UE8M0 scales and
      dequantized exactly on-the-fly into shared scratch buffers for FP16 cuBLAS.
    """

    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            self._register(f"{name}_weight", (n, k // pack), weight_dtype)
            self._register(f"{name}_weight_scale", (n, k // group), scale_dtype)
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        start = self.ep_rank * self.num_experts
        self.expert_lo = start
        self.expert_hi = start + self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        self._g13_t: torch.Tensor | None = None
        self._g2_t: torch.Tensor | None = None
        self._mxfp8_w13_buf: torch.Tensor | None = None
        self._mxfp8_w2_buf: torch.Tensor | None = None
        self._use_cuda_mxfp8: bool = (
            not self.nvfp4 and os.environ.get("INKLING_USE_CUDA_MXFP8", "1") != "0"
        )
        self._use_cuda_nvfp4_pf: bool = (
            self.nvfp4 and os.environ.get("INKLING_USE_CUDA_NVFP4_PF", "1") != "0"
        )
        self._use_ep4_moe_fused: bool = (
            self.nvfp4
            and is_ep4_moe_fused_enabled()
            and self.num_experts == 32
            and self.hidden_size == 2304
            and self.intermediate_size == 1152
            and self.topk == 8
        )
        self._mx_w13_qpn: torch.Tensor | None = None
        self._mx_s13_qpn: torch.Tensor | None = None
        self._mx_g13: torch.Tensor | None = None
        self._mx_wide13: bool = False
        self._mx_w2_qpn: torch.Tensor | None = None
        self._mx_s2_qpn: torch.Tensor | None = None
        self._mx_g2: torch.Tensor | None = None
        self._mx_wide2: bool = False
        self._postprocessed: bool = False

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def process_weights_after_loading(self) -> None:
        if self._postprocessed:
            return
        self._postprocessed = True
        if os.environ.get("INKLING_USE_CUDA_ROUTE", "1") != "0":
            load_ep4_route_ext()
            preallocate_ep4_route_scratch(self.w13_weight.device)
        if self._use_cuda_nvfp4_pf:
            load_nvfp4_prefill_ext()
            preallocate_nvfp4_prefill_scratch(
                self.w13_weight.device,
                h=self.hidden_size,
                inter=self.intermediate_size,
                e_local=self.num_experts,
                topk=self.topk,
            )
        if self._use_ep4_moe_fused:
            load_ep4_moe_fused_ext()
            preallocate_ep4_moe_fused_scratch(self.w13_weight.device)
        if not self.nvfp4:
            if self._use_cuda_mxfp8:
                load_mxfp8_moe_ext()
                (
                    self._mx_w13_qpn,
                    self._mx_s13_qpn,
                    self._mx_g13,
                    self._mx_wide13,
                ) = mxfp8_qpn8_prepack(
                    self.w13_weight.data,
                    self.w13_weight_scale.data,
                    is_w13=True,
                    inplace=True,
                )
                (
                    self._mx_w2_qpn,
                    self._mx_s2_qpn,
                    self._mx_g2,
                    self._mx_wide2,
                ) = mxfp8_qpn8_prepack(
                    self.w2_weight.data,
                    self.w2_weight_scale.data,
                    is_w13=False,
                    inplace=True,
                )
            return
        g13 = (self.w13_weight_amax.data / self.w13_weight_outer_scale_denominator.data).to(
            torch.float32
        ).contiguous()
        g2 = (self.w2_weight_amax.data / self.w2_weight_outer_scale_denominator.data).to(
            torch.float32
        ).contiguous()
        reason = nvfp4_skinny_scale_reason(g13, g2)
        if reason is not None:
            raise RuntimeError(f"Inkling2 NVFP4 global scale rejected by skinny QPN: {reason}")
        self._g13_t = g13
        self._g2_t = g2

        # In-place byte-equal permutation into Volta mma.m8n8k4 QPN fragment order.
        w13, s13 = self.w13_weight.data, self.w13_weight_scale.data.view(torch.uint8)
        w2, s2 = self.w2_weight.data, self.w2_weight_scale.data.view(torch.uint8)
        for e in range(self.num_experts):
            qc, qs = qpn_prepack(w13[e], s13[e], 16)
            w13[e].view(-1).copy_(qc)
            s13[e].view(-1).copy_(qs)
            qc, qs = qpn_prepack(w2[e], s2[e], 16)
            w2[e].view(-1).copy_(qc)
            s2[e].view(-1).copy_(qs)

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        reduce: bool = True,
        precomputed_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_unreduced_pair: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.nvfp4:
            assert self._g13_t is not None and self._g2_t is not None
            if self._use_cuda_nvfp4_pf:
                if return_unreduced_pair:
                    down = run_nvfp4_experts_prefill_cuda(
                        x,
                        self.w13_weight.data,
                        self.w13_weight_scale.data.view(torch.uint8),
                        self._g13_t,
                        self.w2_weight.data,
                        self.w2_weight_scale.data.view(torch.uint8),
                        self._g2_t,
                        ids,
                        weights,
                        src2dst,
                        self.gate_cap,
                        self.up_cap,
                        precomputed_route=precomputed_route,
                        combine=False,
                    )
                    return down, src2dst
                out = run_nvfp4_experts_prefill_cuda(
                    x,
                    self.w13_weight.data,
                    self.w13_weight_scale.data.view(torch.uint8),
                    self._g13_t,
                    self.w2_weight.data,
                    self.w2_weight_scale.data.view(torch.uint8),
                    self._g2_t,
                    ids,
                    weights,
                    src2dst,
                    self.gate_cap,
                    self.up_cap,
                    precomputed_route=precomputed_route,
                    combine=True,
                )
                if not reduce or self.ep_size == 1:
                    return out
                return tensor_model_parallel_all_reduce(out)
            down = run_nvfp4_experts_sm70(
                x,
                self.w13_weight.data,
                self.w13_weight_scale.data.view(torch.uint8),
                self._g13_t,
                self.w2_weight.data,
                self.w2_weight_scale.data.view(torch.uint8),
                self._g2_t,
                ids,
                weights,
                src2dst,
                self.gate_cap,
                self.up_cap,
                precomputed_route=precomputed_route,
            )
        elif self._use_cuda_mxfp8:
            assert (
                self._mx_w13_qpn is not None
                and self._mx_s13_qpn is not None
                and self._mx_g13 is not None
                and self._mx_w2_qpn is not None
                and self._mx_s2_qpn is not None
                and self._mx_g2 is not None
            )
            down = run_mxfp8_experts_cuda(
                x,
                self._mx_w13_qpn,
                self._mx_s13_qpn,
                self._mx_g13,
                self._mx_w2_qpn,
                self._mx_s2_qpn,
                self._mx_g2,
                ids,
                weights,
                src2dst,
                self.gate_cap,
                self.up_cap,
                wide_scale_13=self._mx_wide13,
                wide_scale_2=self._mx_wide2,
                precomputed_route=precomputed_route,
            )
        else:
            if self._mxfp8_w13_buf is None:
                self._mxfp8_w13_buf = torch.empty(
                    (2 * self.intermediate_size, self.hidden_size),
                    dtype=torch.float16,
                    device=x.device,
                )
                self._mxfp8_w2_buf = torch.empty(
                    (self.hidden_size, self.intermediate_size),
                    dtype=torch.float16,
                    device=x.device,
                )
            down = run_mxfp8_experts_sm70(
                x,
                self.w13_weight.data.view(torch.uint8),
                self.w13_weight_scale.data,
                self.w2_weight.data.view(torch.uint8),
                self.w2_weight_scale.data,
                ids,
                weights,
                src2dst,
                self.gate_cap,
                self.up_cap,
                self._mxfp8_w13_buf,
                self._mxfp8_w2_buf,
            )
        if return_unreduced_pair:
            return down, src2dst
        out = combine_sm70(down, src2dst, self.topk, x.dtype)
        if not reduce or self.ep_size == 1:
            return out
        return tensor_model_parallel_all_reduce(out)


class Inkling2MoE(nn.Module):
    def __init__(
        self, config, quant_format: QuantFormat, visit_ids: Sequence[int] = (0,)
    ) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Sm70Experts(config, quant_format)
        self._use_cuda_route = os.environ.get("INKLING_USE_CUDA_ROUTE", "1") != "0"

    def can_use_fused_m8(self, x: torch.Tensor) -> bool:
        return (
            self.experts._use_ep4_moe_fused
            and self.experts._g13_t is not None
            and self.experts._g2_t is not None
            and is_ep4_moe_fused_eligible(
                x,
                self.gate.weight,
                self.experts.w13_weight.data,
                self.experts.w2_weight.data,
                self.experts.expert_lo,
                self.experts.expert_hi,
            )
        )

    def _forward_chunk(
        self,
        x: torch.Tensor,
        is_padding: torch.Tensor | None,
        reduce: bool,
        return_unreduced_pair: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not return_unreduced_pair and self.can_use_fused_m8(x):
            assert self.experts._g13_t is not None and self.experts._g2_t is not None
            out = ep4_nvfp4_moe_fused_cuda(
                x,
                self.gate.weight,
                self.gate.bias,
                self.experts.w13_weight.data,
                self.experts.w13_weight_scale.data.view(torch.uint8),
                self.experts._g13_t,
                self.experts.w2_weight.data,
                self.experts.w2_weight_scale.data.view(torch.uint8),
                self.experts._g2_t,
                self.gate.route_scale,
                self.experts.gate_cap,
                self.experts.up_cap,
                self.experts.expert_lo,
                self.experts.expert_hi,
                is_padding=is_padding,
            )
            if not reduce or self.experts.ep_size == 1:
                return out
            return tensor_model_parallel_all_reduce(out)
        if self._use_cuda_route:
            ids, weights, src2dst, perm, gids, goff = select_and_route_ep4_cuda(
                x,
                self.gate.weight,
                self.gate.bias,
                self.gate.topk,
                self.gate.route_scale,
                self.experts.expert_lo,
                self.experts.expert_hi,
                is_padding,
            )
            return self.experts(
                x,
                ids,
                weights,
                src2dst,
                reduce=reduce,
                precomputed_route=(perm, gids, goff),
                return_unreduced_pair=return_unreduced_pair,
            )
        ids, weights, src2dst = select_experts_sm70(
            x,
            self.gate.weight,
            self.gate.bias,
            self.gate.topk,
            self.gate.route_scale,
            self.experts.expert_bounds,
            is_padding,
        )
        return self.experts(
            x, ids, weights, src2dst, reduce=reduce, return_unreduced_pair=return_unreduced_pair
        )

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        pass_idx: int = 0,
        return_unreduced_pair: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x.contiguous()
        is_padding = None
        if getattr(envs, "VLLM_MOE_SKIP_PADDING", False) and is_forward_context_available():
            is_padding = getattr(get_forward_context(), "is_padding", None)
        if return_unreduced_pair:
            return self._forward_chunk(x, is_padding, reduce=False, return_unreduced_pair=True)
        num_tokens = x.shape[0]
        chunk_cap = 2048 if self.experts._use_cuda_nvfp4_pf else 1024
        if num_tokens > chunk_cap:
            out = torch.empty_like(x)
            for t0 in range(0, num_tokens, chunk_cap):
                t1 = min(t0 + chunk_cap, num_tokens)
                pad_c = None if is_padding is None else is_padding[t0:t1]
                out[t0:t1] = self._forward_chunk(x[t0:t1].contiguous(), pad_c, reduce=False)
            if not reduce or self.experts.ep_size == 1:
                return out
            return tensor_model_parallel_all_reduce(out)
        return self._forward_chunk(x, is_padding, reduce=reduce)
