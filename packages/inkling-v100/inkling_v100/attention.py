# SPDX-License-Identifier: Apache-2.0
"""Inkling2 relative-bias GQA attention & 1Cat-vLLM v1.5.3 backend for Volta sm_70."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .configs import Inkling2ModelConfig
from .conv_state import Inkling2ShortConv, InklingConvState, K_STREAM, V_STREAM, conv_metadata
from .ops.attention_sm70 import (
    MODE_OUT_LSE,
    gate_sink,
    plan_decode_attention,
    rel_attention,
    rel_decode_attention,
)
from .ops.attn_rel_dec_cuda import (
    load_attn_rel_dec_ext,
    plan_rel_decode_splits,
    preallocate_workspace as preallocate_attn_workspace,
    rel_decode_attention_cuda,
)
from .ops.attn_rel_dec2_cuda import (
    load_attn_rel_dec2_ext,
    plan_dec2_splits,
    preallocate_dec2_workspace,
    rel_decode_attention2_cuda,
    supports_dec2,
    usable_dec2,
)
from .ops.attn_rel_gla_cuda import (
    elr_global_kv_enabled,
    get_gla_linear_slots,
    get_gla_safetensors,
    gla_gate_epilogue_cuda,
    gla_global_enabled,
    gla_global_path,
    gla_pack_q_cuda,
    gla_scatter_fp8_cuda,
    load_attn_rel_gla_ext,
    plan_rel_gla_decode_splits,
    preallocate_gla_workspace,
    rel_gla_decode_attention_cuda,
    rel_gla_prefill_attention_cuda,
)
from .ops.attn_rel_pf_cuda import (
    load_attn_rel_pf_ext,
    rel_prefill_attention_cuda,
)
from .ops.attn_rel_pf_wmma_cuda import (
    clip_qr,
    load_attn_rel_pf_wmma_ext,
    rel_prefill_attention_wmma_cuda,
    supports_prefill_wmma,
)
from .ops.fp8_dense_cuda import (
    fp8_enabled_tiers,
    fp8_qpn8_linear,
    get_fp8_safetensors,
    load_fp8_dense_ext,
    preallocate_fp8_dequant_ws,
    prepare_qpn8_weight,
)
from .ops.sconv import (
    attention_prologue,
    kv_conv_prologue,
    qr_clip_rel_only,
    shift_rel_proj,
)

log = init_logger("vllm.inkling_v100.attention")
_GLOBAL_OPT_LOGGED = False


@dataclass
class Inkling2AttentionMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor  # [num_reqs + 1] int32
    seq_lens: torch.Tensor  # [num_reqs] int32
    block_table: torch.Tensor  # [num_reqs, max_blocks] int32
    slot_mapping: torch.Tensor  # [T] int64
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0


class Inkling2AttentionMetadataBuilder(AttentionMetadataBuilder[Inkling2AttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2AttentionMetadata:
        cm = common_attn_metadata
        thresh = self.reorder_batch_threshold or 1
        if cm.max_query_len <= thresh:
            n_dec_r, n_pf_r, n_dec_t, n_pf_t = (
                cm.num_reqs,
                0,
                cm.num_actual_tokens,
                0,
            )
        else:
            n_dec_r, n_pf_r, n_dec_t, n_pf_t = split_decodes_and_prefills(
                cm, decode_threshold=thresh
            )
        return Inkling2AttentionMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            num_decode_reqs=n_dec_r,
            num_prefill_reqs=n_pf_r,
            num_decode_tokens=n_dec_t,
            num_prefill_tokens=n_pf_t,
        )


class Inkling2AttentionBackend(AttentionBackend):
    """Standalone 1Cat-vLLM v1.5.3 backend for Inkling2 relative-bias attention on sm_70."""

    @staticmethod
    def get_name() -> str:
        return "INKLING2_REL_SM70"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Inkling2Attention runs its own sm_70 kernels directly.")

    @staticmethod
    def get_builder_cls() -> type[Inkling2AttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # [num_blocks, 2, block_size, num_kv_heads, head_size]
        # -> key_cache = kv_cache[:, 0], value_cache = kv_cache[:, 1]
        # both have shape [num_blocks, block_size, num_kv_heads, head_size], stride(3)==1!
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16, 32, 64, 128, 256]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 128, 256)

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in (torch.float16, torch.bfloat16)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype in (None, "auto", "float16", "half", "fp8", "fp8_e4m3")

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True


def _make_kv_cache_spec(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheSpec:
    kv_dtype = kv_cache_dtype_str_to_dtype(
        vllm_config.cache_config.cache_dtype, vllm_config.model_config
    )
    if kv_dtype == torch.bfloat16:
        kv_dtype = torch.float16
    if sliding_window is not None:
        return SlidingWindowSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=kv_dtype,
            sliding_window=sliding_window,
        )
    return FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_dtype,
    )


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """Per-pass KV cache owner for loop core layers (`pass_idx >= 1`)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _make_kv_cache_spec(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention on Volta sm_70 (FP16 compute, FP16 or unit-scale FP8 E4M3 KV)."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self._fp8_attn = False
        self._fp8_qkvr: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        self._fp8_qside: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        self._fp8_k: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        self._fp8_v: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        self._fp8_wo: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
        self._fp8_fused_splits: list[int] | None = None
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)

        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent, dtype=torch.float16),
            requires_grad=False,
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, dtype=torch.float16),
            requires_grad=False,
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel, dtype=torch.float16),
            requires_grad=False,
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w, r=tp_rank: p.data.copy_(
                        w.narrow(0, r * p.shape[0], p.shape[0]).to(p.dtype)
                    )
                },
            )
        set_weight_attrs(
            self.rel_logits_proj,
            {"weight_loader": lambda p, w: p.data.copy_(w.to(p.dtype))},
        )

        vllm_config = get_current_vllm_config()
        spec = vllm_config.speculative_config
        self.max_decode_query_len = 1 + (0 if spec is None else spec.num_speculative_tokens)
        cache_dtype = vllm_config.cache_config.cache_dtype
        self.fp8_kv = cache_dtype in ("fp8", "fp8_e4m3")

        # Optional composable global-KV lossy optimizations:
        #   - INKLING_ELR_GLOBAL_KV=1: ELR Last-Loop KV sharing on looped global layers (L11, L17)
        #   - INKLING_GLA_GLOBAL=1 / INKLING_GLA_PATH=...: Post-prologue TP4 GLA-g4 (r=256, FP8)
        self.elr_global_kv = (not is_local) and elr_global_kv_enabled()
        self.use_gla = (not is_local) and gla_global_enabled()
        if self.use_gla:
            if not self.fp8_kv:
                raise ValueError("INKLING_GLA_GLOBAL / INKLING_GLA_PATH requires --kv-cache-dtype fp8")
            if not (self.num_heads == 6 and self.num_kv_heads == 2 and self.head_dim == 128):
                raise ValueError(
                    f"Post-prologue GLA-g4 r=256 requires TP4 Inkling2 global shape "
                    f"(num_heads=6, num_kv_heads=2, head_dim=128), got "
                    f"({self.num_heads}, {self.num_kv_heads}, {self.head_dim})"
                )
        self.kv_cache_num_heads = 1 if self.use_gla else self.num_kv_heads
        self._gla_projs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        global _GLOBAL_OPT_LOGGED
        if (self.elr_global_kv or self.use_gla) and not _GLOBAL_OPT_LOGGED:
            _GLOBAL_OPT_LOGGED = True
            log.info(
                "Inkling2 global KV lossy opts active: ELR_GLOBAL_KV=%s | GLA_G4_R256=%s (path=%s)",
                self.elr_global_kv,
                self.use_gla,
                gla_global_path() if self.use_gla else None,
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        share_kv = config.loop_share_cache or self.elr_global_kv
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.kv_cache_num_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if share_kv else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        self._use_cuda_attn = (
            self.head_dim == 128 and os.environ.get("INKLING_USE_CUDA_ATTN", "1") != "0"
        )
        self._use_cuda_attn_dec2 = (
            self._use_cuda_attn
            and not self.use_gla
            and supports_dec2(
                self.num_heads, self.num_kv_heads, self.head_dim, self.d_rel, torch.float16
            )
        )
        self._dec2_max_reqs = int(
            os.environ.get(
                "INKLING_DEC2_MAX_REQS",
                os.environ.get(
                    "INKLING_DEC2_MAX_REQS_LOCAL" if is_local else "INKLING_DEC2_MAX_REQS_GLOBAL",
                    "8" if is_local else "1",
                ),
            )
        )
        self._use_cuda_attn_pf = (
            self.head_dim == 128 and os.environ.get("INKLING_USE_CUDA_ATTN_PF", "1") != "0"
        )
        self._use_cuda_attn_pf_wmma = (
            self._use_cuda_attn_pf
            and not self.use_gla
            and os.environ.get("INKLING_USE_CUDA_ATTN_PF_WMMA", "1") != "0"
            and supports_prefill_wmma(
                self.num_heads, self.num_kv_heads, self.head_dim, self.d_rel, torch.float16
            )
        )
        self.rel_proj_t: torch.Tensor | None = None
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def process_weights_after_loading(self) -> None:
        dev = self.qkvr.weight.device
        if len(self.split_sizes) > 6 and self.split_sizes[6] > 0 and self.qkvr.weight.numel() > 0:
            self.qkvr.weight.data[-self.split_sizes[6] :].zero_()
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
        self.rel_proj_t = self.rel_logits_proj.t().contiguous()
        if self._use_cuda_attn:
            load_attn_rel_dec_ext()
            preallocate_attn_workspace(slots=65536, device=dev)
        if self._use_cuda_attn_dec2:
            load_attn_rel_dec2_ext()
            preallocate_dec2_workspace(device=dev, max_slots=8192)
        if self._use_cuda_attn_pf:
            load_attn_rel_pf_ext()
        if self._use_cuda_attn_pf_wmma:
            load_attn_rel_pf_wmma_ext()
        if "attn" in fp8_enabled_tiers() and not self._fp8_attn:
            self._load_offline_fp8_weights(dev)
        if self.use_gla and not self._gla_projs:
            self._load_gla_projectors(dev)

    def _load_gla_projectors(self, dev: torch.device) -> None:
        load_attn_rel_gla_ext()
        preallocate_gla_workspace(slots=32768, max_tokens=8192, device=dev)
        sf = get_gla_safetensors()
        assert sf is not None
        m = re.search(r"layers\.(\d+)\.", self.prefix)
        assert m is not None, f"Cannot parse layer id from prefix {self.prefix}"
        layer_id = int(m.group(1))
        r = self.tp_rank
        num_passes = len(self.conv_states)
        if self.elr_global_kv and num_passes > 1:
            key = f"l{layer_id}_elr"
            wd = sf.get_tensor(f"{key}.w_down")[r].to(dev).contiguous()
            ukt = sf.get_tensor(f"{key}.u_k_t")[r].to(dev).contiguous()
            uv = sf.get_tensor(f"{key}.u_v")[r].to(dev).contiguous()
            shared = (wd, ukt, uv)
            self._gla_projs = [shared for _ in range(num_passes)]
        else:
            projs = []
            for p in range(num_passes):
                key = f"l{layer_id}_p{p}"
                wd = sf.get_tensor(f"{key}.w_down")[r].to(dev).contiguous()
                ukt = sf.get_tensor(f"{key}.u_k_t")[r].to(dev).contiguous()
                uv = sf.get_tensor(f"{key}.u_v")[r].to(dev).contiguous()
                projs.append((wd, ukt, uv))
            self._gla_projs = projs

    def _load_offline_fp8_weights(self, dev: torch.device) -> None:
        sf = get_fp8_safetensors()
        assert sf is not None
        load_fp8_dense_ext()
        preallocate_fp8_dequant_ws(dev, 2176 * 2304)
        ckpt_pfx = self.prefix.replace("model.layers.", "model.llm.layers.", 1)
        r, tp = self.tp_rank, self.tp_size

        def load_col_shard(leaf: str) -> tuple[torch.Tensor, torch.Tensor]:
            key = f"{ckpt_pfx}.{leaf}.weight"
            w_full = sf.get_tensor(key)
            s_full = sf.get_tensor(f"{key}_scale")
            n_sh = w_full.shape[0] // tp
            w_sh = w_full[r * n_sh : (r + 1) * n_sh].to(dev)
            s_sh = s_full[r * n_sh : (r + 1) * n_sh].to(dev)
            return w_sh, s_sh

        parts = [load_col_shard(k) for k in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")]
        # Pack ONE zero-copy contiguous QPN8 buffer: [q_side padded to 64-align | k (256) | v (256)].
        # Because QPN8 stores 32-column tiles contiguously along dim N, the 3 sub-slabs serve
        # both fused single-launch (Pass 0 local) AND 3-input split launches (global + looped passes 1-2)
        # with ZERO weight duplication!
        w_q_raw = torch.cat([p[0] for p in parts[:4]], dim=0)
        s_q_raw = torch.cat([p[1] for p in parts[:4]], dim=0)
        nq_out, k_dim = w_q_raw.shape
        nq_pad = ((nq_out + 63) // 64) * 64  # 1638 -> 1664 (26*64)
        q_pad_rows = nq_pad - nq_out
        w_q_pad = torch.zeros((nq_pad, k_dim), dtype=torch.uint8, device=dev)
        w_q_pad[:nq_out].copy_(w_q_raw.view(torch.uint8))
        s_q_pad = torch.zeros((nq_pad,), dtype=torch.float32, device=dev)
        s_q_pad[:nq_out].copy_(s_q_raw.float())

        wk, sk = parts[4]
        wv, sv = parts[5]
        nk, nv = wk.shape[0], wv.shape[0]
        assert nk % 64 == 0 and nv % 64 == 0, f"Expected k/v multiples of 64, got {nk}, {nv}"

        w_cat = torch.cat([w_q_pad, wk.view(torch.uint8), wv.view(torch.uint8)], dim=0).view(torch.float8_e4m3fn)
        s_cat = torch.cat([s_q_pad, sk.float(), sv.float()], dim=0)
        c_all, s16_all, s32_all, n_all_pad, _ = prepare_qpn8_weight(w_cat, s_cat, pad_align=64)
        assert n_all_pad == nq_pad + nk + nv

        # Zero-copy views into the single QPN8 buffer (tile_bytes = k_dim * 32 per 32 cols -> k_dim bytes/col):
        c_flat = c_all.view(-1)
        b_q = nq_pad * k_dim
        b_k = nk * k_dim
        b_v = nv * k_dim
        self._fp8_qkvr = (c_all, s16_all, s32_all, n_all_pad)
        self._fp8_qside = (
            c_flat[:b_q].view(k_dim, nq_pad),
            s16_all[:nq_pad],
            s32_all[:nq_pad],
            nq_out,
        )
        self._fp8_k = (
            c_flat[b_q : b_q + b_k].view(k_dim, nk),
            s16_all[nq_pad : nq_pad + nk],
            s32_all[nq_pad : nq_pad + nk],
            nk,
        )
        self._fp8_v = (
            c_flat[b_q + b_k : b_q + b_k + b_v].view(k_dim, nv),
            s16_all[nq_pad + nk : n_all_pad],
            s32_all[nq_pad + nk : n_all_pad],
            nv,
        )
        self._fp8_fused_splits = [*self.split_sizes[:4], q_pad_rows, nk, nv]

        # RowParallel wo_ud: shard along dim=1 (K), shared per-output-channel scale!
        wo_key = f"{ckpt_pfx}.wo_ud.weight"
        wo_full = sf.get_tensor(wo_key)
        wo_s = sf.get_tensor(f"{wo_key}_scale").to(dev)
        k_sh = wo_full.shape[1] // tp
        wo_sh = wo_full[:, r * k_sh : (r + 1) * k_sh].to(dev).contiguous()
        cwo, swo16, swo32, _, nwo = prepare_qpn8_weight(wo_sh, wo_s, pad_align=64)
        self._fp8_wo = (cwo, swo16, swo32, nwo)

        # Free duplicate FP16 weights cleanly before KV cache sizing!
        self.qkvr.weight.data = torch.empty(0, dtype=torch.float16, device=dev)
        self.wo_ud.weight.data = torch.empty(0, dtype=torch.float16, device=dev)
        self._fp8_attn = True

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _make_kv_cache_spec(
            vllm_config,
            self.kv_cache_num_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        if self._fp8_attn:
            if x_k is x_q and x_v is x_q and self._fp8_qkvr is not None:
                c, s16, s32, n_out = self._fp8_qkvr
                assert self._fp8_fused_splits is not None
                projected = fp8_qpn8_linear(x_q, c, s16, s32, n_out)
                sp = projected.split(self._fp8_fused_splits, dim=-1)
                return [sp[0], sp[1], sp[2], sp[3], sp[5], sp[6]]
            assert self._fp8_qside is not None and self._fp8_k is not None and self._fp8_v is not None
            cq, sq16, sq32, nq = self._fp8_qside
            ck, sk16, sk32, nk = self._fp8_k
            cv, sv16, sv32, nv = self._fp8_v
            projected = fp8_qpn8_linear(x_q, cq, sq16, sq32, nq)
            k = fp8_qpn8_linear(x_k, ck, sk16, sk32, nk)
            v = fp8_qpn8_linear(x_v, cv, sv16, sv32, nv)
            return [*projected.split(self.q_side_splits[:4], dim=-1), k, v]
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = torch.nn.functional.linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = torch.nn.functional.linear(x_q, weight[: sum(self.q_side_splits)])
        k = torch.nn.functional.linear(x_k, weight[q_side:k_end])
        v = torch.nn.functional.linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _split_kv_cache(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Layout from Inkling2AttentionBackend.get_kv_cache_shape:
        # [num_blocks, 2, block_size, num_kv_heads, head_size]
        return kv_cache[:, 0], kv_cache[:, 1]

    def _plan_decode(
        self, md: Inkling2AttentionMetadata, num_tokens: int
    ) -> tuple[int, int, int]:
        num_reqs = md.seq_lens.shape[0]
        if self.use_gla:
            if md.max_query_len <= 8:
                return num_reqs, num_tokens, plan_rel_gla_decode_splits(num_reqs, md.max_query_len)
            num_decodes, num_decode_tokens = md.num_decode_reqs, md.num_decode_tokens
            if not 0 < num_decodes < num_reqs:
                return 0, 0, 0
            max_q = min(self.max_decode_query_len, num_decode_tokens - num_decodes + 1)
            if max_q > 8:
                return 0, 0, 0
            return num_decodes, num_decode_tokens, plan_rel_gla_decode_splits(num_decodes, max_q)
        planner = plan_rel_decode_splits if self._use_cuda_attn else plan_decode_attention
        if self._use_cuda_attn and md.max_query_len > 8:
            num_splits = 0
        else:
            num_splits = planner(
                self.is_local,
                num_reqs=num_reqs,
                max_query_len=md.max_query_len,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rel_extent=self.rel_extent,
            )
        if num_splits:
            return num_reqs, num_tokens, num_splits
        num_decodes, num_decode_tokens = md.num_decode_reqs, md.num_decode_tokens
        if not 0 < num_decodes < num_reqs:
            return 0, 0, 0
        max_q = min(self.max_decode_query_len, num_decode_tokens - num_decodes + 1)
        if self._use_cuda_attn and max_q > 8:
            return 0, 0, 0
        num_splits = planner(
            self.is_local,
            num_reqs=num_decodes,
            max_query_len=max_q,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rel_extent=self.rel_extent,
        )
        if not num_splits:
            return 0, 0, 0
        return num_decodes, num_decode_tokens, num_splits

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        num_tokens = q.shape[0]
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q.view(num_tokens, self.num_heads, self.head_dim),
            r.view(num_tokens, self.num_heads, self.d_rel),
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        if self._fp8_attn and self._fp8_wo is not None:
            cwo, swo16, swo32, nwo = self._fp8_wo
            return fp8_qpn8_linear(output, cwo, swo16, swo32, nwo, out=out)
        if out is not None:
            return torch.mm(output, self.wo_ud.weight.t(), out=out)
        return torch.nn.functional.linear(output, self.wo_ud.weight)

    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int = 0,
    ) -> None:
        num_tokens = q.shape[0]
        attn_metadata = get_forward_context().attn_metadata
        layer_name, kv_cache = self._pass_cache(pass_idx)
        conv_state = self.conv_states[pass_idx]
        md = None
        if isinstance(attn_metadata, dict) and kv_cache.numel() > 0:
            md = cast(Inkling2AttentionMetadata | None, attn_metadata.get(layer_name))
        if md is None:
            # Profile run before KV allocation: reserve transient attention workspace peak.
            if self._use_cuda_attn_pf_wmma:
                _q_tmp = torch.empty(q.shape, dtype=q.dtype, device=q.device)
                _r_tmp = torch.empty(r.shape, dtype=r.dtype, device=r.device)
                del _q_tmp, _r_tmp
            else:
                _rel_tmp = q.new_empty((num_tokens + 128, self.num_heads, self.rel_extent + 256))
                _pf_tmp = q.new_empty(
                    (num_tokens, self.num_heads, self.head_dim), dtype=torch.float32
                )
                del _rel_tmp, _pf_tmp
            output.zero_()
            return

        conv_md = conv_metadata(conv_state)
        assert conv_md is not None
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        # Fast path 1: 1-launch short-SASS fused kv_conv_prologue + decode_attention2 (C<=max_reqs, Q<=8).
        if (
            self._use_cuda_attn_dec2
            and self.rel_proj_t is not None
            and conv_md.num_tail_writes == 0
            and conv_md.num_ckpt_writes == 0
            and md.max_query_len <= 8
            and int(md.seq_lens.shape[0]) <= self._dec2_max_reqs
        ):
            dec2_splits = plan_dec2_splits(
                num_reqs=int(md.seq_lens.shape[0]),
                num_kv_heads=self.num_kv_heads,
                max_query_len=md.max_query_len,
                is_local=self.is_local,
            )
            if dec2_splits > 0 and usable_dec2(
                q,
                k,
                v,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                key_cache,
                value_cache,
                conv_state.kv_cache,
                positions,
                conv_md.slot_mapping,
                md.slot_mapping,
                r=r,
                gate=gate,
                sinks=sinks,
                rel_t=self.rel_proj_t,
                out=output,
            ):
                rel_decode_attention2_cuda(
                    q,
                    r,
                    self.q_scale,
                    self.r_scale,
                    tau,
                    self.rel_proj_t,
                    key_cache,
                    value_cache,
                    gate,
                    sinks,
                    output,
                    k=k,
                    v=v,
                    k_weight=self.k_sconv.conv_weight,
                    v_weight=self.v_sconv.conv_weight,
                    k_dynamic=k_dynamic,
                    v_dynamic=v_dynamic,
                    conv_cache=conv_state.kv_cache,
                    positions=positions,
                    conv_block_table=conv_md.block_table,
                    seq_idx=conv_md.seq_idx,
                    conv_slot_mapping=conv_md.slot_mapping,
                    query_start=conv_md.query_start,
                    kv_slot_mapping=md.slot_mapping,
                    block_table=md.block_table,
                    seq_lens=md.seq_lens,
                    cu_seqlens_q=md.query_start_loc,
                    max_query_len=md.max_query_len,
                    num_actual_tokens=md.num_actual_tokens,
                    softmax_scale=self.scaling,
                    window=self.sliding_window if self.is_local else 0,
                    num_splits=dec2_splits,
                    qk_clip=self.qk_clip,
                    rel_clip=self.rel_clip,
                    norm_eps=self.eps,
                )
                return

        num_decodes, num_decode_tokens, decode_splits = self._plan_decode(md, num_tokens)

        # Fast path 2: On-the-fly Tensor-Core R @ rel_proj WMMA prefill (eliminates global rel_logits!).
        if (
            self._use_cuda_attn_pf_wmma
            and not self.use_gla
            and num_decode_tokens < num_tokens
        ):
            kv_conv_prologue(
                k,
                v,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                key_cache,
                value_cache,
                md.slot_mapping,
                clip_threshold=self.qk_clip,
                norm_eps=self.eps,
                tail_src_toks=conv_md.tail_src_toks,
                tail_req_slots=conv_md.tail_req_slots,
                num_tail_writes=conv_md.num_tail_writes,
                ckpt_dst=conv_md.layer_ckpt_dst(conv_state.ring_layer_idx),
                ckpt_b_pos=conv_md.ckpt_b_pos,
                ckpt_qstart=conv_md.ckpt_qstart,
                ckpt_start_pos=conv_md.ckpt_start_pos,
                ckpt_req_slots=conv_md.ckpt_req_slots,
                num_ckpt_writes=conv_md.num_ckpt_writes,
            )
            if decode_splits:
                # Mixed batch: materialize tiny sheared rel_logits ONLY for the decode prefix.
                dec_q = torch.empty(
                    (num_decode_tokens, self.num_heads, self.head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                dec_rel = q.new_empty(
                    (num_decode_tokens + 128, self.num_heads, self.rel_extent + 256)
                )
                qr_clip_rel_only(
                    q[:num_decode_tokens],
                    r[:num_decode_tokens],
                    self.q_scale,
                    self.r_scale,
                    None if tau is None else tau[:num_decode_tokens],
                    self.rel_proj_shifted,
                    dec_q,
                    dec_rel,
                    conv_md.seq_idx[:num_decode_tokens],
                    md.query_start_loc[: num_decodes + 1],
                    md.seq_lens[:num_decodes],
                    clip_threshold=self.qk_clip,
                    rel_threshold=self.rel_clip,
                )
                dec_actual = md.num_decode_tokens
                max_q = min(self.max_decode_query_len, dec_actual - num_decodes + 1)
                dec_fn = rel_decode_attention_cuda if self._use_cuda_attn else rel_decode_attention
                dec_fn(
                    dec_q,
                    key_cache,
                    value_cache,
                    dec_rel,
                    gate[:num_decode_tokens],
                    sinks[:num_decode_tokens],
                    output[:num_decode_tokens],
                    block_table=md.block_table[:num_decodes],
                    cache_seqlens=md.seq_lens[:num_decodes],
                    cu_seqlens_q=md.query_start_loc[: num_decodes + 1],
                    max_query_len=max_q,
                    num_actual_tokens=dec_actual,
                    softmax_scale=self.scaling,
                    is_local=self.is_local,
                    num_splits=decode_splits,
                )
            pf_q, pf_r = clip_qr(
                q[num_decode_tokens:],
                r[num_decode_tokens:],
                self.q_scale,
                self.r_scale,
                None if tau is None else tau[num_decode_tokens:],
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
            )
            nt = md.num_actual_tokens - (md.num_decode_tokens if num_decodes else 0)
            cu_q = md.query_start_loc[num_decodes:]
            if num_decodes:
                cu_q = cu_q - md.num_decode_tokens
            rel_prefill_attention_wmma_cuda(
                pf_q,
                pf_r,
                self.rel_logits_proj,
                key_cache,
                value_cache,
                gate[num_decode_tokens:],
                sinks[num_decode_tokens:],
                output[num_decode_tokens:],
                block_table=md.block_table[num_decodes:],
                cache_seqlens=md.seq_lens[num_decodes:],
                cu_seqlens_q=cu_q,
                max_query_len=md.max_query_len,
                softmax_scale=self.scaling,
                window=self.sliding_window if self.is_local else 0,
                num_actual_tokens=max(nt, 0),
            )
            return

        clipped_q = q.new_empty(q.shape)
        rel_logits = q.new_empty((num_tokens + 128, self.num_heads, self.rel_extent + 256))

        if self.use_gla:
            w_down, u_k_t, u_v = self._gla_projs[pass_idx]
            kv_post = q.new_empty((num_tokens, 512))
            kv_view = kv_post.view(1, num_tokens, 2, 2, 128)
            k_scratch = kv_view[:, :, 0, :, :]
            v_scratch = kv_view[:, :, 1, :, :]
            lin_slots = get_gla_linear_slots(num_tokens, q.device)
            attention_prologue(
                q,
                r,
                k,
                v,
                self.q_scale,
                self.r_scale,
                tau,
                self.rel_proj_shifted,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                k_scratch,
                v_scratch,
                lin_slots,
                clipped_q,
                rel_logits,
                md.query_start_loc,
                md.seq_lens,
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
                norm_eps=self.eps,
                tail_src_toks=conv_md.tail_src_toks,
                tail_req_slots=conv_md.tail_req_slots,
                num_tail_writes=conv_md.num_tail_writes,
                ckpt_dst=conv_md.layer_ckpt_dst(conv_state.ring_layer_idx),
                ckpt_b_pos=conv_md.ckpt_b_pos,
                ckpt_qstart=conv_md.ckpt_qstart,
                ckpt_start_pos=conv_md.ckpt_start_pos,
                ckpt_req_slots=conv_md.ckpt_req_slots,
                num_ckpt_writes=conv_md.num_ckpt_writes,
            )
            c_raw = torch.mm(kv_post, w_down)
            gla_scatter_fp8_cuda(c_raw, key_cache, value_cache, md.slot_mapping)

            q_pack = q.new_empty((2, num_tokens, 3, 128))
            gla_pack_q_cuda(clipped_q, q_pack)
            q_lat = torch.bmm(q_pack.view(2, 3 * num_tokens, 128), u_k_t).view(
                2, num_tokens, 3, 256
            )
            o_lat = q.new_empty((2, num_tokens, 3, 256))
            is_mixed = decode_splits > 0 and num_decode_tokens < num_tokens
            if decode_splits:
                dec_actual = md.num_decode_tokens if is_mixed else md.num_actual_tokens
                max_q = (
                    min(self.max_decode_query_len, dec_actual - num_decodes + 1)
                    if is_mixed
                    else md.max_query_len
                )
                q_dec = q_lat[:, :num_decode_tokens].contiguous() if is_mixed else q_lat
                o_dec = q.new_empty(q_dec.shape) if is_mixed else o_lat
                rel_gla_decode_attention_cuda(
                    q_dec,
                    key_cache,
                    value_cache,
                    rel_logits,
                    sinks[:num_decode_tokens],
                    o_dec,
                    block_table=md.block_table[:num_decodes],
                    cache_seqlens=md.seq_lens[:num_decodes],
                    cu_seqlens_q=md.query_start_loc[: num_decodes + 1],
                    max_query_len=max_q,
                    num_actual_tokens=dec_actual,
                    softmax_scale=self.scaling,
                    num_splits=decode_splits,
                )
                if is_mixed:
                    o_lat[:, :num_decode_tokens].copy_(o_dec)
            if num_decode_tokens < num_tokens:
                nt = md.num_actual_tokens - (md.num_decode_tokens if num_decodes else 0)
                cu_q = md.query_start_loc[num_decodes:]
                if num_decodes:
                    cu_q = cu_q - md.num_decode_tokens
                q_pf = q_lat[:, num_decode_tokens:].contiguous() if num_decode_tokens else q_lat
                o_pf = q.new_empty(q_pf.shape) if num_decode_tokens else o_lat
                rel_gla_prefill_attention_cuda(
                    q_pf,
                    key_cache,
                    value_cache,
                    rel_logits[num_decode_tokens:],
                    sinks[num_decode_tokens:],
                    o_pf,
                    block_table=md.block_table[num_decodes:],
                    cache_seqlens=md.seq_lens[num_decodes:],
                    cu_seqlens_q=cu_q,
                    max_query_len=md.max_query_len,
                    num_actual_tokens=max(nt, 0),
                    softmax_scale=self.scaling,
                )
                if num_decode_tokens:
                    o_lat[:, num_decode_tokens:].copy_(o_pf)
            v_out = torch.bmm(o_lat.view(2, 3 * num_tokens, 256), u_v).view(
                2, num_tokens, 3, 128
            )
            gla_gate_epilogue_cuda(v_out, gate, output)
            return

        attention_prologue(
            q,
            r,
            k,
            v,
            self.q_scale,
            self.r_scale,
            tau,
            self.rel_proj_shifted,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            conv_state.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clipped_q,
            rel_logits,
            md.query_start_loc,
            md.seq_lens,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
            norm_eps=self.eps,
            tail_src_toks=conv_md.tail_src_toks,
            tail_req_slots=conv_md.tail_req_slots,
            num_tail_writes=conv_md.num_tail_writes,
            ckpt_dst=conv_md.layer_ckpt_dst(conv_state.ring_layer_idx),
            ckpt_b_pos=conv_md.ckpt_b_pos,
            ckpt_qstart=conv_md.ckpt_qstart,
            ckpt_start_pos=conv_md.ckpt_start_pos,
            ckpt_req_slots=conv_md.ckpt_req_slots,
            num_ckpt_writes=conv_md.num_ckpt_writes,
        )
        if decode_splits:
            mixed = num_decodes < md.seq_lens.shape[0]
            dec_actual = md.num_decode_tokens if mixed else md.num_actual_tokens
            max_q = (
                min(self.max_decode_query_len, dec_actual - num_decodes + 1)
                if mixed
                else md.max_query_len
            )
            dec_fn = rel_decode_attention_cuda if self._use_cuda_attn else rel_decode_attention
            dec_fn(
                clipped_q[:num_decode_tokens],
                key_cache,
                value_cache,
                rel_logits,
                gate[:num_decode_tokens],
                sinks[:num_decode_tokens],
                output[:num_decode_tokens],
                block_table=md.block_table[:num_decodes],
                cache_seqlens=md.seq_lens[:num_decodes],
                cu_seqlens_q=md.query_start_loc[: num_decodes + 1],
                max_query_len=max_q,
                num_actual_tokens=dec_actual,
                softmax_scale=self.scaling,
                is_local=self.is_local,
                num_splits=decode_splits,
            )
        if num_decode_tokens < num_tokens:
            nt = md.num_actual_tokens - (md.num_decode_tokens if num_decodes else 0)
            cu_q = md.query_start_loc[num_decodes:]
            if num_decodes:
                cu_q = cu_q - md.num_decode_tokens
            if self._use_cuda_attn_pf:
                rel_prefill_attention_cuda(
                    clipped_q[num_decode_tokens:],
                    key_cache,
                    value_cache,
                    rel_logits[num_decode_tokens:],
                    gate[num_decode_tokens:],
                    sinks[num_decode_tokens:],
                    output[num_decode_tokens:],
                    block_table=md.block_table[num_decodes:],
                    cache_seqlens=md.seq_lens[num_decodes:],
                    cu_seqlens_q=cu_q,
                    max_query_len=md.max_query_len,
                    softmax_scale=self.scaling,
                    window=self.sliding_window if self.is_local else 0,
                    num_actual_tokens=max(nt, 0),
                )
            else:
                pf_out = q.new_empty(
                    (num_tokens - num_decode_tokens, self.num_heads, self.head_dim),
                    dtype=torch.float32,
                )
                lse = q.new_empty((self.num_heads, pf_out.shape[0]), dtype=torch.float32)
                if nt > 0:
                    rel_attention(
                        clipped_q[num_decode_tokens : num_decode_tokens + nt],
                        key_cache,
                        value_cache,
                        pf_out[:nt],
                        lse[:, :nt],
                        rel_logits[num_decode_tokens:],
                        block_table=md.block_table[num_decodes:],
                        cache_seqlens=md.seq_lens[num_decodes:],
                        cu_seqlens_q=cu_q,
                        max_query_len=md.max_query_len,
                        softmax_scale=self.scaling,
                        window=self.sliding_window if self.is_local else 0,
                        num_splits=1,
                        mode=MODE_OUT_LSE,
                    )
                if nt < pf_out.shape[0]:
                    pf_out[nt:].zero_()
                    lse[:, nt:].zero_()
                gate_sink(
                    pf_out,
                    gate[num_decode_tokens:],
                    lse.t(),
                    sinks[num_decode_tokens:],
                    output[num_decode_tokens:],
                )
