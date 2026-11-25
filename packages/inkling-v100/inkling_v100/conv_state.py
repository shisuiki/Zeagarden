# SPDX-License-Identifier: Apache-2.0
"""Inkling2 short-conv state & sliding-window cache backend for 1Cat-vLLM v1.5.3 (sm_70)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec

from .ops.sconv import ring_conv_metadata, sconv_seq_metadata, short_conv

log = init_logger("vllm.inkling_v100.conv_state")

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)
RING_CONV_SLOTS = 16  # >= 3 (W-1 history) + 8 (max DSpark verify Q) => zero overwrite on rejection
RING_MAX_INPLACE_LEN = 12  # 3 + 12 = 15 < 16 => provably disjoint read/write sets


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks]
    slot_mapping: torch.Tensor  # [T] int64 (-1 => pad, -2 => long prefill read-only)
    seq_idx: torch.Tensor  # [T] int32
    query_start: torch.Tensor  # [T] int32
    tail_src_toks: torch.Tensor | None = None  # [max_reqs * 3] int32
    tail_req_slots: torch.Tensor | None = None  # [max_reqs * 3] int32
    num_tail_writes: int = 0


def ring_conv_enabled(vllm_config: VllmConfig) -> bool:
    """Return True when Option 1B fixed per-request 16-slot ring conv buffer is active.

    Default (`auto`): automatically enables when `--no-enable-prefix-caching` is set,
    removing all 48 conv_state layers from vLLM's paged KV pool (~15-20x KV capacity).
    Explicit `INKLING_RING_CONV=1` enforces it (erroring if prefix caching is enabled);
    `INKLING_RING_CONV=0` forces legacy `SlidingWindowSpec` paged conv cache.
    """
    raw = os.environ.get("INKLING_RING_CONV", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    apc = bool(vllm_config.cache_config.enable_prefix_caching)
    if raw in ("1", "true", "yes", "on"):
        if apc:
            raise ValueError(
                "INKLING_RING_CONV=1 requires --no-enable-prefix-caching "
                "(fixed ring conv state does not checkpoint per-block APC prefixes)."
            )
        return True
    return not apc


class _RingConvRuntime:
    """Persistent per-device metadata buffers for Option 1B fixed 16-slot ring conv state."""

    def __init__(self, max_num_seqs: int, max_num_tokens: int, device: torch.device) -> None:
        self.max_num_seqs = max_num_seqs
        self.dummy_slot = max_num_seqs
        self.max_num_tokens = max_num_tokens
        self.device = device
        self.block_table_buf = torch.full(
            (max_num_seqs, 1), self.dummy_slot, dtype=torch.int32, device=device
        )
        self.seq_idx_buf = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.query_start_buf = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.slot_mapping_buf = torch.full(
            (max_num_tokens,), -1, dtype=torch.int64, device=device
        )
        self.tail_src_toks = torch.zeros(max_num_seqs * 3, dtype=torch.int32, device=device)
        self.tail_req_slots = torch.zeros(max_num_seqs * 3, dtype=torch.int32, device=device)
        self.current_meta: InklingSconvMetadata | None = None

    def update(self, input_batch, cudagraph_mode: CUDAGraphMode, for_capture: bool) -> None:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs_p = input_batch.num_reqs_after_padding
            num_toks_p = input_batch.num_tokens_after_padding
        else:
            num_reqs_p = input_batch.num_reqs
            num_toks_p = input_batch.num_tokens
        num_reqs = input_batch.num_reqs
        num_actual = input_batch.num_tokens
        is_dummy = for_capture or bool(getattr(input_batch, "is_dummy_batch", False))
        if is_dummy:
            self.block_table_buf[:num_reqs_p, 0].fill_(self.dummy_slot)
        else:
            if num_reqs > 0:
                self.block_table_buf[:num_reqs, 0].copy_(input_batch.idx_mapping[:num_reqs])
            if num_reqs_p > num_reqs:
                self.block_table_buf[num_reqs:num_reqs_p, 0].fill_(self.dummy_slot)

        ring_conv_metadata(
            input_batch.query_start_loc,
            input_batch.positions,
            self.block_table_buf[:, 0],
            num_reqs,
            num_actual,
            self.seq_idx_buf,
            self.query_start_buf,
            self.slot_mapping_buf,
            num_toks_p,
            ring_n=RING_CONV_SLOTS,
            max_inplace_len=RING_MAX_INPLACE_LEN,
        )

        num_tail = 0
        if not is_dummy and num_reqs > 0:
            sched = input_batch.num_scheduled_tokens
            if int(sched.max()) > RING_MAX_INPLACE_LEN:
                qsl_np = input_batch.query_start_loc_np
                idx_np = input_batch.idx_mapping_np
                src_list: list[int] = []
                slot_list: list[int] = []
                for r in range(num_reqs):
                    if int(sched[r]) > RING_MAX_INPLACE_LEN:
                        qend = int(qsl_np[r + 1])
                        rslot = int(idx_np[r])
                        for tok in range(qend - 3, qend):
                            src_list.append(tok)
                            slot_list.append(rslot)
                num_tail = len(src_list)
                if num_tail > 0:
                    self.tail_src_toks[:num_tail].copy_(
                        torch.tensor(src_list, dtype=torch.int32), non_blocking=True
                    )
                    self.tail_req_slots[:num_tail].copy_(
                        torch.tensor(slot_list, dtype=torch.int32), non_blocking=True
                    )

        self.current_meta = InklingSconvMetadata(
            block_table=self.block_table_buf[:num_reqs_p],
            slot_mapping=self.slot_mapping_buf[:num_toks_p],
            seq_idx=self.seq_idx_buf[:num_toks_p],
            query_start=self.query_start_buf[:num_toks_p],
            tail_src_toks=self.tail_src_toks,
            tail_req_slots=self.tail_req_slots,
            num_tail_writes=num_tail,
        )


_RING_RUNTIME_BY_DEV: dict[torch.device, _RingConvRuntime] = {}
_RING_LOGGED = False


def _ensure_ring_runtime(vllm_config: VllmConfig, device: torch.device) -> _RingConvRuntime:
    rt = _RING_RUNTIME_BY_DEV.get(device)
    if rt is None:
        rt = _RingConvRuntime(
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            device=device,
        )
        _RING_RUNTIME_BY_DEV[device] = rt
    return rt


def install_ring_conv_hook() -> None:
    """Hook `DefaultModelState.prepare_attn` once to refresh persistent ring-conv metadata."""
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    if getattr(DefaultModelState.prepare_attn, "_inkling_ring_conv", False):
        return
    orig = DefaultModelState.prepare_attn

    def _wrapped_prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture: bool = False,
    ):
        attn_metadata = orig(
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )
        if _RING_RUNTIME_BY_DEV:
            rt = _RING_RUNTIME_BY_DEV.get(input_batch.positions.device)
            if rt is not None:
                rt.update(input_batch, cudagraph_mode, for_capture)
        return attn_metadata

    _wrapped_prepare_attn._inkling_ring_conv = True  # type: ignore[attr-defined]
    DefaultModelState.prepare_attn = _wrapped_prepare_attn


class _SharedSconvSeqMetaCache:
    """Share one `(seq_idx_buffer, query_start_buffer)` per device across all conv-state KV groups.

    When a mixed-attention drafter (e.g. DSpark with 1 global + 2 SWA layers) forces vLLM's
    hybrid KV planner to split the 48 InklingConvState layers into 24 two-layer KV groups,
    all 24 InklingSconvMetadataBuilder instances observe the exact same `query_start_loc`
    in a given step. Deduplicating `_seq_metadata_kernel` eliminates 23 redundant kernel
    launches per step while keeping identical persistent addresses for CUDA graph replay.
    """

    def __init__(self, max_num_tokens: int, device: torch.device) -> None:
        self.seq_idx_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        self.query_start_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        self.last_key: tuple[int, int, int, int] | None = None
        self.seen_builders: set[int] = set()


_SHARED_SCONV_CACHE: dict[tuple[torch.device, int], _SharedSconvSeqMetaCache] = {}


class InklingSconvMetadataBuilder(AttentionMetadataBuilder[InklingSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: SlidingWindowSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        cache_key = (device, max_num_tokens)
        cache = _SHARED_SCONV_CACHE.get(cache_key)
        if cache is None:
            cache = _SharedSconvSeqMetaCache(max_num_tokens, device)
            _SHARED_SCONV_CACHE[cache_key] = cache
        self._shared_cache = cache
        self.seq_idx_buffer = cache.seq_idx_buffer
        self.query_start_buffer = cache.query_start_buffer

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens
        key = (
            common_attn_metadata.query_start_loc.data_ptr(),
            num_reqs,
            num_actual_tokens,
            num_padded_tokens,
        )
        bid = id(self)
        cache = self._shared_cache
        if bid in cache.seen_builders or cache.last_key != key:
            cache.seen_builders.clear()
            sconv_seq_metadata(
                common_attn_metadata.query_start_loc,
                num_reqs,
                num_actual_tokens,
                self.seq_idx_buffer,
                self.query_start_buffer,
                num_padded_tokens,
            )
            cache.last_key = key
        cache.seen_builders.add(bid)
        return InklingSconvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class InklingSconvBackend(AttentionBackend):
    """Custom dummy backend for Inkling2 sliding-window short-conv state management."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "InklingSconvBackend has no attention impl; the conv runs out-of-band."
        )

    @staticmethod
    def get_builder_cls() -> type[InklingSconvMetadataBuilder]:
        return InklingSconvMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Head-major layout expected by ops/sconv.py: [num_blocks, H, N, D]
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [4, 8, 16, 32, 64, 128]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0 and (head_size & (head_size - 1)) == 0

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in (torch.float16, torch.bfloat16, torch.float32)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer (per-pass) owner emitting one sliding-window conv-state spec."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
        replicate_hidden: bool = True,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= num_kv_heads and num_kv_heads % tp_size == 0
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // (
            self.num_kv_heads if replicate_hidden else num_kv_heads
        )
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),
            (head_dim, head_dim),
            (2 * head_dim, hidden_per_head),
            (2 * head_dim + hidden_per_head, hidden_per_head),
        )
        vllm_config = get_current_vllm_config()
        self._dtype = torch.float16
        self.use_ring = ring_conv_enabled(vllm_config)
        if self.use_ring:
            global _RING_LOGGED
            dev = torch.device("cuda", torch.cuda.current_device())
            _ensure_ring_runtime(vllm_config, dev)
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            # Dedicated dummy slot at index `max_seqs` keeps CUDA graph capture & padded
            # slots strictly isolated from live request slots 0..max_seqs-1.
            self.kv_cache = torch.zeros(
                (max_seqs + 1, self.num_kv_heads, RING_CONV_SLOTS, raw_head_size),
                dtype=self._dtype,
                device=dev,
            )
            if not _RING_LOGGED:
                _RING_LOGGED = True
                mb_per_layer = self.kv_cache.numel() * self.kv_cache.element_size() / (1024**2)
                log.info(
                    "Inkling2 RingConv (Option 1B) active: %d+1 slots x %d ring x %d dim FP16 "
                    "(%.2f MiB/layer, 0 paged conv KV groups).",
                    max_seqs,
                    RING_CONV_SLOTS,
                    raw_head_size,
                    mb_per_layer,
                )
        else:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

    def forward(self):
        return None

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )


def conv_metadata(owner: InklingConvState) -> InklingSconvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    if owner.use_ring:
        rt = _RING_RUNTIME_BY_DEV.get(owner.kv_cache.device)
        return None if rt is None else rt.current_meta
    return cast(InklingSconvMetadata | None, attn_metadata.get(owner.prefix))


class Inkling2ShortConv(nn.Module):
    """Causal depthwise conv (+ residual, scaled) over one conv-state stream."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: InklingConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.stream = stream
        self.dynamic = dynamic_input_size is not None
        self.tp_rank = get_tensor_model_parallel_rank()
        if dynamic_input_size is not None:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim, dtype=torch.float16), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size, dtype=torch.float16),
                requires_grad=False,
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size, dtype=torch.float16), requires_grad=False
            )
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        param.data.copy_(loaded.to(param.dtype))

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def conv_weight(self) -> torch.Tensor:
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.dynamic:
            return None
        return torch.nn.functional.linear(x, self.weight_proj)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        assert not self.dynamic
        if owner is None:
            owner = self.owner
        m = conv_metadata(owner)
        if m is None:
            # Profile run (no conv state yet): skip conv, preserve FP32 residual sum.
            if addend is not None:
                acc = addend.float() if residual is None else residual.float() + addend.float()
                res = acc + x.float()
            else:
                res = x.float()
            if out is None:
                return res
            return out.copy_(res.to(out.dtype))
        offset, width = owner.stream_ranges[self.stream]
        return short_conv(
            x,
            self.conv_weight,
            owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            addend,
            residual,
            out,
            tail_src_toks=m.tail_src_toks,
            tail_req_slots=m.tail_req_slots,
            num_tail_writes=m.num_tail_writes,
        )
