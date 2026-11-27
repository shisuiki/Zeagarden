# SPDX-License-Identifier: Apache-2.0
"""Inkling2 short-conv state, RingConv (Option 1B), and RingAPC prefix-caching backend (sm_70).

Modes:
- `INKLING_RING_CONV=auto` (default): uses fixed per-request 16-slot FP16 ring buffers
  (`247.5 MiB/GPU`, 0 paged conv KV groups -> ~16-20x attention KV capacity).
  When `--enable-prefix-caching` is active, automatically enables **RingAPC**:
  lossless 3-token stride checkpoints (`INKLING_RING_APC_STRIDE`, default 512 tokens)
  stored in lazily allocated pinned host RAM (~1.41 MiB/checkpoint across all 48 layers),
  paired with EngineCore prefix-hit alignment to stride boundaries.
- `INKLING_RING_CONV=1`: enforces RingConv (with RingAPC if prefix caching is enabled).
- `INKLING_RING_CONV=0`: forces legacy `SlidingWindowSpec` paged conv cache.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
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
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .ops.sconv import ring_conv_metadata, sconv_seq_metadata, short_conv

log = init_logger("vllm.inkling_v100.conv_state")

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)
RING_CONV_SLOTS = 16  # >= 3 (W-1 history) + 8 (max DSpark verify Q) => zero overwrite on rejection
RING_MAX_INPLACE_LEN = 12  # 3 + 12 = 15 < 16 => provably disjoint read/write sets
RING_APC_DEFAULT_STRIDE = 512
RING_APC_DEFAULT_MAX_SLOTS = 3072
RING_APC_CORE_MARGIN = 256  # EngineCore keeps a strict subset so Worker LRU never misses a hit
RING_APC_CHUNK_SLOTS = 64

_INKLING_CONFIG_LOADED = False


def mark_inkling_config_loaded() -> None:
    global _INKLING_CONFIG_LOADED
    _INKLING_CONFIG_LOADED = True


def _ring_conv_env_enabled() -> bool:
    raw = os.environ.get("INKLING_RING_CONV", "auto").strip().lower()
    return raw not in ("0", "false", "no", "off")


def ring_conv_enabled(vllm_config: VllmConfig) -> bool:
    """Return True when Option 1B fixed per-request 16-slot ring conv buffer is active.

    Default (`auto`): enabled both with `--no-enable-prefix-caching` (pure RingConv) and
    with `--enable-prefix-caching` (RingAPC: stride-checkpointed pinned-CPU conv states),
    removing all 48 conv_state layers from vLLM's paged KV pool (~16-20x KV capacity).
    `INKLING_RING_CONV=0` forces legacy `SlidingWindowSpec` paged conv cache.
    """
    del vllm_config
    return _ring_conv_env_enabled()


def ring_apc_enabled(vllm_config: VllmConfig) -> bool:
    return ring_conv_enabled(vllm_config) and bool(
        vllm_config.cache_config.enable_prefix_caching
    )


def ring_apc_max_slots() -> int:
    raw = os.environ.get("INKLING_RING_APC_MAX_SLOTS", "").strip()
    if raw:
        val = int(raw)
        if val < RING_APC_CORE_MARGIN + 64:
            raise ValueError(
                f"INKLING_RING_APC_MAX_SLOTS must be >= {RING_APC_CORE_MARGIN + 64}, got {val}"
            )
        return val
    return RING_APC_DEFAULT_MAX_SLOTS


def select_ring_apc_anchor_group(kv_cache_config: KVCacheConfig) -> tuple[int, int] | None:
    """Deterministic target FullAttention group `(group_id, block_size)` shared by EngineCore and Workers."""
    groups = kv_cache_config.kv_cache_groups
    if not groups:
        return None
    # Prefer a target FullAttention group (excludes DSpark draft `self_attn` groups).
    for gid, g in enumerate(groups):
        if isinstance(g.kv_cache_spec, FullAttentionSpec) and not any(
            "self_attn" in ln for ln in g.layer_names
        ):
            return gid, g.kv_cache_spec.block_size
    for gid, g in enumerate(groups):
        if isinstance(g.kv_cache_spec, FullAttentionSpec):
            return gid, g.kv_cache_spec.block_size
    return 0, groups[0].kv_cache_spec.block_size


def resolve_ring_apc_stride(kv_cache_config: KVCacheConfig) -> int:
    """Resolve checkpoint stride in tokens as a positive multiple of all KV group block sizes."""
    groups = kv_cache_config.kv_cache_groups
    lcm_bs = 1
    for g in groups:
        lcm_bs = math.lcm(lcm_bs, g.kv_cache_spec.block_size)
    raw = os.environ.get("INKLING_RING_APC_STRIDE", "").strip()
    req = int(raw) if raw else RING_APC_DEFAULT_STRIDE
    if req <= 0:
        raise ValueError(f"INKLING_RING_APC_STRIDE must be > 0, got {req}")
    return max(lcm_bs, ((req + lcm_bs - 1) // lcm_bs) * lcm_bs)


@dataclass
class InklingSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks]
    slot_mapping: torch.Tensor  # [T] int64 (-1 => pad, -2 => long prefill read-only)
    seq_idx: torch.Tensor  # [T] int32
    query_start: torch.Tensor  # [T] int32
    tail_src_toks: torch.Tensor | None = None  # [max_reqs * 3] int32
    tail_req_slots: torch.Tensor | None = None  # [max_reqs * 3] int32
    num_tail_writes: int = 0
    ckpt_staging: torch.Tensor | None = None  # [cap, num_layers, 3, H, D]
    ckpt_b_pos: torch.Tensor | None = None  # [cap] int32
    ckpt_qstart: torch.Tensor | None = None  # [cap] int32
    ckpt_start_pos: torch.Tensor | None = None  # [cap] int32
    ckpt_req_slots: torch.Tensor | None = None  # [cap] int32
    num_ckpt_writes: int = 0

    def layer_ckpt_dst(self, layer_idx: int) -> torch.Tensor | None:
        if self.ckpt_staging is None or self.num_ckpt_writes <= 0 or layer_idx < 0:
            return None
        return self.ckpt_staging[:, layer_idx]


class _RingAPCCheckpointPool:
    """Lazily allocated pinned-CPU LRU pool storing 48-layer 3-token sconv checkpoints.

    Each checkpoint has shape `[num_layers, 3, num_kv_heads, raw_head_size]` in FP16
    (~1.41 MiB per checkpoint for Inkling-2-Mini at TP=4).
    Keyed by physical anchor-group `block_id`.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        raw_head_size: int,
        dtype: torch.dtype,
        max_slots: int,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.raw_head_size = raw_head_size
        self.dtype = dtype
        self.max_slots = max_slots
        self.device = device
        self.pin_memory = device.type == "cuda"

        self._chunks: list[torch.Tensor] = []
        self._allocated = 0
        self._free_slots: list[int] = []
        self._block_to_slot: OrderedDict[int, int] = OrderedDict()

        # Reusable GPU staging buffer for batched prefill capture, decode gather, and H2D restore.
        self._gpu_staging_cap = 0
        self._gpu_staging: torch.Tensor | None = None

    def ensure_gpu_staging(self, count: int) -> torch.Tensor:
        if count <= self._gpu_staging_cap and self._gpu_staging is not None:
            return self._gpu_staging
        new_cap = max(8, count)
        self._gpu_staging = torch.empty(
            (new_cap, self.num_layers, 3, self.num_kv_heads, self.raw_head_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._gpu_staging_cap = new_cap
        return self._gpu_staging

    def _slot_tensor(self, slot: int) -> torch.Tensor:
        c_idx = slot // RING_APC_CHUNK_SLOTS
        offset = slot % RING_APC_CHUNK_SLOTS
        return self._chunks[c_idx][offset]

    def _alloc_slot(self, block_id: int) -> torch.Tensor:
        existing = self._block_to_slot.get(block_id)
        if existing is not None:
            self._block_to_slot.move_to_end(block_id)
            return self._slot_tensor(existing)

        if self._free_slots:
            slot = self._free_slots.pop()
        elif self._allocated < self.max_slots:
            chunk_n = min(RING_APC_CHUNK_SLOTS, self.max_slots - self._allocated)
            chunk = torch.empty(
                (chunk_n, self.num_layers, 3, self.num_kv_heads, self.raw_head_size),
                dtype=self.dtype,
                pin_memory=self.pin_memory,
            )
            self._chunks.append(chunk)
            base = self._allocated
            self._allocated += chunk_n
            slot = base
            if chunk_n > 1:
                self._free_slots.extend(range(base + chunk_n - 1, base, -1))
        else:
            _, slot = self._block_to_slot.popitem(last=False)

        self._block_to_slot[block_id] = slot
        return self._slot_tensor(slot)

    def has(self, block_id: int) -> bool:
        return block_id in self._block_to_slot

    def get(self, block_id: int) -> torch.Tensor | None:
        slot = self._block_to_slot.get(block_id)
        if slot is None:
            return None
        self._block_to_slot.move_to_end(block_id)
        return self._slot_tensor(slot)

    def put_from_gpu(self, block_id: int, gpu_tensor: torch.Tensor) -> None:
        dst = self._alloc_slot(block_id)
        # Blocking D2H only when committing checkpoints (stride crossings are rare;
        # ~0.12 ms per 1.4 MiB checkpoint) so staging buffer is immediately safe to reuse.
        dst.copy_(gpu_tensor, non_blocking=False)

    def __len__(self) -> int:
        return len(self._block_to_slot)


class _RingConvRuntime:
    """Persistent per-device metadata buffers for Option 1B RingConv + RingAPC."""

    def __init__(
        self,
        max_num_seqs: int,
        max_num_tokens: int,
        device: torch.device,
        apc_enabled: bool = False,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.dummy_slot = max_num_seqs
        self.max_num_tokens = max_num_tokens
        self.device = device
        self.apc_enabled = apc_enabled

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

        # RingAPC state (active only when `apc_enabled=True`)
        self.layers: list[InklingConvState] = []
        self._apc_pool: _RingAPCCheckpointPool | None = None
        self._anchor_gid: int | None = None
        self._anchor_bs: int = 32
        self._stride: int = RING_APC_DEFAULT_STRIDE
        self._apc_logged: bool = False

        # Per-slot tracking of requests needing prefix restore on first forward
        self._pending_restore_slots: set[int] = set()
        self._slot_req_id: dict[int, str] = {}
        self._slot_last_end: dict[int, int] = {}

        # Per-step checkpoint capture plans (drained in `commit_step()` right after forward)
        self._ckpt_meta_cap = 0
        self._ckpt_b_pos: torch.Tensor | None = None
        self._ckpt_qstart: torch.Tensor | None = None
        self._ckpt_start_pos: torch.Tensor | None = None
        self._ckpt_req_slots: torch.Tensor | None = None
        self._prefill_ckpt_blocks: list[int] = []
        self._post_ring_saves: list[tuple[int, int, int]] = []  # (req_slot, b_pos, phys_block_id)

    def register_layer(self, layer: InklingConvState) -> int:
        idx = len(self.layers)
        self.layers.append(layer)
        return idx

    def on_add_request(self, req_index: int, num_computed_tokens: int) -> None:
        if not self.apc_enabled:
            return
        self._slot_req_id.pop(req_index, None)
        self._slot_last_end.pop(req_index, None)
        if num_computed_tokens > 0:
            self._pending_restore_slots.add(req_index)
        else:
            self._pending_restore_slots.discard(req_index)

    def on_remove_request(self, req_index: int) -> None:
        self._pending_restore_slots.discard(req_index)
        self._slot_req_id.pop(req_index, None)
        self._slot_last_end.pop(req_index, None)

    def _ensure_apc_ready(self, kv_cache_config: KVCacheConfig) -> _RingAPCCheckpointPool | None:
        if not self.apc_enabled or not self.layers:
            return None
        if self._anchor_gid is None:
            anchor = select_ring_apc_anchor_group(kv_cache_config)
            if anchor is None:
                return None
            self._anchor_gid, self._anchor_bs = anchor
            self._stride = resolve_ring_apc_stride(kv_cache_config)
        if self._apc_pool is None:
            first = self.layers[0]
            self._apc_pool = _RingAPCCheckpointPool(
                num_layers=len(self.layers),
                num_kv_heads=first.kv_cache.shape[1],
                raw_head_size=first.kv_cache.shape[3],
                dtype=first.kv_cache.dtype,
                max_slots=ring_apc_max_slots(),
                device=self.device,
            )
            if not self._apc_logged:
                self._apc_logged = True
                mb_per_ckpt = (
                    len(self.layers)
                    * 3
                    * first.kv_cache.shape[1]
                    * first.kv_cache.shape[3]
                    * first.kv_cache.element_size()
                    / (1024**2)
                )
                log.info(
                    "Inkling2 RingAPC active: stride=%d tok, anchor_group=%d (bs=%d), "
                    "max_slots=%d (%.2f MiB/ckpt in lazy pinned host RAM, 0 paged conv KV groups).",
                    self._stride,
                    self._anchor_gid,
                    self._anchor_bs,
                    self._apc_pool.max_slots,
                    mb_per_ckpt,
                )
        return self._apc_pool

    def _ensure_ckpt_meta_bufs(self, count: int) -> None:
        if count <= self._ckpt_meta_cap and self._ckpt_b_pos is not None:
            return
        new_cap = max(8, count)
        self._ckpt_b_pos = torch.zeros(new_cap, dtype=torch.int32, device=self.device)
        self._ckpt_qstart = torch.zeros(new_cap, dtype=torch.int32, device=self.device)
        self._ckpt_start_pos = torch.zeros(new_cap, dtype=torch.int32, device=self.device)
        self._ckpt_req_slots = torch.zeros(new_cap, dtype=torch.int32, device=self.device)
        self._ckpt_meta_cap = new_cap

    def _restore_slot_checkpoint(
        self,
        pool: _RingAPCCheckpointPool,
        req_slot: int,
        start_pos: int,
        phys_block_id: int,
    ) -> None:
        cpu_ckpt = pool.get(phys_block_id)
        if cpu_ckpt is None:
            raise RuntimeError(
                f"Inkling2 RingAPC missing conv checkpoint for anchor phys_block_id={phys_block_id} "
                f"(start_pos={start_pos}, req_slot={req_slot}, pool_size={len(pool)}). "
                "Increase INKLING_RING_APC_MAX_SLOTS or verify EngineCore/Worker stride alignment."
            )
        staging = pool.ensure_gpu_staging(1)
        staging[0].copy_(cpu_ckpt, non_blocking=True)
        # Copy 3 tokens [start_pos - 3, start_pos - 2, start_pos - 1] into ring slots
        r0 = (start_pos - 3) & (RING_CONV_SLOTS - 1)
        r1 = (start_pos - 2) & (RING_CONV_SLOTS - 1)
        r2 = (start_pos - 1) & (RING_CONV_SLOTS - 1)
        # staging[0] has shape [L, 3, H, D]; owner.kv_cache[req_slot] has shape [H, 16, D]
        if r1 == r0 + 1 and r2 == r1 + 1:
            # Fast contiguous slice (always true when start_pos % 16 == 0 => r0=13, r1=14, r2=15!)
            src_all = staging[0].transpose(1, 2)  # [L, H, 3, D]
            for l_idx, layer in enumerate(self.layers):
                layer.kv_cache[req_slot, :, r0 : r0 + 3, :].copy_(src_all[l_idx])
        else:
            ring_idx = (r0, r1, r2)
            for l_idx, layer in enumerate(self.layers):
                dst_cache = layer.kv_cache[req_slot]
                src_l = staging[0, l_idx]  # [3, H, D]
                for t_i, r_slot in enumerate(ring_idx):
                    dst_cache[:, r_slot, :].copy_(src_l[t_i])

    def update(
        self,
        input_batch,
        cudagraph_mode: CUDAGraphMode,
        for_capture: bool,
        block_tables: tuple[torch.Tensor, ...] | None = None,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
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

        # Reset per-step RingAPC capture plans
        self._prefill_ckpt_blocks.clear()
        self._post_ring_saves.clear()
        ckpt_staging: torch.Tensor | None = None
        num_ckpt_writes = 0

        num_tail = 0
        if not is_dummy and num_reqs > 0:
            sched = input_batch.num_scheduled_tokens
            qsl_np = input_batch.query_start_loc_np
            idx_np = input_batch.idx_mapping_np

            if int(sched.max()) > RING_MAX_INPLACE_LEN:
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

            # RingAPC: restore checkpoints for prefix-hit requests & plan stride captures.
            if (
                self.apc_enabled
                and block_tables is not None
                and kv_cache_config is not None
            ):
                pool = self._ensure_apc_ready(kv_cache_config)
                if pool is not None and self._anchor_gid is not None:
                    anchor_bt = block_tables[self._anchor_gid]
                    anchor_bs = self._anchor_bs
                    stride = self._stride
                    req_ids = input_batch.req_ids
                    comp_np = input_batch.num_computed_tokens_np

                    pf_b_pos: list[int] = []
                    pf_qstart: list[int] = []
                    pf_start_pos: list[int] = []
                    pf_req_slots: list[int] = []

                    for r in range(num_reqs):
                        rslot = int(idx_np[r])
                        rid = req_ids[r]
                        start = int(comp_np[r])
                        rlen = int(sched[r])
                        end = start + rlen

                        # Detect prefix-hit restore:
                        # 1) Slot marked by `add_request` with num_computed_tokens > 0
                        # 2) New req_id occupying this slot with start > 0
                        # 3) Resumed/recomputed request whose start jumped outside live ring
                        prev_rid = self._slot_req_id.get(rslot)
                        prev_end = self._slot_last_end.get(rslot, -1)
                        needs_restore = False
                        if start > 0:
                            if rslot in self._pending_restore_slots:
                                needs_restore = True
                            elif prev_rid != rid:
                                needs_restore = True
                            elif prev_end >= 0 and start > prev_end:
                                needs_restore = True
                        self._pending_restore_slots.discard(rslot)

                        if needs_restore:
                            if start % stride != 0:
                                raise RuntimeError(
                                    f"Inkling2 RingAPC expected prefix hit aligned to stride={stride}, "
                                    f"got start={start} for req_id={rid}"
                                )
                            blk_idx = (start // anchor_bs) - 1
                            phys_id = int(anchor_bt[r, blk_idx].item())
                            self._restore_slot_checkpoint(pool, rslot, start, phys_id)

                        self._slot_req_id[rslot] = rid
                        self._slot_last_end[rslot] = end

                        # Plan stride checkpoints for all `b_pos = k * stride - 1` in `[start, end)`.
                        # Also re-save the immediately preceding boundary during short/decode steps
                        # while slots (13, 14, 15) are still unoverwritten in the 16-slot ring,
                        # guarding against async-scheduled speculative rejection straddling a stride.
                        if rlen <= RING_MAX_INPLACE_LEN and start >= stride:
                            prev_b_pos = (start // stride) * stride - 1
                            if (
                                start - prev_b_pos <= RING_MAX_INPLACE_LEN
                                and end <= prev_b_pos + (RING_CONV_SLOTS - 2)
                            ):
                                prev_phys_id = int(anchor_bt[r, prev_b_pos // anchor_bs].item())
                                self._post_ring_saves.append((rslot, prev_b_pos, prev_phys_id))

                        first_k = (start // stride) + 1
                        b_pos = first_k * stride - 1
                        if b_pos < end:
                            qstart = int(qsl_np[r])
                            while b_pos < end:
                                phys_id = int(anchor_bt[r, b_pos // anchor_bs].item())
                                if rlen <= RING_MAX_INPLACE_LEN:
                                    self._post_ring_saves.append((rslot, b_pos, phys_id))
                                else:
                                    pf_b_pos.append(b_pos)
                                    pf_qstart.append(qstart)
                                    pf_start_pos.append(start)
                                    pf_req_slots.append(rslot)
                                    self._prefill_ckpt_blocks.append(phys_id)
                                b_pos += stride

                    num_ckpt_writes = len(self._prefill_ckpt_blocks)
                    if num_ckpt_writes > 0:
                        self._ensure_ckpt_meta_bufs(num_ckpt_writes)
                        ckpt_staging = pool.ensure_gpu_staging(num_ckpt_writes)
                        assert self._ckpt_b_pos is not None
                        assert self._ckpt_qstart is not None
                        assert self._ckpt_start_pos is not None
                        assert self._ckpt_req_slots is not None
                        self._ckpt_b_pos[:num_ckpt_writes].copy_(
                            torch.tensor(pf_b_pos, dtype=torch.int32), non_blocking=True
                        )
                        self._ckpt_qstart[:num_ckpt_writes].copy_(
                            torch.tensor(pf_qstart, dtype=torch.int32), non_blocking=True
                        )
                        self._ckpt_start_pos[:num_ckpt_writes].copy_(
                            torch.tensor(pf_start_pos, dtype=torch.int32), non_blocking=True
                        )
                        self._ckpt_req_slots[:num_ckpt_writes].copy_(
                            torch.tensor(pf_req_slots, dtype=torch.int32), non_blocking=True
                        )

        self.current_meta = InklingSconvMetadata(
            block_table=self.block_table_buf[:num_reqs_p],
            slot_mapping=self.slot_mapping_buf[:num_toks_p],
            seq_idx=self.seq_idx_buf[:num_toks_p],
            query_start=self.query_start_buf[:num_toks_p],
            tail_src_toks=self.tail_src_toks,
            tail_req_slots=self.tail_req_slots,
            num_tail_writes=num_tail,
            ckpt_staging=ckpt_staging,
            ckpt_b_pos=self._ckpt_b_pos,
            ckpt_qstart=self._ckpt_qstart,
            ckpt_start_pos=self._ckpt_start_pos,
            ckpt_req_slots=self._ckpt_req_slots,
            num_ckpt_writes=num_ckpt_writes,
        )

    def discard_pending(self) -> None:
        self._prefill_ckpt_blocks.clear()
        self._post_ring_saves.clear()

    def commit_step(self) -> None:
        """Persist planned stride checkpoints to pinned host RAM right after forward/graph replay."""
        pool = self._apc_pool
        if pool is None:
            self._prefill_ckpt_blocks.clear()
            self._post_ring_saves.clear()
            return

        # 1. Commit checkpoints captured in-flight during eager long prefill.
        if self._prefill_ckpt_blocks:
            staging = pool.ensure_gpu_staging(len(self._prefill_ckpt_blocks))
            for idx, phys_id in enumerate(self._prefill_ckpt_blocks):
                pool.put_from_gpu(phys_id, staging[idx])
            self._prefill_ckpt_blocks.clear()

        # 2. Commit checkpoints written in-place to ring buffers during decode / short prefill.
        if self._post_ring_saves:
            staging = pool.ensure_gpu_staging(1)
            for req_slot, b_pos, phys_id in self._post_ring_saves:
                r0 = (b_pos - 2) & (RING_CONV_SLOTS - 1)
                r1 = (b_pos - 1) & (RING_CONV_SLOTS - 1)
                r2 = b_pos & (RING_CONV_SLOTS - 1)
                if r1 == r0 + 1 and r2 == r1 + 1:
                    for l_idx, layer in enumerate(self.layers):
                        # kv_cache[req_slot, :, r0:r0+3, :] is [H, 3, D] -> transpose to [3, H, D]
                        staging[0, l_idx].copy_(
                            layer.kv_cache[req_slot, :, r0 : r0 + 3, :].transpose(0, 1)
                        )
                else:
                    ring_idx = (r0, r1, r2)
                    for l_idx, layer in enumerate(self.layers):
                        src_cache = layer.kv_cache[req_slot]
                        for t_i, r_slot in enumerate(ring_idx):
                            staging[0, l_idx, t_i].copy_(src_cache[:, r_slot, :])
                pool.put_from_gpu(phys_id, staging[0])
            self._post_ring_saves.clear()


_RING_RUNTIME_BY_DEV: dict[torch.device, _RingConvRuntime] = {}
_RING_LOGGED = False


def _ensure_ring_runtime(vllm_config: VllmConfig, device: torch.device) -> _RingConvRuntime:
    rt = _RING_RUNTIME_BY_DEV.get(device)
    if rt is None:
        rt = _RingConvRuntime(
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            device=device,
            apc_enabled=ring_apc_enabled(vllm_config),
        )
        _RING_RUNTIME_BY_DEV[device] = rt
    return rt


class _RingAPCCoreTracker:
    """EngineCore-side stride-aligned prefix checkpoint tracker.

    Mirrors the set of anchor-group physical block IDs whose conv checkpoints are
    guaranteed to be resident in every TP worker's `_RingAPCCheckpointPool`.
    """

    def __init__(
        self,
        anchor_gid: int,
        anchor_bs: int,
        stride_tokens: int,
        max_slots: int,
    ) -> None:
        self.anchor_gid = anchor_gid
        self.anchor_bs = anchor_bs
        self.stride_tokens = stride_tokens
        self.stride_blocks = stride_tokens // anchor_bs
        self.max_slots = max(64, max_slots - RING_APC_CORE_MARGIN)
        self.valid_blocks: OrderedDict[int, None] = OrderedDict()

    def on_block_cached(self, group_id: int, logical_block_idx: int, block_id: int) -> None:
        if group_id != self.anchor_gid:
            return
        if (logical_block_idx + 1) % self.stride_blocks != 0:
            return
        if block_id in self.valid_blocks:
            self.valid_blocks.move_to_end(block_id)
            return
        if len(self.valid_blocks) >= self.max_slots:
            self.valid_blocks.popitem(last=False)
        self.valid_blocks[block_id] = None

    def on_block_evicted(self, block_id: int) -> None:
        self.valid_blocks.pop(block_id, None)

    def has_and_touch(self, block_id: int) -> bool:
        if block_id not in self.valid_blocks:
            return False
        self.valid_blocks.move_to_end(block_id)
        return True


def _is_inkling_ring_apc_kv_config(kv_cache_config: KVCacheConfig) -> bool:
    if not _ring_conv_env_enabled():
        return False
    groups = kv_cache_config.kv_cache_groups
    if not groups:
        return False
    # If legacy paged conv_state layers are present, RingConv is NOT active.
    for g in groups:
        if any("conv_state" in ln for ln in g.layer_names):
            return False
    if _INKLING_CONFIG_LOADED:
        return True
    # Fallback signature for looped Inkling-2 checkpoints in EngineCore subprocesses.
    return any(".attn.pass" in ln for g in groups for ln in g.layer_names)


def _install_ring_apc_engine_core_hooks() -> None:
    """Hook EngineCore's KVCacheManager/Coordinator/BlockPool to align APC hits for RingAPC."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import (
        HybridKVCacheCoordinator,
        UnitaryKVCacheCoordinator,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager.__init__, "_inkling_ring_apc", False):
        return

    orig_mgr_init = KVCacheManager.__init__

    def _wrapped_mgr_init(self, *args, **kwargs):
        orig_mgr_init(self, *args, **kwargs)
        if self.enable_caching and _is_inkling_ring_apc_kv_config(self.kv_cache_config):
            anchor = select_ring_apc_anchor_group(self.kv_cache_config)
            if anchor is not None:
                anchor_gid, anchor_bs = anchor
                stride = resolve_ring_apc_stride(self.kv_cache_config)
                tracker = _RingAPCCoreTracker(
                    anchor_gid=anchor_gid,
                    anchor_bs=anchor_bs,
                    stride_tokens=stride,
                    max_slots=ring_apc_max_slots(),
                )
                self.coordinator._inkling_ring_apc_tracker = tracker
                self.block_pool._inkling_ring_apc_tracker = tracker
                # Align SWA managers to RingAPC's stride and retain SWA replay windows
                # in normal LRU (instead of eager `reuse_first=True` recycling) so
                # stride-aligned prompt ends, multi-turn reply ends, and branching
                # shared prefixes all hit at the maximum theoretical prefix length.
                for mgr in getattr(self.coordinator, "single_type_managers", ()):
                    if hasattr(mgr, "alignment_tokens"):
                        mgr.alignment_tokens = stride
                    reuse_win = getattr(mgr, "reuse_window", None)
                    if reuse_win is not None:
                        if hasattr(mgr, "hold_reply_window"):
                            mgr.hold_reply_window = True
                        if (
                            hasattr(mgr, "retention_interval")
                            and mgr.retention_interval is None
                        ):
                            win_tok = int(reuse_win)
                            mgr.retention_interval = max(
                                stride, ((win_tok + stride - 1) // stride) * stride
                            )
                log.info(
                    "Inkling2 RingAPC EngineCore enabled: stride=%d tok, anchor_group=%d (bs=%d), "
                    "core_tracked_slots=%d (worker_max=%d).",
                    stride,
                    anchor_gid,
                    anchor_bs,
                    tracker.max_slots,
                    ring_apc_max_slots(),
                )

    _wrapped_mgr_init._inkling_ring_apc = True  # type: ignore[attr-defined]
    KVCacheManager.__init__ = _wrapped_mgr_init

    # Hook BlockPool.cache_full_blocks and _maybe_evict_cached_block
    orig_cache_full = BlockPool.cache_full_blocks

    def _wrapped_cache_full_blocks(
        self,
        request,
        blocks,
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask=None,
    ):
        orig_cache_full(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask=block_mask,
        )
        tracker: _RingAPCCoreTracker | None = getattr(self, "_inkling_ring_apc_tracker", None)
        if (
            tracker is not None
            and kv_cache_group_id == tracker.anchor_gid
            and num_cached_blocks < num_full_blocks
        ):
            for i in range(num_cached_blocks, num_full_blocks):
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                blk = blocks[i]
                if not blk.is_null and blk.block_hash is not None:
                    tracker.on_block_cached(kv_cache_group_id, i, blk.block_id)

    BlockPool.cache_full_blocks = _wrapped_cache_full_blocks

    orig_evict = BlockPool._maybe_evict_cached_block

    def _wrapped_evict(self, block):
        evicted = orig_evict(self, block)
        if evicted:
            tracker: _RingAPCCoreTracker | None = getattr(
                self, "_inkling_ring_apc_tracker", None
            )
            if tracker is not None:
                tracker.on_block_evicted(block.block_id)
        return evicted

    BlockPool._maybe_evict_cached_block = _wrapped_evict

    # Hook coordinator.find_longest_cache_hit so hits are aligned to stride boundaries
    # AND re-queried through the coordinator (keeping SWA-1024 + DSpark SWA windows exact!).
    def _wrap_find_longest(cls):
        orig_find = cls.find_longest_cache_hit

        def _empty_hit(coord):
            if hasattr(coord, "shared_prefix_boundary"):
                coord.shared_prefix_boundary = 0
            n_groups = len(coord.single_type_managers)
            return tuple([] for _ in range(n_groups)), 0

        def _wrapped_find(self, block_hashes, max_cache_hit_length: int):
            tracker: _RingAPCCoreTracker | None = getattr(
                self, "_inkling_ring_apc_tracker", None
            )
            if tracker is None:
                return orig_find(self, block_hashes, max_cache_hit_length)
            if max_cache_hit_length < tracker.stride_tokens:
                return _empty_hit(self)

            stride = tracker.stride_tokens
            # Account for EAGLE/DFlash +1 block lookahead drop so aligned hits
            # do not cascade downward by `eagle_pad` on re-query.
            eagle_pad = 0
            attn_groups = getattr(self, "attention_groups", None)
            if attn_groups:
                for ag in attn_groups:
                    if getattr(ag, "use_eagle", False):
                        eagle_pad = max(eagle_pad, ag.spec.block_size)
            elif getattr(self, "drop_eagle_group_ids", None):
                eagle_pad = getattr(self, "block_size", 0)

            eff_max = max_cache_hit_length - eagle_pad
            if eff_max < stride:
                return _empty_hit(self)

            cap = (eff_max // stride) * stride
            while cap >= stride:
                query_max = min(max_cache_hit_length, cap + eagle_pad)
                blocks, hit_len = orig_find(self, block_hashes, query_max)
                aligned = (min(hit_len, cap) // stride) * stride
                if aligned < stride:
                    return _empty_hit(self)
                if aligned < hit_len:
                    # Re-query with cap=aligned so SlidingWindowManager fetches the exact
                    # [aligned - window, aligned) blocks instead of slicing null-padded tails.
                    cap = aligned
                    continue
                # Verify the anchor physical block has a live conv checkpoint.
                anchor_blocks = blocks[tracker.anchor_gid]
                blk_idx = (aligned // tracker.anchor_bs) - 1
                if blk_idx < len(anchor_blocks):
                    phys_id = anchor_blocks[blk_idx].block_id
                    if tracker.has_and_touch(phys_id):
                        return blocks, aligned
                cap = aligned - stride

            return _empty_hit(self)

        cls.find_longest_cache_hit = _wrapped_find

    _wrap_find_longest(HybridKVCacheCoordinator)
    _wrap_find_longest(UnitaryKVCacheCoordinator)


def install_ring_conv_hook() -> None:
    """Install worker & EngineCore hooks for RingConv and RingAPC."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    _install_ring_apc_engine_core_hooks()

    if getattr(DefaultModelState.prepare_attn, "_inkling_ring_conv", False):
        return

    orig_prepare = DefaultModelState.prepare_attn
    orig_add_req = DefaultModelState.add_request
    orig_rm_req = DefaultModelState.remove_request
    orig_exec_model = GPUModelRunner.execute_model

    def _wrapped_add_request(self, req_index: int, new_req_data) -> None:
        orig_add_req(self, req_index, new_req_data)
        if _RING_RUNTIME_BY_DEV:
            rt = _RING_RUNTIME_BY_DEV.get(self.device)
            if rt is not None:
                rt.on_add_request(req_index, int(new_req_data.num_computed_tokens))

    def _wrapped_remove_request(self, req_index: int) -> None:
        orig_rm_req(self, req_index)
        if _RING_RUNTIME_BY_DEV:
            rt = _RING_RUNTIME_BY_DEV.get(self.device)
            if rt is not None:
                rt.on_remove_request(req_index)

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
        attn_metadata = orig_prepare(
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
                rt.update(
                    input_batch,
                    cudagraph_mode,
                    for_capture,
                    block_tables=block_tables,
                    kv_cache_config=kv_cache_config,
                )
        return attn_metadata

    def _wrapped_execute_model(self, *args, **kwargs):
        result = orig_exec_model(self, *args, **kwargs)
        if _RING_RUNTIME_BY_DEV:
            rt = _RING_RUNTIME_BY_DEV.get(self.device)
            if rt is not None and rt.apc_enabled:
                if kwargs.get("dummy_run", False):
                    rt.discard_pending()
                else:
                    rt.commit_step()
        return result

    _wrapped_prepare_attn._inkling_ring_conv = True  # type: ignore[attr-defined]
    DefaultModelState.add_request = _wrapped_add_request
    DefaultModelState.remove_request = _wrapped_remove_request
    DefaultModelState.prepare_attn = _wrapped_prepare_attn
    GPUModelRunner.execute_model = _wrapped_execute_model


class _SharedSconvSeqMetaCache:
    """Share one `(seq_idx_buffer, query_start_buffer)` per device across all conv-state KV groups."""

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
    """Per-decoder-layer (per-pass) owner managing RingConv/RingAPC or legacy paged conv state."""

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
        self.ring_layer_idx: int = -1
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
            rt = _ensure_ring_runtime(vllm_config, dev)
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            # Dedicated dummy slot at index `max_seqs` keeps CUDA graph capture & padded
            # slots strictly isolated from live request slots 0..max_seqs-1.
            self.kv_cache = torch.zeros(
                (max_seqs + 1, self.num_kv_heads, RING_CONV_SLOTS, raw_head_size),
                dtype=self._dtype,
                device=dev,
            )
            self.ring_layer_idx = rt.register_layer(self)
            if not _RING_LOGGED:
                _RING_LOGGED = True
                mb_per_layer = self.kv_cache.numel() * self.kv_cache.element_size() / (1024**2)
                log.info(
                    "Inkling2 RingConv (Option 1B%s) active: %d+1 slots x %d ring x %d dim FP16 "
                    "(%.2f MiB/layer, 0 paged conv KV groups).",
                    "+RingAPC" if rt.apc_enabled else "",
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
            ckpt_dst=m.layer_ckpt_dst(owner.ring_layer_idx),
            ckpt_b_pos=m.ckpt_b_pos,
            ckpt_qstart=m.ckpt_qstart,
            ckpt_start_pos=m.ckpt_start_pos,
            ckpt_req_slots=m.ckpt_req_slots,
            num_ckpt_writes=m.num_ckpt_writes,
        )
