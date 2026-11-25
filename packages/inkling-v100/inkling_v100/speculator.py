# SPDX-License-Identifier: Apache-2.0
"""1Cat-vLLM V2 speculator subclass driving Inkling2's native DSpark 7-token drafter."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.utils import AttentionGroup

from .spec_k import capture_counts, k_for, k_ranges, spec_k_config

log = init_logger("vllm.inkling_v100.speculator")


class Inkling2DSparkSpeculator(DFlashSpeculator):
    """Subclass of 1Cat's `DFlashSpeculator` with DFlare raw-tap storage & recurrent Markov head.

    Optimizations for Volta (sm_70):
    - Runs the entire 3-layer draft forward + 7-step recurrent Markov head + TP-local
      argmax inside FULL CUDA graphs (`DFlashCudaGraphManager`).
    - Captures `FlashAttnV100MetadataBuilder.copy_dflash_graph_metadata` CUDA graphs so
      FULL draft graph replays skip 2x `FlashAttnV100MetadataBuilder.build()` (~8.5 ms/step).
    - Captures `precompute_and_store_context_kv` CUDA graphs for uniform decode sizes
      (~1.2 ms/step CPU dispatch eliminated).
    - Strips draft-only attention groups from `target_attn_groups` after KV initialization
      so target `prepare_attn()` never builds unused draft metadata (~5.5 ms/step saved).
    """

    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf_cfg = self.draft_model_config.hf_config
        target_layer_ids = getattr(hf_cfg, "target_layer_ids", None) or getattr(
            hf_cfg, "eagle_aux_hidden_state_layer_ids", [5, 23, 41, 48]
        )
        num_taps = len(target_layer_ids)
        hidden_size = int(getattr(hf_cfg, "hidden_size", self.hidden_size))
        # Each draft layer mixes the raw 4 target hidden taps via its own DFlare norm.
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            num_taps * hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # Persistent GPU buffer populated right before FULL draft graph replay.
        self.f_rnn = torch.zeros(
            self.max_num_reqs,
            hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._cur_num_reqs: int = 0
        self._cur_qsl: torch.Tensor | None = None
        self._cur_num_rejected: torch.Tensor | None = None
        self._target_attn_groups: list[list[AttentionGroup]] | None = None
        self._draft_metadata_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._context_kv_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._spec_k_cfg = spec_k_config()
        self._active_k: int = self.draft_block
        self._draft_graph_req_sizes: list[int] = []

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        return self.dtype, 0.0

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = load_dflash_model(target_model, self.vllm_config)
        target_lm = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        model.attach_target(target_lm)
        return model

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._target_attn_groups = target_attn_groups

    def capture(self) -> None:
        if self.query_cudagraph_manager is not None:
            descs = getattr(self.query_cudagraph_manager, "_capture_descs", {})
            req_set: set[int] = set()
            for dl in descs.values():
                for d in dl:
                    nr = d.num_reqs or (d.num_tokens // self.num_query_per_req)
                    if nr > 0:
                        req_set.add(nr)
            self._draft_graph_req_sizes = sorted(req_set)
        super().capture()
        self._strip_draft_groups_from_target()
        self._capture_fast_refresh_graphs()

    def next_num_draft_tokens(self) -> int:
        return self._active_k

    def _draft_k_for_bucket(self, num_reqs_bucket: int) -> int:
        if self._spec_k_cfg is None:
            return self.draft_block
        # Cover the maximum k needed by any live request count that pads up to `num_reqs_bucket`.
        prev = 0
        for sz in self._draft_graph_req_sizes:
            if sz < num_reqs_bucket:
                prev = sz
        lo = prev + 1
        k_req = max(
            k_for(r, self.draft_block, self._spec_k_cfg)
            for r in range(lo, max(lo, num_reqs_bucket) + 1)
        )
        return max(1, min(k_req, self.draft_block))

    def propose(self, input_batch: InputBatch, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self._spec_k_cfg is not None:
            k = k_for(input_batch.num_reqs, self.draft_block, self._spec_k_cfg)
            self._active_k = k
            if k == 0 and not kwargs.get("dummy_run") and not kwargs.get("is_profile"):
                return self.draft_tokens[: input_batch.num_reqs]
        else:
            self._active_k = self.draft_block
        return super().propose(input_batch, *args, **kwargs)

    def _strip_draft_groups_from_target(self) -> None:
        """Remove draft-only AttentionGroups from target `model_runner.attn_groups`.

        Runs strictly after `init_kv_cache` and `_init_kv_zero_meta` (inside `capture()`).
        Target forwards never consume draft attention metadata; skipping draft builders
        avoids two redundant `FlashAttnV100MetadataBuilder.build()` calls every step.
        """
        if self._target_attn_groups is None or not self.draft_attn_layer_names:
            return
        stripped = 0
        for gid, groups in enumerate(self._target_attn_groups):
            kept = []
            for group in groups:
                if set(group.layer_names).issubset(self.draft_attn_layer_names):
                    stripped += 1
                else:
                    kept.append(group)
            if len(kept) != len(groups):
                self._target_attn_groups[gid] = kept
        if stripped:
            log.info(
                "Inkling2 DSpark: stripped %d draft-only AttentionGroup(s) from target prepare_attn.",
                stripped,
            )

    def _capture_fast_refresh_graphs(self) -> None:
        """Capture tiny CUDA graphs for draft metadata refresh & uniform context KV projection."""
        if self.query_cudagraph_manager is None:
            return
        shapes: list[tuple[int, int]] = []
        for desc in self.query_cudagraph_manager.graphs:
            if desc.cg_mode == CUDAGraphMode.FULL:
                nr = desc.num_reqs or (desc.num_tokens // self.num_query_per_req)
                shapes.append((nr, desc.num_tokens))
        if not shapes:
            return

        # 1. Capture draft metadata copy graphs (eliminates FlashAttnV100MetadataBuilder.build per step).
        try:
            from vllm.v1.attention.backends.flash_v100 import (
                FlashAttnV100Impl,
                FlashAttnV100MetadataBuilder,
            )

            has_causal = (
                any(self._group_causal.values())
                if isinstance(self._group_causal, dict)
                else bool(self._group_causal)
            )
            layers_ok = all(
                isinstance(layer.self_attn.attn.impl, FlashAttnV100Impl)
                and not layer.self_attn.attn.impl.use_triton_prefill
                and layer.self_attn.attn.impl.use_flash_v100_prefill_paged
                and layer.self_attn.attn.impl.prefix_anchored_decode_window is None
                for layer in self.model.model.layers
            )
            builders: list[tuple[int, Any]] = []
            if self.block_tables.cp_size == 1 and not has_causal and layers_ok:
                for gid, groups in enumerate(self.attn_groups):
                    for group in groups:
                        builder = group.get_metadata_builder(0)
                        if (
                            isinstance(builder, FlashAttnV100MetadataBuilder)
                            and builder.metadata_workspace.draft.shape is not None
                        ):
                            builders.append((gid, builder))
                        else:
                            builders.clear()
                            break
                    else:
                        continue
                    break
            for num_reqs, num_tokens in shapes:
                if not builders:
                    break
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for gid, builder in builders:
                        builder.copy_dflash_graph_metadata(
                            self.block_tables.input_block_tables[gid][:num_reqs],
                            self.input_buffers.seq_lens[:num_reqs],
                            self.input_buffers.query_start_loc[: num_reqs + 1],
                        )
                self._draft_metadata_graphs[(num_reqs, num_tokens)] = graph
        except Exception as exc:
            log.warning("Inkling2 DSpark: skipped draft metadata graph capture: %s", exc)
            self._draft_metadata_graphs.clear()

        # 2. Capture context KV projection + cache-update graphs for uniform verify sizes.
        if self._layer_group_idx is not None:
            nt_set = {nt for _, nt in shapes if nt <= self.max_num_tokens}
            if self._spec_k_cfg is not None:
                for k, lo, hi in k_ranges(
                    self._spec_k_cfg, self.draft_block, self.max_num_reqs
                ):
                    q = k + 1
                    for b in capture_counts(lo, hi):
                        nt = b * q
                        if 0 < nt <= self.max_num_tokens:
                            nt_set.add(nt)
            token_counts = sorted(nt_set)
            # Warmup once on persistent buffers (slots filled with PAD_SLOT_ID so KV stays clean).
            self._context_slot_mappings.fill_(-1)
            for nt in token_counts:
                slots = [
                    self._context_slot_mappings[gidx][:nt]
                    for gidx in self._layer_group_idx
                ]
                # Warmup before capture
                self.model.precompute_and_store_context_kv(
                    self.hidden_states[:nt], self.context_positions[:nt], slots
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.model.precompute_and_store_context_kv(
                        self.hidden_states[:nt], self.context_positions[:nt], slots
                    )
                self._context_kv_graphs[nt] = graph

        log.info(
            "Inkling2 DSpark: captured %d draft metadata refresh graphs and %d context KV graphs.",
            len(self._draft_metadata_graphs),
            len(self._context_kv_graphs),
        )

    def _refresh_draft_graph_metadata(self, num_reqs: int, num_tokens: int) -> bool:
        graph = self._draft_metadata_graphs.get((num_reqs, num_tokens))
        if graph is None:
            return False
        graph.replay()
        return True

    def _prepare_proposal_runtime(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        self._cur_num_reqs = input_batch.num_reqs
        self._cur_qsl = input_batch.query_start_loc
        self._cur_num_rejected = num_rejected

    def _precompute_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        nt = hidden_states.shape[0]
        graph = (
            self._context_kv_graphs.get(nt)
            if slots is not None
            and hidden_states.data_ptr() == self.hidden_states.data_ptr()
            and positions.data_ptr() == self.context_positions.data_ptr()
            else None
        )
        if graph is not None:
            graph.replay()
        else:
            self.model.precompute_and_store_context_kv(hidden_states, positions, slots)
        # Populate `f_rnn` from each request's last accepted target hidden state
        # before FULL draft CUDA graph replay reads `self.f_rnn[:num_reqs_padded]`.
        if (
            self._cur_num_reqs > 0
            and self._cur_qsl is not None
            and self._cur_num_rejected is not None
        ):
            nr = self._cur_num_reqs
            ends = self._cur_qsl[1 : nr + 1] - self._cur_num_rejected[:nr]
            rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
            self.f_rnn[:nr].copy_(self.model.compute_f_rnn(rows))

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._debug_proposal_stage("dspark draft forward begin")
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._debug_proposal_stage("dspark markov head begin")
        n_full = self.draft_block
        n_spec = (
            self._draft_k_for_bucket(num_reqs)
            if self._spec_k_cfg is not None
            else n_full
        )
        if n_spec == n_full:
            num_sample = num_reqs * n_full
            hidden = last_hidden_states[self.sample_indices[:num_sample]]
            idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_full)
            sample_pos = (self.sample_pos[:num_sample] - 1).view(num_reqs, n_full)
            sample_col = self.sample_col[:num_sample].view(num_reqs, n_full)
        else:
            sidx = self.sample_indices[: num_reqs * n_full].view(num_reqs, n_full)[:, :n_spec].reshape(-1)
            hidden = last_hidden_states[sidx]
            idx_map = self.sample_idx_mapping[: num_reqs * n_full].view(num_reqs, n_full)[:, :n_spec]
            sample_pos = (self.sample_pos[: num_reqs * n_full] - 1).view(num_reqs, n_full)[:, :n_spec]
            sample_col = self.sample_col[: num_reqs * n_full].view(num_reqs, n_full)[:, :n_spec]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        state = hidden.new_zeros((num_reqs, self.model.config.markov_rank))
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n_spec):
            state, bias = self.model.markov_step(state, prev, context[i])
            if self.draft_logits is None:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            else:
                logits = self.model.markov_logits(base_logits[:, i], bias)
                prev = gumbel_sample(
                    logits,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i],
                    apply_temperature=True,
                    is_drafting=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=sample_col[:, i],
                    use_fp64=self.use_fp64_gumbel,
                )
            self.draft_tokens[:num_reqs, i] = prev
        self._debug_proposal_stage("dspark markov head end")
