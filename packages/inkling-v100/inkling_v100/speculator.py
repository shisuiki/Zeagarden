# SPDX-License-Identifier: Apache-2.0
"""1Cat-vLLM V2 speculator subclass driving Inkling2's native DSpark 7-token drafter."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.utils import AttentionGroup

from .ops.dspark_markov import shard_argmax
from .ops.dspark_verify import greedy_verify_tokens, shard_rejection_sample
from .ops.fp8_dense_cuda import fp8_qpn8_linear
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
        # Shard-local Markov sampling + shard-local rejection verification (zero vocab all-gather!).
        self._shard_markov = os.environ.get("INKLING_DSPARK_SHARD_MARKOV", "1") != "0"
        self._shard_verify = os.environ.get("INKLING_DSPARK_SHARD_VERIFY", "1") != "0"
        self._zero_temperature = torch.zeros_like(self.temperature)
        self._dense_local: torch.Tensor | None = None
        self._row_gathered = np.ones(self.max_num_reqs, dtype=np.bool_)
        self._sampler: Any = None
        self._shard_verify_logged: bool = False
        self._shard_greedy_logged: bool = False

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Match target LM head FP32 precision so probabilistic rejection compares on equal footing.
        return torch.float32, 0.0

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
        if self._shard_markov and self.speculative_config.draft_sample_method == "probabilistic":
            shard = int(model.vocab_shard_size)
            self._dense_local = torch.zeros(
                (self.max_num_reqs * self.draft_block, shard),
                dtype=torch.float32,
                device=self.device,
            )
            log.info(
                "Inkling2 DSpark: shard-local probabilistic Markov + verify enabled "
                "(local shard=%d, %.1f MiB/GPU).",
                shard,
                self._dense_local.numel() * 4 / (1024**2),
            )
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
        out = super().propose(input_batch, *args, **kwargs)
        if (
            self._shard_markov
            and self._dense_local is not None
            and not kwargs.get("dummy_run")
            and not kwargs.get("is_profile")
            and input_batch.num_reqs > 0
        ):
            self._row_gathered[input_batch.idx_mapping_np] = False
        return out

    # ------------------------------------------------- Shard-local target verification
    def _all_greedy(self, input_batch: InputBatch) -> bool:
        sampler = self._sampler
        idx = input_batch.idx_mapping_np
        if sampler is None or sampler.compute_nans or idx.size == 0:
            return False
        states = sampler.sampling_states
        return bool(
            np.all(states.temperature.np[idx] == 0.0)
            and states.max_num_logprobs(idx) == NO_LOGPROBS
            and sampler.logprob_token_ids_state.max_num_token_ids(idx) == 0
            and not np.any(sampler.logit_bias_state.use_logit_bias[idx])
            and not np.any(sampler.penalties_state.use_penalty[idx])
            and not np.any(sampler.bad_words_state.num_bad_words.np[idx])
        )

    def _local_verify_ok(self, input_batch: InputBatch) -> bool:
        sampler = self._sampler
        idx = input_batch.idx_mapping_np
        if sampler is None or sampler.compute_nans or idx.size == 0:
            return False
        states = sampler.sampling_states
        return bool(
            np.all(states.top_k.np[idx] == states.vocab_size)
            and np.all(states.top_p.np[idx] == 1.0)
            and np.all(states.min_p.np[idx] == 0.0)
            and states.max_num_logprobs(idx) == NO_LOGPROBS
            and sampler.logprob_token_ids_state.max_num_token_ids(idx) == 0
            and not np.any(sampler.logit_bias_state.use_logit_bias[idx])
            and not np.any(sampler.penalties_state.use_penalty[idx])
            and not np.any(sampler.bad_words_state.num_bad_words.np[idx])
        )

    def _samples(self, input_batch: InputBatch) -> bool:
        if self._sampler is None:
            return True
        temps = self._sampler.sampling_states.temperature.np[input_batch.idx_mapping_np]
        return bool(np.any(temps != 0.0))

    @staticmethod
    def _target_head_info(model: nn.Module):
        lm = model.get_language_model() if hasattr(model, "get_language_model") else model
        processor = getattr(lm, "logits_processor", None)
        lm_head = getattr(lm, "lm_head", None)
        if (
            processor is None
            or lm_head is None
            or processor.scale != 1.0
            or processor.logits_as_input
            or getattr(lm_head, "bias", None) is not None
        ):
            return None
        fp8_ready = bool(getattr(lm_head, "_fp8_ready", False))
        shard = int(lm_head._fp8_n_out) if fp8_ready else int(lm_head.weight.shape[0])
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
        id_base = rank * shard
        num_valid = min(max(int(processor.org_vocab_size) - id_base, 0), shard)
        return lm_head, processor.soft_cap, num_valid, id_base, rank, world

    @staticmethod
    def _target_local_logits(lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(lm_head, "_fp8_ready", False):
            return fp8_qpn8_linear(
                hidden_states,
                lm_head._fp8_codes,
                lm_head._fp8_s16,
                lm_head._fp8_s32,
                lm_head._fp8_n_out,
                out_dtype=torch.float32,
                is_lm_head=True,
            )
        return torch.mm(hidden_states, lm_head.weight.t(), out_dtype=torch.float32)

    def _gather_dense_for(self, input_batch: InputBatch) -> None:
        assert self._dense_local is not None and self.draft_logits is not None
        per_req = input_batch.num_draft_tokens_per_req
        if per_req is None:
            return
        n_spec = self.draft_block
        rows = [
            int(state) * n_spec + i
            for state, n in zip(input_batch.idx_mapping_np.tolist(), per_req.tolist())
            for i in range(n)
        ]
        if not rows:
            return
        index = torch.tensor(rows, dtype=torch.int64, device=self.device)
        gathered = self.model.gather_dense_logits(self._dense_local[index])
        vocab = self.draft_logits.shape[-1]
        self.draft_logits.view(-1, vocab)[index] = gathered[:, :vocab]
        self._row_gathered[input_batch.idx_mapping_np] = True

    def try_sample_target(
        self,
        model: nn.Module,
        rejection_sampler: Any,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any,
        *,
        allow_graph: bool = True,
    ) -> SamplerOutput | None:
        del allow_graph
        if rejection_sampler is not None and self._sampler is None:
            self._sampler = rejection_sampler.sampler

        output: SamplerOutput | None = None
        if (
            self._shard_verify
            and grammar_output is None
            and rejection_sampler is not None
            and getattr(rejection_sampler, "synthetic_conditional_rates", None) is None
        ):
            head = self._target_head_info(model)
            if head is not None:
                lm_head, softcap, num_valid, id_base, rank, world = head
                if self._all_greedy(input_batch):
                    local_logits = self._target_local_logits(lm_head, hidden_states)
                    target_ids = shard_argmax(
                        local_logits,
                        softcap,
                        num_valid,
                        id_base,
                        rank,
                        world,
                        tensor_model_parallel_all_reduce,
                    )
                    if not self._shard_greedy_logged:
                        self._shard_greedy_logged = True
                        log.info(
                            "Inkling2 DSpark: shard-local greedy target verification active."
                        )
                    if input_batch.num_draft_tokens > 0:
                        sampled, num_sampled = greedy_verify_tokens(
                            target_ids,
                            input_batch.input_ids[input_batch.logits_indices],
                            input_batch.cu_num_logits,
                            self.draft_block,
                        )
                    else:
                        sampled = target_ids.view(-1, 1)
                        num_sampled = input_batch.seq_lens.new_ones(input_batch.num_reqs)
                    output = SamplerOutput(
                        sampled_token_ids=sampled,
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                    )
                elif self._local_verify_ok(input_batch) and (
                    input_batch.num_draft_tokens == 0
                    or (self._shard_markov and self._dense_local is not None)
                ):
                    local_logits = self._target_local_logits(lm_head, hidden_states)
                    states = self._sampler.sampling_states
                    sampled, num_sampled = shard_rejection_sample(
                        local_logits,
                        self._dense_local if input_batch.num_draft_tokens > 0 else None,
                        input_batch.input_ids,
                        input_batch.logits_indices,
                        input_batch.cu_num_logits,
                        input_batch.positions[input_batch.logits_indices],
                        input_batch.idx_mapping,
                        input_batch.expanded_idx_mapping,
                        input_batch.expanded_local_pos,
                        states.temperature.gpu,
                        states.seeds.gpu,
                        self.draft_block,
                        softcap,
                        num_valid,
                        id_base,
                        rank,
                        world,
                        self.use_fp64_gumbel,
                        tensor_model_parallel_all_reduce,
                    )
                    if not self._shard_verify_logged:
                        self._shard_verify_logged = True
                        log.info(
                            "Inkling2 DSpark: shard-local probabilistic rejection sampling active."
                        )
                    output = SamplerOutput(
                        sampled_token_ids=sampled,
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                    )

        if (
            output is None
            and self._shard_markov
            and self._dense_local is not None
            and self.draft_logits is not None
            and input_batch.num_draft_tokens > 0
            and not self._row_gathered[input_batch.idx_mapping_np].all()
            and self._samples(input_batch)
        ):
            self._gather_dense_for(input_batch)
        return output

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
        if self._shard_markov:
            probabilistic = self.draft_logits is not None and self._dense_local is not None
            if n_spec == n_full:
                hidden = last_hidden_states[self.sample_indices[: num_reqs * n_full]]
            else:
                sidx = (
                    self.sample_indices[: num_reqs * n_full]
                    .view(num_reqs, n_full)[:, :n_spec]
                    .reshape(-1)
                )
                hidden = last_hidden_states[sidx]
            self.model.markov_sample(
                hidden,
                self.f_rnn[:num_reqs],
                self.draft_tokens,
                self.input_buffers.input_ids,
                self.num_query_per_req,
                self.sample_idx_mapping,
                self.sample_pos,
                self.temperature if probabilistic else self._zero_temperature,
                self.seeds,
                self._dense_local if probabilistic else None,
                n_full,
                self.use_fp64_gumbel,
            )
            self._debug_proposal_stage("dspark markov head end")
            return

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
