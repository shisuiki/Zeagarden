# SPDX-License-Identifier: Apache-2.0
"""Volta (sm_70) TP4 runtime & speculator for Inkling-2-Mini DFly2-KV drafter.

Zero draft KV cache, zero draft conv state, zero draft prefill:
  - Stage A: MTPFusion (anchor normalized embedding + target final normed hidden) + mask_embedding
  - Stage B: 3 parallel backbone layers (block_size=8) sharing target epilogue KV caches:
      Layer 0 (local SWA-1024)  -> target L20
      Layer 1 (global + tau)    -> target L23 (Exact FP16/FP8 KV OR zero-retrain GLA-g4 r=256!)
      Layer 2 (local SWA-1024)  -> target L22
  - Stage C: Sequential AttentiveCorrection (slots 0..K_active, K_active <= 7) reading ONLY
      target L22 SWA-1024 KV (O(1) bandwidth at any context length) + causal block_sink-gated
      intra-block read -> shared target lm_head (softcap=30.0).
  - Shard-local greedy (T=0) AND shard-local probabilistic (T>0) drafting & rejection sampling
    inside FULL CUDA graphs (zero 201K-vocab all-gathers!).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from .ops.attn_rel_gla_cuda import gla_pack_q_cuda
from .ops.dfly2_ops_cuda import (
    empty_f32,
    load_dfly2_ops_ext,
    plan_dfly2_splits,
    preallocate_dfly2_workspace,
)
from .ops.dspark_markov import shard_argmax, shard_sample_slot
from .ops.dspark_sm70 import silu_and_mul
from .ops.dspark_verify import greedy_verify_tokens, shard_rejection_sample
from .ops.fp8_dense_cuda import (
    fp8_qpn8_linear,
    load_fp8_dense_ext,
    preallocate_fp8_dequant_ws,
    prepare_qpn8_weight,
)
from .spec_k import capture_counts, k_for, k_ranges, spec_k_config

log = init_logger("vllm.inkling_v100.dfly2")

TARGET_KV_LAYERS = (20, 23, 22)
LOCAL_LAYER_IDS = (0, 2)
BLK = 8
MAX_GAMMA = 7
H_DIM = 2304
I_FULL = 8192
I_LOCAL = 2048
HC_I_LOCAL = 576
HQ_L = 6
HKV_L = 2
HD = 128
D_REL = 16
E_REL = 1024
SWA = 1024
EPS = 1e-6
SCALE = HD**-0.5  # 0.08838834764831845

_SEEN: set = set()


class _DFly2Linear(nn.Module):
    """Per-channel FP8 QPN8 (default) or FP16 linear layer with CUDA-graph-safe `forward_into`."""

    def __init__(
        self,
        weight_cpu: torch.Tensor,
        device: torch.device,
        use_fp8: bool,
    ) -> None:
        super().__init__()
        self.out_features = int(weight_cpu.shape[0])
        self.in_features = int(weight_cpu.shape[1])
        self.use_fp8 = bool(use_fp8 and (self.in_features % 32 == 0))
        if self.use_fp8:
            wf = weight_cpu.to(device=device, dtype=torch.float32)
            sc = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
            w8 = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            codes, s16, s32, _n_pad, n_out = prepare_qpn8_weight(w8, sc, pad_align=64)
            self.register_buffer("codes", codes, persistent=False)
            self.register_buffer("s16", s16, persistent=False)
            self.register_buffer("s32", s32, persistent=False)
            self._n_out = n_out
            self.register_buffer("weight_t", None, persistent=False)
        else:
            self.register_buffer("codes", None, persistent=False)
            self.register_buffer("s16", None, persistent=False)
            self.register_buffer("s32", None, persistent=False)
            self._n_out = self.out_features
            self.register_buffer(
                "weight_t",
                weight_cpu.to(device=device, dtype=torch.float16).t().contiguous(),
                persistent=False,
            )

    def forward_into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return out
        if self.use_fp8:
            return fp8_qpn8_linear(
                x, self.codes, self.s16, self.s32, self._n_out, out=out
            )
        torch.mm(x, self.weight_t, out=out)
        return out


class _DFly2BackboneLayer(nn.Module):
    def __init__(
        self,
        idx: int,
        is_local: bool,
        target_kv_layer: int,
        qrgkv_proj: _DFly2Linear,
        sink_w_t: torch.Tensor,
        wo_ud: _DFly2Linear,
        gate_up_proj: _DFly2Linear,
        down_proj: _DFly2Linear,
        input_ln: torch.Tensor,
        post_ln: torch.Tensor,
        q_scale: torch.Tensor,
        r_scale: torch.Tensor,
        rel_proj: torch.Tensor,
        intra_rel_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.idx = idx
        self.is_local = is_local
        self.target_kv_layer = target_kv_layer
        self.qrgkv_proj = qrgkv_proj
        self.wo_ud = wo_ud
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.register_buffer("sink_w_t", sink_w_t, persistent=False)
        self.register_buffer("input_ln", input_ln, persistent=False)
        self.register_buffer("post_ln", post_ln, persistent=False)
        self.register_buffer("q_scale", q_scale, persistent=False)
        self.register_buffer("r_scale", r_scale, persistent=False)
        self.register_buffer("rel_proj", rel_proj, persistent=False)
        self.register_buffer("intra_rel_bias", intra_rel_bias, persistent=False)


class Inkling2DFly2DraftModel(nn.Module):
    """TP4 Volta sm_70 Inkling-2-Mini DFly2-KV draft model (0 B draft KV, 0 B conv state)."""

    def __init__(
        self,
        ckpt: str,
        device: torch.device,
        tp_rank: int,
        tp_size: int,
        use_fp8: bool = True,
    ) -> None:
        super().__init__()
        assert tp_size == 4, f"Inkling2DFly2DraftModel requires TP4, got tp_size={tp_size}"
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.use_fp8 = use_fp8

        p = Path(ckpt)
        st_path = (p / "model.safetensors") if p.is_dir() else p
        cfg_path = (p / "config.json") if p.is_dir() else (p.parent / "config.json")
        assert st_path.is_file(), f"Inkling DFly2 checkpoint not found: {st_path}"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        self.target_kv_layers = tuple(int(x) for x in cfg.get("target_kv_layers", TARGET_KV_LAYERS))
        self.local_layer_ids = set(int(x) for x in cfg.get("local_layer_ids", LOCAL_LAYER_IDS))
        self.block_size = int(cfg.get("block_size", BLK))
        self.soft_cap = float(cfg.get("final_logit_softcapping", 30.0))
        assert self.block_size == BLK, f"Expected block_size={BLK}, got {self.block_size}"

        if use_fp8:
            load_fp8_dense_ext()
            preallocate_fp8_dequant_ws(device, 4608 * 2560)

        from safetensors.torch import load_file

        sd = load_file(str(st_path), device="cpu")
        r = tp_rank

        def _buf(name: str, sl: Any = None) -> torch.Tensor:
            t = sd[name] if sl is None else sd[name][sl]
            return t.to(device=device, dtype=torch.float16).contiguous()

        # Stage A: MTPFusion + mask_embedding
        self.register_buffer("mask_embedding", _buf("mask_embedding"), persistent=False)
        self.register_buffer("mtp_enorm", _buf("mtp_fusion.enorm.weight"), persistent=False)
        self.register_buffer("mtp_hnorm", _buf("mtp_fusion.hnorm.weight"), persistent=False)
        self.register_buffer("mtp_gate", _buf("mtp_fusion.gate"), persistent=False)
        eh_w = sd["mtp_fusion.eh_proj.weight"][:, r * 1152 : (r + 1) * 1152].contiguous()
        self.eh_proj = _DFly2Linear(eh_w, device, use_fp8)

        # Stage B: 3 backbone layers
        layers = []
        for li in range(3):
            pfx = f"layers.{li}"
            wq = sd[f"{pfx}.self_attn.wq_du.weight"][r * 768 : (r + 1) * 768]
            wr = sd[f"{pfx}.self_attn.wr_du.weight"][r * 96 : (r + 1) * 96]
            wg = sd[f"{pfx}.self_attn.gate_attn.weight"][r * 768 : (r + 1) * 768]
            wk = sd[f"{pfx}.self_attn.wk_dv.weight"][r * 256 : (r + 1) * 256]
            wv = sd[f"{pfx}.self_attn.wv_dv.weight"][r * 256 : (r + 1) * 256]
            qrgkv_w = torch.cat([wq, wr, wg, wk, wv], dim=0).contiguous()  # [2144, 2304]
            qrgkv_proj = _DFly2Linear(qrgkv_w, device, use_fp8)

            sink_w_t = (
                sd[f"{pfx}.self_attn.attn_sink_proj.weight"][r * HQ_L : (r + 1) * HQ_L]
                .to(device=device, dtype=torch.float16)
                .t()
                .contiguous()
            )
            wo_w = sd[f"{pfx}.self_attn.wo_ud.weight"][:, r * 768 : (r + 1) * 768].contiguous()
            wo_ud = _DFly2Linear(wo_w, device, use_fp8)

            mg = sd[f"{pfx}.mlp.gate_proj.weight"][r * I_LOCAL : (r + 1) * I_LOCAL]
            mu = sd[f"{pfx}.mlp.up_proj.weight"][r * I_LOCAL : (r + 1) * I_LOCAL]
            gate_up_proj = _DFly2Linear(torch.cat([mg, mu], dim=0).contiguous(), device, use_fp8)
            md = sd[f"{pfx}.mlp.down_proj.weight"][:, r * I_LOCAL : (r + 1) * I_LOCAL].contiguous()
            down_proj = _DFly2Linear(md, device, use_fp8)

            layers.append(
                _DFly2BackboneLayer(
                    idx=li,
                    is_local=(li in self.local_layer_ids),
                    target_kv_layer=self.target_kv_layers[li],
                    qrgkv_proj=qrgkv_proj,
                    sink_w_t=sink_w_t,
                    wo_ud=wo_ud,
                    gate_up_proj=gate_up_proj,
                    down_proj=down_proj,
                    input_ln=_buf(f"{pfx}.input_layernorm.weight"),
                    post_ln=_buf(f"{pfx}.post_attention_layernorm.weight"),
                    q_scale=_buf(f"{pfx}.self_attn.q_scale", slice(r * HQ_L, (r + 1) * HQ_L)),
                    r_scale=_buf(f"{pfx}.self_attn.r_scale", slice(r * HQ_L, (r + 1) * HQ_L)),
                    rel_proj=_buf(f"{pfx}.self_attn.rel_logits_proj"),
                    intra_rel_bias=_buf(
                        f"{pfx}.self_attn.intra_rel_bias", slice(r * HQ_L, (r + 1) * HQ_L)
                    ),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.register_buffer("final_norm", _buf("norm.weight"), persistent=False)

        # Stage C: HiddenStatesCorrection + I2MAttentiveCorrection (reads target L22 SWA-1024)
        self.register_buffer("hc_hidden_norm", _buf("hidden_correction.hidden_norm.weight"), persistent=False)
        self.register_buffer("hc_embed_norm", _buf("hidden_correction.embed_norm.weight"), persistent=False)
        self.register_buffer(
            "hc_q_scale",
            _buf("hidden_correction.attention.q_scale", slice(r * HQ_L, (r + 1) * HQ_L)),
            persistent=False,
        )
        self.register_buffer(
            "hc_r_scale",
            _buf("hidden_correction.attention.r_scale", slice(r * HQ_L, (r + 1) * HQ_L)),
            persistent=False,
        )
        self.register_buffer(
            "hc_rel_proj",
            _buf("hidden_correction.attention.rel_logits_proj"),
            persistent=False,
        )
        self.register_buffer(
            "hc_block_gate_w_t",
            sd["hidden_correction.attention.block_gate.weight"][r * HQ_L : (r + 1) * HQ_L]
            .to(device=device, dtype=torch.float16)
            .t()
            .contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "hc_block_gate_bias",
            _buf("hidden_correction.attention.block_gate_bias", slice(r * HQ_L, (r + 1) * HQ_L)),
            persistent=False,
        )
        self.register_buffer(
            "hc_block_rel_bias",
            _buf("hidden_correction.attention.block_rel_bias", slice(r * HQ_L, (r + 1) * HQ_L)),
            persistent=False,
        )
        self.register_buffer(
            "hc_block_sink",
            _buf("hidden_correction.attention.block_sink", slice(r * HQ_L, (r + 1) * HQ_L)),
            persistent=False,
        )

        hc_g = sd["hidden_correction.gate_proj.weight"][r * HC_I_LOCAL : (r + 1) * HC_I_LOCAL]
        hc_u = sd["hidden_correction.up_proj.weight"][r * HC_I_LOCAL : (r + 1) * HC_I_LOCAL]
        hc_q = sd["hidden_correction.attention.q_in.weight"][r * 768 : (r + 1) * 768]
        hc_r = sd["hidden_correction.attention.r_in.weight"][r * 96 : (r + 1) * 96]
        hc_k = sd["hidden_correction.attention.k_blk_proj.weight"][r * 256 : (r + 1) * 256]
        hc_v = sd["hidden_correction.attention.v_blk_proj.weight"][r * 256 : (r + 1) * 256]
        # [576 + 576 + 768 + 96 + 256 + 256 = 2528, 4608]
        self.hc_mega_proj = _DFly2Linear(
            torch.cat([hc_g, hc_u, hc_q, hc_r, hc_k, hc_v], dim=0).contiguous(), device, use_fp8
        )
        self.hc_kv0_proj = _DFly2Linear(
            torch.cat([hc_k, hc_v], dim=0).contiguous(), device, use_fp8
        )
        self.hc_down_proj = _DFly2Linear(
            sd["hidden_correction.down_proj.weight"][:, r * HC_I_LOCAL : (r + 1) * HC_I_LOCAL].contiguous(),
            device,
            use_fp8,
        )
        self.hc_o_proj = _DFly2Linear(
            sd["hidden_correction.attention.o_proj.weight"][:, r * 768 : (r + 1) * 768].contiguous(),
            device,
            use_fp8,
        )

        del sd
        if r == 0:
            nbytes = sum(b.numel() * b.element_size() for b in self.buffers() if b is not None)
            log.info(
                "inkling_v100.dfly2: loaded %s (dtype=%s, kv_layers=%s, %.1f MB/GPU TP4, 0 B draft KV)",
                st_path.name,
                "fp8_qpn8" if use_fp8 else "fp16",
                self.target_kv_layers,
                nbytes / 1e6,
            )


def _resolve_ckpt_path(vllm_config: VllmConfig) -> str:
    env_p = os.environ.get("INKLING_DFLY2_CKPT", "").strip()
    if env_p:
        return env_p
    sc = vllm_config.speculative_config
    assert sc is not None and sc.draft_model_config is not None
    return str(sc.draft_model_config.model)


class Inkling2DFly2Speculator(BaseSpeculator):
    """1Cat-vLLM V2 BaseSpeculator driving Inkling2DFly2DraftModel with shared target KV."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        sc = vllm_config.speculative_config
        assert sc is not None
        self.speculative_config = sc
        self.method = sc.method
        self.num_speculative_steps = int(sc.num_speculative_tokens)
        assert 1 <= self.num_speculative_steps <= MAX_GAMMA, (
            f"Inkling2 DFly2-KV supports K in 1..{MAX_GAMMA}, got {self.num_speculative_steps}"
        )

        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = bool(getattr(vllm_config.model_config, "use_fp64_gumbel", False))
        self.supports_mm_inputs = False
        self.draft_step_listener = None
        self.query_cudagraph_manager = None

        self._spec_k_cfg = spec_k_config()
        self._last_k = self.num_speculative_steps
        self._ckpt_path = _resolve_ckpt_path(vllm_config)
        self._use_fp8 = os.environ.get("INKLING_DFLY2_DTYPE", "fp16").strip().lower() == "fp8"
        self._use_graphs = os.environ.get("INKLING_DFLY2_GRAPH", "1").strip() != "0"
        self._shard_verify = os.environ.get("INKLING_DFLY2_SHARD_VERIFY", "1").strip() != "0"

        self.model: Inkling2DFly2DraftModel | None = None
        self.target_model: nn.Module | None = None
        self.target_lm: nn.Module | None = None
        self.target_attn_mods: list[Any] = []  # [L20, L23, L22] Inkling2Attention
        self.target_kv_group_ids: list[int] = []
        self.block_tables = None
        self.kv_cache_config = None
        self.model_state = None
        self.req_states = None
        self.tp_rank = 0
        self.tp_size = 4
        self._sampler: Any = None
        self._head_info: tuple[Any, float | None, int, int, int, int] | None = None

        # Sampling & verification buffers
        self.draft_tokens = torch.zeros(
            self.max_num_reqs, self.num_speculative_steps, dtype=torch.int64, device=device
        )
        self.temperature = torch.zeros(self.max_num_reqs, dtype=torch.float32, device=device)
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self._probabilistic = sc.draft_sample_method == "probabilistic"
        self._dense_local: torch.Tensor | None = None
        self.draft_logits: torch.Tensor | None = None
        self._row_gathered = np.ones(self.max_num_reqs, dtype=np.bool_)
        self._shard_verify_logged = False
        self._shard_greedy_logged = False

        self._bufs: dict[str, torch.Tensor] = {}
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_pool = None
        self._capture_counts_by_k: dict[int, list[int]] = {}
        self._cudagraph_mode = CUDAGraphMode.NONE
        self._capture_context = None
        self._kv_ready = False

    def set_req_states(self, req_states) -> None:
        self.req_states = req_states

    def next_num_draft_tokens(self) -> int:
        return self._last_k

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

    def load_model(self, target_model: nn.Module) -> None:
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.target_model = target_model
        self.target_lm = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )

        # Keep only a zero-cost reference sentinel `(48,)` when launched under `method="dflash"`
        # so MRV2 unpacks `(hidden_states, [hidden_states])` with 0 extra tensor copies.
        if self.method == "dflash" and hasattr(self.target_lm, "set_aux_hidden_state_layers"):
            self.target_lm.set_aux_hidden_state_layers((48,))

        load_dfly2_ops_ext()
        self.model = Inkling2DFly2DraftModel(
            self._ckpt_path, self.device, self.tp_rank, self.tp_size, use_fp8=self._use_fp8
        )

        decoder_layers = self.target_lm.model.layers
        self.target_attn_mods = [decoder_layers[li].attn for li in self.model.target_kv_layers]
        self._head_info = self._target_head_info(target_model)
        assert self._head_info is not None, "Inkling2DFly2Speculator requires standard ParallelLMHead"

        if self._probabilistic:
            lm_head = self._head_info[0]
            shard = (
                int(lm_head._fp8_n_out)
                if getattr(lm_head, "_fp8_ready", False)
                else int(lm_head.weight.shape[0])
            )
            self._dense_local = torch.zeros(
                (self.max_num_reqs * self.num_speculative_steps, shard),
                dtype=torch.float32,
                device=self.device,
            )
            if self.tp_rank == 0:
                log.info(
                    "inkling_v100.dfly2: shard-local probabilistic drafting + verify active "
                    "(local shard=%d, %.1f MiB/GPU).",
                    shard,
                    self._dense_local.numel() * 4 / (1024**2),
                )

        self._alloc_scratch_buffers()

    def _alloc_scratch_buffers(self) -> None:
        dev = self.device
        B = self.max_num_reqs
        N8 = B * BLK
        has_gla = any(bool(getattr(m, "use_gla", False)) for m in self.target_attn_mods)

        bufs = {
            # Input extraction
            "anchor_tok": torch.zeros(B, dtype=torch.int32, device=dev),
            "ctx_len": torch.zeros(B, dtype=torch.int32, device=dev),
            "anchor_pos": torch.zeros(B, dtype=torch.int32, device=dev),
            "prev_hidden": torch.zeros(B, H_DIM, dtype=torch.float16, device=dev),
            "req_slots": torch.zeros(B, dtype=torch.int32, device=dev),
            # Stage A
            "eh_in_local": torch.empty(B, H_DIM // 2, dtype=torch.float16, device=dev),
            "eh_out": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            # Stage B
            "res_f32": torch.empty(N8, H_DIM, dtype=torch.float32, device=dev),
            "x_norm": torch.empty(N8, H_DIM, dtype=torch.float16, device=dev),
            "qrgkv_b": torch.empty(N8, 2144, dtype=torch.float16, device=dev),
            "sinks_b": torch.empty(N8, HQ_L, dtype=torch.float16, device=dev),
            "q_b": torch.empty(N8, HQ_L, HD, dtype=torch.float16, device=dev),
            "r_b": torch.empty(N8, HQ_L, D_REL, dtype=torch.float16, device=dev),
            "rel_all_b": torch.empty(N8, HQ_L, E_REL, dtype=torch.float16, device=dev),
            "k_blk_b": torch.empty(B, BLK, HKV_L, HD, dtype=torch.float16, device=dev),
            "v_blk_b": torch.empty(B, BLK, HKV_L, HD, dtype=torch.float16, device=dev),
            "ao_ctx_b": torch.empty(N8, HQ_L, HD, dtype=torch.float16, device=dev),
            "lse_ctx_b": torch.empty(N8, HQ_L, dtype=torch.float32, device=dev),
            "attn_out_b": torch.empty(N8, HQ_L * HD, dtype=torch.float16, device=dev),
            "o_part_b": torch.empty(N8, H_DIM, dtype=torch.float16, device=dev),
            "gu_b": torch.empty(N8, 2 * I_LOCAL, dtype=torch.float16, device=dev),
            "mlp_part_b": torch.empty(N8, H_DIM, dtype=torch.float16, device=dev),
            # Stage C
            "inp_c": torch.empty(B, 2 * H_DIM, dtype=torch.float16, device=dev),
            "kv0_c": torch.empty(B, 512, dtype=torch.float16, device=dev),
            "mega_c": torch.empty(B, 2528, dtype=torch.float16, device=dev),
            "gate_raw_c": torch.empty(B, HQ_L, dtype=torch.float16, device=dev),
            "mid_c": torch.empty(B, HC_I_LOCAL, dtype=torch.float16, device=dev),
            "q_c": torch.empty(B, HQ_L, HD, dtype=torch.float16, device=dev),
            "r_c": torch.empty(B, HQ_L, D_REL, dtype=torch.float16, device=dev),
            "rel_all_c": torch.empty(B, HQ_L, E_REL, dtype=torch.float16, device=dev),
            "blk_k_c": torch.empty(B, BLK, HKV_L, HD, dtype=torch.float16, device=dev),
            "blk_v_c": torch.empty(B, BLK, HKV_L, HD, dtype=torch.float16, device=dev),
            "o_ctx_c": torch.empty(B, HQ_L, HD, dtype=torch.float16, device=dev),
            "attn_out_c": torch.empty(B, HQ_L * HD, dtype=torch.float16, device=dev),
            "delta_mlp_c": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            "delta_attn_c": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            "h_corr_c": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
        }
        if has_gla:
            bufs["q_pack_b"] = torch.empty(2 * N8 * 3 * HD, dtype=torch.float16, device=dev)
            bufs["q_lat_b"] = torch.empty(2 * N8 * 3 * 256, dtype=torch.float16, device=dev)
            bufs["o_lat_b"] = torch.empty(2 * N8 * 3 * 256, dtype=torch.float16, device=dev)
            bufs["v_out_b"] = torch.empty(2 * N8 * 3 * HD, dtype=torch.float16, device=dev)

        self._bufs = bufs
        preallocate_dfly2_workspace(slots=max(32768, N8 * HQ_L * 10), head_dim=256 if has_gla else HD, device=dev)
        if self.tp_rank == 0:
            mb = sum(t.numel() * t.element_size() for t in bufs.values()) / 1e6
            log.info(
                "inkling_v100.dfly2: preallocated %.1f MB scratch buffers (max_num_reqs=%d, gla=%s)",
                mb,
                B,
                has_gla,
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode, capture_context=None) -> None:
        self._cudagraph_mode = cudagraph_mode
        self._capture_context = capture_context

    def set_attn(self, model_state, kv_cache_config, block_tables, *extra) -> None:
        del extra
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        gids = []
        for m in self.target_attn_mods:
            lname = m.prefix
            found = -1
            for gi, grp in enumerate(kv_cache_config.kv_cache_groups):
                if lname in grp.layer_names:
                    found = gi
                    break
            assert found >= 0, f"inkling_v100.dfly2: target layer {lname} not found in kv_cache_groups"
            gids.append(found)
        self.target_kv_group_ids = gids
        if self.tp_rank == 0:
            log.info(
                "inkling_v100.dfly2: bound shared target KV layers %s -> kv groups %s (0 B draft KV!)",
                [m.prefix for m in self.target_attn_mods],
                gids,
            )

    def _ensure_ready(self) -> None:
        if self._kv_ready:
            return
        has_gla = any(bool(getattr(m, "use_gla", False)) for m in self.target_attn_mods)
        preallocate_dfly2_workspace(
            slots=max(32768, self.max_num_reqs * BLK * HQ_L * 10),
            head_dim=256 if has_gla else HD,
            device=self.device,
        )
        if self._use_fp8:
            preallocate_fp8_dequant_ws(self.device, 4608 * 2560)
        self._kv_ready = True

    def _run_target_kv_attn(
        self,
        li_idx: int,
        q: torch.Tensor,
        rel_all: torch.Tensor,
        B: int,
        Q: int,
        slot_fixed: int,
        out: torch.Tensor,
        lse_out: torch.Tensor | None,
        skip_attn: bool = False,
    ) -> None:
        ext = load_dfly2_ops_ext()
        m = self.target_attn_mods[li_idx]
        is_local = bool(m.is_local)
        is_gla = bool(getattr(m, "use_gla", False))

        if skip_attn or m.kv_cache.numel() == 0:
            out.zero_()
            if lse_out is not None:
                lse_out.fill_(float("-inf"))
            return

        kv_cache = m.kv_cache
        key_cache, value_cache = m._split_kv_cache(kv_cache)
        if key_cache.element_size() == 1:
            key_cache = key_cache.view(torch.uint8)
            value_cache = value_cache.view(torch.uint8)
        bt = self.block_tables.input_block_tables[self.target_kv_group_ids[li_idx]][:B]
        ctx_len = self._bufs["ctx_len"][:B]
        anchor_pos = self._bufs["anchor_pos"][:B]

        splits = plan_dfly2_splits(is_local, B, Q, is_gla=is_gla)
        if splits > 1:
            o_part, ml_part = preallocate_dfly2_workspace(
                slots=B * Q * HQ_L * splits, head_dim=256 if is_gla else HD, device=self.device
            )
        else:
            o_part = ml_part = empty_f32(self.device)

        if is_gla:
            assert Q == BLK and lse_out is not None
            _, u_k_t, u_v = m._gla_projs[0]
            N8 = B * BLK
            q_pack = self._bufs["q_pack_b"][: 2 * N8 * 3 * HD].view(2, N8, 3, HD)
            q_lat = self._bufs["q_lat_b"][: 2 * N8 * 3 * 256].view(2, N8, 3, 256)
            o_lat = self._bufs["o_lat_b"][: 2 * N8 * 3 * 256].view(2, N8, 3, 256)
            v_out = self._bufs["v_out_b"][: 2 * N8 * 3 * HD].view(2, N8, 3, HD)
            gla_pack_q_cuda(q, q_pack)
            torch.bmm(q_pack.view(2, 3 * N8, HD), u_k_t, out=q_lat.view(2, 3 * N8, 256))
            ext.dfly2_gla_ctx_attn_forward(
                q_lat,
                key_cache,
                value_cache,
                rel_all,
                o_lat,
                lse_out,
                bt,
                ctx_len,
                anchor_pos,
                o_part,
                ml_part,
                SCALE,
                splits,
            )
            torch.bmm(o_lat.view(2, 3 * N8, 256), u_v, out=v_out.view(2, 3 * N8, HD))
            out.view(N8, 2, 3, HD).copy_(v_out.permute(1, 0, 2, 3))
            return

        ext.dfly2_ctx_attn_forward(
            q,
            key_cache,
            value_cache,
            rel_all,
            out,
            lse_out,
            bt,
            ctx_len,
            anchor_pos,
            o_part,
            ml_part,
            Q,
            slot_fixed,
            SCALE,
            SWA if is_local else 0,
            splits,
        )

    def _sample_slot(self, h_corr: torch.Tensor, B: int, step: int) -> torch.Tensor:
        assert self._head_info is not None
        lm_head, softcap, num_valid, id_base, rank, world = self._head_info
        local_logits = self._target_local_logits(lm_head, h_corr)
        if self._probabilistic:
            return shard_sample_slot(
                local_logits,
                self._bufs["req_slots"][:B],
                self._bufs["anchor_pos"][:B],
                self.temperature,
                self.seeds,
                self._dense_local,
                step,
                self.num_speculative_steps,
                softcap,
                num_valid,
                id_base,
                rank,
                world,
                self.use_fp64_gumbel,
                tensor_model_parallel_all_reduce,
            )
        return shard_argmax(
            local_logits,
            softcap,
            num_valid,
            id_base,
            rank,
            world,
            tensor_model_parallel_all_reduce,
        )

    def _run_dfly2_core(self, B: int, K_active: int, skip_attn: bool = False) -> None:
        ext = load_dfly2_ops_ext()
        m = self.model
        assert m is not None and self.target_lm is not None
        bufs = self._bufs
        N8 = B * BLK

        anchor_tok = bufs["anchor_tok"][:B]
        anchor_pos = bufs["anchor_pos"][:B]
        prev_hidden = bufs["prev_hidden"][:B]

        # ---- Stage A: normalized anchor embed + MTPFusion + block init + L0 input_layernorm ----
        e0 = self.target_lm.model.embed_input_ids(anchor_tok).to(torch.float16).contiguous()
        eh_in = bufs["eh_in_local"][:B]
        eh_out = bufs["eh_out"][:B]
        ext.stage_a_prep(e0, prev_hidden, m.mtp_enorm, m.mtp_hnorm, eh_in, self.tp_rank, EPS)
        m.eh_proj.forward_into(eh_in, eh_out)
        eh_red = tensor_model_parallel_all_reduce(eh_out)

        res_f32 = bufs["res_f32"][:N8]
        x_norm = bufs["x_norm"][:N8]
        ext.stage_a_fuse_norm(
            e0, eh_red, m.mtp_gate, m.mask_embedding, m.layers[0].input_ln, res_f32, x_norm, EPS
        )

        # ---- Stage B: 3 parallel backbone layers reading target [L20, L23, L22] KV ----
        qrgkv_b = bufs["qrgkv_b"][:N8]
        sinks_b = bufs["sinks_b"][:N8]
        q_b = bufs["q_b"][:N8]
        r_b = bufs["r_b"][:N8]
        rel_all_b = bufs["rel_all_b"][:N8]
        k_blk_b = bufs["k_blk_b"][:B]
        v_blk_b = bufs["v_blk_b"][:B]
        ao_ctx_b = bufs["ao_ctx_b"][:N8]
        lse_ctx_b = bufs["lse_ctx_b"][:N8]
        attn_out_b = bufs["attn_out_b"][:N8]
        o_part_b = bufs["o_part_b"][:N8]
        gu_b = bufs["gu_b"][:N8]
        mlp_part_b = bufs["mlp_part_b"][:N8]

        for li in range(3):
            layer = m.layers[li]
            layer.qrgkv_proj.forward_into(x_norm, qrgkv_b)
            torch.mm(x_norm, layer.sink_w_t, out=sinks_b)
            ext.stage_b_qkrv_prep(
                qrgkv_b,
                anchor_pos,
                layer.q_scale,
                layer.r_scale,
                q_b,
                r_b,
                k_blk_b,
                v_blk_b,
                not layer.is_local,
                EPS,
            )
            torch.mm(r_b.view(N8 * HQ_L, D_REL), layer.rel_proj, out=rel_all_b.view(N8 * HQ_L, E_REL))
            self._run_target_kv_attn(
                li, q_b, rel_all_b, B, BLK, 0, ao_ctx_b, lse_ctx_b, skip_attn=skip_attn
            )
            ext.stage_b_merge_gate_sink(
                ao_ctx_b,
                lse_ctx_b,
                q_b,
                k_blk_b,
                v_blk_b,
                rel_all_b,
                layer.intra_rel_bias,
                qrgkv_b,
                sinks_b,
                attn_out_b,
                SCALE,
            )
            layer.wo_ud.forward_into(attn_out_b, o_part_b)
            o_red = tensor_model_parallel_all_reduce(o_part_b)
            ext.rmsnorm_res(o_red, res_f32, layer.post_ln, x_norm, EPS)

            layer.gate_up_proj.forward_into(x_norm, gu_b)
            mid_b = silu_and_mul(gu_b)
            layer.down_proj.forward_into(mid_b, mlp_part_b)
            mlp_red = tensor_model_parallel_all_reduce(mlp_part_b)
            next_ln = m.layers[li + 1].input_ln if li < 2 else m.final_norm
            ext.rmsnorm_res(mlp_red, res_f32, next_ln, x_norm, EPS)

        # x_norm is now h_bb [B*8, 2304] post-final-norm!
        h_bb = x_norm

        # ---- Stage C: Slot 0 KV-only init (reads nothing from L22, skips MLP/Q/O/lm_head!) ----
        inp_c = bufs["inp_c"][:B]
        kv0_c = bufs["kv0_c"][:B]
        blk_k_c = bufs["blk_k_c"][:B]
        blk_v_c = bufs["blk_v_c"][:B]
        ext.stage_c_prep(h_bb, e0, m.hc_hidden_norm, m.hc_embed_norm, inp_c, 0, EPS)
        m.hc_kv0_proj.forward_into(inp_c, kv0_c)
        ext.stage_c_slot0_kv(kv0_c, blk_k_c, blk_v_c, EPS)

        # ---- Stage C: Sequential AttentiveCorrection slots 1..K_active (L22 SWA-1024!) ----
        mega_c = bufs["mega_c"][:B]
        gate_raw_c = bufs["gate_raw_c"][:B]
        mid_c = bufs["mid_c"][:B]
        q_c = bufs["q_c"][:B]
        r_c = bufs["r_c"][:B]
        rel_all_c = bufs["rel_all_c"][:B]
        o_ctx_c = bufs["o_ctx_c"][:B]
        attn_out_c = bufs["attn_out_c"][:B]
        delta_mlp_c = bufs["delta_mlp_c"][:B]
        delta_attn_c = bufs["delta_attn_c"][:B]
        h_corr_c = bufs["h_corr_c"][:B]

        e_prev = e0
        for slot in range(1, K_active + 1):
            ext.stage_c_prep(h_bb, e_prev, m.hc_hidden_norm, m.hc_embed_norm, inp_c, slot, EPS)
            m.hc_mega_proj.forward_into(inp_c, mega_c)
            torch.mm(inp_c, m.hc_block_gate_w_t, out=gate_raw_c)
            ext.stage_c_qkrv_silu(
                mega_c,
                m.hc_q_scale,
                m.hc_r_scale,
                mid_c,
                q_c,
                r_c,
                blk_k_c,
                blk_v_c,
                slot,
                EPS,
            )
            torch.mm(r_c.view(B * HQ_L, D_REL), m.hc_rel_proj, out=rel_all_c.view(B * HQ_L, E_REL))
            self._run_target_kv_attn(
                2, q_c, rel_all_c, B, 1, slot, o_ctx_c, None, skip_attn=skip_attn
            )
            ext.stage_c_combine(
                o_ctx_c,
                q_c,
                blk_k_c,
                blk_v_c,
                gate_raw_c,
                m.hc_block_gate_bias,
                m.hc_block_rel_bias,
                m.hc_block_sink,
                attn_out_c,
                slot,
                SCALE,
            )
            m.hc_down_proj.forward_into(mid_c, delta_mlp_c)
            m.hc_o_proj.forward_into(attn_out_c, delta_attn_c)
            delta_mlp_c.add_(delta_attn_c)
            delta_red = tensor_model_parallel_all_reduce(delta_mlp_c)
            ext.add_slot_residual(h_bb, delta_red, h_corr_c, slot)

            tok = self._sample_slot(h_corr_c, B, slot - 1)
            self.draft_tokens[:B, slot - 1].copy_(tok)
            if slot < K_active:
                e_prev = self.target_lm.model.embed_input_ids(tok).to(torch.float16).contiguous()

    def _k_for(self, num_reqs: int) -> int:
        if self._spec_k_cfg is None:
            return self.num_speculative_steps
        return k_for(num_reqs, self.num_speculative_steps, self._spec_k_cfg)

    def capture(self, captured_attn_states=None) -> None:
        del captured_attn_states
        self._ensure_ready()
        if not self._use_graphs or self._cudagraph_mode == CUDAGraphMode.NONE:
            return

        if self._spec_k_cfg is not None:
            rng = k_ranges(self._spec_k_cfg, self.num_speculative_steps, self.max_num_reqs)
        else:
            rng = [(self.num_speculative_steps, 1, self.max_num_reqs)]

        pairs: list[tuple[int, int]] = []
        by_k: dict[int, list[int]] = {}
        for k, lo, hi in rng:
            if k <= 0:
                continue
            counts = capture_counts(lo, hi)
            by_k[k] = counts
            for b in counts:
                pairs.append((b, k))
        self._capture_counts_by_k = by_k

        # Safe dummy inputs (ctx_len=0, req_slots=-1 so _dense_local is untouched!)
        self._bufs["anchor_tok"].zero_()
        self._bufs["ctx_len"].zero_()
        self._bufs["anchor_pos"].zero_()
        self._bufs["prev_hidden"].zero_()
        self._bufs["req_slots"].fill_(-1)

        from vllm.distributed.parallel_state import graph_capture

        ordered = sorted(pairs, key=lambda bk: (bk[0], bk[1]), reverse=True)
        for b, k in ordered:
            self._run_dfly2_core(b, k, skip_attn=False)
        torch.cuda.synchronize(self.device)

        self._graph_pool = torch.cuda.graph_pool_handle()
        with graph_capture(device=self.device, context=self._capture_context) as ctx:
            for b, k in ordered:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._graph_pool, stream=ctx.stream):
                    self._run_dfly2_core(b, k, skip_attn=False)
                self._graphs[(b, k)] = g
        torch.cuda.synchronize(self.device)
        if self.tp_rank == 0:
            log.info(
                "inkling_v100.dfly2: captured %d FULL CUDA graphs (%s, probabilistic=%s)",
                len(self._graphs),
                ", ".join(f"K={k}:{cs}" for k, cs in sorted(by_k.items())),
                self._probabilistic,
            )

    def _pick_graph_b(self, num_reqs: int, k: int) -> int | None:
        if not self._graphs:
            return None
        for b in self._capture_counts_by_k.get(k, ()):
            if b >= num_reqs and (b, k) in self._graphs:
                return b
        return None

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata,
        slot_mappings,
        last_hidden_states: torch.Tensor,
        aux_hidden_states,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
        output_copy_event=None,
        sampled_token_ids_cpu=None,
        num_sampled_tokens_cpu=None,
        all_token_ids_cpu=None,
        active_spec_steps: int | None = None,
    ) -> torch.Tensor:
        del attn_metadata, slot_mappings, aux_hidden_states, num_tokens_across_dp
        del mm_inputs, output_copy_event, sampled_token_ids_cpu, num_sampled_tokens_cpu, all_token_ids_cpu

        num_reqs = int(input_batch.num_reqs)
        k = self._k_for(num_reqs)
        if active_spec_steps is not None:
            k = min(k, int(active_spec_steps))
        self._last_k = k

        if num_reqs <= 0 or k <= 0:
            return self.draft_tokens[:num_reqs]

        if is_profile or skip_attn_for_dummy_run:
            self._ensure_ready()
            b_run = min(num_reqs, self.max_num_reqs)
            self._bufs["anchor_tok"][:b_run].zero_()
            self._bufs["ctx_len"][:b_run].zero_()
            self._bufs["anchor_pos"][:b_run].zero_()
            self._bufs["prev_hidden"][:b_run].zero_()
            self._bufs["req_slots"][:b_run].fill_(-1)
            self._run_dfly2_core(b_run, k, skip_attn=True)
            return self.draft_tokens[:num_reqs]

        self._ensure_ready()
        if self._probabilistic:
            self.temperature.copy_(temperature)
            self.seeds.copy_(seeds)

        b_graph = None if dummy_run else self._pick_graph_b(num_reqs, k)
        b_pad = b_graph if b_graph is not None else num_reqs

        ext = load_dfly2_ops_ext()
        ext.extract_inputs(
            input_batch.query_start_loc,
            input_batch.seq_lens,
            input_batch.idx_mapping,
            input_batch.positions,
            last_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self._bufs["anchor_tok"],
            self._bufs["ctx_len"],
            self._bufs["anchor_pos"],
            self._bufs["prev_hidden"],
            self._bufs["req_slots"],
            num_reqs,
            b_pad,
            self.max_model_len,
        )

        if b_graph is not None:
            self._graphs[(b_graph, k)].replay()
            if ("graph", b_graph, k) not in _SEEN and self.tp_rank == 0 and len(_SEEN) < 32:
                _SEEN.add(("graph", b_graph, k))
                log.info("inkling_v100.dfly2: FULL graph replay B=%d (pad=%d) K=%d", num_reqs, b_graph, k)
        else:
            self._run_dfly2_core(b_pad, k, skip_attn=False)
            if ("eager", b_pad, k) not in _SEEN and self.tp_rank == 0 and len(_SEEN) < 32:
                _SEEN.add(("eager", b_pad, k))
                log.info("inkling_v100.dfly2: eager run B=%d K=%d (dummy=%s)", b_pad, k, dummy_run)

        if self._probabilistic and self._dense_local is not None and not dummy_run:
            self._row_gathered[input_batch.idx_mapping_np] = False
        return self.draft_tokens[:num_reqs]

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

    def _gather_dense_for(self, input_batch: InputBatch) -> None:
        if self._dense_local is None:
            return
        per_req = input_batch.num_draft_tokens_per_req
        if per_req is None:
            return
        n_spec = self.num_speculative_steps
        rows = [
            int(state) * n_spec + i
            for state, n in zip(input_batch.idx_mapping_np.tolist(), per_req.tolist())
            for i in range(n)
        ]
        if not rows:
            return
        if self.draft_logits is None:
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                n_spec,
                self.vocab_size,
                dtype=torch.float32,
                device=self.device,
            )
        index = torch.tensor(rows, dtype=torch.int64, device=self.device)
        gathered = tensor_model_parallel_all_gather(self._dense_local[index], dim=-1)
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
            head = self._head_info or self._target_head_info(model)
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
                        log.info("inkling_v100.dfly2: shard-local greedy target verification active.")
                    if input_batch.num_draft_tokens > 0:
                        sampled, num_sampled = greedy_verify_tokens(
                            target_ids,
                            input_batch.input_ids[input_batch.logits_indices],
                            input_batch.cu_num_logits,
                            self.num_speculative_steps,
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
                    or (self._probabilistic and self._dense_local is not None)
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
                        self.num_speculative_steps,
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
                            "inkling_v100.dfly2: shard-local probabilistic rejection sampling active."
                        )
                    output = SamplerOutput(
                        sampled_token_ids=sampled,
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                    )

        if (
            output is None
            and self._probabilistic
            and self._dense_local is not None
            and input_batch.num_draft_tokens > 0
            and not self._row_gathered[input_batch.idx_mapping_np].all()
            and self._samples(input_batch)
        ):
            self._gather_dense_for(input_batch)
        return output
