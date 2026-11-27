"""q27fuse.dfly2 - Q27 DFly2-KV block-diffusion + AttentiveCorrection drafter runtime (TP4, Volta sm_70).

Activated only when Q27_DFLY2_CKPT=<dir or model.safetensors> is set alongside
--speculative-config '{"method":"mtp","num_speculative_tokens":5}' (zero impact otherwise).

Key properties:
  - 0 bytes of draft context KV: backbone layers {0, 1, 2} read target post-RoPE KV directly from
    Q27 layers [47, 55, 63] and AttentiveCorrection reads target layer 63 KV (with 0 writes to target KV).
  - Exact zero-retrain compatibility with Q27_GLA_CKPT (GLA-g4 r=192 / r=256 compressed target KV):
    projects draft q_nope through target's _gla_k_up_sc_cm, runs attn_gla_mq (+ optional LSE), and
    fuses o_lat @ _gla_v_up_cm inside the ephemeral block merge / correction combine kernels.
  - Supports quantized checkpoints produced by tools/quantize_dfly2.py:
      * nvfp4_fp8 (~267 MB/GPU TP4): NVFP4 backbone MLPs (mlp_fp4.cu) + FP8 QPN8 projections (fp8_lin.cu)
      * fp8_all   (~355 MB/GPU TP4): FP8 QPN8 across all 2D projections
      * raw bf16  (fallback, quantized on-the-fly to FP8 QPN8 at load time)
  - Full CUDA-graph capture of the entire Stage A + Stage B + Stage C + lm_head draft step per
    (B_pad, K_active) bucket, with adaptive Q27_SPEC_K slot truncation and draftscreen 1-bit top-1 head.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

log = logging.getLogger("q27fuse.dfly2")

TARGET_KV_LAYERS = (47, 55, 63)
BLK = 6
H_DIM = 5120
I_LOCAL = 4352
NQ_LOCAL = 6
HD = 256
EPS = 1e-6
SCALE = 1.0 / (HD ** 0.5)   # 0.0625

_EXT = None
_FP8_CFG_IDX: dict[str, int] = {}
_SEEN: set = set()


def ckpt_path() -> str:
    return os.environ.get("Q27_DFLY2_CKPT", "").strip()


def enabled() -> bool:
    return bool(ckpt_path())


def dfly2_ext():
    global _EXT
    if _EXT is None:
        from .ops import _load
        _EXT = _load("q27fuse_dfly2", "dfly2_ops.cu", "_dfly2")
    return _EXT


def _split_k_for_qpn8(k: int) -> int:
    # Choose largest power/divisor in (32, 16, 12, 8, 4) such that k % (16 * sk) == 0
    for sk in (32, 16, 12, 8, 4, 2, 1):
        if k % (16 * sk) == 0:
            return sk
    return 1


def _fp8_cfg_for(m: int, k: int) -> int | None:
    if m <= 8 or m > 64:
        return None
    from . import ops, smallrows
    if not _FP8_CFG_IDX:
        _FP8_CFG_IDX.update({c.split(":", 2)[2]: i for i, c in enumerate(ops.fp8_ext().fp8_configs())})
    for (lo, hi), by_k, default in smallrows.FP8_CFG:
        if lo <= m <= hi:
            return _FP8_CFG_IDX[by_k.get(k, default)]
    return None


class QPN8Linear(nn.Module):
    """Per-channel FP8 E4M3FN linear stored in 1Cat's QPN8 layout (single copy, zero overhead)."""

    def __init__(self, codes: torch.Tensor, s16: torch.Tensor, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.split_k = _split_k_for_qpn8(self.in_features)
        # QPN8 codes [K, N] uint8, scales [1, N] fp16 (= scale * 256)
        self.register_buffer("codes", codes.reshape(self.in_features, self.out_features).contiguous(), persistent=False)
        self.register_buffer("s16", s16.reshape(1, self.out_features).contiguous(), persistent=False)

    @classmethod
    def from_fp8(cls, w8: torch.Tensor, scale: torch.Tensor, device: torch.device) -> "QPN8Linear":
        import vllm._C  # noqa: F401
        out_f, in_f = int(w8.shape[0]), int(w8.shape[1])
        w8_g = w8.to(device=device, dtype=torch.float8_e4m3fn).contiguous()
        sc_g = scale.to(device=device, dtype=torch.float32).reshape(out_f, 1).contiguous()
        codes, s16 = torch.ops._C.fp8_qpn8_prepare_sm70(w8_g, sc_g)[:2]
        return cls(codes, s16, in_f, out_f)

    @classmethod
    def from_dense(cls, w: torch.Tensor, device: torch.device) -> "QPN8Linear":
        wf = w.to(device=device, dtype=torch.float32)
        sc = (wf.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
        w8 = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return cls.from_fp8(w8, sc, device)

    def forward_into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        m = int(x.shape[0])
        if m == 0:
            return out
        from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
        from . import ops

        if m <= 8:
            dispatch_channel_rows(out, x, self.codes, self.s16, self.split_k, 1, False, False, None)
            return out
        if m <= 64:
            cfg = _fp8_cfg_for(m, self.in_features)
            if cfg is not None:
                ops.fp8_ext().fp8_linear(out, x, self.codes.view(-1), self.s16.view(-1), cfg)
                return out
            dispatch_channel_rows(out, x, self.codes, self.s16, self.split_k, 1, False, False, None)
            return out
        # M > 64 (e.g. Stage B at B > 10): 64-row chunks via fp8_lin.cu (zero alloc, CUDA-graph safe)
        cfg64 = _fp8_cfg_for(64, self.in_features)
        ext = ops.fp8_ext()
        cflat = self.codes.view(-1)
        sflat = self.s16.view(-1)
        for r0 in range(0, m, 64):
            r1 = min(m, r0 + 64)
            cm = r1 - r0
            xc = x[r0:r1]
            oc = out[r0:r1]
            if cm <= 8:
                dispatch_channel_rows(oc, xc, self.codes, self.s16, self.split_k, 1, False, False, None)
            else:
                cfg = cfg64 if cm == 64 else _fp8_cfg_for(cm, self.in_features)
                ext.fp8_linear(oc, xc, cflat, sflat, cfg)
        return out


class NVFP4MLP(nn.Module):
    """TP4 NVFP4 backbone MLP backed directly by q27fuse/csrc/mlp_fp4.cu."""

    def __init__(self, wgu: torch.Tensor, wd: torch.Tensor, epi_gu: float, epi_d: float, ir: int):
        super().__init__()
        self.register_buffer("wgu", wgu, persistent=False)
        self.register_buffer("wd", wd, persistent=False)
        self.epi_gu = float(epi_gu)
        self.epi_d = float(epi_d)
        self.ir = int(ir)

    def forward(self, x: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        from . import ops
        return ops.mlp_ext().mlp(x, self.wgu, self.wd, self.epi_gu, self.epi_d, self.ir, ws)


class FP8MLP(nn.Module):
    """TP4 FP8-all fallback backbone MLP using QPN8Linear."""

    def __init__(self, gate_up: QPN8Linear, down: QPN8Linear):
        super().__init__()
        self.gate_up = gate_up
        self.down = down

    def forward(self, x: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        del ws
        m = int(x.shape[0])
        gu = torch.empty((m, self.gate_up.out_features), dtype=x.dtype, device=x.device)
        self.gate_up.forward_into(x, gu)
        half = self.gate_up.out_features // 2
        mid = (F.silu(gu[:, :half].float()) * gu[:, half:].float()).to(x.dtype).contiguous()
        out = torch.empty((m, self.down.out_features), dtype=x.dtype, device=x.device)
        self.down.forward_into(mid, out)
        return out


class DFly2BackboneLayer(nn.Module):
    def __init__(self, idx: int, target_kv_layer: int, qkv_proj: QPN8Linear, o_proj: QPN8Linear,
                 mlp: nn.Module, input_ln: torch.Tensor, post_ln: torch.Tensor,
                 q_norm: torch.Tensor, k_norm: torch.Tensor):
        super().__init__()
        self.idx = idx
        self.target_kv_layer = target_kv_layer
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.mlp = mlp
        self.register_buffer("input_ln", input_ln, persistent=False)
        self.register_buffer("post_ln", post_ln, persistent=False)
        self.register_buffer("q_norm", q_norm, persistent=False)
        self.register_buffer("k_norm", k_norm, persistent=False)


def _load_linear_shard(sd: dict[str, torch.Tensor], base: str, dim: int, start: int, end: int,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (weight_shard, scale_or_None) for an FP8 or dense 2D linear."""
    w = sd[f"{base}.weight"]
    sc_key = f"{base}.weight_scale"
    if dim == 0:
        ws = w[start:end, :].contiguous()
        sc = sd[sc_key][start:end].contiguous() if sc_key in sd else None
    else:
        ws = w[:, start:end].contiguous()
        sc = sd[sc_key].contiguous() if sc_key in sd else None
    return ws, sc


def _make_qpn8(ws: torch.Tensor, sc: torch.Tensor | None, device: torch.device) -> QPN8Linear:
    if sc is not None and ws.dtype == torch.float8_e4m3fn:
        return QPN8Linear.from_fp8(ws, sc, device)
    return QPN8Linear.from_dense(ws, device)


def _concat_qpn8(parts: list[tuple[torch.Tensor, torch.Tensor | None]], device: torch.device) -> QPN8Linear:
    if all(sc is not None and w.dtype == torch.float8_e4m3fn for w, sc in parts):
        w_cat = torch.cat([w for w, _ in parts], dim=0).contiguous()
        s_cat = torch.cat([sc for _, sc in parts], dim=0).contiguous()
        return QPN8Linear.from_fp8(w_cat, s_cat, device)
    # Mixed or dense fallback
    dense = []
    for w, sc in parts:
        if sc is not None and w.dtype == torch.float8_e4m3fn:
            dense.append(w.float() * sc.float()[:, None])
        else:
            dense.append(w.float())
    return QPN8Linear.from_dense(torch.cat(dense, dim=0), device)


class Q27DFly2DraftModel(nn.Module):
    """TP4 Volta sm_70 DFly2-KV draft model (zero own KV cache, zero own embed/lm_head)."""

    def __init__(self, ckpt: str, device: torch.device, tp_rank: int, tp_size: int):
        super().__init__()
        assert tp_size == 4, f"q27fuse.dfly2 requires TP4, got tp_size={tp_size}"
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        p = Path(ckpt)
        if p.is_dir():
            st_path = p / "model.safetensors"
            cfg_path = p / "config.json"
        else:
            st_path = p
            cfg_path = p.parent / "config.json"
        assert st_path.is_file(), f"Q27_DFLY2_CKPT file not found: {st_path}"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        self.target_kv_layers = tuple(cfg.get("target_kv_layers", TARGET_KV_LAYERS))
        self.gamma = int(cfg.get("gamma", 5))
        self.block_size = int(cfg.get("block_size", 6))
        assert self.block_size == BLK and self.gamma == 5

        from safetensors.torch import load_file
        from . import ops

        sd = load_file(str(st_path), device="cpu")
        r = tp_rank

        # Stage A: MTPFusion + mask_embedding
        self.register_buffer("mask_embedding", sd["mask_embedding"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("mtp_enorm", sd["mtp_fusion.enorm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("mtp_hnorm", sd["mtp_fusion.hnorm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("mtp_gate", sd["mtp_fusion.gate"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        eh_w, eh_s = _load_linear_shard(sd, "mtp_fusion.eh_proj", 1, r * 2560, (r + 1) * 2560, device)
        self.eh_proj = _make_qpn8(eh_w, eh_s, device)

        # Stage B: 3 backbone layers
        layers = []
        mlp_mode = "fp8"
        for li in range(3):
            q_part = _load_linear_shard(sd, f"layers.{li}.self_attn.q_proj", 0, r * 3072, (r + 1) * 3072, device)
            k_part = _load_linear_shard(sd, f"layers.{li}.self_attn.k_proj", 0, r * 256, (r + 1) * 256, device)
            v_part = _load_linear_shard(sd, f"layers.{li}.self_attn.v_proj", 0, r * 256, (r + 1) * 256, device)
            qkv_proj = _concat_qpn8([q_part, k_part, v_part], device)

            o_w, o_s = _load_linear_shard(sd, f"layers.{li}.self_attn.o_proj", 1, r * 1536, (r + 1) * 1536, device)
            o_proj = _make_qpn8(o_w, o_s, device)

            if f"layers.{li}.mlp.gate_proj.weight_packed" in sd:
                mlp_mode = "nvfp4"
                gp = sd[f"layers.{li}.mlp.gate_proj.weight_packed"][r * I_LOCAL:(r + 1) * I_LOCAL, :]
                gs = sd[f"layers.{li}.mlp.gate_proj.weight_scale"][r * I_LOCAL:(r + 1) * I_LOCAL, :]
                gg = float(sd[f"layers.{li}.mlp.gate_proj.weight_global_scale"].max().item())
                up = sd[f"layers.{li}.mlp.up_proj.weight_packed"][r * I_LOCAL:(r + 1) * I_LOCAL, :]
                us = sd[f"layers.{li}.mlp.up_proj.weight_scale"][r * I_LOCAL:(r + 1) * I_LOCAL, :]
                ug = float(sd[f"layers.{li}.mlp.up_proj.weight_global_scale"].max().item())
                gu_codes = torch.cat([gp, up], dim=0).to(device=device, dtype=torch.uint8).contiguous()
                gu_scales = torch.cat([gs, us], dim=0).to(device=device).view(torch.uint8).contiguous()
                wgu = ops.mlp_ext().prepack(gu_codes, gu_scales, True)
                epi_gu = 16384.0 / max(gg, ug)

                dp = sd[f"layers.{li}.mlp.down_proj.weight_packed"][:, r * (I_LOCAL // 2):(r + 1) * (I_LOCAL // 2)]
                ds = sd[f"layers.{li}.mlp.down_proj.weight_scale"][:, r * (I_LOCAL // 16):(r + 1) * (I_LOCAL // 16)]
                dg = float(sd[f"layers.{li}.mlp.down_proj.weight_global_scale"].max().item())
                wd = ops.mlp_ext().prepack(dp.to(device=device, dtype=torch.uint8).contiguous(),
                                           ds.to(device=device).view(torch.uint8).contiguous(), False)
                epi_d = 16384.0 / dg
                mlp_mod: nn.Module = NVFP4MLP(wgu, wd, epi_gu, epi_d, I_LOCAL)
            else:
                g_part = _load_linear_shard(sd, f"layers.{li}.mlp.gate_proj", 0, r * I_LOCAL, (r + 1) * I_LOCAL, device)
                u_part = _load_linear_shard(sd, f"layers.{li}.mlp.up_proj", 0, r * I_LOCAL, (r + 1) * I_LOCAL, device)
                gu_lin = _concat_qpn8([g_part, u_part], device)
                d_w, d_s = _load_linear_shard(sd, f"layers.{li}.mlp.down_proj", 1, r * I_LOCAL, (r + 1) * I_LOCAL, device)
                d_lin = _make_qpn8(d_w, d_s, device)
                mlp_mod = FP8MLP(gu_lin, d_lin)

            layers.append(DFly2BackboneLayer(
                idx=li,
                target_kv_layer=self.target_kv_layers[li],
                qkv_proj=qkv_proj,
                o_proj=o_proj,
                mlp=mlp_mod,
                input_ln=sd[f"layers.{li}.input_layernorm.weight"].to(device=device, dtype=torch.float16).contiguous(),
                post_ln=sd[f"layers.{li}.post_attention_layernorm.weight"].to(device=device, dtype=torch.float16).contiguous(),
                q_norm=sd[f"layers.{li}.self_attn.q_norm.weight"].to(device=device, dtype=torch.float16).contiguous(),
                k_norm=sd[f"layers.{li}.self_attn.k_norm.weight"].to(device=device, dtype=torch.float16).contiguous(),
            ))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("final_norm", sd["norm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)

        # Stage C: HiddenCorrection + AttentiveCorrection
        self.register_buffer("hc_hidden_norm", sd["hidden_correction.hidden_norm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("hc_embed_norm", sd["hidden_correction.embed_norm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("hc_q_norm", sd["hidden_correction.attention.q_norm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("hc_k_blk_norm", sd["hidden_correction.attention.k_blk_norm.weight"].to(device=device, dtype=torch.float16).contiguous(), persistent=False)

        self.register_buffer("hc_block_gate_w", sd["hidden_correction.attention.block_gate.weight"][r * 6:(r + 1) * 6, :].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("hc_block_gate_bias", sd["hidden_correction.attention.block_gate_bias"][r * 6:(r + 1) * 6, :].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("hc_block_rel_bias", sd["hidden_correction.attention.block_rel_bias"][r * 6:(r + 1) * 6, :].to(device=device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("hc_block_sink", sd["hidden_correction.attention.block_sink"][r * 6:(r + 1) * 6].to(device=device, dtype=torch.float16).contiguous(), persistent=False)

        hc_gate_p = _load_linear_shard(sd, "hidden_correction.gate_proj", 0, r * 1280, (r + 1) * 1280, device)
        hc_up_p = _load_linear_shard(sd, "hidden_correction.up_proj", 0, r * 1280, (r + 1) * 1280, device)
        hc_q_p = _load_linear_shard(sd, "hidden_correction.attention.q_in", 0, r * 1536, (r + 1) * 1536, device)
        hc_k_p = _load_linear_shard(sd, "hidden_correction.attention.k_blk_proj", 0, r * 256, (r + 1) * 256, device)
        hc_v_p = _load_linear_shard(sd, "hidden_correction.attention.v_blk_proj", 0, r * 256, (r + 1) * 256, device)

        self.hc_mega_proj = _concat_qpn8([hc_gate_p, hc_up_p, hc_q_p, hc_k_p, hc_v_p], device)
        self.hc_kv0_proj = _concat_qpn8([hc_k_p, hc_v_p], device)

        hc_dn_w, hc_dn_s = _load_linear_shard(sd, "hidden_correction.down_proj", 1, r * 1280, (r + 1) * 1280, device)
        self.hc_down_proj = _make_qpn8(hc_dn_w, hc_dn_s, device)
        hc_o_w, hc_o_s = _load_linear_shard(sd, "hidden_correction.attention.o_proj", 1, r * 1536, (r + 1) * 1536, device)
        self.hc_o_proj = _make_qpn8(hc_o_w, hc_o_s, device)

        del sd
        if r == 0:
            nbytes = sum(b.numel() * b.element_size() for b in self.buffers())
            log.warning("q27fuse.dfly2: loaded %s (mlp=%s, kv_layers=%s, %.1f MB/GPU TP4, 0 B draft KV)",
                        st_path.name, mlp_mode, self.target_kv_layers, nbytes / 1e6)


def _resolve_target_lm(target_model: nn.Module) -> nn.Module:
    lm = getattr(target_model, "language_model", None)
    return lm if lm is not None else target_model


class Q27DFly2Speculator(BaseSpeculator):
    """vLLM v1.5.4 BaseSpeculator driving Q27DFly2DraftModel with shared target KV and FULL CUDA graphs."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        sc = vllm_config.speculative_config
        assert sc is not None
        self.speculative_config = sc
        self.method = sc.method
        self.num_speculative_steps = int(sc.num_speculative_tokens)
        assert 1 <= self.num_speculative_steps <= 5, f"DFly2 supports K in 1..5, got {self.num_speculative_steps}"

        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = bool(getattr(vllm_config.model_config, "use_fp64_gumbel", False))
        self.supports_mm_inputs = False
        self.draft_step_listener = None
        self.query_cudagraph_manager = None

        from . import spec_k
        self._spec_k_cfg = spec_k.config()
        self._last_k = self.num_speculative_steps

        self.model: Q27DFly2DraftModel | None = None
        self.target_model: nn.Module | None = None
        self.target_lm: nn.Module | None = None
        self.target_attn_mods: list[Any] = []   # [L47, L55, L63] Qwen3NextAttention
        self.target_kv_group_ids: list[int] = []
        self.block_tables = None
        self.kv_cache_config = None
        self.model_state = None
        self.req_states = None
        self.screen_head = None
        self.tp_group_name = ""
        self.tp_rank = 0
        self.tp_size = 4

        # Sampling state buffers
        self.draft_tokens = torch.zeros(self.max_num_reqs, self.num_speculative_steps, dtype=torch.int64, device=device)
        self.temperature = torch.zeros(self.max_num_reqs, dtype=torch.float32, device=device)
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_logits: torch.Tensor | None = None
        if sc.draft_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs, self.num_speculative_steps, self.vocab_size,
                dtype=vllm_config.model_config.head_dtype, device=device,
            )

        # Persistent GPU scratch buffers (allocated in load_model before KV profiling!)
        self._bufs: dict[str, torch.Tensor] = {}
        self._slot_idx_tensors: list[torch.Tensor] = []
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_pool = None
        self._capture_counts_by_k: dict[int, list[int]] = {}
        self._use_graphs = (os.environ.get("Q27_DFLY2_GRAPH", "1") or "1").strip() != "0"
        self._head_mode = (os.environ.get("Q27_DFLY2_HEAD", "auto") or "auto").strip().lower()
        self._kv_ready = False

    def set_req_states(self, req_states) -> None:
        self.req_states = req_states

    def next_num_draft_tokens(self) -> int:
        return self._last_k

    def load_model(self, target_model: nn.Module) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group
        from . import draftscreen

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group_name = get_tp_group().unique_name
        self.target_model = target_model
        self.target_lm = _resolve_target_lm(target_model)

        # Compile dfly2_ops.cu outside capture
        dfly2_ext()

        # Load quantized DFly2-KV weights
        self.model = Q27DFly2DraftModel(ckpt_path(), self.device, self.tp_rank, self.tp_size)

        # Resolve target [47, 55, 63] Qwen3NextAttention modules
        decoder_layers = self.target_lm.model.layers
        self.target_attn_mods = [decoder_layers[li].self_attn for li in self.model.target_kv_layers]

        # Build fast 1-bit draftscreen head on target's lm_head (40 MB, reuses unified FP8 HeadCopy!)
        if self._head_mode != "exact" and self.draft_logits is None and draftscreen.enabled():
            try:
                draftscreen.register_op()
                lm_head = self.target_lm.lm_head
                data = draftscreen.prepare(lm_head, self.vllm_config, draftscreen.RERANK)
                self.screen_head = draftscreen.ScreenHead(lm_head, None, data)
                if self.tp_rank == 0:
                    log.warning("q27fuse.dfly2: Stage C head = draftscreen (%s, maxr=%d)",
                                data["rerank"], self.screen_head.maxr)
            except Exception as e:  # noqa: BLE001
                log.warning("q27fuse.dfly2: draftscreen init failed (%r); using exact target head", e)
                self.screen_head = None

        # Allocate static GPU scratch buffers NOW (before KV cache sizing!)
        self._alloc_scratch_buffers()

    def _alloc_scratch_buffers(self) -> None:
        dev = self.device
        B = self.max_num_reqs
        N6 = B * BLK
        r_lat_max = 256
        for m in self.target_attn_mods:
            if getattr(m, "_gla_active", False):
                r_lat_max = max(r_lat_max, int(m._gla_rank))
        has_gla = any(getattr(m, "_gla_active", False) for m in self.target_attn_mods)

        bufs = {
            # Extracted inputs
            "anchor_tok": torch.zeros(B, dtype=torch.int32, device=dev),
            "ctx_len": torch.zeros(B, dtype=torch.int32, device=dev),
            "ctx_len_6": torch.zeros(N6, dtype=torch.int32, device=dev),
            "anchor_pos": torch.zeros(B, dtype=torch.int32, device=dev),
            "prev_hidden": torch.zeros(B, H_DIM, dtype=torch.float16, device=dev),
            "req_slots": torch.zeros(B, dtype=torch.int32, device=dev),
            # Stage A
            "eh_in_local": torch.empty(B, 2560, dtype=torch.float16, device=dev),
            "eh_out": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            # Stage B
            "h_block": torch.empty(N6, H_DIM, dtype=torch.float16, device=dev),
            "res_f32": torch.empty(N6, H_DIM, dtype=torch.float32, device=dev),
            "x_norm": torch.empty(N6, H_DIM, dtype=torch.float16, device=dev),
            "qkv_b": torch.empty(N6, 3584, dtype=torch.float16, device=dev),
            "q_b": torch.empty(N6, NQ_LOCAL, HD, dtype=torch.float16, device=dev),
            "k_blk_b": torch.empty(B, BLK, HD, dtype=torch.float16, device=dev),
            "v_blk_b": torch.empty(B, BLK, HD, dtype=torch.float16, device=dev),
            "o_ctx_b": torch.empty(N6, NQ_LOCAL, HD, dtype=torch.float16, device=dev),
            "lse_ctx_b": torch.empty(N6, NQ_LOCAL, dtype=torch.float32, device=dev),
            "attn_out_b": torch.empty(N6, NQ_LOCAL * HD, dtype=torch.float16, device=dev),
            "o_part_b": torch.empty(N6, H_DIM, dtype=torch.float16, device=dev),
            # Stage C
            "inp_c": torch.empty(B, 2 * H_DIM, dtype=torch.float16, device=dev),
            "kv0_c": torch.empty(B, 512, dtype=torch.float16, device=dev),
            "mega_c": torch.empty(B, 4608, dtype=torch.float16, device=dev),
            "q_c": torch.empty(B, NQ_LOCAL, HD, dtype=torch.float16, device=dev),
            "blk_k_c": torch.empty(B, BLK, HD, dtype=torch.float16, device=dev),
            "blk_v_c": torch.empty(B, BLK, HD, dtype=torch.float16, device=dev),
            "o_ctx_c": torch.empty(B, NQ_LOCAL, HD, dtype=torch.float16, device=dev),
            "mid_c": torch.empty(B, 1280, dtype=torch.float16, device=dev),
            "attn_out_c": torch.empty(B, 1536, dtype=torch.float16, device=dev),
            "delta_mlp": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            "delta_attn": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            "h_corr": torch.empty(B, H_DIM, dtype=torch.float16, device=dev),
            "gate_raw": torch.empty(B, NQ_LOCAL, dtype=torch.float16, device=dev),
            "top1_pairs": torch.empty(B, 2, dtype=torch.float32, device=dev),
            "slot_pos_tmp": torch.empty(B, dtype=torch.int32, device=dev),
        }
        if has_gla:
            bufs["q_cat_b"] = torch.empty(N6, NQ_LOCAL, r_lat_max + 64, dtype=torch.float16, device=dev)
            bufs["o_lat_b"] = torch.empty(N6, NQ_LOCAL, r_lat_max, dtype=torch.float16, device=dev)
            bufs["q_cat_c"] = torch.empty(B, NQ_LOCAL, r_lat_max + 64, dtype=torch.float16, device=dev)
            bufs["o_lat_c"] = torch.empty(B, NQ_LOCAL, r_lat_max, dtype=torch.float16, device=dev)
        self._bufs = bufs
        self._slot_idx_tensors = [torch.tensor(i, dtype=torch.int64, device=dev) for i in range(self.num_speculative_steps)]
        if self.tp_rank == 0:
            mb = sum(t.numel() * t.element_size() for t in bufs.values()) / 1e6
            log.warning("q27fuse.dfly2: preallocated %.1f MB scratch buffers (max_num_reqs=%d, gla=%s)",
                        mb, B, has_gla)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode, capture_context=None) -> None:
        self._cudagraph_mode = cudagraph_mode
        self._capture_context = capture_context

    def set_attn(self, model_state, kv_cache_config, block_tables, *extra) -> None:
        del extra
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        # Map each target KV layer name -> kv_cache_group index (for block_tables.input_block_tables[gid])
        gids = []
        for m in self.target_attn_mods:
            lname = m.attn.layer_name
            found = -1
            for gi, grp in enumerate(kv_cache_config.kv_cache_groups):
                if lname in grp.layer_names:
                    found = gi
                    break
            assert found >= 0, f"q27fuse.dfly2: target layer {lname} not found in kv_cache_groups"
            gids.append(found)
        self.target_kv_group_ids = gids
        if self.tp_rank == 0:
            log.warning("q27fuse.dfly2: bound shared target KV layers %s -> groups %s",
                        [m.attn.layer_name for m in self.target_attn_mods], gids)

    def _ensure_kv_ready(self) -> None:
        if self._kv_ready:
            return
        from . import attnv, gla
        attnv.workspace(self.device, 1)
        if any(getattr(m, "_gla_active", False) for m in self.target_attn_mods):
            gla._ensure_workspace(self.device, 1)
            for m in self.target_attn_mods:
                if getattr(m, "_gla_active", False):
                    gla._ensure_runtime_buffers(m, self.device, torch.float16)
        self._kv_ready = True

    def _run_target_kv_attn(
        self,
        layer_idx_in_list: int,
        q: torch.Tensor,
        B: int,
        Q: int,
        lens: torch.Tensor,
        out_exact: torch.Tensor,
        lse_out: torch.Tensor | None,
        anchor_pos: torch.Tensor,
        stride_blk: int,
        slot_fixed: int,
        skip_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read target layer's shared KV cache (Exact E4M3 or GLA-g4) without writing anything.

        Returns (o_tensor, w_v_up_cm_or_None).
        """
        m = self.target_attn_mods[layer_idx_in_list]
        is_gla = bool(getattr(m, "_gla_active", False))
        if skip_attn:
            if lse_out is not None:
                lse_out.fill_(float("-inf"))
            if is_gla:
                r_lat = int(m._gla_rank)
                o_lat = (self._bufs["o_lat_b"] if Q == BLK else self._bufs["o_lat_c"])[:B * Q, :, :r_lat]
                o_lat.zero_()
                return o_lat, m._gla_v_up_cm
            out_exact.zero_()
            return out_exact, None

        from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
        kv_cache = m.attn.kv_cache[0]
        k_cache, v_cache = _kv_layout.split_paged_kv_cache(kv_cache)
        bt = self.block_tables.input_block_tables[self.target_kv_group_ids[layer_idx_in_list]][:B]

        if is_gla:
            from . import gla
            r_lat = int(m._gla_rank)
            row_b = r_lat + 64
            q_cat_full = self._bufs["q_cat_b"] if Q == BLK else self._bufs["q_cat_c"]
            o_lat_full = self._bufs["o_lat_b"] if Q == BLK else self._bufs["o_lat_c"]
            q_cat = q_cat_full[:B * Q, :, :row_b].contiguous()
            o_lat = o_lat_full[:B * Q, :, :r_lat].contiguous()
            dfly2_ext().gla_q_absorb(q, anchor_pos, stride_blk, slot_fixed, m._gla_k_up_sc_cm, q_cat)
            po, pml, cnt = gla._ensure_workspace(self.device, 1)[:3]
            s_req = gla._pick_splits(B, Q, r_lat)
            sep = gla._gla_sep_cfg()
            gla.gla_ext().attn_gla_mq(
                q_cat, k_cache, v_cache, bt, lens, o_lat, po, pml, cnt,
                SCALE, float(m._gla_kv_scale), Q, s_req, 1, sep, lse_out,
            )
            return o_lat, m._gla_v_up_cm

        from . import attnv
        po, pml, cnt = attnv.workspace(self.device, 1)
        if Q == BLK:
            nw, S, mt, ng, pd, pf, rs, sep = (8, 64, 1, 1, 1, 0, 0, 1)
        else:
            nw, S, mt, ng, pd, pf, rs, sep = (4, 64 if B <= 15 else 32, 1, 2, 1, 0, 0, 1)
        attnv.mq_ext().attn_mq(
            q, k_cache, v_cache, bt, lens, out_exact, po, pml, cnt,
            SCALE, float(m.attn._k_scale_float), float(m.attn._v_scale_float),
            Q, nw, S, mt, ng, pd, pf, rs, sep, lse_out,
        )
        return out_exact, None

    def _top1_draft_token(self, h_corr: torch.Tensor, B: int, slot_idx: int) -> torch.Tensor:
        """Select next draft token from h_corr [B, 5120] using draftscreen, HeadCopy.top1, or gumbel."""
        if self.draft_logits is not None:
            from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
            logits = self.target_lm.compute_logits(h_corr)
            pos = self._bufs["slot_pos_tmp"][:B]
            pos.copy_(self._bufs["anchor_pos"][:B]).add_(slot_idx)
            return gumbel_sample(
                logits,
                self._bufs["req_slots"][:B],
                self.temperature,
                self.seeds,
                pos,
                apply_temperature=True,
                is_drafting=True,
                output_processed_logits=self.draft_logits,
                output_processed_logits_col=self._slot_idx_tensors[slot_idx],
                use_fp64=self.use_fp64_gumbel,
            )

        lp = self.target_lm.logits_processor
        if self.screen_head is not None and B <= self.screen_head.maxr:
            return lp.get_top_tokens(self.screen_head, h_corr)

        # Exact FP8 single-pass top-1 (CUDA-graph safe when using HeadCopy.top1 directly into static buffer)
        lm_head = self.target_lm.lm_head
        hc = getattr(lm_head, "q27_head", None)
        if hc is not None and 1 <= B <= 128:
            pairs = self._bufs["top1_pairs"][:B]
            hc.top1(h_corr, out_ptr=pairs.data_ptr())
            custom = lp._maybe_custom_top1_argmax(pairs)
            if custom is not None:
                return custom
            from vllm.distributed import tensor_model_parallel_all_gather
            gathered = tensor_model_parallel_all_gather(pairs, dim=-1).view(B, self.tp_size, 2)
            return gathered[:, :, 1].gather(dim=-1, index=gathered[:, :, 0].argmax(dim=-1, keepdim=True)).squeeze(-1).to(torch.int64)
        return self.target_lm.get_top_tokens(h_corr)

    def _run_dfly2_core(self, B: int, K_active: int, skip_attn: bool = False) -> None:
        """Execute full DFly2-KV draft step (Stage A + Stage B + Stage C slots 0..K_active) on persistent buffers."""
        from vllm.distributed import tensor_model_parallel_all_reduce
        from . import ops

        ext = dfly2_ext()
        m = self.model
        assert m is not None
        bufs = self._bufs
        N6 = B * BLK

        anchor_tok = bufs["anchor_tok"][:B]
        ctx_len = bufs["ctx_len"][:B]
        ctx_len_6 = bufs["ctx_len_6"][:N6]
        anchor_pos = bufs["anchor_pos"][:B]
        prev_hidden = bufs["prev_hidden"][:B]

        # ---- Stage A: anchor embedding + MTPFusion + 6-token block init + Layer 0 input_layernorm ----
        e0 = self.target_lm.model.embed_tokens(anchor_tok)   # [B, 5120] fp16
        eh_in = bufs["eh_in_local"][:B]
        eh_out = bufs["eh_out"][:B]
        ext.stage_a_prep(prev_hidden, e0, m.mtp_hnorm, m.mtp_enorm, eh_in, self.tp_rank, EPS)
        m.eh_proj.forward_into(eh_in, eh_out)
        eh_red = tensor_model_parallel_all_reduce(eh_out)

        h_block = bufs["h_block"][:N6]
        res_f32 = bufs["res_f32"][:N6]
        x_norm = bufs["x_norm"][:N6]
        ext.stage_a_fuse_norm(e0, eh_red, m.mtp_gate, m.mask_embedding, m.layers[0].input_ln,
                              h_block, res_f32, x_norm, EPS)

        # ---- Stage B: 3x bidirectional backbone layers reading target layers [47, 55, 63] KV ----
        qkv_b = bufs["qkv_b"][:N6]
        q_b = bufs["q_b"][:N6]
        k_blk_b = bufs["k_blk_b"][:B]
        v_blk_b = bufs["v_blk_b"][:B]
        o_ctx_b = bufs["o_ctx_b"][:N6]
        lse_ctx_b = bufs["lse_ctx_b"][:N6]
        attn_out_b = bufs["attn_out_b"][:N6]
        o_part_b = bufs["o_part_b"][:N6]
        mlp_ws = ops.workspace(self.device, max(2 * I_LOCAL * H_DIM, H_DIM * I_LOCAL)) if N6 > ops.MLP_MAX else e0.new_empty(0)

        has_fused_arn = hasattr(torch.ops.vllm, "sm70_tp4_all_reduce_gemma_rms_norm")
        for li in range(3):
            layer = m.layers[li]
            layer.qkv_proj.forward_into(x_norm, qkv_b)
            ext.stage_b_qk_prep(qkv_b, anchor_pos, layer.q_norm, layer.k_norm, q_b, k_blk_b, v_blk_b, EPS)
            o_in, w_v_up_cm = self._run_target_kv_attn(
                li, q_b, B, BLK, ctx_len_6, o_ctx_b, lse_ctx_b, anchor_pos, BLK, 0, skip_attn=skip_attn,
            )
            ext.stage_b_merge(o_in, lse_ctx_b, q_b, k_blk_b, v_blk_b, qkv_b, w_v_up_cm, attn_out_b, SCALE)
            layer.o_proj.forward_into(attn_out_b, o_part_b)
            if has_fused_arn:
                x_norm, res_f32 = torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm(
                    o_part_b, res_f32, layer.post_ln, EPS, self.tp_group_name,
                )
            else:
                o_red = tensor_model_parallel_all_reduce(o_part_b)
                ext.gemma_rmsnorm_res(o_red, res_f32, layer.post_ln, x_norm, EPS)
            mlp_part = layer.mlp(x_norm, mlp_ws)
            next_ln = m.layers[li + 1].input_ln if li < 2 else m.final_norm
            if has_fused_arn:
                x_norm, res_f32 = torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm(
                    mlp_part, res_f32, next_ln, EPS, self.tp_group_name,
                )
            else:
                mlp_red = tensor_model_parallel_all_reduce(mlp_part)
                ext.gemma_rmsnorm_res(mlp_red, res_f32, next_ln, x_norm, EPS)

        # x_norm is now h_backbone_normed [B*6, 5120] fp16!
        h_bb = x_norm

        # ---- Stage C: Slot 0 KV-only init (bit-exact skip of MLP/Q/L63/O/lm_head!) ----
        inp_c = bufs["inp_c"][:B]
        kv0_c = bufs["kv0_c"][:B]
        blk_k_c = bufs["blk_k_c"][:B]
        blk_v_c = bufs["blk_v_c"][:B]
        ext.stage_c_prep(h_bb, e0, m.hc_hidden_norm, m.hc_embed_norm, inp_c, 0, EPS)
        m.hc_kv0_proj.forward_into(inp_c, kv0_c)
        ext.stage_c_slot0_kv(kv0_c, anchor_pos, m.hc_k_blk_norm, blk_k_c, blk_v_c, EPS)

        # ---- Stage C: Autoregressive AttentiveCorrection slots 1..K_active ----
        mega_c = bufs["mega_c"][:B]
        q_c = bufs["q_c"][:B]
        o_ctx_c = bufs["o_ctx_c"][:B]
        mid_c = bufs["mid_c"][:B]
        attn_out_c = bufs["attn_out_c"][:B]
        delta_mlp = bufs["delta_mlp"][:B]
        delta_attn = bufs["delta_attn"][:B]
        h_corr = bufs["h_corr"][:B]
        gate_raw = bufs["gate_raw"][:B]

        e_prev = e0
        for slot in range(1, K_active + 1):
            ext.stage_c_prep(h_bb, e_prev, m.hc_hidden_norm, m.hc_embed_norm, inp_c, slot, EPS)
            m.hc_mega_proj.forward_into(inp_c, mega_c)
            torch.matmul(inp_c, m.hc_block_gate_w.T, out=gate_raw)
            ext.stage_c_qk_silu(mega_c, anchor_pos, m.hc_q_norm, m.hc_k_blk_norm,
                                q_c, blk_k_c, blk_v_c, mid_c, slot, EPS)
            o_in_c, w_v_up_c = self._run_target_kv_attn(
                2, q_c, B, 1, ctx_len, o_ctx_c, None, anchor_pos, 0, slot, skip_attn=skip_attn,
            )
            ext.stage_c_combine(o_in_c, q_c, blk_k_c, blk_v_c, gate_raw,
                                m.hc_block_gate_bias, m.hc_block_rel_bias, m.hc_block_sink,
                                w_v_up_c, attn_out_c, slot, SCALE)
            m.hc_down_proj.forward_into(mid_c, delta_mlp)
            m.hc_o_proj.forward_into(attn_out_c, delta_attn)
            delta_mlp.add_(delta_attn)
            delta_red = tensor_model_parallel_all_reduce(delta_mlp)
            ext.add_slot_residual(h_bb, delta_red, h_corr, slot)

            tok = self._top1_draft_token(h_corr, B, slot - 1)
            self.draft_tokens[:B, slot - 1].copy_(tok)
            if slot < K_active:
                e_prev = self.target_lm.model.embed_tokens(tok.to(torch.int32))

    def _k_for(self, num_reqs: int) -> int:
        if self._spec_k_cfg is None:
            return self.num_speculative_steps
        from . import spec_k
        return spec_k.k_for(num_reqs, self.num_speculative_steps, self._spec_k_cfg)

    def capture(self, captured_attn_states=None) -> None:
        del captured_attn_states
        self._ensure_kv_ready()
        if not self._use_graphs or self.draft_logits is not None:
            return
        if getattr(self, "_cudagraph_mode", CUDAGraphMode.NONE) == CUDAGraphMode.NONE:
            return

        from . import spec_k
        if self._spec_k_cfg is not None:
            rng = spec_k.ranges(self._spec_k_cfg, self.num_speculative_steps, self.max_num_reqs)
        else:
            rng = [(self.num_speculative_steps, 1, self.max_num_reqs)]

        pairs: list[tuple[int, int]] = []
        by_k: dict[int, list[int]] = {}
        for k, lo, hi in rng:
            if k <= 0:
                continue
            counts = spec_k.capture_counts(lo, hi)
            by_k[k] = counts
            for b in counts:
                pairs.append((b, k))
        self._capture_counts_by_k = by_k

        # Zero persistent inputs before capture (ctx_len=0 -> safe valid dummy attention)
        self._bufs["anchor_tok"].zero_()
        self._bufs["ctx_len"].zero_()
        self._bufs["ctx_len_6"].zero_()
        self._bufs["anchor_pos"].zero_()
        self._bufs["prev_hidden"].zero_()
        self._bufs["req_slots"].zero_()

        self._graph_pool = torch.cuda.graph_pool_handle()
        # Capture largest first to prime pool memory cleanly
        for b, k in sorted(pairs, key=lambda bk: (bk[0], bk[1]), reverse=True):
            # Warmup once eagerly before capture
            self._run_dfly2_core(b, k, skip_attn=False)
            torch.cuda.synchronize(self.device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._graph_pool):
                self._run_dfly2_core(b, k, skip_attn=False)
            self._graphs[(b, k)] = g
        torch.cuda.synchronize(self.device)
        if self.tp_rank == 0:
            log.warning("q27fuse.dfly2: captured %d FULL CUDA graphs (%s)",
                        len(self._graphs),
                        ", ".join(f"K={k}:{cs}" for k, cs in sorted(by_k.items())))

    def _pick_graph_b(self, num_reqs: int, k: int) -> int | None:
        if self.draft_logits is not None or not self._graphs:
            return None
        for b in self._capture_counts_by_k.get(k, ()):
            if b >= num_reqs and (b, k) in self._graphs:
                return b
        return None

    def propose(
        self,
        input_batch,
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
            # Memory profile / pre-KV-alloc dummy run: run core with skip_attn=True so peak memory is profiled
            b_run = min(num_reqs, self.max_num_reqs)
            self._bufs["anchor_tok"][:b_run].zero_()
            self._bufs["ctx_len"][:b_run].zero_()
            self._bufs["ctx_len_6"][:b_run * BLK].zero_()
            self._bufs["anchor_pos"][:b_run].zero_()
            self._bufs["prev_hidden"][:b_run].zero_()
            self._run_dfly2_core(b_run, k, skip_attn=True)
            return self.draft_tokens[:num_reqs]

        self._ensure_kv_ready()
        if self.draft_logits is not None:
            self.temperature.copy_(temperature)
            self.seeds.copy_(seeds)

        b_graph = None if dummy_run else self._pick_graph_b(num_reqs, k)
        b_pad = b_graph if b_graph is not None else num_reqs

        # Extract per-request inputs on GPU (zero CPU sync!)
        dfly2_ext().extract_inputs(
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
            self._bufs["ctx_len_6"],
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
                log.warning("q27fuse.dfly2: FULL graph replay B=%d (pad=%d) K=%d", num_reqs, b_graph, k)
        else:
            self._run_dfly2_core(b_pad, k, skip_attn=False)
            if ("eager", b_pad, k) not in _SEEN and self.tp_rank == 0 and len(_SEEN) < 32:
                _SEEN.add(("eager", b_pad, k))
                log.warning("q27fuse.dfly2: eager run B=%d K=%d (dummy=%s)", b_pad, k, dummy_run)

        return self.draft_tokens[:num_reqs]


def apply() -> None:
    if not enabled():
        return
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu import spec_decode as SD

    if getattr(SD, "_q27_dfly2_patched", False):
        return

    orig_init = SD.init_speculator

    def init_speculator(vllm_config: VllmConfig, device: torch.device):
        sc = vllm_config.speculative_config
        if sc is not None and sc.method == "mtp" and enabled():
            log.warning("q27fuse.dfly2: replacing EagleSpeculator/Qwen3_5MTP with Q27DFly2Speculator (%s, K=%d)",
                        os.path.basename(ckpt_path().rstrip("/")), sc.num_speculative_tokens)
            return Q27DFly2Speculator(vllm_config, device)
        return orig_init(vllm_config, device)

    SD.init_speculator = init_speculator
    MR.init_speculator = init_speculator
    SD._q27_dfly2_patched = True
    log.warning("q27fuse.dfly2: installed Q27DFly2Speculator hook (Q27_DFLY2_CKPT=%s)", ckpt_path())
