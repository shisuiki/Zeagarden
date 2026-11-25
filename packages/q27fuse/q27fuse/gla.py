"""q27fuse.gla - TP4-native GLA-g4 (Grouped Latent Attention) runtime & Volta sm_70 absorbed FP8-KV backend (v2).

Activated only when Q27_GLA_CKPT=<path to fitted GLA-g4 .pt checkpoint> is set (zero impact otherwise).
Supports:
  - Uniform r_g = 192 (gla_g4_u192_qo_mtp.pt): 256 B/tok on all 16 target + 1 MTP full-attn layers (exact 2.00x KV,
    0 hybrid-allocator padding layers in MTP4 align mode!).
  - Uniform r_g = 192 (gla_g4_u192_qo.pt):     256 B/tok on 16 target full-attn layers.
  - Adaptive 1.6x (gla_g4_adap_16x_qo.pt):     12 layers r_g=192 (256 B/tok) + 4 lossless layers (512 B/tok, untouched).
  - Uniform r_g = 256 (gla_g4_u256_qo*.pt):    320 B/tok on all compressed full-attn layers (exact 1.60x KV).

Zero core vLLM edits & minimal launch count (5 kernels/layer in decode/verify, matching baseline p17):
  1. Single fused QPN8 GEMM qkv_proj -> qckr [N, 3072 + r_g + 64] (slices off 512 unused K/V cols and splices kv_ar
     directly in QPN8 format on both target and MTP layers, saving -7.1% qkv weight bytes and 16 GEMM launches/step).
  2. Route A (Decode & MTP Verify, Q=1..8, FULL CUDA-graph safe, zero allocations inside capture):
       gla_prep_route_a -> attn_gla_mq (+ k_comb) -> gla_vup_gate
     (passes qckr & positions directly through vLLM's native unified_attention_with_output custom op).
  3. Route B (Cold / Warm / Chunked Prefill):
       gla_prep_prefill -> Flash-V100 dense prefill (with gla_gather_dequant for prefix) -> gla_gate_only.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import torch
import torch.nn.functional as F

log = logging.getLogger("q27fuse.gla")

FULL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
FULL_LAYER_SET = set(FULL_LAYERS)
ROPE_DIM = 62
MAX_B = 256
MAX_ROUTE_A_TOK = 2048

_EXT = None
_CKPT_CPU = None
_WS: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_OFFSETS: dict[tuple, torch.Tensor] = {}
_SEEN: set = set()
_ACTIVE_OWNERS: list = []
_QKV_SPECS: dict[str, tuple[object, int, torch.Tensor]] = {}  # qkv_proj.prefix -> (owner_mod, r_g, kv_ar_pad_cpu)
_AOT_TAGS: list[str] = []


def _gla_ckpt_path() -> str:
    return os.environ.get("Q27_GLA_CKPT", "").strip()


def _gla_scale_mult() -> float:
    return float(os.environ.get("Q27_GLA_SCALE_MULT", "1.0") or "1.0")


def _gla_splits_cfg() -> int:
    return int(os.environ.get("Q27_GLA_S", "128") or "128")


def _gla_sep_cfg() -> int:
    return int(os.environ.get("Q27_GLA_SEP", "1") or "1")


def _gla_fuse_qkv() -> bool:
    return (os.environ.get("Q27_GLA_FUSE_QKV", "1") or "1").strip() != "0"


def _gla_mtp_enabled() -> bool:
    return (os.environ.get("Q27_GLA_MTP", "1") or "1").strip() != "0"


def _gla_csrc() -> str:
    env_src = os.environ.get("Q27_GLA_CSRC", "").strip()
    if env_src and os.path.isfile(env_src):
        return env_src
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "attn_gla_mq.cu")


def gla_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        from .ops import default_build_dir
        bdir = os.environ.get("Q27_GLA_BUILD", "").strip() or default_build_dir("_gla")
        os.makedirs(bdir, exist_ok=True)
        _EXT = load(
            name="q27_attn_gla_mq",
            sources=[_gla_csrc()],
            build_directory=bdir,
            extra_cuda_cflags=["-O3", "-arch=sm_70", "-std=c++17", "--use_fast_math", "--ptxas-options=-v"],
            verbose=False,
        )
    return _EXT


def _load_ckpt_cpu() -> dict:
    global _CKPT_CPU
    if _CKPT_CPU is None:
        path = _gla_ckpt_path()
        assert path and os.path.isfile(path), f"Q27_GLA_CKPT not found: {path!r}"
        if path.endswith(".safetensors"):
            import json
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as sf:
                meta = json.loads((sf.metadata() or {})["meta"])
                groups = int(meta["groups"])
                assert groups == 4, f"q27fuse.gla requires TP4-native GLA-g4 (groups=4), got {groups}"
                keys = set(sf.keys())

                def _read_entry(tag: str, info: dict) -> dict:
                    return {
                        "rope_keep": list(info["rope_keep"]),
                        "kv_scale": float(info["kv_scale"]),
                        "kv_a": [sf.get_tensor(f"{tag}.kv_a")[g].contiguous() for g in range(groups)],
                        "k_rope": [sf.get_tensor(f"{tag}.k_rope")[g].contiguous() for g in range(groups)],
                        "k_up_raw": [sf.get_tensor(f"{tag}.k_up_raw")[g].contiguous() for g in range(groups)],
                        "v_up": [sf.get_tensor(f"{tag}.v_up")[g].contiguous() for g in range(groups)],
                    }

                layers = {int(li): _read_entry(f"L{li}", info) for li, info in meta["layers"].items()}
                raw = {"name": meta.get("name", os.path.basename(path)), "groups": groups, "layers": layers}
                if "mtp0" in meta and "mtp0.kv_a" in keys:
                    raw["mtp0"] = _read_entry("mtp0", meta["mtp0"])
        else:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            assert int(raw["groups"]) == 4, f"q27fuse.gla requires TP4-native GLA-g4 (groups=4), got {raw['groups']}"
        _CKPT_CPU = raw
    return _CKPT_CPU


def rope_perm(model_rope_dim: int, head_dim: int, keep: list[int]) -> list[int]:
    half = model_rope_dim // 2
    keep = [int(k) for k in keep]
    drop = [s for s in range(half) if s not in keep]
    perm = keep + [s + half for s in keep] + drop + [s + half for s in drop] + list(range(model_rope_dim, head_dim))
    assert len(perm) == head_dim and sorted(perm) == list(range(head_dim))
    return perm


def _get_offsets(dev: torch.device, Q: int) -> torch.Tensor:
    key = (dev.index, Q)
    t = _OFFSETS.get(key)
    if t is None:
        t = torch.arange(-(Q - 1), 1, dtype=torch.int32, device=dev)
        _OFFSETS[key] = t
    return t


def _pick_splits(B: int, Q: int, r_lat: int) -> int:
    """Choose split count S targeting 160..320 CTAs across V100's 80 SMs (bounding workspace <= ~10 MiB)."""
    s_max = _gla_splits_cfg()
    target_ctas = 320 if Q <= 2 else (240 if Q <= 4 else 160)
    return max(1, min(s_max, target_ctas // max(1, B)))


def _ensure_runtime_buffers(owner_mod, device: torch.device, dtype: torch.dtype) -> None:
    """Build runtime FP32/FP16 derived buffers lazily after checkpoint weights are loaded (and before graph capture)."""
    if owner_mod._gla_rt_ready:
        return
    assert not torch.cuda.is_current_stream_capturing(), (
        "q27fuse.gla: runtime buffers first initialized inside CUDA graph capture!"
    )
    perm_t = owner_mod._gla_perm_t.to(device)
    qw1p = (1.0 + owner_mod.q_norm.weight.detach().to(device=device, dtype=torch.float32)).contiguous()
    kw1p_perm = (1.0 + owner_mod.k_norm.weight.detach().to(device=device, dtype=torch.float32)).index_select(0, perm_t)
    kw_rope_perm = kw1p_perm[:ROPE_DIM].contiguous()
    k_up_raw = owner_mod._gla_k_up_raw.to(device=device, dtype=dtype)
    k_up_raw_cm = k_up_raw.T.contiguous()  # [r_lat, 194] coalesced across 194 rows
    k_up_sc_cm = (k_up_raw.float() * kw1p_perm[ROPE_DIM:, None]).to(dtype).contiguous()  # [194, r_lat]
    v_up = owner_mod._gla_v_up.to(device=device, dtype=dtype).contiguous()  # [256, r_lat]
    v_up_cm = v_up.T.contiguous()  # [r_lat, 256]

    owner_mod._gla_q_norm_w1p = qw1p
    owner_mod._gla_kw_rope_perm = kw_rope_perm
    owner_mod._gla_k_up_raw_cm = k_up_raw_cm
    owner_mod._gla_k_up_sc_cm = k_up_sc_cm
    owner_mod._gla_v_up_cm = v_up_cm
    owner_mod._gla_rt_ready = True


def _ensure_workspace(dev: torch.device, hkv: int = 1):
    key = (dev.index, hkv)
    ws = _WS.get(key)
    if ws is not None:
        return ws
    assert not torch.cuda.is_current_stream_capturing(), (
        "q27fuse.gla: workspace first allocated inside CUDA graph capture (prealloc hook missed!)"
    )
    ext = gla_ext()
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    ext.init(idx)
    sep = _gla_sep_cfg()
    n_po = 16
    n_pml = 16
    with torch.cuda.device(dev):
        for r_lat in (192, 256):
            for Q in range(1, 9):
                for b in range(1, MAX_B + 1):
                    s_req = _pick_splits(b, Q, r_lat)
                    a, c2 = ext.gla_workspace(b * hkv, s_req, r_lat, Q, sep)
                    n_po = max(n_po, int(a))
                    n_pml = max(n_pml, int(c2))
        po = torch.empty((n_po,), dtype=torch.float32, device=dev)
        pml = torch.empty((n_pml,), dtype=torch.float32, device=dev)
        cnt = torch.zeros((MAX_B * hkv,), dtype=torch.int32, device=dev)
        q_cat_buf = torch.empty((MAX_ROUTE_A_TOK * 6 * 320,), dtype=torch.float16, device=dev)
        o_lat_buf = torch.empty((MAX_ROUTE_A_TOK * 6 * 256,), dtype=torch.float16, device=dev)
        for Q in range(1, 9):
            _get_offsets(dev, Q)
    _WS[key] = (po, pml, cnt, q_cat_buf, o_lat_buf)
    if idx == 0:
        tot_mb = ((n_po + n_pml) * 4 + (q_cat_buf.numel() + o_lat_buf.numel()) * 2) / (1 << 20)
        log.warning("q27fuse.gla: preallocated workspace %.2f MiB on cuda:%d (po=%d, pml=%d)",
                    tot_mb, idx, n_po, n_pml)
    return _WS[key]


def _install_prealloc():
    from vllm.v1.worker.gpu import cudagraph_utils as CU
    M = CU.CudaGraphManager
    if getattr(M, "_q27_gla_prealloc", False):
        return
    prev = M.capture

    def capture(self, *a, **kw):
        try:
            dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
            if dev.type == "cuda":
                _ensure_workspace(dev, 1)
                for owner in _ACTIVE_OWNERS:
                    _ensure_runtime_buffers(owner, dev, torch.float16)
        except Exception as e:  # noqa: BLE001
            log.warning("q27fuse.gla: pre-capture init failed (%r)", e)
        return prev(self, *a, **kw)

    M.capture = capture
    M._q27_gla_prealloc = True


def _tag_aot_cache(tag: str) -> None:
    """Mix GLA tag into vLLM's AOT compile-cache key for both target and drafter so caches never collide."""
    from vllm.compilation import decorators as D
    if tag not in _AOT_TAGS:
        _AOT_TAGS.append(tag)
    orig = D._model_hash_key
    if getattr(orig, "_q27_gla_tagged", False):
        return

    def _model_hash_key(fn):
        h = orig(fn)
        if _AOT_TAGS:
            h = hashlib.sha256((h + "|" + "|".join(_AOT_TAGS)).encode()).hexdigest()
        return h

    _model_hash_key._q27_gla_tagged = True
    D._model_hash_key = _model_hash_key


def _install_qkv_splicing():
    """Splice kv_ar into QPN8 qkv_proj weights (3584 -> 3072 + r_g + 64) on both target and MTP layers."""
    from vllm.model_executor.utils import replace_parameter
    fuse_qkv = _gla_fuse_qkv()

    # 1. Target layers: CompressedTensorsW8A16Fp8 (runs after patch.apply_fp8's wrapper too)
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a16_fp8 as S
    cls = S.CompressedTensorsW8A16Fp8
    if not getattr(cls, "_q27_gla_qkv_patched", False):
        orig_pw = cls.process_weights_after_loading

        def process_weights_after_loading(self, layer):
            r = orig_pw(self, layer)
            spec = _QKV_SPECS.get(getattr(layer, "prefix", "") or "")
            if spec is None:
                return r
            owner, r_g, kv_ar_pad_cpu = spec
            w = getattr(layer, "weight", None)
            sc = getattr(layer, "weight_scale_inv", None)
            if (getattr(layer, "sm70_fp8_qpn8", False) and w is not None and sc is not None
                    and w.dtype == torch.uint8 and tuple(w.shape) == (5120, 3584) and sc.numel() == 3584):
                dev = w.device
                k_dim = 5120
                q_cols = 3072
                orig_flat = w.view(-1)[:q_cols * k_dim]
                orig_sc = sc.view(-1)[:q_cols]
                if fuse_qkv:
                    from vllm.model_executor.layers.quantization.sm70_online_qpn8 import prepare_channel_qpn8_weight
                    ar_codes, ar_sc = prepare_channel_qpn8_weight(kv_ar_pad_cpu.to(device=dev, dtype=torch.float16))
                    new_n = q_cols + r_g + 64
                    new_w = torch.cat([orig_flat, ar_codes.view(-1)], dim=0).view(k_dim, new_n).contiguous()
                    new_sc = torch.cat([orig_sc, ar_sc.view(-1)], dim=0).contiguous()
                    owner._gla_qkv_fused = True
                    owner._gla_kv_ar = None  # free FP16 copy!
                else:
                    new_n = q_cols
                    new_w = orig_flat.contiguous().view(k_dim, new_n)
                    new_sc = orig_sc.contiguous()
                    owner._gla_qkv_fused = False
                if sc.ndim == 2:
                    new_sc = new_sc.view(1, new_n)
                replace_parameter(layer, "weight", new_w)
                replace_parameter(layer, "weight_scale_inv", new_sc)
                layer.output_size_per_partition = new_n
                if ("qkv_splice_tgt",) not in _SEEN and dev.index == 0:
                    _SEEN.add(("qkv_splice_tgt",))
                    log.warning("q27fuse.gla: spliced target QPN8 qkv_proj 3584 -> %d (fuse_qkv=%s, %s)",
                                new_n, fuse_qkv, layer.prefix)
            return r

        cls.process_weights_after_loading = process_weights_after_loading
        cls._q27_gla_qkv_patched = True

    # 2. MTP drafter layer: sm70_mtp_draft_qpn8.prepare_mtp_draft_qpn8 (before QPN8 online packing)
    try:
        from vllm.model_executor.layers.quantization import sm70_mtp_draft_qpn8 as MQ
        if not getattr(MQ, "_q27_gla_qkv_patched", False):
            orig_prep = MQ.prepare_mtp_draft_qpn8

            @torch.no_grad()
            def prepare_mtp_draft_qpn8(layer):
                spec = _QKV_SPECS.get(getattr(layer, "prefix", "") or "")
                if spec is not None and not getattr(layer, MQ._READY, False):
                    owner, r_g, kv_ar_pad_cpu = spec
                    w = getattr(layer, "weight", None)
                    if w is not None and w.dtype == torch.float16 and tuple(w.shape) == (3584, 5120):
                        dev = w.device
                        if fuse_qkv:
                            ar_gpu = kv_ar_pad_cpu.to(device=dev, dtype=torch.float16)
                            new_w = torch.cat([w[:3072], ar_gpu], dim=0).contiguous()
                            new_n = 3072 + r_g + 64
                            owner._gla_qkv_fused = True
                            owner._gla_kv_ar = None
                        else:
                            new_w = w[:3072].contiguous()
                            new_n = 3072
                            owner._gla_qkv_fused = False
                        replace_parameter(layer, "weight", new_w)
                        layer.output_size_per_partition = new_n
                        if ("qkv_splice_mtp",) not in _SEEN and dev.index == 0:
                            _SEEN.add(("qkv_splice_mtp",))
                            log.warning("q27fuse.gla: spliced MTP qkv_proj 3584 -> %d before QPN8 pack (fuse_qkv=%s, %s)",
                                        new_n, fuse_qkv, layer.prefix)
                return orig_prep(layer)

            MQ.prepare_mtp_draft_qpn8 = prepare_mtp_draft_qpn8
            MQ._q27_gla_qkv_patched = True
    except Exception as e:  # noqa: BLE001
        log.warning("q27fuse.gla: could not hook sm70_mtp_draft_qpn8 (%r)", e)


def _make_gla_impl_forward(owner_mod, orig_impl_fwd):
    """Wrap Attention.impl.forward for a single compressed GLA-g4 layer (r_g in {192, 256}).

    Caller passes:
      query = qckr [N, stride_qckr] fp16 contiguous (= [q_gate(3072) | c(r_g) | k_rope_raw(62) | pad])
      key   = positions ([N] or [3, N], int64/int32)
      output = [N, 6, 256] fp16 contiguous (written with post-gate attention output!)
    """
    from vllm.v1.attention.backends.flash_v100 import dense_prefill as _dense_prefill
    from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
    from vllm.v1.attention.backends.flash_v100 import ops as _ops

    r_lat = owner_mod._gla_rank
    row_b = r_lat + 64
    scale = float(owner_mod.scaling)
    kv_scale = float(owner_mod._gla_kv_scale)
    eps = float(owner_mod.q_norm.variance_epsilon)
    tag = owner_mod._gla_tag
    sep = _gla_sep_cfg()

    def gla_impl_forward(
        layer,
        query: torch.Tensor,       # qckr [N, stride_qckr]
        key: torch.Tensor,         # positions [N] or [3, N]
        value: torch.Tensor,       # unused
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale=None,
        output_block_scale=None,
    ) -> torch.Tensor:
        assert output is not None
        if attn_metadata is None:
            return output.fill_(0)

        ext = gla_ext()
        qckr = query
        positions = key
        dev = qckr.device
        dtype = qckr.dtype
        _ensure_runtime_buffers(owner_mod, dev, dtype)

        N = int(qckr.shape[0])
        if N == 0:
            return output

        k_cache, v_cache = _kv_layout.split_paged_kv_cache(kv_cache)
        slot_mapping = attn_metadata.slot_mapping[:N].contiguous()
        cos_sin_cache = owner_mod.rotary_emb.cos_sin_cache

        is_capturing = torch.cuda.is_current_stream_capturing()
        max_q = int(attn_metadata.max_query_len)
        B_bt = int(attn_metadata.seq_lens.shape[0])

        # Determine whether this step is Route A (Absorbed Decode / Uniform MTP Verify, Q in 1..8)
        use_route_a = False
        Q = 1
        if N <= MAX_ROUTE_A_TOK:
            if max_q == 1 and N == B_bt:
                use_route_a = True
                Q = 1
            elif 2 <= max_q <= 8 and B_bt > 0 and N == B_bt * max_q:
                if is_capturing:
                    use_route_a = True
                    Q = max_q
                else:
                    qsl_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
                    if qsl_cpu is not None and len(qsl_cpu) == B_bt + 1:
                        qlens = qsl_cpu[1:] - qsl_cpu[:-1]
                        if bool(torch.all(qlens == max_q).item()):
                            use_route_a = True
                            Q = max_q
                    else:
                        qsl = attn_metadata.query_start_loc
                        if len(qsl) == B_bt + 1:
                            qlens = qsl[1:] - qsl[:-1]
                            if bool(torch.all(qlens == max_q).item()):
                                use_route_a = True
                                Q = max_q

        # ---- Route A: 3-Kernel Fused Absorbed Decode & MTP Verify (Zero Allocations Inside Capture!) ----
        if use_route_a:
            po, pml, cnt, q_cat_buf, o_lat_buf = _ensure_workspace(dev, 1)
            if ("route_a", tag, Q) not in _SEEN and dev.index == 0:
                _SEEN.add(("route_a", tag, Q))
                log.warning("q27fuse.gla: Route A fused active (%s, r=%d, B=%d, Q=%d, cap=%s)",
                            tag, r_lat, B_bt, Q, is_capturing)

            q_cat = q_cat_buf[:N * 6 * row_b].view(N, 6, row_b)
            o_lat = o_lat_buf[:N * 6 * r_lat].view(N, 6, r_lat)

            # 1. Fused Q-norm + perm + M-RoPE + Q_abs + K inv_rms + K-RoPE + E4M3 paged cache write
            ext.gla_prep_route_a(
                qckr, positions,
                owner_mod._gla_q_norm_w1p, owner_mod._gla_kw_rope_perm,
                owner_mod._gla_k_up_raw_cm, owner_mod._gla_k_up_sc_cm,
                cos_sin_cache, slot_mapping, k_cache, v_cache, q_cat,
                eps, kv_scale,
            )

            seq_lens = attn_metadata.seq_lens[:B_bt].contiguous()
            if Q == 1:
                per_row_lens = seq_lens
            else:
                offs = _get_offsets(dev, Q)
                per_row_lens = (seq_lens[:, None] + offs[None, :]).clamp_(min=0).reshape(-1).contiguous()

            # 2. Single-read shared-latent Volta HMMA.884 attention
            S_req = _pick_splits(B_bt, Q, r_lat)
            ext.attn_gla_mq(
                q_cat, k_cache, v_cache, attn_metadata.block_table[:B_bt],
                per_row_lens, o_lat, po, pml, cnt,
                scale, kv_scale, Q, S_req, 1, sep,
            )

            # 3. Fused o_lat @ W_v_up^T * sigmoid(gate) directly into output!
            ext.gla_vup_gate(o_lat, owner_mod._gla_v_up_cm, qckr, output)
            return output

        # ---- Route B: Materialized Flash-V100 Prefill (Cold, Warm Prefix-Cached, or Chunked) ----
        q_full = torch.empty((N, 6, 256), dtype=dtype, device=dev)
        k_rot = torch.empty((N, ROPE_DIM), dtype=dtype, device=dev)
        inv_rms_f32 = torch.empty((N,), dtype=torch.float32, device=dev)
        ext.gla_prep_prefill(
            qckr, positions,
            owner_mod._gla_q_norm_w1p, owner_mod._gla_kw_rope_perm,
            owner_mod._gla_k_up_raw_cm, cos_sin_cache,
            slot_mapping, k_cache, v_cache,
            q_full, k_rot, inv_rms_f32,
            eps, kv_scale,
        )

        has_prefix = _kv_layout.has_prefix_context(attn_metadata)
        num_actual = int(attn_metadata.num_actual_tokens)
        qsl_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        qsl = qsl_cpu if qsl_cpu is not None else attn_metadata.query_start_loc.cpu()
        num_seqs = len(qsl) - 1

        if not has_prefix:
            if ("route_b1", tag) not in _SEEN and dev.index == 0:
                _SEEN.add(("route_b1", tag))
                log.warning("q27fuse.gla: Route B1 cold prefill active (%s, r=%d, N=%d)", tag, r_lat, num_actual)
            c_act = qckr[:num_actual, 3072:3072 + r_lat]
            k_nope = F.linear(c_act, owner_mod._gla_k_up_sc_cm)
            k_full = (torch.cat([k_rot[:num_actual], k_nope], dim=-1).float() * inv_rms_f32[:num_actual, None]).to(dtype).unsqueeze(1).contiguous()
            v_full = F.linear(c_act, owner_mod._gla_v_up).unsqueeze(1).contiguous()
            _dense_prefill.flash_v100_dense_prefill(
                query=q_full,
                key=k_full,
                value=v_full,
                output=output,
                query_start_loc=qsl,
                num_actual_tokens=num_actual,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                query_start_loc_device=attn_metadata.query_start_loc,
            )
            ext.gla_gate_only(output, qckr, output)
            return output

        # Route B2: Warm / chunked prefill — gather & dequantize compressed prefix+new tokens from paged cache
        if ("route_b2", tag) not in _SEEN and dev.index == 0:
            _SEEN.add(("route_b2", tag))
            log.warning("q27fuse.gla: Route B2 warm/chunked prefill active (%s, r=%d, seqs=%d)", tag, r_lat, num_seqs)
        seq_lens_gpu = attn_metadata.seq_lens[:num_seqs].contiguous()
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens_gpu.cpu()
        else:
            seq_lens_cpu = seq_lens_cpu[:num_seqs]
        cu_kv_cpu = torch.zeros((num_seqs + 1,), dtype=torch.int32)
        torch.cumsum(seq_lens_cpu, dim=0, out=cu_kv_cpu[1:])
        total_kv = int(cu_kv_cpu[-1].item())
        cu_kv_gpu = cu_kv_cpu.to(dev, non_blocking=True)

        c_all = torch.empty((total_kv, 1, r_lat), dtype=dtype, device=dev)
        kr_all = torch.empty((total_kv, 1, ROPE_DIM), dtype=dtype, device=dev)
        ir_all = torch.empty((total_kv, 1), dtype=torch.float32, device=dev)
        ext.gla_gather_dequant(
            k_cache, v_cache, attn_metadata.block_table[:num_seqs],
            seq_lens_gpu, cu_kv_gpu, c_all, kr_all, ir_all, kv_scale,
        )

        # Materialize K and V one sequence at a time (33x lower peak VRAM at C=32 x 8K chunked prefill!)
        flash_attn_func = _ops.get_flash_ops()[0]
        for b_i in range(num_seqs):
            qs = int(qsl[b_i].item())
            qe = int(qsl[b_i + 1].item())
            ks = int(cu_kv_cpu[b_i].item())
            ke = int(cu_kv_cpu[b_i + 1].item())
            if qe <= qs or ke <= ks:
                continue
            c_b = c_all[ks:ke, 0, :]
            kr_b = kr_all[ks:ke, 0, :]
            ir_b = ir_all[ks:ke, :]
            k_nope_b = F.linear(c_b, owner_mod._gla_k_up_sc_cm)
            k_b = (torch.cat([kr_b, k_nope_b], dim=-1).float() * ir_b).to(dtype).unsqueeze(0).unsqueeze(2).contiguous()
            v_b = F.linear(c_b, owner_mod._gla_v_up).unsqueeze(0).unsqueeze(2).contiguous()
            flash_attn_func(
                q_full[qs:qe].unsqueeze(0),
                k_b,
                v_b,
                softmax_scale=scale,
                causal=True,
                out=output[qs:qe].unsqueeze(0),
            )
        ext.gla_gate_only(output, qckr, output)
        return output

    return gla_impl_forward


def apply() -> None:
    ckpt_path = _gla_ckpt_path()
    if not ckpt_path:
        return

    import dataclasses
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    from vllm.utils.torch_utils import _encode_layer_name

    if getattr(Qwen3NextAttention, "_q27_gla_patched", False):
        return

    ckpt_cpu = _load_ckpt_cpu()
    scale_mult = _gla_scale_mult()
    mtp_on = _gla_mtp_enabled() and ("mtp0" in ckpt_cpu)
    r_tgt = int(ckpt_cpu["layers"][3]["kv_a"][0].shape[0])
    r_mtp = int(ckpt_cpu["mtp0"]["kv_a"][0].shape[0]) if mtp_on else 0
    _tag_aot_cache(f"q27fuse.gla/r{r_tgt}/mtp{r_mtp}/fq{int(_gla_fuse_qkv())}/v2")

    # Add fused qkv_proj shapes (3328 / 3392) to pf2.FP8_ALGO so long prefill also gets cuBLAS algo 6
    try:
        from . import pf2
        for n_col in (3072, 3328, 3392):
            pf2.FP8_ALGO.setdefault((5120, n_col), [(1536, 1 << 30, 0, 6)])
    except Exception:  # noqa: BLE001
        pass

    _install_qkv_splicing()

    orig_init = Qwen3NextAttention.__init__
    orig_fwd = Qwen3NextAttention.forward

    def __init__(self, config, model_config=None, cache_config=None, quant_config=None, prefix=""):
        orig_init(self, config, model_config, cache_config, quant_config, prefix)
        self._gla_active = False
        is_mtp = "mtp" in (prefix or "").lower()
        if is_mtp:
            if not mtp_on:
                return
            ld = ckpt_cpu.get("mtp0")
            tag = "mtp0"
        else:
            if self.layer_idx not in FULL_LAYER_SET:
                return
            ld = ckpt_cpu["layers"].get(self.layer_idx)
            tag = f"L{self.layer_idx}"
        if ld is None:
            return

        tp_rank = get_tensor_model_parallel_rank()
        kv_a_cpu = ld["kv_a"][tp_rank]
        r_g = int(kv_a_cpu.shape[0])
        if r_g not in (192, 256):
            # Lossless r_g == 448 layers stay untouched on native attn_mq.cu!
            return
        rk = [int(x) for x in ld["rope_keep"]]
        assert rk == list(range(31)), f"q27fuse.gla v2 fused kernels require rope_keep=0..30, got {rk[:5]}.. (len={len(rk)})"

        self.sm70_qk_rope_prep = False
        self._gla_active = True
        self._gla_tag = tag
        self._gla_rank = r_g
        self._gla_kv_scale = float(ld.get("kv_scale", 0.05)) * scale_mult
        self._gla_qkv_fused = False
        self._gla_rt_ready = False
        self._gla_encoded_name = _encode_layer_name(self.attn.layer_name)

        half_b = (r_g + 64) // 2
        perm = rope_perm(64, 256, rk)
        dev = self.k_norm.weight.device
        dtype = torch.float16

        # Build padded [kv_a (r_g, 5120); k_rope (62, 5120); zeros (2, 5120)] on CPU for QPN8 qkv_proj splicing
        kv_ar_pad_cpu = torch.cat([
            kv_a_cpu.to(device="cpu", dtype=torch.float16),
            ld["k_rope"][tp_rank].to(device="cpu", dtype=torch.float16),
            torch.zeros((2, 5120), dtype=torch.float16, device="cpu"),
        ], dim=0).contiguous()
        qkv_prefix = getattr(self.qkv_proj, "prefix", "") or f"{prefix}.qkv_proj"
        _QKV_SPECS[qkv_prefix] = (self, r_g, kv_ar_pad_cpu)

        # Keep fallback FP16 _gla_kv_ar until qkv_proj splicing releases it (or when Q27_GLA_FUSE_QKV=0)
        self.register_buffer("_gla_kv_ar", kv_ar_pad_cpu.to(device=dev, dtype=dtype), persistent=False)
        self.register_buffer("_gla_k_up_raw", ld["k_up_raw"][tp_rank].to(device=dev, dtype=dtype).contiguous(), persistent=False)
        self.register_buffer("_gla_v_up", ld["v_up"][tp_rank].to(device=dev, dtype=dtype).contiguous(), persistent=False)
        self.register_buffer("_gla_perm_t", torch.tensor(perm, dtype=torch.long, device=dev), persistent=False)

        _ACTIVE_OWNERS.append(self)

        # Override KV cache spec on this Attention instance -> compressed 256 B / 320 B paged cache!
        attn_mod = self.attn
        orig_get_spec = attn_mod.get_kv_cache_spec

        def _gla_get_kv_cache_spec(vllm_config):
            spec = orig_get_spec(vllm_config)
            if spec is None:
                return None
            return dataclasses.replace(spec, head_size=half_b, head_size_v=half_b)

        attn_mod.get_kv_cache_spec = _gla_get_kv_cache_spec
        attn_mod.impl.forward = _make_gla_impl_forward(self, attn_mod.impl.forward)

        if tp_rank == 0 and (tag == "L3" or tag == "mtp0"):
            comp_layers = [li for li in FULL_LAYERS if int(ckpt_cpu["layers"][li]["kv_a"][0].shape[0]) in (192, 256)]
            log.warning("q27fuse.gla: active on %s (r=%d, kv_scale=%.5f, target=%d/16, mtp0=%s, fuse_qkv=%s)",
                        tag, r_g, self._gla_kv_scale, len(comp_layers), mtp_on, _gla_fuse_qkv())

    def forward(self, positions: torch.Tensor, output: torch.Tensor | None, hidden_states: torch.Tensor):
        if not getattr(self, "_gla_active", False):
            return orig_fwd(self, positions, output, hidden_states)

        n = hidden_states.shape[0]
        qckr, _ = self.qkv_proj(hidden_states)
        if not getattr(self, "_gla_qkv_fused", False):
            ckr = F.linear(hidden_states, self._gla_kv_ar)
            qckr = torch.cat([qckr[:, :self.q_size * 2], ckr], dim=-1).contiguous()

        attn_out = torch.empty((n, self.q_size), dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops.vllm.unified_attention_with_output(
            qckr,
            positions,
            positions,
            attn_out.view(n, self.num_heads, self.head_dim),
            self._gla_encoded_name,
        )
        projected_output, _ = self.o_proj(attn_out)
        if output is not None:
            output[:] = projected_output
        return projected_output

    Qwen3NextAttention.__init__ = __init__
    Qwen3NextAttention.forward = forward
    Qwen3NextAttention._q27_gla_patched = True
    _install_prealloc()
    log.warning("q27fuse.gla: patched Qwen3NextAttention v2 fused (%s, mtp0=%s, fuse_qkv=%s)",
                os.path.basename(ckpt_path), mtp_on, _gla_fuse_qkv())
