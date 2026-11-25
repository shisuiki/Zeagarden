"""q27fuse.gdnv - MTP GDN layers on csrc/gdn_verify.cu, with our own replay records (format in the .cu header).

Under MTP 1Cat commits speculative GDN states by replay ("deferred commit", layers/mamba/gdn/spec_replay.py): a verify
round stores a record per token, the next step replays the accepted ones into the committed state. This module owns
that record format; 1Cat's bookkeeping (SpecReplayState: accepted / verified / side, the per-row tables spec_replay,
spec_map, plain_replay, plain_map, and the prefix-cache checkpoint token/slot) is used unchanged.

Routes (Qwen3_5GatedDeltaNet.forward_cuda runs eagerly inside 1Cat's opaque qwen_gdn_full_forward op under MTP, so the
route is chosen per step, i.e. per captured FULL graph):
  pure verify batch (every request verifies T = K+1 rows, no prefill / plain rows):
      2..32 rows: 1Cat's fused qkvz + b/a projection (apply_gdn_ba_verify) -> gdn_verify VERIFY -> out_proj
      > 32 rows:  in_proj_qkvz (FP8: ours / 1Cat's) -> gdn_verify VERIFY_BA (b/a in the kernel) -> out_proj
  pure draft-free decode batch (K = 0 steps): in_proj_qkvz -> gdn_decode.cu (gdn_core.cu's decode core with the
      commit folded in; Q27_GDNV_DECODE=gdnv: gdn_verify DECODE instead) -> out_proj
  anything else (prefill, mixed batches): 1Cat's forward, whose two replay consumers are replaced by ours:
      _commit_replayed_decodes (plain decode rows; three call sites in qwen_gdn_linear_attn.py) -> gdn_verify COMMIT
      verification_update (speculative rows; _forward_core and qwen_gdn_attention_core_spec_commit) -> VONLY
      _spec_replay_args hands them our buffer.
Nothing of 1Cat reads or writes a record any more: SpecReplayState.layer_buffer, gdn_verify.verify_bound(replay=...)
and commit_replayed raise if reached (1Cat's 204 MB/rank buffer is never allocated; ours is 2 x max_reqs x (K+1) x 4224
B per layer)."""
import functools
import logging
import os
import torch

log = logging.getLogger("q27fuse")

HV, HK, D, DIM, KH = 12, 4, 128, 2560, 5120
VERIFY, VERIFY_BA, DECODE, VONLY, COMMIT = range(5)
CFG_NAMES = ["8x1", "8x2", "8x4", "8x8", "8x16"]          # state columns per warp x warps per CTA
_E = None
_BUF = {}          # layer prefix -> uint8 record buffer
_WS = {}           # device -> zeroed int32 counters
_EMPTY = {}
HITS = {}          # route -> calls (logged on the first)


def ext():
    global _E
    if _E is None:
        from . import ops
        defs = os.environ.get("Q27_GDNV_DEFS", "")      # bench variants only (e.g. -DGDNV_TOKEN_UNROLL=0)
        if defs:
            from torch.utils.cpp_extension import load
            here = os.path.dirname(os.path.abspath(__file__))
            tag = "".join(ch for ch in defs if ch.isalnum())[-24:]
            build = ops.default_build_dir("_gdnv_" + tag)
            os.makedirs(build, exist_ok=True)
            _E = load(name="q27fuse_gdnv_" + tag, sources=[os.path.join(here, "csrc", "gdn_verify.cu")],
                      build_directory=build, extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                                                                "-std=c++17", "--use_fast_math"] + defs.split())
        else:
            _E = ops._load("q27fuse_gdnv", "gdn_verify.cu", "_gdnv")
    return _E


_DX = None


def dext():
    """gdn_decode.cu: gdn_core.cu's fused decode core with the replay of our records folded in"""
    global _DX
    if _DX is None:
        from . import ops
        _DX = ops._load("q27fuse_gdndec", "gdn_decode.cu", "_gdndec")
    return _DX


DECODE_KERNEL = os.environ.get("Q27_GDNV_DECODE", "core")         # core: gdn_decode.cu; gdnv: gdn_verify DECODE

# [ckpt_fix] optional fix of 1Cat's checkpoint-into-the-running-slot case (OFF by default; notes/gdnv.md "1Cat issue").
# When a verify round's LAST token completes a Mamba block (computed + T = multiple of the block), 1Cat's bookkeeping
# points the prefix-cache checkpoint at the running state slot; the speculative state after T tokens then replaces the
# committed one and the next commit replays the accepted tokens on top of it (applied twice). The fix, in the align
# bookkeeping (a wrapper of SpecReplayState.prepare): (1) such a round gets no checkpoint (ckpt token -1), so the
# running slot keeps the committed state; (2) the step whose commit ends exactly on a block boundary (computed % block
# == 0 with tokens to replay) gets a "commit checkpoint" slot = block_table[computed / block - 1]; the kernels store the
# committed state there as well (gdn_verify.cu / gdn_decode.cu `ccs`). Q27_GDNV_CKPT_FIX (p15: default 1).
# [merge] default ON in p15 (Q27_GDNV_CKPT_FIX=0 = 1Cat's behaviour): the corruption is demonstrated in the server
# (notes/merge.md item 4), cost +0.02-0.03 ms per step (one eager Triton launch in SpecReplayState.prepare)
CKPT_FIX = (os.environ.get("Q27_GDNV_CKPT_FIX", "") or "1") == "1"
# [merge] Q27_GDNV_FUSED_MAX: largest pure-verify row count that takes the projection's own a/b (apply_gdn_ba_verify, i.e.
# smallrows' fp8_gdn_in.cu at 17..64 rows / 1Cat's fused op at 2..16) + VERIFY; above it in_proj_qkvz + VERIFY_BA (default 32
# = the gdnv track's routing; p15 default 64: notes/merge.md section 3a, C32 -0.32 ms)
FUSED_MAX = int(os.environ.get("Q27_GDNV_FUSED_MAX", "") or "64")


def rec_bytes() -> int:
    return int(ext().REC)


# kernel configuration (cfg index into CFG_NAMES) by (mode, requests B, tokens T); test_gdn_verify.py bench
# (bench 2026-11-22, rolled loops, results/r1/gdnv/t5_unroll.log + t4_v4.log: B=1 -> 8x4; 2 -> 8x8; 4..16 -> 8x4; b/a in the
# kernel at 32 x 2 -> 8x16; decode 48, 64 -> 8x4)
VERIFY_CFG = [(1, 2), (2, 3), (10 ** 9, 2)]                         # (max B, cfg)
VERIFY_BA_CFG = [(10 ** 9, 4)]                                       # > 32 rows (b/a operands: >= 8 warps)
DECODE_CFG = [(10 ** 9, 2)]                                          # (b/a operands: >= 4 warps)
VERIFY_TM = {1: 2, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 8, 8: 8}        # tokens per request -> template capacity
BA_TM = {1: 2, 2: 2, 3: 4, 4: 4, 5: 5}
BA_CFGS = (3, 4)


def cfg_for(table, B: int) -> int:
    return next(c for m, c in table if B <= m)


def _empty(dev, dtype):
    k = (dev, dtype)
    t = _EMPTY.get(k)
    if t is None:
        t = _EMPTY[k] = torch.empty(0, dtype=dtype, device=dev)
    return t


def workspace(dev) -> torch.Tensor:
    """zeroed int32 counters for the cross-CTA hand-offs (the kernel leaves them zero): [B HK] + [B HV] for B <= 4096"""
    w = _WS.get(dev)
    if w is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("q27fuse gdnv: counter workspace first needed inside a CUDA graph capture")
        w = _WS[dev] = torch.zeros(4096 * (HK + HV), dtype=torch.int32, device=dev)
    return w


def records(layer) -> torch.Tensor:
    """our record buffer of a GDN layer: [2 max_reqs][K+1][REC] bytes, zeroed once (allocated outside any capture)"""
    buf = _BUF.get(layer.prefix)
    if buf is None:
        from vllm.model_executor.layers.mamba.gdn.spec_replay import get_spec_replay_state
        st = get_spec_replay_state()
        assert st is not None, "GDN speculative replay state is not set"
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"q27fuse gdnv: record buffer of {layer.prefix} first needed inside a CUDA graph capture")
        assert layer._replay_stride <= 8
        buf = _BUF[layer.prefix] = torch.zeros(2 * st.max_reqs * layer._replay_stride * rec_bytes(), dtype=torch.uint8,
                                               device=layer.A_log.device)
        if len(_BUF) == 1:
            log.warning("q27fuse gdnv: replay records %.2f MB per GDN layer (max_reqs %d, stride %d)",
                        buf.numel() / 2 ** 20, st.max_reqs, layer._replay_stride)
    return buf


def _hit(route, layer, n):
    c = HITS.get(route, 0)
    HITS[route] = c + 1
    if c == 0:
        log.warning("q27fuse gdnv: %s route first hit (%s, %d rows)", route, layer.prefix, n)


def run(mode, tm, cfg, layer, *, ssm, idx, B, T=0, n=0, qkv=None, z=None, a=None, b=None, x=None, conv=None, cu=None,
        acc=None, rn=None, rmap=None, ckt=None, cks=None, core=None, y=None, rbuf=None, stride=None, dbg=0, ccs=None):
    """one gdn_verify launch with a layer's parameters (tensors a mode does not use may be None)"""
    dev = ssm.device
    eh, ei = _empty(dev, torch.float16), _empty(dev, torch.int32)
    cw = layer.conv1d.weight
    ext().gdnv(mode, tm, cfg, qkv if qkv is not None else eh, z if z is not None else eh, a if a is not None else eh,
               b if b is not None else eh, x if x is not None else eh,
               layer.in_proj_ba.weight if mode in (VERIFY_BA, DECODE) else eh,
               cw.view(cw.size(0), cw.size(-1)) if conv is not None else eh, conv if conv is not None else eh,
               layer.A_log, layer.dt_bias, ssm, idx, cu, acc, rn, rmap, ckt, cks,
               rbuf if rbuf is not None else records(layer), stride if stride is not None else layer._replay_stride,
               layer.norm.weight if y is not None else eh, float(layer.norm.eps), core if core is not None else eh,
               y if y is not None else eh, workspace(dev), B, T, n, dbg, ccs)


# ------------------------------------------------------------------------------------------------ 1Cat consumers -> ours
def commit_replayed_decodes(self, ssm_state, attn_metadata, state_indices=None) -> None:
    """QwenGatedDeltaNetAttention._commit_replayed_decodes on our records (COMMIT: replay rn tokens, store once;
    requests with nothing to replay are not touched)"""
    from vllm.model_executor.layers.mamba.gdn.spec_replay import get_spec_replay_state
    st = get_spec_replay_state()
    assert st is not None, "GDN speculative replay state is not set"
    rows = int(attn_metadata.num_decodes)
    if rows <= 0:
        return
    idx = (attn_metadata.non_spec_state_indices_tensor if state_indices is None else state_indices)[:rows]
    _hit("commit (1Cat route, plain decode rows)", self, rows)
    run(COMMIT, 1, 4, self, ssm=ssm_state, idx=idx, B=rows, rn=st.plain_replay, rmap=st.plain_map,
        ccs=commit_ckpt(st, self, "plain"))


def spec_replay_args(self):
    """QwenGatedDeltaNetAttention._spec_replay_args with our buffer (consumed only by verify_update below)"""
    from vllm.model_executor.layers.mamba.gdn.spec_replay import get_spec_replay_state
    st = get_spec_replay_state()
    assert st is not None, "GDN speculative replay state is not set"
    group = st.group_of[self.prefix]
    return (records(self), st.spec_replay, st.spec_map, st.spec_ckpt_token, st.spec_ckpt_slot[group],
            self._replay_stride)


def verify_update(layer, A_log, dt_bias, requests, mixed_qkv, a, b, state, out, num_actual_tokens, direct_out,
                  query_start, state_indices, slot_selectors, replay=None):
    """layer.verification_update (1Cat's verify_bound contract) on our records: VONLY (convolved qkv in, raw scaled
    core out, replay commit, records, checkpoint)"""
    tokens = mixed_qkv.shape[0]
    if replay is None:
        raise RuntimeError("q27fuse gdnv: verification without replay records is not supported")
    rbuf, rn, rmap, ckt, cks, stride = replay
    if not (requests >= 1 and tokens <= requests * 8 and mixed_qkv.dtype == torch.float16
            and mixed_qkv.stride(1) == 1 and state.dtype == torch.float32 and a.stride(1) == 1 and b.stride(1) == 1
            and a.stride(0) == b.stride(0)):
        return None
    verify_out = (out[:num_actual_tokens].unsqueeze(0) if direct_out
                  else mixed_qkv.new_empty((1, tokens, HV, D)))
    idx = state_indices[:, 0] if state_indices.dim() == 2 else state_indices
    _hit("verify-only (1Cat route, speculative rows)", layer, tokens)
    from vllm.model_executor.layers.mamba.gdn.spec_replay import get_spec_replay_state
    run(VONLY, 8, 4, layer, ssm=state, idx=idx, B=int(requests), T=8, n=tokens, qkv=mixed_qkv, a=a, b=b,
        cu=query_start[:requests + 1], rn=rn, rmap=rmap, ckt=ckt, cks=cks, core=verify_out.view(-1, HV, D),
        rbuf=rbuf, stride=stride, ccs=commit_ckpt(get_spec_replay_state(), layer, "spec"))
    return verify_out, state


# ------------------------------------------------------------------------------------------------------- our routes
def _layer_ok(layer) -> str | None:
    """why gdn_verify cannot serve this layer (None = it can); static shapes, checked once"""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    kv = getattr(layer, "kv_cache", None)
    if kv is None or kv[0].numel() == 0:
        return "no cache yet"
    conv, ssm = kv[0], kv[1]
    cw = layer.conv1d.weight
    if is_conv_state_dim_first():
        return "conv state layout DS"
    if not (layer.num_k_heads // layer.tp_size == HK and layer.num_v_heads // layer.tp_size == HV
            and layer.head_k_dim == layer.head_v_dim == D and layer._q27_qkv == DIM):
        return "heads"
    if not (cw.dtype == torch.float16 and cw.is_contiguous() and tuple(cw.shape) == (DIM, 1, 4)
            and layer.conv1d.bias is None and layer.activation in ("silu", "swish")):
        return "conv weight"
    if not (conv.dtype == torch.float16 and conv.dim() == 3 and conv.size(2) == DIM and conv.stride(2) == 1
            and conv.stride(1) == DIM and conv.stride(0) % 4 == 0 and conv.size(1) >= layer._replay_stride + 2):
        return f"conv state {tuple(conv.shape)} {conv.stride()}"
    if not (ssm.dtype == torch.float32 and ssm.dim() == 4 and tuple(ssm.shape[1:]) == (HV, D, D)
            and ssm.stride(1) == D * D and ssm.stride(2) == D and ssm.stride(3) == 1 and ssm.stride(0) % 4 == 0):
        return f"ssm state {tuple(ssm.shape)} {ssm.stride()}"
    if not (layer.A_log.dtype == torch.float32 and layer.dt_bias.dtype == torch.float16
            and layer.norm.weight.dtype == torch.float16 and layer.norm.weight.numel() == D
            and layer.in_proj_ba.weight.dtype == torch.float16 and layer.in_proj_ba.weight.is_contiguous()
            and tuple(layer.in_proj_ba.weight.shape) == (2 * HV, KH)):
        return "parameter dtypes/shapes"
    if not (1 <= layer._replay_stride <= 8):
        return "replay stride"
    return None


_NO = object()


def route(layer, hidden_states, output):
    """our MTP route for this step, or _NO (the caller then runs the previous forward)"""
    from vllm.forward_context import get_forward_context
    if torch.compiler.is_compiling():
        return _NO
    fc = get_forward_context()
    amd = fc.attn_metadata
    md = amd.get(layer.prefix) if isinstance(amd, dict) else None
    if md is None:
        return _NO
    why = getattr(layer, "_q27v_why", "unset")
    if why == "unset" or why == "no cache yet":
        why = layer._q27v_why = _layer_ok(layer)
        if why is not None and why != "no cache yet":
            log.warning("q27fuse gdnv: not used for %s (%s)", layer.prefix, why)
    if why is not None:
        return _NO
    x = hidden_states
    n = x.size(0)
    if not (x.dtype == torch.float16 and x.is_contiguous() and x.dim() == 2 and x.size(1) == KH and n > 0):
        return _NO
    B = int(md.num_spec_decodes)
    if (B > 0 and md.num_prefills == 0 and md.num_decodes == 0 and md.spec_sequence_masks is not None
            and getattr(md, "ddtree_parent_ids", None) is None and int(md.num_actual_tokens) == n and n % B == 0
            and 1 <= n // B <= 8 and md.spec_query_start_loc is not None and md.spec_query_start_loc.dim() == 1
            and md.spec_query_start_loc.numel() >= B + 1 and md.spec_state_indices_tensor is not None
            and md.spec_state_indices_tensor.dim() == 2 and md.spec_state_indices_tensor.shape[0] >= B
            and md.spec_state_slot_selectors is not None and md.spec_state_slot_selectors.numel() >= B):
        # rows per request come from spec_query_start_loc (device); a FULL graph (or a batch the dispatcher marks
        # uniform) has at most n / B rows per request, otherwise size the kernel for the longest possible round
        from vllm.config import CUDAGraphMode
        # (the GDN layer is a graph-splitting op, so a capture in progress here is a FULL graph: uniform by dispatch)
        bd = getattr(fc, "batch_descriptor", None)
        uniform = (torch.cuda.is_current_stream_capturing()
                   or getattr(fc, "cudagraph_runtime_mode", None) == CUDAGraphMode.FULL
                   or bool(bd is not None and getattr(bd, "uniform", False)))
        cap = n // B if uniform else max(n // B, int(layer._replay_stride))
        return _verify(layer, x, output, md, n, B, n // B, cap)
    if (B == 0 and md.spec_sequence_masks is None and md.num_prefills == 0 and 0 < md.num_decodes <= n
            and md.non_spec_state_indices_tensor is not None):
        return _decode(layer, x, output, md, n)
    return _NO


def _finish(layer, y, output, n):
    """out_proj of the core output; written straight into the decoder layer's buffer when 1Cat's own direct-output
    test (QwenGatedDeltaNetAttention._output_projection) admits it, else projected and copied"""
    if output is not None:
        op = layer.out_proj
        target = getattr(op, "scheme", None) or getattr(op, "quant_method", None)
        sup = getattr(target, "supports_direct_output", None)
        if (output.shape == (n, layer.hidden_size) and output.dtype == torch.float16 and output.is_contiguous()
                and getattr(layer, "output_projection_override", None) is None and op.bias is None
                and not op.reduce_results and callable(sup) and sup(op)):
            return target.apply_weights(op, y, output=output)
    proj, _ = layer.out_proj(y)
    if output is not None:
        output[:n] = proj
    return proj


def _verify(layer, x, output, md, n, B, T, cap):
    from vllm.model_executor.layers.mamba.gdn.spec_replay import get_spec_replay_state
    from vllm.model_executor.layers.quantization.sm70_gdn_ba_verify import apply_gdn_ba_verify
    st = get_spec_replay_state()
    kv = layer.kv_cache
    group = st.group_of[layer.prefix]
    fused = apply_gdn_ba_verify(layer, x) if 2 <= n <= FUSED_MAX else None          # [merge] FUSED_MAX (was 32)
    a = b = xin = None
    if fused is not None:                                  # one 1Cat launch: qkv | z | b | a
        qkv, z, b, a = fused
        mode, tm = VERIFY, VERIFY_TM[cap]
        _hit(f"verify (1Cat fused in-proj, T={T}, capacity {tm})", layer, n)
    else:
        qkvz, _ = layer.in_proj_qkvz(x)
        qkv, z = qkvz, qkvz[:, DIM:]
        if cap in BA_TM:
            mode, tm, xin = VERIFY_BA, BA_TM[cap], x
            _hit(f"verify (b/a in the kernel, T={T}, capacity {tm})", layer, n)
        else:                                              # T > 4 above 32 rows: b/a through cuBLAS
            ba = torch.nn.functional.linear(x, layer.in_proj_ba.weight)
            b, a = ba[:, :HV], ba[:, HV:]
            mode, tm = VERIFY, VERIFY_TM[cap]
            _hit(f"verify (cuBLAS b/a, T={T}, capacity {tm})", layer, n)
    y = x.new_empty((n, HV * D))
    core = x.new_empty((n, HV, D))
    cfg = cfg_for(VERIFY_BA_CFG if mode == VERIFY_BA else VERIFY_CFG, B)
    run(mode, tm, cfg, layer, ssm=kv[1], idx=md.spec_state_indices_tensor[:, 0], B=B, T=T, n=n,
        qkv=qkv, z=z, a=a, b=b, x=xin, conv=kv[0], cu=md.spec_query_start_loc, acc=md.spec_state_slot_selectors,
        rn=st.spec_replay, rmap=st.spec_map, ckt=st.spec_ckpt_token, cks=st.spec_ckpt_slot[group], core=core, y=y,
        ccs=commit_ckpt(st, layer, "spec"))
    return _finish(layer, y, output, n)


def _decode(layer, x, output, md, n):
    from vllm.model_executor.layers.mamba.gdn.spec_replay import get_spec_replay_state
    st = get_spec_replay_state()
    kv = layer.kv_cache
    B = int(md.num_decodes)
    qkvz, _ = layer.in_proj_qkvz(x)
    y = x.new_empty((n, HV * D))
    core = x.new_empty((n, HV, D))
    if DECODE_KERNEL == "core" and n <= 4096 and _core_ok(layer, x, qkvz, kv):
        _hit("draft-free decode (gdn_decode.cu: gdn_core + commit folded in)", layer, n)
        decode_core(layer, x, qkvz, kv[0], kv[1], md.non_spec_state_indices_tensor, B, core, y, st.plain_replay,
                    st.plain_map, ccs=commit_ckpt(st, layer, "plain"))
    else:
        _hit("draft-free decode (gdn_verify DECODE, commit folded in)", layer, n)
        run(DECODE, 1, cfg_for(DECODE_CFG, n), layer, ssm=kv[1], idx=md.non_spec_state_indices_tensor, B=B, T=1, n=n,
            qkv=qkvz, z=qkvz[:, DIM:], x=x, conv=kv[0], rn=st.plain_replay, rmap=st.plain_map, core=core, y=y,
            ccs=commit_ckpt(st, layer, "plain"))
    return _finish(layer, y, output, n)


def _core_ok(layer, x, qkvz, kv) -> bool:
    """gdn_core.cu's own layout test (ops._gdn_core_ok), once per layer"""
    ok = getattr(layer, "_q27v_core_ok", None)
    if ok is None:
        from . import ops
        why = ops._gdn_core_ok(layer, x, qkvz, kv)
        ok = layer._q27v_core_ok = why is None
        if why is not None:
            log.warning("q27fuse gdnv: gdn_decode.cu not used for %s (%s)", layer.prefix, why)
    return ok


def decode_core(layer, x, qkvz, conv, ssm, idx, B, core, y, rn, rmap, rbuf=None, stride=None, cfg=None, dbg=0,
                ccs=None):
    """one gdn_decode.cu launch: replay rn[t] records of side rmap[t] ^ 1, then gdn_core.cu's decode step (gdn_core.cu's
    configuration per batch size)"""
    from . import ops
    cw = layer.conv1d.weight
    dext().gdn_decode(x, qkvz, layer.in_proj_ba.weight, cw.view(cw.size(0), 4), conv, layer.A_log, layer.dt_bias,
                      ssm, idx, int(B), layer.norm.weight, float(layer.norm.eps), core, y, workspace(x.device),
                      ops.gdn_core_cfg(x.size(0)) if cfg is None else cfg, dbg, rn, rmap,
                      rbuf if rbuf is not None else records(layer),
                      stride if stride is not None else layer._replay_stride, ccs)


# ------------------------------------------------------------------------------- [ckpt_fix] align bookkeeping fix
def commit_ckpt(st, layer, kind):
    """this step's commit checkpoint slots of the layer's group ('spec' / 'plain' rows), or None when the fix is off"""
    if not CKPT_FIX or st is None or not hasattr(st, "q27_spec_ccs"):
        return None
    g = st.group_of.get(layer.prefix, 0)
    return (st.q27_spec_ccs if kind == "spec" else st.q27_plain_ccs)[g]


_CKK = None


def _ckfix_kernel():
    global _CKK
    if _CKK is None:
        from vllm.triton_utils import tl, triton

        @triton.jit
        def ckfix(rows_ptr, idx_mapping_ptr, qsl_ptr, num_computed_ptr, spec_replay_ptr, plain_replay_ptr,
                  spec_ckpt_token_ptr, spec_ccs_ptr, plain_ccs_ptr, block_table_ptrs_ptr, block_table_stride,
                  num_spec, num_plain, MAX_ROWS: tl.constexpr, NUM_GROUPS: tl.constexpr,
                  MAMBA_BLOCK_SIZE: tl.constexpr, BLOCK: tl.constexpr):
            which = tl.program_id(0)
            offsets = tl.arange(0, BLOCK)
            in_table = offsets < MAX_ROWS
            if which == 0:                    # speculative rows (1Cat's spec row tables, same order)
                live = offsets < num_spec
                rows = tl.load(rows_ptr + offsets, mask=live, other=0)
                start = tl.load(qsl_ptr + rows, mask=live, other=0)
                end = tl.load(qsl_ptr + rows + 1, mask=live, other=0)
                ckt = tl.load(spec_ckpt_token_ptr + offsets, mask=live, other=-1)
                # (1) the round's last token completes the block: its checkpoint column IS the running column
                tl.store(spec_ckpt_token_ptr + offsets, -1, mask=live & (ckt >= 0) & (ckt == end - start - 1))
                rn = tl.load(spec_replay_ptr + offsets, mask=live, other=0)
                req = tl.load(idx_mapping_ptr + rows, mask=live, other=0)
                computed = tl.load(num_computed_ptr + req, mask=live, other=0)
                # (2) this step's commit ends exactly on a block boundary: checkpoint the committed state
                has = live & (rn > 0) & (computed > 0) & (computed % MAMBA_BLOCK_SIZE == 0)
                column = computed // MAMBA_BLOCK_SIZE - 1
                for group in tl.static_range(NUM_GROUPS):
                    table = tl.load(block_table_ptrs_ptr + group).to(tl.pointer_type(tl.int32))
                    slot = tl.load(table + rows * block_table_stride + column, mask=has, other=-1)
                    tl.store(spec_ccs_ptr + group * MAX_ROWS + offsets, tl.where(has, slot, -1), mask=in_table)
            else:                             # plain rows (commit only)
                live = offsets < num_plain
                rows = tl.load(rows_ptr + MAX_ROWS + offsets, mask=live, other=0)
                rn = tl.load(plain_replay_ptr + offsets, mask=live, other=0)
                req = tl.load(idx_mapping_ptr + rows, mask=live, other=0)
                computed = tl.load(num_computed_ptr + req, mask=live, other=0)
                has = live & (rn > 0) & (computed > 0) & (computed % MAMBA_BLOCK_SIZE == 0)
                column = computed // MAMBA_BLOCK_SIZE - 1
                for group in tl.static_range(NUM_GROUPS):
                    table = tl.load(block_table_ptrs_ptr + group).to(tl.pointer_type(tl.int32))
                    slot = tl.load(table + rows * block_table_stride + column, mask=has, other=-1)
                    tl.store(plain_ccs_ptr + group * MAX_ROWS + offsets, tl.where(has, slot, -1), mask=in_table)

        _CKK = (ckfix, triton)
    return _CKK


def install_ckpt_fix():
    """wrap SpecReplayState: commit-checkpoint tables allocated with the state (before any capture), filled after
    1Cat's own row tables in every prepare (eager, before the forward) while CKPT_FIX is set"""
    import numpy as np
    from vllm.model_executor.layers.mamba.gdn import spec_replay as SR
    S = SR.SpecReplayState
    if getattr(S, "_q27_ckfix", False):
        return
    prev_init, prev_prepare = S.__init__, S.prepare

    def __init__(self, max_reqs, max_rows, num_groups, device):
        prev_init(self, max_reqs, max_rows, num_groups, device)
        self.q27_spec_ccs = torch.full((num_groups, max_rows), -1, dtype=torch.int32, device=device)
        self.q27_plain_ccs = torch.full((num_groups, max_rows), -1, dtype=torch.int32, device=device)

    def prepare(self, idx_mapping, query_start_loc, num_computed_tokens, num_draft_tokens_per_req, is_prefilling,
                num_reqs, block_table_ptrs, block_table_stride, mamba_block_size):
        prev_prepare(self, idx_mapping, query_start_loc, num_computed_tokens, num_draft_tokens_per_req, is_prefilling,
                     num_reqs, block_table_ptrs, block_table_stride, mamba_block_size)
        if not CKPT_FIX or num_reqs == 0:
            return
        if not hasattr(self, "q27_spec_ccs"):           # state built before install_ckpt_fix: allocate now (eager)
            log.warning("q27fuse gdnv: checkpoint-fix tables allocated late (graphs captured before have no ccs)")
            self.q27_spec_ccs = torch.full((self.num_groups, self.max_rows), -1, dtype=torch.int32,
                                           device=self.spec_replay.device)
            self.q27_plain_ccs = torch.full_like(self.q27_spec_ccs, -1)
        drafts = (np.zeros(num_reqs, dtype=np.int64) if num_draft_tokens_per_req is None
                  else np.asarray(num_draft_tokens_per_req[:num_reqs]))
        spec = drafts > 0                                   # the same split as 1Cat's prepare
        if is_prefilling is not None:
            spec &= ~np.asarray(is_prefilling[:num_reqs], dtype=bool)
        num_spec = int(spec.sum())
        k, triton = _ckfix_kernel()
        k[(2,)](self._rows, idx_mapping, query_start_loc, num_computed_tokens, self.spec_replay, self.plain_replay,
                self.spec_ckpt_token, self.q27_spec_ccs, self.q27_plain_ccs, block_table_ptrs, block_table_stride,
                num_spec, num_reqs - num_spec, MAX_ROWS=self.max_rows, NUM_GROUPS=block_table_ptrs.numel(),
                MAMBA_BLOCK_SIZE=mamba_block_size, BLOCK=triton.next_power_of_2(max(self.max_rows, 16)))

    S.__init__, S.prepare = __init__, prepare
    S._q27_ckfix = True
    log.warning("q27fuse gdnv: checkpoint fix ON (Q27_GDNV_CKPT_FIX=1): no prefix-cache checkpoint into the running "
                "state slot; deferred checkpoint at the commit that ends on a block boundary")


# ----------------------------------------------------------------------------------------------------------- install
def _raise(what):
    def f(*args, **kwargs):
        raise RuntimeError(f"q27fuse gdnv: {what} reached; 1Cat's replay records are not kept (our format replaces "
                           f"them), so this path would read records of the other implementation")
    return f


def apply():
    """Install after patch.apply_gdn (wraps its Qwen3_5GatedDeltaNet.__init__ / forward_cuda)."""
    from vllm.model_executor.models import qwen3_5
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as QG, spec_replay as SR
    from vllm.model_executor.layers.fla.ops.sm70 import gdn_verify as GV
    A = qwen3_5.Qwen3_5GatedDeltaNet
    if getattr(A, "_q27v_patched", False):
        return
    # [merge] install_ckpt_fix() is called from __init__.register() AFTER tail.apply(): tail's "rows" piece replaces
    # SpecReplayState.prepare without calling the previous one, so a fix installed here would be dropped
    # 1Cat's replay consumers -> ours (class methods looked up per call)
    QG.QwenGatedDeltaNetAttention._commit_replayed_decodes = commit_replayed_decodes
    QG.QwenGatedDeltaNetAttention._spec_replay_args = spec_replay_args
    # guards: nothing may reach 1Cat's record format any more
    SR.SpecReplayState.layer_buffer = _raise("SpecReplayState.layer_buffer (1Cat's record buffer)")
    orig_vb = GV.verify_bound

    def verify_bound_guard(contract, *args, replay=None, **kwargs):
        if replay is not None:
            _raise("1Cat's verify_bound with replay records")()
        return orig_vb(contract, *args, **kwargs)

    GV.verify_bound = verify_bound_guard            # bind_native_verifier partials made after this point
    GV.commit_replayed = _raise("1Cat's commit_replayed")
    QG.commit_replayed = _raise("1Cat's commit_replayed")

    prev_init, prev_fc = A.__init__, A.forward_cuda

    def __init__(self, *args, **kwargs):
        prev_init(self, *args, **kwargs)
        self._q27_v = bool(getattr(self, "_q27_spec", False) and getattr(self, "_spec_replay", False))
        if getattr(self, "_spec_replay", False):
            # every GDN layer under replay verifies on our records, whatever route the step takes
            self.verification_update = functools.partial(verify_update, self)
        if self._q27_v and not getattr(A, "_q27v_logged", False):
            A._q27v_logged = True
            log.warning("q27fuse gdnv: MTP GDN verify/decode core with our replay records on %s", self.prefix)

    def forward_cuda(self, hidden_states, output):
        if (self._q27_v and getattr(self, "_q27_gdn", False) and self.in_proj_ba.weight.dtype == torch.float16
                and getattr(self, "input_projection_override", None) is None
                and getattr(self, "output_projection_override", None) is None):
            r = route(self, hidden_states, output)
            if r is not _NO:
                return r
        return prev_fc(self, hidden_states, output)

    A.__init__ = __init__
    A.forward_cuda = forward_cuda
    A._q27v_patched = True
