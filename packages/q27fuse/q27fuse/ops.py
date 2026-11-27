"""q27fuse.ops - extension loading and custom ops: NVFP4 MLP (q27_mlp, q27_mlp_local), FP8 projections
(q27_fp8_linear), GDN decode core (q27_gdn_core + glue kernels), and split-KV decode attention."""
import logging
import os
import torch

log = logging.getLogger("q27fuse")


def default_cache_dir(sub: str) -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "q27fuse", sub)


def default_build_dir(suffix: str = "") -> str:
    base = os.environ.get("Q27_FUSE_BUILD", "").strip() or default_cache_dir("build")
    return base + suffix


def _load(name: str, src: str, suffix: str, ldflags=(), std17=True):
    """Build (once, into $Q27_FUSE_BUILD<suffix>) and import csrc/<src>."""
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    build = default_build_dir(suffix)
    os.makedirs(build, exist_ok=True)
    flags = ["-O3", "-gencode=arch=compute_70,code=sm_70"] + (["-std=c++17"] if std17 else []) + ["--use_fast_math"]
    csrc = os.environ.get("Q27_CSRC_DIR", os.path.join(here, "csrc"))
    return load(name=name, sources=[os.path.join(csrc, src)], build_directory=build, extra_cuda_cflags=flags,
                extra_ldflags=list(ldflags), verbose=False)


# ------------------------------------------------------------------------------------------------ GDN glue (gdn_glue.cu)
_G = None
GDN_IN_MAXM = 64  # decode batches; larger (prefill) batches keep cuBLAS for the b/a projection


def gdn_ext():
    """gdn_glue.cu: b/a projection + core zero-fill (gdn_in), gated RMSNorm reading z in place (gdn_gated_norm)."""
    global _G
    if _G is None:
        _G = _load("q27fuse_gdn", "gdn_glue.cu", "_gdn", std17=False)
    return _G


def gdn_in_impl(x: torch.Tensor, w: torch.Tensor, heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """b, a = split(x @ w.T) as contiguous [M, heads] each, and a zeroed core output [M, heads, 128]."""
    m = x.shape[0]
    b = x.new_empty((m, heads))
    a = x.new_empty((m, heads))
    core = x.new_empty((m, heads, 128))
    if m == 0:
        return b, a, core
    if m > GDN_IN_MAXM or not x.is_contiguous():
        ba = torch.nn.functional.linear(x, w)
        b.copy_(ba[:, :heads])
        a.copy_(ba[:, heads:])
        core.zero_()
        return b, a, core
    gdn_ext().gdn_in(x, w, b, a, core)
    return b, a, core


def gdn_gated_norm_impl(core: torch.Tensor, qkvz: torch.Tensor, weight: torch.Tensor, eps: float,
                        zoff: int) -> torch.Tensor:
    """RMSNormGated over each 128-wide head of core, gated by silu(z), z = qkvz[:, zoff:zoff + heads*128]."""
    m, heads, d = core.shape
    y = core.new_empty((m, heads * d))
    if m == 0:
        return y
    gdn_ext().gdn_gated_norm(core.contiguous(), qkvz, weight, y, eps, zoff)
    return y


# ------------------------------------------------------------------------------------------- GDN decode core (gdn_core.cu)
# Everything between a GDN layer's qkvz projection and its output projection is one op, q27_gdn_core, registered as a
# graph-splitting op (like 1Cat's GDN core ops) so it sees each batch's metadata: pure-decode batches run gdn_core.cu
# (b/a projection, conv update, state update, gated norm in one launch); anything else runs the chain gdn_in ->
# 1Cat's recurrent core -> gdn_gated_norm.
GDN_CORE_MAXN = 256
_GC = None
_GC_WS = {}
_GC_LOGGED = set()


def note_alloc(name: str, t: torch.Tensor) -> torch.Tensor:
    """[head] Job 0: a persistent workspace allocated inside a CUDA-graph capture comes out of the graph memory pool and
    may alias other graphs' intermediates (the attnv root cause); log it loudly so a new lazy buffer cannot hide."""
    if torch.cuda.is_current_stream_capturing():
        log.error("q27fuse: persistent workspace %s (%.1f MB on %s) allocated INSIDE a CUDA graph capture",
                  name, t.numel() * t.element_size() / 2 ** 20, t.device)
    return t


def gdn_core_ext():
    """gdn_core.cu: the GDN decode core (b/a projection, conv update, state update, gated norm) in one launch."""
    global _GC
    if _GC is None:
        _GC = _load("q27fuse_gdncore", "gdn_core.cu", "_gdncore")
    return _GC


# gdn_core.cu config per decode batch: [(max tokens, cfg), ...] (cfg = gdn_core dispatch: columns/warp x warps/CTA),
# from test_gdn_core.py bench; the last entry covers the rest.
GDN_CORE_CFG = [(1, 6), (32, 0), (10 ** 9, 7)]


def gdn_core_cfg(t: int) -> int:
    return next(c for m, c in GDN_CORE_CFG if t <= m)


def gdn_core_workspace(dev, n: int) -> torch.Tensor:
    """zeroed int32 counters (the kernel leaves them zero); one buffer per device, grown never shrunk"""
    w = _GC_WS.get(dev)
    if w is None or w.numel() < n * 16:
        w = note_alloc("gdn_core counters", torch.zeros(max(n, 256) * 16, dtype=torch.int32, device=dev))
        _GC_WS[dev] = w
    return w


def _gdn_core_ok(layer, x, qkvz, kv) -> str | None:
    """why the fused core cannot serve this layer (None = it can); checked once per layer, shapes are static"""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    heads, qkv = layer._q27_heads, layer._q27_qkv
    cw = layer.conv1d.weight
    conv, ssm = kv
    if is_conv_state_dim_first():
        return "conv state layout DS"
    if not (cw.dtype == torch.float16 and cw.dim() == 3 and cw.size(1) == 1 and cw.size(2) == 4 and cw.is_contiguous()
            and cw.size(0) == qkv):
        return f"conv weight {tuple(cw.shape)} {cw.dtype}"
    if not (conv.dtype == torch.float16 and conv.dim() == 3 and conv.size(1) >= 3 and conv.size(2) == qkv
            and conv.stride(2) == 1 and conv.stride(1) == qkv and conv.stride(0) % 4 == 0):
        return f"conv state {tuple(conv.shape)} {conv.stride()} {conv.dtype}"
    if not (ssm.dtype == torch.float32 and ssm.dim() == 4 and ssm.size(1) == heads and ssm.size(2) == 128
            and ssm.size(3) == 128 and ssm.stride(1) == 128 * 128 and ssm.stride(2) == 128 and ssm.stride(3) == 1
            and ssm.stride(0) % 4 == 0):
        return f"ssm state {tuple(ssm.shape)} {ssm.stride()} {ssm.dtype}"
    if not (layer.A_log.dtype == torch.float32 and layer.dt_bias.dtype == torch.float16
            and layer.norm.weight.dtype == torch.float16 and layer.norm.weight.numel() == 128
            and layer.in_proj_ba.weight.dtype == torch.float16 and layer.in_proj_ba.weight.is_contiguous()
            and tuple(layer.in_proj_ba.weight.shape) == (2 * heads, 5120)):
        return "parameter dtypes/shapes"
    if not (x.dtype == torch.float16 and x.is_contiguous() and x.size(1) == 5120 and qkvz.dtype == torch.float16
            and qkvz.stride(1) == 1 and qkvz.size(1) == qkv + heads * 128 and qkvz.stride(0) % 4 == 0):
        return "activation layout"
    return None


def gdn_core_impl(x: torch.Tensor, qkvz: torch.Tensor, y: torch.Tensor, layer_name: "LayerNameType") -> None:
    from vllm.forward_context import get_forward_context
    from vllm.utils.torch_utils import _resolve_layer_name
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as G
    name = _resolve_layer_name(layer_name)
    fc = get_forward_context()
    layer = fc.no_compile_layers[name]
    n = x.shape[0]
    ext_ = gdn_core_ext()                            # load (and build) outside any graph capture
    ws = gdn_core_workspace(x.device, GDN_CORE_MAXN)
    if n == 0:
        return
    md = fc.attn_metadata.get(name) if isinstance(fc.attn_metadata, dict) else None
    kv = getattr(layer, "kv_cache", None)
    if (md is not None and md.spec_sequence_masks is None and md.num_prefills == 0 and 0 < md.num_decodes <= n
            and n <= GDN_CORE_MAXN and md.non_spec_state_indices_tensor is not None and kv is not None
            and kv[0].numel() > 0):
        why = getattr(layer, "_q27_core_why", "unset")
        if why == "unset":
            why = _gdn_core_ok(layer, x, qkvz, kv)
            layer._q27_core_why = why
            if why is not None:
                log.warning("q27fuse: fused GDN core not used for %s (%s)", name, why)
            elif "ok" not in _GC_LOGGED:
                _GC_LOGGED.add("ok")
                log.warning("q27fuse: fused GDN decode core (gdn_core.cu) on %s", name)
        if why is None:
            if getattr(layer, "_spec_replay", False):
                # MTP server (patch.apply_gdn routes only draft-free decode steps here): first commit the previous
                # verification of these requests into their states, as every 1Cat path for non-speculative decode
                # rows does (Qwen3_5GatedDeltaNet._commit_replayed_decodes, gdn_verify with no new tokens)
                layer._commit_replayed_decodes(kv[1], md, md.non_spec_state_indices_tensor)
            heads = layer._q27_heads
            core = x.new_empty((n, heads, 128))
            cw = layer.conv1d.weight
            ext_.gdn_core(x, qkvz, layer.in_proj_ba.weight, cw.view(cw.size(0), 4), kv[0], layer.A_log,
                          layer.dt_bias, kv[1], md.non_spec_state_indices_tensor, int(md.num_decodes),
                          layer.norm.weight, float(layer.norm.eps), core, y, ws, gdn_core_cfg(n), 0)
            return
    # everything else (prefill, mixed batches): the chain
    b, a, core = gdn_in_impl(x, layer.in_proj_ba.weight, layer._q27_heads)
    conv_c, ssm_c = G._resolve_qwen_gdn_kv_cache_args(layer_name, core)
    core = G._qwen_gdn_run_recurrent_core(layer, mixed_qkv=qkvz[:, :layer._q27_qkv], b=b, a=a, core_attn_out=core,
                                          layer_name=layer_name, conv_state_cache=conv_c, ssm_state_cache=ssm_c)
    gdn_ext().gdn_gated_norm(core.reshape(n, layer._q27_heads, 128).contiguous(), qkvz, layer.norm.weight, y,
                             float(layer.norm.eps), layer._q27_qkv)


def gdn_core_fake(x, qkvz, y, layer_name) -> None:
    return None


# ----------------------------------------------------------------------------------------- NVFP4 MLP (mlp_fp4.cu)
# Our packed weights are the only copy of the NVFP4 MLP layers (the load hook skips 1Cat's TurboMind/QPN2
# preparation and frees the checkpoint tensors): decode kernels for 1..64 rows, 64-row chunks up to 256, exact
# dequantisation into one shared fp16 workspace + cuBLAS above (prefill).
_M = None
_MLPS = {}                 # decoder layer index -> Qwen2MoeMLP module
MLP_MAX = 64               # decode kernels; larger batches are chunked / dequantised inside mlp_fp4.cu
_WS = {}                   # device -> fp16 dequantisation workspace for prefill


def mlp_ext():
    global _M
    if _M is None:
        _M = _load("q27fuse_mlp", "mlp_fp4.cu", "_mlp", ldflags=["-lcublas"])
    return _M


def mlp_prepack(layer, codes: torch.Tensor, scales: torch.Tensor, global_scale_ct: torch.Tensor, gated: bool):
    """Pack a checkpoint NVFP4 linear (codes [N, K/2] uint8, E4M3 scales [N, K/16], compressed-tensors global scale
    divisor(s)) for mlp_fp4.cu and attach it to the layer: q27_w (uint8 buffer) and q27_epi = 2^14 / g."""
    g = float(global_scale_ct.max().to(torch.float32))         # 1Cat uses 1 / max over the fused shards too
    w = mlp_ext().prepack(codes.contiguous(), scales.contiguous().view(torch.uint8), gated)
    layer.register_buffer("q27_w", w, persistent=False)
    layer.q27_epi = 16384.0 / g
    layer.q27_gated = gated
    layer.q27_k = codes.shape[1] * 2
    if PF_PK:                                                   # [prefill2] Job P / [prefill3]: the packed format's arrays
        from . import pf3
        pf3.prepare(layer, codes, scales, g, gated)


# Opt-in H_32 packed prefill MLP (pf3.py + pk_h32.py + csrc/pk_h32.cu).
# Q27_PF_PK=1 (default 0 = off): runs 3-for-2 INT4-(1,256) FOLD HMMA GEMMs with zero-sync Freivalds guard for
# M >= Q27_PF_PK_MIN (default 1024), using Q27_PF_H32_PLANE=<dir> (1-bit plane) or zero-plane NVFP4 scales if unset.
PF_PK = int(os.environ.get("Q27_PF_PK", "0") or 0)


def workspace(dev: torch.device, numel: int) -> torch.Tensor:
    ws = _WS.get(dev)
    if ws is None or ws.numel() < numel:
        ws = _WS[dev] = note_alloc("prefill dequantisation", torch.empty(numel, dtype=torch.float16, device=dev))
    return ws


def _mlp_run(mlp, x: torch.Tensor, reduce: bool) -> torch.Tensor:
    gu, dn = mlp.gate_up_proj, mlp.down_proj
    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, dn.output_size_per_partition))
    ir = gu.output_size_per_partition // 2
    h = gu.q27_k
    ws = workspace(x.device, max(2 * ir * h, h * ir)) if m > MLP_MAX else x.new_empty(0)
    out = None
    if PF_PK and m > MLP_MAX:                                   # [prefill2] Job P / [prefill3]
        from . import pf3
        out = pf3.run(x.contiguous(), gu, dn, ir, ws)
    if out is None:
        out = mlp_ext().mlp(x.contiguous(), gu.q27_w, dn.q27_w, gu.q27_epi, dn.q27_epi, ir, ws)
    if reduce and dn.reduce_results and dn.tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_reduce
        out = tensor_model_parallel_all_reduce(out)
    return out


def mlp_impl(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """The decoder layer's MLP from our decoder forward (target-only): mlp_fp4.cu, then the TP all-reduce when
    down_proj still reduces (only the last layer: the others leave it to the next layer's fused all-reduce + norm)."""
    return _mlp_run(_MLPS[layer_idx], x, reduce=True)


def mlp_local_impl(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """The MLP on our kernels without the TP all-reduce, called from Qwen2MoeMLP.forward when 1Cat's own decoder-layer
    forward runs (MTP's verifier route); the caller all-reduces in the traced graph, where 1Cat's all-reduce + norm
    fusion can still see it."""
    return _mlp_run(_MLPS[layer_idx], x, reduce=False)


def mlp_fake(x, layer_idx):
    return x.new_empty((x.shape[0], _MLPS[layer_idx].down_proj.output_size_per_partition))


# ---------------------------------------------------------------------------------------- FP8 projections (fp8_lin.cu)
_F8 = None


def fp8_ext():
    """fp8_lin.cu: FP8 W8A16 linear on 1Cat's QPN8 layout, up to 64 rows in one pass."""
    global _F8
    if _F8 is None:
        _F8 = _load("q27fuse_fp8", "fp8_lin.cu", "_fp8")
    return _F8


# The FP8 projections (GDN in_proj_qkvz / out_proj, attention qkv_proj / o_proj, FP8 MLP down_proj) run fp8_lin.cu on
# 1Cat's QPN8 codes in one pass for the row buckets below; other row counts keep 1Cat's dispatch (its GEMV up to 8
# rows, its paired 16-row kernel for 9..16 (ours is 1.10-1.16x faster on one GPU but ties in the server) and its native
# single pass for 33..40; above 40 it would loop 32-row launches). bench_fp8.py:
FP8_CFG = [((17, 32), {}, "s 4x1 wk4 col2"),                             # (rows, {K: config}, default config)
           ((41, 64), {}, "s 4x1 wk3 col2 row2")]
_FP8_IDX = {}


def _fp8_cfg(m: int, k: int):
    for (lo, hi), by_k, default in FP8_CFG:
        if lo <= m <= hi:
            name = by_k.get(k, default)
            if name not in _FP8_IDX:
                _FP8_IDX.update({c.split(":", 2)[2]: i for i, c in enumerate(fp8_ext().fp8_configs())})
            return _FP8_IDX[name]
    return None


def fp8_linear_impl(out: torch.Tensor, x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, split_k: int,
                    nacc: int, prefetch: bool, native_policy: list[str] | None = None) -> None:
    """out = x @ W^T for a QPN8-prepared channel-FP8 linear (codes [K, N] u8, scales [1, N] fp16 = scale * 256)."""
    m = x.shape[0]
    if m == 0:
        return
    cfg = _fp8_cfg(m, x.shape[1])
    if cfg is None:
        from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
        dispatch_channel_rows(out, x, codes, scales, split_k, nacc, prefetch, False, native_policy)
    else:
        fp8_ext().fp8_linear(out, x.contiguous(), codes.view(-1), scales.view(-1), cfg)


def fp8_linear_fake(out, x, codes, scales, split_k, nacc, prefetch, native_policy=None) -> None:
    return None


# ------------------------------------------------------------------------------------------ decode attention (attn_dec.cu)
_AT = None
_AT_WS = {}


def attn_ext():
    """attn_dec.cu: split-KV E4M3 decode attention (6 query heads per KV head, head dim 256), in-kernel combine."""
    global _AT
    if _AT is None:
        _AT = _load("q27fuse_attn", "attn_dec.cu", "_attn")
    return _AT


# (pairs of warps per CTA, splits per sequence, min tiles per pair) by batch size (test_attn.py bench, one GPU):
# [(max B, cfg)]; B below ATTN_MIN_B keeps 1Cat's kernel (a tie at 1K, 1Cat ahead at 8K).
ATTN_MIN_B = 2
ATTN_CFG = [(31, (4, 16, 1)), (10 ** 9, (2, 64, 1))]


def attn_cfg(b: int):
    return next(c for m, c in ATTN_CFG if b <= m)


def attn_workspace(dev):
    """Per device: partial outputs and row stats for one resident wave of the most splitting config, and the
    per-(sequence, head) counters (zeroed once; the kernel leaves them zeroed)."""
    key = (dev.type, dev.index)
    if key not in _AT_WS:
        ext = attn_ext()
        slots = max(ext.attn_capacity(1), ext.attn_capacity(2), ext.attn_capacity(4))
        _AT_WS[key] = (note_alloc("attn_dec partials", torch.empty(slots * 6 * 256, dtype=torch.float32, device=dev)),
                       torch.empty(slots * 12, dtype=torch.float32, device=dev),
                       torch.zeros(8192, dtype=torch.int32, device=dev))
    return _AT_WS[key]


# ------------------------------------------------------------------------------------------------------- registration
_REGISTERED = False


def register_op():
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op, LayerNameType
    gdn_core_impl.__annotations__["layer_name"] = LayerNameType
    direct_register_custom_op(op_name="q27_gdn_core", op_func=gdn_core_impl, mutates_args=["y"],
                              fake_impl=gdn_core_fake)
    direct_register_custom_op(op_name="q27_mlp", op_func=mlp_impl, mutates_args=[], fake_impl=mlp_fake)
    direct_register_custom_op(op_name="q27_mlp_local", op_func=mlp_local_impl, mutates_args=[], fake_impl=mlp_fake)
    # explicit schema (torch's inference lacks optional str lists), as 1Cat registers its own QPN8 op
    from vllm._sm70.policy import register_policy_op
    register_policy_op("q27_fp8_linear", "(Tensor(a!) out, Tensor x, Tensor codes, Tensor scales, int split_k, "
                       "int nacc, bool prefetch, str[]? native_policy=None) -> ()", fp8_linear_impl, fp8_linear_fake)
    _REGISTERED = True
