"""q27fuse.smallrows (track smallrows) - the weight kernels at small row counts: NVFP4 MLP variants at 1..16 rows, the
GDN in-projection of MTP verify batches in one launch of our FP8 tile, and the cutover of the FP8 projections at
9..64 rows (incl. the FP8 MLP gate_up of layers 56-63) to fp8_lin.cu. Defaults = the behaviour adopted from the
server A/B sessions (notes/smallrows.md); every knob is read once per process, and 0 restores p14 for its piece:

Q27_SR_MLP   mlp_fp4.cu variant bits (default 7): 1 gate+up .cg weight loads at 1..8 rows, 2 down .cg at 1..8,
             4 parallel k-slice reduction at 1..16 rows (all bitwise equal to p14's kernels).
Q27_SR_FP8   FP8 projections on fp8_lin.cu (default 56: 48 in p15, 56 again in track head): 16 = rows 9..16 of the target projections (qkv_proj,
             out_proj / o_proj, FP8 down_proj; 1Cat's paired m16 kernel before), 32 = rows 17..64 on FP8_CFG
             (parallel-reduction shapes, ours also at 33..40 where p14 kept 1Cat's, a 48-row shape at 33..48), 8 = the FP8
             MLP gate_up with SiLU-and-mul of layers 56-63 at GATED_MIN..64 rows (1Cat's gated pair kernel before; [head]
             GATED_MIN = Q27_SR_GATED_MIN, default 33). Rows 1..8
             stay on 1Cat's GEMV (ours lost there in the server), and so does the MTP drafter (lost or tied).
Q27_SR_GDNIN 1 (default) = the GDN in-projection (qkv | z | b | a) in one launch of fp8_gdn_in.cu (GDNIN_CFG: .nc weight
             loads at 17..32 rows, .cg at 33..64), replacing 1Cat's apply_gdn_ba_verify (its m32 two-phase kernel at
             17..32 rows) and its split route above 32 rows (FP8 qkvz + cuBLAS b/a + split copy).
Q27_SR_GDNIN_MIN  rows from which the GDN in-projection is ours (default 17: at 5..16 rows it lost in the server).
The GDN projection runs eagerly inside 1Cat's opaque GDN op and the FP8 routing inside the opaque q27_fp8_linear /
q27_fp8_gated ops, so the traced graphs do not depend on the knobs (the gated op is new: a fresh compile cache)."""
import logging
import os
import torch

log = logging.getLogger("q27fuse")

# the adopted behaviour (server A/B, notes/smallrows.md); each knob set to 0 restores p14's behaviour for its piece
# [merge] Q27_SR_FP8 default 56 -> 48: bit 8 (the gated FP8 MLP gate_up of layers 56-63 on fp8_lin.cu) is OFF in p15. With it
# on, together with attnv, MTP servers lose requests to NaN logprobs under staggered load (1-4 of 48 requests in every
# staggered check; Q27_SR_FP8=48, Q27_SR_FP8=0 or Q27_ATTNV=0 -> 0 failures; notes/merge.md section 5). Root cause not found.
# [head] root cause found and fixed in attnv.py (its workspace was allocated inside a CUDA-graph capture and aliased the
# PIECEWISE graphs' memory; the gated kernel only changed the memory layout - notes/head.md Job 0). Default back to 56, with
# the gated kernel from 33 rows only: server A/B with the fix (8K MTP product mapping) -0.27 to -0.35 ms at 48-64 verify
# rows (C 20, C 32; 1K target-only C 48 -0.31), but +0.06 ms at C 8 (32 rows) when it also served 9-32 rows.
DEFAULTS = {"Q27_SR_MLP": 7, "Q27_SR_FP8": 56, "Q27_SR_GDNIN": 1, "Q27_SR_GDNIN_MIN": 17}


def _knob(name: str) -> int:
    v = os.environ.get(name, "")
    return int(v, 0) if v else DEFAULTS[name]


MLPV = _knob("Q27_SR_MLP")
FP8V = _knob("Q27_SR_FP8")
GDNIN = _knob("Q27_SR_GDNIN")
GDNIN_MIN = _knob("Q27_SR_GDNIN_MIN")

# fp8_lin.cu shapes per row bucket (bench_sr_fp8.py; notes/smallrows.md): [((rows lo, hi), {K: name}, default name)]
FP8_CFG = [((9, 16), {1536: "s 2x1 wk4 col2 pr"}, "s 2x1 wk8 col2 pr"),
           ((17, 32), {}, "s 4x1 wk4 col2 pr"),
           ((33, 48), {}, "s 6x1 wk4 col2 pr"),
           ((49, 64), {}, "s 4x1 wk3 col2 row2 pr")]
# fp8_gdn_in.cu shape per row bucket (qkvz: K 5120, N 4096): [(max rows, name)]
# (.cg at 33..64 rows: session 3a, C32 -0.29 vs -0.22 ms with .nc; .nc at 17..32: C8/C16 -0.52/-0.47 vs -0.43/-0.40)
GDNIN_CFG = [(8, "s 1x1 wk8 col1 pr"), (16, "s 2x1 wk8 col2 pr"), (32, "s 4x1 wk4 col2 pr"),
             (48, "s 6x1 wk4 col2 pr cg"), (64, "s 4x1 wk3 col2 row2 pr cg")]
# [merge] Q27_SR_GDNIN_CG: .cg weight loads per in-projection row bucket (bits 1: 17..32, 2: 33..48, 4: 49..64 rows);
# default 6 = the GDNIN_CFG above (.nc at 17..32, .cg at 33..64)
_GDNIN_CG = int(os.environ.get("Q27_SR_GDNIN_CG", "") or "6", 0)
GDNIN_CFG = [(r, n if bit is None else n.removesuffix(" cg") + (" cg" if _GDNIN_CG & bit else ""))
             for (r, n), bit in zip(GDNIN_CFG, (None, None, 1, 2, 4))]
# fp8_lin.cu fp8_gated shape per row bucket (gate_up: K 5120, N 8704), from GATED_MIN rows: [(max rows, name)]
# [head] rows from which the gated op runs fp8_lin.cu (default 33, see DEFAULTS); Q27_DBG_GATED_MIN / _MAX = the Job 0
# experiment names (still honoured)
GATED_MIN = int(os.environ.get("Q27_SR_GATED_MIN", "") or os.environ.get("Q27_DBG_GATED_MIN", "") or "33")
GATED_MAX = int(os.environ.get("Q27_DBG_GATED_MAX", "") or "64")
GATED_CFG = [(16, "g 2x2 wk4 col1 pr"), (32, "g 4x2 wk4 col1 pr"), (48, "g 3x2 wk4 col1 row2 pr"),
             (64, "g 4x2 wk3 col1 row2 pr")]
_GIDX = {}
POST_HOOK = None                                  # [head] debug hook (nandbg.py)


def target_table():
    """the q27_fp8_linear row table for the FP8V bits: p14's entries (ops.FP8_CFG) where a bit is off"""
    from . import ops
    t = [e for e in FP8_CFG if (FP8V & 16 if e[0][1] <= 16 else FP8V & 32)]
    if not FP8V & 32:
        t += [e for e in ops.FP8_CFG if e[0][0] >= 17]
    return t


_GI = None
_GI_IDX = {}


def gdn_in_ext():
    global _GI
    if _GI is None:
        from . import ops
        _GI = ops._load("q27fuse_gdnin", "fp8_gdn_in.cu", "_gdnin")
        _GI_IDX.update({c.split(":", 2)[2]: i for i, c in enumerate(_GI.gdn_in8_configs())})
    return _GI


def gdn_in_cfg(m: int) -> int:
    ext = gdn_in_ext()
    return _GI_IDX[next(n for r, n in GDNIN_CFG if m <= r)]


def _gdn_in_ok(layer) -> str | None:
    """why our in-projection cannot serve this GDN layer (None = it can); 1Cat's own conditions for its fused route
    (apply_gdn_ba_verify) plus our FP8 hook's acceptance of the qkvz codes"""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.platforms import current_platform
    qz, ba = getattr(layer, "in_proj_qkvz", None), getattr(layer, "in_proj_ba", None)
    if not (getattr(layer, "enable_sm70_dflash2_fused_gdn_verify", False)
            and (getattr(layer, "enable_sm70_gdn_ba_verify", False)
                 or getattr(layer, "sm70_gdn_ba_verify_native", None) is not None)):
        return "1Cat's fused verify projection is off (no MTP verifier)"
    if not (isinstance(qz, LinearBase) and isinstance(ba, LinearBase) and current_platform.is_device_capability(70)):
        return "projection modules"
    heads = layer.num_v_heads // layer.tp_size
    width = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size + heads * layer.head_v_dim
    w, s = getattr(qz, "weight", None), getattr(qz, "weight_scale_inv", None)
    if not (getattr(qz, "q27_fp8", False) and w is not None and s is not None and w.dtype == torch.uint8
            and tuple(w.shape) == (5120, width) and w.is_contiguous() and s.dtype == torch.float16
            and s.numel() == width and s.is_contiguous()):
        return "qkvz codes"
    if not (ba.weight.dtype == torch.float16 and ba.weight.is_contiguous() and tuple(ba.weight.shape) == (2 * heads, 5120)
            and 2 * heads == 24 and getattr(ba, "bias", None) is None):
        return "b/a weight"
    if not (layer.head_k_dim == 128 and layer.head_v_dim == 128 and not layer.gqa_interleaved_layout
            and not layer.disable_tp_for_ba_proj and width % 64 == 0
            and ((layer.key_dim * 2 + layer.value_dim) // layer.tp_size) % 64 == 0):
        return "layout"
    return None


def gdn_in_project(layer, x: torch.Tensor, outputs=None):
    """qkv [M, 2560], z [M, 1536], b, a [M, 12] for a GDN layer's input x [M, 5120] (1 <= M <= 64), or None."""
    m = x.shape[0]
    if not (GDNIN and x.dim() == 2 and max(1, GDNIN_MIN) <= m <= 64 and x.dtype == torch.float16 and x.is_contiguous()
            and x.shape[1] == 5120):
        return None
    why = getattr(layer, "_sr_gdnin_why", "unset")
    if why == "unset":
        why = _gdn_in_ok(layer)
        layer._sr_gdnin_why = why
        if why is not None:
            log.warning("q27fuse smallrows: GDN in-projection not taken over for %s (%s)", layer.prefix, why)
        elif not getattr(gdn_in_project, "_logged", False):
            gdn_in_project._logged = True
            log.warning("q27fuse smallrows: GDN in-projection qkv|z|b|a on fp8_gdn_in.cu (%s, mode %d, first %d rows)",
                        layer.prefix, GDNIN, m)
    if why is not None:
        return None
    heads = layer.num_v_heads // layer.tp_size
    zw = heads * layer.head_v_dim
    qw = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    shapes = ((m, qw), (m, zw), (m, heads), (m, heads))
    if outputs is None:
        outputs = tuple(x.new_empty(s) for s in shapes)
    elif any(t.shape != s or t.dtype != x.dtype or not t.is_contiguous() for t, s in zip(outputs, shapes)):
        return None
    qkv, z, b, a = outputs
    qz = layer.in_proj_qkvz
    ext = gdn_in_ext()
    ext.gdn_in8(x, qz.weight, qz.weight_scale_inv, layer.in_proj_ba.weight, qkv, z, b, a, gdn_in_cfg(m), True)
    return outputs


def apply_gdn_in():
    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as V
    if getattr(V, "_q27_sr_patched", False):
        return
    orig = V.apply_gdn_ba_verify

    def apply_gdn_ba_verify(layer, x, outputs=None):
        r = gdn_in_project(layer, x, outputs)
        return r if r is not None else orig(layer, x, outputs)

    V.apply_gdn_ba_verify = apply_gdn_ba_verify   # Qwen3_5GatedDeltaNet.forward_cuda imports it at each call
    V._q27_sr_patched = True


# ------------------------------------------------------------------------- the FP8 MLP gate_up with SiLU-and-mul (GATED)
def gated_impl(out: torch.Tensor, x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, split_k: int, nacc: int,
               prefetch: bool, native_policy: list[str] | None = None) -> None:
    """out [M, I] = silu(x Wg^T) * (x Wu^T), QPN8 codes of the merged [gate | up] weight: fp8_lin.cu at GATED_MIN..64
    rows (Q27_SR_FP8 bit 8), else 1Cat's gated dispatch (what its apply_fused_silu_and_mul runs)"""
    from . import ops
    m = x.shape[0]
    if m == 0:
        return
    if FP8V & 8 and GATED_MIN <= m <= GATED_MAX and (2 * out.shape[1]) % 64 == 0:
        e = ops.fp8_ext()
        if not _GIDX:
            _GIDX.update({c.split(":", 2)[2]: i for i, c in enumerate(e.fp8_gated_configs())})
        e.fp8_gated(out, x.contiguous(), codes.view(-1), scales.view(-1), _GIDX[next(n for r, n in GATED_CFG if m <= r)])
    else:
        from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
        dispatch_channel_rows(out, x, codes, scales, split_k, nacc, prefetch, True, native_policy)
    if POST_HOOK is not None:                     # [head] debug hook (nandbg.py, Q27_NANDBG=1)
        POST_HOOK("gated", out, x, codes)


def gated_fake(out, x, codes, scales, split_k, nacc, prefetch, native_policy=None) -> None:
    return None


def apply_gated():
    """route CompressedTensorsW8A16Fp8.apply_fused_silu_and_mul (the FP8 MLP gate_up of layers 56-63: QPN8 codes, no
    batch TurboMind copy) through op q27_fp8_gated"""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a16_fp8 as S
    cls = S.CompressedTensorsW8A16Fp8
    if getattr(cls, "_q27_sr_gated", False):
        return
    orig = cls.apply_fused_silu_and_mul

    def apply_fused_silu_and_mul(self, layer, x):
        w, sc = getattr(layer, "weight", None), getattr(layer, "weight_scale_inv", None)
        if (getattr(layer, "sm70_fp8_qpn8", False) and getattr(layer, "sm70_fp8_gated_silu", False)
                and not getattr(layer, "sm70_fp8_batch_tm", False) and x.dtype == torch.float16
                and w is not None and sc is not None and w.dtype == torch.uint8 and w.dim() == 2
                and sc.dtype == torch.float16 and sc.numel() == w.shape[1] and w.shape[1] % 64 == 0):
            x2 = x.reshape(-1, x.shape[-1])
            if x2.stride(-1) != 1:
                x2 = x2.contiguous()
            i = layer.output_size_per_partition // 2
            out = torch.empty((x2.shape[0], i), device=x.device, dtype=x.dtype)
            if x2.shape[0] == 0:
                return out.reshape(*x.shape[:-1], i)
            torch.ops.vllm.q27_fp8_gated(out, x2, w, sc, int(layer.sm70_fp8_qpn8_gated_split_k),
                                         int(layer.sm70_fp8_qpn8_gated_nacc),
                                         bool(layer.sm70_fp8_qpn8_gated_prefetch), self.native_ops.arguments)
            return out.reshape(*x.shape[:-1], i)
        return orig(self, layer, x)

    cls.apply_fused_silu_and_mul = apply_fused_silu_and_mul
    cls._q27_sr_gated = True


_REG = False


def apply():
    """called from patch.apply() (after ops.register_op)"""
    global _REG
    from . import ops
    if not _REG:
        from vllm._sm70.policy import register_policy_op
        register_policy_op("q27_fp8_gated", "(Tensor(a!) out, Tensor x, Tensor codes, Tensor scales, int split_k, "
                           "int nacc, bool prefetch, str[]? native_policy=None) -> ()", gated_impl, gated_fake)
        _REG = True
        if not getattr(ops, "_sr_wrapped", False):
            m_ext = ops.mlp_ext

            def mlp_ext():
                e = m_ext()
                if getattr(e, "_sr_v", None) != MLPV:
                    e.set_variant(MLPV)
                    e._sr_v = MLPV
                return e

            ops.mlp_ext = mlp_ext
            ops._sr_wrapped = True
        ops.FP8_CFG = target_table()                 # ops.fp8_linear_impl / _fp8_cfg read it per call
        log.warning("q27fuse smallrows: Q27_SR_MLP=%d Q27_SR_FP8=%d Q27_SR_GDNIN=%d (from %d rows), gated gate_up %d..%d rows",
                    MLPV, FP8V, GDNIN, GDNIN_MIN, GATED_MIN, GATED_MAX)
    apply_gdn_in()
    apply_gated()
