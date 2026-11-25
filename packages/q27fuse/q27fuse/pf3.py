"""q27fuse.pf3 - track prefill3: the packed prefill MLP (prefill2's Job P kernel, csrc/pk_mlp.cu FOLD mode) with
  (1) the selection-bit plane from the BF16 originals: Q27_PF_PK_PLANE=DIR of plane files (work/prefill3/plane/
      study_plane.py --save: alpha', rel and one bit per weight per (layer, rank, matrix), encoded from the BF16 weights by
      replaying the conversion's arithmetic); at load they become the conversion's plane records in the decode array's tile
      order (pk.build_arrays), before the KV cache is sized. Without it the packed weights are derived from NVFP4 alone
      (prefill2's format, "nvd"); Q27_PF_PK_NVD=1 keeps the NVFP4-derived arrays as well (for runtime switching).
  (2) per-matrix / per-layer control and quality emulation (Q27_PF_CTL = JSON file, re-read whenever it changes):
        {"gu": SPEC, "d": SPEC, "layers": null | [layer indices; -1 = the MTP drafter's MLP], "min_rows": 1024,
         "small": 2048}
      SPEC = {"w": "nv" | "nvd" | "pl", "a": 0 | 8 | 9, "k": "kernel" | "emul"}:
        w "nv"   today's NVFP4 weights (exact dequantisation + cuBLAS, mlp_fp4.cu's pieces); a 0 = today exactly,
                 a > 0 = activations rounded to per-token INT-a first (emulation of the activation term alone)
        w "nvd"/"pl" with k "kernel": the packed kernel (per-token INT-a activations, a = 8 or 9)
        w "nvd"/"pl" with k "emul": the packed format's weights unpacked from the conversion's output into fp16 and a
                 cuBLAS GEMM on fp16 (a = 0) or INT-a-rounded (a > 0) activations: the quality of the format without the
                 kernel (gate / up / h rounded to fp16 as the kernel does).
      Without a control file every NVFP4 MLP layer runs both matrices packed with the kernel (w = pl if a plane directory
      is given, else nvd), from Q27_PF_PK_MIN rows (default 1024), cfg 4 below Q27_PF_PK_SMALL rows (default 2048).
  (3) the decode-period table: Q27_PF_PK_CFGFILE (default pk_cfg.json beside this file); per (layer, matrix) FOLD cfg,
      0 = no safe cfg (that matrix stays on today's maths).
"""
import json
import logging
import os
import re

import torch

log = logging.getLogger("q27fuse")

PK_MIN = int(os.environ.get("Q27_PF_PK_MIN", "1024") or 1024)
PK_SMALL = int(os.environ.get("Q27_PF_PK_SMALL", "2048") or 2048)
PK_CFG = int(os.environ.get("Q27_PF_PK_CFG", "0") or 0)
PLANE = os.environ.get("Q27_PF_PK_PLANE", "")
H32_PLANE = os.environ.get("Q27_PF_H32_PLANE", "")          # [prefill4] dir of L{L}_r{r}_{gu|d}.pt (gen_h32_planes.py)
H32_NV = int(os.environ.get("Q27_PF_H32_NV", "0") or 0)     # [prefill4] 1 = build zero-plane h32_nv from NVFP4
_HAS_H32 = bool(H32_PLANE or H32_NV)
NVD = int(os.environ.get("Q27_PF_PK_NVD", "0" if (PLANE or _HAS_H32) else "1") or 0)
ABITS = int(os.environ.get("Q27_PF_PK_ABITS", "7" if (_HAS_H32 and not PLANE) else "8") or 8)
CTL = os.environ.get("Q27_PF_CTL", "")
CFGFILE = os.environ.get("Q27_PF_PK_CFGFILE", "") or os.path.join(os.path.dirname(__file__), "pk_cfg.json")
BF16 = os.environ.get("Q27_PF_BF16", "")          # quality yardstick: dir of the BF16 originals -> w "bf" (pinned host copy)
STATS = os.environ.get("Q27_PF_STATS", "")       # [Job 4] directory: per-launch chain margins + bounds (slow; measurement only)
_S = {"n": 0, "table": None, "cfgs": {}, "ctl": None, "ctl_key": None, "logged": set(), "bytes": 0, "h32_bytes": 0}
FUSEQ = int(os.environ.get("Q27_PF_FUSEQ", "1") or 0)     # [Job 5] quantise x for the packed gate+up inside pf_arn.cu
GUARD = int(os.environ.get("Q27_PF_PK_GUARD", "1") or 0)  # [Job 4] exact output checksum per launch + fallback to today's maths
SYNC_GUARD = int(os.environ.get("Q27_PF_SYNC_GUARD", "0") or 0)  # [prefill4] 0 = zero-sync device-gated guard for H_32; 1 = legacy .tolist() sync
# (the control file may switch guard/fuseq/sync_guard at run time: "guard": 0 / "fuseq": 0 / "sync_guard": 1;
# the checksum arrays are built at load only with Q27_PF_PK_GUARD=1)
POSTNORMS = {}                                           # decoder layer index -> post-attention norm module (patch.py)


def layer_index(prefix: str):
    mt = re.search(r"model\.layers\.(\d+)\.mlp", prefix or "")
    return None if (mt is None or "mtp" in (prefix or "")) else int(mt.group(1))


def _table_cfg(L, gated: bool) -> int:
    if PK_CFG:
        return PK_CFG
    if L is None:
        return 35                                               # other MLPs (MTP drafter): the safest fold cfg
    if _S["table"] is None:
        _S["table"] = json.load(open(CFGFILE))["cfg"]
    return int(_S["table"]["gu" if gated else "d"].get(str(L), 0))


def _load_plane(path: str, dev):
    from . import pf_format as F
    d = torch.load(path, map_location="cpu")
    Np, K, N = d["Np"], d["K"], d["N"]
    _, idx = F.triple_rows(N, d["gated"])
    enc = {"alpha": d["alpha"].to(dev), "rel": d["rel"].to(torch.int32).to(dev), "idx": idx, "Np": Np, "N": N, "K": K}
    pb = d["bits"].to(dev)
    bits = ((pb.to(torch.int32)[..., None] >> torch.arange(8, device=dev, dtype=torch.int32)) & 1).view(Np, K)
    return enc, bits, d["g"]


def _load_bf16(L: int, gated: bool, N: int, K: int) -> torch.Tensor:
    """this rank's slice of the BF16 original of an MLP matrix, merged order (gate rows then up rows / down [H, Ir]), in
    pinned host memory"""
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    r, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    if "bfidx" not in _S:
        _S["bfidx"] = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))["weight_map"]
    m = _S["bfidx"]
    pre = f"model.language_model.layers.{L}.mlp."

    def get(nm, rows=None, cols=None):
        with safe_open(os.path.join(BF16, m[pre + nm]), framework="pt", device="cpu") as f:
            sl = f.get_slice(pre + nm)
            return sl[rows[0]:rows[1]] if rows else sl[:, cols[0]:cols[1]]
    if gated:
        ir = N // 2
        t = torch.cat([get("gate_proj.weight", rows=(r * ir, (r + 1) * ir)), get("up_proj.weight", rows=(r * ir, (r + 1) * ir))])
    else:
        t = get("down_proj.weight", cols=(r * K, (r + 1) * K))
    assert tuple(t.shape) == (N, K), (L, gated, t.shape, N, K)
    out = torch.empty(t.shape, dtype=torch.bfloat16, pin_memory=True)
    out.copy_(t)
    return out


def _checksum_arrays(A: dict, codes, scales, g: float, gated: bool, bits, enc) -> None:
    """[Job 4] the exact output checksum of the packed GEMM (Freivalds over Z / 2^32): random words v per kernel slot and
    u[k] = sum_slot v[slot] w[slot, k] mod 2^32 with w = 2c 2^rel the integer operand the conversion will produce (the
    bit-exact replica's codes: plane bits, or round-to-nearest without a plane). The GEMM epilogue sums v S per row, the
    quant pass sum_k xq u; any wrong S is caught unless its error pattern is orthogonal to v mod 2^32 (probability ~2^-32
    for errors with odd entries)"""
    from . import pf_format as F, pk
    q, s = F.nv_parts(codes, scales.contiguous().view(torch.uint8))
    qs, ss = F.to_slots(q, enc["idx"]), F.to_slots(s, enc["idx"], fill=1.0)
    ia = F.ia_of(enc["alpha"], g)
    if bits is None:
        c, _, _ = F.decode(qs, ss, ia, enc["rel"], 0x100)
    else:
        c0, _, _ = F.decode(qs, ss, ia, enc["rel"], 0)
        c1, _, _ = F.decode(qs, ss, ia, enc["rel"], 0x1FF)
        c = torch.where(bits == 1, c1, c0)
        del c0, c1
    del q, s, qs, ss
    Np, K = c.shape
    w = (2 * c).to(torch.int64) << enc["rel"].to(torch.int64).repeat_interleave(3, 0).repeat_interleave(16, 1)   # [Np, K]
    del c
    gpad = A["gpad"]
    pfs, _ = pk.slot_map(A["N"], gated, gpad)
    pfs = pfs.to(w.device)
    gen = torch.Generator(device="cpu").manual_seed(int.from_bytes(os.urandom(8), "little") & (2 ** 63 - 1))
    # Keep v random nonzero even on padded slots so wrap errors leaking into partially-padded triples stay detectable (2^-32)
    v = torch.randint(0, 2 ** 32, (gpad * 96,), generator=gen, dtype=torch.int64).to(w.device)
    ok = pfs >= 0
    u = torch.zeros(K, dtype=torch.int64, device=w.device)
    idx = torch.nonzero(ok).view(-1)
    for i0 in range(0, idx.numel(), 2048):
        ii = idx[i0:i0 + 2048]
        u += (v[ii, None] * w[pfs[ii]]).sum(0)
    del w
    to32 = lambda t: torch.where((t & 0xFFFFFFFF) >= 2 ** 31, (t & 0xFFFFFFFF) - 2 ** 32, t & 0xFFFFFFFF).to(torch.int32)
    A["chkv"] = to32(v).contiguous()
    A["chku"] = to32(u).contiguous()


def prepare(layer, codes: torch.Tensor, scales: torch.Tensor, g: float, gated: bool) -> None:
    """the packed format's arrays for one NVFP4 MLP matrix (at load, before KV sizing; layer.q27_w already prepacked)"""
    from . import pf_format as F, pk
    L = layer_index(getattr(layer, "prefix", ""))
    N, K = int(codes.shape[0]), int(codes.shape[1]) * 2
    arrs, enc_of, bits_pl = {}, {}, None
    h32_arrs = {}
    with torch.no_grad():
        if PLANE and L is not None:
            from vllm.distributed import get_tensor_model_parallel_rank
            r = get_tensor_model_parallel_rank()
            path = os.path.join(PLANE, f"L{L}_r{r}_{'gu' if gated else 'd'}.pt")
            enc, bits, gf = _load_plane(path, codes.device)
            assert abs(gf - g) <= 1e-6 * abs(g), f"plane {path}: global scale {gf} != {g}"
            assert enc["N"] == N and enc["K"] == K, f"plane {path}: shape"
            arrs["pl"] = pk.build_arrays(enc, g, gated, bits, bn_groups=3)
            enc_of["pl"], bits_pl = enc, bits
        if NVD or (not PLANE and not _HAS_H32) or (L is None and not _HAS_H32):
            V = F.nv_deq(codes, scales.contiguous().view(torch.uint8), g)
            enc = F.encode(V, gated, emode="mse", acands=F.e4m3_acands(g))
            del V
            arrs["nvd"] = pk.build_arrays(enc, g, gated, None, bn_groups=3)
            enc_of["nvd"] = enc
        if _HAS_H32:                                           # [prefill4] H_32 (nosign) + INT4 (1, 256) arrays
            from . import pk_h32
            w_pre = getattr(layer, "q27_w", None)
            if H32_PLANE and L is not None:
                from vllm.distributed import get_tensor_model_parallel_rank
                r = get_tensor_model_parallel_rank()
                mat = "gu" if gated else "d"
                st_path = H32_PLANE if (os.path.isfile(H32_PLANE) and H32_PLANE.endswith(".safetensors")) else os.path.join(
                    H32_PLANE, f"h32_planes_r{r}.safetensors"
                )
                if os.path.isfile(st_path):
                    from safetensors import safe_open
                    sf_key = ("h32_sf", st_path)
                    if sf_key not in _S:
                        sf = safe_open(st_path, framework="pt", device="cpu")
                        _S[sf_key] = (sf, json.loads((sf.metadata() or {})["meta"]), set(sf.keys()))
                    sf, meta_all, sf_keys = _S[sf_key]
                    tag = f"L{L}_{mat}"
                    info = meta_all[tag]
                    d = {
                        "g": float(info["g"]),
                        "N": int(info["N"]),
                        "K": int(info["K"]),
                        "gated": bool(info["gated"]),
                        "alpha": sf.get_tensor(f"{tag}.alpha"),
                        "sel": sf.get_tensor(f"{tag}.sel"),
                    }
                    if f"{tag}.alpha_nv" in sf_keys:
                        d["alpha_nv"] = sf.get_tensor(f"{tag}.alpha_nv")
                    path = f"{st_path}:{tag}"
                else:
                    path = os.path.join(H32_PLANE, f"L{L}_r{r}_{mat}.pt")
                    d = torch.load(path, map_location="cpu")
                assert abs(float(d["g"]) - g) <= 1e-6 * abs(g), f"h32 plane {path}: global scale {d['g']} != {g}"
                assert int(d["N"]) == N and int(d["K"]) == K and bool(d["gated"]) == bool(gated), f"h32 plane {path}: shape"
                h32_arrs["h32_pl"] = pk_h32.arrays_from_alpha(
                    d["alpha"], g, N, K, gated, sel=d["sel"], bn_groups=3, guard=bool(GUARD), w_prepacked=w_pre
                )
                if "alpha_nv" in d:
                    h32_arrs["h32_nv"] = pk_h32.arrays_from_alpha(
                        d["alpha_nv"], g, N, K, gated, sel=None, bn_groups=3, guard=bool(GUARD), w_prepacked=w_pre
                    )
                del d
            if "h32_nv" not in h32_arrs and (H32_NV or (H32_PLANE and L is None)):
                h32_arrs["h32_nv"] = pk_h32.build_h32_arrays(
                    codes, scales, g, gated, bf16=None, bn_groups=3, guard=bool(GUARD), w_prepacked=w_pre
                )
    if BF16 and L is not None:
        layer.q27_bf = _load_bf16(L, gated, N, K)
    if GUARD and arrs:
        for wsrc, a in arrs.items():
            _checksum_arrays(a, codes, scales, g, gated, bits_pl if wsrc == "pl" else None, enc_of[wsrc])
    for a in arrs.values():
        a["gpad2"] = -(-a["groups"] // 2) * 2                    # the 128x128-tile geometry (cfg 4) shares the arrays
        assert a["gpad2"] <= a["gpad"]
        _S["bytes"] += sum(t.numel() * t.element_size() for t in a.values() if torch.is_tensor(t))
    for a in h32_arrs.values():
        nb = sum(t.numel() * t.element_size() for t in a.values() if torch.is_tensor(t))
        _S["h32_bytes"] += nb
        _S["bytes"] += nb
    del enc_of, bits_pl
    layer.q27_pk = arrs
    layer.q27_h32 = h32_arrs
    layer.q27_pkL = -1 if L is None else L
    layer.q27_pkcfg = _table_cfg(L, gated)
    _S["n"] += 1
    _S["cfgs"][layer.q27_pkcfg] = _S["cfgs"].get(layer.q27_pkcfg, 0) + 1
    torch.cuda.empty_cache()


def _default_ctl():
    if H32_PLANE:
        w = "h32_pl"
    elif H32_NV:
        w = "h32_nv"
    elif PLANE:
        w = "pl"
    else:
        w = "nvd"
    sp = {"w": w, "a": ABITS, "k": "kernel"}
    return {"gu": dict(sp), "d": dict(sp), "layers": None, "min_rows": PK_MIN, "small": PK_SMALL}


def ctl():
    if not CTL:
        if _S["ctl"] is None:
            _S["ctl"] = _default_ctl()
        return _S["ctl"]
    try:
        st = os.stat(CTL)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key != _S["ctl_key"] or _S["ctl"] is None:
        c = _default_ctl()
        if key is not None:
            try:
                c.update(json.load(open(CTL)))
            except Exception as e:                              # a half-written file: keep the previous control
                log.warning("q27fuse prefill3: control file %s unreadable (%s); keeping the previous one", CTL, e)
                return _S["ctl"] or c
        if c.get("layers") is not None:
            c["layers"] = set(int(v) for v in c["layers"])
        _S["ctl"], _S["ctl_key"] = c, key
        log.warning("q27fuse prefill3: packed prefill control %s", json.dumps({k: (sorted(v) if isinstance(v, set) else v)
                                                                               for k, v in c.items()}))
    return _S["ctl"]


def fuse_qmax(weight: torch.Tensor, m: int):
    """[Job 5 / prefill4] called by pf2's fused all-reduce + norm: (qmax, u, R, is_h32) for the norm output when it feeds
    a packed gate+up (FP4-331 or H_32 kernel path)"""
    none = (0.0, None, None, False)
    c = ctl()
    if not c.get("fuseq", FUSEQ) or not POSTNORMS:
        return none
    sg = c["gu"]
    is_h32 = sg["w"] in ("h32_nv", "h32_pl")
    fuse_min = (2500 if c.get("guard", GUARD) else 1024) if is_h32 else 1536
    if m < max(c["min_rows"], fuse_min) or sg["k"] != "kernel":
        return none
    if not is_h32 and sg["w"] not in ("nvd", "pl"):
        return none
    if is_h32 and sg["a"] not in (7, 8):
        return none
    if "pn" not in _S:
        _S["pn"] = {mod.weight.data_ptr(): L for L, mod in POSTNORMS.items()}
    L = _S["pn"].get(weight.data_ptr())
    if L is None or (c["layers"] is not None and L not in c["layers"]):
        return none
    from . import ops
    mlp = ops._MLPS.get(L)
    if mlp is None:
        return none
    if is_h32:
        A = _h32_arrays(mlp.gate_up_proj, sg)
        if A is None:
            return none
    else:
        A = _spec_arrays(mlp.gate_up_proj, sg)
        if A is None or (not mlp.gate_up_proj.q27_pkcfg and m >= c["small"]):
            return none                                         # this gate+up does not run packed
    u = R = None
    if c.get("guard", GUARD) and "chku" in A:
        u, R = A["chku"], _guard_bufs(weight.device, m)["R"][0]
    return float(2 ** (sg["a"] - 1) - 1), u, R, is_h32


def stash(out: torch.Tensor, xq: torch.Tensor, sx: torch.Tensor, qm: float, has_ref: bool, h32: bool = False, u=None) -> None:
    # Hold tensor identities (not just data_ptr) so allocator address reuse or mid-layer ctl flip can never alias a stale stash.
    _S["stash"] = (out, tuple(out.shape), xq, sx, qm, has_ref, bool(h32), u)


def clear_stash() -> None:
    _S.pop("stash", None)


def _take_stash(x: torch.Tensor, a: int, h32: bool = False, chku=None):
    """(xq, sx, the reference checksum for chku is already in R[0]) or None"""
    st = _S.pop("stash", None)
    if (st is not None and st[0] is x and st[1] == tuple(x.shape)
            and st[4] == float(2 ** (a - 1) - 1) and st[6] == bool(h32)):
        has_ref = bool(st[5]) and (chku is not None) and (st[7] is chku)
        _log_once(f"fuseq_{int(h32)}", "INT%d%s input of the packed gate+up quantised inside the fused all-reduce + norm "
                  "(%d rows; reference checksum %s)", a, " H_32" if h32 else "", x.shape[0],
                  "fused" if has_ref else "not fused")
        return st[2], st[3], has_ref
    return None


def _qdq(x: torch.Tensor, a: int, clip: bool = False) -> torch.Tensor:
    """per-token INT-a rounding of x, returned as fp16 values (the emulation's activations); clip: per token the scale
    amax * c / qm, c in {1, 0.95, ..., 0.6}, with the least squared error (outliers saturate)"""
    qm = float(2 ** (a - 1) - 1)
    xf = x.float()
    am = xf.abs().amax(1, keepdim=True)
    if not clip:
        return torch.where(am > 0, torch.round(xf * (qm / am)) * (am / qm), torch.zeros_like(am)).half()
    best, berr = None, None
    for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
        sc = torch.where(am > 0, am * c / qm, torch.ones_like(am))
        y = torch.clamp(torch.round(xf / sc), -qm, qm) * sc
        err = (y - xf).pow(2).sum(1, keepdim=True)
        if best is None:
            best, berr = y, err
        else:
            better = err < berr
            best = torch.where(better, y, best)
            berr = torch.where(better, err, berr)
    return torch.where(am > 0, best, torch.zeros_like(best)).half()


def _unpack(P: torch.Tensor, A: dict, gpad: int, gated: bool) -> torch.Tensor:
    """the packed format's weights in fp16 [N, K] (logical order: gate rows then up rows / down rows) from the conversion's
    packed rows P [64 gpad, K]: per triple c = 2 E2M1 recovered from (2 cA + 331 cB) 2^rel, (2 cB + 331 cC) 2^rel; value =
    c 2^rel ae (ae = alpha' / 2)"""
    from . import pk
    N, K = A["N"], A["K"]
    key = ("rows", N, gpad, gated, P.device)
    rows = _S.get(key)
    if rows is None:
        _, rows = pk.slot_map(N, gated, gpad, None if not gated else (-(-(N // 2) // 3) * 3))
        rows = _S[key] = rows.to(P.device)
    out = torch.zeros((N, K), dtype=torch.float16, device=P.device)
    CH = 8
    for g0 in range(0, gpad, CH):
        g1 = min(gpad, g0 + CH)
        rel = A["rel"][g0:g1].to(torch.float32).repeat_interleave(16, -1)          # [G, 32, K]
        Pv = P[64 * g0:64 * g1].view(g1 - g0, 32, 2, K).float()
        sc = torch.exp2(rel)
        n1, n2 = Pv[:, :, 0] / sc, Pv[:, :, 1] / sc
        cB = torch.round(n1 / 331)
        cA = (n1 - 331 * cB) / 2
        cC = torch.round(n2 / 331)
        w = torch.stack([cA, cB, cC], 2) * sc[:, :, None]                          # [G, 32, 3, K] integer operand c 2^rel
        w = w.view((g1 - g0) * 96, K) * A["ae"][96 * g0:96 * g1, None]
        r = rows[96 * g0:96 * g1]
        ok = r >= 0
        out[r[ok]] = w[ok].half()
    return out


def _log_once(tag: str, msg: str, *a) -> None:
    if tag not in _S["logged"]:
        _S["logged"].add(tag)
        log.warning("q27fuse prefill3: " + msg, *a)


def _spec_arrays(lin, sp):
    """the FP4-331 arrays a spec uses on this matrix (None -> today's weights)"""
    if sp["w"] not in ("nvd", "pl"):
        return None
    arrs = getattr(lin, "q27_pk", None) or {}
    return arrs.get(sp["w"]) or arrs.get("nvd")


def _h32_arrays(lin, sp):
    """[prefill4] the H_32 arrays a spec uses on this matrix (None -> unavailable)"""
    if sp["w"] not in ("h32_nv", "h32_pl"):
        return None
    arrs = getattr(lin, "q27_h32", None) or {}
    return arrs.get(sp["w"]) or arrs.get("h32_nv")


def _h32_cfg(c: dict, sp: dict, gated: bool, small: bool) -> int:
    """[prefill4] gemm_i4 cfg: INT7 uses 4 (128x128) for small gu, 40 (r1024u1024) otherwise (down ALWAYS 40);
    INT8 uses 5 for small gu, 42 (r512u512) otherwise. Optional Q27_PF_CTL overrides: h32_cfg_gu / h32_cfg_d."""
    ov = c.get("h32_cfg_gu" if gated else "h32_cfg_d")
    if ov:
        return int(ov)
    if sp["a"] == 8:
        return 5 if (gated and small) else 42
    return 4 if (gated and small) else 40


def _quant(X, t: torch.Tensor, a: int, chk=None):
    tq = torch.empty_like(t)
    s = torch.empty(t.shape[0], dtype=torch.float32, device=t.device)
    if chk is None:
        X.quant_rows(t, tq, s, float(2 ** (a - 1) - 1))
    else:
        X.quant_rows(t, tq, s, float(2 ** (a - 1) - 1), chk[0], chk[1])
    return tq, s


def _quant_h32(X32, t: torch.Tensor, a: int, chk=None):
    tq = torch.empty_like(t)
    s = torch.empty(t.shape[0], dtype=torch.float32, device=t.device)
    qm = float(2 ** (a - 1) - 1)
    if chk is None:
        X32.quant_rows_h32(t, tq, s, qm)
    else:
        X32.quant_rows_h32(t, tq, s, qm, chk[0], chk[1])
    return tq, s


def _guard_bufs(dev, m: int = 0):
    """[Job 4 / prefill4] per device: GEMM-side checksums C (zeroed by check/check_guard after each use), references R,
    per-launch flags [gu mismatch, gu rows, down mismatch, down rows], and persistent device-side mismatch totals
    tot = [total_mismatch_launches, total_mismatch_rows] (updated without host sync by check_guard)."""
    key = ("guard", dev)
    G = _S.get(key)
    if G is None or m > G["C"][0].numel():
        n = max(8192 + 512, m)
        prev = G or {}
        G = _S[key] = {"C": [torch.zeros(n, dtype=torch.int32, device=dev) for _ in range(2)],
                       "R": [torch.zeros(n, dtype=torch.int32, device=dev) for _ in range(2)],
                       "flag": torch.zeros(4, dtype=torch.int32, device=dev),
                       "tot": prev["tot"] if "tot" in prev else torch.zeros(2, dtype=torch.int32, device=dev),
                       "events": prev.get("events", 0), "checked": prev.get("checked", 0)}
    return G


def guard_totals(dev=None):
    """Non-hot-path diagnostic: read device-side zero-sync guard totals [mismatch_launches, mismatch_rows] (syncs!)."""
    out = {}
    for k, G in _S.items():
        if isinstance(k, tuple) and k[0] == "guard" and (dev is None or k[1] == dev):
            out[str(k[1])] = G["tot"].tolist()
    return out


def _stats_all(X, q, lin, ws, K, gated, gpad, small):
    only = os.environ.get("Q27_PF_STATS_W", "")
    for wsrc, A in (getattr(lin, "q27_pk", None) or {}).items():
        if only and wsrc not in only.split(","):
            continue
        P = ws[:64 * gpad * K].view(64 * gpad, K)
        X.convert(lin.q27_w, A["pl"], A["ia"], P, A["ntile"], A["groups"], K, A["plane"], gated, 4)
        _stats(q, P, A, gpad, "gu" if gated else "d", lin.q27_pkL, wsrc)


def _h32_mat(dev):
    H = _S.get(("h32", dev))
    if H is None:
        h = torch.tensor([[1.0]], device=dev)
        for _ in range(5):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = _S[("h32", dev)] = h / (32.0 ** 0.5)
    return H


def _rot32(t: torch.Tensor) -> torch.Tensor:
    return (t.reshape(*t.shape[:-1], t.shape[-1] // 32, 32) @ _h32_mat(t.device)).reshape(t.shape)


def _h32_alpha(R: torch.Tensor) -> torch.Tensor:
    amax = R.abs().amax(1).clamp(min=1e-30)
    best_err, best_s = None, None
    for i in range(17):
        s = amax * (0.60 + 0.025 * i) / 7.0
        q = torch.round(R / s[:, None]).clamp(-7.0, 7.0)
        err = ((q * s[:, None] - R) ** 2).sum(1)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
    return best_s


def _h32_weights(X, lin, N: int, K: int, ws: torch.Tensor, wsrc: str, gated: bool) -> torch.Tensor:
    """[prefill4] Emulated H_32 (nosign) per-channel INT4 weights (rotated domain), chunked to keep peak VRAM < 250 MB."""
    cache = getattr(lin, "_q27_h32s", None)
    if cache is None:
        cache = lin._q27_h32s = {}
    if wsrc in ("h32_nv", "h32_pl"):
        Wt = ws[:N * K].view(N, K)
        X.dequant_nv(lin.q27_w, Wt, N, K)
    else:
        Wt = None
    out = torch.empty((N, K), dtype=torch.float16, device=ws.device)
    CH = 1024
    if wsrc == "h32_nv":
        s = cache.get("nv")
        s_chunks = [] if s is None else None
        epi = float(lin.q27_epi)
        for r0 in range(0, N, CH):
            r1 = min(N, r0 + CH)
            Rnv = _rot32(Wt[r0:r1].float() * epi)
            sc = _h32_alpha(Rnv) if s is None else s[r0:r1]
            if s_chunks is not None:
                s_chunks.append(sc)
            out[r0:r1] = (torch.round(Rnv / sc[:, None]).clamp(-7.0, 7.0) * sc[:, None]).half()
        if s_chunks is not None:
            cache["nv"] = torch.cat(s_chunks)
        return out
    # BF16 rows in physical order (for gated: 32-row blocks of 16 gate + 16 up matching dequant_nv & k_silu_pair)
    s = cache.get("bf")
    s_chunks = [] if s is None else None
    epi = float(lin.q27_epi)
    bf_cpu = lin.q27_bf
    ir = N // 2
    for r0 in range(0, N, CH):
        r1 = min(N, r0 + CH)
        if gated:
            b0, b1 = r0 // 32, r1 // 32
            g_sl = bf_cpu[b0 * 16:b1 * 16].to(ws.device, non_blocking=True).view(b1 - b0, 16, K)
            u_sl = bf_cpu[ir + b0 * 16:ir + b1 * 16].to(ws.device, non_blocking=True).view(b1 - b0, 16, K)
            Vb = torch.cat([g_sl, u_sl], dim=1).reshape(r1 - r0, K).float()
        else:
            Vb = bf_cpu[r0:r1].to(ws.device, non_blocking=True).float()
        Rbf = _rot32(Vb)
        del Vb
        sc = _h32_alpha(Rbf) if s is None else s[r0:r1]
        if s_chunks is not None:
            s_chunks.append(sc)
        if wsrc == "h32_bf":
            out[r0:r1] = (torch.round(Rbf / sc[:, None]).clamp(-7.0, 7.0) * sc[:, None]).half()
        else:
            # h32_pl: 1-bit selection plane choosing floor vs floor+1 (clamped to [-7,7]) of Rnv/s closer to Rbf/s
            Rnv = _rot32(Wt[r0:r1].float() * epi)
            u_bf = Rbf / sc[:, None]
            u_nv = Rnv / sc[:, None]
            del Rnv
            c0 = torch.floor(u_nv).clamp(-7.0, 7.0)
            c1 = (c0 + 1.0).clamp(-7.0, 7.0)
            dec = torch.where((c1 - u_bf).abs() < (c0 - u_bf).abs(), c1, c0)
            out[r0:r1] = (dec * sc[:, None]).half()
        del Rbf
    if s_chunks is not None:
        cache["bf"] = torch.cat(s_chunks)
    return out


def _gate_up(X, x, gu, ir, ws, sp, small, cfg):
    m, H = x.shape
    if sp["w"] in ("h32_nv", "h32_pl") and sp["k"] == "kernel":   # [prefill4] real CUDA H_32 + INT4 (1, 256) FOLD path
        from . import pk_h32
        X32 = pk_h32.ext()
        A = _h32_arrays(gu, sp)
        c = ctl()
        cfg_h = _h32_cfg(c, sp, True, small)
        gpad = A["gpad2"] if pk_h32.BNG[cfg_h] == 2 else A["gpad"]
        P = ws[:64 * gpad * H].view(64 * gpad, H)
        X32.convert_h32(gu.q27_w, A["sel"] if A["plane"] else None, A["ia"], P, A["ntile"], A["groups"], H, A["plane"], True, 2)
        G = _guard_bufs(x.device, m) if c.get("guard", GUARD) and "chkv" in A else None
        xs = _take_stash(x, sp["a"], h32=True, chku=A.get("chku"))
        if xs is not None:
            xq, sx, has_ref = xs
            if G is not None and not has_ref:
                X32.rowdot(xq, A["chku"], G["R"][0])
        else:
            xq, sx = _quant_h32(X32, x, sp["a"], (A["chku"], G["R"][0]) if G is not None else None)
        h = x.new_empty((m, ir))
        if G is not None:
            X32.gemm_i4(xq, P, sx, A["ae"], h, 1, cfg_h, A["chkv"], G["C"][0])
            if c.get("sync_guard", SYNC_GUARD):
                X32.check(G["C"][0], G["R"][0], m, G["flag"][0:2])
            else:
                X32.check_guard(G["C"][0], G["R"][0], m, G["flag"][0:2], G["tot"])
                Wt = ws[:2 * ir * H].view(2 * ir, H)
                X32.dequant_nv_if(G["flag"][0:2], gu.q27_w, Wt, 2 * ir, H)
                X32.gemm_exact_if(G["flag"][0:2], x, Wt, h, float(gu.q27_epi), 2 * ir, H, 1)
                _log_once("h32_zguard", "zero-sync device-gated Freivalds guard active (check_guard + dequant_nv_if + gemm_exact_if)")
        else:
            X32.gemm_i4(xq, P, sx, A["ae"], h, 1, cfg_h)
        return h
    if sp["w"].startswith("h32_"):                             # [prefill4] H_32 INT4-(1,256) emulation (emul / h32_bf)
        xr = _rot32(x.float()).half()
        xin = xr if not sp["a"] else _qdq(xr, sp["a"], sp.get("clip", False))
        W = _h32_weights(X, gu, 2 * ir, H, ws, sp["w"], True)
        g = xin @ W.t()
        h = x.new_empty((m, ir))
        X.silu_pair(g, h)
        return h
    if sp["w"] == "bf":                                        # the BF16 original (fp16 GEMM): quality yardstick
        W = gu.q27_bf.to(x.device, non_blocking=True).half()
        xin = x if not sp["a"] else _qdq(x, sp["a"], sp.get("clip", False))
        gv = xin @ W.t()
        return (torch.nn.functional.silu(gv[:, :ir].float()) * gv[:, ir:].float()).half()
    A = _spec_arrays(gu, sp)
    if A is None:                                              # today's maths, optionally with rounded activations
        xin = x if not sp["a"] else _qdq(x, sp["a"], sp.get("clip", False))
        Wt = ws[:2 * ir * H].view(2 * ir, H)
        X.dequant_nv(gu.q27_w, Wt, 2 * ir, H)
        g = x.new_empty((m, 2 * ir))
        X.gemm_nt(Wt, xin, g, 2 * ir, H, gu.q27_epi)
        h = x.new_empty((m, ir))
        X.silu_pair(g, h)
        return h
    gpad = A["gpad2"] if small else A["gpad"]
    P = ws[:64 * gpad * H].view(64 * gpad, H)
    X.convert(gu.q27_w, A["pl"], A["ia"], P, A["ntile"], A["groups"], H, A["plane"], True, 4)
    if sp["k"] == "kernel":
        G = _guard_bufs(x.device, m) if ctl().get("guard", GUARD) and "chkv" in A else None
        xs = _take_stash(x, sp["a"], h32=False, chku=A.get("chku"))
        if xs is not None:
            xq, sx, has_ref = xs
            if G is not None and not has_ref:
                X.rowdot(xq, A["chku"], G["R"][0])
        else:
            xq, sx = _quant(X, x, sp["a"], (A["chku"], G["R"][0]) if G is not None else None)
        if STATS:
            _stats_all(X, xq, gu, ws, H, True, gpad, small)
            X.convert(gu.q27_w, A["pl"], A["ia"], P, A["ntile"], A["groups"], H, A["plane"], True, 4)
        h = x.new_empty((m, ir))
        if G is not None:
            X.gemm(xq, P, sx, A["ae"], h, 1, cfg, A["chkv"], G["C"][0])
            X.check(G["C"][0], G["R"][0], m, G["flag"][0:2])
        else:
            X.gemm(xq, P, sx, A["ae"], h, 1, cfg)
        return h
    W = _unpack(P, A, gpad, True)
    xin = x if not sp["a"] else _qdq(x, sp["a"], sp.get("clip", False))
    gv = xin @ W.t()
    return (torch.nn.functional.silu(gv[:, :ir].float()) * gv[:, ir:].float()).half()


def _down(X, h, dn, ir, H, ws, sp, small, cfg):
    m = h.shape[0]
    if sp["w"] in ("h32_nv", "h32_pl") and sp["k"] == "kernel":   # [prefill4] real CUDA H_32 + INT4 (1, 256) FOLD path
        from . import pk_h32
        X32 = pk_h32.ext()
        A = _h32_arrays(dn, sp)
        c = ctl()
        cfg_h = _h32_cfg(c, sp, False, small)
        gpad = A["gpad2"] if pk_h32.BNG[cfg_h] == 2 else A["gpad"]
        P = ws[:64 * gpad * ir].view(64 * gpad, ir)
        X32.convert_h32(dn.q27_w, A["sel"] if A["plane"] else None, A["ia"], P, A["ntile"], A["groups"], ir, A["plane"], False, 2)
        G = _guard_bufs(h.device, m) if c.get("guard", GUARD) and "chkv" in A else None
        hq, sh = _quant_h32(X32, h, sp["a"], (A["chku"], G["R"][1]) if G is not None else None)
        out = h.new_empty((m, H))
        if G is not None:
            X32.gemm_i4(hq, P, sh, A["ae"], out, 0, cfg_h, A["chkv"], G["C"][1])
            if c.get("sync_guard", SYNC_GUARD):
                X32.check(G["C"][1], G["R"][1], m, G["flag"][2:4])
            else:
                X32.check_guard(G["C"][1], G["R"][1], m, G["flag"][2:4], G["tot"])
                Wt = ws[:H * ir].view(H, ir)
                X32.dequant_nv_if(G["flag"][2:4], dn.q27_w, Wt, H, ir)
                X32.gemm_exact_if(G["flag"][2:4], h, Wt, out, float(dn.q27_epi), H, ir, 0)
                _log_once("h32_zguard", "zero-sync device-gated Freivalds guard active (check_guard + dequant_nv_if + gemm_exact_if)")
        else:
            X32.gemm_i4(hq, P, sh, A["ae"], out, 0, cfg_h)
        return out
    if sp["w"].startswith("h32_"):                             # [prefill4] H_32 INT4-(1,256) emulation (emul / h32_bf)
        hr = _rot32(h.float()).half()
        hin = hr if not sp["a"] else _qdq(hr, sp["a"], sp.get("clip", False))
        W = _h32_weights(X, dn, H, ir, ws, sp["w"], False)
        return hin @ W.t()
    if sp["w"] == "bf":
        W = dn.q27_bf.to(h.device, non_blocking=True).half()
        hin = h if not sp["a"] else _qdq(h, sp["a"], sp.get("clip", False))
        return hin @ W.t()
    A = _spec_arrays(dn, sp)
    if A is None:
        hin = h if not sp["a"] else _qdq(h, sp["a"], sp.get("clip", False))
        Wt = ws[:H * ir].view(H, ir)
        X.dequant_nv(dn.q27_w, Wt, H, ir)
        out = h.new_empty((m, H))
        X.gemm_nt(Wt, hin, out, H, ir, dn.q27_epi)
        return out
    gpad = A["gpad2"] if small else A["gpad"]
    P = ws[:64 * gpad * ir].view(64 * gpad, ir)
    X.convert(dn.q27_w, A["pl"], A["ia"], P, A["ntile"], A["groups"], ir, A["plane"], False, 4)
    if sp["k"] == "kernel":
        G = _guard_bufs(h.device, m) if ctl().get("guard", GUARD) and "chkv" in A else None
        hq, sh = _quant(X, h, sp["a"], (A["chku"], G["R"][1]) if G is not None else None)
        if STATS:
            _stats_all(X, hq, dn, ws, ir, False, gpad, small)
            X.convert(dn.q27_w, A["pl"], A["ia"], P, A["ntile"], A["groups"], ir, A["plane"], False, 4)
        out = h.new_empty((m, H))
        if G is not None:
            X.gemm(hq, P, sh, A["ae"], out, 0, cfg, A["chkv"], G["C"][1])
            X.check(G["C"][1], G["R"][1], m, G["flag"][2:4])
        else:
            X.gemm(hq, P, sh, A["ae"], out, 0, cfg)
        return out
    W = _unpack(P, A, gpad, False)
    hin = h if not sp["a"] else _qdq(h, sp["a"], sp.get("clip", False))
    return hin @ W.t()


# ------------------------------------------------------------------------------------------ [Job 4] margin statistics
_SP = (256, 512, 1024)                 # reduction periods (dn)
_SU = (512, 1024, 2048)                # unwrap periods (dS1)


def _chain_ops(P, A, gpad):
    """integer chain operands from the conversion output: p [64 gpad, K] (= P, exact integers) and w1 (the pair's S1 weights
    cA 2^rel) as fp32"""
    K = P.shape[1]
    p = P.float()
    rel = A["rel"][:gpad].to(torch.float32).repeat_interleave(16, -1)            # [G, 32, K]
    sc = torch.exp2(rel)
    n1 = p.view(gpad, 32, 2, K)[:, :, 0] / sc
    cB = torch.round(n1 / 331)
    w1 = ((n1 - 331 * cB) / 2 * sc)                                                # [G, 32, K]
    return p, w1.repeat_interleave(2, 1).view(64 * gpad, K)


def _stats(q: torch.Tensor, P: torch.Tensor, A: dict, gpad: int, mat: str, L: int, wsrc: str) -> None:
    """actual accumulator growth per period (64-term tile granularity) and provable Cauchy-Schwarz bounds, max over rows and
    columns, for one launch; appended to STATS/stats_r<rank>.jsonl"""
    import time
    t0 = time.time()
    p, w1 = _chain_ops(P, A, gpad)
    R, K = q.shape
    NT = K // 64
    qf = q.float()
    nx = qf.view(R, NT, 64).pow(2).sum(-1).sqrt()                                  # [R, NT]
    npn = p.view(-1, NT, 64).pow(2).sum(-1).sqrt()                                 # [C, NT]
    nw = w1.view(-1, NT, 64).pow(2).sum(-1).sqrt()
    out = {"L": L, "mat": mat, "w": wsrc, "rows": R, "tag": ctl().get("tag", "")}
    A_dn = {P_: 0.0 for P_ in _SP}
    A_ds = {U: 0.0 for U in _SU}
    A_n = A_s1 = 0.0
    RC = 512
    for r0 in range(0, R, RC):
        qq = qf[r0:r0 + RC]
        rr = qq.shape[0]
        runn = torch.zeros(rr, p.shape[0], device=q.device)
        runs = torch.zeros_like(runn)
        pn = {P_: torch.zeros_like(runn) for P_ in _SP}
        ps = {U: torch.zeros_like(runn) for U in _SU}
        for t in range(NT):
            sl = slice(64 * t, 64 * t + 64)
            dn = qq[:, sl] @ p[:, sl].t()
            ds = qq[:, sl] @ w1[:, sl].t()
            runn += dn
            runs += ds
            for P_ in _SP:
                pn[P_] += dn
                A_dn[P_] = max(A_dn[P_], float(pn[P_].abs().max()))
                if (64 * (t + 1)) % P_ == 0:
                    pn[P_].zero_()
            for U in _SU:
                ps[U] += ds
                if (64 * (t + 1)) % U == 0 or t == NT - 1:
                    A_ds[U] = max(A_ds[U], float(ps[U].abs().max()))
                    ps[U].zero_()
        A_n = max(A_n, float(runn.abs().max()))
        A_s1 = max(A_s1, float(runs.abs().max()))
    out.update({f"A_dn_{k}": v for k, v in A_dn.items()})
    out.update({f"A_dS1_{k}": v for k, v in A_ds.items()})
    out.update({"A_n": A_n, "A_S1": A_s1})
    padt = lambda M_, tp: torch.nn.functional.pad(M_, (0, -(-NT // tp) * tp - NT)).view(M_.shape[0], -(-NT // tp), tp)
    for P_ in _SP:
        tp = P_ // 64
        xs, ps_ = padt(nx, tp), padt(npn, tp)
        out[f"B_ts_{P_}"] = float((xs * ps_.amax(0)[None]).sum(-1).max())
        out[f"B_cs_{P_}"] = float(xs.pow(2).sum(-1).sqrt().amax(0).mul(ps_.pow(2).sum(-1).sqrt().amax(0)).max())
    for U in _SU:
        tp = U // 64
        xs, ws_ = padt(nx, tp), padt(nw, tp)
        out[f"Bs1_ts_{U}"] = float((xs * ws_.amax(0)[None]).sum(-1).max())
    out["B_n_ts"] = float((nx * npn.amax(0)[None]).sum(-1).max())
    out["B_S1_ts"] = float((nx * nw.amax(0)[None]).sum(-1).max())
    out["x_tile_norm_max"] = float(nx.max())
    out["sec"] = round(time.time() - t0, 2)
    from vllm.distributed import get_tensor_model_parallel_rank
    os.makedirs(STATS, exist_ok=True)
    with open(os.path.join(STATS, f"stats_r{get_tensor_model_parallel_rank()}.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


def _needs_bf16(sp: dict) -> bool:
    return sp["w"] in ("bf", "h32_bf") or (sp["w"] == "h32_pl" and sp["k"] != "kernel")


def _is_h32_kern(sp: dict) -> bool:
    return sp["k"] == "kernel" and sp["w"] in ("h32_nv", "h32_pl")


def _is_fp4_kern(sp: dict) -> bool:
    return sp["k"] == "kernel" and sp["w"] in ("nvd", "pl")


def _is_packed_kern(sp: dict) -> bool:
    return _is_fp4_kern(sp) or _is_h32_kern(sp)


def run(x: torch.Tensor, gu, dn, ir: int, ws: torch.Tensor):
    """the NVFP4 MLP for prefill rows on the packed path (FP4-331 or H_32), or None (today's path)"""
    c = ctl()
    m, H = x.shape
    if m < c["min_rows"] or (getattr(gu, "q27_pk", None) is None and getattr(gu, "q27_h32", None) is None):
        clear_stash()
        return None
    if c["layers"] is not None and gu.q27_pkL not in c["layers"]:
        clear_stash()
        return None
    sg, sd = c["gu"], c["d"]
    if sg["w"] == "nv" and sd["w"] == "nv" and not sg["a"] and not sd["a"]:
        clear_stash()
        return None
    if (_needs_bf16(sg) or _needs_bf16(sd)) and getattr(gu, "q27_bf", None) is None:
        clear_stash()
        return None                                            # the drafter's MLP has no BF16 copy: today's path
    small = m < c["small"]
    cg = 4 if small else gu.q27_pkcfg
    cd = 4 if small else dn.q27_pkcfg
    # a matrix whose table entry is 0 (FP4-331) or whose H_32 arrays/width are unsupported stays on today's weights
    if sg["w"] in ("nvd", "pl") and sg["k"] == "kernel" and not cg:
        sg = {"w": "nv", "a": 0, "k": "kernel"}
    if sd["w"] in ("nvd", "pl") and sd["k"] == "kernel" and not cd:
        sd = {"w": "nv", "a": 0, "k": "kernel"}
    if sg["w"] in ("h32_nv", "h32_pl") and sg["k"] == "kernel" and (
            _h32_arrays(gu, sg) is None or sg["a"] not in (7, 8) or H not in (5120, 4352) or (2 * ir) % 128 != 0):
        sg = {"w": "nv", "a": 0, "k": "kernel"}
    if sd["w"] in ("h32_nv", "h32_pl") and sd["k"] == "kernel" and (
            _h32_arrays(dn, sd) is None or sd["a"] not in (7, 8) or ir not in (5120, 4352) or H % 128 != 0):
        sd = {"w": "nv", "a": 0, "k": "kernel"}
    if sg["w"] in ("nvd", "pl") or sd["w"] in ("nvd", "pl"):
        from . import pk
        X = pk.ext()
    else:
        from . import pk_h32
        X = pk_h32.ext()
    gon = bool(c.get("guard", GUARD))
    sync_guard = bool(c.get("sync_guard", SYNC_GUARD))
    check_gu = _is_fp4_kern(sg) or (sync_guard and _is_h32_kern(sg))
    check_d = _is_fp4_kern(sd) or (sync_guard and _is_h32_kern(sd))
    need_sync = gon and (check_gu or check_d)
    G = _S.get(("guard", x.device)) if need_sync else None
    if G is not None:
        G["flag"].zero_()
    h = _gate_up(X, x, gu, ir, ws, sg, small, cg)
    out = _down(X, h, dn, ir, H, ws, sd, small, cd)
    if need_sync:
        G = _S.get(("guard", x.device))
        if G is not None:
            f = G["flag"].tolist()                              # waits for this MLP (legacy sync path: FP4-331 or sync_guard=1)
            G["checked"] += 1
            bad_gu = bool(f[0]) if check_gu else False
            bad_d = bool(f[2]) if check_d else False
            if bad_gu or bad_d:
                # [Job 4] a packed launch returned a wrong exact sum: redo the affected matrices on today's maths
                G["events"] += 1
                log.warning("q27fuse prefill4: packed MLP checksum MISMATCH (layer %s, %d rows: gate+up %d rows, down %d rows); "
                            "recomputing on today's path (event %d of %d checked MLPs)", gu.q27_pkL, m, f[1], f[3],
                            G["events"], G["checked"])
                today = {"w": "nv", "a": 0, "k": "kernel"}
                if bad_gu:
                    h = _gate_up(X, x, gu, ir, ws, today, small, cg)
                out = _down(X, h, dn, ir, H, ws, today if (bad_gu or bad_d) else sd, small, cd)
                G["flag"].zero_()
    _log_once(json.dumps([sg, sd, small]), "packed prefill MLP in use: gate+up %s, down %s, %d rows (%d matrices prepared, "
              "%.1f MB total / %.1f MB H_32 on this rank, fold cfgs %s)", sg, sd, m, _S["n"], _S["bytes"] / 2 ** 20,
              _S["h32_bytes"] / 2 ** 20, _S["cfgs"])
    return out
