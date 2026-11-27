"""q27fuse.pf3 - Opt-in H_32 packed long-prefill MLP (csrc/pk_h32.cu).

Activated when Q27_PF_PK=1 (default OFF). Runs 3-for-2 INT4-(1,256) FOLD HMMA GEMMs with INT7 activations and a
zero-sync device-gated mod 2^32 Freivalds checksum guard for prefill batches M >= Q27_PF_PK_MIN:
  - With Q27_PF_H32_PLANE=<dir>: loads precomputed BF16-guided 1-bit floor/ceil selection planes (h32_pl).
  - Without Q27_PF_H32_PLANE: derives zero-plane per-channel H_32 scales directly from NVFP4 at load (h32_nv).

Optional runtime control file Q27_PF_CTL (JSON, hot-reloaded on mtime/size change):
  {"gu": {"w": "h32_pl"|"h32_nv"|"nv", "a": 7|8}, "d": {"w": "h32_pl"|"h32_nv"|"nv", "a": 7|8},
   "layers": null | [int, ...], "min_rows": 1024, "small": 2048, "guard": 1, "fuseq": 1}
"""
import json
import logging
import os
import re

import torch

log = logging.getLogger("q27fuse")

PK_MIN = int(os.environ.get("Q27_PF_PK_MIN", "1024") or 1024)
PK_SMALL = int(os.environ.get("Q27_PF_PK_SMALL", "2048") or 2048)
H32_PLANE = os.environ.get("Q27_PF_H32_PLANE", "").strip()
H32_NV = int(os.environ.get("Q27_PF_H32_NV", "0") or 0)
ABITS = int(os.environ.get("Q27_PF_PK_ABITS", "7") or 7)
CTL = os.environ.get("Q27_PF_CTL", "")
FUSEQ = int(os.environ.get("Q27_PF_FUSEQ", "1") or 0)
GUARD = int(os.environ.get("Q27_PF_PK_GUARD", "1") or 0)

_S = {"n": 0, "ctl": None, "ctl_key": None, "logged": set(), "bytes": 0}
POSTNORMS = {}


def layer_index(prefix: str):
    mt = re.search(r"model\.layers\.(\d+)\.mlp", prefix or "")
    return None if (mt is None or "mtp" in (prefix or "")) else int(mt.group(1))


def _log_once(tag: str, msg: str, *a) -> None:
    if tag not in _S["logged"]:
        _S["logged"].add(tag)
        log.warning("q27fuse h32: " + msg, *a)


def prepare(layer, codes: torch.Tensor, scales: torch.Tensor, g: float, gated: bool) -> None:
    """Prepare H_32 arrays for one NVFP4 MLP matrix (at load, before KV sizing)."""
    from . import pk_h32
    L = layer_index(getattr(layer, "prefix", ""))
    N, K = int(codes.shape[0]), int(codes.shape[1]) * 2
    w_pre = getattr(layer, "q27_w", None)
    arrs = {}
    with torch.no_grad():
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
                    "g": float(info["g"]), "N": int(info["N"]), "K": int(info["K"]), "gated": bool(info["gated"]),
                    "alpha": sf.get_tensor(f"{tag}.alpha"), "sel": sf.get_tensor(f"{tag}.sel"),
                }
                if f"{tag}.alpha_nv" in sf_keys:
                    d["alpha_nv"] = sf.get_tensor(f"{tag}.alpha_nv")
                path = f"{st_path}:{tag}"
            else:
                path = os.path.join(H32_PLANE, f"L{L}_r{r}_{mat}.pt")
                d = torch.load(path, map_location="cpu")
            assert abs(float(d["g"]) - g) <= 1e-6 * abs(g), f"h32 plane {path}: global scale {d['g']} != {g}"
            assert int(d["N"]) == N and int(d["K"]) == K and bool(d["gated"]) == bool(gated), f"h32 plane {path}: shape"
            arrs["h32_pl"] = pk_h32.arrays_from_alpha(
                d["alpha"], g, N, K, gated, sel=d["sel"], bn_groups=3, guard=bool(GUARD), w_prepacked=w_pre
            )
            if H32_NV and "alpha_nv" in d:
                arrs["h32_nv"] = pk_h32.arrays_from_alpha(
                    d["alpha_nv"], g, N, K, gated, sel=None, bn_groups=3, guard=bool(GUARD), w_prepacked=w_pre
                )
            del d
        if "h32_nv" not in arrs and (H32_NV or not H32_PLANE or L is None):
            arrs["h32_nv"] = pk_h32.build_h32_arrays(
                codes, scales, g, gated, bf16=None, bn_groups=3, guard=bool(GUARD), w_prepacked=w_pre
            )
    for a in arrs.values():
        _S["bytes"] += sum(t.numel() * t.element_size() for t in a.values() if torch.is_tensor(t))
    layer.q27_h32 = arrs
    layer.q27_pkL = -1 if L is None else L
    _S["n"] += 1
    torch.cuda.empty_cache()


def _h32_arrays(lin, sp):
    if sp.get("w") not in ("h32_pl", "h32_nv"):
        return None
    arrs = getattr(lin, "q27_h32", None) or {}
    return arrs.get(sp["w"]) or arrs.get("h32_nv")


def _default_ctl() -> dict:
    sp = {"w": "h32_pl" if H32_PLANE else "h32_nv", "a": ABITS}
    return {"gu": dict(sp), "d": dict(sp), "layers": None, "min_rows": PK_MIN, "small": PK_SMALL}


def ctl() -> dict:
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
            except Exception as e:
                log.warning("q27fuse h32: control file %s unreadable (%s); keeping previous", CTL, e)
                return _S["ctl"] or c
        if c.get("layers") is not None:
            c["layers"] = set(int(v) for v in c["layers"])
        _S["ctl"], _S["ctl_key"] = c, key
        log.warning("q27fuse h32: control %s", json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in c.items()}))
    return _S["ctl"]


def _guard_bufs(dev, m: int = 0) -> dict:
    key = ("guard", dev)
    G = _S.get(key)
    if G is None or m > G["C"][0].numel():
        n = max(8192 + 512, m)
        prev = G or {}
        G = _S[key] = {
            "C": [torch.zeros(n, dtype=torch.int32, device=dev) for _ in range(2)],
            "R": [torch.zeros(n, dtype=torch.int32, device=dev) for _ in range(2)],
            "flag": torch.zeros(4, dtype=torch.int32, device=dev),
            "tot": prev["tot"] if "tot" in prev else torch.zeros(2, dtype=torch.int32, device=dev),
        }
    return G


def guard_totals(dev=None) -> dict:
    out = {}
    for k, G in _S.items():
        if isinstance(k, tuple) and k[0] == "guard" and (dev is None or k[1] == dev):
            out[str(k[1])] = G["tot"].tolist()
    return out


def fuse_qmax(weight: torch.Tensor, m: int):
    """Called by pf2's unified ARN: return (qmax, u, R) to fuse H_32 activation quant into post-attn AR+Norm."""
    none = (0.0, None, None)
    c = ctl()
    if not c.get("fuseq", FUSEQ) or not POSTNORMS:
        return none
    sg = c["gu"]
    if sg.get("w") not in ("h32_pl", "h32_nv") or sg.get("a") not in (7, 8):
        return none
    fuse_min = 2500 if c.get("guard", GUARD) else 1024
    if m < max(c["min_rows"], fuse_min):
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
    A = _h32_arrays(mlp.gate_up_proj, sg)
    if A is None:
        return none
    u = R = None
    if c.get("guard", GUARD) and "chku" in A:
        u, R = A["chku"], _guard_bufs(weight.device, m)["R"][0]
    return float(2 ** (sg["a"] - 1) - 1), u, R


def stash(out: torch.Tensor, xq: torch.Tensor, sx: torch.Tensor, qm: float, has_ref: bool, u=None) -> None:
    _S["stash"] = (out, tuple(out.shape), xq, sx, qm, has_ref, u)


def clear_stash() -> None:
    _S.pop("stash", None)


def _take_stash(x: torch.Tensor, a: int, chku=None):
    st = _S.pop("stash", None)
    if st is not None and st[0] is x and st[1] == tuple(x.shape) and st[4] == float(2 ** (a - 1) - 1):
        has_ref = bool(st[5]) and (chku is not None) and (st[6] is chku)
        _log_once("fuseq", "INT%d H_32 input fused into post-attn AR+Norm (%d rows; ref checksum %s)",
                  a, x.shape[0], "fused" if has_ref else "separate")
        return st[2], st[3], has_ref
    return None


def _h32_cfg(c: dict, sp: dict, gated: bool, small: bool) -> int:
    ov = c.get("h32_cfg_gu" if gated else "h32_cfg_d")
    if ov:
        return int(ov)
    if sp["a"] == 8:
        return 5 if (gated and small) else 42
    return 4 if (gated and small) else 40


def _quant_h32(X32, t: torch.Tensor, a: int, chk=None):
    tq = torch.empty_like(t)
    s = torch.empty(t.shape[0], dtype=torch.float32, device=t.device)
    qm = float(2 ** (a - 1) - 1)
    if chk is None:
        X32.quant_rows_h32(t, tq, s, qm)
    else:
        X32.quant_rows_h32(t, tq, s, qm, chk[0], chk[1])
    return tq, s


def _gate_up(X32, x, gu, ir, ws, sp, small, c):
    m, H = x.shape
    A = _h32_arrays(gu, sp)
    if A is not None:
        from . import pk_h32
        cfg_h = _h32_cfg(c, sp, True, small)
        gpad = A["gpad2"] if pk_h32.BNG[cfg_h] == 2 else A["gpad"]
        P = ws[:64 * gpad * H].view(64 * gpad, H)
        X32.convert_h32(gu.q27_w, A["sel"] if A["plane"] else None, A["ia"], P, A["ntile"], A["groups"], H, A["plane"], True, 2)
        G = _guard_bufs(x.device, m) if c.get("guard", GUARD) and "chkv" in A else None
        xs = _take_stash(x, sp["a"], chku=A.get("chku"))
        if xs is not None:
            xq, sx, has_ref = xs
            if G is not None and not has_ref:
                X32.rowdot(xq, A["chku"], G["R"][0])
        else:
            xq, sx = _quant_h32(X32, x, sp["a"], (A["chku"], G["R"][0]) if G is not None else None)
        h = x.new_empty((m, ir))
        if G is not None:
            X32.gemm_i4(xq, P, sx, A["ae"], h, 1, cfg_h, A["chkv"], G["C"][0])
            X32.check_guard(G["C"][0], G["R"][0], m, G["flag"][0:2], G["tot"])
            Wt = ws[:2 * ir * H].view(2 * ir, H)
            X32.dequant_nv_if(G["flag"][0:2], gu.q27_w, Wt, 2 * ir, H)
            X32.gemm_exact_if(G["flag"][0:2], x, Wt, h, float(gu.q27_epi), 2 * ir, H, 1)
            _log_once("zguard", "zero-sync device-gated Freivalds guard active")
        else:
            X32.gemm_i4(xq, P, sx, A["ae"], h, 1, cfg_h)
        return h
    Wt = ws[:2 * ir * H].view(2 * ir, H)
    X32.dequant_nv(gu.q27_w, Wt, 2 * ir, H)
    g = x.new_empty((m, 2 * ir))
    X32.gemm_nt(Wt, x, g, 2 * ir, H, gu.q27_epi)
    h = x.new_empty((m, ir))
    X32.silu_pair(g, h)
    return h


def _down(X32, h, dn, ir, H, ws, sp, small, c):
    m = h.shape[0]
    A = _h32_arrays(dn, sp)
    if A is not None:
        from . import pk_h32
        cfg_h = _h32_cfg(c, sp, False, small)
        gpad = A["gpad2"] if pk_h32.BNG[cfg_h] == 2 else A["gpad"]
        P = ws[:64 * gpad * ir].view(64 * gpad, ir)
        X32.convert_h32(dn.q27_w, A["sel"] if A["plane"] else None, A["ia"], P, A["ntile"], A["groups"], ir, A["plane"], False, 2)
        G = _guard_bufs(h.device, m) if c.get("guard", GUARD) and "chkv" in A else None
        hq, sh = _quant_h32(X32, h, sp["a"], (A["chku"], G["R"][1]) if G is not None else None)
        out = h.new_empty((m, H))
        if G is not None:
            X32.gemm_i4(hq, P, sh, A["ae"], out, 0, cfg_h, A["chkv"], G["C"][1])
            X32.check_guard(G["C"][1], G["R"][1], m, G["flag"][2:4], G["tot"])
            Wt = ws[:H * ir].view(H, ir)
            X32.dequant_nv_if(G["flag"][2:4], dn.q27_w, Wt, H, ir)
            X32.gemm_exact_if(G["flag"][2:4], h, Wt, out, float(dn.q27_epi), H, ir, 0)
            _log_once("zguard", "zero-sync device-gated Freivalds guard active")
        else:
            X32.gemm_i4(hq, P, sh, A["ae"], out, 0, cfg_h)
        return out
    Wt = ws[:H * ir].view(H, ir)
    X32.dequant_nv(dn.q27_w, Wt, H, ir)
    out = h.new_empty((m, H))
    X32.gemm_nt(Wt, h, out, H, ir, dn.q27_epi)
    return out


def run(x: torch.Tensor, gu, dn, ir: int, ws: torch.Tensor):
    """Run H_32 packed prefill MLP (1-bit plane or zero-plane) when active for this batch/layer, else return None."""
    c = ctl()
    m, H = x.shape
    if m < c["min_rows"] or getattr(gu, "q27_h32", None) is None:
        clear_stash()
        return None
    if c["layers"] is not None and gu.q27_pkL not in c["layers"]:
        clear_stash()
        return None
    sg, sd = c["gu"], c["d"]
    if _h32_arrays(gu, sg) is None or sg.get("a") not in (7, 8) or H not in (5120, 4352) or (2 * ir) % 128 != 0:
        sg = {"w": "nv", "a": 0}
    if _h32_arrays(dn, sd) is None or sd.get("a") not in (7, 8) or ir not in (5120, 4352) or H % 128 != 0:
        sd = {"w": "nv", "a": 0}
    if sg["w"] == "nv" and sd["w"] == "nv":
        clear_stash()
        return None
    from . import pk_h32
    X32 = pk_h32.ext()
    small = m < c["small"]
    h = _gate_up(X32, x, gu, ir, ws, sg, small, c)
    out = _down(X32, h, dn, ir, H, ws, sd, small, c)
    _log_once(json.dumps([sg, sd, small]),
              "packed H_32 MLP active: gate+up %s, down %s, %d rows (%d matrices, %.1f MB/rank)",
              sg, sd, m, _S["n"], _S["bytes"] / 2 ** 20)
    return out
