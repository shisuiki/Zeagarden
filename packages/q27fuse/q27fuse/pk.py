"""q27fuse.pk (tracks prefill, prefill2) - host side of the packed prefill MLP (csrc/pk_mlp.cu): the kernel's group / slot geometry,
the plane + metadata arrays from an encoding (pf_format.encode / plane_bits), and replicas for the tests.

Kernel geometry: group G = decode tiles 3G..3G+2 (96 rows); slot s in 0..95 of a group: gate+up: s < 48 = gate channel
48G + s, s >= 48 = up channel 48G + s - 48; down: channel 96G + s. Triple t of a group = slots 3t..3t+2; packed rows
64G + 2t (slots 3t, 3t+1) and 64G + 2t + 1 (slots 3t+1, 3t+2). Global slot 96G + s = 3 * (packed column // 2) + j."""
import os
import torch

_PK = None


def ext():
    global _PK
    if _PK is None:
        from .ops import _load
        _PK = _load("q27fuse_pk", "pk_mlp.cu", "_pk", ldflags=["-lcublas"])   # [prefill3] cuBLAS for gemm_nt
    return _PK


def geometry(N: int, K: int, gated: bool, bn_groups: int = 3):
    ntile = N // 32
    groups = -(-ntile // 3)
    gpad = -(-groups // bn_groups) * bn_groups
    return ntile, groups, gpad


def slot_map(N: int, gated: bool, gpad: int, Ip: int | None = None):
    """[gpad * 96] long: for each kernel slot, the pf_format slot index (pf_format.triple_rows order) or -1; and the
    matrix row (merged order) or -1."""
    G = torch.arange(gpad).repeat_interleave(96)
    s = torch.arange(96).repeat(gpad)
    if gated:
        I = N // 2
        Ip = -(-I // 3) * 3 if Ip is None else Ip
        up = s >= 48
        ch = 48 * G + torch.where(up, s - 48, s)
        valid = ch < I
        pf = torch.where(up, Ip + ch, ch)
        row = torch.where(up, I + ch, ch)
    else:
        ch = 96 * G + s
        valid = ch < N
        pf = ch
        row = ch
    return torch.where(valid, pf, -1), torch.where(valid, row, -1)


def lane_slot(gated: bool):
    """[3, 32] long: slot of (tile i, lane l)"""
    l = torch.arange(32)
    lc = 8 * ((l >> 2) & 3) + (l & 3) + 4 * (l >> 4)
    out = []
    for i in range(3):
        if gated:
            out.append(torch.where(lc < 16, 16 * i + lc, 48 + 16 * i + lc - 16))
        else:
            out.append(32 * i + lc)
    return torch.stack(out)


def _i32(w: torch.Tensor) -> torch.Tensor:
    """int64 holding uint32 bit patterns -> int32 with the same bits"""
    w = w & 0xFFFFFFFF
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32).contiguous()


def build_arrays(enc: dict, g: float, gated: bool, bits: torch.Tensor | None = None, bn_groups: int = 3):
    """kernel arrays from a pf_format encoding: pl uint8 [gpad][NC][400 | 16], ia f32 [gpad * 96], ae f32 [gpad * 96],
    plus geometry. bits [Np, K] (pf slot order) -> plane mode."""
    from . import pf_format as F   # [prefill2] the format module inside the tree
    N, K, Np = enc["N"], enc["K"], enc["Np"]
    ntile, groups, gpad = geometry(N, K, gated, bn_groups)
    NC, B = K // 32, K // 16
    Ip = Np // 2 if gated else None
    pfs, _ = slot_map(N, gated, gpad, Ip)
    dev = enc["alpha"].device
    pfs = pfs.to(dev)
    valid = pfs >= 0
    alpha = torch.where(valid, enc["alpha"][pfs.clamp(min=0)], torch.zeros((), device=dev))
    ia = torch.where(valid, F.ia_of(alpha.clamp(min=1e-30), g), torch.zeros((), device=dev)).to(torch.float32)
    ae = (alpha / 2).to(torch.float32)
    # rel per kernel triple: the pf triple of the slot 3t (any slot of a triple maps to the same pf triple)
    tri_slot = pfs.view(gpad, 32, 3)[:, :, 0]                                     # pf slot of each kernel triple's first row
    # a triple whose first slot is padding but later slots are real cannot happen (padding only at the end of a half)
    rel = torch.where((tri_slot >= 0)[..., None], enc["rel"][(tri_slot.clamp(min=0) // 3)], torch.zeros((), device=dev, dtype=enc["rel"].dtype))
    rel = rel.to(torch.int64).view(gpad, 32, NC, 2)                              # [G, t, c, b]
    shift = (4 * (torch.arange(32, device=dev) % 8))[None, :, None, None] + 2 * torch.arange(2, device=dev)[None, None, None, :]
    contrib = rel << shift                                                       # [G, t, c, b]
    word = torch.zeros(gpad, NC, 4, dtype=torch.int64, device=dev)
    for q in range(4):
        word[:, :, q] = contrib[:, 8 * q:8 * q + 8].sum((1, 3))
    relrec = _i32(word).view(torch.uint8).view(gpad, NC, 16)
    if bits is None:
        pl = relrec.contiguous()
    else:
        bk = torch.where(valid[:, None], bits[pfs.clamp(min=0)].to(torch.int64), torch.zeros((), device=dev, dtype=torch.int64))
        bk = bk.view(gpad, 96, NC, 16, 2)                                          # [G, s, c, j, h]
        sh = torch.arange(16, device=dev)
        wsel = ((bk[..., 0] << sh) | (bk[..., 1] << (sh + 16))).sum(-1)           # [G, s, c]
        ls = lane_slot(gated).to(dev)                                            # [3, 32]
        sel = wsel[:, ls.view(-1), :].view(gpad, 3, 32, NC).permute(0, 3, 1, 2)   # [G, c, i, l]
        selb = _i32(sel.contiguous()).view(torch.uint8).view(gpad, NC, 384)
        pl = torch.cat([selb, relrec], -1).contiguous()
    return {"pl": pl, "ia": ia.contiguous(), "ae": ae.contiguous(), "ntile": ntile, "groups": groups, "gpad": gpad,
            "N": N, "K": K, "gated": gated, "plane": bits is not None,
            "rel": rel.view(gpad, 32, NC * 2).to(torch.int8).contiguous()}          # [prefill3] for the emulation


def packed_reference(codes_slot: torch.Tensor, rel: torch.Tensor, gpad: int) -> torch.Tensor:
    """replica of the conversion's packed rows from E2M1 code values per kernel slot [gpad * 96, K] and per kernel triple
    rel [gpad, 32, K/16]: [64 gpad, K] fp16 = (2 cA + 331 cB) 2^rel, c = 2 E2M1"""
    K = codes_slot.shape[1]
    c = (2 * codes_slot).view(gpad, 32, 3, K)
    r = torch.exp2(rel.to(torch.float64).repeat_interleave(16, -1))               # [G, 32, K]
    P1 = (2 * c[:, :, 0].double() + 331 * c[:, :, 1].double()) * r
    P2 = (2 * c[:, :, 1].double() + 331 * c[:, :, 2].double()) * r
    return torch.stack([P1, P2], 2).view(64 * gpad, K).to(torch.float16)


def codes_from_packed(P: torch.Tensor, rel: torch.Tensor, gpad: int) -> torch.Tensor:
    """invert the packed rows to 2 E2M1 codes per kernel slot [gpad * 96, K] (exact for valid rows)"""
    K = P.shape[1]
    r = torch.exp2(-rel.to(torch.float64).repeat_interleave(16, -1))              # [G, 32, K]
    Pv = P.view(gpad, 32, 2, K).double()
    n1, n2 = Pv[:, :, 0] * r, Pv[:, :, 1] * r                                    # 2 cA + 331 cB, 2 cB + 331 cC
    cB = torch.round(n1 / 331)
    cA = (n1 - 331 * cB) / 2
    cC = torch.round(n2 / 331)
    cB2 = (n2 - 331 * cC) / 2
    ok = (cB2 == cB)
    return torch.stack([cA, cB, cC], 2).view(gpad * 96, K), bool(ok.all())
