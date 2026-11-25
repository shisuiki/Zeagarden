"""q27fuse.pk_h32 (track prefill4) - host loader, slot/plane layout, and Freivalds array builders for csrc/pk_h32.cu
(online H_32 nosign + INT4 (1, 256) FOLD + INT7/INT8 packed prefill MLP on V100 sm_70).

Kernel geometry (matches q27fuse.pk / pk_mlp.cu):
  - Decode array W: mlp_fp4.cu prepack, 32-column tiles [ntile, K/32, 576] u8.
  - Group G = 3 decode tiles (3G..3G+2 = 96 decode rows):
      gated=True : slots 0..47  = gate channels 48G + 0..47 (row 48G + s)
                   slots 48..95 = up   channels 48G + 0..47 (row I + 48G + s - 48)
      gated=False: slots 0..95  = down channels 96G + 0..95 (row 96G + s)
  - Triple t of group G = slots 3t..3t+2 -> 2 packed rows in (1, 256) chain:
      P1 = c[3t]   + 256 * c[3t+1]   at packed row 64G + 2t
      P2 = c[3t+1] + 256 * c[3t+2]   at packed row 64G + 2t + 1
  - Selection plane (PLANE=1 only, NO rel bytes):
      sel [gpad, K/32, 384] u8 = [gpad, K/32, 3 warps, 32 lanes] uint32 little-endian,
      where bit k (0..31) selects floor(f[k]) + bit (clamped to [-7, 7]).
"""
import math
import os
import torch

_EXT = None

# Exact f32 1 / sqrt(32) = 0x3e3504f3 (matches INV_SQRT32 in pk_h32.cu)
INV_SQRT32_F32 = float(torch.tensor(1.0 / math.sqrt(32.0), dtype=torch.float32).item())

_E2M1_CPU = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# bn_groups required by each gemm_i4 cfg (2 -> 128x128 tile, 3 -> 128x192 tile)
BNG = {4: 2, 5: 2, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3}


def ext():
    global _EXT
    if _EXT is None:
        from .ops import _load
        _EXT = _load("q27_pk_h32", "pk_h32.cu", "_pkh32", ldflags=["-lcublas"])
    return _EXT


# ------------------------------------------------------------------------------------------------ geometry & slot mapping
def geometry(N: int, K: int, gated: bool, bn_groups: int = 3) -> tuple[int, int, int]:
    """Return (ntile, groups, gpad) matching pk.geometry."""
    assert N % 32 == 0 and K % 32 == 0, f"N={N}, K={K} must be multiples of 32"
    ntile = N // 32
    groups = -(-ntile // 3)
    gpad = -(-groups // bn_groups) * bn_groups
    return ntile, groups, gpad


def slot_map_h32(N: int, gated: bool, gpad: int) -> torch.Tensor:
    """[gpad * 96] long: merged matrix row index (0..N-1) for each kernel slot, or -1 for padded slots."""
    G = torch.arange(gpad, dtype=torch.long).repeat_interleave(96)
    s = torch.arange(96, dtype=torch.long).repeat(gpad)
    if gated:
        I = N // 2
        up = s >= 48
        ch = 48 * G + torch.where(up, s - 48, s)
        valid = ch < I
        row = torch.where(up, I + ch, ch)
    else:
        ch = 96 * G + s
        valid = ch < N
        row = ch
    return torch.where(valid, row, torch.full_like(row, -1))


def lane_slot(gated: bool) -> torch.Tensor:
    """[3, 32] long: kernel slot owned by (warp wi in 0..2, lane l in 0..31) in k_conv_h32."""
    l = torch.arange(32, dtype=torch.long)
    lc = 8 * ((l >> 2) & 3) + (l & 3) + 4 * (l >> 4)
    out = []
    for wi in range(3):
        if gated:
            out.append(torch.where(lc < 16, 16 * wi + lc, 48 + 16 * wi + lc - 16))
        else:
            out.append(32 * wi + lc)
    return torch.stack(out)


def _i32(w: torch.Tensor) -> torch.Tensor:
    """int64 holding uint32 bit patterns -> contiguous int32 with identical bits."""
    w = w & 0xFFFFFFFF
    return torch.where(w >= (1 << 31), w - (1 << 32), w).to(torch.int32).contiguous()


# ------------------------------------------------------------------------------------------------ H_32 butterfly & alpha
def fwht32_unnorm(x: torch.Tensor) -> torch.Tensor:
    """Exact 5-stage FP32 Walsh-Hadamard butterfly (strides 1, 2, 4, 8, 16), unnormalized."""
    y = x.float().reshape(-1, 32)
    for stride in (1, 2, 4, 8, 16):
        y = y.view(-1, 32 // (2 * stride), 2, stride)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2).reshape(-1, 32)
    return y.reshape(x.shape)


def fwht32(x: torch.Tensor) -> torch.Tensor:
    """Exact normalized 5-stage FP32 H_32 butterfly (* f32(1/sqrt(32))), matching k_quant_h32."""
    return fwht32_unnorm(x) * INV_SQRT32_F32


def h32_alpha(R: torch.Tensor) -> torch.Tensor:
    """MSE-optimal per-channel INT4 [-7, 7] scale over 17 multipliers 0.60..1.00; 0.0 for all-zero rows."""
    raw_amax = R.abs().amax(1)
    amax = raw_amax.clamp(min=1e-30)
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
    return torch.where(raw_amax > 0, best_s, torch.zeros_like(best_s))


# ------------------------------------------------------------------------------------------------ NVFP4 & packing helpers
def nv_parts_f32(codes: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode NVFP4 packed codes [N, K/2] u8 and E4M3FN scales [N, K/16] -> (q [N, K] f32, s [N, K/16] f32)."""
    lut = _E2M1_CPU.to(codes.device)
    lo = (codes & 0xF).long()
    hi = (codes >> 4).long()
    q = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(codes.shape[0], -1)
    s = scales.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    return q, s


def pack_sel_bits(bits_slot: torch.Tensor, gated: bool, gpad: int) -> torch.Tensor:
    """Pack per-slot selection bits [gpad * 96, K] (0/1) -> [gpad, K/32, 384] u8 in k_conv_h32 (G, chunk, wi, lane) layout."""
    dev = bits_slot.device
    K = bits_slot.shape[1]
    NC = K // 32
    bk = bits_slot.to(torch.int64).view(gpad, 96, NC, 32)
    sh = torch.arange(32, dtype=torch.int64, device=dev)
    wsel = (bk << sh).sum(-1)
    ls = lane_slot(gated).to(dev)
    sel = wsel[:, ls.view(-1), :].view(gpad, 3, 32, NC).permute(0, 3, 1, 2).contiguous()
    return _i32(sel).view(torch.uint8).view(gpad, NC, 384).contiguous()


def unpack_sel_bits(sel: torch.Tensor, gated: bool, gpad: int) -> torch.Tensor:
    """Inverse of pack_sel_bits: [gpad, K/32, 384] u8 -> [gpad * 96, K] uint8 (0/1)."""
    dev = sel.device
    NC = sel.shape[1]
    sel_i32 = sel.contiguous().view(torch.int32).view(gpad, NC, 3, 32)
    ls = lane_slot(gated).to(dev)
    wsel = torch.empty((gpad, 96, NC), dtype=torch.int32, device=dev)
    wsel[:, ls.view(-1), :] = sel_i32.permute(0, 2, 3, 1).reshape(gpad, 96, NC)
    sh = torch.arange(32, dtype=torch.int64, device=dev)
    bits = ((wsel.to(torch.int64)[..., None] >> sh) & 1).reshape(gpad * 96, NC * 32)
    return bits.to(torch.uint8).contiguous()


def unpack_i4_packed(P: torch.Tensor, gpad: int) -> tuple[torch.Tensor, bool]:
    """Invert (1, 256) packed rows P [64 * gpad, K] fp16 -> (c_slot [gpad * 96, K] int64, ok bool)."""
    K = P.shape[1]
    Pv = P.view(gpad, 32, 2, K).to(torch.float64)
    n1, n2 = Pv[:, :, 0], Pv[:, :, 1]
    c2 = torch.round(n1 / 256.0)
    c1 = n1 - 256.0 * c2
    c3 = torch.round(n2 / 256.0)
    c2_chk = n2 - 256.0 * c3
    c = torch.stack([c1, c2, c3], dim=2).view(gpad * 96, K)
    c_i64 = torch.round(c).to(torch.int64)
    ok = bool(
        (c2_chk == c2).all()
        and (c == c_i64.to(torch.float64)).all()
        and (c_i64.abs() <= 7).all()
    )
    return c_i64, ok


def pack_i4_reference(c_slot: torch.Tensor, gpad: int) -> torch.Tensor:
    """Pack integer codes c_slot [gpad * 96, K] in [-7, 7] into (1, 256) fp16 rows [64 * gpad, K]."""
    K = c_slot.shape[1]
    cv = c_slot.view(gpad, 32, 3, K).to(torch.float32)
    P1 = cv[:, :, 0] + 256.0 * cv[:, :, 1]
    P2 = cv[:, :, 1] + 256.0 * cv[:, :, 2]
    return torch.stack([P1, P2], dim=2).view(64 * gpad, K).to(torch.float16).contiguous()


def conv_h32_ref(
    codes: torch.Tensor,
    scales: torch.Tensor,
    ia_slot: torch.Tensor,
    bits_slot: torch.Tensor | None,
    gated: bool,
    gpad: int,
    chunk_rows: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch bit-exact replica of k_conv_h32 arithmetic -> (c_slot [gpad * 96, K] int64, P [64 * gpad, K] fp16)."""
    dev = codes.device
    N, K = codes.shape[0], codes.shape[1] * 2
    rows = slot_map_h32(N, gated, gpad).to(dev)
    c_slot = torch.zeros((gpad * 96, K), dtype=torch.int64, device=dev)
    valid = rows >= 0
    vidx = torch.nonzero(valid).view(-1)
    scales_u8 = scales.contiguous().view(torch.uint8)
    inv2_14 = float(2.0 ** -14)
    for i0 in range(0, vidx.numel(), chunk_rows):
        si = vidx[i0 : i0 + chunk_rows]
        ri = rows[si]
        q, s = nv_parts_f32(codes[ri], scales_u8[ri])
        sc = (s * ia_slot[si, None]).to(torch.float32)
        f = ((q * inv2_14).view(-1, K // 16, 16) * sc[..., None]).view(-1, K)
        u = fwht32_unnorm(f)
        if bits_slot is None:
            c = torch.round(u).clamp(-7.0, 7.0)
        else:
            c = (torch.floor(u) + bits_slot[si].to(torch.float32)).clamp(-7.0, 7.0)
        c_slot[si] = c.to(torch.int64)
    P = pack_i4_reference(c_slot, gpad)
    return c_slot, P


def _freivalds_from_codes(c_slot: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build random chkv [slots] int32 (kept nonzero even on padded slots so wrap errors leaking into
    partially-padded triples stay at 2^-32 miss rate) and chku [K] int32 = sum_{valid} chkv[s] * c_slot[s, k] mod 2^32."""
    dev = c_slot.device
    slots, K = c_slot.shape
    gen = torch.Generator(device="cpu").manual_seed(int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))
    v = torch.randint(0, 1 << 32, (slots,), generator=gen, dtype=torch.int64).to(dev)
    u = torch.zeros(K, dtype=torch.int64, device=dev)
    idx = torch.nonzero(valid).view(-1)
    for i0 in range(0, idx.numel(), 2048):
        ii = idx[i0 : i0 + 2048]
        u += (v[ii, None] * c_slot[ii]).sum(0)
    return _i32(v), _i32(u)


# ------------------------------------------------------------------------------------------------ array builders
@torch.no_grad()
def arrays_from_alpha(
    alpha_row: torch.Tensor,
    g: float,
    N: int,
    K: int,
    gated: bool,
    sel: torch.Tensor | None = None,
    bn_groups: int = 3,
    guard: bool = True,
    w_prepacked: torch.Tensor | None = None,
    X32=None,
) -> dict:
    """Fast load-time builder from precomputed per-channel alpha [N] f32 (+ optional prepacked sel [gpad, K/32, 384] u8
    saved by gen_h32_planes.py). Builds exact Freivalds chkv/chku by running convert_h32 once on w_prepacked."""
    dev = w_prepacked.device if w_prepacked is not None else alpha_row.device
    ntile, groups, gpad = geometry(N, K, gated, bn_groups)
    gpad2 = -(-groups // 2) * 2
    gpad_alloc = max(gpad, gpad2)
    alpha_row = alpha_row.to(dev, dtype=torch.float32).contiguous()
    assert alpha_row.numel() == N, f"alpha length {alpha_row.numel()} != {N}"
    ia_row = torch.where(
        alpha_row > 0,
        (16384.0 / (torch.tensor(float(g), dtype=torch.float64, device=dev) * alpha_row.clamp(min=1e-30).double() * math.sqrt(32.0))).to(
            torch.float32
        ),
        torch.zeros((), device=dev, dtype=torch.float32),
    ).contiguous()
    rows = slot_map_h32(N, gated, gpad_alloc).to(dev)
    valid = rows >= 0
    safe_rows = rows.clamp(min=0)
    ae_slot = torch.where(valid, alpha_row[safe_rows], torch.zeros((), device=dev, dtype=torch.float32)).contiguous()
    ia_slot = torch.where(valid, ia_row[safe_rows], torch.zeros((), device=dev, dtype=torch.float32)).contiguous()

    plane = sel is not None
    if plane:
        sel = sel.to(dev, dtype=torch.uint8).contiguous()
        assert sel.shape[1:] == (K // 32, 384) and sel.shape[0] >= groups, f"sel shape {tuple(sel.shape)} invalid for groups={groups}"
        if sel.shape[0] < gpad_alloc:
            sel = torch.cat([sel, torch.zeros((gpad_alloc - sel.shape[0], K // 32, 384), dtype=torch.uint8, device=dev)], dim=0)
    else:
        sel = torch.empty(0, dtype=torch.uint8, device=dev)

    out = {
        "sel": sel,
        "ia": ia_slot,
        "ae": ae_slot,
        "ntile": ntile,
        "groups": groups,
        "gpad": gpad,
        "gpad2": gpad2,
        "N": N,
        "K": K,
        "gated": gated,
        "plane": plane,
    }
    if guard:
        assert w_prepacked is not None and w_prepacked.is_cuda, "arrays_from_alpha(guard=True) requires CUDA w_prepacked"
        X = ext() if X32 is None else X32
        P_tmp = torch.empty((64 * gpad_alloc, K), dtype=torch.float16, device=dev)
        X.convert_h32(w_prepacked, sel if plane else None, ia_slot, P_tmp, ntile, groups, K, plane, gated, 2)
        c_slot, ok = unpack_i4_packed(P_tmp, gpad_alloc)
        assert ok, "arrays_from_alpha: unpack_i4_packed failed on k_conv_h32 output"
        del P_tmp
        chkv, chku = _freivalds_from_codes(c_slot, valid)
        del c_slot
        out["chkv"] = chkv
        out["chku"] = chku
    return out


@torch.no_grad()
def build_h32_arrays(
    codes: torch.Tensor,
    scales: torch.Tensor,
    g: float,
    gated: bool,
    bf16: torch.Tensor | None = None,
    bn_groups: int = 3,
    guard: bool = True,
    w_prepacked: torch.Tensor | None = None,
    X32=None,
) -> dict:
    """Build H_32 arrays directly from NVFP4 checkpoint tensors (zero-plane h32_nv when bf16 is None, or 1-bit h32_pl
    when bf16 is given). Uses GPU convert_h32 once at load for bit-exact Freivalds chkv/chku."""
    dev = codes.device
    N, K = int(codes.shape[0]), int(codes.shape[1]) * 2
    ntile, groups, gpad = geometry(N, K, gated, bn_groups)
    scales_u8 = scales.contiguous().view(torch.uint8)
    plane = bf16 is not None
    if plane:
        assert tuple(bf16.shape) == (N, K), f"bf16 shape {tuple(bf16.shape)} != {(N, K)}"

    CH = 1024
    alpha_chunks = []
    if not plane:
        inv_g = float(1.0 / float(g))
        for r0 in range(0, N, CH):
            r1 = min(N, r0 + CH)
            q, s = nv_parts_f32(codes[r0:r1], scales_u8[r0:r1])
            Vnv = (q.view(r1 - r0, K // 16, 16) * (s * inv_g)[..., None]).view(r1 - r0, K)
            alpha_chunks.append(h32_alpha(fwht32(Vnv)))
    else:
        for r0 in range(0, N, CH):
            r1 = min(N, r0 + CH)
            Vb = bf16[r0:r1].to(dev, non_blocking=True).float()
            alpha_chunks.append(h32_alpha(fwht32(Vb)))
    alpha_row = torch.cat(alpha_chunks, dim=0).to(torch.float32).contiguous()

    sel = None
    bits_slot = None
    if plane:
        ia_row = torch.where(
            alpha_row > 0,
            (16384.0 / (torch.tensor(float(g), dtype=torch.float64, device=dev) * alpha_row.clamp(min=1e-30).double() * math.sqrt(32.0))).to(
                torch.float32
            ),
            torch.zeros((), device=dev, dtype=torch.float32),
        )
        inv2_14 = float(2.0 ** -14)
        bits_row = torch.zeros((N, K), dtype=torch.uint8, device=dev)
        for r0 in range(0, N, CH):
            r1 = min(N, r0 + CH)
            q, s = nv_parts_f32(codes[r0:r1], scales_u8[r0:r1])
            sc = (s * ia_row[r0:r1, None]).to(torch.float32)
            f = ((q * inv2_14).view(r1 - r0, K // 16, 16) * sc[..., None]).view(r1 - r0, K)
            u_nv = fwht32_unnorm(f)
            Vb = bf16[r0:r1].to(dev, non_blocking=True).float()
            a_ch = alpha_row[r0:r1]
            u_bf = torch.where(a_ch[:, None] > 0, fwht32(Vb) / a_ch.clamp(min=1e-30)[:, None], torch.zeros_like(u_nv))
            fl = torch.floor(u_nv)
            c0 = fl.clamp(-7.0, 7.0)
            c1 = (fl + 1.0).clamp(-7.0, 7.0)
            bit = ((c1 - u_bf).abs() < (c0 - u_bf).abs()) & (a_ch[:, None] > 0)
            bits_row[r0:r1] = bit.to(torch.uint8)
        rows = slot_map_h32(N, gated, gpad).to(dev)
        valid = rows >= 0
        bits_slot = torch.zeros((gpad * 96, K), dtype=torch.uint8, device=dev)
        bits_slot[valid] = bits_row[rows[valid]]
        sel = pack_sel_bits(bits_slot, gated, gpad)
        del bits_row

    if codes.is_cuda:
        X = ext() if X32 is None else X32
        w_pre = w_prepacked if w_prepacked is not None else (X.prepack(codes.contiguous(), scales_u8, gated) if guard else None)
        out = arrays_from_alpha(alpha_row, g, N, K, gated, sel=sel, bn_groups=bn_groups, guard=guard, w_prepacked=w_pre, X32=X)
        if w_prepacked is None and w_pre is not None:
            del w_pre
    else:
        out = arrays_from_alpha(alpha_row, g, N, K, gated, sel=sel, bn_groups=bn_groups, guard=False)
        if guard:
            rows = slot_map_h32(N, gated, gpad).to(dev)
            c_slot, _ = conv_h32_ref(codes, scales_u8, out["ia"], bits_slot, gated, gpad)
            out["chkv"], out["chku"] = _freivalds_from_codes(c_slot, rows >= 0)
            del c_slot
    if bits_slot is not None:
        del bits_slot
    return out
