"""pf_format.py (track prefill) - the packed prefill weight format ("ours") and its derivation from the NVFP4 checkpoint
(+ an optional 1-bit selection plane), in torch (CPU or GPU), bit-exact replicas of the conversion kernel's arithmetic.

FORMAT ("ours", per weight matrix W [N, K], K-constant scales so the 3-in-2 FP4 chain packs and decodes once):
  W_ours[n, k] = alpha'[n] * 2^rel[t(n), k // 16] * E2M1(c[n, k])
    c       E2M1 code (15 values 0, +-0.5 .. +-6)
    rel     0..2, shared by the 3 rows of a chain triple t per 16-k block (2 bits)
    alpha'  per-row FP32 (= alpha[n] * 2^emin[t]; folded into the GEMM epilogue)
  Triples: down: rows (3t, 3t+1, 3t+2); merged gate+up [gate 0..I-1 | up 0..I-1]: gate rows (3t..3t+2), up rows likewise.
  Groups of 96 decode rows = 3 consecutive 32-column decode tiles of mlp_fp4.cu's prepack (gate+up tile p = gate rows
  16p..16p+15 + up rows 16p..16p+15), so a group = 48 gate + 48 up rows (16 + 16 triples) or 96 down rows (32 triples).
  Rows past N are zero padding (one virtual decode tile for gate+up, two for down).

CONVERSION (per weight, in the decode kernels' representation v = E2M1(q) * 2^-14, exact fp16):
  r16  = fp16( f32( f32(s_E4M3 * ia[n]) * 2^-rel ) )      ia[n] = f32(1 / (g * alpha'[n])), per (row, 16-block)
  y16  = fp16( v * r16 )                                   (HMUL2, round to nearest)
  code = (bits(y16) + bias) & 0xFE00                       bias = 0x1FF * sel (plane mode) or 0x100 (no plane: nearest)
  code = code & 0x1000 ? (code & 0x8000) | 0x0E00 : code   saturation at E2M1 6 (valid while |y16| < 30 * 2^-14)
  -> fp16 bit pattern of E2M1(c) * 2^-14 (sign kept).
  The selection bit says floor (0) or ceil (1) in magnitude on the E2M1 grid; the encoder (from the original weights)
  stores the bit that reproduces its own code, or the nearer of the two when neither does.
"""
import math
import os
import torch

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
G1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
CKPT = os.environ.get("Q27_MODEL_CKPT", "")


# ------------------------------------------------------------------------------------------------ checkpoint
_LCACHE = {}


def load_layer(layer: int, rank: int = 0, tp: int = 4, path: str = CKPT):
    """rank shard of one NVFP4 MLP layer: {'gu': (codes [2Ir, H/2] u8, scales [2Ir, H/16] u8 (E4M3 bits), g),
    'down': (codes [H, Ir/2], scales [H, Ir/16], g)} in the plugin's merged order (gate rows then up rows)."""
    from safetensors import safe_open
    pre = f"model.language_model.layers.{layer}.mlp."
    if _LCACHE.get("key") != (layer, path):
        with safe_open(path, framework="pt", device="cpu") as f:
            _LCACHE["T"] = {(p, t): f.get_tensor(pre + p + "." + t) for p in ("gate_proj", "up_proj", "down_proj")
                            for t in ("weight_packed", "weight_scale", "weight_global_scale")}
        _LCACHE["key"] = (layer, path)
    T = _LCACHE["T"]
    I = T["gate_proj", "weight_packed"].shape[0]
    Ir = I // tp
    s = slice(rank * Ir, (rank + 1) * Ir)
    gc = torch.cat([T["gate_proj", "weight_packed"][s], T["up_proj", "weight_packed"][s]]).contiguous()
    gs = torch.cat([T["gate_proj", "weight_scale"][s], T["up_proj", "weight_scale"][s]]).contiguous().view(torch.uint8)
    g_gu = max(float(T["gate_proj", "weight_global_scale"].max()), float(T["up_proj", "weight_global_scale"].max()))
    assert abs(float(T["gate_proj", "weight_global_scale"].max()) - float(T["up_proj", "weight_global_scale"].max())) < 1e-6 * g_gu
    dc = T["down_proj", "weight_packed"][:, rank * Ir // 2:(rank + 1) * Ir // 2].contiguous()
    ds = T["down_proj", "weight_scale"][:, rank * Ir // 16:(rank + 1) * Ir // 16].contiguous().view(torch.uint8)
    g_d = float(T["down_proj", "weight_global_scale"].max())
    return {"gu": (gc, gs, g_gu), "down": (dc, ds, g_d)}


def e4m3_to_f32(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).to(torch.float32)


def nv_parts(codes: torch.Tensor, scales: torch.Tensor):
    """(q E2M1 values [N, K] fp32 with sign, s E4M3 values [N, K/16] fp32)"""
    lo, hi = (codes & 0xF).long(), (codes >> 4).long()
    q = torch.stack([E2M1.to(codes.device)[lo], E2M1.to(codes.device)[hi]], -1).reshape(codes.shape[0], -1)
    return q, e4m3_to_f32(scales)


def nv_deq(codes, scales, g) -> torch.Tensor:
    q, s = nv_parts(codes, scales)
    N, K = q.shape
    return (q.view(N, K // 16, 16) * s[..., None] / g).view(N, K)


# ------------------------------------------------------------------------------------------------ geometry
def triple_rows(N: int, gated: bool):
    """padded row count and the row index of each padded row slot (-1 = padding), in triple order: slot 3t+j"""
    if gated:
        I = N // 2
        Ip = -(-I // 3) * 3
        idx = torch.full((2 * Ip,), -1, dtype=torch.long)
        idx[:I] = torch.arange(I)
        idx[Ip:Ip + I] = torch.arange(I, 2 * I)
        return 2 * Ip, idx
    Np = -(-N // 3) * 3
    idx = torch.full((Np,), -1, dtype=torch.long)
    idx[:N] = torch.arange(N)
    return Np, idx


def to_slots(X: torch.Tensor, idx: torch.Tensor, fill=0.0):
    out = torch.full((idx.numel(),) + tuple(X.shape[1:]), fill, dtype=X.dtype, device=X.device)
    m = idx >= 0
    out[m] = X[idx[m].to(X.device)]
    return out


def from_slots(Xs: torch.Tensor, idx: torch.Tensor, N: int):
    out = torch.zeros((N,) + tuple(Xs.shape[1:]), dtype=Xs.dtype, device=Xs.device)
    m = idx >= 0
    out[idx[m].to(Xs.device)] = Xs[m]
    return out


# ------------------------------------------------------------------------------------------------ rounding helpers
def rnd_e2m1(x: torch.Tensor) -> torch.Tensor:
    """nearest E2M1 value (magnitude clamped to 6), ties to the even code (as torch.round's half-to-even on the grid)"""
    a = x.abs().clamp(max=6.0)
    # grid steps: 0.5 below 2, 1 in [2, 4), 2 in [4, 6]
    r = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2))
    return torch.copysign(r, x)


def rnd_g19(x: torch.Tensor) -> torch.Tensor:
    """nearest of {0, 0.5, 1, ..., 3, 4, 5, 6} (2x = {0..6, 8, 10, 12}: every chain position stays FP16-exact)"""
    a = x.abs().clamp(max=6.0)
    r = torch.where(a < 3, torch.round(a * 2) / 2, torch.round(a))
    return torch.copysign(r, x)


GRIDS = {"e2m1": rnd_e2m1, "g19": rnd_g19}


# ------------------------------------------------------------------------------------------------ encoder
def encode(V: torch.Tensor, gated: bool, spread: int = 2, nalpha: int = 16, emode: str = "amax", acands=(), grid: str = "e2m1"):
    """ours from values V [N, K] fp32 -> dict(alpha [Np] f32 (per slot, incl. 2^emin), rel [Np/3, K/16] int (0..spread),
    code [Np, K] E2M1 values, idx, Np). emode 'amax': e = ceil(log2(amax / 6)) per (triple, block); 'mse': per
    (triple, block) also try e - 1 (clipping the largest) and keep the lower squared error."""
    rnd = GRIDS[grid]
    N, K = V.shape
    Np, idx = triple_rows(N, gated)
    Vs = to_slots(V, idx)                                                       # [Np, K]
    B = K // 16
    Vb = Vs.view(Np, B, 16)
    amax_row_blk = Vb.abs().amax(-1)                                            # [Np, B]
    # alpha per row (fractional octave): per-row exponents for the search, as nvfp4_planes.py
    best_err, best_a = None, None
    cands = sorted({2.0 ** (j / nalpha) for j in range(nalpha)} | {c / 2 ** math.floor(math.log2(c)) for c in acands})
    for a in cands:
        e = torch.ceil(torch.log2(amax_row_blk / a / 6 + 1e-30))
        sc = a * torch.exp2(e)[..., None]
        err = ((rnd(Vb / sc) * sc - Vb) ** 2).sum((1, 2))
        if best_err is None:
            best_err, best_a = err, torch.full((Np,), a, device=V.device)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_a = torch.where(better, torch.full_like(best_a, a), best_a)
    alpha = best_a                                                              # [Np], in [1, 2)
    G = (Vb / alpha[:, None, None]).view(Np // 3, 3, B, 16)
    am = G.abs().amax((1, 3))                                                   # [Np/3, B]
    e = torch.ceil(torch.log2(am / 6 + 1e-30))
    zero = am == 0
    emax = torch.where(zero, torch.full_like(e, -1e9), e).amax(1, keepdim=True)
    emax = torch.where(torch.isinf(emax) | (emax < -1e8), torch.zeros_like(emax), emax)
    e = torch.where(zero, emax, e)
    e = torch.maximum(e, emax - spread)
    if emode == "mse":
        sc3 = lambda ee: (alpha.view(Np // 3, 3, 1, 1) * torch.exp2(ee)[:, None, :, None])
        e1 = torch.maximum(e - 1, emax - spread)
        er0 = ((rnd(G * alpha.view(Np // 3, 3, 1, 1) / sc3(e)) * sc3(e) - G * alpha.view(Np // 3, 3, 1, 1)) ** 2).sum((1, 3))
        er1 = ((rnd(G * alpha.view(Np // 3, 3, 1, 1) / sc3(e1)) * sc3(e1) - G * alpha.view(Np // 3, 3, 1, 1)) ** 2).sum((1, 3))
        e = torch.where(er1 < er0, e1, e)
    emin = e.amin(1, keepdim=True)
    rel = (e - emin).to(torch.int32)                                            # [Np/3, B]
    alpha_p = alpha * torch.exp2(emin).repeat_interleave(3, 0).view(-1)         # alpha' per slot
    sc = alpha_p[:, None, None] * torch.exp2(rel.to(torch.float32)).repeat_interleave(3, 0)[..., None]
    code = rnd(Vb / sc).view(Np, K)
    return {"alpha": alpha_p, "rel": rel, "code": code, "idx": idx, "Np": Np, "N": N, "K": K}


def e4m3_acands(g: float):
    """alpha candidates that make the NVFP4 -> ours ratio an exact power of two for blocks with E4M3 mantissa m"""
    return [(1 + m / 8) / g for m in range(8)]


def ours_values(enc, code=None) -> torch.Tensor:
    """dequantised W_ours [N, K] (fp32) from an encoding (optionally with replaced codes)"""
    Np, K = enc["Np"], enc["K"]
    c = enc["code"] if code is None else code
    sc = enc["alpha"][:, None, None] * torch.exp2(enc["rel"].to(torch.float32)).repeat_interleave(3, 0)[..., None]
    return from_slots((c.view(Np, K // 16, 16) * sc).view(Np, K), enc["idx"], enc["N"])


# ------------------------------------------------------------------------------------------------ decoder replica
def ratio16(s: torch.Tensor, ia: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """r16 [Np, B] = fp16(f32(f32(s * ia) * 2^-rel)); s [Np, B] f32 (E4M3 values), ia [Np] f32, rel [Np/3, B]"""
    r = (s * ia[:, None]).to(torch.float32)
    r = r * torch.exp2(-rel.to(torch.float32)).repeat_interleave(3, 0)
    return r.to(torch.float16)


def decode(qs: torch.Tensor, ss: torch.Tensor, ia: torch.Tensor, rel: torch.Tensor, bias: torch.Tensor | int):
    """replica of the conversion: qs [Np, K] E2M1 values (slots), ss [Np, B] E4M3 values, bias per weight (int16 tensor
    or scalar) -> (E2M1 code values [Np, K] fp32, raw fp16 patterns int32, valid mask)"""
    Np, K = qs.shape
    v = (qs * 2.0 ** -14).to(torch.float16)                                     # exact
    r = ratio16(ss, ia, rel)                                                     # [Np, B]
    y = (v.view(Np, K // 16, 16).to(torch.float32) * r[..., None].to(torch.float32)).to(torch.float16).view(Np, K)
    b = y.view(torch.int16).to(torch.int32) & 0xFFFF
    pat = (b + bias) & 0xFE00
    valid = (b & 0x7FFF) < 0x1E01                                                # y + bias stays below 0x2000
    sat = (pat & 0x1000) != 0                                                    # E2M1 > 6 -> saturate at 6
    pat = torch.where(sat, (pat & 0x8000) | 0x0E00, pat)
    val = (pat.to(torch.int16).view(torch.float16).to(torch.float32)) * 2.0 ** 14
    return val, pat, valid


def ia_of(alpha_p: torch.Tensor, g: float) -> torch.Tensor:
    return (1.0 / (torch.tensor(g, dtype=torch.float64) * alpha_p.to(torch.float64))).to(torch.float32)


def plane_bits(enc, qs, ss, ia, V_target_codes=None):
    """selection bits for the plane: per weight 0 (floor) or 1 (ceil) such that decode == enc code; misses take the
    nearer of the two decoded values to the encoder's unrounded value. Returns (bits int32 [Np, K], decoded codes, stats)"""
    c0, _, v0 = decode(qs, ss, ia, enc["rel"], 0)
    c1, _, v1 = decode(qs, ss, ia, enc["rel"], 0x1FF)
    tgt = enc["code"] if V_target_codes is None else V_target_codes
    hit0 = (c0 == tgt) & v0
    hit1 = (c1 == tgt) & v1
    bits = torch.where(hit0, 0, torch.where(hit1, 1, -1)).to(torch.int32)
    miss = bits < 0
    # misses: nearer valid candidate to the target code (in code units; the scale is common)
    d0 = torch.where(v0, (c0 - tgt).abs(), torch.full_like(c0, 1e9))
    d1 = torch.where(v1, (c1 - tgt).abs(), torch.full_like(c1, 1e9))
    bits = torch.where(miss, torch.where(d1 < d0, 1, 0).to(torch.int32), bits)
    dec = torch.where(bits == 1, c1, c0)
    inval = ~torch.where(bits == 1, v1, v0)
    return bits, dec, {"miss": int(miss.sum()), "invalid": int(inval.sum()), "n": qs.numel()}


def standin_bf16(codes, scales, g, seed=0) -> torch.Tensor:
    """Synthetic stand-in BF16 weights when the original BF16 checkpoint is unavailable: NVFP4 value + uniform noise inside the
    weight's NVFP4 rounding cell (zero cell +-0.25, top cell [5, 6]), rounded to bf16. Re-quantising it with the same
    block scales gives the checkpoint codes back (up to cell-boundary ties)."""
    q, s = nv_parts(codes, scales)
    N, K = q.shape
    lows = torch.tensor([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    highs = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    gen = torch.Generator().manual_seed(seed)
    iq = (q.abs()[..., None] == G1).float().argmax(-1)
    u = torch.rand(q.shape, generator=gen)
    lo, hi = lows[iq], highs[iq]
    mag = lo + u * (hi - lo)
    sign = torch.where(q == 0, torch.where(torch.rand(q.shape, generator=gen) < 0.5, -1.0, 1.0), torch.sign(q))
    sign = torch.where((q == 0) & (torch.signbit(q)), -1.0, sign)
    W = (sign * mag).view(N, K // 16, 16) * s[..., None] / g
    return W.view(N, K).to(torch.bfloat16).to(torch.float32)


def rel_err(A: torch.Tensor, R: torch.Tensor) -> float:
    return float(((A - R).double().pow(2).sum() / R.double().pow(2).sum()).sqrt())
