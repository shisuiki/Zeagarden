#!/usr/bin/env python3
"""test_draftscreen.py - one V100 (gpulock one/gpu N): the draft-head screen of q27fuse/draftscreen.py.

  pack                        (CPU) pack_codes against the bit layout documented in csrc/draft_screen.cu
  cache DDIR                  the codes cache: the build path taken when it is missing (load_or_build on an empty
                              directory builds, saves, equals build_codes), the load path, an unwritable directory,
                              the cache key, the default-on switch
  layout                      accumulator fragment layouts of mma.m8n8k4 (f32 and f16), as the kernels assume them
  check DDIR TABLES           rank 0: codes from dense_rank0.pt; screen keys vs a torch reference of the same arithmetic;
                              merge ids vs reference top-KEEP; fused rerank (FP16 rows and FP8 rows) vs 1Cat's Triton
                              rerank; FP8 rows == dense rows (checkpoint)
  recall DDIR TABLES          all 4 ranks, all dumped draft rows: draft token of the screen head (kernels as served) vs
                              1Cat's recorded token, for NC x KEEP x ACC16 variants
  bench DDIR TABLES [R ...]        per-pass times vs 1Cat's refined head (NVFP4 GEMM + select + rerank), CUDA graphs of 20 calls,
                              >= 1 s of the same graph replayed before timing
Set DDIR and TABLES to optional offline dump directories if running real-dump checks.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from q27fuse import draftscreen as DS  # noqa: E402

dev = torch.device("cuda")
WS = torch.zeros(128, dtype=torch.int64, device=dev)
QG = torch.empty(64 * 5120, dtype=torch.float16, device=dev)   # q' workspace of the 16-query kernel (9+ rows)
QG0 = torch.empty(0, dtype=torch.float16, device=dev)
VS, HD, NR = 62080, 5120, 4
CKPT = os.environ.get("Q27_MODEL_CKPT", "Qwen3.8-27B-NVFP4/model.safetensors")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_dense(ddir, r):
    x = torch.load(os.path.join(ddir, f"dense_rank{r}.pt"), weights_only=False)
    assert x["start"] == r * VS
    return x["dense"].to(dev), x.get("global_scale")


def build(dense):
    t0 = time.time()
    codes, rowf, rowpb = DS.build_codes(lambda b, e: dense[b:e], dense.shape[0], dev)
    torch.cuda.synchronize()
    return (codes, rowf, rowpb), time.time() - t0


def cmd_layout():
    out = DS.ext().layout_probe().cpu()          # [type f32/f16][run row/col][lane][reg]
    for ti, tn in enumerate(("f32", "f16")):
        print(tn)
        for l in range(32):
            rows = [int(out[ti, 0, l, r]) for r in range(8)]
            cols = [int(out[ti, 1, l, r]) for r in range(8)]
            print(f"  lane {l:2d}: " + " ".join(f"({(a - 1) % 10},{(b - 1) % 10}|q{(a - 1) // 10}{(b - 1) // 10})"
                                               for a, b in zip(rows, cols)))
    # what the kernels assume
    bad = 0
    for l in range(32):
        q = (l >> 2) & 3
        for r in range(8):
            a, b = int(out[0, 0, l, r]), int(out[0, 1, l, r])
            er = (l & 1) + 2 * ((r >> 1) & 1) + 4 * ((l >> 4) & 1)
            ec = 2 * ((l >> 1) & 1) + (r & 1) + 4 * (r >> 2)
            bad += (a != er + 1 + 10 * q) or (b != ec + 1 + 10 * q)
            a, b = int(out[1, 0, l, r]), int(out[1, 1, l, r])
            er = (l & 3) + 4 * (l >> 4)
            bad += (a != er + 1 + 10 * q) or (b != r + 1 + 10 * q)
    print("layout assumptions:", "OK" if bad == 0 else f"{bad} MISMATCHES")


def cmd_pack():
    V = 64
    neg = torch.rand(V, 5120, generator=torch.Generator().manual_seed(0)) < 0.5
    c = DS.pack_codes(neg)                       # [u, g, lane, t] int32
    assert tuple(c.shape) == (V // 8, 10, 32, 4), c.shape
    cw = c.to(torch.int64) & 0xFFFFFFFF
    bad = 0
    for u in range(V // 8):
        for g in range(10):
            for l in range(32):
                q = (l >> 2) & 3
                row = 8 * u + (l & 3) + 4 * (l >> 4)
                for t in range(4):
                    w = int(cw[u, g, l, t])
                    for p in range(32):
                        m, hi = p & 15, p >> 4
                        k = 512 * g + 16 * (8 * t + m // 2) + 4 * q + 2 * (m & 1) + hi
                        bad += ((w >> p) & 1) != int(neg[row, k])
    print("pack mismatches:", bad, "PASS" if bad == 0 else "FAIL", flush=True)
    return bad == 0


def cmd_cache(ddir):
    import tempfile
    import types
    dense, _ = load_dense(ddir, 0)
    w8, s8 = DS._load_fp8_rows(CKPT, 0, VS, dev)
    (c0, f0, p0), _ = build(dense)
    res = {}
    tmp = os.environ.get("DS_TEST_TMP", os.path.dirname(os.path.abspath(__file__)))
    calls = []

    def dense_fn():
        calls.append(1)
        return dense

    with tempfile.TemporaryDirectory(dir=tmp, prefix="ds_cache_test_") as td:
        cfile = os.path.join(td, "key", "shard_0_62080.pt")       # the directory does not exist yet
        a = DS.load_or_build(cfile, dense_fn, w8, s8, dev)
        res["missing cache -> built, saved"] = a["cache"] == "built" and os.path.exists(cfile) and len(calls) == 1
        res["built == build_codes"] = all(torch.equal(a[k], v) for k, v in (("codes", c0), ("rowf", f0), ("rowpb", p0)))
        res["fp8 rows == dense rows"] = a["exact_fp8"]
        res["no temporary files left"] = not [f for f in os.listdir(os.path.dirname(cfile)) if ".tmp" in f]
        b = DS.load_or_build(cfile, dense_fn, w8, s8, dev)
        res["present cache -> loaded, no rebuild"] = b["cache"] == "loaded" and len(calls) == 1
        res["loaded == built"] = (all(torch.equal(a[k], b[k]) for k in ("codes", "rowf", "rowpb"))
                                  and b["exact_fp8"] == a["exact_fp8"] and b["codes"].is_cuda)
    with tempfile.NamedTemporaryFile(dir=tmp) as tf:              # a directory below a regular file cannot be made
        c = DS.load_or_build(os.path.join(tf.name, "key", "shard.pt"), lambda: dense, w8, s8, dev)
        res["unwritable cache -> built anyway"] = c["cache"] == "built" and torch.equal(c["codes"], c0)
    cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(model=os.path.dirname(CKPT)))
    cdir, fp, key = DS._ckpt_key(cfg, 0, VS)
    res["cache key: checkpoint file, dir under Q27_DS_CACHE, per shard"] = (
        fp == os.path.realpath(CKPT) and cdir == os.path.join(DS.CACHE, key) and DS._ckpt_key(cfg, VS, 2 * VS)[2] != key)
    old = os.environ.pop("Q27_DS", None)
    try:
        on = DS.enabled()
        os.environ["Q27_DS"] = "0"
        off = not DS.enabled()
    finally:
        os.environ.pop("Q27_DS", None)
        if old is not None:
            os.environ["Q27_DS"] = old
    res["default on, Q27_DS=0 off"] = on and off
    for k, v in res.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}", flush=True)
    print("cache test:", "PASS" if all(res.values()) else "FAIL", f"(default cache dir {DS.CACHE})", flush=True)
    return all(res.values())


def ref_screen(h, codes_src_dense, D, H, rowf):
    """torch reference of the screen arithmetic: FP16 rotated query, +-1 codes, FP32 sums, x f -> FP16 keys"""
    q = DS.rotate(h, D, H).half().float()                      # [R, 5120]
    r = DS.rotate(codes_src_dense, D, H)
    s = torch.where(r < 0, -1.0, 1.0)
    S = (q @ s.T) * rowf.unsqueeze(0)                          # [R, V]
    return S


def ord16(t):
    b = t.half().view(torch.int16).to(torch.int32) & 0xFFFF
    o = torch.where((b & 0x8000) != 0, (~b) & 0xFFFF, b ^ 0x8000)
    o = torch.where((b & 0x7FFF) == 0, torch.full_like(o, 0x8000), o)
    return o


def cmd_check(ddir, tables):
    X = DS.ext()
    dense, _ = load_dense(ddir, 0)
    (codes, rowf, rowpb), tb = build(dense)
    log(f"codes built in {tb:.1f} s: {codes.numel() * 4 / 1e6:.1f} MB")
    T = torch.load(tables, weights_only=False)
    Hd = T["drafts"]["h"]
    D, Hm = DS.rotation_signs().to(dev), DS.hadamard().to(dev)
    db = DS.sign_bits(D)
    G = DS.num_ctas()
    w8, s8 = DS._load_fp8_rows(CKPT, 0, VS, dev)
    ref8 = torch.cat([(w8[b:b + 8192].view(torch.float8_e4m3fn).float() * s8[b:b + 8192, None]).half()
                      for b in range(0, VS, 8192)])
    nbits = int((dense.view(torch.int16) != ref8.view(torch.int16)).sum())
    nval = int((dense.float() != ref8.float()).sum())
    log(f"checkpoint fp16(fp8 x scale) vs dense rows (rank 0): {nbits} elements differ in bits, {nval} in value")
    del ref8
    from vllm.model_executor.models.qwen3_5_mtp_refined_head import _rerank_candidates
    for R in (1, 2, 3, 8, 9, 40):
        h = Hd[1000:1000 + R].to(dev).contiguous()
        S = ref_screen(h, dense, D, Hm, rowf)
        for pc in ((True, False) if R <= 4 else (False,)):
            for nc in (8, 16):
                cand = torch.empty(R, G, nc, dtype=torch.int32, device=dev)
                X.screen(h, db, codes, rowf, rowpb, cand, G, nc, DS.WARPS, pc, 0, QG)
                split = ""
                if pc:      # the popcount kernel's balanced warp split vs the plain split (dbg 32): bitwise equal keys
                    cand2 = torch.empty_like(cand)
                    X.screen(h, db, codes, rowf, rowpb, cand2, G, nc, DS.WARPS, pc, 32, QG)
                    split = f"; balanced split == plain split bitwise {bool(torch.equal(cand, cand2))}"
                torch.cuda.synchronize()
                ck = cand.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
                cid = 65535 - (ck & 0xFFFF)
                cs = ck >> 16
                units = VS // 8
                miss = 0
                maxd = 0.0
                for r in range(R):
                    keys = (ord16(S[r]).to(torch.int64) << 16) | (65535 - torch.arange(VS, device=dev))
                    for c in range(G):
                        lo, hi = c * units // G * 8, (c + 1) * units // G * 8
                        refk = torch.topk(keys[lo:hi], nc).values
                        miss += nc - int(torch.isin(cid[r, c], 65535 - (refk & 0xFFFF)).sum())
                    maxd = max(maxd, float((cs[r].flatten() - ord16(S[r, cid[r].flatten()])).abs().max()))
                tag = f"R={R:2d} {'pc ' if pc else 'mma'} nc={nc:2d}"
                for keep in (16, 32):
                    ids = torch.empty(R, keep, dtype=torch.int32, device=dev)
                    X.merge(cand, G, nc, keep, h, 0, dense, s8, 0, ids, ids, WS)
                    flat = ck.view(R, -1)
                    topref = torch.topk(flat, keep, dim=1).values
                    idref = (65535 - (topref & 0xFFFF)).to(torch.int32)
                    okids = all(set(ids[r].tolist()) == set(idref[r].tolist()) for r in range(R))
                    p1 = _rerank_candidates(h, dense, ids, 0)
                    p2 = torch.empty(R, 2, dtype=torch.float32, device=dev)
                    X.merge(cand, G, nc, keep, h, 1, dense, s8, 0, p2, p2, WS)
                    p3 = torch.empty(R, 2, dtype=torch.float32, device=dev)
                    X.merge(cand, G, nc, keep, h, 2, w8, s8, 0, p3, p3, WS)
                    torch.cuda.synchronize()
                    same2 = int((p1[:, 1] == p2[:, 1]).sum())
                    same3 = int((p1[:, 1] == p3[:, 1]).sum())
                    sd2 = float((p1[:, 0] - p2[:, 0]).abs().max())
                    sd3 = float((p2[:, 0] - p3[:, 0]).abs().max())
                    E = (h.float() @ dense.float().T).half().float()
                    exact = int((E.gather(1, p1[:, 1].long().unsqueeze(1)).squeeze(1) == E.max(1).values).sum())
                    print(f"{tag} keep={keep}: per-CTA misses vs FP16-q' ref {miss} (of {R * G * nc}), max key diff "
                          f"{maxd:.0f}; merge ids == ref {okids}; fused fp16 rerank id == triton {same2}/{R} "
                          f"(max |ds| {sd2:.3g}), fp8 == triton {same3}/{R} (fp8 vs fp16 |ds| {sd3:.3g}); "
                          f"pick is shard argmax {exact}/{R}{split}", flush=True)


def cmd_recall(ddir, tables):
    """variants: (kernel, rows per call, nc, keep); pc runs one row per call"""
    X = DS.ext()
    T = torch.load(tables, weights_only=False)
    Hd, tok = T["drafts"]["h"], T["drafts"]["tok"]
    n = Hd.shape[0]
    G = DS.num_ctas()
    variants = [("mma", 32, 8, 32), ("mma", 32, 16, 32), ("pc", 1, 8, 32), ("pc", 1, 16, 32), ("pc2", 2, 8, 32)]
    best = {v: [] for v in variants}
    for r in range(NR):
        dense, _ = load_dense(ddir, r)
        (codes, rowf, rowpb), tb = build(dense)
        w8, s8 = DS._load_fp8_rows(CKPT, r * VS, (r + 1) * VS, dev)
        db = DS.sign_bits(DS.rotation_signs()).to(dev)
        log(f"rank {r}: codes in {tb:.1f} s")
        for v in variants:
            kern, CH, nc, keep = v
            outs = []
            for b in range(0, n, CH):
                h = Hd[b:b + CH].to(dev).contiguous()
                R = h.shape[0]
                cand = torch.empty(R, G, nc, dtype=torch.int32, device=dev)
                X.screen(h, db, codes, rowf, rowpb, cand, G, nc, DS.WARPS, kern != "mma", 0, QG)
                p = torch.empty(R, 2, dtype=torch.float32, device=dev)
                X.merge(cand, G, nc, keep, h, 2, w8, s8, r * VS, p, p, WS)
                outs.append(p)
            best[v].append(torch.cat(outs).cpu())
            log(f"rank {r} {v} done")
        del dense, codes, rowf, rowpb, w8, s8
        torch.cuda.empty_cache()
    res = {}
    for v in variants:
        P = torch.stack(best[v])                 # [4, n, 2]
        ri = P[:, :, 0].argmax(0)                # first max -> lowest rank on ties (= lowest id: ranks are id-ordered)
        t = P[ri, torch.arange(n), 1].long()
        agree = float((t == tok).float().mean())
        res[v] = t
        print(f"{v[0]:4s} rows/call {v[1]:2d} nc={v[2]:2d} keep={v[3]:2d}: same token as 1Cat {agree:.5f} "
              f"({int((t != tok).sum())} of {n} differ)", flush=True)
    torch.save({"variants": variants, "tokens": res, "tok_1cat": tok}, os.environ.get("DS_RECALL_OUT", "recall.pt"))


def graph_time(fn, reps=20, secs=1.0):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    t0 = time.time()
    while time.time() - t0 < secs:
        g.replay()
        torch.cuda.synchronize()
    out = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / (5 * reps) * 1e3)
    return min(out), max(out)


def cmd_bench(ddir, Rs):
    X = DS.ext()
    dense, gs = load_dense(ddir, 0)
    (codes, rowf, rowpb), _ = build(dense)
    w8, s8 = DS._load_fp8_rows(CKPT, 0, VS, dev)
    db = DS.sign_bits(DS.rotation_signs()).to(dev)
    G = DS.num_ctas()
    # 1Cat's refined head on rank 0's rows, prepared as MTPRefinedHead.__init__
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.models import qwen3_5_mtp_refined_head as RH
    rows, width = dense.shape
    gsc = float(dense.abs().max().float().clamp_min(1e-12) / 2688)
    packed = torch.empty(rows, width // 2, device=dev, dtype=torch.uint8)
    raw = torch.empty(rows, width // 16, device=dev, dtype=torch.float8_e4m3fn)
    th = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
    for b0 in range(0, rows, 512):
        v = dense[b0:b0 + 512].float().reshape(-1, width // 16, 16)
        sc = (v.abs().amax(-1) / 6 / gsc).to(torch.float8_e4m3fn)
        eff = (sc.float() * gsc).half().float().clamp_min(1e-12)
        code = torch.bucketize((v.abs() / eff.unsqueeze(-1)).contiguous(), th).to(torch.uint8)
        code |= (v < 0).to(torch.uint8) * 8
        flat = code.flatten(1)
        packed[b0:b0 + 512] = flat[:, ::2] | (flat[:, 1::2] << 4)
        raw[b0:b0 + 512] = sc
    nc4, nsc = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw)
    if hasattr(torch.ops._C, "nvfp4_qpn2_bundle_sm70"):
        nc4, nsc = torch.ops._C.nvfp4_qpn2_bundle_sm70(nc4, nsc)
    del packed, raw
    T = torch.load(sys.argv[3], weights_only=False)["drafts"]["h"]
    for R in Rs:
        h = T[:R].to(dev).contiguous()
        ref = lambda: RH._refined_head(h, nc4, nsc, gsc, dense, 0)
        sk = 8 if R <= 16 else 16
        logits = torch.empty(R, rows, device=dev, dtype=torch.float16)
        gemm = lambda: sm70_ops.nvfp4_qpn2_gemm_sm70_out(logits, h, nc4, nsc, gsc, sk, 2)
        res = {"1cat_head": graph_time(ref), "1cat_gemm": graph_time(gemm)}
        for w in (12, 16):
            for pc in ((True, False) if R <= 4 else (False,)):
                nc = 8
                cand = torch.empty(R, G, nc, dtype=torch.int32, device=dev)
                p = torch.empty(R, 2, dtype=torch.float32, device=dev)
                scr = lambda: X.screen(h, db, codes, rowf, rowpb, cand, G, nc, w, pc, 0, QG)
                res[f"W{w} {'pc' if pc else 'mma'} screen"] = graph_time(scr)
                if pc and w == 12:
                    res["W12 pc plain split"] = graph_time(
                        lambda: X.screen(h, db, codes, rowf, rowpb, cand, G, nc, w, pc, 32, QG))
        cand = torch.empty(R, G, 8, dtype=torch.int32, device=dev)
        p = torch.empty(R, 2, dtype=torch.float32, device=dev)
        if R > 8:
            res["W12 mma 8-query tiles"] = graph_time(lambda: X.screen(h, db, codes, rowf, rowpb, cand, G, 8, 12, False,
                                                                        0, QG0))
        X.screen(h, db, codes, rowf, rowpb, cand, G, 8, 12, R <= 4, 0, QG)
        res["merge fp8"] = graph_time(lambda: X.merge(cand, G, 8, 32, h, 2, w8, s8, 0, p, p, WS))
        res["merge fp16"] = graph_time(lambda: X.merge(cand, G, 8, 32, h, 1, dense, s8, 0, p, p, WS))
        for pcm in (0, 2, 4):
            res[f"ours pcmax{pcm}"] = graph_time(lambda: DS.draft_head(h, codes, rowf, rowpb, db, w8, s8, WS, 0, 2, G, 8, 32,
                                                                         12, pcm))
        log(f"R={R:2d}: " + ", ".join(f"{k} {v[0]:.1f}" for k, v in res.items()))


def cmd_parts(ddir, Rs):
    """screen kernel with the main loop skipped (dbg 1), the prologue skipped (dbg 2), both (3); merge ids-only vs fused"""
    X = DS.ext()
    dense, _ = load_dense(ddir, 0)
    (codes, rowf, rowpb), _ = build(dense)
    w8, s8 = DS._load_fp8_rows(CKPT, 0, VS, dev)
    db = DS.sign_bits(DS.rotation_signs()).to(dev)
    G = DS.num_ctas()
    T = torch.load(sys.argv[3], weights_only=False)["drafts"]["h"]
    for R in Rs:
        h = T[:R].to(dev).contiguous()
        res = {}
        cand = torch.empty(R, G, 8, dtype=torch.int32, device=dev)
        for pc in ((True, False) if R <= 4 else (False,)):
            for d in (0, 1, 2, 3):
                res[f"{'pc' if pc else 'mma'} dbg{d}"] = graph_time(
                    lambda: X.screen(h, db, codes, rowf, rowpb, cand, G, 8, 12, pc, d, QG))
        X.screen(h, db, codes, rowf, rowpb, cand, G, 8, 12, R <= 4, 0, QG)
        ids = torch.empty(R, 32, dtype=torch.int32, device=dev)
        p = torch.empty(R, 2, dtype=torch.float32, device=dev)
        res["merge ids"] = graph_time(lambda: X.merge(cand, G, 8, 32, h, 0, w8, s8, 0, ids, ids, WS))
        res["merge fp8"] = graph_time(lambda: X.merge(cand, G, 8, 32, h, 2, w8, s8, 0, p, p, WS))
        e = torch.empty(1, device=dev)
        res["empty kernel"] = graph_time(lambda: e.add_(1))
        log(f"R={R:2d}: " + ", ".join(f"{k} {v[0]:.1f}" for k, v in res.items()))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "pack":
        sys.exit(0 if cmd_pack() else 1)
    elif c == "cache":
        sys.exit(0 if cmd_cache(sys.argv[2]) else 1)
    elif c == "layout":
        cmd_layout()
    elif c == "check":
        cmd_check(sys.argv[2], sys.argv[3])
    elif c == "recall":
        cmd_recall(sys.argv[2], sys.argv[3])
    elif c == "parts":
        cmd_parts(sys.argv[2], [int(a) for a in sys.argv[4:]] or [1, 2, 8])
    elif c == "bench":
        Rs = [int(a) for a in sys.argv[4:]] or [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40]
        cmd_bench(sys.argv[2], Rs)
