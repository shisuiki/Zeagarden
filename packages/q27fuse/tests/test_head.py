"""test_head.py (track head) - the unified FP8 LM head (q27fuse/csrc/fp8_head.cu) on one V100 against 1Cat's TurboMind
FP8 head (fp8_sm70_prepare + fp8_gemm_sm70_out, what the target head runs today) and an FP64 reference.

  python test_head.py check [rank]          correctness at 1..128 rows: error vs FP64, vs TurboMind element-wise,
                                            top-1 pairs = torch.max of our logits (incl. NaN / tie / -0 rows), graph replay
  python test_head.py bench [rows] [cfgs]   us per call, cold weights (NCOPY copies cycled), >= 1.2 s clock warm-up
  python test_head.py argmax TABLES.pt      global argmax on the dumped real verify rows (4 shards): ours vs TurboMind
                                            vs the server's chosen tokens
  python test_head.py rerank TABLES.pt      draftscreen's exact rerank reading the unified copy (QPN8 gather, merge
                                            rr 3) vs its own row-major FP8 copy (rr 2): pairs bitwise, merge time
Rows of the shard come from the checkpoint (lm_head.weight / weight_scale, rank r's 62,080 rows)."""
import os
import sys
import time
import torch
import vllm._C  # noqa: F401
import vllm._sm70_ops  # noqa: F401

from q27fuse import head as H

dev = torch.device("cuda")
CKPT = os.environ.get("Q27_MODEL_CKPT", "Qwen3.8-27B-NVFP4/model.safetensors")
VS, HD = 62080, 5120


def load_shard(r: int):
    from safetensors import safe_open
    with safe_open(CKPT, "pt", device="cpu") as f:
        w = f.get_slice("lm_head.weight")[r * VS:(r + 1) * VS]
        s = f.get_slice("lm_head.weight_scale")[r * VS:(r + 1) * VS]
    return w.to(dev).contiguous(), s.float().reshape(-1, 1).to(dev).contiguous()


def tm_prepare(w8, s):
    tw, ts, meta = torch.ops._C.fp8_sm70_prepare(w8, s, 128, False)
    return tw, ts, int(meta[0].item()), int(meta[1].item())


def tm_logits(x, tm):
    tw, ts, kld, qld = tm
    out = torch.empty(x.shape[0], VS, device=dev, dtype=torch.half)
    torch.ops._C.fp8_gemm_sm70_out(out, x, tw, ts, 128, kld, qld, False)
    return out


def warm(fn, secs=1.2):
    t0 = time.time()
    while time.time() - t0 < secs:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()


def timeit(fns, reps=40):
    """fns: list of callables (one per weight copy); us per call, back-to-back, copies cycled"""
    warm(lambda: [f() for f in fns])
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for i in range(reps):
        fns[i % len(fns)]()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / reps


def check(rank=0):
    w8, s = load_shard(rank)
    hc = H.HeadCopy.from_checkpoint(w8, s, vstart=rank * VS, nvalid=VS)
    tm = tm_prepare(w8, s)
    wref = (w8.float() * s).double()                                   # exact code x scale
    w16 = (w8.float() * s).half()                                      # TurboMind's weights (fp16(code x scale))
    print(f"shard {rank}: fp16(code x scale) == fp16 scale route: "
          f"{torch.equal(((w8.view(torch.uint8).view(torch.float8_e4m3fn).float()) * s).half(), w16)}", flush=True)
    ok = True
    torch.manual_seed(0)
    for M in (1, 2, 3, 5, 8, 9, 16, 17, 24, 32, 33, 40, 48, 60, 64, 65, 100, 128, 200):
        x = (torch.randn(M, HD, device=dev) * 0.8).half()
        ref = (x.double() @ wref.t())
        ref16 = (x.double() @ w16.double().t())                       # TurboMind's math in FP64
        a = tm_logits(x, tm)
        for ws in (1, 0):
            b = hc.logits(x, ws=ws)
            e_ref = ((b.double() - ref).norm() / ref.norm()).item()
            e_tm = ((b.double() - a.double()).norm() / a.double().norm()).item()
            e_r16 = ((b.double() - ref16).norm() / ref16.norm()).item()
            mx = (b.float() - a.float()).abs().max().item()
            ta = ((a.double() - ref).norm() / ref.norm()).item()
            pairs = hc.top1(x, ws=ws)
            v, i = b.float().max(dim=-1)
            good_pair = torch.equal(pairs[:, 1].long(), i + rank * VS) and torch.equal(pairs[:, 0], v)
            am = (b.float().argmax(-1) == a.float().argmax(-1)).float().mean().item()
            good = e_ref < 1e-3 and good_pair
            ok &= good
            print(f"M={M:3d} ws={ws}: err vs FP64 {e_ref:.2e} (TurboMind {ta:.2e}), vs TM-math FP64 {e_r16:.2e}, "
                  f"vs TM {e_tm:.2e} max|d| {mx:.2e}; top1 pairs == max(our logits) {good_pair}; argmax == TM {am:.4f}"
                  f"{'' if good else '  <-- FAIL'}", flush=True)
    # special rows: NaN, ties, -0 (top-1 rule = torch.max over our fp16 logits)
    M = 6
    x = (torch.randn(M, HD, device=dev) * 0.8).half()
    x[1] = 0                                   # all logits +-0: index 0
    x[2, 7] = float("nan")                     # NaN everywhere -> first index
    x[3] = x[3] * 1e-6                         # tiny values, many ties at fp16 resolution
    x[4, :] = 0; x[4, 100] = 1                 # logits = code x scale of column 100
    b = hc.logits(x)
    p = hc.top1(x)
    v, i = b.float().max(dim=-1)
    same = torch.equal(p[:, 1].long(), i + rank * VS) and all(
        (torch.isnan(p[r, 0]) and torch.isnan(v[r])) or p[r, 0] == v[r] for r in range(M))
    ok &= bool(same)
    print(f"special rows (zeros, NaN, ties, one-hot): pairs == torch.max {same}; idx {p[:, 1].long().tolist()} vs "
          f"{(i + rank * VS).tolist()}", flush=True)
    # graph capture + replay with unrelated traffic in between (zero-at-rest workspace)
    xs = [(torch.randn(m, HD, device=dev) * 0.8).half() for m in (1, 5, 8, 16, 32, 40, 64)]
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for xx in xs:
            hc.top1(xx); hc.logits(xx)
    torch.cuda.current_stream().wait_stream(s2)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [(hc.top1(xx), hc.logits(xx)) for xx in xs]
    junk = torch.empty(32 << 20, dtype=torch.uint8, device=dev)
    gok = True
    for rep in range(20):
        junk.random_(0, 255)
        g.replay()
        torch.cuda.synchronize()
        for xx, (pp, ll) in zip(xs, outs):
            v, i = ll.float().max(-1)
            gok &= torch.equal(pp[:, 1].long(), i + rank * VS) and torch.equal(pp[:, 0], v)
            if rep == 0:
                gok &= torch.equal(ll, hc.logits(xx))
    ok &= gok
    print(f"graph replay x20 (top-1 + logits, 7 row counts): {gok}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)


def bench(rows=None, cfgs=None):
    w8, s = load_shard(0)
    ncopy = int(os.environ.get("NCOPY", "3"))
    hcs = [H.HeadCopy.from_checkpoint(w8, s, vstart=0, nvalid=VS)]
    tms = [tm_prepare(w8, s)]
    for _ in range(ncopy - 1):
        hcs.append(hcs[0].clone())
        tms.append(tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in tms[0]))
    del w8
    torch.cuda.empty_cache()
    names = H.ext().head_configs()
    rows = [int(r) for r in rows.split(",")] if rows else [1, 2, 4, 5, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    sel = cfgs.split(";") if cfgs else None
    floor = VS * HD / 896e3                                             # us at 896 GB/s
    print(f"weights {VS * HD / 1e6:.1f} MB per copy, {ncopy} copies; HBM floor {floor:.0f} us", flush=True)
    for M in rows:
        x = (torch.randn(M, HD, device=dev) * 0.8).half()
        outs = [torch.empty(M, VS, device=dev, dtype=torch.half) for _ in range(ncopy)]
        t_tm = timeit([lambda i=i: torch.ops._C.fp8_gemm_sm70_out(outs[i], x, tms[i][0], tms[i][1], 128, tms[i][2],
                                                                    tms[i][3], False) for i in range(ncopy)])
        line = [f"M={M:3d} TM {t_tm:7.1f}"]
        best = {}
        for ci, nm in enumerate(names):
            r8, name = nm.split(":", 1)
            if int(r8) < M or (int(r8) >= 2 * M and int(r8) > 8) or (sel and name not in sel):
                continue
            for ws in (1,):
                tl = timeit([lambda i=i: H.ext().head_logits(outs[i], x, hcs[i].codes, hcs[i].s16, hcs[i].s32, ci, ws)
                             for i in range(ncopy)])
                tt = timeit([lambda i=i: hcs[i].top1(x, cfg=ci, ws=ws) for i in range(ncopy)])
                line.append(f"[{name} ws{ws}: logits {tl:6.1f} top1 {tt:6.1f}]")
                if tt < best.get("t", 1e9):
                    best = {"t": tt, "n": name, "l": tl}
        line.append(f"BEST {best.get('n')} top1 {best.get('t', 0):.1f} ({t_tm / best.get('t', 1):.3f}x TM, "
                    f"{floor / best.get('t', 1) * 100:.0f}% floor) logits {best.get('l', 0):.1f}")
        print(" ".join(line), flush=True)


def argmax(tables):
    T = torch.load(tables, map_location="cpu", weights_only=False)
    V = T["verify"]
    h, tgt = V["h"], V["tgt"]
    n = h.shape[0]
    print(f"{n} verify rows, {int((tgt >= 0).sum())} with a server-chosen target token", flush=True)
    best_o = torch.full((n,), -float("inf")); idx_o = torch.zeros(n, dtype=torch.long)
    best_t = best_o.clone(); idx_t = idx_o.clone()
    best_w0 = best_o.clone(); idx_w0 = idx_o.clone()
    gap = torch.full((n,), float("inf"))
    for r in range(4):
        w8, s = load_shard(r)
        hc = H.HeadCopy.from_checkpoint(w8, s, vstart=r * VS, nvalid=VS)
        tm = tm_prepare(w8, s)
        del w8
        for b in range(0, n, 64):
            x = h[b:b + 64].to(dev).half().contiguous()
            p = hc.top1(x).cpu()
            p0 = hc.top1(x, ws=0).cpu()
            lt = tm_logits(x, tm).float()
            vt, it = lt.max(-1)
            top2 = lt.topk(2, dim=-1).values
            vt, it = vt.cpu(), it.cpu() + r * VS
            sl = slice(b, b + x.shape[0])
            upd = p[:, 0] > best_o[sl]
            best_o[sl] = torch.where(upd, p[:, 0], best_o[sl]); idx_o[sl] = torch.where(upd, p[:, 1].long(), idx_o[sl])
            upd = p0[:, 0] > best_w0[sl]
            best_w0[sl] = torch.where(upd, p0[:, 0], best_w0[sl]); idx_w0[sl] = torch.where(upd, p0[:, 1].long(), idx_w0[sl])
            upd = vt > best_t[sl]
            best_t[sl] = torch.where(upd, vt, best_t[sl]); idx_t[sl] = torch.where(upd, it, idx_t[sl])
            gap[sl] = torch.minimum(gap[sl], (top2[:, 0] - top2[:, 1]).cpu())
        del hc, tm
        torch.cuda.empty_cache()
    has = tgt >= 0
    print(f"ours (ws1) == TurboMind argmax: {(idx_o == idx_t).float().mean().item():.6f} "
          f"({int((idx_o != idx_t).sum())} differ)", flush=True)
    print(f"ours (ws0, f32 epilogue scale) == TurboMind: {(idx_w0 == idx_t).float().mean().item():.6f}", flush=True)
    print(f"vs server's chosen token (rows with one): ours {(idx_o[has] == tgt[has]).float().mean().item():.6f}, "
          f"TurboMind {(idx_t[has] == tgt[has]).float().mean().item():.6f}", flush=True)
    d = (idx_o != idx_t).nonzero().flatten()[:20]
    for i in d.tolist():
        print(f"  row {i}: ours {idx_o[i].item()} ({best_o[i].item():.4f}) TM {idx_t[i].item()} ({best_t[i].item():.4f})",
              flush=True)


def rerank(tables):
    from q27fuse import draftscreen as DS
    X = DS.ext()
    w8, s = load_shard(0)
    hc = H.HeadCopy.from_checkpoint(w8, s, vstart=0, nvalid=VS)
    r8, s8 = w8.view(torch.uint8).contiguous(), s.reshape(-1).contiguous()
    dense = torch.cat([(r8[b:b + 8192].view(torch.float8_e4m3fn).float() * s8[b:b + 8192, None]).half()
                       for b in range(0, VS, 8192)])
    codes, rowf, rowpb = DS.build_codes(lambda b, e: dense[b:e], VS, dev)
    del dense
    db = DS.sign_bits(DS.rotation_signs()).to(dev)
    G = DS.num_ctas()
    WS = torch.zeros(128, dtype=torch.int64, device=dev)
    QG = torch.empty(64 * HD, dtype=torch.float16, device=dev)
    T = torch.load(tables, map_location="cpu", weights_only=False)["drafts"]["h"]
    ok = True
    for R in (1, 2, 3, 5, 8, 9, 16, 24, 32, 40):
        nsame = 0
        for b in range(0, 2000, 200):
            h = T[b:b + R].to(dev).half().contiguous()
            cand = torch.empty(R, G, DS.NC, dtype=torch.int32, device=dev)
            X.screen(h, db, codes, rowf, rowpb, cand, G, DS.NC, DS.WARPS, R <= DS.PCMAX, 0, QG)
            p2 = torch.empty(R, 2, dtype=torch.float32, device=dev)
            p3 = torch.empty(R, 2, dtype=torch.float32, device=dev)
            X.merge(cand, G, DS.NC, DS.KEEP, h, 2, r8, s8, 0, p2, p2, WS)
            X.merge(cand, G, DS.NC, DS.KEEP, h, 3, hc.codes, hc.s32, 0, p3, p3, WS)
            nsame += int(torch.equal(p2, p3))
        p = torch.empty(R, 2, dtype=torch.float32, device=dev)
        t2 = timeit([lambda: X.merge(cand, G, DS.NC, DS.KEEP, h, 2, r8, s8, 0, p, p, WS)], reps=200)
        t3 = timeit([lambda: X.merge(cand, G, DS.NC, DS.KEEP, h, 3, hc.codes, hc.s32, 0, p, p, WS)], reps=200)
        ok &= nsame == 10
        print(f"R={R:2d}: rerank pairs bitwise rr3 (unified QPN8 copy) == rr2 (row-major copy) {nsame}/10 batches; "
              f"merge us rr2 {t2:.1f} rr3 {t3:.1f}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif m == "bench":
        bench(sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else None)
    elif m == "argmax":
        argmax(sys.argv[2])
    elif m == "rerank":
        rerank(sys.argv[2])
