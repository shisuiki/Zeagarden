"""test_sr_gdnin.py (track smallrows) - the GDN in-projection on fp8_gdn_in.cu (qkv | z from the FP8 QPN8 codes, b/a
from the fp16 weight, one launch) against: our fp8_lin.cu with the same shape (qkv|z must be bitwise equal), an FP64
reference (error), gdn_glue.cu k_gdn_in (b/a error only), and 1Cat's route (apply_gdn_ba_verify's fused op at 2..32 rows; above
32 rows the split route: QPN8 / p14 fp8_lin qkvz + cuBLAS b/a + split copies). Then us per call in a CUDA graph
cycling 12 weight sets, >= 1.5 s clock warm-up, alternated order, median of 3.
usage: python test_sr_gdnin.py [M list]"""
import sys, time
import torch
import vllm._C  # noqa: F401
import vllm._sm70_ops  # noqa: F401
from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
from q27fuse import ops, smallrows

torch.manual_seed(0)
dev, NCOPY, REPS = "cuda", 12, 36
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 5, 8, 12, 16, 24, 32, 40, 48, 64]
N, K, QW, HEADS = 4096, 5120, 2560, 12
gi = smallrows.gdn_in_ext()
f8 = ops.fp8_ext()
F8IDX = {c.split(":", 2)[2]: i for i, c in enumerate(f8.fp8_configs())}
GIDX = {c.split(":", 2)[2]: i for i, c in enumerate(gi.gdn_in8_configs())}
C = torch.ops._C
sets = []
for c in range(NCOPY):
    w = (torch.randn(N, K, device=dev) * 0.6).to(torch.float8_e4m3fn)
    sc = (torch.rand(N, 1, device=dev) * 2e-3 + 5e-4).float()
    codes, scales = C.fp8_qpn8_prepare_sm70(w, sc)[:2]
    wba = (torch.randn(2 * HEADS, K, device=dev) * 0.02).half()
    if c == 0:
        wref, baref = (w.double() * sc.double()), wba.double()
    sets.append((codes, scales, wba))
print(f"codes {tuple(sets[0][0].shape)} {sets[0][0].dtype}, scales {tuple(sets[0][1].shape)} {sets[0][1].dtype}")


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(REPS):
            fn(i % NCOPY)
    return g


def gtime(g, n=5):
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / REPS)
    return sorted(ts)[n // 2]


def rel(a, r):
    return ((a.double() - r).norm() / r.norm()).item()


ok = True
for M in MS:
    x = (torch.randn(M, K, device=dev) * 0.5).half()
    rq = x.double() @ wref.t()
    rba = x.double() @ baref.t()
    qkv, z = x.new_empty(M, QW), x.new_empty(M, N - QW)
    b, a = x.new_empty(M, HEADS), x.new_empty(M, HEADS)
    name = next(n for r, n in smallrows.GDNIN_CFG if M <= r)
    cfg = GIDX[name]
    fused = lambda i: gi.gdn_in8(x, sets[i][0], sets[i][1], sets[i][2], qkv, z, b, a, cfg, True)
    fused(0)
    full = x.new_empty(M, N)
    twin = name.removesuffix(" cg")                       # .cg: same math; fp8_lin.cu has no 1..8-row twin any more
    same = True
    if twin in F8IDX:
        f8.fp8_linear(full, x, sets[0][0].view(-1), sets[0][1].view(-1), F8IDX[twin])
        same = torch.equal(torch.cat([qkv, z], 1), full)
    eq, eb = rel(torch.cat([qkv, z], 1), rq), rel(torch.cat([b, a], 1), rba)
    b2, a2, core = x.new_empty(M, HEADS), x.new_empty(M, HEADS), x.new_empty(M, HEADS, 128)
    ops.gdn_ext().gdn_in(x, sets[0][2], b2, a2, core)
    e_glue = rel(torch.cat([b2, a2], 1), rba)

    cands = {"ours fused": fused}
    # 1Cat's route
    cq, cz, cb, ca = x.new_empty(M, QW), x.new_empty(M, N - QW), x.new_empty(M, HEADS), x.new_empty(M, HEADS)
    if 2 <= M <= 32:
        op = C.fp8_qpn8_dispatch_ba_masked_sm70_out if M < 8 else C.fp8_qpn8_dispatch_ba_split_sm70_out
        st1, st2 = x.new_empty(M, N), x.new_empty(M, 2 * HEADS)

        def cat(i):
            op(cq, cz, cb, ca, st1, st2, 0, x, sets[i][0], sets[i][1], sets[i][2])
        clabel = "1Cat fused ba"
    else:
        qkvz_o = x.new_empty(M, N)
        p14 = {(17, 32): "s 4x1 wk4 col2", (41, 64): "s 4x1 wk3 col2 row2"}
        pc = next((F8IDX[v] for (lo, hi), v in p14.items() if lo <= M <= hi), None)

        def cat(i):
            if pc is None:
                dispatch_channel_rows(qkvz_o, x, sets[i][0], sets[i][1], 16, 2, False, False)
            else:
                f8.fp8_linear(qkvz_o, x, sets[i][0].view(-1), sets[i][1].view(-1), pc)
            ba = torch.nn.functional.linear(x, sets[i][2])
            cz.copy_(qkvz_o[:, QW:]); cb.copy_(ba[:, :HEADS]); ca.copy_(ba[:, HEADS:])
        clabel = "p14 split route (" + ("1Cat QPN8" if pc is None else "fp8_lin") + " + cuBLAS b/a + copies)"
    cat(0)
    cqz = torch.cat([cq if M <= 32 else qkvz_o[:, :QW], cz], 1)
    e1q, e1b = rel(cqz, rq), rel(torch.cat([cb, ca], 1), rba)
    cands[clabel] = cat
    graphs = {k: graph(f) for k, f in cands.items()}
    t0 = time.time()
    while time.time() - t0 < 1.5:
        graphs[clabel].replay()
    torch.cuda.synchronize()
    res = {k: [] for k in graphs}
    keys = list(graphs)
    for rep in range(3):
        for k in (keys if rep % 2 == 0 else keys[::-1]):
            res[k].append(gtime(graphs[k]))
    med = {k: sorted(v)[1] for k, v in res.items()}
    good = same and eq < 2e-3 and eb < 2e-3
    ok &= good
    print(f"M={M:2d} cfg '{name}': qkv|z bitwise=fp8_lin {same if twin in F8IDX else 'n/a'}, err qkvz {eq:.1e} b/a {eb:.1e} (k_gdn_in {e_glue:.1e}); "
          f"1Cat err qkvz {e1q:.1e} b/a {e1b:.1e} | us: " + " | ".join(f"{k} {med[k]:.2f}" for k in keys)
          + f" | 1Cat/fused {med[clabel] / med['ours fused']:.3f}x", flush=True)
print("PASS" if ok else "FAIL")
