"""test_attn.py MODE ... - our decode attention kernels against an FP32 reference and against 1Cat's shipped E4M3
long-context operator (_vllm_fa2_C.sm70_grouped_long_fwd), on the server's paged layout: kv [blocks][2][page][1][256]
uint8 E4M3, 6 query heads per KV head.
  check|bench [B list] [L list] [configs]      csrc/attn_dec.cu (one query row per request, B >= 2 in the server);
      page 1568 (env PAGE); configs "ng:S:min_tiles,...".
  mqcheck|mqbench [BxQ list] [L list] [configs]  csrc/attn_mq.cu (track attnv): Q query rows per request (MTP verify
      rows, Q = 1 single-query); per-row lengths L - Q + i + 1 (row i of a request), per-request block table; page
      2048 (env PAGE, the MTP server's); configs "nw:S:min_tiles,..." (nw warps per CTA, S splits). 1Cat's operator is
      called through its own wrapper (wrap_long_attention: the server's admission) for Q >= 2, directly (as the
      single-query wrapper does) for Q = 1.
check modes: relative L2 error of each kernel's output against the FP32 reference over ragged lengths (lengths below
one tile, pages crossing, padding rows / requests). bench modes: us per call in a CUDA graph cycling >= 12 distinct KV
caches (>= 48 MB in total, so nothing stays in L2), % of the KV read floor (B ctx 512 B at 898 GB/s)."""
import math, os, sys
import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)
dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
PAGE = int(os.environ.get("PAGE", 2048 if MODE.startswith("mq") else 1568))
HKV, HD = int(os.environ.get("HKV", 1)), 256
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build = os.environ.get("Q27_FUSE_BUILD", os.path.expanduser("~/.cache/q27fuse/build")) + "_attn"
os.makedirs(build, exist_ok=True)
ext = load(name="q27fuse_attn", sources=[os.path.join(here, "q27fuse", "csrc", "attn_dec.cu")], build_directory=build,
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math", "-Xptxas=-v"],
           verbose=False)
mq = None
if MODE.startswith("mq"):
    os.makedirs(build.replace("_attn", "_attnmq"), exist_ok=True)
    minb = os.environ.get("MQ_MINB")             # bench-only: __launch_bounds__ min blocks per SM
    bdir = build.replace("_attn", "_attnmq") + (f"_minb{minb}" if minb else "")
    os.makedirs(bdir, exist_ok=True)
    mq = load(name="q27fuse_attnmq" + (f"_minb{minb}" if minb else ""),
              sources=[os.path.join(here, "q27fuse", "csrc", "attn_mq.cu")], build_directory=bdir,
              extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-std=c++17", "--use_fast_math",
                                 "-Xptxas=-v"] + ([f"-DQ27_MINB={minb}"] if minb else []),
              verbose=bool(os.environ.get("VERBOSE")))

from vllm.v1.attention.ops.sm70_grouped_long import resolve_long_attention, run_six_head_groups
if os.environ.get("ATTN_STOP"):                 # debug timing: kernels return after phase ATTN_STOP
    (mq or ext).set_stop(int(os.environ["ATTN_STOP"]))
OP, MAN = resolve_long_attention()
PAR = MAN.get("kv_heads_revision", 0) >= 1


def make(B, lens, ncopy=1):
    nbps = [math.ceil(l / PAGE) for l in lens]
    W = max(nbps)
    nb = sum(nbps) + 3
    caches = []
    for _ in range(ncopy):
        kv = (torch.randn(nb, 2, PAGE, HKV, HD, device=dev) * 2.0).to(torch.float8_e4m3fn).view(torch.uint8)
        caches.append(kv)
    perm = torch.randperm(nb, device=dev)[: sum(nbps)].int()
    table = torch.zeros(B, W, dtype=torch.int32, device=dev)
    o = 0
    for b, n in enumerate(nbps):
        table[b, :n] = perm[o:o + n]; o += n
    q = (torch.randn(B, 6 * HKV, HD, device=dev)).half()
    return caches, table, torch.tensor(lens, dtype=torch.int32, device=dev), q


def reference(q, kv, table, lens, scale, ks=1.0, vs=1.0):
    k8, v8 = kv[:, 0].view(torch.float8_e4m3fn), kv[:, 1].view(torch.float8_e4m3fn)
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(q.shape[0]):
        L = int(lens[b]); n = math.ceil(L / PAGE)
        blocks = table[b, :n].long()
        K = k8[blocks].float().reshape(-1, HKV, HD)[:L] * ks
        V = v8[blocks].float().reshape(-1, HKV, HD)[:L] * vs
        for h in range(HKV):
            s = (q[b, 6 * h:6 * h + 6].float() @ K[:, h].T) * scale
            out[b, 6 * h:6 * h + 6] = torch.softmax(s, -1) @ V[:, h]
    return out


WS = {}
def ws(bh, S, ng):
    n = max(1, ext.attn_slots(bh, S, ng))
    key = (n,)
    if key not in WS:
        WS[key] = (torch.empty(n * 6 * HD, device=dev), torch.empty(n * 12, device=dev),
                   torch.zeros(4096, dtype=torch.int32, device=dev))
    return WS[key]


def ours(q, kv, table, lens, out, scale, cfg, ks=1.0, vs=1.0):
    ng, S, mt = cfg
    po, pml, cnt = ws(q.shape[0] * HKV, S, ng)
    ext.attn_decode(q, kv[:, 0], kv[:, 1], table, lens, out, po, pml, cnt, scale, ks, vs, ng, S, mt)


CATWS = {}
def cat(q, kv, table, lens, out, scale, ks=1.0, vs=1.0):
    B = q.shape[0]; panels = B * (HKV if PAR else 1)
    if panels not in CATWS:
        CATWS[panels] = (torch.empty(panels, 80, 8, 6, 256, device=dev), torch.empty(panels, 80, 8, 6, 2, device=dev))
    p, l = CATWS[panels]
    pv, lv = (p[0], l[0]) if panels == 1 else (p[:panels], l[:panels])
    run_six_head_groups(OP, q, kv[:, 0], kv[:, 1], out, table, lens, pv, lv, scale, ks, vs, parallel_kv_heads=PAR)


def graph_us(fn, ncopy, reps=12):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            fn(i % ncopy)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i % ncopy)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / reps)
    return sorted(ts)[2]


def parse_cfgs(a):
    return [tuple(int(x) for x in c.split(":")) for c in a.split(",")]


DEFAULT = "1:64:1,2:32:1,4:16:1,1:16:2,2:8:2,4:8:1,4:4:2,4:2:4,2:4:4"
mode = MODE
scale = HD ** -0.5
print(f"1Cat operator: {MAN.get('module_name')} parallel_kv_heads={PAR}; page {PAGE}; capacity ng1/2/4 = "
      f"{ext.attn_capacity(1)}/{ext.attn_capacity(2)}/{ext.attn_capacity(4)}", flush=True)

if mode in ("check", "bench"):
    Bs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 16, 32, 64]
    Ls = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1088, 8256]
    cfgs = parse_cfgs(sys.argv[4]) if len(sys.argv) > 4 else parse_cfgs(DEFAULT)

if mode == "check":
    for B in Bs:
        for L in Ls:
            lens = [max(1, L - (37 * b) % 700) for b in range(B)] if B > 1 else [L]
            caches, table, lt, q = make(B, lens)
            ref = reference(q, caches[0], table, lt, scale, 0.75, 1.25)
            o = torch.empty_like(q)
            cat(q, caches[0], table, lt, o, scale, 0.75, 1.25); torch.cuda.synchronize()
            ec = ((o.float() - ref).norm() / ref.norm()).item()
            res = []
            for cfg in cfgs:
                o2 = torch.full_like(q, float("nan"))
                ours(q, caches[0], table, lt, o2, scale, cfg, 0.75, 1.25)
                ours(q, caches[0], table, lt, o2, scale, cfg, 0.75, 1.25)   # twice: the counters must be reset
                torch.cuda.synchronize()
                res.append(((o2.float() - ref).norm() / ref.norm()).item())
            print(f"B {B:3d} L {L:5d}: 1Cat {ec:.2e} | ours " + " ".join(f"{c}:{e:.2e}" for c, e in zip(cfgs, res)),
                  flush=True)
elif mode == "bench":
    for L in Ls:
        for B in Bs:
            per = B * L * 512
            ncopy = max(1, math.ceil(24e6 / per))
            caches, table, lt, q = make(B, [L] * B, ncopy)
            o = torch.empty_like(q)
            floor = per / 898e3
            tc = graph_us(lambda i: cat(q, caches[i], table, lt, o, scale), ncopy)
            res = [graph_us(lambda i, c=cfg: ours(q, caches[i], table, lt, o, scale, c), ncopy) for cfg in cfgs]
            best = min(range(len(cfgs)), key=lambda i: res[i])
            print(f"L {L:5d} B {B:3d} floor {floor:6.1f} | 1Cat {tc:6.1f} ({100 * floor / tc:3.0f}%) | ours " +
                  " ".join(f"{':'.join(map(str, c))} {t:6.1f}" for c, t in zip(cfgs, res)) +
                  f" | best {':'.join(map(str, cfgs[best]))} {res[best]:.1f} ({100 * floor / res[best]:.0f}%, "
                  f"{tc / res[best]:.2f}x)", flush=True)

# ---------------------------------------------------------------------------- attn_mq.cu: Q query rows per request
def make_mq(B, Q, lens_req, ncopy=1, pad_rows=0):
    """B requests with Q rows each; request b has lens_req[b] tokens incl. its Q new ones (0 = padding request);
    row i sees lens_req[b] - Q + i + 1 tokens. pad_rows: the last pad_rows rows of the last request get length 0."""
    caches, table, _, _ = make(B, [max(1, l) for l in lens_req], ncopy)
    rl = []
    for b, L in enumerate(lens_req):
        rl += [max(0, L - Q + i + 1) if L > 0 else 0 for i in range(Q)]
    for i in range(pad_rows):
        rl[-1 - i] = 0
    q = torch.randn(B * Q, 6 * HKV, HD, device=dev).half()
    return caches, table, torch.tensor(rl, dtype=torch.int32, device=dev), q


def reference_mq(q, kv, table, rl, Q, scale, ks=1.0, vs=1.0):
    k8, v8 = kv[:, 0].view(torch.float8_e4m3fn), kv[:, 1].view(torch.float8_e4m3fn)
    out = torch.zeros_like(q, dtype=torch.float32)
    for r in range(q.shape[0]):
        L = int(rl[r]); b = r // Q
        if L <= 0:
            continue
        blocks = table[b, :math.ceil(L / PAGE)].long()
        K = k8[blocks].float().reshape(-1, HKV, HD)[:L] * ks
        V = v8[blocks].float().reshape(-1, HKV, HD)[:L] * vs
        for h in range(HKV):
            s_ = (q[r, 6 * h:6 * h + 6].float() @ K[:, h].T) * scale
            out[r, 6 * h:6 * h + 6] = torch.softmax(s_, -1) @ V[:, h]
    return out


MQWS = {}
def mq_ws(n_po, n_pml):
    key = "ws"
    if key not in MQWS or MQWS[key][0].numel() < n_po or MQWS[key][1].numel() < n_pml:
        MQWS[key] = (torch.empty(max(1, n_po), device=dev), torch.empty(max(1, n_pml), device=dev),
                     torch.zeros(8192, dtype=torch.int32, device=dev))
    return MQWS[key]


def mq_ok(cfg, Q):
    """configuration exists for Q rows: attn_dec.cu ones only for Q = 1, attn_mq.cu ones when instantiated"""
    if isinstance(cfg[0], str):
        return Q == 1
    c = tuple(cfg) + (1, 1, 0, 0, 0)[len(cfg) - 3:]
    return mq.mq_splits(1, c[1], c[0], Q, c[3], c[4], c[6]) >= 1


def ours_mq(q, kv, table, rl, out, Q, scale, cfg, ks=1.0, vs=1.0):
    if isinstance(cfg[0], str):                  # ("d", ng, S, min_tiles): attn_dec.cu (Q = 1 only)
        assert Q == 1
        return ours(q, kv, table, rl, out, scale, cfg[1:], ks, vs)
    nw, S, mt, ng, pd, pf, rs, sep = tuple(cfg) + (1, 1, 0, 0, 0)[len(cfg) - 3:]
    n_po, n_pml = mq.mq_workspace(table.shape[0] * HKV, S, nw, Q, ng, pd, rs)
    po, pml, cnt = mq_ws(n_po, n_pml)
    mq.attn_mq(q, kv[:, 0], kv[:, 1], table, rl, out, po, pml, cnt, scale, ks, vs, Q, nw, S, mt, ng, pd, pf, rs, sep)


_WRAP = None
def cat_mq(q, kv, table, rl, out, Q, scale, ks=1.0, vs=1.0, via_wrapper=True):
    """1Cat: Q >= 2 through wrap_long_attention (its admission + workspace, as in the server), Q = 1 directly."""
    global _WRAP
    if Q >= 2 and via_wrapper:
        if _WRAP is None:
            from vllm.v1.attention.ops.sm70_grouped_long import wrap_long_attention
            def fallback(*a, **k):
                raise RuntimeError("1Cat's long-attention wrapper rejected the call")
            _WRAP = wrap_long_attention(fallback)
        _WRAP(q, kv[:, 0], kv[:, 1], table, rl, out=out, softmax_scale=scale, k_scale=ks, v_scale=vs)
        return
    B = table.shape[0]; panels = B * (HKV if PAR else 1)
    if panels not in CATWS:
        CATWS[panels] = (torch.empty(panels, 80, 8, 6, 256, device=dev), torch.empty(panels, 80, 8, 6, 2, device=dev))
    p, l = CATWS[panels]
    pv, lv = (p[0], l[0]) if panels == 1 else (p[:panels], l[:panels])
    run_six_head_groups(OP, q, kv[:, 0], kv[:, 1], out, table, rl, pv, lv, scale, ks, vs, parallel_kv_heads=PAR)


def parse_mq_cfgs(a):
    """"nw:S:mt[:ng[:pd[:pf[:rs[:sep]]]]]" -> attn_mq.cu (ng sets of nw warps per CTA, pd tiles in flight in registers
    per set, pf more tiles per set prefetched into L2, rs 1 = row split: the sets share tiles and split the row groups,
    sep 1 = combine in a second kernel instead of the in-kernel split barrier; default 1, 1, 0, 0, 0);
    "dng:S:mt" -> attn_dec.cu (Q = 1)"""
    out = []
    for c in a.split(","):
        x = c.split(":")
        out.append(("d", int(x[0][1:]), int(x[1]), int(x[2])) if x[0].startswith("d") else tuple(int(y) for y in x))
    return out


def cfg_name(c):
    return ":".join(map(str, c)).replace("d:", "d")


def parse_bq(a):
    return [tuple(int(x) for x in c.split("x")) for c in a.split(",")]


MQDEFAULT = "4:64:1,4:32:1,4:16:2,2:64:1,2:32:1"
INTERLEAVE = bool(os.environ.get("INTERLEAVE"))       # mqbench: an unrelated kernel between calls (see there)
if mode in ("mqcheck", "mqbench", "mqone", "mqrace"):
    BQs = parse_bq(sys.argv[2]) if len(sys.argv) > 2 else [(1, 5), (2, 4), (4, 4), (8, 4), (16, 2), (32, 2), (1, 1),
                                                          (2, 1), (8, 1), (32, 1)]
    Ls = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1024, 8192]
    cfgs = parse_mq_cfgs(sys.argv[4]) if len(sys.argv) > 4 else parse_mq_cfgs(MQDEFAULT)

if mode == "mqcheck":
    # ragged lengths: below one tile, around tile and page edges, a padding request, padding rows
    for B, Q in BQs:
        for L in Ls:
            if B == 1:
                cases = [[L]]
                if L == Ls[0]:
                    cases += [[Q + 3], [33], [PAGE + 1], [PAGE + Q - 1]]
            else:
                cases = [[max(Q, L - (37 * b) % 700) for b in range(B)]]
                if L == Ls[0]:
                    cases.append([0 if b == 1 else max(Q, (31 + 97 * b) % (2 * PAGE + 50)) for b in range(B)])
            for ci, lens_req in enumerate(cases):
                pr = 1 if (B > 1 and ci == 0 and Q > 1) else 0
                caches, table, rl, q = make_mq(B, Q, lens_req, pad_rows=pr)
                ref = reference_mq(q, caches[0], table, rl, Q, scale, 0.75, 1.25)
                o = torch.zeros_like(q)
                try:
                    cat_mq(q, caches[0], table, rl, o, Q, scale, 0.75, 1.25, via_wrapper=(0 not in lens_req)); torch.cuda.synchronize()
                    ec = ((o.float() - ref).norm() / ref.norm()).item()
                except RuntimeError as e:
                    ec = float("nan")
                res = []
                o2 = None
                det = True
                for cfg in cfgs:
                    if not mq_ok(cfg, Q):
                        res.append(float("nan")); continue
                    o2 = torch.full_like(q, float("nan"))
                    ours_mq(q, caches[0], table, rl, o2, Q, scale, cfg, 0.75, 1.25)
                    o3 = o2.clone()
                    ours_mq(q, caches[0], table, rl, o2, Q, scale, cfg, 0.75, 1.25)   # twice: counters reset
                    torch.cuda.synchronize()
                    if not torch.equal(o2, o3):                                          # run-to-run bitwise
                        dd = (o2.float() - o3.float()).abs()
                        nz = (dd > 0).nonzero()
                        print(f"   nondeterministic {cfg_name(cfg)}: max |diff| {dd.max().item():.3e}, {nz.shape[0]} elements, "
                              f"rows {sorted(set(nz[:, 0].tolist()))[:12]}, heads {sorted(set(nz[:, 1].tolist()))}, "
                              f"dims {nz[:, 2].min().item()}..{nz[:, 2].max().item()}", flush=True)
                        det = False
                    res.append(((o2.float() - ref).norm() / ref.norm()).item())
                    # padding rows must come out exactly 0
                    zr = (rl == 0).nonzero().flatten()
                    if zr.numel() and o2[zr].abs().max().item() != 0:
                        res[-1] = float("inf")
                d1 = ((o2.float() - o.float()).norm() / o.float().norm()).item() if ec == ec and o2 is not None else float("nan")
                print(f"B {B:3d} Q {Q} lens {str(lens_req[:4])[:-1]}{',..' if B > 4 else ''}] pad_rows {pr}: 1Cat {ec:.2e} | ours "
                      + " ".join(f"{cfg_name(c)} {e:.2e}" for c, e in zip(cfgs, res) if e == e) +
                      f" | ours-1Cat {d1:.2e} | bitwise repeat {det}", flush=True)
elif mode == "mqbench":
    # clocks up before the first point (a cold GPU made the first small points ~3 us slower)
    _w = torch.randn(4096, 4096, device=dev).half()
    for _ in range(200):
        _w @ _w
    torch.cuda.synchronize()
    for L in Ls:
        for B, Q in BQs:
            per = B * L * 512
            ncopy = min(128, max(12, math.ceil(48e6 / per)))
            reps = ncopy                                 # every call of a replay reads a different cache
            caches, table, rl, q = make_mq(B, Q, [L] * B, ncopy)
            o = torch.empty_like(q)
            floor = per / 898e3
            if INTERLEAVE:
                # every call followed by an unrelated elementwise kernel (no shared memory, 16 MB read+write), its own
                # time subtracted: our kernel and 1Cat's between other kernels, as inside the server's graph
                _x = torch.zeros(4 * 2 ** 20, device=dev)
                tx = graph_us(lambda i: _x.add_(1.0), ncopy, reps=reps)
                wrap = lambda f: (lambda i: (f(i), _x.add_(1.0)))
            else:
                tx, wrap = 0.0, (lambda f: f)
            tc = graph_us(wrap(lambda i: cat_mq(q, caches[i], table, rl, o, Q, scale, via_wrapper=False)), ncopy, reps=reps) - tx
            res = [graph_us(wrap(lambda i, c=cfg: ours_mq(q, caches[i], table, rl, o, Q, scale, c)), ncopy, reps=reps) - tx
                   if mq_ok(cfg, Q) else float("inf") for cfg in cfgs]
            best = min(range(len(cfgs)), key=lambda i: res[i])
            print(f"L {L:5d} B {B:3d} Q {Q} floor {floor:6.1f} | 1Cat {tc:6.1f} ({100 * floor / tc:3.0f}%) | ours " +
                  " ".join(f"{cfg_name(c)} {t:6.1f}" for c, t in zip(cfgs, res) if t != float("inf")) +
                  f" | best {cfg_name(cfgs[best])} {res[best]:.1f} ({100 * floor / res[best]:.0f}%, "
                  f"{tc / res[best]:.2f}x)", flush=True)
elif mode == "mqone":
    # eager calls for Nsight Compute (ncu -k regex:k_mq): per (L, B, Q) and config, one warm-up call then one call on a
    # different cache copy (cold KV); 1Cat's operator the same way
    for L in Ls:
        for B, Q in BQs:
            caches, table, rl, q = make_mq(B, Q, [L] * B, 2)
            o = torch.empty_like(q)
            for cfg in cfgs:
                if mq_ok(cfg, Q):
                    ours_mq(q, caches[0], table, rl, o, Q, scale, cfg)
                    ours_mq(q, caches[1], table, rl, o, Q, scale, cfg)
            cat_mq(q, caches[0], table, rl, o, Q, scale, via_wrapper=False)
            cat_mq(q, caches[1], table, rl, o, Q, scale, via_wrapper=False)
            torch.cuda.synchronize()
            print(f"mqone L {L} B {B} Q {Q} cfgs {[cfg_name(c) for c in cfgs]}", flush=True)
elif mode == "mqrace":
    # race detector: per config, 24 calls alternating two query sets and two caches (back to back, no host sync); each
    # output's worst element error against its FP32 reference. Reading another launch's partials shows up as a large
    # element error, a counter left out of step as a hang. Also counts nonzero split counters left after the calls.
    for L in Ls:
        for B, Q in BQs:
            caches, table, rl, q1 = make_mq(B, Q, [L] * B, 2)
            q2 = torch.randn_like(q1)
            refs = {(i, j): reference_mq(qq, caches[j], table, rl, Q, scale) for i, qq in enumerate((q1, q2)) for j in range(2)}
            for cfg in cfgs:
                if not mq_ok(cfg, Q):
                    continue
                outs = []
                for it in range(24):
                    i, j = it % 2, (it // 2) % 2
                    o = torch.empty_like(q1)
                    ours_mq((q1, q2)[i], caches[j], table, rl, o, Q, scale, cfg)
                    outs.append((i, j, o))
                torch.cuda.synchronize()
                worst = max(((o.float() - refs[(i, j)]).abs().max() / refs[(i, j)].abs().max()).item() for i, j, o in outs)
                cnt = MQWS["ws"][2]
                print(f"race L {L} B {B} Q {Q} {cfg_name(cfg)}: worst element rel err {worst:.2e}, "
                      f"counters nonzero {(cnt != 0).sum().item()}", flush=True)
