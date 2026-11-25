"""test_mlp.py - q27fuse NVFP4 MLP (csrc/mlp_fp4.cu) on one V100 with real Qwen3.8-27B weights (TP4 rank-0 shard).

For each M (row-major input, as the server hands it over): our gate+up (fused SiLU-and-mul) + down against an fp32
reference and against 1Cat's own kernels (QPN2 gated + down, TurboMind gated + down), then all three timed per MLP in a
CUDA graph cycling NCOPY weight copies (weights from DRAM, as in decode). Q27_COLD=1 adds cold-cache timings; the last
table checks the prefill paths (64-row chunks, exact dequantisation + cuBLAS) against TurboMind.

usage: python test_mlp.py [layer] [M list] [prefill M list]
"""
import os, sys, itertools
import torch
from safetensors import safe_open
import vllm._C  # noqa: F401
import vllm._sm70_ops  # noqa: F401
from q27fuse import ops

torch.manual_seed(0)
dev = "cuda"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 8, 9, 12, 16, 17, 24, 32, 33, 48, 64]
PREFILL = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [65, 128, 192, 256, 384, 512, 2048, 4096]
NCOPY, REPS, TP, RANK = 8, 32, 4, 0
C = torch.ops._C

path = os.environ.get("Q27_MODEL_CKPT", "Qwen3.8-27B-NVFP4/model.safetensors")
pre = f"model.language_model.layers.{LAYER}.mlp."
T = {}
with safe_open(path, framework="pt", device="cpu") as f:
    for p in ("gate_proj", "up_proj", "down_proj"):
        for t in ("weight_packed", "weight_scale", "weight_global_scale"):
            T[p, t] = f.get_tensor(pre + p + "." + t)
I = T["gate_proj", "weight_packed"].shape[0]
H = T["gate_proj", "weight_packed"].shape[1] * 2
Ir = I // TP
s = slice(RANK * Ir, (RANK + 1) * Ir)
wgu = torch.cat([T["gate_proj", "weight_packed"][s], T["up_proj", "weight_packed"][s]]).contiguous()     # [2 Ir, H/2]
sgu = torch.cat([T["gate_proj", "weight_scale"][s], T["up_proj", "weight_scale"][s]]).contiguous()
dw = T["down_proj", "weight_packed"][:, RANK * Ir // 2:(RANK + 1) * Ir // 2].contiguous()               # [H, Ir/2]
ds = T["down_proj", "weight_scale"][:, RANK * Ir // 16:(RANK + 1) * Ir // 16].contiguous()
g_gu = torch.stack([T["gate_proj", "weight_global_scale"].reshape(-1)[0], T["up_proj", "weight_global_scale"].reshape(-1)[0]])
g_d = T["down_proj", "weight_global_scale"].reshape(-1)[:1]
gg, gd = float(g_gu.max()), float(g_d[0])
print(f"layer {LAYER}: H={H} Ir={Ir}; global scales gate/up {g_gu.tolist()} down {gd:.6g}")

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def dequant(w, sc, g):
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], -1).reshape(w.shape[0], -1)
    return v * sc.to(torch.float32).repeat_interleave(16, dim=1) / g


Wgu = dequant(wgu, sgu, gg).to(dev)
Wd = dequant(dw, ds, gd).to(dev)


def ref(x):
    y = x.float() @ Wgu.t()
    a = (torch.nn.functional.silu(y[:, :Ir]) * y[:, Ir:]).half().float()   # the kernels hand fp16 to the down GEMM
    return a @ Wd.t()


def err(y, r):
    return ((y.float() - r).norm() / r.norm()).item()


# ours: prepack on the GPU from the checkpoint layout, as the server hook does
class L:  # stand-in for a linear layer
    def register_buffer(self, n, t, persistent=False):
        setattr(self, n, t)


ours = []
for _ in range(NCOPY):
    a, b = L(), L()
    ops.mlp_prepack(a, wgu.to(dev), sgu.to(dev), g_gu, True)
    ops.mlp_prepack(b, dw.to(dev), ds.to(dev), g_d, False)
    ours.append((a, b))
print(f"packed bytes: gate+up {ours[0][0].q27_w.numel() / 1e6:.2f} MB, down {ours[0][1].q27_w.numel() / 1e6:.2f} MB")
X = ops.mlp_ext()


WS = ops.workspace(torch.device(dev), max(2 * Ir * H, H * Ir))


def run_ours(x, cp):
    a, b = cp
    return X.mlp(x, a.q27_w, b.q27_w, a.q27_epi, b.q27_epi, Ir, WS)


# 1Cat: QPN2 (gated + down) and TurboMind (gated + down)
from vllm.model_executor.layers.quantization.sm70_turbomind import unpack_mxfp4_weight


def tm_prepare(w, sc, g, gated):
    q = unpack_mxfp4_weight(w.to(dev))
    sc = (sc.to(dev).t().to(torch.float32) * (1.0 / g)).to(torch.float16).contiguous()
    return C.nvfp4_sm70_prepare(q, sc, 16, gated)


tm = [(tm_prepare(wgu, sgu, gg, True), tm_prepare(dw, ds, gd, False)) for _ in range(NCOPY)]
q2 = [(C.nvfp4_qpn2_prepare_sm70(wgu.to(dev), sgu.to(dev)), C.nvfp4_qpn2_prepare_sm70(dw.to(dev), ds.to(dev)))
      for _ in range(NCOPY)]
max_rows = int(C.nvfp4_qpn2_max_rows_sm70())
QCFG = {}   # best (split_k, nacc) per (M, which), found below


def run_tm(x, cp):
    (gw, gs, gm), (dw_, ds_, dm) = cp
    mid = torch.empty(x.shape[0], Ir, device=dev, dtype=torch.half)
    gl = [int(v) for v in gm.tolist()]
    C.nvfp4_gemm_sm70_out(mid, x, gw, gs, 16, gl[0], gl[1], True)
    out = torch.empty(x.shape[0], H, device=dev, dtype=torch.half)
    dl = [int(v) for v in dm.tolist()]
    C.nvfp4_gemm_sm70_out(out, mid, dw_, ds_, 16, dl[0], dl[1], False)
    return out


def run_q2(x, cp, cfg=None):
    (gc, gsc), (dc, dsc) = cp
    M = x.shape[0]
    (sk1, na1), (sk2, na2) = cfg or QCFG[M]
    mid = torch.empty(M, Ir, device=dev, dtype=torch.half)
    out = torch.empty(M, H, device=dev, dtype=torch.half)
    for r0 in range(0, M, max_rows):
        C.nvfp4_qpn2_gated_sm70_out(mid[r0:r0 + max_rows], x[r0:r0 + max_rows], gc, gsc, 1.0 / gg, sk1, na1)
    for r0 in range(0, M, max_rows):
        C.nvfp4_qpn2_gemm_sm70_out(out[r0:r0 + max_rows], mid[r0:r0 + max_rows], dc, dsc, 1.0 / gd, sk2, na2)
    return out


def graph_time(fn, copies):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(copies[i % len(copies)])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(REPS):
            fn(copies[i % len(copies)])
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / REPS)
    return sorted(ts)[2]


# ---- checks and timings (overlap_bench.py execs the file up to here)
ok = True
print(f"\n{'M':>3} {'ours err':>9} {'qpn2 err':>9} {'tm err':>9} | {'ours us':>8} {'qpn2 us':>8} {'tm us':>8} | "
      f"ours/qpn2")
for M in MS:
    x = (torch.randn(M, H, device=dev) * 0.5).half()
    r = ref(x)
    eo = err(run_ours(x, ours[0]), r)
    et = err(run_tm(x, tm[0]), r)
    # QPN2: best split-K x chains for each GEMM (as 1Cat's tuning would), within the exactness bound
    best = []
    for gated, K in ((True, H), (False, Ir)):
        cand = []
        for sk, na in itertools.product((4, 8, 16, 32), (1, 2)):
            if (K // 16) % sk:
                continue
            cfg = ((sk, na), (8, 1)) if gated else ((8, 1), (sk, na))
            try:
                cand.append((graph_time(lambda cp: run_q2(x, cp, cfg), q2), (sk, na)))
            except Exception:
                pass
        best.append(min(cand)[1])
    QCFG[M] = tuple(best)
    eq = err(run_q2(x, q2[0]), r)
    to = graph_time(lambda cp: run_ours(x, cp), ours)
    tq = graph_time(lambda cp: run_q2(x, cp), q2)
    tt = graph_time(lambda cp: run_tm(x, cp), tm)
    ok &= eo < 2e-3
    print(f"{M:3d} {eo:9.2e} {eq:9.2e} {et:9.2e} | {to:8.1f} {tq:8.1f} {tt:8.1f} | {to / tq:6.2f}")
# cold (Q27_COLD=1): each MLP after a TLB + L2 eviction (strided reads over 8 GB, 16 MB written), as in the server,
# where a step streams ~5 GB per GPU between two MLPs; time = graph(flush + MLP) - graph(flush)
if os.environ.get("Q27_COLD") == "1":
    BIG = torch.empty(2 << 30, dtype=torch.int32, device=dev)
    L2B = torch.empty(4 << 20, dtype=torch.int32, device=dev)
    RD = torch.empty(16 << 20, dtype=torch.int32, device=dev)     # 64 MB, streamed like the mixer's FP8 weights
    FLUSH = {"stream 64 MB read": lambda: RD.sum(),
             "TLB + L2 evict": lambda: (BIG[::16384].sum(), L2B.zero_()),
             "16 MB dirty writes": lambda: L2B.zero_()}
    for fname, flush in FLUSH.items():
        tfl = graph_time(lambda cp: flush(), ours)
        print(f"\ncold [{fname}] before each MLP (alone {tfl:.1f} us)")
        for M in MS:
            x = (torch.randn(M, H, device=dev) * 0.5).half()
            tf = graph_time(lambda cp: (flush(), run_ours(x, cp)), ours) - tfl
            tq = graph_time(lambda cp: (flush(), run_q2(x, cp)), q2) - tfl
            print(f"cold {M:3d}  ours {tf:6.1f} us  qpn2 {tq:6.1f} us  ours/qpn2 {tf / tq:.3f}")
    del BIG, L2B, RD


# prefill: exact dequantisation into the workspace + cuBLAS, against TurboMind (1Cat's prefill path)
def ev_time(fn, n=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    return sorted(ts)[n // 2]


print(f"\n{'M':>5} {'chunk err':>9} {'deq err':>9} {'tm err':>9} | {'chunks us':>9} {'deq us':>9} {'tm us':>9}")
for M in PREFILL:
    x = (torch.randn(M, H, device=dev) * 0.5).half()
    r = ref(x)
    X.set_chunked_max(1 << 30)                         # 64-row chunks of the decode kernels
    ec = err(run_ours(x, ours[0]), r); tc = ev_time(lambda: run_ours(x, ours[0]))
    X.set_chunked_max(64)                              # exact dequantisation + cuBLAS
    ed = err(run_ours(x, ours[0]), r); td = ev_time(lambda: run_ours(x, ours[0]))
    X.set_chunked_max(256)
    et, tt = err(run_tm(x, tm[0]), r), ev_time(lambda: run_tm(x, tm[0]))
    ok &= ec < 2e-3 and ed < 2e-3
    print(f"{M:5d} {ec:9.2e} {ed:9.2e} {et:9.2e} | {tc:9.1f} {td:9.1f} {tt:9.1f}")
print("PASS" if ok else "FAIL")
