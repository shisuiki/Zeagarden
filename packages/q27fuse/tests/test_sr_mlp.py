"""test_sr_mlp.py - mlp_fp4.cu kernel variants (set_variant bits: 1 gate+up .cg at 1..8 rows,
2 down .cg at 1..8, 4 parallel reduction at 1..16; default 7) on real layer weights (TP4 rank-0 shard): outputs must
be BITWISE equal to variant 0 at every M; then us per MLP in a CUDA graph cycling 12 weight copies
(cold weights), variants alternated, after >= 1.5 s of clock warm-up.
usage: python test_sr_mlp.py [M list] [variant list]"""
import os, sys, time
import torch
from safetensors import safe_open
from q27fuse import ops

torch.manual_seed(0)
dev = "cuda"
MS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 5, 8, 9, 12, 16, 17, 24, 32, 40, 48, 64]
VS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 4, 5, 6, 7]
NCOPY, REPS, TP, RANK, LAYER = 12, 36, 4, 0, 4
path = os.environ.get("Q27_MODEL_CKPT", "Qwen3.8-27B-NVFP4/model.safetensors")
pre = f"model.language_model.layers.{LAYER}.mlp."
T = {}
with safe_open(path, framework="pt", device="cpu") as f:
    for p in ("gate_proj", "up_proj", "down_proj"):
        for t in ("weight_packed", "weight_scale", "weight_global_scale"):
            T[p, t] = f.get_tensor(pre + p + "." + t)
I = T["gate_proj", "weight_packed"].shape[0]
Ir = I // TP
s = slice(RANK * Ir, (RANK + 1) * Ir)
wgu = torch.cat([T["gate_proj", "weight_packed"][s], T["up_proj", "weight_packed"][s]]).contiguous()
sgu = torch.cat([T["gate_proj", "weight_scale"][s], T["up_proj", "weight_scale"][s]]).contiguous()
dw = T["down_proj", "weight_packed"][:, RANK * Ir // 2:(RANK + 1) * Ir // 2].contiguous()
ds = T["down_proj", "weight_scale"][:, RANK * Ir // 16:(RANK + 1) * Ir // 16].contiguous()
g_gu = torch.stack([T["gate_proj", "weight_global_scale"].reshape(-1)[0], T["up_proj", "weight_global_scale"].reshape(-1)[0]])
g_d = T["down_proj", "weight_global_scale"].reshape(-1)[:1]


class L:
    def register_buffer(self, n, t, persistent=False):
        setattr(self, n, t)


copies = []
for _ in range(NCOPY):
    a, b = L(), L()
    ops.mlp_prepack(a, wgu.to(dev), sgu.to(dev), g_gu, True)
    ops.mlp_prepack(b, dw.to(dev), ds.to(dev), g_d, False)
    copies.append((a, b))
ext = ops.mlp_ext()
ws = torch.empty(0, dtype=torch.half, device=dev)


def run(x, i):
    a, b = copies[i % NCOPY]
    return ext.mlp(x, a.q27_w, b.q27_w, a.q27_epi, b.q27_epi, Ir, ws)


def graph(x):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for i in range(3):
            run(x, i)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(REPS):
            run(x, i)
    return g


def gtime(g, n=7):
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / REPS)
    return sorted(ts)[n // 2]


ok = True
print(f"{'M':>3} " + " ".join(f"{'v' + str(v):>8}" for v in VS) + "   (us per MLP; bitwise = all outputs equal to v0)")
for M in MS:
    x = (torch.randn(M, 5120, device=dev) * 0.5).half()
    outs, graphs = {}, {}
    for v in VS:
        ext.set_variant(v)
        outs[v] = run(x, 0).clone()
        graphs[v] = graph(x)
    same = all(torch.equal(outs[v], outs[VS[0]]) for v in VS)
    ok &= same
    # clock warm-up: >= 1.5 s of the same load
    t0 = time.time()
    while time.time() - t0 < 1.5:
        graphs[VS[0]].replay()
    torch.cuda.synchronize()
    res = {v: [] for v in VS}
    for rep in range(3):
        for v in (VS if rep % 2 == 0 else VS[::-1]):
            res[v].append(gtime(graphs[v]))
    print(f"{M:3d} " + " ".join(f"{sorted(res[v])[1]:8.2f}" for v in VS) + f"   bitwise {'yes' if same else 'NO'}", flush=True)
ext.set_variant(0)
print("PASS" if ok else "FAIL")
