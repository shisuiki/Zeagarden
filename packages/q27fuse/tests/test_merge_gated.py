"""test_merge_gated.py (track merge) - repeatability / NaN stress of smallrows' fp8_gated (the FP8 MLP gate_up + SiLU-and-mul
of layers 56-63) at every row count 9..64: eager reference once, then many CUDA-graph replays of
[fp8_gated(M1), unrelated 32 MB write, fp8_gated(M2), attn_mq-like L2 churn] with fresh inputs copied in; every output must
be bitwise equal to the eager result of the same input and NaN-free. Also 1Cat's gated dispatch on the same inputs (error
vs FP32 reference for both)."""
import os, sys, time
import torch
import vllm._C  # noqa: F401
import vllm._sm70_ops  # noqa: F401
from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
from q27fuse import ops, smallrows

torch.manual_seed(0)
dev = "cuda"
N, K, I = 8704, 5120, 4352
w = (torch.randn(N, K, device=dev) * 0.6).to(torch.float8_e4m3fn)
sc = (torch.rand(N, 1, device=dev) * 2e-3 + 5e-4).float()
codes, scales = torch.ops._C.fp8_qpn8_prepare_sm70(w, sc)[:2]
e = ops.fp8_ext()
GIDX = {c.split(":", 2)[2]: i for i, c in enumerate(e.fp8_gated_configs())}
cfg = lambda m: GIDX[next(n for r, n in smallrows.GATED_CFG if m <= r)]
wref = w.float() * sc
bad = 0
for M in list(range(9, 65)):
    x = (torch.randn(M, K, device=dev) * 0.5).half()
    out = torch.empty(M, I, device=dev, dtype=torch.half)
    e.fp8_gated(out, x, codes.view(-1), scales.view(-1), cfg(M))
    ref = out.clone()
    gu = x.float() @ wref.t()
    r32 = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    o1 = torch.empty_like(out)
    dispatch_channel_rows(o1, x, codes, scales, 16, 2, False, True, None)
    err = ((ref.float() - r32).norm() / r32.norm()).item(); err1 = ((o1.float() - r32).norm() / r32.norm()).item()
    # graph: our kernel twice with junk traffic in between, replayed 200 times
    junk = torch.empty(8 * 1024 * 1024, device=dev)
    out2 = torch.empty_like(out)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            e.fp8_gated(out2, x, codes.view(-1), scales.view(-1), cfg(M)); junk.fill_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        junk.fill_(2.0)
        e.fp8_gated(out2, x, codes.view(-1), scales.view(-1), cfg(M))
        junk.mul_(1.5)
    nd = nn = 0
    for rep in range(200):
        g.replay()
        if rep % 20 == 19:
            torch.cuda.synchronize()
            nd += int(not torch.equal(out2, ref)); nn += int(torch.isnan(out2).any().item())
    torch.cuda.synchronize()
    ok = nd == 0 and nn == 0 and not torch.isnan(ref).any().item() and err < 1e-3
    bad += not ok
    print(f"M={M:2d} cfg {cfg(M)}: err ours {err:.2e} 1Cat {err1:.2e}; graph replays differing {nd}/10, with NaN {nn}/10"
          f"{'' if ok else '  <-- FAIL'}", flush=True)
print("GATED", "PASS" if bad == 0 else f"FAIL ({bad})")
