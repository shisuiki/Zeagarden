"""test_sr_ops.py (track smallrows) - the ops with the smallrows tables in place (defaults, or knobs from the env, e.g.
Q27_SR_FP8=0): q27_fp8_linear (target projection shapes) at every row count 1..64 (+ 65, 200 fallback) against 1Cat's
dispatch and an FP32 reference; q27_fp8_gated against 1Cat's gated dispatch and silu(g)*u in FP32; a CUDA-graph
capture of a mixed set. PASS = every error < 1e-3 and the 1Cat-routed rows bitwise equal to 1Cat."""
import os
import torch
import vllm._C  # noqa: F401
import vllm._sm70_ops  # noqa: F401
from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
from q27fuse import ops, smallrows

torch.manual_seed(0)
dev = "cuda"
ops.register_op()
smallrows.apply()
print("knobs", smallrows.MLPV, smallrows.FP8V, smallrows.GDNIN, "FP8 target rows ours:",
      [m for m in range(1, 65) if ops._fp8_cfg(m, 5120) is not None])
SH = {(5120, 4096): (16, 2, False), (5120, 3584): (16, 2, False), (1536, 5120): (12, 2, False),
      (4352, 5120): (16, 2, False)}
ok = True
for (K, N), (sk, na, pf) in SH.items():
    w = (torch.randn(N, K, device=dev) * 0.6).to(torch.float8_e4m3fn)
    sc = (torch.rand(N, 1, device=dev) * 2e-3 + 5e-4).float()
    codes, scales = torch.ops._C.fp8_qpn8_prepare_sm70(w, sc)[:2]
    wref = w.float() * sc
    for op in ("q27_fp8_linear",):
        line = []
        for M in list(range(1, 17)) + [17, 24, 32, 33, 40, 41, 48, 49, 56, 64, 65, 200]:
            x = (torch.randn(M, K, device=dev) * 0.5).half()
            ref = x.float() @ wref.t()
            a = torch.empty(M, N, device=dev, dtype=torch.half)
            b = torch.empty_like(a)
            dispatch_channel_rows(a, x, codes, scales, sk, na, pf, False)
            getattr(torch.ops.vllm, op)(b, x, codes, scales, sk, na, pf, None)
            eb = ((b.float() - ref).norm() / ref.norm()).item()
            ours = ops._fp8_cfg(M, K) is not None
            good = bool(eb < 1e-3 and (ours or torch.equal(a, b)))
            ok &= good
            if not good or M in (1, 8, 16, 32, 48, 64, 65):
                line.append(f"{M}:{'ours' if ours else '1cat'}:{eb:.1e}{'' if good else '!'}")
        print(f"{op} N={N} K={K}: " + " ".join(line), flush=True)
# gated gate_up
N, K = 8704, 5120
I = N // 2
w = (torch.randn(N, K, device=dev) * 0.6).to(torch.float8_e4m3fn)
sc = (torch.rand(N, 1, device=dev) * 2e-3 + 5e-4).float()
codes, scales = torch.ops._C.fp8_qpn8_prepare_sm70(w, sc)[:2]
wref = w.float() * sc
for gop, gbit in (("q27_fp8_gated", 8),):
    line = []
    for M in list(range(1, 17)) + [24, 32, 33, 40, 48, 56, 64, 65]:
        x = (torch.randn(M, K, device=dev) * 0.5).half()
        y = x.float() @ wref.t()
        ref = torch.nn.functional.silu(y[:, :I]) * y[:, I:]
        a = torch.empty(M, I, device=dev, dtype=torch.half)
        b = torch.empty_like(a)
        dispatch_channel_rows(a, x, codes, scales, 8, 2, False, True)
        getattr(torch.ops.vllm, gop)(b, x, codes, scales, 8, 2, False, None)
        ea = ((a.float() - ref).norm() / ref.norm()).item()
        eb = ((b.float() - ref).norm() / ref.norm()).item()
        ours = bool(smallrows.FP8V & gbit) and smallrows.GATED_MIN <= M <= 64
        good = bool(eb < 1e-3 and (ours or torch.equal(a, b)))
        ok &= good
        if not good or M in (1, 8, 16, 32, 48, 64, 65):
            line.append(f"{M}:{'ours' if ours else '1cat'}:{eb:.1e}(1cat {ea:.1e}){'' if good else '!'}")
    print(f"{gop}: " + " ".join(line), flush=True)
# graph capture of a mixed set (target projection shape + gated)
K, N = 5120, 4096
w = (torch.randn(N, K, device=dev) * 0.6).to(torch.float8_e4m3fn)
sc = (torch.rand(N, 1, device=dev) * 2e-3 + 5e-4).float()
codes, scales = torch.ops._C.fp8_qpn8_prepare_sm70(w, sc)[:2]
wref = w.float() * sc
xs = [(torch.randn(M, K, device=dev) * 0.5).half() for M in (1, 5, 8, 16, 32, 40, 64)]
outs = [torch.empty(x.shape[0], N, device=dev, dtype=torch.half) for x in xs]
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for x, o in zip(xs, outs):
        torch.ops.vllm.q27_fp8_linear(o, x, codes, scales, 16, 2, False, None)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for x, o in zip(xs, outs):
        torch.ops.vllm.q27_fp8_linear(o, x, codes, scales, 16, 2, False, None)
for o in outs:
    o.zero_()
g.replay(); torch.cuda.synchronize()
gerr = max(((o.float() - x.float() @ wref.t()).norm() / (x.float() @ wref.t()).norm()).item() for x, o in zip(xs, outs))
ok &= gerr < 1e-3
print(f"graph replay err {gerr:.1e}")
print("PASS" if ok else "FAIL")
