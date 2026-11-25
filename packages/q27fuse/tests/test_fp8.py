"""test_fp8.py - the q27_fp8_linear op (ops.fp8_linear_impl: fp8_lin.cu for 17..32 and 41..64 rows, 1Cat's QPN8
dispatch otherwise) against 1Cat's dispatch and an FP32 reference, at every row-count boundary, for the per-rank FP8
projection shapes; plus a CUDA-graph capture of each path."""
import torch
import vllm._C  # noqa: F401
import vllm._sm70_ops  # noqa: F401
from vllm.model_executor.kernels.linear.qpn.fp8 import dispatch_channel_rows
from q27fuse import ops

torch.manual_seed(0)
dev = "cuda"
ops.register_op()
QCFG = {(5120, 4096): (16, 2, False), (5120, 3584): (16, 2, False), (1536, 5120): (12, 2, False),
        (4352, 5120): (16, 2, False)}
ok = True
for (K, N), (sk, na, pf) in QCFG.items():
    w = (torch.randn(N, K, device=dev) * 0.6).to(torch.float8_e4m3fn)
    sc = (torch.rand(N, 1, device=dev) * 2e-3 + 5e-4).float()
    codes, scales = torch.ops._C.fp8_qpn8_prepare_sm70(w, sc)[:2]
    wref = w.float() * sc
    line = []
    for M in (1, 8, 9, 12, 16, 17, 24, 32, 33, 40, 41, 48, 64, 65, 200):
        x = (torch.randn(M, K, device=dev) * 0.5).half()
        ref = x.float() @ wref.t()
        a = torch.empty(M, N, device=dev, dtype=torch.half)
        b = torch.empty_like(a)
        dispatch_channel_rows(a, x, codes, scales, sk, na, pf, False)
        torch.ops.vllm.q27_fp8_linear(b, x, codes, scales, sk, na, pf, None)
        ea = ((a.float() - ref).norm() / ref.norm()).item()
        eb = ((b.float() - ref).norm() / ref.norm()).item()
        path = "ours" if ops._fp8_cfg(M, K) is not None else "1cat"
        good = eb < 1e-3 and (path == "ours" or torch.equal(a, b))
        ok &= good
        line.append(f"{M}:{path}:{eb:.1e}{'' if good else '!'}")
    # graph capture of a mixed set of row counts
    g = torch.cuda.CUDAGraph()
    xs = [(torch.randn(M, K, device=dev) * 0.5).half() for M in (8, 16, 32, 48, 64)]
    outs = [torch.empty(x.shape[0], N, device=dev, dtype=torch.half) for x in xs]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for x, o in zip(xs, outs):
            torch.ops.vllm.q27_fp8_linear(o, x, codes, scales, sk, na, pf, None)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for x, o in zip(xs, outs):
            torch.ops.vllm.q27_fp8_linear(o, x, codes, scales, sk, na, pf, None)
    for o in outs:
        o.zero_()
    g.replay(); torch.cuda.synchronize()
    gerr = max(((o.float() - x.float() @ wref.t()).norm() / (x.float() @ wref.t()).norm()).item() for x, o in zip(xs, outs))
    ok &= gerr < 1e-3
    print(f"N={N} K={K}: " + " ".join(line) + f" | graph replay err {gerr:.1e}", flush=True)
print("PASS" if ok else "FAIL")
