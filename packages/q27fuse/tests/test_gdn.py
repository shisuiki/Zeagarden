"""test_gdn.py - check the fused GDN glue kernels against PyTorch references and time them (graph-replayed).

Run on one GPU: python test_gdn.py
"""
import torch
import torch.nn.functional as F
from q27fuse import ops

torch.manual_seed(0)
dev = "cuda"
H, heads, D, QKV = 5120, 12, 128, 2560
ZW = heads * D
G = ops.gdn_ext()


def ref_in(x, w):
    ba = (x.float() @ w.float().t()).half()
    return ba[:, :heads].contiguous(), ba[:, heads:].contiguous()


def ref_norm(core, qkvz, w, eps):
    x = core.float()
    z = qkvz[:, QKV:QKV + ZW].reshape(-1, heads, D).float()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return ((xn * w.float()) * (z * torch.sigmoid(z))).half().reshape(-1, ZW)


def graph_time(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters // 20):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters


wba = (torch.randn(2 * heads, H, device=dev) * 0.02).half()
wn = (torch.rand(D, device=dev) + 0.5).half()
eps = 1e-6
ok = True
print(f"{'M':>4} {'b/a maxerr':>10} {'norm maxerr':>11} {'ref_ba_us':>9} {'gdn_in_us':>9} {'ref_norm_us':>11} "
      f"{'gdn_norm_us':>11}")
for M in [1, 2, 3, 4, 7, 8, 9, 12, 16, 24, 32, 48, 64, 100]:
    x = torch.randn(M, H, device=dev).half()
    qkvz = torch.randn(M, QKV + ZW, device=dev).half()
    core_in = torch.randn(M, heads, D, device=dev).half()
    b, a, core = ops.gdn_in_impl(x, wba, heads)
    rb, ra = ref_in(x, wba)
    e_ba = max((b.float() - rb.float()).abs().max().item(), (a.float() - ra.float()).abs().max().item())
    zero_ok = bool((core == 0).all().item())
    y = ops.gdn_gated_norm_impl(core_in, qkvz, wn, eps, QKV)
    ry = ref_norm(core_in, qkvz, wn, eps)
    e_n = (y.float() - ry.float()).abs().max().item()
    rel_n = e_n / ry.float().abs().max().item()
    good = zero_ok and e_ba < 2e-2 and rel_n < 2e-3
    ok &= good
    # timings: stock-equivalent references (cuBLAS b/a + tail copies + zeros; eager norm chain) vs fused
    def stock_in():
        ba = F.linear(x, wba)
        ba[:, :heads].contiguous()
        ba[:, heads:].contiguous()
        torch.zeros(M, heads, D, device=dev, dtype=torch.half)
    t_ref_in = graph_time(stock_in)
    t_in = graph_time(lambda: ops.gdn_in_impl(x, wba, heads))
    t_ref_n = graph_time(lambda: ref_norm(core_in, qkvz, wn, eps))
    t_n = graph_time(lambda: ops.gdn_gated_norm_impl(core_in, qkvz, wn, eps, QKV))
    print(f"{M:4d} {e_ba:10.2e} {e_n:11.2e} {t_ref_in:9.2f} {t_in:9.2f} {t_ref_n:11.2f} {t_n:11.2f}"
          f"{'' if good else '  FAIL zero=' + str(zero_ok)}")
print("ALL OK" if ok else "FAILURES")
