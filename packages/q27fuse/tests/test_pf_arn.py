#!/usr/bin/env python3
"""test_pf_arn.py (track prefill2; torchrun --nproc-per-node 4, under gpulock all) - csrc/pf_arn.cu, the prefill
all-reduce + residual add + Gemma norm, against (a) a torch reference of the same maths (four partials summed in FP32 in
rank order, rounded to fp16, + FP32 residual, Gemma norm) and (b) 1Cat's path (pynccl all-reduce + its fused add + norm);
back-to-back calls with fresh inputs (race check of the single buffer set); then timing against 1Cat's path
(>= 1.5 s warm-up, interleaved rounds, CUDA events). MODE=check|time|both|quant|h32."""
import os, sys, time, json, statistics
import torch, torch.distributed as dist
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank)
MODE = os.environ.get("MODE", "both")
DEFAULT_ROWS = "512,1024,1536,2048,4096" if MODE == "h32" else "4096,3001,2048,1024,512,260"
ROWS = [int(v) for v in os.environ.get("ROWS", DEFAULT_ROWS).split(",")]
CFGS = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("CFGS", "8:12:40:12,8:8:32:8,8:16:48:16,4:12:40:12,16:12:40:12").split(",")]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group
    init_distributed_environment(world_size=4, rank=rank, local_rank=rank, distributed_init_method="env://", backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=4)
    import vllm.model_executor.kernels.norm.sm70 as N
    tp = get_tp_group()
    from torch.utils.cpp_extension import load
    B = os.environ.get("ARN_BUILD", os.path.expanduser("~/.cache/q27fuse/build_arn")); os.makedirs(B, exist_ok=True)
    fl = ["-O3", "-gencode=arch=compute_70,code=sm_70", "-std=c++17", "--use_fast_math", "-Xptxas", "-v"]
    mk = lambda: load(name="q27fuse_pfarn", sources=[os.path.join(HERE, "q27fuse", "csrc", "pf_arn.cu")], build_directory=B,
                      extra_cuda_cflags=fl, verbose=False)
    X = mk() if rank == 0 else None
    tp.barrier()
    if rank: X = mk()
    tp.barrier()
    mine = X.alloc_buf()
    hs = [None] * 4; dist.all_gather_object(hs, X.ipc_handle(mine), group=tp.cpu_group)
    X.init([mine if r == rank else X.ipc_open(hs[r]) for r in range(4)], rank)
    H, eps = 5120, 1e-6
    ep = [0]

    def ours(x, res, w, cfg=CFGS[0]):
        ep[0] += 1
        rb, gp, gr, ga = cfg
        return X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    def sync():
        torch.cuda.synchronize(); tp.barrier()

    def ref(x, res, w):
        xs = [torch.empty_like(x) for _ in range(4)]
        dist.all_gather(xs, x, group=tp.device_group)
        s = torch.zeros_like(x, dtype=torch.float32)
        for p in range(4):
            s += xs[p].float()
        v = s.half().float() + res
        inv = torch.rsqrt(v.pow(2).mean(1, keepdim=True) + eps)
        return (v * inv * (1 + w.float())).half(), v

    torch.manual_seed(3)
    w = (torch.randn(H, device="cuda") * 0.1).half()
    if MODE in ("check", "both"):
        for M in ROWS:
            MQ = (M + 3) // 4
            q0, q1 = rank * MQ, min(M, (rank + 1) * MQ)
            for cfg in CFGS[:2]:
                worst = {}
                for it in range(6):                     # back-to-back calls, fresh inputs each time
                    torch.manual_seed(1000 * it + 10 * rank + M)
                    x = (torch.randn(M, H, device="cuda") * (0.3 + 0.2 * it)).half()
                    torch.manual_seed(77 + it)
                    res = torch.randn(M, H, device="cuda") * 6
                    o, ro = ours(x, res, w, cfg)
                    ro_q = ro[q0:q1].clone()
                    ro_ref_o, v = ref(x, res, w)
                    ca = N._sm70_dflash2_gemma_fused_add_rms_norm(tensor_model_parallel_all_reduce(x), res, w, eps)
                    torch.cuda.synchronize()
                    d = {"out_vs_ref_ndiff": int((o != ro_ref_o).sum()), "out_vs_ref_max": float((o.float() - ro_ref_o.float()).abs().max()),
                         "res_vs_ref_max": float((ro_q - v[q0:q1]).abs().max()),
                         "out_vs_1cat_ndiff": int((o != ca[0]).sum()), "out_vs_1cat_max": float((o.float() - ca[0].float()).abs().max()),
                         "ref_vs_1cat_ndiff": int((ro_ref_o != ca[0]).sum()),
                         "res_vs_1cat_max": float((ro_q - ca[1][q0:q1]).abs().max()), "nan": int(torch.isnan(o).sum())}
                    for k, val in d.items():
                        worst[k] = max(worst.get(k, 0), val)
                t = torch.tensor([float(v) for v in worst.values()], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp.device_group)
                log(json.dumps({"check": M, "cfg": cfg, "elements": M * H, **{k: float(v) for k, v in zip(worst, t.tolist())}}))
    if MODE == "quant":                                  # [prefill3] Job 5: the fused per-token INT8 output
        sys.path.insert(0, HERE)
        from q27fuse import pk
        PX = pk.ext() if rank == 0 else None
        tp.barrier()
        if rank: PX = pk.ext()
        tp.barrier()
        for M in ROWS:
            for cfg in CFGS[:2]:
                worst = {}
                for it in range(4):
                    torch.manual_seed(1000 * it + 10 * rank + M)
                    x = (torch.randn(M, H, device="cuda") * (0.3 + 0.2 * it)).half()
                    x[:, (7 * it) % H] *= 40                         # an outlier channel
                    torch.manual_seed(77 + it)
                    res = torch.randn(M, H, device="cuda") * 6
                    ep[0] += 1
                    rb, gp, gr, ga = cfg
                    torch.manual_seed(999 + it)
                    u_vec = torch.randint(-2 ** 31, 2 ** 31 - 1, (H,), device="cuda", dtype=torch.int32)
                    R_fused = torch.zeros(M, device="cuda", dtype=torch.int32)
                    R_ref = torch.zeros(M, device="cuda", dtype=torch.int32)
                    o, ro, xq, sx = X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga, 127.0, u_vec, R_fused)
                    o2, _ = ours(x, res, w, cfg)
                    q2 = torch.empty_like(o); s2 = torch.empty(M, device="cuda")
                    PX.quant_rows(o, q2, s2, 127.0, u_vec, R_ref)
                    torch.cuda.synchronize()
                    d = {"out_vs_plain_ndiff": int((o != o2).sum()), "xq_vs_kquant_ndiff": int((xq != q2).sum()),
                         "sx_vs_kquant_ndiff": int((sx != s2).sum()), "R_vs_kquant_ndiff": int((R_fused != R_ref).sum()),
                         "xq_max": float(xq.float().abs().max())}
                    for k, val in d.items():
                        worst[k] = max(worst.get(k, 0), val)
                t = torch.tensor([float(v) for v in worst.values()], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp.device_group)
                log(json.dumps({"quant_check": M, "cfg": cfg, **{k: float(v) for k, v in zip(worst, t.tolist())}}))
        # time: fused vs plain + separate quant pass
        for M in ROWS:
            torch.manual_seed(100 + rank)
            x = (torch.randn(M, H, device="cuda") * 0.5).half()
            torch.manual_seed(7)
            res = torch.randn(M, H, device="cuda") * 8
            q2 = torch.empty_like(x); s2 = torch.empty(M, device="cuda")

            def fused():
                ep[0] += 1
                rb, gp, gr, ga = CFGS[0]
                return X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga, 127.0)

            def plain():
                o, _ = ours(x, res, w)
                PX.quant_rows(o, q2, s2, 127.0)
            V = {"plain arn + quant_rows": plain, "fused arn+quant": fused, "plain arn only": lambda: ours(x, res, w)}
            sync(); t0 = time.time()
            while True:
                for _ in range(10):
                    plain(); fused()
                torch.cuda.synchronize()
                el = torch.tensor([time.time() - t0], device="cuda"); dist.all_reduce(el, op=dist.ReduceOp.MIN, group=tp.device_group)
                if float(el) > 1.5:
                    break
            times = {k: [] for k in V}
            for r in range(6):
                for k in (list(V) if r % 2 == 0 else list(V)[::-1]):
                    sync()
                    for _ in range(3):
                        V[k]()
                    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
                    for e0, e1 in evs:
                        e0.record(); V[k](); e1.record()
                    torch.cuda.synchronize()
                    times[k].append(statistics.median(e0.elapsed_time(e1) * 1e3 for e0, e1 in evs))
            out = {}
            for k, v in times.items():
                med = torch.tensor([statistics.median(v)], device="cuda"); mx = med.clone()
                dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=tp.device_group)
                out[k] = (round(float(med), 1), round(float(mx), 1))
            log(json.dumps({"quant_time": M, "cfg": CFGS[0], "us_rank0_maxrank": out}))
    if MODE == "h32":                                    # [prefill4] fused online H_32 + INT7/INT8 + Freivalds R
        sys.path.insert(0, HERE)
        from q27fuse import pk_h32, pf2
        H32X = pk_h32.ext() if rank == 0 else None
        tp.barrier()
        if rank: H32X = pk_h32.ext()
        tp.barrier()
        for M in ROWS:
            for qmax in (63.0, 127.0):
                for cfg in (pf2._arn_cfg(M), CFGS[0]):
                    worst = {}
                    for it in range(4):
                        torch.manual_seed(2000 * it + 10 * rank + M + int(qmax))
                        x = (torch.randn(M, H, device="cuda") * (0.3 + 0.2 * it)).half()
                        x[:, (7 * it) % H] *= 40                     # outlier channel
                        if it == 3:
                            x[0, :32] = 200.0                        # large coherent 32-block stress
                        torch.manual_seed(77 + it)
                        res = torch.randn(M, H, device="cuda") * 6
                        rb, gp, gr, ga = cfg
                        torch.manual_seed(999 + it)
                        u_vec = torch.randint(-2 ** 31, 2 ** 31 - 1, (H,), device="cuda", dtype=torch.int32)
                        R_fused = torch.zeros(M, device="cuda", dtype=torch.int32)
                        R_ref = torch.zeros(M, device="cuda", dtype=torch.int32)
                        ep[0] += 1
                        o, ro, xq, sx = X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga, qmax, u_vec, R_fused, True)
                        ep[0] += 1
                        o3, _, xq3, sx3 = X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga, qmax, None, None, True)
                        o2, _ = ours(x, res, w, cfg)
                        q2 = torch.empty_like(o); s2 = torch.empty(M, device="cuda")
                        H32X.quant_rows_h32(o2, q2, s2, qmax, u_vec, R_ref)
                        torch.cuda.synchronize()
                        d = {
                            "out_vs_plain_ndiff": int((o != o2).sum()),
                            "xq_vs_h32_ndiff": int((xq != q2).sum()),
                            "sx_vs_h32_ndiff": int((sx != s2).sum()),
                            "R_vs_h32_ndiff": int((R_fused != R_ref).sum()),
                            "xq_nor_ndiff": int((xq != xq3).sum()),
                            "sx_nor_ndiff": int((sx != sx3).sum()),
                            "out_nor_ndiff": int((o != o3).sum()),
                            "xq_max": float(xq.float().abs().max()),
                        }
                        for k, val in d.items():
                            worst[k] = max(worst.get(k, 0), val)
                    t = torch.tensor([float(v) for v in worst.values()], device="cuda")
                    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp.device_group)
                    res_d = {k: float(v) for k, v in zip(worst, t.tolist())}
                    log(json.dumps({"h32_check": M, "qmax": int(qmax), "cfg": cfg, **res_d}))
                    for ck in ("out_vs_plain_ndiff", "xq_vs_h32_ndiff", "sx_vs_h32_ndiff", "R_vs_h32_ndiff",
                               "xq_nor_ndiff", "sx_nor_ndiff", "out_nor_ndiff"):
                        assert res_d[ck] == 0.0, f"pf_arn h32 mismatch {ck}={res_d[ck]} at M={M} qmax={qmax} cfg={cfg}"
        # Timing: plain arn + quant_rows_h32 vs fused arn_h32 across M = 512, 1024, 1536, 2048, 4096
        for M in ROWS:
            cfg = pf2._arn_cfg(M)
            rb, gp, gr, ga = cfg
            torch.manual_seed(100 + rank)
            x = (torch.randn(M, H, device="cuda") * 0.5).half()
            torch.manual_seed(7)
            res = torch.randn(M, H, device="cuda") * 8
            u_vec = torch.randint(-2 ** 31, 2 ** 31 - 1, (H,), device="cuda", dtype=torch.int32)
            R_buf = torch.zeros(M, device="cuda", dtype=torch.int32)
            q2 = torch.empty_like(x); s2 = torch.empty(M, device="cuda")

            def fused_r():
                ep[0] += 1
                return X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga, 63.0, u_vec, R_buf, True)

            def fused_nor():
                ep[0] += 1
                return X.arn(x, res, w, eps, ep[0], rb, gp, gr, ga, 63.0, None, None, True)

            def plain_q():
                o, _ = ours(x, res, w, cfg)
                H32X.quant_rows_h32(o, q2, s2, 63.0, u_vec, R_buf)

            def plain_only():
                return ours(x, res, w, cfg)

            V = {"plain arn only": plain_only, "plain arn + quant_rows_h32": plain_q,
                 "fused arn_h32 (no R)": fused_nor, "fused arn_h32 + R": fused_r}
            sync(); t0 = time.time()
            while True:
                for _ in range(10):
                    plain_q(); fused_r()
                torch.cuda.synchronize()
                el = torch.tensor([time.time() - t0], device="cuda"); dist.all_reduce(el, op=dist.ReduceOp.MIN, group=tp.device_group)
                if float(el) > 1.5:
                    break
            times = {k: [] for k in V}
            for r in range(6):
                for k in (list(V) if r % 2 == 0 else list(V)[::-1]):
                    sync()
                    for _ in range(3):
                        V[k]()
                    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
                    for e0, e1 in evs:
                        e0.record(); V[k](); e1.record()
                    torch.cuda.synchronize()
                    times[k].append(statistics.median(e0.elapsed_time(e1) * 1e3 for e0, e1 in evs))
            out = {}
            for k, v in times.items():
                med = torch.tensor([statistics.median(v)], device="cuda"); mx = med.clone()
                dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=tp.device_group)
                out[k] = (round(float(med), 1), round(float(mx), 1))
            log(json.dumps({"h32_time": M, "cfg": cfg, "us_rank0_maxrank": out}))
        log("PF_ARN_H32_PASS")
    if MODE in ("time", "both"):
        for M in ROWS:
            torch.manual_seed(100 + rank)
            x = (torch.randn(M, H, device="cuda") * 0.5).half()
            torch.manual_seed(7)
            res = torch.randn(M, H, device="cuda") * 8
            V = {"1cat (nccl+norm)": lambda: N._sm70_dflash2_gemma_fused_add_rms_norm(tensor_model_parallel_all_reduce(x), res, w, eps),
                 "nccl": lambda: tensor_model_parallel_all_reduce(x)}
            for cfg in CFGS:
                V[f"ours rb{cfg[0]} p{cfg[1]} r{cfg[2]} a{cfg[3]}"] = (lambda cfg=cfg: ours(x, res, w, cfg))
            sync(); t0 = time.time(); n = 0
            while True:
                for _ in range(10):
                    V["1cat (nccl+norm)"](); ours(x, res, w)
                n += 10; torch.cuda.synchronize()
                el = torch.tensor([time.time() - t0], device="cuda"); dist.all_reduce(el, op=dist.ReduceOp.MIN, group=tp.device_group)
                if float(el) > 1.5:
                    break
            times = {k: [] for k in V}
            for r in range(5):
                for k in (list(V) if r % 2 == 0 else list(V)[::-1]):
                    sync()
                    for _ in range(3):
                        V[k]()
                    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
                    for e0, e1 in ev:
                        e0.record(); V[k](); e1.record()
                    torch.cuda.synchronize()
                    times[k].append(statistics.median(e0.elapsed_time(e1) * 1e3 for e0, e1 in ev))
            out = {}
            for k, v in times.items():
                med = torch.tensor([statistics.median(v)], device="cuda"); mx = med.clone()
                dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=tp.device_group)
                out[k] = (round(float(med), 1), round(float(mx), 1))
            log(json.dumps({"time": M, "us_rank0_maxrank": out}))
    sync()
    log("TEST_DONE")
