"""test_gdn_core.py check|bench [C list] [cfg list] - gdn_core.cu (the fused GDN decode core) at the per-rank (TP4)
Qwen3.8-27B shape against the chain it replaces in the server (mode 3):
  gdn_glue.gdn_in -> 1Cat causal_conv1d_update (triton, silu, in place) -> 1Cat gdn_decode_mixed_qkv_global_state_sm70
  -> gdn_glue.gdn_gated_norm
and the same chain with gdn_state.cu as the state update (whose math gdn_core.cu reuses, so y should match bitwise).
check: y, conv state and SSM state after one step, with padded rows (state index -1, rows beyond Tdec).
bench: µs per call in a CUDA graph cycling NCOPY state copies (states from DRAM, as in decode).
phases: µs with phases skipped (the kernel's debug bits). gdn_fused (fcheck/fbench/ftime): attic/test_gdn_fused.py."""
import os, sys
import torch
from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import gdn_decode_mixed_qkv_global_state_sm70 as cat_gdn
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "attic")]
os.environ.setdefault("Q27_FUSE_BUILD", os.path.expanduser("~/.cache/q27fuse/build_test"))
from q27fuse import ops
import q27attic                                     # gdn_state.cu, the reference whose math gdn_core.cu reuses

torch.manual_seed(0)
dev, HV, HK, D, KH = "cuda", 12, 4, 128, 5120
DIM = (2 * HK + HV) * D
NCOPY, REPS = 8, 20
core_ext, glue, gs = ops.gdn_core_ext(), ops.gdn_ext(), q27attic.gdn_state_ext()
CFGS = ["8x4", "4x4", "2x4", "16x4", "16x8", "8x8", "4x8", "8x16"]


def make(C, slots_extra=5):
    slots = C + slots_extra
    P = dict(
        x=(torch.randn(C, KH, device=dev) * 0.3).half(),
        qkvz=(torch.randn(C, DIM + HV * D, device=dev) * 0.5).half(),
        wba=(torch.randn(2 * HV, KH, device=dev) * 0.02).half(),
        cw=(torch.randn(DIM, 4, device=dev) * 0.5).half(),
        A_log=(torch.rand(HV, device=dev) * 2 - 1).float(),
        dtb=(torch.randn(HV, device=dev) * 0.3).half(),
        nw=(1 + 0.1 * torch.randn(D, device=dev)).half(),
        # page-strided like the server's hybrid cache: one page per slot holding both states
        cst=(torch.randn(slots, 3, DIM, device=dev) * 0.5).half(),
        ssm=(torch.randn(slots, HV, D, D, device=dev) * 0.1).float(),
        idx=torch.randperm(slots, device=dev)[:C].int(),
    )
    return P


def ref_chain(P, cst, ssm, idx, use_gs=False, cfg_gs=0):
    C = P["x"].size(0)
    b = torch.empty(C, HV, device=dev, dtype=torch.half); a = torch.empty_like(b)
    core = torch.empty(C, HV, D, device=dev, dtype=torch.half)
    glue.gdn_in(P["x"], P["wba"], b, a, core)
    mixed = P["qkvz"][:, :DIM].clone()
    causal_conv1d_update(mixed, cst.transpose(-1, -2), P["cw"], None, "silu", conv_state_indices=idx,
                         validate_data=True)
    if use_gs:
        gs.gdn_state(mixed, a, b, P["A_log"], P["dtb"], ssm, idx, core, D ** -0.5, True, cfg_gs)
    else:
        cat_gdn(mixed, a, b, P["A_log"], P["dtb"], ssm, idx, core, None, True)
    y = torch.empty(C, HV * D, device=dev, dtype=torch.half)
    glue.gdn_gated_norm(core, P["qkvz"], P["nw"], y, 1e-6, DIM)
    return y


def fused(P, cst, ssm, idx, Tdec, cfg, core=None, y=None, dbg=0):
    C = P["x"].size(0)
    core = core if core is not None else torch.empty(C, HV, D, device=dev, dtype=torch.half)
    y = y if y is not None else torch.empty(C, HV * D, device=dev, dtype=torch.half)
    core_ext.gdn_core(P["x"], P["qkvz"], P["wba"], P["cw"], cst, P["A_log"], P["dtb"], ssm, idx, Tdec, P["nw"], 1e-6,
                      core, y, ops.gdn_core_workspace(torch.device(dev), C), cfg, dbg)
    return y


def rel(a, b):
    return ((a.float() - b.float()).norm() / max(b.float().norm().item(), 1e-30)).item()


def check(CS, cfgs):
    for C in CS:
        P = make(C)
        # padded rows: one index -1 (when C > 2) and the last row beyond Tdec (when C > 1)
        idx = P["idx"].clone()
        if C > 2:
            idx[1] = -1
        Tdec = C - 1 if C > 1 else C
        idx_ref = idx.clone()
        if Tdec < C:
            idx_ref[Tdec:] = -1
        valid = idx_ref >= 0
        c_ref, s_ref = P["cst"].clone(), P["ssm"].clone()
        y_ref = ref_chain(P, c_ref, s_ref, idx_ref)
        c_gs, s_gs = P["cst"].clone(), P["ssm"].clone()
        y_gs = ref_chain(P, c_gs, s_gs, idx_ref, use_gs=True, cfg_gs=q27attic.gdn_state_cfg(C))
        for cfg in cfgs:
            c_f, s_f = P["cst"].clone(), P["ssm"].clone()
            y_f = fused(P, c_f, s_f, idx, Tdec, cfg)
            torch.cuda.synchronize()
            ok_pad = bool((y_f[~valid] == 0).all())
            same_gs = bool(torch.equal(y_f[valid], y_gs[valid]))
            nd = int((y_f[valid] != y_gs[valid]).sum())
            cst_eq = bool(torch.equal(c_f, c_ref))
            print(f"C={C:3d} cfg {CFGS[cfg]:>5}: y vs 1Cat chain {rel(y_f[valid], y_ref[valid]):.1e}, "
                  f"vs gdn_state chain {'bitwise' if same_gs else f'{rel(y_f[valid], y_gs[valid]):.1e} ({nd} differ)'}; "
                  f"ssm vs 1Cat {rel(s_f, s_ref):.1e}, vs gdn_state {rel(s_f, s_gs):.1e}; "
                  f"conv state {'bitwise' if cst_eq else 'DIFFERS %d' % int((c_f != c_ref).sum())}; pad rows 0: {ok_pad}",
                  flush=True)
        # the counters must be back at zero
        assert int(ops.gdn_core_workspace(torch.device(dev), C).abs().sum()) == 0, "counter workspace not reset"


def graph_time(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            fn(i % NCOPY)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(REPS):
            fn(i % NCOPY)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / REPS)
    return sorted(ts)[3]


def bench(CS, cfgs):
    print(f"{'C':>3} | {'chain (1Cat state)':>18} {'chain (gdn_state)':>18} | " + " ".join(f"{CFGS[c]:>7}" for c in cfgs)
          + " | best, saving per layer, state floor")
    for C in CS:
        P = make(C, slots_extra=0)
        idx = P["idx"]
        csts = [P["cst"].clone() for _ in range(NCOPY)]
        ssms = [P["ssm"].clone() for _ in range(NCOPY)]
        b = torch.empty(C, HV, device=dev, dtype=torch.half); a = torch.empty_like(b)
        core = torch.empty(C, HV, D, device=dev, dtype=torch.half)
        y = torch.empty(C, HV * D, device=dev, dtype=torch.half)
        mixed = P["qkvz"][:, :DIM].clone()
        cfg_gs = q27attic.gdn_state_cfg(C)

        def chain(i, use_gs):
            glue.gdn_in(P["x"], P["wba"], b, a, core)
            mixed.copy_(P["qkvz"][:, :DIM])          # the server's conv runs in place on the projection output
            causal_conv1d_update(mixed, csts[i].transpose(-1, -2), P["cw"], None, "silu", conv_state_indices=idx)
            if use_gs:
                gs.gdn_state(mixed, a, b, P["A_log"], P["dtb"], ssms[i], idx, core, D ** -0.5, True, cfg_gs)
            else:
                cat_gdn(mixed, a, b, P["A_log"], P["dtb"], ssms[i], idx, core, None, True)
            glue.gdn_gated_norm(core, P["qkvz"], P["nw"], y, 1e-6, DIM)

        t1 = graph_time(lambda i: chain(i, False))
        t2 = graph_time(lambda i: chain(i, True))
        tf = [graph_time(lambda i: fused(P, csts[i], ssms[i], idx, C, cfg, core, y)) for cfg in cfgs]
        best = min(range(len(cfgs)), key=lambda j: tf[j])
        floor = C * HV * D * D * 4 * 2 / 900e3     # state read + write at 900 GB/s, µs
        print(f"{C:3d} | {t1:18.1f} {t2:18.1f} | " + " ".join(f"{t:7.1f}" for t in tf)
              + f" | {CFGS[cfgs[best]]} {tf[best]:.1f} us, {t1 - tf[best]:.1f} us, floor {floor:.1f} us", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    CS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16, 32, 64]
    cfgs = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(len(CFGS)))
    if what == "phases":       # µs with phases skipped (debug bits), per config
        for C in CS:
            P = make(C, slots_extra=0)
            csts = [P["cst"].clone() for _ in range(NCOPY)]; ssms = [P["ssm"].clone() for _ in range(NCOPY)]
            core = torch.empty(C, HV, D, device=dev, dtype=torch.half); y = torch.empty(C, HV * D, device=dev, dtype=torch.half)
            for cfg in cfgs:
                r = [graph_time(lambda i: fused(P, csts[i], ssms[i], P["idx"], C, cfg, core, y, d)) for d in (0, 1, 2, 4, 7)]
                print(f"C={C:3d} {CFGS[cfg]:>5}: full {r[0]:6.1f}  -b/a {r[1]:6.1f}  -qk counter {r[2]:6.1f}  -norm {r[3]:6.1f}  "
                      f"-all three {r[4]:6.1f}", flush=True)
    else:
        (check if what == "check" else bench)(CS, cfgs)
