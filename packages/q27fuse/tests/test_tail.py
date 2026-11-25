#!/usr/bin/env python3
"""test_tail.py - single-GPU checks of the track-"tail" kernels against the code they replace (bitwise), plus timings.

  python test_tail.py check   row_argmax_pair vs torch max/add/float/stack; greedy_verify_fused vs 1Cat's greedy_verify
                              + get_num_sampled_and_rejected; tail_mamba precopy/postprocess vs mamba_align's on
                              synthetic state tensors / block tables; run_accept vs postprocess_state; gdnmeta's
                              compute_common_gdn_attn_metadata vs 1Cat's (every field)
  python test_tail.py bench   eager timings (>= 1 s warm-up on the GPU first, brief rule)
Run with PYTHONPATH=<tree> Q27_FUSE_BUILD=<build prefix>, one GPU (gpulock one)."""
import sys
import time

import torch

from q27fuse import tail as T

dev = torch.device("cuda")
torch.manual_seed(0)
V, START = 62080, 3 * 62080


def ref_pair(logits, start):
    v, i = logits.max(dim=-1)
    return torch.stack([v.float(), (i + start).float()], dim=-1)


def our_pair(logits, start):
    out = torch.empty((logits.shape[0], 2), dtype=torch.float32, device=dev)
    part, cnt = T._ws(dev)
    T.ext().row_argmax_pair(logits, start, out.data_ptr(), part, cnt, T.SPLITS)
    return out


def check_argmax():
    bad = 0
    for rows in (1, 2, 5, 8, 16, 32, 40, 64, 80, 128):
        for case in ("rand", "ties", "nan", "zeros", "neg", "inf"):
            x = (torch.randn(rows, V, device=dev) * 4).half()
            if case == "ties":              # equal maxima at two places per row; the lower index must win
                m = x.max() + 1
                for r in range(rows):
                    a, b = sorted(torch.randint(0, V, (2,)).tolist())
                    x[r, b] = m; x[r, a] = m
            elif case == "nan":
                for r in range(0, rows, 2):
                    a, b = sorted(torch.randint(0, V, (2,)).tolist())
                    x[r, b] = float("nan"); x[r, a] = float("nan")
            elif case == "zeros":           # max is 0 with both signs present
                x = -x.abs()
                for r in range(rows):
                    a, b = sorted(torch.randint(0, V, (2,)).tolist())
                    x[r, a] = -0.0; x[r, b] = 0.0
            elif case == "neg":
                x = -x.abs() - 1
            elif case == "inf":
                x[:, :] = -float("inf")
                x[0, 7] = float("inf")
            r = ref_pair(x, START)
            o = our_pair(x, START)
            same = torch.equal(r.view(torch.int32), o.view(torch.int32))
            if not same:
                bad += 1
                d = (r.view(torch.int32) != o.view(torch.int32)).any(dim=-1).nonzero().flatten()[:3].tolist()
                print(f"argmax MISMATCH rows={rows} case={case} rows {d}: ref {r[d].tolist()} ours {o[d].tolist()}")
    # a strided view (a padded logits buffer), offset rows
    big = (torch.randn(70, V + 64, device=dev)).half()
    x = big[3:67, :V]
    if not torch.equal(ref_pair(x, START).view(torch.int32), our_pair(x, START).view(torch.int32)):
        bad += 1; print("argmax MISMATCH strided")
    print(f"row_argmax_pair: {'OK' if not bad else f'{bad} mismatching cases'}")
    return bad


def check_gverify():
    from vllm.v1.worker.gpu.spec_decode import sm70_greedy_verify as GV
    from vllm.v1.worker.gpu import input_batch as IB
    bad = 0
    g = torch.Generator().manual_seed(1)
    for trial in range(300):
        K = int(torch.randint(1, 5, (1,), generator=g))
        reqs = int(torch.randint(1, 33, (1,), generator=g))
        nlog = torch.randint(1, K + 2, (reqs,), generator=g) if trial % 3 == 0 else torch.full((reqs,), K + 1)
        cu = torch.zeros(reqs + 1, dtype=torch.int32); cu[1:] = torch.cumsum(nlog, 0)
        nl = int(cu[-1])
        ntok = nl + int(torch.randint(0, 5, (1,), generator=g))
        input_ids = torch.randint(0, 50, (ntok,), generator=g, dtype=torch.int32)
        logits_indices = torch.sort(torch.randperm(ntok, generator=g)[:nl]).values.to(torch.int64)
        target = torch.randint(0, 50, (nl,), generator=g, dtype=torch.int64)
        # make many drafts match: copy target into the draft positions with probability 0.8
        drafts = input_ids[logits_indices].to(torch.int64)
        keep = torch.rand(nl, generator=g) < 0.8
        drafts = torch.where(keep, torch.roll(target, 1), drafts)
        input_ids[logits_indices] = drafts.to(torch.int32)
        max_reqs = 40
        idx_mapping = torch.randperm(max_reqs, generator=g)[:reqs].to(torch.int32)
        prefill_len = torch.randint(100, 200, (max_reqs,), generator=g, dtype=torch.int32)
        seq_lens = (prefill_len[idx_mapping.long()] + torch.randint(-3, 50, (reqs,), generator=g)).to(torch.int32)
        dv = [t.to(dev) for t in (target, input_ids, logits_indices, cu, seq_lens, prefill_len, idx_mapping)]
        target_d, ids_d, li_d, cu_d, sl_d, pl_d, im_d = dv
        s_ref, n_ref = GV.greedy_verify(target_d, ids_d[li_d], cu_d, K)
        n_ref, r_ref = IB.get_num_sampled_and_rejected(n_ref, sl_d, cu_d, im_d, pl_d)
        s_o = torch.empty((reqs, K + 1), dtype=torch.int64, device=dev)
        n_o = torch.empty(reqs, dtype=torch.int32, device=dev)
        r_o = torch.empty_like(n_o)
        T.ext().greedy_verify_fused(target_d, ids_d, li_d, cu_d, sl_d, pl_d, im_d, s_o, n_o, r_o)
        if not (torch.equal(s_ref, s_o) and torch.equal(n_ref, n_o) and torch.equal(r_ref, r_o)):
            bad += 1
            if bad <= 3:
                print(f"gverify MISMATCH trial {trial} K={K} reqs={reqs}\n ref {s_ref.tolist()} {n_ref.tolist()} "
                      f"{r_ref.tolist()}\n ours {s_o.tolist()} {n_o.tolist()} {r_o.tolist()}")
    print(f"greedy_verify_fused: {'OK' if not bad else f'{bad} mismatching trials'} (300 trials)")
    return bad


class Ctx:
    pass


def make_ctx(num_reqs_max, replay, g):
    """synthetic align-mode context: 2 groups, per group 3 GDN layers x (conv, temporal) states"""
    ctx = Ctx()
    maxb = 8
    nblocks = max(64, num_reqs_max * maxb)
    groups = 2
    bts = [torch.randint(0, nblocks, (num_reqs_max, maxb), generator=g, dtype=torch.int32).to(dev) for _ in range(groups)]
    # unique blocks per (row, col) within a group so different requests never share a destination
    for gi in range(groups):
        perm = torch.randperm(nblocks, generator=g)[: num_reqs_max * maxb] if nblocks >= num_reqs_max * maxb else None
        if perm is None:
            raise SystemExit("nblocks too small")
        bts[gi].copy_(perm.view(num_reqs_max, maxb).to(torch.int32))
    states, meta = [], []
    for gi in range(groups):
        for layer in range(3):
            conv = torch.randn(nblocks, 3, 96, device=dev).half()          # conv width 3, 96 fp16 per token
            temp = torch.randn(nblocks, 4, 33, 31, device=dev)              # fp32 temporal state, odd size
            states += [conv, temp]
            meta += [(conv, 3 * 96 * 2, 2, 96, 3, gi), (temp, 4 * 33 * 31 * 4, 4, 4 * 33 * 31, 0, gi)]
    ctx.block_table_ptrs = torch.tensor([b.data_ptr() for b in bts], dtype=torch.int64, device=dev)
    ctx.block_table_stride_req = maxb
    ctx.state_base_addrs = torch.tensor([m[0].data_ptr() for m in meta], dtype=torch.int64, device=dev)
    ctx.state_block_strides = torch.tensor([m[1] for m in meta], dtype=torch.int64, device=dev)
    ctx.state_elem_sizes = torch.tensor([m[2] for m in meta], dtype=torch.int32, device=dev)
    ctx.state_inner_sizes = torch.tensor([m[3] for m in meta], dtype=torch.int64, device=dev)
    ctx.state_conv_widths = torch.tensor([m[4] for m in meta], dtype=torch.int32, device=dev)
    ctx.state_group_indices = torch.tensor([m[5] for m in meta], dtype=torch.int32, device=dev)
    ctx.num_states = len(meta)
    ctx.block_size = 16
    ctx.temporal_replay = replay
    ctx.is_initialized = True
    ctx.num_accepted_tokens_out = torch.zeros(num_reqs_max, dtype=torch.int32, device=dev)
    return ctx, states, bts


def check_mamba():
    from vllm.v1.worker.gpu import mamba_align as MA
    from q27fuse import tail_mamba as TM
    bad = 0
    g = torch.Generator().manual_seed(2)
    for trial in range(60):
        replay = trial % 2 == 1
        maxr = 8
        ctx, states, bts = make_ctx(maxr, replay, g)
        reqs = int(torch.randint(1, maxr + 1, (1,), generator=g))
        idx = torch.randperm(maxr, generator=g)[:reqs].to(torch.int32)
        if trial % 5 == 0:
            idx[0] = -1
        # precopy inputs
        state_idx = torch.randint(0, 6, (maxr,), generator=g, dtype=torch.int32)
        src_col = torch.where(torch.rand(maxr, generator=g) < 0.7, state_idx,
                              torch.randint(-1, 6, (maxr,), generator=g, dtype=torch.int32))
        if trial % 4 == 0:
            src_col = state_idx.clone()                     # nothing to copy (the steady case)
        token_bias = torch.randint(0, 3, (maxr,), generator=g, dtype=torch.int32)
        qlen = torch.where(torch.rand(reqs, generator=g) < 0.5, torch.ones(reqs, dtype=torch.int32),
                           torch.full((reqs,), 4, dtype=torch.int32))
        qsl = torch.zeros(reqs + 1, dtype=torch.int32); qsl[1:] = torch.cumsum(qlen, 0)
        args = [t.to(dev) for t in (state_idx, src_col, token_bias, idx, qsl)]
        a_states = [s.clone() for s in states]
        b_states = [s.clone() for s in states]
        # original on a_states
        for s, a in zip(states, a_states):
            s.copy_(a)
        MA.run_mamba_align_precopy(ctx, reqs, *args)
        ref = [s.clone() for s in states]
        for s, b in zip(states, b_states):
            s.copy_(b)
        TM.run_precopy(ctx, reqs, *args)
        if not all(torch.equal(r.view(torch.uint8), s.view(torch.uint8)) for r, s in zip(ref, states)):
            bad += 1; print(f"precopy MISMATCH trial {trial} replay={replay}")
        nchg = sum(int((r.view(torch.uint8) != a.view(torch.uint8)).any()) for r, a in zip(ref, a_states))
        # postprocess inputs
        num_acc = torch.randint(1, 4, (maxr,), generator=g, dtype=torch.int32)
        nnc = torch.randint(30, 90, (maxr,), generator=g, dtype=torch.int32)
        if trial % 4 == 0:
            nnc = torch.full((maxr,), 41, dtype=torch.int32); num_acc = torch.ones(maxr, dtype=torch.int32)
        st_idx = torch.randint(0, 5, (maxr,), generator=g, dtype=torch.int32)
        for s, a in zip(states, a_states):
            s.copy_(a)
        na_ref = num_acc.to(dev)
        MA.run_mamba_align_postprocess(ctx, reqs, na_ref, st_idx.to(dev), nnc.to(dev), idx.to(dev))
        ref = [s.clone() for s in states]
        for s, b in zip(states, b_states):
            s.copy_(b)
        na_o = num_acc.to(dev)
        TM.run_postprocess(ctx, reqs, na_o, st_idx.to(dev), nnc.to(dev), idx.to(dev))
        ok = all(torch.equal(r.view(torch.uint8), s.view(torch.uint8)) for r, s in zip(ref, states))
        if not ok or not torch.equal(na_ref, na_o):
            bad += 1; print(f"postprocess MISMATCH trial {trial} replay={replay} states_ok={ok} "
                            f"num_acc ref {na_ref.tolist()} ours {na_o.tolist()}")
        if trial < 4:
            nchg2 = sum(int((r.view(torch.uint8) != a.view(torch.uint8)).any()) for r, a in zip(ref, a_states))
            print(f"  trial {trial}: precopy changed {nchg} state tensors, postprocess {nchg2}")
    print(f"tail_mamba: {'OK' if not bad else f'{bad} mismatching trials'} (60 trials)")
    return bad


def check_accept():
    """run_accept vs 1Cat's postprocess_state sequence: _scatter_num_accepted_kernel, record_accepted
    (index_copy of index_select), run_mamba_align_postprocess"""
    from vllm.v1.worker.gpu import mamba_align as MA
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as MH
    from q27fuse import tail_mamba as TM
    bad = 0
    g = torch.Generator().manual_seed(4)
    for trial in range(60):
        replay = trial % 2 == 1
        maxr = 8
        ctx, states, bts = make_ctx(maxr, replay, g)
        reqs = int(torch.randint(1, maxr + 1, (1,), generator=g))
        idx = torch.randperm(maxr, generator=g)[:reqs].to(torch.int32).to(dev)
        num_sampled = torch.randint(0, 4, (reqs,), generator=g, dtype=torch.int32).to(dev)
        nnc = torch.randint(30, 90, (maxr,), generator=g, dtype=torch.int32)
        if trial % 4 == 0:
            nnc = torch.full((maxr,), 41, dtype=torch.int32)
        nnc = nnc.to(dev)
        st_idx = torch.randint(0, 5, (maxr,), generator=g, dtype=torch.int32).to(dev)
        acc0 = torch.randint(1, 5, (maxr,), generator=g, dtype=torch.int32).to(dev)
        rep0 = torch.randint(1, 5, (maxr,), generator=g, dtype=torch.int32).to(dev)
        base = [s.clone() for s in states]
        # 1Cat sequence
        na_ref, rep_ref = acc0.clone(), rep0.clone()
        MH._scatter_num_accepted_kernel[(reqs,)](idx, num_sampled, na_ref)
        index = idx.long()
        rep_ref.index_copy_(0, index, na_ref.index_select(0, index))
        MA.run_mamba_align_postprocess(ctx, reqs, na_ref, st_idx, nnc, idx)
        ref = [s.clone() for s in states]
        for s, b in zip(states, base):
            s.copy_(b)
        na_o, rep_o = acc0.clone(), rep0.clone()
        TM.run_accept(ctx, reqs, num_sampled, na_o, rep_o if trial % 3 else None, st_idx, nnc, idx)
        ok = all(torch.equal(r.view(torch.uint8), s.view(torch.uint8)) for r, s in zip(ref, states))
        rep_ok = torch.equal(rep_ref, rep_o) if trial % 3 else torch.equal(rep_o, rep0)
        if not ok or not torch.equal(na_ref, na_o) or not rep_ok:
            bad += 1
            print(f"accept MISMATCH trial {trial}: states {ok} num_acc ref {na_ref.tolist()} ours {na_o.tolist()} "
                  f"replay ref {rep_ref.tolist()} ours {rep_o.tolist()}")
    print(f"run_accept: {'OK' if not bad else f'{bad} mismatching trials'} (60 trials)")
    return bad


def warm(seconds=1.2):
    a = torch.randn(4096, 4096, device=dev).half()
    t = time.time()
    while time.time() - t < seconds:
        for _ in range(20):
            a = a @ a
            a = a / a.abs().max()
    torch.cuda.synchronize()


def timeit(fn, reps=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / reps


def bench():
    from vllm.v1.worker.gpu import mamba_align as MA
    from q27fuse import tail_mamba as TM
    warm()
    print("eager, us per call (back to back, host far ahead)")
    for rows in (5, 8, 16, 32, 64):
        x = torch.randn(rows, V, device=dev).half()
        tr = timeit(lambda: ref_pair(x, START))
        to = timeit(lambda: our_pair(x, START))
        print(f"  argmax pair rows {rows:3d}: torch max+add+2 casts+stack {tr:6.1f}   ours {to:6.1f}")
    g = torch.Generator().manual_seed(3)
    for reqs in (1, 8, 16, 32):
        ctx, states, bts = make_ctx(32, True, g)
        ctx.num_states = 96            # the server's count (48 GDN layers x 2); extra entries reuse state 0..11
        for name in ("state_base_addrs", "state_block_strides", "state_elem_sizes", "state_inner_sizes",
                     "state_conv_widths", "state_group_indices"):
            t = getattr(ctx, name)
            setattr(ctx, name, t.repeat(8)[:96].contiguous())
        idx = torch.arange(reqs, dtype=torch.int32, device=dev)
        st = torch.arange(32, dtype=torch.int32, device=dev) % 6
        tb = torch.zeros(32, dtype=torch.int32, device=dev)
        qsl = (torch.arange(reqs + 1, dtype=torch.int32, device=dev) * 2)
        nnc = torch.full((32,), 41, dtype=torch.int32, device=dev)
        na = torch.ones(32, dtype=torch.int32, device=dev)
        t1 = timeit(lambda: MA.run_mamba_align_precopy(ctx, reqs, st, st, tb, idx, qsl))
        t2 = timeit(lambda: TM.run_precopy(ctx, reqs, st, st, tb, idx, qsl))
        t3 = timeit(lambda: MA.run_mamba_align_postprocess(ctx, reqs, na, st, nnc, idx))
        t4 = timeit(lambda: TM.run_postprocess(ctx, reqs, na, st, nnc, idx))
        print(f"  mamba align (no copy due) reqs {reqs:2d}: precopy 1Cat {t1:6.1f} ours {t2:6.1f}   "
              f"postprocess 1Cat {t3:6.1f} ours {t4:6.1f}")


def check_gdnmeta():
    """tail gdnmeta: compute_common_gdn_attn_metadata (ours) vs 1Cat's on random batches - every field equal (tensors:
    dtype, shape, values); pure speculative batches repeated (cached mask path) and mixed ones (1Cat's path)"""
    import random
    from vllm.v1.attention.ops import gdn_state as GS
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as MH
    T.ACTIVE.add("gdnmeta")
    if not getattr(MH.compute_common_gdn_attn_metadata, "__qualname__", "").startswith("_apply_gdnmeta"):
        T._apply_gdnmeta()
    ours = MH.compute_common_gdn_attn_metadata
    ref = GS.compute_common_gdn_attn_metadata
    rnd = random.Random(5)
    bad = n = 0
    prev = None
    for it in range(400):
        K = rnd.choice([1, 2, 3, 4])
        reqs = rnd.choice([1, 2, 3, 4, 8, 16, 32])
        pure = rnd.random() < 0.7
        if prev is not None and rnd.random() < 0.5:
            K, reqs, pure, drafts, qlens = prev
        else:
            drafts, qlens = [], []
            for r in range(reqs):
                if pure or rnd.random() < 0.6:
                    d = rnd.choice([K, K, K, rnd.randint(0, K)])
                    drafts.append(d); qlens.append(d + 1)
                else:
                    drafts.append(-1); qlens.append(rnd.choice([0, 1, 1, rnd.randint(2, 300)]))
        prev = (K, reqs, pure, drafts, qlens)
        nd = torch.tensor(drafts, dtype=torch.int32)
        qsl_cpu = torch.zeros(reqs + 1, dtype=torch.int32)
        qsl_cpu[1:] = torch.cumsum(torch.tensor(qlens, dtype=torch.int32), 0)
        qsl = qsl_cpu.to(dev)
        for legacy in (False, True):
            kw = dict(num_decode_draft_tokens_cpu=nd, query_start_loc=qsl, query_start_loc_cpu=qsl_cpu,
                      num_spec_state_tokens=K, legacy_mixed_decode_routing=legacy)
            a, b = ref(**kw), ours(**kw)
            n += 1
            if (a is None) != (b is None):
                bad += 1; print("  gdnmeta: None mismatch", drafts, qlens); continue
            if a is None:
                continue
            for f in a.__dataclass_fields__:
                x, y = getattr(a, f), getattr(b, f)
                if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                    ok = (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype == y.dtype
                          and x.shape == y.shape and x.device == y.device and torch.equal(x, y))
                else:
                    ok = x == y
                if not ok:
                    bad += 1
                    print(f"  gdnmeta mismatch field {f}: {x} vs {y} (drafts {drafts}, qlens {qlens}, legacy {legacy})")
                    break
    print(f"gdnmeta: {n - bad}/{n} identical CommonGDNSpecMetadata")
    return bad


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    print("GPU", torch.cuda.get_device_name(), torch.cuda.current_device())
    if what == "check":
        b = check_argmax() + check_gverify() + check_mamba() + check_accept() + check_gdnmeta()
        print("ALL OK" if not b else f"FAILURES: {b}")
    else:
        bench()
