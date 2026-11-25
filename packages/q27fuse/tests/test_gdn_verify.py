"""test_gdn_verify.py check|negative|decode|bench - csrc/gdn_verify.cu (q27fuse.gdnv) against 1Cat's own ops at the
per-rank (TP4) Qwen3.8-27B GDN shape (4 key heads, 12 value heads, 128, conv width 4, FP32 state).

check [rounds] [seed]: multi-round oracle. A population of requests runs a random schedule of
  - verify rounds (pure batches; T = K+1 with K from the product mapping 4@1,3@2,1@16, so T changes with the batch:
    2..32 rows take VERIFY with the projection's a/b, > 32 rows VERIFY_BA with the b/a projection in the kernel),
  - draft-free decode rounds (gdn_decode.cu, or with Q27_GDNV_DECODE=gdnv gdn_verify DECODE; the commit folded in),
  - mixed rounds (plain decode rows + speculative rows: 1Cat's route, whose commit / verify consumers are ours: COMMIT
    and VONLY),
  with random accepted counts 1..T, requests joining and leaving, and a Mamba block of 8 tokens so prefix-cache
  checkpoints land often (also into the running slot). Bookkeeping for both sides is 1Cat's (SpecReplayState.prepare =
  its _prepare_replay_rows_kernel, record_accepted); align-mode slot moves (precopy / postprocess conv shift) are
  emulated in torch, identically for both.
  Oracle (1Cat): conv_gate_zero + torch.ops._C.sm70_gdn_verify_out (verify_bound, its own replay buffer) + FLA gated
  RMSNorm (verify rounds); commit_replayed + causal_conv1d_update + gdn_decode_mixed_qkv_global_state_sm70 + norm
  (decode rounds); 1Cat's spec conv + verify_bound / commit_replayed (mixed rounds).
  Compared after every round: outputs (valid rows), every SSM slot (relative error), conv windows (bitwise).
negative [seed]: the same schedule with a deliberately wrong conv offset (accepted count + 1) and with the commit
  skipped once (rn = 0 on our side): both must fail loudly.
decode [C list]: with nothing to replay, gdn_decode.cu against gdn_core.cu (bitwise expected, every config) and
  DECODE against gdn_core.cu (fp32 rounding level: different reduction order).
ckpt: [ckpt_fix] minimal reproduction of 1Cat's checkpoint-into-the-running-slot case (a verify round whose last token
  completes a Mamba block) against a float64 reference that applies every accepted token once, with 1Cat's ops and
  ours, without and with the bookkeeping fix (Q27_GDNV_CKPT_FIX semantics; the 1Cat side gets the same fix as row-table
  edits + a copy of the committed state). check_fix [rounds] [seeds]: the multi-round oracle with the fix on both sides.
bench [list of BxT] [cfg list]: us per layer, CUDA graph over NL layers' distinct states (cold): 1Cat's chain
  (conv_gate_zero + gdn_verify + layer_norm_fwd; > 32 rows + cuBLAS b/a + split) vs gdn_verify; decode (Bx1): 1Cat's
  commit + gdn_core.cu (today's draft-free path) vs DECODE."""
import os
import sys
import types
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE]
os.environ.setdefault("Q27_FUSE_BUILD", os.path.expanduser("~/.cache/q27fuse/build_gdnv_test"))
from q27fuse import gdnv, ops  # noqa: E402
import vllm  # noqa: E402,F401
from vllm.model_executor.layers.fla.ops.sm70 import gdn_verify as GV  # noqa: E402
from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import conv_gate_zero  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.spec_replay import SpecReplayState  # noqa: E402
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn  # noqa: E402
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.sm70.fused_fwd import gdn_decode_mixed_qkv_global_state_sm70 as cat_gdn  # noqa

dev = "cuda"
HV, HK, D, DIM, KH = 12, 4, 128, 2560, 5120
ZW = HV * D
CONTRACT = GdnHeadContract(16, 48, 128, 128, 4)
glue = ops.gdn_ext()
E = gdnv.ext()
REC = int(E.REC)
# [merge] GDNV_INPROJ=1: verify rounds take qkv | z | b | a from the in-projection routes of the merged tree (real QPN8
# codes): 2..16 rows 1Cat's fused op (apply_gdn_ba_verify), 17..32 rows smallrows' fp8_gdn_in.cu (gdn_in_project's
# separate contiguous outputs), > 32 rows the FP8 table of in_proj_qkvz (qkvz [n, 4096], z a strided view, b/a in the
# kernel). GDNV_TAILROWS=1: SpecReplayState.prepare is track tail's "rows" piece (installed before any ckpt fix).
INPROJ = os.environ.get("GDNV_INPROJ", "0") == "1"
if INPROJ:
    from q27fuse import smallrows
    ops.FP8_CFG = smallrows.target_table()
if os.environ.get("GDNV_TAILROWS", "0") == "1":
    from q27fuse import tail
    tail.ACTIVE.add("rows")
    tail.INSTALLED.add("rows")
    tail._apply_rows()
    print("[merge] SpecReplayState.prepare = tail's rows piece", flush=True)
INPROJ_HITS = {"1cat": 0, "smallrows": 0, "fp8 table": 0}


def kmap(B):            # product mapping 4@1,3@2,1@16
    return 4 if B == 1 else (3 if B < 16 else 1)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / max(b.norm().item(), 1e-30)).item()


def make_layer(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g)  # noqa: E731
    L = types.SimpleNamespace()
    L.conv1d = types.SimpleNamespace(weight=(r(DIM, 1, 4) * 0.5).half())
    L.A_log = (torch.rand(HV, device=dev, generator=g) * 2 - 1).float()
    L.dt_bias = (r(HV) * 0.3).half()
    L.norm = types.SimpleNamespace(weight=(1 + 0.1 * r(D)).half(), eps=1e-6)
    L.in_proj_ba = types.SimpleNamespace(weight=(r(2 * HV, KH) * 0.02).half())
    L.prefix = "test.layer"
    return L


class World:
    """requests, slots, both implementations' states, 1Cat's bookkeeping"""

    def __init__(s, seed, kmax=4, max_reqs=None, slots=None, bs=8, ncols=64, neg=None):
        # GDNV_POP: population size (default 24; 64 reaches gdn_decode.cu's > 32-row configuration with replays and
        # VERIFY_BA / VONLY / COMMIT at 33..64 requests)
        max_reqs = max_reqs or int(os.environ.get("GDNV_POP", "24"))
        slots = slots or max(160, 6 * max_reqs)
        s.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        s.K, s.stride, s.max_reqs, s.slots, s.bs, s.ncols, s.neg = kmax, kmax + 1, max_reqs, slots, bs, ncols, neg
        s.layer = make_layer(seed)
        if INPROJ:                                    # [merge] QPN8 codes of a qkvz projection [5120 -> 4096]
            g = torch.Generator(device=dev).manual_seed(seed + 77)
            w8 = (torch.randn(DIM + ZW, KH, device=dev, generator=g) * 0.6).to(torch.float8_e4m3fn)
            sc8 = (torch.rand(DIM + ZW, 1, device=dev, generator=g) * 0.04 + 0.02).float()
            s.codes, s.scales = torch.ops._C.fp8_qpn8_prepare_sm70(w8, sc8)[:2]
        s.layer._replay_stride = s.stride
        W = 3 + kmax
        # page-strided SSM state like the hybrid cache (stride(0) larger than the state)
        base = torch.randn(slots, HV * D * D + 256, device=dev) * 0.1
        s.ssm_cat = base[:, :HV * D * D].view(slots, HV, D, D)
        s.ssm_our = base.clone()[:, :HV * D * D].view(slots, HV, D, D)
        cv = (torch.randn(slots, W, DIM, device=dev) * 0.5).half()
        s.conv_cat, s.conv_our = cv, cv.clone()
        s.st = SpecReplayState(max_reqs, max_reqs, 8, torch.device(dev))
        s.rbuf_cat = s.st.layer_buffer("test.layer", HK, HV, dev)
        s.rbuf_our = torch.zeros(2 * max_reqs * s.stride * REC, dtype=torch.uint8, device=dev)
        s.free_slots = list(range(slots))
        s.rng.shuffle(s.free_slots)
        s.reqs = {}                                   # req index -> dict
        s.num_computed = torch.zeros(max_reqs, dtype=torch.int32, device=dev)
        s.acc_gpu = torch.ones(max_reqs, dtype=torch.int32, device=dev)   # num_accepted_tokens_gpu (conv offset + 1)
        s.stats = dict(rounds=0, verify=0, verify_ba=0, decode=0, mixed=0, ckpt=0, ckpt_running=0, replay_tokens=0,
                       joins=0, leaves=0, max_rows_verify=0, max_rows_verify_ba=0, max_rows_decode=0, max_rows_mixed=0,
                       decode_replay_gt32=0, trigger=0, commit_ckpt=0)
        s.fix = False                                 # [ckpt_fix] fixed bookkeeping on both sides
        s.force = []                                  # [ckpt_fix] forced accepted counts (ckpt reproduction)
        s.worst = dict(state=0.0, out=0.0, conv=0)

    # --- request population
    def join(s):
        free = [r for r in range(s.max_reqs) if r not in s.reqs]
        if not free or len(s.free_slots) < 12:
            return
        r = int(s.rng.choice(free))
        comp = int(s.rng.integers(s.bs, 4 * s.bs))
        col = (comp + s.bs - 1) // s.bs - 1
        blocks = [s.free_slots.pop() for _ in range(col + 1)]
        # a fresh request's running state + conv window (same values in both implementations)
        sl = blocks[col]
        st0 = torch.randn(HV, D, D, device=dev) * 0.1
        s.ssm_cat[sl] = st0; s.ssm_our[sl] = st0
        c0 = (torch.randn(s.conv_cat.shape[1], DIM, device=dev) * 0.5).half()
        s.conv_cat[sl] = c0; s.conv_our[sl] = c0
        s.reqs[r] = dict(blocks=blocks, computed=comp, col=col)
        s.num_computed[r] = comp
        s.acc_gpu[r] = 1
        s.st.reset_request(r)
        s.stats["joins"] += 1

    def leave(s):
        if len(s.reqs) <= 1:
            return
        r = int(s.rng.choice(list(s.reqs)))
        s.free_slots.extend(x for x in s.reqs.pop(r)["blocks"] if x >= 0)
        s.stats["leaves"] += 1

    def slot(s, r, col):
        q = s.reqs[r]
        while len(q["blocks"]) <= col:
            q["blocks"].append(s.free_slots.pop())
        return q["blocks"][col]

    # --- align-mode bookkeeping (emulated identically for both sides)
    def precopy(s, r, q):
        R = s.reqs[r]
        old, comp = R["col"], R["computed"]
        new = (comp + q + s.bs - 1) // s.bs - 1
        tb = max(int(s.acc_gpu[r]) - 1, 0)
        single = q == 1 and tb > 0
        if old != new or single:
            src, dst = s.slot(r, old), s.slot(r, new)
            Wr = s.conv_cat.shape[1]
            for conv in (s.conv_cat, s.conv_our):
                rows = conv[src, tb:Wr].clone()
                conv[dst, :Wr - tb] = rows
            if src != dst:
                s.ssm_cat[dst] = s.ssm_cat[src]
                s.ssm_our[dst] = s.ssm_our[src]
            s.acc_gpu[r] = 1
        R["col"] = new
        for c in range(max(0, new - 2)):                   # blocks no longer reachable: back to the pool
            if R["blocks"][c] >= 0:
                s.free_slots.insert(0, R["blocks"][c])
                R["blocks"][c] = -1

    def postprocess(s, r, acc):
        R = s.reqs[r]
        newc = R["computed"]
        running = newc - acc + 1
        aligned = (newc // s.bs) * s.bs
        if aligned >= running:
            tb = aligned - running
            dst = aligned // s.bs - 1
            src = R["col"]
            if src == dst:
                s.acc_gpu[r] = 1
            if not (src == dst and tb == 0):
                Wr = s.conv_cat.shape[1]
                a, b = s.slot(r, src), s.slot(r, dst)
                for conv in (s.conv_cat, s.conv_our):
                    rows = conv[a, tb:Wr].clone()
                    conv[b, :Wr - tb] = rows

    def prepare(s, batch, drafts):
        """1Cat's SpecReplayState.prepare for this batch (batch = request indices in row order)"""
        nb = len(batch)
        q = [d + 1 for d in drafts]
        qsl = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
        idx_mapping = torch.tensor(batch, dtype=torch.int32, device=dev)
        bt = torch.full((s.max_reqs, s.ncols), -1, dtype=torch.int32, device=dev)
        for i, r in enumerate(batch):
            bl = s.reqs[r]["blocks"]
            bt[i, :len(bl)] = torch.tensor(bl, dtype=torch.int32)
        s._bt = bt                                       # keep alive
        ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=dev)
        s.st.prepare(idx_mapping, torch.tensor(qsl, device=dev), s.num_computed, np.array(drafts),
                     np.zeros(nb, dtype=bool), nb, ptrs, bt.stride(0), s.bs)
        return idx_mapping, qsl

    def sample(s, batch, kinds, T):
        accs = []
        for r, k in zip(batch, kinds):
            a = (s.force.pop(0) if s.force else int(s.rng.integers(1, T + 1))) if k == "v" else 1
            accs.append(a)
            s.acc_gpu[r] = a
        idx_mapping = torch.tensor(batch, dtype=torch.int32, device=dev)
        s.st.record_accepted(idx_mapping, s.acc_gpu)
        for r, a in zip(batch, accs):
            s.reqs[r]["computed"] += a
            s.num_computed[r] = s.reqs[r]["computed"]
            s.postprocess(r, a)
        return accs

    def cat_commit_ckpt(s, idx, ccs):
        """[ckpt_fix] the fixed bookkeeping on 1Cat's ops: after its commit the running slot holds the committed state
        (no checkpoint into it any more), copied to the commit checkpoint slot"""
        if not s.fix:
            return
        for r_, c_ in zip(idx.tolist(), ccs[:idx.numel()].tolist()):
            if c_ >= 0 and c_ != r_:
                s.ssm_cat[c_] = s.ssm_cat[r_]

    def ccs(s, kind, n):
        if not s.fix:
            return None
        return (s.st.q27_spec_ccs if kind == "spec" else s.st.q27_plain_ccs)[0]

    # --- one round
    def round(s, kind):
        batch = list(s.reqs)
        s.rng.shuffle(batch)
        B = len(batch)
        if kind == "mixed" and B < 2:
            kind = "verify"
        if kind == "verify":
            T = kmap(B) + 1
            kinds = ["v"] * B
        elif kind == "decode":
            T = 1
            kinds = ["d"] * B
        else:
            nd = int(s.rng.integers(1, B))
            kinds = ["d"] * nd + ["v"] * (B - nd)          # decodes first, as in a vLLM batch
            T = kmap(B - nd) + 1
        q = [T if k == "v" else 1 for k in kinds]
        s.stats["trigger"] += sum(int(k == "v" and (s.reqs[r]["computed"] + T) % s.bs == 0)
                                  for r, k in zip(batch, kinds))       # last token completes a block
        for r, qq in zip(batch, q):
            s.precopy(r, qq)
        drafts = [qq - 1 for qq in q]
        idx_mapping, qsl = s.prepare(batch, drafts)
        rn_host = s.st.spec_replay[:B].tolist() if kind == "verify" else None
        n = int(qsl[-1])
        running = torch.tensor([s.slot(r, s.reqs[r]["col"]) for r in batch], dtype=torch.int32, device=dev)
        if s.fix:
            s.stats["commit_ckpt"] += int((s.st.q27_spec_ccs[0] >= 0).sum()) + int((s.st.q27_plain_ccs[0] >= 0).sum())
        if kind == "verify":
            out = s.verify_round(batch, running, n, B, T)
            s.stats["verify_ba" if n > 32 else "verify"] += 1
            s.stats["replay_tokens"] += sum(rn_host)
            ckt = s.st.spec_ckpt_token[:B].tolist()
            cks = s.st.spec_ckpt_slot[0][:B].tolist()
            for i in range(B):
                if ckt[i] >= 0:
                    s.stats["ckpt"] += 1
                    s.stats["ckpt_running"] += int(cks[i] == int(running[i]))
        elif kind == "decode":
            s.stats["decode_replay_gt32"] += int(n > 32 and int(s.st.plain_replay[:n].max()) > 0)
            out = s.decode_round(batch, running, n)
            s.stats["decode"] += 1
        else:
            out = s.mixed_round(batch, kinds, running, n, T)
            s.stats["mixed"] += 1
        torch.cuda.synchronize()
        mk = "max_rows_" + ("verify_ba" if kind == "verify" and n > 32 else kind)
        s.stats[mk] = max(s.stats[mk], n)
        s.sample(batch, kinds, T)
        s.stats["rounds"] += 1
        return kind, B, T, n, out

    def inproj(s, x, n):
        """[merge] the merged tree's in-projection route for n rows (see GDNV_INPROJ)"""
        wba = s.layer.in_proj_ba.weight
        if n > gdnv.FUSED_MAX:                              # gdnv: layer.in_proj_qkvz(x) (FP8 table) + VERIFY_BA
            qkvz = x.new_empty(n, DIM + ZW)
            ops.fp8_linear_impl(qkvz, x, s.codes, s.scales, 16, 2, False)
            INPROJ_HITS["fp8 table"] += 1
            return qkvz, qkvz[:, DIM:], None, None
        qkv, z, b, a = x.new_empty(n, DIM), x.new_empty(n, ZW), x.new_empty(n, HV), x.new_empty(n, HV)
        if n >= smallrows.GDNIN_MIN:                        # smallrows' gdn_in_project (17..64)
            smallrows.gdn_in_ext().gdn_in8(x, s.codes, s.scales, wba, qkv, z, b, a, smallrows.gdn_in_cfg(n), True)
            INPROJ_HITS["smallrows"] += 1
        else:                                               # 1Cat's apply_gdn_ba_verify (2..32; masked < 8 rows)
            op = (torch.ops._C.fp8_qpn8_dispatch_ba_masked_sm70_out if n < 8
                  else torch.ops._C.fp8_qpn8_dispatch_ba_split_sm70_out)
            op(qkv, z, b, a, x.new_empty(n, DIM + ZW), x.new_empty(n, 2 * HV), 0, x, s.codes, s.scales, wba)
            INPROJ_HITS["1cat"] += 1
        return qkv, z, b, a

    def verify_round(s, batch, running, n, B, T):
        L, st = s.layer, s.st
        qkv = (torch.randn(n, DIM, device=dev) * 0.5).half()
        z = (torch.randn(n, ZW, device=dev) * 0.5).half()
        x = (torch.randn(n, KH, device=dev) * 0.3).half()
        if INPROJ:                                          # [merge]
            qkv, z, b, a = s.inproj(x, n)
        if n > 32 and not (INPROJ and n <= gdnv.FUSED_MAX):  # b/a from the hidden states (ours: in the kernel)
            b = torch.empty(n, HV, device=dev, dtype=torch.half); a = torch.empty_like(b)
            junk = torch.empty(n, HV, D, device=dev, dtype=torch.half)
            glue.gdn_in(x, L.in_proj_ba.weight, b, a, junk)
        elif not INPROJ:
            a = (torch.randn(n, HV, device=dev) * 0.5).half()
            b = (torch.randn(n, HV, device=dev) * 0.5).half()
        cu = torch.arange(0, n + 1, T, dtype=torch.int32, device=dev)
        idx2 = running.view(B, 1).contiguous()
        sel = s.acc_gpu[torch.tensor(batch, device=dev).long()].contiguous()
        cw = L.conv1d.weight.view(DIM, 4)
        # 1Cat
        qc = qkv[:, :DIM].clone()                          # [merge] (qkvz [n, 4096] under GDNV_INPROJ above 32 rows)
        core = torch.empty(n, HV, D, device=dev, dtype=torch.half)
        conv_gate_zero(qc, s.conv_cat.transpose(-1, -2), cw, idx2[:, 0], sel, cu, L.A_log, a, b, L.dt_bias, core,
                       compute_gating=False)
        r = GV.verify_bound(CONTRACT, L.A_log, L.dt_bias, B, qc, a, b, s.ssm_cat, core, n, True, cu, idx2, sel,
                            replay=(s.rbuf_cat, st.spec_replay, st.spec_map, st.spec_ckpt_token, st.spec_ckpt_slot[0],
                                    s.stride))
        assert r is not None
        s.cat_commit_ckpt(idx2[:, 0], st.q27_spec_ccs[0] if s.fix else None)
        y_cat = rmsnorm_fn(core.view(n * HV, D), L.norm.weight, None, z=z.contiguous().view(n * HV, D), eps=1e-6, group_size=None,
                           norm_before_gate=True, activation="swish").view(n, ZW)
        # ours
        y = torch.empty(n, ZW, device=dev, dtype=torch.half)
        sc = torch.empty(n, HV, D, device=dev, dtype=torch.half)
        sel_o = sel
        rn = st.spec_replay
        if s.neg == "offset":
            sel_o = torch.clamp(sel + 1, max=s.stride)
        if s.neg == "commit" and s.stats["verify"] + s.stats["verify_ba"] == 3:
            rn = torch.zeros_like(rn)
        cap = T if s.rng.random() < 0.5 else max(T, s.stride)     # kernel capacity: exact, or the server's stride
        if n > 32 and T in gdnv.BA_TM and not (INPROJ and n <= gdnv.FUSED_MAX):  # (B >= 9: cfg 3 or 4) [merge]
            gdnv.run(gdnv.VERIFY_BA, gdnv.BA_TM[cap], gdnv.cfg_for(gdnv.VERIFY_BA_CFG, B), L, ssm=s.ssm_our,
                     idx=idx2[:, 0],
                     B=B, T=T, n=n, qkv=qkv, z=z, x=x, conv=s.conv_our, cu=cu, acc=sel_o, rn=rn, rmap=st.spec_map,
                     ckt=st.spec_ckpt_token, cks=st.spec_ckpt_slot[0], core=sc, y=y, rbuf=s.rbuf_our,
                     ccs=s.ccs("spec", B))
        else:
            gdnv.run(gdnv.VERIFY, gdnv.VERIFY_TM[cap], gdnv.cfg_for(gdnv.VERIFY_CFG, B), L, ssm=s.ssm_our,
                     idx=idx2[:, 0], B=B, T=T, n=n, qkv=qkv, z=z, a=a, b=b, conv=s.conv_our, cu=cu, acc=sel_o, rn=rn,
                     rmap=st.spec_map, ckt=st.spec_ckpt_token, cks=st.spec_ckpt_slot[0], core=sc, y=y, rbuf=s.rbuf_our,
                     ccs=s.ccs("spec", B))
        return y_cat, y, None

    def decode_round(s, batch, running, n):
        L, st = s.layer, s.st
        B = n
        qkvz = (torch.randn(n, DIM + ZW, device=dev) * 0.5).half()
        x = (torch.randn(n, KH, device=dev) * 0.3).half()
        idx = running.contiguous()
        # 1Cat: commit, conv, state, norm
        GV.commit_replayed(CONTRACT, L.A_log, L.dt_bias, s.ssm_cat, st.empty_starts[:B + 1], idx, s.rbuf_cat,
                           st.plain_replay, st.plain_map, s.stride)
        s.cat_commit_ckpt(idx, st.q27_plain_ccs[0] if s.fix else None)
        b = torch.empty(n, HV, device=dev, dtype=torch.half); a = torch.empty_like(b)
        core = torch.empty(n, HV, D, device=dev, dtype=torch.half)
        glue.gdn_in(x, L.in_proj_ba.weight, b, a, core)
        mixed = qkvz[:, :DIM].clone()
        causal_conv1d_update(mixed, s.conv_cat.transpose(-1, -2), L.conv1d.weight.view(DIM, 4), None, "silu",
                             conv_state_indices=idx, validate_data=True)
        cat_gdn(mixed, a, b, L.A_log, L.dt_bias, s.ssm_cat, idx, core, None, True)
        y_cat = torch.empty(n, ZW, device=dev, dtype=torch.half)
        glue.gdn_gated_norm(core, qkvz, L.norm.weight, y_cat, 1e-6, DIM)
        # ours
        y = torch.empty(n, ZW, device=dev, dtype=torch.half)
        sc = torch.empty(n, HV, D, device=dev, dtype=torch.half)
        rn = st.plain_replay
        if s.neg == "commit" and s.stats["decode"] == 2:
            rn = torch.zeros_like(rn)
        if gdnv.DECODE_KERNEL == "core":                  # the server's draft-free route: gdn_decode.cu
            gdnv.decode_core(L, x, qkvz, s.conv_our, s.ssm_our, idx, B, sc, y, rn, st.plain_map, rbuf=s.rbuf_our,
                             stride=s.stride, ccs=s.ccs("plain", B))
        else:
            gdnv.run(gdnv.DECODE, 1, gdnv.cfg_for(gdnv.DECODE_CFG, n), L, ssm=s.ssm_our, idx=idx, B=B, T=1, n=n,
                     qkv=qkvz, z=qkvz[:, DIM:], x=x, conv=s.conv_our, rn=rn, rmap=st.plain_map, core=sc, y=y,
                     rbuf=s.rbuf_our, ccs=s.ccs("plain", B))
        return y_cat, y, None

    def mixed_round(s, batch, kinds, running, n, T):
        """plain decode rows first, then speculative rows (1Cat's _forward_core route): commit for the plain rows,
        spec conv + verification for the spec rows, decode conv + state for the plain rows; raw core outputs"""
        L, st = s.layer, s.st
        nd = kinds.count("d")
        Bs = len(batch) - nd
        ns = Bs * T
        qkv_d = (torch.randn(nd, DIM, device=dev) * 0.5).half()
        qkv_s = (torch.randn(ns, DIM, device=dev) * 0.5).half()
        ab_d = (torch.randn(2, nd, HV, device=dev) * 0.5).half()
        ab_s = (torch.randn(2, ns, HV, device=dev) * 0.5).half()
        idx_d = running[:nd].contiguous()
        idx_s2 = running[nd:].view(Bs, 1).contiguous()
        cu_s = torch.arange(0, ns + 1, T, dtype=torch.int32, device=dev)
        sel_s = s.acc_gpu[torch.tensor(batch[nd:], device=dev).long()].contiguous()
        cw = L.conv1d.weight.view(DIM, 4)
        outs = []
        for side in ("cat", "our"):
            ssm = s.ssm_cat if side == "cat" else s.ssm_our
            conv = s.conv_cat if side == "cat" else s.conv_our
            if side == "cat":
                GV.commit_replayed(CONTRACT, L.A_log, L.dt_bias, ssm, st.empty_starts[:nd + 1], idx_d, s.rbuf_cat,
                                   st.plain_replay, st.plain_map, s.stride)
                s.cat_commit_ckpt(idx_d, st.q27_plain_ccs[0] if s.fix else None)
            else:
                rn = st.plain_replay
                if s.neg == "commit" and s.stats["mixed"] == 1:
                    rn = torch.zeros_like(rn)
                gdnv.run(gdnv.COMMIT, 1, 4, L, ssm=ssm, idx=idx_d, B=nd, rn=rn, rmap=st.plain_map, rbuf=s.rbuf_our,
                         ccs=s.ccs("plain", nd))
            # spec rows: 1Cat's spec conv (convolve_decode with the replay stride as max_query_len)
            qs = qkv_s.clone()
            causal_conv1d_update(qs, conv.transpose(-1, -2), cw, None, "silu", conv_state_indices=idx_s2[:, 0],
                                 num_accepted_tokens=sel_s, query_start_loc=cu_s, max_query_len=s.stride,
                                 validate_data=False)
            core_s = torch.zeros(ns, HV, D, device=dev, dtype=torch.half)
            if side == "cat":
                r = GV.verify_bound(CONTRACT, L.A_log, L.dt_bias, Bs, qs, ab_s[0], ab_s[1], ssm, core_s, ns, True,
                                    cu_s, idx_s2, sel_s, replay=(s.rbuf_cat, st.spec_replay, st.spec_map,
                                                                 st.spec_ckpt_token, st.spec_ckpt_slot[0], s.stride))
                s.cat_commit_ckpt(idx_s2[:, 0], st.q27_spec_ccs[0] if s.fix else None)
            else:
                r = gdnv.verify_update(L, L.A_log, L.dt_bias, Bs, qs, ab_s[0], ab_s[1], ssm, core_s, ns, True, cu_s,
                                       idx_s2, sel_s, replay=(s.rbuf_our, st.spec_replay, st.spec_map,
                                                              st.spec_ckpt_token, st.spec_ckpt_slot[0], s.stride))
            assert r is not None
            # plain rows: decode conv + 1Cat's state kernel (both sides: 1Cat's route)
            qd = qkv_d.clone()
            causal_conv1d_update(qd, conv.transpose(-1, -2), cw, None, "silu", conv_state_indices=idx_d,
                                 validate_data=True)
            core_d = torch.empty(nd, HV, D, device=dev, dtype=torch.half)
            cat_gdn(qd, ab_d[0], ab_d[1], L.A_log, L.dt_bias, ssm, idx_d, core_d, None, True)
            outs.append(torch.cat([core_d.view(nd, ZW), core_s.view(ns, ZW)]))
        return outs[0], outs[1], None


def compare(w, tag, y_cat, y):
    eo = rel(y, y_cat)
    mx = (y.float() - y_cat.float()).abs().max().item()
    es = 0.0
    for r, R in w.reqs.items():
        for sl in R["blocks"]:
            if sl < 0:
                continue
            ref = w.ssm_cat[sl]
            if ref.abs().sum() > 0:
                es = max(es, rel(w.ssm_our[sl], ref))
    cd = int((w.conv_our != w.conv_cat).sum())
    w.worst["state"] = max(w.worst["state"], es)
    w.worst["out"] = max(w.worst["out"], eo)
    w.worst["conv"] = max(w.worst["conv"], cd)
    return eo, mx, es, cd


SCHED = ["verify"] * 6 + ["decode"] * 2 + ["mixed"] * 2


def check(rounds=60, seed=1, neg=None, verbose=True, fix=False):
    w = World(seed, neg=neg)
    if fix:                                # [ckpt_fix]
        from vllm.model_executor.layers.mamba.gdn.spec_replay import set_spec_replay_state
        w.fix = True
        set_spec_replay_state(w.st)        # (VONLY through gdnv.verify_update looks the tables up there)
    for _ in range(4):
        w.join()
    fails = 0
    for i in range(rounds):
        if w.rng.random() < 0.25:
            w.join()
        if w.rng.random() < 0.25:
            w.leave()
        if i % 23 == 11:                   # shrink back to a few requests now and then (K = 4 / 3 rounds)
            while len(w.reqs) > int(w.rng.integers(1, 4)):
                w.leave()
        if i % 17 == 9:                    # grow the batch past the K thresholds now and then
            g0, g1 = (int(v) for v in os.environ.get("GDNV_GROW", "4,12").split(","))   # GDNV_GROW: burst size range
            for _ in range(int(w.rng.integers(g0, g1))):
                w.join()
        kind = SCHED[int(w.rng.integers(len(SCHED)))]
        kind, B, T, n, (y_cat, y, _) = w.round(kind)
        eo, mx, es, cd = compare(w, kind, y_cat, y)
        bad = es > 1e-5 or eo > 4e-3 or cd > 0
        fails += bad
        if verbose or bad:
            print(f"round {i:3d} {kind:6s} B={B:2d} T={T} rows={n:3d}: out rel {eo:.1e} (max |d| {mx:.1e}), "
                  f"state rel max {es:.1e}, conv mismatches {cd}{'  <-- FAIL' if bad else ''}", flush=True)
    print(f"summary seed {seed} neg {neg}: {w.stats}; worst out {w.worst['out']:.2e}, state {w.worst['state']:.2e}, "
          f"conv {w.worst['conv']}; failing rounds {fails}")
    return fails


# --------------------------------------------------------------------------- [ckpt_fix] reproduction, float64 reference
def rec_apply(S, rbuf, side, stride, toks):
    """apply records (our format) of one request side to a float64 state S [HV][V][K]: the delta rule once per token"""
    for t in toks:
        o = (side * stride + t) * REC
        r = rbuf[o:o + REC].clone()
        k16 = r[0:1024].view(torch.float16).view(HK, D).double()
        v16 = r[1024:4096].view(torch.float16).view(HV, D).double()
        g = r[4096:4144].view(torch.float32).double()
        be = r[4144:4192].view(torch.float32).double()
        ki = r[4192:4208].view(torch.float32).double()
        for h in range(HV):
            k = k16[h // (HV // HK)] * ki[h // (HV // HK)]
            S[h] *= g[h]
            d = (v16[h] - S[h] @ k) * be[h]
            S[h] += torch.outer(d, k)


def ckpt_scenario(fix, c0, accs, seed=11):
    """one request (B = 1, K = 4: T = 5 rows per round), Mamba block 8, starting at c0 committed tokens, accepted
    counts accs; returns per-round rows: (c, trigger, a, ours vs ref, 1Cat vs ref, note) and block checkpoint checks"""
    w = World(seed)
    w.fix = fix
    r = 0
    comp = c0
    col = (comp + w.bs - 1) // w.bs - 1
    blocks = [w.free_slots.pop() for _ in range(col + 1)]
    sl = blocks[col]
    st0 = torch.randn(HV, D, D, device=dev) * 0.1
    w.ssm_cat[sl] = st0; w.ssm_our[sl] = st0
    c0w = (torch.randn(w.conv_cat.shape[1], DIM, device=dev) * 0.5).half()
    w.conv_cat[sl] = c0w; w.conv_our[sl] = c0w
    w.reqs[r] = dict(blocks=blocks, computed=comp, col=col)
    w.num_computed[r] = comp
    w.acc_gpu[r] = 1
    w.st.reset_request(r)
    ref = st0.double().clone()                  # committed state, every accepted token applied once
    at_boundary = {}                            # block index -> reference state at its end
    rows = []
    for a in accs:
        c = w.reqs[r]["computed"]
        T = kmap(1) + 1
        trig = (c + T) % w.bs == 0
        w.force = [a]
        w.round("verify")
        run_slot = w.slot(r, w.reqs[r]["col"])
        # after this round's kernel the running slot holds the commit of the previous round = ref (unless a
        # checkpoint went into it)
        e_our = rel(w.ssm_our[run_slot], ref); e_cat = rel(w.ssm_cat[run_slot], ref)
        side = int(w.st.spec_map[0])
        for t in range(a):                          # this round's accepted tokens, once
            rec_apply(ref, w.rbuf_our, side, w.stride, [t])
            if (c + t + 1) % w.bs == 0:
                at_boundary[(c + t + 1) // w.bs - 1] = ref.clone()
        rows.append((c, T, trig, a, e_our, e_cat))
    # prefix-cache checkpoints of the blocks whose end was committed (slot of block j vs the state at its end); the
    # last round's commit is not run yet, so a boundary reached by it is checked after one more round with a = 1
    w.force = [1]
    w.round("verify")
    run_slot = w.slot(r, w.reqs[r]["col"])
    final = (rel(w.ssm_our[run_slot], ref), rel(w.ssm_cat[run_slot], ref))
    ck = []
    for j, S in sorted(at_boundary.items()):
        b = w.reqs[r]["blocks"][j] if j < len(w.reqs[r]["blocks"]) else -1
        if b >= 0:
            ck.append((j, rel(w.ssm_our[b], S), rel(w.ssm_cat[b], S)))
    return rows, final, ck, w.stats


def ckpt_repro():
    """[ckpt_fix] 1Cat's checkpoint into the running slot, reproduced; float64 reference = every accepted token once"""
    from vllm.model_executor.layers.mamba.gdn.spec_replay import set_spec_replay_state
    gdnv.install_ckpt_fix()
    scen = {"A: trigger round fully accepted": (9, [2, 5, 3, 2]),
            "B: trigger round partly accepted": (9, [2, 2, 3, 1]),
            "C: no trigger (control)": (9, [3, 4, 1, 2])}
    for name, (c0, accs) in scen.items():
        for fix in (False, True):
            gdnv.CKPT_FIX = fix
            rows, final, ck, stats = ckpt_scenario(fix, c0, accs)
            print(f"{name}, fix {'ON ' if fix else 'off'} (block 8, T 5, start {c0} tokens):")
            for i, (c, T, trig, a, eo, ec) in enumerate(rows):
                print(f"  round {i + 1}: committed {c:2d}, rows {c}..{c + T - 1}{' TRIGGER (c+T = block end)' if trig else '':27s}"
                      f" accepted {a}; running slot after its commit vs float64 ref: ours {eo:.1e}, 1Cat {ec:.1e}")
            print(f"  after the last commit: ours {final[0]:.1e}, 1Cat {final[1]:.1e}; block checkpoints (slot vs ref at "
                  f"the block end): " + ", ".join(f"block {j}: ours {a_:.1e} 1Cat {b_:.1e}" for j, a_, b_ in ck)
                  + f"; commit checkpoints stored {stats['commit_ckpt']}")
    gdnv.CKPT_FIX = False


def negative(seed=3):
    for neg in ("offset", "commit"):
        f = check(30, seed, neg=neg, verbose=False)
        print(f"negative control '{neg}': {f} failing rounds -> {'DETECTED' if f else 'NOT DETECTED (bad test)'}")


def decode_check(CS):
    """DECODE with rn = 0 vs gdn_core.cu (same arithmetic: bitwise expected) and vs 1Cat's chain"""
    L = make_layer(5)
    L._replay_stride = 5
    core_ext = ops.gdn_core_ext()
    for C in CS:
        slots = C + 7
        x = (torch.randn(C, KH, device=dev) * 0.3).half()
        qkvz = (torch.randn(C, DIM + ZW, device=dev) * 0.5).half()
        cst = (torch.randn(slots, 7, DIM, device=dev) * 0.5).half()
        ssm = (torch.randn(slots, HV, D, D, device=dev) * 0.1).float()
        idx = torch.randperm(slots, device=dev)[:C].int()
        if C > 2:
            idx[1] = -1
        cw = L.conv1d.weight.view(DIM, 4)
        c1, s1 = cst.clone(), ssm.clone()
        y1 = torch.empty(C, ZW, device=dev, dtype=torch.half)
        core = torch.empty(C, HV, D, device=dev, dtype=torch.half)
        core_ext.gdn_core(x, qkvz, L.in_proj_ba.weight, cw, c1, L.A_log, L.dt_bias, s1, idx, C, L.norm.weight, 1e-6,
                          core, y1, ops.gdn_core_workspace(torch.device(dev), C), ops.gdn_core_cfg(C), 0)
        c2, s2 = cst.clone(), ssm.clone()
        y2 = torch.empty(C, ZW, device=dev, dtype=torch.half)
        rb = torch.zeros(2 * 64 * 5 * REC, dtype=torch.uint8, device=dev)
        rn = torch.zeros(C, dtype=torch.int32, device=dev)
        for cfg in range(8):                                # gdn_decode.cu with rn = 0: gdn_core.cu bit for bit
            c2.copy_(cst); s2.copy_(ssm)
            gdnv.decode_core(L, x, qkvz, c2, s2, idx, C, torch.empty(C, HV, D, device=dev, dtype=torch.half), y2, rn,
                             rn, rbuf=rb, stride=5, cfg=cfg)
            torch.cuda.synchronize()
            v = idx >= 0
            print(f"gdn_decode C={C:3d} cfg {cfg}: y {'bitwise' if torch.equal(y2[v], y1[v]) else 'DIFF'}, ssm "
                  f"{'bitwise' if torch.equal(s2, s1) else 'DIFF %.1e' % rel(s2, s1)}, conv "
                  f"{'bitwise' if torch.equal(c1, c2) else 'DIFF'}; pad rows zero {bool((y2[~v] == 0).all())}", flush=True)
        for cfg in (2, 3, 4):                               # DECODE configurations
            c2.copy_(cst); s2.copy_(ssm)
            gdnv.run(gdnv.DECODE, 1, cfg, L, ssm=s2, idx=idx, B=C, T=1, n=C, qkv=qkvz, z=qkvz[:, DIM:], x=x, conv=c2,
                     rn=rn, rmap=rn, core=torch.empty(C, HV, D, device=dev, dtype=torch.half), y=y2, rbuf=rb)
            torch.cuda.synchronize()
            v = idx >= 0
            # (the state dot products reduce over 4 lanes x 32 elements, gdn_core.cu over 32 lanes x 4: not bitwise)
            print(f"decode C={C:3d} cfg {gdnv.CFG_NAMES[cfg]:>4}: y vs gdn_core rel {rel(y2[v], y1[v]):.1e} "
                  f"(max |d| {(y2[v].float() - y1[v].float()).abs().max().item():.1e}), ssm rel {rel(s2, s1):.1e}, "
                  f"conv {'bitwise' if torch.equal(c1, c2) else 'DIFF'}; pad rows zero {bool((y2[~v] == 0).all())}",
                  flush=True)


# ---------------------------------------------------------------------------------------------------------- bench
NL, REPS = 12, 10


def graph_time(fn, nl=NL):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            fn(i % nl)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for r in range(REPS):
            for i in range(nl):
                fn(i)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / (REPS * nl))
    return sorted(ts)[3]


def bench(points, cfgs):
    """points: list of (B, T); T = 1 -> decode. Each of NL layers has its own states, conv windows and records; every
    request replays a full previous round (rn = T, or rn = K+1 = 5 for decode) so the commit is always non-trivial."""
    L = make_layer(7)
    print(f"{'BxT:rn':>7} {'rows':>4} | {'1Cat chain':>10} | " + " ".join(f"{gdnv.CFG_NAMES[c]:>7}" for c in cfgs) +
          " | best (cfg), floor 2 x B x 786 KB @ 826 GB/s, best/floor", flush=True)
    res = []
    for B, T, rn_ in points:
        n = B * T
        stride = 5
        L._replay_stride = stride
        slots = B
        lay = []
        for li in range(NL):
            lay.append(dict(ssm=(torch.randn(slots, HV, D, D, device=dev) * 0.1).float(),
                            conv=(torch.randn(slots, 3 + stride - 1, DIM, device=dev) * 0.5).half(),
                            rcat=torch.randn(2 * B * 8 * (HK * D + HV * D + 2 * HV), device=dev) * 0.01,
                            rour=torch.zeros(2 * B * stride * REC, dtype=torch.uint8, device=dev)))
        idx = torch.arange(B, dtype=torch.int32, device=dev)
        idx2 = idx.view(B, 1)
        x = (torch.randn(n, KH, device=dev) * 0.3).half()
        qkvz = (torch.randn(n, DIM + ZW, device=dev) * 0.5).half()
        qkv_src = qkvz[:, :DIM].contiguous()
        qkv = qkv_src.clone()
        z = qkvz[:, DIM:].contiguous()
        a = (torch.randn(n, HV, device=dev) * 0.5).half(); b = (torch.randn(n, HV, device=dev) * 0.5).half()
        cu = torch.arange(0, n + 1, T, dtype=torch.int32, device=dev)
        sel = torch.full((B,), 2, dtype=torch.int32, device=dev)
        rn = torch.full((B,), rn_, dtype=torch.int32, device=dev)
        rmap = (2 * torch.arange(B, device=dev) + 1).int()
        ckt = torch.full((B,), -1, dtype=torch.int32, device=dev); cks = ckt.clone()
        core = torch.empty(n, HV, D, device=dev, dtype=torch.half)
        y = torch.empty(n, ZW, device=dev, dtype=torch.half)
        cw = L.conv1d.weight.view(DIM, 4)
        ba_w = L.in_proj_ba.weight
        empty_starts = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        core_ext = ops.gdn_core_ext()

        if T > 1:
            def cat_chain(i):
                l = lay[i]
                if n > 32:                       # 1Cat > 32 rows: cuBLAS b/a + split kernel (modelled as copies)
                    ba = torch.nn.functional.linear(x, ba_w)
                    b.copy_(ba[:, :HV]); a.copy_(ba[:, HV:])
                qkv.copy_(qkv_src)
                conv_gate_zero(qkv, l["conv"].transpose(-1, -2), cw, idx, sel, cu, L.A_log, a, b, L.dt_bias, core,
                               compute_gating=False)
                torch.ops._C.sm70_gdn_verify_out(qkv, a, b, L.A_log, L.dt_bias, l["ssm"], core, cu, idx2, None, HK, HV,
                                                 D ** -0.5, 1, l["rcat"], rn, None, stride, rmap, ckt, cks)
                rmsnorm_fn(core.view(n * HV, D), L.norm.weight, None, z=z.view(n * HV, D), eps=1e-6, group_size=None,
                           norm_before_gate=True, activation="swish")
        else:
            def cat_chain(i):                    # today's draft-free path: 1Cat's commit + gdn_core.cu
                l = lay[i]
                torch.ops._C.sm70_gdn_verify_out(qkv[:0], a[:0], b[:0], L.A_log, L.dt_bias, l["ssm"], core[:0],
                                                 empty_starts, idx, None, HK, HV, D ** -0.5, 1, l["rcat"], rn, None,
                                                 stride, rmap, None, None)
                core_ext.gdn_core(x, qkvz, ba_w, cw, l["conv"], L.A_log, L.dt_bias, l["ssm"], idx, B, L.norm.weight,
                                  1e-6, core, y, ops.gdn_core_workspace(torch.device(dev), n), ops.gdn_core_cfg(n), 0)
        t_cat = graph_time(cat_chain)
        if T == 1:
            def core_only(i):
                l = lay[i]
                core_ext.gdn_core(x, qkvz, ba_w, cw, l["conv"], L.A_log, L.dt_bias, l["ssm"], idx, B, L.norm.weight,
                                  1e-6, core, y, ops.gdn_core_workspace(torch.device(dev), n), ops.gdn_core_cfg(n), 0)
            print(f"   ({B}x1: gdn_core.cu alone {graph_time(core_only):.2f} us)", flush=True)
            for dcfg in sorted({ops.gdn_core_cfg(n), 0, 6, 7}):
                def dec(i, dcfg=dcfg):
                    l = lay[i]
                    gdnv.decode_core(L, x, qkvz, l["conv"], l["ssm"], idx, B, core, y, rn, rmap, rbuf=l["rour"],
                                     stride=stride, cfg=dcfg)
                print(f"   ({B}x1:{rn_}: gdn_decode.cu cfg {dcfg}{' (gdn_core_cfg)' if dcfg == ops.gdn_core_cfg(n) else ''}"
                      f" {graph_time(dec):.2f} us)", flush=True)
        row = []
        capT = max(T, stride) if os.environ.get("BENCH_CAP") == "stride" else T   # kernel capacity (template TM)
        dbgv = int(os.environ.get("BENCH_DBG", "0"))                               # phase skipping (gdn_verify dbg)
        for cfg in cfgs:
            def ours(i, cfg=cfg):
                l = lay[i]
                if T == 1:
                    gdnv.run(gdnv.DECODE, 1, cfg, L, ssm=l["ssm"], idx=idx, B=B, T=1, n=n, qkv=qkvz, z=qkvz[:, DIM:],
                             x=x, conv=l["conv"], rn=rn, rmap=rmap, core=core, y=y, rbuf=l["rour"], stride=stride,
                             dbg=dbgv)
                elif n > 32 and T in gdnv.BA_TM:
                    gdnv.run(gdnv.VERIFY_BA, gdnv.BA_TM[capT], cfg, L, ssm=l["ssm"], idx=idx, B=B, T=T, n=n, qkv=qkvz,
                             z=qkvz[:, DIM:], x=x, conv=l["conv"], cu=cu, acc=sel, rn=rn, rmap=rmap, ckt=ckt, cks=cks,
                             core=core, y=y, rbuf=l["rour"], stride=stride, dbg=dbgv)
                else:
                    gdnv.run(gdnv.VERIFY, gdnv.VERIFY_TM[capT], cfg, L, ssm=l["ssm"], idx=idx, B=B, T=T, n=n, qkv=qkvz,
                             z=z, a=a, b=b, conv=l["conv"], cu=cu, acc=sel, rn=rn, rmap=rmap, ckt=ckt, cks=cks,
                             core=core, y=y, rbuf=l["rour"], stride=stride, dbg=dbgv)
            try:
                row.append(graph_time(ours))
            except RuntimeError as e:
                row.append(float("nan"))
                print("  ", cfg, e)
        floor = 2 * B * HV * D * D * 4 / 826e9 * 1e6
        best = min(range(len(row)), key=lambda i: row[i] if row[i] == row[i] else 1e9)
        print(f"{B:>2}x{T}:{rn_} {n:>4} | {t_cat:10.2f} | " + " ".join(f"{t:7.2f}" for t in row) +
              f" | {row[best]:.2f} ({gdnv.CFG_NAMES[cfgs[best]]}), floor {floor:.2f}, {row[best] / floor:.2f}x; "
              f"1Cat/ours {t_cat / row[best]:.2f}x", flush=True)
        res.append((B, T, t_cat, row))
    return res


def guards():
    """gdnv.apply() on the real 1Cat classes: every consumer of 1Cat's record format is replaced or refuses to run, and
    the replaced consumers run our kernels on our records (no server needed)"""
    import inspect
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as QG, spec_replay as SR
    gdnv.apply()
    ok = True
    A = QG.QwenGatedDeltaNetAttention
    ok &= A._commit_replayed_decodes is gdnv.commit_replayed_decodes
    ok &= A._spec_replay_args is gdnv.spec_replay_args
    print("class methods replaced:", ok)
    st = SpecReplayState(4, 4, 8, torch.device(dev))
    for name, fn in [("SpecReplayState.layer_buffer", lambda: st.layer_buffer("x", HK, HV, dev)),
                     ("gdn_verify.verify_bound(replay=...)", lambda: GV.verify_bound(CONTRACT, None, None, 1, None, None,
                                                                                    None, None, None, 1, True, None,
                                                                                    None, None, replay=(1,))),
                     ("gdn_verify.commit_replayed", lambda: GV.commit_replayed()),
                     ("qwen_gdn_linear_attn.commit_replayed", lambda: QG.commit_replayed())]:
        try:
            fn()
            print(f"{name}: did NOT raise"); ok = False
        except RuntimeError as e:
            print(f"{name}: raises ({str(e)[:60]}...)")
    # every remaining reference to 1Cat's replay entry points in the GDN layer module goes through the names above
    src = inspect.getsource(QG)
    n_commit = src.count("self._commit_replayed_decodes(")
    n_verify = src.count("self.verification_update(")
    n_args = src.count("self._spec_replay_args()")
    print(f"qwen_gdn_linear_attn.py call sites: _commit_replayed_decodes x{n_commit}, verification_update x{n_verify}, "
          f"_spec_replay_args x{n_args} (all bound to gdnv at runtime)")
    # our COMMIT through 1Cat's own method name, against the float64-free reference: commit k records by hand
    L = make_layer(9); L._replay_stride = 5; L.prefix = "guard.layer"; L.gdn_heads = None
    ssm = torch.randn(6, HV, D, D, device=dev) * 0.1
    ref = ssm.clone()
    rb = torch.zeros(2 * 4 * 5 * REC, dtype=torch.uint8, device=dev)
    gdnv._BUF[L.prefix] = rb
    from vllm.model_executor.layers.mamba.gdn.spec_replay import set_spec_replay_state
    set_spec_replay_state(st)
    md = types.SimpleNamespace(num_decodes=2, non_spec_state_indices_tensor=torch.tensor([3, 5], dtype=torch.int32,
                                                                                          device=dev))
    st.plain_replay[:2] = 0
    A._commit_replayed_decodes(L, ssm, md)            # nothing to replay -> untouched
    torch.cuda.synchronize()
    ok &= torch.equal(ssm, ref)
    print("COMMIT via 1Cat's method name with rn = 0 leaves the state untouched:", torch.equal(ssm, ref))
    print("GUARDS", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what == "check":
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        seeds = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1]
        tot = sum(check(rounds, sd) for sd in seeds)
        if INPROJ:
            print("[merge] in-projection routes used by verify rounds:", INPROJ_HITS)
        print("CHECK", "PASS" if tot == 0 else f"FAIL ({tot} rounds)")
    elif what == "ckpt":                                        # [ckpt_fix]
        ckpt_repro()
    elif what == "check_fix":                                   # [ckpt_fix]
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        seeds = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1]
        gdnv.install_ckpt_fix()
        gdnv.CKPT_FIX = True
        tot = sum(check(rounds, sd, verbose=False, fix=True) for sd in seeds)
        if INPROJ:
            print("[merge] in-projection routes used by verify rounds:", INPROJ_HITS)
        print("CHECK_FIX", "PASS" if tot == 0 else f"FAIL ({tot} rounds)")
    elif what == "negative":
        negative(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif what == "guards":
        guards()
    elif what == "decode":
        decode_check([int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2,5,16,33,48,64").split(",")])
    elif what == "bench":
        pts = []
        for p_ in (sys.argv[2] if len(sys.argv) > 2 else "1x5,2x4,4x4,8x4,16x2,32x2,48x1,64x1,48x1:2,64x1:2").split(","):
            bt, _, r_ = p_.partition(":")
            B_, T_ = (int(u) for u in bt.split("x"))
            pts.append((B_, T_, int(r_) if r_ else (min(T_, 3) if T_ > 1 else 0)))
        cfgs = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(len(gdnv.CFG_NAMES)))
        bench(pts, cfgs)
