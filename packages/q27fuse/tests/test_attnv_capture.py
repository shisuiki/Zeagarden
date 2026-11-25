"""test_attnv_capture.py - [head] Job 0 regression test: the attn_mq.cu workspace (attnv.py) must not come out of the
CUDA-graph memory pool.

Root cause of the NaN logprobs (notes/head.md, Job 0): 1Cat's grouped long-attention wrapper admits a call only under
a captured batch descriptor, so the warm-up runs never reached attnv.run_mq and its first call - which allocated the
persistent workspace (partials po / row stats pml / counters cnt) lazily - happened INSIDE the first FULL CUDA-graph
capture. vLLM captures PIECEWISE graphs first, in the same global pool, and keeps only weak references to the piece
outputs, so their blocks are free to the allocator but live at replay: the workspace landed on such a block. In a mixed
(PIECEWISE) step the eager attention between two pieces then wrote its partials over a tensor that the next piece
reads (the attention output gate), and real decode rows went NaN at the o_proj input.

The test replays that sequence on one GPU with the real kernel:
  piece 1 (pool P) writes `gate` (weak reference only, as vLLM's piecewise backend keeps), piece 2 reads it;
  then the capture loop of the next graph (through vllm's CudaGraphManager.capture, the hook the fix uses) makes the
  first attnv.run_mq call inside a capture in P;
  runtime of a mixed step: replay piece 1, eager run_mq (the attention op between pieces), replay piece 2.
PASS needs: workspace disjoint from the pool block of `gate`, piece 2's output bit-exact, the eager attention output
matching the FP32 reference, and run_mq refusing (False -> 1Cat's operator) instead of allocating when its workspace is
first needed inside a capture.
Before the fix (PYTHONPATH=<p15 tree>): the workspace overlaps `gate` and piece 2's output is corrupted -> FAIL.
Usage: PYTHONPATH=<tree> Q27_FUSE_BUILD=<build> python test_attnv_capture.py"""
import math, sys
import torch

torch.manual_seed(0)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
PAGE, HKV, HD, B, Q, L = 2048, 1, 256, 4, 4, 3000

import q27fuse.attnv as A
from vllm.v1.worker.gpu import cudagraph_utils as CU

print("attnv from", A.__file__, flush=True)
A.mq_ext()


def make():
    nbr = math.ceil(L / PAGE)
    nb = B * nbr + 3
    kv = (torch.randn(nb, 2, PAGE, HKV, HD, device=dev) * 2.0).to(torch.float8_e4m3fn).view(torch.uint8)
    perm = torch.randperm(nb, device=dev)[:B * nbr].int()
    table = perm.view(B, nbr).contiguous()
    rl = torch.tensor([L - Q + i + 1 for _ in range(B) for i in range(Q)], dtype=torch.int32, device=dev)
    q = torch.randn(B * Q, 6 * HKV, HD, device=dev).half()
    return kv, table, rl, q


def reference(q, kv, table, rl, scale):
    k8, v8 = kv[:, 0].view(torch.float8_e4m3fn), kv[:, 1].view(torch.float8_e4m3fn)
    out = torch.zeros_like(q, dtype=torch.float32)
    for r in range(q.shape[0]):
        n, b = int(rl[r]), r // Q
        blocks = table[b, :math.ceil(n / PAGE)].long()
        K = k8[blocks].float().reshape(-1, HKV, HD)[:n]
        V = v8[blocks].float().reshape(-1, HKV, HD)[:n]
        s = (q[r].float() @ K[:, 0].T) * scale
        out[r] = torch.softmax(s, -1) @ V[:, 0]
    return out


kv, table, rl, q = make()
k, v = kv[:, 0], kv[:, 1]
scale = 1.0 / 16
ref = reference(q, kv, table, rl, scale)
out_cap = torch.zeros_like(q)
out_eager = torch.zeros_like(q)
NG = 4 << 20                                     # 16 MiB of float32: larger than the workspace partials (~3.7 MB)
xg = torch.randn(NG, device=dev)
want = torch.sigmoid(xg) * 2.0
torch.cuda.synchronize()

pool = torch.cuda.graph_pool_handle()
s = torch.cuda.Stream()
g1, g2, g3 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
with torch.cuda.graph(g1, pool=pool, stream=s):          # PIECEWISE piece 1: the attention output gate
    gate = torch.sigmoid(xg)
with torch.cuda.graph(g2, pool=pool, stream=s):          # piece 2 (after the eager attention op): reads the gate
    y = gate * 2.0
gate_lo = gate.data_ptr(); gate_hi = gate_lo + gate.numel() * 4
del gate                                                 # vLLM keeps weak references to piece outputs

ok_cap = []


def capture_loop(self, *a, **kw):                        # the next graph's capture: first run_mq call inside it
    with torch.cuda.graph(g3, pool=pool, stream=s):
        ok_cap.append(A.run_mq(q, k, v, out_cap, table, rl, scale, 1.0, 1.0))
    return {}


class _PC: tensor_parallel_size = 4
class _MC:
    @staticmethod
    def get_total_num_kv_heads(): return 4
class _Cfg: parallel_config = _PC(); model_config = _MC()
class _Mgr: device = dev; vllm_config = _Cfg()


CU.CudaGraphManager.capture = capture_loop
if hasattr(A, "install_prealloc"):
    A.install_prealloc()                                 # the fix wraps CudaGraphManager.capture
else:
    print("tree has no attnv.install_prealloc (pre-fix tree)", flush=True)
CU.CudaGraphManager.capture(_Mgr())
torch.cuda.synchronize()

fails = []
key = (dev.type, dev.index, HKV)
ws = A._WS.get(key) or next(iter(A._WS.values()), None)
if ws is None:
    fails.append("no workspace allocated")
else:
    for nm, t in zip(("po", "pml", "cnt"), ws):
        lo, hi = t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()
        if lo < gate_hi and gate_lo < hi:
            fails.append(f"workspace {nm} {lo:x}..{hi:x} overlaps the pool block of piece 1's output {gate_lo:x}..{gate_hi:x}")
print(f"capture-time run_mq -> {ok_cap}", flush=True)

# runtime of a mixed step: piece 1, the eager attention between pieces, piece 2
g1.replay()
ok_eager = A.run_mq(q, k, v, out_eager, table, rl, scale, 1.0, 1.0)
g2.replay()
torch.cuda.synchronize()
if not ok_eager:
    fails.append("eager run_mq did not take the call")
if not torch.equal(y, want):
    bad = (y != want).sum().item()
    fails.append(f"piece 2 output corrupted by the eager attention: {bad} of {NG} values differ "
                 f"(non-finite {(~torch.isfinite(y)).sum().item()})")
err = ((out_eager.float() - ref).norm() / ref.norm()).item()
print(f"eager attention rel. L2 error vs FP32 reference {err:.2e}", flush=True)
if not err < 2e-3:
    fails.append(f"eager attention error {err:.2e}")
g3.replay(); torch.cuda.synchronize()
if ok_cap and ok_cap[0]:
    e3 = ((out_cap.float() - ref).norm() / ref.norm()).item()
    print(f"captured attention rel. L2 error vs FP32 reference {e3:.2e}", flush=True)
    if not e3 < 2e-3:
        fails.append(f"captured attention error {e3:.2e}")

# the second half of the fix: a workspace first needed inside a capture is never allocated there
A._WS.clear()
g4 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g4, pool=pool, stream=s):
    r4 = A.run_mq(q, k, v, out_cap, table, rl, scale, 1.0, 1.0)
if r4 or A._WS:
    fails.append(f"run_mq allocated its workspace inside a capture (returned {r4}, workspaces {len(A._WS)})")

print("FAIL:\n  " + "\n  ".join(fails) if fails else "PASS", flush=True)
sys.exit(1 if fails else 0)
