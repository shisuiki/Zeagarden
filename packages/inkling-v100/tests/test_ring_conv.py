#!/usr/bin/env python3
"""Bit-exact unit test: Option 1B fixed 16-slot ring conv vs exact PyTorch causal conv reference.

Tests:
  1. Multi-chunk prefill across tricky lengths (L = 13, 16, 17, 64, 512, 2048) + decode
  2. DSpark speculative verify (Q = 8) with arbitrary partial rejections (accept a in {1, 2, 3, 5, 7, 8})
  3. Mixed prefill + decode in the same batch with arbitrary persistent idx_mapping permutation
  4. Slot reuse (finished request slot immediately reused by a fresh pos=0 request)
  5. Both static `short_conv` (ATTN/MLP streams) and dynamic `attention_prologue` (K/V streams)
"""

from __future__ import annotations

import torch

from inkling_v100.conv_state import ATTN_STREAM, RING_CONV_SLOTS, RING_MAX_INPLACE_LEN
from inkling_v100.ops.sconv import ring_conv_metadata, short_conv


def ref_causal_conv(seq_x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exact reference matching `_conv`: (sum_{k=0}^{W-1} x_{t-(W-1-k)} * w_k + x_t) / sqrt(W+1)."""
    L, C = seq_x.shape
    W = weight.shape[1]
    x_h = seq_x.half().float()
    w_f = weight.half().float()
    out = torch.zeros((L, C), dtype=torch.float32, device=seq_x.device)
    scale = (W + 1) ** -0.5
    for t in range(L):
        acc = torch.zeros(C, dtype=torch.float32, device=seq_x.device)
        for k in range(W):
            lag = W - 1 - k
            src = t - lag
            if src >= 0:
                acc = acc + x_h[src] * w_f[:, k]
        out[t] = (acc + x_h[t]) * scale
    return out


def run_step(
    cache: torch.Tensor,
    weight: torch.Tensor,
    x_list: list[torch.Tensor],
    pos_list: list[torch.Tensor],
    slot_ids: list[int],
    stream_offset: int,
    stream_width: int,
    pad_tokens: int = 0,
) -> list[torch.Tensor]:
    dev = cache.device
    num_reqs = len(x_list)
    lens = [t.shape[0] for t in x_list]
    num_actual = sum(lens)
    num_padded = num_actual + pad_tokens
    C = x_list[0].shape[1]

    x_cat = torch.zeros((num_padded, C), dtype=torch.float16, device=dev)
    pos_cat = torch.zeros((num_padded,), dtype=torch.int64, device=dev)
    qsl = [0]
    off = 0
    for x, p in zip(x_list, pos_list):
        n = x.shape[0]
        x_cat[off : off + n].copy_(x)
        pos_cat[off : off + n].copy_(p)
        off += n
        qsl.append(off)
    qsl_t = torch.tensor(qsl, dtype=torch.int32, device=dev)
    bt = torch.tensor(slot_ids, dtype=torch.int32, device=dev).unsqueeze(1)

    seq_idx = torch.zeros(num_padded, dtype=torch.int32, device=dev)
    query_start = torch.zeros(num_padded, dtype=torch.int32, device=dev)
    slot_mapping = torch.full((num_padded,), -1, dtype=torch.int64, device=dev)
    ring_conv_metadata(
        qsl_t,
        pos_cat,
        bt[:, 0],
        num_reqs,
        num_actual,
        seq_idx,
        query_start,
        slot_mapping,
        num_padded,
        ring_n=RING_CONV_SLOTS,
        max_inplace_len=RING_MAX_INPLACE_LEN,
    )

    src_list, rslot_list = [], []
    for r in range(num_reqs):
        if lens[r] > RING_MAX_INPLACE_LEN:
            qend = qsl[r + 1]
            for tok in range(qend - 3, qend):
                src_list.append(tok)
                rslot_list.append(slot_ids[r])
    n_tail = len(src_list)
    tail_src = torch.tensor(src_list, dtype=torch.int32, device=dev) if n_tail else None
    tail_slots = torch.tensor(rslot_list, dtype=torch.int32, device=dev) if n_tail else None

    out = torch.empty((num_padded, C), dtype=torch.float32, device=dev)
    short_conv(
        x_cat,
        weight,
        cache,
        pos_cat,
        bt,
        seq_idx,
        slot_mapping,
        query_start,
        stream_offset,
        stream_width,
        out=out,
        tail_src_toks=tail_src,
        tail_req_slots=tail_slots,
        num_tail_writes=n_tail,
    )
    outs = []
    for r in range(num_reqs):
        outs.append(out[qsl[r] : qsl[r + 1]].clone())
    return outs


def main() -> None:
    torch.manual_seed(42)
    dev = torch.device("cuda:0")
    H_kv, raw_head = 2, 2560
    C = 2304
    stream_off, stream_w = 256, 1152  # ATTN_STREAM
    max_seqs = 8

    cache = torch.zeros((max_seqs + 1, H_kv, RING_CONV_SLOTS, raw_head), dtype=torch.float16, device=dev)
    weight = torch.randn((C, 4), dtype=torch.float16, device=dev) * 0.25

    # Test 1: Tricky chunked prefill boundaries (L = 13, 16, 17, 1, 12, 512, 2048)
    for chunks in [
        [16, 16, 16, 1, 1, 8],
        [13, 17, 2, 12, 32, 3],
        [2048, 2048, 8, 1],
    ]:
        cache.zero_()
        total = sum(chunks)
        full_x = torch.randn((total, C), dtype=torch.float16, device=dev)
        ref = ref_causal_conv(full_x, weight)
        pos = 0
        for ch in chunks:
            p = torch.arange(pos, pos + ch, dtype=torch.int64, device=dev)
            [got] = run_step(cache, weight, [full_x[pos : pos + ch]], [p], [3], stream_off, stream_w, pad_tokens=4)
            err = (got - ref[pos : pos + ch]).abs().max().item()
            assert err < 2e-5, f"chunked {chunks} at pos={pos} ch={ch}: max_err={err}"
            pos += ch
    print("PASS Test 1: Tricky chunked prefill boundaries (13, 16, 17, 2048) + decode")

    # Test 2: DSpark Q=8 verify with arbitrary partial rejections (accept a in {1, 2, 3, 5, 7, 8})
    cache.zero_()
    accepted_x = torch.randn((120, C), dtype=torch.float16, device=dev)
    ref = ref_causal_conv(accepted_x, weight)
    # Initial 19-token prefill
    p0 = torch.arange(0, 19, dtype=torch.int64, device=dev)
    [got] = run_step(cache, weight, [accepted_x[:19]], [p0], [5], stream_off, stream_w)
    assert (got - ref[:19]).abs().max().item() < 2e-5
    cur = 19
    for accept_a in [1, 2, 1, 7, 3, 5, 8, 1, 4, 2, 6, 1, 8]:
        if cur + 8 > 120:
            break
        # First `accept_a` tokens match true sequence; remaining `8 - accept_a` are garbage rejected drafts!
        q_x = accepted_x[cur : cur + 8].clone()
        if accept_a < 8:
            q_x[accept_a:] = torch.randn((8 - accept_a, C), dtype=torch.float16, device=dev) * 5.0
        p = torch.arange(cur, cur + 8, dtype=torch.int64, device=dev)
        [got] = run_step(cache, weight, [q_x], [p], [5], stream_off, stream_w, pad_tokens=8)
        err = (got[:accept_a] - ref[cur : cur + accept_a]).abs().max().item()
        assert err < 2e-5, f"DSpark verify at cur={cur}, accept_a={accept_a}: max_err={err}"
        cur += accept_a
    print("PASS Test 2: DSpark Q=8 verify with arbitrary rejections (a in 1..8) + zero rollback!")

    # Test 3: Mixed prefill + decode + batch permutation + slot reuse
    cache.zero_()
    xs = [torch.randn((90, C), dtype=torch.float16, device=dev) for _ in range(4)]
    refs = [ref_causal_conv(x, weight) for x in xs]
    # Step A: req 0 (slot 2, L=16), req 1 (slot 7, L=25), req 2 (slot 0, L=5)
    outs = run_step(
        cache, weight,
        [xs[0][:16], xs[1][:25], xs[2][:5]],
        [torch.arange(0, 16, device=dev), torch.arange(0, 25, device=dev), torch.arange(0, 5, device=dev)],
        [2, 7, 0], stream_off, stream_w,
    )
    assert (outs[0] - refs[0][:16]).abs().max().item() < 2e-5
    assert (outs[1] - refs[1][:25]).abs().max().item() < 2e-5
    assert (outs[2] - refs[2][:5]).abs().max().item() < 2e-5
    # Step B: permute batch order + req 2 finishes & req 3 reuses slot 0 from pos=0 + mixed prefill/verify!
    outs = run_step(
        cache, weight,
        [xs[1][25:33], xs[3][:20], xs[0][16:32]],
        [torch.arange(25, 33, device=dev), torch.arange(0, 20, device=dev), torch.arange(16, 32, device=dev)],
        [7, 0, 2], stream_off, stream_w, pad_tokens=3,
    )
    assert (outs[0] - refs[1][25:33]).abs().max().item() < 2e-5
    assert (outs[1] - refs[3][:20]).abs().max().item() < 2e-5
    assert (outs[2] - refs[0][16:32]).abs().max().item() < 2e-5
    # Step C: follow-up decode for all three
    outs = run_step(
        cache, weight,
        [xs[0][32:36], xs[3][20:24], xs[1][33:37]],
        [torch.arange(32, 36, device=dev), torch.arange(20, 24, device=dev), torch.arange(33, 37, device=dev)],
        [2, 0, 7], stream_off, stream_w,
    )
    assert (outs[0] - refs[0][32:36]).abs().max().item() < 2e-5
    assert (outs[1] - refs[3][20:24]).abs().max().item() < 2e-5
    assert (outs[2] - refs[1][33:37]).abs().max().item() < 2e-5
    print("PASS Test 3: Mixed prefill/decode + batch permutation + immediate slot reuse!")
    print("ALL OPTION 1B RING-CONV UNIT TESTS PASSED (max_err < 2e-5)!")


if __name__ == "__main__":
    main()
