# SPDX-License-Identifier: Apache-2.0
"""Unit & bit-exact parity tests for RingAPC (stride-checkpointed prefix caching)."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

try:
    import pytest
except ImportError:
    class _DummyMark:
        @staticmethod
        def skipif(*_a, **_kw):
            return lambda f: f
    pytest = SimpleNamespace(mark=_DummyMark())

from inkling_v100.conv_state import (
    ATTN_STREAM,
    K_STREAM,
    MLP_STREAM,
    RING_CONV_SLOTS,
    V_STREAM,
    _RingAPCCheckpointPool,
    _RingAPCCoreTracker,
    _RingConvRuntime,
)
from inkling_v100.ops.sconv import ring_capture_ckpt, ring_conv_metadata, short_conv


def _make_fake_layer(num_slots: int, num_heads: int, raw_head_size: int, device: torch.device):
    return SimpleNamespace(
        kv_cache=torch.zeros(
            (num_slots + 1, num_heads, RING_CONV_SLOTS, raw_head_size),
            dtype=torch.float16,
            device=device,
        ),
        ring_layer_idx=0,
    )


def _make_batch(req_specs, device: torch.device):
    """Build a minimal InputBatch-like object from [(req_id, slot, start, length), ...]."""
    num_reqs = len(req_specs)
    qsl = [0]
    pos_list = []
    req_ids = []
    idx_list = []
    comp_list = []
    sched_list = []
    for rid, slot, start, length in req_specs:
        req_ids.append(rid)
        idx_list.append(slot)
        comp_list.append(start)
        sched_list.append(length)
        qsl.append(qsl[-1] + length)
        pos_list.extend(range(start, start + length))
    total = qsl[-1]
    return SimpleNamespace(
        req_ids=req_ids,
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        num_tokens=total,
        num_tokens_after_padding=total,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        query_start_loc_np=np.array(qsl, dtype=np.int32),
        idx_mapping=torch.tensor(idx_list, dtype=torch.int32, device=device),
        idx_mapping_np=np.array(idx_list, dtype=np.int32),
        num_computed_tokens_np=np.array(comp_list, dtype=np.int32),
        num_scheduled_tokens=np.array(sched_list, dtype=np.int32),
        positions=torch.tensor(pos_list, dtype=torch.int64, device=device),
        is_dummy_batch=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ring_apc_bit_exact_prefill_and_chunk_boundary():
    """Verify bit-exact short_conv output when resuming from RingAPC checkpoints across:
    1) Standard prefill crossing multiple stride boundaries (512, 1024)
    2) Chunk boundary straddling `b_pos - start < 2` (start=510, chunk crosses b_pos=511)
    3) Decode-mode in-place step crossing a stride boundary (req_len=4 crossing 511)
    """
    torch.manual_seed(42)
    dev = torch.device("cuda")
    num_heads = 2
    head_dim = 128
    hidden_per_head = 1152
    raw_head_size = 2 * head_dim + 2 * hidden_per_head  # 2560
    stream_ranges = (
        (0, head_dim),
        (head_dim, head_dim),
        (2 * head_dim, hidden_per_head),
        (2 * head_dim + hidden_per_head, hidden_per_head),
    )

    T = 1200
    stride = 512
    anchor_bs = 32
    x_attn = torch.randn((T, num_heads * hidden_per_head), dtype=torch.float16, device=dev)
    w_attn = torch.randn((num_heads * hidden_per_head, 4), dtype=torch.float16, device=dev) * 0.1

    # 1. Cold reference run of all 1200 tokens in one shot
    ref_cache = torch.zeros((2, num_heads, RING_CONV_SLOTS, raw_head_size), dtype=torch.float16, device=dev)
    ref_pos = torch.arange(T, dtype=torch.int64, device=dev)
    ref_qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
    ref_bt = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    ref_seq = torch.zeros(T, dtype=torch.int32, device=dev)
    ref_qs = torch.zeros(T, dtype=torch.int32, device=dev)
    ref_sm = torch.full((T,), -1, dtype=torch.int64, device=dev)
    ring_conv_metadata(ref_qsl, ref_pos, ref_bt[:, 0], 1, T, ref_seq, ref_qs, ref_sm, T)
    off_a, w_a = stream_ranges[ATTN_STREAM]
    ref_out = short_conv(
        x_attn, w_attn, ref_cache, ref_pos, ref_bt, ref_seq, ref_sm, ref_qs, off_a, w_a
    )

    # 2. Chunked run with RingAPC: chunk 1 = [0..510), chunk 2 = [510..1100) (straddles 511 & 1023!)
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    kv_cfg = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.5.attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=anchor_bs,
                    num_kv_heads=num_heads,
                    head_size=head_dim,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    rt = _RingConvRuntime(max_num_seqs=4, max_num_tokens=2048, device=dev, apc_enabled=True)
    layer = _make_fake_layer(4, num_heads, raw_head_size, dev)
    layer.ring_layer_idx = rt.register_layer(layer)

    # Physical block table for req A: block i -> phys_id = 100 + i
    bt_a = torch.arange(100, 100 + 64, dtype=torch.int32, device=dev).unsqueeze(0)

    def run_chunk(req_id, slot, start, length, bt):
        b = _make_batch([(req_id, slot, start, length)], dev)
        rt.update(b, CUDAGraphMode.NONE, False, block_tables=(bt,), kv_cache_config=kv_cfg)
        m = rt.current_meta
        assert m is not None
        out = short_conv(
            x_attn[start : start + length],
            w_attn,
            layer.kv_cache,
            b.positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            off_a,
            w_a,
            tail_src_toks=m.tail_src_toks,
            tail_req_slots=m.tail_req_slots,
            num_tail_writes=m.num_tail_writes,
            ckpt_dst=m.layer_ckpt_dst(layer.ring_layer_idx),
            ckpt_b_pos=m.ckpt_b_pos,
            ckpt_qstart=m.ckpt_qstart,
            ckpt_start_pos=m.ckpt_start_pos,
            ckpt_req_slots=m.ckpt_req_slots,
            num_ckpt_writes=m.num_ckpt_writes,
        )
        rt.commit_step()
        return out

    rt.on_add_request(0, 0)
    out_c1 = run_chunk("reqA", 0, 0, 510, bt_a)
    out_c2 = run_chunk("reqA", 0, 510, 590, bt_a)  # crosses 511 (with start=510!) and 1023!
    torch.testing.assert_close(out_c1, ref_out[:510], atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_c2, ref_out[510:1100], atol=0.0, rtol=0.0)
    rt.on_remove_request(0)

    # Zero out the ENTIRE GPU ring buffer to prove resume relies solely on pinned CPU checkpoint!
    layer.kv_cache.zero_()

    # 3. Brand-new reqB in DIFFERENT slot 2, prefix hit H=512 (using phys_id 100 + 511//32 = 115)
    rt.on_add_request(2, 512)
    out_b = run_chunk("reqB", 2, 512, 688, bt_a)
    torch.testing.assert_close(out_b, ref_out[512:1200], atol=0.0, rtol=0.0)

    # 4. Brand-new reqC in slot 3, prefix hit H=1024 (using phys_id 100 + 1023//32 = 131)
    layer.kv_cache.zero_()
    rt.on_add_request(3, 1024)
    out_c = run_chunk("reqC", 3, 1024, 176, bt_a)
    torch.testing.assert_close(out_c, ref_out[1024:1200], atol=0.0, rtol=0.0)

    # 5. Decode-mode short step crossing b_pos=511 (reqD runs 0..508, then 4-token decode 508..512)
    layer.kv_cache.zero_()
    bt_d = torch.arange(200, 264, dtype=torch.int32, device=dev).unsqueeze(0)
    rt.on_add_request(1, 0)
    run_chunk("reqD", 1, 0, 508, bt_d)
    out_dec = run_chunk("reqD", 1, 508, 4, bt_d)  # req_len=4 <= 12 => in-place + post_ring_save!
    torch.testing.assert_close(out_dec, ref_out[508:512], atol=0.0, rtol=0.0)

    # Resume reqE from reqD's decode-saved checkpoint at H=512!
    layer.kv_cache.zero_()
    rt.on_add_request(0, 512)
    out_e = run_chunk("reqE", 0, 512, 100, bt_d)
    torch.testing.assert_close(out_e, ref_out[512:612], atol=0.0, rtol=0.0)


def test_ring_apc_core_tracker_alignment_and_eviction():
    tracker = _RingAPCCoreTracker(anchor_gid=20, anchor_bs=32, stride_tokens=512, max_slots=320)
    assert tracker.stride_blocks == 16
    assert tracker.max_slots == 64

    # Non-anchor group and non-stride blocks are ignored
    tracker.on_block_cached(0, 15, 999)
    tracker.on_block_cached(20, 14, 998)
    assert not tracker.has_and_touch(999)
    assert not tracker.has_and_touch(998)

    # Logical block 15 (ending at token 511) IS recorded
    tracker.on_block_cached(20, 15, 42)
    assert tracker.has_and_touch(42)

    # Eviction removes it cleanly
    tracker.on_block_evicted(42)
    assert not tracker.has_and_touch(42)

    # LRU evicts oldest once max_slots (64) is exceeded, preserving touched hot entry
    for i in range(64):
        tracker.on_block_cached(20, 15, 1000 + i)
    assert tracker.has_and_touch(1000)  # promote 1000 to MRU
    tracker.on_block_cached(20, 15, 2000)  # should evict 1001, NOT 1000
    assert tracker.has_and_touch(1000)
    assert not tracker.has_and_touch(1001)
    assert tracker.has_and_touch(2000)


if __name__ == "__main__":
    test_ring_apc_core_tracker_alignment_and_eviction()
    print("PASS: test_ring_apc_core_tracker_alignment_and_eviction")
    if torch.cuda.is_available():
        test_ring_apc_bit_exact_prefill_and_chunk_boundary()
        print("PASS: test_ring_apc_bit_exact_prefill_and_chunk_boundary (BIT-EXACT 0.0 diff!)")

