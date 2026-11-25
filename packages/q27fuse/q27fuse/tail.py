"""q27fuse.tail - the MTP engine step outside the verify layers (track "tail"; measurements in notes/tail.md).

Q27_TAIL = comma list of the pieces to install ("all" = every piece, the default; "0" or "" = none). No piece touches
the target's arithmetic, so target outputs are unchanged; drafts are unchanged too.
Q27_TAIL_CTL=<file>: a watcher thread reads "tail <comma list | none | all>" from it (polled 4x per second) and the
pieces outside the list stop acting at the next call (everything but "alias", which is fixed at graph capture), so a
piece's effect can be measured on one server with alternating windows.

  alias    the verify graph writes its hidden states straight into the drafter's input buffer: the CUDA-graph
           manager's persistent output buffer is the speculator's `hidden_states` buffer, so EagleSpeculator.propose
           finds the same pointer and skips its copy (one D2D memcpy of rows x 10 KB per step). Nothing reads the
           verify output after propose has started (the LM-head gather runs before it), and the next verify graph
           rewrites the rows. PIECEWISE / eager verify steps keep their copy.
  noseed   propose copies the sampler's temperature and seed vectors (max_num_seqs each, the temperature from host
           memory) into the drafter every step; only the probabilistic draft sampler reads them (draft_logits is not
           None). With greedy drafting the copies are skipped (two memcpys per step).
  nodec    steps whose draft length is 1 (Q27_SPEC_K, e.g. C >= 16 of the product mapping) run no decode pass, so
           prepare_eagle_decode (the decode-pass inputs) is skipped there (one eager kernel per step).
  argmax   target LM head top-1 (LogitsProcessor.get_top_tokens on the FP8 TurboMind head): the per-row (max, index)
           pair torch's logits.max(dim=-1) gives, offset by the shard start and stacked as fp32, comes from one kernel
           (csrc/tail_ops.cu row_argmax_pair) instead of reduce + add + 2 casts + stack; same values, same tie rule
           (lowest index, NaN first). The cross-device top-1 is 1Cat's op as before.
  regtop1  (with argmax) the pair is written straight into this rank's registered custom-all-reduce staging buffer and
           1Cat's cross-device top-1 op runs in its registered-input mode, so the op's own copy of the pair into that
           buffer (one memcpy per step) goes away. Same kernel, same data. Falls back to the copy mode on any error.
  gverify  greedy MTP verify (sm70_greedy_verify branch of maybe_sample_greedy) + get_num_sampled_and_rejected in one
           kernel (tail_ops.cu greedy_verify_fused) instead of gather + fill + cast + _greedy_verify_kernel +
           _get_num_sampled_and_rejected_kernel; the kernel runs where get_num_sampled_and_rejected is called (the next
           GPU op of GPUModelRunner.sample), with the same arguments.
  uvadev   vLLM keeps prefill_len / prompt_len (request states), num_blocks (block tables) and the M-RoPE delta in
           pinned host memory that kernels read over PCIe (UvaBackedTensor): the eager kernels that read them
           (num_sampled/rejected, combine_sampled_and_draft_tokens, rope positions, gather_block_tables) take ~1 us
           per request. Here each of these four gets a device copy that is refreshed by an H2D copy in stream order
           whenever the host values change (rarely: new requests, new KV blocks), and kernels read that copy. The
           host-memory snapshots are still written as before.
  mamba    align-mode state migration (precopy before the verify, postprocess after it) with a fixed 96 x 16 grid
           instead of requests x 96 x 16 (tail_mamba.py); identical copies.
  h2dc     prepare_inputs copies idx_mapping, cu_num_logits and query_start_loc host-to-device every step (pageable
           copies, ~3-7 us of GPU time each); in steady decode they repeat. Here a copy whose host values equal the
           last copy's is skipped: for the two fresh tensors the previous device tensor is returned (nothing writes
           them on the device), for query_start_loc (copied into the runner's persistent buffer) the buffer still
           holds the values. Dummy batches (InputBatch.make_dummy, which writes that buffer on the device) clear the
           cache.
  dupg     draft pass 1 (EagleSpeculator.prefill, captured in the drafter's first-pass graph) gathered the last
           accepted rows twice from the same MTP output (last_hidden_states and hidden_states are one tensor for this
           MTP): once for the draft head, once into the decode-pass input buffer. Here one index_select writes them into
           the input buffer and the head reads them there (one gather kernel fewer per step). Fixed at graph capture.
  accept   MambaHybridModelState.postprocess_state after a sampled step: _scatter_num_accepted_kernel, the replay
           record (cast + index_select + index_copy), the snapshot copy and the align postprocess become one kernel
           (tail_mamba.run_accept; identical writes; needs align mode with an initialised context, else 1Cat's path).
  rows     GDN replay row tables (SpecReplayState.prepare): the host rebuilds the speculative / plain row lists every
           step and copies them H2D from one pinned buffer; in steady decode they do not change, so the host write and
           the copy are skipped when the lists equal the last ones copied (the device table still holds them; nothing
           else writes it). Same kernel, same arguments otherwise.
  gdnmeta  next-step GDN metadata (MambaHybridModelState.prepare_attn -> compute_common_gdn_attn_metadata): for a pure
           speculative batch (every request verifies, no prefill / plain decode rows) 1Cat copies the request mask
           host-to-device (pageable, every step) and launches torch.arange for spec_token_indx; with FULL graphs the
           fused metadata kernel reads neither (its own pre-initialised buffers), only fallback builders and a debug
           shadow do. Here the mask's device copy is reused while its host values repeat, and spec_token_indx is a
           slice of one persistent arange; same values, same CommonGDNSpecMetadata otherwise (mixed batches: 1Cat's).
"""
import logging
import os
import threading
import time

log = logging.getLogger("q27fuse")

PIECES = ("alias", "noseed", "nodec", "argmax", "regtop1", "gverify", "uvadev", "mamba", "accept", "rows", "h2dc", "dupg",
          "gdnmeta")


def pieces():
    v = os.environ.get("Q27_TAIL", "all").strip()
    if v in ("", "0"):
        return set()
    if v == "all":
        return set(PIECES)
    s = {p.strip() for p in v.split(",") if p.strip()}
    bad = s - set(PIECES)
    assert not bad, f"Q27_TAIL: unknown pieces {sorted(bad)} (known: {PIECES})"
    return s


INSTALLED = set()
ACTIVE = set()        # pieces acting now (Q27_TAIL_CTL can narrow it at run time)
_spec = None          # the speculator whose propose() is running (for the module-level prepare_eagle_decode hook)
_EXT = None
_WS = {}              # device -> (partial uint64 [128 * SPLITS], counter int32 [128])
SPLITS = 8
_pending = None       # greedy verify launched lazily from get_num_sampled_and_rejected
_mirrors = []         # UVA tensors with a device copy (uvadev)
_mirrored = False
_watch = None


def ext():
    global _EXT
    if _EXT is None:
        from .ops import _load
        _EXT = _load("q27fuse_tail", "tail_ops.cu", "_tail")
    return _EXT


def _ws(dev):
    w = _WS.get(dev)
    if w is None:
        import torch
        from .ops import note_alloc                  # [head] logs an allocation inside a CUDA-graph capture
        w = (note_alloc("tail argmax partials", torch.zeros(128 * SPLITS, dtype=torch.int64, device=dev)),
             torch.zeros(128, dtype=torch.int32, device=dev))
        _WS[dev] = w
    return w


def _watcher(path):
    global ACTIVE
    seen = None
    while True:
        time.sleep(0.25)
        try:
            m = os.stat(path).st_mtime
        except OSError:
            continue
        if m == seen:
            continue
        seen = m
        try:
            w = open(path).read().split()
        except OSError:
            continue
        if len(w) >= 2 and w[0] == "tail":
            new = set() if w[1] == "none" else set(INSTALLED) if w[1] == "all" else set(w[1].split(","))
            ACTIVE = (new & INSTALLED) | ({"alias", "dupg"} & INSTALLED)
            log.warning("q27fuse tail: active pieces now %s", ",".join(sorted(ACTIVE)) or "none")


def apply():
    global INSTALLED, ACTIVE
    on = pieces()
    if not on:
        return
    INSTALLED = set(on)
    ACTIVE = set(on)
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as S
    R = MR.GPUModelRunner
    E = S.EagleSpeculator
    if getattr(E, "_q27_tail", False):
        return

    if "alias" in on:
        orig_capture = R.capture_model

        def capture_model(self, *a, **k):
            import torch
            m = getattr(self, "cudagraph_manager", None)
            sp = getattr(self, "speculator", None)
            hs = getattr(sp, "hidden_states", None)
            if (m is not None and isinstance(hs, torch.Tensor) and getattr(m, "hidden_states", 1) is None
                    and type(m).__name__ == "ModelCudaGraphManager"
                    and not getattr(m, "use_aux_hidden_state_outputs", False)):
                m.hidden_states = hs          # [max_num_tokens, hidden] fp16, allocated by the speculator
                log.warning("q27fuse tail: verify graphs write their hidden states into the drafter's buffer "
                            "(%s, %s)", tuple(hs.shape), hs.dtype)
            return orig_capture(self, *a, **k)

        R.capture_model = capture_model

    if "uvadev" in on or os.environ.get("Q27_TAIL_CTL"):
        orig_exec = R.execute_model

        def execute_model(self, *a, **k):
            global _watch
            import torch
            if not k.get("dummy_run") and not torch.cuda.is_current_stream_capturing():
                if _watch is None and os.environ.get("Q27_TAIL_CTL"):
                    _watch = threading.Thread(target=_watcher, args=(os.environ["Q27_TAIL_CTL"],),
                                              name="q27-tail-ctl", daemon=True)
                    _watch.start()
                if "uvadev" in INSTALLED:
                    _uva_sync(self)
            return orig_exec(self, *a, **k)

        R.execute_model = execute_model

    orig_propose = E.propose
    import inspect
    sig = inspect.signature(orig_propose)

    def propose(self, *a, **k):
        global _spec
        if "noseed" in ACTIVE and self.draft_logits is None:
            b = sig.bind_partial(self, *a, **k)
            b.arguments["temperature"] = self.temperature       # copy_ onto itself is a no-op (no memcpy)
            b.arguments["seeds"] = self.seeds
            a, k = b.args[1:], b.kwargs
        _spec = self
        try:
            return orig_propose(self, *a, **k)
        finally:
            _spec = None

    E.propose = propose

    if "nodec" in on:
        orig_ped = S.prepare_eagle_decode

        def prepare_eagle_decode(*a, **k):
            sp = _spec
            if "nodec" in ACTIVE and sp is not None and getattr(sp, "_q27_k", None) == 1:
                return None                   # no decode pass follows (spec_k: k = 1)
            return orig_ped(*a, **k)

        S.prepare_eagle_decode = prepare_eagle_decode
    if "argmax" in on:
        _apply_argmax()
    if "gverify" in on:
        _apply_gverify()
    if "mamba" in on:
        _apply_mamba()
    if "rows" in on:
        _apply_rows()
    if "accept" in on:
        _apply_accept()
    if "dupg" in on:
        _apply_dupg()
    if "h2dc" in on:
        _apply_h2dc()
    if "gdnmeta" in on:
        _apply_gdnmeta()
    E._q27_tail = True
    log.warning("q27fuse tail: %s", ", ".join(sorted(on)))


# ---------------------------------------------------------------------------------------------------------- uvadev
def _mirror(t):
    """give UvaBackedTensor t a device copy that copy_to_uva refreshes (H2D, stream ordered) when the values change"""
    import numpy as np
    import torch
    if getattr(t, "_q27_dev", None) is not None:
        return
    dev = torch.from_numpy(t.np.copy()).to(torch.device("cuda", torch.cuda.current_device()))
    t._q27_dev = dev
    t._q27_last = t.np.copy()
    t._q27_uva = t.gpu
    orig = t.copy_to_uva                    # bound method of the class

    def copy_to_uva(n=None):
        uva = orig(n)                        # the host snapshot as before (sets t.gpu to its UVA view)
        t._q27_uva = uva
        src = t.np if n is None else t.np[:n]
        k = len(src)
        if not np.array_equal(src, t._q27_last[:k]):
            t._q27_dev[:k].copy_(torch.from_numpy(src.copy()), non_blocking=True)
            t._q27_last[:k] = src
        if "uvadev" in ACTIVE:
            t.gpu = t._q27_dev if n is None else t._q27_dev[:n]
        return t.gpu

    t.copy_to_uva = copy_to_uva
    _mirrors.append(t)


def _uva_sync(runner):
    """main thread, before each real step: install the mirrors once, then point .gpu at the device copy or at the
    latest host snapshot according to ACTIVE"""
    global _mirrored
    if not _mirrored:
        _mirrored = True
        rs = getattr(runner, "req_states", None)
        bt = getattr(runner, "block_tables", None)
        rope = getattr(getattr(runner, "model_state", None), "rope_state", None)
        for t in (getattr(rs, "prefill_len", None), getattr(rs, "prompt_len", None), getattr(bt, "num_blocks", None),
                  getattr(rope, "prefill_delta", None)):
            if t is not None and hasattr(t, "np") and hasattr(t, "pool"):
                _mirror(t)
        log.warning("q27fuse tail: device copies of %d UVA-backed tensors", len(_mirrors))
    on = "uvadev" in ACTIVE
    for t in _mirrors:
        want = t._q27_dev if on else t._q27_uva
        if on and want.shape != t._q27_uva.shape:
            want = want[: t._q27_uva.shape[0]]
        if t.gpu.data_ptr() != want.data_ptr() or t.gpu.shape != want.shape:
            t.gpu = want


# ---------------------------------------------------------------------------------------------------------- argmax
def _apply_argmax():
    import torch
    from vllm.model_executor.layers import logits_processor as LP
    from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
    P = LP.LogitsProcessor
    orig = P.get_top_tokens

    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        """LogitsProcessor.get_top_tokens with the local (max, index) pair from row_argmax_pair; every other branch and
        the order of the calls are 1Cat's"""
        tp_size = get_tensor_model_parallel_world_size()
        rows = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
        qm = getattr(lm_head, "quant_method", None)
        if ("argmax" not in ACTIVE or tp_size == 1 or embedding_bias is not None or self.soft_cap is not None
                or self.scale != 1.0 or qm is None or not 1 <= rows <= 128
                or getattr(lm_head, "maybe_get_sm70_lm_head_top1_pair", None) is not None
                or torch.cuda.is_current_stream_capturing() or not hidden_states.is_cuda):
            return orig(self, lm_head, hidden_states, embedding_bias)
        top1_getter = getattr(lm_head, "maybe_get_sm70_lm_head_top1", None)
        if top1_getter is not None and top1_getter(hidden_states, embedding_bias) is not None:
            return orig(self, lm_head, hidden_states, embedding_bias)   # (FP16 heads; not this model's FP8 head)
        logits = qm.apply(lm_head, hidden_states, bias=embedding_bias)
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        if (logits.dtype != torch.float16 or logits.dim() != 2 or logits.stride(-1) != 1 or logits.shape[-1] % 8
                or logits.stride(0) % 8 or logits.data_ptr() % 16):
            local_max_vals, local_max_indices = logits.max(dim=-1)
            local_pair = torch.stack([local_max_vals.float(),
                                      (local_max_indices + lm_head.shard_indices.org_vocab_start_index).float()],
                                     dim=-1)
        else:
            part, cnt = _ws(logits.device)
            start = int(lm_head.shard_indices.org_vocab_start_index)
            ca = _regtop1_comm(self, rows) if "regtop1" in ACTIVE else None
            if ca is not None:
                # the pair goes straight into the registered staging buffer; the top-1 op then reads it in place
                staging = ca.buffer_ptrs[ca.rank]
                ext().row_argmax_pair(logits, start, staging, part, cnt, SPLITS)
                local_pair = ext().from_ptr(staging, 2 * rows, logits).view(rows, 2)
                LP._maybe_sync_top1_all_gather(self, local_pair)
                if self._top1_exchange is None:
                    try:
                        out = torch.empty((rows,), dtype=torch.int64, device=logits.device)
                        ca._native.top1_argmax(ca._ptr, local_pair.reshape(-1), out, 0, 0)
                        self._maybe_dump_top_token_margin(lm_head, hidden_states, embedding_bias, out)
                        return out
                    except Exception as e:          # registered mode refused: copy mode from now on
                        log.warning("q27fuse tail: registered top-1 failed (%r); regtop1 off", e)
                        ACTIVE.discard("regtop1")
                        local_pair = local_pair.clone()
                else:
                    local_pair = local_pair.clone()
            else:
                local_pair = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
                ext().row_argmax_pair(logits, start, local_pair.data_ptr(), part, cnt, SPLITS)
        LP._maybe_sync_top1_all_gather(self, local_pair)
        if self._top1_exchange is not None:
            exchanged_tokens = self._top1_exchange(local_pair)
            if exchanged_tokens is not None:
                return exchanged_tokens
        custom_top_tokens = self._maybe_custom_top1_argmax(local_pair)
        if custom_top_tokens is not None:
            self._maybe_dump_top_token_margin(lm_head, hidden_states, embedding_bias, custom_top_tokens)
            return custom_top_tokens
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        top_tokens = top_tokens.squeeze(-1).to(torch.int64)
        self._maybe_dump_top_token_margin(lm_head, hidden_states, embedding_bias, top_tokens)
        return top_tokens

    P.get_top_tokens = get_top_tokens


def _regtop1_comm(proc, rows):
    """the custom all-reduce communicator if 1Cat's eager top-1 would take its custom path for this pair"""
    from vllm.platforms import current_platform
    from vllm.distributed.parallel_state import get_tp_group
    if not getattr(proc._communication_policy, "top1_custom_ar", False) or not current_platform.is_device_capability(70):
        return None
    if not 2 <= 2 * rows <= 256:
        return None
    try:
        ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    except Exception:
        return None
    if (ca is None or ca.disabled or ca._IS_CAPTURING or (ca.world_size != 2 and not ca.fully_connected)
            or 8 * rows > ca.max_size):
        return None
    return ca


# ---------------------------------------------------------------------------------------------------------- gverify
def _apply_gverify():
    import torch
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu.spec_decode import sm70_greedy_verify as GV
    orig_msg = MR.maybe_sample_greedy
    orig_nsr = MR.get_num_sampled_and_rejected

    def maybe_sample_greedy(model, sampler, rejection_sampler, capability_enabled, verify_enabled,
                            sample_hidden_states, input_batch, grammar_output, sampler_output, cached_logits):
        """1Cat's sm70_greedy_verify branch, with greedy_verify deferred to get_num_sampled_and_rejected (fused)"""
        global _pending
        _pending = None
        ok = ("gverify" in ACTIVE and sampler_output is None and cached_logits is None
              and input_batch.num_draft_tokens > 0 and grammar_output is None and capability_enabled
              and hasattr(model, "get_top_tokens") and rejection_sampler is not None
              and rejection_sampler.synthetic_conditional_rates is None and verify_enabled
              and rejection_sampler.sampler.can_use_sm70_greedy_token_fastpath(input_batch)
              and rejection_sampler.num_speculative_steps + 1 <= 32
              and input_batch.input_ids.dtype == torch.int32 and input_batch.logits_indices.dtype == torch.int64
              and input_batch.cu_num_logits.dtype == torch.int32 and input_batch.seq_lens.dtype == torch.int32
              and input_batch.idx_mapping.dtype == torch.int32)
        if not ok:
            return orig_msg(model, sampler, rejection_sampler, capability_enabled, verify_enabled,
                            sample_hidden_states, input_batch, grammar_output, sampler_output, cached_logits)
        target_ids = model.get_top_tokens(sample_hidden_states).view(-1)
        n = input_batch.cu_num_logits.shape[0] - 1
        sampled = torch.empty((n, rejection_sampler.num_speculative_steps + 1), dtype=torch.int64,
                              device=target_ids.device)
        num_sampled = torch.empty(n, dtype=torch.int32, device=target_ids.device)
        _pending = (num_sampled, sampled, target_ids.to(torch.int64), input_batch.input_ids, input_batch.logits_indices,
                    input_batch.cu_num_logits)
        GV.logger.info_once("SM70 greedy MTP verification from TP-local argmax.")
        return GV.SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                                num_sampled=num_sampled)

    def get_num_sampled_and_rejected(num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len):
        global _pending
        p, _pending = _pending, None
        if p is None or p[0] is not num_sampled or p[5] is not cu_num_logits:
            return orig_nsr(num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len)
        _, sampled, target, input_ids, logits_indices, cu = p
        if prefill_len.dtype != torch.int32 or seq_lens.dtype != torch.int32 or idx_mapping.dtype != torch.int32:
            s2, n2 = GV.greedy_verify(target, input_ids[logits_indices], cu, sampled.shape[1] - 1)
            sampled.copy_(s2)
            num_sampled.copy_(n2)
            return orig_nsr(num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len)
        num_rejected = torch.empty_like(num_sampled)
        ext().greedy_verify_fused(target, input_ids, logits_indices, cu, seq_lens, prefill_len, idx_mapping, sampled,
                                  num_sampled, num_rejected)
        return num_sampled, num_rejected

    MR.maybe_sample_greedy = maybe_sample_greedy
    MR.get_num_sampled_and_rejected = get_num_sampled_and_rejected


# ---------------------------------------------------------------------------------------------------------- mamba
def _apply_mamba():
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as MH
    from . import tail_mamba as TM
    orig_pre = MH.run_mamba_align_precopy
    orig_post = MH.run_mamba_align_postprocess

    def run_mamba_align_precopy(*a, **k):
        return (TM.run_precopy if "mamba" in ACTIVE else orig_pre)(*a, **k)

    def run_mamba_align_postprocess(*a, **k):
        return (TM.run_postprocess if "mamba" in ACTIVE else orig_post)(*a, **k)

    MH.run_mamba_align_precopy = run_mamba_align_precopy
    MH.run_mamba_align_postprocess = run_mamba_align_postprocess


# ---------------------------------------------------------------------------------------------------------- rows
def _apply_rows():
    import numpy as np
    from vllm.model_executor.layers.mamba.gdn import spec_replay as SR
    from vllm.triton_utils import triton
    C = SR.SpecReplayState
    orig = C.prepare

    def prepare(self, idx_mapping, query_start_loc, num_computed_tokens, num_draft_tokens_per_req, is_prefilling,
                num_reqs, block_table_ptrs, block_table_stride, mamba_block_size):
        """SpecReplayState.prepare (1Cat) with the row-table H2D skipped when the tables did not change"""
        if "rows" not in ACTIVE:
            self._q27_last_rows = None
            return orig(self, idx_mapping, query_start_loc, num_computed_tokens, num_draft_tokens_per_req,
                        is_prefilling, num_reqs, block_table_ptrs, block_table_stride, mamba_block_size)
        if num_reqs == 0:
            return
        drafts = (np.zeros(num_reqs, dtype=np.int64) if num_draft_tokens_per_req is None
                  else np.asarray(num_draft_tokens_per_req[:num_reqs]))
        spec = drafts > 0
        if is_prefilling is not None:
            spec &= ~np.asarray(is_prefilling[:num_reqs], dtype=bool)
        spec_rows = np.flatnonzero(spec)
        plain_rows = np.flatnonzero(~spec)
        num_spec, num_plain = spec_rows.size, plain_rows.size
        last = getattr(self, "_q27_last_rows", None)
        if last is None or not (np.array_equal(last[0], spec_rows) and np.array_equal(last[1], plain_rows)):
            rows_np = self._rows_host.numpy()
            rows_np[:num_spec] = spec_rows
            rows_np[self.max_rows: self.max_rows + num_plain] = plain_rows
            self._rows.copy_(self._rows_host, non_blocking=True)
            self._q27_last_rows = (spec_rows.copy(), plain_rows.copy())
        block = triton.next_power_of_2(max(self.max_rows, 16))
        SR._prepare_replay_rows_kernel[(2,)](
            self._rows, idx_mapping, query_start_loc, num_computed_tokens, self.accepted, self.verified, self.side,
            self.spec_replay, self.spec_map, self.spec_ckpt_token, self.spec_ckpt_slot, self.plain_replay,
            self.plain_map, block_table_ptrs, block_table_stride, num_spec, num_plain,
            MAX_ROWS=self.max_rows, NUM_GROUPS=block_table_ptrs.numel(), MAMBA_BLOCK_SIZE=mamba_block_size,
            BLOCK=block)

    C.prepare = prepare


# ---------------------------------------------------------------------------------------------------------- dupg
def _apply_dupg():
    import torch
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as S
    E = S.EagleSpeculator
    orig = E.prefill

    def prefill(self, num_reqs, num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp,
                cudagraph_runtime_mode=S.CUDAGraphMode.NONE, mm_inputs=None):
        """EagleSpeculator.prefill (1Cat) with the two gathers of the last accepted rows done once when the model
        returns one tensor for both"""
        last_token_indices = self.last_token_indices[:num_reqs]
        pos = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]
        last_hidden_states, hidden_states = self.run_model(
            num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode, mm_inputs=mm_inputs)
        if (last_hidden_states is hidden_states and self.hidden_states.dtype == hidden_states.dtype
                and hidden_states.untyped_storage().data_ptr() != self.hidden_states.untyped_storage().data_ptr()):
            dst = self.hidden_states[:num_reqs]
            torch.index_select(hidden_states, 0, last_token_indices, out=dst)
            sample_hidden_states = dst
            same = True
        else:
            sample_hidden_states = last_hidden_states[last_token_indices]
            same = False
        self.draft_tokens[:num_reqs, 0] = self._sample_draft(
            sample_hidden_states, idx_mapping, pos, self.current_draft_step, self.draft_logits)
        if not same:
            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = pos
        if self.draft_step_listener is not None:
            self.draft_step_listener(self.draft_tokens, self.current_draft_step)

    import inspect
    src = inspect.getsource(orig)
    if "sample_hidden_states = last_hidden_states[last_token_indices]" not in src or \
            "self.hidden_states[:num_reqs] = hidden_states[last_token_indices]" not in src:
        log.warning("q27fuse tail: EagleSpeculator.prefill differs from the version dupg was written for; not applied")
        return
    E.prefill = prefill


# ---------------------------------------------------------------------------------------------------------- h2dc
_h2d_new = {}         # (dtype, len) -> (host copy, device tensor) for out=None copies
_h2d_out = {}         # (out data_ptr, len) -> host copy of what was copied there


def _apply_h2dc():
    import numpy as np
    import torch
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu import input_batch as IB
    orig = MR.async_copy_to_gpu

    def async_copy_to_gpu(x, out=None, device=None):
        if "h2dc" not in ACTIVE or torch.cuda.is_current_stream_capturing():
            _h2d_new.clear(); _h2d_out.clear()
            return orig(x, out=out, device=device)
        xn = x if isinstance(x, np.ndarray) else x.numpy() if isinstance(x, torch.Tensor) and x.is_cpu else None
        if xn is None or xn.ndim != 1:
            return orig(x, out=out, device=device)
        if out is None:
            key = (xn.dtype.str, xn.shape[0], str(device))
            c = _h2d_new.get(key)
            if c is not None and np.array_equal(c[0], xn):
                return c[1]
            t = orig(x, out=None, device=device)
            _h2d_new[key] = (xn.copy(), t)
            return t
        key = (out.data_ptr(), xn.shape[0])
        c = _h2d_out.get(key)
        if c is not None and np.array_equal(c, xn) and out.numel() == xn.shape[0]:
            return out
        r = orig(x, out=out, device=device)
        _h2d_out[key] = xn.copy()
        return r

    MR.async_copy_to_gpu = async_copy_to_gpu
    orig_dummy = IB.InputBatch.make_dummy            # classmethod, bound to the class

    def make_dummy(cls, *a, **k):
        _h2d_new.clear(); _h2d_out.clear()
        return orig_dummy(*a, **k)

    IB.InputBatch.make_dummy = classmethod(make_dummy)


# ---------------------------------------------------------------------------------------------------------- accept
def _apply_accept():
    import torch
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as MH
    from . import tail_mamba as TM
    C = MH.MambaHybridModelState
    orig = C.postprocess_state

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        ctx = getattr(self, "_mamba_ctx", None)
        n = idx_mapping.shape[0]
        if ("accept" in ACTIVE and isinstance(num_sampled, torch.Tensor) and num_sampled.dtype == torch.int32
                and getattr(self, "_align_mode", False) and num_computed_tokens is not None and ctx is not None
                and ctx.is_initialized and 0 < n <= 256 and idx_mapping.dtype == torch.int32
                and self.num_accepted_tokens_gpu.dtype == torch.int32
                and not torch.cuda.is_current_stream_capturing()):
            rep = getattr(self, "_spec_replay", None)
            if rep is None or rep.accepted.dtype == torch.int32:
                TM.run_accept(ctx, n, num_sampled, self.num_accepted_tokens_gpu,
                              rep.accepted if rep is not None else None, self._mamba_state_idx_gpu,
                              num_computed_tokens, idx_mapping)
                return
        return orig(self, idx_mapping, num_sampled, num_computed_tokens)

    C.postprocess_state = postprocess_state


# ---------------------------------------------------------------------------------------------------------- gdnmeta
_gm = {}              # "mask": (host bool copy, device tensor); "ar": persistent int32 arange; "e": empty int32


def _apply_gdnmeta():
    import torch
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as MH
    from vllm.v1.attention.ops import gdn_state as GS
    orig = MH.compute_common_gdn_attn_metadata

    def compute_common_gdn_attn_metadata(*, num_decode_draft_tokens_cpu, query_start_loc, query_start_loc_cpu,
                                         num_spec_state_tokens, legacy_mixed_decode_routing):
        """1Cat's compute_common_gdn_attn_metadata; for a pure speculative batch the mask H2D and the arange are
        replaced by a reused device mask (same host values) and a slice of a persistent arange"""
        nd = num_decode_draft_tokens_cpu
        kw = dict(num_decode_draft_tokens_cpu=nd, query_start_loc=query_start_loc,
                  query_start_loc_cpu=query_start_loc_cpu, num_spec_state_tokens=num_spec_state_tokens,
                  legacy_mixed_decode_routing=legacy_mixed_decode_routing)
        if ("gdnmeta" not in ACTIVE or num_spec_state_tokens <= 0 or nd is None or nd.ndim != 1
                or nd.numel() != query_start_loc_cpu.numel() - 1 or torch.cuda.is_current_stream_capturing()):
            _gm.pop("mask", None)
            return orig(**kw)
        masks_cpu = nd >= 0
        n_spec = int(masks_cpu.sum().item())
        if n_spec == 0 or int(nd[masks_cpu].sum().item()) == 0:
            return None
        qlens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        if bool((qlens_cpu[~masks_cpu] != 0).any().item()):       # prefill / plain decode rows: 1Cat's path
            _gm.pop("mask", None)
            return orig(**kw)
        dev = query_start_loc.device
        last = _gm.get("mask")
        if last is not None and last[1].device == dev and torch.equal(last[0], masks_cpu):
            masks = last[1]
        else:
            masks = masks_cpu.to(dev, non_blocking=True)
            _gm["mask"] = (masks_cpu.clone(), masks)
        size = min(n_spec * (num_spec_state_tokens + 1), int(query_start_loc_cpu[-1].item()))
        ar = _gm.get("ar")
        if ar is None or ar.device != dev or ar.numel() < size:
            ar = _gm["ar"] = torch.arange(max(size, 8192), dtype=torch.int32, device=dev)
            _gm["e"] = torch.empty(0, dtype=torch.int32, device=dev)
        return GS.CommonGDNSpecMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0, num_spec_decodes=n_spec,
            num_spec_decode_tokens=int(qlens_cpu.sum().item()), spec_query_start_loc=query_start_loc[: n_spec + 1],
            non_spec_query_start_loc=None, non_spec_query_start_loc_cpu=None, spec_sequence_masks_cpu=masks_cpu,
            spec_sequence_masks=masks, spec_token_indx=ar[:size], non_spec_token_indx=_gm["e"])

    MH.compute_common_gdn_attn_metadata = compute_common_gdn_attn_metadata
