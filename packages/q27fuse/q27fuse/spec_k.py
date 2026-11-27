"""q27fuse.spec_k - MTP with a draft length chosen per engine step from the batch size.

Q27_SPEC_K="K1@n1,K2@n2,...": from n_i requests per step up, draft K_i tokens (the entry with the largest n_i not above
the step's request count; below the first n_i the server's own K). E.g. "4@1,3@2,1@16,0@33". The server must start with
num_speculative_tokens >= every K_i (larger K_i are clamped to it). K_i = 0 is a draft-free step: plain decoding of one
token per request, and the drafter does not run at all. The MTP layer's KV then has no entries for those positions, so
drafts made after the batch shrinks back below the K = 0 threshold attend over stale slots and are accepted less often;
outputs are unaffected (the verify is exact). The older form "k:n" means "K@1,k@n" with the server's K. A single entry "K@1" keeps K everywhere but
still adds the FULL graphs below, which 1Cat captures only up to 16 requests.

The verify step then checks k + 1 positions per request. Both sides decide from the same quantity, the number of
requests in the step being executed, so no message is needed between the scheduler and the workers:
  - scheduler (AsyncScheduler._update_after_schedule): the placeholders it adds for the next step's spec tokens are k
    long;
  - drafter (EagleSpeculator.propose / multi_step_decode): after the first draft pass it runs k - 1 decode passes
    (1Cat's per-step FULL graphs; its single K-1-step graph only when k = K), and reports k through
    next_num_draft_tokens (the column count the runner hands back); the draft buffers keep K columns, of which
    k..K-1 are stale and never scheduled;
  - CUDA graphs (CudaGraphManager._init_candidates): the target verifier and the drafter's first pass get FULL
    uniform-decode graphs for k + 1 tokens per request (1 for K = 0) over the request counts that use k (1, 2, 3, 4, 6, 8, every 8th
    and the range end; 1Cat rounds the verifier's capture sizes to multiples of K + 1 and stops at 16 requests); the
    drafter's single-token decode passes get FULL graphs up to the server's request limit.
Model Runner V2 already handles per-request verify lengths (greedy verify, rejection sampler, GDN replay commit), so a
shorter uniform verify is an ordinary batch shape. Outputs are unchanged under greedy decoding: only how many tokens are
proposed per step changes."""
import logging
import os

log = logging.getLogger("q27fuse")


def config():
    """[(n, k)] ascending in n; k = -1 stands for the server's K"""
    v = os.environ.get("Q27_SPEC_K", "").strip()
    if not v:
        return None
    if ":" in v:                                            # legacy "k:n"
        k, n = (int(x) for x in v.split(":"))
        ent = [(1, -1), (n, k)]
    else:
        ent = []
        for part in v.split(","):
            k, n = part.split("@")
            ent.append((int(n), int(k)))
    ent.sort()
    assert ent and all(n >= 1 and k >= -1 for n, k in ent), f"Q27_SPEC_K={v!r}: K@n entries, K >= 0, n >= 1"
    return ent


def k_for(num_reqs: int, K: int, cfg) -> int:
    k = -1
    for n, kk in cfg:
        if num_reqs >= n:
            k = kk
    return K if k == -1 else min(k, K)


def ranges(cfg, K, max_reqs):
    """[(k, first request count, last request count)] for every entry, k resolved against the server's K"""
    out = []
    lo0 = cfg[0][0]
    if lo0 > 1:
        out.append((K, 1, min(lo0 - 1, max_reqs)))
    for i, (n, k) in enumerate(cfg):
        hi = cfg[i + 1][0] - 1 if i + 1 < len(cfg) else max_reqs
        if n <= min(hi, max_reqs):
            out.append((K if k == -1 else min(k, K), n, min(hi, max_reqs)))
    return out


def capture_counts(lo, hi):
    """request counts that get a graph inside [lo, hi]: 1, 2, 3, 4, 6, 8, 12, then every 8th, and hi (batches in
    between pad up to the next one); one graph per count up to 64 requests ran the 64-sequence server out of memory"""
    return sorted({b for b in range(lo, hi + 1) if b in (1, 2, 3, 4, 6, 8, 12) or b % 8 == 0} | {hi})


def apply():
    cfg = config()
    if cfg is None:
        return
    _patch_scheduler(cfg)
    _patch_speculator(cfg)
    _patch_graphs(cfg)
    _patch_greedy()
    log.warning("q27fuse: MTP draft length by requests per step %s (Q27_SPEC_K)",
                ", ".join(f"{'K' if k == -1 else k} from {n}" for n, k in cfg))


def _patch_scheduler(cfg):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    if getattr(AsyncScheduler, "_q27_spec_k", False):
        return
    orig = AsyncScheduler._update_after_schedule

    def _update_after_schedule(self, scheduler_output):
        K = self.num_spec_tokens
        if K >= 1:
            k = k_for(len(scheduler_output.num_scheduled_tokens), K, cfg)
            if len(self._spec_token_placeholders) != k:
                self._spec_token_placeholders = (-1,) * k
        orig(self, scheduler_output)

    AsyncScheduler._update_after_schedule = _update_after_schedule
    AsyncScheduler._q27_spec_k = True


def _patch_speculator(cfg):
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as S
    E = S.EagleSpeculator
    if getattr(E, "_q27_spec_k", False):
        return
    import inspect
    orig_propose = E.propose
    orig_msd = E.multi_step_decode
    msd_has_active = "active_spec_steps" in inspect.signature(orig_msd).parameters

    def propose(self, input_batch, *args, **kwargs):
        k = k_for(input_batch.num_reqs, self.num_speculative_steps, cfg)
        self._q27_k = k
        if k == 0 and not kwargs.get("dummy_run") and not kwargs.get("is_profile"):
            # draft-free step: the runner takes next_num_draft_tokens() = 0 columns, so the stale buffer is never
            # scheduled; the MTP layer's KV gets no entries for this step's positions
            return self.draft_tokens[: input_batch.num_reqs]
        return orig_propose(self, input_batch, *args, **kwargs)

    def multi_step_decode(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp, active_spec_steps=None):
        """1Cat's per-step loop (EagleSpeculator.multi_step_decode without the single K-1-step graph) for k - 1 passes
        (none for k = 0 or 1). num_speculative_steps stays K: the draft buffers and update_eagle_draft_inputs are laid out
        for K."""
        k = getattr(self, "_q27_k", self.num_speculative_steps)
        if active_spec_steps is not None:
            k = min(k, int(active_spec_steps))
        if k >= self.num_speculative_steps:
            if msd_has_active:
                return orig_msd(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp,
                                active_spec_steps=active_spec_steps)
            return orig_msd(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp)
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        for step in range(1, k):
            attn_metadata = None
            slot_mappings_by_layer = None
            if not skip_attn:
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping, query_start_loc, positions, batch_desc.num_tokens)
                slot_mappings_by_layer = S.build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs, num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens)
            self.current_draft_step.fill_(step)
            if batch_desc.cg_mode == S.CUDAGraphMode.FULL:
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self.generate_draft(num_reqs, batch_desc.num_tokens, attn_metadata, slot_mappings_by_layer,
                                    num_tokens_across_dp=num_tokens_across_dp,
                                    cudagraph_runtime_mode=batch_desc.cg_mode)

    def next_num_draft_tokens(self):
        return getattr(self, "_q27_k", self.num_speculative_steps)

    E.propose = propose
    E.multi_step_decode = multi_step_decode
    E.next_num_draft_tokens = next_num_draft_tokens
    E._q27_spec_k = True


def _patch_greedy():
    """Draft-free steps of several requests take 1Cat's greedy token fast path (TP-local argmax + cross-device top-1,
    model.get_top_tokens) instead of the full sampler: 1Cat gates the no-draft fast path on a single request, so a K = 0
    step of 48 requests ran the Gumbel sampler over the gathered logits (0.37 ms + 0.18 ms of logits processing at C 48,
    trace k0t). The eligibility test is 1Cat's own (sampler.can_use_sm70_greedy_token_fastpath: every request greedy, no
    logprobs, logit bias, penalties or bad words), under which the argmax equals the sampler's token."""
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu.spec_decode import sm70_greedy_verify as GV
    if getattr(GV, "_q27_spec_k", False):
        return
    orig = MR.maybe_sample_greedy

    def maybe_sample_greedy(model, sampler, rejection_sampler, capability_enabled, verify_enabled,
                            sample_hidden_states, input_batch, grammar_output, sampler_output, cached_logits):
        n = input_batch.num_reqs
        if (sampler_output is None and cached_logits is None and n > 1 and input_batch.num_draft_tokens == 0
                and input_batch.num_tokens == n and not input_batch.is_prefilling_np[:n].any()
                and grammar_output is None and capability_enabled and hasattr(model, "get_top_tokens")
                and sampler is not None and sampler.can_use_sm70_greedy_token_fastpath(input_batch)):
            sampled = model.get_top_tokens(sample_hidden_states)
            return GV.SamplerOutput(sampled_token_ids=sampled.view(-1, 1), logprobs_tensors=None, num_nans=None,
                                    num_sampled=input_batch.seq_lens.new_ones(n))
        return orig(model, sampler, rejection_sampler, capability_enabled, verify_enabled, sample_hidden_states,
                    input_batch, grammar_output, sampler_output, cached_logits)

    MR.maybe_sample_greedy = maybe_sample_greedy
    GV._q27_spec_k = True


def _patch_graphs(cfg):
    from vllm.v1.worker.gpu import cudagraph_utils as CU
    from vllm.config.compilation import CUDAGraphMode
    if getattr(CU.CudaGraphManager, "_q27_spec_k", False):
        return
    orig = CU.CudaGraphManager._init_candidates

    def add(self, mode, q, reqs):
        """FULL uniform-decode descriptors of q tokens per request for these request counts, where missing; batches of
        q-token requests between two such sizes pad up to the next one"""
        descs = self._capture_descs.setdefault(mode, [])
        have = {d.num_tokens for d in descs if d.uniform_token_count == q and d.attention_context_bucket is None}
        new = [CU.BatchExecutionDescriptor(cg_mode=mode, num_tokens=q * b, num_reqs=b, uniform_token_count=q)
               for b in reqs if q * b not in have]
        if not new:
            return 0
        top = max(d.num_tokens for d in new)
        if len(self._candidates) <= top:
            self._candidates.extend([] for _ in range(top + 1 - len(self._candidates)))
        sizes = sorted(have | {d.num_tokens for d in new})
        for d in new:
            lo = max([t for t in sizes if t < d.num_tokens], default=0)
            for i in range(lo + 1, d.num_tokens + 1):
                self._candidates[i] = [d] + [c for c in self._candidates[i]
                                             if not (c.uniform_token_count == q and c.num_tokens > d.num_tokens)]
        descs.extend(new)
        descs.sort(key=lambda d: d.num_tokens, reverse=True)
        return len(new)

    def _init_candidates(self):
        """Target verifier and drafter first pass (decode_query_len = K + 1): FULL graphs of k + 1 tokens per request for
        the request counts that use k. 1Cat rounds the verifier's capture sizes up to multiples of K + 1 (24 -> 25,
        48 -> 50) and stops at 16 requests, so without these an 8-request 3-token verify ran PIECEWISE (3x the step).
        Drafter decode passes (decode_query_len = 1): FULL graphs up to the server's request limit."""
        sc = self.vllm_config.speculative_config
        mtp = sc is not None and sc.method == "mtp"
        K = sc.num_speculative_tokens if mtp else 0
        rng = ranges(cfg, K, self.max_num_reqs) if mtp else []
        verifier = mtp and self.decode_query_len == K + 1
        if verifier:
            for k, _, _ in rng:
                if 1 < k + 1 < self.decode_query_len and (k + 1) not in self.decode_query_lens:
                    self.decode_query_lens = (*self.decode_query_lens, k + 1)
        orig(self)
        mode = self.cudagraph_mode.decode_mode() if self.cudagraph_mode else None
        if not (mtp and mode == CUDAGraphMode.FULL and self.cudagraph_mode.separate_routine() and self._candidates):
            return
        if verifier:
            for k, lo, hi in rng:
                if k == 0 and type(self).__name__ == "PrefillEagleCudaGraphManager":
                    continue                                    # the drafter does not run on draft-free steps
                n = add(self, mode, k + 1, capture_counts(lo, hi))   # k = 0: 1-token (draft-free) target steps
                if n:
                    log.warning("q27fuse: %s adds %d FULL %d-token decode graphs (%d..%d requests)",
                                type(self).__name__, n, k + 1, lo, hi)
        elif self.decode_query_len == 1:
            n = add(self, mode, 1, capture_counts(1, self.max_num_reqs))
            if n:
                log.warning("q27fuse: %s adds %d FULL draft decode graphs (up to %d requests)",
                            type(self).__name__, n, self.max_num_reqs)

    CU.CudaGraphManager._init_candidates = _init_candidates
    CU.CudaGraphManager._q27_spec_k = True
    if os.environ.get("Q27_SPEC_K_DEBUG") == "1":
        # which graph each manager hands out, per (manager, request count, uniform query length): the first time each
        # combination occurs and a running count every 2000 dispatches
        import collections
        seen = collections.Counter()
        orig_dispatch = CU.CudaGraphManager.dispatch

        def dispatch(self, num_reqs, num_tokens, uniform_token_count):
            d = orig_dispatch(self, num_reqs, num_tokens, uniform_token_count)
            key = (type(self).__name__, num_reqs, num_tokens, uniform_token_count, str(d.cg_mode), d.num_tokens,
                   d.num_reqs, d.uniform_token_count)
            seen[key] += 1
            if seen[key] == 1:
                log.warning("q27 spec-k dispatch: %s reqs=%d tokens=%d uniform=%s -> %s tokens=%d reqs=%s uniform=%s", *key)
            if sum(seen.values()) % 2000 == 0:
                log.warning("q27 spec-k dispatch counts: %s", seen.most_common(8))
            return d

        CU.CudaGraphManager.dispatch = dispatch
