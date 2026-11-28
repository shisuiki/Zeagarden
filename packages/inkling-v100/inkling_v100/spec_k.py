# SPDX-License-Identifier: Apache-2.0
"""Adaptive speculative verify-width K(C) controller for Inkling-2-Mini (DSpark & DFly2-KV) on 1Cat-vLLM v1.5.4.

Configured via `INKLING_SPEC_K="K1@n1,K2@n2,..."` (same syntax as `q27fuse.spec_k`), e.g.:
  - Static sweep:  `INKLING_SPEC_K="3@1"` (always verify K=3 draft tokens)
  - Adaptive K(C): `INKLING_SPEC_K="7@1,4@7,3@13,2@25"` (shrink K as concurrency rises)

Zero edits to 1Cat-vLLM site-packages:
  1. AsyncScheduler._update_after_schedule: sets `_spec_token_placeholders = (-1,) * k`
     for the step's active request count `num_reqs`.
  2. Target CudaGraphManager._init_candidates: adds FULL uniform-decode graphs for
     `q = k + 1` tokens/req over every active `(k, lo, hi)` range.
  3. Inkling2DSparkSpeculator: runs only `k_eff` Markov steps & `k_eff` LM-head base-logit
     rows per request inside FULL draft graphs (skipping drafting completely when `k == 0`)
     and reports `k` via `next_num_draft_tokens()`.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

log = init_logger("vllm.inkling_v100.spec_k")


def spec_k_config() -> list[tuple[int, int]] | None:
    """Return sorted `[(min_reqs, k)]` from `INKLING_SPEC_K`, or None if unset."""
    raw = os.environ.get("INKLING_SPEC_K", "").strip()
    if not raw or raw == "0":
        return None
    ent: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k_s, n_s = part.split("@")
        ent.append((int(n_s), int(k_s)))
    ent.sort()
    assert ent and all(n >= 1 and k >= -1 for n, k in ent), (
        f"INKLING_SPEC_K={raw!r}: expected 'K@n,...' with K >= 0, n >= 1"
    )
    return ent


def k_for(num_reqs: int, max_k: int, cfg: list[tuple[int, int]] | None) -> int:
    if cfg is None:
        return max_k
    k = -1
    for n, kk in cfg:
        if num_reqs >= n:
            k = kk
    return max_k if k == -1 else min(k, max_k)


def k_ranges(
    cfg: list[tuple[int, int]], max_k: int, max_reqs: int
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    lo0 = cfg[0][0]
    if lo0 > 1:
        out.append((max_k, 1, min(lo0 - 1, max_reqs)))
    for i, (n, k) in enumerate(cfg):
        hi = cfg[i + 1][0] - 1 if i + 1 < len(cfg) else max_reqs
        if n <= min(hi, max_reqs):
            out.append((max_k if k == -1 else min(k, max_k), n, min(hi, max_reqs)))
    return out


def capture_counts(lo: int, hi: int) -> list[int]:
    """Capture exact powers/multiples covering 1..max_reqs without OOMing graph pool."""
    base = {1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32}
    return sorted({b for b in range(lo, hi + 1) if b in base or b % 8 == 0} | {lo, hi})


def apply_spec_k() -> None:
    cfg = spec_k_config()
    if cfg is None:
        return
    _patch_scheduler(cfg)
    _patch_target_graphs(cfg)
    log.info(
        "inkling_v100: adaptive spec K(C) enabled: %s",
        ", ".join(f"{'K' if k == -1 else k}@{n}" for n, k in cfg),
    )


def _patch_scheduler(cfg: list[tuple[int, int]]) -> None:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if getattr(AsyncScheduler, "_inkling_spec_k", False):
        return
    orig = AsyncScheduler._update_after_schedule

    def _update_after_schedule(self, scheduler_output):
        max_k = self.num_spec_tokens
        if max_k >= 1:
            k = k_for(len(scheduler_output.num_scheduled_tokens), max_k, cfg)
            if len(self._spec_token_placeholders) != k:
                self._spec_token_placeholders = (-1,) * k
        orig(self, scheduler_output)

    AsyncScheduler._update_after_schedule = _update_after_schedule
    AsyncScheduler._inkling_spec_k = True  # type: ignore[attr-defined]


def _patch_target_graphs(cfg: list[tuple[int, int]]) -> None:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as CU

    if getattr(CU.CudaGraphManager, "_inkling_spec_k", False):
        return
    orig = CU.CudaGraphManager._init_candidates

    def _add_uniform(self, mode, q: int, reqs: list[int]) -> int:
        descs = self._capture_descs.setdefault(mode, [])
        max_num_tokens = int(
            getattr(
                self,
                "max_num_tokens",
                self.vllm_config.scheduler_config.max_num_batched_tokens,
            )
        )
        have = {
            d.num_tokens
            for d in descs
            if d.uniform_token_count == q and d.attention_context_bucket is None
        }
        new = [
            CU.BatchExecutionDescriptor(
                cg_mode=mode, num_tokens=q * b, num_reqs=b, uniform_token_count=q
            )
            for b in reqs
            if q * b not in have and q * b <= max_num_tokens
        ]
        if not new:
            return 0
        top = max(d.num_tokens for d in new)
        if len(self._candidates) <= top:
            self._candidates.extend([] for _ in range(top + 1 - len(self._candidates)))
        sizes = sorted(have | {d.num_tokens for d in new})
        for d in new:
            lo = max([t for t in sizes if t < d.num_tokens], default=0)
            for i in range(lo + 1, d.num_tokens + 1):
                self._candidates[i] = [d] + [
                    c
                    for c in self._candidates[i]
                    if not (c.uniform_token_count == q and c.num_tokens > d.num_tokens)
                ]
        descs.extend(new)
        descs.sort(key=lambda d: d.num_tokens, reverse=True)
        return len(new)

    def _init_candidates(self):
        # Only patch the target model's ModelCudaGraphManager (not DFlashCudaGraphManager).
        sc = self.vllm_config.speculative_config
        is_target_dspark = (
            isinstance(self, CU.ModelCudaGraphManager)
            and sc is not None
            and sc.method in ("dflash", "draft_model")
        )
        max_k = sc.num_speculative_tokens if is_target_dspark else 0
        rng = k_ranges(cfg, max_k, self.max_num_reqs) if is_target_dspark else []
        if is_target_dspark:
            for k, _, _ in rng:
                q = k + 1
                if 1 <= q <= self.decode_query_len and q not in self.decode_query_lens:
                    self.decode_query_lens = (*self.decode_query_lens, q)
        orig(self)
        mode = self.cudagraph_mode.decode_mode() if self.cudagraph_mode else None
        if not (
            is_target_dspark
            and mode == CUDAGraphMode.FULL
            and self.cudagraph_mode.separate_routine()
            and self._candidates
        ):
            return
        for k, lo, hi in rng:
            added = _add_uniform(self, mode, k + 1, capture_counts(lo, hi))
            if added:
                log.info(
                    "inkling_v100 spec_k: target added %d FULL q=%d decode graphs (%d..%d reqs)",
                    added,
                    k + 1,
                    lo,
                    hi,
                )

    CU.CudaGraphManager._init_candidates = _init_candidates
    CU.CudaGraphManager._inkling_spec_k = True  # type: ignore[attr-defined]

    if os.environ.get("INKLING_SPEC_K_LOG_DISPATCH", "1") != "0":
        orig_dispatch = CU.CudaGraphManager.dispatch

        def _dispatch(self, num_reqs: int, num_tokens: int, uniform_token_count: int | None):
            desc = orig_dispatch(self, num_reqs, num_tokens, uniform_token_count)
            if isinstance(self, CU.ModelCudaGraphManager) and uniform_token_count is not None:
                seen = getattr(self, "_spec_k_seen_dispatch", None)
                if seen is None:
                    seen = set()
                    self._spec_k_seen_dispatch = seen
                key = (desc.cg_mode.name, uniform_token_count, desc.num_reqs, desc.num_tokens)
                if key not in seen:
                    seen.add(key)
                    log.info(
                        "inkling_v100 spec_k dispatch: reqs=%d toks=%d q=%d -> %s (padded reqs=%s toks=%d)",
                        num_reqs,
                        num_tokens,
                        uniform_token_count,
                        desc.cg_mode.name,
                        desc.num_reqs,
                        desc.num_tokens,
                    )
            return desc

        CU.CudaGraphManager.dispatch = _dispatch
