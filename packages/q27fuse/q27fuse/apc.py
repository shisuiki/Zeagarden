"""q27fuse.apc - Hybrid Mamba/Attention partial prefix-cache fixes for Volta sm_70.

Eliminates the multi-turn agentic S>=12 KV eviction cliff on 4x V100-SXM2-16GB (405-450 unified blocks):
  1. Disables per-request Mamba decode-checkpoint double-buffering (Q27_DECODE_CKPT=0 by default):
       1Cat-vLLM's _enable_decode_ckpt reserves 2 extra Mamba blocks per group per active decoding
       request (+6 blocks/req = 96 blocks = 23% of the entire 422-block KV pool at S=16) just to
       cache a few hundred generated tokens (~80 ms prefill), causing 18K-30K multi-turn contexts
       to thrash and evict (>12,000 ms re-prefill penalty). Turning it off frees 72-96 blocks at
       S=12..16 and automatically disables coalesce_gen_tail so the exact prompt-end boundary
       (~97% of the next turn's prompt) is snapshotted cleanly at zero active-decode block cost.
  2. Pure-LRU PartialPrefixCache.evict_one():
       1Cat-vLLM's default evict_one() evicts hit_count==0 entries before hit_count>=1 entries.
       In multi-turn serving, completed sessions from earlier traffic already got hit (hit_count>=1)
       and stay pinned indefinitely while brand-new Turn-1 entries (hit_count==0) are evicted before
       their Turn 2 arrives. Pure LRU retires dead sessions while lookup()'s move_to_end() still
       protects shared system/prompt prefixes.
  3. Deduplicated prompt-tail snapshots + consume-on-hit for single-session turn tails:
       In partial_prefix_stops(), retains shared prefix boundaries (<=2048 tokens or first message
       boundary) plus ONLY the final boundary of each prompt, skipping redundant interior turn
       boundaries that otherwise leak +4 blocks/turn. When the next turn hits that single-session
       turn-tail snapshot in _redirect_partial_hit(), removes it from PartialPrefixCache once CoW
       private copies and _defer_release (3-step async guard) are armed, preventing +4 blocks/session
       accumulation and MRU domino eviction of other sessions' unconsumed turns.
  4. Sized PartialPrefixCache cap (Q27_PARTIAL_CACHE_FRAC=0.70) + demand-driven reclamation:
       Raises max_blocks from 40% (~162-168 blocks, < 12 concurrent 19K sessions) to 70% (~283-295
       blocks) so S=12..16 simultaneous Turn-1->2 handoffs fit without self-eviction, and reclaims
       entries only when actual allocation demand requires free blocks (no speculative soft reserve).

Opt-out: Q27_APC_FIX=0.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("q27fuse.apc")


def enabled() -> bool:
    return (os.environ.get("Q27_APC_FIX", "1") or "1").strip() != "0"


def _partial_frac() -> float:
    raw = os.environ.get("Q27_PARTIAL_CACHE_FRAC", "0.70").strip()
    try:
        return max(0.2, min(0.9, float(raw)))
    except ValueError:
        return 0.70


def _keep_decode_ckpt() -> bool:
    return (os.environ.get("Q27_DECODE_CKPT", "0") or "0").strip() != "0"


def apply() -> None:
    if not enabled():
        return

    # 1. Pure-LRU PartialPrefixCache eviction (fixes dead-session zombie pinning at S>=12)
    try:
        from vllm.v1.core import partial_prefix_cache as PPC

        cls = PPC.PartialPrefixCache
        if not getattr(cls, "_q27_apc_patched", False):

            def _lru_evict_one(self, protected=None) -> bool:
                if not self.entries:
                    return False
                for key, entry in self.entries.items():
                    if entry is not protected:
                        self._remove(key)
                        return True
                return False

            cls.evict_one = _lru_evict_one
            cls._q27_apc_patched = True
    except Exception as e:  # noqa: BLE001
        log.warning("q27fuse.apc: failed to patch PartialPrefixCache (%r)", e)

    # 2. Pool-efficient PartialPrefixCache lifecycle in KVCacheManager
    try:
        from vllm.v1.core import kv_cache_manager as KCM

        M = KCM.KVCacheManager
        if not getattr(M, "_q27_apc_patched", False):
            orig_init_partial = M._init_partial_prefix_cache
            orig_stops = M.partial_prefix_stops
            orig_mat_snaps = M._materialize_snapshots
            orig_mat_decode = M._materialize_decode_snapshots
            orig_redirect = M._redirect_partial_hit

            def _init_partial_prefix_cache(self, hash_block_size: int) -> None:
                orig_init_partial(self, hash_block_size)
                if self.partial_cache is not None:
                    frac = _partial_frac()
                    managers = self.coordinator.single_type_managers
                    num_gpu = int(self.block_pool.num_gpu_blocks)
                    state_blocks = getattr(self.block_pool, "_device_state_blocks", 0) or 0
                    new_max = int(num_gpu * frac)
                    if state_blocks > 0:
                        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

                        mamba_groups = max(1, sum(1 for m in managers if isinstance(m, MambaManager)))
                        max_state_entries = max(2, int(state_blocks * frac) // mamba_groups)
                        new_max = min(new_max, max_state_entries * len(managers))
                    new_max = max(4 * len(managers), new_max)
                    old_max = self.partial_cache.max_blocks
                    self.partial_cache.max_blocks = new_max
                    old_dckpt = bool(getattr(self, "_enable_decode_ckpt", False))
                    if old_dckpt and not _keep_decode_ckpt():
                        self._enable_decode_ckpt = False
                    log.warning(
                        "q27fuse.apc: pure-LRU + turn-tail consume-on-hit PartialPrefixCache active "
                        "(max_blocks %d -> %d / %d, frac=%.2f, decode_ckpt %s -> %s)",
                        old_max,
                        new_max,
                        num_gpu,
                        frac,
                        old_dckpt,
                        bool(getattr(self, "_enable_decode_ckpt", False)),
                    )

            def _partial_prefix_stops(self, request, start: int, end: int) -> tuple[int, ...]:
                res = orig_stops(self, request, start, end)
                if not res or not request.prefix_cache_boundaries:
                    return res
                # Keep short shared prefixes (<=2048), the first boundary (system prompt, <=8192),
                # and ONLY the final turn-end boundary of the prompt (skip stale interior turns).
                first_b = min(request.prefix_cache_boundaries)
                last_b = max(request.prefix_cache_boundaries)
                return tuple(
                    b for b in res
                    if b <= 2048 or (b == first_b and b <= 8192) or b == last_b
                )

            def _materialize_snapshots(self) -> None:
                cache = self.partial_cache
                if cache is None or not self._pending_snapshots:
                    return orig_mat_snaps(self)
                tail_keys = set()
                for req, b, num_preemptions in self._pending_snapshots:
                    if (
                        req.num_preemptions == num_preemptions
                        and not req.is_finished()
                        and b > 2048
                        and req.prefix_cache_boundaries
                        and b == max(req.prefix_cache_boundaries)
                    ):
                        tail_keys.add(self._partial_key(req, b))
                orig_mat_snaps(self)
                for key in tail_keys:
                    entry = cache.entries.get(key)
                    if entry is not None:
                        entry._q27_turn_tail = True

            def _materialize_decode_snapshots(self) -> None:
                cache = self.partial_cache
                if cache is None or not self._pending_decode_snapshots:
                    return orig_mat_decode(self)
                prev_keys = set(cache.entries.keys())
                orig_mat_decode(self)
                for key, entry in cache.entries.items():
                    if key not in prev_keys:
                        entry._q27_turn_tail = True

            def _redirect_partial_hit(self, request, entry) -> None:
                orig_redirect(self, request, entry)
                # orig_redirect already deferred release of copied tail blocks for 3 async steps,
                # and allocate_new_computed_blocks already holds refs on all parent FullAttn blocks.
                if (
                    self.partial_cache is not None
                    and getattr(entry, "_q27_turn_tail", False)
                ):
                    self.partial_cache.remove(entry.key)

            def _reclaim_partial_blocks(
                self,
                num_blocks: int,
                protected=None,
                demand: dict[int, int] | None = None,
            ) -> bool:
                cache = self.partial_cache

                def _satisfied() -> bool:
                    return num_blocks <= self.block_pool.get_num_free_blocks() and (
                        demand is None or self.block_pool.can_allocate_group_blocks(demand)
                    )

                if _satisfied():
                    return True
                while cache is not None and cache.evict_one(protected=protected):
                    if _satisfied():
                        return True
                return _satisfied()

            M._init_partial_prefix_cache = _init_partial_prefix_cache
            M.partial_prefix_stops = _partial_prefix_stops
            M._materialize_snapshots = _materialize_snapshots
            M._materialize_decode_snapshots = _materialize_decode_snapshots
            M._redirect_partial_hit = _redirect_partial_hit
            M._reclaim_partial_blocks = _reclaim_partial_blocks
            M._q27_apc_patched = True
    except Exception as e:  # noqa: BLE001
        log.warning("q27fuse.apc: failed to patch KVCacheManager (%r)", e)
