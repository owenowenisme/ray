"""Global row-budget coordination for a Limit fused into a distributed shuffle.

When a ``Limit`` sits directly upstream of a hash-shuffle, we can avoid
materializing the limited dataset as its own stage and instead apply the limit
*inside* the shuffle-map tasks.  But each map task only sees its own blocks, so a
purely local limit can't enforce a global row cap.  This module provides the
coordination primitive: a single Ray actor that hands out slices of a shared
``remaining_take`` budget.

Design mirrors Daft's distributed-limit counter actor: one actor owns the global
``(remaining_skip, remaining_take)`` state, and every map task ``claim``s its
share before partitioning.  Because all claims funnel through one actor, the
union of rows kept across all tasks never exceeds ``limit`` -- with no locks and
no cross-task barrier.

Unlike Daft, the claim is *per block-batch* (one claim per map task) rather than
per morsel, which matches Ray Data's coarser block granularity and keeps the
actor off the hot path: one RPC per map task, amortized over a whole block.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import ray
from ray.data.block import Block, BlockAccessor


class _LimitCounterImpl:
    """``(skip, take)`` state machine for a distributed Limit.

    Kept as a plain class (no Ray dependency) so the budget logic can be unit
    tested without standing up a cluster; ``LimitCounterActor`` below wraps it
    in ``ray.remote``.
    """

    def __init__(self, limit: int, offset: int = 0) -> None:
        self.remaining_skip = offset
        self.remaining_take = limit
        # task_id -> (cumulative_skip_claimed, cumulative_take_claimed), used to
        # refund a crashed attempt's claims so a retry doesn't double-count.
        self._task_claims: Dict[int, Tuple[int, int]] = {}

    def start_task(self, task_id: int) -> None:
        """Register a map task before it claims.

        If we've seen ``task_id`` before, a prior attempt crashed mid-claim and
        is being retried -- refund its partial claims so the retry's claims
        don't double-count against the global budget.  No-op on the common
        (non-retry) path.
        """
        prior = self._task_claims.pop(task_id, None)
        if prior is not None:
            skip, take = prior
            self.remaining_skip += skip
            self.remaining_take += take

    def claim(self, task_id: int, num_rows: int) -> Tuple[int, int, bool]:
        """Claim up to ``num_rows`` rows from the global budget.

        Returns ``(skip, take, done)`` where the caller should drop the first
        ``skip`` rows (consumed by the global offset), keep the next ``take``
        rows, and ``done`` is True once the global limit is fully satisfied.
        """
        if self.remaining_take == 0:
            return (0, 0, True)

        skip = min(self.remaining_skip, num_rows)
        self.remaining_skip -= skip
        num_rows -= skip

        take = min(self.remaining_take, num_rows)
        self.remaining_take -= take

        if skip > 0 or take > 0:
            prev_skip, prev_take = self._task_claims.get(task_id, (0, 0))
            self._task_claims[task_id] = (prev_skip + skip, prev_take + take)

        return (skip, take, self.remaining_take == 0)

    def is_done(self) -> bool:
        return self.remaining_take == 0


# num_cpus=0: the actor only does bookkeeping; it must not hold a CPU slot away
# from map/reduce tasks.
LimitCounterActor = ray.remote(num_cpus=0)(_LimitCounterImpl)


def make_limit_counter_actor(limit: int, offset: int = 0) -> ray.actor.ActorHandle:
    """Create a ``LimitCounterActor`` pinned to the current (driver) node.

    Pinning keeps the per-task ``claim`` RPCs from cross-hopping nodes, the same
    mitigation Daft uses for its counter actor.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_id = ray.get_runtime_context().get_node_id()
    return LimitCounterActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
    ).remote(limit, offset)


def claim_and_truncate(
    limit_actor: ray.actor.ActorHandle,
    task_id: int,
    blocks: Tuple[Block, ...],
    num_rows: int,
) -> Tuple[List[Block], bool]:
    """Claim a global row budget for this map task and truncate its blocks.

    Drops the first ``skip`` rows (global offset) and keeps the next ``take``
    rows across ``blocks``, returning ``(truncated_blocks, done)``.  An empty
    list means this task claimed nothing (the limit was already satisfied by
    other tasks); the caller should still emit empty shards for schema.
    """
    # start_task first so a retried attempt refunds its predecessor's claims.
    ray.get(limit_actor.start_task.remote(task_id))
    skip, take, done = ray.get(limit_actor.claim.remote(task_id, num_rows))

    if take == 0:
        return [], done

    out: List[Block] = []
    remaining_skip = skip
    remaining_take = take
    for block in blocks:
        if remaining_take <= 0:
            break
        acc = BlockAccessor.for_block(block)
        n = acc.num_rows()
        if remaining_skip >= n:
            remaining_skip -= n
            continue
        start = remaining_skip
        remaining_skip = 0
        count = min(n - start, remaining_take)
        out.append(acc.slice(start, start + count, copy=False))
        remaining_take -= count
    return out, done
