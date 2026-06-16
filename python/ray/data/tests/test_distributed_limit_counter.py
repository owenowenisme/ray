"""Unit tests for the distributed-limit counter state machine.

These exercise ``_LimitCounterImpl`` directly (no Ray cluster), mirroring how
Daft tests its limit counter -- the budget logic is the part that must be
provably correct, and it has no Ray dependency.
"""

import random

import pytest

from ray.data._internal.execution.operators.shuffle_operators.distributed_limit_counter import (  # noqa: E501
    _LimitCounterImpl,
)


def test_single_claim_under_limit():
    counter = _LimitCounterImpl(limit=10)
    assert counter.claim(task_id=0, num_rows=4) == (0, 4, False)
    assert counter.remaining_take == 6


def test_claim_exactly_hits_limit():
    counter = _LimitCounterImpl(limit=10)
    assert counter.claim(task_id=0, num_rows=10) == (0, 10, True)
    assert counter.is_done()


def test_overclaim_is_capped():
    counter = _LimitCounterImpl(limit=5)
    # Asking for more than the budget only grants what remains.
    assert counter.claim(task_id=0, num_rows=100) == (0, 5, True)
    # A subsequent claim gets nothing.
    assert counter.claim(task_id=1, num_rows=100) == (0, 0, True)


@pytest.mark.parametrize("num_tasks", [1, 3, 17])
def test_global_cap_across_many_tasks(num_tasks):
    """The union of takes across many tasks equals min(limit, rows offered)."""
    limit = 50
    counter = _LimitCounterImpl(limit=limit)
    rng = random.Random(0xC0FFEE)

    total_offered = 0
    total_taken = 0
    for task_id in range(num_tasks):
        # Each "task" offers a random block size.
        offered = rng.randint(1, 30)
        total_offered += offered
        _skip, take, _done = counter.claim(task_id, num_rows=offered)
        total_taken += take

    # Never over-take, and take exactly what's available up to the cap.
    assert total_taken == min(limit, total_offered)
    assert counter.is_done() == (total_offered >= limit)


def test_offset_is_skipped_globally():
    counter = _LimitCounterImpl(limit=4, offset=3)
    # First task offers 5 rows: 3 skipped (offset), 2 taken.
    assert counter.claim(task_id=0, num_rows=5) == (3, 2, False)
    # Second task: offset exhausted, takes the remaining 2.
    assert counter.claim(task_id=1, num_rows=10) == (0, 2, True)


def test_offset_spans_multiple_tasks():
    counter = _LimitCounterImpl(limit=2, offset=5)
    # Offset larger than the first task's rows: all skipped, nothing taken.
    assert counter.claim(task_id=0, num_rows=3) == (3, 0, False)
    assert counter.remaining_skip == 2
    # Next task: 2 more skipped, then 1 taken.
    assert counter.claim(task_id=1, num_rows=3) == (2, 1, False)


def test_retry_refunds_prior_claim():
    """A retried task's predecessor claims are refunded, no double-counting."""
    counter = _LimitCounterImpl(limit=10)

    # Task 0's first attempt claims 6 then "crashes".
    counter.start_task(0)
    assert counter.claim(0, num_rows=6) == (0, 6, False)
    assert counter.remaining_take == 4

    # Task 0 is retried: start_task refunds the 6 before it re-claims.
    counter.start_task(0)
    assert counter.remaining_take == 10
    assert counter.claim(0, num_rows=6) == (0, 6, False)
    assert counter.remaining_take == 4

    # A different task can still claim the genuine remainder.
    assert counter.claim(1, num_rows=10) == (0, 4, True)


def test_zero_limit_takes_nothing():
    counter = _LimitCounterImpl(limit=0)
    assert counter.is_done()
    assert counter.claim(task_id=0, num_rows=100) == (0, 0, True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
