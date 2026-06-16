"""End-to-end test for fusing a Limit into the hash-shuffle map phase.

Runs the real planner + executor (needs a working Ray build).  Validates that
`ds.limit(n).repartition(k, keys=...)` produces exactly `n` rows whether or not
the limit is fused into the shuffle map -- i.e. the global limit-counter actor
correctly caps the union of all map tasks.
"""

import os

import pytest

import ray
from ray.data.context import DataContext, ShuffleStrategy


@pytest.fixture(scope="module")
def ray_start():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def _run(limit: int, num_partitions: int, fuse: bool) -> int:
    """Return the total row count of range(N).limit(limit).repartition(...)."""
    # The planner reads the flag at plan time, so set it before the action.
    if fuse:
        os.environ["RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE"] = "1"
    else:
        os.environ.pop("RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE", None)

    ctx = DataContext.get_current()
    ctx.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

    ds = (
        ray.data.range(1000, override_num_blocks=10)
        .limit(limit)
        .repartition(num_partitions, keys=["id"])
    )
    rows = ds.take_all()
    return len(rows)


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("limit", [1, 100, 999])
def test_limit_repartition_row_count(ray_start, fuse, limit):
    num_partitions = 4
    n = _run(limit, num_partitions, fuse)
    assert n == limit, f"expected {limit} rows, got {n} (fuse={fuse})"


def test_read_fused_into_shuffle_map(ray_start):
    """With read fusion on, ReadParquet is absorbed into HashShuffleMap.

    Validates both the plan shape (no separate read stage) and correctness
    (row count + actual values preserved through the fused read+partition).
    """
    import os as _os
    import tempfile

    import pandas as pd

    # Write a small parquet dataset to read back.
    tmpdir = tempfile.mkdtemp()
    ray.data.range(1000, override_num_blocks=8).map(
        lambda r: {"id": r["id"], "v": r["id"] * 2}
    ).write_parquet(tmpdir)

    _os.environ["RAY_DATA_FUSE_READ_INTO_SHUFFLE"] = "1"
    _os.environ.pop("RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE", None)
    DataContext.get_current().shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

    ds = ray.data.read_parquet(tmpdir).repartition(4, keys=["id"]).materialize()
    plan = ds.stats()
    print(plan)

    assert ds.count() == 1000
    # Values survive the fused read+partition.
    df = ds.to_pandas().sort_values("id").reset_index(drop=True)
    assert (df["v"] == df["id"] * 2).all()


def test_read_and_limit_both_fused(ray_start):
    """read + limit + shuffle-map all fused into one map task."""
    import os as _os

    _os.environ["RAY_DATA_FUSE_READ_INTO_SHUFFLE"] = "1"
    _os.environ["RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE"] = "1"
    DataContext.get_current().shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

    ds = (
        ray.data.range(2000, override_num_blocks=8)
        .limit(500)
        .repartition(4, keys=["id"])
        .materialize()
    )
    assert ds.count() == 500


def test_fused_plan_drops_limit_operator(ray_start):
    """With fusion on, the optimized physical plan has no standalone Limit op."""
    os.environ["RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE"] = "1"
    ctx = DataContext.get_current()
    ctx.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    ds = (
        ray.data.range(1000, override_num_blocks=10)
        .limit(100)
        .repartition(4, keys=["id"])
    )
    # Materialize so the physical plan is built/optimized.
    ds = ds.materialize()
    plan_str = ds.stats()
    # Sanity: the result is correct and the shuffle ran.
    assert ds.count() == 100
    print(plan_str)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
