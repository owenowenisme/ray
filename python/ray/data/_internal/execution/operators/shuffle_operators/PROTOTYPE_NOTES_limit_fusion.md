# Prototype: fuse `read → limit → shuffle-map` via a global limit-counter actor

Goal: let a `Limit` sitting directly upstream of a hash-shuffle be absorbed into
the shuffle's map phase, instead of running as its own stage that materializes a
limited dataset before the shuffle. Mirrors how Daft fuses
`read → limit → repartition_write` into one map-side task and coordinates the
global limit with a single counter actor.

## Why an actor

Today Ray Data's `LimitOperator` runs on the driver streaming executor, where it
naturally sees every block and gets a global row count for free — but that's
exactly why it can't fuse into a remote shuffle-map task. To push the limit into
distributed tasks, each task loses the driver's centralized view, so we recover
global coordination with one Ray actor that owns the shared
`(remaining_skip, remaining_take)` budget. Every map task `claim`s its slice
before partitioning; the union of takes can never exceed `limit`.

Granularity: **per map task** (one claim per task, over a whole block batch), not
per morsel. This keeps the actor off the hot path — Ray Data's coarse block model
is a better fit for this than Daft's per-morsel claim.

## What's wired

| Piece | File |
|---|---|
| Counter state machine + Ray actor + claim/truncate helper | `distributed_limit_counter.py` (new) |
| Per-task claim + block truncation before partitioning | `shuffle_tasks.py::_shuffle_map_task` |
| `ShuffleMapOp` accepts `limit`, lazily creates the actor, passes it to tasks, drops the input tail once done | `shuffle_map_operator.py` |
| Planner absorbs a `Limit` directly upstream of the hash shuffle | `plan_all_to_all_op.py::_plan_hash_shuffle_repartition` |
| Pure-Python unit tests for the budget logic | `tests/test_distributed_limit_counter.py` (new) |

Gated behind `RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE=1`; off by default the
`LimitOperator` stays in the plan and nothing changes.

## Validated (python-only build, macOS arm64, myenv)

- **Unit:** `_LimitCounterImpl` budget logic — 10 tests passing: global cap =
  `min(limit, rows offered)`, offset/skip spanning tasks, over-claim capping,
  retry refund via `start_task`, zero-limit.
- **E2E:** `tests/test_limit_shuffle_fusion_e2e.py` — 6 tests passing.
  `range(1000).limit(n).repartition(4, keys=["id"])` returns exactly `n` rows
  for n in {1, 100, 999}, both with and without the fusion flag.
- **Plan confirmed:** with `RAY_DATA_FUSE_LIMIT_INTO_SHUFFLE=1` the optimized
  plan is `ReadRange -> HashShuffleMap -> HashShuffleReduce` (no `LimitOperator`
  stage); with the flag off it is `ReadRange -> LimitOperator[limit=100] ->
  HashShuffleMap -> HashShuffleReduce`. Both yield 100 rows.

  Build note: the env needs a fresh nightly wheel matching current source
  (`build_address` symbol) plus `grpcio`/`aiohttp` (the runtime_env_agent
  segfaults without grpcio, taking the raylet down). `setup-dev.py` was re-run
  from this worktree, so `myenv` currently points at the worktree, not the main
  repo -- re-run it from the main repo to restore that workflow.

## Not yet done (follow-ups)

1. **Active early termination.** We currently drop the input *tail* after a
   completed task reports `done` (passive, like today's `LimitOperator`). The
   real read-side win needs telling the upstream read/map operator to stop
   launching tasks once the budget is spent (cancellation), not just discarding
   what arrives.
3. **Proper fusion-rule integration.** The planner-level absorption is a
   prototype shortcut. The clean home is `FuseOperators._can_fuse` +
   a `_get_fused_*` path, so it composes with the existing
   `MapOperator → AllToAllOperator` fusion and the `_op_map` bookkeeping.
4. **Determinism.** Which rows survive an unordered limit is nondeterministic
   across parallel claims (fine for `LIMIT`; an ordered limit must use top-N).
5. **Actor lifecycle.** No teardown wired yet; add `ray.kill` on operator
   shutdown (see `_do_shutdown`).
