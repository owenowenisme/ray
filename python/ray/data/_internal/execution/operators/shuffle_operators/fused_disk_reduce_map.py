"""PROTOTYPE: fuse a DiskHashShuffleReduceOp into its sole downstream
DiskHashShuffleMapOp.

Motivation (TPC-H q9 timeline): every join level pays two stage barriers --
reduce materializes its output, then a separate shuffle-map stage re-reads it
and re-partitions. Photon runs [join + project + next exchange sink] in one
stage. This operator collapses the pair: one task per upstream partition
fetches the partition's shards, runs the (project-fused) reduce_fn, and
immediately hash-partitions + persists the output as the next stage's shards.

Enabled via RAY_DATA_FUSE_DISK_REDUCE_MAP=1 (see
logical/rules/fuse_disk_reduce_into_map.py).
"""

import collections
import functools
import logging
import os
import re
import struct
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pyarrow as pa

import ray
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    PhysicalOperator,
    RefBundle,
    TaskContext,
)
from ray.data._internal.execution.interfaces.physical_operator import (
    MetadataOpTask,
    OpTask,
)
from ray.data._internal.execution.operators.shuffle_operators.disk_shuffle_map_operator import (  # noqa: E501
    DiskHashShuffleMapOp,
)
from ray.data._internal.execution.operators.shuffle_operators.disk_shuffle_reduce_operator import (  # noqa: E501
    DiskHashShuffleReduceOp,
)
from ray.data._internal.execution.operators.shuffle_operators.disk_shuffle_runtime import (  # noqa: E501
    _SHUFFLE_FILE_SERVER_NAMESPACE,
    ShuffleFileServer,
    ShuffleHandle,
    _build_range_index,
    _compute_prefetch_layout,
    _counts_to_array,
    _fetch_from_file_server,
    SHUFFLE_FILE_SERVERS_PER_NODE,
    _file_server_name,
    _group_by_server,
    _handles_to_sources,
    _PartitionWriter,
    _read_local_group,
    _PwriteSink,
    _read_ipc,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (  # noqa: E501
    extract_partition_id,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_reduce_operator import (  # noqa: E501
    _SHUFFLE_REDUCE_RUNTIME_ENV,
)
from ray.data.block import Block, BlockAccessor, BlockExecStats
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)


@ray.remote
def _disk_shuffle_fused_reduce_map_task(
    *handles_by_input: List[ShuffleHandle],
    partition_id: int,
    reduce_fn: Any,
    max_bytes_per_fetch: int,
    fetch_threads: int,
    map_transformer: Optional[Any],
    map_task_context: Optional["TaskContext"],
    data_context: Optional["DataContext"],
    emit_empty_partition: bool,
    out_partition_fn: Any,
    out_num_partitions: int,
    out_dir: str,
    out_shuffle_id: str,
    out_compression: Optional[str],
    out_block_transformer: Optional[Any],
) -> ShuffleHandle:
    """One task = reduce partition ``partition_id`` + next-stage map.

    Reduce half mirrors ``_disk_shuffle_reduce_task`` (Flight fetch into a
    staging file, decode, reduce_fn, optional fused map transformer) but emits
    plain blocks instead of the streaming-generator protocol. Map half mirrors
    ``_disk_shuffle_map_task`` (partition + persist + sealed ShuffleHandle),
    with ``map_id == partition_id``.
    """
    _maybe_install_polars_partition()
    exec_stats_builder = BlockExecStats.builder()
    start_time_s = time.perf_counter()
    start_thread_s = time.thread_time()
    start_proc_s = time.process_time()

    # EXPERIMENT (RAY_DATA_REDUCE_PROF=1): the trace only sees a task's start
    # and end, so a stage that runs at 50% CPU looks "blocked" with no way to
    # say on what. Accumulate per-phase wall inside the task and print one line
    # at the end; `pull - fetch - decode` is the reduce_fn's own time.
    _prof = collections.defaultdict(float)
    _local_bytes = 0
    # Anyscale's job-log API truncates at ~1150 lines, so one line per task
    # (200 per stage x 5 stages) loses almost all of them. Sample instead:
    # partitions are hash-assigned, so every Nth is representative and
    # percentages are what we are after, not totals.
    _prof_every = int(os.environ.get("RAY_DATA_REDUCE_PROF_EVERY", "1"))
    _prof_on = (
        os.environ.get("RAY_DATA_REDUCE_PROF") == "1"
        and partition_id % _prof_every == 0
    )
    if os.environ.get("RAY_DATA_FLIGHT_PROF") == "1":
        from ray.data._internal.execution.operators.shuffle_operators import (
            disk_shuffle_runtime as _dsr,
        )

        _dsr.flight_prof_take()  # clear anything left by a previous task
    if os.environ.get("RAY_DATA_JOIN_PROF") == "1":
        from ray.data._internal.execution.operators import join as _join_mod

        _join_mod.join_prof_take()  # same: workers are reused across tasks

    # EXPERIMENT (RAY_DATA_REDUCE_SPY=<partition_id>): sample THIS process for
    # the length of this task. Attaching from outside does not work: Ray leaves
    # a finished task's name on the worker's process title, so a PID matched by
    # title is usually running something else by the time the profiler starts.
    _spy_proc = _spy_out = None
    if os.environ.get("RAY_DATA_REDUCE_SPY") == str(partition_id):
        import shutil
        import subprocess

        _exe = shutil.which("py-spy")
        if _exe:
            _spy_out = f"/tmp/spy_reduce_p{partition_id}.folded"
            _spy_proc = subprocess.Popen(
                ["sudo", "-n", _exe, "record", "--pid", str(os.getpid()),
                 "--duration", os.environ.get("RAY_DATA_REDUCE_SPY_SECONDS", "12"),
                 "--format", "raw", "--output", _spy_out,
                 "--nonblocking", "--rate", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

    # Every timer below records THREE clocks for the same region, because the
    # question changed. While the stage was thought to be waiting on the
    # network, wall was the right unit. The shuffle window is now known to be
    # CPU-saturated (FINDINGS section 19), and in that regime wall says how
    # long a step blocked, not what it cost:
    #
    #   wall   perf_counter   elapsed, including every kind of waiting
    #   cpu    thread_time    CPU burned by THIS thread only
    #   proc   process_time   CPU burned by ALL threads in the process
    #
    # proc - cpu is the helper-thread work a step triggers: polars keeps its
    # own rayon pool and Arrow its own, and neither shows up in thread_time.
    # It is contaminated by the fetch threads running concurrently, so it is an
    # upper bound per step; the fetch threads are measured separately below so
    # the contamination can be bounded rather than guessed at.
    def _clocks():
        return time.perf_counter(), time.thread_time(), time.process_time()

    def _charge(key, t0):
        w, c, p = t0
        _prof[key] += time.perf_counter() - w
        _prof[key + "@cpu"] += time.thread_time() - c
        _prof[key + "@proc"] += time.process_time() - p

    def _timed(key):
        class _T:
            def __enter__(self_):
                self_.t = _clocks()

            def __exit__(self_, *e):
                _charge(key, self_.t)

        return _T()

    def _timed_iter(it, key):
        """Charge the time a consumer spends waiting on ``it`` to ``key``."""
        while True:
            t = _clocks()
            try:
                v = next(it)
            except StopIteration:
                _charge(key, t)
                return
            _charge(key, t)
            yield v

    # Needed before the fetch so node-local groups can skip Flight; the map
    # half re-reads it later for the file-server identity.
    _self_node_id = ray.get_runtime_context().get_node_id()
    _local_read = os.environ.get("RAY_DATA_SHUFFLE_LOCAL_READ") == "1"

    num_inputs = len(handles_by_input)
    sources_by_input: List[List[Any]] = []
    schemas_by_input: List[Optional[pa.Schema]] = []
    for handles in handles_by_input:
        sources, schema = _handles_to_sources(handles, partition_id)
        sources_by_input.append(sources)
        schemas_by_input.append(schema)
    reduce_schema: Optional[pa.Schema] = next(
        (s for s in schemas_by_input if s is not None), None
    )

    def _with_typed_empty_inputs(tables_by_input):
        if num_inputs == 1:
            return tables_by_input
        out = []
        for tables, schema in zip(tables_by_input, schemas_by_input):
            if not tables and schema is not None:
                tables = [schema.empty_table()]
            out.append(tables)
        return out

    def _reduce_blocks():
        """Yield this partition's raw reduce output blocks."""
        nonlocal _local_bytes
        if not any(sources_by_input):
            if num_inputs == 1:
                if emit_empty_partition and reduce_schema is not None:
                    yield reduce_schema.empty_table()
                return
            yield from reduce_fn(
                partition_id,
                _with_typed_empty_inputs([[] for _ in range(num_inputs)]),
            )
            return

        shuffle_id = next(s for srcs in sources_by_input for s in srcs).shuffle_id
        staging_dir = os.path.join(
            tempfile.gettempdir(), f"ray_shuffle_disk_{shuffle_id}_reduce"
        )
        os.makedirs(staging_dir, exist_ok=True)
        prefetch_file = os.path.join(staging_dir, f"fusedreduce_p{partition_id}.bin")

        groups_by_input = [_group_by_server(srcs) for srcs in sources_by_input]
        flat_groups = [group for groups in groups_by_input for group in groups]
        group_input_idx = [
            i for i, groups in enumerate(groups_by_input) for _ in groups
        ]
        total_size, base_offsets, node_sizes = _compute_prefetch_layout(flat_groups)
        accum: List[List[pa.Table]] = [[] for _ in range(num_inputs)]
        _compression = (
            data_context if data_context is not None else DataContext.get_current()
        ).shuffle_compression

        try:
            fd = os.open(prefetch_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                if total_size > 0:
                    try:
                        os.posix_fallocate(fd, 0, total_size)
                    except (AttributeError, OSError):
                        os.ftruncate(fd, total_size)

                def _decode_region(base, size, into, schema=None):
                    with _timed("decode"):
                        pos, end = base, base + size
                        while pos < end:
                            length = struct.unpack(">Q", os.pread(fd, 8, pos))[0]
                            into.append(
                                _read_ipc(
                                    os.pread(fd, length, pos + 8),
                                    _compression,
                                    schema,
                                )
                            )
                            pos += 8 + length

                n_threads = min(len(flat_groups), max(1, fetch_threads)) or 1
                work = list(zip(base_offsets, node_sizes, flat_groups, group_input_idx))

                # EXPERIMENT (RAY_DATA_SHUFFLE_PROBE_BATCH_BYTES, 0 = off):
                # decoding every shard of both inputs before calling reduce_fn
                # makes peak memory O(whole partition), which is what forces the
                # 3x sizing and caps this stage's concurrency. A two-input join
                # does not need that: materialise only the smaller (build) side,
                # then feed the larger (probe) side in bounded slices, calling
                # reduce_fn once per slice. Locally this took peak from 2.55x
                # the input to 1.33x at a 128MiB slice, with no extra CPU.
                #
                # ONLY SOUND FOR INNER / LEFT-side joins: a right-outer or
                # right-anti reduce_fn would re-emit unmatched build rows once
                # per slice. Single-input reducers (aggregations) need every row
                # of a key at once and are never streamed.
                probe_batch_bytes = int(
                    os.environ.get("RAY_DATA_SHUFFLE_PROBE_BATCH_BYTES", "0")
                )
                bytes_by_input = [0] * num_inputs
                for _b, size, _g, input_idx in work:
                    bytes_by_input[input_idx] += size
                stream = (
                    probe_batch_bytes > 0
                    and num_inputs == 2
                    and all(bytes_by_input)
                )
                if stream:
                    build_idx = min(range(num_inputs), key=lambda i: bytes_by_input[i])
                    probe_idx = 1 - build_idx

                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    def _submit(group, base):
                        if _local_read and group.node_id == _self_node_id:
                            return ex.submit(
                                _read_local_group, _PwriteSink(fd, base), group
                            )
                        return ex.submit(
                            _fetch_from_file_server,
                            _PwriteSink(fd, base),
                            group.shuffle_id,
                            group.node_id,
                            group.members,
                            max_bytes_per_fetch,
                            # Spread reducers across the node's replicas; every
                            # replica serves the same directory, so any of them
                            # can answer and no file has to move.
                            partition_id,
                        )

                    futs = {
                        _submit(group, base): (base, size, input_idx)
                        for base, size, group, input_idx in work
                    }
                    _local_bytes = sum(
                        size
                        for _b, size, group, _i in work
                        if _local_read and group.node_id == _self_node_id
                    )
                    if not stream:
                        # as_completed() BLOCKS in the for statement until a
                        # future finishes; fut.result() then returns instantly.
                        # Timing only result() reported fetch=0 and pushed the
                        # real wait into reduce_fn via reduce_pull. Drive the
                        # iterator by hand so the wait is charged to "fetch".
                        _ac = as_completed(futs)
                        while True:
                            with _timed("fetch"):
                                try:
                                    fut = next(_ac)
                                except StopIteration:
                                    break
                            base, size, input_idx = futs[fut]
                            if size > 0:
                                _decode_region(
                                    base,
                                    size,
                                    accum[input_idx],
                                    schemas_by_input[input_idx],
                                )
                    else:
                        # Every fetch is already in flight; only the decode is
                        # ordered, so the build side is complete before the
                        # first probe slice is joined.
                        build_futs = {
                            f: v for f, v in futs.items() if v[2] == build_idx
                        }
                        probe_futs = {
                            f: v for f, v in futs.items() if v[2] == probe_idx
                        }
                        _acb = as_completed(build_futs)
                        while True:
                            with _timed("fetch"):
                                try:
                                    fut = next(_acb)
                                except StopIteration:
                                    break
                            base, size, _ = build_futs[fut]
                            if size > 0:
                                _decode_region(
                                    base,
                                    size,
                                    accum[build_idx],
                                    schemas_by_input[build_idx],
                                )

                        slice_tables: List[pa.Table] = []
                        slice_bytes = 0
                        _acp = as_completed(probe_futs)
                        while True:
                            with _timed("fetch"):
                                try:
                                    fut = next(_acp)
                                except StopIteration:
                                    break
                            base, size, _ = probe_futs[fut]
                            if size <= 0:
                                continue
                            _decode_region(
                                base, size, slice_tables, schemas_by_input[probe_idx]
                            )
                            slice_bytes += size
                            if slice_bytes < probe_batch_bytes:
                                continue
                            accum[probe_idx] = slice_tables
                            yield from reduce_fn(
                                partition_id, _with_typed_empty_inputs(accum)
                            )
                            slice_tables, slice_bytes = [], 0
                        if slice_tables:
                            accum[probe_idx] = slice_tables
                            yield from reduce_fn(
                                partition_id, _with_typed_empty_inputs(accum)
                            )
                        return
                yield from reduce_fn(partition_id, _with_typed_empty_inputs(accum))
            finally:
                os.close(fd)
        finally:
            try:
                os.unlink(prefetch_file)
            except OSError:
                pass

    def _post_reduce_blocks():
        """Reduce output -> optional fused map transformer (e.g. Project)."""
        # Inclusive timer one level below "transform": a next() on this
        # charges fetch + decode + reduce_fn, so subtracting the two leaf
        # timers leaves reduce_fn alone, and transform - reduce_pull leaves
        # the fused Project's own machinery. Timing reduce_fn directly would
        # mean wrapping four separate `yield from reduce_fn(...)` sites.
        blocks = _timed_iter(_reduce_blocks(), "reduce_pull")
        if map_transformer is None:
            yield from blocks
            return
        assert map_task_context is not None and data_context is not None
        with DataContext.current(data_context), TaskContext.current(map_task_context):
            from ray.data._internal.execution.operators.map_transformer import (
                TransformClock,
            )

            map_transformer.override_target_max_block_size(
                map_task_context.target_max_block_size_override
            )
            # Charge the fused Project's own machinery (batching, output
            # buffer, per-block dispatch) separately from reduce_fn: with 300+
            # output blocks per partition this plumbing is a candidate for the
            # time neither the join nor the decode accounts for.
            yield from _timed_iter(
                map_transformer.apply_transform(
                    blocks, map_task_context, clock=TransformClock()
                ),
                "transform",
            )

    # ---- map half: partition + persist, mirroring _disk_shuffle_map_task ----
    node_id = ray.get_runtime_context().get_node_id()
    ShuffleFileServer.options(
        name=_file_server_name(
            out_shuffle_id,
            node_id,
            partition_id % SHUFFLE_FILE_SERVERS_PER_NODE,
        ),
        namespace=_SHUFFLE_FILE_SERVER_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        max_restarts=-1,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        num_cpus=0,
    ).remote(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"map_{partition_id}.shf")
    tmp_path = final_path + ".tmp"
    output_schema: Optional[pa.Schema] = None
    final_size_on_close = -1

    def _as_arrow(block: Block) -> pa.Table:
        if not isinstance(block, pa.Table):
            return BlockAccessor.for_block(block).to_arrow()
        return block

    try:
        with open(tmp_path, "wb") as out_file:
            writer = _PartitionWriter(out_file, partition_id, out_compression)
            # "pull" charges everything upstream of the map half: the Flight
            # fetch, the decode, and reduce_fn itself. Subtracting the fetch
            # and decode timers leaves reduce_fn's own cost.
            blocks_iter = _timed_iter(_post_reduce_blocks(), "pull")
            if out_block_transformer is not None:
                arrow_blocks = [
                    _as_arrow(b)
                    for b in blocks_iter
                    if BlockAccessor.for_block(b).num_rows() > 0
                ]
                if arrow_blocks:
                    with _timed("out_block_transform"):
                        combined = transform_pyarrow.concat(
                            arrow_blocks, promote_types=True
                        )
                        blocks_iter = iter([out_block_transformer(combined)])
                else:
                    blocks_iter = iter(())
            for block in blocks_iter:
                block = _as_arrow(block)
                if output_schema is None:
                    output_schema = getattr(block, "schema", None)
                if block.num_rows == 0:
                    continue
                with _timed("partition"):
                    for out_pid, shard in out_partition_fn(block).items():
                        writer.add_shard(out_pid, shard)
            with _timed("encode_write"):
                writer.flush_all()
                out_file.flush()
            final_size_on_close = out_file.tell()
            if writer.index:
                expected = max(
                    off + length
                    for ranges in writer.index.values()
                    for off, length in ranges
                )
            else:
                expected = 0
            assert final_size_on_close == expected
        os.rename(tmp_path, final_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    if _spy_proc is not None:
        out, _ = _spy_proc.communicate(timeout=120)
        if _spy_out and os.path.exists(_spy_out):
            with open(_spy_out) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            print(f"[SPY_SELF] p={partition_id} stacks={len(lines)}", flush=True)
            for ln in lines:
                print(f"[SPY_SELF_FOLD] {ln}", flush=True)
        else:
            print(f"[SPY_SELF] p={partition_id} failed: {(out or '')[-300:]}",
                  flush=True)

    if os.environ.get("RAY_DATA_FLIGHT_PROF") == "1" and (
        partition_id % _prof_every == 0
    ):
        from ray.data._internal.execution.operators.shuffle_operators import (
            disk_shuffle_runtime as _dsr2,
        )

        _fp = _dsr2.flight_prof_take()
        print(
            f"[FLIGHT_PROF] p={partition_id} "
            f"dial={_fp.get('dial', 0):.2f} rpc={_fp.get('rpc', 0):.2f} "
            f"write={_fp.get('write', 0):.2f} "
            f"MiB={_fp.get('bytes', 0) / 2**20:.0f} "
            f"results={int(_fp.get('results', 0))} "
            f"batches={int(_fp.get('batches', 0))} "
            f"calls={int(_fp.get('calls', 0))}",
            flush=True,
        )

    if _prof_on:
        total = time.perf_counter() - start_time_s
        total_cpu = time.thread_time() - start_thread_s
        total_proc = time.process_time() - start_proc_s
        # The generator timers nest: pull > transform > reduce_pull >
        # {fetch, decode, reduce_fn}. Each next() on an outer wrapper charges
        # the whole upstream chain, so a breakdown has to SUBTRACT the inner
        # level, never sum siblings. (The previous formula summed them and
        # printed a negative reduce_fn.)
        reduce_fn_s = _prof["reduce_pull"] - _prof["fetch"] - _prof["decode"]
        project_s = _prof["transform"] - _prof["reduce_pull"]
        pull_overhead_s = _prof["pull"] - _prof["transform"]
        # "other" is everything no timer covered: task setup, the staging-file
        # fallocate, handle bookkeeping, the rename. Disjoint from pull.
        other_s = (
            total
            - _prof["pull"]
            - _prof["partition"]
            - _prof["encode_write"]
            - _prof["out_block_transform"]
        )
        _jp = {}
        if os.environ.get("RAY_DATA_JOIN_PROF") == "1":
            from ray.data._internal.execution.operators import join as _join_mod2

            _jp = _join_mod2.join_prof_take()
        # The join_* fields decompose reduce_fn for join stages; an aggregate
        # stage never enters _join_tables_iter, so they stay 0 there.
        _jp_s = " ".join(
            f"join_{k}={_jp.get(k, 0.0):.2f}"
            for k in ("combine", "setup", "prep", "to_polars", "plan", "exec",
                      "to_arrow", "post")
        )
        # Same subtraction as the wall breakdown, on the CPU clocks. Printed as
        # a separate block rather than folded into the wall fields so the two
        # units can never be read as one number: a step can be 30% of the wall
        # and 5% of the CPU, and in a CPU-bound window only the second matters.
        _rf_cpu = _prof["reduce_pull@cpu"] - _prof["fetch@cpu"] - _prof["decode@cpu"]
        _rf_proc = (_prof["reduce_pull@proc"] - _prof["fetch@proc"]
                    - _prof["decode@proc"])
        _cpu_s = (
            f"CPU[total={total_cpu:.2f}/{total_proc:.2f} "
            f"fetch={_prof['fetch@cpu']:.2f}/{_prof['fetch@proc']:.2f} "
            f"decode={_prof['decode@cpu']:.2f}/{_prof['decode@proc']:.2f} "
            f"reduce_fn={_rf_cpu:.2f}/{_rf_proc:.2f} "
            f"partition={_prof['partition@cpu']:.2f}/{_prof['partition@proc']:.2f} "
            f"encode_write={_prof['encode_write@cpu']:.2f}/"
            f"{_prof['encode_write@proc']:.2f}]"
        )
        print(
            f"[RED_PROF] p={partition_id} total={total:.2f} "
            f"fetch={_prof['fetch']:.2f} decode={_prof['decode']:.2f} "
            f"reduce_fn={reduce_fn_s:.2f} project={project_s:.2f} "
            f"pull_overhead={pull_overhead_s:.2f} "
            f"partition={_prof['partition']:.2f} "
            f"encode_write={_prof['encode_write']:.2f} "
            f"out_block_transform={_prof['out_block_transform']:.2f} "
            f"{_cpu_s} "
            f"other={other_s:.2f} local_mib={_local_bytes / 2**20:.0f} {_jp_s}",
            flush=True,
        )

    return {
        "path": os.path.realpath(final_path),
        "index_ranges": _build_range_index(writer.index, out_num_partitions),
        "shuffle_id": out_shuffle_id,
        "node_id": node_id,
        "num_partitions": out_num_partitions,
        "total_bytes": final_size_on_close,
        "compression": out_compression,
        "num_rows": _counts_to_array(writer.rows_per_partition, out_num_partitions),
        "decoded_bytes": _counts_to_array(
            writer.decoded_bytes_per_partition, out_num_partitions
        ),
        "schema": output_schema,
        "exec_stats": exec_stats_builder.build(
            block_ser_time_s=0.0, task_idx=partition_id
        ),
    }


class FusedDiskReduceMapOp(DiskHashShuffleMapOp):
    """DiskHashShuffleReduceOp fused into its downstream DiskHashShuffleMapOp.

    Downstream-facing it IS a DiskHashShuffleMapOp (same wrapper-bundle
    protocol, handle aggregation, teardown, partition-bytes accounting).
    Upstream-facing it behaves like the reduce op it absorbed: it consumes
    per-partition wrapper bundles from one or more upstream map ops, pairing
    them across inputs, and runs one fused task per partition.
    """

    def __init__(
        self,
        reduce_op: DiskHashShuffleReduceOp,
        map_op: DiskHashShuffleMapOp,
    ):
        # Deliberately bypass DiskHashShuffleMapOp.__init__ (single-input
        # assumption); initialize PhysicalOperator directly, then replicate the
        # map op's state wholesale so every inherited method operates on it.
        PhysicalOperator.__init__(
            self,
            name=f"Fused[{reduce_op.name}->{map_op.name}]",
            input_dependencies=list(reduce_op.input_dependencies),
            data_context=map_op.data_context,
        )

        # ---- state lifted from the absorbed map op ----
        self._num_partitions = map_op._num_partitions
        self._partition_fn = map_op._partition_fn
        self._block_transformer = map_op._block_transformer
        self._shuffle_map_task_num_cpus = map_op._shuffle_map_task_num_cpus
        self._map_runtime_env = map_op._map_runtime_env
        self._shuffle_id = map_op._shuffle_id
        self._map_dir = map_op._map_dir
        self._reduce_dir = map_op._reduce_dir
        # Pre-map merge is irrelevant here (inputs are handle wrappers).
        self._input_batch_bytes = 0
        self._merge_buffer_refs_by_node = {}
        self._merge_buffer_bytes_by_node = {}
        self._merge_buffer_bundles_by_node = {}

        self._next_shuffle_map_task_idx = 0
        self._shuffle_map_tasks = {}
        self._map_resource_usage = ExecutionResources.zero()
        from ray.data._internal.execution.bundle_queue import FIFOBundleQueue

        self._output_queue = FIFOBundleQueue()
        self._partition_bundles_emitted = False
        self._total_input_rows = 0
        self._total_input_bytes = 0
        self._map_blocks_stats = []
        from collections import defaultdict

        self._partition_rows = defaultdict(int)
        self._partition_bytes = defaultdict(int)
        self._map_bar = None
        self._completed_handle_refs = []
        self._shared_handles_ref = None
        self._output_schema = None

        # ---- state lifted from the absorbed reduce op ----
        self._num_inputs = reduce_op._num_inputs
        self._in_num_partitions = reduce_op._num_partitions
        self._reduce_fn = reduce_op._reduce_fn
        self._reduce_ray_remote_args = dict(reduce_op._reduce_ray_remote_args)
        # The planner sizes a disk join reduce at 3x its input
        # (_DISK_JOIN_REDUCE_PEAK_MEMORY_MULTIPLIER) because the task decodes
        # every shard of both inputs before calling reduce_fn, so peak memory
        # is O(whole partition). On this cluster that request is what caps the
        # first reduce stage at ~73 concurrent tasks of 128 cores:
        #   294 GiB of memory resource / (1.34 GiB per partition * 3) = 72.8.
        # Streaming the probe side (RAY_DATA_SHUFFLE_PROBE_BATCH_BYTES) takes
        # the real peak to ~1.33x, so the 3x is no longer the right sizing.
        # Override together with that flag, never alone.
        self._peak_memory_multiplier = float(
            os.environ.get(
                "RAY_DATA_FUSED_REDUCE_MEMORY_MULTIPLIER",
                reduce_op._peak_memory_multiplier,
            )
        )
        self._max_bytes_per_fetch = reduce_op._max_bytes_per_fetch
        self._fetch_threads = reduce_op._fetch_threads
        self._emit_empty_partition_blocks = reduce_op._emit_empty_partitions
        self._fused_output_map_transformer = reduce_op._fused_output_map_transformer
        self._fused_output_map_task_kwargs = dict(
            reduce_op._fused_output_map_task_kwargs
        )
        self._fused_output_map_target_max_block_size_override = (
            reduce_op._fused_output_map_target_max_block_size_override
        )

        # partition_id -> input_index -> wrapper bundle (reduce-style pairing).
        self._pending_inputs: Dict[int, Dict[int, RefBundle]] = {}
        self._output_queue_deque = deque()  # unused; keep attr surface minimal

    # ---- input side: behave like the absorbed reduce op --------------------

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        assert 0 <= input_index < self._num_inputs
        if not refs.block_refs:
            refs.destroy_if_owned()
            return
        partition_id = extract_partition_id(refs)
        pending = self._pending_inputs.setdefault(partition_id, {})
        assert input_index not in pending
        pending[input_index] = refs
        if len(pending) == self._num_inputs:
            del self._pending_inputs[partition_id]
            self._submit_fused_task(
                partition_id, [pending[i] for i in range(self._num_inputs)]
            )

    def all_inputs_done(self) -> None:
        # Skip DiskHashShuffleMapOp.all_inputs_done's merge-buffer flush; just
        # flush partitions missing some inputs, then try emitting wrappers.
        PhysicalOperator.all_inputs_done(self)
        for partition_id in list(self._pending_inputs.keys()):
            pending = self._pending_inputs.pop(partition_id)
            self._submit_fused_task(
                partition_id, [pending.get(i) for i in range(self._num_inputs)]
            )
        self._maybe_emit_partition_bundles()

    def _submit_fused_task(
        self, partition_id: int, bundles: List[Optional[RefBundle]]
    ) -> None:
        handles_refs: List[Any] = []
        estimated_bytes = 0
        for bundle in bundles:
            if bundle is not None and bundle.block_refs:
                handles_refs.append(bundle.block_refs[0])
                estimated_bytes += sum((m.size_bytes or 0) for m in bundle.metadata)
            else:
                handles_refs.append([])

        resources: Dict[str, Any] = {
            "num_cpus": self._shuffle_map_task_num_cpus,
        }
        if estimated_bytes > 0:
            resources["memory"] = int(
                estimated_bytes * self._peak_memory_multiplier
            )
        ray_options: Dict[str, Any] = {
            **resources,
            "scheduling_strategy": "SPREAD",
            "runtime_env": _SHUFFLE_REDUCE_RUNTIME_ENV,
            # Name the task after its operator. Without this Ray falls back to
            # the function name, and anything that attributes state-API task
            # events back to operators (the dashboard, and the trace exporter's
            # backfill for operators that emit handles instead of blocks) can
            # no longer tell which operator a task belongs to. Stock
            # _disk_shuffle_{map,reduce}_task have the same gap.
            "name": self.name,
        }
        user_args = dict(self._reduce_ray_remote_args)
        user_args.pop("runtime_env", None)
        ray_options.update(user_args)

        map_task_context = None
        if self._fused_output_map_transformer is not None:
            map_task_context = TaskContext(
                task_idx=partition_id,
                op_name=self.name,
                target_max_block_size_override=(
                    self._fused_output_map_target_max_block_size_override
                ),
            )
            map_task_context.kwargs.update(self._fused_output_map_task_kwargs)

        handle_ref = _disk_shuffle_fused_reduce_map_task.options(
            **ray_options
        ).remote(
            *handles_refs,
            partition_id=partition_id,
            reduce_fn=self._reduce_fn,
            max_bytes_per_fetch=self._max_bytes_per_fetch,
            fetch_threads=self._fetch_threads,
            map_transformer=self._fused_output_map_transformer,
            map_task_context=map_task_context,
            data_context=self.data_context,
            emit_empty_partition=self._emit_empty_partition_blocks,
            out_partition_fn=self._partition_fn,
            out_num_partitions=self._num_partitions,
            out_dir=self._map_dir,
            out_shuffle_id=self._shuffle_id,
            out_compression=self.data_context.shuffle_compression,
            out_block_transformer=self._block_transformer,
        )

        self._next_shuffle_map_task_idx += 1
        task = MetadataOpTask(
            task_index=partition_id,
            object_ref=handle_ref,
            task_done_callback=functools.partial(
                self._handle_fused_done, partition_id, handle_ref, bundles
            ),
            task_resource_bundle=ExecutionResources.from_resource_dict(resources),
        )
        self._shuffle_map_tasks[partition_id] = task
        requested = task.get_requested_resource_bundle()
        if requested is not None:
            self._map_resource_usage = self._map_resource_usage.add(requested)
        self._metrics.on_task_submitted(
            partition_id,
            RefBundle((), schema=None, owns_blocks=False),
            task_id=task.get_task_id(),
        )

    def _handle_fused_done(
        self,
        partition_id: int,
        handle_ref: Any,
        bundles: List[Optional[RefBundle]],
    ) -> None:
        # Reuse the inherited bookkeeping wholesale.
        self._handle_map_done(
            partition_id,
            handle_ref,
            [b for b in bundles if b is not None],
        )

    # ---- lifecycle ----------------------------------------------------------

    def get_active_tasks(self) -> List[OpTask]:
        return list(self._shuffle_map_tasks.values())

    def has_execution_finished(self) -> bool:
        if (
            self._shuffle_map_tasks
            or self._pending_inputs
            or not self._partition_bundles_emitted
            or self._output_queue.has_next()
        ):
            return False
        return PhysicalOperator.has_execution_finished(self)

    def has_completed(self) -> bool:
        return (
            not self._shuffle_map_tasks
            and not self._pending_inputs
            and self._partition_bundles_emitted
            and not self._output_queue.has_next()
            and PhysicalOperator.has_completed(self)
        )

    def _do_shutdown(self, force: bool = False) -> None:
        for pending in self._pending_inputs.values():
            for bundle in pending.values():
                bundle.destroy_if_owned()
        self._pending_inputs.clear()
        super()._do_shutdown(force)

    def incremental_resource_usage(self) -> ExecutionResources:
        memory = 0
        for upstream in self.input_dependencies:
            if isinstance(upstream, DiskHashShuffleMapOp):
                sizes = [
                    b for b in upstream.get_partition_bytes().values() if b > 0
                ]
                if sizes:
                    memory += int(
                        (sum(sizes) / len(sizes)) * self._peak_memory_multiplier
                    )
        return ExecutionResources(
            cpu=self._shuffle_map_task_num_cpus, memory=memory
        )

    def min_scheduling_resources(self) -> ExecutionResources:
        return self.incremental_resource_usage()

    def num_output_rows_total(self) -> Optional[int]:
        # Reduce (join) can change the row count arbitrarily.
        return None

    def progress_str(self) -> str:
        done = self._next_shuffle_map_task_idx - len(self._shuffle_map_tasks)
        return f"fused reduce+map: {done}/{self._next_shuffle_map_task_idx}"


# =============================================================================
# PROTOTYPE 2: fuse an upstream TaskPoolMapOperator (e.g. parquet read) into a
# DiskHashShuffleMapOp. The fused task runs the upstream MapTransformer over
# the raw input bundles (file manifests for reads) and partitions/persists the
# produced blocks directly -- the read output never enters the object store,
# which removes the read<->map backpressure interlock entirely.
# Enabled via RAY_DATA_FUSE_READ_INTO_SHUFFLE_MAP=1.
# =============================================================================


# EXPERIMENT (RAY_DATA_PLAIN_PARQUET_READ=1): decode filterless parquet
# fragments through ``pq.ParquetFile.read_row_group`` instead of the dataset
# Scanner. The Scanner's fragment/readahead machinery costs 2-3x the plain
# reader's CPU for the same bytes (A/B on the cluster node: 0.22-0.28 vs
# 0.53-0.78 core-s per 16 row groups), and a scan with no predicate and no
# synthesized columns uses none of what it pays for. Applied as a monkeypatch
# at the override seam ``_iter_fragment_tables_without_pickle_check`` so the
# caller's pickle check, retry scoping and position bookkeeping all stay in
# force; any fragment the bypass cannot faithfully serve falls back to the
# original method.
_polars_part_state = {"done": False}


def _maybe_install_polars_partition():
    """Swap transform_pyarrow.hash_partition for a polars partition_by, once.

    The planner's partition closures resolve ``hash_partition`` through
    ``hash_shuffle_v2``'s module globals at CALL time, so patching that one
    name redirects every already-deserialized partition_fn, map and reduce
    side alike. The partition id only needs to be a deterministic function of
    the key columns consistent across all tasks of a run; the reduce side
    groups by partition index, so polars' own hash qualifies. Local bench
    said this LOSES (1-thread 12% slower, free-threaded 2.2x cpu); this gate
    exists to let the cluster contradict the laptop.
    """
    if _polars_part_state["done"]:
        return
    _polars_part_state["done"] = True
    if os.environ.get("RAY_DATA_POLARS_PARTITION") != "1":
        return
    from ray.data._internal.execution.operators import hash_shuffle_v2 as _hsv2

    def _polars_hash_partition(block, *, hash_cols, num_partitions):
        import polars as pl

        if block.num_rows == 0:
            return {}
        if num_partitions == 1:
            return {0: block}
        df = pl.from_arrow(block, rechunk=False)
        pid = (df.select(hash_cols).hash_rows(seed=0) % num_partitions).alias("__p")
        parts = df.with_columns(pid).partition_by(
            "__p", include_key=False, as_dict=True, maintain_order=False
        )
        return {int(k[0]): d.to_arrow() for k, d in parts.items() if d.height}

    _hsv2.hash_partition = _polars_hash_partition
    print(f"[POLARS_PART pid={os.getpid()}] hash_partition swapped", flush=True)


_radix_part_state = {"done": False, "lib": None}
_RADIX_C_SRC = r'''#include <string.h>
#include <stdint.h>
void brbm8(int64_t** cols, int ncols, const int64_t* pids, int64_t N, int P, int B,
           int64_t** outs, int64_t* cursor, int64_t** bufs, int64_t* bcount){
  for(int64_t i=0;i<N;i++){
    int64_t p=pids[i]; int64_t b=bcount[p];
    for(int c=0;c<ncols;c++) bufs[c][(int64_t)p*B+b]=cols[c][i];
    b++;
    if(b==B){ for(int c=0;c<ncols;c++) memcpy(outs[c]+cursor[p], bufs[c]+(int64_t)p*B,(size_t)B*8); cursor[p]+=B; bcount[p]=0; }
    else bcount[p]=b;
  }
  for(int p=0;p<P;p++){int64_t b=bcount[p]; if(b){for(int c=0;c<ncols;c++) memcpy(outs[c]+cursor[p], bufs[c]+(int64_t)p*B,(size_t)b*8); cursor[p]+=b;}}
}
'''


def _maybe_install_radix_partition():
    """Swap hash_partition for a C buffered-radix scatter (all-8-byte fast path).

    Replaces the pyarrow ``take_table`` gather (random-read, ~370 MB/s measured
    on m5d.4xlarge) with a sequential-read scatter into per-partition cache-line
    buffers (~2.3x faster end-to-end on the partition, byte-equal output). Falls
    back to stock ``hash_partition`` for any nullable / non-8-byte / variable-
    width column, so string shuffles and the aggregate are untouched. Partition
    ids come from the same ``_hash_partition_vectorized`` as stock, so reduce-
    side partition indexing stays consistent. Gated by RAY_DATA_RADIX_PARTITION=1.
    """
    if _radix_part_state["done"]:
        return
    _radix_part_state["done"] = True
    if os.environ.get("RAY_DATA_RADIX_PARTITION") != "1":
        return
    import ctypes
    import subprocess

    import numpy as np
    import pyarrow as pa
    from ray.data._internal.arrow_ops import transform_pyarrow as tp
    from ray.data._internal.execution.operators import hash_shuffle_v2 as _hsv2

    cpath = f"/tmp/radix8_{os.getpid()}.c"
    sopath = f"/tmp/radix8_{os.getpid()}.so"
    try:
        with open(cpath, "w") as f:
            f.write(_RADIX_C_SRC)
        subprocess.run(
            ["cc", "-O3", "-march=native", "-shared", "-fPIC", cpath, "-o", sopath],
            check=True, capture_output=True,
        )
        lib = ctypes.CDLL(sopath)
    except Exception as e:  # noqa: BLE001
        print(f"[RADIX_PART pid={os.getpid()}] compile failed, keeping stock: {e}",
              flush=True)
        return
    VP = ctypes.c_void_p
    I64 = ctypes.POINTER(ctypes.c_int64)
    PP = ctypes.POINTER(ctypes.c_void_p)
    lib.brbm8.argtypes = [PP, ctypes.c_int, I64, ctypes.c_int64, ctypes.c_int,
                          ctypes.c_int, PP, I64, PP, I64]
    _radix_part_state["lib"] = lib
    B = 16

    def _fwb(ty):
        if pa.types.is_boolean(ty):
            return None
        try:
            bw = ty.bit_width
        except Exception:  # noqa: BLE001
            return None
        return None if (bw is None or bw % 8) else bw // 8

    def _arr1(c):
        if isinstance(c, pa.ChunkedArray):
            return c.chunk(0) if c.num_chunks == 1 else c.combine_chunks()
        return c

    def _mark(tag):
        key = f"mark_{tag}"
        if _radix_part_state.get(key):
            return
        _radix_part_state[key] = True
        try:
            with open("/mnt/cluster_storage/radix_fired.log", "a") as _mf:
                _mf.write(f"{tag} pid={os.getpid()}\n")
        except Exception:  # noqa: BLE001
            pass

    def radix_hash_partition(block, *, hash_cols, num_partitions):
        N = block.num_rows
        if N == 0 or num_partitions <= 1:
            return tp.hash_partition(block, hash_cols=hash_cols,
                                     num_partitions=num_partitions)
        arrs0 = []
        for c in block.columns:
            if c.null_count > 0:
                _mark("fallback_null")
                return tp.hash_partition(block, hash_cols=hash_cols,
                                         num_partitions=num_partitions)
            a = _arr1(c)
            if _fwb(a.type) != 8:
                _mark(f"fallback_width_{a.type}")
                return tp.hash_partition(block, hash_cols=hash_cols,
                                         num_partitions=num_partitions)
            arrs0.append(a)
        _mark("radix_used")
        names = block.column_names
        nc = len(arrs0)
        pids = np.asarray(
            tp._hash_partition_vectorized(block.select(hash_cols), num_partitions),
            dtype=np.int64,
        )
        counts = np.bincount(pids, minlength=num_partitions).astype(np.int64)
        offs = np.zeros(num_partitions + 1, dtype=np.int64)
        offs[1:] = np.cumsum(counts)
        datas = [
            np.frombuffer(a.buffers()[1], dtype=np.int64, count=a.offset + N)[
                a.offset:a.offset + N
            ]
            for a in arrs0
        ]
        outs = [np.empty(N, dtype=np.int64) for _ in range(nc)]
        bufs = [np.empty(num_partitions * B, dtype=np.int64) for _ in range(nc)]
        bc = np.zeros(num_partitions, dtype=np.int64)
        cur = offs[:num_partitions].copy()
        cpp = (ctypes.c_void_p * nc)(*[d.ctypes.data_as(VP) for d in datas])
        opp = (ctypes.c_void_p * nc)(*[o.ctypes.data_as(VP) for o in outs])
        bpp = (ctypes.c_void_p * nc)(*[b.ctypes.data_as(VP) for b in bufs])
        lib.brbm8(cpp, nc, pids.ctypes.data_as(I64), N, num_partitions, B,
                  opp, cur.ctypes.data_as(I64), bpp, bc.ctypes.data_as(I64))
        full = [
            pa.Array.from_buffers(arrs0[j].type, N, [None, pa.py_buffer(outs[j])])
            for j in range(nc)
        ]
        st = pa.Table.from_arrays(full, names=names)
        return {
            int(p): st.slice(int(offs[p]), int(counts[p]))
            for p in np.nonzero(counts)[0]
        }

    _hsv2.hash_partition = radix_hash_partition
    print(f"[RADIX_PART pid={os.getpid()}] hash_partition swapped (8-byte radix)",
          flush=True)
    try:
        with open("/mnt/cluster_storage/radix_fired.log", "a") as _mf:
            _mf.write(f"swapped pid={os.getpid()}\n")
    except Exception:  # noqa: BLE001
        pass


_plain_patch_state = {"done": False}


def _maybe_install_plain_read():
    """Install the plain-read bypass, once per process, at task runtime.

    At module import time the worker's os.environ does NOT yet carry the
    job's env vars (Ray injects them at lease time, after this module can
    already be imported), so an import-time gate silently never fires: the
    ENVCHK run showed plain_env='1' at task runtime with the module block
    unexecuted in every worker. Checking inside the task sees the injected
    environment.
    """
    if _plain_patch_state["done"]:
        return
    _plain_patch_state["done"] = True
    _mode = os.environ.get("RAY_DATA_PLAIN_PARQUET_READ")
    if _mode in ("1", "pre"):
        from ray.data._internal.datasource_v2.formats.parquet import (
            parquet_file_reader as _pfr,
        )

        _orig_iter_tables = (
            _pfr.ParquetFileReader._iter_fragment_tables_without_pickle_check
        )

        _plain_called = {"plain": False, "fallback": False}

        def _plain_iter_tables(self, fragment, scanner_kwargs):
            import pyarrow.parquet as _pq

            if (
                scanner_kwargs.get("filter") is not None
                or getattr(self, "_synthesized_columns", ())
            ):
                if not _plain_called["fallback"]:
                    _plain_called["fallback"] = True
                    print(f"[PLAIN_BYPASS pid={os.getpid()}] FALLBACK branch "
                          f"(filter={scanner_kwargs.get('filter') is not None})",
                          flush=True)
                _first = [True]
                for _tbl in _orig_iter_tables(self, fragment, scanner_kwargs):
                    if _first[0]:
                        _first[0] = False
                        _rk = [c for c in os.environ.get("RAY_DATA_READ_DICTIONARY","").split(",") if c]
                        _types = {c: str(_tbl.schema.field(c).type)[:12] for c in _rk if c in _tbl.schema.names}
                        print(f"[FALLBACK_DTYPE] {_types}", flush=True)
                    yield _tbl
                return
            if not _plain_called["plain"]:
                _plain_called["plain"] = True
                print(f"[PLAIN_BYPASS pid={os.getpid()}] plain branch ACTIVE "
                      f"path={fragment.path[-40:]}", flush=True)
            # EXPERIMENT (RAY_DATA_READ_DICTIONARY=col,col): read these raw
            # parquet columns as DictionaryArray instead of flattening the
            # file's RLE_DICTIONARY to plain strings. For q1 the group keys
            # (returnflag, linestatus) are 1-char columns: flattened they are
            # ~60 GiB of arrow and re-hashed in group_by; kept as dictionary
            # they are ~6 GiB and group_by hashes small ints. The dictionary
            # already exists in the file, so this is free on decode.
            _rd = [c for c in os.environ.get("RAY_DATA_READ_DICTIONARY", "").split(",") if c]
            pf = _pq.ParquetFile(
                fragment.path, filesystem=fragment.filesystem,
                read_dictionary=_rd or None,
            )
            row_groups = (
                [rg.id for rg in fragment.row_groups]
                if fragment.row_groups is not None
                else list(range(pf.metadata.num_row_groups))
            )
            columns = scanner_kwargs.get("columns")
            # [MAP_PROF] (RAY_DATA_MAP_PROF=1): time pyarrow parquet decode to
            # split the read-stage CPU (trace: 6,791 core-s) into decode vs
            # rest (partition+compress). decode_cpu via process_time is
            # over-counted under the prefetch threadpool; decode_wall
            # (perf_counter) is the clean number -> compare to stage busy-s.
            import threading as _thr
            _gp = globals()
            _mp = _gp.setdefault(
                "_MAP_DECODE_PROF",
                {"cpu": 0.0, "wall": 0.0, "n": 0, "lock": _thr.Lock()},
            )
            _prof_on = os.environ.get("RAY_DATA_MAP_PROF") == "1"

            def _rg_read(_rg):
                if not _prof_on:
                    return pf.read_row_group(_rg, columns=columns)
                _c0 = time.process_time()
                _w0 = time.perf_counter()
                _t = pf.read_row_group(_rg, columns=columns)
                _dc = time.process_time() - _c0
                _dw = time.perf_counter() - _w0
                with _mp["lock"]:
                    _mp["cpu"] += _dc
                    _mp["wall"] += _dw
                    _mp["n"] += 1
                    if _mp["n"] % 400 == 0:
                        print(
                            f"[MAP_DECODE pid={os.getpid()}] rg={_mp['n']} "
                            f"decode_cpu_s={_mp['cpu']:.1f} "
                            f"decode_wall_s={_mp['wall']:.1f}",
                            flush=True,
                        )
                return _t

            if _mode == "pre":
                # Plain decode with a bounded prefetch: the Scanner's only
                # wall advantage is pipelined fetches, so buy that alone.
                # Loop A/B on the node: scanner cpu 0.78 / wall 0.54; plain
                # serial 0.42 / 2.7; this shape 0.46 / 1.07 per 16 row
                # groups at 12 concurrent workers.
                from concurrent.futures import ThreadPoolExecutor

                _n_pre = int(
                    os.environ.get("RAY_DATA_PLAIN_PARQUET_PREFETCH", "3")
                )
                with ThreadPoolExecutor(max_workers=_n_pre) as ex:
                    futures = [
                        ex.submit(_rg_read, rg)
                        for rg in row_groups
                    ]
                    for fut in futures:
                        table = fut.result()
                        if table.num_rows:
                            yield table
                return
            for rg in row_groups:
                table = _rg_read(rg)
                if table.num_rows:
                    yield table

        _pfr.ParquetFileReader._iter_fragment_tables_without_pickle_check = (
            _plain_iter_tables
        )
        print(f"[PLAIN_BYPASS pid={os.getpid()}] patch installed", flush=True)

        # Make the Scanner fallback (filtered reads, e.g. q1's l_shipdate)
        # decode the low-card group-key columns as DictionaryArray too. The
        # plain branch sets read_dictionary on the ParquetFile; the fallback
        # goes through pyarrow.dataset, so inject via the format's read_options.
        _rdcols = [c for c in os.environ.get("RAY_DATA_READ_DICTIONARY", "").split(",") if c]
        if _rdcols:
            import pyarrow.dataset as _pds

            def _make_format_dict(self):
                kw = dict(self._parquet_format_kwargs)
                if "read_options" not in kw:
                    kw["read_options"] = _pds.ParquetReadOptions(
                        dictionary_columns=_rdcols
                    )
                return _pds.ParquetFileFormat(**kw)

            _pfr.ParquetFileReader._make_format = _make_format_dict
            print(f"[READ_DICT_SCANNER] dictionary_columns={_rdcols}", flush=True)


_UNSET = object()


# EXPERIMENT (RAY_DATA_READ_CONCURRENCY_CAPS): cap each read operator's
# concurrency so a scan that is NOT on the critical path cannot take slots from
# one that is. On q9 the five small-table scans held 73-83 of ~90 live slots
# from t=8 to t=24 while lineitem, the only scan that gates the first join, ran
# at 5-18; Photon gives each scan a share proportional to its total work, which
# makes all six finish together instead of leaving lineitem to drain alone.
#
# Format: "frag=cap,frag=cap". A fragment matches the operator name only at a
# token boundary, so "s_suppkey" does not match partsupp's "ps_suppkey".
def _parse_read_concurrency_caps():
    out = []
    for item in os.environ.get("RAY_DATA_READ_CONCURRENCY_CAPS", "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        frag, _, cap = item.partition("=")
        out.append((re.compile(r"(?<![A-Za-z0-9_])" + re.escape(frag.strip())),
                    int(cap)))
    return out


_READ_CONCURRENCY_CAPS = _parse_read_concurrency_caps()


@ray.remote
def _disk_shuffle_fused_upstream_map_task(
    *blocks: Any,
    upstream_transformer: Any,
    map_task_context: "TaskContext",
    data_context: "DataContext",
    partition_fn: Any,
    num_partitions: int,
    out_dir: str,
    map_id: int,
    shuffle_id: str,
    compression: Optional[str],
    block_transformer: Optional[Any],
) -> ShuffleHandle:
    """Upstream map (read) + shuffle map in one task."""
    from ray.data._internal.arrow_ops import transform_pyarrow as _tp
    from ray.data._internal.execution.operators.map_transformer import TransformClock

    _maybe_install_plain_read()
    _maybe_install_polars_partition()

    # EXPERIMENT (RAY_DATA_FUSED_READ_NO_SHAPING=<rows>): stop concatenating
    # the read chain's batches into one 128 MiB block before the shuffle half.
    # Shaping exists for object-store economics and a fused task's blocks never
    # get there; here it degenerates the task into a single 91 MiB unit (one
    # hash_partition call, W_PROF gather=0), so take's random gather runs
    # against DRAM and nothing overlaps the fetch. Two traps found probing the
    # seam locally: `disable_block_shaping` means "no cap" (concat EVERYTHING
    # at finalize), and in a fused chain EVERY fn shapes its own output, with
    # the LAST one (Project, option=None, which also means concat-all) having
    # the final say. So every fn gets an explicit rows-per-block target; one
    # parquet row group is 122,880 rows, making the reader's batches flow
    # through unmerged.
    _ns_rows = int(os.environ.get("RAY_DATA_FUSED_READ_NO_SHAPING", "0"))
    if _ns_rows > 0:
        from ray.data._internal.output_buffer import OutputBlockSizeOption

        upstream_transformer._output_block_size_option_override = None
        for _fn in upstream_transformer.get_transform_fns():
            _fn._output_block_size_option = OutputBlockSizeOption.of(
                target_num_rows_per_block=_ns_rows
            )

    exec_stats_builder = BlockExecStats.builder()

    # EXPERIMENT (RAY_DATA_READMAP_PROF=1): the reduce half has a per-phase
    # breakdown; this half had none, so its 70s were a single opaque bar.
    # Timers nest the same way -- "read" is inclusive of the whole transform
    # chain -- and the chain's own TransformClock splits that further into its
    # stages (parquet read vs the fused Project) when
    # DataContext.per_stage_map_timing is on.
    _rm_prof = collections.defaultdict(float)
    # CPU time per phase, so each phase's cpu/wall says whether it is computing
    # or waiting. The stage as a whole reports cpu_s/busy_s = 1.19, but that is
    # an average over read, partition and encode_write together and cannot say
    # which of them has idle in it. Only a phase whose ratio is well under 1.0
    # has anything a producer thread could overlap with.
    #
    # process_time, not thread_time: it is process-wide and therefore counts
    # pyarrow's and lz4's own thread pools, which is most of the work here.
    # thread_time would see only the task thread and make every phase look
    # idle. A Ray worker runs one task at a time, so attributing the whole
    # process's CPU delta to the running phase is sound; its background gRPC
    # threads are the only contamination and they are negligible.
    _rm_cpu = collections.defaultdict(float)
    _rm_every = int(os.environ.get("RAY_DATA_READMAP_PROF_EVERY", "1"))
    _rm_on = (
        os.environ.get("RAY_DATA_READMAP_PROF") == "1" and map_id % _rm_every == 0
    )
    _rm_t0 = time.perf_counter()
    _rm_c0 = time.process_time()

    if _rm_on:
        # Settles the bypass mystery in one line: what the worker's env says
        # at task runtime, and whether the reader class carries the patched
        # method. Any mismatch between the two names the failure mode.
        from ray.data._internal.datasource_v2.formats.parquet import (
            parquet_file_reader as _pfr_chk,
        )

        _meth = (
            _pfr_chk.ParquetFileReader._iter_fragment_tables_without_pickle_check
        )
        print(
            f"[ENVCHK m={map_id}] plain_env="
            f"{os.environ.get('RAY_DATA_PLAIN_PARQUET_READ')!r} "
            f"method={_meth.__name__} installer_ran="
            f"{_plain_patch_state['done']}",
            flush=True,
        )

    # EXPERIMENT (RAY_DATA_READMAP_THREADPROF=N): per-thread CPU accounting for
    # every Nth map task. process_time says this task burns 2.44 core-s, but
    # the identical work costs 1.31 in every reproduction (fresh subprocess,
    # warmed Ray worker, 12 concurrent, S3, real reader). process_time cannot
    # say WHICH thread burns the difference and py-spy cannot see native pool
    # threads, so read /proc/self/task/*/stat directly: utime+stime per thread,
    # named, snapshotted at task start and end. The delta table either names
    # the thread pool that only spins in production, or disagrees with
    # process_time and convicts the measurement itself.
    _tp_on = False
    _tp_mod = int(os.environ.get("RAY_DATA_READMAP_THREADPROF", "0"))
    if _tp_mod and map_id % _tp_mod == 0:
        _tp_on = True

    def _thread_cpu():
        out = {}
        try:
            tick = os.sysconf("SC_CLK_TCK")
            for tid in os.listdir("/proc/self/task"):
                try:
                    with open(f"/proc/self/task/{tid}/stat", "rb") as f:
                        # comm can contain spaces/parens; split on the LAST ") ".
                        rest = f.read().rsplit(b") ", 1)[1].split()
                    with open(f"/proc/self/task/{tid}/comm") as f:
                        name = f.read().strip()
                    # fields after comm: state=0 ... utime=11 stime=12
                    out[tid] = (name, (int(rest[11]) + int(rest[12])) / tick)
                except OSError:
                    continue
        except (OSError, ValueError):
            pass
        return out

    _tp_before = _thread_cpu() if _tp_on else None
    _tp_c0 = time.process_time()
    _tp_main_tid = str(threading.get_native_id()) if _tp_on else None

    # EXPERIMENT (RAY_DATA_READMAP_SPY=N): native-stack sample of every Nth
    # map task. Subtraction is exhausted: the read phase costs 24 core-s/GiB
    # compressed where pyarrow's reader on the same file costs 4.6-7.5, and
    # the wrapper (block shaping, Project, generator) is under 1%. The missing
    # 2-3x is inside C++, which Python timers cannot split; --native can.
    # Duration covers roughly one task so the samples stay attributable.
    _spy_proc = _spy_out = None
    _spy_mod = int(os.environ.get("RAY_DATA_READMAP_SPY", "0"))
    if _spy_mod and map_id % _spy_mod == 0:
        import shutil
        import subprocess

        _exe = shutil.which("py-spy")
        if _exe:
            _spy_out = f"/tmp/spy_readmap_m{map_id}.folded"
            _spy_proc = subprocess.Popen(
                ["sudo", "-n", _exe, "record", "--pid", str(os.getpid()),
                 "--duration",
                 os.environ.get("RAY_DATA_READMAP_SPY_SECONDS", "2"),
                 "--format", "raw", "--output", _spy_out,
                 "--native", "--rate", "150"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

    def _rm_timed(key):
        class _T:
            def __enter__(self_):
                self_.t = time.perf_counter()
                self_.c = time.process_time()

            def __exit__(self_, *e):
                _rm_prof[key] += time.perf_counter() - self_.t
                _rm_cpu[key] += time.process_time() - self_.c

        return _T()

    def _rm_timed_iter(it, key):
        """Charge the time a consumer spends waiting on ``it`` to ``key``."""
        while True:
            t = time.perf_counter()
            c = time.process_time()
            try:
                v = next(it)
            except StopIteration:
                _rm_prof[key] += time.perf_counter() - t
                _rm_cpu[key] += time.process_time() - c
                return
            _rm_prof[key] += time.perf_counter() - t
            _rm_cpu[key] += time.process_time() - c
            yield v

    # On-disk bytes this task is responsible for, straight from the ListFiles
    # manifest's __file_size column. Without it the only available denominator
    # is the other engine's reported input size, which assumes both engines
    # read the same bytes -- exactly the assumption worth checking. Deduped by
    # path because a chunked file contributes one row per chunk, all carrying
    # the whole file's size. Whole-file size, so an UPPER bound on what a
    # column-pruned scan actually fetches.
    _rm_in_bytes = 0
    if _rm_on:
        _seen_paths = set()
        for _b in blocks:
            try:
                _t = (
                    _b
                    if isinstance(_b, pa.Table)
                    else BlockAccessor.for_block(_b).to_arrow()
                )
                if "__file_size" not in _t.column_names:
                    continue
                for _pth, _sz in zip(
                    _t.column("__path").to_pylist(),
                    _t.column("__file_size").to_pylist(),
                ):
                    if _pth not in _seen_paths:
                        _seen_paths.add(_pth)
                        _rm_in_bytes += int(_sz or 0)
            except Exception:  # noqa: BLE001 - profiling must not break the task
                pass

    # Named rather than inline: draining it at the end is what yields the
    # per-stage split of the read chain.
    _rm_clock = TransformClock()

    def _data_blocks():
        with DataContext.current(data_context), TaskContext.current(map_task_context):
            yield from upstream_transformer.apply_transform(
                iter(blocks), map_task_context, clock=_rm_clock
            )

    def _as_arrow(block: Block) -> pa.Table:
        if not isinstance(block, pa.Table):
            return BlockAccessor.for_block(block).to_arrow()
        return block

    node_id = ray.get_runtime_context().get_node_id()
    ShuffleFileServer.options(
        name=_file_server_name(
            shuffle_id, node_id, map_id % SHUFFLE_FILE_SERVERS_PER_NODE
        ),
        namespace=_SHUFFLE_FILE_SERVER_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        max_restarts=-1,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        num_cpus=0,
    ).remote(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"map_{map_id}.shf")
    tmp_path = final_path + ".tmp"
    output_schema: Optional[pa.Schema] = None
    final_size_on_close = -1
    try:
        with open(tmp_path, "wb") as out_file:
            writer = _PartitionWriter(out_file, map_id, compression)
            blocks_iter = _rm_timed_iter(_data_blocks(), "read")
            if block_transformer is not None:
                arrow_blocks = [
                    _as_arrow(b)
                    for b in blocks_iter
                    if BlockAccessor.for_block(b).num_rows() > 0
                ]
                if arrow_blocks:
                    with _rm_timed("block_transform"):
                        combined = _tp.concat(arrow_blocks, promote_types=True)
                        blocks_iter = iter([block_transformer(combined)])
                else:
                    blocks_iter = iter(())
            for block in blocks_iter:
                block = _as_arrow(block)
                if output_schema is None:
                    output_schema = getattr(block, "schema", None)
                if block.num_rows == 0:
                    continue
                with _rm_timed("partition"):
                    for out_pid, shard in partition_fn(block).items():
                        writer.add_shard(out_pid, shard)
            with _rm_timed("encode_write"):
                writer.flush_all()
                out_file.flush()
            final_size_on_close = out_file.tell()
            if writer.index:
                expected = max(
                    off + length
                    for ranges in writer.index.values()
                    for off, length in ranges
                )
            else:
                expected = 0
            assert final_size_on_close == expected
        os.rename(tmp_path, final_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    if _rm_on:
        _rm_total = time.perf_counter() - _rm_t0
        # "read" is inclusive of the transform chain; the chain's own clock
        # says how that time split. "other" is everything no timer covered:
        # the file-server actor lookup, makedirs, the size assert, the rename.
        _ph = _rm_clock.drain()
        _rm_other = (
            _rm_total
            - _rm_prof["read"]
            - _rm_prof["partition"]
            - _rm_prof["encode_write"]
            - _rm_prof["block_transform"]
        )
        _stages = (
            ",".join(f"{x:.2f}" for x in _ph.stage_s)
            if _ph.stage_s is not None
            else "-"
        )
        _fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(
            f"[RM_PROF] m={map_id} in_mib={_rm_in_bytes / 2 ** 20:.1f} "
            f"total={_rm_total:.2f} "
            f"read={_rm_prof['read']:.2f} "
            f"partition={_rm_prof['partition']:.2f} "
            f"encode_write={_rm_prof['encode_write']:.2f} "
            f"block_transform={_rm_prof['block_transform']:.2f} "
            f"other={_rm_other:.2f} "
            f"chain={_ph.total_s:.2f} prep={_fmt(_ph.input_prep_s)} "
            f"body={_fmt(_ph.function_body_s)} build={_fmt(_ph.output_build_s)} "
            f"stages={_stages}",
            flush=True,
        )
        # Splits the encode_write bucket. write_MBps is per task, so divide the
        # node's concurrent tasks into the NVMe's sustained rate to compare.
        _wp = getattr(writer, "prof", None)
        if _wp:
            _wr = _wp["write"] or 1e-9
            print(
                f"[W_PROF] m={map_id} frames={_wp['frames']} "
                f"in_mib={_wp['bytes_in'] / 2 ** 20:.1f} "
                f"out_mib={_wp['bytes_out'] / 2 ** 20:.1f} "
                f"ratio={_wp['bytes_in'] / max(_wp['bytes_out'], 1):.2f} "
                f"gather={_wp['gather']:.2f}/{_wp['gather_cpu']:.2f} "
                f"encode={_wp['encode']:.2f}/{_wp['encode_cpu']:.2f} "
                f"write={_wp['write']:.2f}/{_wp['write_cpu']:.2f} "
                f"write_MBps={_wp['bytes_out'] / 1e6 / _wr:.0f}",
                flush=True,
            )
        # Printed separately rather than folded into the line above: this is
        # the decision data for whether overlapping read with partition can
        # win anything. A phase at cpu/wall ~1.0 is busy and there is nothing
        # to hide behind; one well under 1.0 is waiting, and that wait is what
        # a bounded producer thread would reclaim.
        _rm_cpu_total = time.process_time() - _rm_c0
        _r = lambda k: (  # noqa: E731
            _rm_cpu[k] / _rm_prof[k] if _rm_prof[k] > 0 else float("nan")
        )
        # op= is load-bearing, not decoration. This task function serves ALL
        # six fused read operators (lineitem's MapLeft plus five MapRights),
        # and each restarts map_id at 0, so a `map_id % N` sample draws from
        # every one of them. Aggregating without the op name mixes a 3046-task
        # scan with a 2-task one whose cpu/busy is 0.218 -- one sampled small
        # task can contribute 13% of the summed wall and it lands almost
        # entirely in `read`, inflating exactly the phase under investigation.
        print(
            f"[RM_CPU] m={map_id} op={getattr(map_task_context, 'op_name', '?')!r} "
            f"total={_rm_cpu_total:.2f}/{_rm_total:.2f}={_rm_cpu_total / _rm_total:.2f} "
            f"read={_rm_cpu['read']:.2f}/{_rm_prof['read']:.2f}={_r('read'):.2f} "
            f"partition={_rm_cpu['partition']:.2f}/{_rm_prof['partition']:.2f}"
            f"={_r('partition'):.2f} "
            f"encode_write={_rm_cpu['encode_write']:.2f}/"
            f"{_rm_prof['encode_write']:.2f}={_r('encode_write'):.2f}",
            flush=True,
        )

    if _tp_on:
        _tp_after = _thread_cpu()
        _tp_total = time.process_time() - _tp_c0
        rows = []
        for tid, (name, cpu) in _tp_after.items():
            before = _tp_before.get(tid, (name, 0.0))[1]
            d = cpu - before
            if d > 0.005:
                rows.append((d, name, tid))
        rows.sort(reverse=True)
        agg = collections.defaultdict(float)
        for d, name, _ in rows:
            agg[name] += d
        per_name = " ".join(
            f"{n}={v:.2f}" for n, v in sorted(agg.items(), key=lambda kv: -kv[1])
        )
        # The comm-name aggregation lumps the main thread with every unnamed
        # pool thread (they inherit the task-title comm), so name the one
        # split that decides where to look next: main vs everyone else.
        main_d = next((d for d, _, tid in rows if tid == _tp_main_tid), 0.0)
        print(
            f"[TP m={map_id}] threads={len(_tp_after)} active={len(rows)} "
            f"sum={sum(d for d, _, _ in rows):.2f} process_time={_tp_total:.2f} "
            f"main={main_d:.2f} "
            f"| {per_name}",
            flush=True,
        )
        # Census of the active threads, tids included: pool threads are
        # spawned in batches, so contiguous tids identify a pool even when
        # every comm reads the same. One line per thread keeps it greppable.
        for d, name, tid in rows[:90]:
            print(f"[TP_T m={map_id}] tid={tid} d={d:.3f} name={name}", flush=True)

    if _spy_proc is not None:
        try:
            out, _ = _spy_proc.communicate(timeout=60)
            if _spy_out and os.path.exists(_spy_out):
                with open(_spy_out) as f:
                    lines = [ln for ln in f.read().splitlines() if ln.strip()]
                print(f"[SPY_RM m={map_id}] samples={len(lines)} lines", flush=True)
                for ln in lines:
                    print(f"[SPY_RM_FOLD m={map_id}] {ln}", flush=True)
            else:
                print(f"[SPY_RM m={map_id}] FAILED rc={_spy_proc.returncode} "
                      f"{(out or '').strip()[-300:]}", flush=True)
        except Exception as e:  # noqa: BLE001 - profiling must not fail the task
            print(f"[SPY_RM m={map_id}] error: {e}", flush=True)

    return {
        "path": os.path.realpath(final_path),
        "index_ranges": _build_range_index(writer.index, num_partitions),
        "shuffle_id": shuffle_id,
        "node_id": node_id,
        "num_partitions": num_partitions,
        "total_bytes": final_size_on_close,
        "compression": compression,
        "num_rows": _counts_to_array(writer.rows_per_partition, num_partitions),
        "decoded_bytes": _counts_to_array(
            writer.decoded_bytes_per_partition, num_partitions
        ),
        "schema": output_schema,
        "exec_stats": exec_stats_builder.build(block_ser_time_s=0.0, task_idx=map_id),
    }


class FusedReadDiskShuffleMapOp(DiskHashShuffleMapOp):
    """An upstream task-pool map (read) fused into a DiskHashShuffleMapOp.

    Inputs are the upstream map's inputs (file-manifest bundles from
    ``ListFiles``), batched by COUNT (their byte size reflects the manifest,
    not the decoded read output, so the inherited byte-based merge buffer
    can't be used). Each fused task runs the upstream transformer, then
    partitions + persists the produced blocks.
    """

    def __init__(self, map_op: DiskHashShuffleMapOp, upstream_op: Any):
        PhysicalOperator.__init__(
            self,
            name=f"Fused[{upstream_op.name}->{map_op.name}]",
            input_dependencies=list(upstream_op.input_dependencies),
            data_context=map_op.data_context,
        )
        # ---- state lifted from the absorbed shuffle map op ----
        self._num_partitions = map_op._num_partitions
        self._partition_fn = map_op._partition_fn
        self._block_transformer = map_op._block_transformer
        self._shuffle_map_task_num_cpus = map_op._shuffle_map_task_num_cpus
        self._map_runtime_env = map_op._map_runtime_env
        self._shuffle_id = map_op._shuffle_id
        self._map_dir = map_op._map_dir
        self._reduce_dir = map_op._reduce_dir
        self._input_batch_bytes = 0
        self._merge_buffer_refs_by_node = {}
        self._merge_buffer_bytes_by_node = {}
        self._merge_buffer_bundles_by_node = {}
        self._next_shuffle_map_task_idx = 0
        self._shuffle_map_tasks = {}
        self._map_resource_usage = ExecutionResources.zero()
        from ray.data._internal.execution.bundle_queue import FIFOBundleQueue

        self._output_queue = FIFOBundleQueue()
        self._partition_bundles_emitted = False
        self._total_input_rows = 0
        self._total_input_bytes = 0
        self._map_blocks_stats = []
        from collections import defaultdict

        self._partition_rows = defaultdict(int)
        self._partition_bytes = defaultdict(int)
        self._map_bar = None
        self._completed_handle_refs = []
        self._shared_handles_ref = None
        self._output_schema = None

        # ---- fused-read specifics ----
        self._upstream_transformer = upstream_op._map_transformer

        # EXPERIMENT (RAY_DATA_Q1_FUSED=1): whole-stage fusion for TPC-H q1.
        # Ray fuses Read->Project->Aggregate into one TASK, but inside the task
        # Project still runs as a separate pyarrow pass that MATERIALISES
        # disc_price/charge as full 43M-row columns before Aggregate reads them.
        # This strips Project to a raw passthrough and replaces the aggregate
        # with a numba loop that computes disc/charge in registers and
        # accumulates in one sweep -- the columns never exist. Columns are q1's
        # TPC-H positions; the output matches the map-side agg contract exactly.
        if os.environ.get("RAY_DATA_Q1_FUSED") == "1" and (
            "l_returnflag" in getattr(map_op, "name", "")
            and "l_linestatus" in getattr(map_op, "name", "")
        ):
            from ray.data._internal.execution.operators.map_transformer import (
                MapTransformer,
            )

            _orig_fns = self._upstream_transformer.get_transform_fns()
            # keep only the ReadFiles stage (fn[0]); drop the Project (fn[1:])
            self._upstream_transformer = MapTransformer([_orig_fns[0]])

            def _q1_block_transformer(block):
                import numba_agg
                cols = block.schema.names
                # raw TPC-H lineitem positions among the read columns
                def pick(cands):
                    for c in cands:
                        if c in cols:
                            return c
                    return None
                qty = pick(["column04", "l_quantity"])
                ext = pick(["column05", "l_extendedprice"])
                dis = pick(["column06", "l_discount"])
                tax = pick(["column07", "l_tax"])
                rf = pick(["column08", "l_returnflag"])
                ls = pick(["column09", "l_linestatus"])
                if None in (qty, ext, dis, tax, rf, ls):
                    print(f"[Q1_FUSED] cols missing in {cols}", flush=True)
                    raise RuntimeError("q1 fused: raw columns not found")
                out = numba_agg.q1_fused_map(block, qty, ext, dis, tax, rf, ls)
                if out is None:
                    raise RuntimeError("q1 fused: kernel declined")
                return out

            self._block_transformer = _q1_block_transformer
            print("[Q1_FUSED] Project stripped, aggregate fused into numba loop",
                  flush=True)
        self._bundles_per_task = int(
            os.environ.get("RAY_DATA_FUSED_READ_MAP_BUNDLES", "12")
        )
        self._fused_task_memory = int(
            os.environ.get("RAY_DATA_FUSED_READ_MAP_MEMORY", str(3 * 1024**3))
        )
        # Per-task sizing from the manifest's __file_size. The flat value above
        # becomes the CAP, so turning this on can only lower a request.
        self._size_from_manifest = (
            os.environ.get("RAY_DATA_FUSED_READ_MAP_MEMORY_FROM_MANIFEST", "0") == "1"
        )
        self._memory_factor = float(
            os.environ.get("RAY_DATA_FUSED_READ_MAP_MEMORY_FACTOR", "1.0")
        )
        self._memory_floor = int(
            os.environ.get("RAY_DATA_FUSED_READ_MAP_MEMORY_FLOOR", str(256 * 1024**2))
        )
        self._manifest_bytes: Dict[Any, int] = {}
        # Running mean of submitted per-task memory, for the backpressure hook.
        self._submitted_memory_sum = 0
        self._submitted_memory_n = 0
        self._pending_bundles: List[RefBundle] = []
        self._cap_cached = _UNSET

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        assert input_index == 0
        self._pending_bundles.append(refs)
        if len(self._pending_bundles) >= self._bundles_per_task:
            batch, self._pending_bundles = self._pending_bundles, []
            self._submit_fused_read_map_task(batch)

    def all_inputs_done(self) -> None:
        PhysicalOperator.all_inputs_done(self)
        if self._pending_bundles:
            batch, self._pending_bundles = self._pending_bundles, []
            self._submit_fused_read_map_task(batch)
        self._maybe_emit_partition_bundles()

    def _estimate_task_memory(self, block_refs: List[Any]) -> int:
        """Size this task's memory request from the files it will actually read.

        The flat ``RAY_DATA_FUSED_READ_MAP_MEMORY`` hands the same reservation
        to a task reading a gigabyte of lineitem and to one that reads nation
        in 0.3 seconds, so the small-table reads sit on budget they never use.
        Measured on q9: the six read+map operators held 762 / 36 / 39 / 18 / 3
        / 3 GiB of reservation for 4362 / 454 / 415 / 114 / 53 / 0.3 busy
        seconds of work, and the cluster pinned at floor(38.4 / 3) * 8 = 96
        concurrent tasks against 128 cores for the whole stage.

        The ListFiles manifest already carries ``__file_size``, so the input is
        known before the task is submitted. Peak is roughly the decoded input
        plus the gather's copy of it: ``hash_partition`` takes once over the
        block, and the staged shards are zero-copy slices pinning that same
        output until ``flush_all``. For q9's lineitem that is ~2.1 GiB peak
        against 2.47 GiB of whole-file bytes, so a factor of 1.0 on the file
        size is a safe over-estimate; a column-pruned scan fetches less than
        the whole file, which is slack in the right direction.

        Falls back to the flat value if the manifest cannot be read, and never
        requests MORE than it, so this can only widen concurrency.
        """
        if not self._size_from_manifest:
            return self._fused_task_memory
        try:
            total = 0
            for ref in block_refs:
                if ref not in self._manifest_bytes:
                    blk = ray.get(ref)
                    tbl = (
                        blk
                        if isinstance(blk, pa.Table)
                        else BlockAccessor.for_block(blk).to_arrow()
                    )
                    if "__file_size" not in tbl.column_names:
                        return self._fused_task_memory
                    # One row per chunk for a chunked file, each carrying the
                    # whole file's size; dedupe so a chunked file counts once.
                    seen, acc = set(), 0
                    for pth, sz in zip(
                        tbl.column("__path").to_pylist(),
                        tbl.column("__file_size").to_pylist(),
                    ):
                        if pth not in seen:
                            seen.add(pth)
                            acc += int(sz or 0)
                    self._manifest_bytes[ref] = acc
                total += self._manifest_bytes[ref]
        except Exception:  # noqa: BLE001 - never let sizing break submission
            return self._fused_task_memory
        if total <= 0:
            return self._fused_task_memory
        est = int(total * self._memory_factor)
        return max(self._memory_floor, min(est, self._fused_task_memory))

    def _submit_fused_read_map_task(self, bundles: List[RefBundle]) -> None:
        cur_task_idx = self._next_shuffle_map_task_idx
        self._next_shuffle_map_task_idx += 1

        block_refs = [ref for b in bundles for ref in b.block_refs]
        task_memory = self._estimate_task_memory(block_refs)
        self._submitted_memory_sum += task_memory
        self._submitted_memory_n += 1
        resources: Dict[str, Any] = {
            "num_cpus": self._shuffle_map_task_num_cpus,
            "memory": task_memory,
        }
        ray_options: Dict[str, Any] = {**resources, "name": self.name}
        if self._map_runtime_env is not None:
            ray_options["runtime_env"] = self._map_runtime_env
        # What each task ACTUALLY asks for. RAY_DATA_FUSED_READ_MAP_MEMORY is a
        # CAP, not the value: _estimate_task_memory sizes from the manifest's
        # __file_size, so lowering the cap may change nothing. Concurrency has
        # been stuck at 74-86 through every knob, and `cluster memory / request`
        # is the one admission limit never actually measured -- this prints the
        # request so the arithmetic can be checked instead of assumed.
        if os.environ.get("RAY_DATA_LOG_TASK_RESOURCES") == "1" and (
            cur_task_idx % 50 == 0
        ):
            import ray as _ray

            _mem = _ray.cluster_resources().get("memory", 0)
            print(
                f"[TASK_RES] op={self.name[:34]!r} idx={cur_task_idx} "
                f"num_cpus={self._shuffle_map_task_num_cpus} "
                f"memory={task_memory / 2 ** 30:.2f}GiB "
                f"blocks={len(block_refs)} "
                f"cluster_memory={_mem / 2 ** 30:.1f}GiB "
                f"=> admits {_mem / max(task_memory, 1):.0f} tasks",
                flush=True,
            )

        map_task_context = TaskContext(task_idx=cur_task_idx, op_name=self.name)

        handle_ref = _disk_shuffle_fused_upstream_map_task.options(
            **ray_options
        ).remote(
            *block_refs,
            upstream_transformer=self._upstream_transformer,
            map_task_context=map_task_context,
            data_context=self.data_context,
            partition_fn=self._partition_fn,
            num_partitions=self._num_partitions,
            out_dir=self._map_dir,
            map_id=cur_task_idx,
            shuffle_id=self._shuffle_id,
            compression=self.data_context.shuffle_compression,
            block_transformer=self._block_transformer,
        )

        task = MetadataOpTask(
            task_index=cur_task_idx,
            object_ref=handle_ref,
            task_done_callback=functools.partial(
                self._handle_map_done, cur_task_idx, handle_ref, bundles
            ),
            task_resource_bundle=ExecutionResources.from_resource_dict(resources),
        )
        self._shuffle_map_tasks[cur_task_idx] = task
        requested = task.get_requested_resource_bundle()
        if requested is not None:
            self._map_resource_usage = self._map_resource_usage.add(requested)
        self._metrics.on_task_submitted(
            cur_task_idx,
            RefBundle((), schema=None, owns_blocks=False),
            task_id=task.get_task_id(),
        )

    def can_add_input(self) -> bool:
        """Return False once this operator is at its configured task cap.

        The executor consults this before handing the operator another bundle
        (streaming_executor_state.py: `op.can_add_input()`), so refusing here
        leaves the slot for another operator rather than queueing work inside
        this one. Uncapped operators keep the base behaviour.
        """
        cap = self._read_concurrency_cap()
        if cap is None:
            return True
        return len(self._shuffle_map_tasks) < cap

    def _read_concurrency_cap(self):
        if self._cap_cached is _UNSET:
            self._cap_cached = None
            for pat, cap in _READ_CONCURRENCY_CAPS:
                if pat.search(self.name):
                    self._cap_cached = cap
                    logger.info(
                        "Read concurrency cap %d applied to %s", cap, self.name
                    )
                    break
        return self._cap_cached

    def get_active_tasks(self) -> List[OpTask]:
        return list(self._shuffle_map_tasks.values())

    def has_execution_finished(self) -> bool:
        if (
            self._shuffle_map_tasks
            or self._pending_bundles
            or not self._partition_bundles_emitted
            or self._output_queue.has_next()
        ):
            return False
        return PhysicalOperator.has_execution_finished(self)

    def has_completed(self) -> bool:
        return (
            not self._shuffle_map_tasks
            and not self._pending_bundles
            and self._partition_bundles_emitted
            and not self._output_queue.has_next()
            and PhysicalOperator.has_completed(self)
        )

    def incremental_resource_usage(self) -> ExecutionResources:
        # Asked BEFORE a task is submitted, so it cannot know that task's own
        # manifest-derived size. Report the running mean of what has been
        # submitted, which converges within the first few tasks; until then the
        # flat value, which is the cap and therefore the conservative answer.
        memory = (
            int(self._submitted_memory_sum / self._submitted_memory_n)
            if self._submitted_memory_n
            else self._fused_task_memory
        )
        return ExecutionResources(
            cpu=self._shuffle_map_task_num_cpus, memory=memory
        )

    def min_scheduling_resources(self) -> ExecutionResources:
        return self.incremental_resource_usage()

    def num_output_rows_total(self) -> Optional[int]:
        return None

    def progress_str(self) -> str:
        done = self._next_shuffle_map_task_idx - len(self._shuffle_map_tasks)
        return f"fused read+map: {done}/{self._next_shuffle_map_task_idx}"
