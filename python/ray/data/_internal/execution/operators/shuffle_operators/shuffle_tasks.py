"""Shared remote tasks + helpers for ShuffleMapOp / ShuffleReduceOp."""

import logging
import os
import pickle
import time
from dataclasses import replace
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import pyarrow as pa

import ray
from ray import ObjectRef
from ray._raylet import (
    StreamingGeneratorStats,  # pyrefly: ignore[missing-module-attribute]
)
from ray.data._internal.output_buffer import BlockOutputBuffer, OutputBlockSizeOption
from ray.data._internal.table_block import TableBlockAccessor
from ray.data.block import (
    Block,
    BlockAccessor,
    BlockExecStats,
    BlockMetadataWithSchema,
    BlockType,
    TaskExecWorkerStats,
)

logger = logging.getLogger(__name__)

# Set RAY_DATA_PROFILE_SHUFFLE_REDUCE=1 to log a per-reduce-task timing
# breakdown (shard fetch/restore vs decode vs reduce+emit).
_PROFILE_REDUCE = bool(int(os.environ.get("RAY_DATA_PROFILE_SHUFFLE_REDUCE", "0")))

PartitionFn = Callable[[pa.Table], Dict[int, pa.Table]]
ReduceFn = Callable[[int, List[pa.Table]], Iterable[Block]]

# When RAY_DATA_LOG_MAP_WORKER_IDLE=1, each map task logs how long this worker
# sat idle since it finished its previous map task.  Map workers are reused
# (the task has no max_calls), so this measures the read->map delivery gap:
# idle ~0 means the worker ran back-to-back (map is throughput/CPU-bound), large
# idle means the worker waited for input between tasks (map is starved).
_LOG_MAP_WORKER_IDLE = bool(int(os.environ.get("RAY_DATA_LOG_MAP_WORKER_IDLE", "0")))
# Per-worker-process timestamp (monotonic) of when this worker's previous map
# task finished. None until this worker has run one map task.
_LAST_MAP_TASK_END_S: Optional[float] = None
# Running totals per worker, so the gap can be read as a fraction of wall time.
_MAP_TASK_COUNT: int = 0
_MAP_IDLE_TOTAL_S: float = 0.0


def _ipc_write_options(compression: Optional[str]) -> pa.ipc.IpcWriteOptions:
    """Arrow IPC write options for the given shard compression codec.

    Args:
        compression: A pyarrow codec name such as "lz4" or "zstd", or "none"
            (or None) to write shards uncompressed. See pyarrow.Codec for the
            full list of supported codecs:
            https://arrow.apache.org/docs/python/generated/pyarrow.Codec.html

    Returns:
        IpcWriteOptions for encoding shards; no compression for "none"/None.
    """
    if not compression or compression == "none":
        return pa.ipc.IpcWriteOptions()
    return pa.ipc.IpcWriteOptions(compression=pa.Codec(compression))


def _partition_blocks_to_shards(
    blocks: Tuple[Block, ...], partition_fn: PartitionFn
) -> Dict[int, List[pa.Table]]:
    """Partition each block independently, grouping shards by partition id.

    Blocks are partitioned one at a time rather than concatenated first, so we
    never hold a single table spanning all inputs in memory.

    We combine_chunks before partitioning because partition_fn's per-column
    take is much slower on chunked input -- enough to roughly halve map
    throughput.
    """
    partition_accumulators: Dict[int, List[pa.Table]] = {}
    for block in blocks:
        block = TableBlockAccessor.try_convert_block_type(
            block, block_type=BlockType.ARROW
        )
        if block.num_rows == 0:
            continue
        assert isinstance(block, pa.Table), f"Expected pa.Table, got {type(block)}"
        if any(col.num_chunks > 1 for col in block.columns):
            block = block.combine_chunks()
        block_partitions = partition_fn(block)
        for partition_id, shard in block_partitions.items():
            if shard.num_rows > 0:
                partition_accumulators.setdefault(partition_id, []).append(shard)
        del block, block_partitions
    return partition_accumulators


def _encode_partition_ipc(
    table: pa.Table,
    ipc_write_options: pa.ipc.IpcWriteOptions,
) -> pa.Buffer:
    """Encode one partition's shard as a single Arrow IPC stream."""
    if table.num_columns > 0:
        table = table.combine_chunks()

    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema, options=ipc_write_options)
    for batch in table.to_batches():
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue()


@ray.remote  # pyrefly: ignore[no-matching-overload]
def _shuffle_map_task(
    *blocks: Block,
    partition_fn: PartitionFn,
    num_partitions: int,
    compression: Optional[str],
    limit_actor: Optional["ray.actor.ActorHandle"] = None,
    limit_task_id: Optional[int] = None,
):
    """Map stage: partition the input blocks and return one shard per partition.

    When ``limit_actor`` is set, a ``Limit`` has been fused into this shuffle:
    before partitioning, the task claims a slice of the global row budget from
    the counter actor and truncates its blocks accordingly, so the union of all
    map outputs respects the limit.  ``limit_done`` is returned in the metadata
    tuple so the operator can stop submitting tasks once the budget is spent.
    """
    if _LOG_MAP_WORKER_IDLE:
        global _LAST_MAP_TASK_END_S, _MAP_TASK_COUNT, _MAP_IDLE_TOTAL_S
        _task_start_s = time.monotonic()
        if _LAST_MAP_TASK_END_S is not None:
            idle_s = _task_start_s - _LAST_MAP_TASK_END_S
            _MAP_TASK_COUNT += 1
            _MAP_IDLE_TOTAL_S += idle_s
            # print (not logger) so it forwards to the driver across all nodes.
            print(
                f"[MAP_IDLE] pid={os.getpid()} idle_s={idle_s:.3f} "
                f"tasks={_MAP_TASK_COUNT} idle_total_s={_MAP_IDLE_TOTAL_S:.3f}",
                flush=True,
            )

    stats = BlockExecStats.builder()

    # Schema is derived from the (untruncated) input so empty shards still carry
    # it even when this task claims zero rows under the fused limit.
    ipc_write_options = _ipc_write_options(compression)
    output_schema = TableBlockAccessor.try_convert_block_type(
        blocks[0], block_type=BlockType.ARROW
    ).schema
    empty_shard = _encode_partition_ipc(output_schema.empty_table(), ipc_write_options)

    # Fused-limit path: claim a slice of the global row budget and truncate this
    # task's blocks to it before partitioning.  See distributed_limit_counter.
    limit_done = False
    if limit_actor is not None:
        from ray.data._internal.execution.operators.shuffle_operators.distributed_limit_counter import (  # noqa: E501
            claim_and_truncate,
        )

        num_rows_in = sum(BlockAccessor.for_block(b).num_rows() for b in blocks)
        truncated, limit_done = claim_and_truncate(
            limit_actor, limit_task_id, blocks, num_rows_in
        )
        blocks = tuple(truncated)

    # Use BlockAccessor so we also work for non-Arrow blocks (pandas, numpy)
    accessors = [BlockAccessor.for_block(b) for b in blocks]
    total_rows = sum(a.num_rows() for a in accessors)
    total_bytes = sum((a.size_bytes() or 0) for a in accessors)

    partition_accumulators = (
        {} if total_rows == 0 else _partition_blocks_to_shards(blocks, partition_fn)
    )

    shard_sizes: Dict[int, Tuple[int, int]] = {}
    partition_bufs: List[pa.Buffer] = []
    for partition_id in range(num_partitions):
        tables = partition_accumulators.pop(partition_id, None)
        if not tables:
            partition_bufs.append(empty_shard)
            continue
        merged = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        shard_sizes[partition_id] = (merged.num_rows, merged.nbytes)
        partition_bufs.append(_encode_partition_ipc(merged, ipc_write_options))
        del merged

    # ``blocks`` may be empty when the fused limit claimed zero rows; fall back
    # to an empty table of the output schema so we still produce metadata.
    meta_block = blocks[0] if blocks else output_schema.empty_table()
    input_meta = BlockAccessor.for_block(meta_block).get_metadata(
        block_exec_stats=stats.build(block_ser_time_s=0),
    )
    input_meta = replace(input_meta, num_rows=total_rows, size_bytes=total_bytes)
    if _LOG_MAP_WORKER_IDLE:
        _LAST_MAP_TASK_END_S = time.monotonic()
    return (input_meta, shard_sizes, output_schema, limit_done), *partition_bufs


def _read_partition_ipc(buf: pa.Buffer) -> Optional[pa.Table]:
    """Decompress one partition shard."""
    if len(buf) == 0:
        return None
    reader = pa.ipc.open_stream(buf)
    schema = reader.schema
    batches: List[pa.RecordBatch] = []
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        if batch.num_rows > 0:
            batches.append(batch)
    return pa.Table.from_batches(batches, schema=schema)


@ray.remote(max_calls=1)
def _shuffle_reduce_task(
    shard_refs: List[ObjectRef],
    partition_id: int,
    reduce_fn: ReduceFn,
    target_max_block_size: Optional[int],
    streaming: bool,
    batch_size: int,
) -> Generator[Union[Block, bytes], None, None]:
    """Reduce stage: fetch one partition's shards and run reduce_fn over them.

    With streaming=True, reduce_fn is called each time the accumulated input
    passes target_max_block_size and its output is reshaped to that size via a
    BlockOutputBuffer; this bounds peak input memory but requires reduce_fn to
    produce valid output from partial input.  With streaming=False, all shards
    are accumulated and reduce_fn is called once, use this when it needs the
    whole partition (sort, aggregate).

    Args:
        shard_refs: ObjectRefs to this partition's IPC shards from every mapper.
            May contain None for mappers that produced no rows here.
        partition_id: Partition this reducer owns.
        reduce_fn: User-supplied reduce callable.
        target_max_block_size: Output block size, and the streaming flush
            threshold.  None emits blocks as-is (no reshaping, no streaming
            flush) -- the "partition = block" contract.
        streaming: Flush incrementally (True) or accumulate then reduce (False).
        batch_size: Number of shard refs to ray.get() at a time.
    """
    start_time_s = time.perf_counter()

    accum_tables: List[pa.Table] = []
    accum_bytes: int = 0
    output_buffer: Optional[BlockOutputBuffer] = None

    # -- Reduce profiling counters (only emitted when _PROFILE_REDUCE) --------
    # fetch_s    : time blocked in ray.get() pulling shards (remote transfer +
    #              spill restore from disk) -- the suspected reduce bottleneck.
    # decode_s   : time decoding shard IPC buffers into Arrow tables.
    # The reduce_fn + output-emit cost is (wall - fetch - decode), and Ray
    # separately reports output-backpressure wait outside this task.
    prof_fetch_s: float = 0.0
    prof_max_batch_fetch_s: float = 0.0
    prof_decode_s: float = 0.0
    prof_num_shards: int = 0
    prof_input_bytes: int = 0
    prof_first_fetch_at_s: Optional[float] = None

    def _yield_with_stats(block: Block):
        """Yield (block, pickled metadata) following the streaming-gen protocol."""
        exec_stats_builder = BlockExecStats.builder()
        exec_stats_builder.finish()
        gen_stats: StreamingGeneratorStats = yield block
        exec_stats = exec_stats_builder.build(
            block_ser_time_s=(gen_stats.object_creation_dur_s if gen_stats else None),
        )
        yield pickle.dumps(
            BlockMetadataWithSchema.from_block(
                block,
                block_exec_stats=exec_stats,
                task_exec_stats=TaskExecWorkerStats(
                    task_wall_time_s=time.perf_counter() - start_time_s,
                ),
            )
        )

    def _flush(tables: List[pa.Table]):
        nonlocal output_buffer
        if output_buffer is None:
            output_buffer = BlockOutputBuffer(
                OutputBlockSizeOption.of(
                    target_max_block_size=target_max_block_size,
                )
            )
        for block in reduce_fn(partition_id, tables):
            output_buffer.add_block(block)
            while output_buffer.has_next():
                yield from _yield_with_stats(output_buffer.next())

    # Step 1: fetch shard refs in batches, decompress, accumulate.  In
    # streaming mode, when the accumulator reaches target_max_block_size,
    # flush through reduce_fn and yield any ready output blocks.
    for batch_start in range(0, len(shard_refs), batch_size):
        batch = shard_refs[batch_start : batch_start + batch_size]
        fetch_t0 = time.perf_counter()
        bufs = ray.get(batch)
        batch_fetch_s = time.perf_counter() - fetch_t0
        prof_fetch_s += batch_fetch_s
        prof_max_batch_fetch_s = max(prof_max_batch_fetch_s, batch_fetch_s)
        if prof_first_fetch_at_s is None:
            prof_first_fetch_at_s = fetch_t0 - start_time_s
        for buf in bufs:
            if buf is None:
                continue
            decode_t0 = time.perf_counter()
            table = _read_partition_ipc(buf)
            prof_decode_s += time.perf_counter() - decode_t0
            if table is None:
                continue
            prof_num_shards += 1
            prof_input_bytes += table.nbytes
            accum_tables.append(table)
            accum_bytes += table.nbytes

            if (
                streaming
                and target_max_block_size is not None
                and accum_bytes >= target_max_block_size
            ):
                tables, accum_tables = accum_tables, []
                accum_bytes = 0
                yield from _flush(tables)

    # Step 2: drain remaining shards through reduce_fn.  This is the only
    # reduce_fn call in blocking mode, and the tail-flush in streaming mode.
    if accum_tables:
        yield from _flush(accum_tables)

    # Step 3: if reduce_fn ran at least once, finalize the buffer to flush
    # any partial block.
    if output_buffer is not None:
        output_buffer.finalize()
        while output_buffer.has_next():
            yield from _yield_with_stats(output_buffer.next())

    if _PROFILE_REDUCE:
        wall_s = time.perf_counter() - start_time_s
        # reduce_fn + emit (incl. output-backpressure wait at the yield points).
        reduce_emit_s = max(0.0, wall_s - prof_fetch_s - prof_decode_s)
        logger.info(
            "[shuffle_reduce] partition=%d shards=%d input_mb=%.1f wall_s=%.2f "
            "fetch_s=%.2f (%.0f%%) max_batch_fetch_s=%.2f first_fetch_at_s=%.2f "
            "decode_s=%.2f (%.0f%%) reduce_emit_s=%.2f (%.0f%%)",
            partition_id,
            prof_num_shards,
            prof_input_bytes / 1e6,
            wall_s,
            prof_fetch_s,
            100.0 * prof_fetch_s / wall_s if wall_s else 0.0,
            prof_max_batch_fetch_s,
            prof_first_fetch_at_s or 0.0,
            prof_decode_s,
            100.0 * prof_decode_s / wall_s if wall_s else 0.0,
            reduce_emit_s,
            100.0 * reduce_emit_s / wall_s if wall_s else 0.0,
        )
