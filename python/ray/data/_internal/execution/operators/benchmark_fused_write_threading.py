"""Microbenchmark: threaded vs sequential datasink write in fused map+write.

Tests whether pipelining the datasink chain in a background thread overlaps
map compute with write I/O, reducing total wall-clock time.

Two levels of benchmarks:
  1. Unit-level: directly exercises MapTransformer.apply_transform with
     simulated CPU (map) and I/O (write) delays.  Clean and reproducible.
  2. End-to-end: full Ray Data pipeline with map_batches + write_parquet.

Usage:
    # Unit benchmark only (no Ray required):
    python benchmark_fused_write_threading.py --unit

    # Full end-to-end benchmark:
    python benchmark_fused_write_threading.py

    # Tune parameters:
    python benchmark_fused_write_threading.py --blocks 16 --cpu-ms 80 --io-ms 60
"""
import argparse
import sys
import time

sys.path.insert(0, "python")


# ---------------------------------------------------------------------------
# Unit-level benchmark — no Ray runtime required
# ---------------------------------------------------------------------------


def _run_unit(n_blocks: int, cpu_ms: float, io_ms: float, threaded: bool) -> float:
    """Measure time for a chain of [map_fn, write_fn] applied to n_blocks items.

    map_fn sleeps for cpu_ms (simulates CPU-bound map).
    write_fn sleeps for io_ms (simulates disk I/O in write).

    With threading: map and write overlap → total ≈ n_blocks * max(cpu_ms, io_ms)
    Without threading: serial → total ≈ n_blocks * (cpu_ms + io_ms)

    Tests the threading mechanism directly via _apply_transform_with_write_thread
    without going through Ray Data's block validation layer.
    """
    from ray.data._internal.execution.interfaces.task_context import TaskContext
    from ray.data._internal.execution.operators.map_transformer import MapTransformer

    ctx = TaskContext(task_idx=0, op_name="bench")

    # Build lightweight callables that just sleep — no block type validation.
    def _sleep_fn(delay_s):
        def _fn(items, ctx):
            for item in items:
                time.sleep(delay_s)
                yield item

        return _fn

    map_callable = _sleep_fn(cpu_ms / 1000)
    write_callable = _sleep_fn(io_ms / 1000)

    # Directly exercise the threading path by building a transformer with two
    # plain callables and the datasink_transform_start_idx set appropriately.
    # We bypass BlockMapTransformFn's block validation intentionally.
    transformer = MapTransformer.__new__(MapTransformer)
    transformer._transform_fns = [map_callable, write_callable]
    transformer._init_fn = lambda: None
    transformer._output_block_size_option_override = None
    transformer._udf_time_s = 0
    transformer._datasink_transform_start_idx = 1 if threaded else None

    # Wrap callables so they look like MapTransformFn objects to _UDFTimingIterator.
    for fn in transformer._transform_fns:
        fn._is_udf = False

    blocks = list(range(n_blocks))

    t0 = time.perf_counter()
    if threaded:
        list(transformer._apply_transform_with_write_thread(iter(blocks), ctx))
    else:
        # Sequential: chain the two callables directly.
        result = iter(blocks)
        for fn in transformer._transform_fns:
            result = fn(result, ctx)
        list(result)
    return time.perf_counter() - t0


def run_unit_benchmark(
    n_blocks: int = 8,
    cpu_ms: float = 80.0,
    io_ms: float = 60.0,
    repeats: int = 3,
):
    print("=" * 60)
    print("Unit benchmark (simulated CPU + I/O delays)")
    print(f"  {n_blocks} blocks, map={cpu_ms}ms/block, write={io_ms}ms/block")
    print(
        f"  theoretical: sequential={n_blocks*(cpu_ms+io_ms):.0f}ms, "
        f"threaded≈{n_blocks*max(cpu_ms,io_ms):.0f}ms"
    )
    print("-" * 60)

    seq_times, thr_times = [], []
    for rep in range(repeats):
        s = _run_unit(n_blocks, cpu_ms, io_ms, threaded=False)
        t = _run_unit(n_blocks, cpu_ms, io_ms, threaded=True)
        seq_times.append(s)
        thr_times.append(t)
        print(
            f"  rep {rep+1}: sequential={s*1000:.0f}ms  "
            f"threaded={t*1000:.0f}ms  speedup={s/t:.2f}x"
        )

    avg_s = sum(seq_times) / len(seq_times)
    avg_t = sum(thr_times) / len(thr_times)
    print("-" * 60)
    print(
        f"  avg:     sequential={avg_s*1000:.0f}ms  "
        f"threaded={avg_t*1000:.0f}ms  speedup={avg_s/avg_t:.2f}x"
    )
    print()
    return avg_s, avg_t


# ---------------------------------------------------------------------------
# End-to-end benchmark — requires Ray
# ---------------------------------------------------------------------------


def _run_e2e(n_blocks: int, block_mb: int, threaded: bool, output_dir: str) -> float:
    """Run map_batches (CPU work) + write_parquet and return wall-clock seconds."""
    import os

    import numpy as np
    import pyarrow as pa

    import ray.data as rd

    # Toggle threading by patching the datasink_transform_start_idx.
    import ray.data._internal.execution.operators.map_transformer as _mt

    _orig_init = _mt.MapTransformer.__init__

    def _patched_init(self, *args, datasink_transform_start_idx=None, **kwargs):
        _orig_init(
            self,
            *args,
            datasink_transform_start_idx=(
                datasink_transform_start_idx if threaded else None
            ),
            **kwargs,
        )

    _mt.MapTransformer.__init__ = _patched_init
    try:
        rows_per_block = max(1, (block_mb * 1024 * 1024) // 8)
        blocks = [
            pa.table(
                {"v": np.random.randint(0, 2**31, rows_per_block, dtype=np.int64)}
            )
            for _ in range(n_blocks)
        ]
        ds = rd.from_arrow(blocks)

        def cpu_heavy_map(batch):
            # Burn ~50ms of actual CPU via numpy sort (I/O-independent work).
            arr = batch["v"].to_pylist()
            arr.sort()
            return {"v": arr}

        out = os.path.join(output_dir, "thr" if threaded else "seq")
        os.makedirs(out, exist_ok=True)

        t0 = time.perf_counter()
        ds.map_batches(cpu_heavy_map, batch_format="pyarrow").write_parquet(out)
        return time.perf_counter() - t0
    finally:
        _mt.MapTransformer.__init__ = _orig_init


def run_e2e_benchmark(
    n_blocks: int = 8,
    block_mb: int = 32,
    repeats: int = 3,
):
    import tempfile

    import ray

    print("=" * 60)
    print("End-to-end benchmark (Ray Data map_batches + write_parquet)")
    print(f"  {n_blocks} blocks × {block_mb} MB")
    print("-" * 60)

    ray.init(ignore_reinit_error=True)
    seq_times, thr_times = [], []

    with tempfile.TemporaryDirectory() as tmpdir:
        for rep in range(repeats):
            s = _run_e2e(n_blocks, block_mb, threaded=False, output_dir=tmpdir)
            t = _run_e2e(n_blocks, block_mb, threaded=True, output_dir=tmpdir)
            seq_times.append(s)
            thr_times.append(t)
            print(
                f"  rep {rep+1}: sequential={s:.2f}s  "
                f"threaded={t:.2f}s  speedup={s/t:.2f}x"
            )

    avg_s = sum(seq_times) / len(seq_times)
    avg_t = sum(thr_times) / len(thr_times)
    print("-" * 60)
    print(
        f"  avg:     sequential={avg_s:.2f}s  "
        f"threaded={avg_t:.2f}s  speedup={avg_s/avg_t:.2f}x"
    )
    ray.shutdown()
    print()
    return avg_s, avg_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark: threaded vs sequential datasink write"
    )
    parser.add_argument("--unit", action="store_true", help="Unit benchmark only")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-mb", type=int, default=32, help="End-to-end only")
    parser.add_argument("--cpu-ms", type=float, default=80.0, help="Unit only")
    parser.add_argument("--io-ms", type=float, default=60.0, help="Unit only")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    run_unit_benchmark(
        n_blocks=args.blocks,
        cpu_ms=args.cpu_ms,
        io_ms=args.io_ms,
        repeats=args.repeats,
    )

    if not args.unit:
        run_e2e_benchmark(
            n_blocks=args.blocks,
            block_mb=args.block_mb,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
