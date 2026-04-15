import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List

import ray
from ray.data.block import BlockAccessor, CallableClass

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import RefBundle

_logger = logging.getLogger(__name__)

_worker_memory_initialized = False


def init_worker_memory():
    """Configure Arrow's memory pool for aggressive page return.

    When ARROW_DEFAULT_MEMORY_POOL=jemalloc is set (recommended),
    this calls pa.jemalloc_set_decay_ms(0) to force jemalloc to
    immediately return freed pages to the OS via madvise(MADV_DONTNEED).
    This is the Arrow-team-recommended approach (apache/arrow#36378).

    Without this, freed Arrow buffers stay in the allocator's cache
    and RSS grows unboundedly across tasks.
    """
    global _worker_memory_initialized
    if _worker_memory_initialized:
        return
    _worker_memory_initialized = True

    import pyarrow as pa

    pool_name = pa.default_memory_pool().backend_name

    if pool_name == "jemalloc":
        try:
            pa.jemalloc_set_decay_ms(0)
            _logger.info("Worker memory init: jemalloc decay_ms=0")
        except Exception as e:
            _logger.warning(f"Worker memory init: jemalloc_set_decay_ms failed: {e}")
    else:
        _logger.info(
            f"Worker memory init: pool={pool_name} (set "
            f"ARROW_DEFAULT_MEMORY_POOL=jemalloc for aggressive page return)"
        )


def get_rss_mb():
    """Return current process RSS in MB."""
    import os

    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (FileNotFoundError, OSError):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_memory_breakdown_mb():
    """Return (rss, private, shared) from /proc/self/smaps_rollup."""
    try:
        with open("/proc/self/smaps_rollup") as f:
            data = f.read()
        vals = {}
        for line in data.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                vals[parts[0].rstrip(":")] = int(parts[1])
        rss = vals.get("Rss", 0) / 1024
        shared = (vals.get("Shared_Clean", 0) + vals.get("Shared_Dirty", 0)) / 1024
        private = (vals.get("Private_Clean", 0) + vals.get("Private_Dirty", 0)) / 1024
        return rss, private, shared
    except (FileNotFoundError, OSError):
        return get_rss_mb(), 0, 0


def release_memory():
    """Force GC + return freed pages to OS via Arrow pool purge and malloc trim."""
    import ctypes
    import gc

    import pyarrow as pa

    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def make_ref_bundles(simple_data: List[List[Any]]) -> List["RefBundle"]:
    """Create ref bundles from a list of block data.

    One bundle is created for each input block.
    """
    import pandas as pd
    import pyarrow as pa

    from ray.data._internal.execution.interfaces import RefBundle

    output = []
    for block in simple_data:
        block = pd.DataFrame({"id": block})
        output.append(
            RefBundle(
                [
                    (
                        ray.put(block),
                        BlockAccessor.for_block(block).get_metadata(),
                    )
                ],
                owns_blocks=True,
                schema=pa.lib.Schema.from_pandas(block, preserve_index=False),
            )
        )
    return output


memory_units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]


def memory_string(num_bytes: float) -> str:
    """Return a human-readable memory string for the given amount of bytes."""
    k = 0
    while num_bytes >= 1024 and k < len(memory_units) - 1:
        num_bytes /= 1024
        k += 1
    return f"{num_bytes:.1f}{memory_units[k]}"


def locality_string(locality_hits: int, locality_misses) -> str:
    """Return a human-readable string for object locality stats."""
    if not locality_misses:
        return "[all objects local]"
    return f"[{locality_hits}/{locality_hits + locality_misses} objects local]"


def make_callable_class_single_threaded(callable_cls: CallableClass) -> CallableClass:
    """Returns a thread-safe CallableClass with the same logic as the provided
    `callable_cls`.

    This function allows the usage of concurrent actors by safeguarding user logic
    behind a separate thread.

    This allows batch slicing and formatting to occur concurrently, to overlap with the
    user provided UDF.
    """

    class _SingleThreadedWrapper(callable_cls):
        def __init__(self, *args, **kwargs):
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return super().__repr__()

        def __call__(self, *args, **kwargs):
            # ThreadPoolExecutor will reuse the same thread for every submit call.
            future = self.thread_pool_executor.submit(super().__call__, *args, **kwargs)
            return future.result()

    return _SingleThreadedWrapper
