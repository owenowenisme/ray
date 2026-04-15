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
    """One-time worker init: switch Arrow to system allocator and tune glibc
    malloc to return freed pages to the OS more aggressively.
    """
    global _worker_memory_initialized
    if _worker_memory_initialized:
        return
    _worker_memory_initialized = True

    import pyarrow as pa

    old_pool = pa.default_memory_pool().backend_name
    if old_pool != "system":
        pa.set_memory_pool(pa.system_memory_pool())

    mallopt_results = {}
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.mallopt.restype = ctypes.c_int

        M_MMAP_THRESHOLD = -3
        M_TRIM_THRESHOLD = -1
        M_ARENA_MAX = -8

        mallopt_results["mmap_thresh"] = libc.mallopt(M_MMAP_THRESHOLD, 128 * 1024)
        mallopt_results["trim_thresh"] = libc.mallopt(M_TRIM_THRESHOLD, 128 * 1024)
        mallopt_results["arena_max"] = libc.mallopt(M_ARENA_MAX, 2)
    except (OSError, AttributeError) as e:
        mallopt_results["error"] = str(e)

    _logger.info(
        f"Worker memory init: Arrow pool {old_pool}->system, "
        f"mallopt={mallopt_results}"
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
