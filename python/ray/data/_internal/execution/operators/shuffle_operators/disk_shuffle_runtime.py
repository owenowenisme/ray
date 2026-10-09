"""Disk-shuffle file-transport runtime: Arrow Flight transport, per-node
ShuffleFileServer actor, prefetch layout, error hierarchy. Imported by the
map/reduce task bodies in ``disk_shuffle_tasks``."""

import errno
import json
import logging
import collections
import os
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import pyarrow as pa

import ray
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.util import MiB
from ray.data._internal.utils.transform_pyarrow import _is_pa_extension_type
from ray.data.block import BlockExecStats
from ray.exceptions import (
    ActorDiedError,
    ActorUnavailableError,
    ActorUnschedulableError,
)

logger = logging.getLogger(__name__)

# ShuffleFileServer actor identity. Name is deterministic in (shuffle_id, node_id).
_SHUFFLE_FILE_SERVER_NAMESPACE = "ray_data_shuffle_disk"

# Endpoint resolution polls while the actor restarts (Ray raises
# ActorUnavailableError): the poll interval, and the warn cadence (in polls) so
# a stuck fetch stays visible.
_ENDPOINT_POLL_INTERVAL_S = 2.0
_ENDPOINT_POLL_WARN_EVERY = 15  # ~30s at the interval above

# Materialized handle metadata per reduce resolve batch.
_DEFAULT_HANDLE_BATCH_BYTES = 64 * MiB

# pyarrow-supported shard codecs (``"none"``/``None`` = uncompressed). Rides
# ``data_context.shuffle_compression``.
Compression = Optional[
    Literal[
        "none", "gzip", "bz2", "brotli", "lz4", "lz4_frame", "lz4_raw", "zstd", "snappy"
    ]
]


# =============================================================================
# SHARED: shard codec, ShuffleFileServer identity/lookup, the
# ShuffleHandle type, and the error hierarchy, which are used by both map and reduce.
# =============================================================================
# Each range's payload length is framed as a u64 in the sink, so a single
# range/IPC frame may be up to 16 EiB. Checked at mapper write time.
_MAX_RANGE_BYTES: int = (1 << 64) - 1


# ----------------------------------------------------------------- Arrow IPC
# Shard wire format: [u64 uncompressed_size][codec(whole IPC stream)]
_WF_HEADER = struct.Struct("<Q")

# EXPERIMENT (RAY_DATA_SHUFFLE_SHARD_FORMAT=batch): a shard frame is currently
# a full IPC STREAM (schema message + batch + EOS). A join-1 reduce partition
# decodes 3,046 of them, re-parsing the same 6-column schema 3,046 times; the
# measured shard-decode pool is the largest CPU sink on the reduce side and
# removing lz4 was already proven neutral, so the stream machinery is the
# suspect. The handle carries the schema once, so write the shard as a
# schema-less record-batch MESSAGE (batch.serialize()) and read it back with
# pa.ipc.read_record_batch(raw, schema). Bit 63 of the u64 header flags the
# format per frame, so stream-format fallbacks (dictionary columns, 0-column
# tables) stay self-describing and both formats can coexist in one file.
_WF_BATCH_FLAG = 1 << 63
_WF_SIZE_MASK = _WF_BATCH_FLAG - 1
# Read lazily inside the task: at module import the worker's os.environ does
# not yet carry the job's env vars (same trap as the plain-read installer).
_shard_format_state = {"checked": False, "batch": False}


def _shard_format_batch() -> bool:
    if not _shard_format_state["checked"]:
        _shard_format_state["checked"] = True
        _shard_format_state["batch"] = (
            os.environ.get("RAY_DATA_SHUFFLE_SHARD_FORMAT") == "batch"
        )
    return _shard_format_state["batch"]


def _codec_for(compression: Optional[str]) -> Optional["pa.Codec"]:
    """Codec name -> pa.Codec; None/"none" -> None (pyarrow has no "none" codec)."""
    if not compression or compression.lower() == "none":
        return None
    return pa.Codec(compression)


def _encode_shard(
    table: pa.Table, compression: Optional[str] = "zstd", combine_native: bool = False
) -> pa.Buffer:
    """Encode a partition shard as a whole-frame blob (one codec frame per shard,
    vs Arrow's per-buffer IPC compression)."""
    if table.num_columns > 0:
        if combine_native:
            try:
                table = table.combine_chunks()
            except pa.ArrowInvalid:
                # >2 GiB string offsets overflow int32 during native combine
                # (esp. nested in struct/list); use the extension-safe path.
                table = transform_pyarrow.combine_chunks(table)
        else:
            table = transform_pyarrow.combine_chunks(table)
    if (
        _shard_format_batch()
        and table.num_columns > 0
        and table.num_rows > 0
        and not any(pa.types.is_dictionary(f.type) for f in table.schema)
    ):
        batches = table.to_batches()
        # combine_chunks above leaves exactly one chunk; guard anyway so a
        # multi-batch table falls back to the stream format rather than
        # silently dropping batches.
        if len(batches) == 1:
            raw = batches[0].serialize()
            codec = _codec_for(compression)
            out = pa.BufferOutputStream()
            out.write(_WF_HEADER.pack(raw.size | _WF_BATCH_FLAG))
            out.write(raw if codec is None else codec.compress(raw))
            return out.getvalue()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as w:  # uncompressed inner IPC
        for batch in table.to_batches():
            w.write_batch(batch)
    raw = sink.getvalue()
    codec = _codec_for(compression)
    out = pa.BufferOutputStream()
    out.write(_WF_HEADER.pack(raw.size))
    out.write(raw if codec is None else codec.compress(raw))
    return out.getvalue()


def _read_ipc(
    buf: Union[bytes, "pa.Buffer", memoryview],
    compression: Optional[str] = "zstd",
    schema: Optional[pa.Schema] = None,
) -> pa.Table:
    """Decode a whole-frame shard: read the u64 size header, one decompress of
    the rest (per ``compression``, same value the map encoded with), then read
    the payload. Bit 63 of the header marks a schema-less record-batch frame
    (RAY_DATA_SHUFFLE_SHARD_FORMAT=batch); ``schema`` comes from the
    ShuffleHandle and is required only for those frames."""
    src = buf if isinstance(buf, pa.Buffer) else pa.py_buffer(buf)
    header = _WF_HEADER.unpack_from(memoryview(src))[0]
    n = header & _WF_SIZE_MASK
    body = src.slice(_WF_HEADER.size)
    codec = _codec_for(compression)
    raw = body if codec is None else codec.decompress(body, decompressed_size=n)
    if header & _WF_BATCH_FLAG:
        if schema is None:
            raise ValueError(
                "shard frame is record-batch format but the caller supplied no "
                "schema; the ShuffleHandle schema must be threaded to _read_ipc"
            )
        batch = pa.ipc.read_record_batch(raw, schema)
        return pa.Table.from_batches([batch], schema=schema)
    with pa.ipc.open_stream(raw) as r:
        return r.read_all()


# EXPERIMENT (RAY_DATA_SHUFFLE_FILE_SERVERS_PER_NODE, default 1): run several
# file servers per node instead of one.
#
# Server-side timers put 44% of a file server's handler time in Python that is
# neither the disk read nor the wait for the transport. With ~33 handler
# threads live in ONE actor process, that number cannot distinguish expensive
# Python from threads queueing on the GIL. Sharding the actor separates them:
# if it is GIL queueing, throughput rises with the replica count; if the code
# is simply expensive, it does not.
#
# Every replica on a node serves the same base_dir, so a request can go to any
# of them and no file needs to move.
SHUFFLE_FILE_SERVERS_PER_NODE = max(
    1, int(os.environ.get("RAY_DATA_SHUFFLE_FILE_SERVERS_PER_NODE", "1"))
)


def _file_server_name(shuffle_id: str, node_id: str, replica: int = 0) -> str:
    # Replica 0 keeps the historical name, so the default single-server
    # deployment is byte-identical to before.
    suffix = f":r{replica}" if replica else ""
    return f"shuffle_file_server:{shuffle_id}:{node_id}{suffix}"


class ShuffleHandle(TypedDict, total=False):
    """Handle written by each mapper task.

    Fetch/runtime fields (``path``, ``index_ranges``, ``shuffle_id``,
    ``node_id``) are consumed by reducers. Driver bookkeeping fields
    (``num_rows``, ``decoded_bytes``, ``schema``, ``num_partitions``,
    ``total_bytes``, ``compression``) are consumed by the map operator
    when emitting partition wrappers.
    """

    path: str
    # Dense per-partition range index (see _build_range_index): row ``p`` is
    # ``(offset, length)`` of partition ``p``'s on-disk IPC frame.
    # ``length == 0`` => this mapper wrote no frame for ``p`` (reducer skips
    # the fetch). Not the same as decoded ``nbytes`` or ``num_rows``.
    index_ranges: "np.ndarray"  # shape [num_partitions, 2], int64
    # Dense per-partition row counts (see _counts_to_array). Empty-partition
    # gating uses this, not ``decoded_bytes``: a ``null``-typed table can
    # have ``num_rows > 0`` with ``tbl.nbytes == 0``.
    num_rows: "np.ndarray"  # shape [num_partitions], int64
    # Dense per-partition decoded-byte counts (see _counts_to_array):
    # pre-compression ``tbl.nbytes``. Used for reduce memory estimates.
    decoded_bytes: "np.ndarray"  # shape [num_partitions], int64
    shuffle_id: str
    node_id: str
    schema: Optional["pa.Schema"]
    # Extra mapper bookkeeping (not required for fetch).
    num_partitions: int
    total_bytes: int
    compression: Optional[str]
    exec_stats: Optional["BlockExecStats"]


class _Endpoint(NamedTuple):
    """A file server's Flight endpoint. ``host``/``port`` are where to connect;
    ``incarnation`` identifies the server process so a restart is detectable."""

    host: str
    port: int
    incarnation: str


class ShuffleDiskError(RuntimeError):
    """Terminal: reducer's local disk exhausted (ENOSPC / EDQUOT).
    Retrying doesn't reclaim space."""


# errno values that indicate the reducer's local disk is exhausted.
# EDQUOT is glibc's quota-exceeded error; not all platforms expose it.
_DISK_EXHAUSTED_ERRNOS = frozenset(
    e
    for e in (
        errno.ENOSPC,
        getattr(errno, "EDQUOT", None),
    )
    if e is not None
)


def _is_disk_exhausted(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in _DISK_EXHAUSTED_ERRNOS


class ShuffleFileServerAnomalyError(RuntimeError):
    """Terminal shuffle file-server failure: a driver-level retry cannot fix it.

    We run with ``max_restarts=-1``, ``lifetime="detached"``,
    ``NodeAffinitySchedulingStrategy(node_id, soft=False)``, this means Ray auto-restarts
    the actor on any mid-life crash and surfaces ``ActorUnavailableError``
    during the restart window. So the anomalous states we key off of are:

    - ``ActorDiedError``          -> ``__init__`` raised, or external
                                    ``ray.kill(actor, no_restart=True)``.
    - ``ActorUnschedulableError`` -> pinned node is gone; ``soft=False`` blocks
                                    relocation. Terminal at the shuffle layer;
                                    an upstream lineage layer must re-execute
                                    the mapper on healthy capacity.
    - ``ValueError`` (from ``ray.get_actor``) -> actor name is not registered
                                                (never created or gc'd).
    - The Flight fetch failed but the server's incarnation is unchanged and Ray
      RPC still works, which is often a network-configuration problem
      (``NetworkPolicy``, firewall, routing).
    """


# =============================================================================
# MAP SIDE (map tasks): buffer per-partition shards, seal them to one file, and
# build the ShuffleHandle's dense range index plus per-partition row/byte
# counts. What this writes is exactly what the SERVER serves and the FETCH
# side reads.
# =============================================================================
def _build_range_index(index, num_partitions):
    """Dense per-partition (offset, length) index: one row per partition.

    Row ``p`` is ``(offset, length)`` of partition ``p``'s on-disk IPC frame;
    mappers that wrote nothing for ``p`` stay ``(0, 0)`` (``length == 0`` =>
    the reducer skips the fetch).
    """
    ranges = np.zeros((num_partitions, 2), dtype=np.int64)
    for partition_id, frame_range in index.items():
        ranges[partition_id] = frame_range[0]
    return ranges


def _counts_to_array(counts, num_partitions):
    """Dense per-partition int64 counts, indexed by partition_id.

    Absent partitions stay 0. Used for both ``num_rows`` (``tbl.num_rows``)
    and ``decoded_bytes`` (``tbl.nbytes``).
    """
    arr = np.zeros(num_partitions, dtype=np.int64)
    for partition_id, n in counts.items():
        arr[partition_id] = n
    return arr


_WRITE_PROF_ON = os.environ.get("RAY_DATA_WRITE_PROF") == "1"


class _PartitionWriter:
    """Per-partition shard buffer, sealed to disk one IPC frame per partition.

    ``add_shard(partition_id, shard)`` buffers a shard into partition ``partition_id``'s list;
    ``flush_all()`` concatenates each partition's shards, IPC-encodes them
    into one whole-frame, writes it to the output file, and records the
    (offset, length) range in ``index``.
    """

    __slots__ = (
        "_out_file",
        "_map_id",
        "_compression",
        "_staging",
        "_index",
        "_rows_per_partition",
        "_decoded_bytes_per_partition",
        "_combine_native_ok",
        "prof",
    )

    def __init__(
        self,
        out_file: BinaryIO,
        map_id: int,
        compression: Optional[str],
    ):
        self._out_file = out_file
        self._map_id = map_id
        # Codec from data_context.shuffle_compression (same field the reduce reads).
        self._compression = compression
        self._staging: Dict[int, List[pa.Table]] = {}
        self._index: Dict[int, List[Tuple[int, int]]] = {}
        self._rows_per_partition: Dict[int, int] = {}
        self._decoded_bytes_per_partition: Dict[int, int] = {}
        # Native combine is safe when there are no extension columns.
        self._combine_native_ok: Optional[bool] = None
        # EXPERIMENT (RAY_DATA_WRITE_PROF=1). The fused map task's phase timers
        # charge gather, encode and write to one "encode_write" bucket whose
        # cpu/wall is 0.78, so 22% of it is waiting on something, and the bucket
        # cannot say what. Split it: encode is CPU (IPC + codec), write is the
        # buffered file write, which only blocks once the kernel throttles dirty
        # pages. Bytes come along so the write time converts to a bandwidth that
        # can be held against the NVMe's sustained rate.
        self.prof = (
            {"gather": 0.0, "encode": 0.0, "write": 0.0,
             "gather_cpu": 0.0, "encode_cpu": 0.0, "write_cpu": 0.0,
             "bytes_out": 0, "bytes_in": 0, "frames": 0}
            if _WRITE_PROF_ON
            else None
        )

    def _flush(self, partition_id: int) -> None:
        shards = self._staging.get(partition_id)
        if not shards:
            return
        if self.prof is not None:
            _t, _c = time.perf_counter(), time.process_time()
        tbl = pa.concat_tables(shards) if len(shards) > 1 else shards[0]
        if self.prof is not None:
            self.prof["gather"] += time.perf_counter() - _t
            self.prof["gather_cpu"] += time.process_time() - _c
        if self._combine_native_ok is None:
            # Once per map: concat above guarantees a uniform schema, so native
            # combine is safe iff no column is an extension type. Avoids the
            # per-shard extension-safe combine (which re-scans + does per-column
            # dispatch + nbytes on every shard).
            self._combine_native_ok = not any(
                _is_pa_extension_type(f.type) for f in tbl.schema
            )
        self._rows_per_partition[partition_id] = tbl.num_rows
        self._decoded_bytes_per_partition[partition_id] = tbl.nbytes
        if self.prof is not None:
            _t, _c = time.perf_counter(), time.process_time()
        buf = _encode_shard(tbl, self._compression, self._combine_native_ok)
        if self.prof is not None:
            self.prof["encode"] += time.perf_counter() - _t
            self.prof["encode_cpu"] += time.process_time() - _c
            self.prof["bytes_in"] += tbl.nbytes
        if buf.size > _MAX_RANGE_BYTES:
            raise RuntimeError(
                f"map_{self._map_id}.shf partition {partition_id}: IPC frame is "
                f"{buf.size} bytes, exceeding the u64 wire-protocol "
                f"per-range limit ({_MAX_RANGE_BYTES}). Increase "
                f"``num_partitions`` or reduce the upstream block size."
            )
        off = self._out_file.tell()
        if self.prof is not None:
            _t, _c = time.perf_counter(), time.process_time()
        self._out_file.write(memoryview(buf))
        if self.prof is not None:
            self.prof["write"] += time.perf_counter() - _t
            self.prof["write_cpu"] += time.process_time() - _c
            self.prof["bytes_out"] += buf.size
            self.prof["frames"] += 1
        self._index.setdefault(partition_id, []).append((off, buf.size))
        self._staging[partition_id] = []

    def add_shard(self, partition_id: int, shard: pa.Table) -> None:
        if not shard.num_rows:
            return
        self._staging.setdefault(partition_id, []).append(shard)

    def flush_all(self) -> None:
        for partition_id in list(self._staging.keys()):
            self._flush(partition_id)

    @property
    def index(self) -> Dict[int, List[Tuple[int, int]]]:
        return self._index

    @property
    def rows_per_partition(self) -> Dict[int, int]:
        return self._rows_per_partition

    @property
    def decoded_bytes_per_partition(self) -> Dict[int, int]:
        return self._decoded_bytes_per_partition


# =============================================================================
# SERVER SIDE (created by map tasks; serves reduce fetches). Arrow Flight over
# gRPC: DoAction streams opaque byte-ranges (the whole-frame shard blob).
# =============================================================================
# Per-Result body size. Each flight.Result buffer is materialized whole in RAM
# when sent, so chunking bounds the ShuffleFileServer actor's memory.
_FLIGHT_CHUNK = MiB


# Flight fetch request (Action body): JSON ``{"s": [[path, [[off, len], ...]], ...]}``.


def _grpc_location(host: str, port) -> str:
    """gRPC URI for host:port. IPv6 literals must be bracketed (grpc://[::1]:0);
    a bare ``:`` in the host is the IPv6 signal."""
    h = f"[{host}]" if ":" in host else host
    return f"grpc://{h}:{port}"


# EXPERIMENT (RAY_DATA_FLIGHT_SERVER_PROF=1): the client side says 96% of a
# reduce task's fetch is spent waiting on `next()` over the DoAction stream.
# That is either the server being slow to produce or the transport being slow
# to carry. These counters separate them:
#   read   time inside f.read(), i.e. local NVMe
#   yield  time the generator is SUSPENDED between yields, i.e. how long gRPC
#          and the consumer take to accept a chunk (back-pressure)
#   busy   wall of the whole handler
# `inflight`/`max_inflight` say whether the one actor per node is serialising
# the cluster's reducers.
_SRV_PROF = collections.defaultdict(float)
_SRV_LOCK = threading.Lock()
# Per-call scratch is passed down explicitly rather than kept in a
# threading.local: Flight drives the handler generator from a thread pool, so
# successive next() calls on ONE request can land on different threads and a
# thread-local reads back empty. It must also not be _SRV_LOCK, because taking
# that per open() or per 1 MiB read serialises every concurrent handler on one
# mutex and inflates exactly the numbers these timers exist to attribute (an
# earlier version did, and cost the query 9%). A generator is never advanced
# concurrently, so a plain dict owned by the call needs no synchronisation.
_SRV_STATE = {"inflight": 0, "max_inflight": 0, "calls": 0}
_SRV_PROF_ON = os.environ.get("RAY_DATA_FLIGHT_SERVER_PROF") == "1"
_SRV_REPORT_EVERY = int(os.environ.get("RAY_DATA_FLIGHT_SERVER_PROF_EVERY", "200"))

# One fd per shuffle file, shared by every DoAction that touches it.
#
# A reducer asks each mapper for its own partition's frame, so without a cache
# the same ``.shf`` is reopened once per reduce partition. Measured on q9
# sf1000 at 200 partitions: 792,224 opens for ~3,961 distinct files, 4.2 ms
# each, 32% of the server's busy time (FINDINGS.md section 17). The per-open
# cost is the same at every frame size, so this is pure repeated work.
#
# ``os.pread`` takes the offset as an argument and never touches the shared
# file position, so one fd serves all concurrent handlers with no lock on the
# read path. The cache is deliberately unbounded: the file server actor is
# registered per (shuffle, node, replica), so it only ever sees the files one
# shuffle wrote on one node (~417 at the observed worst node) and its fds are
# released when the actor dies, which is when that shuffle's dir is removed.
_FD_CACHE: Dict[str, int] = {}
_FD_LOCK = threading.Lock()
_FD_CACHE_ON = os.environ.get("RAY_DATA_SHUFFLE_SERVER_FD_CACHE", "1") != "0"


def _cached_fd(path: str) -> int:
    """Return a read-only fd for ``path``, opening it at most once per process."""
    fd = _FD_CACHE.get(path)
    if fd is not None:
        return fd
    # Opened outside the lock so one slow open cannot block handlers reading
    # other files. Two callers racing on the same path both open; the loser
    # closes its own fd, so no descriptor leaks and neither sees a closed one.
    opened = os.open(path, os.O_RDONLY)
    with _FD_LOCK:
        fd = _FD_CACHE.get(path)
        if fd is None:
            _FD_CACHE[path] = opened
            return opened
    os.close(opened)
    return fd


def _make_flight_server(host: str, base_dir: str):
    """Arrow Flight server serving shuffle byte-ranges via DoAction. Each range
    is framed as ``[u64 length][frame bytes]``."""
    import pyarrow.flight as flight

    class _ShuffleFlightServer(flight.FlightServerBase):
        def do_action(self, context, action):
            if _SRV_PROF_ON:
                yield from self._do_action_profiled(context, action)
                return
            yield from self._do_action(context, action)

        def _do_action_profiled(self, context, action):
            local: Dict[str, float] = collections.defaultdict(float)
            t_call = time.perf_counter()
            with _SRV_LOCK:
                _SRV_STATE["inflight"] += 1
                _SRV_STATE["max_inflight"] = max(
                    _SRV_STATE["max_inflight"], _SRV_STATE["inflight"]
                )
            t_yield = 0.0
            nbytes = n = 0
            try:
                for res in self._do_action(context, action, local):
                    nbytes += res.body.size
                    n += 1
                    t0 = time.perf_counter()
                    yield res
                    # Control returns here only once the consumer has taken the
                    # chunk, so this gap is transport plus client back-pressure.
                    t_yield += time.perf_counter() - t0
            finally:
                with _SRV_LOCK:
                    _SRV_STATE["inflight"] -= 1
                    _SRV_STATE["calls"] += 1
                    _SRV_PROF["busy"] += time.perf_counter() - t_call
                    _SRV_PROF["yield"] += t_yield
                    _SRV_PROF["bytes"] += nbytes
                    _SRV_PROF["results"] += n
                    for _k, _v in local.items():
                        _SRV_PROF[_k] += _v
                    calls = _SRV_STATE["calls"]
                    if calls % _SRV_REPORT_EVERY == 0:
                        print(
                            f"[SRV_PROF] calls={calls} "
                            f"busy={_SRV_PROF['busy']:.1f} "
                            f"read={_SRV_PROF['read']:.1f} "
                            f"yield={_SRV_PROF['yield']:.1f} "
                            f"GiB={_SRV_PROF['bytes'] / 2**30:.2f} "
                            f"results={int(_SRV_PROF['results'])} "
                            f"open={_SRV_PROF['open']:.1f} "
                            f"json={_SRV_PROF['json']:.1f} "
                            f"other={_SRV_PROF['busy'] - _SRV_PROF['read'] - _SRV_PROF['yield'] - _SRV_PROF['open'] - _SRV_PROF['json']:.1f} "
                            f"files={int(_SRV_PROF['members'])} "
                            f"ranges={int(_SRV_PROF['ranges'])} "
                            f"inflight={_SRV_STATE['inflight']} "
                            f"max_inflight={_SRV_STATE['max_inflight']}",
                            flush=True,
                        )

        def _do_action(self, context, action, prof=None):
            _tj = time.perf_counter()
            req = json.loads(action.body.to_pybytes())
            if prof is not None:
                prof["json"] += time.perf_counter() - _tj
                prof["members"] += len(req["s"])
                prof["ranges"] += sum(len(r) for _p, r in req["s"])
            for path, ranges in req["s"]:
                fpath = os.path.join(base_dir, os.path.basename(path))
                # One open() per MAP FILE, which a reducer touches once for
                # every mapper: 254 files at RAY_DATA_FUSED_READ_MAP_BUNDLES=12,
                # ~3,046 at =1, times 200 reducers. _cached_fd collapses that to
                # one open per file per process; the timer stays because the
                # cold path still costs what it always did.
                _to = time.perf_counter()
                _fd = _cached_fd(fpath) if _FD_CACHE_ON else os.open(
                    fpath, os.O_RDONLY
                )
                if prof is not None:
                    prof["open"] += time.perf_counter() - _to
                try:
                    for off, length in ranges:
                        # Empty ranges still emit a u64(0) header so the client's
                        # framed stream stays aligned with _compute_prefetch_layout
                        # (which reserves 8 bytes per range including length==0).
                        if length == 0:
                            yield flight.Result(pa.py_buffer(struct.pack(">Q", 0)))
                            continue
                        # Chunk the frame so no single Result materializes a whole
                        # large frame in the server's RAM. The u64 length header
                        # rides in the first chunk only (its own Result would pay a
                        # full gRPC message's overhead for 8 bytes).
                        remaining = length
                        pos = off
                        first = True
                        while remaining:
                            _tr = time.perf_counter()
                            # pread, not seek+read: the fd is shared by every
                            # concurrent handler, so a read that moved the file
                            # position would hand another request the wrong bytes.
                            buf = os.pread(_fd, min(remaining, _FLIGHT_CHUNK), pos)
                            if prof is not None:
                                prof["read"] += time.perf_counter() - _tr
                            if not buf:
                                # File shorter than the index says (truncated /
                                # stale offset). A short send silently desyncs
                                # every later frame at the client (SPARK-34534).
                                raise ValueError(
                                    f"short read: {fpath} @{off}+{length}, "
                                    f"got {length - remaining}"
                                )
                            remaining -= len(buf)
                            pos += len(buf)
                            if first:
                                buf = struct.pack(">Q", length) + buf
                                first = False
                            yield flight.Result(pa.py_buffer(buf))
                finally:
                    # A cached fd outlives the request by design; an uncached
                    # one must not, including when the consumer abandons the
                    # stream and GeneratorExit unwinds us mid-frame.
                    if not _FD_CACHE_ON:
                        os.close(_fd)

    return _ShuffleFlightServer(_grpc_location(host, 0))


@ray.remote(num_cpus=0, max_restarts=-1)
class ShuffleFileServer:
    """Per-node file fetch service: owns an Arrow Flight server that serves
    byte-ranges of local shuffle files to remote reducers. Survives individual
    map/reduce workers; in a real cluster, one per node (NodeAffinity)."""

    def prof(self) -> Dict[str, float]:
        """Drain this node's Flight-server counters.

        The server runs in a thread of THIS actor process, so the module-level
        _SRV_PROF is readable from here. A detached actor's stdout does not
        reach the submitting job's log, so printing from inside the server is
        invisible -- the driver has to come and ask.

        The module is imported INSIDE the method on purpose. Referencing
        _SRV_LOCK at module scope would make it part of this actor class's
        captured globals, and cloudpickle cannot serialize a threading.Lock --
        which breaks every task in the file that touches this class.
        """
        from ray.data._internal.execution.operators.shuffle_operators import (
            disk_shuffle_runtime as _m,
        )

        with _m._SRV_LOCK:
            out = dict(_m._SRV_PROF)
            out.update({f"state_{k}": float(v) for k, v in _m._SRV_STATE.items()})
        out["_ip"] = ray.util.get_node_ip_address()
        return out

    def __init__(
        self,
        base_dir: str,
    ):
        self.base_dir = os.path.realpath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        ip = ray.util.get_node_ip_address()
        self._server = _make_flight_server(ip, self.base_dir)
        self._host, self._port = ip, self._server.port
        # Unique per actor process; Ray re-runs __init__ on every restart, so
        # this changes on restart. Reducers compare it to detect a restart.
        self._incarnation = uuid.uuid4().hex

    def endpoint(self) -> _Endpoint:
        return _Endpoint(self._host, self._port, self._incarnation)


@ray.remote(num_cpus=0)
def _cleanup_shuffle_dir(map_dir: str, reduce_dir: str) -> None:
    """Best-effort ``rmtree`` of this shuffle's map + reduce staging dirs. Pinned
    to its node with ``NodeAffinity(soft=False)``: on a live node it runs there;
    if the node is gone the task just fails."""
    import shutil

    shutil.rmtree(map_dir, ignore_errors=True)
    shutil.rmtree(reduce_dir, ignore_errors=True)


# =============================================================================
# FETCH SIDE (reduce tasks): resolve file-server endpoints, stream shards from each
# over a Flight client into the prefetch sink at disjoint offsets, then decode.
# =============================================================================
# Process-global endpoint cache, reused across reducers running on the same
# worker. Popped on fetch failure (re-resolved next call); the lock guards the
# concurrent fetch threads.
_ENDPOINT_CACHE: Dict[str, _Endpoint] = {}
_ENDPOINT_CACHE_LOCK = threading.Lock()

# EXPERIMENT (RAY_DATA_SHUFFLE_FLIGHT_CLIENT_CACHE=1): reuse Flight clients
# across fetches instead of dialling per call.
#
# py-spy on a q9 reduce worker put 87% of its non-idle samples on the
# `flight.connect(...)` line of `_stream_members_flight`, not on `do_action`
# and not on the sink write. Every reduce task opens one client per source
# node per call and drops it when the `with` block exits, so the first join
# reduce alone pays 200 partitions x 8 nodes of gRPC channel setup, 128 tasks
# at a time. The endpoint ADDRESS is already cached below; the client is not.
#
# Keyed by incarnation as well as address so a restarted file server cannot be
# served a stale channel: the retry path drops the entry along with the
# endpoint it was built from.
_FLIGHT_CLIENT_CACHE: Dict[Tuple[str, int, str], Any] = {}
_FLIGHT_CLIENT_LOCK = threading.Lock()
# EXPERIMENT (RAY_DATA_FLIGHT_PROF=1): split the Flight fetch into dial,
# server round-trip, and local write. py-spy put 87% of a reduce worker's
# non-idle samples on the `with flight.connect(...)` LINE, but that line is the
# context manager holding the whole loop, so it attributes the block, not the
# dial. Caching the client changed nothing, which rules the dial out; these
# three timers say which of the remaining parts actually costs.
_FLIGHT_PROF: "Dict[str, float]" = collections.defaultdict(float)
_FLIGHT_PROF_LOCK = threading.Lock()
_FLIGHT_PROF_ON = os.environ.get("RAY_DATA_FLIGHT_PROF") == "1"


def flight_prof_take() -> "Dict[str, float]":
    """Read and clear the accumulator. A worker runs one task at a time, so
    taking it at task end attributes cleanly to that task."""
    with _FLIGHT_PROF_LOCK:
        snap = dict(_FLIGHT_PROF)
        _FLIGHT_PROF.clear()
    return snap


_FLIGHT_CLIENT_CACHE_ON = (
    os.environ.get("RAY_DATA_SHUFFLE_FLIGHT_CLIENT_CACHE") == "1"
)


def _flight_client(endpoint: "_Endpoint"):
    """A Flight client for ``endpoint``, reused when caching is enabled.

    Returns ``(client, owned)``. ``owned`` is True when the caller must close
    the client itself, i.e. when caching is off and the client is per-call.
    """
    import pyarrow.flight as flight

    host, port, incarnation = endpoint
    if not _FLIGHT_CLIENT_CACHE_ON:
        return flight.connect(_grpc_location(host, port)), True
    key = (host, port, incarnation)
    with _FLIGHT_CLIENT_LOCK:
        client = _FLIGHT_CLIENT_CACHE.get(key)
        if client is None:
            client = flight.connect(_grpc_location(host, port))
            _FLIGHT_CLIENT_CACHE[key] = client
    return client, False


def _drop_flight_client(endpoint: "_Endpoint") -> None:
    """Forget a cached client, for the transport-fault retry path."""
    if not _FLIGHT_CLIENT_CACHE_ON:
        return
    with _FLIGHT_CLIENT_LOCK:
        client = _FLIGHT_CLIENT_CACHE.pop(tuple(endpoint), None)
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 - a dead channel is what we are dropping
            pass

# --------------------------------------------------------- fetch routing types
# Named containers for reducer fetch orchestration. ``slots=True`` keeps
# per-instance memory small since we allocate one per source/group/member.


@dataclass(slots=True, frozen=True)
class _FileRanges:
    """A shuffle file and the byte ranges to read from it."""

    path: str
    ranges: List[Tuple[int, int]]


@dataclass(slots=True, frozen=True)
class _SourceRef:
    """A ``_FileRanges`` tagged with its ShuffleFileServer identity.

    Built once per input handle for a given partition_id at reducer start.
    The (shuffle_id, node_id) pair is the file server's named-actor identity.
    """

    shuffle_id: str
    node_id: str
    file: _FileRanges


@dataclass(slots=True)
class _NodeGroup:
    """All sources on one ShuffleFileServer, grouped so we open ONE Flight
    connection per file server. Sources collapse to the same group when their
    ``(shuffle_id, node_id)`` (i.e., the file server's named-actor identity) matches.
    """

    shuffle_id: str
    node_id: str
    members: List[_FileRanges] = field(default_factory=list)


class _PwriteSink:
    """A write-only file-like that ``os.pwrite``s sequentially from a
    fixed base offset on a shared fd. Multiple sinks (one per fetch thread) at
    DISJOINT base regions write the same fd concurrently without lock.

    ``reset()`` rewinds ``_pos`` to the base offset so a fetch attempt that
    partially wrote can be retried in-place and subsequent ``os.pwrite``s
    overwrite the partial data (same offsets, idempotent write)."""

    __slots__ = ("_fd", "_base_offset", "_pos")

    def __init__(self, fd: int, base_offset: int):
        self._fd = fd
        self._base_offset = base_offset
        self._pos = base_offset

    def reset(self) -> None:
        self._pos = self._base_offset

    def write(self, data: Union[bytes, bytearray, memoryview, "pa.Buffer"]) -> int:
        mv = memoryview(data)
        bytes_written = 0
        nbytes = len(mv)
        while bytes_written < nbytes:
            n = os.pwrite(self._fd, mv[bytes_written:], self._pos + bytes_written)
            if n <= 0:
                # POSIX pwrite of a nonzero buffer should return >0 or raise;
                # 0 can happen on some FUSE / network filesystems and would
                # spin this loop forever.
                raise OSError(
                    f"pwrite returned {n} on fd {self._fd} "
                    f"(offset={self._pos + bytes_written}, "
                    f"remaining={nbytes - bytes_written})"
                )
            bytes_written += n
        self._pos += bytes_written
        return bytes_written


def _stream_members_flight(
    endpoint: _Endpoint,
    members: List[_FileRanges],
    max_bytes: int,
    sink: _PwriteSink,
) -> None:
    """Arrow Flight DoAction: one client, batched fetch requests. Response
    ``Result`` bodies carry ``[u64 len][frame]`` framing, so they stream verbatim
    into the sink."""
    import pyarrow.flight as flight

    _t = time.perf_counter()
    client, owned = _flight_client(endpoint)
    t_dial = time.perf_counter() - _t
    t_rpc = t_write = 0.0
    n_bytes = n_results = n_batches = 0
    try:
        for batch in _chunk_members_by_bytes(members, max_bytes):
            n_batches += 1
            sources = [(m.path, m.ranges) for m in batch]
            body = json.dumps({"s": sources}).encode("utf-8")
            _t = time.perf_counter()
            it = iter(client.do_action(flight.Action("fetch", body)))
            while True:
                _t2 = time.perf_counter()
                try:
                    result = next(it)
                except StopIteration:
                    t_rpc += time.perf_counter() - _t2
                    break
                t_rpc += time.perf_counter() - _t2
                _t3 = time.perf_counter()
                sink.write(result.body)
                t_write += time.perf_counter() - _t3
                n_results += 1
                n_bytes += result.body.size
            t_rpc += 0.0
    finally:
        if owned:
            client.close()
    if _FLIGHT_PROF_ON:
        with _FLIGHT_PROF_LOCK:
            _FLIGHT_PROF["dial"] += t_dial
            _FLIGHT_PROF["rpc"] += t_rpc
            _FLIGHT_PROF["write"] += t_write
            _FLIGHT_PROF["bytes"] += n_bytes
            _FLIGHT_PROF["results"] += n_results
            _FLIGHT_PROF["batches"] += n_batches
            _FLIGHT_PROF["calls"] += 1


def _fetch_from_file_server(
    out_file_obj: "_PwriteSink",
    shuffle_id: str,
    node_id: str,
    members: List[_FileRanges],
    max_bytes_per_fetch: int,
    replica: int = 0,
) -> None:
    """Stream every member's shards into ``out_file_obj`` over ONE Flight
    client, batched into DoAction requests of ≤ ``max_bytes_per_fetch``.

    Retry only transport faults; propagate everything else. The retriable errors
    are all FlightError and share the single ``except flight.FlightError`` (so
    ``FlightTimedOutError`` and friends need no clause of their own); the terminal
    ones are either split out ahead of it or are not a FlightError at all.

    Retried (transport faults, caught by ``except flight.FlightError``):
      * ``FlightUnavailableError`` -- server down / restarting.
      * ``FlightTimedOutError`` -- hung.
      * ``FlightCancelledError`` -- dropped mid-call.
      * ``FlightInternalError`` and any other transport ``FlightError``.

    Terminal (propagate without retry):
      * ``flight.FlightServerError`` -- server alive but refused the request
        (missing/unreadable file, server-side OSError, or bug). It IS a
        FlightError, so it is re-raised ahead of the broad catch.
      * ``pa.lib.ArrowInvalid`` / ``ArrowKeyError`` -- corrupt / short frame, or a
        server-side ValueError / KeyError. Not a FlightError; propagates as-is.
      * sink ``OSError`` (local write failed, e.g. disk full or I/O error) and
        ``_resolve``'s ``ShuffleFileServerAnomalyError`` (dead / unschedulable /
        unregistered actor). Propagate as-is.

    On a transport fault, re-resolve and compare incarnations (not host:port -- a
    restart can rebind the port). A changed incarnation means the server restarted:
    reset the sink and retry in place. An unchanged incarnation means the process
    stayed up but the fetch still failed: retry once (a blip ``_resolve`` may have
    polled through), then escalate to a terminal ``ShuffleFileServerAnomalyError``
    (likely a network block on the Flight port). Flight error classes / gRPC codes:
    https://arrow.apache.org/docs/python/api/flight.html
    https://grpc.io/docs/guides/status-codes/
    """
    import pyarrow.flight as flight

    # A node only has the replicas its own map tasks created, and an operator
    # whose input is a small dimension table may have run a single map task on
    # that node (q9's nation side has exactly one). So a requested replica can
    # legitimately be absent: prefer it, then take whichever does exist.
    n_replicas = SHUFFLE_FILE_SERVERS_PER_NODE
    preferred = replica % n_replicas
    order = [preferred] + [r for r in range(n_replicas) if r != preferred]
    key = _file_server_name(shuffle_id, node_id, preferred)

    def _resolve() -> _Endpoint:
        # Endpoint cache avoids a blocking Ray RPC per fetch. On miss, ask
        # Ray for actor state and route by outcome (see docstring above).
        with _ENDPOINT_CACHE_LOCK:
            ep = _ENDPOINT_CACHE.get(key)
        if ep is not None:
            return ep
        server = None
        last_err: "Optional[Exception]" = None
        for r in order:
            try:
                server = ray.get_actor(
                    _file_server_name(shuffle_id, node_id, r),
                    namespace=_SHUFFLE_FILE_SERVER_NAMESPACE,
                )
                break
            except ValueError as e:
                last_err = e
        if server is None:
            # No replica is registered (never created or cleaned up)
            raise ShuffleFileServerAnomalyError(
                f"ShuffleFileServer on node {node_id} not found in namespace: "
                f"{last_err}"
            ) from last_err
        poll_count = 0
        while True:
            try:
                ep = cast(_Endpoint, ray.get(server.endpoint.remote()))
                break
            except ActorUnavailableError:
                poll_count += 1
                if poll_count % _ENDPOINT_POLL_WARN_EVERY == 0:
                    logger.warning(
                        f"ShuffleFileServer on node {node_id} unavailable "
                        f"(~{poll_count * _ENDPOINT_POLL_INTERVAL_S:.0f}s), "
                        f"still polling..."
                    )
                time.sleep(_ENDPOINT_POLL_INTERVAL_S)
            except ActorDiedError as e:
                # With max_restarts=-1, Ray auto-restarts on mid-life death and
                # surfaces ActorUnavailableError during the restart. So an
                # ActorDiedError reaching us means init failure or external
                # ray.kill
                raise ShuffleFileServerAnomalyError(
                    f"ShuffleFileServer on node {node_id} is dead: {e}"
                ) from e
            except ActorUnschedulableError as e:
                # Pinned node is gone; soft=False can't relocate the actor.
                # Ray transitions from ActorUnavailableError to this ~10s after
                # heartbeat loss.
                raise ShuffleFileServerAnomalyError(
                    f"ShuffleFileServer on node {node_id} is unschedulable "
                    f"(node likely dead): {e}"
                ) from e
        with _ENDPOINT_CACHE_LOCK:
            _ENDPOINT_CACHE[key] = ep
        return ep

    same_incarnation_retried = False
    while True:
        endpoint = _resolve()
        try:
            _stream_members_flight(endpoint, members, max_bytes_per_fetch, out_file_obj)
            return
        except flight.FlightServerError:
            raise  # terminal: server alive, refused the request (see docstring)
        except flight.FlightError as e:
            # any other FlightError is a transport fault (see docstring)
            out_file_obj.reset()
            _drop_flight_client(endpoint)
            with _ENDPOINT_CACHE_LOCK:
                _ENDPOINT_CACHE.pop(key, None)
            fresh = _resolve()

            if fresh.incarnation != endpoint.incarnation:
                same_incarnation_retried = False
                logger.warning(f"node {node_id}: file server restarted, retrying")
                continue

            if not same_incarnation_retried:
                same_incarnation_retried = True
                logger.warning(
                    f"node {node_id}: Flight fetch failed, retrying once ({e})"
                )
                time.sleep(0.5)
                continue

            raise ShuffleFileServerAnomalyError(
                f"node {node_id}: Flight fetch still failing but file server "
                f"reachable via Ray; likely a network block on the Flight port "
                f"(NetworkPolicy/firewall/routing)."
            ) from e


def _chunk_members_by_bytes(
    members: List[_FileRanges],
    max_bytes: int,
) -> Iterable[List[_FileRanges]]:
    """Yield sub-batches of members whose total requested bytes ≤ ``max_bytes``.

    A source's ranges MAY be split across batches: the source appears as
    multiple pseudo-members with the same ``path`` but disjoint range
    subsets, in the original range order. Individual ranges are NEVER
    split as each range is one Arrow IPC frame at the mapper, so a
    sub-range cut would break the reducer's decode.
    """
    batch: List[_FileRanges] = []
    batch_bytes = 0
    for member in members:
        pending: List[Tuple[int, int]] = []
        for off, length in member.ranges:
            if (batch or pending) and batch_bytes + length > max_bytes:
                if pending:
                    batch.append(_FileRanges(path=member.path, ranges=pending))
                    pending = []
                yield batch
                batch, batch_bytes = [], 0
            pending.append((off, length))
            batch_bytes += length
        if pending:
            batch.append(_FileRanges(path=member.path, ranges=pending))
    if batch:
        yield batch


def _handle_batch_size(handles, batch_bytes):
    """#handles to resolve per batch so materialized metadata stays ≈ batch_bytes.

    Peeks one handle for ``num_partitions`` (each handle ≈ num_partitions × 32
    bytes of dense arrays: ``index_ranges`` is ``[P, 2]`` int64;
    ``num_rows`` and ``decoded_bytes`` are ``[P]`` int64), so in-flight
    handle memory stays constant regardless of #mappers/#partitions.
    """
    if not handles:
        return 1
    probe = handles[0]
    if not isinstance(probe, dict):
        probe = cast(dict, ray.get(probe))
    try:
        npart = int(probe.get("num_partitions") or len(probe["index_ranges"]))
    except Exception:
        npart = 1
    per_handle = max(1, npart * 32)
    return max(1, min(len(handles), batch_bytes // per_handle))


def _handles_to_sources(
    handles: List["ShuffleHandle"],
    partition_id: int,
    batch_bytes: int = _DEFAULT_HANDLE_BATCH_BYTES,
) -> Tuple[List[_SourceRef], Optional[pa.Schema]]:
    """Extract per-partition source refs from a reducer's input handles.

    Resolves handle refs in ≈``batch_bytes`` batches, reads this partition's row
    out of each handle's dense range index, then frees the batch — so in-flight
    handle memory stays constant, not O(maps × partitions). Skips handles with
    ``index_ranges[p].length == 0`` (no on-disk frame); picks the first
    non-None schema.
    """
    sources: List[_SourceRef] = []
    output_schema: Optional[pa.Schema] = None
    if not handles:
        return sources, output_schema

    batch_size = _handle_batch_size(handles, batch_bytes)
    for start in range(0, len(handles), batch_size):
        batch = handles[start : start + batch_size]
        refs = [handle for handle in batch if not isinstance(handle, dict)]
        # pyrefly: ignore[no-matching-overload]
        vals = iter(ray.get(refs)) if refs else iter(())
        resolved = [
            handle if isinstance(handle, dict) else next(vals) for handle in batch
        ]
        for handle in resolved:
            if output_schema is None:
                output_schema = handle.get("schema")
            # Dense index: partition_id is the row; only this row materializes
            # as a Python range, the rest stay as (zero-copy) numpy buffers.
            off, length = handle["index_ranges"][partition_id]
            length = int(length)
            if length > 0:
                sources.append(
                    _SourceRef(
                        shuffle_id=handle["shuffle_id"],
                        node_id=handle["node_id"],
                        file=_FileRanges(
                            path=handle["path"],
                            ranges=[(int(off), length)],
                        ),
                    )
                )
        del resolved, vals
    return sources, output_schema


def _read_local_group(sink, group: "_NodeGroup") -> None:
    """Serve a node-local group by reading the shuffle file directly.

    Byte-for-byte what _ShuffleFlightServer._do_action would have sent for the
    same group -- `[u64 BE length][frame]` per range, a bare u64(0) for empty
    ranges -- so the staging file stays aligned with _compute_prefetch_layout
    and _decode_region cannot tell the difference.

    A reducer fetches from every node including its own, and that self-fetch
    goes out over loopback gRPC today: serialize, copy, chunk into 1 MiB Flight
    Results, copy back. The bytes are already in this node's page cache, so a
    pread is strictly less work, and the region finishes immediately instead of
    queueing behind the remote streams.
    """
    for fr in group.members:
        with open(fr.path, "rb") as f:
            for off, length in fr.ranges:
                if length == 0:
                    sink.write(struct.pack(">Q", 0))
                    continue
                f.seek(off)
                sink.write(struct.pack(">Q", length))
                remaining = length
                while remaining:
                    buf = f.read(min(remaining, _FLIGHT_CHUNK))
                    if not buf:
                        raise ValueError(
                            f"short read: {fr.path} @{off}+{length}, "
                            f"got {length - remaining}"
                        )
                    remaining -= len(buf)
                    sink.write(buf)


def _group_by_server(sources: List[_SourceRef]) -> List[_NodeGroup]:
    """Collapse sources by file server so each file server gets ONE Flight connection."""
    by_key: Dict[Tuple[str, str], _NodeGroup] = {}
    for source in sources:
        key = (source.shuffle_id, source.node_id)
        group = by_key.get(key)
        if group is None:
            group = _NodeGroup(
                shuffle_id=source.shuffle_id,
                node_id=source.node_id,
                members=[],
            )
            by_key[key] = group
        group.members.append(source.file)
    return list(by_key.values())


def _compute_prefetch_layout(
    groups: List[_NodeGroup],
) -> Tuple[int, List[int], List[int]]:
    """Assign each group a contiguous byte region in the reducer's prefetch file.

    Returns ``(total_size, base_offsets, per_group_sizes)`` where sizes are the
    ``8 + length`` framed byte totals (u64 len prefix + IPC bytes per range),
    base offsets are running cumulative sums. Fetch threads then pwrite each
    group's fetched frames at DISJOINT offsets.
    """
    sizes = [
        sum(8 + length for m in g.members for (_off, length) in m.ranges)
        for g in groups
    ]
    base_offsets: List[int] = []
    acc = 0
    for sz in sizes:
        base_offsets.append(acc)
        acc += sz
    return acc, base_offsets, sizes
