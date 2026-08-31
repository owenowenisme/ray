"""PySpark port of ``groupby_benchmark.py`` (aggregate_groups / map_groups).

    # aggregation groups
    python spark_groupby_benchmark.py --sf 100 --group-by column04 --aggregate

    # map groups
    python spark_groupby_benchmark.py --sf 100 --group-by column04 --map-groups

Mirrors the Ray benchmark: read TPC-H lineitem, group by ``--group-by``, then
either mean ``column05`` (l_extendedprice) or normalize every float column
within each group.

map_groups has two implementations -- pick deliberately
-------------------------------------------------------
Ray's ``map_groups(normalize_table, batch_format="pyarrow")`` dispatches a
Python function per group that calls vectorized PyArrow compute. There is no
single Spark construct with the same shape, so both candidates are offered:

  ``--map-groups-impl window`` (default)
      Native window aggregation: ``(x - mean(x) OVER g) / stddev_pop(x) OVER g``.
      This is what a Spark user would actually write, runs entirely in the JVM,
      and needs one shuffle. Compares WHAT EACH ENGINE CAN DO.

  ``--map-groups-impl pandas``
      ``applyInPandas`` -- a real per-group Python callback, the mechanical
      analogue of Ray's map_groups. Pays Arrow<->pandas serialization per group,
      so it is much slower. Compares THE SAME MECHANISM.

Quote which one you used; they answer different questions.

Note on stddev: PyArrow's ``pc.stddev`` defaults to ddof=0 (population), so the
Spark side uses ``stddev_pop``, not ``stddev`` (which is the sample stddev).
Verified locally: the window and pandas implementations agree to 1e-9, and both
match ddof=0 rather than ddof=1.

``--map-groups-impl pandas`` needs pandas + pyarrow importable *in the Python
interpreter Spark launches for its workers*, which is ``PYSPARK_PYTHON`` (not
necessarily the one running this script). If the executors fail with
``ModuleNotFoundError: No module named 'pandas'``, set::

    export PYSPARK_PYTHON=$(which python)

Consumption: Ray materializes the aggregate and iterates every ref bundle for
map_groups. Both outputs can be huge (map_groups output is the same size as its
input), so this uses the ``noop`` sink -- full materialization, no write, nothing
pulled to the driver.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
try:
    from spark_join_benchmark import create_session
except ImportError as e:  # pragma: no cover - setup error
    raise SystemExit(
        f"could not import create_session from spark_join_benchmark: {e}\n"
        "Copy spark_join_benchmark.py next to this script."
    )

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sf", type=int, default=1, choices=[1, 10, 100, 1000, 10000])
    p.add_argument("--group-by", required=True, nargs="+")
    p.add_argument("--base-uri", default="s3a://ray-benchmark-data/tpch/parquet")
    p.add_argument("--output", default="spark_groupby_results.json")
    p.add_argument(
        "--map-groups-impl",
        default="window",
        choices=["window", "pandas"],
        help="window = native Spark; pandas = applyInPandas (Ray-shaped)",
    )
    p.add_argument(
        "--num-partitions",
        type=int,
        default=200,
        help="spark.sql.shuffle.partitions (Ray: default_hash_shuffle_parallelism)",
    )
    p.add_argument(
        "--enable-aqe", "--enable_aqe", action="store_true", dest="enable_aqe"
    )
    p.add_argument(
        "--allow-broadcast",
        "--allow_broadcast_join",
        action="store_true",
        dest="allow_broadcast_join",
        help="accepted for a uniform --spark-args; this benchmark's\n"
        "plan has no joins, so it changes nothing",
    )

    consume = p.add_mutually_exclusive_group(required=True)
    consume.add_argument("--aggregate", action="store_true")
    consume.add_argument("--map-groups", action="store_true")

    p.add_argument("--backend", default="raydp", choices=["raydp", "standalone"])
    p.add_argument("--master", default="local[*]")
    p.add_argument("--num_executors", type=int, default=32)
    p.add_argument("--executor_cores", type=int, default=16)
    p.add_argument("--executor_memory", default="32g")
    p.add_argument("--spark_conf", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    # Names `build_spark_conf` expects.
    args.enable_aqe = args.enable_aqe
    args.num_partitions = args.num_partitions
    return args


def float_columns(df: DataFrame) -> List[str]:
    """Columns Ray's ``normalize_table`` would touch (``types.is_floating``)."""
    return [
        f.name for f in df.schema.fields if f.dataType.typeName() in ("double", "float")
    ]


def map_groups_window(df: DataFrame, group_by: List[str]) -> DataFrame:
    """(x - mean(x)) / stddev_pop(x) within each group, as a window agg."""
    w = Window.partitionBy(*group_by)
    cols = float_columns(df)
    exprs = []
    for name in df.columns:
        if name in cols:
            exprs.append(
                (
                    (F.col(name) - F.mean(name).over(w)) / F.stddev_pop(name).over(w)
                ).alias(name)
            )
        else:
            exprs.append(F.col(name))
    return df.select(*exprs)


def map_groups_pandas(df: DataFrame, group_by: List[str]) -> DataFrame:
    """applyInPandas -- a real per-group Python callback, like Ray's map_groups."""
    schema = df.schema
    numeric = set(float_columns(df))

    def normalize(pdf):
        for name in pdf.columns:
            if name in numeric:
                std = pdf[name].std(ddof=0)  # ddof=0 matches pyarrow pc.stddev
                if std and std == std and std != 0:  # not NaN, not zero
                    pdf[name] = (pdf[name] - pdf[name].mean()) / std
        return pdf

    return df.groupBy(*group_by).applyInPandas(normalize, schema=schema)


def collect_stage_metrics(spark: SparkSession) -> Dict[str, Any]:
    try:
        import requests

        ui = spark.sparkContext.uiWebUrl
        if not ui:
            return {}
        app_id = spark.sparkContext.applicationId
        r = requests.get(f"{ui}/api/v1/applications/{app_id}/stages", timeout=30)
        r.raise_for_status()
        stages = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"  (stage metrics unavailable: {e})")
        return {}

    def total(field: str) -> int:
        return sum(int(s.get(field, 0) or 0) for s in stages)

    return {
        "num_stages": len(stages),
        "input_gb": round(total("inputBytes") / GIB, 3),
        "shuffle_write_gb": round(total("shuffleWriteBytes") / GIB, 3),
        "shuffle_read_gb": round(total("shuffleReadBytes") / GIB, 3),
        "mem_spilled_gb": round(total("memoryBytesSpilled") / GIB, 3),
        "disk_spilled_gb": round(total("diskBytesSpilled") / GIB, 3),
        "executor_run_s": round(total("executorRunTime") / 1000, 1),
        "executor_cpu_s": round(total("executorCpuTime") / 1e9, 1),
    }


def main() -> None:
    args = parse_args()
    try:
        spark = create_session(args, app_name="spark_groupby_benchmark")
    except TypeError:
        spark = create_session(args)

    mode = "aggregate" if args.aggregate else "map_groups"
    result: Dict[str, Any] = {}
    try:
        path = f"{args.base_uri}/sf{args.sf}/lineitem"
        print(f"\n=== groupby {mode} (sf{args.sf}, group_by={args.group_by}) ===")
        print(
            f"  shuffle.partitions={args.num_partitions} "
            f"aqe={'ON' if args.enable_aqe else 'OFF'}"
            + (f" impl={args.map_groups_impl}" if args.map_groups else "")
        )

        # `spark.read.parquet` is EAGER -- it builds the InMemoryFileIndex,
        # listing every file and inferring the schema. Ray times the equivalent
        # work (its plan has an explicit ListFiles operator), so the timer has
        # to start before the read, not after it.
        start = time.perf_counter()
        df = spark.read.parquet(path)
        # Ray casts strings to large_string here to dodge Arrow's int32 offset
        # overflow on low-cardinality keys. Spark's UTF8String has no such
        # limit, so there is no equivalent step.

        if args.aggregate:
            # Ray: grouped_ds.mean("column05").materialize()
            out = df.groupBy(*args.group_by).agg(
                F.mean("column05").alias("mean(column05)")
            )
        else:
            out = (
                map_groups_window(df, args.group_by)
                if args.map_groups_impl == "window"
                else map_groups_pandas(df, args.group_by)
            )
        # noop sink: full materialization, no write, nothing to the driver.
        out.write.format("noop").mode("overwrite").save()
        elapsed = time.perf_counter() - start
        print(f"  elapsed {elapsed:.2f}s")

        stage = collect_stage_metrics(spark)
        if stage:
            print(
                f"  stages={stage['num_stages']} input={stage['input_gb']}GB "
                f"shuffle w/r={stage['shuffle_write_gb']}/"
                f"{stage['shuffle_read_gb']}GB "
                f"spill mem/disk={stage['mem_spilled_gb']}/"
                f"{stage['disk_spilled_gb']}GB"
            )
            print(
                f"  executor run/cpu: {stage['executor_run_s']}/"
                f"{stage['executor_cpu_s']} core-s"
            )

        result = {
            "timestamp": datetime.now().isoformat(),
            "engine": "spark",
            "benchmark": f"groupby_{mode}",
            "sf": args.sf,
            "group_by": args.group_by,
            "elapsed_s": elapsed,
            "config": {
                "map_groups_impl": args.map_groups_impl if args.map_groups else None,
                "num_partitions": args.num_partitions,
                "enable_aqe": args.enable_aqe,
                "backend": args.backend,
            },
            "stage_metrics": stage,
            "status": "ok",
        }
    finally:
        if args.backend == "raydp":
            import raydp

            raydp.stop_spark()
        else:
            spark.stop()

    _outdir = os.path.dirname(os.path.abspath(args.output))
    if _outdir:
        os.makedirs(_outdir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
