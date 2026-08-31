"""Spark port of ``join_benchmark.py``.

Runs the same TPC-H lineitem-x-orders join that the Ray Data join release test
runs, so the two shuffle implementations can be compared apples-to-apples on the
same cluster shape and the same S3 parquet inputs.

Two backends:

* ``--backend raydp``      Spark executors run as Ray actors on the Ray cluster
                           (RayDP). Use this to compare against Ray Data on the
                           exact same nodes.
* ``--backend standalone`` A plain ``SparkSession``. Point ``--master`` at a
                           standalone/YARN/k8s master, or leave the default
                           ``local[*]`` for a smoke test.

The join is forced to be shuffle-based (broadcast join disabled, AQE off by
default) because the point of the benchmark is the shuffle, not the planner.

Results are written in the same JSON shape as ``benchmark.Benchmark``, so
they slot into the same comparison tooling.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

# Spark accepts these names verbatim, but map explicitly so the CLI stays
# identical to join_benchmark.py.
JOIN_TYPES = {
    "inner": "inner",
    "left_outer": "left_outer",
    "right_outer": "right_outer",
    "full_outer": "full_outer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # --- Mirrors join_benchmark.py ---
    parser.add_argument(
        "--left_dataset", required=True, type=str, help="Path to the left dataset"
    )
    parser.add_argument(
        "--right_dataset", required=True, type=str, help="Path to the right dataset"
    )
    parser.add_argument(
        "--num_partitions",
        required=True,
        type=int,
        help="Number of shuffle partitions to use for the join",
    )
    parser.add_argument(
        "--left_join_keys",
        required=True,
        nargs="+",
        type=str,
        help="Join keys for the left dataset",
    )
    parser.add_argument(
        "--right_join_keys",
        required=True,
        nargs="+",
        type=str,
        help="Join keys for the right dataset",
    )
    parser.add_argument(
        "--join_type",
        required=True,
        choices=sorted(JOIN_TYPES),
        help="Type of join operation",
    )

    # --- Spark-specific ---
    parser.add_argument(
        "--backend",
        default="raydp",
        choices=["raydp", "standalone"],
        help="Run Spark executors as Ray actors (raydp) or against a Spark master",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL. Only used by --backend standalone",
    )
    parser.add_argument(
        "--num_executors",
        type=int,
        default=32,
        help="RayDP only: number of Spark executors to request",
    )
    parser.add_argument(
        "--executor_cores",
        type=int,
        default=16,
        help="RayDP only: CPU cores per executor",
    )
    parser.add_argument(
        "--executor_memory",
        default="32g",
        help="RayDP only: heap memory per executor",
    )
    parser.add_argument(
        "--action",
        default="noop",
        choices=["noop", "count"],
        help=(
            "How to materialize the join. 'noop' pushes every output column "
            "through the shuffle (fair vs Ray Data). 'count' matches "
            "join_benchmark.py literally but lets Spark prune all non-key columns"
        ),
    )
    parser.add_argument(
        "--enable_aqe",
        "--enable-aqe",
        action="store_true",
        dest="enable_aqe",
        help=(
            "Enable adaptive query execution. Off by default so the shuffle runs "
            "with exactly --num_partitions partitions, like Ray Data does"
        ),
    )
    parser.add_argument(
        "--allow_broadcast_join",
        "--allow-broadcast",
        action="store_true",
        dest="allow_broadcast_join",
        help=(
            "Allow Spark to broadcast the small side. Off by default; a broadcast "
            "join skips the shuffle entirely and is not comparable to Ray Data"
        ),
    )
    parser.add_argument(
        "--spark_conf",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Spark configs, e.g. --spark_conf spark.sql.files.maxPartitionBytes=134217728",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Where to write results. Defaults to $TEST_OUTPUT_JSON or ./result.json",
    )
    return parser.parse_args()


def _s3a_jars_preinstalled() -> bool:
    """True if hadoop-aws is already on the classpath (baked into the image).

    When it is, we must not set ``spark.jars.packages``: that would send every
    executor through Ivy resolution against Maven Central on startup, which is
    slow and fails outright on clusters without egress.
    """
    import glob

    import pyspark

    jars_dir = os.path.join(os.path.dirname(pyspark.__file__), "jars")
    return bool(glob.glob(os.path.join(jars_dir, "hadoop-aws-*.jar")))


def build_spark_conf(args: argparse.Namespace) -> Dict[str, str]:
    conf = {
        "spark.sql.shuffle.partitions": str(args.num_partitions),
        "spark.sql.adaptive.enabled": str(args.enable_aqe).lower(),
        # -1 disables broadcast, forcing a sort-merge / shuffle-hash join.
        "spark.sql.autoBroadcastJoinThreshold": (
            "-1" if not args.allow_broadcast_join else "10485760"
        ),
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": (
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
        ),
        "spark.eventLog.enabled": "false",
    }

    if not _s3a_jars_preinstalled():
        # Fall back to fetching S3A at runtime. Bundled hadoop with
        # pyspark 3.5.x is 3.3.4, so the connector must match.
        conf["spark.jars.packages"] = (
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )

    for kv in args.spark_conf:
        # Empty values are legal ("spark.jars.packages=" clears a default).
        if "=" not in kv:
            raise ValueError(f"--spark_conf entries must be KEY=VALUE, got: {kv!r}")
        key, _, value = kv.partition("=")
        conf[key] = value
    return conf


_MEMORY_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_memory(value: str) -> int:
    """Parse RayDP-style memory strings ("42g", "512m", "1024") into bytes."""
    text = str(value).strip().lower().rstrip("b")
    if text and text[-1] in _MEMORY_UNITS:
        return int(float(text[:-1]) * _MEMORY_UNITS[text[-1]])
    return int(text)


def check_cluster_fit(args: argparse.Namespace) -> None:
    """Fail fast if Ray cannot place the requested executors.

    Each Spark executor is a single Ray actor, so its whole CPU+memory bundle
    must fit on one node. When it doesn't, Spark does not error -- it parks in
    a 'Initial job has not accepted any resources' loop until the job times
    out, which wastes a full cluster-hour. Check up front instead.
    """
    import ray

    executor_memory = parse_memory(args.executor_memory)
    nodes = [n for n in ray.nodes() if n.get("Alive")]

    placeable = 0
    largest = (0.0, 0)
    for node in nodes:
        resources = node.get("Resources", {})
        cpu = resources.get("CPU", 0)
        memory = resources.get("memory", 0)
        largest = max(largest, (cpu, memory))
        if args.executor_cores > 0 and executor_memory > 0:
            placeable += min(
                int(cpu // args.executor_cores), int(memory // executor_memory)
            )

    if placeable >= args.num_executors:
        return

    gib = 1024**3
    node_cpu, node_memory = largest
    # Ray reserves ~30% of RAM for the object store, so a node's schedulable
    # `memory` resource is well below its physical RAM. That gap is the usual
    # reason a seemingly-fine executor size will not place.
    suggested = int((node_memory / gib) * 0.8)
    raise RuntimeError(
        f"Ray can only place {placeable} of the {args.num_executors} requested "
        f"Spark executors.\n"
        f"  Requested per executor: {args.executor_cores} CPU, "
        f"{executor_memory / gib:.1f} GiB\n"
        f"  Largest node offers:    {node_cpu:g} CPU, "
        f"{node_memory / gib:.1f} GiB schedulable memory "
        f"(across {len(nodes)} alive nodes)\n"
        f"Note that a node's schedulable `memory` is its RAM minus the Ray "
        f"object store (~30% by default), so it is much lower than the "
        f"instance's advertised RAM.\n"
        f"Try: --executor_cores {int(min(args.executor_cores, node_cpu))} "
        f"--executor_memory {max(suggested, 1)}g\n"
        f"Run `ray status` to see the full cluster breakdown."
    )


def create_session(
    args: argparse.Namespace, app_name: str = "spark_join_benchmark"
) -> SparkSession:
    conf = build_spark_conf(args)

    if args.backend == "raydp":
        import raydp

        import ray

        if not ray.is_initialized():
            ray.init(address="auto")
        check_cluster_fit(args)
        return raydp.init_spark(
            app_name=app_name,
            num_executors=args.num_executors,
            executor_cores=args.executor_cores,
            executor_memory=args.executor_memory,
            configs=conf,
        )

    builder = SparkSession.builder.appName(app_name).master(args.master)
    for key, value in conf.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def s3_to_s3a(path: str) -> str:
    """Spark's S3 connector is registered under the ``s3a`` scheme."""
    if path.startswith("s3://"):
        return "s3a://" + path[len("s3://") :]
    return path


def _stage_wall_seconds(submission_time: Any, completion_time: Any) -> Optional[float]:
    """Wall-clock seconds for a stage from the UI's ISO-ish timestamps.

    Spark formats these as "2026-08-27T20:43:36.123GMT", which datetime cannot
    parse directly.
    """
    if not submission_time or not completion_time:
        return None
    fmt = "%Y-%m-%dT%H:%M:%S.%f%Z"
    try:
        start = datetime.strptime(submission_time.replace("GMT", "UTC"), fmt)
        end = datetime.strptime(completion_time.replace("GMT", "UTC"), fmt)
    except ValueError:
        return None
    return round((end - start).total_seconds(), 2)


def collect_shuffle_metrics(spark: SparkSession) -> Dict[str, Any]:
    """Scrape per-stage shuffle metrics from the Spark UI REST API.

    Returns an empty dict if the UI is unreachable (e.g. ``spark.ui.enabled=false``),
    so a metrics failure never fails the benchmark itself.
    """
    try:
        import requests

        ui_url = spark.sparkContext.uiWebUrl
        if not ui_url:
            return {}
        app_id = spark.sparkContext.applicationId
        resp = requests.get(f"{ui_url}/api/v1/applications/{app_id}/stages", timeout=30)
        resp.raise_for_status()
        stages: List[Dict[str, Any]] = resp.json()
    except Exception as e:  # noqa: BLE001 - metrics are best-effort
        print(f"Could not collect Spark shuffle metrics: {e}")
        return {}

    def total(field: str) -> int:
        return sum(int(stage.get(field, 0) or 0) for stage in stages)

    bytes_per_gb = 1024**3

    # Per-stage breakdown. Without this, a wall-clock comparison against Ray
    # Data cannot tell a shuffle difference from a parquet-reader difference:
    # the scan stages are the ones with inputBytes, the shuffle stages are the
    # ones with shuffleWriteBytes.
    per_stage = []
    for stage in sorted(stages, key=lambda s: s.get("stageId", 0)):
        submit, complete = stage.get("submissionTime"), stage.get("completionTime")
        per_stage.append(
            {
                "stage_id": stage.get("stageId"),
                "name": (stage.get("name") or "")[:80],
                "num_tasks": stage.get("numTasks"),
                "wall_s": _stage_wall_seconds(submit, complete),
                "input_gb": round(
                    int(stage.get("inputBytes", 0) or 0) / bytes_per_gb, 3
                ),
                "shuffle_write_gb": round(
                    int(stage.get("shuffleWriteBytes", 0) or 0) / bytes_per_gb, 3
                ),
                "shuffle_read_gb": round(
                    int(stage.get("shuffleReadBytes", 0) or 0) / bytes_per_gb, 3
                ),
                "disk_spilled_gb": round(
                    int(stage.get("diskBytesSpilled", 0) or 0) / bytes_per_gb, 3
                ),
            }
        )

    print("\nPer-stage breakdown:")
    for stage in per_stage:
        print(
            f"  stage {stage['stage_id']:>3} {stage['name'][:40]:<42} "
            f"tasks={stage['num_tasks']:<6} wall={stage['wall_s']}s "
            f"in={stage['input_gb']}GB sw={stage['shuffle_write_gb']}GB "
            f"sr={stage['shuffle_read_gb']}GB spill={stage['disk_spilled_gb']}GB"
        )

    return {
        "spark_stages": per_stage,
        "spark_shuffle_write_gb": round(total("shuffleWriteBytes") / bytes_per_gb, 4),
        "spark_shuffle_read_gb": round(
            (total("shuffleReadBytes") or total("shuffleReadRemoteBytes"))
            / bytes_per_gb,
            4,
        ),
        "spark_memory_spilled_gb": round(total("memoryBytesSpilled") / bytes_per_gb, 4),
        "spark_disk_spilled_gb": round(total("diskBytesSpilled") / bytes_per_gb, 4),
        "spark_input_gb": round(total("inputBytes") / bytes_per_gb, 4),
        "spark_num_stages": len(stages),
        "spark_executor_run_time_s": round(total("executorRunTime") / 1000.0, 2),
        "spark_executor_cpu_time_s": round(total("executorCpuTime") / 1e9, 2),
    }


def run_join(spark: SparkSession, args: argparse.Namespace) -> Dict[str, Any]:
    if len(args.left_join_keys) != len(args.right_join_keys):
        raise ValueError("Number of left and right join keys must match.")

    left: DataFrame = spark.read.parquet(s3_to_s3a(args.left_dataset)).alias("l")
    right: DataFrame = spark.read.parquet(s3_to_s3a(args.right_dataset)).alias("r")

    condition = None
    for left_key, right_key in zip(args.left_join_keys, args.right_join_keys):
        predicate = F.col(f"l.{left_key}") == F.col(f"r.{right_key}")
        condition = predicate if condition is None else (condition & predicate)

    joined = left.join(right, on=condition, how=JOIN_TYPES[args.join_type])

    print("Physical plan:")
    joined.explain(mode="formatted")

    if args.action == "count":
        # Mirrors join_benchmark.py exactly. Caveat: Spark's optimizer prunes
        # every column except the join keys, so far less data crosses the
        # shuffle than Ray Data moves. Good for parity of the written script,
        # bad for parity of the shuffle. Prefer --action noop.
        num_rows = joined.count()
        print(f"Join completed with {num_rows} records.")
        return {"num_rows": num_rows}

    # The "noop" sink materializes every output row and every column but writes
    # nothing, which is the closest analogue to Ray Data materializing the join
    # output. Column pruning cannot kick in, so the shuffle carries the full
    # row payload.
    joined.write.format("noop").mode("overwrite").save()
    print("Join completed (noop sink; pass --action count to get a row count).")
    return {}


def main(args: argparse.Namespace) -> None:
    spark = create_session(args)
    case_name = str(vars(args))
    result: Dict[str, Any] = {}

    try:
        start = time.perf_counter()
        try:
            metrics = run_join(spark, args)
        finally:
            duration = time.perf_counter() - start

        metrics["time"] = duration
        metrics.update(collect_shuffle_metrics(spark))
        result[case_name] = metrics
        print(f"Result of case {case_name}: {metrics}")
    finally:
        if args.backend == "raydp":
            import raydp

            raydp.stop_spark()
        else:
            spark.stop()

    output_json: Optional[str] = args.output_json or os.environ.get(
        "TEST_OUTPUT_JSON", "./result.json"
    )
    _outdir = os.path.dirname(os.path.abspath(output_json))
    if _outdir:
        os.makedirs(_outdir, exist_ok=True)
    with open(output_json, "w") as f:
        f.write(json.dumps(result))
    print(f"Finished benchmark, metrics exported to '{output_json}':")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main(parse_args())
