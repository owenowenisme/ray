"""PySpark port of the Ray Data TPC-H benchmarks (q1, q8, q13, q15, q21, q22).

    python spark_tpch.py --query q1 --sf 100
    for q in q1 q8 q13 q15 q21 q22; do python spark_tpch.py --query $q --sf 100; done

Each query mirrors ``release/nightly_tests/dataset/tpch/tpch_<q>.py`` statement
for statement, including where the Ray version materializes an intermediate
(Ray Data has no CSE, so those become ``.cache()`` here) and where it pulls a
scalar back to the driver.

That mirroring includes the rewrites Ray Data NEEDS because it cannot express
correlated subqueries or non-equi joins -- so this script measures "same plan,
which engine runs it faster?", not "which engine is faster on this query?".
For the latter, run ``spark_tpch_sql.py``, which uses the canonical TPC-H SQL
and lets Catalyst plan it. The gap is query-dependent and large for q21
(221.5s here vs 151.8s canonical at SF1000); quote both where they diverge.

FAIRNESS DEFAULTS -- read before quoting numbers
-----------------------------------------------
Ray executes every join as a 200-partition hash shuffle and executes the join
order as written. Spark would normally broadcast the small dimension tables
(nation, region, filtered part) and reorder joins. Those are real capability
differences, so this script exposes both modes and defaults to the
apples-to-apples one:

  * ``--allow-broadcast`` OFF by default -> ``autoBroadcastJoinThreshold=-1``, so
    every join is a shuffle join like Ray's. Compares SHUFFLE ENGINES.
  * ``--enable-aqe`` OFF by default -> shuffle partitions stay at
    ``--num-partitions`` (200, matching Ray's ``join_num_partitions``) instead of
    being coalesced.

Turn both ON to get the number a Spark user would actually see end-to-end. The
two answer different questions; say which one you are quoting.

Other deliberate differences from the Ray script
------------------------------------------------
  * Ray casts string columns to ``large_string`` to dodge Arrow's int32 offset
    overflow. Spark's UTF8String has no such limit, so there is no equivalent
    step and none is added.
  * Ray's ``to_f64`` casts DECIMAL/int to float64. In this parquet the price
    columns are already DOUBLE and ``l_quantity`` is BIGINT, so the casts are
    kept (no-ops on the doubles) to keep arithmetic types identical.
  * Spark reorders inner joins; Ray does not. The queries are written in Ray's
    order, and each engine plans it its own way. Use ``--explain`` to see what
    Spark chose.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from pyspark.sql import DataFrame, SparkSession, functions as F

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
try:
    from spark_join_benchmark import create_session
except ImportError as e:  # pragma: no cover - setup error
    raise SystemExit(
        f"could not import create_session from spark_join_benchmark: {e}\n"
        "Copy spark_join_benchmark.py next to this script."
    )

GIB = 1024**3

# Identical to tpch/common.py TABLE_COLUMNS.
TABLE_COLUMNS: Dict[str, Dict[str, str]] = {
    "region": {"column0": "r_regionkey", "column1": "r_name", "column2": "r_comment"},
    "nation": {
        "column0": "n_nationkey",
        "column1": "n_name",
        "column2": "n_regionkey",
        "column3": "n_comment",
    },
    "supplier": {
        "column0": "s_suppkey",
        "column1": "s_name",
        "column2": "s_address",
        "column3": "s_nationkey",
        "column4": "s_phone",
        "column5": "s_acctbal",
        "column6": "s_comment",
    },
    "customer": {
        "column0": "c_custkey",
        "column1": "c_name",
        "column2": "c_address",
        "column3": "c_nationkey",
        "column4": "c_phone",
        "column5": "c_acctbal",
        "column6": "c_mktsegment",
        "column7": "c_comment",
    },
    "orders": {
        "column0": "o_orderkey",
        "column1": "o_custkey",
        "column2": "o_orderstatus",
        "column3": "o_totalprice",
        "column4": "o_orderdate",
        "column5": "o_orderpriority",
        "column6": "o_clerk",
        "column7": "o_shippriority",
        "column8": "o_comment",
    },
    "part": {
        "column0": "p_partkey",
        "column1": "p_name",
        "column2": "p_mfgr",
        "column3": "p_brand",
        "column4": "p_type",
        "column5": "p_size",
        "column6": "p_container",
        "column7": "p_retailprice",
        "column8": "p_comment",
    },
    "partsupp": {
        "column0": "ps_partkey",
        "column1": "ps_suppkey",
        "column2": "ps_availqty",
        "column3": "ps_supplycost",
        "column4": "ps_comment",
    },
    "lineitem": {
        "column00": "l_orderkey",
        "column01": "l_partkey",
        "column02": "l_suppkey",
        "column03": "l_linenumber",
        "column04": "l_quantity",
        "column05": "l_extendedprice",
        "column06": "l_discount",
        "column07": "l_tax",
        "column08": "l_returnflag",
        "column09": "l_linestatus",
        "column10": "l_shipdate",
        "column11": "l_commitdate",
        "column12": "l_receiptdate",
        "column13": "l_shipinstruct",
        "column14": "l_shipmode",
        "column15": "l_comment",
    },
}


def load_table(spark: SparkSession, name: str, sf: int, base_uri: str) -> DataFrame:
    df = spark.read.parquet(f"{base_uri}/sf{sf}/{name}")
    mapping = TABLE_COLUMNS.get(name, {})
    if mapping:
        df = df.select(
            *[
                F.col(c).alias(mapping[c]) if c in mapping else F.col(c)
                for c in df.columns
            ]
        )
    return df


def f64(c: str):
    """Ray's ``to_f64``: force float64 so arithmetic types match exactly."""
    return F.col(c).cast("double")


def d(y: int, m: int, day: int):
    return F.lit(datetime(y, m, day).date()).cast("date")


# --------------------------------------------------------------------------
# Queries -- each mirrors tpch/tpch_<q>.py statement for statement
# --------------------------------------------------------------------------


def q1(spark: SparkSession, args) -> DataFrame:
    li = load_table(spark, "lineitem", args.sf, args.base_uri)
    li = li.filter(F.col("l_shipdate") <= d(1998, 9, 2))
    li = (
        li.withColumn("l_quantity_f", f64("l_quantity"))
        .withColumn("l_extendedprice_f", f64("l_extendedprice"))
        .withColumn("l_discount_f", f64("l_discount"))
        .withColumn("l_tax_f", f64("l_tax"))
        .withColumn(
            "disc_price", F.col("l_extendedprice_f") * (1 - F.col("l_discount_f"))
        )
        .withColumn("charge", F.col("disc_price") * (1 + F.col("l_tax_f")))
    ).select(
        "l_returnflag",
        "l_linestatus",
        "l_quantity_f",
        "l_extendedprice_f",
        "l_discount_f",
        "disc_price",
        "charge",
    )
    return (
        li.groupBy("l_returnflag", "l_linestatus")
        .agg(
            F.sum("l_quantity_f").alias("sum_qty"),
            F.sum("l_extendedprice_f").alias("sum_base_price"),
            F.sum("disc_price").alias("sum_disc_price"),
            F.sum("charge").alias("sum_charge"),
            F.mean("l_quantity_f").alias("avg_qty"),
            F.mean("l_extendedprice_f").alias("avg_price"),
            F.mean("l_discount_f").alias("avg_disc"),
            F.count(F.lit(1)).alias("count_order"),
        )
        .orderBy("l_returnflag", "l_linestatus")
    )


def q8(spark: SparkSession, args) -> DataFrame:
    region = load_table(spark, "region", args.sf, args.base_uri).select(
        "r_regionkey", "r_name"
    )
    nation = load_table(spark, "nation", args.sf, args.base_uri).select(
        "n_nationkey", "n_name", "n_regionkey"
    )
    supplier = load_table(spark, "supplier", args.sf, args.base_uri).select(
        "s_suppkey", "s_nationkey"
    )
    customer = load_table(spark, "customer", args.sf, args.base_uri).select(
        "c_custkey", "c_nationkey"
    )
    orders = load_table(spark, "orders", args.sf, args.base_uri).select(
        "o_orderkey", "o_custkey", "o_orderdate"
    )
    lineitem = load_table(spark, "lineitem", args.sf, args.base_uri).select(
        "l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"
    )
    part = load_table(spark, "part", args.sf, args.base_uri).select(
        "p_partkey", "p_type"
    )

    region_name, part_type, nation_name = "AMERICA", "ECONOMY ANODIZED STEEL", "BRAZIL"

    nation_region = region.filter(F.col("r_name") == region_name).join(
        nation, F.col("r_regionkey") == F.col("n_regionkey"), "inner"
    )
    customer_nation = nation_region.join(
        customer, F.col("n_nationkey") == F.col("c_nationkey"), "inner"
    ).select("c_custkey")

    orders_customer = (
        orders.filter(
            (F.col("o_orderdate") >= d(1995, 1, 1))
            & (F.col("o_orderdate") < d(1997, 1, 1))
        )
        .join(customer_nation, F.col("o_custkey") == F.col("c_custkey"), "inner")
        .select("o_orderkey", "o_orderdate")
    )

    lineitem_orders = lineitem.join(
        orders_customer, F.col("l_orderkey") == F.col("o_orderkey"), "inner"
    ).select(
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_extendedprice",
        "l_discount",
        "o_orderdate",
    )

    lineitem_part = lineitem_orders.join(
        part.filter(F.col("p_type") == part_type),
        F.col("l_partkey") == F.col("p_partkey"),
        "inner",
    ).select("l_suppkey", "l_extendedprice", "l_discount", "o_orderdate")

    lineitem_supplier = lineitem_part.join(
        supplier, F.col("l_suppkey") == F.col("s_suppkey"), "inner"
    ).select("l_extendedprice", "l_discount", "o_orderdate", "s_nationkey")

    # Second use of `nation` -- pre-rename instead of aliasing so the join keys
    # and output column are unambiguous (Ray renames n_name -> n_name_supp).
    nation_supp = nation.select(
        F.col("n_nationkey").alias("supp_nationkey"),
        F.col("n_name").alias("n_name_supp"),
    )
    ds = lineitem_supplier.join(
        nation_supp, F.col("s_nationkey") == F.col("supp_nationkey"), "inner"
    )

    ds = (
        ds.withColumn("volume", f64("l_extendedprice") * (1 - f64("l_discount")))
        .withColumn("o_year", F.year("o_orderdate"))
        .withColumn("is_nation", (F.col("n_name_supp") == nation_name).cast("double"))
        .withColumn("nation_volume", F.col("is_nation") * F.col("volume"))
    )

    return (
        ds.groupBy("o_year")
        .agg(
            F.sum("volume").alias("total_volume"),
            F.sum("nation_volume").alias("nation_volume"),
        )
        .withColumn("mkt_share", F.col("nation_volume") / F.col("total_volume"))
        .select("o_year", "mkt_share")
        .orderBy("o_year")
    )


def q13(spark: SparkSession, args) -> DataFrame:
    word1, word2 = "special", "requests"
    customers = load_table(spark, "customer", args.sf, args.base_uri).select(
        "c_custkey"
    )
    orders = load_table(spark, "orders", args.sf, args.base_uri).select(
        "o_orderkey", "o_custkey", "o_comment"
    )
    # NOT LIKE '%special%requests%'
    orders = orders.filter(~F.col("o_comment").rlike(f"{word1}.*{word2}")).select(
        "o_orderkey", "o_custkey"
    )

    joined = customers.join(
        orders, F.col("c_custkey") == F.col("o_custkey"), "left_outer"
    )
    c_orders = joined.groupBy("c_custkey").agg(
        F.count("o_orderkey").alias("c_count")  # count(col) skips nulls, as Ray does
    )
    return (
        c_orders.groupBy("c_count")
        .agg(F.count(F.lit(1)).alias("custdist"))
        .orderBy(F.col("custdist").desc(), F.col("c_count").desc())
    )


def q15(spark: SparkSession, args) -> DataFrame:
    supplier = load_table(spark, "supplier", args.sf, args.base_uri).select(
        "s_suppkey", "s_name", "s_address", "s_phone"
    )
    lineitem = load_table(spark, "lineitem", args.sf, args.base_uri).select(
        "l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"
    )

    lineitem = lineitem.filter(
        (F.col("l_shipdate") >= d(1996, 1, 1)) & (F.col("l_shipdate") < d(1996, 4, 1))
    ).withColumn("rev", f64("l_extendedprice") * (1 - f64("l_discount")))

    # Ray materializes `revenue` and derives the scalar max from it; cache to
    # mirror that (otherwise Spark recomputes it for the max and for the join).
    revenue = (
        lineitem.groupBy("l_suppkey").agg(F.sum("rev").alias("total_revenue")).cache()
    )
    max_revenue = revenue.agg(F.max("total_revenue")).collect()[0][0]
    top = revenue.filter(F.col("total_revenue") == F.lit(max_revenue))

    return (
        supplier.join(top, F.col("s_suppkey") == F.col("l_suppkey"), "inner")
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .orderBy("s_suppkey")
    )


def q21(spark: SparkSession, args) -> DataFrame:
    nation_name = "SAUDI ARABIA"
    supplier = load_table(spark, "supplier", args.sf, args.base_uri).select(
        "s_suppkey", "s_name", "s_nationkey"
    )
    lineitem = load_table(spark, "lineitem", args.sf, args.base_uri).select(
        "l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"
    )
    orders = load_table(spark, "orders", args.sf, args.base_uri).select(
        "o_orderkey", "o_orderstatus"
    )
    nation = load_table(spark, "nation", args.sf, args.base_uri).select(
        "n_nationkey", "n_name"
    )

    # EXISTS: >1 distinct supplier on the order.
    suppliers_per_order = (
        lineitem.select("l_orderkey", "l_suppkey")
        .groupBy("l_orderkey")
        .agg(F.countDistinct("l_suppkey").alias("num_suppliers"))
        .filter(F.col("num_suppliers") > 1)
    )

    # Ray materializes late_lineitem (used twice, and Ray Data has no CSE).
    late_lineitem = (
        lineitem.filter(F.col("l_receiptdate") > F.col("l_commitdate"))
        .select("l_orderkey", "l_suppkey")
        .cache()
    )

    late_suppliers_per_order = (
        late_lineitem.groupBy("l_orderkey")
        .agg(F.countDistinct("l_suppkey").alias("num_late_suppliers"))
        .filter(F.col("num_late_suppliers") == 1)
    )

    saudi_suppliers = supplier.join(
        nation.filter(F.col("n_name") == nation_name),
        F.col("s_nationkey") == F.col("n_nationkey"),
        "inner",
    ).select("s_suppkey", "s_name")

    failed_orders = orders.filter(F.col("o_orderstatus") == "F").select("o_orderkey")

    ds = late_lineitem.join(
        failed_orders, F.col("l_orderkey") == F.col("o_orderkey"), "left_semi"
    )
    ds = ds.join(saudi_suppliers, F.col("l_suppkey") == F.col("s_suppkey"), "inner")
    ds = ds.join(suppliers_per_order, "l_orderkey", "inner")
    ds = ds.join(late_suppliers_per_order, "l_orderkey", "inner")

    return (
        ds.groupBy("s_name")
        .agg(F.count(F.lit(1)).alias("numwait"))
        .orderBy(F.col("numwait").desc(), F.col("s_name").asc())
        .limit(100)
    )


def q22(spark: SparkSession, args) -> DataFrame:
    codes_regex = "^(13|31|23|29|30|18|17)$"
    customer = load_table(spark, "customer", args.sf, args.base_uri).select(
        "c_custkey", "c_phone", "c_acctbal"
    )
    orders = load_table(spark, "orders", args.sf, args.base_uri).select("o_custkey")

    customer = customer.withColumn(
        "cntrycode", F.substring(F.col("c_phone"), 1, 2)
    ).withColumn("c_acctbal_f", f64("c_acctbal"))

    customer_filtered = customer.filter(F.col("cntrycode").rlike(codes_regex))

    # Ray does NOT materialize customer_filtered, so it is recomputed for the
    # scalar average and for the main path. Left uncached to mirror that.
    avg_acctbal = (
        customer_filtered.filter(F.col("c_acctbal_f") > 0.0)
        .agg(F.avg("c_acctbal_f"))
        .collect()[0][0]
    )

    custsale = customer_filtered.filter(F.col("c_acctbal_f") > F.lit(avg_acctbal))
    # NOT EXISTS -> left anti join
    custsale = custsale.join(
        orders, F.col("c_custkey") == F.col("o_custkey"), "left_anti"
    )

    return (
        custsale.groupBy("cntrycode")
        .agg(
            F.count(F.lit(1)).alias("numcust"),
            F.sum("c_acctbal_f").alias("totacctbal"),
        )
        .orderBy("cntrycode")
    )


QUERIES: Dict[str, Callable[[SparkSession, Any], DataFrame]] = {
    "q1": q1,
    "q8": q8,
    "q13": q13,
    "q15": q15,
    "q21": q21,
    "q22": q22,
}


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def collect_stage_metrics(spark: SparkSession) -> Dict[str, Any]:
    """Totals from the Spark UI REST API. Best-effort."""
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


def cluster_cores(spark: SparkSession) -> Optional[int]:
    """Total executor cores, from the REST API (defaultParallelism lies early)."""
    try:
        import requests

        ui = spark.sparkContext.uiWebUrl
        app_id = spark.sparkContext.applicationId
        r = requests.get(f"{ui}/api/v1/applications/{app_id}/executors", timeout=15)
        r.raise_for_status()
        return (
            sum(
                int(e.get("totalCores", 0) or 0)
                for e in r.json()
                if e.get("id") != "driver"
            )
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def warn_if_timer_missed_work(
    spark: SparkSession, elapsed: float, stage: Dict[str, Any]
) -> Optional[str]:
    """Catch a timer that does not span all the work the query actually did.

    Executor run time cannot exceed ``elapsed * total_cores``. If it does, the
    measured window missed a Spark job -- which is exactly what happened when
    the timer started after query construction and q15/q22 ran their scalar
    subquery inside the builder.
    """
    run_s = stage.get("executor_run_s")
    if not run_s or not elapsed:
        return None
    implied = run_s / elapsed
    cores = cluster_cores(spark)
    if cores and implied <= cores * 1.2:
        return None
    if not cores and implied < 2000:
        return None
    msg = (
        f"executor_run_s={run_s:.0f} over elapsed={elapsed:.2f}s implies "
        f"{implied:.0f} busy cores"
        + (f", but the cluster has {cores}" if cores else "")
        + ". The timed window does not cover all the work -- treat this "
        "elapsed time as INVALID."
    )
    print(f"  !! WARNING: {msg}")
    return msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--query", required=True, choices=sorted(QUERIES))
    p.add_argument("--sf", type=int, default=1, choices=[1, 10, 100, 1000, 10000])
    p.add_argument("--base-uri", default="s3a://ray-benchmark-data/tpch/parquet")
    p.add_argument("--output", default=None, help="default: spark_<query>_results.json")
    p.add_argument("--explain", action="store_true", help="print the physical plan")
    p.add_argument("--show", type=int, default=20, help="result rows to print (0=none)")

    # Fairness switches (see module docstring).
    p.add_argument(
        "--allow-broadcast",
        "--allow_broadcast_join",
        action="store_true",
        dest="allow_broadcast",
    )
    p.add_argument(
        "--enable-aqe", "--enable_aqe", action="store_true", dest="enable_aqe"
    )
    p.add_argument(
        "--num-partitions",
        type=int,
        default=200,
        help="spark.sql.shuffle.partitions; 200 matches Ray's join_num_partitions",
    )

    # Spark backend (names match what spark_join_benchmark's helpers expect).
    p.add_argument("--backend", default="raydp", choices=["raydp", "standalone"])
    p.add_argument("--master", default="local[*]")
    p.add_argument("--num_executors", type=int, default=32)
    p.add_argument("--executor_cores", type=int, default=16)
    p.add_argument("--executor_memory", default="32g")
    p.add_argument("--spark_conf", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    # `build_spark_conf` reads these exact attribute names.
    args.enable_aqe = args.enable_aqe
    args.allow_broadcast_join = args.allow_broadcast
    args.num_partitions = args.num_partitions
    if args.output is None:
        args.output = f"spark_{args.query}_results.json"
    return args


def main() -> None:
    args = parse_args()
    try:
        spark = create_session(args, app_name=f"spark_tpch_{args.query}")
    except TypeError:
        spark = create_session(args)

    result: Dict[str, Any] = {}
    try:
        print(f"\n=== {args.query} (sf{args.sf}) ===")
        print(
            f"  broadcast={'ON' if args.allow_broadcast else 'OFF'}  "
            f"aqe={'ON' if args.enable_aqe else 'OFF'}  "
            f"shuffle.partitions={args.num_partitions}"
        )

        # The timer MUST wrap query construction as well as the final collect:
        # q15 and q22 pull a scalar back to the driver mid-query (max revenue /
        # avg balance), which runs a full Spark job inside the builder. Timing
        # only the final collect() attributed ~90GB of work to zero seconds.
        # Ray's `run_tpch_benchmark` wraps its whole `benchmark_fn` for the same
        # reason, so this keeps the two measuring the same span.
        start = time.perf_counter()
        df = QUERIES[args.query](spark, args)
        # TPC-H results are tiny (<=100 rows), so collect() is the faithful
        # analogue of Ray's `.materialize()` on the final Dataset.
        rows = df.collect()
        elapsed = time.perf_counter() - start
        print(f"  elapsed {elapsed:.2f}s, {len(rows)} result rows")

        if args.explain:
            df.explain(mode="formatted")

        if args.show:
            for row in rows[: args.show]:
                print(f"    {row}")

        stage = collect_stage_metrics(spark)
        timer_warning = None
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
            timer_warning = warn_if_timer_missed_work(spark, elapsed, stage)

        result = {
            "timestamp": datetime.now().isoformat(),
            "engine": "spark",
            "timer_warning": timer_warning,
            "query": args.query,
            "sf": args.sf,
            "elapsed_s": elapsed,
            "num_result_rows": len(rows),
            "result_preview": [r.asDict() for r in rows[:20]],
            "config": {
                "impl": "dataframe",
                "allow_broadcast": args.allow_broadcast,
                "enable_aqe": args.enable_aqe,
                "num_partitions": args.num_partitions,
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
