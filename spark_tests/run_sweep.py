#!/usr/bin/env python3
"""Run the Ray Data and Spark benchmark suites sequentially and report a
side-by-side comparison.

    # everything, both engines (long -- see the time estimate it prints)
    python run_sweep.py --engine both

    # just look at the plan first
    python run_sweep.py --engine both --dry-run

    # one group, one engine
    python run_sweep.py --engine ray --only tpch
    python run_sweep.py --engine spark --only joins

    # resume after a crash / interrupt (default: completed runs are skipped)
    python run_sweep.py --engine both

Covers, with parameters taken verbatim from release_data_tests.yaml:

  tpch_{q1,q8,q13,q15,q21,q22}_fixed_size_shuffle_v2      SF1000
  aggregate_groups_fixed_size_shuffle_v2_{2 key sets}     SF1000, 500 partitions
  map_groups_fixed_size_shuffle_v2_{2 key sets}           SF1000 / SF100 (see below)
  joins_sf1000_{inner,left_outer,right_outer,full_outer}_shuffle_v2

NOTE: ``map_groups`` on the 84-group key (``column08 column13 column14``) runs at
SF100, not SF1000, and without ``--num-partitions``. That is not an oversight --
release_data_tests.yaml pins it there with the comment "map_groups v2 on the
84-group key stays at SF100 because there's data skew in partition that makes
the task unschedulable." The SF1000 numbers are therefore NOT comparable across
the two map_groups key sets.

Results are written incrementally, so an interrupted sweep loses at most the run
in flight, and re-running resumes where it stopped.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.realpath(__file__))

# Env the release tests set for every shuffle_v2 run. RAYTEST_FAIL_ON_* are
# consumed by the Anyscale release harness, not by the benchmark scripts, so
# they are omitted here -- they would have no effect.
RAY_BASE_ENV = {
    "RAY_DATA_DEBUG_RESOURCE_MANAGER": "1",
    "RAY_max_direct_call_object_size": "8192",
}
RAY_V2_ENV = dict(RAY_BASE_ENV, RAY_DATA_DEFAULT_SHUFFLE_STRATEGY="shuffle_v2")

TPCH_QUERIES = ["q1", "q8", "q13", "q15", "q21", "q22"]
JOIN_TYPES = ["inner", "left_outer", "right_outer", "full_outer"]
GROUP_KEYS = [
    ("column08 column13 column14", "84groups"),
    ("column02 column14", "7Mgroups"),
]
S3 = "s3://ray-benchmark-data/tpch/parquet"


def build_matrix(tpch_impl: str = "dataframe") -> List[Dict[str, Any]]:
    """One entry per benchmark, with the Ray and Spark invocation for each.

    ``tpch_impl`` selects which Spark script runs the TPC-H arm: ``dataframe``
    mirrors the Ray plan statement for statement (spark_tpch.py), ``sql`` uses
    the canonical TPC-H SQL and lets Catalyst plan it (spark_tpch_sql.py). Both
    take the same ``--query``/``--sf`` flags. Use ``--tag`` to keep the two
    sweeps' results from overwriting each other.
    """
    out: List[Dict[str, Any]] = []
    tpch_script = "spark_tpch.py" if tpch_impl == "dataframe" else "spark_tpch_sql.py"

    for q in TPCH_QUERIES:
        out.append(
            {
                "name": f"tpch_{q}_shuffle_v2_sf1000",
                "group": "tpch",
                "ray": {
                    "script": os.path.join("tpch", f"tpch_{q}.py"),
                    "args": ["--sf", "1000"],
                    "env": RAY_V2_ENV,
                },
                "spark": {
                    "script": tpch_script,
                    "args": ["--query", q, "--sf", "1000"],
                    "env": {},
                },
            }
        )

    for cols, label in GROUP_KEYS:
        out.append(
            {
                "name": f"aggregate_groups_shuffle_v2_{label}_sf1000",
                "group": "aggregate_groups",
                "ray": {
                    "script": "groupby_benchmark.py",
                    "args": [
                        "--sf",
                        "1000",
                        "--aggregate",
                        "--group-by",
                        *cols.split(),
                        "--shuffle-strategy",
                        "shuffle_v2",
                        "--num-partitions",
                        "500",
                    ],
                    "env": RAY_BASE_ENV,
                },
                "spark": {
                    "script": "spark_groupby_benchmark.py",
                    "args": [
                        "--sf",
                        "1000",
                        "--aggregate",
                        "--group-by",
                        *cols.split(),
                        "--num-partitions",
                        "500",
                    ],
                    "env": {},
                },
            }
        )

    # map_groups: the 84-group key is pinned to SF100 with no --num-partitions.
    for cols, label in GROUP_KEYS:
        sf = "100" if label == "84groups" else "1000"
        parts = [] if label == "84groups" else ["--num-partitions", "500"]
        out.append(
            {
                "name": f"map_groups_shuffle_v2_{label}_sf{sf}",
                "group": "map_groups",
                "ray": {
                    "script": "groupby_benchmark.py",
                    "args": [
                        "--sf",
                        sf,
                        "--map-groups",
                        "--group-by",
                        *cols.split(),
                        "--shuffle-strategy",
                        "shuffle_v2",
                        *parts,
                    ],
                    "env": RAY_BASE_ENV,
                },
                "spark": {
                    "script": "spark_groupby_benchmark.py",
                    "args": [
                        "--sf",
                        sf,
                        "--map-groups",
                        "--group-by",
                        *cols.split(),
                        *parts,
                    ],
                    "env": {},
                },
            }
        )

    for jt in JOIN_TYPES:
        out.append(
            {
                "name": f"joins_sf1000_{jt}_shuffle_v2",
                "group": "joins",
                "ray": {
                    "script": "join_benchmark.py",
                    "args": [
                        "--left_dataset",
                        f"{S3}/sf1000/lineitem",
                        "--right_dataset",
                        f"{S3}/sf1000/orders",
                        "--left_join_keys",
                        "column00",
                        "--right_join_keys",
                        "column0",
                        "--join_type",
                        jt,
                        "--num_partitions",
                        "1000",
                    ],
                    "env": RAY_V2_ENV,
                },
                "spark": {
                    "script": "spark_join_benchmark.py",
                    "args": [
                        "--left_dataset",
                        f"{S3}/sf1000/lineitem",
                        "--right_dataset",
                        f"{S3}/sf1000/orders",
                        "--left_join_keys",
                        "column00",
                        "--right_join_keys",
                        "column0",
                        "--join_type",
                        jt,
                        "--num_partitions",
                        "1000",
                    ],
                    "env": {},
                },
            }
        )

    return out


# --------------------------------------------------------------------------
# Result extraction
# --------------------------------------------------------------------------


def extract_elapsed(payload: Any) -> Optional[float]:
    """Pull the wall-clock seconds out of either result format.

    Ray's ``Benchmark.write_result`` emits ``{case_name: {"time": s, ...}}``;
    the Spark scripts emit ``{"elapsed_s": s, ...}`` at the top level.
    """
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("elapsed_s"), (int, float)):
        return float(payload["elapsed_s"])
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("time"), (int, float)):
            return float(value["time"])
    return None


def extract_extras(payload: Any) -> Dict[str, Any]:
    """Secondary metrics worth putting in the report, if present."""
    keys = (
        "object_store_spilled_total_gb",
        "object_store_memory_utilization_peak",
        "num_result_rows",
    )
    out: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out
    for k in keys:
        if k in payload:
            out[k] = payload[k]
    for value in payload.values():
        if isinstance(value, dict):
            for k in keys:
                if k in value:
                    out.setdefault(k, value[k])
    sm = payload.get("stage_metrics") or {}
    for k in ("input_gb", "shuffle_write_gb", "disk_spilled_gb", "executor_cpu_s"):
        if k in sm:
            out[k] = sm[k]
    return out


ELAPSED_RE = re.compile(r"execution finished in ([0-9.]+) seconds")


def elapsed_from_log(text: str) -> Optional[float]:
    m = ELAPSED_RE.findall(text)
    return float(m[-1]) if m else None


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def wait_for_object_store(timeout_s: float, threshold_pct: float = 20.0) -> None:
    """Best-effort: let Ray's object store drain before the next run."""
    if timeout_s <= 0:
        return
    try:
        import ray

        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False)
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            total = ray.cluster_resources().get("object_store_memory", 0)
            avail = ray.available_resources().get("object_store_memory", 0)
            if total <= 0:
                return
            used_pct = 100.0 * (1 - avail / total)
            if used_pct < threshold_pct:
                return
            print(f"      object store {used_pct:.0f}% used, waiting...", flush=True)
            time.sleep(5)
    except Exception:
        return


def run_one(
    bench: Dict[str, Any], engine: str, args: argparse.Namespace
) -> Dict[str, Any]:
    spec = bench[engine]
    workdir = args.ray_dir if engine == "ray" else args.spark_dir
    script = os.path.join(workdir, spec["script"])
    if not os.path.exists(script):
        return {"status": "missing_script", "detail": script}

    out_json = os.path.join(args.results_dir, f"{bench['name']}__{engine}.json")
    log_path = os.path.join(args.log_dir, f"{bench['name']}__{engine}.log")

    cmd = [sys.executable, spec["script"], *spec["args"]]
    if engine == "spark":
        cmd += ["--output", out_json]
        # Pin executor sizing for EVERY Spark run. The per-script defaults
        # disagree (the join benchmark defaults to 15 cores / 42g, the others
        # to 16 / 32g), which would silently give the joins 480 cores and
        # everything else 512 -- not comparable within one sweep.
        extra = shlex.split(args.spark_args)
        for flag, value in (
            ("--num_executors", str(args.executors)),
            ("--executor_cores", str(args.cores)),
            ("--executor_memory", args.memory),
        ):
            if flag not in extra:
                cmd += [flag, value]
        cmd += extra

    env = dict(os.environ)
    env.update(spec.get("env", {}))
    if engine == "ray":
        # Every Ray benchmark writes via Benchmark.write_result(), which honours
        # TEST_OUTPUT_JSON. Point it at a per-run file so runs do not clobber
        # each other's ./result.json.
        env["TEST_OUTPUT_JSON"] = out_json

    print(f"    $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    if args.dry_run:
        return {"status": "dry_run"}

    start = time.perf_counter()
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        rc = proc.returncode
        status = "ok" if rc == 0 else f"exit_{rc}"
    except subprocess.TimeoutExpired:
        rc, status = None, "timeout"
    wall = time.perf_counter() - start

    payload = None
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                payload = json.load(f)
        except Exception:
            payload = None

    elapsed = extract_elapsed(payload)
    if elapsed is None:
        try:
            with open(log_path) as f:
                elapsed = elapsed_from_log(f.read())
        except OSError:
            pass

    result = {
        "status": status,
        "returncode": rc,
        "wall_s": round(wall, 1),
        "elapsed_s": elapsed,
        "log": log_path,
        "result_json": out_json if payload is not None else None,
        **extract_extras(payload),
    }
    marker = "ok" if status == "ok" else status.upper()
    print(
        f"      -> {marker}  elapsed={elapsed if elapsed is not None else 'n/a'}  "
        f"(process wall {wall:.1f}s)",
        flush=True,
    )
    if status != "ok":
        print(f"         see {log_path}", flush=True)
    return result


def save(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def report(data: Dict[str, Any]) -> None:
    """Relative speed, normalized to Spark.

    Reported as speed rather than an elapsed-time ratio because "2x" is
    ambiguous otherwise. Spark is the baseline at 1.00x; Ray's figure is
    ``spark_seconds / ray_seconds``, so >1.00x means Ray is that many times
    FASTER and <1.00x means slower.
    """
    runs = data["runs"]
    print("\n" + "=" * 100)
    print(
        "SWEEP RESULTS  --  relative speed, Spark = 1.00x  "
        "(>1.00x = Ray faster, <1.00x = Ray slower)"
    )
    print("=" * 100)
    print(
        f"  {'benchmark':<44}{'ray s':>9}{'spark s':>10}{'spark':>9}{'ray':>9}  notes"
    )
    print("-" * 100)

    speeds = []
    for name, entry in runs.items():
        ray_r = entry.get("ray") or {}
        spk_r = entry.get("spark") or {}
        rt, st = ray_r.get("elapsed_s"), spk_r.get("elapsed_s")

        def fmt(v, r, width):
            if isinstance(v, (int, float)):
                return f"{v:>{width}.1f}"
            return f"{(r.get('status') or '-'):>{width}}"

        spark_col, ray_col = "", ""
        if isinstance(rt, (int, float)) and isinstance(st, (int, float)) and rt > 0:
            spark_col, ray_col = "1.00x", f"{st / rt:.2f}x"
            speeds.append(st / rt)

        notes = []
        for eng, r in (("ray", ray_r), ("spark", spk_r)):
            if r and r.get("status") not in (None, "ok"):
                notes.append(f"{eng}:{r['status']}")
            if (r or {}).get("timer_warning"):
                notes.append(f"{eng}:INVALID TIMER")
            spill = (r or {}).get("object_store_spilled_total_gb")
            if spill:
                notes.append(f"{eng} spill {spill:.0f}GB")
            read = (r or {}).get("input_gb")
            if read:
                notes.append(f"{eng} read {read:.0f}GB")
        print(
            f"  {name:<44}{fmt(rt, ray_r, 9)}{fmt(st, spk_r, 10)}"
            f"{spark_col:>9}{ray_col:>9}  {', '.join(notes)}"
        )

    print("-" * 100)
    if speeds:
        speeds.sort()
        n = len(speeds)
        mid = speeds[n // 2] if n % 2 else (speeds[n // 2 - 1] + speeds[n // 2]) / 2
        wins = sum(1 for x in speeds if x >= 1.0)
        print(
            f"  {n} paired runs   Ray median {mid:.2f}x of Spark   "
            f"range {speeds[0]:.2f}x - {speeds[-1]:.2f}x   "
            f"Ray at-or-faster in {wins}/{n}"
        )
    print("  NOTE: Spark defaults to broadcast joins OFF and AQE OFF, so joins are")
    print("        shuffle joins like Ray's. Pass --spark-args '--allow-broadcast")
    print("        --enable-aqe' for the number a Spark user would actually see.")
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--engine", default="both", choices=["ray", "spark", "both"])
    p.add_argument(
        "--only",
        default=None,
        help="substring filter on benchmark name or group "
        "(tpch / aggregate_groups / map_groups / joins)",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=90.0,
        help="seconds to idle between runs (default 90)",
    )
    p.add_argument(
        "--drain-wait",
        type=float,
        default=180.0,
        help="extra seconds to wait for Ray's object store to drain",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=7200.0,
        help="per-run timeout in seconds (release tests use 3600-7200)",
    )
    p.add_argument("--ray-dir", default=HERE)
    p.add_argument("--spark-dir", default=HERE)
    p.add_argument("--results-dir", default="sweep_results")
    p.add_argument("--log-dir", default="sweep_logs")
    p.add_argument("--summary", default="sweep_summary.json")
    p.add_argument(
        "--tag",
        default=None,
        help="Isolate a variant: suffixes the summary, results dir and log dir "
        "(e.g. --tag bcast keeps a broadcast+AQE sweep from overwriting the "
        "shuffle-only one). Without it, --rerun clobbers the previous run.",
    )
    p.add_argument(
        "--tpch-impl",
        default="dataframe",
        choices=["dataframe", "sql"],
        help="Which Spark script runs the TPC-H arm. dataframe (default) mirrors "
        "Ray's plan statement for statement; sql runs canonical TPC-H SQL. "
        "Ray's side is identical either way. Pair with --tag to keep the "
        "two sweeps separate.",
    )
    p.add_argument(
        "--executors",
        type=int,
        default=32,
        help="Spark executors (pinned for every run in the sweep)",
    )
    p.add_argument("--cores", type=int, default=16, help="cores per executor")
    p.add_argument("--memory", default="32g", help="heap per executor")
    p.add_argument(
        "--spark-args",
        default="",
        help="extra args appended to every Spark run, e.g. "
        '"--num_executors 32 --executor_cores 16 '
        '--executor_memory 32g"',
    )
    p.add_argument(
        "--rerun", action="store_true", help="re-run benchmarks already recorded as ok"
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Children run with cwd=<ray-dir|spark-dir>, so every path handed to them
    # (TEST_OUTPUT_JSON, --output) must be absolute or it resolves against the
    # wrong directory and the benchmark dies trying to write its result.
    if args.tag:
        args.results_dir = f"{args.results_dir}_{args.tag}"
        args.log_dir = f"{args.log_dir}_{args.tag}"
        base, ext = os.path.splitext(args.summary)
        args.summary = f"{base}_{args.tag}{ext}"
    args.results_dir = os.path.abspath(args.results_dir)
    args.log_dir = os.path.abspath(args.log_dir)
    args.summary = os.path.abspath(args.summary)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    matrix = build_matrix(args.tpch_impl)
    if args.only:
        matrix = [
            b for b in matrix if args.only in b["name"] or args.only == b["group"]
        ]
    if not matrix:
        raise SystemExit(f"no benchmarks matched --only {args.only!r}")

    engines = ["ray", "spark"] if args.engine == "both" else [args.engine]

    data: Dict[str, Any] = {"started": datetime.now().isoformat(), "runs": {}}
    if os.path.exists(args.summary) and not args.rerun:
        try:
            with open(args.summary) as f:
                data = json.load(f)
            data.setdefault("runs", {})
        except Exception:
            pass

    total = len(matrix) * len(engines)
    print(
        f"\n{total} runs ({len(matrix)} benchmarks x {len(engines)} engine(s)), "
        f"cooldown {args.cooldown:g}s, per-run timeout {args.timeout:g}s"
    )
    print("Rough estimate at SF1000: several hours. Interrupt and re-run to resume.\n")

    n = 0
    for bench in matrix:
        entry = data["runs"].setdefault(bench["name"], {})
        for engine in engines:
            n += 1
            prev = entry.get(engine)
            if prev and prev.get("status") == "ok" and not args.rerun:
                print(
                    f"[{n}/{total}] {bench['name']} [{engine}] -- already ok, "
                    f"skipping (--rerun to force)"
                )
                continue

            print(f"[{n}/{total}] {bench['name']} [{engine}]")
            entry[engine] = run_one(bench, engine, args)
            data["updated"] = datetime.now().isoformat()
            save(args.summary, data)

            if not args.dry_run and n < total:
                if engine == "ray":
                    wait_for_object_store(args.drain_wait)
                if args.cooldown:
                    print(f"      cooldown {args.cooldown:g}s", flush=True)
                    time.sleep(args.cooldown)

    report(data)
    print(
        f"\nSummary: {args.summary}   raw: {args.results_dir}/   logs: {args.log_dir}/"
    )


if __name__ == "__main__":
    main()
