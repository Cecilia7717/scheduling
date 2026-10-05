#!/usr/bin/env python3
"""
Corrected three-algorithm CPU scaling benchmark for the EXISTING instances.

Runs, on the same stored jobs and energy intervals:

    1. MaxFlow-Passes
    2. MaxFlow-Restart (correct cold-prefix replay)
    3. Pure-MinCostFlow

Why this version is different
-----------------------------
The previous Restart implementation rebuilt a fresh graph and used the
previous phase's occupied cheap interval amounts only as UPPER capacities.
That does not force the rebuilt max-flow to preserve the same amount of
cheap flow, so it can replace green flow with brown/red flow and obtain a
larger carbon cost.

This corrected Restart implements the intended ablation:

    Prefix 1 (green):
        build fresh -> solve green

    Prefix 2 (green+brown):
        build fresh -> solve green -> lock green -> open brown -> continue

    Prefix 3 (green+brown+red), if needed:
        build fresh -> solve green -> lock green -> open brown -> continue
                    -> lock brown -> open red -> continue

Thus each prefix is recomputed FROM SCRATCH, but within that fresh prefix
solve we use the same cheapest-first residual-flow logic as MaxFlow-Passes.
The final carbon cost must therefore match MaxFlow-Passes and MinCostFlow.

Existing JSON contents are preserved.  This script adds/replaces:

    "cpu_exact_scaling": {...}

Use --overwrite when rerunning after the old Restart implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from compare_maxflow_mincost.maxflow import (
    EPS,
    Job,
    EnergyInterval,
    build_incremental_dinic_network,
    build_timeline,
    max_flow_passes_schedule,
    pure_min_cost_schedule,
    recover_result,
    schedule_cost,
    split_intervals_at_job_boundaries,
)


FOLDER_RE = re.compile(
    r"^jobs_(?P<jobs>\d+)_intervals_(?P<intervals>\d+)_(?P<density>.+)$"
)


# ============================================================
# Input conversion
# ============================================================

def build_jobs(data: Dict[str, Any]) -> List[Job]:
    return [
        Job(
            name=x["name"],
            release=float(x["release"]),
            deadline=float(x["deadline"]),
            processing=float(x["processing"]),
        )
        for x in data["jobs"]
    ]


def build_intervals(data: Dict[str, Any]) -> List[EnergyInterval]:
    return [
        EnergyInterval(
            name=x["name"],
            start=float(x["start"]),
            end=float(x["end"]),
            energy=x["energy"],
        )
        for x in data["energy_intervals"]
    ]


# ============================================================
# Correct restart ablation
# ============================================================

def _solve_prefix_from_scratch(
    jobs: List[Job],
    intervals: List[EnergyInterval],
    total_processing: float,
    prefix_level: int,
) -> Dict[str, Any]:
    """
    Solve one carbon-cost prefix on a FRESH network.

    prefix_level = 1:
        green only

    prefix_level = 2:
        green, then brown using the same residual graph created for THIS
        prefix solve

    prefix_level = 3:
        green, then brown, then red using the same residual graph created
        for THIS prefix solve

    The network is fresh at entry to this function.  This is the state
    discarded by MaxFlow-Restart between prefix problems.
    """
    if prefix_level not in (1, 2, 3):
        raise ValueError(f"Invalid prefix_level={prefix_level}")

    network, metadata = build_incremental_dinic_network(
        jobs,
        intervals,
    )

    source = metadata["source"]
    sink = metadata["sink"]
    out_edges = metadata["interval_out_edges"]

    # --------------------------------------------------------
    # Internal Pass 1: green only
    # --------------------------------------------------------
    added1 = network.max_flow(
        source,
        sink,
        flow_limit=total_processing,
    )
    total_flow1 = added1

    pass1 = recover_result(
        network,
        metadata,
        jobs,
        intervals,
        total_flow1,
        0.0,
    )

    if prefix_level == 1:
        return {
            "network": network,
            "metadata": metadata,
            "pass1": pass1,
            "pass2": None,
            "pass3": None,
            "final": pass1,
            "final_pass": 1,
        }

    # --------------------------------------------------------
    # Internal Pass 2:
    #   preserve current green flow
    #   lock each green interval to its occupied amount
    #   open brown
    #   keep red closed
    # --------------------------------------------------------
    for i, interval in enumerate(intervals):
        edge = out_edges[i]
        inode = metadata["interval_offset"] + i

        if interval.energy == "green":
            occupied = pass1["interval_usage"][i]
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                occupied,
            )
        elif interval.energy == "brown":
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                interval.length,
            )
        else:
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                0.0,
            )

    added2 = network.max_flow(
        source,
        sink,
        flow_limit=max(
            0.0,
            total_processing - total_flow1,
        ),
    )
    total_flow2 = total_flow1 + added2

    pass2 = recover_result(
        network,
        metadata,
        jobs,
        intervals,
        total_flow2,
        0.0,
    )

    if prefix_level == 2:
        return {
            "network": network,
            "metadata": metadata,
            "pass1": pass1,
            "pass2": pass2,
            "pass3": None,
            "final": pass2,
            "final_pass": 2,
        }

    # --------------------------------------------------------
    # Internal Pass 3:
    #   preserve green+brown flow
    #   keep green locked at the internal Pass-1 usage
    #   lock brown to the internal Pass-2 usage
    #   open red
    # --------------------------------------------------------
    for i, interval in enumerate(intervals):
        edge = out_edges[i]
        inode = metadata["interval_offset"] + i

        if interval.energy == "brown":
            occupied = pass2["interval_usage"][i]
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                occupied,
            )
        elif interval.energy == "red":
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                interval.length,
            )
        # Green remains at the capacity established above.

    added3 = network.max_flow(
        source,
        sink,
        flow_limit=max(
            0.0,
            total_processing - total_flow2,
        ),
    )
    total_flow3 = total_flow2 + added3

    pass3 = recover_result(
        network,
        metadata,
        jobs,
        intervals,
        total_flow3,
        0.0,
    )

    return {
        "network": network,
        "metadata": metadata,
        "pass1": pass1,
        "pass2": pass2,
        "pass3": pass3,
        "final": pass3,
        "final_pass": 3,
    }


def max_flow_restart_schedule(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Correct no-cross-prefix-residual-reuse ablation.

    The atomic intervals are constructed once, as in MaxFlow-Passes.
    We then recompute each required cheapest-first prefix from a fresh
    Dinic network.

    This intentionally repeats work:
        green prefix:              G
        green+brown prefix:        G -> B
        green+brown+red prefix:    G -> B -> R

    In contrast, MaxFlow-Passes performs only:
        G -> B -> R

    while retaining the same residual graph across the outer phases.
    """
    intervals = split_intervals_at_job_boundaries(
        jobs,
        original_intervals,
    )
    total_processing = sum(job.processing for job in jobs)

    # --------------------------------------------------------
    # Outer prefix 1: recompute G from scratch.
    # --------------------------------------------------------
    prefix1 = _solve_prefix_from_scratch(
        jobs,
        intervals,
        total_processing,
        prefix_level=1,
    )

    # --------------------------------------------------------
    # Outer prefix 2: discard prefix1 graph, rebuild, replay G,
    # then continue to B.
    # --------------------------------------------------------
    prefix2 = _solve_prefix_from_scratch(
        jobs,
        intervals,
        total_processing,
        prefix_level=2,
    )

    if prefix2["final"]["flow"] + EPS >= total_processing:
        final_prefix = prefix2
        final_pass = 2
        prefix3 = None
    else:
        # ----------------------------------------------------
        # Outer prefix 3: discard prefix2 graph, rebuild, replay
        # G -> B, then continue to R.
        # ----------------------------------------------------
        prefix3 = _solve_prefix_from_scratch(
            jobs,
            intervals,
            total_processing,
            prefix_level=3,
        )
        final_prefix = prefix3
        final_pass = 3

    final = final_prefix["final"]
    feasible = abs(final["flow"] - total_processing) <= EPS

    result = {
        "algorithm": "MaxFlow-Restart-Cold-Prefix-Replay",
        "restart_definition": "cold_prefix_replay_v2",
        "jobs": jobs,
        "intervals": intervals,
        "total_processing": total_processing,

        # These are the FINAL prefix's internal pass states.
        "pass1": final_prefix["pass1"],
        "pass2": final_prefix["pass2"],
        "pass3": final_prefix["pass3"],

        "outer_prefix1_flow": prefix1["final"]["flow"],
        "outer_prefix2_flow": prefix2["final"]["flow"],
        "outer_prefix3_flow": (
            None if prefix3 is None
            else prefix3["final"]["flow"]
        ),

        "final_pass": final_pass,
        "final": final,
        "feasible": feasible,
        "timeline": build_timeline(
            jobs,
            intervals,
            final["assignment"],
        ),
        "normalized_cost": schedule_cost(final),
    }

    return result


# ============================================================
# Optional timeout
# ============================================================

class SolverTimeout(RuntimeError):
    pass


def _alarm_handler(signum, frame):
    raise SolverTimeout("solver timed out")


def run_with_timeout(fn, timeout_seconds: float):
    if timeout_seconds <= 0:
        return fn()

    old_handler = signal.signal(
        signal.SIGALRM,
        _alarm_handler,
    )
    signal.setitimer(
        signal.ITIMER_REAL,
        timeout_seconds,
    )

    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================
# Benchmarking
# ============================================================

def adaptive_repeats(num_jobs: int) -> int:
    if num_jobs <= 200:
        return 10
    if num_jobs <= 1000:
        return 5
    if num_jobs <= 5000:
        return 3
    return 1


def compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    final = result["final"]

    payload = {
        "algorithm": result.get("algorithm"),
        "feasible": bool(result["feasible"]),
        "total_processing": float(result["total_processing"]),
        "flow": float(final["flow"]),
        "normalized_cost": float(result["normalized_cost"]),
        "energy_usage": {
            k: float(v)
            for k, v in final["energy_usage"].items()
        },
    }

    if "final_pass" in result:
        payload["final_pass"] = int(result["final_pass"])

    if "restart_definition" in result:
        payload["restart_definition"] = result[
            "restart_definition"
        ]

    return payload


def benchmark_one(
    solver,
    jobs,
    intervals,
    repeats: int,
    timeout_seconds: float,
):
    times = []
    retained = None

    for _ in range(repeats):
        start = time.perf_counter()

        result = run_with_timeout(
            lambda: solver(
                jobs,
                intervals,
                verbose=False,
            ),
            timeout_seconds,
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if retained is None:
            retained = result

    return {
        **compact_result(retained),
        "timing": {
            "repeats": repeats,
            "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
        },
    }


def folder_metadata(path: Path):
    m = FOLDER_RE.fullmatch(path.parent.name)

    if m is None:
        return None

    return {
        "folder": path.parent.name,
        "num_jobs": int(m.group("jobs")),
        "num_original_intervals": int(
            m.group("intervals")
        ),
        "density": m.group("density"),
    }


def process_file(
    path: Path,
    *,
    fixed_repeats: Optional[int],
    timeout_seconds: float,
    overwrite: bool,
    dry_run: bool,
):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        "cpu_exact_scaling" in data
        and not overwrite
    ):
        return {
            "status": "skipped",
            "reason": (
                "cpu_exact_scaling already exists"
            ),
        }

    jobs = build_jobs(data)
    intervals = build_intervals(data)

    repeats = (
        fixed_repeats
        if fixed_repeats is not None
        else adaptive_repeats(len(jobs))
    )

    algorithms = [
        (
            "max_flow_passes",
            max_flow_passes_schedule,
        ),
        (
            "max_flow_restart",
            max_flow_restart_schedule,
        ),
        (
            "pure_min_cost",
            pure_min_cost_schedule,
        ),
    ]

    payload = {
        "format_version": 2,
        "restart_definition": (
            "cold_prefix_replay_v2"
        ),
        "timing_policy": {
            "repeats": repeats,
            "timeout_seconds_per_execution": (
                timeout_seconds
            ),
        },
    }

    for name, solver in algorithms:
        try:
            payload[name] = benchmark_one(
                solver,
                jobs,
                intervals,
                repeats,
                timeout_seconds,
            )
            payload[name]["status"] = "ok"

        except SolverTimeout:
            payload[name] = {
                "status": "timeout",
                "timing": {
                    "repeats_requested": repeats,
                    "timeout_seconds_per_execution": (
                        timeout_seconds
                    ),
                },
            }

        except MemoryError:
            payload[name] = {
                "status": "memory_error"
            }

        except Exception as exc:
            payload[name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    completed = [
        payload[name]
        for name, _ in algorithms
        if payload[name].get("status") == "ok"
    ]

    if len(completed) == 3:
        if not all(x["feasible"] for x in completed):
            raise RuntimeError(
                f"{path}: at least one exact solver "
                f"reported infeasible"
            )

        costs = [
            x["normalized_cost"]
            for x in completed
        ]
        flows = [
            x["flow"]
            for x in completed
        ]

        if max(costs) - min(costs) > 1e-7:
            raise RuntimeError(
                f"{path}: exact-solver cost mismatch: "
                f"{costs}"
            )

        if max(flows) - min(flows) > 1e-7:
            raise RuntimeError(
                f"{path}: exact-solver flow mismatch: "
                f"{flows}"
            )

        payload["correctness_check"] = {
            "all_completed": True,
            "same_flow": True,
            "same_cost": True,
            "cost": costs[0],
            "flow": flows[0],
        }

    else:
        payload["correctness_check"] = {
            "all_completed": False,
            "same_flow": None,
            "same_cost": None,
        }

    if not dry_run:
        data["cpu_exact_scaling"] = payload

        path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "status": "processed",
        "payload": payload,
        "repeats": repeats,
    }


# ============================================================
# Compact CSV collection
# ============================================================

def collect_rows(
    root: Path,
) -> List[Dict[str, Any]]:
    rows = []

    for path in sorted(
        root.rglob("instance_*.json")
    ):
        meta = folder_metadata(path)
        if meta is None:
            continue

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        exact = data.get("cpu_exact_scaling")
        if exact is None:
            continue

        # Ignore stale v1 successful records left by the
        # previous incorrect Restart implementation.
        if (
            exact.get("restart_definition")
            != "cold_prefix_replay_v2"
        ):
            continue

        row = {
            **meta,
            "file": str(
                path.relative_to(root)
            ),
            "instance_id": data.get(
                "instance_id"
            ),
        }

        for key, prefix in [
            (
                "max_flow_passes",
                "passes",
            ),
            (
                "max_flow_restart",
                "restart",
            ),
            (
                "pure_min_cost",
                "mincost",
            ),
        ]:
            item = exact.get(key, {})

            row[f"{prefix}_status"] = (
                item.get("status")
            )

            timing = item.get("timing", {})

            row[
                f"{prefix}_mean_seconds"
            ] = timing.get("mean_seconds")

            row[
                f"{prefix}_median_seconds"
            ] = timing.get(
                "median_seconds"
            )

            row[
                f"{prefix}_min_seconds"
            ] = timing.get("min_seconds")

            row[
                f"{prefix}_max_seconds"
            ] = timing.get("max_seconds")

            row[f"{prefix}_cost"] = (
                item.get("normalized_cost")
            )

            row[f"{prefix}_flow"] = (
                item.get("flow")
            )

        if (
            row["passes_mean_seconds"]
            is not None
            and row["restart_mean_seconds"]
            is not None
        ):
            row["restart_over_passes"] = (
                row["restart_mean_seconds"]
                / row["passes_mean_seconds"]
            )
        else:
            row["restart_over_passes"] = None

        if (
            row["passes_mean_seconds"]
            is not None
            and row["mincost_mean_seconds"]
            is not None
        ):
            row["mincost_over_passes"] = (
                row["mincost_mean_seconds"]
                / row["passes_mean_seconds"]
            )
        else:
            row["mincost_over_passes"] = None

        rows.append(row)

    return rows


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "job_interval_scaling_results_gpu_ecl_timing_adjusted"
        ),
    )

    p.add_argument(
        "--timing-repeats",
        type=int,
        default=None,
        help=(
            "If omitted: 10 repeats <=200 jobs, "
            "5 <=1000, 3 <=5000, 1 above 5000."
        ),
    )

    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help=(
            "Per solver execution; "
            "0 disables timeout."
        ),
    )

    p.add_argument(
        "--min-jobs",
        type=int,
        default=None,
    )

    p.add_argument(
        "--max-jobs",
        type=int,
        default=None,
    )

    p.add_argument(
        "--density",
        action="append",
        default=None,
        help=(
            "Restrict to a density. Repeat flag "
            "for multiple values."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Required when replacing results from "
            "the previous incorrect Restart version."
        ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    files = []

    for path in sorted(
        args.root.rglob("instance_*.json")
    ):
        meta = folder_metadata(path)
        if meta is None:
            continue

        if (
            args.min_jobs is not None
            and meta["num_jobs"] < args.min_jobs
        ):
            continue

        if (
            args.max_jobs is not None
            and meta["num_jobs"] > args.max_jobs
        ):
            continue

        if (
            args.density is not None
            and meta["density"]
            not in set(args.density)
        ):
            continue

        files.append(path)

    print("=" * 80)
    print(
        "THREE-ALGORITHM CPU SCALING "
        "(CORRECTED RESTART)"
    )
    print("=" * 80)
    print(f"Root:       {args.root}")
    print(f"Files:      {len(files)}")
    print(
        f"Repeats:    "
        f"{args.timing_repeats or 'adaptive'}"
    )
    print(
        f"Timeout:    "
        f"{args.timeout_seconds or 'disabled'}"
    )
    print(
        "Restart:    cold_prefix_replay_v2"
    )
    print()

    processed = 0
    skipped = 0
    failed = 0

    for i, path in enumerate(
        files,
        start=1,
    ):
        rel = path.relative_to(args.root)

        try:
            info = process_file(
                path,
                fixed_repeats=(
                    args.timing_repeats
                ),
                timeout_seconds=(
                    args.timeout_seconds
                ),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            if info["status"] == "skipped":
                skipped += 1
                print(
                    f"[{i:>4}/{len(files)}] "
                    f"SKIP {rel}"
                )
                continue

            processed += 1
            payload = info["payload"]

            def short(name):
                x = payload[name]

                if x.get("status") != "ok":
                    return x.get("status")

                return (
                    f"{x['timing']['mean_seconds']:.6g}s"
                )

            print(
                f"[{i:>4}/{len(files)}] "
                f"OK   {rel} | "
                f"P={short('max_flow_passes')} | "
                f"R={short('max_flow_restart')} | "
                f"MCF={short('pure_min_cost')}"
            )

        except Exception as exc:
            failed += 1

            print(
                f"[{i:>4}/{len(files)}] "
                f"ERROR {rel}: "
                f"{type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    if not args.dry_run:
        rows = collect_rows(args.root)

        csv_path = (
            args.root
            / "cpu_three_algorithms_all_instances.csv"
        )

        write_csv(csv_path, rows)

        print(f"CSV:       {csv_path}")


if __name__ == "__main__":
    main()
