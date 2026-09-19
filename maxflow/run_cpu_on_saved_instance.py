#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


from compare_maxflow_mincost.maxflow import (
    Job,
    EnergyInterval,
    max_flow_passes_schedule,
)


EPS = 1e-9


# ============================================================
# Helpers
# ============================================================

def clean_number(x: float):
    """
    Store integer-valued floats as ints when possible.
    Example:
        3.0 -> 3
        3.5 -> 3.5
    """
    if abs(x - round(x)) <= EPS:
        return int(round(x))
    return float(x)


def build_jobs(data: Dict[str, Any]) -> List[Job]:
    jobs = []

    for item in data["jobs"]:
        jobs.append(
            Job(
                name=item["name"],
                release=float(item["release"]),
                deadline=float(item["deadline"]),
                processing=float(item["processing"]),
            )
        )

    return jobs


def build_energy_intervals(data: Dict[str, Any]) -> List[EnergyInterval]:
    intervals = []

    for item in data["energy_intervals"]:
        intervals.append(
            EnergyInterval(
                name=item["name"],
                start=float(item["start"]),
                end=float(item["end"]),
                energy=item["energy"],
            )
        )

    return intervals


# ============================================================
# Convert CPU result into JSON-safe representation
# ============================================================

def assignment_to_schedule_rows(
    assignment,
    jobs,
    intervals,
):
    rows = []

    for (j, i), amount in assignment.items():
        if amount <= EPS:
            continue

        interval = intervals[i]

        rows.append(
            {
                "Job": jobs[j].name,
                "Interval": interval.name,
                "Start": clean_number(interval.start),
                "End": clean_number(interval.end),
                "Energy": interval.energy,
                "Flow": clean_number(amount),
            }
        )

    return rows


def serialize_pass(
    pass_result,
    jobs,
    intervals,
    previous_flow: float = 0.0,
):
    if pass_result is None:
        return None

    flow = pass_result["flow"]

    return {
        "added_flow": clean_number(flow - previous_flow),
        "flow": clean_number(flow),

        "energy_usage": {
            energy: clean_number(amount)
            for energy, amount in pass_result["energy_usage"].items()
        },

        "interval_usage": {
            str(i): clean_number(amount)
            for i, amount in pass_result["interval_usage"].items()
        },

        "job_usage": {
            str(j): clean_number(amount)
            for j, amount in pass_result["job_usage"].items()
        },

        "schedule_rows": assignment_to_schedule_rows(
            pass_result["assignment"],
            jobs,
            intervals,
        ),
    }


def serialize_cpu_result(
    result: Dict[str, Any],
    wall_seconds: float,
) -> Dict[str, Any]:

    jobs = result["jobs"]
    intervals = result["intervals"]

    pass1 = result["pass1"]
    pass2 = result["pass2"]
    pass3 = result["pass3"]

    p1_flow = pass1["flow"]
    p2_flow = pass2["flow"]

    serialized_pass1 = serialize_pass(
        pass1,
        jobs,
        intervals,
        previous_flow=0.0,
    )

    serialized_pass2 = serialize_pass(
        pass2,
        jobs,
        intervals,
        previous_flow=p1_flow,
    )

    serialized_pass3 = None
    if pass3 is not None:
        serialized_pass3 = serialize_pass(
            pass3,
            jobs,
            intervals,
            previous_flow=p2_flow,
        )

    final = result["final"]

    return {
        "feasible": bool(result["feasible"]),
        "total_processing": clean_number(result["total_processing"]),
        "flow": clean_number(final["flow"]),
        "final_pass": int(result["final_pass"]),
        "normalized_cost": clean_number(result["normalized_cost"]),

        "energy_usage": {
            energy: clean_number(amount)
            for energy, amount in final["energy_usage"].items()
        },

        # Entire CPU algorithm runtime:
        # splitting intervals + network construction +
        # pass 1 + pass 2 + pass 3 + final schedule construction
        "wall_seconds": wall_seconds,

        "pass1": serialized_pass1,
        "pass2": serialized_pass2,
        "pass3": serialized_pass3,

        "final_schedule_rows": assignment_to_schedule_rows(
            final["assignment"],
            jobs,
            intervals,
        ),

        "timeline": [
            {
                "start": clean_number(row["start"]),
                "end": clean_number(row["end"]),
                "job": row["job"],
                "energy": row["energy"],
                "interval": row["interval"],
            }
            for row in result["timeline"]
        ],
    }


# ============================================================
# Read only the input portion
# ============================================================

def get_input_before_gpu(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take only top-level fields appearing before gpu_max_flow_passes.

    For your current JSON this includes things such as:

        instance_id
        horizon
        jobs
        energy_intervals
        split_intervals
        construction_witness_schedule

    and stops before:

        gpu_max_flow_passes

    The CPU solver itself only needs:
        jobs
        energy_intervals
    """

    result = {}

    for key, value in data.items():
        if key == "gpu_max_flow_passes":
            break

        result[key] = value

    return result


# ============================================================
# Process one file
# ============================================================

def process_instance(
    path: Path,
    *,
    overwrite_cpu: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:

    with path.open("r", encoding="utf-8") as f:
        full_data = json.load(f)

    # --------------------------------------------------------
    # Skip if already processed unless explicitly requested.
    # --------------------------------------------------------
    if "cpu_max_flow_passes" in full_data and not overwrite_cpu:
        return {
            "status": "skipped",
            "file": str(path),
            "reason": "cpu_max_flow_passes already exists",
        }

    # --------------------------------------------------------
    # Everything before gpu_max_flow_passes is considered input.
    # --------------------------------------------------------
    input_data = get_input_before_gpu(full_data)

    if "jobs" not in input_data:
        raise ValueError(f"{path}: missing jobs")

    if "energy_intervals" not in input_data:
        raise ValueError(f"{path}: missing energy_intervals")

    jobs = build_jobs(input_data)
    intervals = build_energy_intervals(input_data)

    # --------------------------------------------------------
    # Run CPU max-flow passes.
    # --------------------------------------------------------
    start = time.perf_counter()

    cpu_result = max_flow_passes_schedule(
        jobs,
        intervals,
        verbose=False,
    )

    wall_seconds = time.perf_counter() - start

    cpu_json = serialize_cpu_result(
        cpu_result,
        wall_seconds,
    )

    # --------------------------------------------------------
    # Do NOT remove GPU results.
    #
    # Just add:
    #
    #     "cpu_max_flow_passes": {...}
    #
    # to the original JSON.
    # --------------------------------------------------------
    full_data["cpu_max_flow_passes"] = cpu_json

    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                full_data,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

    return {
        "status": "processed",
        "file": str(path),
        "feasible": cpu_json["feasible"],
        "flow": cpu_json["flow"],
        "cost": cpu_json["normalized_cost"],
        "wall_seconds": wall_seconds,
    }


# ============================================================
# Process entire directory tree
# ============================================================

def process_all(
    root: Path,
    *,
    overwrite_cpu: bool = False,
    dry_run: bool = False,
):
    if not root.exists():
        raise FileNotFoundError(
            f"Root directory does not exist: {root}"
        )

    # Recursively finds files such as:
    #
    # jobs_10_intervals_3_light/instance_0001.json
    # jobs_10_intervals_3_light/instance_0002.json
    # jobs_20_intervals_5_medium/instance_0001.json
    # ...
    #
    instance_files = sorted(
        p
        for p in root.rglob("instance_*.json")
        if p.is_file()
    )

    print("=" * 80)
    print("CPU MAX FLOW BATCH RUN")
    print("=" * 80)
    print(f"Root:  {root}")
    print(f"Files: {len(instance_files)}")
    print()

    processed = 0
    skipped = 0
    failed = 0

    total_cpu_seconds = 0.0

    for index, path in enumerate(instance_files, start=1):

        relative = path.relative_to(root)

        try:
            info = process_instance(
                path,
                overwrite_cpu=overwrite_cpu,
                dry_run=dry_run,
            )

            if info["status"] == "skipped":
                skipped += 1

                print(
                    f"[{index:>5}/{len(instance_files)}] "
                    f"SKIP  {relative}"
                )

                continue

            processed += 1
            total_cpu_seconds += info["wall_seconds"]

            print(
                f"[{index:>5}/{len(instance_files)}] "
                f"OK    {relative} | "
                f"feasible={info['feasible']} | "
                f"flow={info['flow']} | "
                f"cost={info['cost']} | "
                f"time={info['wall_seconds']:.6f}s"
            )

        except Exception as exc:
            failed += 1

            print(
                f"[{index:>5}/{len(instance_files)}] "
                f"ERROR {relative}"
            )
            print(f"        {type(exc).__name__}: {exc}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Found:       {len(instance_files)}")
    print(f"Processed:   {processed}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {failed}")

    if processed:
        print(f"CPU seconds: {total_cpu_seconds:.6f}")
        print(
            f"Mean/file:   "
            f"{total_cpu_seconds / processed:.6f} s"
        )


# ============================================================
# Timing collection / plotting
# ============================================================

FOLDER_RE = re.compile(
    r"^jobs_(?P<jobs>\d+)_intervals_(?P<intervals>\d+)_(?P<density>.+)$"
)


def _experiment_metadata(path: Path, root: Path) -> Dict[str, Any]:
    """Parse jobs / intervals / density from the instance's parent folder."""
    folder = path.parent.name
    match = FOLDER_RE.match(folder)

    if match is None:
        return {
            "folder": folder,
            "num_jobs": None,
            "num_intervals": None,
            "density": "unknown",
        }

    return {
        "folder": folder,
        "num_jobs": int(match.group("jobs")),
        "num_intervals": int(match.group("intervals")),
        "density": match.group("density"),
    }


def collect_timing_data(root: Path) -> List[Dict[str, Any]]:
    """
    Read timing values already stored in every instance JSON.

    Collected values:
        CPU wall       = cpu_max_flow_passes.wall_seconds
        GPU max-flow   = gpu_max_flow_passes.gpu_maxflow_seconds
        GPU wall       = gpu_max_flow_passes.wall_seconds
    """
    rows: List[Dict[str, Any]] = []

    instance_files = sorted(
        p for p in root.rglob("instance_*.json") if p.is_file()
    )

    for path in instance_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cpu = data.get("cpu_max_flow_passes")
        gpu = data.get("gpu_max_flow_passes")

        if cpu is None or gpu is None:
            print(
                f"WARNING: skipping timing collection for {path}: "
                f"cpu={cpu is not None}, gpu={gpu is not None}"
            )
            continue

        if "wall_seconds" not in cpu:
            print(f"WARNING: missing CPU wall_seconds in {path}")
            continue

        if "wall_seconds" not in gpu or "gpu_maxflow_seconds" not in gpu:
            print(f"WARNING: missing GPU timing field(s) in {path}")
            continue

        meta = _experiment_metadata(path, root)

        rows.append(
            {
                "file": str(path.relative_to(root)),
                "instance_id": data.get("instance_id"),
                **meta,
                "cpu_wall_seconds": float(cpu["wall_seconds"]),
                "gpu_maxflow_seconds": float(gpu["gpu_maxflow_seconds"]),
                "gpu_wall_seconds": float(gpu["wall_seconds"]),
            }
        )

    return rows


def write_timing_csv(root: Path, rows: List[Dict[str, Any]]) -> Path:
    path = root / "cpu_gpu_timing_all_instances.csv"

    fieldnames = [
        "file",
        "instance_id",
        "folder",
        "num_jobs",
        "num_intervals",
        "density",
        "cpu_wall_seconds",
        "gpu_maxflow_seconds",
        "gpu_wall_seconds",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def summarize_timings(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate timing by experiment folder using means and medians."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(row["folder"], []).append(row)

    summary: List[Dict[str, Any]] = []

    for folder, group in grouped.items():
        first = group[0]

        cpu = [r["cpu_wall_seconds"] for r in group]
        gpu_alg = [r["gpu_maxflow_seconds"] for r in group]
        gpu_wall = [r["gpu_wall_seconds"] for r in group]

        summary.append(
            {
                "folder": folder,
                "num_jobs": first["num_jobs"],
                "num_intervals": first["num_intervals"],
                "density": first["density"],
                "num_instances": len(group),
                "mean_cpu_wall_seconds": statistics.mean(cpu),
                "median_cpu_wall_seconds": statistics.median(cpu),
                "mean_gpu_maxflow_seconds": statistics.mean(gpu_alg),
                "median_gpu_maxflow_seconds": statistics.median(gpu_alg),
                "mean_gpu_wall_seconds": statistics.mean(gpu_wall),
                "median_gpu_wall_seconds": statistics.median(gpu_wall),
            }
        )

    density_order = {
        "sparse": 0,
        "light": 1,
        "medium": 2,
        "dense": 3,
        "very_dense": 4,
    }

    summary.sort(
        key=lambda r: (
            density_order.get(r["density"], 999),
            r["num_jobs"] if r["num_jobs"] is not None else 10**18,
            r["num_intervals"] if r["num_intervals"] is not None else 10**18,
            r["folder"],
        )
    )

    return summary


def write_timing_summary_csv(
    root: Path,
    summary: List[Dict[str, Any]],
) -> Path:
    path = root / "cpu_gpu_timing_summary.csv"

    fieldnames = [
        "folder",
        "num_jobs",
        "num_intervals",
        "density",
        "num_instances",
        "mean_cpu_wall_seconds",
        "median_cpu_wall_seconds",
        "mean_gpu_maxflow_seconds",
        "median_gpu_maxflow_seconds",
        "mean_gpu_wall_seconds",
        "median_gpu_wall_seconds",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    return path


def plot_timings_by_density(
    root: Path,
    summary: List[Dict[str, Any]],
) -> List[Path]:
    """
    Produce one scaling plot for each density.

    X axis: number of jobs
    Y axis: mean runtime in seconds, logarithmic scale

    Lines:
        CPU wall
        GPU max-flow algorithm time
        GPU whole wall time
    """
    by_density: Dict[str, List[Dict[str, Any]]] = {}

    for row in summary:
        if row["num_jobs"] is None:
            continue
        by_density.setdefault(row["density"], []).append(row)

    outputs: List[Path] = []

    for density, group in sorted(by_density.items()):
        group.sort(key=lambda r: r["num_jobs"])

        x = [r["num_jobs"] for r in group]
        cpu = [r["mean_cpu_wall_seconds"] for r in group]
        gpu_alg = [r["mean_gpu_maxflow_seconds"] for r in group]
        gpu_wall = [r["mean_gpu_wall_seconds"] for r in group]

        plt.figure(figsize=(8.5, 5.5))
        plt.plot(x, cpu, marker="o", label="CPU wall")
        plt.plot(x, gpu_alg, marker="o", label="GPU max-flow algorithm")
        plt.plot(x, gpu_wall, marker="o", label="GPU wall")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of jobs")
        plt.ylabel("Mean runtime (seconds)")
        plt.title(f"CPU vs GPU runtime — {density.replace('_', ' ')}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output = root / f"cpu_gpu_runtime_{density}.png"
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        outputs.append(output)

    return outputs


def collect_write_and_plot(root: Path) -> None:
    rows = collect_timing_data(root)

    if not rows:
        print("No instances containing both CPU and GPU timings were found.")
        return

    all_csv = write_timing_csv(root, rows)
    summary = summarize_timings(rows)
    summary_csv = write_timing_summary_csv(root, summary)
    plots = plot_timings_by_density(root, summary)

    print()
    print("=" * 80)
    print("CPU/GPU TIMING COLLECTION")
    print("=" * 80)
    print(f"Instances collected: {len(rows)}")
    print(f"Raw timing CSV:       {all_csv}")
    print(f"Summary CSV:          {summary_csv}")
    for plot in plots:
        print(f"Plot:                 {plot}")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Recursively run CPU incremental max-flow on "
            "all saved scheduling instances."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "job_interval_scaling_results_gpu_ecl_timing_adjusted"
        ),
        help="Root folder containing jobs_* subdirectories.",
    )

    parser.add_argument(
        "--overwrite-cpu",
        action="store_true",
        help=(
            "Recompute CPU results even when cpu_max_flow_passes "
            "already exists."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the CPU algorithm and print results, "
            "but do not modify JSON files."
        ),
    )

    args = parser.parse_args()

    process_all(
        args.root,
        overwrite_cpu=args.overwrite_cpu,
        dry_run=args.dry_run,
    )

    # After the CPU run, collect CPU/GPU timing data already stored
    # in the JSON files and generate CSV summaries + plots.
    if not args.dry_run:
        collect_write_and_plot(args.root)


if __name__ == "__main__":
    main()