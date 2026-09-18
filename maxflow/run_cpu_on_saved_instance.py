#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


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


if __name__ == "__main__":
    main()