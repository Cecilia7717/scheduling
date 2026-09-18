from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Load GPU Max Flow + Passes implementation
# ============================================================

HERE = Path(__file__).resolve().parent
ALGORITHM_FILE = HERE / "maxflow_gpu_passes.py"


def load_algorithm():
    if not ALGORITHM_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {ALGORITHM_FILE.name} next to this script."
        )

    spec = importlib.util.spec_from_file_location(
        "gpu_maxflow_passes",
        ALGORITHM_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ALGORITHM_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


alg = load_algorithm()

Job = alg.Job
EnergyInterval = alg.EnergyInterval
EPS = alg.EPS


# ============================================================
# Random feasible instance generation
# ============================================================

def random_partition(
    rng: random.Random,
    total: int,
    pieces: int,
    minimum: int = 1,
) -> List[int]:
    if pieces * minimum > total:
        raise ValueError(
            "Not enough total length for requested number of pieces."
        )

    remaining = total - pieces * minimum

    if pieces == 1:
        return [total]

    cuts = sorted(
        rng.randint(0, remaining)
        for _ in range(pieces - 1)
    )
    cuts = [0] + cuts + [remaining]

    extras = [
        cuts[i + 1] - cuts[i]
        for i in range(pieces)
    ]

    return [minimum + x for x in extras]


def generate_energy_intervals(
    rng: random.Random,
    horizon: int,
    min_intervals: int = 5,
    max_intervals: int = 12,
    green_share_range: Tuple[float, float] = (0.65, 0.75),
    brown_share_range: Optional[Tuple[float, float]] = (0.20, 0.25),
    fixed_intervals: Optional[int] = None,
) -> List[EnergyInterval]:
    if horizon < 3:
        raise ValueError("horizon must be at least 3.")

    if fixed_intervals is not None:
        if fixed_intervals < 3:
            raise ValueError("fixed_intervals must be at least 3.")
        if fixed_intervals > horizon:
            raise ValueError(
                f"fixed_intervals={fixed_intervals} "
                f"cannot exceed horizon={horizon}."
            )
        count = fixed_intervals
    else:
        max_possible = min(max_intervals, horizon)
        min_possible = min(min_intervals, max_possible)
        count = rng.randint(
            max(3, min_possible),
            max_possible,
        )

    g_low, g_high = green_share_range

    if not (0.0 < g_low <= g_high < 1.0):
        raise ValueError(
            "green_share_range must satisfy "
            "0 < low <= high < 1."
        )

    min_green = max(
        1,
        int(round(g_low * horizon)),
    )
    max_green = min(
        horizon - 2,
        int(round(g_high * horizon)),
    )

    if min_green > max_green:
        raise ValueError(
            "green_share_range is incompatible with this horizon."
        )

    green_total = rng.randint(
        min_green,
        max_green,
    )
    remaining = horizon - green_total

    if brown_share_range is None:
        brown_total = rng.randint(
            1,
            remaining - 1,
        )
        red_total = remaining - brown_total

    else:
        b_low, b_high = brown_share_range

        if not (0.0 < b_low <= b_high < 1.0):
            raise ValueError(
                "brown_share_range must satisfy "
                "0 < low <= high < 1."
            )

        min_brown = max(
            1,
            int(round(b_low * horizon)),
        )
        max_brown = min(
            remaining - 1,
            int(round(b_high * horizon)),
        )

        if min_brown > max_brown:
            raise ValueError(
                "Selected green/brown ranges leave no positive red time."
            )

        brown_total = rng.randint(
            min_brown,
            max_brown,
        )
        red_total = remaining - brown_total

    color_counts = [1, 1, 1]

    for _ in range(count - 3):
        color_counts[rng.randrange(3)] += 1

    totals = [
        green_total,
        brown_total,
        red_total,
    ]

    changed = True

    while changed:
        changed = False

        for c in range(3):
            while color_counts[c] > totals[c]:
                recipients = [
                    k
                    for k in range(3)
                    if k != c
                    and color_counts[k] < totals[k]
                ]

                if not recipients:
                    raise RuntimeError(
                        "Could not allocate energy interval counts."
                    )

                color_counts[c] -= 1
                color_counts[
                    rng.choice(recipients)
                ] += 1
                changed = True

    green_lengths = random_partition(
        rng,
        green_total,
        color_counts[0],
    )

    brown_lengths = random_partition(
        rng,
        brown_total,
        color_counts[1],
    )

    red_lengths = random_partition(
        rng,
        red_total,
        color_counts[2],
    )

    pieces = (
        [(x, "green") for x in green_lengths]
        + [(x, "brown") for x in brown_lengths]
        + [(x, "red") for x in red_lengths]
    )

    rng.shuffle(pieces)

    intervals: List[EnergyInterval] = []
    t = 0

    for i, (length, energy) in enumerate(pieces):
        intervals.append(
            EnergyInterval(
                name=f"E{i}",
                start=float(t),
                end=float(t + length),
                energy=energy,
            )
        )
        t += length

    return intervals


def generate_feasible_jobs(
    rng: random.Random,
    horizon: int,
    num_jobs: int,
    min_processing: int = 1,
    max_processing: int = 6,
    target_utilization_range: Tuple[float, float] = (0.55, 0.90),
) -> Tuple[List[Job], List[Dict[str, float]]]:
    low_util, high_util = target_utilization_range

    min_total = max(
        num_jobs * min_processing,
        int(round(low_util * horizon)),
    )

    max_total = min(
        num_jobs * max_processing,
        int(round(high_util * horizon)),
        horizon,
    )

    if min_total > max_total:
        min_total = num_jobs * min_processing
        max_total = min(
            num_jobs * max_processing,
            horizon,
        )

    total_processing = rng.randint(
        min_total,
        max_total,
    )

    processing_times = [
        min_processing
    ] * num_jobs

    remaining_processing = (
        total_processing
        - num_jobs * min_processing
    )

    while remaining_processing > 0:
        eligible = [
            j
            for j, p in enumerate(processing_times)
            if p < max_processing
        ]

        if not eligible:
            raise RuntimeError(
                "Requested total processing exceeds "
                "bounded job capacity."
            )

        j = rng.choice(eligible)
        processing_times[j] += 1
        remaining_processing -= 1

    idle_total = horizon - total_processing

    idle_gaps = [0] * (num_jobs + 1)

    for _ in range(idle_total):
        idle_gaps[
            rng.randrange(num_jobs + 1)
        ] += 1

    witness: List[Dict[str, float]] = []
    cursor = idle_gaps[0]

    for j, p in enumerate(processing_times):
        start = cursor
        end = start + p

        witness.append(
            {
                "job": f"J{j + 1}",
                "start": float(start),
                "end": float(end),
                "processing": float(p),
            }
        )

        cursor = end + idle_gaps[j + 1]

    jobs: List[Job] = []

    for j, piece in enumerate(witness):
        start = int(piece["start"])
        end = int(piece["end"])
        p = int(piece["processing"])

        release = rng.randint(0, start)
        deadline = rng.randint(end, horizon)

        jobs.append(
            Job(
                name=f"J{j + 1}",
                release=float(release),
                deadline=float(deadline),
                processing=float(p),
            )
        )

    rng.shuffle(jobs)

    return jobs, witness


def generate_feasible_instance(
    rng: random.Random,
    instance_id: int,
    min_jobs: int = 5,
    max_jobs: int = 15,
    horizon_range: Tuple[int, int] = (20, 60),
    green_share_range: Tuple[float, float] = (0.65, 0.75),
    brown_share_range: Optional[Tuple[float, float]] = (0.20, 0.25),
    target_utilization_range: Tuple[float, float] = (0.55, 0.90),
    fixed_jobs: Optional[int] = None,
    horizon_per_job: int = 6,
    fixed_intervals: Optional[int] = None,
) -> Dict[str, Any]:
    if fixed_jobs is not None:
        num_jobs = fixed_jobs
        horizon = horizon_per_job * num_jobs
    else:
        horizon = rng.randint(*horizon_range)

        max_jobs_for_horizon = min(
            max_jobs,
            horizon,
        )
        min_jobs_for_horizon = min(
            min_jobs,
            max_jobs_for_horizon,
        )

        num_jobs = rng.randint(
            min_jobs_for_horizon,
            max_jobs_for_horizon,
        )

    jobs, witness = generate_feasible_jobs(
        rng,
        horizon=horizon,
        num_jobs=num_jobs,
        target_utilization_range=target_utilization_range,
    )

    energy_intervals = generate_energy_intervals(
        rng,
        horizon=horizon,
        green_share_range=green_share_range,
        brown_share_range=brown_share_range,
        fixed_intervals=fixed_intervals,
    )

    return {
        "instance_id": instance_id,
        "horizon": horizon,
        "jobs": jobs,
        "energy_intervals": energy_intervals,
        "witness_schedule": witness,
    }


# ============================================================
# Serialization
# ============================================================

def serializable_job(job: Job) -> Dict[str, Any]:
    return {
        "name": job.name,
        "release": job.release,
        "deadline": job.deadline,
        "processing": job.processing,
    }


def serializable_interval(
    interval: EnergyInterval,
) -> Dict[str, Any]:
    return {
        "name": interval.name,
        "start": interval.start,
        "end": interval.end,
        "energy": interval.energy,
        "length": interval.length,
    }


def pass_summary(p: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if p is None:
        return None

    return {
        "added_flow": p["added_flow"],
        "flow": p["flow"],
        "gpu_seconds": p["gpu_seconds"],
        "energy_usage": p["energy_usage"],
        "interval_usage": {
            str(k): v
            for k, v in p["interval_usage"].items()
        },
        "job_usage": {
            str(k): v
            for k, v in p["job_usage"].items()
        },
        "schedule_rows": p["schedule_rows"],
    }


def save_instance_json(
    output_dir: Path,
    instance: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    payload = {
        "instance_id": instance["instance_id"],
        "horizon": instance["horizon"],
        "jobs": [
            serializable_job(j)
            for j in instance["jobs"]
        ],
        "energy_intervals": [
            serializable_interval(i)
            for i in instance["energy_intervals"]
        ],
        "split_intervals": [
            serializable_interval(i)
            for i in result["intervals"]
        ],
        "construction_witness_schedule": (
            instance["witness_schedule"]
        ),
        "gpu_max_flow_passes": {
            "feasible": result["feasible"],
            "total_processing": result["total_processing"],
            "flow": result["final"]["flow"],
            "final_pass": result["final_pass"],
            "normalized_cost": result["normalized_cost"],
            "energy_usage": result["final"]["energy_usage"],
            "gpu_maxflow_seconds": result["gpu_maxflow_seconds"],
            "wall_seconds": result["wall_seconds"],
            "pass1": pass_summary(result["pass1"]),
            "pass2": pass_summary(result["pass2"]),
            "pass3": pass_summary(result["pass3"]),
            "final_schedule_rows": result["schedule_rows"],
            "timeline": result["timeline"],
        },
    }

    path = output_dir / (
        f"instance_{instance['instance_id']:04d}.json"
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_schedule_csv(
    output_dir: Path,
    instance_id: int,
    result: Dict[str, Any],
) -> None:
    path = output_dir / (
        f"instance_{instance_id:04d}_schedule.csv"
    )

    fieldnames = [
        "Job",
        "Interval",
        "Start",
        "End",
        "Energy",
        "Flow",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            result["schedule_rows"]
        )


def write_results_csv(
    output_dir: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path = output_dir / "results.csv"

    fieldnames = [
        "instance_id",
        "horizon",
        "num_jobs",
        "num_original_energy_intervals",
        "num_split_intervals",
        "total_processing",
        "feasible",
        "final_flow",
        "final_pass",
        "normalized_cost",
        "green_units",
        "brown_units",
        "red_units",
        "pass1_added_flow",
        "pass2_added_flow",
        "pass3_added_flow",
        "gpu_maxflow_ms",
        "wall_ms",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    num_instances: int,
    seed: int,
    green_share_range: Tuple[float, float],
    brown_share_range: Optional[Tuple[float, float]],
    target_utilization_range: Tuple[float, float],
    ecl_dir: Path,
) -> None:
    all_feasible = all(
        row["feasible"]
        for row in rows
    )

    mean_gpu_ms = statistics.mean(
        row["gpu_maxflow_ms"]
        for row in rows
    )

    median_gpu_ms = statistics.median(
        row["gpu_maxflow_ms"]
        for row in rows
    )

    mean_wall_ms = statistics.mean(
        row["wall_ms"]
        for row in rows
    )

    median_wall_ms = statistics.median(
        row["wall_ms"]
        for row in rows
    )

    mean_green = statistics.mean(
        row["green_units"]
        for row in rows
    )

    mean_brown = statistics.mean(
        row["brown_units"]
        for row in rows
    )

    mean_red = statistics.mean(
        row["red_units"]
        for row in rows
    )

    mean_cost = statistics.mean(
        row["normalized_cost"]
        for row in rows
    )

    if brown_share_range is None:
        brown_text = "uncontrolled"
    else:
        brown_text = (
            f"{brown_share_range[0]:.4f} to "
            f"{brown_share_range[1]:.4f}"
        )

    summary = f"""GPU Max Flow + Passes Benchmark
===============================

Solver
------
ECL-MaxFlow directory: {ecl_dir}

Pass semantics
--------------
Pass 1: green only
Pass 2: preserve pass-1 flow, lock green usage, open brown
Pass 3: preserve pass-2 flow, lock brown usage, open red

The pass-2/pass-3 GPU calls run on the exact residual graph of the
flow carried from the preceding pass. Reverse residual arcs are included.

Experiment
----------
Random seed: {seed}
Instances: {num_instances}
Green share: {green_share_range[0]:.4f} to {green_share_range[1]:.4f}
Brown share: {brown_text}
Target utilization: {target_utilization_range[0]:.4f} to {target_utilization_range[1]:.4f}

Feasibility
-----------
All generated instances are feasible by construction.
GPU Max Flow + Passes feasible on every instance: {all_feasible}

Runtime
-------
Mean sum of ECL-reported GPU max-flow runtimes: {mean_gpu_ms:.6f} ms
Median sum of ECL-reported GPU max-flow runtimes: {median_gpu_ms:.6f} ms

Mean whole-workflow wall time: {mean_wall_ms:.6f} ms
Median whole-workflow wall time: {median_wall_ms:.6f} ms

Whole-workflow wall time includes Python graph construction, DIMACS writing,
DIMACS->EGR conversion, subprocess launch, CSV reading, and all GPU passes.

Final Schedule
--------------
Average normalized cost: {mean_cost:.6f}
Average green units: {mean_green:.6f}
Average brown units: {mean_brown:.6f}
Average red units: {mean_red:.6f}

Files
-----
results.csv
    One summary row per instance.

instance_XXXX.json
    Full input instance, all three pass results, and final schedule.

instance_XXXX_schedule.csv
    Final Job / Interval / Start / End / Energy / Flow allocation.
"""

    (output_dir / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )


# ============================================================
# Experiment
# ============================================================

def run_random_experiment(
    *,
    ecl_dir: str,
    num_instances: int = 30,
    seed: int = 42,
    output_folder: str = "gpu_passes_results",
    min_jobs: int = 5,
    max_jobs: int = 15,
    min_horizon: int = 20,
    max_horizon: int = 60,
    green_share_range: Tuple[float, float] = (0.65, 0.75),
    brown_share_range: Optional[Tuple[float, float]] = (0.20, 0.25),
    target_utilization_range: Tuple[float, float] = (0.55, 0.90),
    fixed_jobs: Optional[int] = None,
    horizon_per_job: int = 6,
    fixed_intervals: Optional[int] = None,
    verbose_ecl: bool = False,
    keep_work_files: bool = False,
) -> Path:
    rng = random.Random(seed)

    output_dir = HERE / output_folder
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in output_dir.glob("instance_*.json"):
        old_file.unlink()

    for old_file in output_dir.glob(
        "instance_*_schedule.csv"
    ):
        old_file.unlink()

    for old_name in (
        "results.csv",
        "summary.txt",
    ):
        old_file = output_dir / old_name
        if old_file.exists():
            old_file.unlink()

    ecl_dir_path = Path(ecl_dir).resolve()

    rows: List[Dict[str, Any]] = []

    print("=" * 72)
    print("GPU ECL MAX FLOW + PASSES")
    print("=" * 72)
    print(f"ECL directory: {ecl_dir_path}")
    print(f"Instances: {num_instances}")
    print(f"Seed: {seed}")

    if fixed_jobs is None:
        print(
            f"Jobs per instance: "
            f"{min_jobs} to {max_jobs}"
        )
        print(
            f"Horizon: "
            f"{min_horizon} to {max_horizon}"
        )
    else:
        print(
            f"Jobs per instance: "
            f"{fixed_jobs} (fixed)"
        )
        print(
            f"Horizon: {horizon_per_job} * "
            f"{fixed_jobs} = "
            f"{horizon_per_job * fixed_jobs}"
        )

    if fixed_intervals is None:
        print(
            "Original energy intervals: "
            "random default range"
        )
    else:
        print(
            f"Original energy intervals: "
            f"{fixed_intervals} (fixed)"
        )

    print(
        f"Green share: "
        f"{green_share_range[0]:.2f} to "
        f"{green_share_range[1]:.2f}"
    )

    if brown_share_range is None:
        print("Brown share: uncontrolled")
    else:
        print(
            f"Brown share: "
            f"{brown_share_range[0]:.2f} to "
            f"{brown_share_range[1]:.2f}"
        )

    print(
        f"Target utilization: "
        f"{target_utilization_range[0]:.2f} to "
        f"{target_utilization_range[1]:.2f}"
    )

    print(f"Output folder: {output_dir}")
    print()

    for instance_id in range(
        1,
        num_instances + 1,
    ):
        instance = generate_feasible_instance(
            rng,
            instance_id=instance_id,
            min_jobs=min_jobs,
            max_jobs=max_jobs,
            horizon_range=(
                min_horizon,
                max_horizon,
            ),
            green_share_range=green_share_range,
            brown_share_range=brown_share_range,
            target_utilization_range=target_utilization_range,
            fixed_jobs=fixed_jobs,
            horizon_per_job=horizon_per_job,
            fixed_intervals=fixed_intervals,
        )

        result = alg.max_flow_passes_schedule_gpu(
            instance["jobs"],
            instance["energy_intervals"],
            ecl_dir=ecl_dir_path,
            verbose=False,
            verbose_ecl=verbose_ecl,
            keep_work_files=keep_work_files,
            work_folder=(
                output_dir
                / f"instance_{instance_id:04d}_ecl"
                if keep_work_files
                else None
            ),
        )

        if not result["feasible"]:
            raise RuntimeError(
                "GPU Max Flow + Passes failed on "
                f"feasible instance {instance_id}."
            )

        usage = result["final"]["energy_usage"]

        p3_added = (
            result["pass3"]["added_flow"]
            if result["pass3"] is not None
            else 0
        )

        row = {
            "instance_id": instance_id,
            "horizon": instance["horizon"],
            "num_jobs": len(instance["jobs"]),
            "num_original_energy_intervals": len(
                instance["energy_intervals"]
            ),
            "num_split_intervals": len(
                result["intervals"]
            ),
            "total_processing": result["total_processing"],
            "feasible": result["feasible"],
            "final_flow": result["final"]["flow"],
            "final_pass": result["final_pass"],
            "normalized_cost": result["normalized_cost"],
            "green_units": usage["green"],
            "brown_units": usage["brown"],
            "red_units": usage["red"],
            "pass1_added_flow": result["pass1"]["added_flow"],
            "pass2_added_flow": result["pass2"]["added_flow"],
            "pass3_added_flow": p3_added,
            "gpu_maxflow_ms": (
                1000.0
                * result["gpu_maxflow_seconds"]
            ),
            "wall_ms": (
                1000.0
                * result["wall_seconds"]
            ),
        }

        rows.append(row)

        save_instance_json(
            output_dir,
            instance,
            result,
        )

        save_schedule_csv(
            output_dir,
            instance_id,
            result,
        )

        print(
            f"[{instance_id:04d}/{num_instances:04d}] "
            f"jobs={len(instance['jobs']):4d} "
            f"horizon={instance['horizon']:6d} | "
            f"flow={result['final']['flow']:6d}/"
            f"{result['total_processing']:6d} | "
            f"G/B/R="
            f"{usage['green']}/"
            f"{usage['brown']}/"
            f"{usage['red']} | "
            f"GPU={row['gpu_maxflow_ms']:.4f} ms | "
            f"wall={row['wall_ms']:.4f} ms"
        )

    write_results_csv(
        output_dir,
        rows,
    )

    write_summary(
        output_dir,
        rows,
        num_instances=num_instances,
        seed=seed,
        green_share_range=green_share_range,
        brown_share_range=brown_share_range,
        target_utilization_range=target_utilization_range,
        ecl_dir=ecl_dir_path,
    )

    print()
    print("=" * 72)
    print("FINISHED")
    print("=" * 72)
    print(
        f"Results: {output_dir / 'results.csv'}"
    )
    print(
        f"Summary: {output_dir / 'summary.txt'}"
    )

    return output_dir


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate feasible scheduling instances and run "
            "GPU ECL Max Flow + Passes. "
            "No min-cost-flow comparison."
        )
    )

    parser.add_argument(
        "--ecl-dir",
        type=str,
        required=True,
        help=(
            "Path to ECL-MaxFlow, e.g. "
            "~/scheduling/maxflow_gpu/ECL-MaxFlow"
        ),
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="gpu_passes_results",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Fixed number of jobs in every instance."
        ),
    )

    parser.add_argument(
        "--horizon-per-job",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--intervals",
        type=int,
        default=None,
        help=(
            "Fixed number of original energy intervals."
        ),
    )

    parser.add_argument(
        "--min-jobs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-horizon",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-horizon",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--green-min",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--green-max",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--brown-min",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--brown-max",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--random-brown",
        action="store_true",
    )

    parser.add_argument(
        "--util-min",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--util-max",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--verbose-ecl",
        action="store_true",
        help="Print ECL output for every pass.",
    )

    parser.add_argument(
        "--keep-work-files",
        action="store_true",
        help=(
            "Keep each pass DIMACS, EGR, and raw ECL flow CSV."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.instances <= 0:
        raise ValueError("--instances must be > 0.")

    if args.jobs is not None and args.jobs <= 0:
        raise ValueError("--jobs must be > 0.")

    if args.horizon_per_job <= 0:
        raise ValueError(
            "--horizon-per-job must be > 0."
        )

    if args.intervals is not None and args.intervals < 3:
        raise ValueError(
            "--intervals must be >= 3."
        )

    if args.min_jobs <= 0 or args.max_jobs < args.min_jobs:
        raise ValueError(
            "Require 0 < min-jobs <= max-jobs."
        )

    if (
        args.min_horizon <= 0
        or args.max_horizon < args.min_horizon
    ):
        raise ValueError(
            "Require 0 < min-horizon <= max-horizon."
        )

    if not (
        0.0
        < args.green_min
        <= args.green_max
        < 1.0
    ):
        raise ValueError(
            "Require 0 < green-min <= green-max < 1."
        )

    if not args.random_brown:
        if not (
            0.0
            < args.brown_min
            <= args.brown_max
            < 1.0
        ):
            raise ValueError(
                "Require "
                "0 < brown-min <= brown-max < 1."
            )

    if not (
        0.0
        < args.util_min
        <= args.util_max
        <= 1.0
    ):
        raise ValueError(
            "Require 0 < util-min <= util-max <= 1."
        )

    brown_share_range = (
        None
        if args.random_brown
        else (
            args.brown_min,
            args.brown_max,
        )
    )

    run_random_experiment(
        ecl_dir=args.ecl_dir,
        num_instances=args.instances,
        seed=args.seed,
        output_folder=args.output,
        min_jobs=args.min_jobs,
        max_jobs=args.max_jobs,
        min_horizon=args.min_horizon,
        max_horizon=args.max_horizon,
        green_share_range=(
            args.green_min,
            args.green_max,
        ),
        brown_share_range=brown_share_range,
        target_utilization_range=(
            args.util_min,
            args.util_max,
        ),
        fixed_jobs=args.jobs,
        horizon_per_job=args.horizon_per_job,
        fixed_intervals=args.intervals,
        verbose_ecl=args.verbose_ecl,
        keep_work_files=args.keep_work_files,
    )
