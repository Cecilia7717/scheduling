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
# Load the comparison implementation from the sibling file.
# This avoids duplicating either scheduling algorithm.
# ============================================================

HERE = Path(__file__).resolve().parent
ALGORITHM_FILE = HERE / "maxflow.py"


def load_algorithms():
    if not ALGORITHM_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {ALGORITHM_FILE.name} next to this benchmark script."
        )

    spec = importlib.util.spec_from_file_location(
        "flow_algorithm_comparison",
        ALGORITHM_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ALGORITHM_FILE}")

    module = importlib.util.module_from_spec(spec)

    # Register before execution so decorators such as @dataclass can
    # resolve module-level annotations correctly on Python 3.13.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


alg = load_algorithms()

Job = alg.Job
EnergyInterval = alg.EnergyInterval
EPS = alg.EPS


# ============================================================
# Random feasible instance generation
#
# Feasibility is guaranteed BY CONSTRUCTION:
#
# 1. Create a single-machine witness schedule consisting of
#    non-overlapping job pieces.
# 2. For every job, choose a release time no later than its
#    witness start and a deadline no earlier than its witness end.
# 3. Therefore the witness schedule is always feasible.
#
# The algorithms do NOT receive the witness schedule. They only
# receive r_j, d_j, processing time, and the energy intervals.
# ============================================================

ENERGY_TYPES = ("green", "brown", "red")


def random_partition(
    rng: random.Random,
    total: int,
    pieces: int,
    minimum: int = 1,
) -> List[int]:
    """
    Return positive integer pieces summing to total.
    """
    if pieces * minimum > total:
        raise ValueError("Not enough total length for requested number of pieces.")

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
) -> List[EnergyInterval]:
    """
    Generate random green/brown/red intervals.

    green_share_range:
        Fraction of the entire horizon that is green.

    brown_share_range:
        Optional fraction of the entire horizon that is brown.
        If None, the non-green portion is split randomly between
        brown and red.

    Red receives the remaining horizon.
    All three energy types have positive total length.
    """
    if horizon < 3:
        raise ValueError("horizon must be at least 3.")

    max_possible = min(max_intervals, horizon)
    min_possible = min(min_intervals, max_possible)
    count = rng.randint(max(3, min_possible), max_possible)

    g_low, g_high = green_share_range
    if not (0.0 < g_low <= g_high < 1.0):
        raise ValueError(
            "green_share_range must satisfy 0 < low <= high < 1."
        )

    min_green = max(1, int(round(g_low * horizon)))
    max_green = min(horizon - 2, int(round(g_high * horizon)))

    if min_green > max_green:
        raise ValueError(
            "green_share_range is incompatible with this horizon."
        )

    green_total = rng.randint(min_green, max_green)
    remaining = horizon - green_total

    if brown_share_range is None:
        # No brown control requested: randomly split non-green time.
        brown_total = rng.randint(1, remaining - 1)
        red_total = remaining - brown_total
    else:
        b_low, b_high = brown_share_range

        if not (0.0 < b_low <= b_high < 1.0):
            raise ValueError(
                "brown_share_range must satisfy 0 < low <= high < 1."
            )

        min_brown = max(1, int(round(b_low * horizon)))
        max_brown = min(
            remaining - 1,
            int(round(b_high * horizon)),
        )

        if min_brown > max_brown:
            raise ValueError(
                "The selected green and brown ranges can leave no "
                "positive red time. Reduce the ranges."
            )

        brown_total = rng.randint(min_brown, max_brown)
        red_total = remaining - brown_total

    # Decide how many distinct intervals each energy type gets.
    color_counts = [1, 1, 1]  # green, brown, red
    for _ in range(count - 3):
        color_counts[rng.randrange(3)] += 1

    totals = [green_total, brown_total, red_total]

    # No color can have more positive-length pieces than total length.
    changed = True
    while changed:
        changed = False
        for c in range(3):
            while color_counts[c] > totals[c]:
                recipients = [
                    k for k in range(3)
                    if k != c and color_counts[k] < totals[k]
                ]
                if not recipients:
                    raise RuntimeError(
                        "Could not allocate energy interval counts."
                    )
                color_counts[c] -= 1
                color_counts[rng.choice(recipients)] += 1
                changed = True

    green_lengths = random_partition(
        rng, green_total, color_counts[0], minimum=1
    )
    brown_lengths = random_partition(
        rng, brown_total, color_counts[1], minimum=1
    )
    red_lengths = random_partition(
        rng, red_total, color_counts[2], minimum=1
    )

    pieces = (
        [(length, "green") for length in green_lengths]
        + [(length, "brown") for length in brown_lengths]
        + [(length, "red") for length in red_lengths]
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
    """
    Build jobs around a known feasible non-overlapping witness schedule.
    """

    low_util, high_util = target_utilization_range

    min_total = max(num_jobs * min_processing, int(round(low_util * horizon)))
    max_total = min(
        num_jobs * max_processing,
        int(round(high_util * horizon)),
        horizon,
    )

    if min_total > max_total:
        # Fall back to the largest guaranteed feasible total.
        min_total = num_jobs * min_processing
        max_total = min(num_jobs * max_processing, horizon)

    total_processing = rng.randint(min_total, max_total)

    # Construct bounded processing times directly.
    # Start every job at min_processing, then distribute the remaining
    # processing units only to jobs that are still below max_processing.
    processing_times = [min_processing] * num_jobs
    remaining_processing = total_processing - num_jobs * min_processing

    while remaining_processing > 0:
        eligible = [
            j
            for j, p in enumerate(processing_times)
            if p < max_processing
        ]

        if not eligible:
            raise RuntimeError(
                "Requested total processing exceeds bounded job capacity."
            )

        j = rng.choice(eligible)
        processing_times[j] += 1
        remaining_processing -= 1

    idle_total = horizon - total_processing

    # Divide idle time into n+1 gaps:
    # before J1, between jobs, and after Jn.
    idle_gaps = [0] * (num_jobs + 1)
    for _ in range(idle_total):
        idle_gaps[rng.randrange(num_jobs + 1)] += 1

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

    # Randomize release/deadline slack around the witness piece.
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

    # Shuffle job order so the generated order itself carries no schedule.
    rng.shuffle(jobs)

    return jobs, witness


def generate_feasible_instance(
    rng: random.Random,
    instance_id: int,
    min_jobs: int = 25,
    max_jobs: int = 50,
    horizon_range: Tuple[int, int] = (20, 60),
    green_share_range: Tuple[float, float] = (0.65, 0.75),
    brown_share_range: Optional[Tuple[float, float]] = (0.20, 0.25),
    target_utilization_range: Tuple[float, float] = (0.55, 0.90),
) -> Dict[str, Any]:
    horizon = rng.randint(*horizon_range)

    max_jobs_for_horizon = min(max_jobs, horizon)
    min_jobs_for_horizon = min(min_jobs, max_jobs_for_horizon)
    num_jobs = rng.randint(min_jobs_for_horizon, max_jobs_for_horizon)

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
    )

    return {
        "instance_id": instance_id,
        "horizon": horizon,
        "jobs": jobs,
        "energy_intervals": energy_intervals,
        "witness_schedule": witness,
    }


# ============================================================
# Timing
# ============================================================

def time_solver(
    solver,
    jobs: List[Job],
    intervals: List[EnergyInterval],
    repeats: int,
):
    """
    Run once for the retained result, then time repeated fresh executions.
    """
    result = solver(jobs, intervals, verbose=False)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        solver(jobs, intervals, verbose=False)
        times.append(time.perf_counter() - start)

    return result, {
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
    }


# ============================================================
# Serialization helpers
# ============================================================

def serializable_job(job: Job) -> Dict[str, Any]:
    return {
        "name": job.name,
        "release": job.release,
        "deadline": job.deadline,
        "processing": job.processing,
    }


def serializable_interval(interval: EnergyInterval) -> Dict[str, Any]:
    return {
        "name": interval.name,
        "start": interval.start,
        "end": interval.end,
        "energy": interval.energy,
        "length": interval.length,
    }


def serializable_timeline(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "start": row["start"],
            "end": row["end"],
            "job": row["job"],
            "energy": row["energy"],
            "interval": row["interval"],
        }
        for row in timeline
    ]


def result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    final = result["final"]
    usage = final["energy_usage"]

    return {
        "feasible": result["feasible"],
        "total_processing": result["total_processing"],
        "flow": final["flow"],
        "normalized_cost": result["normalized_cost"],
        "green_units": usage["green"],
        "brown_units": usage["brown"],
        "red_units": usage["red"],
        "timeline": serializable_timeline(result["timeline"]),
    }


# ============================================================
# Output
# ============================================================

def save_instance_json(
    output_dir: Path,
    instance: Dict[str, Any],
    max_result: Dict[str, Any],
    pure_result: Dict[str, Any],
    max_timing: Dict[str, float],
    pure_timing: Dict[str, float],
) -> None:
    payload = {
        "instance_id": instance["instance_id"],
        "horizon": instance["horizon"],
        "jobs": [serializable_job(j) for j in instance["jobs"]],
        "energy_intervals": [
            serializable_interval(i)
            for i in instance["energy_intervals"]
        ],
        "construction_witness_schedule": instance["witness_schedule"],
        "max_flow_passes": {
            **result_summary(max_result),
            "timing": max_timing,
        },
        "pure_min_cost": {
            **result_summary(pure_result),
            "timing": pure_timing,
        },
    }

    path = output_dir / f"instance_{instance['instance_id']:02d}.json"
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
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
        "num_energy_intervals",
        "total_processing",
        "max_flow_feasible",
        "pure_min_cost_feasible",
        "max_flow_cost",
        "pure_min_cost_cost",
        "cost_difference_max_minus_pure",
        "max_flow_green",
        "max_flow_brown",
        "max_flow_red",
        "pure_green",
        "pure_brown",
        "pure_red",
        "max_flow_mean_ms",
        "max_flow_median_ms",
        "pure_mean_ms",
        "pure_median_ms",
        "median_speed_ratio_max_over_pure",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    num_instances: int,
    timing_repeats: int,
    seed: int,
    min_jobs: int,
    max_jobs: int,
    min_horizon: int,
    max_horizon: int,
    green_share_range: Tuple[float, float],
    brown_share_range: Optional[Tuple[float, float]],
    target_utilization_range: Tuple[float, float],
) -> None:
    max_mean_ms = statistics.mean(
        row["max_flow_mean_ms"]
        for row in rows
    )
    pure_mean_ms = statistics.mean(
        row["pure_mean_ms"]
        for row in rows
    )

    max_median_ms = statistics.mean(
        row["max_flow_median_ms"]
        for row in rows
    )
    pure_median_ms = statistics.mean(
        row["pure_median_ms"]
        for row in rows
    )

    max_cost = statistics.mean(
        row["max_flow_cost"]
        for row in rows
    )
    pure_cost = statistics.mean(
        row["pure_min_cost_cost"]
        for row in rows
    )

    cost_diff = statistics.mean(
        row["cost_difference_max_minus_pure"]
        for row in rows
    )

    max_green = statistics.mean(
        row["max_flow_green"]
        for row in rows
    )
    pure_green = statistics.mean(
        row["pure_green"]
        for row in rows
    )

    max_brown = statistics.mean(
        row["max_flow_brown"]
        for row in rows
    )
    pure_brown = statistics.mean(
        row["pure_brown"]
        for row in rows
    )

    max_red = statistics.mean(
        row["max_flow_red"]
        for row in rows
    )
    pure_red = statistics.mean(
        row["pure_red"]
        for row in rows
    )

    all_max_feasible = all(row["max_flow_feasible"] for row in rows)
    all_pure_feasible = all(row["pure_min_cost_feasible"] for row in rows)

    lower_cost_count = sum(
        1
        for row in rows
        if row["pure_min_cost_cost"] + EPS < row["max_flow_cost"]
    )
    equal_cost_count = sum(
        1
        for row in rows
        if abs(row["pure_min_cost_cost"] - row["max_flow_cost"]) <= EPS
    )
    higher_cost_count = num_instances - lower_cost_count - equal_cost_count

    if brown_share_range is None:
        brown_text = "uncontrolled (random split of non-green time)"
    else:
        brown_text = (
            f"{brown_share_range[0]:.4f} to "
            f"{brown_share_range[1]:.4f}"
        )

    summary = f"""Random Scheduling Benchmark
===========================

Experiment Settings
-------------------
Random seed: {seed}
Instances: {num_instances}
Timing repetitions per algorithm per instance: {timing_repeats}

Jobs per instance: {min_jobs} to {max_jobs}
Horizon: {min_horizon} to {max_horizon}
Green share range: {green_share_range[0]:.4f} to {green_share_range[1]:.4f}
Brown share range: {brown_text}
Target utilization range: {target_utilization_range[0]:.4f} to {target_utilization_range[1]:.4f}

Feasibility
-----------
All generated instances are feasible by construction.
Max Flow + Passes feasible on every instance: {all_max_feasible}
Pure Min-Cost Flow feasible on every instance: {all_pure_feasible}

Average Runtime
---------------
Max Flow + Passes mean runtime: {max_mean_ms:.6f} ms
Pure Min-Cost Flow mean runtime: {pure_mean_ms:.6f} ms

Average of per-instance median runtimes:
Max Flow + Passes: {max_median_ms:.6f} ms
Pure Min-Cost Flow: {pure_median_ms:.6f} ms

Mean runtime ratio (Max Flow + Passes / Pure Min-Cost Flow):
{max_mean_ms / pure_mean_ms:.6f}x

Normalized Final-Schedule Cost
------------------------------
Common cost function:
    green = 0 per unit
    brown = 1 per unit
    red   = 2 per unit

Average Max Flow + Passes cost: {max_cost:.6f}
Average Pure Min-Cost Flow cost: {pure_cost:.6f}
Average cost difference (Max Flow + Passes - Pure): {cost_diff:.6f}

Pure Min-Cost Flow lower cost: {lower_cost_count}/{num_instances}
Equal cost: {equal_cost_count}/{num_instances}
Pure Min-Cost Flow higher cost: {higher_cost_count}/{num_instances}

Average Energy Usage
--------------------
                    Max Flow + Passes    Pure Min-Cost Flow
Green units         {max_green:.6f}              {pure_green:.6f}
Brown units         {max_brown:.6f}              {pure_brown:.6f}
Red units           {max_red:.6f}              {pure_red:.6f}

Files
-----
results.csv
    One row per random instance.

instance_XX.json
    Full generated instance, construction witness schedule,
    both final schedules, costs, energy usage, and timings.
"""

    (output_dir / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )


# ============================================================
# Main experiment
# ============================================================

def run_random_experiment(
    num_instances: int = 30,
    timing_repeats: int = 20,
    seed: int = 42,
    output_folder: str = "random_benchmark_results",
    min_jobs: int = 5,
    max_jobs: int = 15,
    min_horizon: int = 20,
    max_horizon: int = 60,
    green_share_range: Tuple[float, float] = (0.65, 0.75),
    brown_share_range: Optional[Tuple[float, float]] = (0.20, 0.25),
    target_utilization_range: Tuple[float, float] = (0.55, 0.90),
) -> Path:
    rng = random.Random(seed)

    output_dir = HERE / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove files from a previous run using the same output folder so the
    # directory always represents exactly this experiment.
    for old_file in output_dir.glob("instance_*.json"):
        old_file.unlink()

    for old_name in ("results.csv", "summary.txt"):
        old_file = output_dir / old_name
        if old_file.exists():
            old_file.unlink()

    rows: List[Dict[str, Any]] = []

    print("=" * 72)
    print("RANDOM FEASIBLE INSTANCE BENCHMARK")
    print("=" * 72)
    print(f"Instances: {num_instances}")
    print(f"Timing repetitions per solver/instance: {timing_repeats}")
    print(f"Seed: {seed}")
    print(f"Jobs per instance: {min_jobs} to {max_jobs}")
    print(f"Horizon: {min_horizon} to {max_horizon}")
    print(
        "Green share: "
        f"{green_share_range[0]:.2f} to {green_share_range[1]:.2f}"
    )

    if brown_share_range is None:
        print("Brown share: uncontrolled")
    else:
        print(
            "Brown share: "
            f"{brown_share_range[0]:.2f} to {brown_share_range[1]:.2f}"
        )

    print(
        "Target utilization: "
        f"{target_utilization_range[0]:.2f} "
        f"to {target_utilization_range[1]:.2f}"
    )
    print(f"Output folder: {output_dir}")
    print()

    for instance_id in range(1, num_instances + 1):
        instance = generate_feasible_instance(
            rng,
            instance_id=instance_id,
            min_jobs=min_jobs,
            max_jobs=max_jobs,
            horizon_range=(min_horizon, max_horizon),
            green_share_range=green_share_range,
            brown_share_range=brown_share_range,
            target_utilization_range=target_utilization_range,
        )

        jobs = instance["jobs"]
        intervals = instance["energy_intervals"]

        max_result, max_timing = time_solver(
            alg.max_flow_passes_schedule,
            jobs,
            intervals,
            repeats=timing_repeats,
        )

        pure_result, pure_timing = time_solver(
            alg.pure_min_cost_schedule,
            jobs,
            intervals,
            repeats=timing_repeats,
        )

        # Construction guarantees feasibility, so a solver failure is important.
        if not max_result["feasible"]:
            raise RuntimeError(
                f"Max Flow + Passes failed on feasible instance {instance_id}."
            )

        if not pure_result["feasible"]:
            raise RuntimeError(
                f"Pure Min-Cost Flow failed on feasible instance {instance_id}."
            )

        max_usage = max_result["final"]["energy_usage"]
        pure_usage = pure_result["final"]["energy_usage"]

        max_median_ms = 1000.0 * max_timing["median_seconds"]
        pure_median_ms = 1000.0 * pure_timing["median_seconds"]

        row = {
            "instance_id": instance_id,
            "horizon": instance["horizon"],
            "num_jobs": len(jobs),
            "num_energy_intervals": len(intervals),
            "total_processing": max_result["total_processing"],
            "max_flow_feasible": max_result["feasible"],
            "pure_min_cost_feasible": pure_result["feasible"],
            "max_flow_cost": max_result["normalized_cost"],
            "pure_min_cost_cost": pure_result["normalized_cost"],
            "cost_difference_max_minus_pure": (
                max_result["normalized_cost"]
                - pure_result["normalized_cost"]
            ),
            "max_flow_green": max_usage["green"],
            "max_flow_brown": max_usage["brown"],
            "max_flow_red": max_usage["red"],
            "pure_green": pure_usage["green"],
            "pure_brown": pure_usage["brown"],
            "pure_red": pure_usage["red"],
            "max_flow_mean_ms": 1000.0 * max_timing["mean_seconds"],
            "max_flow_median_ms": max_median_ms,
            "pure_mean_ms": 1000.0 * pure_timing["mean_seconds"],
            "pure_median_ms": pure_median_ms,
            "median_speed_ratio_max_over_pure": (
                max_median_ms / pure_median_ms
                if pure_median_ms > 0
                else float("inf")
            ),
        }

        rows.append(row)

        save_instance_json(
            output_dir,
            instance,
            max_result,
            pure_result,
            max_timing,
            pure_timing,
        )

        print(
            f"[{instance_id:02d}/{num_instances}] "
            f"jobs={len(jobs):2d} "
            f"horizon={instance['horizon']:2d} | "
            f"cost: passes={max_result['normalized_cost']:.2f}, "
            f"pure={pure_result['normalized_cost']:.2f} | "
            f"median ms: passes={max_median_ms:.4f}, "
            f"pure={pure_median_ms:.4f}"
        )

    write_results_csv(output_dir, rows)

    write_summary(
        output_dir,
        rows,
        num_instances=num_instances,
        timing_repeats=timing_repeats,
        seed=seed,
        min_jobs=min_jobs,
        max_jobs=max_jobs,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        green_share_range=green_share_range,
        brown_share_range=brown_share_range,
        target_utilization_range=target_utilization_range,
    )

    print()
    print("=" * 72)
    print("FINISHED")
    print("=" * 72)
    print(f"Results saved to: {output_dir}")
    print(f"Summary:          {output_dir / 'summary.txt'}")
    print(f"CSV:              {output_dir / 'results.csv'}")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate random feasible scheduling instances and compare "
            "Max Flow + Passes against Pure Min-Cost Flow."
        )
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=30,
        help="Number of random feasible instances. Default: 30",
    )

    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=20,
        help=(
            "Number of timed executions of each solver per instance. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="random_benchmark_results",
        help="Output folder name. Default: random_benchmark_results",
    )

    parser.add_argument(
        "--min-jobs",
        type=int,
        default=5,
        help="Minimum jobs per instance. Default: 5",
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=15,
        help="Maximum jobs per instance. Default: 15",
    )

    parser.add_argument(
        "--min-horizon",
        type=int,
        default=20,
        help="Minimum scheduling horizon. Default: 20",
    )

    parser.add_argument(
        "--max-horizon",
        type=int,
        default=60,
        help="Maximum scheduling horizon. Default: 60",
    )

    parser.add_argument(
        "--green-min",
        type=float,
        default=0.65,
        help="Minimum fraction of the horizon that is green. Default: 0.65",
    )

    parser.add_argument(
        "--green-max",
        type=float,
        default=0.75,
        help="Maximum fraction of the horizon that is green. Default: 0.75",
    )

    parser.add_argument(
        "--brown-min",
        type=float,
        default=0.20,
        help="Minimum fraction of the horizon that is brown. Default: 0.20",
    )

    parser.add_argument(
        "--brown-max",
        type=float,
        default=0.25,
        help="Maximum fraction of the horizon that is brown. Default: 0.25",
    )

    parser.add_argument(
        "--random-brown",
        action="store_true",
        help=(
            "Do not control the brown share. Instead, randomly split all "
            "non-green time between brown and red."
        ),
    )

    parser.add_argument(
        "--util-min",
        type=float,
        default=0.55,
        help="Minimum target machine utilization. Default: 0.55",
    )

    parser.add_argument(
        "--util-max",
        type=float,
        default=0.90,
        help="Maximum target machine utilization. Default: 0.90",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.instances <= 0:
        raise ValueError("Require instances > 0.")

    if args.timing_repeats <= 0:
        raise ValueError("Require timing-repeats > 0.")

    if args.min_jobs <= 0 or args.max_jobs < args.min_jobs:
        raise ValueError("Require 0 < min-jobs <= max-jobs.")

    if args.min_horizon <= 0 or args.max_horizon < args.min_horizon:
        raise ValueError("Require 0 < min-horizon <= max-horizon.")

    if not (0.0 < args.green_min <= args.green_max < 1.0):
        raise ValueError(
            "Require 0 < green-min <= green-max < 1."
        )

    if not args.random_brown:
        if not (0.0 < args.brown_min <= args.brown_max < 1.0):
            raise ValueError(
                "Require 0 < brown-min <= brown-max < 1."
            )

    if not (0.0 < args.util_min <= args.util_max <= 1.0):
        raise ValueError(
            "Require 0 < util-min <= util-max <= 1."
        )

    if args.max_jobs > args.min_horizon:
        print(
            "Warning: some small horizons may cap the number of jobs "
            "because each job needs at least one unit of processing."
        )

    if (
        not args.random_brown
        and args.green_max + args.brown_max >= 1.0
    ):
        print(
            "Warning: some green/brown combinations may leave no "
            "positive red time. Individual generated instances may fail "
            "if the selected shares are incompatible."
        )

    brown_share_range = (
        None
        if args.random_brown
        else (args.brown_min, args.brown_max)
    )

    run_random_experiment(
        num_instances=args.instances,
        timing_repeats=args.timing_repeats,
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
    )
