from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Files
# ============================================================

HERE = Path(__file__).resolve().parent

# Change these names if your files have different names.
GENERATOR_FILE = HERE / "compare_algorithms.py"
MAXFLOW_FILE = HERE / "maxflow.py"
LAWLER_FILE = HERE / "dp.py"


# ============================================================
# Dynamic module loading
# ============================================================

def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


generator = load_module(GENERATOR_FILE, "synthetic_generator")
flow = load_module(MAXFLOW_FILE, "flow_algorithm")
lawler = load_module(LAWLER_FILE, "lawler_algorithm")


# ============================================================
# Configuration
# ============================================================

# Your current MaxFlow experiments use:
#
#   green = 0
#   brown = 1
#   red   = 2
#
# We use exactly the same costs in the Lawler reduction.

ENERGY_COST = {
    "green": 0,
    "brown": 1,
    "red": 2,
}


# ============================================================
# Original cloud jobs
# ============================================================

def assign_outsourcing_penalties(
    jobs,
    rng: random.Random,
    min_penalty: int,
    max_penalty: int,
) -> Dict[str, int]:
    """
    Assign integer cloud outsourcing penalty kappa_j to each original job.

    Returns:
        {
            job_name: kappa_j,
            ...
        }
    """

    return {
        job.name: rng.randint(min_penalty, max_penalty)
        for job in jobs
    }


# ============================================================
# Binary dummy decomposition
# ============================================================

def binary_dummy_lengths(length: int) -> List[int]:
    """
    Construct the dummy processing lengths from the reduction.

    For interval length ell:

        1, 2, 4, ..., 2^(m-1)

    plus

        b = ell - (2^m - 1)

    if b > 0.

    The resulting processing times:
      1. sum to ell
      2. have subset sums covering every integer 0,...,ell
    """

    if length <= 0:
        raise ValueError("Interval length must be positive.")

    lengths = []

    power = 1
    covered = 0

    # Add 1,2,4,... while their cumulative sum does not exceed length.
    while covered + power <= length:
        lengths.append(power)
        covered += power
        power *= 2

    remainder = length - covered

    if remainder > 0:
        lengths.append(remainder)

    assert sum(lengths) == length

    return lengths


# ============================================================
# Cloud -> penalty-instance reduction
# ============================================================

def construct_penalty_instance(
    jobs,
    intervals,
    kappa: Dict[str, int],
):
    """
    Construct the penalty-scheduling instance from the reduction.

    Original job j:
        (p_j, R_j, D_j, w_j = kappa_j)

    Since the current synthetic generator has no precedence constraints:
        R_j = r_j
        D_j = d_j

    Dummy job g_(k,h):
        p = b_(k,h)
        r = interval.start
        d = interval.end
        w = c_k * b_(k,h)

    Zero-cost dummy jobs are omitted because rejecting/accepting them
    has zero penalty and therefore they do not affect the objective.
    """

    penalty_jobs = []
    original_penalty_jobs = []
    dummy_jobs = []

    # --------------------------------------------------------
    # Original jobs
    # --------------------------------------------------------

    for job in jobs:
        pjob = lawler.Job(
            id=job.name,
            release=float(job.release),
            processing=float(job.processing),
            due=float(job.deadline),
            weight=int(kappa[job.name]),
        )

        penalty_jobs.append(pjob)
        original_penalty_jobs.append(pjob)

    # --------------------------------------------------------
    # Dummy jobs
    # --------------------------------------------------------

    for k, interval in enumerate(intervals):

        start = int(round(interval.start))
        end = int(round(interval.end))
        length = end - start

        if length <= 0:
            raise ValueError(
                f"Invalid interval {interval.name}: "
                f"[{interval.start}, {interval.end})"
            )

        carbon_cost = ENERGY_COST[interval.energy]

        dummy_lengths = binary_dummy_lengths(length)

        for h, b in enumerate(dummy_lengths):

            weight = carbon_cost * b

            # Green intervals have c_k = 0.
            #
            # The current Lawler Job class requires positive weights.
            # Zero-weight dummy jobs can safely be omitted because
            # they never contribute to the rejection penalty.
            if weight == 0:
                continue

            dummy = lawler.Job(
                id=f"G{k}_{h}",
                release=float(start),
                processing=float(b),
                due=float(end),
                weight=int(weight),
            )

            penalty_jobs.append(dummy)
            dummy_jobs.append(dummy)

    return penalty_jobs, original_penalty_jobs, dummy_jobs


# ============================================================
# Solve penalty instance using Lawler
# ============================================================

def solve_with_lawler(penalty_jobs):
    """
    Lawler returns maximum ON-TIME weight.

    Our penalty objective is minimum REJECTED weight:

        OPT_pen = total_weight - maximum_on_time_weight
    """

    total_weight = sum(job.weight for job in penalty_jobs)

    start = time.perf_counter()

    maximum_on_time_weight = (
        lawler.lawler_optimal_on_time_weight(penalty_jobs)
    )

    runtime = time.perf_counter() - start

    minimum_rejected_penalty = (
        total_weight - maximum_on_time_weight
    )

    return {
        "total_weight": total_weight,
        "maximum_on_time_weight": maximum_on_time_weight,
        "minimum_rejected_penalty": minimum_rejected_penalty,
        "runtime_seconds": runtime,
    }


# ============================================================
# Exact cloud optimum by exhaustive enumeration
# ============================================================

def exact_cloud_optimum(
    jobs,
    intervals,
    kappa: Dict[str, int],
):
    """
    Enumerate every subset A of original jobs.

    A = jobs kept locally on the edge server.
    J \\ A = jobs outsourced to the cloud.

    For every A:

        cost(A)
          = minimum carbon cost of scheduling A locally
            + sum_{j not in A} kappa_j

    The local scheduling cost is obtained from the existing
    pure min-cost-flow implementation.

    This is exponential and is ONLY for Stage-1 validation.
    """

    n = len(jobs)

    best_cost = float("inf")
    best_local_jobs = None

    subsets_checked = 0
    feasible_subsets = 0

    start_total = time.perf_counter()

    # --------------------------------------------------------
    # Enumerate all 2^n subsets
    # --------------------------------------------------------

    for mask in range(1 << n):

        subsets_checked += 1

        local_jobs = [
            jobs[i]
            for i in range(n)
            if mask & (1 << i)
        ]

        local_names = {job.name for job in local_jobs}

        outsourcing_cost = sum(
            kappa[job.name]
            for job in jobs
            if job.name not in local_names
        )

        # ----------------------------------------------------
        # Empty local subset
        # ----------------------------------------------------

        if not local_jobs:
            local_carbon_cost = 0.0
            feasible = True

        else:
            result = flow.pure_min_cost_schedule(
                local_jobs,
                intervals,
                verbose=False,
            )

            feasible = result["feasible"]

            if feasible:
                local_carbon_cost = result["normalized_cost"]

        # ----------------------------------------------------
        # Infeasible local subset cannot be used
        # ----------------------------------------------------

        if not feasible:
            continue

        feasible_subsets += 1

        total_cost = (
            float(local_carbon_cost)
            + float(outsourcing_cost)
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_local_jobs = sorted(local_names)

    runtime = time.perf_counter() - start_total

    if best_local_jobs is None:
        raise RuntimeError(
            "No feasible cloud solution found. "
            "This should be impossible because outsourcing all jobs "
            "is always feasible."
        )

    return {
        "optimal_cost": best_cost,
        "best_local_jobs": best_local_jobs,
        "subsets_checked": subsets_checked,
        "feasible_subsets": feasible_subsets,
        "runtime_seconds": runtime,
    }


# ============================================================
# Validate one instance
# ============================================================

def validate_instance(
    instance,
    penalty_rng: random.Random,
    min_penalty: int,
    max_penalty: int,
):
    jobs = instance["jobs"]
    intervals = instance["energy_intervals"]

    # --------------------------------------------------------
    # Assign kappa_j
    # --------------------------------------------------------

    kappa = assign_outsourcing_penalties(
        jobs,
        penalty_rng,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # --------------------------------------------------------
    # Reduction
    # --------------------------------------------------------

    (
        penalty_jobs,
        original_penalty_jobs,
        dummy_jobs,
    ) = construct_penalty_instance(
        jobs,
        intervals,
        kappa,
    )

    # --------------------------------------------------------
    # Lawler
    # --------------------------------------------------------

    lawler_result = solve_with_lawler(
        penalty_jobs
    )

    # --------------------------------------------------------
    # Independent exact cloud solution
    # --------------------------------------------------------

    cloud_result = exact_cloud_optimum(
        jobs,
        intervals,
        kappa,
    )

    # Costs are integral under this experiment.
    lawler_opt = lawler_result[
        "minimum_rejected_penalty"
    ]

    cloud_opt = cloud_result[
        "optimal_cost"
    ]

    match = abs(lawler_opt - cloud_opt) <= 1e-9

    # --------------------------------------------------------
    # Useful structural statistics
    # --------------------------------------------------------

    distinct_release_dates = len(
        {
            job.release
            for job in penalty_jobs
        }
    )

    original_weight = sum(
        job.weight
        for job in original_penalty_jobs
    )

    dummy_weight = sum(
        job.weight
        for job in dummy_jobs
    )

    return {
        "instance_id": instance["instance_id"],

        "num_original_jobs": len(jobs),
        "num_energy_intervals": len(intervals),

        "num_dummy_jobs": len(dummy_jobs),
        "num_penalty_jobs": len(penalty_jobs),

        "distinct_release_dates_k": distinct_release_dates,

        "original_weight": original_weight,
        "dummy_weight": dummy_weight,
        "total_weight_W": lawler_result["total_weight"],

        "lawler_penalty_optimum": lawler_opt,
        "exact_cloud_optimum": cloud_opt,

        "match": match,

        "lawler_runtime_seconds":
            lawler_result["runtime_seconds"],

        "cloud_enumeration_runtime_seconds":
            cloud_result["runtime_seconds"],

        "subsets_checked":
            cloud_result["subsets_checked"],

        "feasible_subsets":
            cloud_result["feasible_subsets"],

        "best_local_jobs":
            ",".join(cloud_result["best_local_jobs"]),
    }


# ============================================================
# Main Stage-1 experiment
# ============================================================

def run_stage1(
    job_counts: List[int],
    instances_per_size: int,
    seed: int,
    output_folder: str,
    horizon_per_job: int,
    interval_ratio: float,
    min_penalty: int,
    max_penalty: int,
):
    output_dir = HERE / output_folder
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = output_dir / "stage1_validation.csv"

    rows = []

    total_tests = 0
    total_matches = 0

    print("=" * 80)
    print("STAGE 1: LAWLER REDUCTION VALIDATION")
    print("=" * 80)

    print(f"Job counts:          {job_counts}")
    print(f"Instances per size:  {instances_per_size}")
    print(f"Seed:                {seed}")
    print(f"Horizon/job:         {horizon_per_job}")
    print(f"Interval ratio:      {interval_ratio}")
    print(
        f"Outsourcing penalty: "
        f"{min_penalty} to {max_penalty}"
    )
    print()

    # --------------------------------------------------------
    # Different n
    # --------------------------------------------------------

    for n in job_counts:

        horizon = horizon_per_job * n

        # At least 3 because your existing energy generator
        # requires green/brown/red all to have positive length.
        num_intervals = max(
            3,
            int(round(interval_ratio * n)),
        )

        print("-" * 80)
        print(
            f"n={n}, horizon={horizon}, "
            f"intervals={num_intervals}"
        )
        print("-" * 80)

        # Separate deterministic random streams for each n.
        generation_rng = random.Random(
            seed + n * 100000
        )

        penalty_rng = random.Random(
            seed + n * 200000
        )

        for instance_index in range(
            1,
            instances_per_size + 1,
        ):

            # ------------------------------------------------
            # Generate exactly the same type of synthetic
            # instance as your existing experiments.
            # ------------------------------------------------

            instance = generator.generate_feasible_instance(
                generation_rng,
                instance_id=instance_index,

                fixed_jobs=n,
                horizon_per_job=horizon_per_job,
                fixed_intervals=num_intervals,

                # Match your previous experiment settings.
                green_share_range=(0.50, 0.60),
                brown_share_range=(0.20, 0.25),
                target_utilization_range=(0.70, 0.80),
            )

            # ------------------------------------------------
            # Validate reduction
            # ------------------------------------------------

            result = validate_instance(
                instance,
                penalty_rng,
                min_penalty=min_penalty,
                max_penalty=max_penalty,
            )

            result["job_count_group"] = n
            result["interval_ratio"] = interval_ratio

            rows.append(result)

            total_tests += 1

            if result["match"]:
                total_matches += 1

            status = (
                "MATCH"
                if result["match"]
                else "MISMATCH"
            )

            print(
                f"[n={n:2d}] "
                f"[{instance_index:02d}/"
                f"{instances_per_size:02d}] "
                f"{status} | "
                f"cloud={result['exact_cloud_optimum']:.0f} "
                f"lawler={result['lawler_penalty_optimum']:.0f} | "
                f"original={result['num_original_jobs']} "
                f"dummy={result['num_dummy_jobs']} "
                f"N={result['num_penalty_jobs']} "
                f"k={result['distinct_release_dates_k']} "
                f"W={result['total_weight_W']} | "
                f"Lawler="
                f"{result['lawler_runtime_seconds']:.4f}s "
                f"enumeration="
                f"{result['cloud_enumeration_runtime_seconds']:.4f}s"
            )

            # ------------------------------------------------
            # Stop immediately on a mismatch.
            #
            # For validation, a mismatch is something we want
            # to inspect rather than average away.
            # ------------------------------------------------

            if not result["match"]:
                print()
                print("ERROR: reduction validation failed.")
                print(
                    "Stopping so this instance can be inspected."
                )

                write_csv(output_csv, rows)

                raise AssertionError(
                    f"Instance n={n}, "
                    f"id={instance_index}: "
                    f"Lawler={result['lawler_penalty_optimum']}, "
                    f"cloud={result['exact_cloud_optimum']}"
                )

    write_csv(output_csv, rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        "Stage 1: Lawler Reduction Validation\n"
        "====================================\n\n"
        f"Seed: {seed}\n"
        f"Job counts: {job_counts}\n"
        f"Instances per size: {instances_per_size}\n"
        f"Total instances: {total_tests}\n"
        f"Matching instances: {total_matches}\n"
        f"Mismatching instances: "
        f"{total_tests - total_matches}\n\n"
        f"Validation rate: "
        f"{100.0 * total_matches / total_tests:.2f}%\n"
    )

    summary_path = output_dir / "summary.txt"

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(
        f"Matches: {total_matches}/{total_tests}"
    )
    print(f"CSV:     {output_csv}")
    print(f"Summary: {summary_path}")


# ============================================================
# CSV
# ============================================================

def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
):
    if not rows:
        return

    fieldnames = [
        "job_count_group",
        "instance_id",
        "interval_ratio",

        "num_original_jobs",
        "num_energy_intervals",

        "num_dummy_jobs",
        "num_penalty_jobs",

        "distinct_release_dates_k",

        "original_weight",
        "dummy_weight",
        "total_weight_W",

        "lawler_penalty_optimum",
        "exact_cloud_optimum",

        "match",

        "lawler_runtime_seconds",
        "cloud_enumeration_runtime_seconds",

        "subsets_checked",
        "feasible_subsets",

        "best_local_jobs",
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


# ============================================================
# Command line
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 validation of the cloud-outsourcing "
            "to penalty-scheduling reduction using Lawler."
        )
    )

    parser.add_argument(
        "--jobs",
        nargs="+",
        type=int,
        default=[5, 8, 10],
        help=(
            "Original job counts to validate. "
            "Default: 5 8 10"
        ),
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=30,
        help=(
            "Number of random instances for each job count. "
            "Default: 30"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--horizon-per-job",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--interval-ratio",
        type=float,
        default=0.50,
        help=(
            "Number of carbon intervals relative to original "
            "job count. Default: 0.50"
        ),
    )

    parser.add_argument(
        "--min-penalty",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-penalty",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="lawler_stage1_validation",
    )

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    if any(n <= 0 for n in args.jobs):
        raise ValueError(
            "All job counts must be positive."
        )

    if args.instances <= 0:
        raise ValueError(
            "--instances must be positive."
        )

    if args.horizon_per_job <= 0:
        raise ValueError(
            "--horizon-per-job must be positive."
        )

    if args.interval_ratio <= 0:
        raise ValueError(
            "--interval-ratio must be positive."
        )

    if (
        args.min_penalty <= 0
        or args.max_penalty < args.min_penalty
    ):
        raise ValueError(
            "Require 0 < min-penalty <= max-penalty."
        )

    run_stage1(
        job_counts=args.jobs,
        instances_per_size=args.instances,
        seed=args.seed,
        output_folder=args.output,
        horizon_per_job=args.horizon_per_job,
        interval_ratio=args.interval_ratio,
        min_penalty=args.min_penalty,
        max_penalty=args.max_penalty,
    )