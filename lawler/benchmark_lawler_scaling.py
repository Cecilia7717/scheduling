from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Files
# ============================================================

HERE = Path(__file__).resolve().parent

# CHANGE THIS to the filename containing your synthetic generator.
GENERATOR_FILE = HERE / "compare_algorithms.py"

LAWLER_FILE = HERE / "dp.py"

# ============================================================
# Experiment settings
# ============================================================

    # 5,
    # 10,
    # 20,
    # 30,
JOB_COUNTS = [
    40,
    50,
]

# Same style as your previous MaxFlow scaling experiments.
INTERVAL_LEVELS = [
    (0.10, "sparse"),
    (0.25, "light"),
    (0.50, "medium"),
    (1.00, "dense"),
]

INSTANCES = 30
SEED = 42

HORIZON_PER_JOB = 6

UTIL_MIN = 0.70
UTIL_MAX = 0.80

GREEN_MIN = 0.50
GREEN_MAX = 0.60

BROWN_MIN = 0.20
BROWN_MAX = 0.25

KAPPA_MIN = 1
KAPPA_MAX = 10


# Carbon costs from your current experiments.
ENERGY_COST = {
    "green": 0,
    "brown": 1,
    "red": 2,
}


# ============================================================
# Dynamic module loading
# ============================================================

def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


generator = load_module(
    GENERATOR_FILE,
    "synthetic_generator",
)

lawler = load_module(
    LAWLER_FILE,
    "lawler_algorithm",
)


# ============================================================
# Outsourcing penalties
# ============================================================

def assign_outsourcing_penalties(
    jobs,
    rng: random.Random,
) -> Dict[str, int]:

    return {
        job.name: rng.randint(
            KAPPA_MIN,
            KAPPA_MAX,
        )
        for job in jobs
    }


# ============================================================
# Dummy-job binary decomposition
# ============================================================

def binary_dummy_lengths(
    length: int,
) -> List[int]:

    if length <= 0:
        raise ValueError(
            "Interval length must be positive."
        )

    lengths = []

    power = 1
    covered = 0

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
# Cloud -> penalty reduction
# ============================================================

def construct_penalty_instance(
    jobs,
    intervals,
    kappa: Dict[str, int],
):
    """
    No precedence constraints in this synthetic experiment, so:

        R_j = r_j
        D_j = d_j.

    Original job:
        p = p_j
        r = R_j
        d = D_j
        w = kappa_j

    Dummy job:
        p = b_kh
        r = t_(k-1)
        d = t_k
        w = c_k * b_kh
    """

    penalty_jobs = []
    original_jobs = []
    dummy_jobs = []

    # --------------------------------------------------------
    # Original jobs
    # --------------------------------------------------------

    for job in jobs:

        new_job = lawler.Job(
            id=job.name,
            release=float(job.release),
            processing=float(job.processing),
            due=float(job.deadline),
            weight=int(kappa[job.name]),
        )

        penalty_jobs.append(new_job)
        original_jobs.append(new_job)

    # --------------------------------------------------------
    # Dummy jobs
    # --------------------------------------------------------

    for k, interval in enumerate(intervals):

        start = int(round(interval.start))
        end = int(round(interval.end))
        length = end - start

        if length <= 0:
            raise ValueError(
                f"Invalid interval: {interval.name}"
            )

        carbon_cost = ENERGY_COST[
            interval.energy
        ]

        pieces = binary_dummy_lengths(
            length
        )

        for h, piece in enumerate(pieces):

            weight = carbon_cost * piece

            # Lawler implementation requires positive weight.
            # c_k = 0 dummy jobs contribute no rejection cost,
            # so they can be omitted.
            if weight == 0:
                continue

            dummy = lawler.Job(
                id=f"G{k}_{h}",
                release=float(start),
                processing=float(piece),
                due=float(end),
                weight=int(weight),
            )

            penalty_jobs.append(dummy)
            dummy_jobs.append(dummy)

    return (
        penalty_jobs,
        original_jobs,
        dummy_jobs,
    )


# ============================================================
# Run Lawler once
# ============================================================

def run_lawler(
    penalty_jobs,
) -> Dict[str, Any]:

    total_weight = sum(
        job.weight
        for job in penalty_jobs
    )

    start = time.perf_counter()

    maximum_on_time_weight = (
        lawler.lawler_optimal_on_time_weight(
            penalty_jobs
        )
    )

    runtime = (
        time.perf_counter() - start
    )

    minimum_rejected_penalty = (
        total_weight
        - maximum_on_time_weight
    )

    return {
        "total_weight": total_weight,
        "maximum_on_time_weight":
            maximum_on_time_weight,
        "minimum_rejected_penalty":
            minimum_rejected_penalty,
        "runtime_seconds": runtime,
    }


# ============================================================
# Save results
# ============================================================

def write_results(
    output_dir: Path,
    rows: List[Dict[str, Any]],
):

    path = output_dir / "results.csv"

    fieldnames = [
        "num_original_jobs",
        "interval_ratio",
        "density",
        "instance_id",
        "seed",

        "horizon",
        "num_energy_intervals",

        "num_dummy_jobs",
        "num_penalty_jobs",

        "distinct_release_dates_k",

        "original_weight",
        "dummy_weight",
        "total_weight_W",

        "maximum_on_time_weight",
        "minimum_rejected_penalty",

        "lawler_runtime_seconds",
        "lawler_runtime_ms",
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
# Aggregate results
# ============================================================

def write_summary(
    output_dir: Path,
    rows: List[Dict[str, Any]],
):

    path = output_dir / "summary.csv"

    grouped = {}

    for row in rows:

        key = (
            row["num_original_jobs"],
            row["interval_ratio"],
            row["density"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    summary_rows = []

    for (
        num_jobs,
        ratio,
        density,
    ), group in sorted(grouped.items()):

        runtimes = [
            x["lawler_runtime_seconds"]
            for x in group
        ]

        dummy_counts = [
            x["num_dummy_jobs"]
            for x in group
        ]

        total_jobs = [
            x["num_penalty_jobs"]
            for x in group
        ]

        k_values = [
            x["distinct_release_dates_k"]
            for x in group
        ]

        weights = [
            x["total_weight_W"]
            for x in group
        ]

        summary_rows.append(
            {
                "num_original_jobs":
                    num_jobs,

                "interval_ratio":
                    ratio,

                "density":
                    density,

                "num_instances":
                    len(group),

                "mean_dummy_jobs":
                    statistics.mean(
                        dummy_counts
                    ),

                "median_dummy_jobs":
                    statistics.median(
                        dummy_counts
                    ),

                "mean_penalty_jobs":
                    statistics.mean(
                        total_jobs
                    ),

                "median_penalty_jobs":
                    statistics.median(
                        total_jobs
                    ),

                "mean_k":
                    statistics.mean(
                        k_values
                    ),

                "median_k":
                    statistics.median(
                        k_values
                    ),

                "mean_W":
                    statistics.mean(
                        weights
                    ),

                "median_W":
                    statistics.median(
                        weights
                    ),

                "mean_runtime_seconds":
                    statistics.mean(
                        runtimes
                    ),

                "median_runtime_seconds":
                    statistics.median(
                        runtimes
                    ),

                "min_runtime_seconds":
                    min(runtimes),

                "max_runtime_seconds":
                    max(runtimes),
            }
        )

    fieldnames = list(
        summary_rows[0].keys()
    )

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
            summary_rows
        )


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    output_folder: str,
):

    output_dir = (
        HERE / output_folder
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 90)
    print("LAWLER SCALABILITY EXPERIMENT")
    print("=" * 90)

    print(
        f"Job counts: {JOB_COUNTS}"
    )

    print(
        f"Instances/configuration: "
        f"{INSTANCES}"
    )

    print(
        f"Horizon/job: "
        f"{HORIZON_PER_JOB}"
    )

    print(
        f"Utilization: "
        f"{UTIL_MIN:.2f}-"
        f"{UTIL_MAX:.2f}"
    )

    print(
        f"Kappa: "
        f"{KAPPA_MIN}-"
        f"{KAPPA_MAX}"
    )

    print()

    # ========================================================
    # Job-count sweep
    # ========================================================

    for num_jobs in JOB_COUNTS:

        horizon = (
            HORIZON_PER_JOB
            * num_jobs
        )

        for (
            interval_ratio,
            density,
        ) in INTERVAL_LEVELS:

            num_intervals = max(
                3,
                int(
                    round(
                        interval_ratio
                        * num_jobs
                    )
                ),
            )

            # Cannot have more positive intervals
            # than integer time units.
            num_intervals = min(
                num_intervals,
                horizon,
            )

            print()
            print("-" * 90)

            print(
                f"jobs={num_jobs} | "
                f"density={density} | "
                f"ratio={interval_ratio} | "
                f"intervals={num_intervals}"
            )

            print("-" * 90)

            # =================================================
            # Instances
            # =================================================

            for instance_id in range(
                1,
                INSTANCES + 1,
            ):

                # ---------------------------------------------
                # Deterministic seed for this exact instance.
                #
                # Same configuration + seed always regenerates
                # the same instance.
                # ---------------------------------------------

                instance_seed = (
                    SEED
                    + num_jobs * 1_000_000
                    + int(
                        interval_ratio
                        * 100
                    ) * 10_000
                    + instance_id
                )

                rng = random.Random(
                    instance_seed
                )

                penalty_rng = random.Random(
                    instance_seed
                    + 500_000_000
                )

                # ---------------------------------------------
                # Generate original synthetic cloud instance
                # ---------------------------------------------

                instance = (
                    generator.generate_feasible_instance(
                        rng,
                        instance_id=instance_id,

                        fixed_jobs=num_jobs,

                        horizon_per_job=
                            HORIZON_PER_JOB,

                        fixed_intervals=
                            num_intervals,

                        green_share_range=(
                            GREEN_MIN,
                            GREEN_MAX,
                        ),

                        brown_share_range=(
                            BROWN_MIN,
                            BROWN_MAX,
                        ),

                        target_utilization_range=(
                            UTIL_MIN,
                            UTIL_MAX,
                        ),
                    )
                )

                jobs = instance["jobs"]

                intervals = (
                    instance[
                        "energy_intervals"
                    ]
                )

                # ---------------------------------------------
                # Generate outsourcing penalties
                # ---------------------------------------------

                kappa = (
                    assign_outsourcing_penalties(
                        jobs,
                        penalty_rng,
                    )
                )

                # ---------------------------------------------
                # Apply your reduction
                # ---------------------------------------------

                (
                    penalty_jobs,
                    original_penalty_jobs,
                    dummy_jobs,
                ) = (
                    construct_penalty_instance(
                        jobs,
                        intervals,
                        kappa,
                    )
                )

                # ---------------------------------------------
                # Structural values
                # ---------------------------------------------

                k_distinct = len(
                    {
                        job.release
                        for job
                        in penalty_jobs
                    }
                )

                original_weight = sum(
                    job.weight
                    for job
                    in original_penalty_jobs
                )

                dummy_weight = sum(
                    job.weight
                    for job
                    in dummy_jobs
                )

                # ---------------------------------------------
                # Run Lawler
                # ---------------------------------------------

                result = run_lawler(
                    penalty_jobs
                )

                # ---------------------------------------------
                # Record
                # ---------------------------------------------

                row = {
                    "num_original_jobs":
                        num_jobs,

                    "interval_ratio":
                        interval_ratio,

                    "density":
                        density,

                    "instance_id":
                        instance_id,

                    "seed":
                        instance_seed,

                    "horizon":
                        horizon,

                    "num_energy_intervals":
                        len(intervals),

                    "num_dummy_jobs":
                        len(dummy_jobs),

                    "num_penalty_jobs":
                        len(penalty_jobs),

                    "distinct_release_dates_k":
                        k_distinct,

                    "original_weight":
                        original_weight,

                    "dummy_weight":
                        dummy_weight,

                    "total_weight_W":
                        result[
                            "total_weight"
                        ],

                    "maximum_on_time_weight":
                        result[
                            "maximum_on_time_weight"
                        ],

                    "minimum_rejected_penalty":
                        result[
                            "minimum_rejected_penalty"
                        ],

                    "lawler_runtime_seconds":
                        result[
                            "runtime_seconds"
                        ],

                    "lawler_runtime_ms":
                        1000.0
                        * result[
                            "runtime_seconds"
                        ],
                }

                rows.append(row)

                # Save after EVERY instance.
                #
                # If a later large instance takes forever,
                # completed results are not lost.
                write_results(
                    output_dir,
                    rows,
                )

                print(
                    f"[{instance_id:02d}/"
                    f"{INSTANCES:02d}] "
                    f"N={len(penalty_jobs):3d} "
                    f"dummy={len(dummy_jobs):3d} "
                    f"k={k_distinct:3d} "
                    f"W={result['total_weight']:5d} | "
                    f"OPT="
                    f"{result['minimum_rejected_penalty']:5d} | "
                    f"time="
                    f"{result['runtime_seconds']:.6f}s"
                )

    # ========================================================
    # Final output
    # ========================================================

    write_results(
        output_dir,
        rows,
    )

    write_summary(
        output_dir,
        rows,
    )

    print()
    print("=" * 90)
    print("FINISHED")
    print("=" * 90)

    print(
        f"Raw results: "
        f"{output_dir / 'results.csv'}"
    )

    print(
        f"Summary:     "
        f"{output_dir / 'summary.csv'}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Lawler's exact algorithm on "
            "penalty instances produced by the "
            "cloud-outsourcing reduction."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="lawler_scaling_results",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    run_experiment(
        output_folder=args.output,
    )