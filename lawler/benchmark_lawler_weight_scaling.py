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

# CHANGE this to your synthetic-generator filename.
GENERATOR_FILE = HERE / "compare_algorithms.py"

LAWLER_FILE = HERE / "dp.py"


# ============================================================
# Stage-3 settings
# ============================================================

JOB_COUNTS = [
    5,
    10,
    15,
    20,
]

WEIGHT_SCALES = [
    1,
    2,
    4,
    8,
    16,
]

INSTANCES = 10
SEED = 42

HORIZON_PER_JOB = 6

# Keep interval structure fixed relative to n.
INTERVAL_RATIO = 0.50

UTIL_MIN = 0.70
UTIL_MAX = 0.80

GREEN_MIN = 0.50
GREEN_MAX = 0.60

BROWN_MIN = 0.20
BROWN_MAX = 0.25

# Base outsourcing penalties.
KAPPA_MIN = 1
KAPPA_MAX = 10


ENERGY_COST = {
    "green": 0,
    "brown": 1,
    "red": 2,
}


# ============================================================
# Module loading
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
    "synthetic_generator_stage3",
)

lawler = load_module(
    LAWLER_FILE,
    "lawler_algorithm_stage3",
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
# Dummy-job decomposition
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

    remainder = (
        length - covered
    )

    if remainder > 0:
        lengths.append(
            remainder
        )

    assert sum(lengths) == length

    return lengths


# ============================================================
# Construct BASE penalty instance
# ============================================================

def construct_base_penalty_instance(
    jobs,
    intervals,
    kappa: Dict[str, int],
):
    """
    Construct the reduced penalty instance ONCE.

    The returned weights correspond to alpha = 1.

    Later we multiply every positive weight by alpha while
    keeping p, r, and d exactly unchanged.
    """

    penalty_jobs = []

    num_original = 0
    num_dummy = 0

    original_weight = 0
    dummy_weight = 0

    # --------------------------------------------------------
    # Original jobs
    # --------------------------------------------------------

    for job in jobs:

        weight = int(
            kappa[job.name]
        )

        penalty_job = lawler.Job(
            id=job.name,
            release=float(
                job.release
            ),
            processing=float(
                job.processing
            ),
            due=float(
                job.deadline
            ),
            weight=weight,
        )

        penalty_jobs.append(
            penalty_job
        )

        num_original += 1

        original_weight += weight

    # --------------------------------------------------------
    # Dummy jobs
    # --------------------------------------------------------

    for k, interval in enumerate(
        intervals
    ):

        start = int(
            round(interval.start)
        )

        end = int(
            round(interval.end)
        )

        length = end - start

        if length <= 0:
            raise ValueError(
                f"Invalid interval "
                f"{interval.name}: "
                f"[{start}, {end})"
            )

        carbon_cost = ENERGY_COST[
            interval.energy
        ]

        pieces = (
            binary_dummy_lengths(
                length
            )
        )

        for h, piece in enumerate(
            pieces
        ):

            weight = (
                carbon_cost
                * piece
            )

            # c_k = 0 gives zero-weight dummy jobs.
            # They do not affect the rejection objective and
            # the current Lawler implementation requires
            # positive weights.
            if weight == 0:
                continue

            dummy = lawler.Job(
                id=f"G{k}_{h}",
                release=float(start),
                processing=float(piece),
                due=float(end),
                weight=int(weight),
            )

            penalty_jobs.append(
                dummy
            )

            num_dummy += 1

            dummy_weight += weight

    return {
        "jobs": penalty_jobs,

        "num_original":
            num_original,

        "num_dummy":
            num_dummy,

        "original_weight":
            original_weight,

        "dummy_weight":
            dummy_weight,
    }


# ============================================================
# Scale ONLY weights
# ============================================================

def scale_penalty_instance(
    base_jobs,
    alpha: int,
):
    """
    Create an identical scheduling instance with every weight
    multiplied by alpha.

    IMPORTANT:

        processing  unchanged
        release     unchanged
        due         unchanged
        job IDs     unchanged

    Only:

        weight <- alpha * weight

    changes.
    """

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    scaled_jobs = []

    for job in base_jobs:

        scaled_jobs.append(
            lawler.Job(
                id=job.id,
                release=job.release,
                processing=job.processing,
                due=job.due,
                weight=(
                    alpha
                    * job.weight
                ),
            )
        )

    return scaled_jobs


# ============================================================
# Run Lawler
# ============================================================

def run_lawler(
    jobs,
):

    total_weight = sum(
        job.weight
        for job in jobs
    )

    start = time.perf_counter()

    maximum_on_time_weight = (
        lawler.lawler_optimal_on_time_weight(
            jobs
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    minimum_rejected_penalty = (
        total_weight
        - maximum_on_time_weight
    )

    return {
        "W":
            total_weight,

        "max_on_time_weight":
            maximum_on_time_weight,

        "minimum_rejected_penalty":
            minimum_rejected_penalty,

        "runtime_seconds":
            elapsed,
    }


# ============================================================
# CSV output
# ============================================================

RESULT_FIELDS = [
    "num_original_jobs",
    "instance_id",
    "instance_seed",

    "weight_scale_alpha",

    "horizon",
    "num_energy_intervals",

    "num_dummy_jobs",
    "num_penalty_jobs",

    "distinct_release_dates_k",

    "base_original_weight",
    "base_dummy_weight",
    "base_total_weight_W",

    "scaled_total_weight_W",

    "maximum_on_time_weight",
    "minimum_rejected_penalty",

    "baseline_optimum",
    "expected_scaled_optimum",
    "objective_scale_correct",

    "lawler_runtime_seconds",
]


def write_results(
    output_dir: Path,
    rows: List[Dict[str, Any]],
):

    path = (
        output_dir
        / "results.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Summary
# ============================================================

def write_summary(
    output_dir: Path,
    rows: List[Dict[str, Any]],
):

    groups = {}

    for row in rows:

        key = (
            row[
                "num_original_jobs"
            ],
            row[
                "weight_scale_alpha"
            ],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    summary_rows = []

    for (
        num_jobs,
        alpha,
    ), group in sorted(
        groups.items()
    ):

        runtimes = [
            row[
                "lawler_runtime_seconds"
            ]
            for row in group
        ]

        weights = [
            row[
                "scaled_total_weight_W"
            ]
            for row in group
        ]

        total_jobs = [
            row[
                "num_penalty_jobs"
            ]
            for row in group
        ]

        k_values = [
            row[
                "distinct_release_dates_k"
            ]
            for row in group
        ]

        all_correct = all(
            row[
                "objective_scale_correct"
            ]
            for row in group
        )

        summary_rows.append(
            {
                "num_original_jobs":
                    num_jobs,

                "weight_scale_alpha":
                    alpha,

                "num_instances":
                    len(group),

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

                "all_objective_scaling_correct":
                    all_correct,
            }
        )

    path = (
        output_dir
        / "summary.csv"
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
# Main experiment
# ============================================================

def run_experiment(
    output_folder: str,
):

    output_dir = (
        HERE
        / output_folder
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 90)
    print(
        "STAGE 3: LAWLER WEIGHT-SCALING EXPERIMENT"
    )
    print("=" * 90)

    print(
        f"Job counts: "
        f"{JOB_COUNTS}"
    )

    print(
        f"Weight scales: "
        f"{WEIGHT_SCALES}"
    )

    print(
        f"Instances/job count: "
        f"{INSTANCES}"
    )

    print(
        f"Interval ratio: "
        f"{INTERVAL_RATIO}"
    )

    print()

    # ========================================================
    # Original job counts
    # ========================================================

    for num_jobs in JOB_COUNTS:

        horizon = (
            HORIZON_PER_JOB
            * num_jobs
        )

        num_intervals = max(
            3,
            int(
                round(
                    INTERVAL_RATIO
                    * num_jobs
                )
            ),
        )

        num_intervals = min(
            num_intervals,
            horizon,
        )

        print()
        print("-" * 90)

        print(
            f"Original jobs={num_jobs} | "
            f"horizon={horizon} | "
            f"intervals={num_intervals}"
        )

        print("-" * 90)

        # ====================================================
        # Independent base instances
        # ====================================================

        for instance_id in range(
            1,
            INSTANCES + 1,
        ):

            # -----------------------------------------------
            # Fixed deterministic seed.
            # -----------------------------------------------

            instance_seed = (
                SEED
                + num_jobs * 1_000_000
                + instance_id
            )

            generation_rng = (
                random.Random(
                    instance_seed
                )
            )

            penalty_rng = (
                random.Random(
                    instance_seed
                    + 500_000_000
                )
            )

            # -----------------------------------------------
            # Generate scheduling instance ONCE.
            # -----------------------------------------------

            instance = (
                generator.generate_feasible_instance(
                    generation_rng,

                    instance_id=
                        instance_id,

                    fixed_jobs=
                        num_jobs,

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

            jobs = (
                instance["jobs"]
            )

            intervals = (
                instance[
                    "energy_intervals"
                ]
            )

            # -----------------------------------------------
            # Generate kappa ONCE.
            # -----------------------------------------------

            kappa = (
                assign_outsourcing_penalties(
                    jobs,
                    penalty_rng,
                )
            )

            # -----------------------------------------------
            # Apply reduction ONCE.
            # -----------------------------------------------

            base = (
                construct_base_penalty_instance(
                    jobs,
                    intervals,
                    kappa,
                )
            )

            base_jobs = (
                base["jobs"]
            )

            base_W = sum(
                job.weight
                for job in base_jobs
            )

            N = len(
                base_jobs
            )

            k_distinct = len(
                {
                    job.release
                    for job
                    in base_jobs
                }
            )

            print(
                f"\nInstance "
                f"{instance_id:02d}/"
                f"{INSTANCES:02d}: "
                f"N={N}, "
                f"k={k_distinct}, "
                f"base W={base_W}"
            )

            # -----------------------------------------------
            # Run scales in increasing order.
            # -----------------------------------------------

            baseline_optimum = None

            for alpha in WEIGHT_SCALES:

                scaled_jobs = (
                    scale_penalty_instance(
                        base_jobs,
                        alpha,
                    )
                )

                result = (
                    run_lawler(
                        scaled_jobs
                    )
                )

                # -------------------------------------------
                # At alpha=1 establish the baseline optimum.
                # -------------------------------------------

                if alpha == 1:

                    baseline_optimum = (
                        result[
                            "minimum_rejected_penalty"
                        ]
                    )

                if baseline_optimum is None:
                    raise RuntimeError(
                        "WEIGHT_SCALES must "
                        "begin with 1."
                    )

                expected_optimum = (
                    alpha
                    * baseline_optimum
                )

                scale_correct = (
                    result[
                        "minimum_rejected_penalty"
                    ]
                    == expected_optimum
                )

                expected_W = (
                    alpha
                    * base_W
                )

                if (
                    result["W"]
                    != expected_W
                ):
                    raise AssertionError(
                        "Weight scaling failed: "
                        f"expected W="
                        f"{expected_W}, "
                        f"got "
                        f"{result['W']}"
                    )

                # -------------------------------------------
                # Save result.
                # -------------------------------------------

                row = {
                    "num_original_jobs":
                        num_jobs,

                    "instance_id":
                        instance_id,

                    "instance_seed":
                        instance_seed,

                    "weight_scale_alpha":
                        alpha,

                    "horizon":
                        horizon,

                    "num_energy_intervals":
                        len(intervals),

                    "num_dummy_jobs":
                        base[
                            "num_dummy"
                        ],

                    "num_penalty_jobs":
                        N,

                    "distinct_release_dates_k":
                        k_distinct,

                    "base_original_weight":
                        base[
                            "original_weight"
                        ],

                    "base_dummy_weight":
                        base[
                            "dummy_weight"
                        ],

                    "base_total_weight_W":
                        base_W,

                    "scaled_total_weight_W":
                        result["W"],

                    "maximum_on_time_weight":
                        result[
                            "max_on_time_weight"
                        ],

                    "minimum_rejected_penalty":
                        result[
                            "minimum_rejected_penalty"
                        ],

                    "baseline_optimum":
                        baseline_optimum,

                    "expected_scaled_optimum":
                        expected_optimum,

                    "objective_scale_correct":
                        scale_correct,

                    "lawler_runtime_seconds":
                        result[
                            "runtime_seconds"
                        ],
                }

                rows.append(row)

                # -------------------------------------------
                # Save after every Lawler execution.
                # -------------------------------------------

                write_results(
                    output_dir,
                    rows,
                )

                status = (
                    "OK"
                    if scale_correct
                    else "ERROR"
                )

                print(
                    f"    alpha={alpha:2d} | "
                    f"W={result['W']:6d} | "
                    f"OPT="
                    f"{result['minimum_rejected_penalty']:6d} | "
                    f"time="
                    f"{result['runtime_seconds']:10.6f}s | "
                    f"{status}"
                )

                if not scale_correct:

                    raise AssertionError(
                        "Scaling all weights "
                        "changed the optimal "
                        "objective by something "
                        "other than alpha."
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
        f"Summary: "
        f"{output_dir / 'summary.csv'}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Controlled total-weight scaling "
            "experiment for Lawler's exact DP."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "lawler_weight_scaling_results"
        ),
    )

    return parser.parse_args()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    if (
        not WEIGHT_SCALES
        or WEIGHT_SCALES[0] != 1
    ):
        raise ValueError(
            "WEIGHT_SCALES must start with 1."
        )

    run_experiment(
        output_folder=args.output,
    )