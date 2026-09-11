#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Directory name format:
#
# jobs_1000_intervals_500_medium
# jobs_200_intervals_20_sparse
# ============================================================

DIR_PATTERN = re.compile(
    r"^jobs_(\d+)_intervals_(\d+)_(sparse|light|medium|dense|very_dense)$"
)


LEVEL_ORDER = [
    "sparse",
    "light",
    "medium",
    "dense",
    "very_dense",
]

LEVEL_LABELS = {
    "sparse": "Sparse (0.10 × jobs)",
    "light": "Light (0.25 × jobs)",
    "medium": "Medium (0.50 × jobs)",
    "dense": "Dense (1.00 × jobs)",
    "very_dense": "Very dense (2.00 × jobs)",
}


def read_gpu_times(results_csv: Path):
    """
    Read gpu_maxflow_ms from one experiment's results.csv.

    Returns:
        list of GPU runtimes in milliseconds
    """

    times = []

    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "gpu_maxflow_ms" not in reader.fieldnames:
            raise ValueError(
                f"{results_csv} does not contain gpu_maxflow_ms.\n"
                f"Columns found: {reader.fieldnames}"
            )

        for row in reader:
            value = row.get("gpu_maxflow_ms")

            if value is None or value == "":
                continue

            times.append(float(value))

    return times


def load_results(root: Path):
    """
    Scan directories such as:

        jobs_100_intervals_10_sparse
        jobs_100_intervals_25_light
        jobs_100_intervals_50_medium
        jobs_100_intervals_100_dense
        jobs_100_intervals_200_very_dense

    Returns:

        data[level][jobs] = {
            "intervals": ...,
            "mean_ms": ...,
            "median_ms": ...,
            "count": ...
        }
    """

    data = {
        level: {}
        for level in LEVEL_ORDER
    }

    for folder in root.iterdir():

        if not folder.is_dir():
            continue

        match = DIR_PATTERN.match(folder.name)

        if match is None:
            continue

        jobs = int(match.group(1))
        intervals = int(match.group(2))
        level = match.group(3)

        results_csv = folder / "results.csv"

        if not results_csv.exists():
            print(
                f"Skipping {folder.name}: "
                f"results.csv not found"
            )
            continue

        times = read_gpu_times(results_csv)

        if not times:
            print(
                f"Skipping {folder.name}: "
                f"no gpu_maxflow_ms values"
            )
            continue

        sorted_times = sorted(times)

        n = len(sorted_times)

        if n % 2 == 1:
            median_ms = sorted_times[n // 2]
        else:
            median_ms = (
                sorted_times[n // 2 - 1]
                + sorted_times[n // 2]
            ) / 2.0

        mean_ms = sum(times) / len(times)

        data[level][jobs] = {
            "intervals": intervals,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "count": len(times),
        }

    return data


def print_table(data, metric):
    print()
    print("=" * 90)
    print("GPU MAX-FLOW + PASSES SCALING RESULTS")
    print("=" * 90)

    all_jobs = sorted(
        {
            jobs
            for level in LEVEL_ORDER
            for jobs in data[level]
        }
    )

    print(
        f"{'Jobs':>8} "
        f"{'Sparse':>14} "
        f"{'Light':>14} "
        f"{'Medium':>14} "
        f"{'Dense':>14} "
        f"{'Very dense':>14}"
    )

    print("-" * 90)

    for jobs in all_jobs:

        values = []

        for level in LEVEL_ORDER:

            if jobs in data[level]:
                value = data[level][jobs][metric]
                values.append(f"{value:14.6f}")
            else:
                values.append(f"{'-':>14}")

        print(
            f"{jobs:8d} "
            + " ".join(values)
        )

    print()
    print("All times are milliseconds.")
    print()


def plot_results(
    data,
    metric,
    output_file,
    log_x=False,
    log_y=False,
):
    plt.figure(figsize=(11, 7))

    for level in LEVEL_ORDER:

        points = data[level]

        if not points:
            continue

        jobs = sorted(points.keys())

        times = [
            points[j][metric]
            for j in jobs
        ]

        plt.plot(
            jobs,
            times,
            marker="o",
            linewidth=2,
            markersize=6,
            label=LEVEL_LABELS[level],
        )

    plt.xlabel(
        "Number of Jobs",
        fontsize=13,
    )

    if metric == "mean_ms":
        plt.ylabel(
            "Mean GPU Max-Flow Runtime (ms)",
            fontsize=13,
        )

        plt.title(
            "GPU Max-Flow + Passes Runtime vs. Number of Jobs",
            fontsize=15,
        )

    else:
        plt.ylabel(
            "Median GPU Max-Flow Runtime (ms)",
            fontsize=13,
        )

        plt.title(
            "GPU Max-Flow + Passes Runtime vs. Number of Jobs",
            fontsize=15,
        )

    if log_x:
        plt.xscale("log")

    if log_y:
        plt.yscale("log")

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.legend(
        title="Original Energy Intervals",
        fontsize=10,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved plot to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="job_interval_scaling_results_gpu_ecl",
        help=(
            "Root directory containing "
            "jobs_*_intervals_* experiment folders."
        ),
    )

    parser.add_argument(
        "--metric",
        choices=[
            "mean",
            "median",
        ],
        default="mean",
        help=(
            "Aggregate runtime across the 30 instances. "
            "Default: mean."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="gpu_runtime_vs_jobs.png",
    )

    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic x-axis.",
    )

    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Result directory does not exist: {root}"
        )

    metric = (
        "mean_ms"
        if args.metric == "mean"
        else "median_ms"
    )

    data = load_results(root)

    total_configs = sum(
        len(data[level])
        for level in LEVEL_ORDER
    )

    if total_configs == 0:
        raise RuntimeError(
            f"No experiment results found under {root}"
        )

    print(f"Found {total_configs} experiment configurations.")

    print_table(
        data,
        metric,
    )

    plot_results(
        data=data,
        metric=metric,
        output_file=args.output,
        log_x=args.log_x,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()