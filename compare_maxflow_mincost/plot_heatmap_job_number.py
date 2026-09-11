#!/usr/bin/env python3
"""
Draw a heatmap of runtime ratio:

    Max Flow + Passes runtime / Pure Min-Cost Flow runtime

Expected directory structure, for example:

job_interval_scaling_results/
├── jobs_10_intervals_3_light/
│   └── summary.txt
├── jobs_10_intervals_5_medium/
│   └── summary.txt
├── jobs_20_intervals_3_sparse/
│   └── summary.txt
└── ...

The script reads the "Mean runtime ratio" from each summary.txt.

It also:
    1. Prints a raw data table before plotting.
    2. Saves the raw table as CSV.
    3. Produces the heatmap.

Usage:
    python plot_runtime_ratio_heatmap.py

    python plot_runtime_ratio_heatmap.py \
        --root job_interval_scaling_results

    python plot_runtime_ratio_heatmap.py \
        --root job_interval_scaling_results \
        --output runtime_ratio_heatmap.png

    python plot_runtime_ratio_heatmap.py \
        --root job_interval_scaling_results \
        --table-output runtime_ratio_table.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Folder / summary parsing
# ============================================================

FOLDER_RE = re.compile(
    r"^jobs_(?P<jobs>\d+)_intervals_(?P<intervals>\d+)(?:_(?P<label>.+))?$"
)

RATIO_RE = re.compile(
    r"Max Flow \+ Passes:\s*"
    r"([0-9]*\.?[0-9]+)\s*ms?",
    re.IGNORECASE,
)


# ============================================================
# Read one summary
# ============================================================

def read_ratio(summary_file: Path) -> float:
    text = summary_file.read_text(encoding="utf-8")

    match = RATIO_RE.search(text)

    if not match:
        raise ValueError(
            f"Could not find runtime ratio in {summary_file}\n"
            'Expected something like: "0.666314x" after '
            '"Mean runtime ratio '
            '(Max Flow + Passes / Pure Min-Cost Flow):"'
        )

    return float(match.group(1))


# ============================================================
# Collect all experiment results
# ============================================================

def collect_results(root: Path):
    results = []

    for folder in sorted(root.iterdir()):

        if not folder.is_dir():
            continue

        match = FOLDER_RE.match(folder.name)

        if not match:
            continue

        summary_file = folder / "summary.txt"

        if not summary_file.exists():
            print(
                f"Skipping {folder.name}: "
                f"no summary.txt"
            )
            continue

        jobs = int(match.group("jobs"))
        intervals = int(match.group("intervals"))
        label = match.group("label") or ""

        try:
            ratio = read_ratio(summary_file)

        except ValueError as exc:
            print(
                f"Skipping {folder.name}: {exc}"
            )
            continue

        results.append(
            {
                "jobs": jobs,
                "intervals": intervals,
                "label": label,
                "ratio": ratio,
                "folder": folder.name,
            }
        )

    if not results:
        raise RuntimeError(
            f"No usable experiment folders found under "
            f"{root.resolve()}"
        )

    return results


# ============================================================
# Print raw data table
# ============================================================

def print_raw_table(results):
    """
    Print one row for every experiment configuration.
    """

    sorted_results = sorted(
        results,
        key=lambda x: (
            x["jobs"],
            x["intervals"],
        ),
    )

    print()
    print("=" * 80)
    print("RAW RUNTIME RATIO DATA")
    print("=" * 80)

    print(
        f"{'Jobs':>10} "
        f"{'Intervals':>12} "
        f"{'Level':>15} "
        f"{'Runtime Ratio':>18}"
    )

    print("-" * 80)

    for r in sorted_results:
        print(
            f"{r['jobs']:>10d} "
            f"{r['intervals']:>12d} "
            f"{r['label']:>15} "
            f"{r['ratio']:>17.6f}x"
        )

    print("=" * 80)

    print()
    print(
        "Runtime ratio = "
        "Max Flow + Passes / Pure Min-Cost Flow"
    )

    print(
        "ratio < 1 : Max Flow + Passes is faster"
    )

    print(
        "ratio > 1 : Pure Min-Cost Flow is faster"
    )

    print()


# ============================================================
# Print matrix-style table
# ============================================================

def print_matrix_table(results):
    """
    Print a table where:

        rows    = number of jobs
        columns = interval level

    This is often easier to compare with the plot.
    """

    labels = sorted(
        {
            r["label"]
            for r in results
        }
    )

    jobs = sorted(
        {
            r["jobs"]
            for r in results
        }
    )

    lookup = {
        (
            r["jobs"],
            r["label"],
        ): r["ratio"]
        for r in results
    }

    print()
    print("=" * 100)
    print("RUNTIME RATIO BY JOB COUNT AND INTERVAL LEVEL")
    print("=" * 100)

    header = f"{'Jobs':>10}"

    for label in labels:
        header += f"{label:>18}"

    print(header)

    print("-" * max(100, len(header)))

    for job_count in jobs:

        row = f"{job_count:>10d}"

        for label in labels:

            value = lookup.get(
                (
                    job_count,
                    label,
                )
            )

            if value is None:
                row += f"{'-':>18}"

            else:
                row += f"{value:>17.6f}x"

        print(row)

    print("=" * max(100, len(header)))
    print()


# ============================================================
# Save raw data table to CSV
# ============================================================

def save_table_csv(
    results,
    output: Path,
):
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["jobs"],
            x["intervals"],
        ),
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "jobs",
                "intervals",
                "level",
                "runtime_ratio",
                "folder",
            ],
        )

        writer.writeheader()

        for r in sorted_results:
            writer.writerow(
                {
                    "jobs": r["jobs"],
                    "intervals": r["intervals"],
                    "level": r["label"],
                    "runtime_ratio": r["ratio"],
                    "folder": r["folder"],
                }
            )

    print(
        f"Saved raw data table to: "
        f"{output.resolve()}"
    )


# ============================================================
# Heatmap
# ============================================================

def make_heatmap(
    results,
    output: Path,
    show: bool,
):

    job_values = sorted(
        {
            r["jobs"]
            for r in results
        }
    )

    interval_values = sorted(
        {
            r["intervals"]
            for r in results
        }
    )

    job_to_row = {
        v: i
        for i, v in enumerate(job_values)
    }

    interval_to_col = {
        v: i
        for i, v in enumerate(interval_values)
    }

    matrix = np.full(
        (
            len(job_values),
            len(interval_values),
        ),
        np.nan,
    )

    labels = {}

    for r in results:

        row = job_to_row[
            r["jobs"]
        ]

        col = interval_to_col[
            r["intervals"]
        ]

        matrix[row, col] = r["ratio"]

        labels[
            (
                row,
                col,
            )
        ] = r["label"]

    masked = np.ma.masked_invalid(
        matrix
    )

    fig_width = max(
        8,
        0.9 * len(interval_values) + 3,
    )

    fig_height = max(
        5,
        0.7 * len(job_values) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            fig_height,
        )
    )


    # ========================================================
    # Center color scale around 1
    #
    # < 1 => Max Flow + Passes faster
    # > 1 => Pure Min-Cost Flow faster
    # ========================================================

    finite = matrix[
        np.isfinite(matrix)
    ]

    max_distance = max(
        abs(
            float(finite.min()) - 1.0
        ),
        abs(
            float(finite.max()) - 1.0
        ),
        0.05,
    )

    im = ax.imshow(
        masked,
        aspect="auto",
        vmin=1.0 - max_distance,
        vmax=1.0 + max_distance,
    )


    # ========================================================
    # Axis labels
    # ========================================================

    ax.set_xticks(
        np.arange(
            len(interval_values)
        )
    )

    ax.set_xticklabels(
        interval_values
    )

    ax.set_yticks(
        np.arange(
            len(job_values)
        )
    )

    ax.set_yticklabels(
        job_values
    )

    ax.set_xlabel(
        "Number of Energy Intervals"
    )

    ax.set_ylabel(
        "Number of Jobs"
    )

    ax.set_title(
        "Runtime Ratio: "
        "Max Flow + Passes / Pure Min-Cost Flow"
    )


    # ========================================================
    # Put numerical value into each cell
    # ========================================================

    for row in range(
        len(job_values)
    ):

        for col in range(
            len(interval_values)
        ):

            value = matrix[
                row,
                col,
            ]

            if np.isnan(value):
                continue

            text = (
                f"{value:.3f}"
            )

            label = labels.get(
                (
                    row,
                    col,
                ),
                "",
            )

            if label:
                text += (
                    f"\n{label}"
                )

            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
            )


    # ========================================================
    # Grid
    # ========================================================

    ax.set_xticks(
        np.arange(
            -0.5,
            len(interval_values),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(job_values),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.6,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    # ========================================================
    # Color bar
    # ========================================================

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Runtime ratio\n"
        "< 1: Max Flow + Passes faster     "
        "> 1: Pure Min-Cost Flow faster"
    )


    # ========================================================
    # Save
    # ========================================================

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved heatmap to: "
        f"{output.resolve()}"
    )

    if show:
        plt.show()

    else:
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "job_interval_scaling_results"
        ),
        help=(
            "Root folder containing "
            "jobs_*_intervals_* experiment folders."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime_ratio_heatmap.png"
        ),
        help=(
            "Output heatmap image filename."
        ),
    )

    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path(
            "runtime_ratio_table.csv"
        ),
        help=(
            "Output CSV containing the raw "
            "runtime-ratio data."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Also display the heatmap interactively."
        ),
    )

    args = parser.parse_args()


    if not args.root.exists():

        raise FileNotFoundError(
            f"Root directory not found: "
            f"{args.root}"
        )


    # ========================================================
    # Read data
    # ========================================================

    results = collect_results(
        args.root
    )


    print()
    print(
        f"Found {len(results)} "
        f"experiment folders."
    )


    # ========================================================
    # 1. Print raw numerical data
    # ========================================================

    print_raw_table(
        results
    )


    # ========================================================
    # 2. Print matrix-style table
    # ========================================================

    print_matrix_table(
        results
    )


    # ========================================================
    # 3. Save table
    # ========================================================

    save_table_csv(
        results,
        args.table_output,
    )


    # ========================================================
    # 4. Plot heatmap
    # ========================================================

    make_heatmap(
        results,
        args.output,
        args.show,
    )


if __name__ == "__main__":
    main()