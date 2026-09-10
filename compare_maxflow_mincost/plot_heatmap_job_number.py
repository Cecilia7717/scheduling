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

Usage:
    python plot_runtime_ratio_heatmap.py
    python plot_runtime_ratio_heatmap.py --root job_interval_scaling_results
    python plot_runtime_ratio_heatmap.py --root job_interval_scaling_results --output runtime_ratio_heatmap.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FOLDER_RE = re.compile(
    r"^jobs_(?P<jobs>\d+)_intervals_(?P<intervals>\d+)(?:_(?P<label>.+))?$"
)

RATIO_RE = re.compile(
    r"Mean runtime ratio\s*\(Max Flow \+ Passes / Pure Min-Cost Flow\):\s*"
    r"([0-9]*\.?[0-9]+)\s*x?",
    re.IGNORECASE,
)


def read_ratio(summary_file: Path) -> float:
    text = summary_file.read_text(encoding="utf-8")
    match = RATIO_RE.search(text)
    if not match:
        raise ValueError(
            f"Could not find runtime ratio in {summary_file}\n"
            'Expected something like: "0.666314x" after '
            '"Mean runtime ratio (Max Flow + Passes / Pure Min-Cost Flow):"'
        )
    return float(match.group(1))


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
            print(f"Skipping {folder.name}: no summary.txt")
            continue

        jobs = int(match.group("jobs"))
        intervals = int(match.group("intervals"))
        label = match.group("label") or ""

        try:
            ratio = read_ratio(summary_file)
        except ValueError as exc:
            print(f"Skipping {folder.name}: {exc}")
            continue

        results.append(
            {
                "jobs": jobs,
                "intervals": intervals,
                "label": label,
                "ratio": ratio,
            }
        )

    if not results:
        raise RuntimeError(
            f"No usable experiment folders found under {root.resolve()}"
        )

    return results


def make_heatmap(results, output: Path, show: bool):
    job_values = sorted({r["jobs"] for r in results})
    interval_values = sorted({r["intervals"] for r in results})

    job_to_row = {v: i for i, v in enumerate(job_values)}
    interval_to_col = {v: i for i, v in enumerate(interval_values)}

    matrix = np.full((len(job_values), len(interval_values)), np.nan)

    labels = {}
    for r in results:
        row = job_to_row[r["jobs"]]
        col = interval_to_col[r["intervals"]]
        matrix[row, col] = r["ratio"]
        labels[(row, col)] = r["label"]

    masked = np.ma.masked_invalid(matrix)

    fig_width = max(8, 0.9 * len(interval_values) + 3)
    fig_height = max(5, 0.7 * len(job_values) + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Center the color scale around ratio = 1 whenever possible:
    # < 1 => Max Flow + Passes is faster
    # > 1 => Pure Min-Cost Flow is faster
    finite = matrix[np.isfinite(matrix)]
    max_distance = max(
        abs(float(finite.min()) - 1.0),
        abs(float(finite.max()) - 1.0),
        0.05,
    )

    im = ax.imshow(
        masked,
        aspect="auto",
        vmin=1.0 - max_distance,
        vmax=1.0 + max_distance,
    )

    ax.set_xticks(np.arange(len(interval_values)))
    ax.set_xticklabels(interval_values)
    ax.set_yticks(np.arange(len(job_values)))
    ax.set_yticklabels(job_values)

    ax.set_xlabel("Number of energy intervals")
    ax.set_ylabel("Number of jobs")
    ax.set_title(
        "Runtime Ratio: Max Flow + Passes / Pure Min-Cost Flow"
    )

    # Put the numerical ratio in every available cell.
    for row in range(len(job_values)):
        for col in range(len(interval_values)):
            value = matrix[row, col]
            if np.isnan(value):
                continue

            text = f"{value:.3f}x"
            label = labels.get((row, col), "")
            if label:
                text += f"\n{label}"

            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
            )

    # Grid lines between cells.
    ax.set_xticks(np.arange(-0.5, len(interval_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(job_values), 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Runtime ratio\n"
        "< 1: Max Flow + Passes faster     "
        "> 1: Pure Min-Cost Flow faster"
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to: {output.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("job_interval_scaling_results"),
        help="Root folder containing jobs_*_intervals_* experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_ratio_heatmap.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the heatmap interactively.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Root directory not found: {args.root}")

    results = collect_results(args.root)

    print(f"Found {len(results)} experiment folders.")
    for r in sorted(results, key=lambda x: (x["jobs"], x["intervals"])):
        print(
            f"jobs={r['jobs']:>4}, intervals={r['intervals']:>4}, "
            f"ratio={r['ratio']:.6f}x, label={r['label']}"
        )

    make_heatmap(results, args.output, args.show)


if __name__ == "__main__":
    main()
