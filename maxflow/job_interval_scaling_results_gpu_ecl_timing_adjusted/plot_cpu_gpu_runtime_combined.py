#!/usr/bin/env python3

import argparse
import html
from io import StringIO
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DENSITY_ORDER = ["sparse", "light", "medium", "dense", "very_dense"]

DENSITY_LABELS = {
    "sparse": "Sparse",
    "light": "Light",
    "medium": "Medium",
    "dense": "Dense",
    "very_dense": "Very Dense",
}


def load_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read the timing summary CSV.

    This also cleans formatting artifacts such as:
      **median_gpu_maxflow_seconds**
      &#x6D;ean_gpu_wall_seconds
      &#x30;.715...
    which can appear when copying CSV data through Markdown/HTML.
    """
    text = path.read_text(encoding="utf-8")

    # Decode HTML entities, e.g. &#x6D; -> m, &#x30; -> 0.
    text = html.unescape(text)

    # Remove Markdown bold markers.
    text = text.replace("**", "")

    df = pd.read_csv(StringIO(text))

    # Clean column names.
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "num_jobs",
        "density",
        "mean_cpu_wall_seconds",
        "mean_gpu_maxflow_seconds",
        "mean_gpu_wall_seconds",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n  "
            + "\n  ".join(missing)
            + "\n\nColumns found:\n  "
            + "\n  ".join(df.columns)
        )

    # Ensure numeric columns are numeric.
    numeric_cols = [
        "num_jobs",
        "num_intervals",
        "num_instances",
        "mean_cpu_wall_seconds",
        "median_cpu_wall_seconds",
        "mean_gpu_maxflow_seconds",
        "median_gpu_maxflow_seconds",
        "mean_gpu_wall_seconds",
        "median_gpu_wall_seconds",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["density"] = df["density"].astype(str).str.strip()

    return df


def plot_runtime_panels(df: pd.DataFrame, output_pdf: Path, output_png: Path):
    """
    Plot CPU wall time, GPU algorithm time, and GPU full wall time
    against number of jobs, with one panel per interval density.
    """

    densities = [d for d in DENSITY_ORDER if d in set(df["density"])]

    if not densities:
        raise ValueError("No recognized density values found.")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.3),
        sharex=False,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, density in zip(axes, densities):
        sub = df[df["density"] == density].copy()
        sub = sub.sort_values("num_jobs")

        ax.plot(
            sub["num_jobs"],
            sub["mean_cpu_wall_seconds"],
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            label="CPU",
        )

        ax.plot(
            sub["num_jobs"],
            sub["mean_gpu_maxflow_seconds"],
            marker="s",
            linewidth=1.7,
            markersize=4.5,
            label="GPU algorithm",
        )

        ax.plot(
            sub["num_jobs"],
            sub["mean_gpu_wall_seconds"],
            marker="^",
            linewidth=1.7,
            markersize=4.5,
            label="GPU wall",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(DENSITY_LABELS.get(density, density.replace("_", " ").title()))
        ax.set_xlabel("Number of jobs")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

    # Remove unused panel(s).
    for ax in axes[len(densities):]:
        fig.delaxes(ax)

    # Y label only on left-side panels.
    for i, ax in enumerate(fig.axes):
        if i % 3 == 0:
            ax.set_ylabel("Mean runtime (seconds)")

    # Shared legend.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot CPU vs GPU MaxFlow-Passes runtime from the timing summary CSV."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("cpu_gpu_timing_summary.csv"),
        help="Input CSV file (default: cpu_gpu_timing_summary.csv)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gpu_cpu_runtime_vs_jobs.pdf"),
        help="Output PDF path (default: gpu_cpu_runtime_vs_jobs.pdf)",
    )

    args = parser.parse_args()

    df = load_csv_robust(args.csv)

    output_pdf = args.output
    output_png = output_pdf.with_suffix(".png")

    plot_runtime_panels(
        df,
        output_pdf=output_pdf,
        output_png=output_png,
    )

    print(f"Saved PDF: {output_pdf}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()
