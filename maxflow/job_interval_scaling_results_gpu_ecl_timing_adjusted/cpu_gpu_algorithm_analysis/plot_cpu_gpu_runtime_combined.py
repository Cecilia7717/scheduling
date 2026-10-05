#!/usr/bin/env python3

import argparse
import html
from io import StringIO
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DENSITY_ORDER = [
    "sparse",
    "light",
    "medium",
    "dense",
    "very_dense",
]

DENSITY_LABELS = {
    "sparse": "Sparse",
    "light": "Light",
    "medium": "Medium",
    "dense": "Dense",
    "very_dense": "Very Dense",
}


# ============================================================
# Load CSV
# ============================================================

def load_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read cpu_gpu_algorithms_summary.csv.

    Expected timing columns:
        mean_passes_seconds
        mean_restart_seconds
        mean_mincost_seconds
        mean_gpu_algorithm_seconds
        mean_gpu_wall_seconds
    """

    text = path.read_text(encoding="utf-8")

    # Decode possible HTML entities.
    text = html.unescape(text)

    # Remove possible Markdown bold markers.
    text = text.replace("**", "")

    df = pd.read_csv(StringIO(text))

    # Clean column names.
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    required = [
        "num_jobs",
        "density",
        "mean_passes_seconds",
        "mean_restart_seconds",
        "mean_mincost_seconds",
        "mean_gpu_algorithm_seconds",
        "mean_gpu_wall_seconds",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n  "
            + "\n  ".join(missing)
            + "\n\nColumns found:\n  "
            + "\n  ".join(df.columns)
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_cols = [
        "num_jobs",
        "num_original_intervals",
        "num_instances",
        "mean_atomic_intervals",
        "mean_job_interval_edges",

        "mean_passes_seconds",
        "median_passes_seconds",

        "mean_restart_seconds",
        "median_restart_seconds",

        "mean_mincost_seconds",
        "median_mincost_seconds",

        "mean_gpu_algorithm_seconds",
        "median_gpu_algorithm_seconds",

        "mean_gpu_wall_seconds",
        "median_gpu_wall_seconds",

        "mean_restart_over_passes",
        "mean_mincost_over_passes",
        "mean_gpu_algorithm_over_passes",
        "mean_gpu_wall_over_passes",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["density"] = (
        df["density"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# Plot
# ============================================================

def plot_runtime_panels(
    df: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
):
    """
    Plot all CPU and GPU algorithm runtimes against number of jobs.

    One panel per original interval density.

    Curves:
        CPU MaxFlow-Passes
        CPU MaxFlow-Restart
        CPU MinCostFlow
        GPU MaxFlow-Passes algorithm time
        GPU MaxFlow-Passes wall time
    """

    densities = [
        d
        for d in DENSITY_ORDER
        if d in set(df["density"])
    ]

    if not densities:
        raise ValueError(
            "No recognized density values found."
        )

    # --------------------------------------------------------
    # 2 x 3 layout
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.5, 6.6),
        sharex=False,
        sharey=True,
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # Plot one density per panel
    # --------------------------------------------------------

    for ax, density in zip(
        axes,
        densities,
    ):

        sub = (
            df[
                df["density"] == density
            ]
            .copy()
            .sort_values("num_jobs")
        )

        # ====================================================
        # CPU MaxFlow-Passes
        # ====================================================

        ax.plot(
            sub["num_jobs"],
            sub["mean_passes_seconds"],
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            label="CPU MaxFlow-Passes",
        )

        # ====================================================
        # CPU MaxFlow-Restart
        # ====================================================

        ax.plot(
            sub["num_jobs"],
            sub["mean_restart_seconds"],
            marker="s",
            linewidth=1.7,
            markersize=4.5,
            label="CPU MaxFlow-Restart",
        )

        # ====================================================
        # CPU MinCostFlow
        # ====================================================

        ax.plot(
            sub["num_jobs"],
            sub["mean_mincost_seconds"],
            marker="^",
            linewidth=1.7,
            markersize=4.5,
            label="CPU MinCostFlow",
        )

        # ====================================================
        # GPU algorithm time
        # ====================================================

        ax.plot(
            sub["num_jobs"],
            sub["mean_gpu_algorithm_seconds"],
            marker="D",
            linewidth=1.7,
            markersize=4.5,
            label="GPU MaxFlow-Passes (algorithm)",
        )

        # ====================================================
        # GPU wall time
        # ====================================================

        ax.plot(
            sub["num_jobs"],
            sub["mean_gpu_wall_seconds"],
            marker="v",
            linewidth=1.7,
            markersize=4.5,
            linestyle="--",
            label="GPU MaxFlow-Passes (wall)",
        )

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(
            DENSITY_LABELS.get(
                density,
                density.replace(
                    "_",
                    " ",
                ).title(),
            )
        )

        ax.set_xlabel(
            "Number of jobs"
        )

        ax.grid(
            True,
            which="both",
            linestyle=":",
            linewidth=0.6,
            alpha=0.6,
        )

    # --------------------------------------------------------
    # Remove unused panel
    # --------------------------------------------------------

    for ax in axes[len(densities):]:
        fig.delaxes(ax)

    # --------------------------------------------------------
    # Y-axis labels
    # --------------------------------------------------------

    for i, ax in enumerate(fig.axes):
        if i % 3 == 0:
            ax.set_ylabel(
                "Mean runtime (seconds)"
            )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles, labels = (
        fig.axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.91)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot CPU MaxFlow-Passes, MaxFlow-Restart, MinCostFlow, "
            "GPU algorithm time, and GPU wall time."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path(
            "cpu_gpu_algorithms_summary.csv"
        ),
        help=(
            "Input CSV file "
            "(default: cpu_gpu_algorithms_summary.csv)"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "cpu_gpu_all_runtime_vs_jobs.pdf"
        ),
        help=(
            "Output PDF path "
            "(default: cpu_gpu_all_runtime_vs_jobs.pdf)"
        ),
    )

    args = parser.parse_args()

    df = load_csv_robust(
        args.csv
    )

    output_pdf = args.output

    output_png = (
        output_pdf
        .with_suffix(".png")
    )

    plot_runtime_panels(
        df,
        output_pdf=output_pdf,
        output_png=output_png,
    )

    print(
        f"Saved PDF: {output_pdf}"
    )

    print(
        f"Saved PNG: {output_png}"
    )


if __name__ == "__main__":
    main()