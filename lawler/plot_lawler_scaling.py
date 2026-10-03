#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
INPUT = HERE / "lawler_scaling_results" / "results.csv"
OUTPUT_DIR = HERE / "lawler_scaling_results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot settings
# ============================================================

DENSITY_ORDER = ["sparse", "light", "medium", "dense"]

MARKERS = {
    "sparse": "o",
    "light": "s",
    "medium": "^",
    "dense": "D",
}


def save_figure(fig, name):
    pdf = OUTPUT_DIR / f"{name}.pdf"
    png = OUTPUT_DIR / f"{name}.png"

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")

    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


# ============================================================
# 1. Main paper plot:
#    Lawler runtime vs original number of jobs
# ============================================================

def plot_runtime_vs_jobs(df):

    summary = (
        df.groupby(["num_original_jobs", "density"])
        .agg(
            median_runtime=("lawler_runtime_seconds", "median"),
            mean_runtime=("lawler_runtime_seconds", "mean"),
            q25=("lawler_runtime_seconds", lambda x: x.quantile(0.25)),
            q75=("lawler_runtime_seconds", lambda x: x.quantile(0.75)),
            num_instances=("lawler_runtime_seconds", "count"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    for density in DENSITY_ORDER:
        d = summary[summary["density"] == density].sort_values(
            "num_original_jobs"
        )

        if d.empty:
            continue

        ax.plot(
            d["num_original_jobs"],
            d["median_runtime"],
            marker=MARKERS[density],
            linewidth=1.8,
            markersize=6,
            label=density.replace("_", " ").title(),
        )

        # Interquartile range across instances
        ax.fill_between(
            d["num_original_jobs"],
            d["q25"],
            d["q75"],
            alpha=0.15,
        )

    ax.set_xlabel("Number of original jobs")
    ax.set_ylabel("Lawler runtime (seconds)")
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Interval density", frameon=False)

    fig.tight_layout()
    save_figure(fig, "lawler_runtime_vs_jobs")
    plt.close(fig)

    return summary


# ============================================================
# 2. Runtime vs number of jobs AFTER reduction
# ============================================================

def plot_runtime_vs_penalty_jobs(df):

    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    for density in DENSITY_ORDER:
        d = df[df["density"] == density]

        if d.empty:
            continue

        ax.scatter(
            d["num_penalty_jobs"],
            d["lawler_runtime_seconds"],
            marker=MARKERS[density],
            s=28,
            alpha=0.65,
            label=density.replace("_", " ").title(),
        )

    ax.set_xlabel("Number of jobs after reduction, $N$")
    ax.set_ylabel("Lawler runtime (seconds)")
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Interval density", frameon=False)

    fig.tight_layout()
    save_figure(fig, "lawler_runtime_vs_penalty_jobs")
    plt.close(fig)


# ============================================================
# 3. Runtime vs k
# ============================================================

def plot_runtime_vs_k(df):

    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    for density in DENSITY_ORDER:
        d = df[df["density"] == density]

        if d.empty:
            continue

        ax.scatter(
            d["distinct_release_dates_k"],
            d["lawler_runtime_seconds"],
            marker=MARKERS[density],
            s=28,
            alpha=0.65,
            label=density.replace("_", " ").title(),
        )

    ax.set_xlabel("Number of distinct release dates, $k$")
    ax.set_ylabel("Lawler runtime (seconds)")
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Interval density", frameon=False)

    fig.tight_layout()
    save_figure(fig, "lawler_runtime_vs_k")
    plt.close(fig)


# ============================================================
# 4. Runtime vs W
# ============================================================

def plot_runtime_vs_W(df):

    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    for density in DENSITY_ORDER:
        d = df[df["density"] == density]

        if d.empty:
            continue

        ax.scatter(
            d["total_weight_W"],
            d["lawler_runtime_seconds"],
            marker=MARKERS[density],
            s=28,
            alpha=0.65,
            label=density.replace("_", " ").title(),
        )

    ax.set_xlabel("Total weight, $W$")
    ax.set_ylabel("Lawler runtime (seconds)")
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Interval density", frameon=False)

    fig.tight_layout()
    save_figure(fig, "lawler_runtime_vs_W")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print(f"Reading: {INPUT}")

    df = pd.read_csv(INPUT)

    print(f"Rows: {len(df)}")
    print(f"Original job counts: {sorted(df['num_original_jobs'].unique())}")
    print(f"Densities: {list(df['density'].unique())}")

    # Remove incomplete rows if the experiment is still running.
    df = df.dropna(
        subset=[
            "num_original_jobs",
            "density",
            "lawler_runtime_seconds",
        ]
    )

    summary = plot_runtime_vs_jobs(df)

    plot_runtime_vs_penalty_jobs(df)
    plot_runtime_vs_k(df)
    plot_runtime_vs_W(df)

    # Save aggregated values used in the main figure.
    summary_file = OUTPUT_DIR / "runtime_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved: {summary_file}")
    print("Done.")


if __name__ == "__main__":
    main()