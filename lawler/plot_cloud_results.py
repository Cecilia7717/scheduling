
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(df, group_cols, value_col):
    return (
        df.groupby(group_cols)[value_col]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def plot_with_iqr(ax, x, median, q25, q75, label=None):
    ax.plot(
        x,
        median,
        marker="o",
        linewidth=2,
        markersize=6,
        label=label,
    )

    ax.fill_between(
        x,
        q25,
        q75,
        alpha=0.2,
    )


def save_figure(fig, output_dir, filename):
    fig.tight_layout()

    fig.savefig(
        output_dir / f"{filename}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_cloud_jobs(df, output_dir):
    data = df[df["weight_scale_alpha"] == 1]

    stats = summarize(
        data,
        ["num_original_jobs"],
        "num_cloud_jobs",
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    plot_with_iqr(
        ax,
        stats["num_original_jobs"],
        stats["median"],
        stats["q25"],
        stats["q75"],
    )

    ax.set_xlabel("Number of Original Jobs")
    ax.set_ylabel("Number of Cloud Jobs")
    ax.set_title("Cloud Outsourcing vs. Job Count")
    ax.grid(alpha=0.3)

    save_figure(
        fig,
        output_dir,
        "cloud_jobs_vs_jobs",
    )


def plot_cloud_ratio(df, output_dir):
    data = df[df["weight_scale_alpha"] == 1].copy()

    data["cloud_percentage"] = (
        data["cloud_job_ratio"] * 100
    )

    stats = summarize(
        data,
        ["num_original_jobs"],
        "cloud_percentage",
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    plot_with_iqr(
        ax,
        stats["num_original_jobs"],
        stats["median"],
        stats["q25"],
        stats["q75"],
    )

    ax.set_xlabel("Number of Original Jobs")
    ax.set_ylabel("Jobs Outsourced (%)")
    ax.set_title("Cloud Outsourcing Ratio")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    save_figure(
        fig,
        output_dir,
        "cloud_ratio_vs_jobs",
    )


def plot_cloud_cost(df, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))

    for alpha in sorted(
        df["weight_scale_alpha"].unique()
    ):
        data = df[
            df["weight_scale_alpha"] == alpha
        ]

        stats = summarize(
            data,
            ["num_original_jobs"],
            "cloud_outsourcing_cost",
        )

        plot_with_iqr(
            ax,
            stats["num_original_jobs"],
            stats["median"],
            stats["q25"],
            stats["q75"],
            label=rf"$\alpha={alpha}$",
        )

    ax.set_xlabel("Number of Original Jobs")
    ax.set_ylabel("Cloud Outsourcing Cost")
    ax.set_title("Cloud Cost Under Weight Scaling")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(
        fig,
        output_dir,
        "cloud_cost_vs_jobs",
    )


def plot_cloud_vs_runtime(df, output_dir):
    data = df[df["weight_scale_alpha"] == 1]

    stats = summarize(
        data,
        ["num_cloud_jobs"],
        "lawler_runtime_seconds",
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    plot_with_iqr(
        ax,
        stats["num_cloud_jobs"],
        stats["median"],
        stats["q25"],
        stats["q75"],
    )

    ax.set_xlabel("Number of Cloud Jobs")
    ax.set_ylabel("Lawler Runtime (seconds)")
    ax.set_title("Cloud Jobs vs. Lawler Runtime")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    save_figure(
        fig,
        output_dir,
        "cloud_jobs_vs_runtime",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="lawler_cloud_results/results.csv",
    )

    parser.add_argument(
        "--output",
        default="lawler_cloud_results/figures",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = [
        "num_original_jobs",
        "weight_scale_alpha",
        "num_cloud_jobs",
        "cloud_job_ratio",
        "cloud_outsourcing_cost",
        "lawler_runtime_seconds",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    plot_cloud_jobs(df, output_dir)
    plot_cloud_ratio(df, output_dir)
    plot_cloud_cost(df, output_dir)
    plot_cloud_vs_runtime(df, output_dir)

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
