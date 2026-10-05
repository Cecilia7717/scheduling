#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

# Good for running on a remote server without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Expected directory structure
# ============================================================
#
# experiment2_pass_mechanism/
# |
# +-- util_0.30_0.35_green_0.20_0.25/
# |   +-- results.csv
# |
# +-- util_0.30_0.35_green_0.25_0.30/
# |   +-- results.csv
# |
# ...
#
# Each results.csv is the output produced by
# compare_algorithms_exp2.py.
# ============================================================


CELL_PATTERN = re.compile(
    r"util_([0-9.]+)_([0-9.]+)_green_([0-9.]+)_([0-9.]+)"
)


# ============================================================
# Loading
# ============================================================


def parse_cell_name(path: Path):
    """
    Extract configured utilization and green-share ranges
    from a directory such as:

        util_0.60_0.65_green_0.70_0.75
    """
    m = CELL_PATTERN.fullmatch(path.name)

    if m is None:
        return None

    util_min = float(m.group(1))
    util_max = float(m.group(2))
    green_min = float(m.group(3))
    green_max = float(m.group(4))

    return {
        "util_min": util_min,
        "util_max": util_max,
        "green_min": green_min,
        "green_max": green_max,
        "util_mid": (util_min + util_max) / 2.0,
        "green_mid": (green_min + green_max) / 2.0,
        "util_label": f"{util_min:.2f}-{util_max:.2f}",
        "green_label": f"{green_min:.2f}-{green_max:.2f}",
    }


def load_all_results(root: Path) -> pd.DataFrame:
    frames = []

    csv_files = sorted(root.rglob("results.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No results.csv files found under {root}"
        )

    for csv_path in csv_files:
        info = parse_cell_name(csv_path.parent)

        if info is None:
            print(
                f"Skipping {csv_path}: directory name does not "
                f"match expected experiment format."
            )
            continue

        df = pd.read_csv(csv_path)

        for key, value in info.items():
            df[key] = value

        df["source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "Found results.csv files, but none were under "
            "recognized utilization/green-share directories."
        )

    data = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(data)} instance results.")
    print(
        f"Configurations: "
        f"{data[['util_label', 'green_label']].drop_duplicates().shape[0]}"
    )

    return data


# ============================================================
# Aggregation
# ============================================================


def aggregate_cells(data: pd.DataFrame) -> pd.DataFrame:

    group_cols = [
        "util_min",
        "util_max",
        "green_min",
        "green_max",
        "util_mid",
        "green_mid",
        "util_label",
        "green_label",
    ]

    metrics = [
        # Runtime
        "reuse_mean_ms",
        "restart_mean_ms",
        "mincost_mean_ms",
        "restart_over_reuse_mean_ratio",
        "mincost_over_reuse_mean_ratio",

        # Flow by pass
        "pass1_flow_fraction",
        "pass2_flow_fraction",
        "pass3_flow_fraction",

        # Rerouting
        "pass2_green_rerouted_fraction",
        "pass3_prior_rerouted_fraction",

        # Dinic work
        "reuse_pass1_bfs_calls",
        "reuse_pass2_bfs_calls",
        "reuse_pass3_bfs_calls",

        "reuse_pass1_augmenting_pushes",
        "reuse_pass2_augmenting_pushes",
        "reuse_pass3_augmenting_pushes",

        "reuse_pass1_bfs_edge_scans",
        "reuse_pass2_bfs_edge_scans",
        "reuse_pass3_bfs_edge_scans",

        "reuse_pass1_dfs_edge_scans",
        "reuse_pass2_dfs_edge_scans",
        "reuse_pass3_dfs_edge_scans",

        # Pass timing
        "reuse_pass1_maxflow_mean_ms",
        "reuse_pass2_maxflow_mean_ms",
        "reuse_pass3_maxflow_mean_ms",

        "reuse_pass2_transition_mean_ms",
        "reuse_pass3_transition_mean_ms",

        # MCF internals
        "mincost_shortest_path_calls",
        "mincost_augmentations",
        "mincost_edge_scans",
        "mincost_solve_mean_ms",

        # Network size
        "atomic_intervals",
        "network_vertices",
        "job_interval_edges",
        "forward_edges",

        # Actual workload characteristics
        "actual_utilization",
        "green_share",
        "brown_share",
        "red_share",
    ]

    existing = [x for x in metrics if x in data.columns]

    grouped = data.groupby(group_cols, as_index=False)

    mean_df = grouped[existing].mean()

    counts = (
        grouped.size()
        .rename(columns={"size": "instances"})
    )

    result = mean_df.merge(counts, on=group_cols)

    # Probability that red pass was actually needed.
    pass3 = (
        data.assign(
            pass3_required=(data["final_pass"] == 3).astype(float)
        )
        .groupby(group_cols, as_index=False)["pass3_required"]
        .mean()
    )

    result = result.merge(
        pass3,
        on=group_cols,
        how="left",
    )

    return result


# ============================================================
# Heatmap
# ============================================================


def make_heatmap(
    cell_df: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    output: Path,
    fmt: str = ".2f",
):

    pivot = cell_df.pivot(
        index="util_label",
        columns="green_label",
        values=value_col,
    )

    # Preserve numeric order rather than alphabetical order.
    util_order = (
        cell_df[
            ["util_label", "util_mid"]
        ]
        .drop_duplicates()
        .sort_values("util_mid")["util_label"]
        .tolist()
    )

    green_order = (
        cell_df[
            ["green_label", "green_mid"]
        ]
        .drop_duplicates()
        .sort_values("green_mid")["green_label"]
        .tolist()
    )

    pivot = pivot.reindex(
        index=util_order,
        columns=green_order,
    )

    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.5, 8.5))

    image = ax.imshow(
        values,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(green_order)))
    ax.set_xticklabels(
        green_order,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(len(util_order)))
    ax.set_yticklabels(util_order)

    ax.set_xlabel("Green energy share range")
    ax.set_ylabel("Target utilization range")
    ax.set_title(title)

    # Write values inside cells.
    finite_values = values[np.isfinite(values)]

    if len(finite_values) > 0:
        threshold = (
            np.nanmin(finite_values)
            + np.nanmax(finite_values)
        ) / 2.0
    else:
        threshold = 0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            if not np.isfinite(value):
                continue

            # No explicit colors are necessary for the figure itself;
            # use contrasting text based on the normalized background.
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Fixed-utilization extraction
# ============================================================


def select_utilization(
    cell_df: pd.DataFrame,
    util_min: float,
    util_max: float,
) -> pd.DataFrame:

    mask = (
        np.isclose(cell_df["util_min"], util_min)
        & np.isclose(cell_df["util_max"], util_max)
    )

    selected = (
        cell_df.loc[mask]
        .sort_values("green_mid")
        .copy()
    )

    if selected.empty:
        available = (
            cell_df[
                ["util_min", "util_max"]
            ]
            .drop_duplicates()
            .sort_values(["util_min", "util_max"])
        )

        raise ValueError(
            f"No data for utilization "
            f"{util_min:.2f}-{util_max:.2f}.\n"
            f"Available:\n{available.to_string(index=False)}"
        )

    return selected


# ============================================================
# Flow fractions
# ============================================================


def plot_flow_fractions(
    selected: pd.DataFrame,
    output: Path,
):

    x = np.arange(len(selected))

    p1 = 100.0 * selected["pass1_flow_fraction"].to_numpy()
    p2 = 100.0 * selected["pass2_flow_fraction"].to_numpy()
    p3 = 100.0 * selected["pass3_flow_fraction"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    ax.bar(
        x,
        p1,
        label="Green pass",
    )

    ax.bar(
        x,
        p2,
        bottom=p1,
        label="Brown pass",
    )

    ax.bar(
        x,
        p3,
        bottom=p1 + p2,
        label="Red pass",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        selected["green_label"],
        rotation=45,
        ha="right",
    )

    util_label = selected.iloc[0]["util_label"]

    ax.set_xlabel("Green energy share range")
    ax.set_ylabel("Fraction of total processing (%)")
    ax.set_ylim(0, 100)

    ax.set_title(
        "Flow Introduced by Each Max-Flow Phase\n"
        f"Target Utilization = {util_label}"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Pass runtime
# ============================================================


def plot_pass_runtime(
    selected: pd.DataFrame,
    output: Path,
):

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    x = np.arange(len(selected))

    ax.plot(
        x,
        selected["reuse_pass1_maxflow_mean_ms"],
        marker="o",
        label="Green pass",
    )

    ax.plot(
        x,
        selected["reuse_pass2_maxflow_mean_ms"],
        marker="o",
        label="Brown pass",
    )

    ax.plot(
        x,
        selected["reuse_pass3_maxflow_mean_ms"],
        marker="o",
        label="Red pass",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        selected["green_label"],
        rotation=45,
        ha="right",
    )

    util_label = selected.iloc[0]["util_label"]

    ax.set_xlabel("Green energy share range")
    ax.set_ylabel("Mean max-flow time (ms)")
    ax.set_title(
        "Max-Flow Runtime by Phase\n"
        f"Target Utilization = {util_label}"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Rerouting
# ============================================================


def plot_rerouting(
    selected: pd.DataFrame,
    output: Path,
):

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    x = np.arange(len(selected))

    p2 = (
        100.0
        * selected[
            "pass2_green_rerouted_fraction"
        ].to_numpy()
    )

    p3 = (
        100.0
        * selected[
            "pass3_prior_rerouted_fraction"
        ].to_numpy()
    )

    ax.plot(
        x,
        p2,
        marker="o",
        label="Green rerouted when brown opens",
    )

    ax.plot(
        x,
        p3,
        marker="o",
        label="Green/brown rerouted when red opens",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        selected["green_label"],
        rotation=45,
        ha="right",
    )

    util_label = selected.iloc[0]["util_label"]

    ax.set_xlabel("Green energy share range")
    ax.set_ylabel("Previously assigned flow rerouted (%)")
    ax.set_title(
        "Residual Rerouting Between Passes\n"
        f"Target Utilization = {util_label}"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# BFS / augmentation work
# ============================================================


def plot_augmenting_work(
    selected: pd.DataFrame,
    output: Path,
):

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    x = np.arange(len(selected))

    ax.plot(
        x,
        selected["reuse_pass1_augmenting_pushes"],
        marker="o",
        label="Green pass",
    )

    ax.plot(
        x,
        selected["reuse_pass2_augmenting_pushes"],
        marker="o",
        label="Brown pass",
    )

    ax.plot(
        x,
        selected["reuse_pass3_augmenting_pushes"],
        marker="o",
        label="Red pass",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        selected["green_label"],
        rotation=45,
        ha="right",
    )

    util_label = selected.iloc[0]["util_label"]

    ax.set_xlabel("Green energy share range")
    ax.set_ylabel("Successful augmenting pushes")
    ax.set_title(
        "Augmenting Work by Max-Flow Phase\n"
        f"Target Utilization = {util_label}"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Pass-3 probability
# ============================================================


def plot_pass3_probability(
    cell_df: pd.DataFrame,
    output: Path,
):

    temp = cell_df.copy()
    temp["pass3_required_percent"] = (
        100.0 * temp["pass3_required"]
    )

    make_heatmap(
        temp,
        value_col="pass3_required_percent",
        title="Fraction of Instances Requiring the Red Pass",
        colorbar_label="Instances requiring Pass 3 (%)",
        output=output,
        fmt=".0f",
    )


# ============================================================
# Runtime vs network size
# ============================================================


def plot_network_size_runtime(
    data: pd.DataFrame,
    output: Path,
):

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.scatter(
        data["job_interval_edges"],
        data["reuse_mean_ms"],
        alpha=0.45,
        label="MaxFlow-Passes",
    )

    ax.scatter(
        data["job_interval_edges"],
        data["restart_mean_ms"],
        alpha=0.45,
        label="MaxFlow-Restart",
    )

    ax.scatter(
        data["job_interval_edges"],
        data["mincost_mean_ms"],
        alpha=0.45,
        label="MinCostFlow",
    )

    ax.set_xlabel("Number of job-interval edges")
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_title("Runtime vs. Flow-Network Size")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Overall summary table
# ============================================================


def metric_summary(
    data: pd.DataFrame,
    column: str,
    name: str,
    scale: float = 1.0,
):

    values = data[column].dropna() * scale

    return {
        "Metric": name,
        "Mean": values.mean(),
        "Median": values.median(),
        "Std.": values.std(),
        "Min": values.min(),
        "Max": values.max(),
    }


def make_overall_table(
    data: pd.DataFrame,
    output_dir: Path,
):

    rows = [
        metric_summary(
            data,
            "mincost_over_reuse_mean_ratio",
            "MinCost / MaxFlow-Passes runtime",
        ),
        metric_summary(
            data,
            "restart_over_reuse_mean_ratio",
            "Restart / MaxFlow-Passes runtime",
        ),
        metric_summary(
            data,
            "pass1_flow_fraction",
            "Flow added in green pass (\\%)",
            100.0,
        ),
        metric_summary(
            data,
            "pass2_flow_fraction",
            "Flow added in brown pass (\\%)",
            100.0,
        ),
        metric_summary(
            data,
            "pass3_flow_fraction",
            "Flow added in red pass (\\%)",
            100.0,
        ),
        metric_summary(
            data,
            "pass2_green_rerouted_fraction",
            "Green allocation rerouted in Pass 2 (\\%)",
            100.0,
        ),
        metric_summary(
            data,
            "pass3_prior_rerouted_fraction",
            "Prior allocation rerouted in Pass 3 (\\%)",
            100.0,
        ),
    ]

    table = pd.DataFrame(rows)

    table.to_csv(
        output_dir / "overall_summary_table.csv",
        index=False,
    )

    latex = table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.3f}",
        escape=False,
        caption=(
            "Aggregate behavior of the exact solvers and "
            "incremental max-flow phases."
        ),
        label="tab:pass-mechanism-summary",
    )

    (
        output_dir
        / "overall_summary_table.tex"
    ).write_text(
        latex,
        encoding="utf-8",
    )

    return table


# ============================================================
# Representative configuration table
# ============================================================


def nearest_cell(
    cell_df: pd.DataFrame,
    util_target: float,
    green_target: float,
):

    temp = cell_df.copy()

    temp["_distance"] = (
        (temp["util_mid"] - util_target) ** 2
        + (temp["green_mid"] - green_target) ** 2
    )

    return temp.sort_values("_distance").iloc[0]


def make_representative_table(
    cell_df: pd.DataFrame,
    output_dir: Path,
):

    util_values = sorted(cell_df["util_mid"].unique())
    green_values = sorted(cell_df["green_mid"].unique())

    u_low = util_values[1] if len(util_values) > 2 else util_values[0]
    u_mid = util_values[len(util_values) // 2]
    u_high = util_values[-2] if len(util_values) > 2 else util_values[-1]

    g_low = green_values[1] if len(green_values) > 2 else green_values[0]
    g_mid = green_values[len(green_values) // 2]
    g_high = green_values[-2] if len(green_values) > 2 else green_values[-1]

    targets = [
        ("Low util / low green", u_low, g_low),
        ("Low util / high green", u_low, g_high),
        ("Mid util / mid green", u_mid, g_mid),
        ("High util / low green", u_high, g_low),
        ("High util / high green", u_high, g_high),
    ]

    rows = []

    for label, u, g in targets:
        r = nearest_cell(
            cell_df,
            util_target=u,
            green_target=g,
        )

        rows.append(
            {
                "Configuration": label,
                "Utilization": r["util_label"],
                "Green share": r["green_label"],
                "MCF / Reuse": r[
                    "mincost_over_reuse_mean_ratio"
                ],
                "Restart / Reuse": r[
                    "restart_over_reuse_mean_ratio"
                ],
                "Green flow (%)": (
                    100.0 * r["pass1_flow_fraction"]
                ),
                "Brown flow (%)": (
                    100.0 * r["pass2_flow_fraction"]
                ),
                "Red flow (%)": (
                    100.0 * r["pass3_flow_fraction"]
                ),
                "Pass 3 required (%)": (
                    100.0 * r["pass3_required"]
                ),
            }
        )

    table = pd.DataFrame(rows)

    table.to_csv(
        output_dir / "representative_cells_table.csv",
        index=False,
    )

    latex = table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.2f}",
        escape=False,
        caption=(
            "Representative utilization and green-share "
            "configurations."
        ),
        label="tab:representative-configurations",
    )

    (
        output_dir
        / "representative_cells_table.tex"
    ).write_text(
        latex,
        encoding="utf-8",
    )

    return table


# ============================================================
# Correlation analysis
# ============================================================


def make_correlation_table(
    data: pd.DataFrame,
    output_dir: Path,
):

    columns = [
        "mincost_over_reuse_mean_ratio",
        "restart_over_reuse_mean_ratio",
        "pass1_flow_fraction",
        "pass2_flow_fraction",
        "pass3_flow_fraction",
        "pass2_green_rerouted_fraction",
        "pass3_prior_rerouted_fraction",
        "job_interval_edges",
        "actual_utilization",
        "green_share",
    ]

    columns = [
        c for c in columns
        if c in data.columns
    ]

    corr = data[columns].corr()

    corr.to_csv(
        output_dir / "correlation_matrix.csv"
    )


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate plots and tables for Experiment 2 "
            "(MaxFlow-Passes mechanism experiment)."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "experiment2_pass_mechanism"
        ),
        help=(
            "Root directory containing "
            "util_*_green_* experiment directories."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiment2_analysis"
        ),
        help="Directory for plots and tables.",
    )

    parser.add_argument(
        "--focus-util-min",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--focus-util-max",
        type=float,
        default=0.65,
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and aggregate
    # --------------------------------------------------------

    data = load_all_results(args.root)

    cells = aggregate_cells(data)

    data.to_csv(
        args.output / "all_instance_results.csv",
        index=False,
    )

    cells.to_csv(
        args.output / "cell_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figure 1:
    # MCF / Reuse heatmap
    # --------------------------------------------------------

    make_heatmap(
        cells,
        value_col="mincost_over_reuse_mean_ratio",
        title=(
            "Runtime Ratio: "
            "Pure Min-Cost Flow / MaxFlow-Passes"
        ),
        colorbar_label=(
            "Runtime ratio "
            "(MinCostFlow / MaxFlow-Passes)"
        ),
        output=(
            args.output
            / "fig1_mincost_over_reuse_heatmap"
        ),
    )

    # --------------------------------------------------------
    # Figure 2:
    # Restart / Reuse heatmap
    # --------------------------------------------------------

    make_heatmap(
        cells,
        value_col="restart_over_reuse_mean_ratio",
        title=(
            "Benefit of Residual Reuse: "
            "MaxFlow-Restart / MaxFlow-Passes"
        ),
        colorbar_label=(
            "Runtime ratio "
            "(Restart / Reuse)"
        ),
        output=(
            args.output
            / "fig2_restart_over_reuse_heatmap"
        ),
    )

    # --------------------------------------------------------
    # Focused utilization
    # --------------------------------------------------------

    selected = select_utilization(
        cells,
        args.focus_util_min,
        args.focus_util_max,
    )

    # --------------------------------------------------------
    # Figure 3:
    # Flow fractions
    # --------------------------------------------------------

    plot_flow_fractions(
        selected,
        args.output
        / "fig3_flow_fraction_by_pass",
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    plot_pass_runtime(
        selected,
        args.output
        / "diag_pass_runtime",
    )

    plot_rerouting(
        selected,
        args.output
        / "diag_rerouting",
    )

    plot_augmenting_work(
        selected,
        args.output
        / "diag_augmenting_pushes",
    )

    plot_pass3_probability(
        cells,
        args.output
        / "diag_pass3_probability_heatmap",
    )

    plot_network_size_runtime(
        data,
        args.output
        / "diag_runtime_vs_network_size",
    )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    overall = make_overall_table(
        data,
        args.output,
    )

    representative = make_representative_table(
        cells,
        args.output,
    )

    make_correlation_table(
        data,
        args.output,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EXPERIMENT 2 ANALYSIS COMPLETE")
    print("=" * 72)

    print()
    print("Overall summary:")
    print(overall.to_string(index=False))

    print()
    print("Representative configurations:")
    print(representative.to_string(index=False))

    print()
    print("Main paper figures:")
    print(
        "  fig1_mincost_over_reuse_heatmap.pdf"
    )
    print(
        "  fig2_restart_over_reuse_heatmap.pdf"
    )
    print(
        "  fig3_flow_fraction_by_pass.pdf"
    )

    print()
    print("Suggested paper table:")
    print(
        "  overall_summary_table.tex"
    )

    print()
    print("Additional diagnostics:")
    print(
        "  diag_pass_runtime.pdf"
    )
    print(
        "  diag_rerouting.pdf"
    )
    print(
        "  diag_augmenting_pushes.pdf"
    )
    print(
        "  diag_pass3_probability_heatmap.pdf"
    )
    print(
        "  diag_runtime_vs_network_size.pdf"
    )

    print()
    print(
        f"All outputs saved under: {args.output}"
    )


if __name__ == "__main__":
    main()