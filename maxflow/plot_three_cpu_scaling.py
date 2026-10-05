#!/usr/bin/env python3
"""
Plot CPU exact-solver scaling together with GPU MaxFlow-Passes timing.

Compared runtimes:
    1. CPU MaxFlow-Passes
    2. CPU MaxFlow-Restart
    3. CPU MinCostFlow
    4. GPU MaxFlow-Passes algorithm time
    5. GPU MaxFlow-Passes full wall time

Outputs:
    cpu_gpu_algorithms_all_instances.csv
    cpu_gpu_algorithms_summary.csv

    runtime_all_<density>.pdf/png
    runtime_all_vs_edges.pdf/png

    mincost_over_cpu_passes_vs_edges.pdf/png
    restart_over_cpu_passes_vs_edges.pdf/png
    gpu_algorithm_over_cpu_passes_vs_edges.pdf/png
    gpu_wall_over_cpu_passes_vs_edges.pdf/png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-9


FOLDER_RE = re.compile(
    r"^jobs_(?P<jobs>\d+)_intervals_(?P<intervals>\d+)_(?P<density>.+)$"
)


DENSITY_ORDER = {
    "sparse": 0,
    "light": 1,
    "medium": 2,
    "dense": 3,
    "very_dense": 4,
}


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "job_interval_scaling_results_gpu_ecl_timing_adjusted"
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return p.parse_args()


# ============================================================
# Metadata
# ============================================================

def meta(path: Path):

    m = FOLDER_RE.fullmatch(path.parent.name)

    if m is None:
        return None

    return {
        "folder": path.parent.name,
        "num_jobs": int(m.group("jobs")),
        "num_original_intervals": int(m.group("intervals")),
        "density": m.group("density"),
    }


# ============================================================
# Job -> interval edge count
# ============================================================

def edge_count(data):

    jobs = data.get("jobs", [])
    intervals = data.get("split_intervals", [])

    count = 0

    for job in jobs:

        r = float(job["release"])
        d = float(job["deadline"])

        for interval in intervals:

            if (
                r <= float(interval["start"]) + EPS
                and
                float(interval["end"]) <= d + EPS
            ):
                count += 1

    return count


# ============================================================
# Collect data
# ============================================================

def collect(root):

    rows = []

    for path in sorted(root.rglob("instance_*.json")):

        m = meta(path)

        if m is None:
            continue

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        # ----------------------------------------------------
        # CPU exact algorithms
        # ----------------------------------------------------

        exact = data.get("cpu_exact_scaling")

        if exact is None:
            continue

        algs = {}

        complete = True

        for key in (
            "max_flow_passes",
            "max_flow_restart",
            "pure_min_cost",
        ):

            result = exact.get(key, {})

            if result.get("status") != "ok":
                complete = False
                break

            algs[key] = result

        if not complete:
            continue

        # ----------------------------------------------------
        # GPU MaxFlow-Passes
        # ----------------------------------------------------

        gpu = data.get("gpu_max_flow_passes")

        if gpu is None:
            print(
                f"WARNING: no gpu_max_flow_passes in {path}"
            )
            continue

        gpu_alg_seconds = gpu.get(
            "gpu_maxflow_seconds"
        )

        gpu_wall_seconds = gpu.get(
            "wall_seconds"
        )

        if (
            gpu_alg_seconds is None
            or gpu_wall_seconds is None
        ):
            print(
                f"WARNING: missing GPU timing fields in {path}"
            )
            continue

        # ----------------------------------------------------
        # CPU runtimes
        # ----------------------------------------------------

        passes_seconds = (
            algs["max_flow_passes"]
            ["timing"]
            ["mean_seconds"]
        )

        restart_seconds = (
            algs["max_flow_restart"]
            ["timing"]
            ["mean_seconds"]
        )

        mincost_seconds = (
            algs["pure_min_cost"]
            ["timing"]
            ["mean_seconds"]
        )

        gpu_alg_seconds = float(
            gpu_alg_seconds
        )

        gpu_wall_seconds = float(
            gpu_wall_seconds
        )

        # ----------------------------------------------------
        # Store row
        # ----------------------------------------------------

        rows.append({
            **m,

            "instance_id":
                data.get("instance_id"),

            "num_atomic_intervals":
                len(
                    data.get(
                        "split_intervals",
                        [],
                    )
                ),

            "job_interval_edges":
                edge_count(data),

            # CPU
            "passes_seconds":
                passes_seconds,

            "restart_seconds":
                restart_seconds,

            "mincost_seconds":
                mincost_seconds,

            # GPU
            "gpu_algorithm_seconds":
                gpu_alg_seconds,

            "gpu_wall_seconds":
                gpu_wall_seconds,

            # CPU algorithm ratios
            "restart_over_passes":
                restart_seconds
                / passes_seconds,

            "mincost_over_passes":
                mincost_seconds
                / passes_seconds,

            # GPU ratios
            "gpu_algorithm_over_passes":
                gpu_alg_seconds
                / passes_seconds,

            "gpu_wall_over_passes":
                gpu_wall_seconds
                / passes_seconds,
        })

    return rows


# ============================================================
# Summary
# ============================================================

def summarize(rows):

    groups = {}

    for row in rows:
        groups.setdefault(
            row["folder"],
            [],
        ).append(row)

    out = []

    for folder, group in groups.items():

        first = group[0]

        def mean(name):
            return statistics.mean(
                x[name]
                for x in group
            )

        def median(name):
            return statistics.median(
                x[name]
                for x in group
            )

        out.append({

            "folder":
                folder,

            "num_jobs":
                first["num_jobs"],

            "num_original_intervals":
                first["num_original_intervals"],

            "density":
                first["density"],

            "num_instances":
                len(group),

            "mean_atomic_intervals":
                mean(
                    "num_atomic_intervals"
                ),

            "mean_job_interval_edges":
                mean(
                    "job_interval_edges"
                ),

            # ================================================
            # CPU times
            # ================================================

            "mean_passes_seconds":
                mean(
                    "passes_seconds"
                ),

            "median_passes_seconds":
                median(
                    "passes_seconds"
                ),

            "mean_restart_seconds":
                mean(
                    "restart_seconds"
                ),

            "median_restart_seconds":
                median(
                    "restart_seconds"
                ),

            "mean_mincost_seconds":
                mean(
                    "mincost_seconds"
                ),

            "median_mincost_seconds":
                median(
                    "mincost_seconds"
                ),

            # ================================================
            # GPU times
            # ================================================

            "mean_gpu_algorithm_seconds":
                mean(
                    "gpu_algorithm_seconds"
                ),

            "median_gpu_algorithm_seconds":
                median(
                    "gpu_algorithm_seconds"
                ),

            "mean_gpu_wall_seconds":
                mean(
                    "gpu_wall_seconds"
                ),

            "median_gpu_wall_seconds":
                median(
                    "gpu_wall_seconds"
                ),

            # ================================================
            # Ratios
            # ================================================

            "mean_restart_over_passes":
                mean(
                    "restart_over_passes"
                ),

            "mean_mincost_over_passes":
                mean(
                    "mincost_over_passes"
                ),

            "mean_gpu_algorithm_over_passes":
                mean(
                    "gpu_algorithm_over_passes"
                ),

            "mean_gpu_wall_over_passes":
                mean(
                    "gpu_wall_over_passes"
                ),
        })

    out.sort(
        key=lambda r: (
            DENSITY_ORDER.get(
                r["density"],
                999,
            ),
            r["num_jobs"],
        )
    )

    return out


# ============================================================
# CSV
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Figure save
# ============================================================

def save(fig, base):

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Runtime vs jobs
# One plot per density
# ============================================================

def plot_density_runtime(
    summary,
    output,
):

    by_density = {}

    for row in summary:

        by_density.setdefault(
            row["density"],
            [],
        ).append(row)

    for density, group in by_density.items():

        g = sorted(
            group,
            key=lambda x: x["num_jobs"],
        )

        fig, ax = plt.subplots(
            figsize=(8.3, 5.4)
        )

        x = [
            r["num_jobs"]
            for r in g
        ]

        # ----------------------------------------------------
        # CPU algorithms
        # ----------------------------------------------------

        ax.plot(
            x,
            [
                r["mean_passes_seconds"]
                for r in g
            ],
            marker="o",
            label="CPU MaxFlow-Passes",
        )

        ax.plot(
            x,
            [
                r["mean_restart_seconds"]
                for r in g
            ],
            marker="s",
            label="CPU MaxFlow-Restart",
        )

        ax.plot(
            x,
            [
                r["mean_mincost_seconds"]
                for r in g
            ],
            marker="^",
            label="CPU MinCostFlow",
        )

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        ax.plot(
            x,
            [
                r["mean_gpu_algorithm_seconds"]
                for r in g
            ],
            marker="D",
            label="GPU MaxFlow-Passes (algorithm)",
        )

        ax.plot(
            x,
            [
                r["mean_gpu_wall_seconds"]
                for r in g
            ],
            marker="v",
            linestyle="--",
            label="GPU MaxFlow-Passes (wall)",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(
            "Number of jobs"
        )

        ax.set_ylabel(
            "Mean runtime (s)"
        )

        ax.set_title(
            "Exact-solver scalability — "
            + density.replace(
                "_",
                " ",
            )
        )

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save(
            fig,
            output
            / f"runtime_all_{density}",
        )


# ============================================================
# Runtime vs edges
# ============================================================

def plot_runtime_edges(
    summary,
    output,
):

    fig, ax = plt.subplots(
        figsize=(8.6, 5.7)
    )

    ordered = sorted(
        summary,
        key=lambda r:
            r["mean_job_interval_edges"],
    )

    x = [
        r["mean_job_interval_edges"]
        for r in ordered
    ]

    ax.scatter(
        x,
        [
            r["mean_passes_seconds"]
            for r in ordered
        ],
        label="CPU MaxFlow-Passes",
        alpha=0.7,
    )

    ax.scatter(
        x,
        [
            r["mean_restart_seconds"]
            for r in ordered
        ],
        label="CPU MaxFlow-Restart",
        alpha=0.7,
    )

    ax.scatter(
        x,
        [
            r["mean_mincost_seconds"]
            for r in ordered
        ],
        label="CPU MinCostFlow",
        alpha=0.7,
    )

    ax.scatter(
        x,
        [
            r["mean_gpu_algorithm_seconds"]
            for r in ordered
        ],
        label="GPU MaxFlow-Passes (algorithm)",
        alpha=0.7,
    )

    ax.scatter(
        x,
        [
            r["mean_gpu_wall_seconds"]
            for r in ordered
        ],
        label="GPU MaxFlow-Passes (wall)",
        alpha=0.7,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "Mean number of job-interval edges"
    )

    ax.set_ylabel(
        "Mean runtime (s)"
    )

    ax.set_title(
        "Exact-solver runtime vs. flow-network size"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    save(
        fig,
        output
        / "runtime_all_vs_edges",
    )


# ============================================================
# Ratio vs edges
# ============================================================

def plot_ratio_edges(
    summary,
    output,
    field,
    title,
    filename,
    ylabel,
):

    fig, ax = plt.subplots(
        figsize=(8.6, 5.7)
    )

    by_density = {}

    for r in summary:

        by_density.setdefault(
            r["density"],
            [],
        ).append(r)

    ordered_densities = sorted(
        by_density,
        key=lambda d:
            DENSITY_ORDER.get(
                d,
                999,
            ),
    )

    for density in ordered_densities:

        g = sorted(
            by_density[density],
            key=lambda r:
                r["mean_job_interval_edges"],
        )

        ax.plot(
            [
                r["mean_job_interval_edges"]
                for r in g
            ],
            [
                r[field]
                for r in g
            ],
            marker="o",
            label=density.replace(
                "_",
                " ",
            ),
        )

    # Ratio = 1 means equal runtime
    ax.axhline(
        1.0,
        linewidth=1.0,
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Mean number of job-interval edges"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(
        title="Original interval density"
    )

    fig.tight_layout()

    save(
        fig,
        output / filename,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    output = (
        args.output
        or (
            args.root
            / "cpu_gpu_algorithm_analysis"
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    rows = collect(
        args.root
    )

    if not rows:

        raise RuntimeError(
            "No instances containing both "
            "cpu_exact_scaling and "
            "gpu_max_flow_passes were found."
        )

    # --------------------------------------------------------
    # Summarize
    # --------------------------------------------------------

    summary = summarize(
        rows
    )

    # --------------------------------------------------------
    # CSV files
    # --------------------------------------------------------

    write_csv(
        output
        / "cpu_gpu_algorithms_all_instances.csv",
        rows,
    )

    write_csv(
        output
        / "cpu_gpu_algorithms_summary.csv",
        summary,
    )

    # --------------------------------------------------------
    # Absolute runtime plots
    # --------------------------------------------------------

    plot_density_runtime(
        summary,
        output,
    )

    plot_runtime_edges(
        summary,
        output,
    )

    # --------------------------------------------------------
    # CPU comparison ratios
    # --------------------------------------------------------

    plot_ratio_edges(
        summary,
        output,
        "mean_mincost_over_passes",
        "MinCostFlow / CPU MaxFlow-Passes vs. network size",
        "mincost_over_cpu_passes_vs_edges",
        "Runtime ratio (MinCostFlow / CPU MaxFlow-Passes)",
    )

    plot_ratio_edges(
        summary,
        output,
        "mean_restart_over_passes",
        "MaxFlow-Restart / CPU MaxFlow-Passes vs. network size",
        "restart_over_cpu_passes_vs_edges",
        "Runtime ratio (Restart / CPU MaxFlow-Passes)",
    )

    # --------------------------------------------------------
    # GPU comparison ratios
    # --------------------------------------------------------

    plot_ratio_edges(
        summary,
        output,
        "mean_gpu_algorithm_over_passes",
        "GPU algorithm / CPU MaxFlow-Passes vs. network size",
        "gpu_algorithm_over_cpu_passes_vs_edges",
        "Runtime ratio (GPU algorithm / CPU MaxFlow-Passes)",
    )

    plot_ratio_edges(
        summary,
        output,
        "mean_gpu_wall_over_passes",
        "GPU wall time / CPU MaxFlow-Passes vs. network size",
        "gpu_wall_over_cpu_passes_vs_edges",
        "Runtime ratio (GPU wall / CPU MaxFlow-Passes)",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 80)
    print("CPU + GPU ALGORITHM SCALING ANALYSIS")
    print("=" * 80)

    print(
        f"Instances: {len(rows)}"
    )

    print(
        f"Cells:     {len(summary)}"
    )

    print(
        f"Output:    {output}"
    )

    print(
        "Summary:   "
        f"{output / 'cpu_gpu_algorithms_summary.csv'}"
    )


if __name__ == "__main__":
    main()