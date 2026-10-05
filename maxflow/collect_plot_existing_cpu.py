#!/usr/bin/env python3
"""
Task 1: collect and plot the CPU MaxFlow-Passes runtimes that are already
stored in the existing scaling-result JSON files.

Expected folder names:
    jobs_100_intervals_10_sparse/
    jobs_100_intervals_25_light/
    jobs_100_intervals_50_medium/
    jobs_100_intervals_100_dense/
    jobs_100_intervals_200_very_dense/

Expected JSON field:
    cpu_max_flow_passes.wall_seconds

Outputs:
    existing_cpu_all_instances.csv
    existing_cpu_scaling_summary.csv
    existing_cpu_runtime_vs_jobs.{pdf,png}
    existing_cpu_runtime_vs_edges.{pdf,png}
    existing_network_edges_vs_jobs.{pdf,png}
    existing_cpu_runtime_<density>.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("job_interval_scaling_results_gpu_ecl_timing_adjusted"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to ROOT/cpu_scaling_analysis",
    )
    return p.parse_args()


def folder_metadata(path: Path) -> Dict[str, Any] | None:
    m = FOLDER_RE.fullmatch(path.parent.name)
    if m is None:
        return None
    return {
        "folder": path.parent.name,
        "num_jobs": int(m.group("jobs")),
        "num_original_intervals": int(m.group("intervals")),
        "density": m.group("density"),
    }


def compute_job_interval_edges(data: Dict[str, Any]) -> int:
    """
    Count edges j -> I in the CPU flow network using the already stored
    atomic/split intervals.
    """
    jobs = data.get("jobs", [])
    intervals = data.get("split_intervals", [])

    count = 0
    for job in jobs:
        r = float(job["release"])
        d = float(job["deadline"])
        for interval in intervals:
            a = float(interval["start"])
            b = float(interval["end"])
            if r <= a + EPS and b <= d + EPS:
                count += 1
    return count


def collect(root: Path) -> List[Dict[str, Any]]:
    rows = []

    for path in sorted(root.rglob("instance_*.json")):
        meta = folder_metadata(path)
        if meta is None:
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: cannot read {path}: {exc}")
            continue

        cpu = data.get("cpu_max_flow_passes")
        if cpu is None:
            print(f"WARNING: no cpu_max_flow_passes in {path}")
            continue

        wall = cpu.get("wall_seconds")
        if wall is None:
            print(f"WARNING: no cpu wall_seconds in {path}")
            continue

        split_intervals = data.get("split_intervals", [])

        rows.append({
            **meta,
            "file": str(path.relative_to(root)),
            "instance_id": data.get("instance_id"),
            "horizon": data.get("horizon"),
            "num_atomic_intervals": len(split_intervals),
            "job_interval_edges": compute_job_interval_edges(data),
            "cpu_wall_seconds": float(wall),
            "cpu_feasible": cpu.get("feasible"),
            "cpu_cost": cpu.get("normalized_cost"),
        })

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["folder"], []).append(row)

    out = []
    for folder, group in groups.items():
        first = group[0]
        runtimes = [x["cpu_wall_seconds"] for x in group]
        edges = [x["job_interval_edges"] for x in group]
        atomic = [x["num_atomic_intervals"] for x in group]

        out.append({
            "folder": folder,
            "num_jobs": first["num_jobs"],
            "num_original_intervals": first["num_original_intervals"],
            "density": first["density"],
            "num_instances": len(group),

            "mean_atomic_intervals": statistics.mean(atomic),
            "median_atomic_intervals": statistics.median(atomic),

            "mean_job_interval_edges": statistics.mean(edges),
            "median_job_interval_edges": statistics.median(edges),

            "mean_cpu_wall_seconds": statistics.mean(runtimes),
            "median_cpu_wall_seconds": statistics.median(runtimes),
            "min_cpu_wall_seconds": min(runtimes),
            "max_cpu_wall_seconds": max(runtimes),
            "std_cpu_wall_seconds": (
                statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
            ),
        })

    out.sort(
        key=lambda r: (
            DENSITY_ORDER.get(r["density"], 999),
            r["num_jobs"],
            r["num_original_intervals"],
        )
    )
    return out


def save_both(fig, base: Path):
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_jobs(summary, output: Path):
    by_density = {}
    for r in summary:
        by_density.setdefault(r["density"], []).append(r)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    for density in sorted(by_density, key=lambda d: DENSITY_ORDER.get(d, 999)):
        g = sorted(by_density[density], key=lambda x: x["num_jobs"])
        ax.plot(
            [x["num_jobs"] for x in g],
            [x["mean_cpu_wall_seconds"] for x in g],
            marker="o",
            label=density.replace("_", " "),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Mean CPU MaxFlow-Passes wall time (s)")
    ax.set_title("CPU MaxFlow-Passes scalability")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Original interval density")
    fig.tight_layout()
    save_both(fig, output / "existing_cpu_runtime_vs_jobs")


def plot_runtime_vs_edges(summary, output: Path):
    by_density = {}
    for r in summary:
        by_density.setdefault(r["density"], []).append(r)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    for density in sorted(by_density, key=lambda d: DENSITY_ORDER.get(d, 999)):
        g = sorted(by_density[density], key=lambda x: x["mean_job_interval_edges"])
        ax.plot(
            [x["mean_job_interval_edges"] for x in g],
            [x["mean_cpu_wall_seconds"] for x in g],
            marker="o",
            label=density.replace("_", " "),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean number of job-interval edges")
    ax.set_ylabel("Mean CPU MaxFlow-Passes wall time (s)")
    ax.set_title("CPU runtime vs. flow-network edge count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Original interval density")
    fig.tight_layout()
    save_both(fig, output / "existing_cpu_runtime_vs_edges")


def plot_edges_vs_jobs(summary, output: Path):
    by_density = {}
    for r in summary:
        by_density.setdefault(r["density"], []).append(r)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    for density in sorted(by_density, key=lambda d: DENSITY_ORDER.get(d, 999)):
        g = sorted(by_density[density], key=lambda x: x["num_jobs"])
        ax.plot(
            [x["num_jobs"] for x in g],
            [x["mean_job_interval_edges"] for x in g],
            marker="o",
            label=density.replace("_", " "),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Mean number of job-interval edges")
    ax.set_title("Flow-network growth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Original interval density")
    fig.tight_layout()
    save_both(fig, output / "existing_network_edges_vs_jobs")


def plot_each_density(summary, output: Path):
    by_density = {}
    for r in summary:
        by_density.setdefault(r["density"], []).append(r)

    for density, group in by_density.items():
        g = sorted(group, key=lambda x: x["num_jobs"])

        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.plot(
            [x["num_jobs"] for x in g],
            [x["mean_cpu_wall_seconds"] for x in g],
            marker="o",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of jobs")
        ax.set_ylabel("Mean CPU MaxFlow-Passes wall time (s)")
        ax.set_title(
            f"CPU MaxFlow-Passes — {density.replace('_', ' ')} intervals"
        )
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_both(fig, output / f"existing_cpu_runtime_{density}")


def main():
    args = parse_args()
    root = args.root
    output = args.output or (root / "cpu_scaling_analysis")
    output.mkdir(parents=True, exist_ok=True)

    rows = collect(root)
    if not rows:
        raise RuntimeError(
            f"No JSON files containing cpu_max_flow_passes.wall_seconds "
            f"were found under {root}"
        )

    summary = summarize(rows)

    write_csv(output / "existing_cpu_all_instances.csv", rows)
    write_csv(output / "existing_cpu_scaling_summary.csv", summary)

    plot_runtime_vs_jobs(summary, output)
    plot_runtime_vs_edges(summary, output)
    plot_edges_vs_jobs(summary, output)
    plot_each_density(summary, output)

    print("=" * 80)
    print("EXISTING CPU SCALING DATA COLLECTED")
    print("=" * 80)
    print(f"Instances: {len(rows)}")
    print(f"Cells:     {len(summary)}")
    print(f"Output:    {output}")
    print()
    print("Main outputs:")
    print(f"  {output / 'existing_cpu_scaling_summary.csv'}")
    print(f"  {output / 'existing_cpu_runtime_vs_jobs.pdf'}")
    print(f"  {output / 'existing_cpu_runtime_vs_edges.pdf'}")
    print(f"  {output / 'existing_network_edges_vs_jobs.pdf'}")


if __name__ == "__main__":
    main()
