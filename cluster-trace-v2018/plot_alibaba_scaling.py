#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT = Path("alibaba_scaling_jobs_all/runtime_results/runtime_summary.csv")
OUTPUT_DIR = Path("alibaba_scaling_jobs_all/runtime_results/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)

# Sort only to make connected lines progress from smaller to larger x.
df_tasks = df.sort_values("num_tasks")
df_edges = df.sort_values("num_job_interval_edges")

print(f"Loaded {len(df)} workloads")
print(f"Task range: {df['num_tasks'].min()} - {df['num_tasks'].max()}")
print(
    "Job-interval edge range: "
    f"{df['num_job_interval_edges'].min()} - "
    f"{df['num_job_interval_edges'].max()}"
)


# ============================================================
# Figure 1: Runtime vs. number of DAG tasks
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 4.2))

ax.scatter(
    df_tasks["num_tasks"],
    df_tasks["maxflow_median_seconds"],
    marker="o",
    s=40,
    label="MaxFlow + Passes",
)

ax.scatter(
    df_tasks["num_tasks"],
    df_tasks["mincost_median_seconds"],
    marker="s",
    s=40,
    label="Min-Cost Flow",
)

ax.set_yscale("log")

ax.set_xlabel("Number of DAG Tasks")
ax.set_ylabel("Median Runtime (s)")

ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_vs_tasks.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_vs_tasks.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 2: Runtime vs. number of job-interval edges
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 4.2))

ax.plot(
    df_edges["num_job_interval_edges"],
    df_edges["maxflow_median_seconds"],
    marker="o",
    markersize=5,
    linewidth=1.5,
    label="MaxFlow + Passes",
)

ax.plot(
    df_edges["num_job_interval_edges"],
    df_edges["mincost_median_seconds"],
    marker="s",
    markersize=5,
    linewidth=1.5,
    label="Min-Cost Flow",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of Job-Interval Edges")
ax.set_ylabel("Median Runtime (s)")

ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_vs_edges.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_vs_edges.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 3: Min-Cost / MaxFlow runtime ratio vs. DAG tasks
# ============================================================

df_ratio = df.sort_values("num_tasks").copy()

# Recompute using median runtimes. This is preferable for the figure
# because the existing mincost_over_maxflow column may have been
# computed from mean runtimes.
df_ratio["median_runtime_ratio"] = (
    df_ratio["mincost_median_seconds"]
    / df_ratio["maxflow_median_seconds"]
)

fig, ax = plt.subplots(figsize=(6.0, 4.2))

ax.scatter(
    df_ratio["num_tasks"],
    df_ratio["median_runtime_ratio"],
    s=40,
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
)

ax.set_xlabel("Number of DAG Tasks")
ax.set_ylabel("Runtime Ratio (Min-Cost / MaxFlow-Passes)")

ax.grid(True, linestyle="--", alpha=0.35)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_ratio_vs_tasks.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_ratio_vs_tasks.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 4: Min-Cost / MaxFlow ratio vs. network edges
# ============================================================

df_ratio_edges = df.sort_values("num_job_interval_edges").copy()

df_ratio_edges["median_runtime_ratio"] = (
    df_ratio_edges["mincost_median_seconds"]
    / df_ratio_edges["maxflow_median_seconds"]
)

fig, ax = plt.subplots(figsize=(6.0, 4.2))

ax.scatter(
    df_ratio_edges["num_job_interval_edges"],
    df_ratio_edges["median_runtime_ratio"],
    s=40,
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
)

ax.set_xscale("log")

ax.set_xlabel("Number of Job-Interval Edges")
ax.set_ylabel("Runtime Ratio (Min-Cost / MaxFlow-Passes)")

ax.grid(True, which="both", linestyle="--", alpha=0.35)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_ratio_vs_edges.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "alibaba_runtime_ratio_vs_edges.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Summary statistics
# ============================================================

ratio = (
    df["mincost_median_seconds"]
    / df["maxflow_median_seconds"]
)

print()
print("Runtime ratio (Min-Cost / MaxFlow-Passes)")
print(f"  Mean:   {ratio.mean():.3f}x")
print(f"  Median: {ratio.median():.3f}x")
print(f"  Min:    {ratio.min():.3f}x")
print(f"  Max:    {ratio.max():.3f}x")

print()
print("Figures written to:")
print(OUTPUT_DIR)