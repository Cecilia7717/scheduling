#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT = HERE / "lawler_weight_scaling_results" / "results.csv"
OUTPUT = HERE / "lawler_weight_scaling_results" / "lawler_runtime_vs_weight.pdf"

df = pd.read_csv(INPUT)

# Aggregate repeated instances.
summary = (
    df.groupby(["num_original_jobs", "weight_scale_alpha"])
    ["lawler_runtime_seconds"]
    .agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(6.4, 4.3))

markers = ["o", "s", "^", "D", "v"]

for marker, n in zip(markers, sorted(summary["num_original_jobs"].unique())):
    d = summary[
        summary["num_original_jobs"] == n
    ].sort_values("weight_scale_alpha")

    ax.plot(
        d["weight_scale_alpha"],
        d["median"],
        marker=marker,
        linewidth=1.8,
        markersize=6,
        label=f"$n={n}$",
    )

    ax.fill_between(
        d["weight_scale_alpha"],
        d["q25"],
        d["q75"],
        alpha=0.15,
    )

ax.set_xlabel(r"Weight scale $\alpha$")
ax.set_ylabel("Lawler runtime (seconds)")

# Runtime spans several orders of magnitude.
ax.set_yscale("log")

ax.set_xticks(sorted(df["weight_scale_alpha"].unique()))

ax.grid(
    True,
    which="both",
    linestyle="--",
    linewidth=0.5,
    alpha=0.5,
)

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(OUTPUT, bbox_inches="tight")
fig.savefig(
    OUTPUT.with_suffix(".png"),
    dpi=300,
    bbox_inches="tight",
)

summary.to_csv(
    OUTPUT.parent / "weight_scaling_summary.csv",
    index=False,
)

print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
print("\nSummary:")
print(summary.to_string(index=False))

plt.close(fig)