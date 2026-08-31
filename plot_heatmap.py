from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "heatmap_results_1"

MAXFLOW_OUTPUT = BASE_DIR / "maxflow_runtime_heatmap_1.png"
MINCOST_OUTPUT = BASE_DIR / "mincost_runtime_heatmap_1.png"
RATIO_OUTPUT = BASE_DIR / "runtime_ratio_heatmap_1.png"


GREEN_RANGES = [
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, 0.35),
    (0.35, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
]


UTIL_RANGES = [
    (0.30, 0.35),
    (0.35, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.00),
]


def read_mean_runtimes(summary_path):
    text = summary_path.read_text(encoding="utf-8")

    maxflow_match = re.search(
        r"Max Flow \+ Passes mean runtime:\s*([0-9.]+)\s*ms",
        text,
    )

    mincost_match = re.search(
        r"Pure Min-Cost Flow mean runtime:\s*([0-9.]+)\s*ms",
        text,
    )

    if maxflow_match is None:
        raise ValueError(
            f"Could not find Max Flow runtime in {summary_path}"
        )

    if mincost_match is None:
        raise ValueError(
            f"Could not find Pure Min-Cost runtime in {summary_path}"
        )

    maxflow = float(maxflow_match.group(1))
    mincost = float(mincost_match.group(1))

    return maxflow, mincost


# ============================================================
# Build matrices
# ============================================================

shape = (
    len(UTIL_RANGES),
    len(GREEN_RANGES),
)

maxflow_matrix = np.full(
    shape,
    np.nan,
)

mincost_matrix = np.full(
    shape,
    np.nan,
)

ratio_matrix = np.full(
    shape,
    np.nan,
)


for i, (util_min, util_max) in enumerate(UTIL_RANGES):

    for j, (green_min, green_max) in enumerate(GREEN_RANGES):

        folder_name = (
            f"util_{util_min:.2f}_{util_max:.2f}"
            f"_green_{green_min:.2f}_{green_max:.2f}"
        )

        summary_path = (
            RESULTS_DIR
            / folder_name
            / "summary.txt"
        )

        if not summary_path.exists():
            print(f"Missing: {summary_path}")
            continue

        maxflow, mincost = read_mean_runtimes(
            summary_path
        )

        maxflow_matrix[i, j] = maxflow
        mincost_matrix[i, j] = mincost

        if maxflow > 0:
            ratio_matrix[i, j] = (
                mincost / maxflow
            )


# ============================================================
# Labels
# ============================================================

green_labels = [
    f"{a:.2f}-{b:.2f}"
    for a, b in GREEN_RANGES
]

util_labels = [
    f"{a:.2f}-{b:.2f}"
    for a, b in UTIL_RANGES
]


# ============================================================
# General heatmap function
# ============================================================

def draw_heatmap(
    matrix,
    title,
    colorbar_label,
    output_file,
    value_format=".2f",
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(
        figsize=(13, 10)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(len(GREEN_RANGES))
    )

    ax.set_xticklabels(
        green_labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(UTIL_RANGES))
    )

    ax.set_yticklabels(
        util_labels
    )

    ax.set_xlabel(
        "Green Energy Share Range"
    )

    ax.set_ylabel(
        "Target Utilization Range"
    )

    ax.set_title(title)

    # Put the value inside each cell.
    for i in range(len(UTIL_RANGES)):

        for j in range(len(GREEN_RANGES)):

            value = matrix[i, j]

            if np.isnan(value):
                text = "N/A"
            else:
                text = format(
                    value,
                    value_format,
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        colorbar_label
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
    )

    print(
        f"Saved: {output_file}"
    )

    plt.show()


# ============================================================
# Use the same color scale for the two raw-runtime heatmaps
# ============================================================

all_runtime_values = np.concatenate(
    [
        maxflow_matrix[
            ~np.isnan(maxflow_matrix)
        ],
        mincost_matrix[
            ~np.isnan(mincost_matrix)
        ],
    ]
)

if len(all_runtime_values) > 0:
    runtime_min = float(
        np.min(all_runtime_values)
    )

    runtime_max = float(
        np.max(all_runtime_values)
    )
else:
    runtime_min = None
    runtime_max = None


# ============================================================
# Heatmap 1: Max Flow + Passes
# ============================================================

draw_heatmap(
    matrix=maxflow_matrix,
    title="Max Flow + Passes Mean Runtime",
    colorbar_label="Mean Runtime (ms)",
    output_file=MAXFLOW_OUTPUT,
    value_format=".2f",
    vmin=runtime_min,
    vmax=runtime_max,
)


# ============================================================
# Heatmap 2: Pure Min-Cost Flow
# ============================================================

draw_heatmap(
    matrix=mincost_matrix,
    title="Pure Min-Cost Flow Mean Runtime",
    colorbar_label="Mean Runtime (ms)",
    output_file=MINCOST_OUTPUT,
    value_format=".2f",
    vmin=runtime_min,
    vmax=runtime_max,
)


# ============================================================
# Heatmap 3: Runtime ratio
#
# ratio < 1:
#     Max Flow + Passes is faster
#
# ratio = 1:
#     Same runtime
#
# ratio > 1:
#     Pure Min-Cost Flow is faster
# ============================================================

draw_heatmap(
    matrix=ratio_matrix,
    title=(
        "Runtime Ratio: "
        "Pure Min-Cost Flow / Max Flow + Passes"
    ),
    colorbar_label=(
        "Runtime Ratio "
        "(Pure Min-Cost Flow / Max Flow + Passes)"
    ),
    output_file=RATIO_OUTPUT,
    value_format=".2f",
)