from pathlib import Path
import re
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# Put this script in the same directory containing:
#
# benchmark_green_0.20_0.25/
# benchmark_green_0.25_0.30/
# ...
# benchmark_green_0.80_0.85/
#
# Each folder must contain summary.txt.

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "runtime_vs_green_share.png"


# ============================================================
# Read one summary.txt
# ============================================================

def read_summary(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")

    green_match = re.search(
        r"Green share range:\s*([0-9.]+)\s+to\s+([0-9.]+)",
        text,
    )

    util_match = re.search(
        r"Target utilization range:\s*([0-9.]+)\s+to\s+([0-9.]+)",
        text,
    )

    maxflow_match = re.search(
        r"Max Flow \+ Passes mean runtime:\s*([0-9.]+)\s*ms",
        text,
    )

    mincost_match = re.search(
        r"Pure Min-Cost Flow mean runtime:\s*([0-9.]+)\s*ms",
        text,
    )

    if green_match is None:
        raise ValueError(
            f"Could not find green share range in {summary_path}"
        )

    if maxflow_match is None:
        raise ValueError(
            f"Could not find Max Flow + Passes mean runtime in {summary_path}"
        )

    if mincost_match is None:
        raise ValueError(
            f"Could not find Pure Min-Cost Flow mean runtime in {summary_path}"
        )

    green_min = float(green_match.group(1))
    green_max = float(green_match.group(2))

    if util_match is not None:
        util_min = float(util_match.group(1))
        util_max = float(util_match.group(2))
    else:
        util_min = None
        util_max = None

    maxflow_runtime = float(maxflow_match.group(1))
    mincost_runtime = float(mincost_match.group(1))

    return {
        "green_min": green_min,
        "green_max": green_max,
        "label": f"{green_min:.2f}-{green_max:.2f}",
        "util_min": util_min,
        "util_max": util_max,
        "maxflow_runtime": maxflow_runtime,
        "mincost_runtime": mincost_runtime,
    }


# ============================================================
# Find experiment folders
# ============================================================

def collect_results():
    results = []

    for folder in BASE_DIR.glob("benchmark_green_*"):
        if not folder.is_dir():
            continue

        summary_path = folder / "summary.txt"

        if not summary_path.exists():
            print(f"Skipping {folder.name}: no summary.txt")
            continue

        try:
            result = read_summary(summary_path)
            results.append(result)
        except ValueError as exc:
            print(f"Skipping {folder.name}: {exc}")

    results.sort(
        key=lambda x: (x["green_min"], x["green_max"])
    )

    return results


# ============================================================
# Plot
# ============================================================

def plot_results(results):
    if not results:
        raise RuntimeError(
            "No benchmark results found. Expected directories such as "
            "benchmark_green_0.20_0.25 containing summary.txt."
        )

    labels = [r["label"] for r in results]

    maxflow_times = [
        r["maxflow_runtime"]
        for r in results
    ]

    mincost_times = [
        r["mincost_runtime"]
        for r in results
    ]

    x = list(range(len(results)))

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        maxflow_times,
        marker="o",
        label="Max Flow + Passes",
    )

    plt.plot(
        x,
        mincost_times,
        marker="o",
        label="Pure Min-Cost Flow",
    )

    plt.xticks(
        x,
        labels,
        rotation=45,
    )

    plt.xlabel("Green Share Range")
    plt.ylabel("Mean Runtime (ms)")
    plt.title(
        "Mean Runtime vs. Green Share\n"
        "Target Utilization = 0.60-0.65, Brown Share Uncontrolled"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
    )

    print()
    print(f"Plot saved to: {OUTPUT_FILE}")

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = collect_results()

    print("=" * 80)
    print("RUNTIME BY GREEN SHARE")
    print("=" * 80)

    for result in results:
        print(
            f"Green {result['label']} | "
            f"Max Flow + Passes: "
            f"{result['maxflow_runtime']:.6f} ms | "
            f"Pure Min-Cost Flow: "
            f"{result['mincost_runtime']:.6f} ms"
        )

    plot_results(results)
