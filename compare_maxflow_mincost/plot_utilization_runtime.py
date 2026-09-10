from pathlib import Path
import re
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# Directory containing folders such as:
# benchmark_util_0.50_0.55/
# benchmark_util_0.55_0.60/
# ...
BASE_DIR = Path(__file__).resolve().parent

# Output image
OUTPUT_FILE = BASE_DIR / "runtime_vs_utilization.png"


# ============================================================
# Read one summary.txt
# ============================================================

def read_summary(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")

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

    if util_match is None:
        raise ValueError(
            f"Could not find utilization range in {summary_path}"
        )

    if maxflow_match is None:
        raise ValueError(
            f"Could not find Max Flow + Passes mean runtime in {summary_path}"
        )

    if mincost_match is None:
        raise ValueError(
            f"Could not find Pure Min-Cost Flow mean runtime in {summary_path}"
        )

    util_min = float(util_match.group(1))
    util_max = float(util_match.group(2))

    maxflow_runtime = float(maxflow_match.group(1))
    mincost_runtime = float(mincost_match.group(1))

    return {
        "util_min": util_min,
        "util_max": util_max,
        "label": f"{util_min:.2f}-{util_max:.2f}",
        "maxflow_runtime": maxflow_runtime,
        "mincost_runtime": mincost_runtime,
    }


# ============================================================
# Find all experiment folders
# ============================================================

def collect_results():
    results = []

    for folder in BASE_DIR.glob("benchmark_util_*"):
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

    # Sort by lower endpoint of utilization range.
    results.sort(key=lambda x: (x["util_min"], x["util_max"]))

    return results


# ============================================================
# Plot
# ============================================================

def plot_results(results):
    if not results:
        raise RuntimeError(
            "No benchmark results found. Expected directories such as "
            "benchmark_util_0.50_0.55 containing summary.txt."
        )

    labels = [r["label"] for r in results]
    maxflow_times = [r["maxflow_runtime"] for r in results]
    mincost_times = [r["mincost_runtime"] for r in results]

    x = list(range(len(results)))

    plt.figure(figsize=(10, 6))

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

    plt.xticks(x, labels, rotation=45)

    plt.xlabel("Target Utilization Range")
    plt.ylabel("Mean Runtime (ms)")
    plt.title("Mean Runtime vs. Target Utilization")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300)

    print()
    print(f"Plot saved to: {OUTPUT_FILE}")

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = collect_results()

    print("=" * 72)
    print("RUNTIME RESULTS")
    print("=" * 72)

    for result in results:
        print(
            f"{result['label']} | "
            f"Max Flow + Passes: {result['maxflow_runtime']:.6f} ms | "
            f"Pure Min-Cost Flow: {result['mincost_runtime']:.6f} ms"
        )

    plot_results(results)
