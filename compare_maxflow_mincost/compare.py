from maxflow import (
    Job,
    EnergyInterval,
    max_flow_passes_schedule,
    pure_min_cost_schedule,
    compare_algorithms,
)

jobs = [
    # Early jobs: all gone by time 16
    Job("J1", release=0,  deadline=8,  processing=2),
    Job("J2", release=0,  deadline=15, processing=5),
    Job("J3", release=0,  deadline=16, processing=2),
    Job("J6", release=0,  deadline=16, processing=5),

    # Later jobs: none released before time 18
    Job("J4", release=18, deadline=30, processing=5),
    Job("J5", release=20, deadline=35, processing=3),
    Job("J7", release=25, deadline=40, processing=10),
]

energy_intervals = [
    EnergyInterval("E0", 0,  5,  "brown"),
    EnergyInterval("E1", 5,  10, "brown"),

    # Split the old [10,20] green region so that [16,18]
    # is a separate green interval.
    EnergyInterval("E2", 10, 16, "green"),

    # Nobody is active here:
    # early jobs have deadline <= 16
    # later jobs have release >= 18
    EnergyInterval("E3", 16, 18, "green"),

    EnergyInterval("E4", 18, 20, "green"),

    EnergyInterval("E5", 20, 25, "brown"),
    EnergyInterval("E6", 25, 30, "red"),
    EnergyInterval("E7", 30, 35, "brown"),
    EnergyInterval("E8", 35, 40, "green"),
]

if __name__ == "__main__":
    print("\nRUN 1: INCREMENTAL MAX FLOW + PASSES")
    max_flow_passes_schedule(
        jobs,
        energy_intervals,
        verbose=True,
    )

    print("\n\nRUN 2: PURE MIN-COST FLOW")
    pure_min_cost_schedule(
        jobs,
        energy_intervals,
        verbose=True,
    )

    print("\n\nSIDE-BY-SIDE SUMMARY")
    compare_algorithms(
        jobs,
        energy_intervals,
        benchmark_repeats=0,
        verbose=True,
    )