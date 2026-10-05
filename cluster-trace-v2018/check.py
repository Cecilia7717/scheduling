#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv("all_job_stats.csv")

df = df[
    (df["num_tasks"] >= 2) &
    (df["horizon"] >= 6) &
    (df["total_instances"] > 0) &
    (df["invalid_task_rows"] == 0)
].copy()

bins = [
    (80, 99),
    (100, 119),
    (120, 139),
    (140, 159),
    (160, 179),
    (180, 199),
    (200, 219),
    (220, 239),
    (240, 260),
    (261, 300),
    (301, 400),
    (401, 500),
    (501, 750),
    (751, 1000),
    (1001, 2000),
]

print()
print("Cheap candidate availability")
print("=" * 65)

for lo, hi in bins:
    x = df[
        (df["num_tasks"] >= lo) &
        (df["num_tasks"] <= hi)
    ]

    print(
        f"{lo:4d}-{hi:<4d}: "
        f"{len(x):6,d} jobs"
    )