import csv


# ------------------------------------------------------------
# Same job order used when constructing the ECL graph
# ------------------------------------------------------------

jobs = [
    ("J1", 0, 8, 2),
    ("J2", 0, 15, 5),
    ("J3", 0, 16, 2),
    ("J6", 0, 16, 5),
    ("J4", 18, 30, 5),
    ("J5", 20, 35, 3),
    ("J7", 25, 40, 10),
]


# ------------------------------------------------------------
# Same energy intervals used when constructing the graph
# ------------------------------------------------------------

intervals = [
    ("E0", 0, 5, "brown"),
    ("E1", 5, 10, "brown"),
    ("E2", 10, 15, "green"),
    ("E3", 15, 20, "green"),
    ("E4", 20, 25, "brown"),
    ("E5", 25, 30, "green"),
    ("E6", 30, 35, "green"),
    ("E7", 35, 40, "brown"),
]


# ------------------------------------------------------------
# Node numbering used in make_scheduling_graph.py
#
# 0                     = source
# 1 ... len(jobs)       = jobs
# following nodes       = intervals
# last node             = sink
# ------------------------------------------------------------

job_start = 1
interval_start = job_start + len(jobs)

job_nodes = {
    job_start + i: job[0]
    for i, job in enumerate(jobs)
}

interval_nodes = {
    interval_start + i: interval
    for i, interval in enumerate(intervals)
}


# ------------------------------------------------------------
# Read ECL output
# ------------------------------------------------------------

input_file = "flow_output.csv"
output_file = "schedule_output.csv"

schedule_rows = []

with open(input_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        u = int(row["from"])
        v = int(row["to"])
        flow = int(row["flow"])

        # We only care about positive Job -> Interval flow.
        if flow <= 0:
            continue

        if u not in job_nodes:
            continue

        if v not in interval_nodes:
            continue

        job_name = job_nodes[u]

        interval_name, start, end, energy = interval_nodes[v]

        schedule_rows.append(
            {
                "Job": job_name,
                "Interval": interval_name,
                "Start": start,
                "End": end,
                "Energy": energy,
                "Flow": flow,
            }
        )


# ------------------------------------------------------------
# Sort by job, then time
# ------------------------------------------------------------

job_order = {
    job[0]: i
    for i, job in enumerate(jobs)
}

schedule_rows.sort(
    key=lambda x: (
        job_order[x["Job"]],
        x["Start"],
    )
)


# ------------------------------------------------------------
# Write clean scheduling CSV
# ------------------------------------------------------------

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Job",
            "Interval",
            "Start",
            "End",
            "Energy",
            "Flow",
        ],
    )

    writer.writeheader()
    writer.writerows(schedule_rows)


# ------------------------------------------------------------
# Also print a readable table
# ------------------------------------------------------------

print()
print(
    f"{'Job':<6}"
    f"{'Interval':<10}"
    f"{'Start':<8}"
    f"{'End':<8}"
    f"{'Energy':<10}"
    f"{'Flow':<8}"
)

print("-" * 50)

for row in schedule_rows:
    print(
        f"{row['Job']:<6}"
        f"{row['Interval']:<10}"
        f"{row['Start']:<8}"
        f"{row['End']:<8}"
        f"{row['Energy']:<10}"
        f"{row['Flow']:<8}"
    )

print()
print(f"Schedule written to: {output_file}")