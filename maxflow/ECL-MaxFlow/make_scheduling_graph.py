jobs = [
    ("J1", 0, 8, 2),
    ("J2", 0, 15, 5),
    ("J3", 0, 16, 2),
    ("J6", 0, 16, 5),
    ("J4", 18, 30, 5),
    ("J5", 20, 35, 3),
    ("J7", 25, 40, 10),
]

intervals = [
    ("E0", 0, 5),
    ("E1", 5, 10),
    ("E2", 10, 15),
    ("E3", 15, 20),
    ("E4", 20, 25),
    ("E5", 25, 30),
    ("E6", 30, 35),
    ("E7", 35, 40),
]

source = 0

job_start = 1
interval_start = job_start + len(jobs)
sink = interval_start + len(intervals)

edges = []

# source -> jobs
for j, (_, r, d, p) in enumerate(jobs):
    job_node = job_start + j
    edges.append((source, job_node, p))

# jobs -> intervals
for j, (_, r, d, p) in enumerate(jobs):
    job_node = job_start + j

    for i, (_, a, b) in enumerate(intervals):
        interval_node = interval_start + i

        overlap = max(0, min(d, b) - max(r, a))

        if overlap > 0:
            edges.append(
                (job_node, interval_node, overlap)
            )

# intervals -> sink
for i, (_, a, b) in enumerate(intervals):
    interval_node = interval_start + i
    capacity = b - a

    edges.append(
        (interval_node, sink, capacity)
    )

print("source =", source)
print("sink =", sink)
print("nodes =", sink + 1)
print("edges =", len(edges))

# DIMACS is 1-based
with open("scheduling.dimacs", "w") as f:
    f.write("c scheduling max flow instance\n")
    f.write(f"p max {sink + 1} {len(edges)}\n")

    f.write(f"n {source + 1} s\n")
    f.write(f"n {sink + 1} t\n")

    for u, v, cap in edges:
        f.write(f"a {u + 1} {v + 1} {cap}\n")