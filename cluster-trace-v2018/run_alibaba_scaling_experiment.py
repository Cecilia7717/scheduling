#!/usr/bin/env python3

"""
run_alibaba_scaling_experiment.py

Runtime/scaling experiment on trace-derived Alibaba DAG workloads.

For each materialized workload:

    1. Reconstruct task release/deadline/CPU work from instance CSVs.
    2. Scale processing times to fixed utilization U=0.70.
    3. Tighten task windows using DAG precedence.
    4. Construct the exact scheduling jobs used by maxflow.py.
    5. Measure actual flow-network dimensions.
    6. Run:
           - Incremental MaxFlow + Passes
           - Pure Min-Cost Flow
    7. Repeat each solver several times.
    8. Save both aggregate and per-repetition timing results.

Primary scaling variable:
    actual number of DAG tasks.

Additional network-size variables:
    number of elementary intervals
    number of job-interval edges
    number of flow-network vertices
    number of forward flow-network edges

The complete solver call is timed. Therefore runtime includes:
    interval splitting,
    network construction,
    flow computation,
    result recovery,
    timeline construction.

Alibaba preprocessing itself is NOT included in solver runtime.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


from maxflow import (
    Job,
    EnergyInterval,
    max_flow_passes_schedule,
    pure_min_cost_schedule,
    split_intervals_at_job_boundaries,
)


EPS = 1e-9

DEFAULT_UTILIZATION = 0.70
DEFAULT_REPEATS = 5


# ============================================================
# Alibaba task representation
# ============================================================

@dataclass
class AlibabaTask:
    task_name: str
    node_id: int
    predecessors: List[int]

    expected_instances: int
    usable_instances: int

    raw_release: float
    raw_deadline: float
    cpu_work: float

    processing: float = 0.0

    effective_release: float = 0.0
    effective_deadline: float = 0.0


# ============================================================
# Basic helpers
# ============================================================

def as_float(value, default=None):
    try:
        x = float(value)
        if not math.isfinite(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_predecessors(value: str) -> List[int]:

    if value is None:
        return []

    s = str(value).strip()

    if not s or s == "[]":
        return []

    s = (
        s.strip("[](){}")
        .replace(";", ",")
        .replace("_", ",")
    )

    result = []

    for piece in s.split(","):

        piece = piece.strip().strip("'\"")

        if not piece:
            continue

        result.append(int(piece))

    return result


def read_csv(path: Path) -> List[dict]:

    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# Instance loading
# ============================================================

def find_instance_file(job_dir: Path, task_name: str) -> Path:

    direct = job_dir / "instances" / f"{task_name}.csv"

    if direct.exists():
        return direct

    instance_dir = job_dir / "instances"

    if not instance_dir.exists():
        raise FileNotFoundError(
            f"Missing instance directory: {instance_dir}"
        )

    for path in instance_dir.glob("*.csv"):

        if path.stem == task_name:
            return path

    raise FileNotFoundError(
        f"No instance CSV found for task "
        f"{task_name!r} in {instance_dir}"
    )


def compute_instance_stats(
    instance_path: Path,
) -> Tuple[float, float, float, int, int]:

    raw_release = math.inf
    raw_deadline = -math.inf

    cpu_work = 0.0

    usable = 0
    rejected = 0

    with instance_path.open("r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            start = as_float(row.get("start_time"))
            end = as_float(row.get("end_time"))
            cpu = as_float(row.get("cpu_avg"))

            if (
                start is None
                or end is None
                or cpu is None
                or end <= start + EPS
                or cpu <= 0.0
            ):
                rejected += 1
                continue

            duration = end - start

            raw_release = min(raw_release, start)
            raw_deadline = max(raw_deadline, end)

            cpu_work += duration * cpu

            usable += 1

    if usable == 0:
        raise ValueError(
            f"No usable instances in {instance_path}"
        )

    if cpu_work <= EPS:
        raise ValueError(
            f"Non-positive CPU work in {instance_path}"
        )

    return (
        raw_release,
        raw_deadline,
        cpu_work,
        usable,
        rejected,
    )


# ============================================================
# Load Alibaba DAG
# ============================================================

def load_alibaba_tasks(
    job_dir: Path,
) -> Tuple[List[AlibabaTask], dict]:

    tasks_path = job_dir / "tasks.csv"

    if not tasks_path.exists():
        raise FileNotFoundError(
            f"Missing {tasks_path}"
        )

    task_rows = read_csv(tasks_path)

    if not task_rows:
        raise ValueError(
            f"No tasks in {tasks_path}"
        )

    tasks: List[AlibabaTask] = []

    node_ids = set()

    total_instance_rows = 0
    total_rejected_instance_rows = 0

    for row in task_rows:

        task_name = row["task_name"].strip()

        node_id = as_int(
            row.get("dag_node_id")
        )

        if node_id is None:
            raise ValueError(
                f"{job_dir.name}/{task_name}: "
                f"missing/invalid dag_node_id"
            )

        if node_id in node_ids:
            raise ValueError(
                f"{job_dir.name}: duplicate DAG node id "
                f"{node_id}"
            )

        node_ids.add(node_id)

        predecessors = parse_predecessors(
            row.get("predecessors", "")
        )

        expected_instances = as_int(
            row.get("expected_instances"),
            0,
        )

        instance_path = find_instance_file(
            job_dir,
            task_name,
        )

        (
            release,
            deadline,
            cpu_work,
            usable,
            rejected,
        ) = compute_instance_stats(instance_path)

        total_instance_rows += usable + rejected
        total_rejected_instance_rows += rejected

        tasks.append(
            AlibabaTask(
                task_name=task_name,
                node_id=node_id,
                predecessors=predecessors,
                expected_instances=expected_instances,
                usable_instances=usable,
                raw_release=release,
                raw_deadline=deadline,
                cpu_work=cpu_work,
            )
        )

    id_set = {
        t.node_id
        for t in tasks
    }

    for task in tasks:

        missing = [
            p
            for p in task.predecessors
            if p not in id_set
        ]

        if missing:
            raise ValueError(
                f"{job_dir.name}/{task.task_name}: "
                f"missing predecessor node(s) "
                f"{missing}"
            )

    metadata = {
        "raw_instance_rows":
            total_instance_rows,

        "rejected_instance_rows":
            total_rejected_instance_rows,
    }

    return tasks, metadata


# ============================================================
# DAG operations
# ============================================================

def build_dag(
    tasks: List[AlibabaTask],
):

    by_id = {
        t.node_id: t
        for t in tasks
    }

    successors: Dict[int, List[int]] = defaultdict(list)

    indegree = {
        t.node_id: 0
        for t in tasks
    }

    for task in tasks:

        for pred in task.predecessors:

            successors[pred].append(
                task.node_id
            )

            indegree[task.node_id] += 1

    q = deque(
        sorted(
            node
            for node, deg in indegree.items()
            if deg == 0
        )
    )

    topo = []

    while q:

        u = q.popleft()

        topo.append(u)

        for v in sorted(successors[u]):

            indegree[v] -= 1

            if indegree[v] == 0:
                q.append(v)

    if len(topo) != len(tasks):

        raise ValueError(
            "DAG contains a cycle."
        )

    return (
        by_id,
        successors,
        topo,
    )


# ============================================================
# Processing-time scaling
# ============================================================

def scale_processing_times(
    tasks: List[AlibabaTask],
    utilization: float,
    horizon_start: float,
    horizon_end: float,
) -> float:

    horizon = (
        horizon_end
        - horizon_start
    )

    if horizon <= EPS:
        raise ValueError(
            "Non-positive job horizon."
        )

    total_work = sum(
        t.cpu_work
        for t in tasks
    )

    if total_work <= EPS:
        raise ValueError(
            "Total CPU work is non-positive."
        )

    target_processing = (
        utilization
        * horizon
    )

    alpha = (
        target_processing
        / total_work
    )

    for task in tasks:

        task.processing = (
            alpha
            * task.cpu_work
        )

    return alpha


# ============================================================
# Effective windows
# ============================================================

def tighten_effective_windows(
    tasks: List[AlibabaTask],
) -> Tuple[bool, str]:

    (
        by_id,
        successors,
        topo,
    ) = build_dag(tasks)

    # Forward pass.
    for node in topo:

        task = by_id[node]

        r = task.raw_release

        for pred in task.predecessors:

            pred_task = by_id[pred]

            r = max(
                r,
                pred_task.effective_release
                + pred_task.processing,
            )

        task.effective_release = r

    # Backward pass.
    for node in reversed(topo):

        task = by_id[node]

        d = task.raw_deadline

        for succ in successors[node]:

            succ_task = by_id[succ]

            d = min(
                d,
                succ_task.effective_deadline
                - succ_task.processing,
            )

        task.effective_deadline = d

    # Validate windows.
    for task in tasks:

        window = (
            task.effective_deadline
            - task.effective_release
        )

        if window <= EPS:

            return (
                False,
                f"{task.task_name}: "
                f"non-positive effective window "
                f"[{task.effective_release}, "
                f"{task.effective_deadline}]",
            )

        if task.processing > window + EPS:

            return (
                False,
                f"{task.task_name}: "
                f"processing="
                f"{task.processing:.6f} "
                f"> effective window="
                f"{window:.6f}",
            )

    return True, ""


# ============================================================
# Energy intervals
# ============================================================

def load_energy_intervals(
    energy_path: Path,
    horizon_start: float,
    horizon_end: float,
) -> List[EnergyInterval]:

    rows = read_csv(energy_path)

    if not rows:
        raise ValueError(
            f"No energy intervals in {energy_path}"
        )

    intervals = []

    for idx, row in enumerate(rows):

        start = as_float(
            row.get("start_time")
        )

        end = as_float(
            row.get("end_time")
        )

        energy = (
            row.get("energy_type")
            or row.get("energy")
            or ""
        ).strip().lower()

        if (
            start is None
            or end is None
            or end <= start + EPS
        ):
            raise ValueError(
                f"Invalid energy interval row "
                f"{idx} in {energy_path}"
            )

        if energy not in {
            "green",
            "brown",
            "red",
        }:
            raise ValueError(
                f"Invalid energy type "
                f"{energy!r}"
            )

        intervals.append(
            EnergyInterval(
                name=f"E{idx}",
                start=start,
                end=end,
                energy=energy,
            )
        )

    intervals.sort(
        key=lambda x: (
            x.start,
            x.end,
        )
    )

    if (
        horizon_start
        < intervals[0].start - EPS
    ):
        raise ValueError(
            "Task horizon starts before "
            "energy horizon."
        )

    if (
        horizon_end
        > intervals[-1].end + EPS
    ):
        raise ValueError(
            "Task horizon ends after "
            "energy horizon."
        )

    return intervals


# ============================================================
# Convert to MaxFlow jobs
# ============================================================

def make_maxflow_jobs(
    tasks: List[AlibabaTask],
) -> List[Job]:

    return [
        Job(
            name=t.task_name,
            release=t.effective_release,
            deadline=t.effective_deadline,
            processing=t.processing,
        )
        for t in tasks
    ]


# ============================================================
# Network statistics
# ============================================================

def compute_network_stats(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
):

    elementary = split_intervals_at_job_boundaries(
        jobs,
        original_intervals,
    )

    num_jobs = len(jobs)

    num_intervals = len(elementary)

    num_job_interval_edges = 0

    for job in jobs:

        for interval in elementary:

            if (
                job.release
                <= interval.start + EPS
                and
                interval.end
                <= job.deadline + EPS
            ):
                num_job_interval_edges += 1

    # Forward edges:
    #
    # source -> job
    # job -> elementary interval
    # elementary interval -> sink
    #
    num_forward_edges = (
        num_jobs
        + num_job_interval_edges
        + num_intervals
    )

    # source + jobs + intervals + sink
    num_vertices = (
        num_jobs
        + num_intervals
        + 2
    )

    possible_job_interval_edges = (
        num_jobs
        * num_intervals
    )

    if possible_job_interval_edges > 0:

        edge_density = (
            num_job_interval_edges
            / possible_job_interval_edges
        )

    else:

        edge_density = 0.0

    return {
        "num_elementary_intervals":
            num_intervals,

        "num_job_interval_edges":
            num_job_interval_edges,

        "num_network_vertices":
            num_vertices,

        "num_forward_edges":
            num_forward_edges,

        "job_interval_edge_density":
            edge_density,
    }


# ============================================================
# Timing
# ============================================================

def benchmark_solver(
    solver,
    jobs,
    intervals,
    repeats: int,
):

    # --------------------------------------------------------
    # Untimed correctness/result run
    # --------------------------------------------------------

    result = solver(
        jobs,
        intervals,
        verbose=False,
    )

    # --------------------------------------------------------
    # Timed repetitions
    # --------------------------------------------------------

    times = []

    for _ in range(repeats):

        start = time.perf_counter()

        solver(
            jobs,
            intervals,
            verbose=False,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

    if not times:

        raise ValueError(
            "At least one timing repetition is required."
        )

    return {
        "result": result,

        "times": times,

        "mean_seconds":
            statistics.mean(times),

        "median_seconds":
            statistics.median(times),

        "min_seconds":
            min(times),

        "max_seconds":
            max(times),

        "std_seconds":
            (
                statistics.stdev(times)
                if len(times) >= 2
                else 0.0
            ),
    }


# ============================================================
# Job discovery
# ============================================================

def discover_jobs(root: Path):

    jobs = []

    # Search recursively because the builder may organize
    # workloads by sampling band.
    for tasks_path in root.rglob("tasks.csv"):

        job_dir = tasks_path.parent

        if not (
            job_dir / "energy.csv"
        ).exists():
            continue

        if not (
            job_dir / "instances"
        ).is_dir():
            continue

        jobs.append(job_dir)

    # Remove accidental duplicates.
    unique = {
        str(path.resolve()): path
        for path in jobs
    }

    jobs = list(unique.values())

    # Sort initially by path.
    jobs.sort(
        key=lambda p: str(p)
    )

    return jobs


# ============================================================
# One workload
# ============================================================

def process_one(
    job_dir: Path,
    utilization: float,
    repeats: int,
):

    # --------------------------------------------------------
    # Load trace-derived workload
    # --------------------------------------------------------

    (
        tasks,
        instance_metadata,
    ) = load_alibaba_tasks(job_dir)

    actual_start = min(
        t.raw_release
        for t in tasks
    )

    actual_end = max(
        t.raw_deadline
        for t in tasks
    )

    horizon = (
        actual_end
        - actual_start
    )

    total_cpu_work = sum(
        t.cpu_work
        for t in tasks
    )

    total_usable_instances = sum(
        t.usable_instances
        for t in tasks
    )

    # --------------------------------------------------------
    # Processing scaling
    # --------------------------------------------------------

    alpha = scale_processing_times(
        tasks,
        utilization,
        actual_start,
        actual_end,
    )

    total_processing = sum(
        t.processing
        for t in tasks
    )

    # --------------------------------------------------------
    # Effective windows
    # --------------------------------------------------------

    window_ok, reason = (
        tighten_effective_windows(tasks)
    )

    if not window_ok:

        raise RuntimeError(
            f"Previously selected workload became "
            f"window-infeasible: {reason}"
        )

    # --------------------------------------------------------
    # Energy intervals
    # --------------------------------------------------------

    intervals = load_energy_intervals(
        job_dir / "energy.csv",
        actual_start,
        actual_end,
    )

    # --------------------------------------------------------
    # Convert to scheduling jobs
    # --------------------------------------------------------

    jobs = make_maxflow_jobs(tasks)

    # --------------------------------------------------------
    # Network statistics
    # --------------------------------------------------------

    network_stats = compute_network_stats(
        jobs,
        intervals,
    )

    # --------------------------------------------------------
    # MaxFlow + Passes
    # --------------------------------------------------------

    passes = benchmark_solver(
        max_flow_passes_schedule,
        jobs,
        intervals,
        repeats,
    )

    # --------------------------------------------------------
    # Pure Min-Cost Flow
    # --------------------------------------------------------

    mincost = benchmark_solver(
        pure_min_cost_schedule,
        jobs,
        intervals,
        repeats,
    )

    p_result = passes["result"]
    m_result = mincost["result"]

    # --------------------------------------------------------
    # Correctness checks
    # --------------------------------------------------------

    if (
        p_result["feasible"]
        != m_result["feasible"]
    ):

        raise RuntimeError(
            "MaxFlow and MinCost disagree "
            "on feasibility."
        )

    cost_difference = abs(
        p_result["normalized_cost"]
        - m_result["normalized_cost"]
    )

    cost_scale = max(
        1.0,
        abs(p_result["normalized_cost"]),
        abs(m_result["normalized_cost"]),
    )

    cost_agree = (
        cost_difference
        <= 1e-7 * cost_scale
    )

    if not cost_agree:

        raise RuntimeError(
            "MaxFlow and MinCost disagree "
            f"on cost: "
            f"{p_result['normalized_cost']} vs "
            f"{m_result['normalized_cost']}"
        )

    # --------------------------------------------------------
    # Runtime ratio
    # --------------------------------------------------------

    if passes["median_seconds"] > 0:

        runtime_ratio = (
            mincost["median_seconds"]
            / passes["median_seconds"]
        )

    else:

        runtime_ratio = math.inf

    # --------------------------------------------------------
    # Energy usage
    # --------------------------------------------------------

    pu = (
        p_result["final"]
        ["energy_usage"]
    )

    mu = (
        m_result["final"]
        ["energy_usage"]
    )

    # --------------------------------------------------------
    # Aggregate result
    # --------------------------------------------------------

    row = {

        "job_name":
            job_dir.name,

        "job_path":
            str(job_dir),

        "num_tasks":
            len(tasks),

        "usable_instances":
            total_usable_instances,

        "raw_instance_rows":
            instance_metadata[
                "raw_instance_rows"
            ],

        "rejected_instance_rows":
            instance_metadata[
                "rejected_instance_rows"
            ],

        "actual_start":
            actual_start,

        "actual_end":
            actual_end,

        "horizon":
            horizon,

        "utilization":
            utilization,

        "total_cpu_work":
            total_cpu_work,

        "scale_alpha":
            alpha,

        "total_processing":
            total_processing,

        **network_stats,

        "maxflow_feasible":
            p_result["feasible"],

        "mincost_feasible":
            m_result["feasible"],

        "maxflow_cost":
            p_result[
                "normalized_cost"
            ],

        "mincost_cost":
            m_result[
                "normalized_cost"
            ],

        "cost_difference":
            cost_difference,

        "maxflow_green":
            pu["green"],

        "maxflow_brown":
            pu["brown"],

        "maxflow_red":
            pu["red"],

        "mincost_green":
            mu["green"],

        "mincost_brown":
            mu["brown"],

        "mincost_red":
            mu["red"],

        "maxflow_mean_seconds":
            passes["mean_seconds"],

        "maxflow_median_seconds":
            passes["median_seconds"],

        "maxflow_min_seconds":
            passes["min_seconds"],

        "maxflow_max_seconds":
            passes["max_seconds"],

        "maxflow_std_seconds":
            passes["std_seconds"],

        "mincost_mean_seconds":
            mincost["mean_seconds"],

        "mincost_median_seconds":
            mincost["median_seconds"],

        "mincost_min_seconds":
            mincost["min_seconds"],

        "mincost_max_seconds":
            mincost["max_seconds"],

        "mincost_std_seconds":
            mincost["std_seconds"],

        "mincost_over_maxflow":
            runtime_ratio,
    }

    # --------------------------------------------------------
    # Raw repetition rows
    # --------------------------------------------------------

    timing_rows = []

    for algorithm, timing in (
        ("maxflow_passes", passes),
        ("mincost", mincost),
    ):

        for repeat_index, seconds in enumerate(
            timing["times"],
            start=1,
        ):

            timing_rows.append(
                {
                    "job_name":
                        job_dir.name,

                    "num_tasks":
                        len(tasks),

                    "utilization":
                        utilization,

                    "num_elementary_intervals":
                        network_stats[
                            "num_elementary_intervals"
                        ],

                    "num_job_interval_edges":
                        network_stats[
                            "num_job_interval_edges"
                        ],

                    "algorithm":
                        algorithm,

                    "repeat":
                        repeat_index,

                    "seconds":
                        seconds,
                }
            )

    return row, timing_rows


# ============================================================
# CSV writing
# ============================================================

SUMMARY_FIELDS = [

    "job_name",
    "job_path",

    "num_tasks",
    "usable_instances",

    "raw_instance_rows",
    "rejected_instance_rows",

    "actual_start",
    "actual_end",
    "horizon",

    "utilization",

    "total_cpu_work",
    "scale_alpha",
    "total_processing",

    "num_elementary_intervals",
    "num_job_interval_edges",
    "num_network_vertices",
    "num_forward_edges",
    "job_interval_edge_density",

    "maxflow_feasible",
    "mincost_feasible",

    "maxflow_cost",
    "mincost_cost",
    "cost_difference",

    "maxflow_green",
    "maxflow_brown",
    "maxflow_red",

    "mincost_green",
    "mincost_brown",
    "mincost_red",

    "maxflow_mean_seconds",
    "maxflow_median_seconds",
    "maxflow_min_seconds",
    "maxflow_max_seconds",
    "maxflow_std_seconds",

    "mincost_mean_seconds",
    "mincost_median_seconds",
    "mincost_min_seconds",
    "mincost_max_seconds",
    "mincost_std_seconds",

    "mincost_over_maxflow",
]


TIMING_FIELDS = [

    "job_name",
    "num_tasks",
    "utilization",

    "num_elementary_intervals",
    "num_job_interval_edges",

    "algorithm",
    "repeat",
    "seconds",
]


def write_rows(
    path: Path,
    fields,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "alibaba_scaling_jobs"
        ),
        help=(
            "Root containing materialized "
            "Alibaba scaling workloads."
        ),
    )

    parser.add_argument(
        "--utilization",
        type=float,
        default=DEFAULT_UTILIZATION,
        help=(
            "Fixed utilization for the "
            "scaling experiment."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=(
            "Timed repetitions per solver."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<root>/runtime_results"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if not (
        0.0
        < args.utilization
        <= 1.0
    ):
        raise ValueError(
            "Utilization must satisfy "
            "0 < U <= 1."
        )

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )

    root = args.root

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else root / "runtime_results"
    )

    # --------------------------------------------------------
    # Discover workloads
    # --------------------------------------------------------

    job_dirs = discover_jobs(root)

    if not job_dirs:

        raise RuntimeError(
            f"No materialized workloads "
            f"found under {root}"
        )

    # --------------------------------------------------------
    # Determine task counts first so execution order is
    # increasing DAG size.
    # --------------------------------------------------------

    jobs_with_sizes = []

    for job_dir in job_dirs:

        tasks_path = (
            job_dir / "tasks.csv"
        )

        rows = read_csv(tasks_path)

        jobs_with_sizes.append(
            (
                len(rows),
                job_dir.name,
                job_dir,
            )
        )

    jobs_with_sizes.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 100)

    print(
        "ALIBABA REAL-DATA "
        "SCALING EXPERIMENT"
    )

    print("=" * 100)

    print(
        f"Root:          {root}"
    )

    print(
        f"Workloads:     "
        f"{len(jobs_with_sizes)}"
    )

    print(
        f"Utilization:   "
        f"{args.utilization:.2f}"
    )

    print(
        f"Timed repeats: "
        f"{args.repeats}"
    )

    print(
        f"Output:        "
        f"{output_dir}"
    )

    print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    summary_rows = []
    timing_rows = []
    failure_rows = []

    total = len(jobs_with_sizes)

    for index, (
        expected_tasks,
        job_name,
        job_dir,
    ) in enumerate(
        jobs_with_sizes,
        start=1,
    ):

        print(
            f"[{index:>3}/{total}] "
            f"{job_name:<15} "
            f"tasks={expected_tasks:<4}",
            end="  ",
            flush=True,
        )

        try:

            (
                row,
                raw_timings,
            ) = process_one(
                job_dir=job_dir,
                utilization=args.utilization,
                repeats=args.repeats,
            )

            summary_rows.append(row)
            timing_rows.extend(
                raw_timings
            )

            print(
                f"I={row['num_elementary_intervals']:<4} "
                f"E={row['num_job_interval_edges']:<7} "
                f"passes="
                f"{1000 * row['maxflow_median_seconds']:.3f} ms  "
                f"mincost="
                f"{1000 * row['mincost_median_seconds']:.3f} ms  "
                f"ratio="
                f"{row['mincost_over_maxflow']:.2f}x"
            )

        except Exception as exc:

            failure_rows.append(
                {
                    "job_name":
                        job_name,

                    "job_path":
                        str(job_dir),

                    "expected_tasks":
                        expected_tasks,

                    "error":
                        str(exc),
                }
            )

            print(
                f"ERROR: {exc}"
            )

    # --------------------------------------------------------
    # Sort final results by actual task count
    # --------------------------------------------------------

    summary_rows.sort(
        key=lambda r: (
            r["num_tasks"],
            r["job_name"],
        )
    )

    timing_rows.sort(
        key=lambda r: (
            r["num_tasks"],
            r["job_name"],
            r["algorithm"],
            r["repeat"],
        )
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "runtime_summary.csv"
    )

    timing_path = (
        output_dir
        / "runtime_repetitions.csv"
    )

    failure_path = (
        output_dir
        / "runtime_failures.csv"
    )

    write_rows(
        summary_path,
        SUMMARY_FIELDS,
        summary_rows,
    )

    write_rows(
        timing_path,
        TIMING_FIELDS,
        timing_rows,
    )

    if failure_rows:

        write_rows(
            failure_path,
            [
                "job_name",
                "job_path",
                "expected_tasks",
                "error",
            ],
            failure_rows,
        )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"Successful workloads: "
        f"{len(summary_rows)}"
    )

    print(
        f"Failed workloads:     "
        f"{len(failure_rows)}"
    )

    if summary_rows:

        min_tasks = min(
            r["num_tasks"]
            for r in summary_rows
        )

        max_tasks = max(
            r["num_tasks"]
            for r in summary_rows
        )

        ratios = [
            r["mincost_over_maxflow"]
            for r in summary_rows
            if math.isfinite(
                r["mincost_over_maxflow"]
            )
        ]

        print(
            f"DAG-size range:       "
            f"{min_tasks} to "
            f"{max_tasks}"
        )

        if ratios:

            print(
                f"Median MinCost/"
                f"MaxFlow ratio: "
                f"{statistics.median(ratios):.3f}x"
            )

    print()
    print(
        f"Summary:     "
        f"{summary_path}"
    )

    print(
        f"Repetitions: "
        f"{timing_path}"
    )

    if failure_rows:

        print(
            f"Failures:    "
            f"{failure_path}"
        )


if __name__ == "__main__":
    main()