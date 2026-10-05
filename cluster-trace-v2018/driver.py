#!/usr/bin/env python3

import argparse
import csv
import importlib.util
from collections import deque
from pathlib import Path


ALPHA = 2.0
EPS = 1e-9


# ============================================================
# Load existing MaxFlow-Passes implementation
# ============================================================

def load_solver(path: Path):
    import sys

    spec = importlib.util.spec_from_file_location(
        "maxflow_impl",
        path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load solver from {path}")

    module = importlib.util.module_from_spec(spec)

    # Important for @dataclass
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================
# Read tasks.csv
# ============================================================

def read_tasks(job_dir: Path):

    tasks_file = job_dir / "tasks.csv"

    tasks = {}

    with tasks_file.open(newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            task_name = row["task_name"]

            node_id = None

            if row["dag_node_id"].strip():
                node_id = int(row["dag_node_id"])

            predecessors = []

            pred_text = row["predecessors"].strip()

            if pred_text:
                predecessors = [
                    int(x)
                    for x in pred_text.split(";")
                    if x.strip()
                ]

            tasks[task_name] = {
                "task_name": task_name,
                "node_id": node_id,
                "predecessor_ids": predecessors,

                "instance_num": int(row["instance_num"]),
                "start_time": float(row["start_time"]),
                "original_end_time": float(row["end_time"]),

                "plan_cpu": (
                    float(row["plan_cpu"])
                    if row["plan_cpu"].strip()
                    else None
                ),

                "task_type": row["task_type"],
                "status": row["status"],
            }

    return tasks


# ============================================================
# Compute processing time from instance CSV
#
# Also records exactly why individual instances are skipped.
# ============================================================

def compute_processing(job_dir: Path, task):

    task_name = task["task_name"]

    instance_file = (
        job_dir
        / "instances"
        / f"{task_name}.csv"
    )

    if not instance_file.exists():
        raise FileNotFoundError(
            f"No instance file for {task_name}: "
            f"{instance_file}"
        )

    processing = 0.0
    valid_instances = 0
    missing_cpu = 0
    bad_duration = 0
    invalid_time = 0

    invalid_instances = []

    with instance_file.open(
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):

            # Try to identify the Alibaba instance.
            # If neither column exists, use its CSV row number.
            instance_name = (
                row.get("instance_name")
                or row.get("instance_id")
                or f"row_{row_number}"
            )

            # ------------------------------------------------
            # 1. Validate start/end time
            # ------------------------------------------------

            try:
                start = float(row["start_time"])
                end = float(row["end_time"])

            except (ValueError, TypeError, KeyError):

                invalid_time += 1

                invalid_instances.append({
                    "task_name": task_name,
                    "instance": instance_name,
                    "row_number": row_number,
                    "reason": "invalid_start_or_end_time",
                    "start_time": row.get("start_time", ""),
                    "end_time": row.get("end_time", ""),
                    "cpu_avg": row.get("cpu_avg", ""),
                })

                continue

            # ------------------------------------------------
            # 2. Validate duration
            # ------------------------------------------------

            duration = end - start

            if duration < 0:

                bad_duration += 1

                invalid_instances.append({
                    "task_name": task_name,
                    "instance": instance_name,
                    "row_number": row_number,
                    "reason": "negative_duration",
                    "start_time": start,
                    "end_time": end,
                    "cpu_avg": row.get("cpu_avg", ""),
                })

                continue

            # ------------------------------------------------
            # 3. Validate cpu_avg
            # ------------------------------------------------

            cpu_text = row.get("cpu_avg", "").strip()

            if cpu_text == "":

                missing_cpu += 1

                invalid_instances.append({
                    "task_name": task_name,
                    "instance": instance_name,
                    "row_number": row_number,
                    "reason": "missing_cpu_avg",
                    "start_time": start,
                    "end_time": end,
                    "cpu_avg": "",
                })

                continue

            try:
                cpu_avg = float(cpu_text)

            except ValueError:

                missing_cpu += 1

                invalid_instances.append({
                    "task_name": task_name,
                    "instance": instance_name,
                    "row_number": row_number,
                    "reason": "invalid_cpu_avg",
                    "start_time": start,
                    "end_time": end,
                    "cpu_avg": cpu_text,
                })

                continue

            # ------------------------------------------------
            # Valid instance
            # ------------------------------------------------

            processing += duration * cpu_avg / 100.0
            valid_instances += 1

    return {
        "processing": processing,
        "valid_instances": valid_instances,
        "missing_cpu_instances": missing_cpu,
        "bad_duration_instances": bad_duration,
        "invalid_time_instances": invalid_time,
        "invalid_instances": invalid_instances,
    }


# ============================================================
# Build DAG
# ============================================================

def build_dag(tasks):

    # node ID -> task name
    id_to_task = {}

    for name, task in tasks.items():

        node_id = task["node_id"]

        if node_id is not None:
            id_to_task[node_id] = name

    predecessors = {
        name: []
        for name in tasks
    }

    successors = {
        name: []
        for name in tasks
    }

    for name, task in tasks.items():

        for pred_id in task["predecessor_ids"]:

            # Sometimes an encoded predecessor might not be
            # present in the selected job data.
            if pred_id not in id_to_task:
                continue

            pred_name = id_to_task[pred_id]

            predecessors[name].append(pred_name)
            successors[pred_name].append(name)

    return predecessors, successors


# ============================================================
# Topological order
# ============================================================

def topological_order(tasks, predecessors, successors):

    indegree = {
        name: len(predecessors[name])
        for name in tasks
    }

    q = deque(
        name
        for name, degree in indegree.items()
        if degree == 0
    )

    order = []

    while q:

        u = q.popleft()
        order.append(u)

        for v in successors[u]:

            indegree[v] -= 1

            if indegree[v] == 0:
                q.append(v)

    if len(order) != len(tasks):

        raise ValueError(
            "DAG contains a cycle or invalid dependency structure."
        )

    return order


# ============================================================
# Effective release/deadline computation
# ============================================================

def compute_effective_windows(
    tasks,
    predecessors,
    successors,
    topo_order,
):

    # --------------------------------------------------------
    # Forward pass:
    #
    # R_v = max(
    #     original r_v,
    #     already-computed R of each predecessor
    # )
    # --------------------------------------------------------

    for name in topo_order:

        task = tasks[name]

        values = [task["release"]]

        for pred in predecessors[name]:

            values.append(
                tasks[pred]["effective_release"]
            )

        task["effective_release"] = max(values)

    # --------------------------------------------------------
    # Reverse pass:
    #
    # D_v = min(
    #     original d_v,
    #     already-computed D of each successor
    # )
    # --------------------------------------------------------

    for name in reversed(topo_order):

        task = tasks[name]

        values = [task["deadline"]]

        for succ in successors[name]:

            values.append(
                tasks[succ]["effective_deadline"]
            )

        task["effective_deadline"] = min(values)


# ============================================================
# Write invalid instance details
# ============================================================

def write_invalid_instances(job_dir, invalid_instances):

    output = job_dir / "invalid_instances.csv"

    fields = [
        "task_name",
        "instance",
        "row_number",
        "reason",
        "start_time",
        "end_time",
        "cpu_avg",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(invalid_instances)

    return output


# ============================================================
# Prepare one Alibaba job
# ============================================================

def preprocess_job(job_dir: Path):

    tasks = read_tasks(job_dir)

    all_invalid_instances = []

    # --------------------------------------------------------
    # Compute r, p, d
    # --------------------------------------------------------

    for name, task in tasks.items():

        stats = compute_processing(
            job_dir,
            task
        )

        # Collect detailed invalid-instance information.
        all_invalid_instances.extend(
            stats["invalid_instances"]
        )

        p = stats["processing"]

        task["processing"] = p

        task["valid_instances"] = (
            stats["valid_instances"]
        )

        task["missing_cpu_instances"] = (
            stats["missing_cpu_instances"]
        )

        task["bad_duration_instances"] = (
            stats["bad_duration_instances"]
        )

        task["invalid_time_instances"] = (
            stats["invalid_time_instances"]
        )

        # release = Alibaba stage/task start time
        r = task["start_time"]

        # deadline = r + alpha * p
        d = r + ALPHA * p

        task["release"] = r
        task["deadline"] = d

    # --------------------------------------------------------
    # Write detailed invalid-instance information
    # --------------------------------------------------------

    write_invalid_instances(
        job_dir,
        all_invalid_instances
    )

    # --------------------------------------------------------
    # DAG tightening
    # --------------------------------------------------------

    predecessors, successors = build_dag(tasks)

    topo_order = topological_order(
        tasks,
        predecessors,
        successors,
    )

    compute_effective_windows(
        tasks,
        predecessors,
        successors,
        topo_order,
    )

    return (
        tasks,
        predecessors,
        successors,
        topo_order,
    )


# ============================================================
# Save preprocessing results
# ============================================================

def write_processed_tasks(
    job_dir,
    tasks,
    predecessors,
    successors,
):

    output = job_dir / "processed_tasks.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fields = [
            "task_name",
            "instance_num",

            "raw_release",
            "raw_deadline",
            "processing",

            "effective_release",
            "effective_deadline",

            "predecessors",
            "successors",

            "valid_instances",
            "missing_cpu_instances",
            "bad_duration_instances",
            "invalid_time_instances",

            "original_start_time",
            "original_end_time",
            "plan_cpu",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for name, task in tasks.items():

            writer.writerow({
                "task_name":
                    name,

                "instance_num":
                    task["instance_num"],

                "raw_release":
                    task["release"],

                "raw_deadline":
                    task["deadline"],

                "processing":
                    task["processing"],

                "effective_release":
                    task["effective_release"],

                "effective_deadline":
                    task["effective_deadline"],

                "predecessors":
                    ";".join(predecessors[name]),

                "successors":
                    ";".join(successors[name]),

                "valid_instances":
                    task["valid_instances"],

                "missing_cpu_instances":
                    task["missing_cpu_instances"],

                "bad_duration_instances":
                    task["bad_duration_instances"],

                "invalid_time_instances":
                    task["invalid_time_instances"],

                "original_start_time":
                    task["start_time"],

                "original_end_time":
                    task["original_end_time"],

                "plan_cpu":
                    task["plan_cpu"],
            })

    return output


# ============================================================
# Convert to MaxFlow Job objects
# ============================================================

def make_solver_jobs(tasks, solver):

    jobs = []
    skipped_zero = []

    for name, task in tasks.items():

        p = task["processing"]

        # Solver requires deadline > release.
        #
        # p=0 gives d=r under the current definition.
        # Such a task requires no capacity, so it does not need
        # a source->job flow edge.
        if p <= EPS:

            skipped_zero.append(name)
            continue

        R = task["effective_release"]
        D = task["effective_deadline"]

        if D <= R + EPS:

            raise ValueError(
                f"{name}: DAG tightening produced invalid "
                f"window [{R}, {D}] with p={p}"
            )

        if p > D - R + EPS:

            raise ValueError(
                f"{name}: processing={p} exceeds effective "
                f"window length={D-R}"
            )

        jobs.append(
            solver.Job(
                name=name,
                release=R,
                deadline=D,
                processing=p,
            )
        )

    return jobs, skipped_zero


# ============================================================
# All-green energy profile
# ============================================================

def make_all_green_interval(jobs, solver):

    start = min(
        job.release
        for job in jobs
    )

    end = max(
        job.deadline
        for job in jobs
    )

    return [
        solver.EnergyInterval(
            name="all_green",
            start=start,
            end=end,
            energy="green",
        )
    ]


# ============================================================
# Run one Alibaba job folder
# ============================================================

def run_one_job(job_dir, solver):

    print()
    print("=" * 70)
    print(f"Alibaba instance: {job_dir.name}")
    print("=" * 70)

    (
        tasks,
        predecessors,
        successors,
        topo_order,
    ) = preprocess_job(job_dir)

    write_processed_tasks(
        job_dir,
        tasks,
        predecessors,
        successors,
    )

    solver_jobs, zero_tasks = make_solver_jobs(
        tasks,
        solver,
    )

    # Count invalid raw Alibaba instances.
    total_invalid_instances = sum(
        task["missing_cpu_instances"]
        + task["bad_duration_instances"]
        + task["invalid_time_instances"]
        for task in tasks.values()
    )

    print(
        f"Invalid raw instances: "
        f"{total_invalid_instances}"
    )

    if total_invalid_instances > 0:

        print(
            f"Invalid-instance details: "
            f"{job_dir / 'invalid_instances.csv'}"
        )

    if not solver_jobs:

        print(
            "No positive-processing tasks. "
            "Skipping MaxFlow run."
        )

        return {
            "alibaba_job":
                job_dir.name,

            "num_tasks":
                len(tasks),

            "num_solver_tasks":
                0,

            "zero_processing_tasks":
                len(zero_tasks),

            "invalid_instances":
                total_invalid_instances,

            "total_processing":
                0.0,

            "feasible":
                "",

            "flow":
                "",

            "runtime_seconds":
                "",
        }

    intervals = make_all_green_interval(
        solver_jobs,
        solver,
    )

    import time

    t0 = time.perf_counter()

    result = solver.max_flow_passes_schedule(
        solver_jobs,
        intervals,
        verbose=False,
    )

    runtime = time.perf_counter() - t0

    print(
        f"Tasks: {len(tasks)}, "
        f"solver tasks: {len(solver_jobs)}, "
        f"zero-work tasks: {len(zero_tasks)}"
    )

    print(
        f"Total processing: "
        f"{result['total_processing']:.6f}"
    )

    print(
        f"Flow: "
        f"{result['final']['flow']:.6f}"
    )

    print(
        f"Feasible: "
        f"{result['feasible']}"
    )

    print(
        f"Runtime: "
        f"{runtime:.6f} sec"
    )

    return {
        "alibaba_job":
            job_dir.name,

        "num_tasks":
            len(tasks),

        "num_solver_tasks":
            len(solver_jobs),

        "zero_processing_tasks":
            len(zero_tasks),

        "invalid_instances":
            total_invalid_instances,

        "total_processing":
            result["total_processing"],

        "feasible":
            result["feasible"],

        "flow":
            result["final"]["flow"],

        "runtime_seconds":
            runtime,
    }


# ============================================================
# Main driver
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="selected_jobs",
        help="Directory containing j_*/ folders",
    )

    parser.add_argument(
        "--solver",
        default="maxflow.py",
        help="Python file containing MaxFlow implementation",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Deadline slack multiplier: d = r + alpha * p",
    )

    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help=(
            "Skip an Alibaba job if processed_tasks.csv "
            "already exists in its folder."
        ),
    )

    args = parser.parse_args()

    global ALPHA
    ALPHA = args.alpha

    dataset_dir = Path(args.dataset)
    solver_path = Path(args.solver)

    solver = load_solver(solver_path)

    job_dirs = sorted(
        p
        for p in dataset_dir.iterdir()
        if p.is_dir()
        and (p / "tasks.csv").exists()
    )

    if not job_dirs:

        raise RuntimeError(
            f"No Alibaba job folders found in {dataset_dir}"
        )

    print(
        f"Found {len(job_dirs)} Alibaba experiment instances."
    )

    if args.skip_processed:
        print(
            "Resume mode enabled: previously processed "
            "jobs will be skipped."
        )

    results = []

    skipped_processed = 0

    for job_dir in job_dirs:

        # ----------------------------------------------------
        # Resume / skip previously processed jobs
        # ----------------------------------------------------

        processed_file = (
            job_dir
            / "processed_tasks.csv"
        )

        if (
            args.skip_processed
            and processed_file.exists()
        ):

            print(
                f"Skipping {job_dir.name}: "
                f"already processed."
            )

            skipped_processed += 1
            continue

        # ----------------------------------------------------
        # Process job
        # ----------------------------------------------------

        try:

            row = run_one_job(
                job_dir,
                solver,
            )

            results.append(row)

        except Exception as exc:

            print(
                f"ERROR for {job_dir.name}: {exc}"
            )

            results.append({
                "alibaba_job":
                    job_dir.name,

                "num_tasks":
                    "",

                "num_solver_tasks":
                    "",

                "zero_processing_tasks":
                    "",

                "invalid_instances":
                    "",

                "total_processing":
                    "",

                "feasible":
                    False,

                "flow":
                    "",

                "runtime_seconds":
                    "",

                "error":
                    str(exc),
            })

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_file = (
        dataset_dir
        / "maxflow_summary.csv"
    )

    fields = [
        "alibaba_job",
        "num_tasks",
        "num_solver_tasks",
        "zero_processing_tasks",
        "invalid_instances",
        "total_processing",
        "feasible",
        "flow",
        "runtime_seconds",
        "error",
    ]

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in results:

            row.setdefault(
                "error",
                ""
            )

            writer.writerow(row)

    print()
    print("=" * 70)
    print("Experiment complete")
    print("=" * 70)

    print(
        f"New jobs processed: "
        f"{len(results)}"
    )

    print(
        f"Previously processed jobs skipped: "
        f"{skipped_processed}"
    )

    print(
        f"Summary written to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()