#!/usr/bin/env python3
"""
run_alibaba_experiment.py

Convert the selected Alibaba DAG jobs into single-machine scheduling instances
and run the existing maxflow.py algorithms.

Expected input layout:

alibaba_30_jobs/
    small/
        j_xxx/
            tasks.csv
            instances/
                <task_name>.csv
            energy.csv
    medium/
        ...
    large/
        ...

For each Alibaba task i:
    raw release  r_i = min usable instance start_time
    raw deadline d_i = max usable instance end_time
    CPU work     w_i = sum_k (end_k - start_k) * cpu_avg_k

For each requested utilization U, CPU work is scaled proportionally:
    p_i = alpha * w_i
    alpha = U * H / sum_i w_i
where H is the actual job horizon.

Precedence is incorporated by tightening windows:
    effective release:
        R_i = max(r_i, max_{pred j}(R_j + p_j))
    effective deadline:
        D_i = min(d_i, min_{succ j}(D_j - p_j))

The resulting (R_i, D_i, p_i) jobs are passed to maxflow.py.

NOTE:
The effective-window formulas above assume the precedence interpretation
"predecessor must complete before successor starts." If your theoretical
transformation uses a different formula, edit tighten_effective_windows().
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
)

EPS = 1e-9
DEFAULT_UTILIZATIONS = (0.50, 0.70, 0.90)


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
    """
    Accept formats such as:
        ""
        "[]"
        "[1, 2]"
        "1,2"
        "1_2"
        "1;2"
    """
    if value is None:
        return []

    s = str(value).strip()
    if not s or s == "[]":
        return []

    s = s.strip("[](){}").replace(";", ",").replace("_", ",")
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


def find_instance_file(job_dir: Path, task_name: str) -> Path:
    direct = job_dir / "instances" / f"{task_name}.csv"
    if direct.exists():
        return direct

    # Fallback in case filenames were sanitized during preprocessing.
    instance_dir = job_dir / "instances"
    if not instance_dir.exists():
        raise FileNotFoundError(f"Missing instance directory: {instance_dir}")

    for path in instance_dir.glob("*.csv"):
        if path.stem == task_name:
            return path

    raise FileNotFoundError(
        f"No instance CSV found for task {task_name!r} in {instance_dir}"
    )


def compute_instance_stats(instance_path: Path) -> Tuple[float, float, float, int, int]:
    """
    Return:
        raw_release, raw_deadline, cpu_work,
        usable_rows, rejected_rows
    """
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
        raise ValueError(f"No usable instances in {instance_path}")

    if cpu_work <= EPS:
        raise ValueError(f"Non-positive CPU work in {instance_path}")

    return raw_release, raw_deadline, cpu_work, usable, rejected


def load_alibaba_tasks(job_dir: Path) -> Tuple[List[AlibabaTask], dict]:
    tasks_path = job_dir / "tasks.csv"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing {tasks_path}")

    task_rows = read_csv(tasks_path)
    if not task_rows:
        raise ValueError(f"No tasks in {tasks_path}")

    tasks: List[AlibabaTask] = []
    node_ids = set()

    total_instance_rows = 0
    total_rejected_instance_rows = 0

    for row in task_rows:
        task_name = row["task_name"].strip()

        node_id = as_int(row.get("dag_node_id"))
        if node_id is None:
            raise ValueError(
                f"{job_dir.name}/{task_name}: missing/invalid dag_node_id"
            )

        if node_id in node_ids:
            raise ValueError(
                f"{job_dir.name}: duplicate DAG node id {node_id}"
            )
        node_ids.add(node_id)

        predecessors = parse_predecessors(row.get("predecessors", ""))

        expected_instances = as_int(row.get("expected_instances"), 0)
        stored_usable = as_int(row.get("usable_instances"), 0)

        instance_path = find_instance_file(job_dir, task_name)
        release, deadline, cpu_work, usable, rejected = compute_instance_stats(
            instance_path
        )

        total_instance_rows += usable + rejected
        total_rejected_instance_rows += rejected

        # We recompute usable rows from the actual instance file rather than
        # trusting the stored count.
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

        if stored_usable and stored_usable != usable:
            print(
                f"WARNING {job_dir.name}/{task_name}: "
                f"tasks.csv usable_instances={stored_usable}, "
                f"recomputed={usable}"
            )

    id_set = {t.node_id for t in tasks}
    for task in tasks:
        missing = [p for p in task.predecessors if p not in id_set]
        if missing:
            raise ValueError(
                f"{job_dir.name}/{task.task_name}: "
                f"missing predecessor node(s) {missing}"
            )

    metadata = {
        "raw_instance_rows": total_instance_rows,
        "rejected_instance_rows": total_rejected_instance_rows,
    }

    return tasks, metadata


def build_dag(tasks: List[AlibabaTask]):
    by_id = {t.node_id: t for t in tasks}
    successors: Dict[int, List[int]] = defaultdict(list)
    indegree = {t.node_id: 0 for t in tasks}

    for task in tasks:
        for pred in task.predecessors:
            successors[pred].append(task.node_id)
            indegree[task.node_id] += 1

    q = deque(sorted(node for node, deg in indegree.items() if deg == 0))
    topo = []

    while q:
        u = q.popleft()
        topo.append(u)

        for v in sorted(successors[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    if len(topo) != len(tasks):
        raise ValueError("DAG contains a cycle.")

    return by_id, successors, topo


def scale_processing_times(
    tasks: List[AlibabaTask],
    utilization: float,
    horizon_start: float,
    horizon_end: float,
) -> float:
    """
    Scale CPU work so total processing equals utilization * job horizon.

    Returns alpha.
    """
    horizon = horizon_end - horizon_start
    if horizon <= EPS:
        raise ValueError("Non-positive job horizon.")

    total_work = sum(t.cpu_work for t in tasks)
    if total_work <= EPS:
        raise ValueError("Total CPU work is non-positive.")

    target_processing = utilization * horizon
    alpha = target_processing / total_work

    for task in tasks:
        task.processing = alpha * task.cpu_work

    return alpha


def tighten_effective_windows(tasks: List[AlibabaTask]) -> Tuple[bool, str]:
    """
    Tighten releases/deadlines using precedence.

    Forward:
        R_i = max(raw_r_i, R_pred + p_pred)

    Backward:
        D_i = min(raw_d_i, D_succ - p_succ)

    This enforces completion-before-start precedence after window tightening.
    """
    by_id, successors, topo = build_dag(tasks)

    # Forward pass.
    for node in topo:
        task = by_id[node]
        r = task.raw_release

        for pred in task.predecessors:
            pred_task = by_id[pred]
            r = max(r, pred_task.effective_release + pred_task.processing)

        task.effective_release = r

    # Backward pass.
    for node in reversed(topo):
        task = by_id[node]
        d = task.raw_deadline

        for succ in successors[node]:
            succ_task = by_id[succ]
            d = min(d, succ_task.effective_deadline - succ_task.processing)

        task.effective_deadline = d

    # Check each tightened task window.
    for task in tasks:
        window = task.effective_deadline - task.effective_release

        if window <= EPS:
            return (
                False,
                f"{task.task_name}: non-positive effective window "
                f"[{task.effective_release}, {task.effective_deadline}]",
            )

        if task.processing > window + EPS:
            return (
                False,
                f"{task.task_name}: processing={task.processing:.6f} "
                f"> effective window={window:.6f}",
            )

    return True, ""


def load_energy_intervals(
    energy_path: Path,
    horizon_start: float,
    horizon_end: float,
) -> List[EnergyInterval]:
    """
    Read energy.csv.

    Expected columns include:
        start_time,end_time,energy_type

    The selected-job preprocessing created six intervals covering the
    actual usable-instance horizon.
    """
    rows = read_csv(energy_path)
    if not rows:
        raise ValueError(f"No energy intervals in {energy_path}")

    intervals = []

    for idx, row in enumerate(rows):
        start = as_float(row.get("start_time"))
        end = as_float(row.get("end_time"))
        energy = (row.get("energy_type") or row.get("energy") or "").strip().lower()

        if start is None or end is None or end <= start + EPS:
            raise ValueError(f"Invalid energy interval row {idx} in {energy_path}")

        if energy not in {"green", "brown", "red"}:
            raise ValueError(
                f"Invalid energy type {energy!r} in {energy_path}, row {idx}"
            )

        intervals.append(
            EnergyInterval(
                name=f"E{idx}",
                start=start,
                end=end,
                energy=energy,
            )
        )

    intervals.sort(key=lambda x: (x.start, x.end))

    # The MaxFlow code requires the scheduling windows to lie inside the
    # energy horizon.
    if horizon_start < intervals[0].start - EPS:
        raise ValueError(
            f"Task horizon begins at {horizon_start}, "
            f"before energy horizon {intervals[0].start}"
        )

    if horizon_end > intervals[-1].end + EPS:
        raise ValueError(
            f"Task horizon ends at {horizon_end}, "
            f"after energy horizon {intervals[-1].end}"
        )

    return intervals


def clone_tasks(tasks: List[AlibabaTask]) -> List[AlibabaTask]:
    return [
        AlibabaTask(
            task_name=t.task_name,
            node_id=t.node_id,
            predecessors=list(t.predecessors),
            expected_instances=t.expected_instances,
            usable_instances=t.usable_instances,
            raw_release=t.raw_release,
            raw_deadline=t.raw_deadline,
            cpu_work=t.cpu_work,
        )
        for t in tasks
    ]


def make_maxflow_jobs(tasks: List[AlibabaTask]) -> List[Job]:
    return [
        Job(
            name=t.task_name,
            release=t.effective_release,
            deadline=t.effective_deadline,
            processing=t.processing,
        )
        for t in tasks
    ]


def time_solver(solver, jobs, intervals, repeats: int):
    """
    Run once for the actual result, then benchmark fresh solver calls.

    The solver reconstructs its network each call, so this is complete
    algorithm wall time for the Python implementation.
    """
    t0 = time.perf_counter()
    result = solver(jobs, intervals, verbose=False)
    first_seconds = time.perf_counter() - t0

    timings = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        solver(jobs, intervals, verbose=False)
        timings.append(time.perf_counter() - t0)

    if timings:
        mean_seconds = statistics.mean(timings)
        median_seconds = statistics.median(timings)
    else:
        mean_seconds = first_seconds
        median_seconds = first_seconds

    return result, first_seconds, mean_seconds, median_seconds


def write_scheduling_csv(
    path: Path,
    tasks: List[AlibabaTask],
    utilization: float,
    alpha: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "task_name",
        "node_id",
        "predecessors",
        "expected_instances",
        "usable_instances",
        "raw_release",
        "raw_deadline",
        "raw_window",
        "effective_release",
        "effective_deadline",
        "effective_window",
        "cpu_work",
        "scale_alpha",
        "utilization",
        "processing_time",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for t in sorted(tasks, key=lambda x: x.node_id):
            writer.writerow(
                {
                    "task_name": t.task_name,
                    "node_id": t.node_id,
                    "predecessors": ",".join(map(str, t.predecessors)),
                    "expected_instances": t.expected_instances,
                    "usable_instances": t.usable_instances,
                    "raw_release": t.raw_release,
                    "raw_deadline": t.raw_deadline,
                    "raw_window": t.raw_deadline - t.raw_release,
                    "effective_release": t.effective_release,
                    "effective_deadline": t.effective_deadline,
                    "effective_window": (
                        t.effective_deadline - t.effective_release
                    ),
                    "cpu_work": t.cpu_work,
                    "scale_alpha": alpha,
                    "utilization": utilization,
                    "processing_time": t.processing,
                }
            )


def discover_jobs(root: Path):
    jobs = []

    for group in ("small", "medium", "large"):
        group_dir = root / group
        if not group_dir.exists():
            continue

        for job_dir in sorted(group_dir.iterdir()):
            if (
                job_dir.is_dir()
                and (job_dir / "tasks.csv").exists()
                and (job_dir / "energy.csv").exists()
            ):
                jobs.append((group, job_dir))

    return jobs


def process_one(
    size_group: str,
    job_dir: Path,
    utilizations: Tuple[float, ...],
    repeats: int,
):
    base_tasks, instance_metadata = load_alibaba_tasks(job_dir)

    actual_start = min(t.raw_release for t in base_tasks)
    actual_end = max(t.raw_deadline for t in base_tasks)
    horizon = actual_end - actual_start

    total_cpu_work = sum(t.cpu_work for t in base_tasks)
    total_usable_instances = sum(t.usable_instances for t in base_tasks)

    intervals = load_energy_intervals(
        job_dir / "energy.csv",
        actual_start,
        actual_end,
    )

    rows = []

    for utilization in utilizations:
        tasks = clone_tasks(base_tasks)

        alpha = scale_processing_times(
            tasks,
            utilization,
            actual_start,
            actual_end,
        )

        window_ok, window_reason = tighten_effective_windows(tasks)

        scheduling_dir = job_dir / "scheduling"
        scheduling_path = scheduling_dir / f"util_{utilization:.2f}.csv"
        write_scheduling_csv(
            scheduling_path,
            tasks,
            utilization,
            alpha,
        )

        total_processing = sum(t.processing for t in tasks)

        common = {
            "size_group": size_group,
            "job_name": job_dir.name,
            "utilization": utilization,
            "num_tasks": len(tasks),
            "usable_instances": total_usable_instances,
            "raw_instance_rows": instance_metadata["raw_instance_rows"],
            "rejected_instance_rows": instance_metadata[
                "rejected_instance_rows"
            ],
            "actual_start": actual_start,
            "actual_end": actual_end,
            "horizon": horizon,
            "total_cpu_work": total_cpu_work,
            "scale_alpha": alpha,
            "total_processing": total_processing,
            "window_feasible": window_ok,
            "window_failure_reason": window_reason,
        }

        if not window_ok:
            rows.append(
                {
                    **common,
                    "maxflow_feasible": False,
                    "mincost_feasible": False,
                    "maxflow_cost": "",
                    "mincost_cost": "",
                    "maxflow_green": "",
                    "maxflow_brown": "",
                    "maxflow_red": "",
                    "mincost_green": "",
                    "mincost_brown": "",
                    "mincost_red": "",
                    "maxflow_first_seconds": "",
                    "maxflow_mean_seconds": "",
                    "maxflow_median_seconds": "",
                    "mincost_first_seconds": "",
                    "mincost_mean_seconds": "",
                    "mincost_median_seconds": "",
                }
            )

            print(
                f"{job_dir.name:>12}  U={utilization:.2f}  "
                f"WINDOW-INFEASIBLE  {window_reason}"
            )
            continue

        jobs = make_maxflow_jobs(tasks)

        passes, p_first, p_mean, p_median = time_solver(
            max_flow_passes_schedule,
            jobs,
            intervals,
            repeats,
        )

        pure, m_first, m_mean, m_median = time_solver(
            pure_min_cost_schedule,
            jobs,
            intervals,
            repeats,
        )

        pu = passes["final"]["energy_usage"]
        mu = pure["final"]["energy_usage"]

        rows.append(
            {
                **common,
                "maxflow_feasible": passes["feasible"],
                "mincost_feasible": pure["feasible"],
                "maxflow_cost": passes["normalized_cost"],
                "mincost_cost": pure["normalized_cost"],
                "maxflow_green": pu["green"],
                "maxflow_brown": pu["brown"],
                "maxflow_red": pu["red"],
                "mincost_green": mu["green"],
                "mincost_brown": mu["brown"],
                "mincost_red": mu["red"],
                "maxflow_first_seconds": p_first,
                "maxflow_mean_seconds": p_mean,
                "maxflow_median_seconds": p_median,
                "mincost_first_seconds": m_first,
                "mincost_mean_seconds": m_mean,
                "mincost_median_seconds": m_median,
            }
        )

        print(
            f"{job_dir.name:>12}  U={utilization:.2f}  "
            f"passes={passes['feasible']}  "
            f"mincost={pure['feasible']}  "
            f"cost={passes['normalized_cost']:.4f}  "
            f"passes={1000*p_median:.3f} ms  "
            f"mincost={1000*m_median:.3f} ms"
        )

    return rows


def write_summary(path: Path, rows: List[dict]):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "size_group",
        "job_name",
        "utilization",
        "num_tasks",
        "usable_instances",
        "raw_instance_rows",
        "rejected_instance_rows",
        "actual_start",
        "actual_end",
        "horizon",
        "total_cpu_work",
        "scale_alpha",
        "total_processing",
        "window_feasible",
        "window_failure_reason",
        "maxflow_feasible",
        "mincost_feasible",
        "maxflow_cost",
        "mincost_cost",
        "maxflow_green",
        "maxflow_brown",
        "maxflow_red",
        "mincost_green",
        "mincost_brown",
        "mincost_red",
        "maxflow_first_seconds",
        "maxflow_mean_seconds",
        "maxflow_median_seconds",
        "mincost_first_seconds",
        "mincost_mean_seconds",
        "mincost_median_seconds",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("alibaba_30_jobs"),
        help="Root containing small/, medium/, and large/ selected jobs.",
    )

    parser.add_argument(
        "--utilizations",
        type=float,
        nargs="+",
        default=list(DEFAULT_UTILIZATIONS),
        help="Target single-machine utilizations.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timing repetitions per solver after the first run.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output summary CSV. Default: <root>/experiment_summary.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    utilizations = tuple(args.utilizations)

    for u in utilizations:
        if not (0.0 < u <= 1.0):
            raise ValueError(f"Invalid utilization {u}; expected 0 < U <= 1.")

    if args.repeats < 0:
        raise ValueError("--repeats must be >= 0")

    root = args.root
    summary_path = (
        args.summary
        if args.summary is not None
        else root / "experiment_summary.csv"
    )

    selected = discover_jobs(root)

    if not selected:
        raise RuntimeError(
            f"No selected jobs found under {root}. "
            "Expected small/, medium/, and large/ directories."
        )

    print("=" * 88)
    print("ALIBABA SINGLE-MACHINE MAXFLOW EXPERIMENT")
    print("=" * 88)
    print(f"Root:          {root}")
    print(f"Jobs found:    {len(selected)}")
    print(f"Utilizations:  {', '.join(f'{u:.2f}' for u in utilizations)}")
    print(f"Timing repeats:{args.repeats}")
    print()

    all_rows = []
    failed_jobs = []

    for index, (size_group, job_dir) in enumerate(selected, start=1):
        print(
            f"\n[{index}/{len(selected)}] "
            f"{size_group}/{job_dir.name}"
        )

        try:
            rows = process_one(
                size_group=size_group,
                job_dir=job_dir,
                utilizations=utilizations,
                repeats=args.repeats,
            )
            all_rows.extend(rows)

        except Exception as exc:
            failed_jobs.append(
                {
                    "size_group": size_group,
                    "job_name": job_dir.name,
                    "error": str(exc),
                }
            )
            print(f"ERROR: {exc}")

    write_summary(summary_path, all_rows)

    if failed_jobs:
        failed_path = root / "experiment_failures.csv"
        with failed_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["size_group", "job_name", "error"],
            )
            writer.writeheader()
            writer.writerows(failed_jobs)

        print(f"\nFailures written to: {failed_path}")

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Experiment rows: {len(all_rows)}")
    print(f"Failed jobs:     {len(failed_jobs)}")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
