#!/usr/bin/env python3
"""
build_alibaba_scaling_jobs.py

Build a real-data Alibaba scaling benchmark for:

    MaxFlow + Passes vs. Min-Cost Flow

The benchmark is NOT organized around artificial target task counts.

Instead:

    * Each selected Alibaba job is one natural scheduling workload.
    * Its actual DAG task count is preserved.
    * Runtime should later be plotted against the ACTUAL number of DAG tasks.
    * Dense small-size regions are sampled.
    * Sparse large-size regions are searched much more aggressively / exhaustively.
    * A job is selected only if it is actually feasible under MaxFlow + Passes
      at the fixed utilization U = 0.70.

Scheduling conversion is kept consistent with run_alibaba_experiment.py.

For each Alibaba task i:

    raw release:
        r_i = min usable instance start_time

    raw deadline:
        d_i = max usable instance end_time

    CPU work:
        w_i = sum_k (end_k - start_k) * cpu_avg_k

Processing times:

    p_i = alpha * w_i

where:

    alpha = U * H / sum_i w_i

and H is the actual usable-instance horizon.

Precedence tightening:

    R_i = max(r_i, max_pred(R_pred + p_pred))

    D_i = min(d_i, min_succ(D_succ - p_succ))

A candidate is accepted only if:

    1. Task names expose the DAG node/predecessor structure.
    2. DAG is structurally valid and acyclic.
    3. Every task has at least one usable instance.
    4. Actual usable-instance horizon >= 6.
    5. Effective windows are valid.
    6. Every task individually fits its effective window.
    7. MaxFlow + Passes reports the complete workload feasible.

Default sampling/search bands:

      10-19      -> keep up to 5 feasible jobs
      20-39      -> keep up to 5 feasible jobs
      40-79      -> keep up to 5 feasible jobs
      80-139     -> keep up to 5 feasible jobs

     140-300     -> search ALL cheap candidates, keep ALL feasible jobs
     301-500     -> search ALL cheap candidates, keep ALL feasible jobs
     501-750     -> search ALL cheap candidates, keep ALL feasible jobs
     751-1000    -> search ALL cheap candidates, keep ALL feasible jobs
    1001-2000    -> search ALL cheap candidates, keep ALL feasible jobs

For small/dense bands, the script first builds a larger feasible pool and
then chooses jobs spread across actual DAG size and horizon.

For large/sparse bands, every feasible natural Alibaba job is retained.

Output:

    alibaba_scaling_jobs/
        j_xxx/
            tasks.csv
            instances/
                <task>.csv
            energy.csv
            selection_info.csv

        summary.csv
        feasible_candidate_pool.csv
        rejected_candidates.csv
        band_summary.csv

Expected inputs:

    all_job_stats.csv

    data/batch_task.csv

    data/instances_by_job/
        j_xxx.csv
        ...

Per-job instance CSV expected columns:

    instance_name
    task_name
    start_time
    end_time
    cpu_avg
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from maxflow import (
    Job,
    EnergyInterval,
    max_flow_passes_schedule,
)


# ============================================================
# Constants
# ============================================================

EPS = 1e-9

DEFAULT_UTILIZATION = 0.70


# These are SAMPLING/SEARCH BANDS only.
#
# They are NOT target task counts and should not be used as the x-axis.
#
# Final experiments should use:
#
#     x = actual_tasks
#
# for every selected Alibaba workload.
#
#
# mode:
#
#   "sample"
#       Dense region. Build a feasible pool and select a representative
#       subset.
#
#   "all"
#       Sparse region. Evaluate every cheap candidate and retain every
#       feasible workload.
#
BANDS = [
    {
        "name": "10_19",
        "min_tasks": 10,
        "max_tasks": 19,
        "mode": "sample",
        "keep": 20,
        "preferred_feasible_pool": 100,
        "max_candidates": 5000,
    },
    {
        "name": "20_39",
        "min_tasks": 20,
        "max_tasks": 39,
        "mode": "sample",
        "keep": 20,
        "preferred_feasible_pool": 100,
        "max_candidates": 5000,
    },
    {
        "name": "40_59",
        "min_tasks": 40,
        "max_tasks": 59,
        "mode": "sample",
        "keep": 20,
        "preferred_feasible_pool": 100,
        "max_candidates": 5000,
    },
    {
        "name": "60_79",
        "min_tasks": 60,
        "max_tasks": 79,
        "mode": "sample",
        "keep": 20,
        "preferred_feasible_pool": 100,
        "max_candidates": 5000,
    },
    {
        "name": "80_99",
        "min_tasks": 80,
        "max_tasks": 99,
        "mode": "sample",
        "keep": 20,
        "preferred_feasible_pool": 100,
        "max_candidates": None,
    },

    # Sparse ranges: test everything and keep everything feasible.
    {
        "name": "100_119",
        "min_tasks": 100,
        "max_tasks": 119,
        "mode": "all",
        "keep": 0,
    },
    {
        "name": "120_139",
        "min_tasks": 120,
        "max_tasks": 139,
        "mode": "all",
        "keep": 0,
    },
    {
        "name": "140_plus",
        "min_tasks": 140,
        "max_tasks": 1000000,
        "mode": "all",
        "keep": 0,
    },
]
TASK_COLUMNS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]


INSTANCE_OUTPUT_FIELDS = [
    "instance_name",
    "start_time",
    "end_time",
    "cpu_avg",
]


# ============================================================
# Data classes
# ============================================================

@dataclass
class TaskRecord:
    task_name: str
    node_id: int
    predecessors: List[int]

    expected_instances: int

    task_type: str
    status: str

    plan_cpu: float
    plan_mem: float

    raw_release: float = 0.0
    raw_deadline: float = 0.0

    cpu_work: float = 0.0

    usable_instances: int = 0
    rejected_instances: int = 0

    processing: float = 0.0

    effective_release: float = 0.0
    effective_deadline: float = 0.0


@dataclass
class CandidateResult:
    band_name: str

    job_name: str

    actual_tasks: int
    total_instances_stats: int
    cheap_horizon: float

    valid: bool = False
    feasible: bool = False

    reason: str = ""

    actual_start: float = 0.0
    actual_end: float = 0.0
    actual_horizon: float = 0.0

    usable_instances: int = 0
    rejected_instances: int = 0

    total_cpu_work: float = 0.0
    total_processing: float = 0.0

    scale_alpha: float = 0.0

    maxflow_cost: float = 0.0

    maxflow_green: float = 0.0
    maxflow_brown: float = 0.0
    maxflow_red: float = 0.0

    tasks: Optional[List[TaskRecord]] = None

    instance_rows_by_task: Optional[
        Dict[str, List[dict]]
    ] = None


# ============================================================
# Generic helpers
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


def numeric_job_id(job_name: str) -> int:
    match = re.search(
        r"(\d+)$",
        str(job_name),
    )

    if not match:
        return 10**18

    return int(match.group(1))


# ============================================================
# Alibaba task-name parser
# ============================================================

def parse_task_name(task_name: str):
    """
    Parse DAG task names such as:

        M1
        R2_1
        M5_1_2

        J99_1_8_11_13_16_18_21_24_27_29_32_35_42_44_46_49_

    A trailing underscore is tolerated.

    Opaque names such as:

        task_LTQzNDAxMzg3NzYwNDk5MzI1ODE=

    are intentionally rejected because they do not expose the DAG
    predecessor structure required by this experiment.
    """

    if task_name is None:
        return None, None

    task_name = str(task_name).strip()

    if not task_name:
        return None, None

    # Alibaba contains otherwise valid names ending with "_".
    task_name = task_name.rstrip("_")

    match = re.match(
        r"^[A-Za-z](\d+)(?:_(.*))?$",
        task_name,
    )

    if not match:
        return None, None

    node_id = int(
        match.group(1)
    )

    predecessor_text = match.group(2)

    if (
        predecessor_text is None
        or predecessor_text == ""
    ):
        return node_id, []

    predecessors = []

    for piece in predecessor_text.split("_"):

        piece = piece.strip()

        if not piece:
            continue

        try:
            predecessors.append(
                int(piece)
            )

        except ValueError:
            return None, None

    return node_id, predecessors


# ============================================================
# Band helper
# ============================================================

def find_band(num_tasks: int):
    for band in BANDS:

        if (
            band["min_tasks"]
            <= num_tasks
            <= band["max_tasks"]
        ):
            return band

    return None


# ============================================================
# Load cheap candidates
# ============================================================

def load_candidate_stats(
    stats_path: Path,
):
    """
    Cheap filters:

        num_tasks >= 10
        horizon >= 6
        total_instances > 0
        invalid_task_rows == 0

    Candidate is then assigned to exactly one search band.
    """

    candidates_by_band = {
        band["name"]: []
        for band in BANDS
    }

    with stats_path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            job_name = (
                row.get("job_name", "")
                .strip()
            )

            num_tasks = as_int(
                row.get("num_tasks")
            )

            total_instances = as_int(
                row.get("total_instances"),
                0,
            )

            horizon = as_float(
                row.get("horizon"),
                0.0,
            )

            invalid_rows = as_int(
                row.get(
                    "invalid_task_rows"
                ),
                0,
            )

            if not job_name:
                continue

            if (
                num_tasks is None
                or num_tasks < 10
            ):
                continue

            if total_instances <= 0:
                continue

            if (
                horizon is None
                or horizon < 6
            ):
                continue

            if invalid_rows != 0:
                continue

            band = find_band(
                num_tasks
            )

            if band is None:
                continue

            candidates_by_band[
                band["name"]
            ].append(
                {
                    "job_name": job_name,
                    "num_tasks": num_tasks,
                    "total_instances": (
                        total_instances
                    ),
                    "horizon": horizon,
                }
            )

    return candidates_by_band


# ============================================================
# Candidate ordering
# ============================================================

def order_sample_candidates(
    rows: List[dict],
):
    """
    Dense bands.

    We want candidates spread across:

        * actual task count
        * horizon

    rather than simply taking the first N rows.

    The ordering is deterministic.
    """

    if not rows:
        return []

    rows = sorted(
        rows,
        key=lambda r: (
            r["num_tasks"],
            r["horizon"],
            r["total_instances"],
            numeric_job_id(
                r["job_name"]
            ),
        ),
    )

    # Divide into task-count buckets.
    by_tasks = defaultdict(list)

    for row in rows:
        by_tasks[
            row["num_tasks"]
        ].append(row)

    task_counts = sorted(
        by_tasks
    )

    # Within each task count, alternate low/high horizons.
    per_task_order = {}

    for task_count in task_counts:

        group = sorted(
            by_tasks[task_count],
            key=lambda r: (
                r["horizon"],
                r["total_instances"],
                numeric_job_id(
                    r["job_name"]
                ),
            ),
        )

        ordered_group = []

        left = 0
        right = len(group) - 1

        while left <= right:

            if left == right:
                ordered_group.append(
                    group[left]
                )

            else:
                ordered_group.append(
                    group[left]
                )

                ordered_group.append(
                    group[right]
                )

            left += 1
            right -= 1

        per_task_order[
            task_count
        ] = ordered_group

    # Round-robin across task counts.
    result = []

    indices = {
        task_count: 0
        for task_count in task_counts
    }

    while True:

        added = False

        for task_count in task_counts:

            idx = indices[
                task_count
            ]

            group = per_task_order[
                task_count
            ]

            if idx < len(group):

                result.append(
                    group[idx]
                )

                indices[
                    task_count
                ] += 1

                added = True

        if not added:
            break

    return result


def order_all_candidates(
    rows: List[dict],
):
    """
    Sparse large-job bands.

    Evaluate all cheap candidates.

    Sorting only affects execution order.
    """

    return sorted(
        rows,
        key=lambda r: (
            r["num_tasks"],
            r["horizon"],
            r["total_instances"],
            numeric_job_id(
                r["job_name"]
            ),
        ),
    )


# ============================================================
# Scan batch_task.csv once
# ============================================================

def load_task_rows_for_candidates(
    batch_task_path: Path,
    candidate_names: set,
):
    """
    Scan batch_task.csv once and retain only candidate jobs.
    """

    result = defaultdict(list)

    print()
    print(
        "Scanning batch_task.csv "
        "for candidate jobs..."
    )

    print(
        f"Candidate jobs: "
        f"{len(candidate_names):,}"
    )

    with batch_task_path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            fieldnames=TASK_COLUMNS,
        )

        for row_number, row in enumerate(
            reader,
            start=1,
        ):

            job_name = (
                row.get("job_name")
                or ""
            ).strip()

            if job_name in candidate_names:

                result[
                    job_name
                ].append(row)

            if (
                row_number
                % 5_000_000
                == 0
            ):
                print(
                    f"  scanned "
                    f"{row_number:,} "
                    f"task rows"
                )

    print(
        f"Found task rows for "
        f"{len(result):,} "
        f"candidate jobs."
    )

    return result


# ============================================================
# DAG construction / validation
# ============================================================

def build_dag(
    tasks: List[TaskRecord],
):

    by_id = {
        task.node_id: task
        for task in tasks
    }

    if len(by_id) != len(tasks):
        raise ValueError(
            "duplicate DAG node id"
        )

    successors = defaultdict(list)

    indegree = {
        task.node_id: 0
        for task in tasks
    }

    for task in tasks:

        for pred in task.predecessors:

            if pred not in by_id:

                raise ValueError(
                    f"{task.task_name}: "
                    f"missing predecessor "
                    f"{pred}"
                )

            successors[
                pred
            ].append(
                task.node_id
            )

            indegree[
                task.node_id
            ] += 1

    q = deque(
        sorted(
            node
            for node, degree
            in indegree.items()
            if degree == 0
        )
    )

    topo = []

    while q:

        node = q.popleft()

        topo.append(node)

        for succ in sorted(
            successors[node]
        ):

            indegree[
                succ
            ] -= 1

            if (
                indegree[succ]
                == 0
            ):
                q.append(
                    succ
                )

    if len(topo) != len(tasks):

        raise ValueError(
            "DAG contains a cycle"
        )

    return (
        by_id,
        successors,
        topo,
    )


# ============================================================
# Load instance rows
# ============================================================

def load_instances_for_job(
    instance_path: Path,
    valid_task_names: set,
):
    """
    Load:

        data/instances_by_job/<job>.csv

    Usable row:

        valid start
        valid end
        end > start
        cpu_avg > 0
    """

    if not instance_path.exists():

        raise FileNotFoundError(
            "missing per-job "
            "instance file: "
            f"{instance_path}"
        )

    rows_by_task = defaultdict(list)

    rejected_by_task = defaultdict(int)

    with instance_path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            task_name = (
                row.get("task_name")
                or ""
            ).strip()

            if (
                task_name
                not in valid_task_names
            ):
                continue

            start = as_float(
                row.get("start_time")
            )

            end = as_float(
                row.get("end_time")
            )

            cpu = as_float(
                row.get("cpu_avg")
            )

            if (
                start is None
                or end is None
                or cpu is None
                or end <= start + EPS
                or cpu <= 0.0
            ):

                rejected_by_task[
                    task_name
                ] += 1

                continue

            rows_by_task[
                task_name
            ].append(
                {
                    "instance_name": (
                        row.get(
                            "instance_name"
                        )
                        or ""
                    ).strip(),

                    "task_name": (
                        task_name
                    ),

                    "start_time": (
                        start
                    ),

                    "end_time": (
                        end
                    ),

                    "cpu_avg": (
                        cpu
                    ),
                }
            )

    return (
        rows_by_task,
        rejected_by_task,
    )


# ============================================================
# Build scheduling tasks
# ============================================================

def build_tasks_from_raw(
    job_name: str,
    task_rows: List[dict],
    instance_root: Path,
):
    """
    Deep validation for one natural Alibaba job.
    """

    if not task_rows:

        raise ValueError(
            "no batch_task rows"
        )

    tasks = []

    seen_nodes = set()
    seen_names = set()

    for row in task_rows:

        task_name = (
            row.get("task_name")
            or ""
        ).strip()

        if not task_name:

            raise ValueError(
                "empty task_name"
            )

        if task_name in seen_names:

            raise ValueError(
                f"duplicate task name "
                f"{task_name}"
            )

        node_id, predecessors = (
            parse_task_name(
                task_name
            )
        )

        if node_id is None:

            raise ValueError(
                "unparseable task name "
                f"{task_name!r}"
            )

        if node_id in seen_nodes:

            raise ValueError(
                "duplicate DAG node id "
                f"{node_id}"
            )

        seen_names.add(
            task_name
        )

        seen_nodes.add(
            node_id
        )

        expected_instances = as_int(
            row.get("instance_num"),
            0,
        )

        if (
            expected_instances is None
            or expected_instances <= 0
        ):

            raise ValueError(
                f"{task_name}: "
                f"non-positive "
                f"instance_num "
                f"{row.get('instance_num')!r}"
            )

        tasks.append(
            TaskRecord(
                task_name=task_name,

                node_id=node_id,

                predecessors=(
                    predecessors
                ),

                expected_instances=(
                    expected_instances
                ),

                task_type=(
                    row.get(
                        "task_type"
                    )
                    or ""
                ).strip(),

                status=(
                    row.get(
                        "status"
                    )
                    or ""
                ).strip(),

                plan_cpu=as_float(
                    row.get(
                        "plan_cpu"
                    ),
                    0.0,
                ),

                plan_mem=as_float(
                    row.get(
                        "plan_mem"
                    ),
                    0.0,
                ),
            )
        )

    # Structural DAG validation.
    build_dag(tasks)

    instance_path = (
        instance_root
        / f"{job_name}.csv"
    )

    (
        rows_by_task,
        rejected_by_task,
    ) = load_instances_for_job(
        instance_path,
        seen_names,
    )

    for task in tasks:

        rows = rows_by_task.get(
            task.task_name,
            [],
        )

        if not rows:

            raise ValueError(
                f"{task.task_name}: "
                f"no usable instances"
            )

        task.raw_release = min(
            row["start_time"]
            for row in rows
        )

        task.raw_deadline = max(
            row["end_time"]
            for row in rows
        )

        task.cpu_work = sum(
            (
                row["end_time"]
                - row["start_time"]
            )
            * row["cpu_avg"]

            for row in rows
        )

        task.usable_instances = (
            len(rows)
        )

        task.rejected_instances = (
            rejected_by_task.get(
                task.task_name,
                0,
            )
        )

        if (
            task.raw_deadline
            <= task.raw_release
            + EPS
        ):

            raise ValueError(
                f"{task.task_name}: "
                f"non-positive raw window"
            )

        if task.cpu_work <= EPS:

            raise ValueError(
                f"{task.task_name}: "
                f"non-positive CPU work"
            )

    actual_start = min(
        task.raw_release
        for task in tasks
    )

    actual_end = max(
        task.raw_deadline
        for task in tasks
    )

    actual_horizon = (
        actual_end
        - actual_start
    )

    if actual_horizon < 6:

        raise ValueError(
            "actual usable-instance "
            f"horizon "
            f"{actual_horizon:.6f} "
            f"< 6"
        )

    return (
        tasks,
        rows_by_task,
    )


# ============================================================
# Processing-time scaling
# ============================================================

def scale_processing_times(
    tasks: List[TaskRecord],
    utilization: float,
    horizon_start: float,
    horizon_end: float,
):
    """
    Same scaling rule as run_alibaba_experiment.py.

        alpha = U * H / sum(work)

        p_i = alpha * w_i
    """

    horizon = (
        horizon_end
        - horizon_start
    )

    if horizon <= EPS:

        raise ValueError(
            "non-positive job horizon"
        )

    total_work = sum(
        task.cpu_work
        for task in tasks
    )

    if total_work <= EPS:

        raise ValueError(
            "total CPU work "
            "is non-positive"
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
    tasks: List[TaskRecord],
):
    """
    Same transformation as current run_alibaba_experiment.py.

    Forward:

        R_i =
            max(
                raw_r_i,
                R_pred + p_pred
            )

    Backward:

        D_i =
            min(
                raw_d_i,
                D_succ - p_succ
            )
    """

    (
        by_id,
        successors,
        topo,
    ) = build_dag(tasks)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    for node in topo:

        task = by_id[node]

        release = (
            task.raw_release
        )

        for pred in task.predecessors:

            pred_task = (
                by_id[pred]
            )

            release = max(
                release,

                pred_task.effective_release
                + pred_task.processing,
            )

        task.effective_release = (
            release
        )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    for node in reversed(topo):

        task = by_id[node]

        deadline = (
            task.raw_deadline
        )

        for succ in successors[node]:

            succ_task = (
                by_id[succ]
            )

            deadline = min(
                deadline,

                succ_task.effective_deadline
                - succ_task.processing,
            )

        task.effective_deadline = (
            deadline
        )

    # --------------------------------------------------------
    # Individual-window feasibility
    # --------------------------------------------------------

    for task in tasks:

        window = (
            task.effective_deadline
            - task.effective_release
        )

        if window <= EPS:

            return (
                False,

                f"{task.task_name}: "
                f"non-positive "
                f"effective window "
                f"["
                f"{task.effective_release:.6f}, "
                f"{task.effective_deadline:.6f}"
                f"]",
            )

        if (
            task.processing
            > window + EPS
        ):

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

def create_energy_intervals(
    start_time: float,
    end_time: float,
):
    """
    Six equal-duration energy periods:

        green
        brown
        red
        green
        brown
        red

    Costs:

        green = 0
        brown = 1
        red   = 2
    """

    horizon = (
        end_time
        - start_time
    )

    if horizon < 6:

        raise ValueError(
            f"horizon {horizon} < 6"
        )

    pattern = [
        "green",
        "brown",
        "red",
        "green",
        "brown",
        "red",
    ]

    boundaries = [
        start_time
        + horizon * i / 6.0

        for i in range(7)
    ]

    boundaries[0] = (
        start_time
    )

    boundaries[-1] = (
        end_time
    )

    intervals = []

    for i, energy in enumerate(
        pattern
    ):

        start = boundaries[i]
        end = boundaries[i + 1]

        if end <= start + EPS:

            raise ValueError(
                "zero-length "
                "energy interval"
            )

        intervals.append(
            EnergyInterval(
                name=f"E{i}",
                start=start,
                end=end,
                energy=energy,
            )
        )

    return intervals


# ============================================================
# MaxFlow conversion
# ============================================================

def make_maxflow_jobs(
    tasks: List[TaskRecord],
):
    return [
        Job(
            name=task.task_name,

            release=(
                task.effective_release
            ),

            deadline=(
                task.effective_deadline
            ),

            processing=(
                task.processing
            ),
        )

        for task in tasks
    ]


# ============================================================
# Candidate evaluation
# ============================================================

def evaluate_candidate(
    band_name: str,
    stat_row: dict,
    task_rows: List[dict],
    instance_root: Path,
    utilization: float,
):
    """
    Fully validate and test one natural Alibaba job.

    MaxFlow feasibility is the final acceptance criterion.
    """

    result = CandidateResult(
        band_name=band_name,

        job_name=(
            stat_row["job_name"]
        ),

        actual_tasks=(
            stat_row["num_tasks"]
        ),

        total_instances_stats=(
            stat_row[
                "total_instances"
            ]
        ),

        cheap_horizon=(
            stat_row["horizon"]
        ),
    )

    try:

        (
            tasks,
            rows_by_task,
        ) = build_tasks_from_raw(
            result.job_name,
            task_rows,
            instance_root,
        )

        if (
            len(tasks)
            != result.actual_tasks
        ):

            raise ValueError(
                "task-count mismatch: "
                f"stats="
                f"{result.actual_tasks}, "
                f"loaded="
                f"{len(tasks)}"
            )

        actual_start = min(
            task.raw_release
            for task in tasks
        )

        actual_end = max(
            task.raw_deadline
            for task in tasks
        )

        actual_horizon = (
            actual_end
            - actual_start
        )

        alpha = (
            scale_processing_times(
                tasks,
                utilization,
                actual_start,
                actual_end,
            )
        )

        (
            window_ok,
            window_reason,
        ) = tighten_effective_windows(
            tasks
        )

        if not window_ok:

            result.reason = (
                "effective_window_"
                "infeasible: "
                f"{window_reason}"
            )

            return result

        intervals = (
            create_energy_intervals(
                actual_start,
                actual_end,
            )
        )

        jobs = (
            make_maxflow_jobs(
                tasks
            )
        )

        result.valid = True

        # ----------------------------------------------------
        # Actual MaxFlow feasibility test
        # ----------------------------------------------------

        maxflow_result = (
            max_flow_passes_schedule(
                jobs,
                intervals,
                verbose=False,
            )
        )

        result.actual_start = (
            actual_start
        )

        result.actual_end = (
            actual_end
        )

        result.actual_horizon = (
            actual_horizon
        )

        result.usable_instances = sum(
            task.usable_instances
            for task in tasks
        )

        result.rejected_instances = sum(
            task.rejected_instances
            for task in tasks
        )

        result.total_cpu_work = sum(
            task.cpu_work
            for task in tasks
        )

        result.total_processing = sum(
            task.processing
            for task in tasks
        )

        result.scale_alpha = (
            alpha
        )

        if not maxflow_result[
            "feasible"
        ]:

            result.reason = (
                "maxflow_infeasible"
            )

            return result

        # ----------------------------------------------------
        # Feasible
        # ----------------------------------------------------

        result.feasible = True

        result.reason = ""

        result.maxflow_cost = (
            maxflow_result[
                "normalized_cost"
            ]
        )

        usage = (
            maxflow_result[
                "final"
            ][
                "energy_usage"
            ]
        )

        result.maxflow_green = (
            usage["green"]
        )

        result.maxflow_brown = (
            usage["brown"]
        )

        result.maxflow_red = (
            usage["red"]
        )

        result.tasks = tasks

        result.instance_rows_by_task = (
            rows_by_task
        )

        return result

    except Exception as exc:

        result.reason = str(exc)

        return result


# ============================================================
# Representative selection for dense bands
# ============================================================

def select_representative_feasible(
    feasible: List[CandidateResult],
    count: int,
):
    """
    Select representative jobs across BOTH:

        actual DAG task count
        actual horizon

    from a feasible pool.

    This is only used for dense small/medium bands.

    Sparse large bands keep all feasible jobs.
    """

    if len(feasible) <= count:

        return sorted(
            feasible,
            key=lambda x: (
                x.actual_tasks,
                x.actual_horizon,
                numeric_job_id(
                    x.job_name
                ),
            ),
        )

    # Normalize task count and log horizon.
    task_values = [
        r.actual_tasks
        for r in feasible
    ]

    horizon_values = [
        math.log1p(
            r.actual_horizon
        )
        for r in feasible
    ]

    task_min = min(
        task_values
    )

    task_max = max(
        task_values
    )

    horizon_min = min(
        horizon_values
    )

    horizon_max = max(
        horizon_values
    )

    def normalized_task(r):

        if task_max == task_min:
            return 0.5

        return (
            r.actual_tasks
            - task_min
        ) / (
            task_max
            - task_min
        )

    def normalized_horizon(r):

        value = math.log1p(
            r.actual_horizon
        )

        if (
            horizon_max
            == horizon_min
        ):
            return 0.5

        return (
            value
            - horizon_min
        ) / (
            horizon_max
            - horizon_min
        )

    # Start from a central workload.
    center = min(
        feasible,
        key=lambda r: (
            (
                normalized_task(r)
                - 0.5
            ) ** 2
            +
            (
                normalized_horizon(r)
                - 0.5
            ) ** 2
        ),
    )

    selected = [
        center
    ]

    selected_names = {
        center.job_name
    }

    # Greedy farthest-point selection.
    while (
        len(selected)
        < count
    ):

        best = None
        best_distance = -1.0

        for candidate in feasible:

            if (
                candidate.job_name
                in selected_names
            ):
                continue

            cx = normalized_task(
                candidate
            )

            cy = normalized_horizon(
                candidate
            )

            min_distance = (
                math.inf
            )

            for chosen in selected:

                sx = normalized_task(
                    chosen
                )

                sy = normalized_horizon(
                    chosen
                )

                distance = (
                    (cx - sx) ** 2
                    +
                    (cy - sy) ** 2
                )

                min_distance = min(
                    min_distance,
                    distance,
                )

            if (
                min_distance
                > best_distance
            ):

                best_distance = (
                    min_distance
                )

                best = candidate

        if best is None:
            break

        selected.append(
            best
        )

        selected_names.add(
            best.job_name
        )

    return sorted(
        selected,
        key=lambda r: (
            r.actual_tasks,
            r.actual_horizon,
            numeric_job_id(
                r.job_name
            ),
        ),
    )


# ============================================================
# Write selected workload
# ============================================================

def write_selected_job(
    output_root: Path,
    result: CandidateResult,
    utilization: float,
):
    """
    Output layout:

        alibaba_scaling_jobs/
            j_xxx/
                tasks.csv
                instances/
                energy.csv
                selection_info.csv
    """

    job_dir = (
        output_root
        / result.job_name
    )

    instances_dir = (
        job_dir
        / "instances"
    )

    instances_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks = result.tasks

    # --------------------------------------------------------
    # tasks.csv
    # --------------------------------------------------------

    task_fields = [
        "task_name",
        "dag_node_id",
        "predecessors",
        "expected_instances",
        "usable_instances",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
        "task_type",
        "status",
    ]

    tasks_path = (
        job_dir
        / "tasks.csv"
    )

    with tasks_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=task_fields,
        )

        writer.writeheader()

        for task in sorted(
            tasks,
            key=lambda x: x.node_id,
        ):

            writer.writerow(
                {
                    "task_name": (
                        task.task_name
                    ),

                    "dag_node_id": (
                        task.node_id
                    ),

                    "predecessors": (
                        ",".join(
                            map(
                                str,
                                task.predecessors,
                            )
                        )
                    ),

                    "expected_instances": (
                        task.expected_instances
                    ),

                    "usable_instances": (
                        task.usable_instances
                    ),

                    "start_time": (
                        task.raw_release
                    ),

                    "end_time": (
                        task.raw_deadline
                    ),

                    "plan_cpu": (
                        task.plan_cpu
                    ),

                    "plan_mem": (
                        task.plan_mem
                    ),

                    "task_type": (
                        task.task_type
                    ),

                    "status": (
                        task.status
                    ),
                }
            )

    # --------------------------------------------------------
    # Per-task instance CSV
    # --------------------------------------------------------

    for task in tasks:

        rows = (
            result
            .instance_rows_by_task[
                task.task_name
            ]
        )

        path = (
            instances_dir
            / f"{task.task_name}.csv"
        )

        with path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=(
                    INSTANCE_OUTPUT_FIELDS
                ),
            )

            writer.writeheader()

            for row in rows:

                writer.writerow(
                    {
                        "instance_name": (
                            row[
                                "instance_name"
                            ]
                        ),

                        "start_time": (
                            row[
                                "start_time"
                            ]
                        ),

                        "end_time": (
                            row[
                                "end_time"
                            ]
                        ),

                        "cpu_avg": (
                            row[
                                "cpu_avg"
                            ]
                        ),
                    }
                )

    # --------------------------------------------------------
    # energy.csv
    # --------------------------------------------------------

    energy_fields = [
        "start_time",
        "end_time",
        "energy_type",
        "cost",
        "capacity",
    ]

    pattern = [
        ("green", 0),
        ("brown", 1),
        ("red", 2),
        ("green", 0),
        ("brown", 1),
        ("red", 2),
    ]

    start = (
        result.actual_start
    )

    end = (
        result.actual_end
    )

    horizon = (
        end - start
    )

    boundaries = [
        start
        + horizon * i / 6.0

        for i in range(7)
    ]

    boundaries[0] = start
    boundaries[-1] = end

    energy_path = (
        job_dir
        / "energy.csv"
    )

    with energy_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=energy_fields,
        )

        writer.writeheader()

        for i, (
            energy,
            cost,
        ) in enumerate(
            pattern
        ):

            interval_start = (
                boundaries[i]
            )

            interval_end = (
                boundaries[i + 1]
            )

            writer.writerow(
                {
                    "start_time": (
                        interval_start
                    ),

                    "end_time": (
                        interval_end
                    ),

                    "energy_type": (
                        energy
                    ),

                    "cost": (
                        cost
                    ),

                    "capacity": (
                        interval_end
                        - interval_start
                    ),
                }
            )

    # --------------------------------------------------------
    # selection_info.csv
    # --------------------------------------------------------

    info_fields = [
        "band_name",
        "job_name",
        "actual_tasks",
        "utilization",
        "actual_start",
        "actual_end",
        "actual_horizon",
        "total_instances_stats",
        "usable_instances",
        "rejected_instances",
        "total_cpu_work",
        "scale_alpha",
        "total_processing",
        "maxflow_cost",
        "maxflow_green",
        "maxflow_brown",
        "maxflow_red",
    ]

    info_path = (
        job_dir
        / "selection_info.csv"
    )

    with info_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=info_fields,
        )

        writer.writeheader()

        writer.writerow(
            {
                "band_name": (
                    result.band_name
                ),

                "job_name": (
                    result.job_name
                ),

                "actual_tasks": (
                    result.actual_tasks
                ),

                "utilization": (
                    utilization
                ),

                "actual_start": (
                    result.actual_start
                ),

                "actual_end": (
                    result.actual_end
                ),

                "actual_horizon": (
                    result.actual_horizon
                ),

                "total_instances_stats": (
                    result
                    .total_instances_stats
                ),

                "usable_instances": (
                    result.usable_instances
                ),

                "rejected_instances": (
                    result.rejected_instances
                ),

                "total_cpu_work": (
                    result.total_cpu_work
                ),

                "scale_alpha": (
                    result.scale_alpha
                ),

                "total_processing": (
                    result.total_processing
                ),

                "maxflow_cost": (
                    result.maxflow_cost
                ),

                "maxflow_green": (
                    result.maxflow_green
                ),

                "maxflow_brown": (
                    result.maxflow_brown
                ),

                "maxflow_red": (
                    result.maxflow_red
                ),
            }
        )


# ============================================================
# CSV output helpers
# ============================================================

SUMMARY_FIELDS = [
    "band_name",
    "job_name",
    "actual_tasks",
    "total_instances_stats",
    "usable_instances",
    "rejected_instances",
    "cheap_horizon",
    "actual_horizon",
    "utilization",
    "total_cpu_work",
    "scale_alpha",
    "total_processing",
    "maxflow_cost",
    "maxflow_green",
    "maxflow_brown",
    "maxflow_red",
]


def result_to_summary_row(
    result: CandidateResult,
    utilization: float,
):

    return {
        "band_name": (
            result.band_name
        ),

        "job_name": (
            result.job_name
        ),

        "actual_tasks": (
            result.actual_tasks
        ),

        "total_instances_stats": (
            result.total_instances_stats
        ),

        "usable_instances": (
            result.usable_instances
        ),

        "rejected_instances": (
            result.rejected_instances
        ),

        "cheap_horizon": (
            result.cheap_horizon
        ),

        "actual_horizon": (
            result.actual_horizon
        ),

        "utilization": (
            utilization
        ),

        "total_cpu_work": (
            result.total_cpu_work
        ),

        "scale_alpha": (
            result.scale_alpha
        ),

        "total_processing": (
            result.total_processing
        ),

        "maxflow_cost": (
            result.maxflow_cost
        ),

        "maxflow_green": (
            result.maxflow_green
        ),

        "maxflow_brown": (
            result.maxflow_brown
        ),

        "maxflow_red": (
            result.maxflow_red
        ),
    }


def write_dict_rows(
    path: Path,
    fields: List[str],
    rows: List[dict],
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

        writer.writerows(
            rows
        )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(
            "all_job_stats.csv"
        ),
    )

    parser.add_argument(
        "--batch-task",
        type=Path,
        default=Path(
            "data/batch_task.csv"
        ),
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        default=Path(
            "data/instances_by_job"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "alibaba_scaling_jobs"
        ),
    )

    parser.add_argument(
        "--utilization",
        type=float,
        default=(
            DEFAULT_UTILIZATION
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing output "
            "directory before building."
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
            "utilization must satisfy "
            "0 < U <= 1"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if (
        args.overwrite
        and args.output.exists()
    ):

        print(
            "Removing existing output: "
            f"{args.output}"
        )

        shutil.rmtree(
            args.output
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "ALIBABA REAL-DATA "
        "SCALING DATASET BUILDER"
    )
    print("=" * 88)

    print(
        f"Utilization: "
        f"{args.utilization:.2f}"
    )

    print(
        "Reported size: "
        "ACTUAL DAG task count"
    )

    print()

    # --------------------------------------------------------
    # Load cheap candidates
    # --------------------------------------------------------

    candidates_by_band = (
        load_candidate_stats(
            args.stats
        )
    )

    # --------------------------------------------------------
    # Decide which candidates to deep-check
    # --------------------------------------------------------

    ordered_by_band = {}

    all_candidate_names = set()

    print(
        "Cheap candidate availability:"
    )

    print("-" * 88)

    for band in BANDS:

        name = band["name"]

        rows = (
            candidates_by_band[
                name
            ]
        )

        if (
            band["mode"]
            == "sample"
        ):

            ordered = (
                order_sample_candidates(
                    rows
                )
            )

            max_candidates = (
                band[
                    "max_candidates"
                ]
            )

            if (
                max_candidates
                is not None
            ):

                ordered = (
                    ordered[
                        :max_candidates
                    ]
                )

        else:

            ordered = (
                order_all_candidates(
                    rows
                )
            )

        ordered_by_band[
            name
        ] = ordered

        for row in ordered:

            all_candidate_names.add(
                row["job_name"]
            )

        print(
            f"{name:>10}: "
            f"{len(rows):6,d} "
            f"cheap candidates, "
            f"{len(ordered):6,d} "
            f"scheduled for "
            f"deep validation, "
            f"mode={band['mode']}"
        )

    if not all_candidate_names:

        raise RuntimeError(
            "No candidate jobs found."
        )

    # --------------------------------------------------------
    # Scan task file once
    # --------------------------------------------------------

    task_rows_by_job = (
        load_task_rows_for_candidates(
            args.batch_task,
            all_candidate_names,
        )
    )

    # --------------------------------------------------------
    # Deep validation
    # --------------------------------------------------------

    all_feasible_results = []

    all_rejected_rows = []

    selected_results = []

    band_summary_rows = []

    selected_job_names = set()

    for band in BANDS:

        band_name = (
            band["name"]
        )

        candidates = (
            ordered_by_band[
                band_name
            ]
        )

        print()
        print("=" * 88)

        print(
            f"BAND {band_name} "
            f"("
            f"{band['min_tasks']}"
            f"-"
            f"{band['max_tasks']} "
            f"tasks"
            f")"
        )

        print("=" * 88)

        feasible = []

        tested = 0

        rejected = 0

        for index, stat_row in enumerate(
            candidates,
            start=1,
        ):

            job_name = (
                stat_row["job_name"]
            )

            print(
                f"["
                f"{index:4d}/"
                f"{len(candidates):4d}"
                f"] "
                f"{job_name:>12} "
                f"tasks="
                f"{stat_row['num_tasks']:4d} "
                f"horizon="
                f"{stat_row['horizon']:10.2f}",
                end="  ",
                flush=True,
            )

            task_rows = (
                task_rows_by_job.get(
                    job_name,
                    [],
                )
            )

            result = (
                evaluate_candidate(
                    band_name=band_name,
                    stat_row=stat_row,
                    task_rows=task_rows,
                    instance_root=(
                        args.instance_root
                    ),
                    utilization=(
                        args.utilization
                    ),
                )
            )

            tested += 1

            if result.feasible:

                feasible.append(
                    result
                )

                all_feasible_results.append(
                    result
                )

                print(
                    "FEASIBLE "
                    f"(tasks="
                    f"{result.actual_tasks}, "
                    f"horizon="
                    f"{result.actual_horizon:.2f}"
                    f")"
                )

            else:

                rejected += 1

                print(
                    "REJECT: "
                    f"{result.reason}"
                )

                all_rejected_rows.append(
                    {
                        "band_name": (
                            band_name
                        ),

                        "job_name": (
                            result.job_name
                        ),

                        "actual_tasks": (
                            result.actual_tasks
                        ),

                        "total_instances_stats": (
                            result
                            .total_instances_stats
                        ),

                        "cheap_horizon": (
                            result.cheap_horizon
                        ),

                        "valid_before_maxflow": (
                            result.valid
                        ),

                        "reason": (
                            result.reason
                        ),
                    }
                )

            # ------------------------------------------------
            # Dense bands:
            #
            # stop after we have a sufficiently large feasible
            # pool.
            #
            # Sparse large bands:
            #
            # NEVER stop early. Evaluate all candidates.
            # ------------------------------------------------

            if (
                band["mode"]
                == "sample"
            ):

                preferred_pool = (
                    band[
                        "preferred_feasible_pool"
                    ]
                )

                if (
                    preferred_pool
                    is not None
                    and len(feasible)
                    >= preferred_pool
                ):

                    print(
                        "\nPreferred feasible "
                        f"pool of "
                        f"{preferred_pool} "
                        f"reached."
                    )

                    break

        # ----------------------------------------------------
        # Select final jobs from this band
        # ----------------------------------------------------

        if (
            band["mode"]
            == "sample"
        ):

            keep = band["keep"]

            if len(feasible) == 0:

                selected = []

                print(
                    "\nWARNING: "
                    f"{band_name} produced "
                    "no feasible workloads."
                )

            else:

                selected = (
                    select_representative_feasible(
                        feasible,
                        min(
                            keep,
                            len(feasible),
                        ),
                    )
                )

                if len(feasible) < keep:

                    print(
                        "\nWARNING: "
                        f"{band_name} produced "
                        f"only "
                        f"{len(feasible)} "
                        f"feasible workload(s); "
                        f"requested {keep}."
                    )

        else:

            # Sparse large band:
            # retain EVERY feasible natural job.
            selected = sorted(
                feasible,
                key=lambda r: (
                    r.actual_tasks,
                    r.actual_horizon,
                    numeric_job_id(
                        r.job_name
                    ),
                ),
            )

        # ----------------------------------------------------
        # Save selected workloads
        # ----------------------------------------------------

        print()
        print(
            f"Band {band_name}: "
            f"tested={tested}, "
            f"feasible={len(feasible)}, "
            f"selected={len(selected)}"
        )

        if selected:

            print(
                "Selected workloads:"
            )

        for result in selected:

            # Defensive duplicate protection.
            if (
                result.job_name
                in selected_job_names
            ):

                continue

            selected_job_names.add(
                result.job_name
            )

            selected_results.append(
                result
            )

            print(
                f"  "
                f"{result.job_name:>12}  "
                f"tasks="
                f"{result.actual_tasks:4d}  "
                f"horizon="
                f"{result.actual_horizon:10.3f}  "
                f"usable_instances="
                f"{result.usable_instances:8d}"
            )

            write_selected_job(
                args.output,
                result,
                args.utilization,
            )

        band_summary_rows.append(
            {
                "band_name": (
                    band_name
                ),

                "min_tasks": (
                    band[
                        "min_tasks"
                    ]
                ),

                "max_tasks": (
                    band[
                        "max_tasks"
                    ]
                ),

                "mode": (
                    band["mode"]
                ),

                "cheap_candidates": (
                    len(
                        candidates_by_band[
                            band_name
                        ]
                    )
                ),

                "deep_tested": (
                    tested
                ),

                "feasible_found": (
                    len(feasible)
                ),

                "selected": (
                    len(selected)
                ),
            }
        )

    # ========================================================
    # Write summary.csv
    # ========================================================

    selected_results = sorted(
        selected_results,
        key=lambda r: (
            r.actual_tasks,
            r.actual_horizon,
            numeric_job_id(
                r.job_name
            ),
        ),
    )

    summary_rows = [
        result_to_summary_row(
            result,
            args.utilization,
        )

        for result in selected_results
    ]

    write_dict_rows(
        args.output
        / "summary.csv",

        SUMMARY_FIELDS,

        summary_rows,
    )

    # ========================================================
    # Write complete feasible pool
    # ========================================================

    all_feasible_results = sorted(
        all_feasible_results,
        key=lambda r: (
            r.actual_tasks,
            r.actual_horizon,
            numeric_job_id(
                r.job_name
            ),
        ),
    )

    feasible_rows = [
        result_to_summary_row(
            result,
            args.utilization,
        )

        for result
        in all_feasible_results
    ]

    write_dict_rows(
        args.output
        / "feasible_candidate_pool.csv",

        SUMMARY_FIELDS,

        feasible_rows,
    )

    # ========================================================
    # Write rejected candidates
    # ========================================================

    rejected_fields = [
        "band_name",
        "job_name",
        "actual_tasks",
        "total_instances_stats",
        "cheap_horizon",
        "valid_before_maxflow",
        "reason",
    ]

    write_dict_rows(
        args.output
        / "rejected_candidates.csv",

        rejected_fields,

        all_rejected_rows,
    )

    # ========================================================
    # Write band summary
    # ========================================================

    band_fields = [
        "band_name",
        "min_tasks",
        "max_tasks",
        "mode",
        "cheap_candidates",
        "deep_tested",
        "feasible_found",
        "selected",
    ]

    write_dict_rows(
        args.output
        / "band_summary.csv",

        band_fields,

        band_summary_rows,
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)

    print(
        f"Selected workloads: "
        f"{len(selected_results)}"
    )

    print(
        f"All feasible candidates found: "
        f"{len(all_feasible_results)}"
    )

    print(
        f"Rejected candidates: "
        f"{len(all_rejected_rows)}"
    )

    print(
        f"Output directory: "
        f"{args.output}"
    )

    if selected_results:

        min_tasks = min(
            result.actual_tasks
            for result
            in selected_results
        )

        max_tasks = max(
            result.actual_tasks
            for result
            in selected_results
        )

        print(
            f"Selected DAG-size range: "
            f"{min_tasks} "
            f"to "
            f"{max_tasks} tasks"
        )

        print()
        print(
            "Selected workloads "
            "by actual DAG size:"
        )

        for result in selected_results:

            print(
                f"  "
                f"{result.actual_tasks:5d} tasks  "
                f"{result.job_name:>12}  "
                f"horizon="
                f"{result.actual_horizon:10.3f}"
            )

    print()
    print(
        "Files:"
    )

    print(
        f"  {args.output / 'summary.csv'}"
    )

    print(
        f"  "
        f"{args.output / 'feasible_candidate_pool.csv'}"
    )

    print(
        f"  "
        f"{args.output / 'rejected_candidates.csv'}"
    )

    print(
        f"  "
        f"{args.output / 'band_summary.csv'}"
    )


if __name__ == "__main__":
    main()