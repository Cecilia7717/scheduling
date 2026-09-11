from __future__ import annotations

import csv
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EPS = 1e-9

ENERGY_COST = {
    "green": 0.0,
    "brown": 1.0,
    "red": 2.0,
}


# ============================================================
# Scheduling data structures
# ============================================================

@dataclass(frozen=True)
class Job:
    name: str
    release: float
    deadline: float
    processing: float


@dataclass(frozen=True)
class EnergyInterval:
    name: str
    start: float
    end: float
    energy: str  # green / brown / red

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class OriginalEdge:
    """
    One edge in the scheduling flow network.

    capacity is the CURRENT total capacity for the current pass.
    flow is the flow already present and carried into the next pass.
    """
    u: int
    v: int
    capacity: int
    flow: int
    kind: str
    job_index: Optional[int] = None
    interval_index: Optional[int] = None


@dataclass(frozen=True)
class ResidualArc:
    """
    Mapping from one residual-graph arc back to an original edge.

    sign = +1: residual flow increases original flow
    sign = -1: residual flow decreases original flow
    """
    u: int
    v: int
    capacity: int
    original_edge_index: int
    sign: int


# ============================================================
# Input validation / interval splitting
# ============================================================

def _to_int(x: float, label: str) -> int:
    """
    ECL-MaxFlow uses integer capacities. This wrapper therefore requires
    integer-valued scheduling times and processing requirements.
    """
    r = round(x)
    if abs(x - r) > EPS:
        raise ValueError(
            f"{label}={x} is not integer-valued. "
            "The current ECL wrapper uses integer capacities."
        )
    return int(r)


def validate_input(
    jobs: List[Job],
    intervals: List[EnergyInterval],
) -> None:
    if not jobs:
        raise ValueError("At least one job is required.")
    if not intervals:
        raise ValueError("At least one energy interval is required.")

    for job in jobs:
        if job.deadline <= job.release + EPS:
            raise ValueError(f"{job.name}: deadline must exceed release.")
        if job.processing < -EPS:
            raise ValueError(f"{job.name}: processing cannot be negative.")
        if job.processing > job.deadline - job.release + EPS:
            raise ValueError(
                f"{job.name}: processing={job.processing} exceeds "
                f"window length={job.deadline - job.release}."
            )
        _to_int(job.release, f"{job.name}.release")
        _to_int(job.deadline, f"{job.name}.deadline")
        _to_int(job.processing, f"{job.name}.processing")

    valid = {"green", "brown", "red"}
    ordered = sorted(intervals, key=lambda x: (x.start, x.end))

    for interval in ordered:
        if interval.energy not in valid:
            raise ValueError(
                f"{interval.name}: energy must be one of {sorted(valid)}."
            )
        if interval.end <= interval.start + EPS:
            raise ValueError(f"{interval.name}: invalid interval.")
        _to_int(interval.start, f"{interval.name}.start")
        _to_int(interval.end, f"{interval.name}.end")

    for prev, cur in zip(ordered, ordered[1:]):
        if abs(prev.end - cur.start) > EPS:
            raise ValueError(
                "Energy intervals must form a continuous, "
                "non-overlapping partition."
            )


def split_intervals_at_job_boundaries(
    jobs: List[Job],
    intervals: List[EnergyInterval],
) -> List[EnergyInterval]:
    """
    Split energy intervals at every release time and deadline.

    Then a Job -> Interval edge exists exactly when the entire elementary
    interval lies inside the job's [release, deadline] window.
    """
    validate_input(jobs, intervals)

    ordered = sorted(intervals, key=lambda x: (x.start, x.end))
    horizon_start = ordered[0].start
    horizon_end = ordered[-1].end

    boundaries = {horizon_start, horizon_end}

    for interval in ordered:
        boundaries.add(interval.start)
        boundaries.add(interval.end)

    for job in jobs:
        if job.release < horizon_start - EPS or job.deadline > horizon_end + EPS:
            raise ValueError(
                f"{job.name}: window [{job.release}, {job.deadline}] "
                f"outside [{horizon_start}, {horizon_end}]."
            )
        boundaries.add(job.release)
        boundaries.add(job.deadline)

    points = sorted(boundaries)
    result: List[EnergyInterval] = []

    for k in range(len(points) - 1):
        a = points[k]
        b = points[k + 1]

        if b <= a + EPS:
            continue

        parent = None
        for interval in ordered:
            if interval.start <= a + EPS and b <= interval.end + EPS:
                parent = interval
                break

        if parent is None:
            raise RuntimeError(
                f"Could not map elementary interval [{a}, {b}] "
                "to an energy interval."
            )

        result.append(
            EnergyInterval(
                name=f"I{len(result)}",
                start=a,
                end=b,
                energy=parent.energy,
            )
        )

    return result


# ============================================================
# Scheduling flow network
# ============================================================

def build_scheduling_network(
    jobs: List[Job],
    intervals: List[EnergyInterval],
) -> Tuple[List[OriginalEdge], Dict[str, Any]]:
    """
    Construct the full scheduling graph.

        source -> jobs -> elementary intervals -> sink

    Interval -> sink capacities are initialized for PASS 1:
        green = interval length
        brown = 0
        red   = 0
    """
    n_jobs = len(jobs)
    n_intervals = len(intervals)

    source = 0
    job_offset = 1
    interval_offset = job_offset + n_jobs
    sink = interval_offset + n_intervals
    nodes = sink + 1

    edges: List[OriginalEdge] = []

    source_job_edge: Dict[int, int] = {}
    job_interval_edge: Dict[Tuple[int, int], int] = {}
    interval_sink_edge: Dict[int, int] = {}

    # source -> jobs
    for j, job in enumerate(jobs):
        edge_index = len(edges)
        source_job_edge[j] = edge_index
        edges.append(
            OriginalEdge(
                u=source,
                v=job_offset + j,
                capacity=_to_int(job.processing, f"{job.name}.processing"),
                flow=0,
                kind="source_job",
                job_index=j,
            )
        )

    # jobs -> elementary intervals
    for j, job in enumerate(jobs):
        jnode = job_offset + j

        for i, interval in enumerate(intervals):
            if (
                job.release <= interval.start + EPS
                and interval.end <= job.deadline + EPS
            ):
                edge_index = len(edges)
                job_interval_edge[(j, i)] = edge_index
                edges.append(
                    OriginalEdge(
                        u=jnode,
                        v=interval_offset + i,
                        capacity=_to_int(
                            interval.length,
                            f"{interval.name}.length",
                        ),
                        flow=0,
                        kind="job_interval",
                        job_index=j,
                        interval_index=i,
                    )
                )

    # elementary intervals -> sink
    for i, interval in enumerate(intervals):
        initial_capacity = (
            _to_int(interval.length, f"{interval.name}.length")
            if interval.energy == "green"
            else 0
        )

        edge_index = len(edges)
        interval_sink_edge[i] = edge_index

        edges.append(
            OriginalEdge(
                u=interval_offset + i,
                v=sink,
                capacity=initial_capacity,
                flow=0,
                kind="interval_sink",
                interval_index=i,
            )
        )

    metadata = {
        "nodes": nodes,
        "source": source,
        "sink": sink,
        "job_offset": job_offset,
        "interval_offset": interval_offset,
        "source_job_edge": source_job_edge,
        "job_interval_edge": job_interval_edge,
        "interval_sink_edge": interval_sink_edge,
    }

    return edges, metadata


# ============================================================
# ECL-MaxFlow wrapper
# ============================================================

_RUNTIME_RE = re.compile(r"^runtime:\s*([0-9.eE+-]+)s\s*$", re.MULTILINE)
_MAXFLOW_RE = re.compile(
    r"Maximum flow from nodes\s+\d+\s+to\s+\d+:\s+(-?\d+)"
)


def _write_dimacs(
    path: Path,
    nodes: int,
    arcs: List[Tuple[int, int, int]],
    source: int,
    sink: int,
) -> None:
    """
    Write a directed DIMACS max-flow graph.

    DIMACS node numbers are 1-based.
    """
    positive = [(u, v, c) for u, v, c in arcs if c > 0]

    with path.open("w", encoding="utf-8") as f:
        f.write("c generated scheduling residual graph\n")
        f.write(f"p max {nodes} {len(positive)}\n")
        f.write(f"n {source + 1} s\n")
        f.write(f"n {sink + 1} t\n")

        for u, v, capacity in positive:
            f.write(f"a {u + 1} {v + 1} {capacity}\n")


def _read_ecl_flow_csv(path: Path) -> Dict[Tuple[int, int], int]:
    """
    Read the CSV emitted by the modified ECL executable:

        from,to,flow,capacity

    Scheduling/residual graphs constructed here have no parallel arcs
    with the same ordered (u,v) pair.
    """
    result: Dict[Tuple[int, int], int] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"from", "to", "flow", "capacity"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: "
                "from,to,flow,capacity"
            )

        for row in reader:
            u = int(row["from"])
            v = int(row["to"])
            flow = int(row["flow"])

            key = (u, v)
            if key in result:
                raise ValueError(
                    "Parallel arcs detected in ECL output for "
                    f"{u}->{v}. This wrapper currently expects unique pairs."
                )

            result[key] = flow

    return result


def run_ecl_maxflow(
    *,
    nodes: int,
    arcs: List[Tuple[int, int, int]],
    source: int,
    sink: int,
    ecl_dir: str | Path,
    work_dir: Path,
    stem: str,
    keep_files: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    DIMACS -> EGR -> ECL GPU max-flow -> edge-flow CSV.

    Requires the user's modified ECL executable supporting:

        ./maxflow graph.egr source sink 1 output.csv
    """
    ecl_dir = Path(ecl_dir).resolve()
    converter = ecl_dir / "convert_dimacs_to_eclgraph" / "dimacsflow2ecl"
    executable = ecl_dir / "maxflow"

    if not converter.exists():
        raise FileNotFoundError(
            f"ECL DIMACS converter not found: {converter}"
        )

    if not executable.exists():
        raise FileNotFoundError(
            f"ECL maxflow executable not found: {executable}"
        )

    dimacs_path = work_dir / f"{stem}.dimacs"
    egr_path = work_dir / f"{stem}.egr"
    csv_path = work_dir / f"{stem}_flow.csv"

    positive_arcs = [(u, v, c) for u, v, c in arcs if c > 0]

    _write_dimacs(
        dimacs_path,
        nodes,
        positive_arcs,
        source,
        sink,
    )

    subprocess.run(
        [
            str(converter),
            str(dimacs_path),
            str(egr_path),
        ],
        check=True,
        cwd=ecl_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    completed = subprocess.run(
        [
            str(executable),
            str(egr_path),
            str(source),
            str(sink),
            "1",
            str(csv_path),
        ],
        check=True,
        cwd=ecl_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stdout = completed.stdout

    if verbose:
        print(stdout, end="" if stdout.endswith("\n") else "\n")

    maxflow_match = _MAXFLOW_RE.search(stdout)
    if maxflow_match is None:
        raise RuntimeError(
            "Could not parse maximum-flow value from ECL output.\n"
            + stdout
        )

    runtime_match = _RUNTIME_RE.search(stdout)
    gpu_seconds = (
        float(runtime_match.group(1))
        if runtime_match is not None
        else None
    )

    edge_flows = _read_ecl_flow_csv(csv_path)

    result = {
        "flow": int(maxflow_match.group(1)),
        "edge_flows": edge_flows,
        "gpu_seconds": gpu_seconds,
        "stdout": stdout,
        "dimacs_path": dimacs_path if keep_files else None,
        "egr_path": egr_path if keep_files else None,
        "flow_csv_path": csv_path if keep_files else None,
    }

    if not keep_files:
        for p in (dimacs_path, egr_path, csv_path):
            if p.exists():
                p.unlink()

    return result


# ============================================================
# Residual-network passes
# ============================================================

def _current_total_flow(
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
) -> int:
    return sum(
        edges[edge_index].flow
        for edge_index in metadata["source_job_edge"].values()
    )


def _set_pass2_capacities(
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
    intervals: List[EnergyInterval],
) -> None:
    """
    PASS 2:
      green: lock total capacity to exactly PASS-1 usage
      brown: open full interval capacity
      red:   remain closed
    """
    for i, interval in enumerate(intervals):
        edge = edges[metadata["interval_sink_edge"][i]]

        if interval.energy == "green":
            edge.capacity = edge.flow
        elif interval.energy == "brown":
            edge.capacity = _to_int(
                interval.length,
                f"{interval.name}.length",
            )
        else:
            edge.capacity = 0

        if edge.flow > edge.capacity:
            raise RuntimeError(
                f"Invalid pass-2 capacity on {interval.name}: "
                f"flow={edge.flow}, capacity={edge.capacity}"
            )


def _set_pass3_capacities(
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
    intervals: List[EnergyInterval],
) -> None:
    """
    PASS 3:
      green: stays locked at PASS-1 usage from pass 2
      brown: lock total capacity to exactly PASS-2 usage
      red:   open full interval capacity
    """
    for i, interval in enumerate(intervals):
        edge = edges[metadata["interval_sink_edge"][i]]

        if interval.energy == "brown":
            edge.capacity = edge.flow
        elif interval.energy == "red":
            edge.capacity = _to_int(
                interval.length,
                f"{interval.name}.length",
            )
        # green stays unchanged

        if edge.flow > edge.capacity:
            raise RuntimeError(
                f"Invalid pass-3 capacity on {interval.name}: "
                f"flow={edge.flow}, capacity={edge.capacity}"
            )


def _build_residual_arcs(
    edges: List[OriginalEdge],
) -> List[ResidualArc]:
    """
    Build the exact residual network for the currently carried flow.

    For original edge u->v with capacity C and current flow f:
        forward residual u->v has capacity C-f
        reverse residual v->u has capacity f
    """
    residual: List[ResidualArc] = []

    seen_pairs: set[Tuple[int, int]] = set()

    for edge_index, edge in enumerate(edges):
        if edge.flow < 0 or edge.flow > edge.capacity:
            raise RuntimeError(
                f"Invalid flow state on edge {edge.u}->{edge.v}: "
                f"flow={edge.flow}, capacity={edge.capacity}"
            )

        forward_capacity = edge.capacity - edge.flow
        if forward_capacity > 0:
            key = (edge.u, edge.v)
            if key in seen_pairs:
                raise RuntimeError(
                    f"Parallel residual arc detected: {key}"
                )
            seen_pairs.add(key)
            residual.append(
                ResidualArc(
                    u=edge.u,
                    v=edge.v,
                    capacity=forward_capacity,
                    original_edge_index=edge_index,
                    sign=+1,
                )
            )

        if edge.flow > 0:
            key = (edge.v, edge.u)
            if key in seen_pairs:
                raise RuntimeError(
                    f"Parallel residual arc detected: {key}"
                )
            seen_pairs.add(key)
            residual.append(
                ResidualArc(
                    u=edge.v,
                    v=edge.u,
                    capacity=edge.flow,
                    original_edge_index=edge_index,
                    sign=-1,
                )
            )

    return residual


def _apply_residual_augmentation(
    edges: List[OriginalEdge],
    residual_arcs: List[ResidualArc],
    residual_flow: Dict[Tuple[int, int], int],
) -> None:
    """
    Convert the GPU flow on the residual graph into an update of the
    original carried flow.
    """
    delta = [0] * len(edges)

    mapping = {
        (arc.u, arc.v): arc
        for arc in residual_arcs
    }

    for pair, amount in residual_flow.items():
        if amount == 0:
            continue

        arc = mapping.get(pair)
        if arc is None:
            raise RuntimeError(
                f"ECL returned flow for unknown residual arc {pair}."
            )

        if amount < 0 or amount > arc.capacity:
            raise RuntimeError(
                f"Invalid ECL residual flow on {pair}: "
                f"{amount}/{arc.capacity}"
            )

        delta[arc.original_edge_index] += arc.sign * amount

    for i, change in enumerate(delta):
        edges[i].flow += change

        if edges[i].flow < 0 or edges[i].flow > edges[i].capacity:
            raise RuntimeError(
                f"Residual augmentation produced invalid original flow "
                f"on {edges[i].u}->{edges[i].v}: "
                f"{edges[i].flow}/{edges[i].capacity}"
            )


def _run_initial_pass(
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
    *,
    ecl_dir: Path,
    work_dir: Path,
    keep_files: bool,
    verbose_ecl: bool,
) -> Dict[str, Any]:
    """
    Pass 1 starts from zero flow, so run ECL directly on the pass-1 graph.
    """
    arcs = [
        (edge.u, edge.v, edge.capacity)
        for edge in edges
        if edge.capacity > 0
    ]

    ecl = run_ecl_maxflow(
        nodes=metadata["nodes"],
        arcs=arcs,
        source=metadata["source"],
        sink=metadata["sink"],
        ecl_dir=ecl_dir,
        work_dir=work_dir,
        stem="pass1",
        keep_files=keep_files,
        verbose=verbose_ecl,
    )

    pair_to_edge = {
        (edge.u, edge.v): i
        for i, edge in enumerate(edges)
        if edge.capacity > 0
    }

    for pair, amount in ecl["edge_flows"].items():
        edge_index = pair_to_edge.get(pair)

        if edge_index is None:
            if amount != 0:
                raise RuntimeError(
                    f"ECL returned flow for unknown pass-1 arc {pair}."
                )
            continue

        edges[edge_index].flow = amount

    total = _current_total_flow(edges, metadata)

    if total != ecl["flow"]:
        raise RuntimeError(
            f"Pass-1 flow mismatch: source flow={total}, "
            f"ECL maxflow={ecl['flow']}"
        )

    return {
        "added_flow": ecl["flow"],
        "gpu_seconds": ecl["gpu_seconds"],
        "stdout": ecl["stdout"],
    }


def _run_augmentation_pass(
    pass_name: str,
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
    *,
    ecl_dir: Path,
    work_dir: Path,
    keep_files: bool,
    verbose_ecl: bool,
) -> Dict[str, Any]:
    """
    Run ECL on the current residual network and add that augmentation
    to the carried original flow.
    """
    residual_arcs = _build_residual_arcs(edges)

    arcs = [
        (arc.u, arc.v, arc.capacity)
        for arc in residual_arcs
        if arc.capacity > 0
    ]

    if not arcs:
        return {
            "added_flow": 0,
            "gpu_seconds": 0.0,
            "stdout": "",
        }

    ecl = run_ecl_maxflow(
        nodes=metadata["nodes"],
        arcs=arcs,
        source=metadata["source"],
        sink=metadata["sink"],
        ecl_dir=ecl_dir,
        work_dir=work_dir,
        stem=pass_name,
        keep_files=keep_files,
        verbose=verbose_ecl,
    )

    before = _current_total_flow(edges, metadata)

    _apply_residual_augmentation(
        edges,
        residual_arcs,
        ecl["edge_flows"],
    )

    after = _current_total_flow(edges, metadata)

    if after - before != ecl["flow"]:
        raise RuntimeError(
            f"{pass_name} augmentation mismatch: "
            f"source increase={after - before}, "
            f"ECL residual maxflow={ecl['flow']}"
        )

    return {
        "added_flow": ecl["flow"],
        "gpu_seconds": ecl["gpu_seconds"],
        "stdout": ecl["stdout"],
    }


# ============================================================
# Recover scheduling information
# ============================================================

def recover_result(
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
    jobs: List[Job],
    intervals: List[EnergyInterval],
) -> Dict[str, Any]:
    assignment: Dict[Tuple[int, int], int] = {}

    for key, edge_index in metadata["job_interval_edge"].items():
        amount = edges[edge_index].flow
        if amount > 0:
            assignment[key] = amount

    interval_usage: Dict[int, int] = {}
    energy_usage = {
        "green": 0,
        "brown": 0,
        "red": 0,
    }

    for i, interval in enumerate(intervals):
        amount = edges[metadata["interval_sink_edge"][i]].flow
        interval_usage[i] = amount
        energy_usage[interval.energy] += amount

    job_usage: Dict[int, int] = {}

    for j in range(len(jobs)):
        job_usage[j] = edges[metadata["source_job_edge"][j]].flow

    return {
        "flow": _current_total_flow(edges, metadata),
        "assignment": assignment,
        "interval_usage": interval_usage,
        "energy_usage": energy_usage,
        "job_usage": job_usage,
    }


def build_schedule_rows(
    jobs: List[Job],
    intervals: List[EnergyInterval],
    assignment: Dict[Tuple[int, int], int],
) -> List[Dict[str, Any]]:
    """
    Desired interval-allocation format:

        Job, Interval, Start, End, Energy, Flow
    """
    rows: List[Dict[str, Any]] = []

    for (j, i), amount in assignment.items():
        if amount <= 0:
            continue

        interval = intervals[i]

        rows.append(
            {
                "Job": jobs[j].name,
                "Interval": interval.name,
                "Start": interval.start,
                "End": interval.end,
                "Energy": interval.energy,
                "Flow": amount,
            }
        )

    job_order = {
        job.name: index
        for index, job in enumerate(jobs)
    }

    rows.sort(
        key=lambda row: (
            job_order[row["Job"]],
            row["Start"],
            row["End"],
        )
    )

    return rows


def build_timeline(
    jobs: List[Job],
    intervals: List[EnergyInterval],
    assignment: Dict[Tuple[int, int], int],
) -> List[Dict[str, Any]]:
    """
    Produce one concrete non-overlapping ordering inside each elementary
    interval. Flow itself only determines how much of each interval each
    job receives; this deterministic ordering turns that into timestamps.
    """
    timeline: List[Dict[str, Any]] = []

    for i, interval in enumerate(intervals):
        cursor = interval.start

        pieces = [
            (j, amount)
            for (j, ii), amount in assignment.items()
            if ii == i and amount > 0
        ]

        pieces.sort(
            key=lambda x: (
                jobs[x[0]].deadline,
                jobs[x[0]].name,
            )
        )

        for j, amount in pieces:
            timeline.append(
                {
                    "start": cursor,
                    "end": cursor + amount,
                    "job": jobs[j].name,
                    "energy": interval.energy,
                    "interval": interval.name,
                }
            )
            cursor += amount

        if cursor < interval.end - EPS:
            timeline.append(
                {
                    "start": cursor,
                    "end": interval.end,
                    "job": "IDLE",
                    "energy": interval.energy,
                    "interval": interval.name,
                }
            )

    return timeline


def schedule_cost(flow_result: Dict[str, Any]) -> float:
    usage = flow_result["energy_usage"]

    return (
        ENERGY_COST["green"] * usage["green"]
        + ENERGY_COST["brown"] * usage["brown"]
        + ENERGY_COST["red"] * usage["red"]
    )


def _snapshot_pass(
    edges: List[OriginalEdge],
    metadata: Dict[str, Any],
    jobs: List[Job],
    intervals: List[EnergyInterval],
    added_flow: int,
    gpu_seconds: Optional[float],
) -> Dict[str, Any]:
    result = recover_result(
        edges,
        metadata,
        jobs,
        intervals,
    )

    result["added_flow"] = added_flow
    result["gpu_seconds"] = gpu_seconds
    result["schedule_rows"] = build_schedule_rows(
        jobs,
        intervals,
        result["assignment"],
    )

    return result


# ============================================================
# Main GPU Max Flow + Passes algorithm
# ============================================================

def max_flow_passes_schedule_gpu(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    *,
    ecl_dir: str | Path,
    verbose: bool = True,
    verbose_ecl: bool = False,
    keep_work_files: bool = False,
    work_folder: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    GPU implementation of the incremental three-pass algorithm.

    PASS 1:
        Only green interval->sink capacity is available.

    PASS 2:
        Carry pass-1 flow forward.
        Lock each green interval->sink capacity to its pass-1 occupied flow.
        Open brown intervals.
        Run GPU max-flow on the exact residual graph.

    PASS 3:
        Carry pass-2 flow forward.
        Green remains locked.
        Lock each brown interval->sink capacity to its pass-2 occupied flow.
        Open red intervals.
        Run GPU max-flow on the exact residual graph.

    This reproduces the residual-network logic of the CPU Dinic version,
    but ECL-MaxFlow is used for each max-flow computation.
    """
    wall_start = time.perf_counter()

    intervals = split_intervals_at_job_boundaries(
        jobs,
        original_intervals,
    )

    total_processing = sum(
        _to_int(job.processing, f"{job.name}.processing")
        for job in jobs
    )

    edges, metadata = build_scheduling_network(
        jobs,
        intervals,
    )

    ecl_dir = Path(ecl_dir).resolve()

    if work_folder is not None:
        work_dir = Path(work_folder).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    elif keep_work_files:
        work_dir = Path.cwd() / "ecl_pass_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(
            prefix="ecl_maxflow_passes_"
        )
        work_dir = Path(temp_context.name)

    try:
        # ----------------------------------------------------
        # PASS 1: green only
        # ----------------------------------------------------
        p1_run = _run_initial_pass(
            edges,
            metadata,
            ecl_dir=ecl_dir,
            work_dir=work_dir,
            keep_files=keep_work_files,
            verbose_ecl=verbose_ecl,
        )

        pass1 = _snapshot_pass(
            edges,
            metadata,
            jobs,
            intervals,
            added_flow=p1_run["added_flow"],
            gpu_seconds=p1_run["gpu_seconds"],
        )

        # ----------------------------------------------------
        # PASS 2: lock green, open brown
        # ----------------------------------------------------
        _set_pass2_capacities(
            edges,
            metadata,
            intervals,
        )

        p2_run = _run_augmentation_pass(
            "pass2",
            edges,
            metadata,
            ecl_dir=ecl_dir,
            work_dir=work_dir,
            keep_files=keep_work_files,
            verbose_ecl=verbose_ecl,
        )

        pass2 = _snapshot_pass(
            edges,
            metadata,
            jobs,
            intervals,
            added_flow=p2_run["added_flow"],
            gpu_seconds=p2_run["gpu_seconds"],
        )

        # ----------------------------------------------------
        # PASS 3: lock brown, open red, only if needed
        # ----------------------------------------------------
        if pass2["flow"] < total_processing:
            _set_pass3_capacities(
                edges,
                metadata,
                intervals,
            )

            p3_run = _run_augmentation_pass(
                "pass3",
                edges,
                metadata,
                ecl_dir=ecl_dir,
                work_dir=work_dir,
                keep_files=keep_work_files,
                verbose_ecl=verbose_ecl,
            )

            pass3 = _snapshot_pass(
                edges,
                metadata,
                jobs,
                intervals,
                added_flow=p3_run["added_flow"],
                gpu_seconds=p3_run["gpu_seconds"],
            )

            final = pass3
            final_pass = 3

        else:
            pass3 = None
            final = pass2
            final_pass = 2

        feasible = final["flow"] == total_processing

        timeline = build_timeline(
            jobs,
            intervals,
            final["assignment"],
        )

        gpu_times = [
            x
            for x in (
                pass1.get("gpu_seconds"),
                pass2.get("gpu_seconds"),
                pass3.get("gpu_seconds") if pass3 is not None else None,
            )
            if x is not None
        ]

        result = {
            "algorithm": "GPU ECL Max Flow + Passes",
            "jobs": jobs,
            "intervals": intervals,
            "total_processing": total_processing,
            "pass1": pass1,
            "pass2": pass2,
            "pass3": pass3,
            "final_pass": final_pass,
            "final": final,
            "feasible": feasible,
            "schedule_rows": final["schedule_rows"],
            "timeline": timeline,
            "normalized_cost": schedule_cost(final),
            "gpu_maxflow_seconds": sum(gpu_times),
            "wall_seconds": time.perf_counter() - wall_start,
        }

        if verbose:
            print_gpu_passes_result(result)

        return result

    finally:
        if temp_context is not None:
            temp_context.cleanup()


# Keep the old public function name convenient for callers.
def max_flow_passes_schedule(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    *,
    ecl_dir: str | Path,
    verbose: bool = True,
    verbose_ecl: bool = False,
    keep_work_files: bool = False,
    work_folder: Optional[str | Path] = None,
) -> Dict[str, Any]:
    return max_flow_passes_schedule_gpu(
        jobs,
        original_intervals,
        ecl_dir=ecl_dir,
        verbose=verbose,
        verbose_ecl=verbose_ecl,
        keep_work_files=keep_work_files,
        work_folder=work_folder,
    )


# ============================================================
# Printing / CSV helpers
# ============================================================

def fmt(x: float) -> str:
    if abs(x - round(x)) <= EPS:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def print_pass(
    title: str,
    p: Dict[str, Any],
    intervals: List[EnergyInterval],
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Added flow in this pass = {fmt(p['added_flow'])}")
    print(f"Total flow currently present = {fmt(p['flow'])}")
    print(
        "Energy usage: "
        f"green={fmt(p['energy_usage']['green'])}, "
        f"brown={fmt(p['energy_usage']['brown'])}, "
        f"red={fmt(p['energy_usage']['red'])}"
    )

    for i, interval in enumerate(intervals):
        print(
            f"  {interval.name:>4} "
            f"[{fmt(interval.start)}, {fmt(interval.end)}] "
            f"{interval.energy:<5} "
            f"used={fmt(p['interval_usage'][i])}/{fmt(interval.length)}"
        )


def print_schedule_table(rows: List[Dict[str, Any]]) -> None:
    print(
        f"{'Job':<8}"
        f"{'Interval':<10}"
        f"{'Start':<10}"
        f"{'End':<10}"
        f"{'Energy':<10}"
        f"{'Flow':<8}"
    )
    print("-" * 56)

    for row in rows:
        print(
            f"{row['Job']:<8}"
            f"{row['Interval']:<10}"
            f"{fmt(row['Start']):<10}"
            f"{fmt(row['End']):<10}"
            f"{row['Energy']:<10}"
            f"{fmt(row['Flow']):<8}"
        )


def print_gpu_passes_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("GPU ECL MAX FLOW + PASSES")
    print("=" * 72)

    print_pass(
        "PASS 1: GREEN ONLY",
        result["pass1"],
        result["intervals"],
    )

    print_pass(
        "PASS 2: KEEP PASS-1 FLOW, LOCK GREEN, OPEN BROWN",
        result["pass2"],
        result["intervals"],
    )

    if result["pass3"] is not None:
        print_pass(
            "PASS 3: KEEP PASS-2 FLOW, LOCK BROWN, OPEN RED",
            result["pass3"],
            result["intervals"],
        )

    final = result["final"]
    usage = final["energy_usage"]

    print("\nFINAL")
    print("-" * 72)
    print(f"Feasible: {result['feasible']}")
    print(
        f"Flow: {fmt(final['flow'])}/"
        f"{fmt(result['total_processing'])}"
    )
    print(f"Final pass: {result['final_pass']}")
    print(f"Normalized cost: {fmt(result['normalized_cost'])}")
    print(
        "Energy usage: "
        f"green={fmt(usage['green'])}, "
        f"brown={fmt(usage['brown'])}, "
        f"red={fmt(usage['red'])}"
    )
    print(
        f"Sum of ECL-reported GPU max-flow runtimes: "
        f"{1000.0 * result['gpu_maxflow_seconds']:.6f} ms"
    )
    print(
        f"Whole Python/DIMACS/EGR/subprocess wall time: "
        f"{1000.0 * result['wall_seconds']:.6f} ms"
    )

    print("\nFinal interval allocation:")
    print_schedule_table(result["schedule_rows"])


def save_schedule_csv(
    result: Dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)

    fieldnames = [
        "Job",
        "Interval",
        "Start",
        "End",
        "Energy",
        "Flow",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(result["schedule_rows"])
