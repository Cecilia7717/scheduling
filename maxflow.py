from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import statistics
import time

EPS = 1e-9
INF = float("inf")


# ============================================================
# Data structures
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
    energy: str  # "green", "brown", or "red"

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class FlowEdge:
    to: int
    rev: int
    capacity: float          # current residual capacity
    original_capacity: float # current total capacity of this edge


@dataclass
class CostEdge:
    to: int
    rev: int
    capacity: float
    cost: float
    original_capacity: float


# ============================================================
# Dinic max flow
# ============================================================

class Dinic:
    def __init__(self, vertices: int):
        self.V = vertices
        self.graph: List[List[FlowEdge]] = [[] for _ in range(vertices)]
        self.level = [-1] * vertices
        self.it = [0] * vertices

    def add_edge(self, u: int, v: int, capacity: float) -> FlowEdge:
        forward = FlowEdge(
            to=v,
            rev=len(self.graph[v]),
            capacity=capacity,
            original_capacity=capacity,
        )
        backward = FlowEdge(
            to=u,
            rev=len(self.graph[u]),
            capacity=0.0,
            original_capacity=0.0,
        )
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        return forward

    def _bfs(self, source: int, sink: int) -> bool:
        self.level = [-1] * self.V
        self.level[source] = 0
        q = deque([source])

        while q:
            u = q.popleft()
            for edge in self.graph[u]:
                if edge.capacity > EPS and self.level[edge.to] < 0:
                    self.level[edge.to] = self.level[u] + 1
                    q.append(edge.to)

        return self.level[sink] >= 0

    def _dfs(self, u: int, sink: int, pushed: float) -> float:
        if u == sink:
            return pushed

        while self.it[u] < len(self.graph[u]):
            edge = self.graph[u][self.it[u]]

            if edge.capacity > EPS and self.level[edge.to] == self.level[u] + 1:
                flow = self._dfs(edge.to, sink, min(pushed, edge.capacity))
                if flow > EPS:
                    edge.capacity -= flow
                    self.graph[edge.to][edge.rev].capacity += flow
                    return flow

            self.it[u] += 1

        return 0.0

    def max_flow(self, source: int, sink: int, flow_limit: float = INF) -> float:
        """
        Return ONLY the additional flow added during this call.

        This is important for the incremental three-pass algorithm:
        the same residual network is reused across all passes.
        """
        added_flow = 0.0

        while added_flow + EPS < flow_limit and self._bfs(source, sink):
            self.it = [0] * self.V

            while added_flow + EPS < flow_limit:
                pushed = self._dfs(source, sink, flow_limit - added_flow)
                if pushed <= EPS:
                    break
                added_flow += pushed

        return added_flow

    @staticmethod
    def edge_flow(edge: FlowEdge) -> float:
        return edge.original_capacity - edge.capacity

    def set_total_capacity_preserve_flow(
        self,
        u: int,
        edge: FlowEdge,
        new_total_capacity: float,
    ) -> None:
        """
        Change an edge's TOTAL capacity without deleting its existing flow.

        If the edge currently carries f units, after this operation:
            total capacity    = new_total_capacity
            residual capacity = new_total_capacity - f

        Existing flow is therefore kept as the starting flow for the next pass.
        """
        reverse = self.graph[edge.to][edge.rev]
        current_flow = reverse.capacity

        if new_total_capacity + EPS < current_flow:
            raise ValueError(
                f"Cannot reduce capacity below existing flow: "
                f"new cap={new_total_capacity}, current flow={current_flow}"
            )

        edge.original_capacity = new_total_capacity
        edge.capacity = max(0.0, new_total_capacity - current_flow)


# ============================================================
# Min-cost max-flow
# ============================================================

class MinCostMaxFlow:
    def __init__(self, vertices: int):
        self.V = vertices
        self.graph: List[List[CostEdge]] = [[] for _ in range(vertices)]

    def add_edge(self, u: int, v: int, capacity: float, cost: float = 0.0) -> CostEdge:
        forward = CostEdge(
            to=v,
            rev=len(self.graph[v]),
            capacity=capacity,
            cost=cost,
            original_capacity=capacity,
        )
        backward = CostEdge(
            to=u,
            rev=len(self.graph[u]),
            capacity=0.0,
            cost=-cost,
            original_capacity=0.0,
        )
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        return forward

    def _shortest_path(self, source: int, sink: int):
        dist = [INF] * self.V
        in_queue = [False] * self.V
        parent_node = [-1] * self.V
        parent_edge = [-1] * self.V

        dist[source] = 0.0
        q = deque([source])
        in_queue[source] = True

        while q:
            u = q.popleft()
            in_queue[u] = False

            for ei, edge in enumerate(self.graph[u]):
                if edge.capacity <= EPS:
                    continue

                nd = dist[u] + edge.cost
                if nd + EPS < dist[edge.to]:
                    dist[edge.to] = nd
                    parent_node[edge.to] = u
                    parent_edge[edge.to] = ei

                    if not in_queue[edge.to]:
                        q.append(edge.to)
                        in_queue[edge.to] = True

        return dist, parent_node, parent_edge

    def min_cost_max_flow(
        self,
        source: int,
        sink: int,
        flow_limit: float = INF,
    ) -> Tuple[float, float]:
        total_flow = 0.0
        total_cost = 0.0

        while total_flow + EPS < flow_limit:
            dist, parent_node, parent_edge = self._shortest_path(source, sink)

            if dist[sink] == INF:
                break

            add = flow_limit - total_flow
            v = sink

            while v != source:
                u = parent_node[v]
                if u == -1:
                    add = 0.0
                    break
                edge = self.graph[u][parent_edge[v]]
                add = min(add, edge.capacity)
                v = u

            if add <= EPS:
                break

            v = sink
            while v != source:
                u = parent_node[v]
                ei = parent_edge[v]
                edge = self.graph[u][ei]
                reverse = self.graph[v][edge.rev]

                edge.capacity -= add
                reverse.capacity += add
                v = u

            total_flow += add
            total_cost += add * dist[sink]

        return total_flow, total_cost

    @staticmethod
    def edge_flow(edge: CostEdge) -> float:
        return edge.original_capacity - edge.capacity


# ============================================================
# Input validation and interval splitting
# ============================================================

def validate_input(jobs: List[Job], intervals: List[EnergyInterval]) -> None:
    if not jobs:
        raise ValueError("At least one job is required.")
    if not intervals:
        raise ValueError("At least one energy interval is required.")

    for job in jobs:
        if job.deadline <= job.release + EPS:
            raise ValueError(f"{job.name}: deadline must be greater than release.")
        if job.processing < -EPS:
            raise ValueError(f"{job.name}: processing cannot be negative.")
        if job.processing > job.deadline - job.release + EPS:
            raise ValueError(
                f"{job.name}: processing={job.processing} exceeds "
                f"window length={job.deadline - job.release}."
            )

    valid = {"green", "brown", "red"}
    ordered = sorted(intervals, key=lambda x: (x.start, x.end))

    for interval in ordered:
        if interval.energy not in valid:
            raise ValueError(
                f"{interval.name}: energy must be one of {sorted(valid)}."
            )
        if interval.end <= interval.start + EPS:
            raise ValueError(f"{interval.name}: invalid interval.")

    for prev, cur in zip(ordered, ordered[1:]):
        if abs(prev.end - cur.start) > EPS:
            raise ValueError(
                "Energy intervals must form a continuous, non-overlapping partition."
            )


def split_intervals_at_job_boundaries(
    jobs: List[Job],
    intervals: List[EnergyInterval],
) -> List[EnergyInterval]:
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
                f"{job.name}: window [{job.release}, {job.deadline}] outside "
                f"[{horizon_start}, {horizon_end}]."
            )
        boundaries.add(job.release)
        boundaries.add(job.deadline)

    points = sorted(boundaries)
    result: List[EnergyInterval] = []

    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        if b <= a + EPS:
            continue

        parent = None
        for interval in ordered:
            if interval.start <= a + EPS and b <= interval.end + EPS:
                parent = interval
                break

        if parent is None:
            raise RuntimeError("Could not map split interval to original interval.")

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
# Incremental max-flow network
#
# PASS 1:
#   green -> sink capacity = interval length
#   brown/red -> sink capacity = 0
#
# PASS 2:
#   keep the Pass-1 flow in the SAME residual graph
#   lock every green interval to exactly its Pass-1 occupied amount
#   restore every brown interval to its full interval length
#   red remains 0
#
# PASS 3:
#   keep the Pass-2 flow in the SAME residual graph
#   lock every brown interval to exactly its Pass-2 occupied amount
#   green remains locked at its Pass-1 occupied amount
#   restore every red interval to full interval length
# ============================================================

def build_incremental_dinic_network(
    jobs: List[Job],
    intervals: List[EnergyInterval],
):
    n_jobs = len(jobs)
    n_intervals = len(intervals)

    source = 0
    job_offset = 1
    interval_offset = job_offset + n_jobs
    sink = interval_offset + n_intervals

    network = Dinic(sink + 1)

    source_job_edges: Dict[int, FlowEdge] = {}
    job_interval_edges: Dict[Tuple[int, int], FlowEdge] = {}
    interval_out_edges: Dict[int, FlowEdge] = {}

    for j, job in enumerate(jobs):
        source_job_edges[j] = network.add_edge(
            source,
            job_offset + j,
            job.processing,
        )

    for j, job in enumerate(jobs):
        jnode = job_offset + j

        for i, interval in enumerate(intervals):
            if (
                job.release <= interval.start + EPS
                and interval.end <= job.deadline + EPS
            ):
                inode = interval_offset + i
                job_interval_edges[(j, i)] = network.add_edge(
                    jnode,
                    inode,
                    interval.length,
                )

    # Pass 1 capacities.
    for i, interval in enumerate(intervals):
        inode = interval_offset + i
        initial_cap = interval.length if interval.energy == "green" else 0.0
        interval_out_edges[i] = network.add_edge(inode, sink, initial_cap)

    metadata = {
        "source": source,
        "sink": sink,
        "job_offset": job_offset,
        "interval_offset": interval_offset,
        "source_job_edges": source_job_edges,
        "job_interval_edges": job_interval_edges,
        "interval_out_edges": interval_out_edges,
    }

    return network, metadata


def recover_result(
    network,
    metadata: Dict[str, Any],
    jobs: List[Job],
    intervals: List[EnergyInterval],
    flow: float,
    cost: float = 0.0,
) -> Dict[str, Any]:
    assignment: Dict[Tuple[int, int], float] = {}

    for key, edge in metadata["job_interval_edges"].items():
        amount = network.edge_flow(edge)
        if amount > EPS:
            assignment[key] = amount

    interval_usage: Dict[int, float] = {}
    energy_usage = {"green": 0.0, "brown": 0.0, "red": 0.0}

    for i, interval in enumerate(intervals):
        edge = metadata["interval_out_edges"][i]
        amount = network.edge_flow(edge)
        interval_usage[i] = amount
        energy_usage[interval.energy] += amount

    job_usage: Dict[int, float] = {}
    for j in range(len(jobs)):
        edge = metadata["source_job_edges"][j]
        job_usage[j] = network.edge_flow(edge)

    return {
        "flow": flow,
        "cost": cost,
        "assignment": assignment,
        "interval_usage": interval_usage,
        "energy_usage": energy_usage,
        "job_usage": job_usage,
    }


def build_timeline(
    jobs: List[Job],
    intervals: List[EnergyInterval],
    assignment: Dict[Tuple[int, int], float],
) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []

    for i, interval in enumerate(intervals):
        cursor = interval.start

        pieces = [
            (j, amount)
            for (j, ii), amount in assignment.items()
            if ii == i and amount > EPS
        ]

        # A deterministic concrete ordering inside each elementary interval.
        pieces.sort(key=lambda x: (jobs[x[0]].deadline, jobs[x[0]].name))

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


ENERGY_COST = {
    "green": 0.0,
    "brown": 1.0,
    "red": 2.0,
}


def schedule_cost(flow_result: Dict[str, Any]) -> float:
    usage = flow_result["energy_usage"]
    return (
        ENERGY_COST["green"] * usage["green"]
        + ENERGY_COST["brown"] * usage["brown"]
        + ENERGY_COST["red"] * usage["red"]
    )


def max_flow_passes_schedule(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    verbose: bool = True,
) -> Dict[str, Any]:
    intervals = split_intervals_at_job_boundaries(jobs, original_intervals)
    total_processing = sum(job.processing for job in jobs)

    network, metadata = build_incremental_dinic_network(jobs, intervals)
    source = metadata["source"]
    sink = metadata["sink"]
    out_edges = metadata["interval_out_edges"]

    # --------------------------------------------------------
    # PASS 1: green only
    # --------------------------------------------------------
    added1 = network.max_flow(source, sink)
    total_flow1 = added1

    pass1 = recover_result(
        network, metadata, jobs, intervals, total_flow1, 0.0
    )

    # --------------------------------------------------------
    # PASS 2:
    #   - preserve the exact flow already present
    #   - lock each green interval to its Pass-1 occupied value
    #   - restore each brown interval to full capacity
    #   - red remains unavailable
    # --------------------------------------------------------
    for i, interval in enumerate(intervals):
        edge = out_edges[i]
        inode = metadata["interval_offset"] + i

        if interval.energy == "green":
            occupied = pass1["interval_usage"][i]
            network.set_total_capacity_preserve_flow(
                inode, edge, occupied
            )
        elif interval.energy == "brown":
            network.set_total_capacity_preserve_flow(
                inode, edge, interval.length
            )
        else:  # red
            network.set_total_capacity_preserve_flow(
                inode, edge, 0.0
            )

    added2 = network.max_flow(
        source,
        sink,
        flow_limit=max(0.0, total_processing - total_flow1),
    )
    total_flow2 = total_flow1 + added2

    pass2 = recover_result(
        network, metadata, jobs, intervals, total_flow2, 0.0
    )

    # --------------------------------------------------------
    # PASS 3:
    #   - preserve the exact Pass-2 flow
    #   - green stays locked to Pass-1 usage
    #   - lock each brown interval to its Pass-2 occupied value
    #   - restore red intervals to full capacity
    # --------------------------------------------------------
    if total_flow2 + EPS < total_processing:
        for i, interval in enumerate(intervals):
            edge = out_edges[i]
            inode = metadata["interval_offset"] + i

            if interval.energy == "brown":
                occupied = pass2["interval_usage"][i]
                network.set_total_capacity_preserve_flow(
                    inode, edge, occupied
                )
            elif interval.energy == "red":
                network.set_total_capacity_preserve_flow(
                    inode, edge, interval.length
                )
            # green is intentionally unchanged from the Pass-2 lock

        added3 = network.max_flow(
            source,
            sink,
            flow_limit=max(0.0, total_processing - total_flow2),
        )
        total_flow3 = total_flow2 + added3

        pass3 = recover_result(
            network, metadata, jobs, intervals, total_flow3, 0.0
        )
        final = pass3
        final_pass = 3
    else:
        pass3 = None
        final = pass2
        final_pass = 2

    feasible = abs(final["flow"] - total_processing) <= EPS
    timeline = build_timeline(jobs, intervals, final["assignment"])

    result = {
        "algorithm": "Incremental Max Flow + Passes",
        "jobs": jobs,
        "intervals": intervals,
        "total_processing": total_processing,
        "pass1": pass1,
        "pass2": pass2,
        "pass3": pass3,
        "final_pass": final_pass,
        "final": final,
        "feasible": feasible,
        "timeline": timeline,
        "normalized_cost": schedule_cost(final),
    }

    if verbose:
        print_incremental_result(result)

    return result


# ============================================================
# Pure min-cost flow: unchanged comparison method
# ============================================================

def build_pure_min_cost_network(
    jobs: List[Job],
    intervals: List[EnergyInterval],
):
    n_jobs = len(jobs)
    n_intervals = len(intervals)

    source = 0
    job_offset = 1
    interval_offset = job_offset + n_jobs
    sink = interval_offset + n_intervals

    network = MinCostMaxFlow(sink + 1)

    source_job_edges: Dict[int, CostEdge] = {}
    job_interval_edges: Dict[Tuple[int, int], CostEdge] = {}
    interval_out_edges: Dict[int, CostEdge] = {}

    for j, job in enumerate(jobs):
        source_job_edges[j] = network.add_edge(
            source, job_offset + j, job.processing, 0.0
        )

    for j, job in enumerate(jobs):
        jnode = job_offset + j
        for i, interval in enumerate(intervals):
            if (
                job.release <= interval.start + EPS
                and interval.end <= job.deadline + EPS
            ):
                job_interval_edges[(j, i)] = network.add_edge(
                    jnode,
                    interval_offset + i,
                    interval.length,
                    0.0,
                )

    for i, interval in enumerate(intervals):
        interval_out_edges[i] = network.add_edge(
            interval_offset + i,
            sink,
            interval.length,
            ENERGY_COST[interval.energy],
        )

    metadata = {
        "source": source,
        "sink": sink,
        "job_offset": job_offset,
        "interval_offset": interval_offset,
        "source_job_edges": source_job_edges,
        "job_interval_edges": job_interval_edges,
        "interval_out_edges": interval_out_edges,
    }

    return network, metadata


def pure_min_cost_schedule(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    verbose: bool = True,
) -> Dict[str, Any]:
    intervals = split_intervals_at_job_boundaries(jobs, original_intervals)
    total_processing = sum(job.processing for job in jobs)

    network, metadata = build_pure_min_cost_network(jobs, intervals)

    flow, cost = network.min_cost_max_flow(
        metadata["source"],
        metadata["sink"],
        flow_limit=total_processing,
    )

    final = recover_result(
        network, metadata, jobs, intervals, flow, cost
    )

    result = {
        "algorithm": "Pure Min-Cost Flow",
        "jobs": jobs,
        "intervals": intervals,
        "total_processing": total_processing,
        "final": final,
        "feasible": abs(flow - total_processing) <= EPS,
        "timeline": build_timeline(jobs, intervals, final["assignment"]),
        "normalized_cost": schedule_cost(final),
    }

    if verbose:
        print_single_result(result)

    return result


# ============================================================
# Comparison and printing
# ============================================================

def benchmark_solver(solver, jobs, intervals, repeats=100):
    solver(jobs, intervals, verbose=False)  # warm-up
    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        solver(jobs, intervals, verbose=False)
        times.append(time.perf_counter() - t0)

    return {
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
    }


def compare_algorithms(
    jobs: List[Job],
    intervals: List[EnergyInterval],
    benchmark_repeats: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    passes = max_flow_passes_schedule(jobs, intervals, verbose=False)
    pure = pure_min_cost_schedule(jobs, intervals, verbose=False)

    comparison = {
        "max_flow_passes": passes,
        "pure_min_cost": pure,
    }

    if benchmark_repeats > 0:
        comparison["max_flow_passes_timing"] = benchmark_solver(
            max_flow_passes_schedule, jobs, intervals, benchmark_repeats
        )
        comparison["pure_min_cost_timing"] = benchmark_solver(
            pure_min_cost_schedule, jobs, intervals, benchmark_repeats
        )

    if verbose:
        print_comparison(comparison)

    return comparison


def fmt(x: float) -> str:
    if abs(x - round(x)) <= EPS:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def print_pass(title, p, intervals):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Total flow currently present = {fmt(p['flow'])}")
    print(
        "Energy usage: "
        f"green={fmt(p['energy_usage']['green'])}, "
        f"brown={fmt(p['energy_usage']['brown'])}, "
        f"red={fmt(p['energy_usage']['red'])}"
    )
    for i, interval in enumerate(intervals):
        print(
            f"  {interval.name:>3} "
            f"[{fmt(interval.start)}, {fmt(interval.end)}] "
            f"{interval.energy:<5} "
            f"used={fmt(p['interval_usage'][i])}/{fmt(interval.length)}"
        )


def print_incremental_result(result):
    print("\n" + "=" * 72)
    print("INCREMENTAL MAX FLOW + PASSES")
    print("=" * 72)

    print_pass("PASS 1: GREEN ONLY", result["pass1"], result["intervals"])
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

    print("\nFINAL")
    print("-" * 72)
    print(f"Feasible: {result['feasible']}")
    print(f"Flow: {fmt(result['final']['flow'])}/{fmt(result['total_processing'])}")
    print(f"Normalized cost: {fmt(result['normalized_cost'])}")
    print(
        "Energy usage: "
        f"green={fmt(result['final']['energy_usage']['green'])}, "
        f"brown={fmt(result['final']['energy_usage']['brown'])}, "
        f"red={fmt(result['final']['energy_usage']['red'])}"
    )

    print("\nFinal schedule:")
    for row in result["timeline"]:
        print(
            f"  [{fmt(row['start'])}, {fmt(row['end'])}] "
            f"{row['job']:<5} {row['energy']:<5} ({row['interval']})"
        )


def print_single_result(result):
    print("\n" + "=" * 72)
    print(result["algorithm"].upper())
    print("=" * 72)
    print(f"Feasible: {result['feasible']}")
    print(f"Flow: {fmt(result['final']['flow'])}/{fmt(result['total_processing'])}")
    print(f"Normalized cost: {fmt(result['normalized_cost'])}")
    print(
        "Energy usage: "
        f"green={fmt(result['final']['energy_usage']['green'])}, "
        f"brown={fmt(result['final']['energy_usage']['brown'])}, "
        f"red={fmt(result['final']['energy_usage']['red'])}"
    )

    print("\nFinal schedule:")
    for row in result["timeline"]:
        print(
            f"  [{fmt(row['start'])}, {fmt(row['end'])}] "
            f"{row['job']:<5} {row['energy']:<5} ({row['interval']})"
        )


def print_comparison(comparison):
    passes = comparison["max_flow_passes"]
    pure = comparison["pure_min_cost"]

    pu = passes["final"]["energy_usage"]
    mu = pure["final"]["energy_usage"]

    print("\n" + "=" * 88)
    print("COMPARISON")
    print("=" * 88)
    print(f"{'Metric':<28}{'Incremental Max Flow':>28}{'Pure Min-Cost':>28}")
    print("-" * 88)

    rows = [
        ("Feasible", str(passes["feasible"]), str(pure["feasible"])),
        ("Total flow", fmt(passes["final"]["flow"]), fmt(pure["final"]["flow"])),
        ("Green units", fmt(pu["green"]), fmt(mu["green"])),
        ("Brown units", fmt(pu["brown"]), fmt(mu["brown"])),
        ("Red units", fmt(pu["red"]), fmt(mu["red"])),
        ("Normalized cost", fmt(passes["normalized_cost"]), fmt(pure["normalized_cost"])),
    ]

    for label, a, b in rows:
        print(f"{label:<28}{a:>28}{b:>28}")

    if "max_flow_passes_timing" in comparison:
        pt = comparison["max_flow_passes_timing"]
        mt = comparison["pure_min_cost_timing"]
        print(
            f"{'Median runtime (ms)':<28}"
            f"{1000*pt['median_seconds']:>28.6f}"
            f"{1000*mt['median_seconds']:>28.6f}"
        )
