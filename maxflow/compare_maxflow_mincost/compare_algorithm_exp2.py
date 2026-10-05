#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
BASE_BENCH_FILE = HERE / "compare_algorithms.py"


# ============================================================
# Load the user's existing benchmark module.
#
# compare_algorithms.py already loads maxflow_gpu.py and exposes it
# as `alg`, so this experiment reuses the exact same Job,
# EnergyInterval, splitting, recovery, and cost logic.
# ============================================================

def load_base_benchmark():
    if not BASE_BENCH_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {BASE_BENCH_FILE.name} next to this script."
        )

    spec = importlib.util.spec_from_file_location(
        "base_compare_algorithms",
        BASE_BENCH_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_BENCH_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_base_benchmark()
alg = bench.alg

Job = alg.Job
EnergyInterval = alg.EnergyInterval
EPS = alg.EPS
INF = alg.INF


# ============================================================
# Instrumented Dinic
#
# We preserve the behavior of the user's Dinic implementation,
# but count the work performed by each max-flow call.
# ============================================================

class InstrumentedDinic(alg.Dinic):
    def __init__(self, vertices: int):
        super().__init__(vertices)
        self.last_stats: Dict[str, Any] = {}

    def _reset_call_stats(self) -> None:
        self._bfs_calls = 0
        self._successful_bfs = 0
        self._bfs_edge_scans = 0
        self._dfs_calls = 0
        self._dfs_edge_scans = 0
        self._root_dfs_calls = 0
        self._augmenting_pushes = 0

    def _bfs(self, source: int, sink: int) -> bool:
        self._bfs_calls += 1

        self.level = [-1] * self.V
        self.level[source] = 0
        q = deque([source])

        while q:
            u = q.popleft()
            for edge in self.graph[u]:
                self._bfs_edge_scans += 1
                if edge.capacity > EPS and self.level[edge.to] < 0:
                    self.level[edge.to] = self.level[u] + 1
                    q.append(edge.to)

        success = self.level[sink] >= 0
        if success:
            self._successful_bfs += 1
        return success

    def _dfs(self, u: int, sink: int, pushed: float) -> float:
        self._dfs_calls += 1

        if u == sink:
            return pushed

        while self.it[u] < len(self.graph[u]):
            edge = self.graph[u][self.it[u]]
            self._dfs_edge_scans += 1

            if edge.capacity > EPS and self.level[edge.to] == self.level[u] + 1:
                flow = self._dfs(edge.to, sink, min(pushed, edge.capacity))
                if flow > EPS:
                    edge.capacity -= flow
                    self.graph[edge.to][edge.rev].capacity += flow
                    return flow

            self.it[u] += 1

        return 0.0

    def max_flow(
        self,
        source: int,
        sink: int,
        flow_limit: float = INF,
    ) -> float:
        """
        Return the additional flow added in this call and expose detailed
        counters through self.last_stats.
        """
        self._reset_call_stats()

        added_flow = 0.0

        while added_flow + EPS < flow_limit and self._bfs(source, sink):
            self.it = [0] * self.V

            while added_flow + EPS < flow_limit:
                self._root_dfs_calls += 1
                pushed = self._dfs(
                    source,
                    sink,
                    flow_limit - added_flow,
                )
                if pushed <= EPS:
                    break

                self._augmenting_pushes += 1
                added_flow += pushed

        self.last_stats = {
            "added_flow": added_flow,
            "bfs_calls": self._bfs_calls,
            "successful_bfs": self._successful_bfs,
            "bfs_edge_scans": self._bfs_edge_scans,
            "dfs_calls": self._dfs_calls,
            "dfs_edge_scans": self._dfs_edge_scans,
            "root_dfs_calls": self._root_dfs_calls,
            "augmenting_pushes": self._augmenting_pushes,
        }
        return added_flow


# ============================================================
# Instrumented min-cost flow
#
# This counts the number of shortest-path searches and successful
# augmentations. It uses exactly the same SPFA-style shortest-path
# method as the user's current implementation.
# ============================================================

class InstrumentedMinCostMaxFlow(alg.MinCostMaxFlow):
    def __init__(self, vertices: int):
        super().__init__(vertices)
        self.last_stats: Dict[str, Any] = {}

    def _reset_call_stats(self) -> None:
        self._shortest_path_calls = 0
        self._edge_scans = 0
        self._augmentations = 0

    def _shortest_path(self, source: int, sink: int):
        self._shortest_path_calls += 1

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
                self._edge_scans += 1

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
        self._reset_call_stats()

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

            self._augmentations += 1
            total_flow += add
            total_cost += add * dist[sink]

        self.last_stats = {
            "flow": total_flow,
            "cost": total_cost,
            "shortest_path_calls": self._shortest_path_calls,
            "edge_scans": self._edge_scans,
            "augmentations": self._augmentations,
        }
        return total_flow, total_cost


# ============================================================
# Network construction
#
# Same orientation as the user's existing code:
#
#     source -> job -> interval -> sink
#
# interval_capacities lets us build each restart pass from zero.
# ============================================================

def build_dinic_network_with_caps(
    jobs: List[Job],
    intervals: List[EnergyInterval],
    interval_capacities: Dict[int, float],
):
    n_jobs = len(jobs)
    n_intervals = len(intervals)

    source = 0
    job_offset = 1
    interval_offset = job_offset + n_jobs
    sink = interval_offset + n_intervals

    network = InstrumentedDinic(sink + 1)

    source_job_edges: Dict[int, Any] = {}
    job_interval_edges: Dict[Tuple[int, int], Any] = {}
    interval_out_edges: Dict[int, Any] = {}

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
                job_interval_edges[(j, i)] = network.add_edge(
                    jnode,
                    interval_offset + i,
                    interval.length,
                )

    for i, interval in enumerate(intervals):
        interval_out_edges[i] = network.add_edge(
            interval_offset + i,
            sink,
            interval_capacities.get(i, 0.0),
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

    network_stats = {
        "vertices": sink + 1,
        "jobs": n_jobs,
        "atomic_intervals": n_intervals,
        "job_interval_edges": len(job_interval_edges),
        "forward_edges": (
            len(source_job_edges)
            + len(job_interval_edges)
            + len(interval_out_edges)
        ),
        "residual_edge_entries": 2 * (
            len(source_job_edges)
            + len(job_interval_edges)
            + len(interval_out_edges)
        ),
    }

    return network, metadata, network_stats


def build_min_cost_network(
    jobs: List[Job],
    intervals: List[EnergyInterval],
):
    n_jobs = len(jobs)
    n_intervals = len(intervals)

    source = 0
    job_offset = 1
    interval_offset = job_offset + n_jobs
    sink = interval_offset + n_intervals

    network = InstrumentedMinCostMaxFlow(sink + 1)

    source_job_edges: Dict[int, Any] = {}
    job_interval_edges: Dict[Tuple[int, int], Any] = {}
    interval_out_edges: Dict[int, Any] = {}

    for j, job in enumerate(jobs):
        source_job_edges[j] = network.add_edge(
            source,
            job_offset + j,
            job.processing,
            0.0,
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
            alg.ENERGY_COST[interval.energy],
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

    network_stats = {
        "vertices": sink + 1,
        "jobs": n_jobs,
        "atomic_intervals": n_intervals,
        "job_interval_edges": len(job_interval_edges),
        "forward_edges": (
            len(source_job_edges)
            + len(job_interval_edges)
            + len(interval_out_edges)
        ),
        "residual_edge_entries": 2 * (
            len(source_job_edges)
            + len(job_interval_edges)
            + len(interval_out_edges)
        ),
    }

    return network, metadata, network_stats


# ============================================================
# Rerouting measurement
# ============================================================

def assignment_rerouting(
    previous: Dict[Tuple[int, int], float],
    current: Dict[Tuple[int, int], float],
    intervals: List[EnergyInterval],
    allowed_energy_types: Iterable[str],
) -> Dict[str, float]:
    """
    Measure how much existing interval->job allocation changed.

    The L1 difference counts a moved unit twice:
       -1 from the old job and +1 to the new job.
    Therefore rerouted_amount = 0.5 * L1 difference.
    """
    allowed = set(allowed_energy_types)
    keys = set(previous) | set(current)

    l1 = 0.0
    previous_total = 0.0

    for key in keys:
        _, interval_idx = key
        if intervals[interval_idx].energy not in allowed:
            continue

        before = previous.get(key, 0.0)
        after = current.get(key, 0.0)

        l1 += abs(after - before)
        previous_total += before

    rerouted = 0.5 * l1
    fraction = rerouted / previous_total if previous_total > EPS else 0.0

    return {
        "rerouted_amount": rerouted,
        "previous_flow": previous_total,
        "rerouted_fraction": fraction,
    }


# ============================================================
# Solver 1: proposed incremental MaxFlow-Passes with residual reuse
# ============================================================

def max_flow_passes_instrumented(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    verbose: bool = False,
) -> Dict[str, Any]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    intervals = alg.split_intervals_at_job_boundaries(
        jobs,
        original_intervals,
    )
    split_seconds = time.perf_counter() - t0

    total_processing = sum(job.processing for job in jobs)

    initial_caps = {
        i: interval.length if interval.energy == "green" else 0.0
        for i, interval in enumerate(intervals)
    }

    t0 = time.perf_counter()
    network, metadata, network_stats = build_dinic_network_with_caps(
        jobs,
        intervals,
        initial_caps,
    )
    initial_build_seconds = time.perf_counter() - t0

    source = metadata["source"]
    sink = metadata["sink"]
    out_edges = metadata["interval_out_edges"]

    # -------------------------
    # Pass 1: green
    # -------------------------
    t0 = time.perf_counter()
    added1 = network.max_flow(source, sink)
    p1_maxflow_seconds = time.perf_counter() - t0
    p1_stats = dict(network.last_stats)

    total_flow1 = added1

    t0 = time.perf_counter()
    pass1 = alg.recover_result(
        network,
        metadata,
        jobs,
        intervals,
        total_flow1,
        0.0,
    )
    p1_recover_seconds = time.perf_counter() - t0

    # -------------------------
    # Pass 2: open brown, preserve residual graph
    # -------------------------
    t0 = time.perf_counter()
    for i, interval in enumerate(intervals):
        edge = out_edges[i]
        inode = metadata["interval_offset"] + i

        if interval.energy == "green":
            occupied = pass1["interval_usage"][i]
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                occupied,
            )
        elif interval.energy == "brown":
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                interval.length,
            )
        else:
            network.set_total_capacity_preserve_flow(
                inode,
                edge,
                0.0,
            )
    p2_transition_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    added2 = network.max_flow(
        source,
        sink,
        flow_limit=max(0.0, total_processing - total_flow1),
    )
    p2_maxflow_seconds = time.perf_counter() - t0
    p2_stats = dict(network.last_stats)

    total_flow2 = total_flow1 + added2

    t0 = time.perf_counter()
    pass2 = alg.recover_result(
        network,
        metadata,
        jobs,
        intervals,
        total_flow2,
        0.0,
    )
    p2_recover_seconds = time.perf_counter() - t0

    reroute_p2 = assignment_rerouting(
        pass1["assignment"],
        pass2["assignment"],
        intervals,
        {"green"},
    )

    # -------------------------
    # Pass 3: open red, preserve residual graph
    # -------------------------
    pass3 = None
    added3 = 0.0
    p3_transition_seconds = 0.0
    p3_maxflow_seconds = 0.0
    p3_recover_seconds = 0.0
    p3_stats = {
        "added_flow": 0.0,
        "bfs_calls": 0,
        "successful_bfs": 0,
        "bfs_edge_scans": 0,
        "dfs_calls": 0,
        "dfs_edge_scans": 0,
        "root_dfs_calls": 0,
        "augmenting_pushes": 0,
    }
    reroute_p3 = {
        "rerouted_amount": 0.0,
        "previous_flow": total_flow2,
        "rerouted_fraction": 0.0,
    }

    if total_flow2 + EPS < total_processing:
        t0 = time.perf_counter()
        for i, interval in enumerate(intervals):
            edge = out_edges[i]
            inode = metadata["interval_offset"] + i

            if interval.energy == "brown":
                occupied = pass2["interval_usage"][i]
                network.set_total_capacity_preserve_flow(
                    inode,
                    edge,
                    occupied,
                )
            elif interval.energy == "red":
                network.set_total_capacity_preserve_flow(
                    inode,
                    edge,
                    interval.length,
                )
        p3_transition_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        added3 = network.max_flow(
            source,
            sink,
            flow_limit=max(0.0, total_processing - total_flow2),
        )
        p3_maxflow_seconds = time.perf_counter() - t0
        p3_stats = dict(network.last_stats)

        total_flow3 = total_flow2 + added3

        t0 = time.perf_counter()
        pass3 = alg.recover_result(
            network,
            metadata,
            jobs,
            intervals,
            total_flow3,
            0.0,
        )
        p3_recover_seconds = time.perf_counter() - t0

        reroute_p3 = assignment_rerouting(
            pass2["assignment"],
            pass3["assignment"],
            intervals,
            {"green", "brown"},
        )

        final = pass3
        final_pass = 3
    else:
        final = pass2
        final_pass = 2

    feasible = abs(final["flow"] - total_processing) <= EPS

    t0 = time.perf_counter()
    timeline = alg.build_timeline(
        jobs,
        intervals,
        final["assignment"],
    )
    timeline_seconds = time.perf_counter() - t0

    total_seconds = time.perf_counter() - total_start

    def frac(x: float) -> float:
        return x / total_processing if total_processing > EPS else 0.0

    result = {
        "algorithm": "MaxFlow-Passes-Reuse",
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
        "normalized_cost": alg.schedule_cost(final),
        "diagnostics": {
            "network": network_stats,
            "flow": {
                "pass1_added": added1,
                "pass2_added": added2,
                "pass3_added": added3,
                "pass1_fraction": frac(added1),
                "pass2_fraction": frac(added2),
                "pass3_fraction": frac(added3),
                "pass1_cumulative": total_flow1,
                "pass2_cumulative": total_flow2,
                "final_flow": final["flow"],
            },
            "pass1": p1_stats,
            "pass2": p2_stats,
            "pass3": p3_stats,
            "rerouting": {
                "pass2_green": reroute_p2,
                "pass3_green_brown": reroute_p3,
            },
            "timing": {
                "split_seconds": split_seconds,
                "initial_build_seconds": initial_build_seconds,
                "pass1_transition_seconds": 0.0,
                "pass1_maxflow_seconds": p1_maxflow_seconds,
                "pass1_recover_seconds": p1_recover_seconds,
                "pass2_transition_seconds": p2_transition_seconds,
                "pass2_maxflow_seconds": p2_maxflow_seconds,
                "pass2_recover_seconds": p2_recover_seconds,
                "pass3_transition_seconds": p3_transition_seconds,
                "pass3_maxflow_seconds": p3_maxflow_seconds,
                "pass3_recover_seconds": p3_recover_seconds,
                "timeline_seconds": timeline_seconds,
                "internal_total_seconds": total_seconds,
            },
        },
    }

    if verbose:
        print(
            "MaxFlow-Passes-Reuse:",
            f"flow={final['flow']:.2f}/{total_processing:.2f}",
            f"cost={result['normalized_cost']:.2f}",
        )

    return result


# ============================================================
# Solver 2: MaxFlow-Restart ablation
#
# Same three carbon phases, but every phase rebuilds the network
# and recomputes a maximum flow from zero.
#
# This isolates the benefit of retaining the residual graph.
# ============================================================

def max_flow_restart_schedule(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Restart-from-zero ablation.

    The important point is that a naive cumulative max-flow restart is NOT
    equivalent to the cheapest-first algorithm: a fresh max-flow solve on
    green+brown could give up previously maximized green usage.  To preserve
    the exact lexicographic/cheapest-first objective, this ablation rebuilds
    the network at every phase and REPLAYS all cheaper passes from zero.

    Thus:
      phase 1: build -> green
      phase 2: rebuild -> green -> brown
      phase 3: rebuild -> green -> brown -> red

    The final schedule is still optimal, but previous work is discarded and
    recomputed.  This isolates the benefit of carrying the residual graph
    forward between phases.
    """
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    intervals = alg.split_intervals_at_job_boundaries(
        jobs,
        original_intervals,
    )
    split_seconds = time.perf_counter() - t0

    total_processing = sum(job.processing for job in jobs)

    def run_prefix(max_level: int) -> Dict[str, Any]:
        """Run cheapest-first passes 1..max_level on a brand-new graph."""
        initial_caps = {
            i: interval.length if interval.energy == "green" else 0.0
            for i, interval in enumerate(intervals)
        }

        t_build = time.perf_counter()
        network, metadata, network_stats = build_dinic_network_with_caps(
            jobs,
            intervals,
            initial_caps,
        )
        build_seconds = time.perf_counter() - t_build

        source = metadata["source"]
        sink = metadata["sink"]
        out_edges = metadata["interval_out_edges"]

        # Green pass.
        t = time.perf_counter()
        added1 = network.max_flow(source, sink)
        p1_maxflow_seconds = time.perf_counter() - t
        p1_stats = dict(network.last_stats)
        flow1 = added1

        t = time.perf_counter()
        pass1_local = alg.recover_result(
            network, metadata, jobs, intervals, flow1, 0.0
        )
        p1_recover_seconds = time.perf_counter() - t

        pass2_local = None
        pass3_local = None
        p2_transition_seconds = 0.0
        p2_maxflow_seconds = 0.0
        p2_recover_seconds = 0.0
        p3_transition_seconds = 0.0
        p3_maxflow_seconds = 0.0
        p3_recover_seconds = 0.0

        zero_stats = {
            "added_flow": 0.0,
            "bfs_calls": 0,
            "successful_bfs": 0,
            "bfs_edge_scans": 0,
            "dfs_calls": 0,
            "dfs_edge_scans": 0,
            "root_dfs_calls": 0,
            "augmenting_pushes": 0,
        }
        p2_stats = dict(zero_stats)
        p3_stats = dict(zero_stats)
        added2 = 0.0
        added3 = 0.0
        flow2 = flow1
        flow3 = flow1

        if max_level >= 2:
            t = time.perf_counter()
            for i, interval in enumerate(intervals):
                edge = out_edges[i]
                inode = metadata["interval_offset"] + i

                if interval.energy == "green":
                    network.set_total_capacity_preserve_flow(
                        inode,
                        edge,
                        pass1_local["interval_usage"][i],
                    )
                elif interval.energy == "brown":
                    network.set_total_capacity_preserve_flow(
                        inode,
                        edge,
                        interval.length,
                    )
                else:
                    network.set_total_capacity_preserve_flow(
                        inode,
                        edge,
                        0.0,
                    )
            p2_transition_seconds = time.perf_counter() - t

            t = time.perf_counter()
            added2 = network.max_flow(
                source,
                sink,
                flow_limit=max(0.0, total_processing - flow1),
            )
            p2_maxflow_seconds = time.perf_counter() - t
            p2_stats = dict(network.last_stats)
            flow2 = flow1 + added2

            t = time.perf_counter()
            pass2_local = alg.recover_result(
                network, metadata, jobs, intervals, flow2, 0.0
            )
            p2_recover_seconds = time.perf_counter() - t

        if (
            max_level >= 3
            and flow2 + EPS < total_processing
            and pass2_local is not None
        ):
            t = time.perf_counter()
            for i, interval in enumerate(intervals):
                edge = out_edges[i]
                inode = metadata["interval_offset"] + i

                if interval.energy == "brown":
                    network.set_total_capacity_preserve_flow(
                        inode,
                        edge,
                        pass2_local["interval_usage"][i],
                    )
                elif interval.energy == "red":
                    network.set_total_capacity_preserve_flow(
                        inode,
                        edge,
                        interval.length,
                    )
            p3_transition_seconds = time.perf_counter() - t

            t = time.perf_counter()
            added3 = network.max_flow(
                source,
                sink,
                flow_limit=max(0.0, total_processing - flow2),
            )
            p3_maxflow_seconds = time.perf_counter() - t
            p3_stats = dict(network.last_stats)
            flow3 = flow2 + added3

            t = time.perf_counter()
            pass3_local = alg.recover_result(
                network, metadata, jobs, intervals, flow3, 0.0
            )
            p3_recover_seconds = time.perf_counter() - t
        else:
            flow3 = flow2

        final_local = (
            pass3_local
            if pass3_local is not None
            else pass2_local
            if pass2_local is not None
            else pass1_local
        )

        return {
            "network_stats": network_stats,
            "pass1": pass1_local,
            "pass2": pass2_local,
            "pass3": pass3_local,
            "final": final_local,
            "flow1": flow1,
            "flow2": flow2,
            "flow3": flow3,
            "added1": added1,
            "added2": added2,
            "added3": added3,
            "pass1_stats": p1_stats,
            "pass2_stats": p2_stats,
            "pass3_stats": p3_stats,
            "timing": {
                "build_seconds": build_seconds,
                "pass1_transition_seconds": 0.0,
                "pass1_maxflow_seconds": p1_maxflow_seconds,
                "pass1_recover_seconds": p1_recover_seconds,
                "pass2_transition_seconds": p2_transition_seconds,
                "pass2_maxflow_seconds": p2_maxflow_seconds,
                "pass2_recover_seconds": p2_recover_seconds,
                "pass3_transition_seconds": p3_transition_seconds,
                "pass3_maxflow_seconds": p3_maxflow_seconds,
                "pass3_recover_seconds": p3_recover_seconds,
            },
        }

    # Phase 1 result is intentionally discarded before Phase 2.
    phase1 = run_prefix(1)

    # Phase 2 starts from a fresh graph and replays green before brown.
    phase2 = run_prefix(2)

    # Match the proposed algorithm: only open red if green+brown is not enough.
    if phase2["flow2"] + EPS < total_processing:
        # Phase 3 again starts from a fresh graph and replays green and brown.
        phase3 = run_prefix(3)
        chosen = phase3
        final_pass = 3
    else:
        phase3 = None
        chosen = phase2
        final_pass = 2

    pass1 = chosen["pass1"]
    pass2 = chosen["pass2"]
    pass3 = chosen["pass3"]
    final = chosen["final"]

    # Rerouting is measured inside the chosen replayed prefix so the compared
    # assignments belong to the same residual-flow sequence.
    reroute_p2 = assignment_rerouting(
        pass1["assignment"],
        pass2["assignment"],
        intervals,
        {"green"},
    )

    if pass3 is not None:
        reroute_p3 = assignment_rerouting(
            pass2["assignment"],
            pass3["assignment"],
            intervals,
            {"green", "brown"},
        )
    else:
        reroute_p3 = {
            "rerouted_amount": 0.0,
            "previous_flow": chosen["flow2"],
            "rerouted_fraction": 0.0,
        }

    feasible = abs(final["flow"] - total_processing) <= EPS

    t0 = time.perf_counter()
    timeline = alg.build_timeline(
        jobs,
        intervals,
        final["assignment"],
    )
    timeline_seconds = time.perf_counter() - t0

    total_seconds = time.perf_counter() - total_start

    def frac(x: float) -> float:
        return x / total_processing if total_processing > EPS else 0.0

    # Aggregate wall-clock work done by each restart phase.  Phase 2 includes
    # replaying green; Phase 3 includes replaying green and brown.
    phase1_total_maxflow = phase1["timing"]["pass1_maxflow_seconds"]
    phase2_total_maxflow = (
        phase2["timing"]["pass1_maxflow_seconds"]
        + phase2["timing"]["pass2_maxflow_seconds"]
    )
    phase3_total_maxflow = 0.0
    if phase3 is not None:
        phase3_total_maxflow = (
            phase3["timing"]["pass1_maxflow_seconds"]
            + phase3["timing"]["pass2_maxflow_seconds"]
            + phase3["timing"]["pass3_maxflow_seconds"]
        )

    phase1_total_recover = phase1["timing"]["pass1_recover_seconds"]
    phase2_total_recover = (
        phase2["timing"]["pass1_recover_seconds"]
        + phase2["timing"]["pass2_recover_seconds"]
    )
    phase3_total_recover = 0.0
    if phase3 is not None:
        phase3_total_recover = (
            phase3["timing"]["pass1_recover_seconds"]
            + phase3["timing"]["pass2_recover_seconds"]
            + phase3["timing"]["pass3_recover_seconds"]
        )

    result = {
        "algorithm": "MaxFlow-Restart",
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
        "normalized_cost": alg.schedule_cost(final),
        "diagnostics": {
            "network": chosen["network_stats"],
            "flow": {
                "pass1_added": chosen["added1"],
                "pass2_added": chosen["added2"],
                "pass3_added": chosen["added3"],
                "pass1_fraction": frac(chosen["added1"]),
                "pass2_fraction": frac(chosen["added2"]),
                "pass3_fraction": frac(chosen["added3"]),
                "pass1_cumulative": chosen["flow1"],
                "pass2_cumulative": chosen["flow2"],
                "final_flow": final["flow"],
            },
            "pass1": chosen["pass1_stats"],
            "pass2": chosen["pass2_stats"],
            "pass3": chosen["pass3_stats"],
            "rerouting": {
                "pass2_green": reroute_p2,
                "pass3_green_brown": reroute_p3,
            },
            "restart_replay": {
                "phase1_prefix_levels": 1,
                "phase2_prefix_levels": 2,
                "phase3_prefix_levels": 3 if phase3 is not None else 0,
                "total_network_builds": 3 if phase3 is not None else 2,
                "total_maxflow_calls": (
                    6 if phase3 is not None else 3
                ),
            },
            "timing": {
                "split_seconds": split_seconds,
                "pass1_build_seconds": phase1["timing"]["build_seconds"],
                "pass1_maxflow_seconds": phase1_total_maxflow,
                "pass1_recover_seconds": phase1_total_recover,
                "pass2_build_seconds": phase2["timing"]["build_seconds"],
                "pass2_maxflow_seconds": phase2_total_maxflow,
                "pass2_recover_seconds": phase2_total_recover,
                "pass3_build_seconds": (
                    phase3["timing"]["build_seconds"]
                    if phase3 is not None else 0.0
                ),
                "pass3_maxflow_seconds": phase3_total_maxflow,
                "pass3_recover_seconds": phase3_total_recover,
                "timeline_seconds": timeline_seconds,
                "internal_total_seconds": total_seconds,
            },
        },
    }

    if verbose:
        print(
            "MaxFlow-Restart:",
            f"flow={final['flow']:.2f}/{total_processing:.2f}",
            f"cost={result['normalized_cost']:.2f}",
        )

    return result


# ============================================================
# Solver 3: instrumented pure min-cost flow
# ============================================================

def pure_min_cost_instrumented(
    jobs: List[Job],
    original_intervals: List[EnergyInterval],
    verbose: bool = False,
) -> Dict[str, Any]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    intervals = alg.split_intervals_at_job_boundaries(
        jobs,
        original_intervals,
    )
    split_seconds = time.perf_counter() - t0

    total_processing = sum(job.processing for job in jobs)

    t0 = time.perf_counter()
    network, metadata, network_stats = build_min_cost_network(
        jobs,
        intervals,
    )
    build_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    flow, cost = network.min_cost_max_flow(
        metadata["source"],
        metadata["sink"],
        flow_limit=total_processing,
    )
    solve_seconds = time.perf_counter() - t0
    solve_stats = dict(network.last_stats)

    t0 = time.perf_counter()
    final = alg.recover_result(
        network,
        metadata,
        jobs,
        intervals,
        flow,
        cost,
    )
    recover_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    timeline = alg.build_timeline(
        jobs,
        intervals,
        final["assignment"],
    )
    timeline_seconds = time.perf_counter() - t0

    result = {
        "algorithm": "Pure-MinCostFlow",
        "jobs": jobs,
        "intervals": intervals,
        "total_processing": total_processing,
        "final": final,
        "feasible": abs(flow - total_processing) <= EPS,
        "timeline": timeline,
        "normalized_cost": alg.schedule_cost(final),
        "diagnostics": {
            "network": network_stats,
            "solve": solve_stats,
            "timing": {
                "split_seconds": split_seconds,
                "build_seconds": build_seconds,
                "solve_seconds": solve_seconds,
                "recover_seconds": recover_seconds,
                "timeline_seconds": timeline_seconds,
                "internal_total_seconds": (
                    time.perf_counter() - total_start
                ),
            },
        },
    }

    if verbose:
        print(
            "Pure-MinCostFlow:",
            f"flow={flow:.2f}/{total_processing:.2f}",
            f"cost={result['normalized_cost']:.2f}",
        )

    return result


# ============================================================
# Instance generation
#
# IMPORTANT DIFFERENCE FROM THE OLD utilization script:
# when jobs are randomly chosen from [min_jobs, max_jobs],
# horizon = horizon_per_job * num_jobs.
#
# This keeps high utilization ranges achievable instead of allowing
# a small horizon to silently cap the number of jobs or a large horizon
# to make 0.95-1.00 utilization impossible under p_j <= 6.
# ============================================================

def generate_scaled_instance(
    rng: random.Random,
    instance_id: int,
    min_jobs: int,
    max_jobs: int,
    horizon_per_job: int,
    green_share_range: Tuple[float, float],
    brown_share_range: Optional[Tuple[float, float]],
    target_utilization_range: Tuple[float, float],
    fixed_jobs: Optional[int] = None,
    fixed_intervals: Optional[int] = None,
) -> Dict[str, Any]:
    num_jobs = (
        fixed_jobs
        if fixed_jobs is not None
        else rng.randint(min_jobs, max_jobs)
    )
    horizon = horizon_per_job * num_jobs

    jobs, witness = bench.generate_feasible_jobs(
        rng,
        horizon=horizon,
        num_jobs=num_jobs,
        target_utilization_range=target_utilization_range,
    )

    energy_intervals = bench.generate_energy_intervals(
        rng,
        horizon=horizon,
        green_share_range=green_share_range,
        brown_share_range=brown_share_range,
        fixed_intervals=fixed_intervals,
    )

    return {
        "instance_id": instance_id,
        "horizon": horizon,
        "jobs": jobs,
        "energy_intervals": energy_intervals,
        "witness_schedule": witness,
    }


# ============================================================
# Benchmark helpers
# ============================================================

def summarize_times(times: List[float]) -> Dict[str, float]:
    return {
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
    }


def mean_nested_timing(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    keys = set()
    for result in results:
        keys.update(result["diagnostics"]["timing"].keys())

    out: Dict[str, float] = {}
    for key in sorted(keys):
        vals = [
            result["diagnostics"]["timing"][key]
            for result in results
            if key in result["diagnostics"]["timing"]
        ]
        if vals:
            out[key] = statistics.mean(vals)

    return out


def benchmark_solver(
    solver,
    jobs: List[Job],
    intervals: List[EnergyInterval],
    repeats: int,
):
    # Warm-up and retained structural result.
    retained = solver(jobs, intervals, verbose=False)

    times: List[float] = []
    timed_results: List[Dict[str, Any]] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        result = solver(jobs, intervals, verbose=False)
        times.append(time.perf_counter() - t0)
        timed_results.append(result)

    return (
        retained,
        summarize_times(times),
        mean_nested_timing(timed_results),
    )


# ============================================================
# Serialization
# ============================================================

def serializable_job(job: Job) -> Dict[str, Any]:
    return {
        "name": job.name,
        "release": job.release,
        "deadline": job.deadline,
        "processing": job.processing,
    }


def serializable_interval(interval: EnergyInterval) -> Dict[str, Any]:
    return {
        "name": interval.name,
        "start": interval.start,
        "end": interval.end,
        "energy": interval.energy,
        "length": interval.length,
    }


def pass_summary(pass_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if pass_result is None:
        return None
    return {
        "flow": pass_result["flow"],
        "energy_usage": dict(pass_result["energy_usage"]),
    }


def solver_json_summary(
    result: Dict[str, Any],
    benchmark_timing: Dict[str, float],
    mean_internal_timing: Dict[str, float],
) -> Dict[str, Any]:
    payload = {
        "algorithm": result["algorithm"],
        "feasible": result["feasible"],
        "total_processing": result["total_processing"],
        "normalized_cost": result["normalized_cost"],
        "final_energy_usage": dict(result["final"]["energy_usage"]),
        "diagnostics": result["diagnostics"],
        "benchmark_timing": benchmark_timing,
        "mean_internal_timing": mean_internal_timing,
    }

    if "pass1" in result:
        payload["pass1"] = pass_summary(result.get("pass1"))
        payload["pass2"] = pass_summary(result.get("pass2"))
        payload["pass3"] = pass_summary(result.get("pass3"))
        payload["final_pass"] = result.get("final_pass")

    return payload


def energy_shares(
    intervals: List[EnergyInterval],
    horizon: float,
) -> Dict[str, float]:
    totals = {"green": 0.0, "brown": 0.0, "red": 0.0}

    for interval in intervals:
        totals[interval.energy] += interval.length

    return {
        key: value / horizon
        for key, value in totals.items()
    }


def safe_ratio(a: float, b: float) -> float:
    return a / b if b > 0.0 else float("inf")


# ============================================================
# One experiment configuration
# ============================================================

def run_experiment(
    num_instances: int,
    timing_repeats: int,
    seed: int,
    output_folder: str,
    min_jobs: int,
    max_jobs: int,
    fixed_jobs: Optional[int],
    horizon_per_job: int,
    fixed_intervals: Optional[int],
    green_share_range: Tuple[float, float],
    brown_share_range: Optional[Tuple[float, float]],
    target_utilization_range: Tuple[float, float],
) -> Path:
    rng = random.Random(seed)

    output_dir = HERE / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("instance_*.json"):
        old_file.unlink()

    for old_name in ("results.csv", "summary.txt"):
        p = output_dir / old_name
        if p.exists():
            p.unlink()

    rows: List[Dict[str, Any]] = []

    print("=" * 78)
    print("EXPERIMENT 2: WHY MAXFLOW-PASSES IS FASTER")
    print("=" * 78)
    print(f"Instances:              {num_instances}")
    print(f"Timing repeats:         {timing_repeats}")
    print(f"Seed:                   {seed}")
    if fixed_jobs is None:
        print(f"Jobs:                   {min_jobs}-{max_jobs}")
    else:
        print(f"Jobs:                   {fixed_jobs} fixed")
    print(f"Horizon per job:        {horizon_per_job}")
    print(
        "Green share:            "
        f"{green_share_range[0]:.2f}-{green_share_range[1]:.2f}"
    )
    print(
        "Utilization:            "
        f"{target_utilization_range[0]:.2f}-"
        f"{target_utilization_range[1]:.2f}"
    )
    print(
        "Brown share:            "
        + (
            "uncontrolled"
            if brown_share_range is None
            else f"{brown_share_range[0]:.2f}-{brown_share_range[1]:.2f}"
        )
    )
    print(f"Output:                 {output_dir}")
    print()

    for instance_id in range(1, num_instances + 1):
        instance = generate_scaled_instance(
            rng=rng,
            instance_id=instance_id,
            min_jobs=min_jobs,
            max_jobs=max_jobs,
            horizon_per_job=horizon_per_job,
            green_share_range=green_share_range,
            brown_share_range=brown_share_range,
            target_utilization_range=target_utilization_range,
            fixed_jobs=fixed_jobs,
            fixed_intervals=fixed_intervals,
        )

        jobs = instance["jobs"]
        original_intervals = instance["energy_intervals"]
        horizon = float(instance["horizon"])

        reuse_result, reuse_timing, reuse_internal = benchmark_solver(
            max_flow_passes_instrumented,
            jobs,
            original_intervals,
            timing_repeats,
        )

        restart_result, restart_timing, restart_internal = benchmark_solver(
            max_flow_restart_schedule,
            jobs,
            original_intervals,
            timing_repeats,
        )

        mincost_result, mincost_timing, mincost_internal = benchmark_solver(
            pure_min_cost_instrumented,
            jobs,
            original_intervals,
            timing_repeats,
        )

        for result in (reuse_result, restart_result, mincost_result):
            if not result["feasible"]:
                raise RuntimeError(
                    f"{result['algorithm']} failed on feasible "
                    f"instance {instance_id}."
                )

        # All exact formulations should have the same normalized cost.
        costs = [
            reuse_result["normalized_cost"],
            restart_result["normalized_cost"],
            mincost_result["normalized_cost"],
        ]
        if max(costs) - min(costs) > 1e-7:
            raise RuntimeError(
                f"Cost mismatch on instance {instance_id}: {costs}"
            )

        total_processing = reuse_result["total_processing"]
        actual_util = total_processing / horizon
        shares = energy_shares(original_intervals, horizon)

        d = reuse_result["diagnostics"]
        net = d["network"]
        flow = d["flow"]
        r2 = d["rerouting"]["pass2_green"]
        r3 = d["rerouting"]["pass3_green_brown"]

        reuse_mean_ms = 1000.0 * reuse_timing["mean_seconds"]
        restart_mean_ms = 1000.0 * restart_timing["mean_seconds"]
        mincost_mean_ms = 1000.0 * mincost_timing["mean_seconds"]

        row = {
            "instance_id": instance_id,
            "num_jobs": len(jobs),
            "horizon": instance["horizon"],
            "actual_utilization": actual_util,
            "green_share": shares["green"],
            "brown_share": shares["brown"],
            "red_share": shares["red"],
            "original_energy_intervals": len(original_intervals),
            "atomic_intervals": net["atomic_intervals"],
            "network_vertices": net["vertices"],
            "job_interval_edges": net["job_interval_edges"],
            "forward_edges": net["forward_edges"],
            "total_processing": total_processing,

            "pass1_added_flow": flow["pass1_added"],
            "pass2_added_flow": flow["pass2_added"],
            "pass3_added_flow": flow["pass3_added"],
            "pass1_flow_fraction": flow["pass1_fraction"],
            "pass2_flow_fraction": flow["pass2_fraction"],
            "pass3_flow_fraction": flow["pass3_fraction"],
            "final_pass": reuse_result["final_pass"],

            "pass2_green_rerouted_amount": r2["rerouted_amount"],
            "pass2_green_rerouted_fraction": r2["rerouted_fraction"],
            "pass3_prior_rerouted_amount": r3["rerouted_amount"],
            "pass3_prior_rerouted_fraction": r3["rerouted_fraction"],

            "reuse_pass1_bfs_calls": d["pass1"]["bfs_calls"],
            "reuse_pass1_successful_bfs": d["pass1"]["successful_bfs"],
            "reuse_pass1_bfs_edge_scans": d["pass1"]["bfs_edge_scans"],
            "reuse_pass1_dfs_calls": d["pass1"]["dfs_calls"],
            "reuse_pass1_dfs_edge_scans": d["pass1"]["dfs_edge_scans"],
            "reuse_pass1_augmenting_pushes": d["pass1"]["augmenting_pushes"],

            "reuse_pass2_bfs_calls": d["pass2"]["bfs_calls"],
            "reuse_pass2_successful_bfs": d["pass2"]["successful_bfs"],
            "reuse_pass2_bfs_edge_scans": d["pass2"]["bfs_edge_scans"],
            "reuse_pass2_dfs_calls": d["pass2"]["dfs_calls"],
            "reuse_pass2_dfs_edge_scans": d["pass2"]["dfs_edge_scans"],
            "reuse_pass2_augmenting_pushes": d["pass2"]["augmenting_pushes"],

            "reuse_pass3_bfs_calls": d["pass3"]["bfs_calls"],
            "reuse_pass3_successful_bfs": d["pass3"]["successful_bfs"],
            "reuse_pass3_bfs_edge_scans": d["pass3"]["bfs_edge_scans"],
            "reuse_pass3_dfs_calls": d["pass3"]["dfs_calls"],
            "reuse_pass3_dfs_edge_scans": d["pass3"]["dfs_edge_scans"],
            "reuse_pass3_augmenting_pushes": d["pass3"]["augmenting_pushes"],

            "reuse_pass1_maxflow_mean_ms": (
                1000.0 * reuse_internal["pass1_maxflow_seconds"]
            ),
            "reuse_pass2_transition_mean_ms": (
                1000.0 * reuse_internal["pass2_transition_seconds"]
            ),
            "reuse_pass2_maxflow_mean_ms": (
                1000.0 * reuse_internal["pass2_maxflow_seconds"]
            ),
            "reuse_pass3_transition_mean_ms": (
                1000.0 * reuse_internal["pass3_transition_seconds"]
            ),
            "reuse_pass3_maxflow_mean_ms": (
                1000.0 * reuse_internal["pass3_maxflow_seconds"]
            ),

            "reuse_mean_ms": reuse_mean_ms,
            "reuse_median_ms": 1000.0 * reuse_timing["median_seconds"],
            "restart_mean_ms": restart_mean_ms,
            "restart_median_ms": 1000.0 * restart_timing["median_seconds"],
            "mincost_mean_ms": mincost_mean_ms,
            "mincost_median_ms": 1000.0 * mincost_timing["median_seconds"],

            "restart_over_reuse_mean_ratio": safe_ratio(
                restart_mean_ms,
                reuse_mean_ms,
            ),
            "mincost_over_reuse_mean_ratio": safe_ratio(
                mincost_mean_ms,
                reuse_mean_ms,
            ),

            "restart_pass1_maxflow_mean_ms": (
                1000.0 * restart_internal["pass1_maxflow_seconds"]
            ),
            "restart_pass2_build_mean_ms": (
                1000.0 * restart_internal["pass2_build_seconds"]
            ),
            "restart_pass2_maxflow_mean_ms": (
                1000.0 * restart_internal["pass2_maxflow_seconds"]
            ),
            "restart_pass3_build_mean_ms": (
                1000.0 * restart_internal["pass3_build_seconds"]
            ),
            "restart_pass3_maxflow_mean_ms": (
                1000.0 * restart_internal["pass3_maxflow_seconds"]
            ),

            "mincost_shortest_path_calls": (
                mincost_result["diagnostics"]["solve"]["shortest_path_calls"]
            ),
            "mincost_edge_scans": (
                mincost_result["diagnostics"]["solve"]["edge_scans"]
            ),
            "mincost_augmentations": (
                mincost_result["diagnostics"]["solve"]["augmentations"]
            ),
            "mincost_solve_mean_ms": (
                1000.0 * mincost_internal["solve_seconds"]
            ),

            "reuse_cost": reuse_result["normalized_cost"],
            "restart_cost": restart_result["normalized_cost"],
            "mincost_cost": mincost_result["normalized_cost"],
        }

        rows.append(row)

        payload = {
            "instance_id": instance_id,
            "horizon": instance["horizon"],
            "jobs": [serializable_job(j) for j in jobs],
            "energy_intervals": [
                serializable_interval(x)
                for x in original_intervals
            ],
            "construction_witness_schedule": instance["witness_schedule"],
            "actual_utilization": actual_util,
            "energy_shares": shares,
            "maxflow_passes_reuse": solver_json_summary(
                reuse_result,
                reuse_timing,
                reuse_internal,
            ),
            "maxflow_restart": solver_json_summary(
                restart_result,
                restart_timing,
                restart_internal,
            ),
            "pure_min_cost": solver_json_summary(
                mincost_result,
                mincost_timing,
                mincost_internal,
            ),
        }

        (output_dir / f"instance_{instance_id:04d}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print(
            f"[{instance_id:03d}/{num_instances}] "
            f"jobs={len(jobs):2d} "
            f"util={actual_util:.3f} "
            f"green={shares['green']:.3f} | "
            f"flow fractions="
            f"{flow['pass1_fraction']:.2f}/"
            f"{flow['pass2_fraction']:.2f}/"
            f"{flow['pass3_fraction']:.2f} | "
            f"ms reuse={reuse_mean_ms:.4f}, "
            f"restart={restart_mean_ms:.4f}, "
            f"mcf={mincost_mean_ms:.4f}"
        )

    # Write CSV.
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate summary.
    def avg(key: str) -> float:
        return statistics.mean(float(row[key]) for row in rows)

    pass3_used_count = sum(
        1 for row in rows if int(row["final_pass"]) == 3
    )

    summary = f"""Experiment 2: Why MaxFlow-Passes Is Faster
==========================================

Configuration
-------------
Seed: {seed}
Instances: {num_instances}
Timing repeats per solver per instance: {timing_repeats}
Jobs: {fixed_jobs if fixed_jobs is not None else f"{min_jobs}-{max_jobs}"}
Horizon rule: horizon = {horizon_per_job} * number_of_jobs
Green share range: {green_share_range[0]:.4f}-{green_share_range[1]:.4f}
Brown share range: {
    "uncontrolled/random split of non-green time"
    if brown_share_range is None
    else f"{brown_share_range[0]:.4f}-{brown_share_range[1]:.4f}"
}
Target utilization range:
    {target_utilization_range[0]:.4f}-{target_utilization_range[1]:.4f}

Algorithms
----------
1. MaxFlow-Passes-Reuse
   Same residual graph is retained across green -> brown -> red.

2. MaxFlow-Restart
   Same pass logic, but the flow network is rebuilt and solved from zero
   at every carbon level.

3. Pure-MinCostFlow
   Direct min-cost flow baseline.

Average Runtime
---------------
MaxFlow-Passes-Reuse: {avg("reuse_mean_ms"):.6f} ms
MaxFlow-Restart:      {avg("restart_mean_ms"):.6f} ms
Pure-MinCostFlow:     {avg("mincost_mean_ms"):.6f} ms

Restart / Reuse ratio:
    {avg("restart_over_reuse_mean_ratio"):.6f}x

MinCost / Reuse ratio:
    {avg("mincost_over_reuse_mean_ratio"):.6f}x

Where the Final Flow Is Added
-----------------------------
Pass 1 (green) fraction:
    {avg("pass1_flow_fraction"):.6f}

Pass 2 (brown) additional fraction:
    {avg("pass2_flow_fraction"):.6f}

Pass 3 (red) additional fraction:
    {avg("pass3_flow_fraction"):.6f}

Instances that required Pass 3:
    {pass3_used_count}/{num_instances}

Residual Rerouting
------------------
Fraction of Pass-1 green allocation rerouted during Pass 2:
    {avg("pass2_green_rerouted_fraction"):.6f}

Fraction of existing green+brown allocation rerouted during Pass 3:
    {avg("pass3_prior_rerouted_fraction"):.6f}

Incremental Max-Flow Work
-------------------------
Pass 1 mean BFS calls:
    {avg("reuse_pass1_bfs_calls"):.3f}
Pass 2 mean BFS calls:
    {avg("reuse_pass2_bfs_calls"):.3f}
Pass 3 mean BFS calls:
    {avg("reuse_pass3_bfs_calls"):.3f}

Pass 1 mean augmenting pushes:
    {avg("reuse_pass1_augmenting_pushes"):.3f}
Pass 2 mean augmenting pushes:
    {avg("reuse_pass2_augmenting_pushes"):.3f}
Pass 3 mean augmenting pushes:
    {avg("reuse_pass3_augmenting_pushes"):.3f}

Min-Cost-Flow Work
------------------
Mean shortest-path searches:
    {avg("mincost_shortest_path_calls"):.3f}

Mean augmentations:
    {avg("mincost_augmentations"):.3f}

Mean edge scans:
    {avg("mincost_edge_scans"):.3f}

Correctness
-----------
All three methods are required to be feasible on every generated instance.
The script also checks that all three exact methods return identical
normalized carbon cost for every instance. A mismatch aborts the run.

Files
-----
results.csv
    One row per instance with flow fractions, rerouting, graph size,
    per-pass Dinic work, min-cost-flow work, and runtime ratios.

instance_XXXX.json
    Full generated instance plus diagnostics for all three algorithms.

summary.txt
    Aggregate statistics for this configuration.
"""

    (output_dir / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("FINISHED")
    print("=" * 78)
    print(f"CSV:     {csv_path}")
    print(f"Summary: {output_dir / 'summary.txt'}")

    return output_dir


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2: compare residual-reuse MaxFlow-Passes, "
            "a restart-from-zero max-flow ablation, and pure min-cost flow."
        )
    )

    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="experiment2_results",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Use exactly this many jobs. If omitted, use a random number "
            "between --min-jobs and --max-jobs."
        ),
    )
    parser.add_argument("--min-jobs", type=int, default=35)
    parser.add_argument("--max-jobs", type=int, default=50)
    parser.add_argument(
        "--horizon-per-job",
        type=int,
        default=6,
        help=(
            "Horizon is this value times the generated number of jobs. "
            "Default: 6."
        ),
    )
    parser.add_argument(
        "--intervals",
        type=int,
        default=None,
        help="Use exactly this many original energy intervals.",
    )

    parser.add_argument("--green-min", type=float, default=0.65)
    parser.add_argument("--green-max", type=float, default=0.75)

    parser.add_argument("--brown-min", type=float, default=0.20)
    parser.add_argument("--brown-max", type=float, default=0.25)
    parser.add_argument(
        "--random-brown",
        action="store_true",
        help=(
            "Randomly split non-green time between brown and red. "
            "Recommended for wide green-share sweeps."
        ),
    )

    parser.add_argument("--util-min", type=float, default=0.55)
    parser.add_argument("--util-max", type=float, default=0.90)

    return parser.parse_args()


def validate_args(args) -> None:
    if args.instances <= 0:
        raise ValueError("--instances must be > 0.")

    if args.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be > 0.")

    if args.jobs is not None and args.jobs <= 0:
        raise ValueError("--jobs must be > 0.")

    if args.min_jobs <= 0 or args.max_jobs < args.min_jobs:
        raise ValueError("Require 0 < min-jobs <= max-jobs.")

    if args.horizon_per_job <= 0:
        raise ValueError("--horizon-per-job must be > 0.")

    if args.intervals is not None and args.intervals < 3:
        raise ValueError("--intervals must be >= 3.")

    if not (0.0 < args.green_min <= args.green_max < 1.0):
        raise ValueError(
            "Require 0 < green-min <= green-max < 1."
        )

    if not args.random_brown:
        if not (0.0 < args.brown_min <= args.brown_max < 1.0):
            raise ValueError(
                "Require 0 < brown-min <= brown-max < 1."
            )

        if args.green_max + args.brown_max >= 1.0:
            raise ValueError(
                "green-max + brown-max must be < 1 when brown is fixed. "
                "Use --random-brown for high green-share ranges."
            )

    if not (0.0 < args.util_min <= args.util_max <= 1.0):
        raise ValueError(
            "Require 0 < util-min <= util-max <= 1."
        )

    max_jobs = args.jobs if args.jobs is not None else args.max_jobs
    if (
        args.intervals is not None
        and args.intervals > args.horizon_per_job * max_jobs
    ):
        raise ValueError(
            "--intervals cannot exceed the maximum generated horizon."
        )


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    brown_share_range = (
        None
        if args.random_brown
        else (args.brown_min, args.brown_max)
    )

    run_experiment(
        num_instances=args.instances,
        timing_repeats=args.timing_repeats,
        seed=args.seed,
        output_folder=args.output,
        min_jobs=args.min_jobs,
        max_jobs=args.max_jobs,
        fixed_jobs=args.jobs,
        horizon_per_job=args.horizon_per_job,
        fixed_intervals=args.intervals,
        green_share_range=(args.green_min, args.green_max),
        brown_share_range=brown_share_range,
        target_utilization_range=(args.util_min, args.util_max),
    )
