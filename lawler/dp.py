
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf
import bisect
import heapq
from typing import Iterable


EPS = 1e-12


@dataclass(frozen=True)
class Job:
    id: str
    release: float
    processing: float
    due: float
    weight: int = 1

    def __post_init__(self):
        if self.processing <= 0:
            raise ValueError(f"{self.id}: processing time must be positive")
        if self.weight <= 0 or int(self.weight) != self.weight:
            raise ValueError(f"{self.id}: weight must be a positive integer")
        if self.due < self.release:
            raise ValueError(f"{self.id}: due date must be >= release date")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    job_id: str


def preemptive_edd_schedule(jobs: Iterable[Job]) -> tuple[list[Segment], bool]:
    """
    Construct the preemptive EDD schedule for the supplied jobs.

    Returns:
        (segments, all_on_time)
    """
    jobs = sorted(jobs, key=lambda x: (x.release, x.due, x.id))
    if not jobs:
        return [], True

    remaining = {j.id: float(j.processing) for j in jobs}
    completion: dict[str, float] = {}
    ready: list[tuple[float, str, Job]] = []
    segments: list[Segment] = []

    i = 0
    t = min(j.release for j in jobs)

    while i < len(jobs) or ready:
        if not ready:
            t = max(t, jobs[i].release)
            while i < len(jobs) and jobs[i].release <= t + EPS:
                j = jobs[i]
                heapq.heappush(ready, (j.due, j.id, j))
                i += 1

        _, _, j = heapq.heappop(ready)
        next_release = jobs[i].release if i < len(jobs) else inf

        run = remaining[j.id]
        if next_release < inf:
            run = min(run, max(0.0, next_release - t))

        # If a release occurs exactly now, first insert it into the ready queue.
        if run <= EPS and next_release < inf:
            while i < len(jobs) and jobs[i].release <= t + EPS:
                x = jobs[i]
                heapq.heappush(ready, (x.due, x.id, x))
                i += 1
            heapq.heappush(ready, (j.due, j.id, j))
            continue

        start = t
        end = t + run
        if segments and segments[-1].job_id == j.id and abs(segments[-1].end - start) <= EPS:
            prev = segments[-1]
            segments[-1] = Segment(prev.start, end, j.id)
        else:
            segments.append(Segment(start, end, j.id))

        t = end
        remaining[j.id] -= run

        while i < len(jobs) and jobs[i].release <= t + EPS:
            x = jobs[i]
            heapq.heappush(ready, (x.due, x.id, x))
            i += 1

        if remaining[j.id] <= EPS:
            completion[j.id] = t
        else:
            heapq.heappush(ready, (j.due, j.id, j))

    all_on_time = all(completion[j.id] <= j.due + EPS for j in jobs)
    return segments, all_on_time


def lawler_optimal_on_time_weight(jobs: Iterable[Job]) -> int:
    """
    Lawler's dynamic program for

        1 | pmtn, r_j | sum w_j U_j

    Returns the maximum total weight of a subset that can be completed on time.

    The jobs are internally sorted by nondecreasing due date.

    Complexity:
        O(n k^2 W^2) time in the paper's analysis,
        where k = number of distinct release dates and W = sum of integer weights.

    Notes on two boundary details in the scanned paper:
      1) In the third branch of (4.2), this implementation allows
         w' = w - w_j. This is needed when the set before the last block is empty.
      2) P counts processing inside [r_j, r']; therefore, if its first block
         starts at release r, its contribution is
             max(0, C(r,w') - max(r, r_j)).
         This follows the definition of P as interval overlap.
    """
    jobs = list(jobs)
    if not jobs:
        return 0

    jobs = sorted(jobs, key=lambda x: (x.due, x.release, x.id))

    W = sum(int(j.weight) for j in jobs)
    releases = sorted(set(j.release for j in jobs))
    k = len(releases)
    release_index = {r: i for i, r in enumerate(releases)}

    # C_prev[a][w] = C_{j-1}(releases[a], w)
    C_prev = [[inf] * (W + 1) for _ in range(k)]
    for a, r in enumerate(releases):
        C_prev[a][0] = r

    for j_idx, job in enumerate(jobs, start=1):
        rj = job.release
        pj = job.processing
        dj = job.due
        wj = int(job.weight)

        previous_release_dates = set(x.release for x in jobs[:j_idx - 1])

        @lru_cache(maxsize=None)
        def P(a: int, b: int, target_weight: int) -> float:
            """
            P_{j-1}(r, r', target_weight).

            releases[a] is the lower bound on the minimum release date of
            selected old jobs; releases[b] = r'.

            The value is the minimum amount of OLD-job processing that must
            occur inside [r_j, r'].
            """
            if target_weight <= 0:
                return 0.0

            if a >= k or b >= k or a >= b:
                return inf

            r = releases[a]
            r_prime = releases[b]

            # Case: the selected set actually starts after r.
            best = P(a + 1, b, target_weight) if a + 1 < k else inf

            # Case: the first block starts at r.
            for block_weight in range(1, target_weight + 1):
                c = C_prev[a][block_weight]
                if c == inf or c > r_prime + EPS:
                    continue

                # r'' = smallest release date >= c.
                next_idx = bisect.bisect_left(releases, c - EPS)

                remaining_weight = target_weight - block_weight
                if remaining_weight <= 0:
                    tail = 0.0
                elif next_idx < k:
                    tail = P(next_idx, b, remaining_weight)
                else:
                    tail = inf

                if tail == inf:
                    continue

                overlap_of_first_block = max(0.0, c - max(r, rj))
                candidate = overlap_of_first_block + tail
                if candidate < best:
                    best = candidate

            return best

        C_new = [row[:] for row in C_prev]

        for a, r in enumerate(releases):
            if r > rj + EPS:
                # Job j cannot belong to any set with r(S) >= r.
                continue

            for target_weight in range(W + 1):
                # Case 1: skip job j.
                best = C_prev[a][target_weight]

                # Case 2: job j starts after all previously selected jobs.
                needed_before_j = max(0, target_weight - wj)
                c_before = C_prev[a][needed_before_j]

                if c_before < inf:
                    candidate = max(rj, c_before) + pj
                    if candidate < best:
                        best = candidate

                # Case 3: some of job j is processed before the last old block.
                if target_weight > wj:
                    for r_prime in previous_release_dates:
                        if r_prime <= rj + EPS:
                            continue

                        b = release_index[r_prime]

                        # w' is the target weight of the last old block.
                        # Equality is included so that the prefix before that
                        # block may have weight 0.
                        max_last_block_weight = target_weight - wj
                        for last_block_weight in range(max_last_block_weight + 1):
                            c_last_block = C_prev[b][last_block_weight]
                            if c_last_block == inf:
                                continue

                            prefix_weight = (
                                target_weight - wj - last_block_weight
                            )
                            occupied = P(a, b, prefix_weight)
                            if occupied == inf:
                                continue

                            remaining_j_after_r_prime = max(
                                0.0,
                                pj - r_prime + rj + occupied,
                            )

                            candidate = c_last_block + remaining_j_after_r_prime
                            if candidate < best:
                                best = candidate

                # Since j has the largest due date among jobs 1..j,
                # any state finishing after d_j is infeasible.
                C_new[a][target_weight] = (
                    best if best <= dj + EPS else inf
                )

        C_prev = C_new

    r_min_idx = release_index[min(releases)]
    return max(
        w for w in range(W + 1)
        if C_prev[r_min_idx][w] < inf
    )


def lawler_solve(
    jobs: Iterable[Job],
    reconstruct: bool = True,
) -> dict:
    """
    Solve the weighted late-job problem.

    Returns:
        optimal_on_time_weight
        optimal_late_weight
        on_time_jobs
        late_jobs
        schedule

    Reconstruction:
        The paper states that a pointer-based reconstruction can retain the
        asymptotic space bound, but omits the details. For clarity, this
        reference implementation reconstructs a maximum-weight feasible subset
        by repeated exact DP calls. This is slower than the objective-only DP,
        but easy to verify and suitable for small/medium examples.
    """
    jobs = list(jobs)
    total_weight = sum(j.weight for j in jobs)
    optimum = lawler_optimal_on_time_weight(jobs)

    result = {
        "optimal_on_time_weight": optimum,
        "optimal_late_weight": total_weight - optimum,
        "on_time_jobs": None,
        "late_jobs": None,
        "schedule": None,
    }

    if not reconstruct:
        return result

    # Find one optimal subset by deleting jobs whenever the optimum remains.
    active = list(jobs)
    i = 0
    while i < len(active):
        candidate = active[:i] + active[i + 1:]
        if lawler_optimal_on_time_weight(candidate) == optimum:
            active = candidate
        else:
            i += 1

    on_time_ids = {j.id for j in active}
    late = [j for j in jobs if j.id not in on_time_ids]

    segments, feasible = preemptive_edd_schedule(active)
    if not feasible:
        raise RuntimeError("Internal error: reconstructed subset is not EDD-feasible.")

    result["on_time_jobs"] = active
    result["late_jobs"] = late
    result["schedule"] = segments
    return result


def print_solution(title: str, jobs: Iterable[Job], result: dict) -> None:
    jobs = list(jobs)
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Jobs:")
    for j in jobs:
        print(
            f"  {j.id}: r={j.release}, p={j.processing}, "
            f"d={j.due}, w={j.weight}"
        )

    print(f"\nMaximum on-time weight: {result['optimal_on_time_weight']}")
    print(f"Minimum late weight:    {result['optimal_late_weight']}")

    if result["on_time_jobs"] is not None:
        print(
            "On-time subset:         "
            + ", ".join(j.id for j in result["on_time_jobs"])
        )
        print(
            "Late jobs:              "
            + (", ".join(j.id for j in result["late_jobs"]) or "(none)")
        )
        print("Preemptive EDD schedule:")
        for s in result["schedule"]:
            print(f"  [{s.start:g}, {s.end:g})  {s.job_id}")
    print()


def example_1_paper() -> None:
    jobs = [
        Job("J1", 3, 2, 7, 2),
        Job("J2", 2, 2, 8, 4),
        Job("J3", 2, 3, 9, 1),
        Job("J4", 4, 2, 9, 1),
        Job("J5", 4, 2, 11, 6),
        Job("J6", 0, 1, 12, 1),
    ]
    print_solution(
        "Example 1: six-job instance from Lawler's paper",
        jobs,
        lawler_solve(jobs),
    )


def example_2_preemption_is_necessary() -> None:
    jobs = [
        Job("A", 2, 2, 5, 1),
        Job("B", 1, 3, 6, 3),
    ]
    print_solution(
        "Example 2: preemption is needed to keep both jobs on time",
        jobs,
        lawler_solve(jobs),
    )


def example_3_one_job_must_be_late() -> None:
    jobs = [
        Job("A", 0, 3, 4, 4),
        Job("B", 1, 3, 5, 2),
        Job("C", 2, 2, 6, 5),
    ]
    print_solution(
        "Example 3: choose the maximum-weight feasible subset",
        jobs,
        lawler_solve(jobs),
    )


if __name__ == "__main__":
    example_1_paper()
    example_2_preemption_is_necessary()
    example_3_one_job_must_be_late()
