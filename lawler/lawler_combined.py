from __future__ import annotations



import argparse

import csv

import importlib.util

import os

import random

import signal

import statistics

import sys

import time

from pathlib import Path



HERE = Path(__file__).resolve().parent

JOB_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40]

DENSITIES = [(0.10, 'sparse'), (0.25, 'light'), (0.50, 'medium'), (1.00, 'dense')]

KAPPA_MAX_VALUES = [5, 10, 20, 40, 80]

INSTANCES_PER_CONFIGURATION = 5

SEED = 42

ENERGY_COST = {'green': 0, 'brown': 1, 'red': 2}





def load_module(filename, module_name):

    path = HERE / filename

    if not path.exists():

        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module





generator = load_module('compare_algorithms.py', 'combined_generator')

lawler = load_module('dp.py', 'combined_lawler')





def binary_dummy_lengths(length):

    if length <= 0:

        raise ValueError('Nonpositive interval length')

    result, covered, power = [], 0, 1

    while covered + power <= length:

        result.append(power)

        covered += power

        power *= 2

    if covered < length:

        result.append(length - covered)

    return result





def make_reduction(jobs, intervals, penalties):

    reduced, originals, dummies = [], set(), {}

    for job in jobs:

        jid = str(job.name)

        if jid in originals:

            raise ValueError(f'Duplicate original job ID: {jid}')

        originals.add(jid)

        reduced.append(lawler.Job(

            id=jid, release=float(job.release),

            processing=float(job.processing), due=float(job.deadline),

            weight=int(penalties[jid])))

    for i, interval in enumerate(intervals):

        start, end = int(round(interval.start)), int(round(interval.end))

        cost = ENERGY_COST[interval.energy]

        for j, piece in enumerate(binary_dummy_lengths(end - start)):

            weight = cost * piece

            if not weight:

                continue

            jid = f'G{i}_{j}'

            if jid in originals or jid in dummies:

                raise ValueError(f'Duplicate dummy job ID: {jid}')

            dummies[jid] = weight

            reduced.append(lawler.Job(

                id=jid, release=float(start), processing=float(piece),

                due=float(end), weight=int(weight)))

    return reduced, originals, dummies





class SolverTimeout(Exception):

    pass





def timeout_handler(signum, frame):

    raise SolverTimeout('Solver timed out')





def timed_call(fn, timeout):

    previous = signal.signal(signal.SIGALRM, timeout_handler)

    start = time.perf_counter()

    try:

        if timeout > 0:

            signal.setitimer(signal.ITIMER_REAL, timeout)

        return fn(), time.perf_counter() - start

    finally:

        signal.setitimer(signal.ITIMER_REAL, 0)

        signal.signal(signal.SIGALRM, previous)





def solve_once(reduced, W, originals, dummies, penalties, args):

    """One solve per instance, with a single 1000-second total budget."""

    start = time.perf_counter()

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:

        signal.setitimer(signal.ITIMER_REAL, args.timeout)

        dp_start = time.perf_counter()

        best = lawler.lawler_optimal_on_time_weight(reduced)

        dp_seconds = time.perf_counter() - dp_start

        reconstruction_start = time.perf_counter()

        solution = lawler.lawler_solve(reduced, reconstruct=True)

        reconstruction_seconds = time.perf_counter() - reconstruction_start

        optimum = W - best

        if int(solution['optimal_late_weight']) != int(optimum):

            raise ValueError('Reconstructed objective does not match DP objective')

        late = {str(job.id) for job in solution['late_jobs']}

        on_time = {str(job.id) for job in solution['on_time_jobs']}

        cloud = sorted(originals & late)

        edge = sorted(originals & on_time)

        if set(cloud) | set(edge) != originals or set(cloud) & set(edge):

            raise ValueError('Original job partition is invalid')

        cloud_cost = sum(penalties[j] for j in cloud)

        dummy_cost = sum(w for j, w in dummies.items() if j in late)

        if cloud_cost + dummy_cost != optimum:

            raise ValueError('Reconstructed costs do not match optimum')

        return {

            'maximum_on_time_weight': best,

            'minimum_rejected_penalty': optimum,

            'num_cloud_jobs': len(cloud),

            'num_edge_jobs': len(edge),

            'cloud_job_ratio': len(cloud) / len(originals),

            'cloud_outsourcing_cost': cloud_cost,

            'num_rejected_dummy_jobs': len(set(dummies) & late),

            'rejected_dummy_cost': dummy_cost,

            'cloud_job_ids': ';'.join(cloud),

            'edge_job_ids': ';'.join(edge),

            'lawler_runtime_seconds': dp_seconds,

            'lawler_runtime_ms': 1000 * dp_seconds,

            'reconstruction_seconds': reconstruction_seconds,

            'total_solver_seconds': time.perf_counter() - start,

            'retry_count': 0,

            'effective_timeout': args.timeout,

            'status': 'ok', 'error': '',

        }

    except SolverTimeout:

        elapsed = time.perf_counter() - start

        print(f'    TIMEOUT after {elapsed:.1f}s (limit {args.timeout:g}s); moving on', flush=True)

        return {

            'status': 'timeout', 'retry_count': 0,

            'effective_timeout': args.timeout,

            'total_solver_seconds': elapsed,

            'error': f'Exceeded {args.timeout:g}s total solver limit',

        }

    finally:

        signal.setitimer(signal.ITIMER_REAL, 0)

        signal.signal(signal.SIGALRM, previous_handler)





FIELDS = [

    'num_original_jobs', 'interval_ratio', 'density', 'kappa_max',

    'instance_id', 'seed', 'status', 'error', 'horizon',

    'num_energy_intervals', 'num_dummy_jobs', 'num_penalty_jobs',

    'distinct_release_dates_k', 'mean_kappa', 'original_weight',

    'dummy_weight', 'total_weight_W', 'maximum_on_time_weight',

    'minimum_rejected_penalty', 'num_cloud_jobs', 'num_edge_jobs',

    'cloud_job_ratio', 'cloud_outsourcing_cost',

    'num_rejected_dummy_jobs', 'rejected_dummy_cost',

    'cloud_job_ids', 'edge_job_ids', 'lawler_runtime_seconds',

    'lawler_runtime_ms', 'reconstruction_seconds',

    'total_solver_seconds', 'retry_count', 'effective_timeout',

    'N_times_W', 'k_times_W', 'N_times_k_times_W',

]





def key(row):

    return (int(row['num_original_jobs']), round(float(row['interval_ratio']), 8),

            int(row['kappa_max']), int(row['instance_id']))





def atomic_csv(path, fields, rows):

    tmp = path.with_name(path.name + '.tmp')

    with tmp.open('w', newline='', encoding='utf-8') as f:

        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')

        writer.writeheader()

        writer.writerows(rows)

        f.flush()

        os.fsync(f.fileno())

    os.replace(tmp, path)





def load_existing(path):

    if not path.exists():

        return {}

    with path.open(newline='', encoding='utf-8') as f:

        reader = csv.DictReader(f)

        required = {'num_original_jobs', 'interval_ratio', 'kappa_max', 'instance_id'}

        if not required.issubset(set(reader.fieldnames or [])):

            raise ValueError(

                f'{path} is not a full-factorial results CSV. '

                'Use a different --output directory; the existing file is unchanged.')

        saved = {}

        duplicates = 0

        for row in reader:

            if not row.get('instance_id'):

                continue

            k = key(row)

            if k in saved:

                duplicates += 1

                # Prefer a completed result over a failed one.

                if saved[k].get('status', 'ok') == 'ok':

                    continue

            saved[k] = row

        if duplicates:

            print(f'Found {duplicates} repeated configuration keys; '

                  'kept one row per configuration (preferring success).')

        return saved





def save_all(output_dir, saved):

    rows = [saved[k] for k in sorted(saved)]

    atomic_csv(output_dir / 'results.csv', FIELDS, rows)

    groups = {}

    for row in rows:

        group = (int(row['num_original_jobs']), float(row['interval_ratio']),

                 int(row['kappa_max']))

        groups.setdefault(group, []).append(row)

    summary = []

    for (n, density, K), group in sorted(groups.items()):

        ok = [r for r in group if r.get('status', 'ok') == 'ok'

              and r.get('num_cloud_jobs', '') not in ('', None)]

        def mean(field):

            return statistics.mean(float(r[field]) for r in ok) if ok else ''

        def median(field):

            return statistics.median(float(r[field]) for r in ok) if ok else ''

        summary.append({

            'num_original_jobs': n, 'interval_ratio': density, 'kappa_max': K,

            'num_attempted': len(group), 'num_completed': len(ok),

            'num_timeouts': sum(r.get('status') == 'timeout' for r in group),

            'median_runtime_seconds': median('lawler_runtime_seconds'),

            'mean_runtime_seconds': mean('lawler_runtime_seconds'),

            'median_total_weight_W': median('total_weight_W'),

            'median_num_penalty_jobs': median('num_penalty_jobs'),

            'mean_cloud_jobs': mean('num_cloud_jobs'),

            'mean_cloud_ratio': mean('cloud_job_ratio'),

            'mean_cloud_outsourcing_cost': mean('cloud_outsourcing_cost'),

        })

    summary_fields = [

        'num_original_jobs', 'interval_ratio', 'kappa_max',

        'num_attempted', 'num_completed', 'num_timeouts',

        'median_runtime_seconds', 'mean_runtime_seconds',

        'median_total_weight_W', 'median_num_penalty_jobs',

        'mean_cloud_jobs', 'mean_cloud_ratio',

        'mean_cloud_outsourcing_cost',

    ]

    atomic_csv(output_dir / 'summary.csv', summary_fields, summary)





def run(args):
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = HERE / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = load_existing(output_dir / 'results.csv')

    target_keys = {
        (n, round(ratio, 8), K, i)
        for n in JOB_COUNTS
        for ratio, _ in DENSITIES
        for K in KAPPA_MAX_VALUES
        for i in range(1, INSTANCES_PER_CONFIGURATION + 1)
    }
    unrelated = set(saved) - target_keys
    if unrelated:
        raise ValueError(
            f'{len(unrelated)} existing rows are outside the requested '
            'factorial grid. Existing CSV left unchanged. '
            'Use a separate --output folder for this experiment.'
        )

    def is_complete(row):
        return (row.get('status', 'ok') == 'ok'
                and row.get('num_cloud_jobs', '') not in ('', None))

    # Freeze the phase-2 queue at startup: only PREVIOUS timeouts are retried.
    # Newly timed-out uncovered cases are not immediately attempted again.
    uncovered = sorted(target_keys - set(saved))
    incomplete = sorted(k for k in target_keys & set(saved)
                        if not is_complete(saved[k])
                        and saved[k].get('status') != 'timeout')
    previous_timeouts = sorted(k for k in target_keys & set(saved)
                               if saved[k].get('status') == 'timeout')
    complete = sum(is_complete(saved[k]) for k in target_keys & set(saved))
    print(f'Full factorial: {len(target_keys)} runs; {complete} complete; '
          f'{len(uncovered)} uncovered; {len(incomplete)} incomplete/error; '
          f'{len(previous_timeouts)} previous timeouts', flush=True)

    def process(keys, phase):
        # Group by shared generated instance to avoid regenerating for each K.
        groups = {}
        for n, ratio, K, i in keys:
            groups.setdefault((n, ratio, i), []).append(K)

        for (n, ratio, i), kappas in sorted(groups.items()):
            label = next(name for value, name in DENSITIES
                         if round(value, 8) == ratio)
            seed = SEED + n * 1_000_000 + round(ratio * 100) * 10_000 + i
            horizon = 6 * n
            num_intervals = min(horizon, max(3, round(ratio * n)))
            instance = generator.generate_feasible_instance(
                random.Random(seed), instance_id=i,
                fixed_jobs=n, horizon_per_job=6,
                fixed_intervals=num_intervals,
                green_share_range=(0.50, 0.60),
                brown_share_range=(0.20, 0.25),
                target_utilization_range=(0.70, 0.80))
            jobs = instance['jobs']
            intervals = instance['energy_intervals']

            for K in sorted(kappas):
                k = (n, ratio, K, i)
                # Do not overwrite a successful result.
                if k in saved and is_complete(saved[k]):
                    continue
                print(f'[{phase}] n={n} density={label} K={K} '
                      f'instance={i}', flush=True)
                rng = random.Random(seed + 500_000_000)
                penalties = {str(j.name): 1 + int(rng.random() * K)
                             for j in jobs}
                reduced, originals, dummies = make_reduction(
                    jobs, intervals, penalties)
                N = len(reduced)
                W = sum(int(j.weight) for j in reduced)
                release_k = len({j.release for j in reduced})
                original_weight = sum(penalties.values())
                row = {field: '' for field in FIELDS}
                row.update({
                    'num_original_jobs': n, 'interval_ratio': ratio,
                    'density': label, 'kappa_max': K, 'instance_id': i,
                    'seed': seed, 'horizon': horizon,
                    'num_energy_intervals': len(intervals),
                    'num_dummy_jobs': len(dummies),
                    'num_penalty_jobs': N,
                    'distinct_release_dates_k': release_k,
                    'mean_kappa': statistics.mean(penalties.values()),
                    'original_weight': original_weight,
                    'dummy_weight': W - original_weight,
                    'total_weight_W': W,
                    'N_times_W': N * W, 'k_times_W': release_k * W,
                    'N_times_k_times_W': N * release_k * W,
                })
                try:
                    row.update(solve_once(
                        reduced, W, originals, dummies, penalties, args))
                except Exception as exc:
                    row.update(status='error',
                               error=f'{type(exc).__name__}: {exc}')
                    print(f'    ERROR: {exc}', flush=True)
                saved[k] = row
                save_all(output_dir, saved)
                print(f'    {row["status"]} cloud={row["num_cloud_jobs"]} '
                      f'DP={row["lawler_runtime_seconds"]}s', flush=True)

    try:
        print('PHASE 1: uncovered cases', flush=True)
        process(uncovered, 'uncovered')
        if incomplete:
            print('PHASE 1b: existing incomplete/error cases', flush=True)
            process(incomplete, 'incomplete')
        print('PHASE 2: previously timed-out cases', flush=True)
        process(previous_timeouts, 'retry-timeout')
    finally:
        save_all(output_dir, saved)

    complete = sum(is_complete(saved[k]) for k in target_keys & set(saved))
    print(f'FINISHED: {complete}/{len(target_keys)} completed')
    print(f'Results: {output_dir / "results.csv"}')
    print(f'Summary: {output_dir / "summary.csv"}')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', default='lawler_combined_results')

    parser.add_argument('--timeout', type=float, default=1000,

                        help='Maximum total solver seconds per instance (default: 1000)')

    args = parser.parse_args()

    if args.timeout <= 0:

        parser.error('--timeout must be positive')

    main_args = args

    run(main_args)





if __name__ == '__main__':

    main()
