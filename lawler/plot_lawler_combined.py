#!/usr/bin/env python3
"""Plot Lawler factorial experiment results from results.csv."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, directory, stem):
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(directory / f'{stem}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {stem}.pdf and {stem}.png')


def median_iqr(ax, data, xcol, ycol, label=None, logy=False):
    grouped = data.groupby(xcol)[ycol]
    stats = grouped.agg(median='median', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).sort_index()
    if stats.empty:
        return
    x = stats.index.to_numpy(dtype=float)
    mid = stats['median'].to_numpy(dtype=float)
    low = stats['q1'].to_numpy(dtype=float)
    high = stats['q3'].to_numpy(dtype=float)
    if logy:
        low = np.maximum(low, np.finfo(float).tiny)
    ax.plot(x, mid, marker='o', markersize=4, linewidth=1.8, label=label)
    ax.fill_between(x, low, high, alpha=.18)
    if logy:
        ax.set_yscale('log')
    ax.grid(True, alpha=.25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='lawler_combined_results/results.csv')
    p.add_argument('--output', default='lawler_combined_results/figures')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if 'status' not in df:
        raise ValueError('Missing status column')
    for col in ('num_original_jobs', 'kappa_max', 'total_weight_W', 'num_cloud_jobs',
                'cloud_job_ratio', 'lawler_runtime_seconds', 'total_solver_seconds', 'interval_ratio'):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    ok = df[df['status'].eq('ok')].copy()
    if ok.empty:
        raise ValueError('No successful runs to plot')
    runtime = 'total_solver_seconds' if 'total_solver_seconds' in ok and ok['total_solver_seconds'].notna().any() else 'lawler_runtime_seconds'
    ok = ok.dropna(subset=['num_original_jobs', runtime])
    ok = ok[ok[runtime] > 0]
    if ok.empty:
        raise ValueError('No successful runs with positive runtimes')
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    median_iqr(ax, ok, 'num_original_jobs', runtime, logy=True)
    ax.set(xlabel='Number of original jobs', ylabel='Solver runtime (seconds)', title='Lawler runtime scaling')
    save(fig, output, 'lawler_runtime_vs_jobs')

    if 'total_weight_W' in ok:
        subset = ok.dropna(subset=['total_weight_W'])
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.scatter(subset['total_weight_W'], subset[runtime], s=17, alpha=.45)
            ax.set_yscale('log')
            ax.set(xlabel='Total penalty weight W', ylabel='Solver runtime (seconds)', title='Runtime vs. penalty weight')
            ax.grid(True, alpha=.25)
            save(fig, output, 'lawler_runtime_vs_weight')

    cloud = ok.dropna(subset=['cloud_job_ratio']) if 'cloud_job_ratio' in ok else pd.DataFrame()
    if not cloud.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        median_iqr(ax, cloud.assign(cloud_percent=100 * cloud['cloud_job_ratio']),
                   'num_original_jobs', 'cloud_percent')
        ax.set(xlabel='Number of original jobs', ylabel='Cloud outsourcing ratio (%)',
               title='Cloud outsourcing vs. problem size', ylim=(0, 100))
        save(fig, output, 'lawler_cloud_ratio_vs_jobs')

    if 'num_cloud_jobs' in ok:
        counts = ok.dropna(subset=['num_cloud_jobs'])
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            median_iqr(ax, counts, 'num_original_jobs', 'num_cloud_jobs')
            ax.set(xlabel='Number of original jobs', ylabel='Number of cloud jobs',
                   title='Cloud job count vs. problem size')
            save(fig, output, 'lawler_cloud_jobs_vs_jobs')

    if 'kappa_max' in ok and 'cloud_job_ratio' in ok:
        subset = ok.dropna(subset=['kappa_max', 'cloud_job_ratio'])
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            median_iqr(ax, subset.assign(cloud_percent=100 * subset['cloud_job_ratio']),
                       'kappa_max', 'cloud_percent')
            ax.set(xlabel='Maximum outsourcing penalty κ', ylabel='Cloud outsourcing ratio (%)',
                   title='Outsourcing vs. penalty range', ylim=(0, 100))
            save(fig, output, 'lawler_cloud_ratio_vs_kappa')

    if 'interval_ratio' in ok:
        subset = ok.dropna(subset=['interval_ratio'])
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            for k, group in subset.groupby('kappa_max'):
                median_iqr(ax, group, 'interval_ratio', runtime, label=f'κ max = {int(k)}', logy=True)
            ax.set(xlabel='Interval density ratio', ylabel='Solver runtime (seconds)',
                   title='Runtime vs. interval density')
            ax.legend(fontsize=8)
            save(fig, output, 'lawler_runtime_vs_density')

    print(f'Rows: {len(df)} | successful: {(df.status == "ok").sum()} | '
          f'timeout: {(df.status == "timeout").sum()} | other: {(~df.status.isin(["ok", "timeout"])).sum()}')
    print(f'Figures in: {output.resolve()}')


if __name__ == '__main__':
    main()
