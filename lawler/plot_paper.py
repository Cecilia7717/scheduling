#!/usr/bin/env python3
"""Three paper-ready Lawler plots, grouped by carbon-interval density.

python3 plot_lawler_paper.py --input lawler_combined_results/results.csv \
    --output lawler_combined_results/figures
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DENSITIES = [(0.10, 'Sparse'), (0.25, 'Light'), (0.50, 'Medium'), (1.00, 'Dense')]


def load(path, runtime_preference):
    df = pd.read_csv(path)
    required = ['status', 'num_original_jobs', 'total_weight_W', 'kappa_max']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    for col in ['num_original_jobs', 'total_weight_W', 'kappa_max', 'interval_ratio',
                'interval_density', 'lawler_runtime_seconds', 'total_solver_seconds',
                'cloud_job_ratio', 'num_cloud_jobs']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'interval_ratio' in df:
        df['_density'] = df['interval_ratio']
    elif 'interval_density' in df:
        df['_density'] = df['interval_density']
    elif 'density' in df:
        df['_density'] = df['density'].astype(str).str.lower().map(
            {'sparse': .1, 'light': .25, 'medium': .5, 'dense': 1.0})
    else:
        raise ValueError('Need interval_ratio, interval_density, or density column')
    df['_density'] = df['_density'].apply(
        lambda x: min(DENSITIES, key=lambda d: abs(d[0] - x))[1] if pd.notna(x) and
        min(abs(d[0] - x) for d in DENSITIES) < .025 else np.nan)
    df['status'] = df['status'].fillna('').astype(str).str.strip().str.lower()
    if runtime_preference == 'total':
        runtime = 'total_solver_seconds'
    elif runtime_preference == 'dp':
        runtime = 'lawler_runtime_seconds'
    else:
        runtime = 'total_solver_seconds' if 'total_solver_seconds' in df else 'lawler_runtime_seconds'
    if runtime not in df:
        raise ValueError(f'Missing runtime column {runtime}')
    df['_runtime'] = df[runtime]
    if 'cloud_job_ratio' in df:
        df['_cloud_pct'] = df['cloud_job_ratio'] * 100
    elif 'num_cloud_jobs' in df:
        df['_cloud_pct'] = 100 * df['num_cloud_jobs'] / df['num_original_jobs']
    else:
        df['_cloud_pct'] = np.nan
    ok = df[(df.status == 'ok') & df._runtime.gt(0) & df._density.notna()].copy()
    return df, ok, runtime


def plot_grouped(ax, data, xcol, ycol, *, logy=False):
    drawn = 0
    for _, label in DENSITIES:
        sub = data[data['_density'] == label].dropna(subset=[xcol, ycol])
        if logy:
            sub = sub[sub[ycol] > 0]
        if sub.empty:
            continue
        stats = sub.groupby(xcol)[ycol].agg(
            median='median', q1=lambda s: s.quantile(.25),
            q3=lambda s: s.quantile(.75)).sort_index()
        x = stats.index.to_numpy(dtype=float)
        y = stats['median'].to_numpy(dtype=float)
        lo = stats['q1'].to_numpy(dtype=float)
        hi = stats['q3'].to_numpy(dtype=float)
        line, = ax.plot(x, y, '-o', lw=1.8, ms=4, label=label)
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.14)
        drawn += 1
    if logy:
        ax.set_yscale('log')
    ax.grid(alpha=.23, which='major')
    if drawn:
        ax.legend(frameon=False, fontsize=9, title='Interval density', title_fontsize=9)
    return drawn


def save(fig, output, stem):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {stem}.pdf and {stem}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='lawler_combined_results/results.csv')
    parser.add_argument('--output', default='lawler_combined_results/figures')
    parser.add_argument('--runtime', choices=['auto', 'dp', 'total'], default='dp',
                        help='dp = Lawler DP time; total = DP plus reconstruction (default: dp)')
    parser.add_argument('--weight-bins', type=int, default=10,
                        help='Number of global quantile bins for W (default: 10)')
    args = parser.parse_args()
    if args.weight_bins < 2:
        parser.error('--weight-bins must be at least 2')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df, ok, runtime = load(args.input, args.runtime)
    if ok.empty:
        raise ValueError('No successful observations with positive runtime and recognized density')
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})

    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    plot_grouped(ax, ok, 'num_original_jobs', '_runtime', logy=True)
    ax.set(xlabel='Number of original jobs ($n$)', ylabel='Lawler runtime (s)')
    save(fig, out, 'lawler_runtime_vs_jobs_density')

    weighted = ok.dropna(subset=['total_weight_W']).copy()
    weighted = weighted[weighted.total_weight_W > 0]
    if not weighted.empty:
        # Common quantile bins across ALL densities: curves are comparable on the same W axis.
        weighted['_w_bin'] = pd.qcut(weighted['total_weight_W'], q=args.weight_bins,
                                     labels=False, duplicates='drop')
        # Bin x-position is the global median W; avoids a different x-axis for each density.
        centers = weighted.groupby('_w_bin')['total_weight_W'].median()
        weighted['_w_center'] = weighted['_w_bin'].map(centers)
        fig, ax = plt.subplots(figsize=(5.7, 3.7))
        plot_grouped(ax, weighted, '_w_center', '_runtime', logy=True)
        ax.set(xlabel='Total reduced-instance weight ($W$)', ylabel='Lawler runtime (s)')
        save(fig, out, 'lawler_runtime_vs_weight_density')

    cloud = ok.dropna(subset=['_cloud_pct', 'kappa_max']).copy()
    if not cloud.empty:
        fig, ax = plt.subplots(figsize=(5.7, 3.7))
        plot_grouped(ax, cloud, 'kappa_max', '_cloud_pct')
        ax.set(xlabel=r'Maximum outsourcing penalty ($\kappa_{\max}$)',
               ylabel='Original jobs outsourced (%)', ylim=(0, 100))
        ax.set_xticks(sorted(cloud.kappa_max.unique()))
        save(fig, out, 'lawler_cloud_ratio_vs_kappa_density')
    else:
        print('Cloud figure skipped: no successful cloud allocation data')

    print(f'Rows: {len(df)}; successful: {(df.status == "ok").sum()}; '
          f'timeouts: {(df.status == "timeout").sum()}; '
          f'other: {(~df.status.isin(["ok", "timeout"])).sum()}')
    print(f'Runtime column: {runtime}; plotted successful rows: {len(ok)}')
    print('Caution: runtime plots exclude timed-out runs; report timeout counts separately.')


if __name__ == '__main__':
    main()
