#!/usr/bin/env python3
"""Plot Lawler experiment heatmaps from results.csv.

Usage:
  python3 plot_lawler_heatmaps.py --input lawler_combined_results/results.csv \
      --output lawler_combined_results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    required = ['num_original_jobs', 'kappa_max', 'status', 'lawler_runtime_seconds']
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in {path}: {missing}')
    for col in ['num_original_jobs', 'kappa_max', 'lawler_runtime_seconds',
                'interval_ratio', 'interval_density', 'cloud_job_ratio',
                'num_cloud_jobs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['status'] = df['status'].fillna('').str.lower().str.strip()
    df['_success'] = df['status'].eq('ok') & df['lawler_runtime_seconds'].gt(0)
    df['_timeout'] = df['status'].eq('timeout')
    if 'interval_ratio' in df.columns:
        df['_density'] = df['interval_ratio']
    elif 'interval_density' in df.columns:
        df['_density'] = df['interval_density']
    elif 'density' in df.columns:
        lookup = {'sparse': .10, 'light': .25, 'medium': .50,
                  'dense': 1.00, 'very_dense': 2.00}
        df['_density'] = df['density'].astype(str).str.lower().map(lookup)
    else:
        df['_density'] = np.nan
    if 'cloud_job_ratio' in df.columns:
        df['_cloud_pct'] = df['cloud_job_ratio'] * 100
    elif 'num_cloud_jobs' in df.columns:
        df['_cloud_pct'] = 100 * df['num_cloud_jobs'] / df['num_original_jobs']
    else:
        df['_cloud_pct'] = np.nan
    return df


def draw_heatmap(df, xcol, ycol, metric, title, colorbar, outfile, log=False,
                 percentage=False):
    valid = df.dropna(subset=[xcol, ycol]).copy()
    if valid.empty:
        print(f'SKIP {outfile.name}: no data for axes')
        return
    xs = sorted(valid[xcol].unique())
    ys = sorted(valid[ycol].unique())
    values = np.full((len(ys), len(xs)), np.nan)
    timeouts = np.zeros((len(ys), len(xs)), dtype=int)
    counts = np.zeros((len(ys), len(xs)), dtype=int)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            subset = valid[(valid[xcol] == x) & (valid[ycol] == y)]
            timeouts[i, j] = int(subset['_timeout'].sum())
            good = subset[subset['_success']]
            if percentage:
                good = good[good[metric].notna()]
            counts[i, j] = len(good)
            if len(good):
                values[i, j] = (good[metric].mean() if percentage
                                else good[metric].median())
    finite = values[np.isfinite(values)]
    if not len(finite):
        print(f'SKIP {outfile.name}: no successful observations')
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.85 * len(xs) + 2),
                                    max(3.8, 0.65 * len(ys) + 1.5)))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    if log:
        positive = finite[finite > 0]
        if not len(positive):
            plt.close(fig)
            return
        lo, hi = float(positive.min()), float(positive.max())
        norm = LogNorm(vmin=max(lo, 1e-9), vmax=max(hi, lo * 1.001))
    else:
        norm = Normalize(vmin=0, vmax=max(100 if percentage else 0, float(finite.max()), 1))
    image = ax.imshow(np.ma.masked_invalid(values), origin='lower', aspect='auto',
                      cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xs)), [f'{x:g}' for x in xs])
    ax.set_yticks(range(len(ys)), [f'{y:g}' for y in ys])
    ax.set_xlabel('Number of original jobs ($n$)')
    ax.set_ylabel(r'Maximum outsourcing penalty ($\kappa_{\max}$)' if ycol == 'kappa_max'
                  else 'Interval density')
    ax.set_title(title)
    ax.set_xticks(np.arange(-.5, len(xs), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ys), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.3)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i in range(len(ys)):
        for j in range(len(xs)):
            val = values[i, j]
            if np.isfinite(val):
                label = f'{val:.0f}%' if percentage else f'{val:.2g}'
            else:
                label = 'TO' if timeouts[i, j] else '—'
            if timeouts[i, j]:
                label += f'\n({timeouts[i,j]} TO)'
            text_color = 'white' if np.isfinite(val) and norm(val) > .55 else 'black'
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color=text_color)
    cb = fig.colorbar(image, ax=ax, fraction=.045, pad=.025)
    cb.set_label(colorbar)
    fig.text(.5, .015, 'TO = timed out; gray = no successful run. '
             'Runtime values use successful runs only.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, 1])
    for ext in ['pdf', 'png']:
        fig.savefig(outfile.with_suffix('.' + ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}.pdf and .png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='lawler_combined_results/results.csv')
    parser.add_argument('--output', default='lawler_combined_results/figures')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input)
    draw_heatmap(df, 'num_original_jobs', 'kappa_max', 'lawler_runtime_seconds',
                 'Lawler runtime by jobs and outsourcing penalty',
                 'Median runtime (seconds, log scale)',
                 out / 'lawler_runtime_heatmap_jobs_kappa', log=True)
    if df['_density'].notna().any():
        draw_heatmap(df, 'num_original_jobs', '_density', 'lawler_runtime_seconds',
                     'Lawler runtime by jobs and interval density',
                     'Median runtime (seconds, log scale)',
                     out / 'lawler_runtime_heatmap_jobs_density', log=True)
    if df['_cloud_pct'].notna().any():
        draw_heatmap(df, 'num_original_jobs', 'kappa_max', '_cloud_pct',
                     'Cloud outsourcing by jobs and penalty',
                     'Mean original jobs outsourced (%)',
                     out / 'lawler_cloud_heatmap_jobs_kappa', percentage=True)
    else:
        print('Cloud heatmap skipped: cloud_job_ratio and num_cloud_jobs unavailable')
    print('Total rows:', len(df), '| successful:', int(df['_success'].sum()),
          '| timeouts:', int(df['_timeout'].sum()))


if __name__ == '__main__':
    main()
