# Lawler Experiments

This directory contains the implementation and experiments for evaluating Lawler's exact dynamic program and the reduction from the cloud-outsourcing problem to weighted penalty scheduling.

## Files

### Core implementation

- `dp.py` — implementation of Lawler's exact dynamic program for $1|pmtn,r_j|\sum w_j U_j$.
- `maxflow.py` — flow-based scheduling routines used during reduction validation.
- `compare_algorithms.py` — utilities for comparing scheduling algorithms.

## Experiment 1: Reduction Validation

`validate_lawler_reduction.py` validates the cloud-outsourcing reduction on small instances.

For each generated instance, the script constructs the corresponding weighted penalty-scheduling instance and solves it using Lawler's exact dynamic program. It independently computes the cloud optimum by enumerating all local/cloud assignments and solving each local scheduling problem, then checks whether

$$
\mathrm{OPT}_{\mathrm{pen}} = \mathrm{OPT}_{\mathrm{cld}}.
$$

Settings:

- Original jobs: `5, 8, 10`
- Instances per job count: `30`
- Total instances: `90`
- Random seed: `42`

Results:

```text
lawler_stage1_validation/
├── stage1_validation.csv
└── summary.txt
```

All 90 validation instances produced matching optimal objective values.

## Experiment 2: Lawler Scalability

Scripts:

- `benchmark_lawler_scaling.py`
- `plot_lawler_scaling.py`

This experiment measures the runtime of Lawler's exact dynamic program on instances produced by the cloud-outsourcing reduction. It varies the number of original jobs and carbon intervals.

Interval configurations are approximately:

```text
0.10 n   sparse
0.25 n   light
0.50 n   medium
1.00 n   dense
```

The benchmark records the number of original jobs, carbon intervals, dummy jobs, total penalty jobs $N$, distinct release dates $k$, total weight $W$, and Lawler runtime.

Run:

```bash
python benchmark_lawler_scaling.py
python plot_lawler_scaling.py
```

Results:

```text
lawler_scaling_results/
├── results.csv
└── plots/
    ├── lawler_runtime_vs_jobs.pdf
    ├── lawler_runtime_vs_jobs.png
    ├── lawler_runtime_vs_k.pdf
    ├── lawler_runtime_vs_k.png
    ├── lawler_runtime_vs_penalty_jobs.pdf
    ├── lawler_runtime_vs_penalty_jobs.png
    ├── lawler_runtime_vs_W.pdf
    ├── lawler_runtime_vs_W.png
    └── runtime_summary.csv
```

`lawler_runtime_vs_jobs.pdf` is the main scalability figure used in the paper.

## Experiment 3: Weight Scaling

Scripts:

- `benchmark_lawler_weight_scaling.py`
- `plot_lawler_weight_scaling.py`

This experiment isolates the pseudopolynomial dependence of Lawler's dynamic program on total weight $W$. For each base instance, the scheduling structure is fixed while every job weight is multiplied by

```text
alpha = 1, 2, 4, 8, 16
```

Thus,

$$
W_\alpha = \alpha W,
$$

while processing times, release times, deadlines, dummy jobs, $N$, and $k$ remain unchanged.

Settings:

- Original jobs: `5, 10, 15, 20`
- Base instances per job count: `10`
- Carbon interval density: approximately `0.5 n`

Run:

```bash
python benchmark_lawler_weight_scaling.py
python plot_lawler_weight_scaling.py
```

Results:

```text
lawler_weight_scaling_results/
├── results.csv
├── weight_scaling_summary.csv
├── lawler_runtime_vs_weight.pdf
└── lawler_runtime_vs_weight.png
```

The benchmark also checks that scaling all weights by $\alpha$ preserves the optimal solution and scales the optimal penalty by the same factor.

## Reproducing the Experiments

From the `lawler` directory:

```bash
# Reduction validation
python validate_lawler_reduction.py

# Scalability experiment
python benchmark_lawler_scaling.py
python plot_lawler_scaling.py

# Weight-scaling experiment
python benchmark_lawler_weight_scaling.py
python plot_lawler_weight_scaling.py
```
