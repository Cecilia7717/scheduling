# Carbon-Aware Scheduling Experiments

This repository contains implementations and experiments for **carbon-aware scheduling with time-varying energy costs**.

The main goal is to evaluate exact scheduling algorithms based on maximum flow and compare them with alternative formulations, GPU implementations, and real-world workloads.

## Problem

We consider preemptive scheduling on a single edge server. Each job \(j\) has:

- release time \(r_j\),
- deadline \(d_j\),
- processing requirement \(p_j\).

The scheduling horizon is divided into intervals with different carbon costs:

| Energy | Cost |
|---|---:|
| Green | 0 |
| Brown | 1 |
| Red | 2 |

The objective is to complete all required processing while minimizing total carbon cost.

## Algorithms

The main exact algorithms are:

**MaxFlow-Passes** — progressively enables green, brown, and red capacity while preserving the current flow and residual network between phases.

**MaxFlow-Restart** — follows the same sequence but rebuilds and resolves the flow network at each phase. This is used to measure the benefit of residual reuse.

**MinCostFlow** — directly solves the scheduling problem as a minimum-cost flow problem on the same job-interval network.

For feasible instances, the exact algorithms are expected to produce the same optimal carbon cost.

## Repository Structure

```text
scheduling/
│
├── maxflow/
│   ├── compare_maxflow_mincost/
│   ├── ECL-MaxFlow/
│   └── ...
│
├── cluster-trace-v2018/
│   ├── alibaba_scaling_jobs/
│   ├── run_alibaba_experiment.py
│   ├── run_alibaba_scaling_experiment.py
│   └── ...
│
├── lawler/
│   ├── validate_lawler_reduction.py
│   ├── plot_lawler_scaling.py
│   └── ...
│
├── Platforms/
│
└── .gitignore
```

### `maxflow/`

Contains the main synthetic experiments and CPU/GPU implementations.

Experiments include:

- MaxFlow-Passes vs. MinCostFlow
- residual-network reuse
- flow distribution across carbon phases
- scalability with jobs and job-interval edges
- CPU vs. GPU max-flow performance

The GPU implementation uses **ECL-MaxFlow**.

### `cluster-trace-v2018/`

Contains experiments based on the **Alibaba cluster trace**.

Alibaba jobs are converted into scheduling instances using observed task execution information. Tasks within a job form a DAG, and their scheduling windows are tightened according to precedence constraints.

These experiments test whether the performance observed on synthetic instances also holds for production-derived workloads.

### `lawler/`

Contains experiments for the cloud-outsourcing reduction and **Lawler's dynamic programming algorithm**.

The experiments include:

- correctness validation against exhaustive enumeration,
- scaling with the number of jobs,
- scaling with interval density,
- scaling with total penalty weight.

### `Platforms/`

Contains earlier platform-related experiments and supporting notebooks.

## Main Experimental Questions

The experiments study:

1. How does MaxFlow-Passes compare with direct MinCostFlow?
2. How much does residual-network reuse improve runtime?
3. How does runtime scale with the size of the constructed flow network?
4. Can GPU max-flow accelerate the scheduling algorithm?
5. Do the results carry over to Alibaba trace-derived workloads?
6. How does Lawler's exact dynamic program scale?

## Running Experiments

Clone the repository:

```bash
git clone https://github.com/Cecilia7717/scheduling.git
cd scheduling
```

Create a Python environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then enter the relevant experiment directory:

```bash
cd maxflow
```

or

```bash
cd cluster-trace-v2018
```

or

```bash
cd lawler
```

See the README inside each subdirectory for experiment-specific details and commands.

## Notes

Some datasets and experiment outputs are too large to store in GitHub and are excluded through `.gitignore`. These files may need to be downloaded or regenerated locally.
