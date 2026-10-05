# Max-Flow Scheduling Experiments

This directory contains the implementations, synthetic benchmarks, CPU/GPU experiments, and analysis scripts used to evaluate the flow-based algorithms for carbon-aware scheduling.

The experiments focus on three main questions:

1. How does **MaxFlow-Passes** compare with **Pure-MinCostFlow**?
2. How much does **residual-network reuse** improve MaxFlow-Passes?
3. Does a GPU implementation of MaxFlow-Passes outperform the CPU implementation as the scheduling network grows?

The main algorithms considered in this directory are:

- **MaxFlow-Passes**
- **MaxFlow-Restart**
- **Pure-MinCostFlow**
- **GPU MaxFlow-Passes using ECL-MaxFlow**

---

## 1. Problem Setting

We consider preemptive single-machine scheduling with time-varying energy costs.

Each job has:

- release time `r_j`,
- deadline `d_j`,
- processing requirement `p_j`.

The time horizon is divided into energy intervals. Each interval belongs to one of three cost levels:

| Energy type | Cost |
|---|---:|
| Green | 0 |
| Brown | 1 |
| Red | 2 |

A feasible schedule must complete every job within its release-deadline window while respecting the capacity of the single machine.

The objective is to minimize total energy/carbon cost.

---

## 2. Algorithms

### 2.1 MaxFlow-Passes

`MaxFlow-Passes` solves the scheduling problem using a sequence of ordinary maximum-flow computations.

The algorithm progressively enables increasingly expensive energy intervals.

### Pass 1

Only green intervals are available:

```text
Green
```

### Pass 2

Brown intervals are enabled while preserving the flow obtained during the green pass:

```text
Green -> Brown
```

### Pass 3

If the schedule is still incomplete, red intervals are enabled:

```text
Green -> Brown -> Red
```

The important implementation feature is that the **same residual network is retained across passes**.

Conceptually:

```text
Build graph once

Pass 1: Green
          |
          v
      residual graph
          |
Pass 2: + Brown
          |
          v
      residual graph
          |
Pass 3: + Red
```

This avoids recomputing the cheaper-energy prefix from scratch.

---

### 2.2 MaxFlow-Restart

`MaxFlow-Restart` is an ablation used to measure the benefit of residual-network reuse.

Instead of keeping the residual graph between the outer passes, each prefix problem starts from a fresh network.

Conceptually:

```text
Prefix 1:
    G

Prefix 2:
    G -> B

Prefix 3:
    G -> B -> R
```

Therefore:

```text
MaxFlow-Passes:
    G -> B -> R

MaxFlow-Restart:
    G
    G -> B
    G -> B -> R
```

The current implementation uses the corrected:

```text
cold_prefix_replay_v2
```

definition.

Each prefix is recomputed from scratch, but within a prefix the same cheapest-first residual-flow logic as MaxFlow-Passes is used.

Therefore, when all exact solvers complete successfully, the final flow and carbon cost should agree between:

```text
MaxFlow-Passes
MaxFlow-Restart
Pure-MinCostFlow
```

The difference is the amount of computation required to obtain the solution.

---

### 2.3 Pure-MinCostFlow

`Pure-MinCostFlow` solves the problem using a single min-cost-flow network.

The energy costs are encoded directly into the network:

```text
Green = 0
Brown = 1
Red   = 2
```

This provides an exact baseline against which MaxFlow-Passes is compared.

The experiments compare both correctness and runtime.

---

## 3. Repository Structure

The main structure of this directory is:

```text
maxflow/
│
├── compare_maxflow_mincost/
│   ├── maxflow.py
│   ├── compare.py
│   ├── compare_algorithms.py
│   ├── compare_algorithm_exp2.py
│   ├── experiment2.py
│   ├── experiment_scaling.py
│   │
│   ├── green_interval_test.sh
│   ├── green_util_test.sh
│   ├── util_test.sh
│   ├── job_number_test.sh
│   ├── run_experiment2_pass_mechanism.sh
│   │
│   ├── experiment2_analysis/
│   ├── experiment2_pass_mechanism/
│   ├── heatmap_results/
│   ├── heatmap_results_25_50_jobs/
│   ├── job_interval_scaling_results/
│   ├── plot/
│   └── result/
│
├── ECL-MaxFlow/
│   ├── benchmark_gpu_passes.py
│   ├── maxflow_gpu_passes.py
│   ├── make_scheduling_graph.py
│   ├── convert_flow_to_schedule.py
│   ├── plot_job_interval_scaling_gpu.py
│   │
│   ├── job_number_test_gpu_ecl.sh
│   ├── convert_dimacs_to_eclgraph
│   ├── preprocess_eclgraph
│   ├── maxflow
│   │
│   ├── src/
│   ├── lib/
│   ├── test/
│   │
│   ├── gpu_passes_results/
│   ├── job_interval_ecl_new/
│   ├── job_interval_gpu_ecl_timing_adjusted/
│   └── job_interval_scaling_results_gpu_ecl/
│
├── job_interval_scaling_results_gpu_ecl_timing_adjusted/
│   ├── cpu_scaling_analysis/
│   ├── cpu_three_algorithm_analysis/
│   ├── cpu_gpu_algorithm_analysis/
│   │
│   ├── cpu_three_algorithms_all_instances.csv
│   ├── cpu_gpu_timing_all_instances.csv
│   ├── cpu_gpu_timing_summary.csv
│   │
│   ├── cpu_gpu_runtime_sparse.png
│   ├── cpu_gpu_runtime_light.png
│   ├── cpu_gpu_runtime_medium.png
│   ├── cpu_gpu_runtime_dense.png
│   ├── cpu_gpu_runtime_very_dense.png
│   ├── gpu_cpu_runtime_vs_jobs.png
│   └── gpu_cpu_runtime_vs_jobs.pdf
│
├── collect_plot_existing_cpu.py
├── plot_three_cpu_scaling.py
├── run_cpu_on_saved_instance.py
└── run_three_cpu_algorithms_existing_*.py
```

---

# 4. `compare_maxflow_mincost/`

This directory contains the main CPU implementation and the synthetic experiments comparing MaxFlow-Passes and MinCostFlow.

```text
compare_maxflow_mincost/
```

The core scheduling implementation is:

```text
compare_maxflow_mincost/maxflow.py
```

It contains the flow-network construction and CPU scheduling algorithms used throughout the experiments.

---

## 4.1 Runtime Comparison

The first group of experiments compares:

```text
MaxFlow-Passes
vs.
Pure-MinCostFlow
```

under different synthetic workload configurations.

Relevant scripts include:

```text
compare.py
compare_algorithms.py
compare_algorithm_exp2.py
experiment2.py
experiment_scaling.py
```

The experiments vary parameters including:

- target utilization,
- fraction of green energy,
- number of jobs,
- number of energy intervals,
- interval density.

The purpose is to determine under which workload conditions MaxFlow-Passes provides a runtime advantage over directly solving MinCostFlow.

---

## 4.2 Utilization Experiments

The utilization experiments vary the total workload relative to the available scheduling horizon.

Relevant scripts include:

```text
util_test.sh
green_util_test.sh
plot_utilization_runtime.py
```

These experiments study how solver runtime changes as the scheduling instance becomes increasingly constrained.

---

## 4.3 Green-Energy Experiments

The green-energy experiments vary the fraction of the time horizon assigned to low-cost green energy.

Relevant scripts include:

```text
green_interval_test.sh
green_util_test.sh
plot_green_runtime.py
```

These experiments examine how the availability of cheap energy affects the relative runtime of MaxFlow-Passes and MinCostFlow.

---

## 4.4 Heatmap Experiments

Heatmap experiments jointly vary workload parameters to identify regions where one algorithm is faster than another.

Relevant output directories include:

```text
heatmap_results/
heatmap_results_25_50_jobs/
```

Relevant plotting scripts include:

```text
plot_heatmap.py
plot_heatmap_job_number.py
```

An example generated result is:

```text
runtime_max_heatmap.png
runtime_ratio_table.csv
```

The heatmaps visualize runtime ratios between the compared algorithms.

---

# 5. Pass-Mechanism Experiment

The pass-mechanism experiment examines how much flow is added during each carbon-cost pass.

The relevant experiment is run using:

```text
run_experiment2_pass_mechanism.sh
```

with results stored under:

```text
experiment2_pass_mechanism/
```

The experiment measures how much of the final schedule is constructed during:

```text
Pass 1: Green
Pass 2: Brown
Pass 3: Red
```

This helps explain the behavior of MaxFlow-Passes and how much additional work is required after each cost level is enabled.

---

# 6. Synthetic Scaling Experiment

The main scaling experiment studies how the algorithms behave as the scheduling network becomes larger.

The synthetic generator varies:

```text
number of jobs
number of energy intervals
```

The interval-density levels are:

```text
sparse
light
medium
dense
very_dense
```

A typical generated instance directory has a name such as:

```text
jobs_1000_intervals_500_medium/
```

and contains files such as:

```text
instance_0001.json
instance_0002.json
...
```

---

## 6.1 Network Size

The number of original energy intervals is not necessarily the final number of interval nodes in the flow network.

Intervals are split at job release times and deadlines.

The experiments therefore also track:

```text
num_jobs
num_original_intervals
num_atomic_intervals
num_job_interval_edges
```

The number of `job_interval_edges` is particularly useful as a direct measure of the size of the scheduling network.

---

# 7. Three-Algorithm CPU Scaling Experiment

The main CPU scaling comparison evaluates:

```text
1. MaxFlow-Passes
2. MaxFlow-Restart
3. Pure-MinCostFlow
```

on the **same saved synthetic instances**.

This experiment is used to separate two effects:

```text
MaxFlow-Passes vs. Pure-MinCostFlow
    -> benefit of replacing MinCostFlow with ordinary MaxFlow passes

MaxFlow-Passes vs. MaxFlow-Restart
    -> benefit of residual-network reuse
```

The corrected Restart implementation performs a cold-prefix replay rather than merely using the previous cheap-flow amount as an upper capacity. This ensures that the restart baseline preserves the same cheapest-first behavior as MaxFlow-Passes. :chatgpt-content-reference{index="0"}

---

## 7.1 Correctness Check

For instances where all three algorithms complete, the benchmark verifies that they produce the same:

```text
total flow
normalized carbon cost
```

A mismatch is treated as an error.

This ensures that the runtime comparison is between exact algorithms solving the same optimization problem.

---

## 7.2 Timing Repetitions

The benchmark uses adaptive timing repetitions based on the number of jobs:

```text
jobs <= 200        : 10 repetitions
jobs <= 1000       : 5 repetitions
jobs <= 5000       : 3 repetitions
jobs > 5000        : 1 repetition
```

This reduces the cost of repeatedly solving very large instances while providing more stable measurements for small instances.

---

## 7.3 Stored Results

The three-algorithm experiment adds:

```text
cpu_exact_scaling
```

to the existing instance JSON.

The original instance data are preserved.

The combined CSV is:

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
    cpu_three_algorithms_all_instances.csv
```

Important columns include timing results for:

```text
passes_mean_seconds
restart_mean_seconds
mincost_mean_seconds
```

and runtime ratios such as:

```text
restart_over_passes
mincost_over_passes
```

---

## 7.4 Analysis

The analysis output is stored under:

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
    cpu_three_algorithm_analysis/
```

The plotting script is:

```bash
python3 plot_three_cpu_scaling.py
```

---

# 8. GPU Implementation

The GPU implementation is contained in:

```text
ECL-MaxFlow/
```

It uses **ECL-MaxFlow** as the underlying GPU maximum-flow implementation.

The scheduling-specific GPU pipeline consists of graph generation, graph conversion, GPU maximum flow, and schedule recovery.

Conceptually:

```text
Scheduling instance
        |
        v
make_scheduling_graph.py
        |
        v
scheduling.dimacs
        |
        v
convert_dimacs_to_eclgraph
        |
        v
scheduling.egr
        |
        v
ECL-MaxFlow
        |
        v
flow_output.csv
        |
        v
convert_flow_to_schedule.py
        |
        v
schedule_output.csv
```

The scheduling wrapper is:

```text
maxflow_gpu_passes.py
```

The main GPU benchmark script is:

```text
benchmark_gpu_passes.py
```

---

# 9. Building ECL-MaxFlow

Move to the GPU implementation directory:

```bash
cd ~/scheduling/maxflow/ECL-MaxFlow
```

The experiments were run on an NVIDIA Quadro RTX 8000.

The CUDA architecture used for compilation is:

```text
sm_75
```

For example:

```bash
nvcc -O3 -arch=sm_75 ...
```

The provided `Makefile` can also be used where appropriate.

To select GPU 0:

```bash
export CUDA_VISIBLE_DEVICES=0
```

---

# 10. GPU Scaling Experiment

GPU scaling results are generated using the ECL-MaxFlow implementation.

Relevant scripts include:

```text
benchmark_gpu_passes.py
job_number_test_gpu_ecl.sh
plot_job_interval_scaling_gpu.py
```

GPU results are stored in directories including:

```text
gpu_passes_results/
job_interval_ecl_new/
job_interval_gpu_ecl_timing_adjusted/
job_interval_scaling_results_gpu_ecl/
```

The experiments vary both:

```text
number of jobs
interval density
```

using the same general instance-generation settings as the CPU scaling experiments.

---

# 11. CPU vs. GPU Experiment

The CPU/GPU experiment compares MaxFlow-Passes running on the CPU with the ECL-MaxFlow-based GPU implementation.

The comparison uses the **same saved scheduling instances**.

The main combined result directory is:

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
```

---

## 11.1 Adding CPU Results to Existing GPU Instances

The script:

```text
run_cpu_on_saved_instance.py
```

recursively searches the experiment directory for:

```text
instance_*.json
```

and runs CPU MaxFlow-Passes on the stored jobs and energy intervals.

Existing GPU results are preserved.

The CPU result is added to the JSON as:

```json
"cpu_max_flow_passes": {
    ...
}
```

If `cpu_max_flow_passes` already exists, the instance is skipped unless overwrite mode is requested.

The CPU wall-clock measurement includes interval splitting, network construction, all required passes, and final schedule construction. :chatgpt-content-reference{index="1"}

---

## 11.2 Running the CPU Solver on Saved Instances

From:

```bash
cd ~/scheduling/maxflow
```

run:

```bash
python3 run_cpu_on_saved_instance.py \
    --root job_interval_scaling_results_gpu_ecl_timing_adjusted
```

To recompute CPU results that already exist:

```bash
python3 run_cpu_on_saved_instance.py \
    --root job_interval_scaling_results_gpu_ecl_timing_adjusted \
    --overwrite-cpu
```

To execute the CPU solver without modifying the JSON files:

```bash
python3 run_cpu_on_saved_instance.py \
    --root job_interval_scaling_results_gpu_ecl_timing_adjusted \
    --dry-run
```

---

# 12. CPU/GPU Timing Metrics

The CPU/GPU comparison records three main timing quantities.

### CPU wall time

```text
cpu_max_flow_passes.wall_seconds
```

This represents the complete CPU MaxFlow-Passes execution.

### GPU algorithm time

```text
gpu_max_flow_passes.gpu_maxflow_seconds
```

This measures the GPU-side MaxFlow-Passes computation used in the experiment.

### GPU wall time

```text
gpu_max_flow_passes.wall_seconds
```

This measures the complete GPU workflow.

The analysis collects these values from each saved instance and groups them by experiment configuration.

---

# 13. CPU/GPU Result Files

The main combined result directory is:

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
```

---

## Raw CPU/GPU Timing Data

```text
cpu_gpu_timing_all_instances.csv
```

This file contains one row for each instance.

Important fields include:

```text
file
instance_id
folder
num_jobs
num_intervals
density
cpu_wall_seconds
gpu_maxflow_seconds
gpu_wall_seconds
```

---

## Aggregated CPU/GPU Timing Data

```text
cpu_gpu_timing_summary.csv
```

Results are grouped by experiment configuration.

The summary contains:

```text
folder
num_jobs
num_intervals
density
num_instances

mean_cpu_wall_seconds
median_cpu_wall_seconds

mean_gpu_maxflow_seconds
median_gpu_maxflow_seconds

mean_gpu_wall_seconds
median_gpu_wall_seconds
```

The collection code explicitly reads the CPU wall time, GPU max-flow time, and GPU wall time from the same saved instances. :chatgpt-content-reference{index="2"}

---

# 14. CPU/GPU Plots

Separate runtime plots are generated for each interval density:

```text
cpu_gpu_runtime_sparse.png
cpu_gpu_runtime_light.png
cpu_gpu_runtime_medium.png
cpu_gpu_runtime_dense.png
cpu_gpu_runtime_very_dense.png
```

The plots compare:

```text
CPU MaxFlow-Passes wall time
GPU MaxFlow-Passes algorithm time
GPU complete wall time
```

against the number of jobs.

Both axes are typically shown on logarithmic scales for the scaling experiments.

The combined paper figure is stored as:

```text
gpu_cpu_runtime_vs_jobs.png
gpu_cpu_runtime_vs_jobs.pdf
```

---

# 15. Analysis Directories

The combined experiment directory contains several analysis subdirectories.

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
│
├── cpu_scaling_analysis/
├── cpu_three_algorithm_analysis/
└── cpu_gpu_algorithm_analysis/
```

### `cpu_scaling_analysis/`

Contains CPU scaling analysis based on the saved synthetic instances.

### `cpu_three_algorithm_analysis/`

Contains the comparison among:

```text
MaxFlow-Passes
MaxFlow-Restart
Pure-MinCostFlow
```

### `cpu_gpu_algorithm_analysis/`

Contains analysis comparing the CPU and GPU implementations.

---

# 16. Main Experiment Summary

The experiments in this directory can be summarized as follows:

| Experiment | Comparison | Main purpose |
|---|---|---|
| Runtime comparison | MaxFlow-Passes vs. Pure-MinCostFlow | Compare exact solver runtime |
| Utilization sweep | MaxFlow-Passes vs. Pure-MinCostFlow | Study effect of workload pressure |
| Green-share sweep | MaxFlow-Passes vs. Pure-MinCostFlow | Study effect of cheap-energy availability |
| Heatmap | MaxFlow-Passes vs. Pure-MinCostFlow | Identify workload regimes with different runtime behavior |
| Pass mechanism | Green/Brown/Red passes | Measure where flow is added |
| Residual reuse | Passes vs. Restart | Measure benefit of residual-network reuse |
| CPU scaling | Passes vs. Restart vs. MinCostFlow | Compare exact algorithms as network size grows |
| GPU scaling | CPU Passes vs. GPU Passes | Evaluate GPU acceleration |
| Real data | Alibaba cluster trace | Evaluate the scheduling method on realistic workloads |

---

# 17. Relationship Between the Main CPU Algorithms

The three CPU algorithms can be viewed as follows:

```text
                    Exact scheduling problem
                            |
             +--------------+--------------+
             |                             |
             v                             v
      Pure-MinCostFlow              Cheapest-first MaxFlow
                                           |
                                +----------+----------+
                                |                     |
                                v                     v
                         MaxFlow-Passes       MaxFlow-Restart
                         reuse residual       rebuild prefixes
                            graph
```

This allows the experiments to distinguish between:

1. the benefit of replacing MinCostFlow with ordinary MaxFlow computations, and
2. the additional benefit of preserving the residual graph across passes.

---

# 18. Reproducing the Main Experiments

## MaxFlow-Passes vs. MinCostFlow

Move to:

```bash
cd ~/scheduling/maxflow/compare_maxflow_mincost
```

The scripts in this directory reproduce the synthetic runtime,
utilization, green-share, and heatmap experiments.

The shell scripts provide convenient entry points for parameter sweeps.

---

## Three CPU Algorithms

Move to:

```bash
cd ~/scheduling/maxflow
```

Run the corresponding `run_three_cpu_algorithms_existing_*.py` script on:

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
```

The benchmark runs:

```text
MaxFlow-Passes
MaxFlow-Restart
Pure-MinCostFlow
```

on the same saved instances.

The implementation supports options for:

```text
--root
--timing-repeats
--timeout-seconds
--min-jobs
--max-jobs
--density
--overwrite
--dry-run
```

When no fixed timing repetition count is provided, adaptive repetitions are used.

---

## CPU/GPU Comparison

First generate the GPU results using the ECL-MaxFlow implementation.

Then run:

```bash
cd ~/scheduling/maxflow

python3 run_cpu_on_saved_instance.py \
    --root job_interval_scaling_results_gpu_ecl_timing_adjusted
```

This adds CPU measurements to the existing instances and collects CPU/GPU timing data.

---

# 19. Current GPU Result

The GPU implementation was tested across increasing job counts and interval densities.

For the scheduling networks evaluated in these experiments, the ECL-MaxFlow GPU implementation did **not** outperform the CPU implementation.

The GPU implementation incurs additional overhead from the GPU workflow and ECL integration, while the CPU implementation remains faster over the tested range.

Therefore:

- the **CPU implementation** is used as the primary implementation for the main algorithmic experiments;
- the **GPU implementation** is retained as a separate acceleration/scalability experiment.

---

# 20. Real-Data Experiments

The synthetic experiments in this directory are complemented by experiments using the Alibaba cluster trace.

Those experiments are maintained separately under the Alibaba dataset directory.

The real-data experiment converts selected Alibaba workloads into scheduling instances and evaluates the proposed scheduling algorithm on workloads derived from real cluster traces.

The purpose of the real-data experiment is different from the synthetic scaling experiments:

```text
Synthetic experiments:
    controlled parameter sweeps
    algorithmic scaling
    runtime comparisons

Alibaba experiments:
    realistic workload characteristics
    real job/task timing information
    practical scheduling behavior
```

---

# 21. Lawler Baseline

The Lawler baseline is maintained separately under:

```text
~/scheduling/lawler/
```

That directory contains the implementation and experiments for Lawler's exact dynamic-programming algorithm.

The Lawler experiments are separate from the flow-solver experiments in this directory.

---

# 22. Generated Data

Many experiment directories contain generated JSON instances.

A typical instance contains:

```text
instance description
jobs
energy intervals
split intervals
construction/witness information
GPU results
CPU results
three-algorithm CPU results
```

Different experiments may append additional result sections to the same instance.

For example:

```text
gpu_max_flow_passes
cpu_max_flow_passes
cpu_exact_scaling
```

Existing instance contents should generally be preserved when adding new experimental results.

---

# 23. Rerunning Experiments

Generated result directories can be large.

Before rerunning an experiment, check whether the relevant result is already stored in the instance JSON.

Most batch scripts skip existing results by default.

Where supported, use:

```text
--overwrite
```

or:

```text
--overwrite-cpu
```

to intentionally recompute existing results.

For testing without modifying existing files, use:

```text
--dry-run
```

where available.

Avoid manually modifying individual generated JSON files unless necessary, because several later analysis scripts expect the stored result structure to remain consistent.

---

# 24. Main Paper Figures and Data

The most important result files for the paper are located under:

```text
job_interval_scaling_results_gpu_ecl_timing_adjusted/
```

In particular:

```text
cpu_three_algorithms_all_instances.csv
cpu_gpu_timing_all_instances.csv
cpu_gpu_timing_summary.csv
gpu_cpu_runtime_vs_jobs.pdf
gpu_cpu_runtime_vs_jobs.png
```

The corresponding analysis directories are:

```text
cpu_scaling_analysis/
cpu_three_algorithm_analysis/
cpu_gpu_algorithm_analysis/
```

The earlier MaxFlow-vs-MinCostFlow parameter-sweep results are under:

```text
compare_maxflow_mincost/
```

including:

```text
experiment2_analysis/
experiment2_pass_mechanism/
heatmap_results/
heatmap_results_25_50_jobs/
job_interval_scaling_results/
plot/
result/
```

---

# 25. Experimental Workflow

The overall experimental workflow is:

```text
                    Synthetic instance generation
                              |
                              v
                     Saved instance JSON
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   MaxFlow-Passes      MaxFlow-Restart     Pure-MinCostFlow
          |
          |
          +---------------- GPU implementation
                              |
                              v
                         ECL-MaxFlow
                              |
                              v
                     Save timing/results
                              |
                              v
                       Collect CSV data
                              |
                              v
                         Plot results
                              |
                              v
                        Paper figures
```

For the real-data evaluation:

```text
Alibaba cluster trace
        |
        v
workload extraction
        |
        v
scheduling instances
        |
        v
CPU scheduling algorithm
        |
        v
real-data evaluation
```

---

# 26. Notes

- `MaxFlow-Passes`, `MaxFlow-Restart`, and `Pure-MinCostFlow` are intended to be exact algorithms for the tested scheduling formulation.
- Completed three-algorithm runs are checked for equal flow and equal carbon cost.
- `MaxFlow-Restart` uses the corrected `cold_prefix_replay_v2` implementation.
- CPU and GPU comparisons should use the same saved scheduling instances whenever possible.
- Runtime results should be interpreted together with network size, especially the number of atomic intervals and job-interval edges.
- Generated experiment directories should not normally be edited manually.
- GPU results include both algorithm-level and complete wall-clock measurements because ECL integration introduces additional overhead.