# Lawler Experiments

This directory contains the implementation and experiments for evaluating
Lawler's exact dynamic program and the reduction from the cloud-outsourcing
problem to weighted penalty scheduling.

## Files

### Core implementation

- `dp.py`  
  Implementation of Lawler's exact dynamic program for
  \(1|pmtn,r_j|\sum w_j U_j\).

- `maxflow.py`  
  Max-flow/min-cost-flow scheduling routines used to independently solve
  the local scheduling problem during reduction validation.

- `compare_algorithms.py`  
  Utilities for comparing scheduling algorithms.

### Experiment 1: Reduction validation

- `validate_lawler_reduction.py`

Validates the cloud-outsourcing reduction on small instances.

For each generated instance:

1. Assign outsourcing penalties to the original jobs.
2. Construct the corresponding weighted penalty-scheduling instance.
3. Solve the reduced instance using Lawler's exact dynamic program.
4. Independently compute the cloud optimum by enumerating all local/cloud
   assignments and solving the local scheduling problem.
5. Check whether
   \[
   \mathrm{OPT}_{\mathrm{pen}}=\mathrm{OPT}_{\mathrm{cld}}.
   \]

Experimental setting:

- Original jobs: `5, 8, 10`
- Instances per job count: `30`
- Total instances: `90`
- Random seed: `42`

Results are stored in:

```text
lawler_stage1_validation/
├── stage1_validation.csv
└── summary.txt