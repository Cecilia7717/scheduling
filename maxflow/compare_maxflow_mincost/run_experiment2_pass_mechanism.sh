#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 2:
#   Why is MaxFlow-Passes faster?
#
# For every (utilization, green-share) cell, compare:
#
#   1. MaxFlow-Passes-Reuse
#      Reuses the same residual graph across green/brown/red.
#
#   2. MaxFlow-Restart
#      Rebuilds the network and solves from zero at each pass.
#
#   3. Pure-MinCostFlow
#      Direct min-cost flow baseline.
#
# The Python script additionally records:
#   - flow added in green / brown / red
#   - fraction of total demand added in each pass
#   - pass-by-pass runtime
#   - BFS counts
#   - DFS counts
#   - edge scans
#   - successful augmenting pushes
#   - residual rerouting between passes
#   - network size
#   - restart/reuse runtime ratio
#   - mincost/reuse runtime ratio
#
# Brown share is intentionally uncontrolled during this 2-D sweep,
# matching the earlier green-share experiment and avoiding invalid
# green+brown combinations at high green shares.
# ============================================================


# ----------------------------
# Publication settings
# ----------------------------

INSTANCES="${INSTANCES:-30}"
TIMING_REPEATS="${TIMING_REPEATS:-20}"
SEED="${SEED:-42}"

MIN_JOBS="${MIN_JOBS:-35}"
MAX_JOBS="${MAX_JOBS:-50}"

# With p_j in [1,6], horizon = 6 * jobs makes the full utilization
# range through 1.00 achievable without silently falling back to a
# different utilization target.
HORIZON_PER_JOB="${HORIZON_PER_JOB:-6}"

ROOT_OUTPUT="${ROOT_OUTPUT:-experiment2_pass_mechanism}"


# ----------------------------
# Utilization sweep
# ----------------------------

UTIL_RANGES=(
    "0.30 0.35"
    "0.35 0.40"
    "0.40 0.45"
    "0.45 0.50"
    "0.50 0.55"
    "0.55 0.60"
    "0.60 0.65"
    "0.65 0.70"
    "0.70 0.75"
    "0.75 0.80"
    "0.80 0.85"
    "0.85 0.90"
    "0.90 0.95"
    "0.95 1.00"
)


# ----------------------------
# Green-share sweep
# ----------------------------

GREEN_RANGES=(
    "0.20 0.25"
    "0.25 0.30"
    "0.30 0.35"
    "0.35 0.40"
    "0.40 0.45"
    "0.45 0.50"
    "0.50 0.55"
    "0.55 0.60"
    "0.60 0.65"
    "0.65 0.70"
    "0.70 0.75"
    "0.75 0.80"
    "0.80 0.85"
)


mkdir -p "$ROOT_OUTPUT"

echo "============================================================"
echo "EXPERIMENT 2: MAX-FLOW PASS MECHANISM"
echo "============================================================"
echo "Instances/cell:       $INSTANCES"
echo "Timing repeats:       $TIMING_REPEATS"
echo "Seed:                 $SEED"
echo "Jobs:                 $MIN_JOBS-$MAX_JOBS"
echo "Horizon/job:          $HORIZON_PER_JOB"
echo "Output root:          $ROOT_OUTPUT"
echo "Utilization cells:    ${#UTIL_RANGES[@]}"
echo "Green-share cells:    ${#GREEN_RANGES[@]}"
echo "Total cells:          $((${#UTIL_RANGES[@]} * ${#GREEN_RANGES[@]}))"
echo "============================================================"
echo


for UTIL_RANGE in "${UTIL_RANGES[@]}"; do
    read -r UTIL_MIN UTIL_MAX <<< "$UTIL_RANGE"

    for GREEN_RANGE in "${GREEN_RANGES[@]}"; do
        read -r GREEN_MIN GREEN_MAX <<< "$GREEN_RANGE"

        OUTPUT_NAME="${ROOT_OUTPUT}/util_${UTIL_MIN}_${UTIL_MAX}_green_${GREEN_MIN}_${GREEN_MAX}"

        echo "============================================================"
        echo "Utilization:  ${UTIL_MIN} - ${UTIL_MAX}"
        echo "Green share:  ${GREEN_MIN} - ${GREEN_MAX}"
        echo "Output:       ${OUTPUT_NAME}"
        echo "============================================================"

        python3 compare_algorithm_exp2.py \
            --instances "$INSTANCES" \
            --timing-repeats "$TIMING_REPEATS" \
            --seed "$SEED" \
            --min-jobs "$MIN_JOBS" \
            --max-jobs "$MAX_JOBS" \
            --horizon-per-job "$HORIZON_PER_JOB" \
            --util-min "$UTIL_MIN" \
            --util-max "$UTIL_MAX" \
            --green-min "$GREEN_MIN" \
            --green-max "$GREEN_MAX" \
            --random-brown \
            --output "$OUTPUT_NAME"

        echo
    done
done


echo "============================================================"
echo "ALL EXPERIMENT-2 CELLS FINISHED"
echo "============================================================"
echo "Results root: $ROOT_OUTPUT"
echo
echo "Each cell contains:"
echo "  results.csv"
echo "  summary.txt"
echo "  instance_XXXX.json"
echo "============================================================"
