#!/bin/bash
set -euo pipefail

JOB_COUNTS=(10 20 50 100 200 500 1000 2000 5000 10000)

INTERVAL_LEVELS=(
    "0.10 sparse"
    "0.25 light"
    "0.50 medium"
    "1.00 dense"
    "2.00 very_dense"
)

INSTANCES=30
TIMING_REPEATS=5
SEED=42

UTIL_MIN=0.70
UTIL_MAX=0.80
GREEN_MIN=0.50
GREEN_MAX=0.60
BROWN_MIN=0.20
BROWN_MAX=0.25

RESULT_ROOT="job_interval_scaling_results_gpu"
mkdir -p "$RESULT_ROOT"

for JOBS in "${JOB_COUNTS[@]}"; do
    HORIZON=$((6 * JOBS))

    for LEVEL in "${INTERVAL_LEVELS[@]}"; do
        read RATIO LABEL <<< "$LEVEL"

        INTERVALS=$(python3 -c "print(max(3, int(round(${JOBS} * ${RATIO}))))")
        OUTPUT="${RESULT_ROOT}/jobs_${JOBS}_intervals_${INTERVALS}_${LABEL}"

        echo "================================================================"
        echo "Jobs:      ${JOBS}"
        echo "Horizon:   ${HORIZON}"
        echo "Intervals: ${INTERVALS} (${LABEL}, ${RATIO} x jobs)"
        echo "Output:    ${OUTPUT}"
        echo "================================================================"

        CUDA_VISIBLE_DEVICES=0 python compare_algorithms.py \
            --jobs "$JOBS" \
            --horizon-per-job 6 \
            --intervals "$INTERVALS" \
            --instances "$INSTANCES" \
            --timing-repeats "$TIMING_REPEATS" \
            --seed "$SEED" \
            --util-min "$UTIL_MIN" \
            --util-max "$UTIL_MAX" \
            --green-min "$GREEN_MIN" \
            --green-max "$GREEN_MAX" \
            --brown-min "$BROWN_MIN" \
            --brown-max "$BROWN_MAX" \
            --output "$OUTPUT"
    done
done

echo "ALL JOB/INTERVAL SCALING EXPERIMENTS FINISHED"
echo "Plot with:"
echo "python plot_job_interval_scaling.py --root ${RESULT_ROOT}"
