#!/bin/bash
set -euo pipefail


# ============================================================
# Job counts
# ============================================================
# 
    # 10
    # 20
    # 50
    # 100
    # 200
    # 500
    # 1000
    # 2000
    
JOB_COUNTS=(
    5000
    10000
)


# ============================================================
# Number of original energy intervals relative to job count
#
# Format:
#     ratio label
# ============================================================

INTERVAL_LEVELS=(
    "0.10 sparse"
    "0.25 light"
    "0.50 medium"
    "1.00 dense"
    "2.00 very_dense"
)


# ============================================================
# Experiment settings
# ============================================================

INSTANCES=5
SEED=42

HORIZON_PER_JOB=6

UTIL_MIN=0.70
UTIL_MAX=0.80

GREEN_MIN=0.50
GREEN_MAX=0.60

BROWN_MIN=0.20
BROWN_MAX=0.25


# ============================================================
# ECL-MaxFlow location
# ============================================================

ECL_DIR="/home/nbl0582/scheduling/maxflow_gpu/ECL-MaxFlow"


# ============================================================
# Python GPU-pass benchmark
# ============================================================

BENCHMARK_SCRIPT="${ECL_DIR}/benchmark_gpu_passes.py"


# ============================================================
# Output root
# ============================================================

RESULT_ROOT="${ECL_DIR}/job_interval_scaling_results_gpu_ecl"

mkdir -p "$RESULT_ROOT"


# ============================================================
# GPU
# ============================================================

export CUDA_VISIBLE_DEVICES=0


# ============================================================
# Print experiment configuration
# ============================================================

echo "================================================================"
echo "GPU ECL MAX-FLOW + PASSES SCALING EXPERIMENT"
echo "================================================================"
echo "ECL directory:      ${ECL_DIR}"
echo "Instances / config: ${INSTANCES}"
echo "Seed:               ${SEED}"
echo "Horizon / job:      ${HORIZON_PER_JOB}"
echo "Utilization:        ${UTIL_MIN} - ${UTIL_MAX}"
echo "Green share:        ${GREEN_MIN} - ${GREEN_MAX}"
echo "Brown share:        ${BROWN_MIN} - ${BROWN_MAX}"
echo "GPU:                CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Result root:        ${RESULT_ROOT}"
echo "================================================================"
echo


# ============================================================
# Run experiment
# ============================================================

for JOBS in "${JOB_COUNTS[@]}"; do

    HORIZON=$((HORIZON_PER_JOB * JOBS))

    for LEVEL in "${INTERVAL_LEVELS[@]}"; do

        read -r RATIO LABEL <<< "$LEVEL"

        INTERVALS=$(
            python3 -c \
            "print(max(3, int(round(${JOBS} * ${RATIO}))))"
        )


        # ----------------------------------------------------
        # Safety check:
        #
        # Number of intervals cannot exceed horizon because
        # every generated interval has positive integer length.
        # ----------------------------------------------------

        if (( INTERVALS > HORIZON )); then
            INTERVALS=$HORIZON
        fi


        OUTPUT="${RESULT_ROOT}/jobs_${JOBS}_intervals_${INTERVALS}_${LABEL}"


        echo
        echo "================================================================"
        echo "Jobs:               ${JOBS}"
        echo "Horizon:            ${HORIZON}"
        echo "Original intervals: ${INTERVALS}"
        echo "Interval density:   ${LABEL} (${RATIO} x jobs)"
        echo "Instances:          ${INSTANCES}"
        echo "Output:             ${OUTPUT}"
        echo "================================================================"


        python3 "$BENCHMARK_SCRIPT" \
            --ecl-dir "$ECL_DIR" \
            --jobs "$JOBS" \
            --horizon-per-job "$HORIZON_PER_JOB" \
            --intervals "$INTERVALS" \
            --instances "$INSTANCES" \
            --seed "$SEED" \
            --util-min "$UTIL_MIN" \
            --util-max "$UTIL_MAX" \
            --green-min "$GREEN_MIN" \
            --green-max "$GREEN_MAX" \
            --brown-min "$BROWN_MIN" \
            --brown-max "$BROWN_MAX" \
            --output "$OUTPUT"


        echo
        echo "Finished:"
        echo "  jobs=${JOBS}"
        echo "  intervals=${INTERVALS}"
        echo "  density=${LABEL}"
        echo

    done

done


echo
echo "================================================================"
echo "ALL GPU ECL JOB/INTERVAL SCALING EXPERIMENTS FINISHED"
echo "================================================================"
echo
echo "Results:"
echo "${RESULT_ROOT}"
echo