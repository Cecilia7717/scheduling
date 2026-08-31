#!/bin/bash 
# Run utilization experiments from 0.50-0.55 through 0.95-1.00. 
# Jobs are fixed to the range 35-50.
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

for RANGE in "${UTIL_RANGES[@]}"; do
    read UTIL_MIN UTIL_MAX <<< "$RANGE"

    OUTPUT_NAME="benchmark_util_${UTIL_MIN}_${UTIL_MAX}"

    echo "============================================================"
    echo "Running utilization range: ${UTIL_MIN} - ${UTIL_MAX}"
    echo "Output: ${OUTPUT_NAME}"
    echo "============================================================"

    python compare_algorithms.py \
        --min-jobs 35 \
        --max-jobs 50 \
        --util-min "$UTIL_MIN" \
        --util-max "$UTIL_MAX" \
        --output "$OUTPUT_NAME"

    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment ${UTIL_MIN}-${UTIL_MAX} failed."
        exit 1
    fi

    echo
done

echo "============================================================"
echo "ALL UTILIZATION EXPERIMENTS FINISHED"
echo "============================================================"
