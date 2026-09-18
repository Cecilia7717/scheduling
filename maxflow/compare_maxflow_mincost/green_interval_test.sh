#!/bin/bash

# Sweep green share from 0.20-0.25 through 0.80-0.85.
# Keep target utilization fixed at 0.60-0.65.
# Do not control the brown share.
# All other benchmark settings use their existing defaults.

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

for RANGE in "${GREEN_RANGES[@]}"; do
    read GREEN_MIN GREEN_MAX <<< "$RANGE"

    OUTPUT_NAME="benchmark_green_${GREEN_MIN}_${GREEN_MAX}"

    echo "============================================================"
    echo "Running green range: ${GREEN_MIN} - ${GREEN_MAX}"
    echo "Target utilization: 0.60 - 0.65"
    echo "Brown share: uncontrolled"
    echo "Output: ${OUTPUT_NAME}"
    echo "============================================================"

    python compare_algorithms.py \
        --green-min "$GREEN_MIN" \
        --green-max "$GREEN_MAX" \
        --util-min 0.60 \
        --util-max 0.65 \
        --random-brown \
        --output "$OUTPUT_NAME"

    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment ${GREEN_MIN}-${GREEN_MAX} failed."
        exit 1
    fi

    echo
done

echo "============================================================"
echo "ALL GREEN-SHARE EXPERIMENTS FINISHED"
echo "============================================================"
