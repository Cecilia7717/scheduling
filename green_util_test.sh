#!/bin/bash

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

TOTAL=$((${#GREEN_RANGES[@]} * ${#UTIL_RANGES[@]}))
COUNT=0

for UTIL_RANGE in "${UTIL_RANGES[@]}"; do

    read UTIL_MIN UTIL_MAX <<< "$UTIL_RANGE"

    for GREEN_RANGE in "${GREEN_RANGES[@]}"; do

        read GREEN_MIN GREEN_MAX <<< "$GREEN_RANGE"

        COUNT=$((COUNT + 1))

        OUTPUT_NAME="heatmap_results/util_${UTIL_MIN}_${UTIL_MAX}_green_${GREEN_MIN}_${GREEN_MAX}"

        echo
        echo "============================================================"
        echo "Experiment ${COUNT}/${TOTAL}"
        echo "Utilization: ${UTIL_MIN}-${UTIL_MAX}"
        echo "Green:       ${GREEN_MIN}-${GREEN_MAX}"
        echo "Brown:       uncontrolled"
        echo "============================================================"

        python compare_algorithms.py \
            --green-min "$GREEN_MIN" \
            --green-max "$GREEN_MAX" \
            --util-min "$UTIL_MIN" \
            --util-max "$UTIL_MAX" \
            --random-brown \
            --output "$OUTPUT_NAME"

        if [ $? -ne 0 ]; then
            echo
            echo "ERROR:"
            echo "Utilization ${UTIL_MIN}-${UTIL_MAX}"
            echo "Green ${GREEN_MIN}-${GREEN_MAX}"
            exit 1
        fi

    done
done

echo
echo "============================================================"
echo "ALL ${TOTAL} HEATMAP EXPERIMENTS FINISHED"
echo "============================================================"