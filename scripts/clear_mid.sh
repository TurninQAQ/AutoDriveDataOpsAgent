#!/usr/bin/env bash

set -euo pipefail
DATASET_PATH=/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-27_095504
# DATASET_PATH=/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-27_094426
# DATASET_PATH=/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-17_153655
MAX_PARALLEL=8

clean_clip() {
    local clip_dir="$1"
    echo "Cleaning: $clip_dir"
    # rm -rf "$clip_dir"/calib
    rm -rf "$clip_dir"/camera
    rm -rf "$clip_dir"/labels
    rm -rf "$clip_dir"/lidar
    rm -rf "$clip_dir"/map
    rm -rf "$clip_dir"/pose
    rm -rf "$clip_dir"/temp_data
    rm -rf "$clip_dir"/*.json
}

export -f clean_clip

if command -v parallel &>/dev/null; then
    find "$DATASET_PATH" -maxdepth 1 -type d -name "clip_*" | parallel -j "$MAX_PARALLEL" clean_clip {}
else
    while [ $(jobs | wc -l) -ge "$MAX_PARALLEL" ]; do
        sleep 0.1
    done
    for clip_dir in "$DATASET_PATH"/clip_*; do
        if [ -d "$clip_dir" ]; then
            clean_clip "$clip_dir" &
        fi
    done
    wait
fi

echo "Batch clean completed."

