#!/bin/bash

set -euo pipefail

DATASET_PATH=/home/cidi/data_pipeline/data_set_SQ_1/2026-07-14-xw/record_CLOUD_MAPPING_2026-07-14_170929-csc
dirs=(
    "clip_000_20260714_170932"
    "clip_001_20260714_171002"
    "clip_002_20260714_171033"
    )

# DATASET_PATH=/home/cidi/data_pipeline/data_set_SQ_1/2026-07-14/record_CLOUD_MAPPING_2026-07-14_091228/
# dirs=(
#     "clip_000_20260714_091231"
#     "clip_001_20260714_091301"
#     )

# DATASET_PATH=/home/cidi/data_pipeline/data_set_SQ_1/2026-07-14/record_CLOUD_MAPPING_2026-07-14_091708/
# dirs=(
#     "clip_000_20260714_091711"
#     "clip_001_20260714_091741"
#     "clip_002_20260714_091815"
#     )

# DATASET_PATH=/home/cidi/data_pipeline/data_set_SQ_1/2026-07-14-xw/record_CLOUD_MAPPING_2026-07-14_171415
# dirs=(
#     "clip_000_20260714_171418"
#     "clip_001_20260714_171449"
#     "clip_002_20260714_171520"
#     "clip_003_20260714_171550"
#     "clip_004_20260714_171624"
#     "clip_005_20260714_171655"
#     "clip_006_20260714_171725"
#     "clip_007_20260714_171755"
#     "clip_008_20260714_171826"
#     "clip_009_20260714_171902"
#     "clip_010_20260714_171932"
#     "clip_011_20260714_172003"
#     "clip_012_20260714_172033"
#     "clip_013_20260714_172104"
#     "clip_014_20260714_172134"
#     "clip_015_20260714_172204"
#     "clip_016_20260714_172236"
#     "clip_017_20260714_172307"
#     "clip_018_20260714_172337"
#     )

for dir in "${dirs[@]}"; do
    clip_dir="$DATASET_PATH/$dir"
    if [[ ! -d $clip_dir ]]; then
        echo "not exist: $dir"
        continue
    fi

    echo "[rm dir] $clip_dir"
    rm -rf "$clip_dir"/calib
    rm -rf "$clip_dir"/calibration
    rm -rf "$clip_dir"/camera
    rm -rf "$clip_dir"/labels
    rm -rf "$clip_dir"/lidar
    rm -rf "$clip_dir"/map
    rm -rf "$clip_dir"/pose
    rm -rf "$clip_dir"/temp_data
    rm -rf "$clip_dir"/*.json

    mkdir -p "$clip_dir"/camera/raw/back_normal
    mkdir -p "$clip_dir"/camera/infer/undistort_images/back_normal

    for file in "$clip_dir"/*.log "$clip_dir"/*.tmp; do
        [[ -e "$file" ]] || continue
        echo "  ??: $file"
        rm -f "$file"
    done
done


echo "Batch clean completed."

