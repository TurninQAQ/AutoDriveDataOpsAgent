#!/usr/bin/env bash
set -euo pipefail

# DATASET_PATH=/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-27_095504
# CLIPS=(
#     "clip_000_20260627_095507"
#     "clip_001_20260627_095537"
#     "clip_002_20260627_095608"
#     "clip_003_20260627_095639"
#     "clip_004_20260627_095711"
# )

# DATASET_PATH=/home/cidi/data_pipeline/data_set_SQ_1/2026-07-06/record_CLOUD_MAPPING_2026-07-06_111723
# CLIPS=(
#     "clip_000_20260706_111727"
#     "clip_001_20260706_111757"
# )

DATASET_PATH=/home/cidi/data_pipeline/data_set_SQ_1/2026-07-14-xw/record_CLOUD_MAPPING_2026-07-14_170929-csc
CLIPS=(
    "clip_000_20260714_170932"
    "clip_001_20260714_171002"
    "clip_002_20260714_171033"
)

run_stage() {
    local stage=$1
    local image_tag=$2
    local script=$3
    
    echo "=== Starting $stage ==="
    for i in "${!CLIPS[@]}"; do
        local clip="${CLIPS[$i]}"
        local gpu=$((i + 5))
        local data_dir="${DATASET_PATH}/${clip}"
        
        echo "  ${clip} (GPU: ${gpu})"
        DATASET_PATH="${DATASET_PATH}" \
        DATASET_NAME="${clip}" \
        DATA_DIR="${data_dir}" \
        IMAGE_TAG="${image_tag}" \
        GPU_IDS="${gpu}" \
        ./${script} &
    done
    wait
    echo "=== $stage completed ==="
}

# run_stage "data_parser" \
#     "172.16.201.100:5000/data_parser:v1.1.0_cidi_07-13_10_57_06" \
#     "run_parser.sh"

run_stage "run_map" \
    "172.16.201.100:5000/offline_mapping:v1.1.2_cidi_07-14_12_39_10" \
    "run_map.sh"

# run_stage "run_od" \
#     "172.16.201.100:5000/label_od:v1.1.12_nty_06-29_16_11_23" \
#     "run_od.sh"

# run_stage "run_coloration" \
#     "172.16.201.100:5000/pointcloud_coloration:v1.0.7_cidi_07-03_11_29_27" \
#     "run_coloration.sh"

# run_stage "run_occ" \
#     "172.16.201.100:5000/label_occ:v1.0.23_ni.xs_06-30_09_28_03" \
#     "run_occ.sh"

echo "All stages completed!"