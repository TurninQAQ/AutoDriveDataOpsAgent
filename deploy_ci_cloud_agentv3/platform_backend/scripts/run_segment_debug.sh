#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET_PATH="${DATASET_PATH:-/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-17_153655}"
export DATASET_NAME="${DATASET_NAME:-clip_007_20260617_154032}"
export DATA_DIR="${DATA_DIR:-${DATASET_PATH}/${DATASET_NAME}}"
export IMAGE_TAG="${IMAGE_TAG:-172.16.201.100:5000/sam31:v1.0.9_cfy_07-13_11_17_09}"
export GPU_IDS="${GPU_IDS:-5}"

exec bash "${SCRIPT_DIR}/run_segment.sh" "$@"
