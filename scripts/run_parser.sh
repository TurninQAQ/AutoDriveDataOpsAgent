#!/usr/bin/env bash
set -euo pipefail

if [ "${PLATFORM_STAGE_RUNTIME:-real}" = "mock" ]; then
    exec bash "$(dirname "$0")/run_mock_stage.sh" "parser"
fi


# DATASET_PATH=/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-17_153655
# DATASET_NAME=clip_009_20260617_154135
# DATA_DIR=DATASET_PATH+/+DATASET_NAME
# IMAGE_TAG=172.16.201.100:5000/data_parser:v1.0.7_cidi_06-27_09_27_35
# GPU_IDS="2"

: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"
: "${CONTAINER_NAME:?ENV MISSING}"

echo "[INFO] Stage PARSER | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
    --name "${CONTAINER_NAME}" \
    -v "${DATASET_PATH}":/data_pipeline \
    "${IMAGE_TAG}" \
    release \
    "/data_pipeline/${DATASET_NAME}"
