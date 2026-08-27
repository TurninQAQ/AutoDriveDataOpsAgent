#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=/home/cidi/data_pipeline/pipeline-test-fix-data/shanqi_1/record_1
DATASET_NAME=clip_018_20260627_100427_test
DATA_DIR=$DATASET_PATH+/+$DATASET_NAME
IMAGE_TAG=172.16.201.100:5000/data_parser:v1.0.10_cidi_07-03_10_17_52
GPU_IDS="2"

echo "[INFO] Stage PARSER | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
    -v "${DATASET_PATH}":/data_pipeline \
    "${IMAGE_TAG}" \
    release \
    "/data_pipeline/${DATASET_NAME}"