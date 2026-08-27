#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=/home/cidi/data_pipeline/pipeline-test-fix-data/shanqi_1/record_1
DATASET_NAME=clip_018_20260627_100427_test
DATA_DIR=$DATASET_PATH+/+$DATASET_NAME
IMAGE_TAG=172.16.201.100:5000/offline_mapping:v1.0.7_root_07-01_14_25_10
GPU_IDS="2"

: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"

echo "[INFO] Stage MAPPING | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
    -v "${DATASET_PATH}":/data_pipeline \
    "${IMAGE_TAG}" \
    release \
    "/data_pipeline/${DATASET_NAME}"