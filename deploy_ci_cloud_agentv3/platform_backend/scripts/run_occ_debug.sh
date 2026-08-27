#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=/home/cidi/data_pipeline/pipeline-test-fix-data/shanqi_1/record_1
DATASET_NAME=clip_018_20260627_100427_test
DATA_DIR=$DATASET_PATH+/+$DATASET_NAME
IMAGE_TAG=172.16.201.100:5000/label_occ:v1.0.23_ni.xs_06-30_09_28_03
GPU_IDS="5"

: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"
: "${GPU_IDS:?ENV MISSING}"

echo "[INFO] Stage OCC | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
  -v "${DATASET_PATH}":/data_pipeline \
  -e "DATASET_PATH=${DATASET_PATH}" \
  -e "DATASET_NAME=${DATASET_NAME}" \
  "${IMAGE_TAG}" \
  release \
  --data-root "/data_pipeline/${DATASET_NAME}" \
  --camera-visibility-gpu-id ${GPU_IDS} \
  --config config/gen_occ_example.yaml \
  --skip-voxel-plot
