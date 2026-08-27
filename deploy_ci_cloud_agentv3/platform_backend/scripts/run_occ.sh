#!/usr/bin/env bash
set -euo pipefail

if [ "${PLATFORM_STAGE_RUNTIME:-real}" = "mock" ]; then
    exec bash "$(dirname "$0")/run_mock_stage.sh" "occ"
fi


# DATASET_PATH=/home/cidi/data_pipeline/2026-06-17/record_CLOUD_MAPPING_2026-06-17_153655
# DATASET_NAME=clip_007_20260617_154032
# DATA_DIR=DATASET_PATH+/+DATASET_NAME
# IMAGE_TAG=172.16.201.100:5000/label_occ:v1.0.22_ni.xs_06-29_19_55_53
# GPU_IDS="5"

: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"
: "${GPU_IDS:?ENV MISSING}"
: "${CONTAINER_NAME:?ENV MISSING}"

echo "[INFO] Stage OCC | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  -v "${DATASET_PATH}":/data_pipeline \
  -e "DATASET_PATH=${DATASET_PATH}" \
  -e "DATASET_NAME=${DATASET_NAME}" \
  "${IMAGE_TAG}" \
  release \
  --data-root "/data_pipeline/${DATASET_NAME}" \
  --camera-visibility-gpu-id ${GPU_IDS} \
  --config config/gen_occ_example.yaml \
  --skip-voxel-plot
