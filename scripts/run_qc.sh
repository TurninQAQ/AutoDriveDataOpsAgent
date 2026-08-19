#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"

echo "[INFO] Stage PARSER | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
    -v "${DATASET_PATH}":/data_pipeline \
    "${IMAGE_TAG}" \
    release \
    "/data_pipeline/${DATASET_NAME}"