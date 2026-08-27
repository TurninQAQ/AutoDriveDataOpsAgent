#!/usr/bin/env bash
set -euo pipefail

if [ "${PLATFORM_STAGE_RUNTIME:-real}" = "mock" ]; then
    exec bash "$(dirname "$0")/run_mock_stage.sh" "coloration"
fi


: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"
: "${CONTAINER_NAME:?ENV MISSING}"

echo "[INFO] Stage COLORATION | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

docker run --rm \
    --name "${CONTAINER_NAME}" \
    -v "${DATASET_PATH}":/data_pipeline \
    "${IMAGE_TAG}" \
    release \
    "/data_pipeline/${DATASET_NAME}"
