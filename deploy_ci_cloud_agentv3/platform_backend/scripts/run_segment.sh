#!/usr/bin/env bash
set -euo pipefail

if [ "${PLATFORM_STAGE_RUNTIME:-real}" = "mock" ]; then
    exec bash "$(dirname "$0")/run_mock_stage.sh" "segment"
fi


: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"
: "${GPU_IDS:?ENV MISSING}"
: "${CONTAINER_NAME:?ENV MISSING}"

DEFAULT_CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR:-/home/cfy/sam3___1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/sam3.1_multiplex.pt"
CONTAINER_CHECKPOINT_FILE="/app/checkpoints/sam3.1_multiplex.pt"
SHM_SIZE="${SHM_SIZE:-16g}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
SAM31_USE_FA3="${SAM31_USE_FA3:-1}"
read -r -a DOCKER <<< "${DOCKER_CMD}"

[ -d "${DATASET_PATH}" ] || {
    echo "[ERROR] DATASET_PATH not found: ${DATASET_PATH}" >&2
    exit 1
}
[ -d "${DATA_DIR}" ] || {
    echo "[ERROR] DATA_DIR not found: ${DATA_DIR}" >&2
    exit 1
}
[ -d "${CHECKPOINT_DIR}" ] || {
    echo "[ERROR] CHECKPOINT_DIR not found: ${CHECKPOINT_DIR}" >&2
    exit 1
}
[ -f "${CHECKPOINT_FILE}" ] || {
    echo "[ERROR] SAM3.1 checkpoint not found: ${CHECKPOINT_FILE}" >&2
    exit 1
}

DATASET_PATH="$(cd "${DATASET_PATH}" && pwd)"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/sam3.1_multiplex.pt"

WORK_DIR="${DATA_DIR}/.sam31_work"
DATA_MOUNT_ROOT="/data_pipeline"
CONTAINER_DATA_DIR="${DATA_MOUNT_ROOT}/${DATASET_NAME}"
CONTAINER_WORK_DIR="/app/workspace/sam3.1/work"
RESULT_FILE="${CONTAINER_DATA_DIR}/results_segment.json"

GPU_ARGS=()
if [ -n "${GPU_IDS}" ] && [[ "${DOCKER_CMD}" != *"nvidia-docker"* ]]; then
    GPU_REQUEST="${GPU_IDS}"
    if [[ "${GPU_REQUEST}" != "all" && "${GPU_REQUEST}" != device=* ]]; then
        GPU_REQUEST="device=${GPU_REQUEST}"
    fi
    if [[ "${GPU_REQUEST}" == device=*,* ]]; then
        GPU_REQUEST="\"${GPU_REQUEST}\""
    fi
    GPU_ARGS=(--gpus "${GPU_REQUEST}")
fi

cleanup_work_dir() {
    [ -d "${WORK_DIR}" ] || return 0
    if rm -rf "${WORK_DIR}" 2>/dev/null; then
        return 0
    fi

    echo "[WARN] Normal cleanup failed, retrying with docker: ${WORK_DIR}" >&2
    if "${DOCKER[@]}" run --rm \
        --entrypoint bash \
        -v "${WORK_DIR}:/cleanup_target" \
        "${IMAGE_TAG}" \
        -lc 'find /cleanup_target -mindepth 1 -exec rm -rf {} +'; then
        rmdir "${WORK_DIR}" 2>/dev/null || true
    else
        echo "[WARN] Failed to cleanup WORK_DIR: ${WORK_DIR}" >&2
    fi
}
trap cleanup_work_dir EXIT

mkdir -p "${WORK_DIR}"

echo "[INFO] Stage SEGMENT | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"
echo "[INFO] Image: ${IMAGE_TAG}"
echo "[INFO] GPUs: ${GPU_IDS}"
echo "[INFO] Checkpoint dir: ${CHECKPOINT_DIR}"
echo "[INFO] Checkpoint file: ${CHECKPOINT_FILE}"

"${DOCKER[@]}" run --rm \
    --name "${CONTAINER_NAME}" \
    "${GPU_ARGS[@]}" \
    --ipc=host \
    --shm-size "${SHM_SIZE}" \
    -v "${CHECKPOINT_DIR}:/app/checkpoints:ro" \
    -v "${DATASET_PATH}:${DATA_MOUNT_ROOT}" \
    -v "${WORK_DIR}:${CONTAINER_WORK_DIR}" \
    -e "SAM31_WORK_DIR=${CONTAINER_WORK_DIR}" \
    -e "DATASET_PATH=${CONTAINER_DATA_DIR}" \
    -e "DATASET_NAME=${DATASET_NAME}" \
    -e "SAM31_DATASET_DIR=${CONTAINER_DATA_DIR}" \
    -e "RESULT_FILE=${RESULT_FILE}" \
    -e SAM31_USE_FA3="${SAM31_USE_FA3}" \
    "${IMAGE_TAG}" \
    release \
    batch \
    --checkpoint_path "${CONTAINER_CHECKPOINT_FILE}"
