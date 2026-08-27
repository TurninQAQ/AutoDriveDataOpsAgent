#!/bin/bash
set -euo pipefail

if [ "${PLATFORM_STAGE_RUNTIME:-real}" = "mock" ]; then
    exec bash "$(dirname "$0")/run_mock_stage.sh" "od"
fi


: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"
: "${IMAGE_TAG:?ENV MISSING}"
: "${GPU_IDS:?ENV MISSING}"
: "${CONTAINER_NAME:?ENV MISSING}"


echo "[INFO] Stage OD | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"
# 如果想推理单个/多个指定的clip，可以在pipeline_config.yaml中指定clip列表:
DATA_DIR=${DATASET_PATH}
DOCKER_DATA_DIR="/workspace/data/input_data"
SOURCE_DATA_DIR="$DATA_DIR/$DATASET_NAME"
# 如果需要删除临时目录，可以在pipeline_config.yaml中指定delete_temp_dir: true (实验性功能，不建议使用，因为会删除所有中间结果，以及存在误删风险)
TEMP_DIR="${DATASET_PATH}/${DATASET_NAME}/temp_data"
DOCKER_TEMP_DIR="/workspace/data/temp"
# 外部模型文件和配置文件目录：
MODEL_DIR="${OD_MODEL_DIR:-${MODEL_DIR:-/home/cidi/data/nty/projects/models}}"
DOCKER_MODEL_DIR="/workspace/data/models"
# run_result.json 文件路径
RUN_STATUS_FILE_PATH="$DOCKER_DATA_DIR/$DATASET_NAME/results_od.json"

docker run --rm  \
    --name "${CONTAINER_NAME}" \
    --gpus device=$GPU_IDS  \
    --shm-size=16g  \
    -v $DATA_DIR:$DOCKER_DATA_DIR  \
    -v $TEMP_DIR:$DOCKER_TEMP_DIR  \
    -v $MODEL_DIR:$DOCKER_MODEL_DIR  \
    -e SOURCE_DATA_DIR="$SOURCE_DATA_DIR" \
    "${IMAGE_TAG}"  \
    debug  \
    --data_dir="$DOCKER_DATA_DIR"  \
    --clip_name="$DATASET_NAME" \
    --temp_dir="$DOCKER_TEMP_DIR"  \
    --model_dir="$DOCKER_MODEL_DIR"  \
    --output_dir="$DOCKER_DATA_DIR"  \
    --run_status_file_path="$RUN_STATUS_FILE_PATH"  \
    --set "del_temp_files=true"

#rm -rf $TEMP_DIR
