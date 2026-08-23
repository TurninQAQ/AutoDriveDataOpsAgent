#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?stage required}"
: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PYTHON_BIN="${AIRFLOW_PYTHON:-python3}"
STAGE_KEY=$(printf '%s' "$STAGE" | tr '[:lower:]-' '[:upper:]_')
RESULT_VAR="MOCK_STAGE_RESULT_${STAGE_KEY}"
DURATION_VAR="MOCK_STAGE_DURATION_SEC_${STAGE_KEY}"
RESULT="${!RESULT_VAR:-${MOCK_STAGE_RESULT:-success}}"
DURATION="${!DURATION_VAR:-${MOCK_STAGE_DURATION_SEC:-0}}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/mock_stage.py" \
  --stage "$STAGE" \
  --dataset-path "$DATASET_PATH" \
  --dataset-name "$DATASET_NAME" \
  --duration-sec "$DURATION" \
  --result "$RESULT"
