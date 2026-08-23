#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"

RESULT_FILE="${DATA_DIR}/results_tail.json"
PYTHON_BIN="${AIRFLOW_PYTHON:-python3}"

echo "[INFO] Stage TAIL | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

RESULT_FILE="$RESULT_FILE" \
DATASET_NAME="$DATASET_NAME" \
DATASET_PATH="$DATASET_PATH" \
DATA_DIR="$DATA_DIR" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "dataset_path": os.environ["DATA_DIR"],
    "record_path": os.environ["DATASET_PATH"],
    "dataset_name": os.environ["DATASET_NAME"],
    "status": "success",
    "reason": "priority_tail",
    "error_message": "",
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

result_file = Path(os.environ["RESULT_FILE"])
result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
result_file.chmod(0o666)
print(f"[INFO] Wrote result JSON: {result_file}", flush=True)
PY
