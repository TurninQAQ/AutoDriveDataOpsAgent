#!/usr/bin/env bash
set -euo pipefail
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"

echo "[INFO] Cleaning up data dir: ${DATA_DIR}"
rm -rf "${DATA_DIR}"