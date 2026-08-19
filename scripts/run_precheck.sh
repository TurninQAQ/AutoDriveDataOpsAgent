#!/usr/bin/env bash
set -uo pipefail

if [ "${PLATFORM_STAGE_RUNTIME:-real}" = "mock" ]; then
    exec bash "$(dirname "$0")/run_mock_stage.sh" "precheck"
fi


: "${DATASET_PATH:?ENV MISSING}"
: "${DATASET_NAME:?ENV MISSING}"
: "${DATA_DIR:?ENV MISSING}"

echo "[INFO] Stage PRECHECK | dataset=${DATASET_NAME} | data_dir=${DATA_DIR}"

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m'

CLIP_PATH="$DATA_DIR"
CALIB_PATH="${CLIP_PATH}/calibration"
RESULT_FILE="${CLIP_PATH}/results_precheck.json"
PYTHON_BIN="${AIRFLOW_PYTHON:-python3}"

TMP_DIR=$(mktemp -d "${CLIP_PATH}/.precheck_tmp.XXXXXX" 2>/dev/null || mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

CALIB_MISSING_FILE="${TMP_DIR}/calib_missing.txt"
TOPIC_MISSING_FILE="${TMP_DIR}/topic_missing.txt"
INVALID_COUNTS_FILE="${TMP_DIR}/invalid_counts.txt"
TOPIC_COUNTS_FILE="${TMP_DIR}/topic_counts.tsv"

: > "$CALIB_MISSING_FILE"
: > "$TOPIC_MISSING_FILE"
: > "$INVALID_COUNTS_FILE"
: > "$TOPIC_COUNTS_FILE"

REQUIRED_FILES=(
    "camera/camera_info.pb.txt"
    "camera/extrinsic/front_normal_extrinsic.yaml"
    "camera/extrinsic/front_wide_extrinsic.yaml"
    "camera/extrinsic/left_back_extrinsic.yaml"
    "camera/extrinsic/left_front_extrinsic.yaml"
    "camera/extrinsic/right_back_extrinsic.yaml"
    "camera/extrinsic/right_front_extrinsic.yaml"
    "camera/intrinsic/front_normal_intrinsic.yaml"
    "camera/intrinsic/front_wide_intrinsic.yaml"
    "camera/intrinsic/left_back_intrinsic.yaml"
    "camera/intrinsic/left_front_intrinsic.yaml"
    "camera/intrinsic/right_back_intrinsic.yaml"
    "camera/intrinsic/right_front_intrinsic.yaml"
    "imu/imu2car.yaml"
    "lidar/extrinsic/helios32_left_novatel_extrinsics.yaml"
    "lidar/extrinsic/helios32_right_novatel_extrinsics.yaml"
    "lidar/extrinsic/hesai_back_novatel_extrinsics.yaml"
    "lidar/extrinsic/hesai_front_novatel_extrinsics.yaml"
    "lidar/extrinsic/rsem4_top_novatel_extrinsics.yaml"
)

CHECK_TOPICS=(
    "/lidar/right_raw_data"
    "/lidar/left_raw_data"
    "/camera/compress/back_normal"
    "/camera/compress/front_wide"
    "/camera/compress/left_front"
    "/camera/compress/right_back"
    "/camera/compress/front_normal"
    "/camera/compress/left_back"
    "/camera/compress/right_front"
)

MIN_COUNT=200
MAX_COUNT=350
MAX_DIFF=25

write_result() {
    local status="$1"
    local error_message="$2"
    local reason="$3"

    STATUS="$status" \
    ERROR_MESSAGE="$error_message" \
    REASON="$reason" \
    RESULT_FILE="$RESULT_FILE" \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_PATH="$DATASET_PATH" \
    DATA_DIR="$DATA_DIR" \
    METADATA_FILE="${METADATA_FILE:-}" \
    CALIB_REQUIRED="${#REQUIRED_FILES[@]}" \
    CALIB_FOUND="${calib_found:-0}" \
    CALIB_MISSING_FILE="$CALIB_MISSING_FILE" \
    TOPIC_MISSING_FILE="$TOPIC_MISSING_FILE" \
    INVALID_COUNTS_FILE="$INVALID_COUNTS_FILE" \
    TOPIC_COUNTS_FILE="$TOPIC_COUNTS_FILE" \
    MIN_COUNT="$MIN_COUNT" \
    MAX_COUNT="$MAX_COUNT" \
    MAX_DIFF="$MAX_DIFF" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def read_lines(path):
    p = Path(path)
    if not p.exists():
        return []
    return [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_topic_counts(path):
    result = {}
    p = Path(path)
    if not p.exists():
        return result
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        topic, count = line.split("\t", 1)
        try:
            result[topic] = int(count)
        except ValueError:
            result[topic] = count
    return result


payload = {
    "dataset_path": os.environ["DATA_DIR"],
    "record_path": os.environ["DATASET_PATH"],
    "dataset_name": os.environ["DATASET_NAME"],
    "status": os.environ["STATUS"],
    "reason": os.environ["REASON"],
    "error_message": os.environ["ERROR_MESSAGE"],
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "checks": {
        "calibration": {
            "required": int(os.environ["CALIB_REQUIRED"]),
            "found": int(os.environ["CALIB_FOUND"]),
            "missing": read_lines(os.environ["CALIB_MISSING_FILE"]),
        },
        "metadata": {
            "file": os.environ.get("METADATA_FILE") or "",
            "min_count": int(os.environ["MIN_COUNT"]),
            "max_count": int(os.environ["MAX_COUNT"]),
            "max_diff": int(os.environ["MAX_DIFF"]),
            "missing_topics": read_lines(os.environ["TOPIC_MISSING_FILE"]),
            "invalid_counts": read_lines(os.environ["INVALID_COUNTS_FILE"]),
            "topic_counts": read_topic_counts(os.environ["TOPIC_COUNTS_FILE"]),
        },
    },
}

result_file = Path(os.environ["RESULT_FILE"])
result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[INFO] Wrote result JSON: {result_file}", flush=True)
PY
}

echo "=========================================="
echo " Clip 完整性检查"
echo " 目标路径: $CLIP_PATH"
echo "=========================================="

calib_pass=true
calib_found=0
calib_missing=0

echo ""
echo -e "${YELLOW}[1/2] 标定文件检查${NC}"
echo "------------------------------------------"

if [ ! -d "$CALIB_PATH" ]; then
    echo -e "${RED}✘ calibration 目录不存在: $CALIB_PATH${NC}"
    calib_pass=false
    calib_missing=${#REQUIRED_FILES[@]}
    printf "%s\n" "${REQUIRED_FILES[@]}" >> "$CALIB_MISSING_FILE"
else
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "${CALIB_PATH}/${file}" ]; then
            echo -e "  [${GREEN}✓${NC}] $file"
            calib_found=$((calib_found + 1))
        else
            echo -e "  [${RED}✗${NC}] $file  ${RED}-- 缺失${NC}"
            calib_missing=$((calib_missing + 1))
            echo "$file" >> "$CALIB_MISSING_FILE"
        fi
    done

    echo "------------------------------------------"
    echo " 标定文件: 总计 ${#REQUIRED_FILES[@]}  存在: ${calib_found}  缺失: ${calib_missing}"

    if [ "$calib_missing" -ne 0 ]; then
        calib_pass=false
    fi
fi

echo ""
echo -e "${YELLOW}[2/2] metadata.yaml 检查${NC}"
echo "------------------------------------------"

METADATA_FILE="${CLIP_PATH}/metadata.yaml"
if [ ! -f "$METADATA_FILE" ]; then
    METADATA_FILE="${CLIP_PATH}/raw_bag/metadata.yaml"
    if [ ! -f "$METADATA_FILE" ]; then
        echo -e "${RED}✘ metadata.yaml 不存在 (根目录和 raw_bag 均未找到)${NC}"
        printf "%s\n" "${CHECK_TOPICS[@]}" >> "$TOPIC_MISSING_FILE"
        METADATA_FILE=""
        meta_pass=false
    else
        echo -e "${YELLOW}⚠ metadata.yaml 在 raw_bag 中找到${NC}"
        meta_pass=true
    fi
else
    meta_pass=true
fi

declare -A topic_counts
counts=()
if $meta_pass; then
    current_topic=""
    while IFS= read -r line; do
        if [[ "$line" =~ name:\ (.+) ]]; then
            current_topic="${BASH_REMATCH[1]}"
            current_topic=$(echo "$current_topic" | xargs)
        fi
        if [[ -n "$current_topic" && "$line" =~ ^[[:space:]]+message_count:\ ([0-9]+) ]]; then
            topic_counts["$current_topic"]="${BASH_REMATCH[1]}"
            current_topic=""
        fi
    done < "$METADATA_FILE"

    for topic in "${CHECK_TOPICS[@]}"; do
        count="${topic_counts[$topic]:-}"
        if [ -z "$count" ]; then
            echo "$topic" >> "$TOPIC_MISSING_FILE"
            meta_pass=false
            continue
        fi

        printf "%s\t%s\n" "$topic" "$count" >> "$TOPIC_COUNTS_FILE"
        echo -e "  [${GREEN}✓${NC}] $topic: $count 条消息"

        if [ "$count" -lt "$MIN_COUNT" ] || [ "$count" -gt "$MAX_COUNT" ]; then
            echo "$topic: $count (应为 $MIN_COUNT~$MAX_COUNT)" >> "$INVALID_COUNTS_FILE"
            meta_pass=false
        else
            counts+=("$count")
        fi
    done

    if [ ${#counts[@]} -ge 2 ]; then
        min_val=$(printf "%s\n" "${counts[@]}" | sort -n | head -1)
        max_val=$(printf "%s\n" "${counts[@]}" | sort -n | tail -1)
        diff=$((max_val - min_val))

        if [ "$diff" -gt "$MAX_DIFF" ]; then
            echo "各topic数量差异过大 ($min_val~$max_val, 差距=$diff > $MAX_DIFF)" >> "$INVALID_COUNTS_FILE"
            meta_pass=false
        fi
    fi
fi

echo ""
echo "=========================================="
echo " 汇总报告"
echo "=========================================="

if $meta_pass && $calib_pass; then
    echo -e "${GREEN}🎉 所有检查通过！${NC}"
    write_result "success" "" ""
    exit 0
fi

error_message="precheck failed"
if [ "$calib_pass" = false ]; then
    error_message="${error_message}; 标定文件缺失: ${calib_missing} 个"
fi
if [ "$meta_pass" = false ]; then
    error_message="${error_message}; metadata/topic 检查未通过"
fi

echo -e "${RED}❌ 检查未通过${NC}"
echo "------------------------------------------"
if [ -s "$TOPIC_MISSING_FILE" ]; then
    echo -e "${RED}缺失 topics:${NC}"
    sed 's/^/  - /' "$TOPIC_MISSING_FILE"
fi
if [ -s "$INVALID_COUNTS_FILE" ]; then
    echo -e "${RED}数量异常:${NC}"
    sed 's/^/  - /' "$INVALID_COUNTS_FILE"
fi
if [ "$calib_pass" = false ]; then
    echo -e "${RED}标定文件缺失: ${calib_missing} 个${NC}"
fi

write_result "failed" "$error_message" "$error_message"
exit 1
