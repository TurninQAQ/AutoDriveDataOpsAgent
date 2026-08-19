#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DEFAULT_TEMPLATE="$SCRIPT_DIR/../config/task_full_serial_template.yaml"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/../config/generated_task_yamls"

RECORD_DIRS=()
TASK_PREFIX=""
TEMPLATE_YAML="${TASK_TEMPLATE_YAML:-$DEFAULT_TEMPLATE}"
OUTPUT_DIR="${TASK_YAML_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
OUTPUT_YAML=""
TIMEOUT_MIN=""
DRY_RUN=0
NO_SUBMIT=0

usage() {
    cat <<'EOF'
Usage:
  start_task.sh --dataset <record_dir> [--dataset <record_dir> ...] [options]

Options:
  -d, --dataset <record_dir>   Record directory. Can be provided multiple times.
  -n, --name <name>            Task name prefix. Default: sanitized record folder name.
                               For multiple records: first record folder name.
  -y, --yaml <path>            Stable YAML template to inherit stages/images/GPU/default timeout.
                               Default: ../config/task_full_serial_template.yaml
  -o, --output-yaml <path>     YAML output path. Default: ../config/generated_task_yamls/<prefix>_<time>.yaml
      --output-dir <dir>       YAML output directory when --output-yaml is not set.
      --timeout-min <minutes>  Override timeout_min in generated YAML.
      --no-submit              Only generate YAML, do not submit to Airflow.
      --dry-run                Print commands without running them.
  -h, --help                   Show this help.

Examples:
  start_task.sh --dataset /data/record_xxx
  start_task.sh --dataset /data/record_xxx --name sq_full_cfy
  start_task.sh --dataset /data/record_001 --dataset /data/record_002 --name sq_multi_cfy
  start_task.sh --dataset /data/record_xxx --name sq_full_cfy --yaml /path/stable_task.yaml
EOF
}

sanitize_prefix() {
    local raw="$1"
    local lowered
    local cleaned
    lowered=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
    cleaned=$(printf '%s' "$lowered" \
        | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')
    if [[ -z "$cleaned" ]]; then
        cleaned="task"
    fi
    if [[ ! "$cleaned" =~ ^[a-z] ]]; then
        cleaned="task_${cleaned}"
    fi
    cleaned="${cleaned:0:32}"
    cleaned=$(printf '%s' "$cleaned" | sed -E 's/_+$//')
    if [[ -z "$cleaned" ]]; then
        cleaned="task"
    fi
    printf '%s' "$cleaned"
}

quote_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

default_task_prefix() {
    local first_base
    first_base=$(basename "${RECORD_DIRS[0]}")
    sanitize_prefix "$first_base"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "[ERROR] --dataset requires a record directory" >&2
                exit 2
            fi
            RECORD_DIRS+=("${2:-}")
            shift 2
            ;;
        -n|--name)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "[ERROR] --name requires a task name prefix" >&2
                exit 2
            fi
            TASK_PREFIX="${2:-}"
            shift 2
            ;;
        -y|--yaml)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "[ERROR] --yaml requires a template YAML path" >&2
                exit 2
            fi
            TEMPLATE_YAML="${2:-}"
            shift 2
            ;;
        -o|--output-yaml)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "[ERROR] --output-yaml requires a path" >&2
                exit 2
            fi
            OUTPUT_YAML="${2:-}"
            shift 2
            ;;
        --output-dir)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "[ERROR] --output-dir requires a directory" >&2
                exit 2
            fi
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --timeout-min)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "[ERROR] --timeout-min requires minutes" >&2
                exit 2
            fi
            TIMEOUT_MIN="${2:-}"
            shift 2
            ;;
        --no-submit)
            NO_SUBMIT=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            echo "[ERROR] Positional arguments are not supported. Use --dataset for record paths and --name for the task name." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ${#RECORD_DIRS[@]} -lt 1 ]]; then
    usage >&2
    exit 2
fi

for idx in "${!RECORD_DIRS[@]}"; do
    if [[ ! -d "${RECORD_DIRS[$idx]}" ]]; then
        echo "[ERROR] record_dir not found: ${RECORD_DIRS[$idx]}" >&2
        exit 2
    fi
    RECORD_DIRS[$idx]=$(readlink -f "${RECORD_DIRS[$idx]}")
done

TEMPLATE_YAML=$(readlink -m "$TEMPLATE_YAML")
OUTPUT_DIR=$(readlink -m "$OUTPUT_DIR")

if [[ ! -f "$TEMPLATE_YAML" ]]; then
    echo "[ERROR] template YAML not found: $TEMPLATE_YAML" >&2
    exit 2
fi

if [[ -z "$TASK_PREFIX" ]]; then
    TASK_PREFIX=$(default_task_prefix)
else
    TASK_PREFIX=$(sanitize_prefix "$TASK_PREFIX")
fi

if [[ -n "$TIMEOUT_MIN" && ! "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --timeout-min must be a positive integer: $TIMEOUT_MIN" >&2
    exit 2
fi

if [[ -z "$OUTPUT_YAML" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_YAML="$OUTPUT_DIR/${TASK_PREFIX}_$(date +%Y%m%d_%H%M%S).yaml"
else
    OUTPUT_YAML=$(readlink -m "$OUTPUT_YAML")
    mkdir -p "$(dirname "$OUTPUT_YAML")"
fi

GENERATE_CMD=("$SCRIPT_DIR/generate.py")
for record_dir in "${RECORD_DIRS[@]}"; do
    GENERATE_CMD+=("$record_dir")
done
GENERATE_CMD+=(--base-yaml "$TEMPLATE_YAML" -o "$OUTPUT_YAML")
if [[ -n "$TIMEOUT_MIN" ]]; then
    GENERATE_CMD+=(--timeout-min "$TIMEOUT_MIN")
fi

SUBMIT_CMD=(
    "$SCRIPT_DIR/manage_task.sh"
    submit
    --name "$TASK_PREFIX"
    --yaml "$OUTPUT_YAML"
)

echo "[INFO] record_dirs=${RECORD_DIRS[*]}"
echo "[INFO] name=$TASK_PREFIX"
echo "[INFO] template_yaml=$TEMPLATE_YAML"
echo "[INFO] output_yaml=$OUTPUT_YAML"
echo "[INFO] generate_cmd=$(quote_cmd "${GENERATE_CMD[@]}")"
if [[ "$NO_SUBMIT" -eq 0 ]]; then
    echo "[INFO] submit_cmd=$(quote_cmd "${SUBMIT_CMD[@]}")"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

"${GENERATE_CMD[@]}"

if [[ "$NO_SUBMIT" -eq 1 ]]; then
    echo "[INFO] YAML generated. Submit skipped by --no-submit."
    exit 0
fi

"${SUBMIT_CMD[@]}"
