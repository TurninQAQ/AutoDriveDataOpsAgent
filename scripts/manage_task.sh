#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"
AIRFLOW_PYTHON="${AIRFLOW_PYTHON:-/home/cidi/miniforge3/envs/airflow/bin/python}"
PLATFORM_HOME="${PLATFORM_HOME:-$(dirname "${AIRFLOW_HOME%/}")}"
AIRFLOW_STATE_DIR="${AIRFLOW_STATE_DIR:-$PLATFORM_HOME/state}"
AIRFLOW_TASK_QUEUE_DIR="${AIRFLOW_TASK_QUEUE_DIR:-$AIRFLOW_STATE_DIR/task_queue}"
AIRFLOW_TASK_LOCK_DIR="${AIRFLOW_TASK_LOCK_DIR:-$AIRFLOW_STATE_DIR/task_locks}"
AIRFLOW_GPU_LOCK_DIR="${AIRFLOW_GPU_LOCK_DIR:-$AIRFLOW_STATE_DIR/gpu_locks}"
TASK_COMMAND="${AIRFLOW_TASK_COMMAND:-manage_task.sh}"
export AIRFLOW_HOME AIRFLOW_PYTHON PLATFORM_HOME AIRFLOW_STATE_DIR
export AIRFLOW_TASK_QUEUE_DIR AIRFLOW_TASK_LOCK_DIR AIRFLOW_GPU_LOCK_DIR

usage() {
    cat <<EOF
任务命令 ${TASK_COMMAND}

提交任务:
  ${TASK_COMMAND} submit --name <任务名前缀> --yaml <任务yaml>
  ${TASK_COMMAND} submit --name <任务名前缀> --yaml <任务yaml> --schedule "YYYY-MM-DD HH:MM"

查看/取消定时任务:
  ${TASK_COMMAND} schedule list
  ${TASK_COMMAND} schedule list --all
  ${TASK_COMMAND} schedule list --status scheduled
  ${TASK_COMMAND} schedule remove <schedule_id> --yes

管理任务:
  ${TASK_COMMAND} stop <完整任务名> --yes
  ${TASK_COMMAND} stop <完整任务名> <clip_1> <clip_2> --yes
  ${TASK_COMMAND} resume <完整任务名>
  ${TASK_COMMAND} resume <完整任务名> <clip_1> <clip_2>
  ${TASK_COMMAND} priority <完整任务名> --priority <数字>
  ${TASK_COMMAND} delete <完整任务名> --yes

危险提示:
  ${TASK_COMMAND} delete <完整任务名> --yes 会删除 generated DAG、任务配置和 Airflow 元数据，并停止匹配任务容器。
  ${TASK_COMMAND} stop <完整任务名> --yes 会标记运行中 DagRun 失败，并默认停止匹配任务容器。

常用示例:
  ${TASK_COMMAND} submit --name reprocess --yaml /tmp/reprocess.yaml
  ${TASK_COMMAND} submit --name reprocess --yaml /tmp/reprocess.yaml --schedule "2026-07-31 23:00"
  ${TASK_COMMAND} schedule list
  ${TASK_COMMAND} schedule remove sched_xxx --yes
  ${TASK_COMMAND} priority reprocess_20260731_153000 --priority 5
  ${TASK_COMMAND} delete reprocess_20260731_153000 --yes

帮助:
  ${TASK_COMMAND} --help
  ${TASK_COMMAND} submit --help
  ${TASK_COMMAND} schedule --help
EOF
}

if [ "$#" -lt 1 ]; then
    usage >&2
    exit 2
fi

case "$1" in
    submit|schedule|stop|resume|priority|delete)
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "[ERROR] Unsupported action: $1" >&2
        usage >&2
        exit 2
        ;;
esac

exec "${AIRFLOW_PYTHON}" "${SCRIPT_DIR}/task_manager.py" "$@"
