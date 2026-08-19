#!/usr/bin/env bash
# airflow_ctl.sh - Airflow 3.2+ 完整启停管理脚本
# 用法: ./airflow_ctl.sh {start|stop|status}

set -euo pipefail

# ==================== 配置区 ====================
AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"
AIRFLOW_BIN="${AIRFLOW_BIN:-/home/cidi/miniforge3/envs/airflow/bin/airflow}"
AIRFLOW_PYTHON="${AIRFLOW_PYTHON:-$(dirname "$AIRFLOW_BIN")/python}"
AIRFLOW_SCRIPTS_DIR="${AIRFLOW_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)}"
PLATFORM_HOME="${PLATFORM_HOME:-$(dirname "${AIRFLOW_HOME%/}")}"
AIRFLOW_STATE_DIR="${AIRFLOW_STATE_DIR:-$PLATFORM_HOME/state}"
AIRFLOW_TASK_QUEUE_DIR="${AIRFLOW_TASK_QUEUE_DIR:-$AIRFLOW_STATE_DIR/task_queue}"
AIRFLOW_TASK_LOCK_DIR="${AIRFLOW_TASK_LOCK_DIR:-$AIRFLOW_STATE_DIR/task_locks}"
AIRFLOW_GPU_LOCK_DIR="${AIRFLOW_GPU_LOCK_DIR:-$AIRFLOW_STATE_DIR/gpu_locks}"
LOG_DIR="$AIRFLOW_HOME/logs"
API_SERVER_PORT="${AIRFLOW_PORT:-${API_SERVER_PORT:-8080}}"
AIRFLOW_PORT="$API_SERVER_PORT"
AIRFLOW_EXECUTION_API_PORT="${AIRFLOW_EXECUTION_API_PORT:-$((API_SERVER_PORT + 1))}"
AIRFLOW_EXECUTION_API_BASE="${AIRFLOW_EXECUTION_API_BASE:-http://127.0.0.1:${AIRFLOW_EXECUTION_API_PORT}}"
AIRFLOW_WORKER_LOG_SERVER_PORT="${AIRFLOW_WORKER_LOG_SERVER_PORT:-8793}"
AIRFLOW_TRIGGER_LOG_SERVER_PORT="${AIRFLOW_TRIGGER_LOG_SERVER_PORT:-8794}"
AIRFLOW_SCHEDULER_HEALTH_CHECK_SERVER_PORT="${AIRFLOW_SCHEDULER_HEALTH_CHECK_SERVER_PORT:-8974}"
AIRFLOW_PUBLIC_URL="${AIRFLOW_PUBLIC_URL:-}"
AIRFLOW_API_BASE="${AIRFLOW_API_BASE:-http://127.0.0.1:${API_SERVER_PORT}}"
AIRFLOW__API__BASE_URL="$AIRFLOW_API_BASE"
AIRFLOW__CORE__EXECUTION_API_SERVER_URL="${AIRFLOW_EXECUTION_API_BASE%/}/execution/"
HEALTH_CHECK_TIMEOUT=30
HEALTH_CHECK_INTERVAL=2

# 所有需要管理的组件列表（顺序即启动顺序）
COMPONENTS=(scheduler dag_processor triggerer api_server execution_api_server task_submit_scheduler)

export AIRFLOW_HOME
export PLATFORM_HOME AIRFLOW_STATE_DIR
export AIRFLOW_PYTHON AIRFLOW_SCRIPTS_DIR
export AIRFLOW_TASK_QUEUE_DIR AIRFLOW_TASK_LOCK_DIR AIRFLOW_GPU_LOCK_DIR
export AIRFLOW_PORT API_SERVER_PORT AIRFLOW_EXECUTION_API_PORT AIRFLOW_EXECUTION_API_BASE
export AIRFLOW_WORKER_LOG_SERVER_PORT AIRFLOW_TRIGGER_LOG_SERVER_PORT AIRFLOW_SCHEDULER_HEALTH_CHECK_SERVER_PORT
export AIRFLOW_API_BASE AIRFLOW__API__BASE_URL AIRFLOW__CORE__EXECUTION_API_SERVER_URL
export PATH="$(dirname "$AIRFLOW_BIN"):$PATH"
# ================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }

mkdir -p "$LOG_DIR" "$AIRFLOW_STATE_DIR" "$AIRFLOW_TASK_QUEUE_DIR" "$AIRFLOW_TASK_LOCK_DIR" "$AIRFLOW_GPU_LOCK_DIR"

# ---------- 核心工具函数 ----------
advertised_url() {
    if [[ -n "${AIRFLOW_PUBLIC_URL:-}" ]]; then
        printf '%s\n' "$AIRFLOW_PUBLIC_URL"
        return 0
    fi

    local ip
    ip=$(
        hostname -I 2>/dev/null \
            | tr ' ' '\n' \
            | grep -Ev '^(127\.|172\.17\.|169\.254\.|$)' \
            | head -n 1
    )
    if [[ -z "$ip" ]]; then
        ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -Ev '^(127\.|$)' | head -n 1)
    fi
    if [[ -n "$ip" ]]; then
        printf 'http://%s:%s\n' "$ip" "$API_SERVER_PORT"
    else
        printf 'http://127.0.0.1:%s\n' "$API_SERVER_PORT"
    fi
}

get_pid() {
    local component="$1"
    local pid_file="$AIRFLOW_HOME/airflow-${component}.pid"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

format_file_mtime() {
    local path="$1"
    date -r "$path" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || printf 'unknown'
}

print_platform_health() {
    local pid
    local schedule_file="$AIRFLOW_TASK_QUEUE_DIR/scheduled_submits.lock"
    local scheduler_log="$LOG_DIR/task_submit_scheduler.log"

    echo "------------------------------"
    echo " Platform Health"

    if pid=$(get_pid task_submit_scheduler 2>/dev/null); then
        echo -e "  task submit scheduler: ${GREEN}正常${NC} (PID: $pid)"
    else
        echo -e "  task submit scheduler: ${RED}异常${NC} (未运行)"
    fi

    if [[ -r "$schedule_file" ]]; then
        echo -e "  schedule file: ${GREEN}可读${NC} ($schedule_file)"
    elif [[ -e "$schedule_file" ]]; then
        echo -e "  schedule file: ${RED}不可读${NC} ($schedule_file)"
    else
        echo -e "  schedule file: ${YELLOW}未创建${NC} ($schedule_file)"
    fi

    if [[ -e "$scheduler_log" ]]; then
        echo "  latest scheduler log: $(format_file_mtime "$scheduler_log") ($scheduler_log)"
    else
        echo -e "  latest scheduler log: ${YELLOW}未创建${NC} ($scheduler_log)"
    fi
}

start_detached() {
    local log_file="$1"
    shift
    if command -v setsid >/dev/null 2>&1; then
        nohup setsid "$@" >> "$log_file" 2>&1 &
    else
        nohup "$@" >> "$log_file" 2>&1 &
    fi
    STARTED_PID=$!
}

# 通用组件启动器
# 参数: $1=组件名(下划线格式) $2=airflow子命令 $3=额外参数(可选) $4=就绪检测端口(可选,留空则用健康检查)
start_component() {
    local comp_name="$1"
    local cli_cmd="$2"
    local extra_args="${3:-}"
    local ready_port="${4:-}"
    local display_name="${comp_name//_/ }"

    if get_pid "$comp_name" > /dev/null 2>&1; then
        log_info "$display_name 已在运行 (PID: $(get_pid "$comp_name"))"
        return 0
    fi

    log_info "启动 $display_name..."
    local log_file="$LOG_DIR/${comp_name}.log"
    
    # shellcheck disable=SC2086
    start_detached "$log_file" "$AIRFLOW_BIN" "$cli_cmd" $extra_args
    local comp_pid="$STARTED_PID"

    log_info "⏳ 等待 $display_name 就绪 (PID: $comp_pid)..."
    local ready=false
    for i in $(seq 1 "$HEALTH_CHECK_TIMEOUT"); do
        sleep 1

        # 进程已退出则立即报错
        if ! kill -0 "$comp_pid" 2>/dev/null; then
            log_error "❌ $display_name 在 ${i}s 内异常退出，请检查 $log_file"
            tail -n 20 "$log_file" >&2
            return 1
        fi

        # 端口检测模式
        if [[ -n "$ready_port" ]]; then
            if ss -tlnp 2>/dev/null | grep -q ":${ready_port} "; then
                echo "$comp_pid" > "$AIRFLOW_HOME/airflow-${comp_name}.pid"
                log_info "✅ $display_name 已启动 (PID: $comp_pid, 耗时 ${i}s)"
                ready=true
                break
            fi
        else
            # 无端口检测时，只要进程存活超过 3s 即视为就绪
            if (( i >= 3 )); then
                echo "$comp_pid" > "$AIRFLOW_HOME/airflow-${comp_name}.pid"
                log_info "✅ $display_name 已启动 (PID: $comp_pid, 耗时 ${i}s)"
                ready=true
                break
            fi
        fi
    done

    if [ "$ready" = false ]; then
        if kill -0 "$comp_pid" 2>/dev/null; then
            echo "$comp_pid" > "$AIRFLOW_HOME/airflow-${comp_name}.pid"
            log_warn "⚠️ $display_name 就绪检测超时，但进程存活，已写入 PID"
        else
            log_error "❌ $display_name 启动超时且进程已退出"
            tail -n 30 "$log_file" >&2
            return 1
        fi
    fi
}

start_api_server_instance() {
    local comp_name="$1"
    local display_name="$2"
    local host="$3"
    local port="$4"
    local apps="$5"
    local health_path="$6"
    local log_file="$LOG_DIR/${comp_name}.log"
    local health_host="$host"
    if [[ "$health_host" == "0.0.0.0" ]]; then
        health_host="127.0.0.1"
    fi

    if get_pid "$comp_name" > /dev/null 2>&1; then
        log_info "$display_name 已在运行 (PID: $(get_pid "$comp_name"))"
        return 0
    fi

    log_info "启动 $display_name (host: $host, 端口: $port, apps: $apps)..."
    start_detached "$log_file" "$AIRFLOW_BIN" api-server --apps "$apps" -H "$host" -p "$port"
    local api_pid="$STARTED_PID"

    log_info "⏳ 等待 $display_name 就绪 (PID: $api_pid)..."
    local api_ready=false
    for i in $(seq 1 "$HEALTH_CHECK_TIMEOUT"); do
        sleep 1
        if ! kill -0 "$api_pid" 2>/dev/null; then
            log_error "❌ $display_name 在 ${i}s 内异常退出"
            tail -n 20 "$log_file" >&2
            return 1
        fi
        if curl -sf "http://${health_host}:${port}${health_path}" > /dev/null 2>&1; then
            echo "$api_pid" > "$AIRFLOW_HOME/airflow-${comp_name}.pid"
            log_info "✅ $display_name 已启动 (PID: $api_pid, 耗时 ${i}s)"
            api_ready=true
            break
        fi
    done

    if [ "$api_ready" = false ] && kill -0 "$api_pid" 2>/dev/null; then
        echo "$api_pid" > "$AIRFLOW_HOME/airflow-${comp_name}.pid"
        log_warn "⚠️ $display_name 健康检查超时，但进程存活，已写入 PID"
    elif [ "$api_ready" = false ]; then
        log_error "❌ $display_name 启动失败"
        return 1
    fi
}

start_task_submit_scheduler() {
    local comp_name="task_submit_scheduler"
    local display_name="task submit scheduler"
    local log_file="$LOG_DIR/${comp_name}.log"

    if get_pid "$comp_name" > /dev/null 2>&1; then
        log_info "$display_name 已在运行 (PID: $(get_pid "$comp_name"))"
        return 0
    fi

    log_info "启动 $display_name..."
    start_detached "$log_file" "$AIRFLOW_PYTHON" "$AIRFLOW_SCRIPTS_DIR/task_manager.py" submit scheduler
    local comp_pid="$STARTED_PID"

    sleep 2
    if kill -0 "$comp_pid" 2>/dev/null; then
        echo "$comp_pid" > "$AIRFLOW_HOME/airflow-${comp_name}.pid"
        log_info "✅ $display_name 已启动 (PID: $comp_pid)"
    else
        log_error "❌ $display_name 启动失败，请检查 $log_file"
        tail -n 20 "$log_file" >&2
        return 1
    fi
}

validate_runtime_api_config() {
    local expected_execution_url="${AIRFLOW_EXECUTION_API_BASE%/}/execution/"
    local configured_execution_url=""
    local configured_api_port=""

    if [[ ! -f "$AIRFLOW_HOME/airflow.cfg" ]]; then
        log_error "❌ airflow.cfg 不存在: $AIRFLOW_HOME/airflow.cfg"
        return 1
    fi

    configured_execution_url=$(
        "$AIRFLOW_BIN" config get-value core execution_api_server_url 2>/dev/null || true
    )
    configured_api_port=$(
        "$AIRFLOW_BIN" config get-value api port 2>/dev/null || true
    )

    if [[ "$configured_execution_url" != "$expected_execution_url" ]]; then
        log_error "❌ execution_api_server_url 配置错误: 当前=$configured_execution_url 期望=$expected_execution_url"
        log_error "请重新执行源码目录 ./platform install 生成 runtime 配置"
        return 1
    fi
    if [[ "$configured_api_port" != "$API_SERVER_PORT" ]]; then
        log_error "❌ API Server 端口配置错误: 当前=$configured_api_port 期望=$API_SERVER_PORT"
        log_error "请重新执行源码目录 ./platform install 生成 runtime 配置"
        return 1
    fi
}

# ---------- 命令实现 ----------
do_start() {
    validate_runtime_api_config
    log_info "🚀 启动 Airflow 3.2 全组件..."

    # 按依赖顺序启动各组件
    # 1. Public UI/Core API Server
    start_api_server_instance "api_server" "API Server" "0.0.0.0" "$API_SERVER_PORT" "core" "/public/health"

    # 2. Internal Execution API Server
    start_api_server_instance "execution_api_server" "Execution API Server" "127.0.0.1" "$AIRFLOW_EXECUTION_API_PORT" "execution" "/execution/health"

    # 3. Scheduler
    start_component "scheduler" "scheduler" "" "$AIRFLOW_WORKER_LOG_SERVER_PORT"

    # 4. DAG Processor (无独立端口，进程存活即就绪)
    start_component "dag_processor" "dag-processor" "" ""

    # 5. Triggerer (无独立端口，进程存活即就绪)
    start_component "triggerer" "triggerer" "" ""

    # 6. Task submit scheduler (定时提交任务扫描器)
    start_task_submit_scheduler

    log_info "🎉 Airflow 3.2 全组件启动完成 | 访问地址: $(advertised_url)"
}

do_stop() {
    log_info "🛑 停止 Airflow 3.2 全组件..."
    local killed=0

    # 1. 按当前 runtime 的 PID 文件停止组件，避免误杀同机其他 Airflow 实例。
    for component in "${COMPONENTS[@]}"; do
        local pid_file="$AIRFLOW_HOME/airflow-${component}.pid"
        local pid=""
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file" 2>/dev/null || true)
        fi
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log_info "停止 ${component//_/ } (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            for _ in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "${component//_/ } 未正常退出，强制结束 (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
            fi
            killed=1
        fi
    done

    # 2. 只清理当前 runtime 的 API 端口，防止残留 API Server 占用 0731 端口。
    local managed_ports=(
        "$API_SERVER_PORT"
        "$AIRFLOW_EXECUTION_API_PORT"
        "$AIRFLOW_WORKER_LOG_SERVER_PORT"
        "$AIRFLOW_TRIGGER_LOG_SERVER_PORT"
        "$AIRFLOW_SCHEDULER_HEALTH_CHECK_SERVER_PORT"
    )
    for port in "${managed_ports[@]}"; do
        if command -v lsof &> /dev/null; then
            local api_pids
            api_pids=$(lsof -ti :"$port" 2>/dev/null || true)
            if [[ -n "$api_pids" ]]; then
                log_info "发现占用端口 $port 的进程: $api_pids"
                echo "$api_pids" | xargs kill -9 2>/dev/null || true
                killed=1
            fi
        elif command -v fuser &> /dev/null; then
            if fuser -k -9 "$port"/tcp 2>/dev/null; then
                killed=1
            fi
        else
            log_warn "⚠️ 未安装 lsof/fuser，跳过端口反查"
            break
        fi
    done

    # 3. 验证当前 runtime 记录的 PID 已不存在
    sleep 2
    local remaining=""
    for component in "${COMPONENTS[@]}"; do
        local pid_file="$AIRFLOW_HOME/airflow-${component}.pid"
        if [[ -f "$pid_file" ]]; then
            local pid
            pid=$(cat "$pid_file" 2>/dev/null || true)
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                remaining+="${pid}"$'\n'
            fi
        fi
    done
    remaining=$(printf '%s' "$remaining" | sort -u | sed '/^$/d')
    if [[ -z "$remaining" ]]; then
        log_info "✅ Airflow 所有进程已彻底清除"
    else
        log_error "❌ 仍有残留进程: $remaining"
    fi

    # 4. 清理当前 runtime 的残留 PID 文件
    rm -f "$AIRFLOW_HOME"/airflow-*.pid 2>/dev/null || true
    log_info "已清理所有残留 PID 文件"

    if (( killed == 0 )); then
        log_warn "未发现任何运行中的 Airflow 进程"
    fi
}

do_status() {
    validate_runtime_api_config || true
    echo "=============================="
    echo " Airflow 3.2 服务状态"
    echo " AIRFLOW_HOME: $AIRFLOW_HOME"
    echo " 访问地址: $(advertised_url)"
    echo "=============================="

    for component in "${COMPONENTS[@]}"; do
        local display_name="${component//_/ }"
        if pid=$(get_pid "$component" 2>/dev/null); then
            echo -e "  $display_name: ${GREEN}运行中${NC} (PID: $pid)"
        else
            echo -e "  $display_name: ${RED}已停止${NC}"
        fi
    done

    # API Server 健康检查
    echo "------------------------------"
    if get_pid api_server > /dev/null 2>&1; then
        if curl -sf "http://127.0.0.1:${API_SERVER_PORT}/public/health" > /dev/null 2>&1; then
            echo -e "  Public API Health: ${GREEN}正常${NC}"
        else
            echo -e "  Public API Health: ${RED}无响应${NC} (进程存活但接口未就绪)"
        fi
    else
        echo -e "  Public API Health: ${YELLOW}跳过${NC} (API Server 未运行)"
    fi
    if get_pid execution_api_server > /dev/null 2>&1; then
        if curl -sf "http://127.0.0.1:${AIRFLOW_EXECUTION_API_PORT}/execution/health" > /dev/null 2>&1; then
            echo -e "  Execution API Health: ${GREEN}正常${NC} (127.0.0.1:${AIRFLOW_EXECUTION_API_PORT})"
        else
            echo -e "  Execution API Health: ${RED}无响应${NC} (进程存活但接口未就绪)"
        fi
    else
        echo -e "  Execution API Health: ${YELLOW}跳过${NC} (Execution API Server 未运行)"
    fi
    print_platform_health
    echo "=============================="

    local health_json
    if health_json=$(curl -sf "http://localhost:${API_SERVER_PORT}/api/v2/monitor/health" 2>/dev/null); then
        printf '%s\n' "$health_json" | python3 -m json.tool || printf '%s\n' "$health_json"
    else
        log_warn "Health API JSON 不可用，跳过详细健康信息"
    fi

    echo "=============================="
}

# ---------- 入口 ----------
case "${1:-}" in
    start)   do_start ;;
    stop)    do_stop ;;
    status)  do_status ;;
    *)
        echo "用法: $0 {start|stop|status}"
        exit 1
        ;;
esac
