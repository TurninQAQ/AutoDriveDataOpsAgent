#!/usr/bin/env bash
set -euo pipefail

LATEST_VERSION="agent-v1.2.0  2026-08-19:(new:gemini_provider_embedding base:agent-v1.1.0)"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
DAGS_DIR="${AIRFLOW_DAGS_DIR:-/home/cidi/airflow/dags/data_center}"
HOST_DATA_ROOT="${AIRFLOW_HOST_DATA_ROOT:-/opt/airflow/data}"
CONFIG_DIR="${AIRFLOW_CONFIG_DIR:-/opt/airflow/config}"
SCRIPTS_DIR="${AIRFLOW_SCRIPTS_DIR:-/opt/airflow/scripts}"
TOOLS_DIR="$SCRIPTS_DIR/tools"
PLATFORM_CORE_DIR="${AIRFLOW_PLATFORM_CORE_DIR:-$(dirname "$SCRIPTS_DIR")/platform_core}"
PLATFORM_MCP_DIR="${AIRFLOW_PLATFORM_MCP_DIR:-$(dirname "$SCRIPTS_DIR")/platform_mcp}"
PLATFORM_AGENT_DIR="${AIRFLOW_PLATFORM_AGENT_DIR:-$(dirname "$SCRIPTS_DIR")/platform_agent}"
PLATFORM_PLANNING_DIR="${AIRFLOW_PLATFORM_PLANNING_DIR:-$(dirname "$SCRIPTS_DIR")/platform_planning}"
PLATFORM_RAG_DIR="${AIRFLOW_PLATFORM_RAG_DIR:-$(dirname "$SCRIPTS_DIR")/platform_rag}"
PLATFORM_OBSERVABILITY_DIR="${AIRFLOW_PLATFORM_OBSERVABILITY_DIR:-$(dirname "$SCRIPTS_DIR")/platform_observability}"
PLATFORM_EVAL_DIR="${AIRFLOW_PLATFORM_EVAL_DIR:-$(dirname "$SCRIPTS_DIR")/platform_eval}"
PLATFORM_HARDENING_DIR="${AIRFLOW_PLATFORM_HARDENING_DIR:-$(dirname "$SCRIPTS_DIR")/platform_hardening}"
KNOWLEDGE_DIR="${AIRFLOW_KNOWLEDGE_DIR:-$(dirname "$SCRIPTS_DIR")/knowledge}"
AGENT_EVAL_DIR="${AIRFLOW_AGENT_EVAL_DIR:-$(dirname "$SCRIPTS_DIR")/eval}"
TASK_CONFIG_ROOT="${AIRFLOW_TASK_CONFIG_ROOT:-$CONFIG_DIR/tasks}"
GENERATED_DAGS_DIR="$DAGS_DIR/generated"
TEMPLATE_DAGS_DIR="$DAGS_DIR/templates"
AIRFLOW_BIN="${AIRFLOW_BIN:-/home/cidi/miniforge3/envs/airflow/bin/airflow}"
AIRFLOW_HOME="${AIRFLOW_HOME:-/home/cidi/airflow}"
AIRFLOW_LOCAL_SETTINGS_DIR="$AIRFLOW_HOME/config"
AIRFLOW_PORT="${AIRFLOW_PORT:-8080}"
export AIRFLOW_HOME
export PATH="$(dirname "$AIRFLOW_BIN"):$PATH"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

info "Latest version: $LATEST_VERSION"
info "=== Starting deployment of custom2 batch pipeline ==="

# Step 1: Create directories
info "Creating directory structure..."
mkdir -p "$HOST_DATA_ROOT"
mkdir -p "$CONFIG_DIR"
mkdir -p "$TASK_CONFIG_ROOT"
mkdir -p "$DAGS_DIR"
mkdir -p "$AIRFLOW_LOCAL_SETTINGS_DIR"
mkdir -p "$GENERATED_DAGS_DIR"
mkdir -p "$TEMPLATE_DAGS_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$TOOLS_DIR"
mkdir -p "$PLATFORM_CORE_DIR"
mkdir -p "$PLATFORM_MCP_DIR"
mkdir -p "$PLATFORM_AGENT_DIR"
mkdir -p "$PLATFORM_PLANNING_DIR"
mkdir -p "$PLATFORM_RAG_DIR"
mkdir -p "$PLATFORM_OBSERVABILITY_DIR"
mkdir -p "$PLATFORM_EVAL_DIR"
mkdir -p "$PLATFORM_HARDENING_DIR"
mkdir -p "$KNOWLEDGE_DIR"
mkdir -p "$AGENT_EVAL_DIR"
success "Directories created"

# Step 2: Copy DAGs
info "Copying DAG files..."
cp -f "$PROJECT_DIR/dags/"*.py "$DAGS_DIR/" || error "Failed to copy DAGs"
# The new production flow uses dynamic task DAGs submitted by manage_task.sh.
# Legacy scheduler DAGs read fixed /opt/airflow/config/datasets_config*.yaml
# at import time and break clean runtime installs that do not use /opt/airflow.
rm -f "$DAGS_DIR"/dataset_schedulers*.py
if [ -d "$PROJECT_DIR/dags/templates" ]; then
    cp -f "$PROJECT_DIR/dags/templates/"*.py "$TEMPLATE_DAGS_DIR/" || error "Failed to copy DAG templates"
    if [ -f "$PROJECT_DIR/dags/templates/.airflowignore" ]; then
        cp -f "$PROJECT_DIR/dags/templates/.airflowignore" "$TEMPLATE_DAGS_DIR/.airflowignore" || error "Failed to copy DAG template ignore file"
    fi
fi
success "DAG files copied"

# Step 3: Copy scripts
info "Copying shell scripts..."
cp -f "$PROJECT_DIR/scripts/"*.sh "$SCRIPTS_DIR/" || error "Failed to copy shell scripts"
cp -f "$PROJECT_DIR/scripts/"*.py "$SCRIPTS_DIR/" || error "Failed to copy Python scripts"
cp -f "$PROJECT_DIR/scripts/tools/"*.py "$TOOLS_DIR/" || error "Failed to copy script tools"
# Task submission and management are now unified under manage_task.sh.
rm -f "$SCRIPTS_DIR/task.sh" "$SCRIPTS_DIR/submit_task.sh"
chmod +x "$SCRIPTS_DIR/"*.sh || error "Failed to make scripts executable"
chmod +x "$SCRIPTS_DIR/generate.py" || error "Failed to make generate.py executable"

info "Copying platform_core package..."
rm -rf "$PLATFORM_CORE_DIR"
mkdir -p "$PLATFORM_CORE_DIR"
mkdir -p "$PLATFORM_MCP_DIR"
cp -R "$PROJECT_DIR/platform_core/." "$PLATFORM_CORE_DIR/" || error "Failed to copy platform_core"
find "$PLATFORM_CORE_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_mcp package..."
rm -rf "$PLATFORM_MCP_DIR"
mkdir -p "$PLATFORM_MCP_DIR"
cp -R "$PROJECT_DIR/platform_mcp/." "$PLATFORM_MCP_DIR/" || error "Failed to copy platform_mcp"
find "$PLATFORM_MCP_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_agent package..."
rm -rf "$PLATFORM_AGENT_DIR"
mkdir -p "$PLATFORM_AGENT_DIR"
cp -R "$PROJECT_DIR/platform_agent/." "$PLATFORM_AGENT_DIR/" || error "Failed to copy platform_agent"
find "$PLATFORM_AGENT_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_planning package..."
rm -rf "$PLATFORM_PLANNING_DIR"
mkdir -p "$PLATFORM_PLANNING_DIR"
cp -R "$PROJECT_DIR/platform_planning/." "$PLATFORM_PLANNING_DIR/" || error "Failed to copy platform_planning"
find "$PLATFORM_PLANNING_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_rag package..."
rm -rf "$PLATFORM_RAG_DIR"
mkdir -p "$PLATFORM_RAG_DIR"
cp -R "$PROJECT_DIR/platform_rag/." "$PLATFORM_RAG_DIR/" || error "Failed to copy platform_rag"
find "$PLATFORM_RAG_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_observability package..."
rm -rf "$PLATFORM_OBSERVABILITY_DIR"
mkdir -p "$PLATFORM_OBSERVABILITY_DIR"
cp -R "$PROJECT_DIR/platform_observability/." "$PLATFORM_OBSERVABILITY_DIR/" || error "Failed to copy platform_observability"
find "$PLATFORM_OBSERVABILITY_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_eval package..."
rm -rf "$PLATFORM_EVAL_DIR"
mkdir -p "$PLATFORM_EVAL_DIR"
mkdir -p "$PLATFORM_HARDENING_DIR"
cp -R "$PROJECT_DIR/platform_eval/." "$PLATFORM_EVAL_DIR/" || error "Failed to copy platform_eval"
find "$PLATFORM_EVAL_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform_hardening package..."
rm -rf "$PLATFORM_HARDENING_DIR"
mkdir -p "$PLATFORM_HARDENING_DIR"
cp -R "$PROJECT_DIR/platform_hardening/." "$PLATFORM_HARDENING_DIR/" || error "Failed to copy platform_hardening"
find "$PLATFORM_HARDENING_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
info "Copying platform knowledge sources..."
rm -rf "$KNOWLEDGE_DIR"
mkdir -p "$KNOWLEDGE_DIR"
cp -R "$PROJECT_DIR/knowledge/." "$KNOWLEDGE_DIR/" || error "Failed to copy knowledge sources"
mkdir -p "$KNOWLEDGE_DIR/repository"
for doc in README.md skill.md usage_guide.md deploy_guide.md version.md V0.1_REFACTOR.md V0.2_GPU_SIMULATION.md V0.3_PLATFORM_MCP.md V0.4_READ_ONLY_AGENT.md V0.5_RAG_RUNBOOK.md V0.6_TASK_PLANNING.md V0.7_WRITE_AGENT_HITL.md V0.8_ACTION_VERIFICATION.md V0.9_EVALUATION_OBSERVABILITY.md V1.0_HARDENING_E2E.md V1.1_EVALUATION_ALIGNMENT.md V1.2_GEMINI_PROVIDER.md; do
    if [ -f "$PROJECT_DIR/$doc" ]; then
        cp -f "$PROJECT_DIR/$doc" "$KNOWLEDGE_DIR/repository/$doc" || error "Failed to copy knowledge document: $doc"
    fi
done
for cfg in task_submit_template.yaml task_types.yaml task_planning_defaults.yaml; do
    if [ -f "$PROJECT_DIR/config/$cfg" ]; then
        cp -f "$PROJECT_DIR/config/$cfg" "$KNOWLEDGE_DIR/repository/$cfg" || error "Failed to copy knowledge config: $cfg"
    fi
done
info "Copying Agent evaluation fixtures..."
rm -rf "$AGENT_EVAL_DIR"
mkdir -p "$AGENT_EVAL_DIR"
cp -R "$PROJECT_DIR/eval/." "$AGENT_EVAL_DIR/" || error "Failed to copy Agent evaluation fixtures"
if [ -f "$PROJECT_DIR/requirements-eval.txt" ]; then
    cp -f "$PROJECT_DIR/requirements-eval.txt" "$(dirname "$SCRIPTS_DIR")/requirements-eval.txt" || error "Failed to copy requirements-eval.txt"
fi
success "Scripts, platform_core, platform_mcp, platform_agent, platform_planning, platform_rag, platform_observability, platform_eval, platform_hardening, knowledge, eval fixtures and optional eval requirements copied"

# Step 4: Copy config
info "Copying configuration files..."
cp -f "$PROJECT_DIR/config/"*.yaml "$CONFIG_DIR/" || error "Failed to copy config"
if [ -f "$PROJECT_DIR/config/airflow_local_settings.py" ]; then
    cp -f "$PROJECT_DIR/config/airflow_local_settings.py" "$AIRFLOW_LOCAL_SETTINGS_DIR/airflow_local_settings.py" || error "Failed to copy airflow local settings"
fi
success "Configuration files copied"
info "Preserved task configs under: $TASK_CONFIG_ROOT"
info "Preserved generated DAGs under: $GENERATED_DAGS_DIR"

if [ "${DEPLOY_SKIP_VERIFY:-0}" = "1" ]; then
    info "DEPLOY_SKIP_VERIFY=1, skipping DAG parse/status verification"
    info "=== Deployment files copied successfully ==="
    exit 0
fi

# Step 5: Create Airflow pools
# info "Creating Airflow pools..."

# create_pool() {
#     local pool_name=$1
#     local slots=$2
#     local description=$3
#     if ! airflow pools list | grep -q "^$pool_name"; then
#         airflow pools create "$pool_name" "$slots" "$description"
#         info "Created pool: $pool_name ($slots slots)"
#     else
#         info "Pool $pool_name already exists"
#     fi
# }

# # Main pipeline pools
# create_pool "batch_pipeline_pool" 20 "Main pool for batch pipeline tasks"
# create_pool "batch_pipeline_pool_b" 20 "Pool for stage B (parallel)"
# create_pool "batch_pipeline_pool_c" 20 "Pool for stage C (parallel)"

# # Tier-based pools (for dataset_schedulers)
# create_pool "pool_small" 20 "Small dataset pool"
# create_pool "pool_medium" 15 "Medium dataset pool"
# create_pool "pool_large" 5 "Large dataset pool"

# # Global scheduler pool
# create_pool "global_dataset_pool" 10 "Global pool for dataset schedulers"

# success "Airflow pools created"

# Step 6: Initialize data directories from config
# info "Initializing dataset directories..."
# if [ -f "$CONFIG_DIR/datasets_config.yaml" ]; then
#     python3 << 'PYTHON_SCRIPT'
# import yaml
# import os

# CONFIG_PATH = "/opt/airflow/config/datasets_config.yaml"
# HOST_DATA_ROOT = "/opt/airflow/data"

# with open(CONFIG_PATH) as f:
#     config = yaml.safe_load(f)

# for ds in config.get("datasets", []):
#     ds_name = ds["dataset_name"]
#     custom_path = ds.get("dataset_path")
#     if custom_path:
#         data_dir = custom_path
#     else:
#         data_dir = os.path.join(HOST_DATA_ROOT, ds_name)
    
#     os.makedirs(data_dir, exist_ok=True)
#     print(f"Created directory: {data_dir}")
# PYTHON_SCRIPT
#     success "Dataset directories initialized"
# else
#     info "No datasets_config.yaml found, skipping directory initialization"
# fi

# Step 7: Restart Airflow services
# info "Restarting Airflow services..."
# if command -v systemctl &>/dev/null; then
#     if systemctl is-active --quiet airflow-scheduler; then
#         systemctl restart airflow-scheduler
#     fi
#     if systemctl is-active --quiet airflow-webserver; then
#         systemctl restart airflow-webserver
#     fi
# elif pgrep -f "airflow scheduler" &>/dev/null; then
#     pkill -f "airflow scheduler"
#     pkill -f "airflow webserver"
#     pkill -f "airflow api-server"
#     # Give time for processes to exit
#     sleep 3
#     # Start in background
#     airflow standalone start &>/tmp/airflow_start.log &
#     info "Airflow started in background, check /tmp/airflow_start.log for details"
# else
#     info "No running Airflow processes found, please start manually"
# fi
# success "Airflow services restarted"

# Step 8: Verify deployment
info "Verifying deployment..."

# Match a DAG ID exactly in the default Airflow table output.
dag_exists() {
    local expected_dag_id="$1"
    "$AIRFLOW_BIN" dags list 2>/dev/null | awk -F '|' -v expected="$expected_dag_id" '
        {
            dag_id = $1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", dag_id)
            if (dag_id == expected) {
                found = 1
            }
        }
        END { exit(found ? 0 : 1) }
    '
}

# Check the shared Universal runtime exactly. Generated task DAGs are created
# later by manage_task.sh submit, never by deployment.
dag_loaded=0
for _ in $(seq 1 30); do
    if dag_exists "batch_pipeline_universal"; then
        dag_loaded=1
        break
    fi
    sleep 5
done
if [ "$dag_loaded" -eq 1 ]; then
    success "Shared DAG batch_pipeline_universal loaded successfully"
else
    error "Shared DAG batch_pipeline_universal not found"
fi

# Check scripts are in place
if [ -f "$SCRIPTS_DIR/run_precheck.sh" ] \
    && [ -f "$SCRIPTS_DIR/run_parser.sh" ] \
    && [ -f "$SCRIPTS_DIR/validate_json.py" ] \
    && [ -x "$SCRIPTS_DIR/generate.py" ] \
    && [ -x "$SCRIPTS_DIR/manage_task.sh" ] \
    && [ -x "$SCRIPTS_DIR/start_task.sh" ] \
    && [ -f "$SCRIPTS_DIR/task_manager.py" ] \
    && [ -f "$PLATFORM_CORE_DIR/__init__.py" ] \
    && [ -f "$PLATFORM_CORE_DIR/services/task_service.py" ] \
    && [ -f "$TOOLS_DIR/genarate_dataset_config.py" ]; then
    success "Scripts are in place"
else
    error "Some scripts are missing"
fi

# Check config is in place
if [ -f "$CONFIG_DIR/task_submit_template.yaml" ] \
    && [ -f "$TEMPLATE_DAGS_DIR/batch_pipeline_universal_template.py" ]; then
    success "Task template files are in place"
else
    error "Task YAML or DAG template is missing"
fi

# Check airflow is running
AIRFLOW_HOME="$AIRFLOW_HOME" AIRFLOW_BIN="$AIRFLOW_BIN" AIRFLOW_PORT="$AIRFLOW_PORT" "$SCRIPT_DIR/airflow_ctl.sh" status

info "=== Deployment completed successfully ==="
echo ""
echo "Next steps:"
if [ -n "${AIRFLOW_PUBLIC_URL:-}" ]; then
    echo "1. Check Airflow UI at ${AIRFLOW_PUBLIC_URL}"
else
    HOST_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -Ev '^(127\.|172\.17\.|169\.254\.|$)' | head -n 1 || true)
    if [ -n "${HOST_IP:-}" ]; then
        echo "1. Check Airflow UI at http://${HOST_IP}:${AIRFLOW_PORT}"
    else
        echo "1. Check Airflow UI at http://<machine-ip>:${AIRFLOW_PORT}"
    fi
fi
echo "2. Generate YAML: $SCRIPTS_DIR/generate.py <record_dir ...> -o <yaml_path>"
echo "3. Submit YAML: $SCRIPTS_DIR/manage_task.sh submit --name <task_name_prefix> --yaml <yaml_path>"
echo "4. Test one-command submit: $SCRIPTS_DIR/start_task.sh --dataset <record_dir> --name <task_name_prefix> --yaml $CONFIG_DIR/task_full_serial_template.yaml"
echo "5. Manage task: $SCRIPTS_DIR/manage_task.sh <action> <task_name> [clip ...]"
echo "6. Generated task DAGs: $GENERATED_DAGS_DIR"
echo "7. Submitted task configs: $TASK_CONFIG_ROOT"
