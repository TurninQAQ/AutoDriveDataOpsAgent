#!/usr/bin/env bash
set -euo pipefail

LATEST_VERSION="v1.1.0  2027-07-15:(new:None base:None)"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
DAGS_DIR="/home/cidi/airflow/dags/data_center"
HOST_DATA_ROOT="/opt/airflow/data"
CONFIG_DIR="/opt/airflow/config"
SCRIPTS_DIR="/opt/airflow/scripts"

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
mkdir -p "$DAGS_DIR"
mkdir -p "$SCRIPTS_DIR"
success "Directories created"

# Step 2: Copy DAGs
info "Copying DAG files..."
cp -f "$PROJECT_DIR/dags/"*.py "$DAGS_DIR/" || error "Failed to copy DAGs"
cp -f "$PROJECT_DIR/dags/"dataset_schedulers_segment.py "$DAGS_DIR/" || error "Failed to copy DAGs"
cp -f "$PROJECT_DIR/dags/"batch_pipeline_universal_segment.py "$DAGS_DIR/" || error "Failed to copy DAGs"
# cp -f "$PROJECT_DIR/dags/"dataset_schedulers.py "$DAGS_DIR/" || error "Failed to copy DAGs"
# cp -f "$PROJECT_DIR/dags/"batch_pipeline_universal.py "$DAGS_DIR/" || error "Failed to copy DAGs"

success "DAG files copied"

# Step 3: Copy scripts
info "Copying shell scripts..."
cp -f "$PROJECT_DIR/scripts/"*.sh "$SCRIPTS_DIR/" || error "Failed to copy shell scripts"
cp -f "$PROJECT_DIR/scripts/"*.py "$SCRIPTS_DIR/" || error "Failed to copy Python scripts"
chmod +x "$SCRIPTS_DIR/"*.sh || error "Failed to make scripts executable"
success "Shell scripts copied and made executable"

# Step 4: Copy config
info "Copying configuration files..."
cp -f "$PROJECT_DIR/config/"datasets_config_segment.yaml "$CONFIG_DIR/" || error "Failed to copy config"
success "Configuration files copied"

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
sleep 5

# Check DAGs are loaded
if airflow dags list | grep -q "batch_pipeline_universal"; then
    success "DAG batch_pipeline_universal loaded successfully"
else
    error "DAG batch_pipeline_universal not found"
fi

# Check scripts are in place
if [ -f "$SCRIPTS_DIR/run_parser.sh" ] && [ -f "$SCRIPTS_DIR/validate_json.py" ]; then
    success "Scripts are in place"
else
    error "Some scripts are missing"
fi

# Check config is in place
if [ -f "$CONFIG_DIR/datasets_config_segment.yaml" ]; then
    success "Configuration file is in place"
else
    error "Configuration file is missing"
fi

# Check airflow is running
./airflow_ctl.sh status

info "=== Deployment completed successfully ==="
echo ""
echo "Next steps:"
HOST_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -Ev '^(127\.|172\.17\.|169\.254\.|$)' | head -n 1 || true)
if [ -n "${HOST_IP:-}" ]; then
    echo "1. Check Airflow UI at http://${HOST_IP}:${AIRFLOW_PORT:-8080}"
else
    echo "1. Check Airflow UI at http://<machine-ip>:${AIRFLOW_PORT:-8080}"
fi
echo "2. Unpause DAGs in the UI"
echo "3. Verify pools are created in Admin > Pools"
echo "4. Trigger scheduler_<dataset_name> DAG to start pipeline"
