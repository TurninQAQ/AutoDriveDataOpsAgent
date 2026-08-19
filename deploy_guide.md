# deploy_ci_cloud 0731 部署指南

本文只说明从 0 部署平台。日常提交任务、定时任务、优先级、停止恢复删除等使用说明见 `usage_guide.md`。

## 1. 基础环境

```bash
sudo apt update
sudo apt install -y git curl vim python3 python3-venv python3-pip postgresql postgresql-contrib docker.io lsof
sudo systemctl enable --now postgresql
sudo systemctl enable --now docker
sudo usermod -aG docker "$(whoami)"
```

加入 docker 组后需要退出当前登录会话再重新登录。

## 2. 准备 PostgreSQL

数据库用户名、密码、库名要和 `.env` 中的 `AIRFLOW_DB_URI` 保持一致。

示例：

```bash
sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE USER airflow_0731 WITH PASSWORD 'airflow_0731_password';"
sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE airflow_0731 OWNER airflow_0731;"
sudo -u postgres psql -v ON_ERROR_STOP=1 -c "GRANT ALL PRIVILEGES ON DATABASE airflow_0731 TO airflow_0731;"
sudo -u postgres psql -v ON_ERROR_STOP=1 -d airflow_0731 -c "GRANT ALL ON SCHEMA public TO airflow_0731;"
```

验证：

```bash
psql postgresql://airflow_0731:airflow_0731_password@127.0.0.1:5432/airflow_0731 -tAc "SELECT current_database(), current_user;"
```

期望输出：

```text
airflow_0731|airflow_0731
```

## 3. 拉取代码

```bash
git clone -b feature/0724-deploy-optimize http://172.16.201.6/BCD/deploy_ci_cloud.git deploy_ci_cloud_0731
cd deploy_ci_cloud_0731
```

如果代码已经在机器上，直接进入源码目录即可。

## 4. 配置运行目录

查看本机 IP：

```bash
hostname -I
```

编辑 `runtime.path`：

```bash
vim runtime.path
```

格式必须是：

```text
<本机IP>=<运行目录>
```

示例：

```text
172.16.201.103=/home/cidi/deploy_ci_cloud_runtime
```

创建运行目录并确保运行用户可写：

```bash
mkdir -p /home/cidi/deploy_ci_cloud_runtime
sudo chown -R cidi:cidi /home/cidi/deploy_ci_cloud_runtime
```

运行目录是平台真实写入 Airflow、DAG、脚本、任务配置、日志、队列和 GPU 锁的地方。

## 5. 配置 `.env`

```bash
vim .env
```

示例：

```bash
AIRFLOW_PORT=8081
AIRFLOW_PUBLIC_URL=http://172.16.201.103:8081
AIRFLOW_API_BASE=
AIRFLOW_EXECUTION_API_PORT=8085
AIRFLOW_EXECUTION_API_BASE=
AIRFLOW_WORKER_LOG_SERVER_PORT=8082
AIRFLOW_TRIGGER_LOG_SERVER_PORT=8083
AIRFLOW_SCHEDULER_HEALTH_CHECK_SERVER_PORT=8084
AIRFLOW_DB_URI=postgresql+psycopg2://airflow_0731:airflow_0731_password@127.0.0.1:5432/airflow_0731
AIRFLOW_ADMIN_USER=cidi
AIRFLOW_ADMIN_PASSWORD=cidi123456
```

说明：

- `AIRFLOW_PORT` 是 Airflow 页面/API 端口。
- `AIRFLOW_PUBLIC_URL` 是浏览器访问地址。
- `AIRFLOW_EXECUTION_API_PORT` 是本机任务执行/心跳 API 端口，默认监听 `127.0.0.1`，要和其他内部端口错开。
- `AIRFLOW_DB_URI` 是 Airflow 元数据库连接。
- `AIRFLOW_ADMIN_USER` 和 `AIRFLOW_ADMIN_PASSWORD` 是 Airflow 页面/API 登录账号。
- 同机多实例部署时，内部端口 `AIRFLOW_EXECUTION_API_PORT`、`AIRFLOW_WORKER_LOG_SERVER_PORT`、`AIRFLOW_TRIGGER_LOG_SERVER_PORT`、`AIRFLOW_SCHEDULER_HEALTH_CHECK_SERVER_PORT` 都要错开。

## 6. 安装平台

建议切到运行用户执行，例如：

```bash
sudo -iu cidi
cd /home/cfy/project/two/deploy_ci_cloud_0731
```

安装：

```bash
chmod +x platform task
chmod +x scripts/*.sh
chmod +x scripts/generate.py
chmod +x scripts/task_manager.py
./platform install
```

`install` 会创建 runtime venv、生成 `airflow.cfg`、应用 Airflow runtime 补丁、执行 `airflow db migrate`，并把 DAG/scripts/platform_core/config 部署到 `runtime.path` 指向的运行目录。

## 7. 启动和验证

```bash
./platform start
./platform status
```

确认 execution API 已经和页面 API 分离：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/airflow config get-value core execution_api_server_url
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python /home/cfy/project/two/deploy_ci_cloud_0731/scripts/patch_airflow_grid_session.py --check
```

预期：

```text
http://127.0.0.1:8085/execution/
airflow_grid_session_patch=already_patched ...
```

正常状态应看到：

```text
scheduler: 运行中
dag processor: 运行中
triggerer: 运行中
api server: 运行中
execution api server: 运行中
task submit scheduler: 运行中
Public API Health: 正常
Execution API Health: 正常
```

访问 Airflow：

```text
http://<本机IP>:<AIRFLOW_PORT>
```

登录账号密码来自 `.env`：

```text
AIRFLOW_ADMIN_USER / AIRFLOW_ADMIN_PASSWORD
```

## 8. 日常更新部署

代码更新后：

```bash
cd /path/to/deploy_ci_cloud_0731
git pull --rebase
./platform deploy
./platform status
```

`deploy` 只同步 DAG/scripts/platform_core/config 到 runtime，不重建 venv，不改数据库，不重启组件。

执行 deploy 前，平台会检查 Airflow 元数据库中的全部 `running/queued` DagRun，
并检查名称以 `airflow-` 或 `airflow-task-` 开头的运行容器。任一存在都会直接拒绝
deploy；它不会停止或修改这些任务。

deploy 会覆盖 runtime 中的公共 DAG、DAG 模板、`scripts/*.sh`、`scripts/*.py`、
`scripts/tools/*.py`、`config/*.yaml` 和 `airflow_local_settings.py`，同时更新
runtime 的 `bin/task`、`bin/platform` 入口。以下运行态会被保留：

- `$RUNTIME_DIR/opt_airflow/config/tasks/<task_name>/` 下已提交任务的配置；
- `$RUNTIME_DIR/airflow/dags/data_center/generated/` 下已生成 DAG；
- `state/` 下的队列、任务锁、GPU reservation、定时记录及实验日志；
- Airflow 元数据库和现有 venv。

因此，根目录 Markdown 文档以及 `/home/cfy/project/two/test` 下的实验脚本并不会被
复制到 runtime；仅更新它们不会影响部署中的平台任务。修改 DAG、任务管理脚本、
生成器或 `config/task_types.yaml` 后，才需要在平台空闲时执行 deploy。

`./platform deploy` 内部以文件同步为主，完成后应执行 `./platform status`，并在
Airflow 页面确认新 DAG 已被解析。

如果平台组件异常：

```bash
./platform status
./platform start
```

谨慎使用：

```bash
./platform restart
```

`restart` 是故障恢复入口，会停止 active/queued/scheduled 任务，清队列、锁和 GPU reservation 后重启组件。

不要为了更新文档或测试脚本执行 restart；它会终止当前正在运行的全流程实验。

## V0.6 Task Planning 配置

`./platform install/deploy` 会同步：

```text
$RUNTIME_DIR/opt_airflow/platform_planning/
$RUNTIME_DIR/opt_airflow/config/task_planning_defaults.yaml
```

默认无需额外配置。

如果需要使用其他平台默认配置文件，可在 `.env` 指定：

```bash
PLATFORM_TASK_PLANNING_DEFAULTS=/absolute/path/task_planning_defaults.yaml
```

安装后验证：

```bash
$RUNTIME_DIR/bin/dataops-agent plan-task-eval
$RUNTIME_DIR/bin/dataops-agent plan-task \
  '创建一个test任务，只运行 precheck，数据 /tmp/record_a'
```

## V0.9 Agent Trace / Audit

`platform install` 会将以下配置写入 Runtime `config/platform.env`：

```text
PLATFORM_AGENT_TRACE_ENABLED
PLATFORM_AGENT_TRACE_DIR
PLATFORM_AGENT_AUDIT_FILE
PLATFORM_AGENT_TRACE_MAX_VALUE_CHARS
```

默认：

```text
PLATFORM_AGENT_TRACE_DIR=$AIRFLOW_STATE_DIR/agent_traces
PLATFORM_AGENT_AUDIT_FILE=$AIRFLOW_STATE_DIR/agent_audit/audit.jsonl
```

`platform deploy` 会同步：

```text
platform_observability/
platform_eval/
eval/
```

部署后可用：

```bash
$RUNTIME_DIR/bin/dataops-agent traces --limit 20
$RUNTIME_DIR/bin/dataops-agent trace <trace_id>
$RUNTIME_DIR/bin/dataops-agent eval
```

## V1.1 Evaluation Alignment

默认部署会同步 `platform_eval/`、`eval/v1_1/` 和 `requirements-eval.txt`，但不会自动安装 Ragas/DeepEval。dependency-light 主门禁无需外部模型：

```bash
dataops-agent eval-aligned
dataops-agent eval-frameworks
```

若需要运行 Ragas / DeepEval semantic judge，在首次安装前设置：

```bash
export PLATFORM_INSTALL_EVAL_DEPS=1
./platform install
export OPENAI_API_KEY=...
dataops-agent eval-ragas
dataops-agent eval-deepeval
```

Promptfoo 为独立 CLI；安装后可执行：

```bash
dataops-agent eval-promptfoo
# 有攻击模型 provider 时：
dataops-agent eval-promptfoo --redteam
```

默认质量门槛由 `eval/v1_1/thresholds.json` 管理。评测依赖不进入默认生产 Runtime，避免影响 Airflow 调度依赖。


## V1.2 Gemini Provider 与真实 Embedding

V1.2 可在无真实 GPU 的机器上使用 Gemini 作为 Agent 模型，并使用 Gemini Embedding 作为 RAG 向量侧。

先 rotate 任何已经外泄的 API Key，然后只通过环境变量配置：

```bash
export GEMINI_API_KEY='<ROTATED_KEY>'
export PLATFORM_AGENT_PROVIDER=gemini
export PLATFORM_AGENT_MODEL=gemini-3.7-flash
export PLATFORM_AGENT_RUNTIME=langgraph
export PLATFORM_GPU_RUNTIME=simulated
export PLATFORM_STAGE_RUNTIME=mock
export PLATFORM_RAG_EMBED_PROVIDER=gemini
export PLATFORM_RAG_EMBED_MODEL=gemini-embedding-2
export PLATFORM_RAG_EMBED_DIM=768
```

执行：

```bash
./platform install
./platform deploy
```

检查：

```bash
dataops-agent doctor --strict --json
dataops-agent knowledge build --force --json
dataops-agent knowledge status --json
```

默认不配置 `PLATFORM_RAG_EMBED_PROVIDER=gemini` 时仍使用 V1.1 的 hash hybrid，不需要 Gemini Embedding API。

详细设计与本地验收见 `V1.2_GEMINI_PROVIDER.md`。
