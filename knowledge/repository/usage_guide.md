# deploy_ci_cloud 0731 使用指南

本文只说明平台已经部署后的日常使用。全新机器部署见 `deploy_guide.md`。

## 1. 入口

进入源码目录：

```bash
cd /home/cfy/project/two/deploy_ci_cloud_0731
```

查看平台状态：

```bash
./platform status
```

提交和管理任务统一使用源码目录入口：

```bash
./task --help
```

`./task` 会读取 `runtime.path`，再转发到当前机器 runtime 下的任务管理脚本。

## 2. 日常更新代码

```bash
git pull --rebase
./platform deploy
./platform status
```

`deploy` 只同步 DAG/scripts/platform_core/config 到 runtime，不重建 venv，不改数据库，不重启组件。

有任务正在运行时，`deploy` 会拒绝执行：它检查全部 `running/queued` DagRun 和
`airflow-`/`airflow-task-` 容器。先等任务结束，或明确停止/删除任务后再部署；不要在
抢占恢复或全流程实验期间 deploy。deploy 保留已提交任务配置、generated DAG、队列、
GPU 锁和 state 日志，但根目录文档与 `/home/cfy/project/two/test` 下的实验脚本不会部署
到 runtime。

## 3. 任务类型和优先级

任务类型配置：

```text
config/task_types.yaml
```

示例：

```yaml
default_priority: 100

task_types:
  release:
    priority: 10
  reprocess:
    priority: 20
  test:
    priority: 50
  debug:
    priority: 80
```

规则：

- 数字越小，优先级越高。
- 任务 YAML 可写 `task_type`。
- 如果只写 `task_type`，优先级从 `config/task_types.yaml` 读取。
- 如果任务 YAML 直接写 `priority`，会覆盖 `task_type` 默认优先级。
- 如果两个都不写，使用 `default_priority`。
- 如果写了不存在的 `task_type`，提交时报错。

新增任务类型后需要同步 runtime：

```bash
./platform deploy
```

## 4. 任务 YAML 核心字段

示例：

```yaml
task_type: reprocess

pipeline_stages:
  - precheck
  - parser
  - segment
  - map
  - od
  - coloration
  - occ

max_active_runs: 5
task_exclusive: true
task_lock_wait_interval_sec: 10
preempt_grace_timeout_min: 60

gpu_ids: "0,1"
gpu_stages: "segment,od,occ"
exclusive_gpu_stages: "segment,od"
exclusive_gpu_idle_used_max_mb: 512
gpu_stage_memory_mb:
  segment: 24000
  od: 24000
  occ: 4000
gpu_wait_interval_sec: 10
gpu_reservation_pending_sec: 60

datasets:
  - dataset_name: clip_001
    tier: small
    pool: default_pool
    dataset_path: /path/to/record_dir
    image_parser: 172.16.201.100:5000/data_parser:xxx
    image_segment: 172.16.201.100:5000/sam31:xxx
    image_map: 172.16.201.100:5000/offline_mapping:xxx
    image_od: 172.16.201.100:5000/label_od:xxx
    image_coloration: 172.16.201.100:5000/pointcloud_coloration:xxx
    image_occ: 172.16.201.100:5000/label_occ:xxx
    timeout_min: 60
```

重点：

- `dataset_path` 是 record 根目录，平台实际处理 `<dataset_path>/<dataset_name>`。
- `pipeline_stages` 中普通项串行执行，列表项表示并行阶段，例如 `[od, occ]`。
- `task_exclusive: true` 表示进入全局任务队列，优先级和抢占才有意义。
- `timeout_min` 是每个阶段拿到资源并开始运行后的最大运行时间，GPU 等待不计入。
- 当前 OD 需求按 segment 级别处理，建议 `exclusive_gpu_stages` 包含 `od`，且 `od` 显存配置与 `segment` 一致。

## 5. 生成 YAML

标准任务和回归实验的 YAML 统一只由生成器产生，不手写 YAML。主线方式是编辑生成器
顶部参数后无参数生成，或直接在命令行传 record 目录和参数。

当前脚本文件名是：

```text
scripts/tools/genarate_dataset_config.py
```


优先修改文件顶部这些参数：

```python
TARGET_DIRS = [
    "/path/record_a",
    "/path/record_b",
]

OUTPUT_YAML = "/tmp/task.yaml"

TASK_TYPE = None
PRIORITY = None
TIER = "small"
POOL = "default_pool"

TIMEOUT_MIN = 60
MAX_ACTIVE_RUNS = 2
TASK_EXCLUSIVE = True
TASK_LOCK_WAIT_INTERVAL_SEC = 10
PREEMPT_GRACE_TIMEOUT_MIN = 60

PIPELINE_STAGES = [
    "precheck",
    "parser",
    "segment",
    "map",
    "od",
    "coloration",
    "occ",
]

GPU_IDS = "0,1"
GPU_STAGES = "segment,od,occ"
EXCLUSIVE_GPU_STAGES = "segment,od"
EXCLUSIVE_GPU_IDLE_USED_MAX_MB = 512
GPU_STAGE_MEMORY_MB = OrderedDict([
    ("segment", 24000),
    ("od", 24000),
    ("occ", 4000),
])
GPU_WAIT_INTERVAL_SEC = 10
GPU_RESERVATION_PENDING_SEC = 60

DEFAULT_IMAGES = OrderedDict([
    ...
])
```

生成：

```bash
cd /home/cfy/project/two/deploy_ci_cloud_0731
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python scripts/tools/genarate_dataset_config.py
```

补充用法：也可以不改脚本顶部参数，直接从命令行传 record 目录。

单个 record：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python \
  scripts/tools/genarate_dataset_config.py \
  /path/to/record_dir --output /tmp/task.yaml
```

多个 record：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python \
  scripts/tools/genarate_dataset_config.py \
  /path/record_a /path/record_b --output /tmp/task.yaml
```

命令行指定阶段和 GPU 配置：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python \
  scripts/tools/genarate_dataset_config.py \
  /path/to/record_dir --output /tmp/task.yaml \
  --pipeline-stages precheck,parser,segment,map,od,coloration,occ \
  --gpu-stages segment,od,occ \
  --gpu-ids 0,1 \
  --exclusive-gpu-stages segment,od \
  --gpu-stage-memory-mb segment:24000,od:24000,occ:4000 \
  --max-active-runs 2 \
  --timeout-min 60
```

## 6. 提交任务

```bash
./task submit --name reprocess --yaml /tmp/task.yaml
```

`--name` 是任务名前缀，平台自动追加北京时间秒级时间戳：

```text
reprocess_20260803_103000
```

提交成功后会打印：

```text
task_name=<完整任务名>
dag_id=batch_pipeline_universal_<完整任务名>
task_type=<类型> priority=<优先级> priority_source=<来源>
queue_state=<start|queued|preempt_requested>
```

后续管理必须使用完整任务名。

## 7. 定时提交

```bash
./task submit --name reprocess --yaml /tmp/task.yaml --schedule "2026-08-03 23:00"
```

查看定时任务：

```bash
./task schedule list
./task schedule list --all
```

取消未到点的定时任务：

```bash
./task schedule remove <schedule_id> --yes
```

手动触发一次定时扫描：

```bash
./task submit scheduler --scheduler-once
```

## 8. 调整优先级

```bash
./task priority <完整任务名> --priority 5
```

该命令会修改 runtime 中该任务的 YAML，并刷新队列。

如果新优先级高于当前 active 任务，会发出阶段边界抢占请求：

- 当前 active 任务先跑完正在执行的 stage 和对应 validate。
- 完成 validate 后，该 clip 以恢复点重新入队。
- 所有需要让出的 clip 都到达阶段边界后，高优先级任务变为 active。
- 低优先级任务保留在平台队列中，后续从恢复点继续，不重跑已完成 stage。
- 低优先级任务重新 active 后，只触发 recovery DagRun；被抢占前遗留的原始
  DagRun 会跳过，不计入任务完成数，避免同一 clip 重复执行。
- Airflow 页面可能把原始 DagRun 显示为 success/skipped；平台排队状态以
  `state/task_queue/queue.lock` 为准。

## 9. 停止、恢复、删除

停止整个任务：

```bash
./task stop <完整任务名> --yes
```

停止指定 clip：

```bash
./task stop <完整任务名> clip_001 clip_002 --yes
```

恢复失败 clip：

```bash
./task resume <完整任务名>
```

恢复指定 clip：

```bash
./task resume <完整任务名> clip_001 clip_002
```

删除整个任务：

```bash
./task delete <完整任务名> --yes
```

不停止容器的删除：

```bash
./task delete <完整任务名> --yes --no-stop-containers
```

如果停止或删除的是当前 active 任务，平台会从 `queue.lock.queue` 里取下一个任务并触发它。
因此被抢占后排队的任务不需要手工 resume；active 任务自然完成、stop 或 delete 后都会走同一套队列推进逻辑。

## 10. 查看运行状态

平台状态：

```bash
./platform status
```

查看队列：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python -m json.tool \
  /home/cidi/deploy_ci_cloud_runtime/state/task_queue/queue.lock
```

Airflow 页面中关注：

- `batch_pipeline_universal`：公共模板 DAG。
- `batch_pipeline_universal_<完整任务名>`：每次 submit 生成的任务 DAG。

## 11. 轻量全流程验证实验

该实验用于验证 task_type 默认优先级、定时提交、手动 priority 调整、阶段边界抢占、被抢占任务恢复，以及删除 active 后队列自动推进。

实验 YAML 建议使用：

```text
/home/cfy/project/two/test/0731_full_validation_yamls/minimal/01_taska_two_clips.yaml
/home/cfy/project/two/test/0731_full_validation_yamls/minimal/02_taskb_one_clip.yaml
```

这两个 YAML 不占 GPU，阶段顺序是：

```yaml
pipeline_stages:
  - parser
  - precheck
gpu_stages: ""
exclusive_gpu_stages: ""
timeout_min: 240
```

`parser` 是真实数据解析，`precheck` 是轻量恢复锚点。taskA 被抢占后应从 `precheck` 恢复，不重跑 parser。

先清理旧实验并部署当前代码：

```bash
cd /home/cfy/project/two/deploy_ci_cloud_0731

./task delete val_taska_20260804_134514 --yes || true
./task delete val_taskb_20260804_134733 --yes || true

./platform deploy
./platform restart
./platform status
```

提交 taskA：

```bash
./task submit \
  --name val_taska \
  --yaml /home/cfy/project/two/test/0731_full_validation_yamls/minimal/01_taska_two_clips.yaml
```

定时 1 分钟后提交 taskB：

```bash
RUN_AFTER=$(date -d '+1 minute' '+%F %T')

./task submit \
  --name val_taskb \
  --yaml /home/cfy/project/two/test/0731_full_validation_yamls/minimal/02_taskb_one_clip.yaml \
  --schedule "$RUN_AFTER"

./task schedule list --all
```

taskB 到点后调高优先级：

```bash
TASKB=$(basename "$(ls -td /home/cidi/deploy_ci_cloud_runtime/opt_airflow/config/tasks/val_taskb_* | head -1)")
echo "$TASKB"

./task priority "$TASKB" --priority 5
```

查看平台队列：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python -m json.tool \
  /home/cidi/deploy_ci_cloud_runtime/state/task_queue/queue.lock
```

预期：

- taskA 先是 active，priority=80。
- taskB 到点后先 queued，priority=81。
- 执行 priority 后 taskA 变为 `status=draining`，taskB 仍 queued 等阶段边界。
- taskA 的两个 parser 完成并 validate 后，taskB 变为 active。
- taskA 进入 queue，`pending_run_confs` 中应有 `_platform_resume_from_stage: precheck`。

可选验证：删除 active taskB 后，taskA 应自动接管并从 precheck 恢复：

```bash
./task delete "$TASKB" --yes

/home/cidi/deploy_ci_cloud_runtime/venv/bin/python -m json.tool \
  /home/cidi/deploy_ci_cloud_runtime/state/task_queue/queue.lock
```

预期：`active.task_name` 变为最新的 `val_taska_*`。

实验结束清理：

```bash
TASKA=$(basename "$(ls -td /home/cidi/deploy_ci_cloud_runtime/opt_airflow/config/tasks/val_taska_* | head -1)")
echo "$TASKA"

./task delete "$TASKA" --yes || true
./task delete "$TASKB" --yes || true
```

## 12. 全流程定时、优先级与手动提权回归实验

真实 full record 的自动化实验位于：

```text
/home/cfy/project/two/test/0731_full_pipeline_schedule_priority_experiment/
```

它只调用 `scripts/tools/genarate_dataset_config.py` 生成 YAML，并通过 `./task submit`、
`./task submit --schedule`、`./task priority` 操作平台。阶段严格串行：

```text
precheck -> parser -> segment -> map -> od -> coloration -> occ
```

实验会立即提交 A（`taska=80`），定时提交 B（`taskb=81`），随后自动把 B 提到
priority `5`。因此定时提交本身不会抢占 A，抢占来源可唯一归因于手动提权。

仅在专用、空闲的 runtime 运行。全量 5 clip 可能超过 12 小时，建议显式给出 24 小时
的监控时限：

```bash
sudo -iu cidi
cd /home/cfy/project/two/deploy_ci_cloud_0731

EXPERIMENT_TIMEOUT_MIN=1440 \
bash /home/cfy/project/two/test/0731_full_pipeline_schedule_priority_experiment/run_full_pipeline_schedule_priority_experiment.sh
```

如果确认可以停止当前任务，才可以加 `RESET_PLATFORM=1`；该选项会调用
`./platform restart` 并清理 active、queued、scheduled 任务。实验启动后无需手动提交
或调优先级，脚本会自动监控队列并在结束时生成验证日志。

正确的抢占恢复现象是：A 的 drain target 在已完成 `parser` 后以
`_platform_resume_from_stage=segment` 恢复，尚未开始的 clip 从头执行；原始 A DagRun
在 Airflow 页面可能显示 `success`，但其 `verify_pipeline_status` 为 `skipped`，不计入
平台完成数，也不会在恢复点之后重复执行业务阶段。

日志和生成器输出位于：

```text
/home/cidi/deploy_ci_cloud_runtime/state/experiment_logs/
  0731_full_pipeline_schedule_priority_experiment/<时间戳>/
```

## V0.4 Read-only Agent

Agent 只提供读取和诊断能力，不会提交、停止、删除、恢复任务，也不会修改优先级。

安装后：

```bash
$RUNTIME_DIR/bin/dataops-agent ask "release_xxx 现在是什么状态？"
$RUNTIME_DIR/bin/dataops-agent ask "release_xxx 为什么没跑？" --json
$RUNTIME_DIR/bin/dataops-agent chat --thread-id incident-001
$RUNTIME_DIR/bin/dataops-agent tools
```

默认 `PLATFORM_AGENT_PROVIDER=auto`：如果配置了 OpenAI-compatible 模型通道则使用 LLM，否则使用 deterministic heuristic provider 完成本地只读开发和回归。

正式 Agent Runtime 默认：

```bash
PLATFORM_AGENT_RUNTIME=langgraph
```

依赖和架构细节见 `V0.4_READ_ONLY_AGENT.md`。

## Agent 静态知识与 Runbook（V0.5）

V0.5 的 RAG 只用于平台规则和 Runbook；当前 Task/GPU/Container/Airflow 状态仍通过 MCP Tool 获取。

```bash
# 查看知识索引
$RUNTIME_DIR/bin/dataops-agent knowledge status

# 构建/刷新知识索引
$RUNTIME_DIR/bin/dataops-agent knowledge build

# 直接检索平台知识
$RUNTIME_DIR/bin/dataops-agent knowledge search "draining 为什么迟迟不能结束"

# 运行固定 Retrieval Eval
$RUNTIME_DIR/bin/dataops-agent knowledge eval

# 静态机制问题可以不访问实时 MCP
PLATFORM_AGENT_PROVIDER=heuristic PLATFORM_AGENT_RUNTIME=sequential \
  $RUNTIME_DIR/bin/dataops-agent ask "软抢占机制是什么？"
```

索引默认位于：

```text
$AIRFLOW_STATE_DIR/agent_knowledge/index.json
```

## 自然语言任务规划（V0.6）

V0.6 可以把自然语言转换为经过现有平台规则校验的 Task YAML，但不会提交任务。

```bash
$RUNTIME_DIR/bin/dataops-agent plan-task \
  '创建一个release任务，把 /data/record_001 做完整流程，最多同时4个clip，segment和od独占GPU，occ共享GPU'
```

写本地 YAML：

```bash
$RUNTIME_DIR/bin/dataops-agent plan-task \
  '创建一个test任务，只运行 precheck，数据 /tmp/record_a' \
  --output /tmp/planned_task.yaml
```

执行固定规划评测：

```bash
$RUNTIME_DIR/bin/dataops-agent plan-task-eval
```

说明：

- Task type 默认优先级来自 `config/task_types.yaml`。
- Pipeline、GPU、镜像等确定性默认值来自 `config/task_planning_defaults.yaml`。
- 最终 YAML 会调用现有 `platform_core.config.validate_config()`。
- 缺少 dataset path、Stage 非法、镜像/GPU 配置不完整时结果为 invalid，不允许写 YAML。
- `创建任务/生成 YAML` 只表示规划；`提交/触发/启动任务` 在 V0.6 仍被 Agent Policy 禁止。


## V0.7 Write Agent / HITL

V0.7 开放 submit/resume/priority/stop/delete，但 `ask` 本身不会直接执行写操作。

例如：

```bash
$RUNTIME_DIR/bin/dataops-agent ask '把 release_xxx 优先级改成5'
```

Agent 先读取当前 Task/Queue 状态并生成 Pending Approval。

查看：

```bash
$RUNTIME_DIR/bin/dataops-agent approvals
```

显式审批后才执行：

```bash
$RUNTIME_DIR/bin/dataops-agent approve <approval_id>
```

拒绝：

```bash
$RUNTIME_DIR/bin/dataops-agent reject <approval_id>
```

Approval 保存 queue/task-config fingerprint。Human review 期间平台状态变化时，执行返回 `PRECONDITION_FAILED`，需要重新发起请求和影响分析。

`创建任务并提交` 会先经过 V0.6 TaskPlanningService 和平台 validator；无效 TaskSpec 不产生 submit approval。

V0.7 尚不做统一写后 Verification，写后重新 Observe 并确认目标状态属于 V0.8。


## V0.8 Action Verification

V0.8 起，`dataops-agent approve <approval_id>` 在 Write MCP Tool 返回后会重新采集平台状态并执行 action-specific deterministic verification。

验证通过：

```text
approval_id=... status=executed
verification_status=verified attempts=2
```

如果写操作实际运行过，但后验状态未达到目标：

```text
approval_id=... status=verification_failed
```

此时 `execution_result` 和 `verification_result` 都会持久化，不能将操作视为成功。

配置：

```bash
export PLATFORM_AGENT_VERIFY_ATTEMPTS=5
export PLATFORM_AGENT_VERIFY_INTERVAL_SEC=1.0
```

V0.8 只重试 Observe，不重复执行 Write Tool。


## Agent Trace 与 Eval（V0.9）

查看最近请求：

```bash
dataops-agent traces --limit 50
```

查看单个 Trace：

```bash
dataops-agent trace <trace_id>
```

运行确定性 Agent 回归：

```bash
dataops-agent eval
```

Trace 默认写入 `$AIRFLOW_STATE_DIR/agent_traces`，Audit 默认写入 `$AIRFLOW_STATE_DIR/agent_audit/audit.jsonl`。写盘前会统一过滤 password/token/api_key/authorization 等敏感字段。

## Agent V1.0 本地硬化检查

不依赖真实 GPU 的本地完整控制链回归：

```bash
dataops-agent doctor
dataops-agent e2e
```

需要检查完整 Airflow/MCP/LangGraph/Docker/GPU Runtime 时：

```bash
dataops-agent doctor --strict
```

清理长期运行产生的 Trace/Audit：

```bash
dataops-agent observability-maintenance
```

详细说明见 `V1.0_HARDENING_E2E.md`。

## Agent V1.1 评测

### dependency-light 统一门禁

```bash
dataops-agent eval-aligned
```

该命令同时运行：

- chunk-level RAG retrieval eval
- Tool / Argument correctness regression
- environment-first Hard Task Success
- Task Planning regression
- curated security attacks

门槛读取 `eval/v1_1/thresholds.json`。

### 可选主流评测框架

```bash
dataops-agent eval-frameworks
dataops-agent eval-ragas
dataops-agent eval-deepeval
dataops-agent eval-promptfoo
```

Ragas/DeepEval 需要可选 Python 依赖及 Judge API Key；Promptfoo dynamic red team 需要 Promptfoo CLI 和攻击模型。详细数据集与指标见 `V1.1_EVALUATION_ALIGNMENT.md` 和 `eval/v1_1/README.md`。


## Agent V1.2 Gemini 使用

### Agent 主模型

```bash
export GEMINI_API_KEY='<ROTATED_KEY>'
export PLATFORM_AGENT_PROVIDER=gemini
export PLATFORM_AGENT_MODEL=gemini-3.7-flash
```

不要把 API Key 写入 YAML、源码、测试或命令历史共享文件。

### RAG 使用 Gemini Embedding

```bash
export PLATFORM_RAG_EMBED_PROVIDER=gemini
export PLATFORM_RAG_EMBED_MODEL=gemini-embedding-2
export PLATFORM_RAG_EMBED_DIM=768
```

首次构建：

```bash
dataops-agent knowledge build --force --json
```

检查：

```bash
dataops-agent knowledge status --json
```

Gemini 模式会持久化 document embedding sidecar；普通 Query 只计算 query embedding 并复用已缓存的 document vectors。

### A/B

```bash
PLATFORM_RAG_EMBED_PROVIDER=hash dataops-agent eval-aligned --json
PLATFORM_RAG_EMBED_PROVIDER=gemini dataops-agent eval-aligned --json
```

使用同一 V1.1 Golden Set 比较，不允许为了提高得分修改标注。
