# deploy_ci_cloud 开发交接说明

本文用于让下一个开发者快速接手 `deploy_ci_cloud`。修改任务提交、Airflow
部署、全局队列、优先级、抢占、定时提交、GPU 锁、删除逻辑前，先读这个
文件。

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 当前机器信息](#2-当前机器信息)
- [3. 关键文件](#3-关键文件)
- [4. Runtime 目录结构](#4-runtime-目录结构)
- [5. platform 命令语义](#5-platform-命令语义)
- [6. Airflow 配置和端口](#6-airflow-配置和端口)
- [7. 任务 YAML](#7-任务-yaml)
- [8. 任务类型和优先级](#8-任务类型和优先级)
- [9. 全局队列和软抢占](#9-全局队列和软抢占)
- [10. 定时提交](#10-定时提交)
- [11. GPU 锁](#11-gpu-锁)
- [12. 阶段执行和结果校验](#12-阶段执行和结果校验)
- [13. 停止、恢复、删除](#13-停止恢复删除)
- [14. 开发和验证](#14-开发和验证)
- [15. 人工冒烟测试](#15-人工冒烟测试)
- [16. 常见问题](#16-常见问题)
- [17. 最近新增能力](#17-最近新增能力)
- [18. 修改准则](#18-修改准则)

## 1. 项目定位

`deploy_ci_cloud` 是一个基于 Airflow 3.2 的数据处理平台。它把一批
dataset/clip 按 YAML 描述提交成独立的动态 DAG，并通过平台自己的
`./task` 命令管理任务生命周期。

当前主线流程：

```text
源码目录 -> ./platform install/deploy/start -> 任务 YAML -> ./task submit -> generated Airflow DAG
```

核心原则：

- `./platform install` 负责首次安装和配置生成：创建 venv、渲染
  `airflow.cfg`、写 runtime 环境、执行 `airflow db migrate`、部署文件。
- `./platform deploy` 只同步 DAG/scripts/platform_core/config 到 runtime，不改
  `airflow.cfg`、不重建 venv、不迁移数据库、不重启 Airflow。
- 源码目录里的 `./task` 只是转发入口，真实执行的是
  `$RUNTIME_DIR/bin/task`。
- 每次任务提交都会生成一个独立 DAG：
  `batch_pipeline_universal_<task_name>`。
- 全局排队、优先级、软抢占、定时提交、停止、恢复、删除，都是平台逻辑，
  不要只按 Airflow 原生调度理解。

## 2. 当前机器信息

源码目录：

```text
/home/cidi/project/one/deploy_ci_cloud
```

当前开发分支：

```text
feature/0724-deploy-optimize
```

远程仓库：

```text
origin http://172.16.201.6/BCD/deploy_ci_cloud.git
```

`runtime.path` 当前配置：

```text
172.16.34.131=/home/cidi/deploy_ci_cloud_runtime
```

当前 Airflow 页面：

```text
http://172.16.34.131:8081
```

日常操作先进入源码目录：

```bash
cd /home/cidi/project/one/deploy_ci_cloud
```

## 3. 关键文件

入口脚本：

- `platform`：源码侧平台命令，读取 `runtime.path`，install 时读取
  `.env`，deploy/start/stop/restart/status 时读取 runtime 的
  `platform.env`。
- `task`：源码侧任务命令，读取 `runtime.path` 后转发到
  `$RUNTIME_DIR/bin/task`。

平台脚本：

- `scripts/deploy_ci_cloud.sh`：把 DAG、脚本、工具、YAML 配置复制到
  runtime。
- `scripts/airflow_ctl.sh`：启动、停止、检查 Airflow 组件。
- `scripts/manage_task.sh`：runtime 里的任务 shell wrapper。
- `scripts/task_manager.py`：任务提交、队列、优先级、定时、删除等核心逻辑。

DAG 逻辑：

- `dags/batch_pipeline_universal.py`：公共 DAG 和 generated DAG 运行时函数。
- `dags/templates/batch_pipeline_universal_template.py`：动态 DAG 模板。
- `config/airflow_local_settings.py`：Airflow 启动 hook，用于接管平台任务
  DAG 的页面删除按钮。

任务 YAML：

- `config/task_submit_template.yaml`：主任务 YAML 模板。
- `config/task_full_serial_template.yaml`：全串行任务模板。
- `config/task_types.yaml`：任务类型和默认优先级。
- `scripts/generate.py`：生成 YAML 的短入口。
- `scripts/tools/genarate_dataset_config.py`：真正的 YAML 生成逻辑。注意文件名
  是 `genarate`，不是 `generate`。

测试：

- `tests/test_task_priority_queue.py`：优先级队列和动态优先级。
- `tests/test_task_submit_scheduler.py`：定时提交扫描器和
  `schedule list/remove`。
- `tests/test_platform_restart_cleanup.py`：restart 清理队列、锁和定时记录。
- `tests/test_exclusive_gpu_stages.py`：独占 GPU 配置和运行时锁。
- `tests/test_airflow_page_delete_hook.py`：Airflow 页面删除 hook。
- `tests/test_dag_preemption_queue.py`：DAG 侧抢占队列行为。

用户文档：

- `README.md`：部署概览。
- `usage_guide.md`：当前完整使用教程。
- `deploy_guide.md`：新机器从 0 部署流程。

## 4. Runtime 目录结构

当前机器 runtime：

```text
/home/cidi/deploy_ci_cloud_runtime/
  airflow/
    airflow.cfg
    config/airflow_local_settings.py
    dags/data_center/
      batch_pipeline_universal.py
      generated/
      templates/
    logs/
  bin/
    platform
    task
  config/
    platform.env
  opt_airflow/
    config/
      task_types.yaml
      tasks/<task_name>/datasets_config.yaml
    scripts/
      task_manager.py
      manage_task.sh
      airflow_ctl.sh
  state/task_queue/
    queue.lock
    scheduled_submits.lock
  venv/
```

锁路径：

```text
$RUNTIME_DIR/state/task_locks/active_task.lock
$RUNTIME_DIR/state/gpu_locks/gpu_<id>.lock
```

正常开发不要手改 runtime 文件。改源码，然后执行：

```bash
./platform deploy
```

### 日志位置

平台日志根目录：

```text
$RUNTIME_DIR/airflow/logs/
```

当前机器实际路径：

```text
/home/cidi/deploy_ci_cloud_runtime/airflow/logs/
```

核心组件日志：

```text
$RUNTIME_DIR/airflow/logs/scheduler.log
$RUNTIME_DIR/airflow/logs/dag_processor.log
$RUNTIME_DIR/airflow/logs/triggerer.log
$RUNTIME_DIR/airflow/logs/api_server.log
$RUNTIME_DIR/airflow/logs/task_submit_scheduler.log
```

常用查看命令：

```bash
tail -n 200 /home/cidi/deploy_ci_cloud_runtime/airflow/logs/scheduler.log
tail -n 200 /home/cidi/deploy_ci_cloud_runtime/airflow/logs/dag_processor.log
tail -n 200 /home/cidi/deploy_ci_cloud_runtime/airflow/logs/triggerer.log
tail -n 200 /home/cidi/deploy_ci_cloud_runtime/airflow/logs/api_server.log
tail -n 200 /home/cidi/deploy_ci_cloud_runtime/airflow/logs/task_submit_scheduler.log
```

实时跟随：

```bash
tail -f /home/cidi/deploy_ci_cloud_runtime/airflow/logs/scheduler.log
tail -f /home/cidi/deploy_ci_cloud_runtime/airflow/logs/dag_processor.log
tail -f /home/cidi/deploy_ci_cloud_runtime/airflow/logs/task_submit_scheduler.log
```

任务 DAG 的具体 task log 在：

```text
$RUNTIME_DIR/airflow/logs/dag_id=batch_pipeline_universal_<task_name>/
```

查某个任务的 attempt 日志：

```bash
find /home/cidi/deploy_ci_cloud_runtime/airflow/logs \
  -path '*batch_pipeline_universal_<task_name>*' \
  -type f \
  -name 'attempt=*.log'
```

dag processor 还有按日期分的细日志：

```text
$RUNTIME_DIR/airflow/logs/dag_processor/latest
```

如果只是排查平台组件是否异常，先看
`scheduler.log`、`dag_processor.log`、`api_server.log` 和
`task_submit_scheduler.log`，通常不需要额外开发日志查看功能。

## 5. platform 命令语义

```bash
./platform install
./platform deploy
./platform start
./platform stop
./platform restart
./platform status
```

语义：

- `install`：首次安装或明确需要重新生成配置时使用。会创建 runtime、venv、
  `airflow.cfg`、密码文件、`platform.env`，应用 Airflow runtime 补丁，
  执行 DB migrate，并部署文件。
- `deploy`：日常同步代码。只复制 DAG/scripts/platform_core/config；如果有 active
  DagRun 或平台任务容器，会拒绝部署；不重渲染 `airflow.cfg`，不修补 venv。
- `start`：启动缺失组件；已运行组件不会重启。
- `stop`：停止平台组件。
- `restart`：故障恢复入口。会停止组件、停止 active/queued/scheduled 任务、
  停容器、清 GPU reservation、清 task lock、清 task queue，再启动组件。
- `status`：输出 runtime 路径、磁盘、组件 PID、Airflow URL、Public API
  Health、Execution API Health、Platform Health 和 Airflow health JSON。
  Platform Health 会额外展示
  `task submit scheduler` 是否运行、定时提交状态文件是否可读、调度组件日志
  最近更新时间。

`restart` 会清理任务状态，不是普通部署命令。只有接受清理当前任务时再用。

正式帮助入口：

```bash
./platform --help
./task --help
```

`./platform --help` 重点看日常更新、首次安装、组件管理、故障恢复和日志路径。
`./task --help` 重点看提交任务、定时任务、停止/恢复/删除和优先级调整。
不要把 `-help` 当成正式入口。

## 6. Airflow 配置和端口

`airflow.cfg` 由 `./platform install` 根据 `config/airflow.cfg.base` 或
`recover/airflow.cfg` 渲染。

端口相关配置来自 `.env`：

```text
AIRFLOW_PORT
AIRFLOW_API_BASE
AIRFLOW_EXECUTION_API_PORT
AIRFLOW_EXECUTION_API_BASE
AIRFLOW_PUBLIC_URL
AIRFLOW__API__BASE_URL
AIRFLOW__CORE__EXECUTION_API_SERVER_URL
```

install 会更新这些关键 cfg：

- `[api] port`
- `[api] base_url`
- `[core] execution_api_server_url`
- DAG、plugins、logs、数据库、auth users、password file 等路径。

规则：

- 改端口要改 `.env` 后跑 `./platform install`。
- 不要把 8080/8081 修补逻辑塞进 `deploy`。
- 如果 base cfg 缺少必须存在的配置项，应报错，不要加隐式兜底。
- runtime 的 `platform.env`、`airflow.cfg`、`AIRFLOW_API_BASE` 和
  `AIRFLOW_EXECUTION_API_BASE` 必须一致。
- public API 只启动 core app；execution API 单独监听
  `AIRFLOW_EXECUTION_API_PORT`。

## 7. 任务 YAML

常用结构：

```yaml
task_type: reprocess

pipeline_stages:
  - precheck
  - parser
  - segment
  - map
  - [od, occ]
  - coloration

max_active_runs: 5
task_exclusive: true
task_lock_wait_interval_sec: 10
preempt_grace_timeout_min: 60

gpu_ids: "5,6,7,8,9"
gpu_stages: "segment,od,occ"
exclusive_gpu_stages: "segment,od"
exclusive_gpu_idle_used_max_mb: 512
gpu_stage_memory_mb:
  segment: 24000
  od: 24000
  occ: 4000

datasets:
  - dataset_name: clip_001
    dataset_path: /home/cidi/data/test/test1
    image_parser: 172.16.201.100:5000/data_parser:tag
    image_segment: 172.16.201.100:5000/sam31:tag
    image_map: 172.16.201.100:5000/offline_mapping:tag
    image_od: 172.16.201.100:5000/label_od:tag
    image_occ: 172.16.201.100:5000/label_occ:tag
    image_coloration: 172.16.201.100:5000/pointcloud_coloration:tag
    timeout_min: 60
```

规则：

- `pipeline_stages` 定义阶段顺序。
- 普通列表项串行执行。
- 内层列表表示并行阶段，例如 `[od, occ]`。
- 只需要提供用到的 `image_<stage>`。
- `dataset_path` 是 record 根目录，实际数据目录是
  `<dataset_path>/<dataset_name>`。
- `max_active_runs` 控制同一个任务 DAG 内最多并发多少个 clip。
- `task_exclusive: true` 才会进入平台全局队列、优先级和抢占逻辑。
- `exclusive_gpu_stages` 不写时，所有 `gpu_stages` 默认独占。
- `exclusive_gpu_stages: ""` 表示关闭独占 GPU。
- `exclusive_gpu_idle_used_max_mb` 默认 512 MB。

生成 YAML：

```bash
./scripts/generate.py /path/to/record_dir -o /tmp/task.yaml
```

提交 YAML：

```bash
./task submit --name reprocess --yaml /tmp/task.yaml
```

## 8. 任务类型和优先级

配置文件：

```text
config/task_types.yaml
```

当前内容：

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

语义：

- 数字越小，优先级越高。
- 抢占条件是 `new_task.priority < active_task.priority`。
- YAML 里直接写 `priority` 时，优先级以 `priority` 为准。
- 只写 `task_type` 时，从 `config/task_types.yaml` 读取默认优先级。
- `task_type` 和 `priority` 都不写时，使用 `default_priority: 100`。
- 写了不存在的 `task_type` 会提交失败。

调整已提交任务优先级：

```bash
./task priority <task_name> --priority <number>
```

这个命令会改 runtime 里的任务 YAML：

```text
$RUNTIME_DIR/opt_airflow/config/tasks/<task_name>/datasets_config.yaml
```

然后刷新队列。如果该任务新优先级高于当前 active 任务，会触发软抢占。

新增任务类型：

```yaml
task_types:
  release:
    priority: 10
  reprocess:
    priority: 20
  test:
    priority: 50
  debug:
    priority: 80
  scheduled:
    priority: 30
```

改完后：

```bash
./platform deploy
```

删除任务类型前，确认没有未完成任务还引用该 `task_type`，否则后续刷新优先级
时会报未知类型。

## 9. 全局队列和软抢占

队列状态：

```text
$RUNTIME_DIR/state/task_queue/queue.lock
```

基本 schema：

```json
{"active": null, "queue": [], "version": 2}
```

队列排序：

```text
priority 升序 -> queued/submitted 时间升序 -> task_name 升序
```

提交行为：

- 没有 active exclusive 任务时，新任务直接 active 并触发 DagRun。
- 当前 active 优先级更高或相等时，新任务进入 queue。
- 新任务优先级更高时，只发出 `preempt_requested`，当前 active 任务进入
  `draining`，高优先级任务继续留在 queue，等待阶段边界接管。

软抢占规则：

- 当前 active clip 先跑完正在执行的 stage 和对应 validate。
- validate 成功后记录 `_platform_resume_from_stage`，旧 run 跳过后续 stage。
- 所有 draining run 到达边界后，低优先级任务以恢复点重新入队并暂停，
  高优先级任务变为 active。
- 高优先级任务完成后，低优先级任务从恢复点继续，不从头重跑已完成 stage。
- 被抢占任务恢复 active 后，原始旧 DagRun 必须保持跳过、不计数；实际补跑只
  能由 recovery DagRun 承担，避免旧 run 和 recovery run 同时执行业务阶段。
- recovery DagRun 不在 drain 阶段提前触发；`queue.lock` 是平台队列的唯一
  状态源。被抢占任务重新成为 active 时，才触发对应 recovery conf。
- stop/delete 当前 active 任务时，也走同一套队列推进逻辑；如果 queue 里有
  被抢占任务，它会自动接管并从 `_platform_resume_from_stage` 继续。
- `preempt_grace_timeout_min` 只作为异常兜底，用于锁泄漏或旧抢占状态残留时的
  强制清理，不是正常抢占路径。

关键实现：

- `scripts/task_manager.py`
  - `register_task_queue()`
  - `mark_runs_completed_in_queue()`
  - `set_task_priority()`
  - `remove_task_from_queue()`
  - `pending_activation_plan()`
  - `run_matches_pending_conf()`
- `dags/batch_pipeline_universal.py`
  - `record_stage_checkpoint_after_validate()`
  - `finalize_drained_run_and_maybe_advance_queue()`
  - `complete_task_run_and_advance_queue()`
  - `prepare_queued_task_activation()`
  - `pending_activation_plan()`
  - `preempt_cleanup_plan()`

## 10. 定时提交

注册定时任务：

```bash
./task submit --name reprocess --yaml /tmp/reprocess.yaml --schedule "2026-07-30 23:00"
```

支持时间格式：

```text
YYYY-MM-DD HH:MM
YYYY-MM-DD HH:MM:SS
YYYY-MM-DDTHH:MM
YYYY-MM-DDTHH:MM:SS
```

状态文件：

```text
$RUNTIME_DIR/state/task_queue/scheduled_submits.lock
```

后台组件：

```text
task submit scheduler
```

由 `./platform start` 拉起，内部运行：

```bash
task_manager.py submit scheduler
```

默认每 30 秒扫描一次：

```text
AIRFLOW_SUBMIT_SCHEDULER_INTERVAL_SEC=30
```

到时间后，扫描器把记录标记为 `running`，执行一次普通 submit，然后标记为
`submitted`，并写入 `result_task_name` 和 `result_dag_id`。

管理命令：

```bash
./task schedule list
./task schedule list --all
./task schedule list --status scheduled
./task schedule list --json
./task schedule remove <schedule_id>
./task schedule remove <schedule_id> --yes
```

规则：

- `schedule list` 默认显示 pending，也就是 `scheduled/running`。
- `--all` 显示 `submitted/failed/stopped/removed` 等历史。
- `remove` 不加 `--yes` 是 dry-run。
- 只有 `status=scheduled` 的记录能取消。
- `running/submitted` 不允许取消，因为扫描器可能已经创建真实任务。

## 11. GPU 锁

GPU 锁文件：

```text
$RUNTIME_DIR/state/gpu_locks/gpu_<id>.lock
```

每个 GPU stage 运行时会写 reservation，包含 PID、任务、dataset、stage、
预留显存和是否独占。

共享模式：

- 多个非独占阶段可以共享 GPU。
- 判断逻辑是 `nvidia-smi free_mb - active_reserved_mb >= required_mb`。

独占模式：

- 当前 stage 在 `exclusive_gpu_stages` 中时启用。
- 目标 GPU 必须没有 active reservation。
- 目标 GPU 实际已用显存必须小于等于 `exclusive_gpu_idle_used_max_mb`。
- 如果某 GPU 已有独占 reservation，非独占任务也不能使用它。

关键函数：

```text
dags/batch_pipeline_universal.py::acquire_gpu_from_pool()
scripts/task_manager.py::normalize_gpu_config()
scripts/task_manager.py::normalize_exclusive_gpu_config()
```

## 12. 阶段执行和结果校验

阶段脚本：

```text
scripts/run_precheck.sh
scripts/run_parser.sh
scripts/run_segment.sh
scripts/run_map.sh
scripts/run_od.sh
scripts/run_coloration.sh
scripts/run_occ.sh
scripts/run_qc.sh
```

DAG 通过 `dags/batch_pipeline_universal.py::run_shell_script()` 执行阶段。

每个阶段成功后，validate 前会处理：

```text
<dataset_path>/<dataset_name>/results_<stage>.json
```

权限修复逻辑：

- 先在宿主机尝试 `chmod`，给结果文件加可读位。
- 如果文件是 root 创建导致宿主机 chmod 失败，则用同一个阶段镜像启动临时
  root 容器，对挂载目录里的结果文件执行 `chmod a+r`。
- 如果结果文件不存在，不吞错误，仍由 `scripts/validate_json.py` 报错。

不要把 `validate_json.py` 改成忽略读不到文件。读不到文件时无法判断算法是否
真的成功，跳过会掩盖问题。

## 13. 停止、恢复、删除

命令：

```bash
./task stop <task_name> --yes
./task stop <task_name> <clip_1> <clip_2> --yes
./task resume <task_name>
./task resume <task_name> <clip_1> <clip_2>
./task delete <task_name> --yes
```

删除逻辑会做：

- pause DAG。
- 标记 active DagRun/task instance failed。
- 停止匹配的 Docker 任务容器，除非传 `--no-stop-containers`。
- 等待或清理 GPU reservation。
- 清 task lock。
- 从 task queue 移除。
- 如果删除的是 active 任务，会自动激活 queue 中的下一个任务并触发它的
  pending/recovery DagRun。
- 删除 generated DAG 文件。
- 删除 runtime 任务配置目录。
- 删除 Airflow DAG metadata。

关键实现：

```text
scripts/task_manager.py::delete_task_by_name()
scripts/task_manager.py::delete_task()
```

Airflow 页面删除按钮：

- `config/airflow_local_settings.py` patch 了 Airflow 原生 delete DAG API。
- generated 平台任务 DAG 会走平台删除逻辑，等价于：
  `./task delete <task_name> --yes`。
- 公共 DAG `batch_pipeline_universal` 禁止从页面删除。
- 普通非平台 DAG 继续走 Airflow 原生删除。
- `AIRFLOW_PLATFORM_DELETE_BYPASS=1` 只用于受控调试。

## 14. 开发和验证

改代码前：

```bash
git status --short
```

基础检查：

```bash
git diff --check
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python -m py_compile scripts/task_manager.py
bash -n task
bash -n scripts/manage_task.sh
```

定向测试：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python tests/test_task_priority_queue.py
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python tests/test_task_submit_scheduler.py
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python tests/test_platform_restart_cleanup.py
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python tests/test_exclusive_gpu_stages.py
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python tests/test_airflow_page_delete_hook.py
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python tests/test_dag_preemption_queue.py
```

不要默认使用 shell 里的 `python3`。当前 conda 环境可能是 Python 3.8，会因为
`zoneinfo` 等依赖失败。优先使用 runtime venv：

```text
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python
```

部署修改：

```bash
./platform deploy
./platform status
```

如果改了 Airflow API server 启动时才加载的逻辑，例如
`config/airflow_local_settings.py`，`deploy` 只会复制文件，不会重启 API
server。只有接受清理任务状态时才用：

```bash
./platform restart
```

## 15. 人工冒烟测试

检查平台：

```bash
./platform status
```

检查定时任务：

```bash
./task schedule list
./task schedule list --all
```

注册并取消未来定时任务：

```bash
./task submit --name sched_probe --yaml /tmp/task.yaml --schedule "2099-01-01 00:00"
./task schedule list
./task schedule remove <schedule_id>
./task schedule remove <schedule_id> --yes
./task schedule list
./task schedule list --all
```

优先级/抢占测试建议：

- 用一个慢的 local stage 做低优先级任务。
- 用一个短的 local stage 做高优先级任务。
- 先提交低优先级任务，再提交高优先级任务。
- 高优先级输出应包含 `queue_state=preempt_requested`。
- 高优先级完成后，低优先级应从 `_platform_resume_from_stage` 继续。

清理测试任务：

```bash
./task delete <task_name> --yes
```

确认无残留：

```bash
sed -n '1,200p' /home/cidi/deploy_ci_cloud_runtime/state/task_queue/queue.lock
find /home/cidi/deploy_ci_cloud_runtime/opt_airflow/config/tasks -maxdepth 1 -mindepth 1 -type d -printf '%f\n'
find /home/cidi/deploy_ci_cloud_runtime/airflow/dags/data_center/generated -maxdepth 1 -type f -printf '%f\n'
docker ps --filter name=airflow-task --format '{{.Names}}'
```

期望队列：

```json
{"active": null, "queue": [], "version": 2}
```

## 16. 常见问题

端口不一致：

- 端口在 `.env` 里配置 `AIRFLOW_PORT` 和 `AIRFLOW_EXECUTION_API_PORT`。
- 运行 `./platform install` 生成 `airflow.cfg`。
- 不要在 `deploy` 里加 8080/8081 修补。
- 确认 runtime 的 `platform.env`、`airflow.cfg`、API base、execution API
  URL 一致。

Airflow UI/API 卡住：

- Airflow 3.2.0 的 `/ui/grid/ti_summaries/{dag_id}` streaming endpoint
  需要 runtime 补丁，避免 UI Grid 刷新留下 `idle in transaction`。
- 该问题根因是 UI streaming session 泄漏，不是 task submit scheduler 心跳。
- 任务执行心跳必须走 execution API，不要回落到 public API。
- 验证：

```bash
/home/cidi/deploy_ci_cloud_runtime/venv/bin/airflow config get-value core execution_api_server_url
/home/cidi/deploy_ci_cloud_runtime/venv/bin/python scripts/patch_airflow_grid_session.py --check
```

- 用 `./platform status` 看 api server、execution api server、scheduler、
  triggerer、dag processor 是否 healthy；同时看 Platform Health 里的
  `task submit scheduler`、
  schedule file 和 latest scheduler log。
- 如果组件异常且可以清任务，使用 `./platform restart`。

删除超时或残留：

- 优先用 `./task delete <task_name> --yes`。
- 检查容器：
  `docker ps --filter name=airflow-task-<task_name>`。
- 检查队列：
  `$RUNTIME_DIR/state/task_queue/queue.lock`。
- 检查 generated DAG 和 task config 目录。

GPU 看起来被打满：

- 先用 `nvidia-smi` 看真实显存。
- 再看 `$RUNTIME_DIR/state/gpu_locks` 是否有 stale reservation。
- `./platform restart` 会清 GPU reservation。

结果 JSON 权限失败：

- 镜像用 root 写出的 `results_<stage>.json` 可能导致 validate 读不到。
- 当前 DAG 已在阶段成功后、validate 前修可读权限。
- 已经失败的历史任务需要人工确认文件权限后再重跑。

未知 stage：

- `pipeline_stages` 只能写已有 runtime 实现的阶段。
- 不想跑某阶段，就从 YAML 的 `pipeline_stages` 删除。

未知 task type：

- 在 `config/task_types.yaml` 添加类型。
- 执行 `./platform deploy`。
- 不要在未完成老任务还引用某类型时删除该类型。

## 17. 最近新增能力

任务类型和优先级：

- 配置：`config/task_types.yaml`
- YAML：`task_type`、`priority`
- CLI：`./task priority <task_name> --priority <number>`

软抢占：

- 高优先级任务可抢占低优先级 active 任务。
- 已成功 clip 不重复跑。
- 被抢占 clip 从已验证 stage 后的下一个 stage 继续。
- `preempt_grace_timeout_min` 只用于异常兜底清理。

定时提交：

- 注册：`./task submit --schedule`
- 后台组件：`task submit scheduler`
- 状态文件：`scheduled_submits.lock`
- 管理：`./task schedule list/remove`

独占 GPU 阶段：

- YAML 字段：`exclusive_gpu_stages`
- 字符串形式，当前推荐 `"segment,od"`。
- 不写时运行时所有 `gpu_stages` 默认独占。
- 空字符串关闭独占。
- 空闲阈值：`exclusive_gpu_idle_used_max_mb`。

Airflow 页面删除 hook：

- 平台任务 DAG 的页面删除按钮走平台 delete。
- 公共 DAG 禁止页面删除。

结果 JSON 权限修复：

- 阶段成功后、validate 前修 `results_<stage>.json` 可读权限。

## 18. 修改准则

- 先读源码，不要凭记忆改。
- 队列/定时/删除/GPU 相关逻辑优先看 `scripts/task_manager.py`。
- DAG 运行时、stage、GPU 分配、抢占完成推进优先看
  `dags/batch_pipeline_universal.py`。
- 配置渲染保持在 `install`，不要放到 `deploy`。
- generated DAG 的删除必须走平台 delete，不要只依赖 Airflow native delete。
- 不要直接手改 queue/lock 文件，除非是明确的故障排查。
- 尽量保持旧 YAML 兼容。
- 改队列、定时、删除、GPU、DAG runtime 时必须补或跑对应测试。
- 部署后要通过源码目录的 `./task` 验证，而不是只直接调用 Python 模块。
