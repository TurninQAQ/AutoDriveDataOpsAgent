# 平台架构与状态来源

## 执行面与智能控制面

平台执行面由 Airflow、PostgreSQL、Docker、业务任务全局队列、GPU Reservation、Stage Validate、Soft Preemption 和 Recovery 组成。Agent 不能替代这些确定性逻辑。

Agent 的实时事实必须来自 MCP Tool。RAG 只能提供静态规则、运行手册和历史经验，不能把文档内容当作当前任务、当前 GPU 或当前容器状态。

## 业务任务模型

一个业务 Task 包含多个 Clip/Dataset。每个 Clip 对应一个 DagRun。DagRun 按 `pipeline_stages` 执行 precheck、parser、segment、map、od、occ、coloration 等 Stage。外层列表表示串行依赖，内层列表表示并行 Stage。

任务 YAML 还可以配置 `max_active_runs`、镜像、GPU ID、GPU Stage、显存需求、独占 GPU Stage、超时和业务优先级。

## 状态来源

任务实时状态主要来自：

- Airflow DagRun / TaskInstance
- 全局业务任务队列 active / queued / draining
- Stage checkpoint 与 Recovery 状态
- Docker inspect 与 Container 生命周期
- GPU Runtime 实际显存
- GPU Reservation
- Stage 日志和 Validate 结果

诊断时应优先用 `diagnose_task` 聚合证据，再按需要补充 `get_stage_logs`、`get_gpu_pool`、`get_task_detail` 等只读 Tool。
