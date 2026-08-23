# Runbook：Airflow 组件或 Metadata Database 异常

## 适用现象

多个无关业务任务同时出现调度延迟、API 查询失败、DagRun/TaskInstance 状态读取异常。

## 诊断顺序

1. 使用 `get_platform_health` 检查 Airflow API、Queue、Docker、Task Config Root 和 GPU Runtime。
2. 如果 Airflow health 不正常，优先处理平台组件问题，不要逐个任务猜测根因。
3. 检查 Scheduler / API Server 日志以及 PostgreSQL 连通性。

## 背景

平台多任务化后 DagRun 和 TaskInstance 数量增加，Metadata Database 并发访问量明显上升。平台已经从 SQLite 迁移到 PostgreSQL，用于避免 SQLite 在该并发场景下成为稳定性瓶颈。
