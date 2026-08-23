# Runbook：GPU Stage 长时间等待

## 适用现象

Segment、OD、Occ 等 GPU Stage 处于等待状态，Airflow 本身未报告明确失败。

## 诊断顺序

1. 用 `get_task_detail` 确认 Stage 的 `required_memory_mb` 和是否 exclusive。
2. 用 `get_gpu_pool` 查看每张 GPU 的 total/used/free memory 与 Reservation。
3. 用 `diagnose_task` 确认该 Clip 当前 DagRun/TaskInstance 是否真的处于资源等待，而不是 Stage 已失败。
4. 检查 stale reservation 是否已经清理。

## 常见根因

- 所有 GPU 的实际 free memory 都低于 Stage 显存需求。
- 独占 Stage 遇到其他 Reservation，即使 free memory 足够也不能启动。
- GPU 上存在平台外进程，实际显存不满足空闲阈值。
- Reservation 对应 PID 已死亡但状态未及时清理。

## 注意

GPU 等待时间不计入算法 Stage runtime timeout。不要仅因为等待时间长就判断 Stage timeout。
