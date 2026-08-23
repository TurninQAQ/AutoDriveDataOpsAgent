# Runbook：Stage 执行失败或 Validate 失败

## 算法进程失败

如果 `run_<stage>` TaskInstance failed，应查看对应 Stage 日志。常见证据包括非零退出码、Python exception、CUDA OOM、输入数据错误、镜像或挂载问题。

## Validate 失败

算法进程退出成功不代表 Stage 完成。必须通过 Validate 才能形成 Stage checkpoint。

如果 Validate 失败：

- 当前 Stage 不应记为安全恢复点。
- Soft Preemption 不能从该 Stage 之后恢复。
- Recovery 应重新执行该 Stage，而不是跳过。

## GPU OOM

日志包含 `CUDA out of memory`、`CUDA OOM` 或明确的 GPU OOM 文本时，优先检查 Stage 显存需求、独占策略、当前 Reservation 和实际显存，而不是直接判断 Airflow Scheduler 故障。
