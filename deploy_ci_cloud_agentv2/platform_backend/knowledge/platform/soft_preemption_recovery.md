# 任务优先级、软抢占与 Recovery

## 全局业务任务优先级

平台在 Airflow 之上维护业务任务级全局队列。队列状态包括 active、queued、draining，并按照 priority、提交时间和 task name 排序。priority 数值越小表示优先级越高。

## 阶段边界软抢占

高优任务到达后，不直接杀掉低优任务正在运行的 Stage。低优任务先从 active 进入 draining，并记录 preempt request。当前 Stage 允许正常执行并完成 Validate。

只有 Stage 执行成功且 Validate 成功，才把该位置视为安全恢复点。对于多 Clip 并行任务，需要等待本次需要排空的所有 Run 到达安全 Stage 边界或终态，才允许高优任务真正成为 active。

## Recovery

低优任务被抢占时记录已通过 Validate 的 Stage checkpoint。高优任务结束后，原任务重新 active，并创建 Recovery DagRun。Recovery DagRun 使用 `_platform_resume_from_stage` 等恢复信息，从断点位置继续，而不是从头重算。

Validate 失败的 Stage 不能形成 checkpoint，因此 Recovery 不能跳过该 Stage。
