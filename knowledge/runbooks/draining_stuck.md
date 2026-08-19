# Runbook：任务长期处于 draining

## 适用现象

业务任务进入 draining 后，长时间未被高优任务完全替换或未进入 Recovery。

## 诊断顺序

1. `get_queue_state` 确认任务确实为 draining，并确认触发抢占的高优任务。
2. `diagnose_task` 检查涉及的 DagRun 和 TaskInstance。
3. 确认正在运行的 Stage 是否已经结束。
4. 确认 Stage 结束后 Validate 是否成功。
5. 检查所有本次需要排空的 Clip 是否都达到安全边界或终态。

## 常见根因

- 某个长 Stage 仍在运行，因此软抢占不会强杀。
- Stage 运行成功但 Validate 失败，不能形成 checkpoint。
- 多 Clip 中仍有一个 Run 未达到安全边界。
- 某 Run 状态异常，导致 drain barrier 迟迟不能满足。

## 原则

draining 本身不代表故障。它是阶段边界软抢占的中间状态，必须结合当前 Stage、Validate 和多 Run 排空状态判断。
