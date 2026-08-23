# Runbook：Container 泄漏或停止不准确

## 诊断

使用 `inspect_task_containers` 获取指定 task，必要时进一步指定 dataset。不要用镜像名或模糊字符串直接判断容器归属。

## 精准匹配

平台使用 task 前缀 + Docker inspect token 匹配。dataset 必须完整匹配 Name、Env、Cmd、Entrypoint、Bind/Mount 等证据，以避免 `clip_001` 与 `clip_0010` 混淆。

## 清理

平台停止流程会先 stop/remove，再 inspect 验证。普通停止失败时才进入强制清理。Agent V0.5 仍然是只读版本，只能诊断和给出建议，不能执行 stop/remove。
