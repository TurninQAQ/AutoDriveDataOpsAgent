# Docker Container 生命周期

## 唯一标识

每个 Stage 启动时生成唯一 container name，其中包含 task、stage、dataset、PID 和高精度时间戳。

## 精准归属判断

停止或清理时，不能只按镜像名或 Stage 名筛选。平台先按 task 前缀缩小容器范围，再结合 Docker inspect 中的 Name、Env、Cmd、Entrypoint、Bind/Mount 信息对 dataset 做完整 token 匹配。

完整 token 匹配用于避免 `clip_001` 错误匹配到 `clip_0010`。

## 清理语义

确认 Container ID 后执行 stop/remove，并再次 inspect 确认容器真正退出。正常停止失败时进行强制清理。相同 Container 标识和回收机制用于用户停止、Stage timeout、任务失败、任务删除和平台故障恢复。
