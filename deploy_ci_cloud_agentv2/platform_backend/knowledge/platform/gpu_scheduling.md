# GPU 调度规则

## Reservation 与真实显存

GPU 分配不能只看 `nvidia-smi` 的 free memory，也不能只看平台 Reservation。平台同时检查实际显存与 Reservation，避免外部进程占用显存或平台内部并发分配造成冲突。

Reservation 记录 GPU ID、PID、task、dataset、stage、required memory 和 exclusive。死亡 PID 对应的 stale reservation 应在分配或查询过程中清理。

## 共享 GPU

Occ 等显存需求较小的 Stage 可以共享 GPU。共享前仍必须满足：当前实际剩余显存和已有 Reservation 扣减后的可用资源足够。

## 独占 GPU

Segment、OD 等模型加载峰值明显的 Stage 可以配置为独占。独占 Stage 只有在 GPU 没有其他 Reservation，并且实际显存接近空闲时才允许启动。即使总剩余显存看起来足够，只要存在其他 Reservation，也不能满足独占条件。

## 超时语义

GPU 排队时间不计入算法真正的 `timeout_min`。只有成功获得 GPU 并开始 Stage 进程后，才开始计算 Stage runtime timeout。

## 无物理 GPU 开发环境

开发环境使用 `SimulatedGPURuntime` 模拟硬件状态，但共享/独占、Reservation、stale cleanup、显存判断、GPU 选择和等待逻辑仍使用生产同一套 `GPUAllocator`。
