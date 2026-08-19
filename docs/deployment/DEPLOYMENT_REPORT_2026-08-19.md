# AutoDrive DataOps Agent V1.2 本次本地部署报告

## 1. 报告信息

| 项目 | 内容 |
|---|---|
| 部署日期 | 2026-08-19 |
| 部署类型 | 本地单机部署、模拟 GPU、Mock Stage、Airflow 全运行时 |
| 源码目录 | `/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agent_v1.2` |
| Runtime 目录 | `/home/ubuntu/project/autodrive_dataops_runtime` |
| Git 分支 | `main` |
| 项目版本 | AutoDrive DataOps Agent V1.2，附本地 Runtime 集成修复 |
| 部署目标 | Airflow + Platform Core + MCP + Agent + RAG + 任务提交/调度/审批闭环 |

本报告记录的是本机实际执行结果，不是只根据部署指南推演的结果。报告不包含任何真实 API Key、数据库密码或 SSH 私钥。

## 2. 总体结论

核心平台已经部署完成并保持运行：`doctor --strict` 通过，Airflow、Execution API、任务提交调度器、MCP 和 Docker 运行依赖均正常；依赖轻量回归测试 163 项全部通过，端到端控制链 10/10 步骤通过，V1.1 对齐评测门禁通过。

Gemini 的网络和地区限制已经通过代理解决。当前 `新加坡-优化-GPT` 节点下，用户提供形态的 Gemini 简单请求连续 3 次返回 `HTTP 200`。但是完整 Agent 请求会触发多次结构化生成和 Embedding 调用，期间出现 Gemini 上游 `429 Too Many Requests` 和 `503 UNAVAILABLE`，因此“真实 Gemini Agent + Gemini Embedding 的稳定生产验收”暂未宣称通过。

当前状态可分为：

| 能力 | 状态 | 说明 |
|---|---|---|
| Platform Core | 通过 | 任务、队列、GPU Reservation、软抢占和恢复链路可运行 |
| Airflow 全运行时 | 通过 | Scheduler、DAG Processor、Triggerer、API Server、Execution API 均健康 |
| MCP | 通过 | 官方 MCP 握手成功，16 个 Tool 可发现，`get_gpu_pool` 调用成功 |
| Docker | 通过 | Docker daemon 和 Python Docker 依赖正常；拉取 Docker Hub `hello-world` 时发生外部 registry 超时 |
| 模拟 GPU | 通过 | 设备 `0/1/2` 可由 `SimulatedGPURuntime` 提供 |
| Mock Stage | 通过 | 本地端到端测试使用 Mock Stage，默认结果为 `success` |
| RAG Hash 基线 | 通过 | V1.1 对齐评测全部满足门槛 |
| Gemini 直连 | 通过 | 代理切换后简单生成请求连续返回 200 |
| Gemini 完整 Agent | 部分通过 | 请求能抵达 API，但结构化规划/多次 Embedding 链路遇到上游 429/503 |
| Gemini 向量索引 | 未完成 | 语料索引新鲜，Gemini 向量索引因外部限流未建立 |

## 3. 部署目录和网络入口

### 3.1 源码与 Runtime

源码仓库已经上传到：

`https://github.com/TurninQAQ/AutoDriveDataOpsAgent`

本机目录关系如下：

```text
/home/ubuntu/project/AutoDriveDataOpsAgent/
├── AutoDrive_DataOps_Agent_V1.2_Local_Deployment_Guide.md
├── deploy_ci_cloud_agent_v1.2.zip
└── deploy_ci_cloud_agent_v1.2/          # Git 源码仓库

/home/ubuntu/project/autodrive_dataops_runtime/  # 实际运行时
├── config/platform.env
├── opt_airflow/
├── airflow/
├── state/
└── venv/
```

`runtime.path` 已按本机环境设置为：

```text
10.1.0.4=/home/ubuntu/project/autodrive_dataops_runtime
```

### 3.2 Airflow 和 Platform 地址

| 服务 | 地址 | 验收结果 |
|---|---|---|
| Airflow Public API | `http://10.1.0.4:8080` | 健康 |
| Execution API | `http://127.0.0.1:8081` | 健康 |
| Gemini/Mihomo HTTP Proxy | `http://127.0.0.1:7890` | 可用 |
| Mihomo Controller | `http://127.0.0.1:9090` | 可切换 `PROXY` 节点 |

Airflow 的 Runtime 组件均在 `/home/ubuntu/project/autodrive_dataops_runtime` 下运行，没有把运行时虚拟环境或数据库文件提交到 GitHub。

## 4. 实际安装的软件和依赖

已安装或确认可用的系统依赖包括：

- Git、curl、Vim
- Python 3.12.3、`python3-venv`、`python3-pip`
- PostgreSQL 客户端/服务端依赖
- Docker Engine 29.1.3
- lsof

Runtime 虚拟环境中确认的关键 Python 依赖：

| 依赖 | 版本/状态 | 用途 |
|---|---|---|
| Apache Airflow | 3.2.0 | DAG 调度与执行 API |
| `mcp` | 2.0.0 | Platform MCP Server/Client |
| `langgraph` | 1.2.11 | Agent workflow |
| `langchain-openai` | 1.5.2 | 兼容性依赖 |
| `google-genai` | 2.18.1 | Gemini Agent 和 Embedding |
| pytest / pytest-asyncio | 已安装 | 自动化回归和异步测试 |
| trio | 已安装 | MCP CLI/协议客户端支持 |

由于部署时配置的镜像源不提供 Apache Airflow 3.2.0，Airflow 依赖最终从官方 PyPI 安装；这不是源码修改，也没有把安装缓存放入 Git 仓库。

## 5. 数据库和权限配置

已创建本地 PostgreSQL 数据库及角色：

- 数据库：`autodrive_agent`
- 角色：`autodrive_agent`
- Airflow metadata database migration：已完成

Runtime 配置文件为：

```text
/home/ubuntu/project/autodrive_dataops_runtime/config/platform.env
```

该文件包含 API Key 和数据库连接等敏感配置，权限已限制为 `600`，没有提交到 GitHub。

Docker 方面已完成：

- Docker daemon 已启动
- Airflow 相关进程已使用包含 `docker` 组的环境重启
- Runtime 中 `docker` 检查通过
- Docker Hub `hello-world` 拉取测试因 registry 网络超时失败；这只说明外部镜像仓库访问不稳定，不代表本地 Docker daemon 故障

## 6. 运行配置

本次本地验证使用了以下非敏感配置：

| 配置 | 值 | 含义 |
|---|---|---|
| `PLATFORM_AGENT_PROVIDER` | `gemini` | 使用原生 Gemini Agent Provider |
| `PLATFORM_AGENT_MODEL` | `gemini-flash-latest` | Agent 规划和回答模型 |
| `PLATFORM_RAG_EMBED_PROVIDER` | `gemini` | 使用 Gemini Embedding |
| `PLATFORM_RAG_EMBED_MODEL` | `gemini-embedding-2` | RAG 向量模型 |
| `PLATFORM_RAG_EMBED_DIM` | `768` | 向量维度 |
| `PLATFORM_RAG_LEXICAL_WEIGHT` | `0.5` | Hybrid RAG 词法权重 |
| `PLATFORM_RAG_VECTOR_WEIGHT` | `0.5` | Hybrid RAG 向量权重 |
| `PLATFORM_GPU_RUNTIME` | `simulated` | 无物理 GPU 时使用模拟设备 |
| `PLATFORM_STAGE_RUNTIME` | `mock` | 本地使用 Mock Stage |
| `MOCK_STAGE_RESULT` | `success` | Mock Stage 默认成功 |
| `MOCK_STAGE_DURATION_SEC` | `0` | Mock Stage 不等待真实算法执行 |
| 模拟设备 | `0,1,2` | 提供 3 个逻辑 GPU |

代码还修复了 Runtime 配置传播问题，使 GPU Simulator、Mock Stage 和可选的 `MOCK_STAGE_*` 覆盖项能够从源码配置写入 `platform.env`。

## 7. 本次部署中完成的代码修复

本地部署过程中发现并修复了以下问题：

1. `platform` 原先没有把 GPU Simulator、Mock Stage 相关环境变量完整写入 Runtime `platform.env`，导致配置在安装后丢失；现在已补齐。
2. `PLATFORM_RAG_LEXICAL_WEIGHT` 或 `PLATFORM_RAG_VECTOR_WEIGHT` 为空字符串时，原逻辑会执行 `float('')` 并启动失败；现在空值会回退到 Provider 默认权重。
3. 增加 Gemini 适配器的空权重回归测试和部署契约检查。
4. `version.md` 增加本地 Runtime 集成修复记录。
5. 项目文档已整理到 `docs/`，并同步修复 `scripts/deploy_ci_cloud.sh`，确保下次部署仍会把文档复制到 Runtime 的 `knowledge/repository/`。

## 8. 服务启动和健康检查

执行 `./platform status` 时确认以下组件均为运行中或健康：

- Airflow Scheduler
- Airflow DAG Processor
- Airflow Triggerer
- Airflow API Server
- Execution API Server
- Task Submit Scheduler
- Metadatabase
- Scheduler/Triggerer/DAG Processor heartbeat
- Public API Health
- Execution API Health
- Scheduled submit lock 文件可读

启动后成功解析的 DAG 包括：

```text
batch_pipeline_universal
batch_pipeline_universal_segment
batch_pipeline_universal_test
```

DAG 解析过程只出现 Graphviz 相关提示，没有阻断解析的错误。

## 9. 自动化验收结果

### 9.1 Dependency-light 回归

依赖轻量测试结果：

```text
163 passed in 5.18s
```

测试覆盖 Platform Core、GPU Simulator、Mock Stage、MCP Contract、Agent Policy、RAG、Task Planning、HITL、Action Verification、Observability、Hardening 和 Gemini adapter 等模块。

### 9.2 Doctor

`doctor --strict --json` 结果：

```text
ready_dependency_light = true
ready_full_runtime     = true
errors                  = []
warnings                = []
```

检查通过的项目包括 Python、Runtime state 目录、任务配置目录、Agent session/approval/trace/audit 目录、Mock Stage、Task Planning defaults、Knowledge source、Simulated GPU、Docker、Airflow、MCP、LangGraph、Google GenAI 和 Gemini 配置。

### 9.3 E2E 控制链

E2E 共 10 个步骤，全部通过：

1. Mock Stage 输出验证
2. Submit 计划和 HITL Approval
3. Submit 执行及后验验证
4. Submit 后读取任务状态
5. GPU 诊断
6. 高优先级任务提交并触发软抢占
7. Stage boundary 切换
8. 高优先级任务完成后的恢复
9. Priority HITL、前置条件和后验验证
10. Trace/Audit 持久化

最终记录：

```text
trace_count = 8
audit_count = 8
```

E2E 验证了软抢占不是直接 `kill` 当前 Stage，而是在 Stage 边界切换，并在高优先级任务完成后恢复原任务。

### 9.4 V1.1 对齐评测

Hash/BM25 baseline 评测门禁通过：

| 指标 | 实际值 | 门槛/要求 |
|---|---:|---:|
| RAG context recall | 0.8167 | >= 0.75 |
| RAG context precision | 0.7417 | >= 0.65 |
| Tool F1 | 1.0000 | >= 0.95 |
| Argument accuracy | 1.0000 | >= 0.90 |
| Hard task success rate | 1.0000 | >= 1.00 |
| Task planning accuracy | 1.0000 | >= 1.00 |
| Security attack success rate | 0.0000 | <= 0.00 |

结论：不依赖真实 LLM API 的回归质量门禁通过。

### 9.5 MCP 协议验证

- MCP 官方 CLI 初始化/握手成功
- 自定义 stdio Client 初始化成功
- 成功列出 16 个 Tool
- 成功调用 `get_gpu_pool`
- Tool 执行仍受只读/写操作策略和 HITL 边界约束

## 10. Knowledge/RAG 状态

当前知识库状态：

| 项目 | 状态 |
|---|---|
| Runtime knowledge source | 存在 |
| 文档数 | 30 |
| Chunk 数 | 443 |
| 词法权重 | 0.5 |
| 向量权重 | 0.5 |
| Source fingerprint | 与索引一致，索引新鲜 |
| Retrieval mode | `gemini_hybrid` |
| Embedding provider | `gemini` |
| Embedding model | `gemini-embedding-2` |
| Embedding dimension | 768 |
| Embedding index | 尚未建立 |
| Vector count | 0 |

Hash baseline 的本地检索和评测已经通过；Gemini 向量索引因为外部 Embedding 请求出现 `429 Too Many Requests`，暂未形成可用的向量文件。当前不应把“语料索引新鲜”误认为“真实 Gemini 向量 A/B 已完成”。

## 11. Proxy 和 Gemini 真实网络验证

### 11.1 原始问题

未切换节点时，Gemini API 返回：

```text
400 FAILED_PRECONDITION
User location is not supported for the API use.
```

这说明请求已经到达 Google API，但出口节点地区不被支持。

### 11.2 节点选择

执行的代理入口为：

```text
/usr/local/bin/proxy
```

该命令控制本机 Mihomo 的 `PROXY` 选择器。Gemini 相关节点实际探测结果：

| 节点 | 结果 |
|---|---|
| 台湾-优化 | 请求超时 |
| 印度-优化 | Gemini 简单请求 200，约 3.7 秒 |
| 新加坡-优化-GPT | Gemini 简单请求 200，约 5.1 秒 |
| 日本-优化 | 仍被地区限制 |

之后切换到 `新加坡-优化-GPT`。在该节点上，用户提供形态的最小 Gemini `generateContent` 请求连续 3 次返回 `HTTP 200`，说明代理、API Key 配置和地区出口已经有效。

### 11.3 完整 Agent 的限制

完整 `dataops-agent ask` 需要依次执行结构化规划、知识检索、若干 Embedding 请求和结构化回答。实际日志中出现过：

- `gemini-flash-latest:generateContent` 返回过 `200 OK`
- `gemini-embedding-2:batchEmbedContents` 多次返回过 `200 OK`
- 后续 Embedding 请求返回 `429 Too Many Requests`
- 规划或最终结构化回答返回 `503 UNAVAILABLE: This model is currently experiencing high demand`

因此当前证据支持以下判断：

1. 代理地区问题已解决。
2. API Key 已被 Gemini API 接受。
3. 简单 Gemini 请求可以成功。
4. 完整 Agent 的失败点是上游限流/高峰或结构化长请求稳定性，不是本地部署脚本找不到 Key。
5. 真实 Gemini Agent 和 Embedding 的生产级稳定性仍需使用更高配额或稍后重试验证。

## 12. 未完成项和风险

### 12.1 Gemini 配额和模型高峰

建议使用专用的、有稳定配额的 Gemini API Project/Key，并在 Agent Provider 增加针对 `429/503` 的指数退避、请求级重试上限和可观测错误分类。当前简单请求可用，不代表包含多次 Embedding 和 Structured Output 的完整工作流具备稳定吞吐。

### 12.2 Embedding A/B 尚未完成

当前 `knowledge status` 显示 Embedding 文件不存在、向量数为 0。需要在 API 限流恢复后执行：

```bash
cd /home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agent_v1.2
. /home/ubuntu/project/autodrive_dataops_runtime/config/platform.env
PATH=/home/ubuntu/project/autodrive_dataops_runtime/bin:$PATH \
  dataops-agent knowledge build --force
PATH=/home/ubuntu/project/autodrive_dataops_runtime/bin:$PATH \
  dataops-agent knowledge status --json
```

完成后再执行 Gemini RAG eval，并与已经通过的 Hash baseline 对比；不要覆盖 V1.1 Golden Set。

### 12.3 GPU 和 Stage 仍为本地模拟

本次部署验证的是平台控制链，不是真实自动驾驶算法生产执行：

- GPU 使用 `SimulatedGPURuntime`
- Stage 使用 `Mock Stage`
- Mock Stage 默认立即返回成功

切换生产环境前，应准备真实 GPU、算法镜像、数据路径和相应的 `PLATFORM_GPU_RUNTIME`/`PLATFORM_STAGE_RUNTIME` 配置，并重新执行完整验收。

### 12.4 Docker 镜像仓库网络

本地 Docker daemon 正常，但 Docker Hub `hello-world` 拉取遇到 registry timeout。真实 Stage 镜像部署前，应先确认镜像仓库、代理、登录凭据和镜像缓存策略。

### 12.5 API Key 安全

API Key 只写入 Runtime 的 `platform.env`，报告、源码和 GitHub 提交均未包含实际值。但该 Key 曾经出现在聊天输入中，建议在完成验证后立即轮换，并重新运行 `doctor --strict` 和 Gemini smoke test。

## 13. 日常运维命令

```bash
cd /home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agent_v1.2

# 查看平台状态
./platform status

# 严格依赖检查
. /home/ubuntu/project/autodrive_dataops_runtime/config/platform.env
PATH=/home/ubuntu/project/autodrive_dataops_runtime/bin:$PATH \
  dataops-agent doctor --strict --json

# 查看知识库状态
PATH=/home/ubuntu/project/autodrive_dataops_runtime/bin:$PATH \
  dataops-agent knowledge status --json

# 运行本地 E2E
PATH=/home/ubuntu/project/autodrive_dataops_runtime/bin:$PATH \
  dataops-agent e2e --json

# 重新切换代理节点
/usr/local/bin/proxy
```

如果要直接使用当前代理环境运行 Agent，需要确认当前 shell 已包含 `http_proxy`、`https_proxy` 和 `all_proxy`，且不要把 `platform.env` 内容打印到终端或日志。

## 14. 验收材料位置

本机验收材料暂存于源码目录的 `local_acceptance/`，包括：

```text
doctor.json
doctor_strict.json
e2e.json
hash_eval.json
knowledge_status.json
platform_status.txt
```

这些材料含有本机 Runtime 的绝对路径和运行状态，因此没有提交到 GitHub。报告中只保留了可复核的摘要结果。

## 15. 最终结论

本次部署已经完成本地可运行平台的主要目标：Airflow、任务调度、GPU 模拟、Mock Stage、MCP、HITL、软抢占、恢复、审计追踪和基线评测全部通过。Gemini 出口节点也已经切换到能够访问 Google API 的新加坡节点。

当前唯一未闭环的部分是外部 Gemini 服务的稳定性：简单 API 请求正常，但完整 Agent 的多请求结构化链路仍受 `429/503` 影响，Embedding 向量索引尚未生成。生产启用前应先解决 API 配额/限流，并完成 Gemini Embedding A/B 和真实 Stage/GPU 验收。
