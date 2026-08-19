# deploy_ci_cloud 0731

## Agent 化版本进度

- V0.1：Platform Core 重构，CLI 与核心业务能力解耦。
- V0.2：GPU Runtime 抽象、Simulated GPU、共享 GPUAllocator、Mock Stage。
- V0.3：Platform MCP Server，只读领域 Tool、Airflow/Queue/Docker/GPU Evidence 聚合。
- V0.4：Read-only Agent，LangGraph workflow、MCP Tool Planning、Evidence Diagnosis、只读 Policy、Thread Memory。
- V0.5：RAG + Runbook，本地 Hybrid Retrieval、知识索引、Grounding、Retrieval Eval。
- V0.6：Natural Language Task Planning，Structured TaskSpec、确定性 defaults、平台原生校验、YAML 生成。
- V0.7：Write Agent，Write MCP Tools、PolicyEngine、持久化 HITL Approval、Precondition 和 exactly-once approval claim。
- V0.8：Action Verification，写操作后重新 Observe，并对 submit/priority/stop/resume/delete 执行确定性后验验证。
- V0.9：Evaluation + Observability，Trace/Audit、Secret Redaction 和统一 Agent Eval。
- V1.0：Hardening + E2E，dependency-light 完整控制链回归、doctor、Trace/Audit retention/rotation 和并发审批硬化。
- V1.1：Evaluation Alignment，新增 chunk-level RAG Golden、Agent tool/task eval、Promptfoo 风格安全集、可选 Ragas/DeepEval/Promptfoo 集成与统一质量门禁。
- V1.2：Gemini Provider，新增原生 google-genai Structured Output Agent Provider，并可选使用 gemini-embedding-2 + BM25 做真正语义 Hybrid RAG。

无真实 GPU 的本地开发方式见：`V0.2_GPU_SIMULATION.md`。

MCP Tool 设计和运行方式见：`V0.3_PLATFORM_MCP.md`。

Read-only Agent 设计和运行方式见：`V0.4_READ_ONLY_AGENT.md`。

RAG / Runbook 设计和运行方式见：`V0.5_RAG_RUNBOOK.md`。

Task Planning 设计和运行方式见：`V0.6_TASK_PLANNING.md`。

Write Agent/HITL 设计见：`V0.7_WRITE_AGENT_HITL.md`。

Action Verification 设计见：`V0.8_ACTION_VERIFICATION.md`。

V1.0 硬化/E2E 设计见：`V1.0_HARDENING_E2E.md`。

V1.1 评测体系见：`V1.1_EVALUATION_ALIGNMENT.md`。

Gemini 模型/RAG 适配见：`V1.2_GEMINI_PROVIDER.md`。


本项目用于把数据处理 pipeline 部署到 Airflow，并通过任务 YAML 提交和管理多任务动态 DAG。

主要文档：

- 从 0 部署平台：[deploy_guide.md](deploy_guide.md)
- 日常使用平台：[usage_guide.md](usage_guide.md)

实验材料位于源码外的 `/home/cfy/project/two/test`。其中全量数据、定时提交和
手动提权的回归实验见：

```text
/home/cfy/project/two/test/0731_full_pipeline_schedule_priority_experiment/
```

该实验严格串行执行 `precheck -> parser -> segment -> map -> od -> coloration -> occ`，
任务 YAML 仅由 `scripts/tools/genarate_dataset_config.py` 生成，并只通过
`./task` 的提交、定时和优先级入口操作平台。

常用入口：

```bash
./platform status
./platform deploy
./task --help
# install 后 Runtime 中可启动只读 MCP Server
$RUNTIME_DIR/bin/mcp-server
$RUNTIME_DIR/bin/dataops-agent ask "release_xxx 现在是什么状态？"
$RUNTIME_DIR/bin/dataops-agent knowledge search "GPU 排队为什么不计入 timeout"
$RUNTIME_DIR/bin/dataops-agent knowledge eval
$RUNTIME_DIR/bin/dataops-agent plan-task "创建一个release任务，把 /data/record_001 做完整流程"
$RUNTIME_DIR/bin/dataops-agent plan-task-eval
$RUNTIME_DIR/bin/dataops-agent doctor
$RUNTIME_DIR/bin/dataops-agent e2e
$RUNTIME_DIR/bin/dataops-agent observability-maintenance
```

注意：`./platform deploy` 会同步 runtime 中的公共 DAG、脚本、Platform Core/MCP/Agent/Planning/RAG/Observability/Eval/Hardening 包、静态配置和 Agent knowledge；
知识目录会包含选定的根目录文档/YAML 快照，但仍不会复制 `/home/cfy/project/two/test` 下的实验脚本。运行中存在
Airflow `running/queued` DagRun 或平台容器时，deploy 会拒绝执行，避免覆盖运行时代码。


## Gemini 本地配置（V1.2）

真实 Key 只放环境变量，不要提交到仓库。推荐先在 Google AI Studio 轮换已暴露过的 Key。

```bash
export GEMINI_API_KEY='YOUR_NEW_KEY'
export PLATFORM_AGENT_PROVIDER=gemini
export PLATFORM_AGENT_MODEL=gemini-3.7-flash
export PLATFORM_AGENT_RUNTIME=langgraph

# 无物理 GPU
export PLATFORM_GPU_RUNTIME=simulated
export PLATFORM_STAGE_RUNTIME=mock

# 可选：启用真正 Gemini Embedding 语义检索
export PLATFORM_RAG_EMBED_PROVIDER=gemini
export PLATFORM_RAG_EMBED_MODEL=gemini-embedding-2
export PLATFORM_RAG_EMBED_DIM=768

./platform install
dataops-agent doctor --strict --json
dataops-agent knowledge build --force --json
dataops-agent knowledge status --json
dataops-agent ask "软抢占为什么不直接 kill 当前 Stage？" --json
dataops-agent ask "当前 GPU Reservation 和显存情况怎么样？" --json
```

若 `PLATFORM_RAG_EMBED_PROVIDER=hash`，则继续使用 V1.1 的 BM25 + feature-hashing baseline，不调用 embedding API。


## Agent V0.9 Observability / Evaluation

V0.9 adds persistent request traces and audit logs without changing the platform execution state machine.

```bash
dataops-agent traces --limit 20
dataops-agent trace <trace_id>
dataops-agent eval
```

Default runtime paths:

```text
$AIRFLOW_STATE_DIR/agent_traces
$AIRFLOW_STATE_DIR/agent_audit/audit.jsonl
```

All persisted trace/audit values pass through secret redaction before disk write. The deterministic V0.9 eval suite covers intent, tool selection, diagnosis, safety, task planning and action verification and does not require a physical GPU or LLM API key.
