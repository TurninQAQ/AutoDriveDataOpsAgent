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
- V1.3.3：Qwen Plus Primary Agent，保留 qwen3.7-flash 为 legacy/fallback，继续使用 qwen3.7-text-embedding。
- V1.7.0：Policy-gated bounded autonomy，仅允许满足 deterministic 条件的 `resume_task` 进入 AUTO；其他写操作继续 HITL。

无真实 GPU 的本地开发方式见：[docs/V0.2_GPU_SIMULATION.md](docs/V0.2_GPU_SIMULATION.md)。

MCP Tool 设计和运行方式见：[docs/V0.3_PLATFORM_MCP.md](docs/V0.3_PLATFORM_MCP.md)。

Read-only Agent 设计和运行方式见：[docs/V0.4_READ_ONLY_AGENT.md](docs/V0.4_READ_ONLY_AGENT.md)。

RAG / Runbook 设计和运行方式见：[docs/V0.5_RAG_RUNBOOK.md](docs/V0.5_RAG_RUNBOOK.md)。

Task Planning 设计和运行方式见：[docs/V0.6_TASK_PLANNING.md](docs/V0.6_TASK_PLANNING.md)。

Write Agent/HITL 设计见：[docs/V0.7_WRITE_AGENT_HITL.md](docs/V0.7_WRITE_AGENT_HITL.md)。

Action Verification 设计见：[docs/V0.8_ACTION_VERIFICATION.md](docs/V0.8_ACTION_VERIFICATION.md)。

V1.0 硬化/E2E 设计见：[docs/V1.0_HARDENING_E2E.md](docs/V1.0_HARDENING_E2E.md)。

V1.1 评测体系见：[docs/V1.1_EVALUATION_ALIGNMENT.md](docs/V1.1_EVALUATION_ALIGNMENT.md)。

Gemini 模型/RAG 适配见：[docs/V1.2_GEMINI_PROVIDER.md](docs/V1.2_GEMINI_PROVIDER.md)。

Qwen V1.3.1 迁移与验收见：[docs/V1.3.1_QWEN_RUNTIME_MIGRATION.md](docs/V1.3.1_QWEN_RUNTIME_MIGRATION.md)、[docs/V1.3.1_TEST_REPORT.md](docs/V1.3.1_TEST_REPORT.md) 和 [部署报告](docs/deployment/CODEX_LUNA_V1.3.1_QWEN_DEPLOYMENT_REPORT_2026-08-20.md)。

V1.3.3 qwen-plus Primary Agent 评测见：[主评测报告](docs/evaluation/V1.3.3_QWEN_PLUS_PRIMARY_EVALUATION.md)、[当前状态](docs/evaluation/V1.3.3_CURRENT_STATE.md)。


本项目用于把数据处理 pipeline 部署到 Airflow，并通过任务 YAML 提交和管理多任务动态 DAG。

主要文档：

- 文档总览：[docs/README.md](docs/README.md)
- 本次部署报告：[docs/deployment/DEPLOYMENT_REPORT_2026-08-19.md](docs/deployment/DEPLOYMENT_REPORT_2026-08-19.md)
- 从 0 部署平台：[docs/deploy_guide.md](docs/deploy_guide.md)
- 日常使用平台：[docs/usage_guide.md](docs/usage_guide.md)

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

## Qwen 本地 Runtime（V1.3.3）

正式 runtime 默认使用 Qwen；密钥只通过环境变量或本机 secure env 注入，不要提交仓库，也不要把 `/home/ubuntu/project/auth/ali.api` 内容复制到源码或报告。

```bash
export DASHSCOPE_API_KEY='YOUR_KEY'
export DASHSCOPE_OPENAI_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export DASHSCOPE_API_BASE_URL='https://dashscope.aliyuncs.com/api/v1'
export PLATFORM_AGENT_PROVIDER=qwen
export PLATFORM_AGENT_MODEL=qwen-plus
export PLATFORM_RAG_EMBED_PROVIDER=qwen
export PLATFORM_RAG_EMBED_MODEL=qwen3.7-text-embedding
export PLATFORM_RAG_EMBED_DIM=1024
export PLATFORM_RAG_EMBED_BATCH_SIZE=20

./platform install
dataops-agent doctor --strict --json
dataops-agent knowledge build --json
dataops-agent knowledge status --json
dataops-agent eval-aligned --json
```

Qwen dense sidecar 与历史 Gemini sidecar 分开保存；第一轮固定 lexical/dense=0.50/0.50，不启用 instruct 或 reranker。

## V1.7 Bounded Autonomy

Autonomy 默认关闭。开启后也只有 deterministic policy 允许的安全
`resume_task` 可以 AUTO；`submit_task`、优先级、停止和删除继续要求 HITL。
AUTO resume 会先冻结目标 task 和当前确实失败的 dataset 集合，再共同经过
Precondition、Mutation、Action Verification 和 Goal Verification；没有自动重试，
跨任务抢占、非失败 dataset、未知 dataset、关键读证据缺失或预算超限不会 AUTO。

```bash
export PLATFORM_AGENT_AUTONOMY_ENABLED=0
export PLATFORM_AGENT_AUTO_ACTIONS_PER_REQUEST=1
export PLATFORM_AGENT_AUTO_RESUME_MAX_DATASETS=3
```

实现与验收记录见：[V1.7.0 Bounded Autonomy](docs/evaluation/V1.7.0_BOUNDED_AUTONOMY.md)。

## V1.8 A+ Final Hardening

V1.8 保持只有 `resume_task` 可进入 deterministic AUTO，并封闭并发
reservation race、重复 action fingerprint、同一 AUTO record 的重复 claim、
重启 replay、scope drift 和 failure retry。AUTO 记录先持久化为
`authorized`，再通过原子 claim 进入 `executing`，并继续经过
Precondition、Mutation、Action Verification 和 Goal Verification。

实现与验收记录见：[V1.8.0 A+ Final Hardening](docs/evaluation/V1.8.0_A_PLUS_FINAL_HARDENING.md)。


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
