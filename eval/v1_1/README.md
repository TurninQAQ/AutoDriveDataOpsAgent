# V1.1 Evaluation Dataset

V1.1 将“开发回归”与“Agent/RAG 质量评测”分离。数据均基于当前自动驾驶离线处理平台的真实对象模型、状态机和故障语义构造；没有真实线上数据的部分使用可复现的 realistic fixtures，而不是随机文本。

## 数据集

| 文件 | 数量 | 用途 |
|---|---:|---|
| `rag_retrieval.jsonl` | 30 | Chunk-level RAG retrieval golden set |
| `rag_generation_cases.jsonl` | 12 | 可选 Ragas semantic/judge 子集 |
| `agent_tool_cases.jsonl` | 21 | Tool selection / argument / forbidden capability |
| `agent_task_cases.jsonl` | 13 | Environment-first diagnosis / safety / verification |
| `security/curated_attacks.jsonl` | 12 | dependency-light curated red-team regression |
| `../task_planning_cases.json` | 8 | Structured TaskSpec planning regression |

## Golden 数据原则

### RAG

每条 retrieval case 包含：

- `query`
- `reference_context_ids`：精确到 `source#section`，必要时可带 `::chunkN`
- `reference_answer`
- `required_facts`
- `top_k`

不再使用“只要命中某个文件就算正确”的 source-level 标注。

### Agent Tool

每条 case 区分：

- `required_tools`
- `optional_tools`
- `forbidden_tools`
- `expected_arguments`（subset matching）
- `required_order`（仅在业务合同真的要求顺序时使用）

因此不强制 Agent 匹配唯一 trajectory。

### Agent Task

环境状态 fixture 模拟真实平台字段，包括：

- Airflow DagRun / TaskInstance
- Queue active/draining
- GPU free memory / Reservation
- Stage logs / OOM
- Container residue
- Action Verification snapshot

最终成功优先由 deterministic environment state 判定，而不是相信 Agent 自己说“成功”。

### Security

curated cases 与 Promptfoo 风险类别对齐，重点检查：

- excessive agency
- system prompt override
- indirect prompt injection
- RAG poisoning
- MCP/tool abuse
- shell injection
- secret disclosure
- cross-task access

dependency-light scorer 的硬标准是“禁止能力是否真的被调用”，而不是要求模型必须输出某个固定 intent。

## 指标

### Retrieval

- Hit@K（辅助）
- Precision@K（辅助）
- Context Recall
- Ragas-style Context Precision（ranking-aware）
- MRR
- nDCG@K

### Agent component

- Intent Accuracy（诊断辅助）
- Tool Precision / Recall / F1
- Argument Accuracy
- Forbidden Tool Call Rate
- Ordering Accuracy（只在显式 required_order case 上）

### Task / Safety

- Hard Task Success Rate
- Task Planning Accuracy
- Security Attack Success Rate

## Gate

默认门槛在 `thresholds.json`。RAG gate 故意不是 1.0：它用于发现 retrieval 真实差距，并为后续 embedding/reranker 改造提供可比较 baseline。

## 可选主流框架

- Ragas：ContextPrecision/Recall、Faithfulness、AnswerRelevancy、AnswerCorrectness
- DeepEval：ToolCorrectness、ArgumentCorrectness；TaskCompletion 应绑定真实 Agent trace
- Promptfoo：curated safety eval + dynamic red team

`eval-aligned` 不依赖这些外部包；Ragas/DeepEval/Promptfoo 是第二层 semantic/security evaluation，按需安装和运行。
