# CODEX LUNA V1.3.1 Qwen Deployment Report

日期：2026-08-20

## 结论

Qwen runtime 已部署到 `/home/ubuntu/project/autodrive_dataops_runtime`，源码改动位于 `v1.3.1-qwen-runtime` 工作树。Qwen generation、native embedding、503 条 dense vectors、Agent tool grounding 和 HITL verification 已经真实跑通。Hash/Qwen 同 Golden Set A/B 两套 deterministic gates 均通过。

## Gemini previous state

迁移前保留证据显示：generation 只能作为历史 partial evidence；Gemini 768 维 dense sidecar 为 416/503，缺失 87，`complete=false`，可恢复但没有完成。之前的 runtime 证据包含 429/503；旧报告 `docs/deployment/CODEX_LUNA_V1.3_DEPLOYMENT_REPORT_2026-08-19.md` 未覆盖或删除。

## Qwen new state

- Model：`qwen3.7-flash`。
- Embedding：`qwen3.7-text-embedding`，1024 维，batch size 20。
- Generation smoke：3/3。
- Embedding smoke：document/query 均通过。
- Dense sidecar：503/503，`missing=0`，`complete=true`。
- Doctor strict：dependency-light/full-runtime ready；Qwen dependency、key presence 和 endpoints 均 configured。key 值未进入任何输出。

## A/B outcome

| Metric | Hash | Qwen | Delta |
|---|---:|---:|---:|
| Context Recall | 0.816667 | 0.833333 | +0.016667 |
| Context Precision | 0.741667 | 0.780556 | +0.038889 |
| MRR | 0.741667 | 0.788889 | +0.047222 |
| nDCG | 0.746500 | 0.790052 | +0.043552 |

Per case 记录在 `local_acceptance/v1.3.1_after/rag_hash_vs_qwen.md`：improved=5、unchanged=24、regressed=1。第一轮没有改变 lexical/dense 权重，没有开启 instruct，没有添加 reranker。

## Agent / MCP / HITL

1. 静态问题“软抢占为什么不直接 kill 当前 Stage？”返回 `platform_knowledge`，使用 Qwen Hybrid RAG，未调用 MCP。
2. 实时 GPU 问题 trace 含 `get_gpu_pool`。
3. 平台健康问题 trace 含 `get_platform_health`。
4. `release_demo` 诊断返回 MCP 支撑的任务配置缺失根因，同时记录当前 GPU 资源池证据；没有仅凭 RAG 推断当前状态。
5. runtime 临时 `test_task` fixture 上执行 priority 20→5：pending approval、impact/precondition、mutation、observe 和 deterministic verification 全部存在，完成后 fixture config/DAG 已精确清理。

## Evaluation framework status

- Ragas：Qwen adapter 已实现；真实 sample collection 成功，但 0.4.3 judge/embedding score 在当前代理链路受控 90 秒内没有返回，记录 `BLOCKED_NOT_VALIDATED`，原始证据在 `local_acceptance/v1.3.1_after/ragas_qwen.json`。
- DeepEval：默认 OpenAI 字符串 model 失败后，custom `QwenDeepEvalModel` 真实执行 21 case；ArgumentCorrectness=1.0，ToolCorrectness=0.142857。TaskCompletion 未虚构。
- Promptfoo：配置为 12 个 curated security cases，使用真实 Qwen planner；执行被 npm 镜像 `ECONNRESET` 阻塞，没有生成 JSON 结果，原始 stderr 保存在 `local_acceptance/v1.3.1_after/promptfoo_qwen.stderr`，记为 `BLOCKED_NOT_VALIDATED`。

## Final regression evidence

- Provider/retry/embedding/security 专项：43 passed。
- Hardening：7 passed。
- Dependency-light：194 passed、1 skipped；隔离了 DeepEval 等可选 pytest 插件并保留 asyncio 插件，排除了 3 个历史 Airflow 直导测试。无 NVIDIA 环境下会因 `nvidia-smi` 缺失而等待的历史 GPU DAG 测试单独保留，未计入该汇总。
- Runtime doctor strict：两套 ready，0 errors、0 warnings。
- Runtime E2E：10/10。
- Runtime final aligned gates：Tool F1、Argument Accuracy、Hard Task Success、Task Planning Accuracy 均为 1.0，Security ASR 为 0；Qwen RAG 指标见上表。

## Security and secrets

- Qwen key 只通过 `DASHSCOPE_API_KEY` 注入；`/home/ubuntu/project/auth/ali.api` 权限为 600。
- redaction 覆盖 `DASHSCOPE_API_KEY`、Gemini、OpenAI、Airflow token/password 等模式。
- 未 rotate Airflow secrets；未做 history rewrite 或 force push。
- 旧 Gemini provider/report 保留，Qwen 1024 sidecar 与 Gemini 768 sidecar 分离。

## Handoff

原始测试和命令输出位于 `local_acceptance/v1.3.1_after/`；建议交接时先查看 `docs/V1.3.1_TEST_REPORT.md` 与 `docs/V1.3.1_QWEN_RUNTIME_MIGRATION.md`，再按 Phase Q 单独处理远端旧 secret history。当前没有执行 force push。
