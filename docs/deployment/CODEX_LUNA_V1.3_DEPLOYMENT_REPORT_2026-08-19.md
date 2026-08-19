# CODEX LUNA V1.3 部署与验收报告

日期：2026-08-19（Asia/Shanghai）

源码目录：`/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agent_v1.2`

Runtime：`/home/ubuntu/project/autodrive_dataops_runtime`
工作分支：`v1.3-runtime-reliability-security`

## 结论

V1.3 的安全整改、Airflow 运行时密钥轮换、Gemini 重试层、增量可恢复 dense sidecar 和依赖轻量回归已经完成并通过本地测试。

真实 Gemini 生成和 Embedding 小请求在切换到 `印度-优化` 节点后均曾成功；但随后真实结构化生成出现 503/429，完整 dense build 在有限续建后停在 288/503 个当前 chunk。因此 Phase 4 的“全量完成”条件未满足，依赖完整 Gemini 数据的 A/B、真实 Agent grounding 和语义评测没有被伪造执行。

用户明确要求 Gemini API key 保持不变，本次没有修改或轮换该 key。Airflow 的 Fernet/API/JWT 密钥则已按计划轮换。

## Phase 0：基线和备份

- `platform status`：通过。
- `doctor --strict`：通过，Runtime 完整就绪。
- E2E：通过，10 个步骤、8 条 trace、8 条 audit。
- 依赖轻量基线回归：修正测试环境后通过 163 项。
- Hash aligned eval：通过，基线指标为：
  - context recall：0.8166666667
  - context precision：0.7416666667
  - tool F1：1.0
  - argument accuracy：1.0
  - hard task success：1.0
  - task planning accuracy：1.0
  - security attack success rate：0.0
- 元数据库备份成功，文件位于 Runtime 外部备份目录：
  `/home/ubuntu/project/autodrive_dataops_runtime/backups/v1.3_pre_rotation_20260819_2320/airflow_metadata.sql`。

## Phase 1：仓库脱敏与 Airflow 密钥轮换

完成内容：

- `config/airflow.cfg.base` 中的 `fernet_key`、`sql_alchemy_conn`、`secret_key`、`jwt_secret` 已改为空模板值，由 Runtime 注入。
- `recover/airflow.cfg` 和 `recover/simple_auth_manager_passwords.json.generated` 已从 Git 跟踪中移除。
- `recover/README.md` 改为只描述恢复流程，不保存真实配置或密码。
- `.gitignore` 已覆盖 `platform.env`、`runtime_secrets.env`、`local_acceptance/`、`recover/*` 和生成文件。
- 新增 `scripts/runtime_secrets.py`：首次安装原子生成并保持 `0600`；重复安装复用；显式 `rotate` 才轮换。
- `platform` 安装流程会生成或读取 Runtime `config/runtime_secrets.env`，并将 Airflow secrets 注入最终 `airflow.cfg`。
- 轮换前已确认没有运行中的 DagRun；使用双 Fernet key 迁移后切换到新 Fernet key，并将最终 `airflow.cfg` 和 Runtime secret 文件权限设为 `0600`。
- 轮换完成后服务恢复健康，API、Execution API、scheduler、dag processor、triggerer 均正常。

敏感值没有写入本报告、源码 diff 或验收输出。需要注意：旧 secret 仍存在于远端 Git 历史中；本次没有执行历史重写或 force push。

## Phase 2：Gemini 429/503 重试

新增 `platform_integrations/gemini_retry.py` 共享层，覆盖：

- 408、429、500、502、503、504 和临时传输错误。
- 非重试类 400、401、403 不重复请求。
- 默认 5 次尝试、指数退避、最大延迟和 jitter，可由环境变量调整。
- 支持 `Retry-After`，支持注入 sleep/jitter，测试不真实等待。
- 同时接入结构化生成和每个 Embedding API batch。
- 最终异常只包含 operation、attempts、status code 和 retryable 分类，不包含 prompt、Authorization 或 API key。

## Phase 3：可恢复增量 dense sidecar

`DenseEmbeddingIndex` 已升级到 schema v2，增加：

- `expected_chunk_count`
- `complete`
- `content_hashes`
- `updated_at`
- `missing_vector_count` 和 `resumable` 状态输出

行为已改为按 `chunk_id + content_hash` 复用向量；删除 chunk 会清理 stale vector；model/dimension 变化会使旧向量失效；`--force` 不会无故清空可复用 dense vector；新增 `--reset-embeddings` 才执行全量 dense 重建；每个成功 batch 都以临时文件加原子替换 checkpoint。

同步 Embedding build 通过 `asyncio.to_thread` 离开 Agent async event loop。

真实验收中当前知识库实际为 503 个 chunk。首次失败后 sidecar 保留 96 个向量；续建后曾达到 192、288，且每次没有归零，证明断点续建和 partial sidecar 生效。外部服务持续 429/503 后停止继续尝试，当前状态为：

```text
provider=gemini
retrieval_mode=gemini_hybrid
vector_count=288
expected_vector_count=503
missing_vector_count=215
complete=false
resumable=true
```

## 真实 Gemini 验收

代理探测结果：

- 日本节点：Gemini 返回区域限制。
- 新加坡节点：曾返回区域限制，随后出现配额 429。
- 台湾节点：连接超时。
- 印度节点：最小 `generateContent` 请求 HTTP 200 且有内容；最小 `gemini-embedding-2` 请求成功并返回 768 维向量。

后续 3 次结构化生成 smoke 中出现 503→429，重试层执行到最大尝试后安全失败；这是外部节点/配额状态，已保留失败现场，没有把它报告为 3/3 通过。

## 测试结果

- V1.3 新增安全、重试、embedding resume 测试：24 passed。
- V1.2 Gemini、V1.3 reliability/resume 和 RAG 相关定向测试：41 passed。
- 修正测试环境变量污染后，完整 dependency-light 回归：187 passed。
- 真实 Runtime 在轮换和源码同步后仍保持健康。

验收材料保存在被 Git 忽略的 `local_acceptance/v1.3_before/` 和 `local_acceptance/v1.3_after/`，运行时备份保存在 Runtime 外部 `backups/` 目录。

## 未完成项与下一步

以下项目因真实 Gemini 全量 dense build 被外部 429/503 阻塞，暂不标记通过：

1. dense sidecar 完成到当前 503/503。
2. Hash vs Gemini 完整 A/B 报告。
3. 真实 Gemini Agent tool grounding / HITL acceptance。
4. Ragas、DeepEval 等真实语义评测。

外部 Gemini 恢复可用且配额允许后，直接执行 `dataops-agent knowledge build --json` 继续当前 sidecar；不要使用 `--reset-embeddings`，除非明确需要清空现有向量。
