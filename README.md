# AutoDriveDataOpsAgent V3.9 Release Candidate

AutoDriveDataOpsAgent is a single-agent DataOps assistant for a **single-node simulated AutoDrive training platform**. V3.9 keeps the V3.5 core architecture frozen and adds durable persistence, persistent write safety, real-dense-capable hybrid retrieval, FastAPI/SSE serving, deterministic evaluation, and release packaging.

## 1. Project Overview

The system lets a language model inspect task/GPU/queue/runtime state and search runbooks, while platform mutations remain outside model authority. The validated core is:

```text
User / API
  -> LangGraph Single-Agent Guarded ReAct
     -> agent
     -> model_tools
     -> review
     -> execute_write
  -> Standard MCP
     -> Agent profile: READ / PREPARE / PROPOSAL
     -> Runtime profile: READ / real WRITE
  -> simulated AutoDrive platform
```

The production graph still has exactly four business nodes. V3.9 does not introduce a planner agent, intent router, verifier agent, multi-agent topology, or a separate RAG node.

## 2. Problem

READ operations are safe to automate, but WRITE operations such as priority changes, resume/stop/delete, and task submission need explicit authorization and deterministic correctness checks. The core design therefore separates semantic choice from mutation authority.

## 3. Guarded ReAct

The LLM reasons over observations and emits native tool calls. READ/PREPARE tools execute directly. A WRITE request is represented by a side-effect-free Proposal, converted by the runtime into a frozen `PendingAction`, then reviewed through LangGraph HITL.

Core rules:

- LLM controls semantic decisions; Runtime controls deterministic execution.
- Proposal is not execution.
- API success is not business success.
- Thin Graph, Rich Service.

## 4. Standard MCP

The project uses the official MCP Client/Server model with two capability profiles:

- `/mcp/agent`: READ, PREPARE, PROPOSAL only.
- `/mcp/runtime`: READ plus real WRITE tools.

Local mainline uses the official in-process `Client(MCPServer)` path. Remote mainline uses Streamable HTTP. Mounted Agent/Runtime servers are hosted by one Starlette lifespan which starts both MCP session managers.

## 5. Native Function Calling

MCP `tools/list` schemas are adapted into provider-native tool schemas. Qwen receives `tools`, `tool_choice=auto`, and native parallel tool calls. Proposal policy rejects mixed/multiple Proposal rounds while returning one Tool message for every emitted `tool_call_id`.

## 6. Proposal + HITL

Real WRITE tools are never exposed to the model. Proposal tools return structured intent only. Runtime then assigns a unique runtime `proposal_id`, captures before-state/preconditions, freezes arguments/artifacts, computes a fingerprint over both the approval identity and frozen semantic content, and exposes a safe review payload. Approve authorizes exactly the frozen fingerprint; edit rebuilds a new fingerprint; reject produces zero mutation.

## 7. WRITE Reliability

`WriteService` owns the mutation boundary:

```text
fingerprint validation
-> global precondition recheck
-> action-specific revalidation
-> persistent idempotency claim (DISPATCHING)
-> one mutation attempt
-> observe again
-> action-specific verification
-> persistent result + audit
```

V3.9 persists idempotency in SQLite. A restart that sees `DISPATCHING` or `UNKNOWN_OUTCOME` reconciles by READ and does not blindly retry the mutation. This is **at-most-one mutation attempt per approved fingerprint**, not an exactly-once claim. Because the runtime-generated `proposal_id` is included in the fingerprint, a later independent human approval may execute the same semantic action again after platform state legitimately returns to the same value; only retries of the same approved action are deduplicated.

## 8. Persistence

Responsibilities are deliberately separated:

- LangGraph checkpointer: workflow state / interrupt-resume only.
- `AuditStore`: append-only business events.
- `WriteExecutionStore`: persistent idempotency and mutation-attempt state.
- `RunStore`: API run metadata.

Production configuration uses the official `langgraph-checkpoint-sqlite` `AsyncSqliteSaver`; the project does not implement its own LangGraph checkpoint protocol.

Important environment variables:

```text
AUTODRIVE_STATE_DIR=./runtime_state
AUTODRIVE_CHECKPOINT_BACKEND=sqlite
AUTODRIVE_DB_PATH=./runtime_state/autodrive_state.sqlite
AUTODRIVE_CHECKPOINT_PATH=./runtime_state/checkpoints.sqlite
AUTODRIVE_WRITE_STORE=sqlite
```

## 9. Hybrid RAG

`search_knowledge` remains a normal MCP READ tool. Retrieval modes are explicit:

- `bm25`: lexical retrieval.
- `dense`: real configured embedding index only.
- `hybrid`: BM25 + Dense rank fusion using RRF.

Dense index files:

```text
runtime_state/knowledge_index/
  chunks.jsonl
  embeddings.npy
  manifest.json
```

The manifest binds embedding model, dimension, chunk count, knowledge hash, and build time. A changed corpus/model/dimension invalidates the index.

`RAG_DENSE_PROVIDER=disabled` falls back to **BM25 and reports BM25**. It never calls feature hashing a Dense Embedding. `GeminiEmbeddingProvider` supports `gemini-embedding-2` through `google-genai`, default dimension 768. The deterministic embedding provider exists only for tests/offline correctness checks.

## 10. Retrieval Evaluation

`deploy_ci_cloud_agentv3/evaluation/rag_cases.jsonl` contains 30 retrieval cases. Evaluation reports Recall@1/3/5 and MRR to `artifacts/rag_eval.json` and `artifacts/rag_eval.csv`.

Without a Google API key, real external dense evaluation is `NOT_RUN_EXTERNAL`; BM25 and deterministic embedding tests still run offline.

## 11. FastAPI + SSE

The API is a transport layer around `AgentRuntime`:

```text
POST /runs
GET  /runs/{run_id}
GET  /runs/{run_id}/events
POST /runs/{run_id}/approve
POST /runs/{run_id}/reject
POST /runs/{run_id}/edit
GET  /health
GET  /ready
```

SSE replays durable AuditStore events first, then uses an in-memory queue only for live delivery. It exposes safe status/event summaries, not hidden chain-of-thought, system prompts, provider reasoning, or API keys.

## 12. Benchmark

The repository includes 40 deterministic cases (10 READ, 10 WRITE, 10 MIXED, 10 FAULT) and three scripted baselines:

- Naive ReAct
- Generic HITL
- Guarded ReAct

Metrics include Task Success, False Success, Unsafe Write, Wrong Target, Tool Selection Accuracy, Verification Success, average LLM/tool calls, and latency fields. The offline benchmark uses a deterministic `ScriptedProvider`, but it executes the real Guarded `AgentRuntime`/LangGraph/MCP path against an isolated simulated platform. Naive ReAct directly dispatches its selected mutation. Generic HITL has a distinct candidate → human approval → execution control flow, but intentionally omits frozen fingerprint binding, precondition/revalidation, persistent idempotency, and post-write verification. Cases load isolated platform fixtures, and fault injection changes actual platform/transport behavior. False Success is computed from the final success claim versus the observable business effect (`expected_final_state`), not from mutation-call counts. This remains an offline deterministic benchmark, not a live external-Qwen benchmark.

## 13. Simulated Platform Scope

Validated deployment target: **single-node simulated/mock AutoDrive training platform** with Task, GPU, Queue, Airflow-like, Docker-like, and knowledge abstractions. This repository does not claim a production autonomous-driving cluster or physical multi-GPU production validation.

## 14. Quick Start

```bash
python -m pip install -e '.[test]'
cp .env.example .env

autodrive-agent --version
autodrive-agent health
autodrive-agent ready
```

Run one Agent request:

```bash
autodrive-agent run "task_A 为什么失败？"
```

Serve API:

```bash
autodrive-agent serve
```

Serve MCP profiles:

```bash
autodrive-agent mcp-serve
```

Build a real Dense index after configuring Gemini:

```bash
export RAG_DENSE_PROVIDER=gemini
export GOOGLE_API_KEY=...
autodrive-agent rag build-index
```

Offline retrieval evaluation / benchmark:

```bash
autodrive-agent rag eval
autodrive-agent benchmark
```

## 15. Tests

Release gates in a complete dependency environment:

```bash
pytest -q deploy_ci_cloud_agentv3/tests
python -m compileall -q deploy_ci_cloud_agentv3
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Core regression coverage retains fingerprint/artifact tamper, persistent idempotency, stale approval, resume TOCTOU, verification false-success attacks, mixed Proposal protocol, official MCP profiles/HTTP, mounted MCP lifespan, LangGraph approve/reject/edit, persistence, RAG, API, and benchmark tests.

## 16. Demo

Start the API, then run:

```bash
python scripts/demo_v39.py
```

The demo covers READ diagnosis plus Proposal -> review -> verified write. Stale-approval behavior is covered by tests and can also be demonstrated by mutating simulated platform state between review and approve.

## 17. Security / Error Handling

Critical MCP/persistence/verification paths fail closed. Audit payloads redact common secret/token fields. Observation errors are not treated as evidence of resource absence. Release artifacts exclude `.env`, runtime databases, Python caches, and test state.

## 18. Limitations / Future Work

- SQLite is appropriate for this local/single-node project, not active-active distributed deployment.
- Real Qwen/Gemini tests require credentials and are reported separately from offline correctness.
- No vector database, distributed lock, Kafka/Redis, OpenTelemetry, LangSmith, multi-agent planner, or exactly-once protocol is included.
- Benchmark results in `artifacts/` should be interpreted according to their recorded benchmark type/provider mode.
