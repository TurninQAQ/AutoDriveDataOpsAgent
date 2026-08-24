# V2 Production Integration Gap Matrix

This matrix records the boundary between the restored architecture reference
implementation and production integration work. It is intentionally scoped to
adapters, hosting, observability, and deployment; it does not change the
frozen semantic authority model.

The complete 43-item V2 requirement mapping is maintained in
`V2_REQUIREMENT_TRACEABILITY_MATRIX.md`.

| Area | Frozen/reference state | Production gap | Closure evidence |
|---|---|---|---|
| Agent authority | One visible Agent loop; Runtime validates decisions | None in the authority model | Real LangGraph graph tests |
| LangGraph | Real `StateGraph`/`interrupt()` source; offline tests had a compatibility harness | Real `langgraph==1.2.11` checkpointer serialization and interrupt/resume needed validation | Pinned runtime environment; marked real-runtime tests pass |
| Provider | Deterministic and scripted offline providers | Structured HTTP/Qwen adapter, timeout, retry, rate-limit handling, auth isolation, telemetry | Strict json_schema request with qwen3.7-plus-2026-05-26 and `enable_thinking=false`, valid first decision, fresh single-READ and GPU/queue/knowledge multi-READ CompletionGate passes, and bounded five-run sample 5/5 completed with zero schema-invalid, zero regeneration, zero Provider errors, and zero WRITE; local fake HTTP provider tests pass |
| RAG embedding | Offline BM25 + feature-hashing HybridRetriever | Existing dense providers/index were not constructed by the canonical platform runtime | `PLATFORM_RAG_EMBED_PROVIDER` factory now injects local/Qwen/Gemini selection into `KnowledgeService`; fake-provider/index compatibility tests pass; real Qwen embedding smoke, 443-vector sidecar reuse, and canonical 30-case A/B pass. Local Top-5=`0.8667`, Qwen Top-5=`0.8667`; Qwen MRR=`0.7722` vs local=`0.7417`; no Top-5 regressions. Real Qwen chat/Agent READ E2E is separately validated with `qwen-plus-2025-07-28`. |
| Platform | In-memory READ/WRITE facades | V2-owned platform execution layer plus custom JSON-RPC-over-HTTP transport | V2 in-process platform backend is packaged under `platform_backend`; localhost 5/5 READ, real-Qwen Agent READ E2E for GPU/queue/knowledge, queue-file-to-V2 normalization, task-state/priority normalization, missing-task normalization, approval-bound precondition forwarding, rejected WRITE, approved reversible WRITE, approval replay, TOCTOU, uncertain response/reconciliation, and approved cleanup pass in mock/simulated sandbox; production WRITE remains 0 |
| Non-mock AutoDrive staging | No non-mock endpoint is configured | OUT_OF_SCOPE for the single-node simulated release profile | The product target intentionally uses PLATFORM_STAGE_RUNTIME=mock and PLATFORM_GPU_RUNTIME=simulated; no non-mock cluster or physical multi-GPU validation is required |
| Platform HTTP bridge | stdio canonical MCP implementation | Localhost-only V2 `tools/call` HTTP transport | `/health`, stdio/in-process gateway tests, V2 adapter READ smoke, and narrow NOT_FOUND mapping pass |
| Result boundary | Typed result normalization, provenance, evidence qualification | Adapter must preserve raw result boundary and transport semantics | MCP mutation/identity/error tests |
| Persistence | SQLite event/checkpoint/claim/approval durability exists | Runtime-root layout and host bootstrap | SQLite path/readiness tests |
| Host API | Python `invoke`/`resume`/`reconcile` | Operator-facing CLI, pending approval inspection, health/readiness | CLI health/readiness smoke pass |
| Configuration | Hard-coded test defaults accepted by reference tests | Strict typed environment/JSON configuration without secrets in state/context/logs | Config validation tests and `.env.example` |
| Observability | Audit events carry provenance | Provider-safe telemetry and correlated production logs | Redaction/telemetry tests |
| CI | Local regression report | Pinned real dependency CI, shim detection, wheel/import/static audit | Hosted run #27 (`32680483362`) passed Python 3.11/3.12, wheel/import, compile, real-LangGraph, and static checks |
| Container/deployment | No production image | Non-root image, volumes, single-instance SQLite deployment contract | Hosted run #27 (`32680483362`) passed image build, non-root identity, health, no-secret readiness, and same-volume SQLite smoke; local daemon registry timeout still blocks local build |

## Single-node simulated release scope

The product target for this release is the validated single-node simulated
deployment. The stage and GPU behavior are intentionally simulated while the
Agent, Qwen Provider, LangGraph Runtime, gateway, platform execution layer,
RAG, SQLite/PostgreSQL backing services, and safety transactions remain real.

```text
NON_MOCK_AUTODRIVE=OUT_OF_SCOPE
PHYSICAL_MULTI_GPU=OUT_OF_SCOPE
MULTI_NODE_CLUSTER=OUT_OF_SCOPE
```

Absence of a non-mock AutoDrive endpoint is therefore not a release blocker.
The release boundary is `PLATFORM_STAGE_RUNTIME=mock`,
`PLATFORM_GPU_RUNTIME=simulated`, and
`AUTODRIVE_GATEWAY_BACKEND=in_process`. A fresh clean-restart model smoke that
produces a schema-invalid Qwen proposal remains a separate external Provider
revalidation item; strict DecisionIngress rejection is the expected safety
behavior and is not repaired by weakening the parser.

The V2 package includes a localhost-only HTTP bridge at `127.0.0.1:8765/mcp`.
It can either start the configured canonical stdio MCP command or use the
V2-owned in-process platform backend (`AUTODRIVE_GATEWAY_BACKEND=in_process`).
The current in-process validation uses a mock/no-trigger disposable task through
the normal V2 approval path. It proves five transport reads, rejected WRITE
with zero attempts, one approved priority change, no approval replay, stale
transaction rejection, and a mock-only post-dispatch response-drop fault that
enters `OUTCOME_UNKNOWN` and is reconciled with READ-only verification. The
task is restored and removed through fresh approved transactions; no production
mutation is performed. The response-drop hook is disabled by default and is
honored only when `PLATFORM_STAGE_RUNTIME=mock`. The bridge does not make the
stdio MCP server public, and the migrated backend excludes the old project's
semantic Agent/planning/evaluation packages.

## Non-goals retained

This integration work does not add a Planner, Router, second semantic model,
AUTO WRITE, distributed execution, multi-operator approval, or active-active
SQLite deployment. READ remains autonomous under deterministic Runtime guards;
WRITE remains frozen, human-approved, single-attempt execution.

## Retrieval evaluation evidence

The five-case `retrieval_golden.json` is retained as a smoke test. The final
V1.1 alignment set is the immutable 30-case JSONL under
`platform_backend/knowledge/eval/v1_1/`; its SHA-256 is
`7cb32e25fbb35a62274732558ed00f42aa98f20c871c7281127247efcb19f7ed`.
The evaluator calls the canonical V2 retriever and emits per-case artifacts
under runtime evaluation state. It does not import or restore the legacy
runtime evaluator. On 2026-08-24, local hashing and real Qwen dense both
executed 30/30 cases; Qwen preserved Top-5 recall, improved aggregate MRR and
had no Top-5 retrieval regression. `RERANKER_NOT_REQUIRED` is the current
corpus decision; local remains the default embedding mode.
