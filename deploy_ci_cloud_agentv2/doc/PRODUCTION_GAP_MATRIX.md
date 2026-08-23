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
| Provider | Deterministic and scripted offline providers | Structured HTTP/Qwen adapter, timeout, retry, rate-limit handling, auth isolation, telemetry | Local fake HTTP provider tests and malformed/429 coverage pass |
| Platform | In-memory READ/WRITE facades | JSON-RPC/MCP-over-HTTP facade with strict transport/error mapping | Local fake MCP transport and approved WRITE sandbox pass |
| Result boundary | Typed result normalization, provenance, evidence qualification | Adapter must preserve raw result boundary and transport semantics | MCP mutation/identity/error tests |
| Persistence | SQLite event/checkpoint/claim/approval durability exists | Runtime-root layout and host bootstrap | SQLite path/readiness tests |
| Host API | Python `invoke`/`resume`/`reconcile` | Operator-facing CLI, pending approval inspection, health/readiness | CLI health/readiness smoke pass |
| Configuration | Hard-coded test defaults accepted by reference tests | Strict typed environment/JSON configuration without secrets in state/context/logs | Config validation tests and `.env.example` |
| Observability | Audit events carry provenance | Provider-safe telemetry and correlated production logs | Redaction/telemetry tests |
| CI | Local regression report | Pinned real dependency CI, shim detection, wheel/import/static audit | GitHub Actions workflow added; local workflow syntax/static checks pass |
| Container/deployment | No production image | Non-root image, volumes, single-instance SQLite deployment contract | Dockerfile and deployment docs added; local daemon registry timeout prevented build |

## Non-goals retained

This integration work does not add a Planner, Router, second semantic model,
AUTO WRITE, distributed execution, multi-operator approval, or active-active
SQLite deployment. READ remains autonomous under deterministic Runtime guards;
WRITE remains frozen, human-approved, single-attempt execution.
