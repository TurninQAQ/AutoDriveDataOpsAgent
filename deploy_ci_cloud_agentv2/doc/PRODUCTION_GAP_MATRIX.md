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
| Platform | In-memory READ/WRITE facades | V2-owned platform execution layer plus custom JSON-RPC-over-HTTP transport | V2 in-process platform backend is packaged under `platform_backend`; localhost READ smoke and missing-task normalization pass; sandbox task target remains pending |
| Platform HTTP bridge | stdio canonical MCP implementation | Localhost-only V2 `tools/call` HTTP transport | `/health`, stdio/in-process gateway tests, V2 adapter READ smoke, and narrow NOT_FOUND mapping pass |
| Result boundary | Typed result normalization, provenance, evidence qualification | Adapter must preserve raw result boundary and transport semantics | MCP mutation/identity/error tests |
| Persistence | SQLite event/checkpoint/claim/approval durability exists | Runtime-root layout and host bootstrap | SQLite path/readiness tests |
| Host API | Python `invoke`/`resume`/`reconcile` | Operator-facing CLI, pending approval inspection, health/readiness | CLI health/readiness smoke pass |
| Configuration | Hard-coded test defaults accepted by reference tests | Strict typed environment/JSON configuration without secrets in state/context/logs | Config validation tests and `.env.example` |
| Observability | Audit events carry provenance | Provider-safe telemetry and correlated production logs | Redaction/telemetry tests |
| CI | Local regression report | Pinned real dependency CI, shim detection, wheel/import/static audit | Hosted run #12 passed Python 3.11/3.12, wheel/import, compile, real-LangGraph, and static checks |
| Container/deployment | No production image | Non-root image, volumes, single-instance SQLite deployment contract | Hosted run #12 passed image build, non-root identity, health, no-secret readiness, and same-volume SQLite smoke; local daemon registry timeout still blocks local build |

The V2 package includes a localhost-only HTTP bridge at `127.0.0.1:8765/mcp`.
It can either start the configured canonical stdio MCP command or use the
V2-owned in-process platform backend (`AUTODRIVE_GATEWAY_BACKEND=in_process`).
Initial validation is READ-only; it does not make the stdio MCP server public
and it does not create a sandbox task. The migrated backend excludes the old
project's semantic Agent/planning/evaluation packages.

## Non-goals retained

This integration work does not add a Planner, Router, second semantic model,
AUTO WRITE, distributed execution, multi-operator approval, or active-active
SQLite deployment. READ remains autonomous under deterministic Runtime guards;
WRITE remains frozen, human-approved, single-attempt execution.
