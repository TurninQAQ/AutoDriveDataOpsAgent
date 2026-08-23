# AutoDriveDataOpsAgent V2.0

AutoDriveDataOpsAgent V2.0 is a **single-loop DataOps Agent with autonomous READs and human-approved WRITEs**. The Agent is the only semantic next-action authority; deterministic Runtime components own validation, evidence, completion, approval, execution safety, audit, and recovery.

The frozen architecture contract is `doc/AutoDriveDataOpsAgent_V2_ARCHITECTURE.md`. `doc/Luna_OPERATING_PRINCIPLES.md` is advisory operating guidance and never overrides Runtime safety invariants.

## Host API

```python
from deploy_ci_cloud_agentv2 import build_system_context, invoke, resume
from deploy_ci_cloud_agentv2.safety.approval import ApprovalDecision, ResumeInput

context = build_system_context(provider, read_facade=platform_facade)
result = await invoke("resume task_A", thread_id="thread-1", system_context=context)

if result.status == "INTERRUPTED":
    pending = result.pending_interrupt
    result = await resume(
        thread_id="thread-1",
        resume_input=ResumeInput(
            ApprovalDecision.APPROVE,
            pending.approval_request_id,
            pending.transaction_id,
            pending.fingerprint,
        ),
        system_context=context,
    )
```

## Production host

The concrete production assembly is in `host.py`: a strict environment/JSON
`RuntimeConfig`, `QwenProvider` over an OpenAI-compatible endpoint, and the
custom AutoDrive JSON-RPC `MCPPlatformFacade` (not a standards-compliant MCP
transport claim). Runtime state defaults to
`/home/ubuntu/project/autodrive_dataops_runtimev2`, outside the source tree.
When a durable path is configured, Runtime operations are protected by a
kernel advisory lock at `<runtime_root>/run/runtime.lock` for the full
`invoke`/`resume`/`reconcile` lifetime. The deployment is deliberately
single-instance; a competing process receives
`RUNTIME_INSTANCE_ALREADY_ACTIVE`, and `single_instance=false` is rejected.

The minimal operator boundary is the `autodrive-agent` CLI:

```bash
autodrive-agent health
autodrive-agent ready
autodrive-agent invoke "task_A status" --thread-id task-A
autodrive-agent pending --thread-id task-A
autodrive-agent approve --thread-id task-A --approval-request-id ... --transaction-id ... --fingerprint ...
autodrive-agent reject --thread-id task-A --approval-request-id ... --transaction-id ... --fingerprint ...
autodrive-agent reconcile --thread-id task-A
```

When the platform is available through its canonical stdio MCP launcher, start
the V2 transport bridge with:

```bash
export AUTODRIVE_STDIO_MCP_COMMAND=mcp-server
python -m deploy_ci_cloud_agentv2.platform.http_gateway
```

For a single-tree deployment using the platform execution code packaged inside
V2, use the in-process backend instead:

```bash
export AUTODRIVE_GATEWAY_BACKEND=in_process
export AUTODRIVE_RUNTIME_ROOT=/home/ubuntu/project/autodrive_dataops_runtimev2
python -m deploy_ci_cloud_agentv2.platform.http_gateway
```

The migrated `platform_backend` contains deterministic platform core/services,
RAG/knowledge, observability, mutation mechanics, stage scripts, and DAG
templates. The previous project's semantic Agent/planning/evaluation layers
are intentionally not imported into V2.

For the mock/simulated sandbox only, setting
`AUTODRIVE_PLATFORM_SUBMIT_NO_TRIGGER=1` makes `submit_task` create the task
configuration and DAG without starting a scheduler run. The normal trigger
behavior remains the default when that flag is absent. Approved WRITE
preconditions are forwarded as detached data to the backend, where the V2
target and precondition fingerprint are recomputed before the platform handler
is called.

It binds to `127.0.0.1:8765` and serves the V2-compatible `POST /mcp`
JSON-RPC `tools/call` endpoint plus non-mutating `GET /health`. The bridge is
transport-only; approval, mutation claims, verification, and completion remain
in V2 Runtime.

The CLI delegates only to `invoke()`, `resume()`, and `reconcile()`; it has no
direct WRITE-tool bypass. `health` is liveness only. `ready` checks local
configuration, sealed catalog/hash, and SQLite writability without contacting
or mutating the platform.

`invoke()` never bypasses an outstanding approval. `resume()` can authorize only the exact frozen `WriteTransaction` represented by the pending interrupt. A WRITE requires explicit approval, revalidated preconditions, one single-use `ExecutionClaim`, at most one mutation attempt, and deterministic verification. There is no AUTO WRITE path.

## Canonical loop

```text
START -> Agent
Agent -> READ Runtime -> Observation -> Agent
Agent -> WRITE Guard -> Approval interrupt
Approval -> Revalidate -> ExecutionClaim -> Mutation
         -> ActionVerifier -> OperationalGoalVerifier -> Goal Resolution -> Agent
Agent -> ResponseCompletionGate -> Agent / END
Runtime terminal -> END
```

WRITE guard outcomes are exactly `INVALID`, `DENIED`, and `APPROVAL_REQUIRED`. Human rejection and deterministic policy denial are goal-level outcomes. Runtime-level failures use `ControlledTerminalOutcome`.

## Durability and recovery

Pass `durable_path=` to `build_system_context()` to use the V2-local SQLite durability boundary. The event log is immutable audit truth; safety-critical capability transitions use database CAS and crash-consistent audit events. Runtime checkpoints carry an integrity digest and must agree with the durable event tail.

If a process dies after `MutationStarted` but before a durable mutation result, the Runtime does **not** replay the WRITE. It transitions to `REQUIRES_RECONCILIATION`. Reconciliation uses only verifier reads predeclared by the frozen `ToolSpec`; any later execution requires a new `WriteTransaction` and new human approval.

## Tool model

READ tools may run autonomously and in bounded parallel batches when every member is `parallel_safe`. Successful siblings survive partial batch failures. External observations are untrusted data and must pass strict result-contract, scope, identity, provenance, freshness, and evidence qualification.

WRITE tools are serialized. The built-in V2 platform facade exposes:

- `resume_task`
- `submit_task`
- `stop_task`
- `delete_task`
- `set_task_priority`

WRITE `ToolSpec` metadata includes risk, idempotency/reconciliation policy, affected entities, preconditions, and deterministic verification reads.

## Evaluation

`deploy_ci_cloud_agentv2.evaluation` derives Section 70 metrics from immutable audit events. Metrics that require semantic ground truth (`False Success Rate`, `Goal State Macro-F1`, wrong-target evaluation) require explicit evaluator labels instead of treating model output as truth.

## Tests

From the repository root:

```bash
pytest -q deploy_ci_cloud_agentv2/tests
python -m compileall -q deploy_ci_cloud_agentv2
```

Correctness tests make no real provider/API calls. The production package requires
the pinned `langgraph==1.2.11`, `httpx`, and the declared optional-RAG adapter
libraries. The marked real-runtime tests assert
that LangGraph is loaded from site-packages and exercise real `interrupt()` /
`Command(resume=...)`. When LangGraph is unavailable in an offline review
environment, `tests/conftest.py` installs a **test-only compatibility harness**
so node/routing invariants can still be exercised; production Runtime code has
no fallback semantic loop.

The 43-item V2 requirement-to-evidence mapping is maintained in
`doc/V2_REQUIREMENT_TRACEABILITY_MATRIX.md`.

## V1 boundary

V2 has no runtime imports from `deploy_ci_cloud_agent`. V1 may be read as historical/reference material only; autonomy-specific V1 logic is not migrated into V2.

Real external provider and platform smoke remain separate from local tests. In
the current review environment no provider API key or AutoDrive endpoint is
configured, so those results are reported as pending. Hosted GitHub Actions
run #21 (`32656130224`, commit `d9f06bf`) validated the Python 3.11/3.12 matrix, wheel/import/static checks, and
the non-root container health/readiness/persistent-volume smoke. Local Docker
build/run remains dependent on access to the `python:3.12-slim` registry image.

Optional dense retrieval configuration, sidecar compatibility, and validation
evidence are documented in `doc/DENSE_EMBEDDING_RUNTIME.md`. The offline default
remains BM25 + feature hashing; an external embedding secret is never required
for normal platform startup.
