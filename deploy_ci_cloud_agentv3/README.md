# AutoDriveDataOpsAgent V3.5

V3.5 is the current correctness-closure release of the V3 main architecture. V2 is preserved under `deploy_ci_cloud_agentv2/` for comparison and reuse of the simulated AutoDrive platform backend.

## Architecture

V3.5 is a **Single-Agent Guarded ReAct** system with four LangGraph control nodes:

```text
agent -> model_tools -> review -> execute_write
  ^          |                         |
  |          +-------------------------+
  +------------------------------------+
```

The graph is intentionally thin. Deterministic write reliability lives in `services/write_service.py`.

- `agent`: LLM reasoning and native tool-call selection.
- `model_tools`: executes model-visible safe MCP tools. READ/PREPARE calls can execute together; a Proposal must be the only tool call in its round.
- `review`: LangGraph `interrupt()` human review of a frozen `PendingAction`.
- `execute_write`: deterministic `WriteService` execution after exact fingerprint approval.

## Standard MCP boundary

A single shared tool implementation is exposed through two Streamable HTTP capability profiles:

```text
/mcp/agent
  READ + PREPARE + PROPOSAL
  no real WRITE tools

/mcp/runtime
  READ + runtime-internal reads + real WRITE
  no proposal tools
```

The server implementation targets the official Python MCP SDK v2 and exposes standard `tools/list` / `tools/call`. Agent and Runtime profiles are registered from the same shared Pydantic-backed ToolDefinition registry, so remote schemas and native-function schemas use one source of truth. The default local Agent path uses the official `Client(MCPServer)` in-process transport; the remote path uses the same official client with Streamable HTTP URLs. The custom registry-only `InProcessMCPClient` remains test-only. The repository includes official-SDK in-process and Streamable HTTP integration tests; they run when the `mcp` dependency is installed.

Run the MCP service after installing dependencies:

```bash
python -m pip install '.[test]'
autodrive-mcp-v3
```

Default endpoints:

```text
http://127.0.0.1:8000/mcp/agent
http://127.0.0.1:8000/mcp/runtime
```

## Write lifecycle

```text
Proposal
-> PendingAction
-> HITL Review
-> Approve exact fingerprint
-> Global Precondition Recheck
-> Action-specific Revalidation
-> runtime-derived idempotency key
-> single mutation attempt
-> Runtime MCP WRITE
-> Observe Again
-> Post-write Verification
-> FinalGuard
```

`Proposal != Execution`: model-visible `propose_*` tools have zero platform side effect. Real write tools are absent from the Agent capability profile.

`API success != business success`: a successful MCP call is followed by action-specific real-platform read-back. Observation errors are not treated as absence evidence. Only `WriteResult(status="VERIFIED", verified=True)` can produce final status `write_verified`.

Potentially applied mutations are not blindly retried. The mutation-attempt key is derived inside `WriteService` from the recomputed approved fingerprint and is not trusted from mutable workflow state. A transport exception after dispatch enters reconciliation by READ; if the resulting business state cannot be confirmed, the result remains `UNKNOWN_OUTCOME`.

`resume_task(datasets=None)` is resolved from one fail-closed Airflow snapshot before review into an explicit failed-dataset list. That same snapshot is reused as the verification baseline, and a resume-specific precondition (including the baseline hash) is fingerprint-bound to the PendingAction. Immediately before mutation, WriteService re-reads the selected datasets: a dynamically resolved target must still have `FAILED` as its latest observed DagRun state or the action returns `PRECONDITION_FAILED` without crossing the mutation boundary. After mutation, verification requires a new progressing run for every approved dataset. Stop verification requires target execution quiescence; whole-task stop additionally requires GPU release and queue removal.

## Task creation pipeline

```text
TaskDraft
-> deterministic platform defaults
-> schema validation
-> PreparedArtifact (YAML + hash)
-> propose_submit_task(artifact_id)
-> HITL review of the frozen artifact
-> submit_task
-> read-back verification
```

GPU IDs, image tags, timeouts, scheduler defaults, and other platform defaults come from the repository-owned platform configuration, not from the model.

## Provider layer

`providers/qwen.py` uses the OpenAI-compatible Qwen chat-completions endpoint with native `tools=[...]` function calling. The legacy V2 Agent Decision JSON DSL is not used by V3.

## Local validation

The V3 unit tests use a fake facade and do not call a paid model or a real cluster:

```bash
python -m pytest -q deploy_ci_cloud_agentv3/tests
```

Coverage includes capability isolation, proposal zero-side-effect behavior, approval-time tamper attacks, runtime-derived idempotency, resume target freezing, resume-specific stale-approval revalidation, fail-closed single-snapshot resume baselines, resume before/after run-identity verification, stop quiescence verification, edit fingerprint invalidation, stale approval blocking, delete/submit verification, UNKNOWN_OUTCOME reconciliation, mixed-proposal native-tool-call rejection completeness, pipeline artifact binding, structured FinalGuard gating, context tool-call grouping, official-MCP mainline wiring, and Graph HITL flows. See `doc/V3.5_MCP_LIFESPAN_CLOSURE.md` for the current mounted-MCP closure and integration-test gate. In this build environment the official MCP/LangGraph E2E tests remain dependency-gated because external package installation is unavailable; the dependency-independent lifespan regression and all other local tests pass.

## Scope

The platform backend remains the single-node mock/simulated AutoDrive training platform reused from V2. V3 does not claim a real production cluster, multi-agent architecture, vector database, or exactly-once distributed mutation protocol.
