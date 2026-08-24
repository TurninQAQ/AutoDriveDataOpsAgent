# V2 Single-Node Simulated Deployment

## Scope

This is the intended release profile for AutoDriveDataOpsAgent V2:

```text
real Alibaba Bailian/Qwen Agent Provider
real LangGraph runtime
real V2 Runtime, persistence, verification, and WRITE safety
real V2 HTTP JSON-RPC gateway
V2-owned in-process platform backend
mock stage runtime
simulated GPU runtime
```

The profile does not claim physical multi-GPU, multi-node scheduling, or a
non-mock AutoDrive cluster. Those are explicitly outside this product scope:

```text
NON_MOCK_AUTODRIVE=OUT_OF_SCOPE
PHYSICAL_MULTI_GPU=OUT_OF_SCOPE
```

## Canonical topology

```text
Qwen/Bailian
    -> V2 Agent and LangGraph Runtime
    -> 127.0.0.1:8765/mcp
    -> in-process platform_backend
    -> mock stage + simulated GPU runtime
    -> Airflow/PostgreSQL backing services
```

The gateway is transport-only. It does not select tools, approve WRITE,
qualify evidence, or decide completion. The Agent remains the semantic
authority and the Runtime remains the safety authority.

## Configuration

The non-secret release profile is kept outside the repository at:

```text
/home/ubuntu/project/autodrive_dataops_runtimev2/config/single_node_simulated.env
```

It defines:

```text
AUTODRIVE_ENVIRONMENT=single-node-simulated
AUTODRIVE_PROVIDER=qwen
AUTODRIVE_MODEL=qwen3.7-plus-2026-05-26
AUTODRIVE_PROVIDER_STRUCTURED_MODE=json_schema
AUTODRIVE_PROVIDER_THINKING_MODE=auto
AUTODRIVE_PROVIDER_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
AUTODRIVE_PROVIDER_API_KEY_ENV=DASHSCOPE_API_KEY
AUTODRIVE_GATEWAY_BACKEND=in_process
AUTODRIVE_PLATFORM_ENDPOINT=http://127.0.0.1:8765/mcp
PLATFORM_STAGE_RUNTIME=mock
PLATFORM_GPU_RUNTIME=simulated
PLATFORM_RAG_EMBED_PROVIDER=local
```

The API key is not part of this file or the repository. The operator loads
`DASHSCOPE_API_KEY` from the external secret file
`/home/ubuntu/project/auth/ali.api` into the current service process only.
The file is mode `600`; its contents must never be logged, committed, or
stored in SQLite/evidence.

For simulated task creation only, the deployment profile sets
`AUTODRIVE_PLATFORM_SUBMIT_NO_TRIGGER=1`. This mock-only setting materializes
the task configuration and DAG without starting an Airflow scheduler run. It
does not bypass V2 approval or transaction safety.

## Start and stop

Start the required single-node backing services from the runtime installation:

```bash
cd /home/ubuntu/project/autodrive_dataops_runtime
bin/platform start
```

Start the one canonical V2 gateway from the external launcher:

```bash
/home/ubuntu/project/autodrive_dataops_runtimev2/bin/start-v2-gateway
```

The launcher loads only the non-secret profile, uses the V2 source tree, and
executes the V2 gateway in `in_process` mode. It binds only to localhost.

Stop the gateway with its service supervisor/process owner, then stop backing
services with:

```bash
cd /home/ubuntu/project/autodrive_dataops_runtime
bin/platform stop
```

The final topology has one gateway on port `8765`; port `8766` is not part of
the release profile. No process may use the deleted `deploy_ci_cloud_agent`
directory.

## Health and readiness

```text
GET http://127.0.0.1:8765/health
GET http://127.0.0.1:8080/api/v2/monitor/health
GET http://127.0.0.1:8081/execution/health
PostgreSQL 127.0.0.1:5432
autodrive-agent health
autodrive-agent ready
```

`ready` requires the configured Provider secret, gateway/platform
configuration, sealed tool catalog, and writable SQLite state. It is a local
configuration/readiness check and is not a substitute for a real model call.

The CLI is supplied by the V2 runtime environment at
`/home/ubuntu/project/autodrive_dataops_runtimev2/.venv/bin`; it is not
required to be globally installed.

## Persistence

Runtime state is outside the source tree:

```text
/home/ubuntu/project/autodrive_dataops_runtimev2/state/autodrive.sqlite3
```

The SQLite event, checkpoint, approval, and execution-claim stores are
protected by the single-instance runtime lock. A restart must restore durable
state and must never replay a mutation after `MutationStarted` without
reconciliation.

## Release validation boundary

The validated release checks cover real Qwen Provider integration, real
LangGraph, the V2 gateway, simulated platform READ/WRITE, approval replay
protection, TOCTOU rejection, mock-only post-dispatch response loss,
`OUTCOME_UNKNOWN`, READ-only reconciliation, cleanup, persistence, packaging,
and hosted CI. Production mutation count is zero.

The current clean-restart Provider smoke must continue to be reported
separately from historical Provider evidence. If a model emits a malformed V2
decision, strict `DecisionIngress` rejection is correct; the parser and safety
boundary must not be weakened to force readiness.

## Intentional limitations

This deployment manages a simulated GPU/runtime environment on one node. It
does not validate physical GPU hardware, a real multi-node cluster, or a
non-mock AutoDrive control plane. Those capabilities are future product
scopes, not unresolved requirements of this release profile.
