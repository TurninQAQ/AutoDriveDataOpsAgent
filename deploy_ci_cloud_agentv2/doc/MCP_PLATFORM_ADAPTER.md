# MCP platform adapter

`platform/mcp.py` implements the concrete **custom AutoDrive JSON-RPC tool
gateway** `tools/call` transport. It is not a claim of standards-compliant
MCP session/transport support; the facade boundary intentionally keeps that
transport detail below Runtime.
It exposes the five READ tools and the five existing WRITE tools from the
sealed ToolCatalog. The facade is synchronous because the frozen precondition
and verification reader protocol is synchronous; ToolRegistry runs sync
handlers in worker threads, so READ batches remain concurrent at the Runtime
boundary.

External responses remain raw untrusted data until Tool Runtime ingress takes
the canonical snapshot and applies the existing strict result/provenance/
evidence contracts. READ transport failures are bounded and side-effect-free.
WRITE timeout/5xx, malformed responses, connection loss after dispatch, and
remote JSON-RPC tool errors are conservatively `OUTCOME_UNKNOWN`. This
adapter has no deterministic proof of non-dispatch for a remote error, so it
never classifies that error as `FAILED_BEFORE_EFFECT`.

The local integration suite uses a fake JSON-RPC server to prove READ, approval
pause, rejected WRITE with zero mutation, approved WRITE exactly once, and
post-write verification.

The V2 package also provides a localhost-only HTTP bridge at
`127.0.0.1:8765/mcp`. Its implementation is
`deploy_ci_cloud_agentv2/platform/http_gateway.py`; it supports two transport
backends:

* `AUTODRIVE_GATEWAY_BACKEND=stdio` starts the configured canonical stdio MCP
  command (`AUTODRIVE_STDIO_MCP_COMMAND`, default `mcp-server`).
* `AUTODRIVE_GATEWAY_BACKEND=in_process` uses the V2-owned platform execution
  layer under `deploy_ci_cloud_agentv2/platform_backend/`.

Both modes forward only the ten V2 tool names. The in-process migration includes
the platform core, read services, RAG index, observability helpers, mutation
mechanics, stage scripts, and DAG template assets. It intentionally does not
bring the previous project's Agent/planning/evaluation semantic layers into V2.
The bridge remains transport-only and introduces no semantic routing or second
approval/completion authority. `GET /health` is a non-mutating liveness endpoint.

The bridge normalizes only transport-shape differences and one deterministic
platform absence contract: an unscoped V2 queue
read (`task_name: null`) becomes the existing stdio tool's empty-string form,
and GPU observation explicitly disables the legacy stale-reservation cleanup
flag. The canonical `Task config not found:` discriminator becomes V2
`status=NOT_FOUND, exists=false`; generic task/config/transport errors remain
bounded platform errors. It never captures a precondition, authorizes a WRITE,
or fabricates a WRITE argument. Direct gateway WRITE calls without an
approval-bound `precondition` are rejected with `WRITE_PRECONDITION_REQUIRED`.

For the mock/simulated sandbox only, `AUTODRIVE_PLATFORM_SUBMIT_NO_TRIGGER=1`
causes `submit_task` to create the task configuration and DAG without starting
a scheduler run. The default, with that flag absent, retains the normal
trigger behavior. The V2 Runtime passes the approved precondition as detached
data; the backend recomputes the V2 target/fingerprint and performs its own
platform-side precondition check before invoking the mutation handler.

## External smoke status

The V2-owned in-process bridge was validated on
`127.0.0.1:8765/mcp` against the mock/simulated platform runtime: GPU, queue,
knowledge, and missing-task normalization passed. A disposable no-trigger
sandbox task was created and removed through the normal V2 approval path; the
production mutation count remains zero. The current clean sandbox has no task
configuration, so task-specific detail/diagnosis for an existing task and a
non-local AutoDrive endpoint remain **UNVERIFIED_EXTERNAL**. The local sandbox
suite independently proves
the implemented five READ tools, approval-before-WRITE, exactly-one sandbox
mutation, post-write verification, connection-drop handling, and
`OUTCOME_UNKNOWN` for uncertain remote WRITE outcomes.

The product release profile is the localhost single-node simulated platform:
`PLATFORM_STAGE_RUNTIME=mock`, `PLATFORM_GPU_RUNTIME=simulated`, and
`AUTODRIVE_GATEWAY_BACKEND=in_process`. Non-mock AutoDrive, physical
multi-GPU hardware, and a multi-node cluster are explicitly `OUT_OF_SCOPE`.
The final clean-restart smoke used one canonical process on port `8765`; port
`8766` was not listening and no process retained a deleted legacy working
directory.
