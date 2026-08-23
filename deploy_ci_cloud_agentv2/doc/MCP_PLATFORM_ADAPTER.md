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

## External smoke status

The V2 stdio bridge was validated on `127.0.0.1:8765/mcp` through the
production V2 adapter and the existing runtime stdio MCP server. The V2-owned
in-process bridge was also validated on a separate localhost port against the
mock/simulated platform runtime: GPU, queue, knowledge, and missing-task
normalization passed. The current sandbox has no task configuration, so
task-specific detail/diagnosis for existing tasks cannot yet be exercised and
no platform mutation has been attempted. The external platform result is
**PENDING**. The local sandbox suite independently proves
the implemented five READ tools, approval-before-WRITE, exactly-one sandbox
mutation, post-write verification, connection-drop handling, and
`OUTCOME_UNKNOWN` for uncertain remote WRITE outcomes.
