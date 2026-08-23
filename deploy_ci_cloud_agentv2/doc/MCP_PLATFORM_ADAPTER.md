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
`deploy_ci_cloud_agentv2/platform/http_gateway.py`; it starts the configured
canonical stdio MCP command (`AUTODRIVE_STDIO_MCP_COMMAND`, default
`mcp-server`) and forwards only the ten V2 tool names. It does not copy
platform business logic or introduce semantic routing or a second
approval/completion authority. `GET /health` is a non-mutating liveness
endpoint.

The bridge normalizes only transport-shape differences: an unscoped V2 queue
read (`task_name: null`) becomes the existing stdio tool's empty-string form,
and GPU observation explicitly disables the legacy stale-reservation cleanup
flag. It never captures a precondition, authorizes a WRITE, or fabricates a
WRITE argument.

## External smoke status

The local V2 bridge was validated on `127.0.0.1:8765/mcp` through the
production V2 adapter and the existing runtime stdio MCP server for the
available global READ surfaces. No configured external AutoDrive endpoint or
task target is present, so task-specific READs return bounded backend errors
and no external platform request or production WRITE was attempted. The
external platform result is **PENDING**. The local sandbox suite independently proves
the implemented five READ tools, approval-before-WRITE, exactly-one sandbox
mutation, post-write verification, connection-drop handling, and
`OUTCOME_UNKNOWN` for uncertain remote WRITE outcomes.
