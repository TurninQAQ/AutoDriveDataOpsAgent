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

## External smoke status

The current environment has no configured `AUTODRIVE_PLATFORM_ENDPOINT` beyond
the default local placeholder and no AutoDrive gateway listening there. No
external platform request or production WRITE was attempted. The external
platform result is **PENDING**. The local sandbox suite independently proves
the implemented five READ tools, approval-before-WRITE, exactly-one sandbox
mutation, post-write verification, connection-drop handling, and
`OUTCOME_UNKNOWN` for uncertain remote WRITE outcomes.
