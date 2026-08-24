# Production runtime boundary

`host.py` is the only production assembly layer. It loads strict typed
configuration, creates the Qwen-compatible structured provider and the MCP
facade, then calls the existing `build_system_context()`. The host does not
select tools or bypass Runtime checks.

Runtime state is outside the source tree:

```text
/home/ubuntu/project/autodrive_dataops_runtimev2/
  config data state logs run secrets
```

The `autodrive-agent` CLI delegates to `invoke`, `resume`, and `reconcile`.
`health` is liveness only. `ready` validates configuration, the sealed tool
catalog/hash, configured endpoints, presence of the referenced provider
secret, and SQLite writability without a platform WRITE. It reports only
secret presence and never returns or logs the secret itself.

The default persistence model is one active process and one local SQLite WAL
database. This is enforced in code: durable Runtime operations acquire a
non-blocking Linux advisory lock at `<runtime_root>/run/runtime.lock` for their
entire lifetime, including precondition checks, mutation, verification, and
durable result recording. The lock is kernel-owned, so a clean exit or process
death releases ownership; the lock file's presence is not treated as
ownership. A competing process receives `RUNTIME_INSTANCE_ALREADY_ACTIVE` and
does not reinterpret the live operation as a crash or start reconciliation.
`single_instance=false` is rejected because no multi-process/HA ownership
protocol exists in this phase. Active-active replicas, NFS state, and multiple
writers remain unsupported.

## Integration status

The frozen Runtime, real LangGraph integration, strict Qwen Provider, custom
JSON-RPC gateway, and mock/simulated platform profile are validated. The
single-node release uses the real Qwen model and real V2 safety/runtime while
the stage and GPU behavior are intentionally simulated. Non-mock AutoDrive and
physical multi-GPU validation are outside this product scope. No production
WRITE is performed by the validation suite.
