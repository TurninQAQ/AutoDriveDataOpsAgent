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
database. Active-active replicas, NFS state, and multiple writers are not
supported by this phase.
