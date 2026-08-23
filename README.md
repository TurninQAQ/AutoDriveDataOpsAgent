# AutoDriveDataOpsAgent V2

AutoDriveDataOpsAgent V2 is a deterministic Runtime around a single semantic
Agent. READ tools are autonomous within Runtime guards. WRITE operations are
human-approved, transaction-bound, revalidated, claimed once, verified, and
recoverable through SQLite-backed audit state.

The repository root is the canonical packaging source:

```bash
python -m pip install '.[test]'
autodrive-agent health
autodrive-agent ready
```

The implementation and tests live under `deploy_ci_cloud_agentv2`. The root
package installs that exact tree; there is no second conflicting package
definition.

## Validation status

The architecture and local correctness suites use scripted providers and fake
platform transports. They do not call a paid model or production platform.
Real LangGraph 1.2.11 tests run only when that package is installed; in a
shim-only environment those tests are skipped and reported as skipped.

The structured HTTP provider and the custom AutoDrive JSON-RPC platform
gateway are implemented, but real external provider/platform smoke tests and
production deployment remain explicit operational steps. First WRITE testing
must use a sandbox.

Runtime state is kept outside the source tree. Set `AUTODRIVE_RUNTIME_ROOT`
or provide a strict JSON config; the default is
`/home/ubuntu/project/autodrive_dataops_runtimev2`.
Durable `invoke`, `resume`, and `reconcile` operations hold a Linux advisory
lock at `<runtime_root>/run/runtime.lock` for their complete lifetime. The
single-instance rule is enforced in code: a competing process gets
`RUNTIME_INSTANCE_ALREADY_ACTIVE`, while process death releases the kernel
lock and leaves normal post-`MutationStarted` reconciliation semantics intact.

See:

- `deploy_ci_cloud_agentv2/doc/PRODUCTION_RUNTIME.md`
- `deploy_ci_cloud_agentv2/doc/REAL_PROVIDER_INTEGRATION.md`
- `deploy_ci_cloud_agentv2/doc/MCP_PLATFORM_ADAPTER.md`
- `deploy_ci_cloud_agentv2/doc/DEPLOYMENT.md`
