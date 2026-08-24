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

The implementation, tests, and the migrated transport-independent platform
execution layer live under `deploy_ci_cloud_agentv2`. The root package installs
that exact tree; there is no second conflicting V2 package definition. The
former `deploy_ci_cloud_agent` tree has been removed from the working tree
after migrating the required platform execution assets into V2; a separately
deployed runtime service remains an external process boundary only when stdio
mode is selected.

## Validation status

The architecture and local correctness suites use scripted providers and fake
platform transports. They do not call a paid model or production platform.
Real LangGraph 1.2.11 tests run only when that package is installed; in a
shim-only environment those tests are skipped and reported as skipped.

The structured HTTP provider and the custom AutoDrive JSON-RPC platform
gateway are implemented. The release target is a single-node simulated
platform; non-mock AutoDrive and physical multi-GPU infrastructure are
explicitly out of scope. First WRITE testing must use the simulated sandbox
and the normal V2 approval path.

Current validation status:

```text
Local correctness / real LangGraph       PASS
Local provider adapter sandbox           PASS
Local platform JSON-RPC sandbox          PASS
Real Qwen Agent Provider                 PASS: qwen-plus-2025-07-28 READ E2E
Single-node simulated platform           PASS: mock stage + simulated GPU runtime
Non-mock AutoDrive / physical GPU        OUT_OF_SCOPE
Fresh clean-restart Qwen E2E             BLOCKED_EXTERNAL: strict ingress rejected malformed proposals
Hosted CI (Python 3.11/3.12)             PASS: hosted run #27 (32680483362)
Hosted Docker build/runtime smoke        PASS: hosted run #27 (32680483362)
Local Docker build/run                   BLOCKED: Docker Hub registry timeout
V2 in-process platform backend           PASS: localhost mock/simulated READ smoke
Missing task contract                    PASS: deterministic NOT_FOUND/exists=false
Sandbox task creation/WRITE               PASS: one mock no-trigger disposable task created and removed through V2 approval; production WRITE 0
```

Runtime state is kept outside the source tree. Set `AUTODRIVE_RUNTIME_ROOT`
or provide a strict JSON config; the default is
`/home/ubuntu/project/autodrive_dataops_runtimev2`.
Durable `invoke`, `resume`, and `reconcile` operations hold a Linux advisory
lock at `<runtime_root>/run/runtime.lock` for their complete lifetime. The
single-instance rule is enforced in code: a competing process gets
`RUNTIME_INSTANCE_ALREADY_ACTIVE`, while process death releases the kernel
lock and leaves normal post-`MutationStarted` reconciliation semantics intact.

See:

- `deploy_ci_cloud_agentv2/doc/V2_REQUIREMENT_TRACEABILITY_MATRIX.md`
- `deploy_ci_cloud_agentv2/doc/PRODUCTION_RUNTIME.md`
- `deploy_ci_cloud_agentv2/doc/REAL_PROVIDER_INTEGRATION.md`
- `deploy_ci_cloud_agentv2/doc/MCP_PLATFORM_ADAPTER.md`
- `deploy_ci_cloud_agentv2/doc/DEPLOYMENT.md`
- `deploy_ci_cloud_agentv2/doc/SINGLE_NODE_SIMULATED_DEPLOYMENT.md`
