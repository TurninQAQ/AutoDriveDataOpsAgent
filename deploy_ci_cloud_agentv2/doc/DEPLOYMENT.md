# Deployment boundary

The repository root is the only supported package build/install path:

```bash
python -m build
python -m pip install dist/autodrive_dataops_agent_v2-2.0.0-py3-none-any.whl
```

Use the `Dockerfile` for a non-root image. Runtime state and logs are mounted
outside the source tree. The current SQLite safety model is single-instance
and this is code-enforced: durable `invoke`, `resume`, and `reconcile`
operations take a non-blocking Linux advisory lock at
`<runtime_root>/run/runtime.lock` for the complete operation. A second process
receives `RUNTIME_INSTANCE_ALREADY_ACTIVE`; it does not mutate state or
manufacture a reconciliation result. Kernel lock release makes the runtime
recoverable after clean exit or process death. Do not run active-active workers
against the same database or put it on a network filesystem.
`single_instance=false` is rejected because a distributed ownership protocol
is not implemented. A future HA deployment requires a separate architecture
phase.

The image healthcheck is liveness-oriented. Readiness additionally requires a
valid sealed catalog, writable SQLite state, configured endpoints, and the
referenced provider secret to be present. Secrets are supplied at runtime and
are never baked into the image, prompts, audit events, or logs.

Real provider and platform connectivity must be validated separately from
local fake-transport tests. The platform adapter in this repository is a
custom AutoDrive JSON-RPC gateway, not a claim of standards-compliant MCP.
The current local validation environment has no configured external provider
secret or AutoDrive endpoint. Docker build was attempted with the installed
daemon but was blocked resolving `docker.io/library/python:3.12-slim`; therefore
external provider, external platform, and Docker are not reported as PASS.
