# Operations and recovery

Use `autodrive-agent health` for liveness and `autodrive-agent ready` for local
readiness. Keep SQLite under the runtime volume, never inside the source tree.

If a process stops after `MutationStarted` but before a durable mutation result,
the Runtime marks the transaction `RECONCILIATION_REQUIRED` and blocks replay.
Run `autodrive-agent reconcile --thread-id ...`; only a deterministic,
predeclared verification read can clear the unknown outcome. Any new mutation
requires a new transaction and new approval according to its ToolSpec
idempotency policy.

The current deployment contract is one active runtime instance on local
storage, and Runtime enforces it with a Linux advisory lock at
`<runtime_root>/run/runtime.lock`. A second process receives
`RUNTIME_INSTANCE_ALREADY_ACTIVE` and must not be interpreted as a crashed
mutation. The kernel releases the lock after clean exit or process death.
Do not run active-active replicas or network-filesystem SQLite; set
`single_instance=false` is unsupported and rejected.
