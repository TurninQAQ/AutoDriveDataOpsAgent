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
storage. Do not run active-active replicas or network-filesystem SQLite.
