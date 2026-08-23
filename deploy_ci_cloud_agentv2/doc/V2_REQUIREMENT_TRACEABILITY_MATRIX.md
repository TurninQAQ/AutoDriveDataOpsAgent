# V2 Requirement Traceability Matrix

This matrix maps the frozen V2 Definition of Done in
`AutoDriveDataOpsAgent_V2_ARCHITECTURE.md`, section 72, to the current
implementation and independently rerun evidence. It does not replace the
architecture contract and does not claim external validation.

| # | V2 requirement | Implementation | Evidence | Status |
|---:|---|---|---|:---:|
| 1 | One explicit Agent loop | `agent/graph.py` | `test_read_loop.py`; real LangGraph run | PASS |
| 2 | No Planner/Router/adaptive authority | `agent/graph.py`; static audit | `test_static_v2_architecture.py` | PASS |
| 3 | Agent declares `GoalDescriptor` | `agent/goals.py`; `agent/decision_ingress.py` | `test_goals_and_contracts.py` | PASS |
| 4 | Runtime compiles `CompletionContract` | `agent/contracts.py` | `test_goals_and_contracts.py` | PASS |
| 5 | Autonomous READ path | `agent/runtime.py`; `tools/runtime.py` | read-loop integration tests | PASS |
| 6 | Every WRITE requires human approval | `safety/write_guard.py`; `agent/graph.py` | WRITE lifecycle, mutation count before approval | PASS |
| 7 | No AUTO WRITE | `safety/write_guard.py`; no autonomy module | static audit | PASS |
| 8 | Write Guard outcomes are only INVALID/DENIED/APPROVAL_REQUIRED | `safety/write_guard.py` | write-admission tests | PASS |
| 9 | WRITE lifecycle is `WriteTransaction` | `safety/write_transaction.py` | write lifecycle tests | PASS |
| 10 | Approval binds frozen proposal and fingerprint | `safety/approval.py` | forged approval/fingerprint tests | PASS |
| 11 | Approval authorizes one protected attempt | `safety/locks.py`; `memory/sqlite.py` | claim/attempt tests | PASS |
| 12 | `ExecutionClaim` prevents duplicates | `safety/locks.py`; SQLite CAS | concurrent resume tests | PASS |
| 13 | Preconditions revalidated after approval | `safety/precondition.py` | TOCTOU test | PASS |
| 14 | `ActionVerifier` remains distinct | `verification/action.py` | direct-effect verification tests | PASS |
| 15 | `OperationalGoalVerifier` remains distinct | `verification/operational_goal.py` | goal verification tests | PASS |
| 16 | `ResponseCompletionGate` remains distinct | `agent/gate.py` | gate tests | PASS |
| 17 | Human rejection has controlled goal semantics | `agent/outcomes.py`; write runtime | reject/no-mutation tests | PASS |
| 18 | Policy denial has controlled goal semantics | `safety/policy.py` | denial/no-interrupt tests | PASS |
| 19 | Read batches preserve partial success | `tools/runtime.py` | partial-batch tests | PASS |
| 20 | Tool idempotency metadata is explicit | `tools/metadata.py`; `tools/catalog.py` | catalog and lifecycle tests | PASS |
| 21 | Evidence models freshness | `agent/evidence.py`; `agent/provenance.py` | freshness/provenance tests | PASS |
| 22 | Mutation invalidates affected mutable evidence | `agent/evidence.py`; write lifecycle | invalidation tests | PASS |
| 23 | Context separates structured state and semantic condensation | `agent/context.py` | bounded projection tests | PASS |
| 24 | LLM summaries are not safety authority | `agent/context.py`; Runtime-owned state | capability-isolation tests | PASS |
| 25 | Event/checkpoint persistence is crash-consistent | `memory/sqlite.py`; `agent/events.py` | crash-consistency tests | PASS |
| 26 | Unknown mutation requires reconciliation | `tools/write_runtime.py`; `agent/runtime.py` | unknown-outcome/replay tests | PASS |
| 27 | No V1 runtime imports | V2-local package tree | AST/static audit | PASS |
| 28 | Autonomy-specific V1 code is not migrated | V2 safety modules | static audit and tree review | PASS |
| 29 | Visible graph equals real Agent loop | `agent/graph.py` | `test_real_langgraph_runtime.py`: 4 passed | PASS |
| 30 | No unapproved or duplicate WRITE execution | safety/runtime boundaries | full adversarial suite | PASS |
| 31 | Every goal has a `GoalOutcome` | `agent/outcomes.py`; `agent/state.py` | multi-goal tests | PASS |
| 32 | Goal revision recompiles contract | `agent/runtime.py`; `agent/contracts.py` | goal revision tests | PASS |
| 33 | Incompatible goal change invalidates WRITE | `safety/write_transaction.py` | goal-drift tests | PASS |
| 34 | One approval cannot authorize a second attempt | `safety/approval.py`; `safety/locks.py` | replay/attempt CAS tests | PASS |
| 35 | Verifier reads are predeclared and deterministic | `tools/metadata.py`; `tools/registry.py` | catalog hash/verifier-read tests | PASS |
| 36 | `SystemContext` is explicit and Runtime-controlled | `agent/runtime.py` | host/runtime integration tests | PASS |
| 37 | `invoke()` and `resume()` are stable APIs | `agent/runtime.py`; `host.py` | host and graph tests | PASS |
| 38 | Approval binds trusted operator identity | `safety/approval.py`; `config.py` | approval/operator tests | PASS |
| 39 | Reject/deny are goal-level reason codes | `agent/outcomes.py`; gate | reject/deny tests | PASS |
| 40 | Runtime terminals use `ControlledTerminalOutcome` | `agent/outcomes.py`; `agent/runtime.py` | terminal routing tests | PASS |
| 41 | Recoverable missing evidence remains PENDING | `agent/evidence.py`; `agent/gate.py` | gate/evidence tests | PASS |
| 42 | A second WRITE requires new transaction and approval | `agent/runtime.py`; safety lifecycle | reconciliation retry tests | PASS |
| 43 | Only resolution returns to Agent; Runtime terminal goes to END | `agent/graph.py` | graph-shape and terminal tests | PASS |

## External evidence boundary

The following are implementation/local evidence, not external closure:

| Area | Current status | Evidence |
|---|---|---|
| Real Provider request | `UNVERIFIED_EXTERNAL` | No non-empty provider secret is configured; no request was sent |
| Real AutoDrive endpoint | `UNVERIFIED_EXTERNAL` | No endpoint/gateway is configured or listening |
| Sandbox platform | PASS | Local fake JSON-RPC adapter and approval/verification suite |
| Docker hosted build/runtime | `PASS` | Hosted run #11 built and ran the non-root image, health/readiness, SQLite, and same-volume smoke |
| Docker local build/run | `BLOCKED_EXTERNAL` | Local Docker Hub `python:3.12-slim` pull timed out |
| Hosted CI execution | `PASS` | Hosted run #11 passed Python 3.11/3.12, real-LangGraph, compile, wheel/import, container, and static jobs |
