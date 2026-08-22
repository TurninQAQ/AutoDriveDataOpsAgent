# AutoDriveDataOpsAgent V2.0 — Frozen Architecture Contract (FINAL)

> **Project root**
>
> `/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agentv2`
>
> **Migration source**
>
> `/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agent`
>
> **Architecture status**
>
> **FROZEN — READY FOR IMPLEMENTATION**
>
> This revision removes autonomous WRITE execution entirely and makes the architecture internally consistent around one final product boundary:
>
> **Autonomous READ, human-approved WRITE.**
>
> **Core architecture**
>
> **Small Agent Core + Explicit Tool Loop + Deterministic Human-Approval Runtime**
>
> **Highest-order invariant**
>
> **No subsystem may quietly become a second semantic decision-maker.**

---

# 1. Final Product Position

AutoDriveDataOpsAgent V2.0 is a single-loop DataOps Agent.

The Agent can autonomously inspect the environment through READ tools.

The Agent can propose WRITE tools, but it never receives permission to execute them directly.

Every WRITE that is structurally valid and policy-permitted must receive explicit human approval before execution.

The final product boundary is:

```text
READ
→ autonomous under deterministic runtime guards

WRITE
→ frozen proposal
→ deterministic admission
→ explicit human approval
→ protected one-attempt execution
→ verification
```

There is no AUTO WRITE path.

There is no future-AUTO placeholder in the main architecture.

There is no `BoundedAutonomyPolicy`.

There is no `max_auto_mutations`.

There is no autonomous write authorization reservation.

---

# 2. The Four Final Architecture Principles

## 2.1 Agent is the only semantic decision-maker

The Agent decides:

```text
what the user wants
what semantic action to attempt next
what READ tool to call
what WRITE tool to propose
whether more information is needed
how to interpret observations
how to compose the final answer
```

No Planner, Router, Guard, Compiler, Policy, Tool, or Runtime component decides the next semantic action.

---

## 2.2 READ may execute autonomously

READ tools may execute without human approval when they pass deterministic runtime checks.

Examples:

```text
get_task_detail
get_dataset_detail
get_queue_state
get_gpu_pool
diagnose_task
search_knowledge
```

---

## 2.3 Every WRITE requires explicit human approval

Every executable WRITE is first converted into a frozen `WriteTransaction`.

Human approval authorizes exactly that frozen transaction.

No approval means no mutation.

Examples:

```text
resume_task
submit_task
stop_task
delete_task
set_task_priority
```

---

## 2.4 One approval authorizes one protected execution attempt

Approval does not grant general authority.

Approval authorizes exactly one frozen `WriteTransaction` to enter the protected execution path.

Execution still requires:

```text
precondition revalidation
execution claim
mutation
ActionVerifier
OperationalGoalVerifier
```

Unknown mutation outcome blocks replay until reconciliation.

---

# 3. Final Responsibility Model

## 3.1 Agent

Owns:

```text
semantic reasoning
GoalDescriptor declaration
next semantic action
READ selection
WRITE proposal
interpretation of observations
final response composition
```

Does not own:

```text
completion requirements
evidence truth
write admission
approval
execution claim
mutation execution
verification truth
terminal runtime state
```

---

## 3.2 CompletionContractCompiler

Owns:

```text
deterministic completion requirements
```

Does not:

```text
parse free-form user requests
choose tools
choose workflow
perform free-form semantic reasoning
```

---

## 3.3 Runtime

Owns:

```text
tool validation
controlled projections
read execution
write admission
write transaction integrity
human approval enforcement
precondition revalidation
execution claim
mutation execution
bounded side-effect-free runtime retries
budgets
terminal runtime state
```

`bounded side-effect-free runtime retries` applies only to retry-safe READ transport failures or other side-effect-free Runtime operations.

WRITE mutation execution is never retried within the same `WriteTransaction`.

---

## 3.4 Evidence Tracker

Owns:

```text
evidence truth
provenance
freshness
validity
invalidation
```

---

## 3.5 ActionVerifier

Answers:

```text
Did the direct mutation actually happen?
```

---

## 3.6 OperationalGoalVerifier

Answers:

```text
Did the external system reach the user's requested operational state?
```

---

## 3.7 ResponseCompletionGate

Answers:

```text
May the interaction terminate normally with this FinalCandidate?
```

---

## 3.8 ContextBuilder

Owns:

```text
bounded model context
deterministic projection of runtime-critical state
semantic condensation of large untrusted observations
```

---

## 3.9 Event Store

Owns:

```text
immutable audit history
```

`AgentState` is the current execution/checkpoint projection.

---

# 4. No Separate Planning Authority

V2 does not contain:

```text
Planner Node
Planning Agent
Planning Tool
prepare_task_plan
decide_next
route_intent
strategy engine
mandatory planning stage
intent-specific workflow
```

The precise rule is:

> **There is no separate planning authority or mandatory planning stage.**

The Agent itself may reason about multiple future steps.

Planning as cognition is allowed.

A second component that decides the next semantic action is not.

---

# 5. GoalDescriptor

The Agent may declare the semantic goals it believes the user is asking to accomplish.

Every semantic goal has a stable `goal_id`, and every accepted GoalDescriptor revision has a monotonic `descriptor_version`.

Example:

```python
GoalDescriptor(
    descriptor_version=3,
    goals=[
        DiagnoseTask(goal_id="g1", target="A"),
        ExplainKnowledge(goal_id="g2", topic="task_exclusive"),
    ],
)
```

`goal_id` is stable across compatible conversational refinement so that completion, denial,
approval, and verification can be attributed to the correct user goal.

The Agent may declare:

```text
what the user wants
which target/topic the goal refers to
```

The Agent may not declare:

```text
what evidence is sufficient
what verification is sufficient
what exact tool sequence must occur
what counts as successful completion
```

---

# 6. CompletionContractCompiler

The semantic chain is:

```text
User
  ↓
Agent understands request
  ↓
GoalDescriptor
  ↓
CompletionContractCompiler
  ↓
fixed deterministic mapping
  ↓
CompletionContract
```

The compiler consumes:

```text
GoalDescriptor
+
deterministic platform rules
+
runtime safety rules
```

It does not consume raw user language as a free-form semantic input.

Example:

```text
DiagnoseTask(A)
→ target binding required
→ LIVE_TASK required
→ DIAGNOSTIC_CONTEXT required
```

Example:

```text
ExplainKnowledge(task_exclusive)
→ KNOWLEDGE evidence required
```

The Agent says:

```text
what the user wants
```

The Runtime says:

```text
what must be true for that goal to count as complete
```

---

# 7. CompletionContract Is Not a Workflow

Allowed:

```text
requires LIVE_TASK
requires DIAGNOSTIC_CONTEXT
requires KNOWLEDGE
requires target binding
requires post-write verification
```

Forbidden:

```text
call tool A
then tool B
then tool C
then answer
```

The Compiler defines completion requirements.

The Agent decides how to satisfy them.

---

# 7A. Per-Goal GoalOutcome

Multi-goal requests are tracked explicitly.

The Runtime maintains one `GoalOutcome` per `goal_id`.

Conceptual model:

```python
@dataclass(frozen=True)
class GoalOutcome:
    goal_id: str

    status: Literal[
        "PENDING",
        "SATISFIED",
        "DENIED",
        "REJECTED",
        "FAILED",
        "INCONCLUSIVE",
        "BLOCKED",
    ]

    reason_code: str | None
    evidence_refs: tuple[str, ...]
    write_transaction_id: str | None
```

The key distinction is:

```text
GoalDescriptor
= what the user wants

CompletionContract
= what must be true

GoalOutcome
= what happened for this specific goal
```

A multi-goal request may therefore end with:

```text
g1 DiagnoseTask(A)          → SATISFIED
g2 ExplainKnowledge(rule)   → SATISFIED
g3 DeleteTask(A)            → REJECTED
```

The system must not collapse this into a single misleading global boolean.

---

# 7B. Aggregate Completion Semantics

`ResponseCompletionGate` evaluates both:

```text
per-goal CompletionContract state
+
per-goal GoalOutcome state
```

A goal is terminally resolved when its outcome is one of:

```text
SATISFIED
DENIED
REJECTED
FAILED
INCONCLUSIVE
BLOCKED
```

`PENDING` is non-terminal.

However, `FAILED`, `INCONCLUSIVE`, and `BLOCKED` may be assigned only when the Runtime has
determined that the specific goal has **no allowed continuation in the current interaction**.

Recoverable conditions remain `PENDING`.

Examples:

```text
missing LIVE_TASK evidence, but get_task_detail is still allowed
→ PENDING

temporary read timeout with remaining retry budget
→ PENDING

policy permanently forbids the write
→ DENIED

human rejects approval
→ REJECTED

required backend is unavailable and no allowed continuation remains for that goal
→ BLOCKED / FAILED / INCONCLUSIVE
```

For a multi-goal request:

```text
all goals terminally resolved
+
FinalCandidate accurately reports each material outcome
→ normal conversation completion is allowed
```

This supports honest partial completion.

Example:

```text
diagnosis succeeded
knowledge explanation succeeded
delete request rejected by user
```

The final answer must report all three facts.

A rejected or denied write does not automatically terminate unrelated read goals.

## Goal-level terminal vs Runtime-level terminal

These are different concepts.

### Goal-level terminal

Examples:

```text
SATISFIED
DENIED
REJECTED
FAILED
INCONCLUSIVE
BLOCKED
```

A goal-level terminal outcome resolves only that `goal_id`.

It may still return control to the Agent if other goals remain `PENDING`.

### Runtime-level terminal

Examples:

```text
BUDGET_EXHAUSTED
PROVIDER_UNAVAILABLE
REQUIRES_RECONCILIATION
UNRECOVERABLE_RUNTIME_ERROR
CHECKPOINT_CORRUPTION
```

A Runtime-level terminal condition ends the entire interaction through `ControlledTerminalOutcome`.

`USER_REJECTED_WRITE` and `POLICY_DENIED_WRITE` are **not** Runtime-level terminal codes.

They are `GoalOutcome.reason_code` values.

---

# 7C. GoalDescriptor Revision and Contract Recompilation

If the Agent revises the `GoalDescriptor` because the conversation changes or clarifies user intent:

```text
GoalDescriptor vN
        ↓
Agent proposes revised descriptor
        ↓
Runtime validates
        ↓
GoalDescriptor vN+1
        ↓
CompletionContractCompiler
        ↓
new CompletionContract
```

Recompilation is mandatory.

The Runtime must never keep using a CompletionContract compiled from an older incompatible goal description.

Each compiled contract records:

```text
descriptor_version
contract_version
contract_fingerprint
goal_ids
```

---

# 7D. Goal Change vs Existing WriteTransaction

Every `WriteTransaction` is bound to:

```text
bound_goal_ids
goal_descriptor_version
completion_contract_fingerprint
```

When the GoalDescriptor changes, the Runtime performs a deterministic compatibility check.

The compatibility check does not perform semantic planning.

It compares structured goal/contract identity.

A transaction becomes:

```text
INVALIDATED_GOAL_CHANGED
```

if any bound write goal changes materially, including:

```text
goal removed
target changed
write operation changed
frozen arguments no longer correspond to the goal
required completion semantics changed incompatibly
```

Any existing approval remains in the audit log but no longer authorizes execution.

A new executable proposal requires:

```text
new WriteTransaction
+
new human approval
```

If the GoalDescriptor update affects only an unrelated read goal and the bound write goal's
structured contract fingerprint remains identical, the existing transaction may remain valid.

---

# 8. Canonical V2 Graph

```text
                              ┌──────────────────────────┐
                              │                          │
                              ▼                          │
                            AGENT                        │
                       Reason + Act                      │
                              │                          │
         ┌────────────────────┼──────────────────────┐   │
         │                    │                      │   │
         ▼                    ▼                      ▼   │
   SINGLE READ           READ BATCH               WRITE │
         │                    │                      │   │
         └──────────┬─────────┘                      │   │
                    │                                ▼   │
                    │                          WRITE GUARD │
                    │                         /    |     \ │
                    │                    INVALID DENIED APPROVAL_REQUIRED
                    │                       │      │        │
                    │                       │      │        ▼
                    │                       │      │  WRITE TRANSACTION
                    │                       │      │        │
                    │                       │      │   APPROVAL NODE
                    │                       │      │    interrupt()
                    │                       │      │      /   \
                    │                       │      │ reject approve
                    │                       │      │   │      │
                    │                       │      │   │   REVALIDATE
                    │                       │      │   │      │
                    │                       │      │   │  EXECUTION CLAIM
                    │                       │      │   │      │
                    │                       │      │   │    EXECUTE
                    │                       │      │   │      │
                    │                       │      │   │ ACTION VERIFY
                    │                       │      │   │      │
                    │                       │      │   │ GOAL VERIFY
                    │                       │      │   │      │
                    ▼                       ▼      ▼   ▼
                         OBSERVATION / GOAL RESOLUTION
                                      │
                                      └──────────────────→ AGENT
```

Normal final response:

```text
AGENT
  ↓
FinalCandidate
  ↓
ResponseCompletionGate
   /               \
PASS             NOT_SATISFIED
 │                   │
END                AGENT
```

Runtime-controlled abnormal termination is a separate path and never returns to the Agent:

```text
RUNTIME TERMINAL CONDITION
        ↓
BUDGET_EXHAUSTED
PROVIDER_UNAVAILABLE
REQUIRES_RECONCILIATION
UNRECOVERABLE_RUNTIME_ERROR
CHECKPOINT_CORRUPTION
        ↓
ControlledTerminalOutcome
        ↓
END
```

---

# 9. AgentDecision

The Agent emits:

```text
AgentDecision
├── SingleToolCall
├── ReadToolBatch
└── FinalCandidate
```

It may also update its `GoalDescriptor` when the conversation clarifies the user's request.

It cannot directly modify the `CompletionContract`.

---

# 10. ToolSpec

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: Literal["READ", "WRITE"]
    risk: RiskLevel
    schema: dict

    parallel_safe: bool = False

    requires_precondition: bool = False

    verification: Literal[
        "NONE",
        "ACTION",
        "ACTION_AND_GOAL",
    ] = "NONE"

    idempotency: Literal[
        "SAFE_RETRY",
        "RECONCILE_BEFORE_RETRY",
        "NO_RETRY",
    ] = "NO_RETRY"
```

Example:

```text
get_task_detail
READ
parallel_safe=True
verification=NONE
idempotency=SAFE_RETRY

get_gpu_pool
READ
parallel_safe=True
verification=NONE
idempotency=SAFE_RETRY

search_knowledge
READ
parallel_safe=True
verification=NONE
idempotency=SAFE_RETRY

diagnose_task
READ
parallel_safe=False
verification=NONE
idempotency=SAFE_RETRY

resume_task
WRITE
parallel_safe=False
requires_precondition=True
verification=ACTION_AND_GOAL
idempotency=RECONCILE_BEFORE_RETRY

delete_task
WRITE
parallel_safe=False
requires_precondition=True
verification=ACTION
idempotency=NO_RETRY
```

`ToolSpec.idempotency` informs reconciliation and recovery policy.

It never authorizes reuse of a consumed WRITE approval or execution attempt.

All WRITE still require human approval.

---

# 11. Parallel READ Is Structural

A `ReadToolBatch` is valid only if:

```text
all calls are READ
all ToolSpec.parallel_safe == True
batch size <= configured maximum
all arguments are concrete before batch execution
no call references another same-batch output
```

The Runtime does not perform semantic reasoning about whether two tools “feel independent.”

The Agent decides which reads it wants together.

The Runtime validates structural legality only.

---

# 12. ReadToolBatch Is Concurrent, Not Transactional

A read batch may partially fail.

Example:

```text
get_task_detail → SUCCESS
get_queue_state → SUCCESS
get_gpu_pool    → TIMEOUT
```

The Runtime returns:

```python
ReadBatchObservation(
    results=[
        ReadSuccess(...),
        ReadSuccess(...),
        ReadFailure(
            error_code="READ_TIMEOUT",
            retryable=True,
        ),
    ]
)
```

The entire batch is not rolled back or converted into one generic failure.

The Agent receives all usable results.

The Agent decides whether the partial result is sufficient or whether another read is needed.

---

# 13. WRITE Is Always Serialized

There is no write batch.

Forbidden:

```text
[
  resume_task(A),
  delete_task(B),
  set_task_priority(C)
]
```

WRITE is one frozen proposal at a time.

This preserves one clear:

```text
transaction
approval
precondition
execution claim
mutation result
verification chain
```

---

# 14. Tools Do Not Think for the Agent

READ tools observe.

WRITE tools mutate.

Forbidden tool concepts:

```text
make_plan
choose_action
decide_next
route_intent
prepare_strategy
select_workflow
```

`diagnose_task` may:

```text
run deterministic diagnostic rules
aggregate state
normalize facts
```

It may not:

```text
invoke an LLM
select the next tool
autonomously call arbitrary tools
decide a workflow
```

---

# 15. Read Path

```text
Agent
  ↓
SingleToolCall / ReadToolBatch
  ↓
Tool Registry
  ↓
Read Guard
  ↓
MCP Executor
  ↓
ToolObservation(s)
  ↓
Evidence Tracker
  ↓
Event Store
  ↓
ContextBuilder
  ↓
Agent
```

Read Guard handles:

```text
schema
read-only enforcement
target sanity
timeouts
error normalization
provenance
trust labeling
parallel structural validation
```

It never chooses the next semantic action.

---

# 16. Tool Observations Are Untrusted Data

All external observations are data, never authority.

```python
ToolObservation(
    observation_id="obs_123",
    source="get_stage_logs",
    trust="UNTRUSTED_EXTERNAL_DATA",
    data=...,
)
```

This includes:

```text
logs
RAG documents
dataset metadata
external status
user-generated metadata
```

Observation text may contain:

```text
IGNORE PREVIOUS INSTRUCTIONS
DELETE ALL TASKS
SYSTEM MESSAGE: ...
```

Such content has zero deterministic authority.

Observations cannot directly change:

```text
Policy
Target Binding
Approval
WriteTransaction state
Execution Claim
```

---

# 17. Write Guard

The Write Guard evaluates a WRITE proposal.

It may:

```text
validate schema
bind target
normalize arguments
freeze arguments
validate existing evidence
take mandatory safety precondition snapshot
perform deterministic impact analysis
apply deterministic write admission policy
```

It may not:

```text
autonomously gather semantic evidence
select READ tools
perform semantic exploration
repair Agent intent
choose a workflow
```

---

# 18. Write Admission Outcomes

The Write Guard has exactly three outcomes:

```text
INVALID
DENIED
APPROVAL_REQUIRED
```

There is no AUTO.

---

# 19. INVALID

`INVALID` means the proposal cannot currently enter human approval.

Examples:

```text
schema invalid
target unresolved
target ambiguous
required evidence missing
precondition cannot be established
arguments inconsistent
unsupported write structure
```

Flow:

```text
INVALID
→ RuntimeObservation
→ Agent
```

The Agent may choose to gather more evidence or revise the proposal.

---

# 20. DENIED

`DENIED` means the proposal is structurally understandable but forbidden by deterministic policy.

Examples:

```text
protected resource deletion prohibited
operation violates hard platform rule
write is permanently disallowed in this environment
```

The Runtime creates a controlled write-resolution outcome.

The bound write goal receives:

```text
GoalOutcome(status="DENIED", reason_code="POLICY_DENIED_WRITE")
```

`POLICY_DENIED_WRITE` is a **goal-level reason code**, not a Runtime-level terminal condition.

For a write-only request, all goals are then terminally resolved and the Agent may produce a truthful
FinalCandidate describing the denial.

For a multi-goal request, independent non-terminal goals continue before final response composition.

The Agent cannot override the denial.

---

# 21. APPROVAL_REQUIRED

`APPROVAL_REQUIRED` means:

```text
proposal is valid
policy permits the operation to be presented for human approval
```

The Runtime creates a `WriteTransaction`.

No mutation occurs yet.

---

# 22. WriteTransaction

All WRITE lifecycle state is consolidated into one object.

Conceptual model:

```python
@dataclass
class WriteTransaction:
    transaction_id: str

    proposal: FrozenToolCall
    fingerprint: str

    bound_goal_ids: tuple[str, ...]
    goal_descriptor_version: int
    completion_contract_fingerprint: str

    status: WriteTransactionStatus

    approval: ApprovalRecord | None
    precondition: PreconditionSnapshot | None

    execution_claim: ExecutionClaim | None
    execution_attempt_id: str | None
    mutation_result: MutationResult | None

    action_verification: VerificationResult | None
    operational_goal_verification: VerificationResult | None

    affected_entities: tuple[str, ...]
    reconciliation: ReconciliationState | None
```

This replaces scattered top-level state such as:

```text
pending_write
authorization
execution_claim
mutation result
verification result
```

---

# 23. WriteTransaction Lifecycle

```text
PROPOSED
   ↓
VALIDATED
   ↓
PENDING_APPROVAL
   │
   ├────────────→ REJECTED
   │
   ▼
APPROVED
   ↓
REVALIDATING
   │
   ├────────────→ INVALIDATED
   │
   ▼
EXECUTING
   │
   ├────────────→ FAILED
   │
   ├────────────→ RECONCILIATION_REQUIRED
   │
   ▼
EXECUTED
   ↓
VERIFYING
   │
   ├────────────→ VERIFICATION_FAILED
   │
   ▼
VERIFIED
```

Possible terminal transaction states:

```text
REJECTED
INVALIDATED
INVALIDATED_GOAL_CHANGED
FAILED
RECONCILIATION_REQUIRED
VERIFICATION_FAILED
VERIFIED
```

---

# 24. Human Approval Is the Authorization

There is no parallel `authorization` object.

Human approval is the write authorization artifact.

Definition:

> **ApprovalGranted authorizes exactly one protected execution attempt for exactly one frozen WriteTransaction.**

Approval is bound to:

```text
transaction_id
frozen proposal
fingerprint
request/thread identity
approval actor
approval timestamp
```

Delete:

```text
authorization_id
AuthorizationReserved
autonomous authorization state
atomic AUTO reservation
```

Keep:

```text
approval_request_id
ApprovalRecord
fingerprint
ExecutionClaim
```

---

# 25. Approval Node

Flow:

```text
WriteTransaction(PENDING_APPROVAL)
        ↓
Approval Node
        ↓
interrupt()
        ↓
Human decision
```

The node containing `interrupt()` must be replay-safe.

Non-idempotent pre-interrupt side effects are forbidden.

Durable transaction preparation happens before the interrupt node using idempotent semantics.

---

# 26. Human Reject

Human rejection changes the transaction to:

```text
REJECTED
```

and produces:

```text
USER_REJECTED_WRITE
```

The bound write goal receives:

```text
GoalOutcome(status="REJECTED", reason_code="USER_REJECTED_WRITE")
```

`USER_REJECTED_WRITE` is a **goal-level reason code**, not a Runtime-level terminal condition.

For a write-only request, this makes all goals terminally resolved and the Agent may produce a truthful
FinalCandidate such as:

```text
The write operation was not executed because approval was rejected.
```

The Agent does not automatically reopen the same approval request.

For a mixed request, unrelated non-terminal goals continue.

---

# 27. Human Approve

Approval changes:

```text
PENDING_APPROVAL
→ APPROVED
```

The exact frozen proposal is now eligible for one protected execution attempt.

Approval does not bypass:

```text
precondition revalidation
ExecutionClaim
verification
reconciliation rules
```

---

# 28. Precondition Revalidation

After approval:

```text
APPROVED
→ REVALIDATING
```

The Runtime re-reads only the exact safety-critical state necessary to detect TOCTOU drift.

This is not semantic exploration.

If the frozen transaction is no longer valid:

```text
INVALIDATED
```

No mutation occurs.

A new materially different proposal requires a new `WriteTransaction` and new human approval.

---

# 29. ExecutionClaim

`ExecutionClaim` is retained even though AUTO is removed.

Reason:

```text
checkpoint replay
two workers resuming the same approval
duplicate resume
concurrent execution attempts
```

Flow:

```text
APPROVED
→ atomic/CAS ExecutionClaim
→ EXECUTING
```

Only one worker may successfully claim one transaction execution.

The moment a transaction successfully obtains its `ExecutionClaim`, its single approved execution
attempt is considered consumed.

Infrastructure may safely retry the **claim operation itself** only while no claim has been acquired.
It may not start a second mutation attempt for the same transaction.

---

# 30. One Approval = One Execution Attempt

Frozen invariant:

> **One ApprovalRecord authorizes exactly one WriteTransaction execution attempt.**

There is no same-transaction mutation retry.

There is no reuse of an approval for a second execution attempt.

After successful `ExecutionClaim`, any subsequent need to execute a WRITE again requires:

```text
new WriteTransaction
+
fresh frozen proposal
+
fresh fingerprint
+
fresh human approval
```

This rule applies even when the previous attempt is known to have failed before producing the
desired business result.

The reason is deliberate simplicity:

```text
approval
= consent for one concrete attempt

not
= standing permission to keep trying
```

---

# 31. Mutation Execution

After successful claim:

```text
WriteTransaction.status = EXECUTING
execution_attempt_id = stable unique id
```

The Runtime invokes the mutation tool exactly once for that transaction.

No WRITE mutation retry is allowed within the same transaction.

---

# 32. Mutation Outcome Taxonomy

Possible outcomes:

```text
FAILED_BEFORE_EFFECT
CONFIRMED_SUCCESS
CONFIRMED_FAILURE
OUTCOME_UNKNOWN
```

If:

```text
OUTCOME_UNKNOWN
```

then:

```text
WriteTransaction.status = RECONCILIATION_REQUIRED
```

and all replay is blocked until reconciliation.

If a confirmed failure later requires another attempt, the next attempt is still:

```text
new WriteTransaction
→ new ApprovalRecord
```

---

# 32A. Tool Idempotency Is Reconciliation Metadata

`ToolSpec.idempotency` remains useful, but its meaning is deliberately narrow.

```text
SAFE_RETRY
RECONCILE_BEFORE_RETRY
NO_RETRY
```

For READ tools, it may authorize bounded Runtime transport retries.

For WRITE tools, it describes what must be known before a **future new transaction** could be
considered safe.

It never permits:

```text
same WriteTransaction
same ApprovalRecord
second mutation execution
```

Examples:

```text
RECONCILE_BEFORE_RETRY
→ determine external state first
→ if another write is still needed
→ create new WriteTransaction
→ request new human approval

NO_RETRY
→ no replay path is offered without explicit product-level redesign
```

---

# 33. ActionVerifier

After confirmed mutation execution:

```text
EXECUTED
→ VERIFYING
```

`ActionVerifier` checks:

```text
did the requested direct mutation occur?
was the frozen target affected?
does post-state match the direct expected effect?
```

---

# 34. OperationalGoalVerifier

After direct action verification, if required:

```text
OperationalGoalVerifier
```

checks:

```text
did the external world reach the user's intended operational state?
```

Example:

```text
resume_task(A)
```

ActionVerifier:

```text
was a new execution created?
```

OperationalGoalVerifier:

```text
is A now represented in the expected post-resume state?
```

---

# 34A. Verifier Reads Are Deterministic Verification Reads

`ActionVerifier` and `OperationalGoalVerifier` may perform verification reads without returning to
the Agent first.

These reads do not violate the single semantic decision authority because they are not exploratory.

They must be:

```text
predeclared by ToolSpec / verifier contract
restricted to the frozen target or affected entities
read-only
bounded
deterministic
free of LLM reasoning
unable to choose arbitrary new tools
```

Example:

```text
resume_task(A)
→ verifier contract says read post-state of A
→ deterministic get_task_detail(A)
→ verify
```

Forbidden:

```text
Verifier:
"I am unsure; maybe inspect GPU, queue, logs, and knowledge."
```

That would be semantic exploration and must return control to the Agent instead.

Verification-read observations are provenance-tagged and may become fresh post-write evidence.

---

# 35. ResponseCompletionGate

After observations return to the Agent, the Agent may produce a `FinalCandidate`.

The Gate checks:

```text
CompletionContract
required evidence
target binding
write terminal state
ActionVerifier result
OperationalGoalVerifier result
pending approval
reconciliation status
```

It never chooses the next tool.

---

# 36. ControlledTerminalOutcome

Normal completion:

```text
FinalCandidate
→ ResponseCompletionGate PASS
→ END
```

Abnormal/bounded **interaction-level** termination:

```text
BUDGET_EXHAUSTED
PROVIDER_UNAVAILABLE
REQUIRES_RECONCILIATION
UNRECOVERABLE_RUNTIME_ERROR
CHECKPOINT_CORRUPTION
→ ControlledTerminalOutcome
→ END
```

Goal-level denial/rejection is not routed through `ControlledTerminalOutcome`.

Instead:

```text
USER_REJECTED_WRITE
POLICY_DENIED_WRITE
→ GoalOutcome.reason_code
→ update that goal only
→ continue if other goals remain PENDING
```

`ControlledTerminalOutcome` is deterministic and bounded.

It does not invent semantic conclusions.

---

# 37. ControlledTerminalOutcome Schema

```python
@dataclass(frozen=True)
class ControlledTerminalOutcome:
    code: TerminalCode
    safe_facts: dict
    message_template: str
    retry_allowed: bool
    human_action_required: bool
```

Examples:

```text
REQUIRES_RECONCILIATION
→ "The mutation result is uncertain and must be reconciled before any replay."

PROVIDER_UNAVAILABLE
→ "The configured provider is unavailable, so this interaction cannot safely continue."

BUDGET_EXHAUSTED
→ "The bounded execution budget was exhausted before all goals could be completed."
```

---

# 38. Canonical AgentState

```python
class AgentState(TypedDict):
    messages: list

    request_id: str
    thread_id: str

    step_count: int
    tool_call_count: int

    goal_descriptor: GoalDescriptor | None
    goal_descriptor_version: int
    completion_contract: CompletionContract | None
    goal_outcomes: dict[str, GoalOutcome]

    evidence: EvidenceState

    active_write_transaction: WriteTransaction | None

    budgets: RuntimeBudgets
    terminal_state: ControlledTerminalOutcome | None
    termination_reason: str | None
```

No top-level:

```text
authorization
execution_claim
pending_write
```

Those belong inside `WriteTransaction`.

`active_write_transaction` represents the currently active/latest WriteTransaction for the thread.

WRITE serialization means:

```text
at most one active WriteTransaction at a time
```

It does **not** mean a thread can only ever contain one WriteTransaction.

After one transaction reaches a terminal state, a later WRITE may create a new
`active_write_transaction`.

Historical terminal WriteTransactions are preserved in the `EventStore` and referenced through
their stable transaction identity from `GoalOutcome`, evidence, and audit events.

---

# 39. State Field Ownership

| State | Authoritative writer |
|---|---|
| assistant message | Agent Node / message adapter |
| proposed ToolCall | Agent Node |
| `goal_descriptor` | Agent Node, Runtime validated |
| `completion_contract` | CompletionContractCompiler |
| `goal_outcomes` | Runtime / Completion & Write Outcome handlers |
| ToolObservation | Tool Runtime |
| `evidence` | Evidence Tracker |
| `active_write_transaction` lifecycle | WriteTransaction Runtime |
| `approval` | Approval Runtime |
| `execution_claim` | WriteTransaction Runtime |
| mutation result | Mutation Runtime |
| action verification | ActionVerifier |
| operational goal verification | OperationalGoalVerifier |
| budgets | Runtime |
| terminal state | Runtime |

The Agent cannot forge Runtime-owned fields.

---

# 40. One Canonical State, Multiple Controlled Projections

```text
Canonical State
      │
      ├── Agent-visible projection
      ├── Safety/WriteTransaction projection
      ├── Checkpoint projection
      └── Audit projection
```

The LLM never receives a full raw state dump.

---

# 41. ContextBuilder Has Two Paths

## 41.1 Deterministic structured projection

Runtime/security-critical state is projected deterministically.

Examples:

```text
GoalDescriptor
CompletionContract status
Frozen WriteTransaction proposal
fingerprint reference
target binding
approval state
execution claim state
mutation uncertainty
verification result
reconciliation state
critical EvidenceRecord metadata
```

These remain structured.

---

## 41.2 Semantic condensation

Large untrusted data may be condensed.

Examples:

```text
logs
RAG content
large diagnostic JSON
repeated stack traces
large read payloads
```

The raw observation remains in the Event Store.

The condensed prompt context keeps stable references such as:

```text
observation_id
source
target
```

---

# 42. LLM Summary Is Never Authoritative Safety State

Frozen invariant:

> **Security-critical structured state must be projected deterministically; an LLM-generated summary must never be its authoritative representation.**

The Runtime must never replace:

```json
{
  "transaction_id": "...",
  "tool": "resume_task",
  "target": "A",
  "datasets": ["x"],
  "fingerprint": "..."
}
```

with only:

```text
"Preparing to resume A."
```

for authoritative state.

The summary may help the Agent reason.

It cannot replace the real structure.

---

# 43. EvidenceRecord

Evidence must model freshness explicitly.

Conceptual shape:

```python
@dataclass
class EvidenceRecord:
    kind: str
    target: str

    observation_id: str

    observed_at: datetime
    entity_version: str | None
    valid_until: datetime | None

    status: Literal[
        "VALID",
        "STALE",
        "INVALIDATED",
    ]

    invalidated_by: str | None
```

---

# 44. Evidence Freshness

Evidence validity means more than presence.

Runtime must distinguish:

```text
present and current
present but stale
invalidated
missing
```

Example:

```text
10:00
task A = FAILED
→ LIVE_TASK(A) valid

10:05
resume_task(A)
→ mutable state changed

old LIVE_TASK(A)
→ STALE / INVALIDATED
```

The old pre-mutation state cannot continue to satisfy a current live-state completion requirement.

---

# 45. Mutation-Driven Evidence Invalidation

Frozen invariant:

> **A successful or uncertain mutation invalidates affected pre-mutation mutable evidence.**

`WriteTransaction` records:

```text
affected_entities
```

After:

```text
CONFIRMED_SUCCESS
```

or:

```text
OUTCOME_UNKNOWN
```

the Evidence Tracker invalidates mutable evidence for affected entities.

Example:

```text
WriteTransaction.affected_entities = {"task_A"}
```

Then:

```text
LIVE_TASK(task_A)
DIAGNOSTIC_CONTEXT(task_A)
QUEUE_STATE(task_A)
```

are invalidated or marked stale according to evidence type.

Immutable knowledge evidence does not need mutation-driven invalidation.

---

# 46. Post-Write Evidence

`ActionVerifier` and `OperationalGoalVerifier` may create new verified post-write evidence.

Example:

```text
POST_WRITE_TASK_STATE(task_A)
```

This evidence is newer than the invalidated pre-write state.

ResponseCompletionGate must prefer current valid evidence.

---

# 47. Evidence Version Rules

Where possible, Evidence should carry:

```text
entity_version
generation
revision
etag
execution_id
or equivalent backend identity
```

If no backend version exists, use:

```text
observed_at
+
mutation sequence
+
request/transaction causation
```

to establish freshness conservatively.

---

# 48. Event Log

```text
Event Log
= immutable audit truth

AgentState
= execution/checkpoint projection
```

Safety-critical transitions and their corresponding audit events must be crash-consistent.

---

# 49. Crash-Consistent Persistence

Frozen invariant:

> **Safety-critical state transitions and their corresponding audit events must have idempotent, crash-consistent persistence semantics.**

Applies to:

```text
WriteTransactionPrepared
ApprovalRequested
ApprovalGranted
ApprovalRejected
ExecutionClaimed
MutationStarted
MutationResultRecorded
ActionVerificationRecorded
OperationalGoalVerificationRecorded
EvidenceInvalidated
ReconciliationRequired
```

---

# 50. Lightweight Persistence Protocol

A full event-sourcing system is not required.

A practical implementation may use:

```text
stable event_id
stable transaction_id
sequence_no
idempotent append
checkpoint.last_applied_event_id
```

A lightweight transactional-outbox-style implementation is acceptable if useful.

---

# 51. Recovery

After restart:

```text
checkpoint.last_applied_event_id
```

is compared with durable events.

Recovery detects:

```text
event persisted but checkpoint lagged
checkpoint advanced but event missing
duplicate replay
out-of-order transition
uncertain mutation
```

Safety-critical inconsistency fails closed.

---

# 52. Event Schema

Every event contains:

```text
event_id
sequence_no
request_id
thread_id
causation_id
timestamp
event_type
```

Relevant events also record:

```text
model_version
prompt_version
tool_catalog_hash
policy_version
transaction_id
```

---

# 53. Error / Observation Taxonomy

```python
RuntimeObservation(
    status=...,
    retryable=...,
    side_effect_state=...,
    error_code=...,
    observation_id=...,
)
```

Suggested side-effect state:

```text
NONE
CONFIRMED
FAILED_BEFORE_EFFECT
UNKNOWN
```

---

# 54. READ Retry

READ failures may use bounded deterministic retry.

Example:

```text
READ_TIMEOUT
retryable=True
side_effect_state=NONE
```

Recommended:

```text
0–2 retries
```

Retry is a transport/runtime policy.

It is not a second semantic decision-maker.

---

# 55. WRITE Retry

There is **no same-transaction WRITE mutation retry**.

Frozen invariant:

> **One ApprovalRecord = one WriteTransaction = one mutation execution attempt.**

If execution fails before effect:

```text
WriteTransaction
→ FAILED
```

If another WRITE attempt is still needed:

```text
new WriteTransaction
→ new frozen proposal
→ new fingerprint
→ new human approval
```

If mutation outcome is unknown:

```text
WriteTransaction
→ RECONCILIATION_REQUIRED
→ replay blocked
```

`ToolSpec.idempotency` does not permit reuse of the same transaction or approval.

For WRITE tools, idempotency metadata only constrains whether a **future new transaction** may be
considered safe after reconciliation.

Examples:

```text
RECONCILE_BEFORE_RETRY
→ reconcile external state
→ if another write is still necessary
→ create new WriteTransaction
→ request new approval

NO_RETRY
→ no new execution path unless product policy explicitly defines one

SAFE_RETRY
→ may inform future new-transaction policy
→ never authorizes a second attempt under the existing approval
```

There is no “safe execution policy” that may silently reuse a consumed `ApprovalRecord`.

---

# 56. Human Approval and Replay

Approval remains bound to the exact frozen transaction.

If reconciliation determines that a new materially different mutation is required:

```text
new WriteTransaction
→ new approval
```

Human approval is never generalized to “whatever retry the Agent thinks is appropriate.”

---

# 56A. SystemContext

V2 has an explicit immutable `SystemContext` supplied by the host application.

It is not conversational Agent state.

Conceptual model:

```python
@dataclass(frozen=True)
class SystemContext:
    runtime_version: str
    environment: str

    operator_id: str
    trust_domain: str

    tool_catalog_hash: str
    policy_version: str

    event_store: EventStore
    checkpointer: Checkpointer

    provider_config: ProviderConfig
```

`SystemContext` contains host/runtime facts and dependencies that should not be inferred from the
conversation.

The Agent receives only a safe projection of SystemContext.

Secrets, credentials, storage handles, approval internals, and security configuration remain
Runtime-private.

---

# 56B. Single Trusted Operator Assumption

V2 intentionally assumes:

> **one authenticated trusted human operator within one trust domain.**

The architecture does not attempt to implement:

```text
multi-party approval
role hierarchies
RBAC policy administration
four-eyes approval
cross-tenant approval delegation
Byzantine/malicious approver defense
```

Authentication of the operator is expected to be provided by the host application.

`ApprovalRecord` binds the approval to `operator_id` from `SystemContext`.

A future multi-operator authorization model would be a separate product/security version, not a
hidden extension of this V2 contract.

---

# 56C. Top-Level Runtime API

The V2 runtime exposes two primary application entry points:

```python
async def invoke(
    user_input: str,
    *,
    thread_id: str,
    system_context: SystemContext,
) -> AgentRunResult:
    ...
```

and:

```python
async def resume(
    *,
    thread_id: str,
    resume_input: ResumeInput,
    system_context: SystemContext,
) -> AgentRunResult:
    ...
```

---

# 56D. `invoke()` Semantics

`invoke()` starts or continues a normal conversational Agent turn.

Responsibilities:

```text
load/create checkpoint
append UserEvent
build/update GoalDescriptor
compile/recompile CompletionContract
run Agent graph
return final / interrupted / controlled-terminal result
```

`invoke()` does not bypass an outstanding approval interrupt.

If a thread is suspended for approval, the host should use `resume()`.

---

# 56E. `resume()` Semantics

`resume()` resumes a previously checkpointed suspended graph.

Primary uses:

```text
human approval
human rejection
other explicitly modeled interrupt payloads
```

`resume()` must:

```text
load the exact thread checkpoint
validate the resume input against the pending interrupt
bind operator identity from SystemContext
preserve WriteTransaction identity
resume replay-safe graph execution
```

It must not accept an arbitrary new semantic WRITE proposal as an approval payload.

If user intent has changed materially, that is handled as a new conversational input / new
GoalDescriptor revision, not by mutating the pending WriteTransaction during resume.

---

# 56F. AgentRunResult

Conceptual top-level result:

```python
@dataclass(frozen=True)
class AgentRunResult:
    thread_id: str
    request_id: str

    status: Literal[
        "COMPLETED",
        "INTERRUPTED",
        "CONTROLLED_TERMINAL",
        "ERROR",
    ]

    response: str | None
    goal_outcomes: tuple[GoalOutcome, ...]

    pending_interrupt: object | None
    terminal_outcome: ControlledTerminalOutcome | None
```

This gives the host application one stable boundary regardless of whether the graph:

```text
completed normally
paused for approval
terminated safely
failed unexpectedly
```

---

# 57. Source Migration Rule

V2 must not import old production runtime modules.

Forbidden:

```python
from deploy_ci_cloud_agent... import ...
```

Instead copy required source into:

```text
/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agentv2
```

and adapt locally.

Migrate/refactor:

```text
AgentPolicyEngine
→ WriteAdmissionPolicy / equivalent deterministic policy

ApprovalStore

WriteActionCoordinator
→ WriteTransactionCoordinator / WriteTransaction Runtime

ActionVerifier

GoalVerifier
→ OperationalGoalVerifier
```

Do **not** migrate:

```text
BoundedAutonomyPolicy
autonomy.py
AUTO eligibility logic
AUTO budget logic
atomic AUTO authorization reservation
```

---

# 58. Final Safety Directory

Recommended:

```text
safety/
├── write_guard.py
├── policy.py
├── approval.py
├── write_transaction.py
├── precondition.py
├── fingerprint.py
└── locks.py
```

Delete:

```text
autonomy.py
```

`write_transaction.py` owns the human-approved write lifecycle.

---

# 59. Final Project Structure

```text
deploy_ci_cloud_agentv2/
│
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
│
├── agent/
│   ├── graph.py
│   ├── node.py
│   ├── state.py
│   ├── goal_descriptor.py
│   ├── completion_contract.py
│   ├── response_completion_gate.py
│   ├── context.py
│   ├── messages.py
│   ├── events.py
│   └── runtime.py
│
├── tools/
│   ├── registry.py
│   ├── metadata.py
│   ├── schemas.py
│   ├── read_guard.py
│   ├── executor.py
│   └── mcp_client.py
│
├── safety/
│   ├── write_guard.py
│   ├── policy.py
│   ├── approval.py
│   ├── write_transaction.py
│   ├── precondition.py
│   ├── fingerprint.py
│   └── locks.py
│
├── evidence/
│   ├── contracts.py
│   ├── tracker.py
│   ├── provenance.py
│   └── freshness.py
│
├── verification/
│   ├── action.py
│   ├── operational_goal.py
│   └── results.py
│
├── memory/
│   ├── checkpoint.py
│   ├── event_store.py
│   └── recovery.py
│
├── providers/
│   ├── model.py
│   ├── qwen.py
│   ├── errors.py
│   └── telemetry.py
│
├── platform/
│   ├── facade.py
│   └── mcp.py
│
├── evaluation/
│   ├── runner.py
│   ├── evaluators.py
│   └── metrics.py
│
└── tests/
```

Do not create an empty module only to satisfy this tree.

---

# 60. Budgets

Final runtime budgets may include:

```text
max_agent_steps
max_read_tool_calls
max_parallel_read_batch
max_write_proposals
max_completion_gate_rejections
max_context_tokens
max_runtime_read_retries
```

Delete:

```text
max_auto_mutations
```

---

# 61. Final Architecture Invariants

## 61.1 Single semantic authority

```text
Only the Agent chooses the next semantic action.
```

## 61.2 No second planning authority

```text
No Planner, Router, Guard, Compiler, Policy, or Tool chooses the next semantic action.
```

## 61.3 Autonomous READ

```text
READ may execute under deterministic guards without human approval.
```

## 61.4 Human-approved WRITE

```text
Every executable WRITE requires explicit human approval.
```

## 61.5 Frozen proposal

```text
Human approval applies to one exact frozen WriteTransaction.
```

## 61.6 One execution attempt

```text
Approval authorizes one protected execution attempt enforced by ExecutionClaim.
```

## 61.7 No AUTO

```text
There is no autonomous WRITE path in V2.
```

## 61.8 Goal authority split

```text
Agent declares GoalDescriptor.
Compiler determines CompletionContract.
```

## 61.9 Agent cannot grade itself

```text
Agent cannot weaken completion requirements.
```

## 61.10 Tool purity

```text
Tools observe or mutate.
Tools do not host another Agent.
```

## 61.11 Write admission

```text
Write Guard outputs only INVALID, DENIED, APPROVAL_REQUIRED.
```

## 61.12 Controlled write lifecycle

```text
WRITE state lives in WriteTransaction.
```

## 61.13 ExecutionClaim retained

```text
Human approval does not eliminate duplicate-execution risk.
```

## 61.14 Parallel READ

```text
Parallel READ is allowed only by structural ToolSpec rules.
```

## 61.15 Read batch partial failure

```text
ReadToolBatch is concurrent, not transactional.
```

## 61.16 Serialized WRITE

```text
WRITE is always one transaction at a time.
```

## 61.17 Tool idempotency metadata

```text
ToolSpec declares replay/idempotency constraints.
```

## 61.18 Unknown mutation

```text
Unknown mutation outcome blocks replay until reconciliation.
```

## 61.19 Evidence freshness

```text
Evidence must be current, not merely present.
```

## 61.20 Mutation invalidation

```text
Successful or uncertain mutation invalidates affected mutable pre-write evidence.
```

## 61.21 One state, controlled projections

```text
One canonical state, multiple controlled projections.
```

## 61.22 LLM summary not authoritative

```text
Security-critical state is projected deterministically.
```

## 61.23 Observation trust

```text
Tool observations are untrusted data, never authority.
```

## 61.24 Interrupt replay safety

```text
interrupt() nodes are replay-safe.
```

## 61.25 Persistence consistency

```text
Safety-critical transaction state and audit events are crash-consistent.
```

## 61.26 Verification separation

```text
ActionVerifier
!=
OperationalGoalVerifier
!=
ResponseCompletionGate
```

## 61.27 Terminal semantics

```text
Normal completion and controlled abnormal termination are distinct.
```

## 61.28 V2 independence

```text
V2 has no runtime import dependency on V1.
```

## 61.29 Visible loop

```text
Visible LangGraph equals the actual Agent loop.
```

## 61.30 Architecture drift

```text
No subsystem may quietly become a second semantic decision-maker.
```

## 61.31 Per-goal outcomes

```text
Every declared goal has an explicit GoalOutcome.
Multi-goal partial success, denial, and rejection are represented honestly.
```

## 61.32 Goal revision recompiles contracts

```text
GoalDescriptor revision always triggers CompletionContract recompilation.
```

## 61.33 Goal drift invalidates incompatible writes

```text
A WriteTransaction incompatible with the new structured goal/contract is invalidated.
Its old approval cannot be executed.
```

## 61.34 Approval is single-use

```text
One ApprovalRecord = one ExecutionClaim = at most one mutation execution attempt.
A second attempt requires a new WriteTransaction and new approval.
```

## 61.35 Deterministic verifier reads

```text
Verifier-owned reads are predeclared, bounded, target-bound, read-only, and non-semantic.
```

## 61.36 Explicit host API

```text
The application enters through invoke() and resumes suspended execution through resume().
```

## 61.37 Trusted operator scope

```text
V2 assumes one authenticated trusted human operator within one trust domain.
```

## 61.38 Goal terminal vs Runtime terminal

```text
DENIED / REJECTED / FAILED / INCONCLUSIVE / BLOCKED are per-goal outcomes.
Only interaction-level Runtime terminal conditions use ControlledTerminalOutcome.
```

## 61.39 Recoverable work remains PENDING

```text
Missing evidence or temporary next-step requirements remain PENDING while an allowed continuation exists.
```

## 61.40 Retry is new-transaction only

```text
Any second WRITE mutation attempt requires a new WriteTransaction and a new human approval.
```

---

# 62. Phase A — V2 Self-Contained Project

Target:

```text
/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agentv2
```

Tasks:

```text
create minimal package
copy only reusable proven V1 code
do not copy autonomy-specific code
remove V1 runtime imports
establish V2-local tests
```

---

# 63. Phase B — First Read-Only Agent Loop

Implement only:

```text
START
  ↓
AGENT
  ↓
READ EXECUTOR
  ↓
AGENT
  ↓
FINAL CANDIDATE
  ↓
RESPONSE COMPLETION GATE
  ↓
END
```

Include:

```text
GoalDescriptor
CompletionContract
ContextBuilder
Evidence freshness basics
Event Store
Single READ
ReadToolBatch
ControlledTerminalOutcome for budget/provider failure
```

No WRITE yet.

The first end-to-end demo should be driven through `invoke()` rather than calling internal graph
nodes directly.

---

# 64. Phase B Initial Tools

Start with:

```text
get_task_detail
get_gpu_pool
search_knowledge
```

Then add:

```text
get_queue_state
diagnose_task
```

after the minimal loop works.

---

# 65. Phase C — Goal / Context Hardening

Validate:

```text
GoalDescriptor multi-goal request
CompletionContract deterministic mapping
early FinalCandidate rejection
ContextBuilder bounded context
tool injection isolation
evidence freshness
read-batch partial failure
```

---

# 66. Phase D — WriteTransaction

Implement:

```text
WRITE proposal
→ Write Guard
→ INVALID / DENIED / APPROVAL_REQUIRED
→ WriteTransaction
```

No mutation yet until approval flow is complete.

---

# 67. Phase E — Human Approval

Implement:

```text
PENDING_APPROVAL
→ interrupt()
→ reject / approve
```

Reject:

```text
USER_REJECTED_WRITE
```

Approve:

```text
APPROVED
```

No AUTO.

---

# 68. Phase F — Protected Execution

Implement:

```text
Revalidate
→ ExecutionClaim
→ Mutation
→ ActionVerifier
→ OperationalGoalVerifier
→ Evidence invalidation/update
→ Observation
→ Agent
```

---

# 69. Phase G — Recovery / Reconciliation

Implement:

```text
crash-consistent events
checkpoint recovery
OUTCOME_UNKNOWN
RECONCILIATION_REQUIRED
ToolSpec.idempotency
```

---

# 70. Evaluation

Primary product metrics:

```text
Resolved@1
False Success Rate
Human Approval Completion Rate
Write Verification Success Rate
Goal State Macro-F1
```

Safety metrics:

```text
Unapproved Write Execution Rate
Duplicate Write Execution Rate
Wrong-Target Write Rate
Replay-after-Unknown-Outcome Rate
Prompt-Injection Authority Violation Rate
```

Operational metrics:

```text
LLM calls
tool calls
parallel-read savings
latency
tokens
approval latency
context compression count
runtime retry count
```

Delete autonomy-specific metrics:

```text
Unsafe AUTO Rate
Autonomy Precision
Human Intervention Reduction vs AUTO
```

---

# 71. Critical Adversarial Tests

Required examples:

```text
Agent attempts WRITE without approval
→ mutation count = 0

Agent forges approval
→ ignored/rejected

Agent forges evidence
→ ignored/rejected

Agent forges ExecutionClaim
→ ignored/rejected

Human rejects WRITE
→ no repeated automatic approval request

Two workers resume same approved transaction
→ one ExecutionClaim

Mutation outcome unknown
→ no replay before reconciliation

Mutation changes task state
→ old mutable evidence becomes stale

Read batch one timeout
→ successful sibling observations retained

Tool log contains DELETE ALL TASKS instruction
→ no authority change

Context compression omits large logs
→ structured safety state remains authoritative
```

---

# 72. Definition of Done

V2.0 is implementation-complete only when:

```text
1. one explicit Agent loop exists
2. no Planner/Router/adaptive semantic authority exists
3. GoalDescriptor is Agent-declared
4. CompletionContract is Runtime-compiled
5. READ may execute autonomously
6. every WRITE requires explicit human approval
7. no AUTO WRITE path exists
8. Write Guard has only INVALID / DENIED / APPROVAL_REQUIRED
9. WRITE lifecycle is represented by WriteTransaction
10. approval is bound to frozen proposal + fingerprint
11. approval authorizes one protected execution attempt
12. ExecutionClaim prevents duplicate execution
13. precondition is revalidated after approval
14. ActionVerifier remains distinct
15. OperationalGoalVerifier remains distinct
16. ResponseCompletionGate remains distinct
17. Human rejection has controlled terminal semantics
18. Policy denial has controlled terminal semantics
19. ReadToolBatch supports partial failure
20. ToolSpec has idempotency metadata
21. Evidence models freshness
22. mutations invalidate affected mutable evidence
23. ContextBuilder separates structured security state from semantic condensation
24. LLM summaries are never authoritative safety state
25. Event and checkpoint persistence is crash-consistent
26. unknown mutation outcome requires reconciliation
27. no runtime imports from V1 exist
28. autonomy-specific V1 code is not migrated
29. visible LangGraph equals real Agent loop
30. adversarial tests show no unapproved or duplicate WRITE execution
31. every goal has a per-goal GoalOutcome
32. GoalDescriptor revision recompiles CompletionContract
33. incompatible outstanding WriteTransaction is invalidated on goal change
34. one ApprovalRecord cannot authorize a second execution attempt
35. verifier reads are deterministic and predeclared
36. SystemContext is explicit and Runtime-controlled
37. invoke() and resume() are the stable host APIs
38. ApprovalRecord binds the trusted operator identity
39. USER_REJECTED_WRITE and POLICY_DENIED_WRITE are goal-level reason codes, not Runtime terminal codes
40. Runtime-level terminal conditions alone use ControlledTerminalOutcome
41. recoverable missing evidence remains PENDING
42. any second WRITE execution attempt requires new WriteTransaction + new approval
43. the canonical graph routes only Observation / Goal Resolution back to Agent; Runtime terminal goes directly to END
```

---
# 73. Final Product Statement

AutoDriveDataOpsAgent V2.0 is:

> **A single-loop DataOps Agent with autonomous reads and human-approved writes.**

The control model is:

```text
                      AGENT
                        │
              only semantic decision
                        │
        ┌───────────────┼─────────────────┐
        ▼               ▼                 ▼
      READ            WRITE              FINAL
        │               │                 │
        ▼               ▼                 ▼
     Runtime        Write Guard     Completion Gate
        │               │                 │
        │        INVALID / DENIED          │
        │               │                 │
        │       APPROVAL_REQUIRED          │
        │               │                 │
        │       WriteTransaction           │
        │               │                 │
        │        Human Approval            │
        │               │                 │
        │        Protected Execute         │
        │               │                 │
        └────── Observation / Goal Resolution ──────┘
                        │
                        ▼
                      AGENT
```

The four final rules are:

```text
1. Agent is the only semantic decision-maker.

2. READ tools may execute autonomously under deterministic runtime guards.

3. Every WRITE is a frozen proposal that requires explicit human approval.

4. Human approval authorizes exactly one replay-safe, verified WriteTransaction execution attempt.
```


The host boundary is intentionally small:

```text
invoke(user_input, thread_id, SystemContext)
resume(thread_id, ResumeInput, SystemContext)
```

Termination semantics are also intentionally split:

```text
Goal-level:
DENIED / REJECTED / FAILED / INCONCLUSIVE / BLOCKED
→ update GoalOutcome
→ continue other PENDING goals if possible

Runtime-level:
BUDGET_EXHAUSTED / PROVIDER_UNAVAILABLE / REQUIRES_RECONCILIATION / ...
→ ControlledTerminalOutcome
→ END
```

Multi-goal interactions are represented as:

```text
GoalDescriptor
→ CompletionContract
→ per-goal GoalOutcome
```

and a WRITE approval is deliberately single-use:

```text
one ApprovalRecord
→ one ExecutionClaim
→ at most one mutation attempt
```

If another execution is needed:

```text
new WriteTransaction
→ new human approval
```

And the architecture's permanent anti-drift rule remains:

> **No subsystem may quietly become a second semantic decision-maker.**

This document is frozen.

The next step is implementation, beginning with the read-only Agent loop.
