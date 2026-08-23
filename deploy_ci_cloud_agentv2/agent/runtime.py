"""Stable V2 host boundary for autonomous READs and human-approved WRITEs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import wraps
import inspect
from pathlib import Path
import uuid
from typing import Any

from .budgets import RuntimeBudgets
from .context import ContextBuilder
from .capabilities import build_capability_projection
from .contracts import CompletionContractCompiler
from .events import EventProvenance, EventStore
from .evidence import EvidenceKind, EvidenceTracker
from .gate import ResponseCompletionGate
from .graph import GraphDependencies, build_graph
from .principles import load_operating_principles
from .state import AgentState, InMemoryCheckpointer, LatestStateHolder, new_state
from .outcomes import ControlledTerminalOutcome, GoalOutcome, GoalStatus, TerminalCode
from ..platform.facade import InMemoryReadFacade, ReadFacade
from ..providers.deterministic import DeterministicReadAgent
from ..providers.model import AgentProvider
from ..safety.approval import (
    ApprovalInterrupt, ApprovalRecordConflict, ApprovalRecordStore, ApprovalValidator, ResumeInput,
)
from ..safety.locks import (
    ExecutionClaimAlreadyExists, ExecutionClaimStore, MutationAttemptAlreadyConsumed,
    active_mutations,
)
from ..safety.runtime_lock import runtime_ownership
from ..safety.policy import WriteAdmissionPolicy
from ..safety.write_guard import WriteGuard
from ..safety.write_transaction import ReconciliationState, WriteTransactionStatus
from ..tools.catalog import build_full_registry, build_read_registry
from ..tools.metadata import Idempotency, ToolKind
from ..tools.registry import ToolCatalogIntegrityError, ToolRegistry
from ..tools.runtime import ReadToolRuntime
from ..tools.write_runtime import WriteToolRuntime
from ..verification.action import ActionVerifier
from ..verification.operational_goal import OperationalGoalVerifier
from ..verification.results import VerificationStatus
from ..memory.sqlite import (
    CheckpointIntegrityError, DurableConcurrencyError, SQLiteApprovalRecordStore,
    SQLiteCheckpointer, SQLiteEventStore, SQLiteExecutionClaimStore,
)
from ..memory.codec import LangGraphCheckpointSerializer


@dataclass(frozen=True)
class SystemContext:
    runtime_version: str
    environment: str
    operator_id: str
    trust_domain: str
    tool_catalog_hash: str
    policy_version: str
    event_store: Any
    checkpointer: Any
    graph_checkpointer: Any
    provider: AgentProvider
    read_facade: ReadFacade
    tool_registry: ToolRegistry
    principles_path: str
    budgets: RuntimeBudgets
    claim_store: Any
    approval_store: Any
    write_policy: WriteAdmissionPolicy
    write_enabled: bool
    context_builder: ContextBuilder | None = None
    runtime_lock_path: str | None = None
    single_instance: bool = True

    def operation_ownership(self):
        if self.single_instance is not True:
            raise ValueError(
                "single_instance=false is unsupported; the current Runtime requires one active instance"
            )
        return runtime_ownership(
            self.runtime_lock_path,
            enabled=True,
        )


@dataclass(frozen=True)
class AgentRunResult:
    thread_id: str
    request_id: str
    status: str
    response: str | None
    goal_outcomes: tuple[Any, ...]
    pending_interrupt: object | None = None
    terminal_outcome: object | None = None
    state: AgentState | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    thread_id: str
    transaction_id: str
    status: str
    effect_confirmed: bool
    replay_allowed: bool
    state: AgentState



def _supports_write(facade: object) -> bool:
    return all(
        callable(getattr(facade, name, None))
        for name in (
            "resume_task",
            "submit_task",
            "stop_task",
            "delete_task",
            "set_task_priority",
        )
    )


def _default_graph_checkpointer() -> Any:
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as exc:  # pragma: no cover - production dependency error
        raise RuntimeError("LangGraph checkpointer support is required for V2 HITL") from exc
    # LangGraph 1.2.x accepts a SerializerProtocol.  Use the same explicit
    # allow-listed V2 codec as durable checkpoints so typed Runtime state is
    # serializable without pickle.  The signature check only keeps the
    # offline test-only harness compatible; production LangGraph always gets
    # the serializer.
    if "serde" in inspect.signature(InMemorySaver).parameters:
        return InMemorySaver(serde=LangGraphCheckpointSerializer())
    return InMemorySaver()


def build_system_context(
    provider: AgentProvider | None = None,
    *,
    read_facade: ReadFacade | None = None,
    event_store: EventStore | None = None,
    checkpointer: InMemoryCheckpointer | None = None,
    graph_checkpointer: Any | None = None,
    claim_store: ExecutionClaimStore | None = None,
    approval_store: ApprovalRecordStore | None = None,
    write_policy: WriteAdmissionPolicy | None = None,
    budgets: RuntimeBudgets | None = None,
    principles_path: str | Path | None = None,
    environment: str = "offline",
    operator_id: str = "trusted-operator",
    trust_domain: str = "default-trust-domain",
    context_builder: ContextBuilder | None = None,
    durable_path: str | Path | None = None,
    runtime_root: str | Path | None = None,
    single_instance: bool = True,
) -> SystemContext:
    """Build one explicit Runtime-controlled context.

    A facade that implements the five WRITE methods receives the complete
    READ+WRITE catalog. A READ-only facade remains a valid read-only host; any WRITE proposal
    fails closed because no WRITE ToolSpec exists.
    """
    if single_instance is not True:
        raise ValueError(
            "single_instance=false is unsupported; the current Runtime requires one active instance"
        )
    facade = read_facade or InMemoryReadFacade()
    write_enabled = _supports_write(facade)
    registry = build_full_registry(facade) if write_enabled else build_read_registry(facade)
    selected_provider = provider or DeterministicReadAgent()
    source = principles_path or Path(__file__).resolve().parents[1] / "doc" / "Luna_OPERATING_PRINCIPLES.md"
    policy = write_policy or WriteAdmissionPolicy()
    if durable_path is not None:
        if event_store is None:
            event_store = SQLiteEventStore(durable_path)
        if checkpointer is None:
            checkpointer = SQLiteCheckpointer(durable_path)
        if claim_store is None:
            claim_store = SQLiteExecutionClaimStore(durable_path)
        if approval_store is None:
            approval_store = SQLiteApprovalRecordStore(durable_path)
    lock_path: Path | None = None
    if runtime_root is not None:
        lock_path = Path(runtime_root).expanduser().resolve() / "run" / "runtime.lock"
    elif durable_path is not None:
        durable = Path(durable_path).expanduser().resolve()
        lock_path = durable.with_name(durable.name + ".runtime.lock")
    return SystemContext(
        runtime_version="autodrive-dataops-agent-v2.0",
        environment=environment,
        operator_id=operator_id,
        trust_domain=trust_domain,
        tool_catalog_hash=registry.catalog_hash(),
        policy_version=policy.version,
        event_store=event_store or EventStore(),
        checkpointer=checkpointer or InMemoryCheckpointer(),
        graph_checkpointer=graph_checkpointer or _default_graph_checkpointer(),
        provider=selected_provider,
        read_facade=facade,
        tool_registry=registry,
        principles_path=str(source),
        budgets=budgets or RuntimeBudgets(),
        claim_store=claim_store or ExecutionClaimStore(),
        approval_store=approval_store or ApprovalRecordStore(),
        write_policy=policy,
        write_enabled=write_enabled,
        context_builder=context_builder,
        runtime_lock_path=str(lock_path) if lock_path is not None else None,
        single_instance=single_instance,
    )


def _event_provenance(system_context: SystemContext, snapshot: Any) -> EventProvenance:
    return EventProvenance(
        model_version=getattr(system_context.provider, "model_version", "unknown"),
        prompt_version=getattr(system_context.provider, "prompt_version", "unknown"),
        tool_catalog_hash=system_context.tool_catalog_hash,
        operating_principles_version=snapshot.version,
        operating_principles_hash=snapshot.content_hash,
        policy_version=system_context.policy_version,
    )


def _append_runtime_event(
    system_context: SystemContext,
    state: AgentState,
    *,
    event_type: str,
    payload: dict[str, Any],
    provenance: EventProvenance,
    causation_id: str | None,
    current=None,
    event_id: str | None = None,
):
    """Append a host/runtime event, atomically checkpointing when durable."""
    active = current or state["current_request"]
    candidate = dict(state)
    candidate["current_request"] = active
    appender = getattr(system_context.event_store, "append_with_checkpoint", None)
    if callable(appender):
        return appender(
            system_context.checkpointer,
            AgentState(**candidate),
            event_type=event_type,
            request_id=active.identity.request_id,
            thread_id=active.identity.thread_id,
            payload=payload,
            provenance=provenance,
            causation_id=causation_id,
            event_id=event_id,
        )
    return system_context.event_store.append(
        event_type=event_type,
        request_id=active.identity.request_id,
        thread_id=active.identity.thread_id,
        payload=payload,
        provenance=provenance,
        causation_id=causation_id,
        event_id=event_id,
    )


def _write_replay_blocked(event_store: Any, tool_name: str, target: str) -> bool:
    blocked = False
    for event in event_store.all():
        if event.event_type not in {"WriteReplayBlocked", "WriteReplayBlockCleared"}:
            continue
        if event.payload.get("tool_name") != tool_name or event.payload.get("target") != target:
            continue
        blocked = event.event_type == "WriteReplayBlocked"
    return blocked


def _dependencies(
    system_context: SystemContext,
    current,
    latest_state: LatestStateHolder,
) -> GraphDependencies:
    read_runtime = ReadToolRuntime(
        system_context.tool_registry,
        current.identity,
        expected_catalog_hash=system_context.tool_catalog_hash,
    )
    write_guard = None
    write_runtime = None
    action_verifier = None
    operational_goal_verifier = None
    if system_context.write_enabled:
        write_guard = WriteGuard(
            system_context.tool_registry,
            system_context.read_facade,
            system_context.write_policy,
            replay_blocker=lambda tool_name, target: _write_replay_blocked(
                system_context.event_store, tool_name, target
            ),
        )
        write_runtime = WriteToolRuntime(
            system_context.tool_registry,
            system_context.claim_store,
            expected_catalog_hash=system_context.tool_catalog_hash,
        )
        action_verifier = ActionVerifier(system_context.read_facade, system_context.tool_registry)
        operational_goal_verifier = OperationalGoalVerifier(system_context.read_facade, system_context.tool_registry)
    selected_context_builder = system_context.context_builder or ContextBuilder(
        capability_projection=build_capability_projection(system_context.tool_registry)
    )
    return GraphDependencies(
        provider=system_context.provider,
        read_runtime=read_runtime,
        compiler=CompletionContractCompiler(),
        evidence_tracker=EvidenceTracker(),
        completion_gate=ResponseCompletionGate(),
        context_builder=selected_context_builder,
        event_store=system_context.event_store,
        model_version=getattr(system_context.provider, "model_version", "unknown"),
        prompt_version=getattr(system_context.provider, "prompt_version", "unknown"),
        tool_catalog_hash=system_context.tool_catalog_hash,
        policy_version=system_context.policy_version,
        latest_state_holder=latest_state,
        write_guard=write_guard,
        claim_store=system_context.claim_store if system_context.write_enabled else None,
        write_runtime=write_runtime,
        action_verifier=action_verifier,
        operational_goal_verifier=operational_goal_verifier,
        operator_id=system_context.operator_id,
        trust_domain=system_context.trust_domain,
        approval_store=system_context.approval_store if system_context.write_enabled else None,
        runtime_checkpointer=system_context.checkpointer,
    )


def _assert_catalog(system_context: SystemContext) -> None:
    if (
        not system_context.tool_registry.is_sealed
        or system_context.tool_registry.catalog_hash() != system_context.tool_catalog_hash
    ):
        raise ToolCatalogIntegrityError(
            "sealed ToolRegistry hash does not match SystemContext tool_catalog_hash"
        )


def _is_interrupted(raw_state: object) -> bool:
    return isinstance(raw_state, dict) and bool(raw_state.get("__interrupt__"))


def _strip_graph_private(state: AgentState | dict[str, Any]) -> AgentState:
    clean = dict(state)
    clean.pop("__interrupt__", None)
    return AgentState(**clean)


def _save_checkpoint(system_context: SystemContext, state: AgentState) -> None:
    saver = getattr(system_context.checkpointer, "save_consistent", None)
    if callable(saver):
        saver(system_context.event_store, state)
    else:
        system_context.checkpointer.save(state)


def _recover_lagged_projection(
    system_context: SystemContext,
    state: AgentState,
    events: tuple[Any, ...],
) -> AgentState:
    """Replay only capability events that intentionally outrank the checkpoint.

    Ordinary graph events are committed with the SQLite checkpoint in the same
    transaction. The only legal lag window is therefore an ApprovalRecord,
    ExecutionClaim, or MutationStarted capability committed atomically with its
    audit event just before the state projection is saved.
    """
    last_event_id = state.get("last_event_id")
    positions = {event.event_id: index for index, event in enumerate(events)}
    if last_event_id not in positions:
        raise CheckpointIntegrityError("checkpoint last event is not present in durable audit history")
    lag = events[positions[last_event_id] + 1 :]
    recovered = dict(state)
    current = recovered["current_request"]

    for event in lag:
        transaction = current.write_transaction
        if transaction is None:
            raise CheckpointIntegrityError("durable safety event exists without WriteTransaction")
        payload = event.payload
        if payload.get("transaction_id") != transaction.transaction_id:
            raise CheckpointIntegrityError("lagging safety event targets a different transaction")

        if event.event_type in {"ApprovalGranted", "ApprovalRejected"}:
            getter = getattr(system_context.approval_store, "get", None)
            approval = getter(transaction.approval_request_id) if callable(getter) else None
            if approval is None:
                raise CheckpointIntegrityError("approval audit event has no durable ApprovalRecord")
            if (
                approval.transaction_id != transaction.transaction_id
                or approval.fingerprint != transaction.fingerprint
                or approval.approval_id != payload.get("approval_id")
                or approval.operator_id != system_context.operator_id
                or approval.trust_domain != system_context.trust_domain
            ):
                raise CheckpointIntegrityError("durable ApprovalRecord does not match checkpoint transaction")
            if event.event_type == "ApprovalGranted":
                if approval.decision.value != "APPROVE":
                    raise CheckpointIntegrityError("ApprovalGranted event conflicts with ApprovalRecord")
                transaction = transaction.transition(
                    WriteTransactionStatus.APPROVED, approval=approval
                )
                current = replace(
                    current,
                    write_transaction=transaction,
                    pending_interrupt=None,
                    resume_input=None,
                )
            else:
                if approval.decision.value != "REJECT":
                    raise CheckpointIntegrityError("ApprovalRejected event conflicts with ApprovalRecord")
                transaction = transaction.transition(
                    WriteTransactionStatus.REJECTED, approval=approval
                )
                outcomes = dict(current.goal_outcomes)
                for goal_id in transaction.bound_goal_ids:
                    outcomes[goal_id] = GoalOutcome(
                        goal_id,
                        status=GoalStatus.REJECTED,
                        reason_code="USER_REJECTED_WRITE",
                        write_transaction_id=transaction.transaction_id,
                    )
                current = replace(
                    current,
                    write_transaction=transaction,
                    pending_interrupt=None,
                    resume_input=None,
                    goal_outcomes=outcomes,
                    gate_feedback=("USER_REJECTED_WRITE",),
                )
        elif event.event_type == "ExecutionClaimed":
            getter = getattr(system_context.claim_store, "claim_for", None)
            claim = getter(transaction.transaction_id) if callable(getter) else None
            if claim is None:
                raise CheckpointIntegrityError("ExecutionClaimed event has no durable claim")
            if (
                claim.claim_id != payload.get("claim_id")
                or claim.approval_id != payload.get("approval_id")
                or claim.fingerprint != transaction.fingerprint
            ):
                raise CheckpointIntegrityError("durable ExecutionClaim does not match audit event")
            transaction = transaction.transition(
                WriteTransactionStatus.EXECUTING, execution_claim=claim
            )
            current = replace(current, write_transaction=transaction)
        elif event.event_type == "MutationStarted":
            getter = getattr(system_context.claim_store, "attempt_for", None)
            attempt_id = getter(transaction.transaction_id) if callable(getter) else None
            if (
                not attempt_id
                or attempt_id != payload.get("execution_attempt_id")
                or transaction.execution_claim is None
                or transaction.execution_claim.claim_id != payload.get("claim_id")
            ):
                raise CheckpointIntegrityError("durable mutation attempt does not match audit event")
            transaction = transaction.transition(
                WriteTransactionStatus.EXECUTING, execution_attempt_id=attempt_id
            )
            current = replace(current, write_transaction=transaction)
        else:
            raise CheckpointIntegrityError(
                f"checkpoint lag contains non-recoverable event {event.event_type}"
            )
        recovered["current_request"] = current
        recovered["last_event_id"] = event.event_id

    recovered_state = AgentState(**recovered)
    _save_checkpoint(system_context, recovered_state)
    return recovered_state


def _promote_inflight_mutation_to_reconciliation(
    system_context: SystemContext,
    state: AgentState,
) -> AgentState:
    """Convert a durably-started but result-less mutation into uncertain state."""
    current = state["current_request"]
    transaction = current.write_transaction
    if not (
        transaction is not None
        and transaction.status is WriteTransactionStatus.EXECUTING
        and transaction.execution_attempt_id is not None
        and transaction.mutation_result is None
    ):
        return state

    # A second resume in this process may observe the durable MutationStarted
    # while the winning worker is still inside the external handler.  That is
    # live in-flight work, not evidence of a crashed process.  A fresh process
    # has no registry entry and therefore follows the conservative restart
    # reconciliation path below.
    if active_mutations.is_active(
        transaction.transaction_id, transaction.execution_attempt_id
    ):
        return state

    tracker = EvidenceTracker()
    evidence, invalidated_refs = tracker.invalidate_entities(
        current.evidence, transaction.affected_entities
    )
    transaction = transaction.transition(WriteTransactionStatus.RECONCILIATION_REQUIRED)
    terminal = ControlledTerminalOutcome(
        code=TerminalCode.REQUIRES_RECONCILIATION,
        safe_facts={
            "transaction_id": transaction.transaction_id,
            "execution_attempt_id": transaction.execution_attempt_id,
            "reason": "PROCESS_RESTART_AFTER_MUTATION_STARTED",
        },
        message_template="The mutation may have taken effect before the process stopped. Reconciliation is required.",
        human_action_required=True,
    )
    current = replace(
        current,
        write_transaction=transaction,
        evidence=evidence,
        terminal_state=terminal,
        termination_reason=terminal.code.value,
        decision=None,
        final_candidate=None,
    )
    working = dict(state)
    working["current_request"] = current
    provenance = _event_provenance(system_context, current.operating_principles_snapshot)
    parent = working.get("last_event_id")

    if invalidated_refs:
        ev = _append_runtime_event(
            system_context,
            AgentState(**working),
            event_type="EvidenceInvalidated",
            payload={
                "transaction_id": transaction.transaction_id,
                "evidence_refs": invalidated_refs,
                "reason": "UNCERTAIN_INFLIGHT_MUTATION",
            },
            provenance=provenance,
            causation_id=parent,
            current=current,
        )
        parent = ev.event_id
        working["last_event_id"] = parent

    spec = system_context.tool_registry.spec(transaction.proposal.tool_name)
    ev = _append_runtime_event(
        system_context,
        AgentState(**working),
        event_type="ReconciliationRequired",
        payload={
            "transaction_id": transaction.transaction_id,
            "execution_attempt_id": transaction.execution_attempt_id,
            "tool_name": transaction.proposal.tool_name,
            "target": transaction.affected_entities[0],
            "idempotency": spec.idempotency.value,
            "reason": "PROCESS_RESTART_AFTER_MUTATION_STARTED",
        },
        provenance=provenance,
        causation_id=parent,
        current=current,
    )
    parent = ev.event_id
    working["last_event_id"] = parent
    ev = _append_runtime_event(
        system_context,
        AgentState(**working),
        event_type="WriteReplayBlocked",
        payload={
            "transaction_id": transaction.transaction_id,
            "tool_name": transaction.proposal.tool_name,
            "target": transaction.affected_entities[0],
            "reason": "PROCESS_RESTART_AFTER_MUTATION_STARTED",
            "idempotency": spec.idempotency.value,
        },
        provenance=provenance,
        causation_id=parent,
        current=current,
    )
    parent = ev.event_id
    working["last_event_id"] = parent
    ev = _append_runtime_event(
        system_context,
        AgentState(**working),
        event_type="ControlledTerminalOutcomeProduced",
        payload={"code": terminal.code.value, "safe_facts": terminal.safe_facts},
        provenance=provenance,
        causation_id=parent,
        current=current,
    )
    working["last_event_id"] = ev.event_id
    return AgentState(**working)


def _load_checkpoint(system_context: SystemContext, thread_id: str) -> AgentState | None:
    try:
        state = system_context.checkpointer.load(thread_id)
    except CheckpointIntegrityError:
        raise
    except Exception as exc:
        raise CheckpointIntegrityError("checkpoint backend could not be read") from exc
    events = system_context.event_store.for_thread(thread_id)
    if state is None:
        if events:
            raise CheckpointIntegrityError("durable events exist without a matching checkpoint")
        return None
    if not events:
        raise CheckpointIntegrityError("checkpoint exists without durable audit history")
    if state.get("last_event_id") != events[-1].event_id:
        state = _recover_lagged_projection(system_context, state, events)
        events = system_context.event_store.for_thread(thread_id)
    _validate_safety_checkpoint(state, events)
    state = _promote_inflight_mutation_to_reconciliation(system_context, state)
    if state.get("last_event_id") != events[-1].event_id:
        events = system_context.event_store.for_thread(thread_id)
        _validate_safety_checkpoint(state, events)
    return state


def _validate_safety_checkpoint(state: AgentState, events: tuple[Any, ...]) -> None:
    current = state["current_request"]
    transaction = current.write_transaction
    if transaction is None:
        if current.pending_interrupt is not None or current.resume_input is not None:
            raise CheckpointIntegrityError("approval state exists without WriteTransaction")
        return
    tx_events = [
        event for event in events
        if event.payload.get("transaction_id") == transaction.transaction_id
        or (
            event.event_type == "WriteTransactionPrepared"
            and event.payload.get("transaction", {}).get("transaction_id") == transaction.transaction_id
        )
    ]
    event_types = {event.event_type for event in tx_events}
    if "WriteTransactionPrepared" not in event_types:
        raise CheckpointIntegrityError("WriteTransaction has no durable preparation event")
    if current.pending_interrupt is not None and "ApprovalRequested" not in event_types:
        raise CheckpointIntegrityError("pending approval has no durable ApprovalRequested event")
    if transaction.approval is not None:
        expected = "ApprovalRejected" if transaction.status is WriteTransactionStatus.REJECTED else "ApprovalGranted"
        if expected not in event_types:
            raise CheckpointIntegrityError("approval state has no matching durable approval event")
    if transaction.execution_claim is not None and "ExecutionClaimed" not in event_types:
        raise CheckpointIntegrityError("ExecutionClaim has no durable ExecutionClaimed event")
    if transaction.execution_attempt_id is not None and "MutationStarted" not in event_types:
        raise CheckpointIntegrityError("execution attempt has no durable MutationStarted event")
    if transaction.mutation_result is not None and "MutationResultRecorded" not in event_types:
        raise CheckpointIntegrityError("mutation result has no durable MutationResultRecorded event")
    if transaction.action_verification is not None and not (
        "ActionVerificationRecorded" in event_types or "ReconciliationChecked" in event_types
    ):
        raise CheckpointIntegrityError("action verification has no durable verification event")
    if transaction.operational_goal_verification is not None and not (
        "OperationalGoalVerificationRecorded" in event_types or "ReconciliationChecked" in event_types
    ):
        raise CheckpointIntegrityError("operational verification has no durable verification event")
    if transaction.status is WriteTransactionStatus.RECONCILIATION_REQUIRED and "ReconciliationRequired" not in event_types:
        raise CheckpointIntegrityError("reconciliation-required state has no durable event")


def _checkpoint_terminal(thread_id: str, code_detail: str, state: AgentState | None = None) -> AgentRunResult:
    terminal = ControlledTerminalOutcome(
        code=TerminalCode.CHECKPOINT_CORRUPTION,
        safe_facts={"reason": code_detail},
        message_template="The durable checkpoint and audit history are inconsistent.",
        human_action_required=True,
    )
    if state is None:
        return AgentRunResult(
            thread_id=thread_id, request_id="unknown", status="CONTROLLED_TERMINAL",
            response=None, goal_outcomes=(), terminal_outcome=terminal, state=None,
        )
    return _result_from_state(thread_id, state, status="CONTROLLED_TERMINAL", terminal_outcome=terminal)


def _recovery_entry_for_state(state: AgentState) -> str:
    current = state["current_request"]
    transaction = current.write_transaction
    if transaction is None:
        return "agent"
    status = transaction.status
    if status is WriteTransactionStatus.APPROVED:
        return "revalidate_write"
    if status is WriteTransactionStatus.REVALIDATING:
        return "execution_claim"
    if status is WriteTransactionStatus.EXECUTING:
        if transaction.execution_attempt_id is not None:
            raise CheckpointIntegrityError(
                "durably started mutation must reconcile rather than re-enter execute_write"
            )
        if transaction.execution_claim is None:
            raise CheckpointIntegrityError("EXECUTING transaction has no ExecutionClaim")
        return "execute_write"
    if status is WriteTransactionStatus.EXECUTED:
        return "action_verify"
    if status is WriteTransactionStatus.VERIFYING:
        if transaction.action_verification is None:
            return "action_verify"
        return "operational_goal_verify"
    if status in {
        WriteTransactionStatus.REJECTED,
        WriteTransactionStatus.INVALIDATED,
        WriteTransactionStatus.INVALIDATED_GOAL_CHANGED,
        WriteTransactionStatus.FAILED,
        WriteTransactionStatus.VERIFICATION_FAILED,
        WriteTransactionStatus.VERIFIED,
    }:
        return "agent"
    if status is WriteTransactionStatus.PENDING_APPROVAL:
        return "approval"
    return "agent"


def _run_is_unfinished(system_context: SystemContext, state: AgentState) -> bool:
    events = system_context.event_store.for_thread(state["thread_id"])
    if not events:
        return False
    if events[-1].event_type == "AgentRunCompleted":
        return False
    transaction = state["current_request"].write_transaction
    if (
        transaction is not None
        and transaction.reconciliation is not None
        and transaction.status in {WriteTransactionStatus.FAILED, WriteTransactionStatus.VERIFIED}
        and state["current_request"].terminal_state is None
    ):
        return False
    return True


async def _continue_checkpointed_run(
    *,
    thread_id: str,
    state: AgentState,
    system_context: SystemContext,
) -> AgentRunResult:
    current = state["current_request"]
    transaction = current.write_transaction
    if (
        transaction is not None
        and transaction.execution_attempt_id is not None
        and active_mutations.is_active(
            transaction.transaction_id, transaction.execution_attempt_id
        )
    ):
        # A live owner is still executing the already-started mutation.  Do
        # not route a concurrent worker into recovery or emit a false crash
        # terminal; the durable owner remains authoritative.
        raise ValueError("mutation is currently in flight in another worker")
    provenance = _event_provenance(system_context, current.operating_principles_snapshot)
    latest_state = LatestStateHolder()
    latest_state.record(state)
    if current.terminal_state is not None or current.gate_passed:
        return _finalize_graph_result(
            thread_id, state, system_context, provenance, latest_state
        )
    entry = _recovery_entry_for_state(state)
    if entry == "approval" and isinstance(current.pending_interrupt, ApprovalInterrupt):
        return AgentRunResult(
            thread_id=thread_id,
            request_id=current.identity.request_id,
            status="INTERRUPTED",
            response=None,
            goal_outcomes=tuple(current.goal_outcomes.values()),
            pending_interrupt=current.pending_interrupt,
            state=state,
        )
    dependencies = _dependencies(system_context, current, latest_state)
    try:
        _assert_catalog(system_context)
        graph = build_graph(
            dependencies,
            checkpointer=system_context.graph_checkpointer,
            entry_node=entry,
        )
        raw_final = await graph.ainvoke(
            state,
            config={
                "configurable": {
                    "thread_id": f"{current.identity.request_id}:recover:{state.get('last_event_id')}"
                },
                "recursion_limit": system_context.budgets.max_agent_steps * 8 + 40,
            },
        )
        interrupted = _is_interrupted(raw_final)
        final_state = _strip_graph_private(raw_final)
        latest_state.record(final_state)
    except (DurableConcurrencyError, ExecutionClaimAlreadyExists, MutationAttemptAlreadyConsumed) as exc:
        raise ValueError("another worker already advanced this durable write") from exc
    except Exception as exc:
        return _runtime_exception_result(
            thread_id, state, system_context, provenance, latest_state, exc
        )
    if interrupted:
        final_current = final_state["current_request"]
        _save_checkpoint(system_context, final_state)
        return AgentRunResult(
            thread_id=thread_id,
            request_id=final_current.identity.request_id,
            status="INTERRUPTED",
            response=None,
            goal_outcomes=tuple(final_current.goal_outcomes.values()),
            pending_interrupt=final_current.pending_interrupt,
            state=final_state,
        )
    return _finalize_graph_result(
        thread_id, final_state, system_context, provenance, latest_state
    )


def _resume_matches_durable_approval(
    current,
    resume_input: ResumeInput,
    system_context: SystemContext,
) -> bool:
    transaction = current.write_transaction
    approval = getattr(transaction, "approval", None) if transaction is not None else None
    return bool(
        transaction is not None
        and approval is not None
        and resume_input.approval_request_id == transaction.approval_request_id
        and resume_input.transaction_id == transaction.transaction_id
        and resume_input.fingerprint == transaction.fingerprint
        and resume_input.decision is approval.decision
        and approval.operator_id == system_context.operator_id
        and approval.trust_domain == system_context.trust_domain
    )


def _runtime_owned(operation):
    """Hold Runtime ownership across the complete async operation."""
    @wraps(operation)
    async def wrapped(*args, **kwargs):
        system_context = kwargs.get("system_context")
        if system_context is None:
            raise TypeError("system_context is required")
        with system_context.operation_ownership():
            return await operation(*args, **kwargs)
    return wrapped


@_runtime_owned
async def invoke(
    user_input: str,
    *,
    thread_id: str,
    system_context: SystemContext,
) -> AgentRunResult:
    """Start a normal conversational turn; never bypass an outstanding approval."""
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must not be empty")

    try:
        prior = _load_checkpoint(system_context, thread_id)
    except CheckpointIntegrityError as exc:
        return _checkpoint_terminal(thread_id, str(exc))
    if prior is not None:
        prior_current = prior["current_request"]
        transaction = prior_current.write_transaction
        if (
            isinstance(prior_current.pending_interrupt, ApprovalInterrupt)
            and transaction is not None
            and transaction.status is WriteTransactionStatus.PENDING_APPROVAL
        ):
            return AgentRunResult(
                thread_id=thread_id,
                request_id=prior_current.identity.request_id,
                status="INTERRUPTED",
                response=None,
                goal_outcomes=tuple(prior_current.goal_outcomes.values()),
                pending_interrupt=prior_current.pending_interrupt,
                state=prior,
            )
        if transaction is not None and transaction.status is WriteTransactionStatus.RECONCILIATION_REQUIRED:
            terminal = prior_current.terminal_state or ControlledTerminalOutcome(
                code=TerminalCode.REQUIRES_RECONCILIATION,
                safe_facts={"transaction_id": transaction.transaction_id},
                message_template="The prior mutation outcome is unknown and must be reconciled before any future write.",
                human_action_required=True,
            )
            return _result_from_state(
                thread_id, prior, status="CONTROLLED_TERMINAL", terminal_outcome=terminal
            )
        if _run_is_unfinished(system_context, prior):
            return await _continue_checkpointed_run(
                thread_id=thread_id, state=prior, system_context=system_context
            )

    snapshot = load_operating_principles(system_context.principles_path)
    state = new_state(
        user_input=user_input,
        thread_id=thread_id,
        snapshot=snapshot,
        budgets=system_context.budgets,
        prior=prior,
    )
    provenance = _event_provenance(system_context, snapshot)
    latest_state = LatestStateHolder()
    latest_state.record(state)
    try:
        started = _append_runtime_event(
            system_context,
            state,
            event_type="AgentRunStarted",
            payload={"user_input_length": len(user_input)},
            provenance=provenance,
            causation_id=prior.get("last_event_id") if prior is not None else None,
        )
    except Exception as exc:
        terminal = ControlledTerminalOutcome(
            code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
            safe_facts={"event_store_error_type": type(exc).__name__},
            message_template="The runtime could not establish an immutable audit boundary.",
        )
        failed_state = dict(state)
        failed_state["current_request"] = replace(
            state["current_request"],
            terminal_state=terminal,
            termination_reason=terminal.code.value,
        )
        return _result_from_state(
            thread_id, failed_state, status="CONTROLLED_TERMINAL", terminal_outcome=terminal
        )

    state["last_event_id"] = started.event_id
    latest_state.record(state)
    dependencies = _dependencies(system_context, state["current_request"], latest_state)
    try:
        _assert_catalog(system_context)
        graph = build_graph(dependencies, checkpointer=system_context.graph_checkpointer)
        raw_final = await graph.ainvoke(
            state,
            config={
                "configurable": {"thread_id": state["current_request"].identity.request_id},
                "recursion_limit": system_context.budgets.max_agent_steps * 8 + 40,
            },
        )
        interrupted = _is_interrupted(raw_final)
        final_state = _strip_graph_private(raw_final)
        latest_state.record(final_state)
    except Exception as exc:
        return _runtime_exception_result(
            thread_id, state, system_context, provenance, latest_state, exc
        )

    if interrupted:
        final_current = final_state["current_request"]
        if not isinstance(final_current.pending_interrupt, ApprovalInterrupt):
            failure = ControlledTerminalOutcome(
                code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                safe_facts={"reason": "GRAPH_INTERRUPTED_WITHOUT_RUNTIME_APPROVAL_STATE"},
                message_template="The graph paused without a valid Runtime approval boundary.",
            )
            return _result_from_state(
                thread_id, final_state, status="CONTROLLED_TERMINAL", terminal_outcome=failure
            )
        # WriteTransactionPrepared + ApprovalRequested are already durable.
        # Checkpoint only after that durable tail exists.
        _save_checkpoint(system_context, final_state)
        return AgentRunResult(
            thread_id=thread_id,
            request_id=final_current.identity.request_id,
            status="INTERRUPTED",
            response=None,
            goal_outcomes=tuple(final_current.goal_outcomes.values()),
            pending_interrupt=final_current.pending_interrupt,
            state=final_state,
        )

    return _finalize_graph_result(
        thread_id, final_state, system_context, provenance, latest_state
    )


@_runtime_owned
async def resume(
    *,
    thread_id: str,
    resume_input: ResumeInput | object,
    system_context: SystemContext,
) -> AgentRunResult:
    """Resume the exact durable suspended transaction through the approval node."""
    try:
        prior = _load_checkpoint(system_context, thread_id)
    except CheckpointIntegrityError as exc:
        return _checkpoint_terminal(thread_id, str(exc))
    if prior is None:
        raise ValueError("no checkpoint exists for thread")
    current = prior["current_request"]
    transaction = current.write_transaction
    pending = current.pending_interrupt
    if not isinstance(resume_input, ResumeInput):
        if not isinstance(resume_input, dict):
            raise ValueError("resume_input must be ResumeInput or a mapping")
        resume_input = ResumeInput(**resume_input)
    if transaction is None:
        raise ValueError("thread is not suspended for approval")
    if (
        transaction.execution_attempt_id is not None
        and active_mutations.is_active(
            transaction.transaction_id, transaction.execution_attempt_id
        )
    ):
        raise ValueError("mutation is currently in flight in another worker")
    if transaction.status is WriteTransactionStatus.RECONCILIATION_REQUIRED:
        terminal = current.terminal_state or ControlledTerminalOutcome(
            code=TerminalCode.REQUIRES_RECONCILIATION,
            safe_facts={"transaction_id": transaction.transaction_id},
            message_template="The prior mutation outcome is unknown and must be reconciled before any future write.",
            human_action_required=True,
        )
        return _result_from_state(
            thread_id, prior, status="CONTROLLED_TERMINAL", terminal_outcome=terminal
        )
    if not isinstance(pending, ApprovalInterrupt):
        if _run_is_unfinished(system_context, prior) and _resume_matches_durable_approval(
            current, resume_input, system_context
        ):
            return await _continue_checkpointed_run(
                thread_id=thread_id, state=prior, system_context=system_context
            )
        raise ValueError("thread is not suspended for approval")
    ApprovalValidator.validate_binding(
        transaction, pending, resume_input,
        operator_id=system_context.operator_id, trust_domain=system_context.trust_domain,
    )
    _assert_catalog(system_context)

    snapshot = current.operating_principles_snapshot
    provenance = _event_provenance(system_context, snapshot)
    resumed_current = replace(current, resume_input=resume_input)
    resume_state = dict(prior)
    resume_state["current_request"] = resumed_current
    latest_state = LatestStateHolder()
    latest_state.record(resume_state)
    dependencies = _dependencies(system_context, resumed_current, latest_state)
    try:
        # Prefer the real LangGraph Command(resume=...) protocol when the
        # in-process graph checkpointer still owns the interrupt checkpoint.
        # The V2 SQLite checkpoint remains the recovery authority after a
        # process restart, in which case the replay-safe approval entry path
        # below rebuilds the graph from Runtime-owned state.
        graph_thread_id = current.identity.request_id
        graph_checkpoint_exists = False
        graph_getter = getattr(system_context.graph_checkpointer, "get_tuple", None)
        if callable(graph_getter):
            try:
                graph_checkpoint_exists = (
                    graph_getter({"configurable": {"thread_id": graph_thread_id}})
                    is not None
                )
            except Exception:
                graph_checkpoint_exists = False

        if graph_checkpoint_exists:
            from langgraph.types import Command

            graph = build_graph(
                dependencies, checkpointer=system_context.graph_checkpointer
            )
            raw_final = await graph.ainvoke(
                Command(resume=resume_input),
                config={
                    "configurable": {"thread_id": graph_thread_id},
                    "recursion_limit": system_context.budgets.max_agent_steps * 8 + 40,
                },
            )
        else:
            graph = build_graph(
                dependencies, checkpointer=system_context.graph_checkpointer, resume_entry=True
            )
            raw_final = await graph.ainvoke(
                resume_state,
                config={
                    "configurable": {
                        "thread_id": f"{current.identity.request_id}:resume:{transaction.transaction_id}"
                    },
                    "recursion_limit": system_context.budgets.max_agent_steps * 8 + 40,
                },
            )
        interrupted = _is_interrupted(raw_final)
        final_state = _strip_graph_private(raw_final)
        latest_state.record(final_state)
    except (DurableConcurrencyError, ExecutionClaimAlreadyExists, MutationAttemptAlreadyConsumed, ApprovalRecordConflict) as exc:
        raise ValueError("another worker already advanced this durable approval") from exc
    except Exception as exc:
        return _runtime_exception_result(
            thread_id, resume_state, system_context, provenance, latest_state, exc
        )

    if interrupted:
        final_current = final_state["current_request"]
        if not isinstance(final_current.pending_interrupt, ApprovalInterrupt):
            failure = ControlledTerminalOutcome(
                code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                safe_facts={"reason": "GRAPH_REINTERRUPTED_WITHOUT_APPROVAL_STATE"},
                message_template="The graph paused without a valid Runtime approval boundary.",
            )
            return _result_from_state(
                thread_id, final_state, status="CONTROLLED_TERMINAL", terminal_outcome=failure
            )
        _save_checkpoint(system_context, final_state)
        return AgentRunResult(
            thread_id=thread_id, request_id=final_current.identity.request_id,
            status="INTERRUPTED", response=None,
            goal_outcomes=tuple(final_current.goal_outcomes.values()),
            pending_interrupt=final_current.pending_interrupt, state=final_state,
        )

    return _finalize_graph_result(
        thread_id, final_state, system_context, provenance, latest_state
    )



@_runtime_owned
async def reconcile(*, thread_id: str, system_context: SystemContext) -> ReconciliationResult:
    """Deterministically reconcile one unknown WRITE outcome.

    This is a Runtime maintenance operation, not an Agent semantic decision.
    It performs only the verifier reads predeclared by the frozen ToolSpec.
    """
    try:
        state = _load_checkpoint(system_context, thread_id)
    except CheckpointIntegrityError as exc:
        raise CheckpointIntegrityError(f"cannot reconcile corrupt checkpoint: {exc}") from exc
    if state is None:
        raise ValueError("no checkpoint exists for thread")
    current = state["current_request"]
    transaction = current.write_transaction
    if transaction is None or transaction.status is not WriteTransactionStatus.RECONCILIATION_REQUIRED:
        raise ValueError("thread has no transaction requiring reconciliation")
    _assert_catalog(system_context)
    spec = system_context.tool_registry.spec(transaction.proposal.tool_name)
    if spec.kind is not ToolKind.WRITE:
        raise ValueError("reconciliation requires a WRITE ToolSpec")

    action = ActionVerifier(system_context.read_facade, system_context.tool_registry).verify(transaction)
    operational = None
    effect_confirmed = action.status is VerificationStatus.VERIFIED
    if effect_confirmed and spec.verification == "ACTION_AND_GOAL":
        operational = OperationalGoalVerifier(system_context.read_facade, system_context.tool_registry).verify(transaction)
        effect_confirmed = operational.status is VerificationStatus.VERIFIED

    tracker = EvidenceTracker()
    evidence = current.evidence
    outcomes = dict(current.goal_outcomes)
    if effect_confirmed:
        evidence, action_record = tracker.record_verification(
            evidence,
            kind=EvidenceKind.ACTION_VERIFIED,
            target=action.target,
            source="reconciliation_action_verifier",
            observation_id=f"reconcile_action_{uuid.uuid4().hex}",
            owner=current.identity,
        )
        if operational is not None:
            evidence, goal_record = tracker.record_verification(
                evidence,
                kind=EvidenceKind.OPERATIONAL_GOAL_VERIFIED,
                target=operational.target,
                source="reconciliation_operational_goal_verifier",
                observation_id=f"reconcile_goal_{uuid.uuid4().hex}",
                owner=current.identity,
            )
        else:
            goal_record = None
        transaction = transaction.transition(
            WriteTransactionStatus.VERIFIED,
            action_verification=action,
            operational_goal_verification=operational,
            reconciliation=ReconciliationState(
                "EFFECT_CONFIRMED",
                detail={"action": action.reason_code, "operational": getattr(operational, "reason_code", None)},
            ),
        )
        outcomes = tracker.refresh_goal_outcomes(
            current.goal_descriptor, current.completion_contract, evidence, outcomes
        )
        replay_allowed = False
        reconciliation_status = "EFFECT_CONFIRMED"
    else:
        if spec.idempotency is Idempotency.NO_RETRY:
            reconciliation_status = "NO_CURRENT_EFFECT_CONFIRMED_NO_RETRY"
            replay_allowed = False
            goal_status = GoalStatus.BLOCKED
            reason = "NO_RETRY_AFTER_UNKNOWN_OUTCOME"
        else:
            reconciliation_status = "NO_CURRENT_EFFECT_CONFIRMED_NEW_TRANSACTION_ALLOWED"
            replay_allowed = True
            goal_status = GoalStatus.INCONCLUSIVE
            reason = "UNKNOWN_OUTCOME_RECONCILED_NO_CURRENT_EFFECT"
        transaction = transaction.transition(
            WriteTransactionStatus.FAILED,
            action_verification=action,
            operational_goal_verification=operational,
            reconciliation=ReconciliationState(
                reconciliation_status,
                detail={"action": action.reason_code, "idempotency": spec.idempotency.value},
            ),
        )
        for goal_id in transaction.bound_goal_ids:
            outcomes[goal_id] = GoalOutcome(
                goal_id,
                status=goal_status,
                reason_code=reason,
                write_transaction_id=transaction.transaction_id,
            )
        action_record = None
        goal_record = None

    updated = replace(
        current,
        write_transaction=transaction,
        evidence=evidence,
        goal_outcomes=outcomes,
        terminal_state=None,
        termination_reason=None,
        decision=None,
        final_candidate=None,
        pending_interrupt=None,
        resume_input=None,
        gate_passed=None,
        gate_feedback=(f"RECONCILIATION:{reconciliation_status}",),
    )
    provenance = _event_provenance(system_context, current.operating_principles_snapshot)
    working = dict(state)
    working["current_request"] = updated
    last_event_id = state.get("last_event_id")

    # Reconciliation is a Runtime maintenance path, but its verification
    # decisions are still safety-relevant audit truth.  Record the same
    # verifier event types used by the normal WRITE lifecycle before the
    # reconciliation summary.  On SQLite durability each append is committed
    # atomically with the corresponding checkpoint projection.
    verification_events = [
        (
            "ActionVerificationRecorded",
            {
                "transaction_id": transaction.transaction_id,
                "status": action.status.value,
                "reason_code": action.reason_code,
                "reconciliation": True,
            },
        )
    ]
    if operational is not None:
        verification_events.append(
            (
                "OperationalGoalVerificationRecorded",
                {
                    "transaction_id": transaction.transaction_id,
                    "status": operational.status.value,
                    "reason_code": operational.reason_code,
                    "reconciliation": True,
                },
            )
        )

    for event_type, payload in verification_events:
        ev = _append_runtime_event(
            system_context,
            AgentState(**working),
            event_type=event_type,
            payload=payload,
            provenance=provenance,
            causation_id=last_event_id,
            current=updated,
        )
        last_event_id = ev.event_id
        working["last_event_id"] = last_event_id

    event = _append_runtime_event(
        system_context,
        AgentState(**working),
        event_type="ReconciliationChecked",
        payload={
            "transaction_id": transaction.transaction_id,
            "tool_name": transaction.proposal.tool_name,
            "target": transaction.affected_entities[0],
            "status": reconciliation_status,
            "effect_confirmed": effect_confirmed,
            "replay_allowed": replay_allowed,
        },
        provenance=provenance,
        causation_id=last_event_id,
        current=updated,
    )
    last_event_id = event.event_id
    working["last_event_id"] = last_event_id

    for record in (action_record, goal_record):
        if record is None:
            continue
        ev = _append_runtime_event(
            system_context,
            AgentState(**working),
            event_type="EvidenceRecorded",
            payload={
                "evidence_id": record.evidence_id,
                "kind": record.kind.value,
                "target": record.target,
                "observation_id": record.observation_id,
                "owner": asdict(record.owner),
                "provenance": asdict(record.provenance),
                "freshness": asdict(record.freshness),
            },
            provenance=provenance,
            causation_id=last_event_id,
            current=updated,
        )
        last_event_id = ev.event_id
        working["last_event_id"] = last_event_id

    if spec.idempotency is not Idempotency.NO_RETRY:
        cleared = _append_runtime_event(
            system_context,
            AgentState(**working),
            event_type="WriteReplayBlockCleared",
            payload={
                "transaction_id": transaction.transaction_id,
                "tool_name": transaction.proposal.tool_name,
                "target": transaction.affected_entities[0],
                "reconciliation_status": reconciliation_status,
            },
            provenance=provenance,
            causation_id=last_event_id,
            current=updated,
        )
        last_event_id = cleared.event_id
        working["last_event_id"] = last_event_id

    final_state = AgentState(**working)
    # In-memory stores do not checkpoint inside _append_runtime_event.  The
    # durable SQLite path has already checkpointed every event atomically;
    # this final save is idempotent for both implementations.
    _save_checkpoint(system_context, final_state)
    return ReconciliationResult(
        thread_id=thread_id,
        transaction_id=transaction.transaction_id,
        status=reconciliation_status,
        effect_confirmed=effect_confirmed,
        replay_allowed=replay_allowed,
        state=final_state,
    )


def _runtime_exception_result(
    thread_id: str,
    initial_state: AgentState,
    system_context: SystemContext,
    provenance: EventProvenance,
    latest_state: LatestStateHolder,
    exc: Exception,
) -> AgentRunResult:
    runtime_terminal = ControlledTerminalOutcome(
        code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
        safe_facts={
            "graph_error_type": type(exc).__name__,
            **(
                {"reason": "TOOL_CATALOG_INTEGRITY_ERROR"}
                if isinstance(exc, ToolCatalogIntegrityError)
                else {}
            ),
        },
        message_template="The runtime could not safely complete this interaction.",
    )
    final_state = latest_state.current() or dict(initial_state)
    terminal_state = replace(
        final_state["current_request"],
        terminal_state=runtime_terminal,
        termination_reason=runtime_terminal.code.value,
        decision=None,
        final_candidate=None,
    )
    terminal_candidate = dict(final_state)
    terminal_candidate["current_request"] = terminal_state
    try:
        terminal_event = _append_runtime_event(
            system_context,
            AgentState(**terminal_candidate),
            event_type="ControlledTerminalOutcomeProduced",
            payload={"code": runtime_terminal.code.value, "safe_facts": runtime_terminal.safe_facts},
            provenance=provenance,
            causation_id=final_state.get("last_event_id"),
            current=terminal_state,
        )
    except Exception:
        return _result_from_state(
            thread_id,
            final_state,
            status="CONTROLLED_TERMINAL",
            terminal_outcome=runtime_terminal,
        )
    final_state = dict(final_state)
    final_state["current_request"] = terminal_state
    final_state["last_event_id"] = terminal_event.event_id
    latest_state.record(final_state)
    return _finalize_graph_result(
        thread_id, final_state, system_context, provenance, latest_state
    )


def _finalize_graph_result(
    thread_id: str,
    final_state: AgentState,
    system_context: SystemContext,
    provenance: EventProvenance,
    latest_state: LatestStateHolder,
) -> AgentRunResult:
    final_current = final_state["current_request"]
    terminal = final_current.terminal_state
    passed = bool(final_current.gate_passed)
    status = "COMPLETED" if passed else "CONTROLLED_TERMINAL" if terminal else "ERROR"
    try:
        completed_event = _append_runtime_event(
            system_context,
            final_state,
            event_type="AgentRunCompleted",
            payload={"status": status, "termination_reason": final_current.termination_reason},
            provenance=provenance,
            causation_id=final_state.get("last_event_id"),
            current=final_current,
        )
    except Exception:
        fallback = latest_state.current() or final_state
        failure = ControlledTerminalOutcome(
            code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
            safe_facts={"runtime_failure": "AgentRunCompleted append failed"},
            message_template="The runtime could not durably record completion.",
        )
        return _result_from_state(
            thread_id, fallback, status="CONTROLLED_TERMINAL", terminal_outcome=failure
        )
    final_state = dict(final_state)
    final_state["last_event_id"] = completed_event.event_id
    latest_state.record(final_state)
    _save_checkpoint(system_context, final_state)
    candidate = final_current.final_candidate
    return AgentRunResult(
        thread_id=thread_id,
        request_id=final_current.identity.request_id,
        status=status,
        response=candidate.response if passed and candidate is not None else None,
        goal_outcomes=tuple(final_current.goal_outcomes.values()),
        pending_interrupt=final_current.pending_interrupt,
        terminal_outcome=terminal,
        state=final_state,
    )


def _result_from_state(
    thread_id: str,
    state: AgentState,
    *,
    status: str,
    terminal_outcome: ControlledTerminalOutcome,
) -> AgentRunResult:
    current = state["current_request"]
    return AgentRunResult(
        thread_id=thread_id,
        request_id=current.identity.request_id,
        status=status,
        response=None,
        goal_outcomes=tuple(current.goal_outcomes.values()),
        pending_interrupt=current.pending_interrupt,
        terminal_outcome=terminal_outcome,
        state=state,
    )
