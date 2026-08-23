"""Canonical visible V2.0 LangGraph Agent loop with one semantic authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any
import uuid

from .budgets import BudgetState
from .context import ContextBudgetExceeded, ContextBuilder
from .contracts import CompletionContractCompiler
from .decision_ingress import AgentDecisionIngressValidator, AgentDecisionValidationError
from .decisions import (
    AcceptedToolCall,
    AcceptedWriteCall,
    AgentDecision,
    FinalCandidate,
    ReadToolBatch,
    SingleToolCall,
)
from .evidence import (
    EvidenceKind,
    EvidenceTracker,
    ObservationDisposition,
    ToolObservation,
    TransportStatus,
)
from .events import EventProvenance, EventStore
from .gate import ResponseCompletionGate
from .identity import RequestIdentity
from .outcomes import ControlledTerminalOutcome, GoalOutcome, GoalStatus, TerminalCode
from .principles import OperatingPrinciplesSnapshot
from .provenance import (
    IdentityStatus,
    ObservationProvenance,
    ObservationScope,
    ScopeKind,
    ScopeStatus,
    canonical_tool_call_fingerprint,
)
from .state import AgentState, CurrentRequestContext, LatestStateHolder
from ..providers.model import ProviderUnavailable
from ..tools.runtime import ReadToolRuntime
from ..tools.metadata import ToolKind
from ..tools.write_runtime import WriteToolRuntime
from ..safety.approval import (
    ApprovalDecision, ApprovalInterrupt, ApprovalRecordStore, ApprovalValidator, ResumeInput,
)
from ..safety.locks import (
    ExecutionClaimAlreadyExists, ExecutionClaimStore, MutationAttemptAlreadyConsumed,
    active_mutations,
)
from ..safety.write_guard import WriteAdmissionOutcome, WriteGuard
from ..safety.write_transaction import MutationOutcome, WriteTransactionStatus
from ..verification.action import ActionVerifier
from ..verification.operational_goal import OperationalGoalVerifier
from ..verification.results import VerificationStatus


@dataclass(frozen=True)
class GraphDependencies:
    provider: Any
    read_runtime: ReadToolRuntime
    compiler: CompletionContractCompiler
    evidence_tracker: EvidenceTracker
    completion_gate: ResponseCompletionGate
    context_builder: ContextBuilder
    event_store: EventStore
    model_version: str
    prompt_version: str
    tool_catalog_hash: str
    policy_version: str
    latest_state_holder: LatestStateHolder | None = None
    write_guard: WriteGuard | None = None
    claim_store: ExecutionClaimStore | None = None
    write_runtime: WriteToolRuntime | None = None
    action_verifier: ActionVerifier | None = None
    operational_goal_verifier: OperationalGoalVerifier | None = None
    operator_id: str = "operator"
    trust_domain: str = "default"
    approval_store: Any | None = None
    runtime_checkpointer: Any | None = None


def build_graph(
    dependencies: GraphDependencies,
    *,
    checkpointer: Any | None = None,
    resume_entry: bool = False,
    entry_node: str | None = None,
):
    """Build the explicit Reason -> Action -> Observation -> Re-reason graph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LangGraph is required for the V2 explicit Agent loop") from exc

    async def agent(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        if current.terminal_state is not None:
            return {"current_request": replace(current, decision=None)}
        budget = current.budgets
        if not budget.has_agent_step():
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={
                        "agent_steps_used": budget.agent_steps_used,
                        "max_agent_steps": budget.limits.max_agent_steps,
                    },
                    message_template="The bounded execution budget was exhausted before all goals could be completed.",
                ),
            )
        budget = budget.with_agent_step()
        current_for_context = replace(current, budgets=budget)
        try:
            context = dependencies.context_builder.build(
                current_for_context, state["thread_history"]
            )
        except ContextBudgetExceeded as exc:
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={"reason": "CONTEXT_BUDGET_EXCEEDED", "detail": str(exc)},
                    message_template="The bounded Agent context could not preserve required runtime state.",
                ),
                budgets=budget,
            )
        context_projection_compressed = (
            context.user_input != current_for_context.user_input
            or len(context.messages) < len(current_for_context.messages)
            or len(context.semantic_observations.observations) < len(current_for_context.observations)
            or len(context.operating_guidance.principles)
            < len(current_for_context.operating_principles_snapshot.principles)
        )
        try:
            proposal = await dependencies.provider.generate(context)
        except ProviderUnavailable as exc:
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.PROVIDER_UNAVAILABLE,
                    safe_facts={"provider_error_type": type(exc).__name__},
                    message_template="The configured provider is unavailable, so this interaction cannot safely continue.",
                ),
                budgets=budget,
            )
        except AgentDecisionValidationError as exc:
            # A structured provider response is still an untrusted proposal.
            # Parse/schema failures are bounded decision rejections, not a
            # Runtime/provider outage, so the Agent may recover within its
            # existing step budget.
            rejection = _agent_decision_rejection_update(
                state,
                dependencies,
                current_for_context,
                reason=str(exc),
                budgets=budget,
                proposal_type="provider_response",
            )
            _remember(dependencies, state, rejection)
            return rejection
        except Exception as exc:
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"provider_error_type": type(exc).__name__},
                    message_template="The Agent provider returned an unusable decision.",
                ),
                budgets=budget,
            )

        try:
            decision: AgentDecision = AgentDecisionIngressValidator().validate(proposal)
        except AgentDecisionValidationError as exc:
            rejection = _agent_decision_rejection_update(
                state,
                dependencies,
                current_for_context,
                reason=str(exc),
                budgets=budget,
                proposal_type=type(proposal).__name__,
            )
            _remember(dependencies, state, rejection)
            return rejection

        current = replace(
            current_for_context,
            step_count=current.step_count + 1,
            decision=None,
            final_candidate=None,
            gate_passed=None,
            new_turn=False,
            continue_after_read_guard=False,
            messages=current.messages
            + (
                {
                    "role": "assistant",
                    "kind": getattr(decision, "kind", type(decision).__name__),
                    "content": decision.response
                    if isinstance(decision, FinalCandidate)
                    else getattr(decision, "kind", type(decision).__name__),
                    "candidate": isinstance(decision, FinalCandidate),
                },
            ),
        )
        last_event_id = state.get("last_event_id")

        descriptor = getattr(decision, "proposed_goal_descriptor", None)
        descriptor_error: str | None = None
        if (
            descriptor is not None
            and current.goal_descriptor is not None
            and descriptor != current.goal_descriptor
        ):
            if descriptor.descriptor_version != current.goal_descriptor.descriptor_version + 1:
                descriptor_error = "non_monotonic_goal_descriptor_revision"
        if (
            descriptor is not None
            and descriptor != current.goal_descriptor
            and descriptor_error is None
        ):
            event_type = (
                "GoalDescriptorDeclared"
                if current.goal_descriptor is None
                else "GoalDescriptorRevised"
            )
            contract = dependencies.compiler.compile(descriptor)
            descriptor_current = replace(
                current,
                goal_descriptor=descriptor,
                completion_contract=None,
                goal_outcomes={},
            )
            event = _emit(
                state,
                dependencies,
                event_type,
                {"descriptor": descriptor.to_dict()},
                current=descriptor_current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
            current = descriptor_current
            contract_current = replace(
                current,
                completion_contract=contract,
                goal_outcomes={goal.goal_id: GoalOutcome(goal.goal_id) for goal in descriptor.goals},
            )
            contract_event = _emit(
                state,
                dependencies,
                "CompletionContractCompiled",
                {
                    "descriptor_version": contract.descriptor_version,
                    "contract_version": contract.contract_version,
                    "contract_fingerprint": contract.contract_fingerprint,
                },
                current=contract_current,
                causation_id=last_event_id,
            )
            last_event_id = contract_event.event_id
            current = contract_current

            # A GoalDescriptor revision cannot silently leave an incompatible
            # pre-execution WriteTransaction/approval live.  This is a
            # deterministic structured-identity check, not semantic planning.
            transaction = current.write_transaction
            goal_change_sensitive = {
                WriteTransactionStatus.PROPOSED,
                WriteTransactionStatus.VALIDATED,
                WriteTransactionStatus.PENDING_APPROVAL,
                WriteTransactionStatus.APPROVED,
                WriteTransactionStatus.REVALIDATING,
            }
            if (
                transaction is not None
                and transaction.status in goal_change_sensitive
                and dependencies.write_guard is not None
                and not dependencies.write_guard.compatible(transaction, descriptor, contract)
            ):
                invalidated = transaction.transition(
                    WriteTransactionStatus.INVALIDATED_GOAL_CHANGED
                )
                current = replace(
                    current,
                    write_transaction=invalidated,
                    pending_interrupt=None,
                    resume_input=None,
                    gate_feedback=("WRITE_INVALIDATED_GOAL_CHANGED",),
                )
                invalidated_event = _emit(
                    state,
                    dependencies,
                    "WriteTransactionInvalidated",
                    {
                        "transaction_id": transaction.transaction_id,
                        "reason": "GOAL_CHANGED",
                        "prior_descriptor_version": transaction.goal_descriptor_version,
                        "new_descriptor_version": descriptor.descriptor_version,
                    },
                    current=current,
                    causation_id=last_event_id,
                )
                last_event_id = invalidated_event.event_id

        accepted_decision: AgentDecision | None = None
        guard_reason = descriptor_error
        if guard_reason is None:
            try:
                accepted_decision = _accept_decision(
                    decision,
                    dependencies.read_runtime,
                    dependencies.write_guard,
                    current.budgets.limits.max_parallel_read_batch,
                )
            except (TypeError, ValueError) as exc:
                guard_reason = f"READ_GUARD_REJECTED: {type(exc).__name__}: {exc}"
        if guard_reason is None and current.goal_descriptor is None and descriptor is None:
            guard_reason = "Agent did not declare a structured GoalDescriptor"
        accepted_current = replace(
            current,
            decision=accepted_decision if guard_reason is None else None,
            final_candidate=(
                accepted_decision
                if guard_reason is None and isinstance(accepted_decision, FinalCandidate)
                else None
            ),
        )
        decision_event = _emit(
            state,
            dependencies,
            "AgentDecisionMade",
            {
                "decision_kind": getattr(decision, "kind", type(decision).__name__),
                "accepted": guard_reason is None,
                "rejection_reason": guard_reason,
                "context_projection_compressed": context_projection_compressed,
                "estimated_context_chars": context.estimated_context_chars,
                **_decision_audit_payload(accepted_decision),
            },
            current=accepted_current,
            causation_id=last_event_id,
        )
        last_event_id = decision_event.event_id
        if guard_reason is not None:
            guard_updates = _read_guard_update(
                state, dependencies, current, guard_reason, budget, last_event_id
            )
            result = {"current_request": current, **guard_updates}
            _remember(dependencies, state, result)
            return result

        current = accepted_current
        if isinstance(accepted_decision, FinalCandidate):
            final_event = _emit(
                state,
                dependencies,
                "FinalCandidateProduced",
                {
                    "referenced_goal_ids": list(accepted_decision.referenced_goal_ids),
                    "response_length": len(accepted_decision.response),
                },
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = final_event.event_id
        result = {"current_request": current, "last_event_id": last_event_id}
        _remember(dependencies, state, result)
        return result

    async def read_executor(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        decision = current.decision
        if not isinstance(decision, (SingleToolCall, ReadToolBatch)):
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "read_executor_received_non_read_decision"},
                    message_template="The read runtime received an invalid AgentDecision.",
                ),
            )
        calls = (decision.call,) if isinstance(decision, SingleToolCall) else decision.calls
        budget = current.budgets
        if not budget.has_read_calls(len(calls)):
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={
                        "read_tool_calls_used": budget.read_tool_calls_used,
                        "requested_calls": len(calls),
                        "max_read_tool_calls": budget.limits.max_read_tool_calls,
                    },
                    message_template="The bounded execution budget was exhausted before all goals could be completed.",
                ),
            )
        # Agent node has already produced the accepted canonical decision.
        # Re-validating/reconstructing a provider proposal here would recreate
        # the TOCTOU boundary this runtime is required to avoid.
        if isinstance(decision, SingleToolCall):
            accepted_calls = (decision.call,)
        else:
            accepted_calls = decision.calls
        if not all(isinstance(call, AcceptedToolCall) for call in accepted_calls):
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "read_executor_received_unaccepted_call"},
                    message_template="The runtime could not execute an unaccepted read decision.",
                ),
            )

        last_event_id = state.get("last_event_id")

        async def on_started(call, attempt):
            nonlocal last_event_id
            event = _emit(
                state,
                dependencies,
                "ToolCallStarted",
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "arguments_fingerprint": canonical_tool_call_fingerprint(
                        call.tool_name, call.arguments
                    ),
                    "attempt": attempt,
                    "request_id": current.identity.request_id,
                    "tool_catalog_hash": dependencies.tool_catalog_hash,
                },
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id

        if isinstance(decision, SingleToolCall):
            observations = (
                await dependencies.read_runtime.execute_single(
                    decision.call,
                    max_retries=budget.limits.max_runtime_read_retries_per_call,
                    on_started=on_started,
                ),
            )
        else:
            observations = (
                await dependencies.read_runtime.execute_batch(
                    decision,
                    max_retries=budget.limits.max_runtime_read_retries_per_call,
                    max_batch=budget.limits.max_parallel_read_batch,
                    on_started=on_started,
                )
            ).results

        new_evidence, created = dependencies.evidence_tracker.record_observations(
            current.evidence, observations, current.identity
        )
        outcomes = current.goal_outcomes
        if current.goal_descriptor is not None and current.completion_contract is not None:
            outcomes = dependencies.evidence_tracker.refresh_goal_outcomes(
                current.goal_descriptor, current.completion_contract, new_evidence, outcomes
            )
        observation_current = replace(
            current,
            budgets=budget.with_read_calls(len(calls)).with_retries(
                sum(item.retry_count for item in observations)
            ),
            tool_call_count=current.tool_call_count + len(calls),
            observations=current.observations,
            evidence=current.evidence,
            goal_outcomes=current.goal_outcomes,
            decision=None,
        )
        for observation in observations:
            observation_current = replace(
                observation_current,
                observations=observation_current.observations + (observation,),
            )
            event = _emit(
                state,
                dependencies,
                "ToolObservationRecorded",
                {
                    "observation_id": observation.observation_id,
                    "call_id": observation.call_id,
                    "source": observation.source,
                    "target": observation.target,
                    "owner": asdict(observation.owner),
                    "transport_status": observation.transport_status.value,
                    "disposition": observation.disposition.value,
                    "trust": observation.trust,
                    "error_code": observation.error_code,
                    "provenance": asdict(observation.provenance)
                    if observation.provenance is not None
                    else None,
                    "data": observation.data,
                },
                current=observation_current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        current = observation_current
        evidence_current = current
        for record in created:
            evidence_current = replace(
                evidence_current,
                evidence=type(current.evidence)(
                    owner=current.identity,
                    records=evidence_current.evidence.records + (record,),
                ),
            )
            event = _emit(
                state,
                dependencies,
                "EvidenceRecorded",
                {
                    "evidence_id": record.evidence_id,
                    "kind": record.kind.value,
                    "target": record.target,
                    "observation_id": record.observation_id,
                    "owner": asdict(record.owner),
                    "provenance": asdict(record.provenance),
                    "freshness": asdict(record.freshness),
                },
                current=evidence_current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        current = evidence_current
        outcome_current = current
        for outcome in outcomes.values():
            goal_outcomes = dict(outcome_current.goal_outcomes)
            goal_outcomes[outcome.goal_id] = outcome
            outcome_current = replace(outcome_current, goal_outcomes=goal_outcomes)
            event = _emit(
                state,
                dependencies,
                "GoalOutcomeUpdated",
                {
                    "goal_id": outcome.goal_id,
                    "status": outcome.status.value,
                    "evidence_refs": list(outcome.evidence_refs),
                },
                current=outcome_current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        current = outcome_current
        result = {"current_request": current, "last_event_id": last_event_id}
        _remember(dependencies, state, result)
        return result

    async def write_guard_node(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        decision = current.decision
        if dependencies.write_guard is None or not isinstance(decision, SingleToolCall) or not isinstance(decision.call, AcceptedWriteCall):
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "write_guard_received_invalid_decision"},
                    message_template="The write guard received an invalid proposal.",
                ),
            )
        admission = dependencies.write_guard.assess(
            decision.call,
            current.goal_descriptor,
            current.completion_contract,
            current.goal_outcomes,
        )
        last_event_id = state.get("last_event_id")
        if admission.outcome is WriteAdmissionOutcome.INVALID:
            observation = _write_observation(
                current,
                decision.call.call_id,
                "write_guard",
                "INVALID_WRITE_PROPOSAL",
                admission.reason or "invalid write proposal",
                ObservationDisposition.WRITE_GUARD_REJECTED,
            )
            updated = replace(
                current,
                decision=None,
                observations=current.observations + (observation,),
                gate_feedback=(f"WRITE_GUARD_INVALID: {admission.reason or 'invalid'}",),
            )
            event = _emit(
                state, dependencies, "ToolObservationRecorded",
                _observation_event_payload(observation), current=updated, causation_id=last_event_id,
            )
            result = {"current_request": updated, "last_event_id": event.event_id}
            _remember(dependencies, state, result)
            return result
        if admission.outcome is WriteAdmissionOutcome.DENIED:
            outcomes = dict(current.goal_outcomes)
            for goal_id in admission.bound_goal_ids:
                outcomes[goal_id] = GoalOutcome(
                    goal_id,
                    status=GoalStatus.DENIED,
                    reason_code="POLICY_DENIED_WRITE",
                )
            observation = _write_observation(
                current,
                decision.call.call_id,
                "write_guard",
                "POLICY_DENIED_WRITE",
                admission.reason or "write denied by deterministic policy",
                ObservationDisposition.WRITE_RESOLUTION,
            )
            updated = replace(
                current,
                decision=None,
                observations=current.observations + (observation,),
                goal_outcomes=outcomes,
                gate_feedback=("POLICY_DENIED_WRITE",),
            )
            event = _emit(
                state, dependencies, "WriteDenied",
                {"call_id": decision.call.call_id, "reason": admission.reason, "bound_goal_ids": admission.bound_goal_ids},
                current=updated, causation_id=last_event_id,
            )
            obs_event = _emit(
                state, dependencies, "ToolObservationRecorded",
                _observation_event_payload(observation), current=updated, causation_id=event.event_id,
            )
            result = {"current_request": updated, "last_event_id": obs_event.event_id}
            _remember(dependencies, state, result)
            return result

        transaction = admission.transaction
        if transaction is None:
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "approval_required_without_transaction"},
                    message_template="The runtime could not prepare a write transaction.",
                ),
            )
        risk = dependencies.read_runtime.registry.spec(transaction.proposal.tool_name).risk.value
        pending = ApprovalInterrupt.from_transaction(transaction, risk)
        updated = replace(
            current,
            decision=None,
            write_transaction=transaction,
            pending_interrupt=pending,
            gate_feedback=(),
        )
        prepared = _emit(
            state, dependencies, "WriteTransactionPrepared",
            {"transaction": transaction.audit_projection()},
            current=updated, causation_id=last_event_id,
        )
        requested = _emit(
            state, dependencies, "ApprovalRequested",
            {
                "transaction_id": transaction.transaction_id,
                "approval_request_id": transaction.approval_request_id,
                "fingerprint": transaction.fingerprint,
                "tool_name": transaction.proposal.tool_name,
                "bound_goal_ids": transaction.bound_goal_ids,
            },
            current=updated, causation_id=prepared.event_id,
        )
        result = {"current_request": updated, "last_event_id": requested.event_id}
        _remember(dependencies, state, result)
        return result

    async def approval_node(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        transaction = current.write_transaction
        pending = current.pending_interrupt
        if transaction is None or not isinstance(pending, ApprovalInterrupt):
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "approval_node_missing_transaction"},
                    message_template="The approval boundary is inconsistent.",
                ),
            )
        resume_raw = current.resume_input
        if resume_raw is None:
            from langgraph.types import interrupt
            resume_raw = interrupt({
                "approval_request_id": pending.approval_request_id,
                "transaction_id": pending.transaction_id,
                "fingerprint": pending.fingerprint,
                "tool_name": pending.tool_name,
                "arguments": pending.arguments,
                "bound_goal_ids": pending.bound_goal_ids,
                "risk": pending.risk,
            })
        if isinstance(resume_raw, ResumeInput):
            resume_input = resume_raw
        elif isinstance(resume_raw, dict):
            resume_input = ResumeInput(**resume_raw)
        else:
            raise ValueError("resume payload must be ResumeInput or mapping")
        candidate = ApprovalValidator.validate_resume(
            transaction,
            pending,
            resume_input,
            operator_id=dependencies.operator_id,
            trust_domain=dependencies.trust_domain,
        )
        if dependencies.approval_store is None:
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "approval_store_missing"},
                    message_template="The approval authority is unavailable.",
                ),
            )
        last_event_id = state.get("last_event_id")
        recorder = getattr(dependencies.approval_store, "record_with_event", None)
        if callable(recorder):
            approval, event = recorder(
                dependencies.event_store,
                candidate,
                request_id=current.identity.request_id,
                thread_id=current.identity.thread_id,
                provenance=_event_provenance_for(state, dependencies, current=current),
                causation_id=last_event_id,
            )
        else:
            approval = dependencies.approval_store.record(candidate)
            event = None

        if approval.decision is ApprovalDecision.REJECT:
            transaction = transaction.transition(
                WriteTransactionStatus.REJECTED,
                approval=approval,
            )
            outcomes = dict(current.goal_outcomes)
            for goal_id in transaction.bound_goal_ids:
                outcomes[goal_id] = GoalOutcome(
                    goal_id,
                    status=GoalStatus.REJECTED,
                    reason_code="USER_REJECTED_WRITE",
                    write_transaction_id=transaction.transaction_id,
                )
            updated = replace(
                current,
                write_transaction=transaction,
                pending_interrupt=None,
                resume_input=None,
                goal_outcomes=outcomes,
                gate_feedback=("USER_REJECTED_WRITE",),
            )
            if event is None:
                event = _emit(
                    state, dependencies, "ApprovalRejected",
                    {
                        "approval_id": approval.approval_id,
                        "approval_request_id": approval.approval_request_id,
                        "transaction_id": approval.transaction_id,
                        "fingerprint": approval.fingerprint,
                        "operator_id": approval.operator_id,
                        "trust_domain": approval.trust_domain,
                        "decision": approval.decision.value,
                    },
                    current=updated,
                    causation_id=last_event_id,
                    event_id=f"evt_approval_{approval.approval_request_id}",
                )
            else:
                _persist_projection_after_atomic_event(
                    dependencies, state, updated, event.event_id
                )
            result = {"current_request": updated, "last_event_id": event.event_id}
            _remember(dependencies, state, result)
            return result

        transaction = transaction.transition(
            WriteTransactionStatus.APPROVED,
            approval=approval,
        )
        updated = replace(current, write_transaction=transaction, pending_interrupt=None, resume_input=None)
        if event is None:
            event = _emit(
                state, dependencies, "ApprovalGranted",
                {
                    "approval_id": approval.approval_id,
                    "approval_request_id": approval.approval_request_id,
                    "transaction_id": approval.transaction_id,
                    "fingerprint": approval.fingerprint,
                    "operator_id": approval.operator_id,
                    "trust_domain": approval.trust_domain,
                    "decision": approval.decision.value,
                },
                current=updated,
                causation_id=last_event_id,
                event_id=f"evt_approval_{approval.approval_request_id}",
            )
        else:
            _persist_projection_after_atomic_event(
                dependencies, state, updated, event.event_id
            )
        result = {"current_request": updated, "last_event_id": event.event_id}
        _remember(dependencies, state, result)
        return result

    async def revalidate_write(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        transaction = current.write_transaction
        if dependencies.write_guard is None or transaction is None or transaction.status is not WriteTransactionStatus.APPROVED:
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "revalidate_without_approved_transaction"},
                    message_template="The write transaction could not be revalidated.",
                ),
            )
        if not dependencies.write_guard.compatible(transaction, current.goal_descriptor, current.completion_contract):
            invalidated = transaction.transition(WriteTransactionStatus.INVALIDATED_GOAL_CHANGED)
            updated = replace(current, write_transaction=invalidated, gate_feedback=("WRITE_INVALIDATED_GOAL_CHANGED",))
            event = _emit(
                state, dependencies, "WriteTransactionInvalidated",
                {"transaction_id": transaction.transaction_id, "reason": "GOAL_CHANGED"},
                current=updated,
            )
            result = {"current_request": updated, "last_event_id": event.event_id}
            _remember(dependencies, state, result)
            return result
        transaction = transaction.transition(WriteTransactionStatus.REVALIDATING)
        base_current = replace(current, write_transaction=transaction)
        if transaction.precondition is None or not dependencies.write_guard.preconditions.matches(transaction.precondition, transaction.proposal):
            new_evidence, invalidated_refs = dependencies.evidence_tracker.invalidate_entities(
                current.evidence, transaction.affected_entities
            )
            invalidated = transaction.transition(WriteTransactionStatus.INVALIDATED)
            updated = replace(
                current,
                write_transaction=invalidated,
                evidence=new_evidence,
                gate_feedback=("WRITE_PRECONDITION_CHANGED",),
            )
            event = _emit(
                state, dependencies, "WriteTransactionInvalidated",
                {"transaction_id": transaction.transaction_id, "reason": "PRECONDITION_CHANGED"},
                current=updated,
            )
            last_event_id = event.event_id
            if invalidated_refs:
                ev = _emit(
                    state, dependencies, "EvidenceInvalidated",
                    {"transaction_id": transaction.transaction_id, "evidence_refs": invalidated_refs},
                    current=updated, causation_id=last_event_id,
                )
                last_event_id = ev.event_id
            result = {"current_request": updated, "last_event_id": last_event_id}
            _remember(dependencies, state, result)
            return result
        event = _emit(
            state, dependencies, "WritePreconditionRevalidated",
            {"transaction_id": transaction.transaction_id, "precondition_fingerprint": transaction.precondition.fingerprint},
            current=base_current,
        )
        result = {"current_request": base_current, "last_event_id": event.event_id}
        _remember(dependencies, state, result)
        return result

    async def execution_claim_node(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        transaction = current.write_transaction
        if dependencies.claim_store is None or transaction is None or transaction.status is not WriteTransactionStatus.REVALIDATING or transaction.approval is None:
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "execution_claim_invalid_state"},
                    message_template="The execution claim could not be established.",
                ),
            )
        try:
            claimer = getattr(dependencies.claim_store, "claim_with_event", None)
            if callable(claimer):
                claim, event = claimer(
                    dependencies.event_store,
                    transaction,
                    transaction.approval,
                    request_id=current.identity.request_id,
                    thread_id=current.identity.thread_id,
                    provenance=_event_provenance_for(state, dependencies, current=current),
                    causation_id=state.get("last_event_id"),
                )
                transaction = transaction.transition(
                    WriteTransactionStatus.EXECUTING, execution_claim=claim
                )
                updated = replace(current, write_transaction=transaction)
                _persist_projection_after_atomic_event(
                    dependencies, state, updated, event.event_id
                )
            else:
                claim = dependencies.claim_store.claim(transaction, transaction.approval)
                transaction = transaction.transition(
                    WriteTransactionStatus.EXECUTING, execution_claim=claim
                )
                updated = replace(current, write_transaction=transaction)
                event = _emit(
                    state,
                    dependencies,
                    "ExecutionClaimed",
                    {
                        "transaction_id": transaction.transaction_id,
                        "claim_id": claim.claim_id,
                        "approval_id": claim.approval_id,
                    },
                    current=updated,
                    event_id=f"evt_claim_{transaction.transaction_id}",
                )
        except ExecutionClaimAlreadyExists:
            # A concurrent/stale resume already acquired the one claim. Do not
            # append a competing terminal event from this worker.
            raise
        result = {"current_request": updated, "last_event_id": event.event_id}
        _remember(dependencies, state, result)
        return result

    async def execute_write(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        transaction = current.write_transaction
        if dependencies.write_runtime is None or transaction is None or transaction.status is not WriteTransactionStatus.EXECUTING or transaction.execution_claim is None:
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "execute_without_claim"},
                    message_template="The mutation boundary is inconsistent.",
                ),
            )
        last_event_id = state.get("last_event_id")
        started_tx = transaction
        attempt_id: str | None = None
        consumer = getattr(dependencies.claim_store, "consume_attempt_with_event", None)
        if callable(consumer):
            active_token: tuple[str, str] | None = None
            try:
                # Register before the atomic durable MutationStarted boundary
                # so another in-process resume cannot mistake this live owner
                # for a crashed worker during the tiny commit window.
                active_mutations.begin(transaction.transaction_id, "pending")
                attempt_id, started_event = consumer(
                    dependencies.event_store,
                    transaction.execution_claim,
                    request_id=current.identity.request_id,
                    thread_id=current.identity.thread_id,
                    tool_name=transaction.proposal.tool_name,
                    fingerprint=transaction.fingerprint,
                    provenance=_event_provenance_for(state, dependencies, current=current),
                    causation_id=last_event_id,
                )
            except MutationAttemptAlreadyConsumed:
                # Another worker already crossed the unique mutation-start
                # boundary. This stale worker must not add a competing event.
                active_mutations.end(transaction.transaction_id, "pending")
                raise
            except BaseException:
                active_mutations.end(transaction.transaction_id, "pending")
                raise
            active_mutations.end(transaction.transaction_id, "pending")
            active_mutations.begin(transaction.transaction_id, attempt_id)
            active_token = (transaction.transaction_id, attempt_id)
            started_tx = transaction.transition(
                WriteTransactionStatus.EXECUTING,
                execution_attempt_id=attempt_id,
            )
            started_current = replace(current, write_transaction=started_tx)
            last_event_id = started_event.event_id
            try:
                _persist_projection_after_atomic_event(
                    dependencies, state, started_current, last_event_id
                )
            except BaseException:
                active_mutations.end(*active_token)
                raise
            transaction = started_tx
            current = started_current
            try:
                mutation = await dependencies.write_runtime.execute_started_once(
                    transaction, transaction.execution_claim, attempt_id
                )
            finally:
                active_mutations.end(*active_token)
        else:
            async def on_started(new_attempt_id: str):
                nonlocal last_event_id, started_tx
                started_tx = transaction.transition(
                    WriteTransactionStatus.EXECUTING,
                    execution_attempt_id=new_attempt_id,
                )
                started_current = replace(current, write_transaction=started_tx)
                event = _emit(
                    state, dependencies, "MutationStarted",
                    {
                        "transaction_id": transaction.transaction_id,
                        "execution_attempt_id": new_attempt_id,
                        "claim_id": transaction.execution_claim.claim_id,
                        "tool_name": transaction.proposal.tool_name,
                        "fingerprint": transaction.fingerprint,
                    },
                    current=started_current,
                    causation_id=last_event_id,
                    event_id=f"evt_mutation_started_{new_attempt_id}",
                )
                last_event_id = event.event_id
            active_mutations.begin(transaction.transaction_id, "pending")
            try:
                attempt_id, mutation = await dependencies.write_runtime.execute_once(
                    transaction, transaction.execution_claim, on_started=on_started
                )
            finally:
                active_mutations.end(transaction.transaction_id, "pending")
        if mutation.outcome is MutationOutcome.CONFIRMED_SUCCESS:
            status = WriteTransactionStatus.EXECUTED
        elif mutation.outcome is MutationOutcome.OUTCOME_UNKNOWN:
            status = WriteTransactionStatus.RECONCILIATION_REQUIRED
        else:
            status = WriteTransactionStatus.FAILED
        transaction = transaction.transition(
            status,
            execution_attempt_id=attempt_id,
            mutation_result=mutation,
        )
        evidence = current.evidence
        invalidated_refs: tuple[str, ...] = ()
        if mutation.outcome is not MutationOutcome.FAILED_BEFORE_EFFECT:
            evidence, invalidated_refs = dependencies.evidence_tracker.invalidate_entities(
                evidence, transaction.affected_entities
            )
        updated = replace(current, write_transaction=transaction, evidence=evidence)
        event = _emit(
            state, dependencies, "MutationResultRecorded",
            {
                "transaction_id": transaction.transaction_id,
                "execution_attempt_id": attempt_id,
                "outcome": mutation.outcome.value,
                "error_code": mutation.error_code,
            },
            current=updated, causation_id=last_event_id,
        )
        last_event_id = event.event_id
        if invalidated_refs:
            ev = _emit(
                state, dependencies, "EvidenceInvalidated",
                {"transaction_id": transaction.transaction_id, "evidence_refs": invalidated_refs},
                current=updated, causation_id=last_event_id,
            )
            last_event_id = ev.event_id
        if mutation.outcome is MutationOutcome.OUTCOME_UNKNOWN:
            spec = dependencies.read_runtime.registry.spec(transaction.proposal.tool_name)
            reconciliation_event = _emit(
                state, dependencies, "ReconciliationRequired",
                {
                    "transaction_id": transaction.transaction_id,
                    "execution_attempt_id": attempt_id,
                    "tool_name": transaction.proposal.tool_name,
                    "target": transaction.affected_entities[0],
                    "idempotency": spec.idempotency.value,
                },
                current=updated, causation_id=last_event_id,
            )
            last_event_id = reconciliation_event.event_id
            block_event = _emit(
                state, dependencies, "WriteReplayBlocked",
                {
                    "transaction_id": transaction.transaction_id,
                    "tool_name": transaction.proposal.tool_name,
                    "target": transaction.affected_entities[0],
                    "reason": "OUTCOME_UNKNOWN",
                    "idempotency": spec.idempotency.value,
                },
                current=updated, causation_id=last_event_id,
            )
            last_event_id = block_event.event_id
            tx_state = dict(state)
            tx_state.update({"current_request": updated, "last_event_id": last_event_id})
            return _terminal_update(
                tx_state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.REQUIRES_RECONCILIATION,
                    safe_facts={"transaction_id": transaction.transaction_id, "execution_attempt_id": attempt_id},
                    message_template="The mutation outcome is unknown. Reconciliation is required before any future write.",
                    human_action_required=True,
                ),
                causation_id=last_event_id,
            )
        if status is WriteTransactionStatus.FAILED:
            outcomes = dict(updated.goal_outcomes)
            for goal_id in transaction.bound_goal_ids:
                outcomes[goal_id] = GoalOutcome(
                    goal_id,
                    status=GoalStatus.PENDING,
                    reason_code="WRITE_ATTEMPT_FAILED_NEW_APPROVAL_REQUIRED",
                    write_transaction_id=transaction.transaction_id,
                )
            updated = replace(updated, goal_outcomes=outcomes, gate_feedback=("WRITE_ATTEMPT_FAILED_NEW_APPROVAL_REQUIRED",))
        result = {"current_request": updated, "last_event_id": last_event_id}
        _remember(dependencies, state, result)
        return result

    async def action_verify_node(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        transaction = current.write_transaction
        if dependencies.action_verifier is None or transaction is None or transaction.status is not WriteTransactionStatus.EXECUTED:
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "action_verifier_invalid_state"},
                    message_template="The action verifier received invalid state.",
                ),
            )
        transaction = transaction.transition(WriteTransactionStatus.VERIFYING)
        verification = dependencies.action_verifier.verify(transaction)
        transaction = transaction.transition(WriteTransactionStatus.VERIFYING, action_verification=verification)
        updated = replace(current, write_transaction=transaction)
        event = _emit(
            state, dependencies, "ActionVerificationRecorded",
            {"transaction_id": transaction.transaction_id, "status": verification.status.value, "reason_code": verification.reason_code},
            current=updated,
        )
        last_event_id = event.event_id
        if verification.status is not VerificationStatus.VERIFIED:
            transaction = transaction.transition(WriteTransactionStatus.VERIFICATION_FAILED)
            outcomes = dict(updated.goal_outcomes)
            for goal_id in transaction.bound_goal_ids:
                outcomes[goal_id] = GoalOutcome(
                    goal_id,
                    status=GoalStatus.INCONCLUSIVE,
                    reason_code="ACTION_VERIFICATION_FAILED",
                    write_transaction_id=transaction.transaction_id,
                )
            updated = replace(updated, write_transaction=transaction, goal_outcomes=outcomes)
            result = {"current_request": updated, "last_event_id": last_event_id}
            _remember(dependencies, state, result)
            return result
        evidence, record = dependencies.evidence_tracker.record_verification(
            updated.evidence,
            kind=EvidenceKind.ACTION_VERIFIED,
            target=verification.target,
            source="action_verifier",
            observation_id=f"action_{uuid.uuid4().hex}",
            owner=updated.identity,
        )
        updated = replace(updated, evidence=evidence)
        ev = _emit(
            state, dependencies, "EvidenceRecorded",
            {"evidence_id": record.evidence_id, "kind": record.kind.value, "target": record.target, "observation_id": record.observation_id, "owner": asdict(record.owner), "provenance": asdict(record.provenance), "freshness": asdict(record.freshness)},
            current=updated, causation_id=last_event_id,
        )
        last_event_id = ev.event_id
        spec = dependencies.read_runtime.registry.spec(transaction.proposal.tool_name)
        if spec.verification == "ACTION":
            transaction = transaction.transition(WriteTransactionStatus.VERIFIED)
            outcomes = dependencies.evidence_tracker.refresh_goal_outcomes(
                updated.goal_descriptor, updated.completion_contract, updated.evidence, updated.goal_outcomes
            )
            updated = replace(updated, write_transaction=transaction, goal_outcomes=outcomes)
        result = {"current_request": updated, "last_event_id": last_event_id}
        _remember(dependencies, state, result)
        return result

    async def operational_goal_verify_node(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        transaction = current.write_transaction
        if dependencies.operational_goal_verifier is None or transaction is None or transaction.status is not WriteTransactionStatus.VERIFYING or transaction.action_verification is None:
            return _terminal_update(
                state, dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "operational_verifier_invalid_state"},
                    message_template="The operational verifier received invalid state.",
                ),
            )
        verification = dependencies.operational_goal_verifier.verify(transaction)
        transaction = transaction.transition(
            WriteTransactionStatus.VERIFYING,
            operational_goal_verification=verification,
        )
        updated = replace(current, write_transaction=transaction)
        event = _emit(
            state, dependencies, "OperationalGoalVerificationRecorded",
            {"transaction_id": transaction.transaction_id, "status": verification.status.value, "reason_code": verification.reason_code},
            current=updated,
        )
        last_event_id = event.event_id
        if verification.status is not VerificationStatus.VERIFIED:
            transaction = transaction.transition(WriteTransactionStatus.VERIFICATION_FAILED)
            outcomes = dict(updated.goal_outcomes)
            for goal_id in transaction.bound_goal_ids:
                outcomes[goal_id] = GoalOutcome(
                    goal_id,
                    status=GoalStatus.INCONCLUSIVE,
                    reason_code="OPERATIONAL_GOAL_VERIFICATION_FAILED",
                    write_transaction_id=transaction.transaction_id,
                )
            updated = replace(updated, write_transaction=transaction, goal_outcomes=outcomes)
            result = {"current_request": updated, "last_event_id": last_event_id}
            _remember(dependencies, state, result)
            return result
        evidence, record = dependencies.evidence_tracker.record_verification(
            updated.evidence,
            kind=EvidenceKind.OPERATIONAL_GOAL_VERIFIED,
            target=verification.target,
            source="operational_goal_verifier",
            observation_id=f"goal_{uuid.uuid4().hex}",
            owner=updated.identity,
        )
        transaction = transaction.transition(WriteTransactionStatus.VERIFIED)
        outcomes = dependencies.evidence_tracker.refresh_goal_outcomes(
            updated.goal_descriptor, updated.completion_contract, evidence, updated.goal_outcomes
        )
        updated = replace(updated, write_transaction=transaction, evidence=evidence, goal_outcomes=outcomes)
        ev = _emit(
            state, dependencies, "EvidenceRecorded",
            {"evidence_id": record.evidence_id, "kind": record.kind.value, "target": record.target, "observation_id": record.observation_id, "owner": asdict(record.owner), "provenance": asdict(record.provenance), "freshness": asdict(record.freshness)},
            current=updated, causation_id=last_event_id,
        )
        result = {"current_request": updated, "last_event_id": ev.event_id}
        _remember(dependencies, state, result)
        return result

    async def response_completion_gate(state: AgentState) -> dict[str, Any]:
        current = state["current_request"]
        candidate = current.final_candidate
        descriptor = current.goal_descriptor
        contract = current.completion_contract
        if not isinstance(candidate, FinalCandidate) or descriptor is None or contract is None:
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "completion_gate_missing_structured_input"},
                    message_template="The completion gate could not evaluate the Agent candidate.",
                ),
            )
        evaluation = dependencies.completion_gate.evaluate(
            candidate, descriptor, contract, current.evidence, current.goal_outcomes
        )
        last_event_id = state.get("last_event_id")
        rejected_budget = current.budgets.with_gate_rejection() if not evaluation.passed else current.budgets
        gate_current = replace(
            current,
            gate_passed=evaluation.passed,
            budgets=rejected_budget,
            gate_feedback=tuple(evaluation.facts + evaluation.missing) if not evaluation.passed else current.gate_feedback,
        )
        gate_event = _emit(
            state,
            dependencies,
            "CompletionGateEvaluated",
            {
                "passed": evaluation.passed,
                "facts": list(evaluation.facts),
                "missing": list(evaluation.missing),
                "referenced_goal_ids": list(candidate.referenced_goal_ids),
            },
            current=gate_current,
            causation_id=last_event_id,
        )
        last_event_id = gate_event.event_id
        current = gate_current
        outcome_current = current
        for outcome in evaluation.goal_outcomes.values():
            goal_outcomes = dict(outcome_current.goal_outcomes)
            goal_outcomes[outcome.goal_id] = outcome
            outcome_current = replace(outcome_current, goal_outcomes=goal_outcomes)
            event = _emit(
                state,
                dependencies,
                "GoalOutcomeUpdated",
                {
                    "goal_id": outcome.goal_id,
                    "status": outcome.status.value,
                    "evidence_refs": list(outcome.evidence_refs),
                },
                current=outcome_current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        current = outcome_current
        if evaluation.passed:
            result = {"current_request": current, "last_event_id": last_event_id}
            _remember(dependencies, state, result)
            return result
        budget = current.budgets
        if budget.completion_gate_rejections > budget.limits.max_completion_gate_rejections:
            return _terminal_update(
                state,
                dependencies,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={
                        "completion_gate_rejections": budget.completion_gate_rejections,
                        "max_completion_gate_rejections": budget.limits.max_completion_gate_rejections,
                    },
                    message_template="The bounded execution budget was exhausted before all goals could be completed.",
                ),
                budgets=budget,
                causation_id=last_event_id,
            )
        result = {"current_request": current, "last_event_id": last_event_id}
        _remember(dependencies, state, result)
        return result

    def after_agent(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        if current.continue_after_read_guard:
            return "agent"
        if isinstance(current.decision, FinalCandidate):
            return "response_completion_gate"
        if isinstance(current.decision, ReadToolBatch):
            return "read_executor"
        if isinstance(current.decision, SingleToolCall):
            return "write_guard" if isinstance(current.decision.call, AcceptedWriteCall) else "read_executor"
        return "terminal"

    def after_write_guard(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        transaction = current.write_transaction
        if transaction is not None and transaction.status is WriteTransactionStatus.PENDING_APPROVAL:
            return "approval"
        return "agent"

    def after_approval(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        transaction = current.write_transaction
        if transaction is None:
            return "terminal"
        if transaction.status is WriteTransactionStatus.APPROVED:
            return "revalidate_write"
        if transaction.status is WriteTransactionStatus.REJECTED:
            return "agent"
        return "terminal"

    def after_revalidation(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        transaction = current.write_transaction
        if transaction is not None and transaction.status is WriteTransactionStatus.REVALIDATING:
            return "execution_claim"
        return "agent"

    def after_claim(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        transaction = current.write_transaction
        if transaction is not None and transaction.status is WriteTransactionStatus.EXECUTING:
            return "execute_write"
        return "terminal"

    def after_execute(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        transaction = current.write_transaction
        if transaction is None:
            return "terminal"
        if transaction.status is WriteTransactionStatus.EXECUTED:
            return "action_verify"
        if transaction.status is WriteTransactionStatus.FAILED:
            return "agent"
        return "terminal"

    def after_action_verify(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        transaction = current.write_transaction
        if transaction is None:
            return "terminal"
        if transaction.status is WriteTransactionStatus.VERIFYING:
            return "operational_goal_verify"
        return "agent"

    def after_operational_verify(state: AgentState) -> str:
        current = state["current_request"]
        return "terminal" if current.terminal_state is not None else "agent"

    def after_gate(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        return "end" if current.gate_passed else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("read_executor", read_executor)
    builder.add_node("write_guard", write_guard_node)
    builder.add_node("approval", approval_node)
    builder.add_node("revalidate_write", revalidate_write)
    builder.add_node("execution_claim", execution_claim_node)
    builder.add_node("execute_write", execute_write)
    builder.add_node("action_verify", action_verify_node)
    builder.add_node("operational_goal_verify", operational_goal_verify_node)
    builder.add_node("response_completion_gate", response_completion_gate)
    builder.add_edge(START, entry_node or ("approval" if resume_entry else "agent"))
    builder.add_conditional_edges(
        "agent",
        after_agent,
        {
            "agent": "agent",
            "read_executor": "read_executor",
            "write_guard": "write_guard",
            "response_completion_gate": "response_completion_gate",
            "terminal": END,
        },
    )
    builder.add_edge("read_executor", "agent")
    builder.add_conditional_edges(
        "write_guard",
        after_write_guard,
        {"approval": "approval", "agent": "agent", "terminal": END},
    )
    builder.add_conditional_edges(
        "approval",
        after_approval,
        {"revalidate_write": "revalidate_write", "agent": "agent", "terminal": END},
    )
    builder.add_conditional_edges(
        "revalidate_write",
        after_revalidation,
        {"execution_claim": "execution_claim", "agent": "agent", "terminal": END},
    )
    builder.add_conditional_edges(
        "execution_claim",
        after_claim,
        {"execute_write": "execute_write", "terminal": END},
    )
    builder.add_conditional_edges(
        "execute_write",
        after_execute,
        {"action_verify": "action_verify", "agent": "agent", "terminal": END},
    )
    builder.add_conditional_edges(
        "action_verify",
        after_action_verify,
        {"operational_goal_verify": "operational_goal_verify", "agent": "agent", "terminal": END},
    )
    builder.add_conditional_edges(
        "operational_goal_verify",
        after_operational_verify,
        {"agent": "agent", "terminal": END},
    )
    builder.add_conditional_edges(
        "response_completion_gate",
        after_gate,
        {"agent": "agent", "end": END, "terminal": END},
    )
    return builder.compile(checkpointer=checkpointer)


def _accept_decision(
    decision: AgentDecision,
    runtime: ReadToolRuntime,
    write_guard: WriteGuard | None,
    max_batch: int,
) -> AgentDecision:
    """Turn one provider proposal into the immutable Runtime decision."""
    if isinstance(decision, SingleToolCall):
        spec = runtime.registry.spec(decision.call.tool_name)
        if spec.kind is ToolKind.READ:
            accepted = runtime.validate_single(decision.call)
        else:
            if write_guard is None:
                raise ValueError("WRITE path is not configured")
            accepted = write_guard.normalize(decision.call)
        return SingleToolCall(accepted, decision.proposed_goal_descriptor)
    if isinstance(decision, ReadToolBatch):
        accepted = runtime.validate_batch(decision, max_batch)
        return ReadToolBatch(accepted.calls, decision.proposed_goal_descriptor)
    if isinstance(decision, FinalCandidate):
        return decision
    raise ValueError("AgentDecision is not SingleToolCall, ReadToolBatch, or FinalCandidate")


def _decision_audit_payload(decision: AgentDecision | None) -> dict[str, Any]:
    """Serialize only the Runtime-accepted decision representation."""

    if isinstance(decision, SingleToolCall):
        call = decision.call
        return {
            "calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "arguments_fingerprint": canonical_tool_call_fingerprint(
                        call.tool_name, call.arguments
                    ),
                }
            ]
        }
    if isinstance(decision, ReadToolBatch):
        return {
            "calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "arguments_fingerprint": canonical_tool_call_fingerprint(
                        call.tool_name, call.arguments
                    ),
                }
                for call in decision.calls
            ]
        }
    if isinstance(decision, FinalCandidate):
        return {"referenced_goal_ids": list(decision.referenced_goal_ids)}
    return {}


def _write_observation(
    current: CurrentRequestContext,
    call_id: str,
    source: str,
    error_code: str,
    reason: str,
    disposition: ObservationDisposition,
) -> ToolObservation:
    """Create a Runtime-owned non-evidence observation for a WRITE resolution."""
    return ToolObservation(
        observation_id=f"obs_write_{uuid.uuid4().hex}",
        call_id=call_id,
        owner=current.identity,
        source=source,
        target="platform",
        transport_status=TransportStatus.ERROR,
        disposition=disposition,
        data={"error_code": error_code, "reason": reason},
        trust="RUNTIME_STRUCTURED",
        error_code=error_code,
        observed_at=datetime.now(timezone.utc),
        provenance=ObservationProvenance(
            source_tool=source,
            arguments_fingerprint="",
            requested_scope=ObservationScope(ScopeKind.PLATFORM),
            observed_scope=ObservationScope(ScopeKind.UNKNOWN),
            requested_identity=None,
            observed_identity=None,
            identity_status=IdentityStatus.NOT_APPLICABLE,
            scope_status=ScopeStatus.UNKNOWN,
        ),
    )


def _observation_event_payload(observation: ToolObservation) -> dict[str, Any]:
    return {
        "observation_id": observation.observation_id,
        "call_id": observation.call_id,
        "source": observation.source,
        "owner": asdict(observation.owner),
        "disposition": observation.disposition.value,
        "trust": observation.trust,
        "error_code": observation.error_code,
        "data": observation.data,
    }


def _read_guard_update(
    state: AgentState,
    dependencies: GraphDependencies,
    current: CurrentRequestContext,
    reason: str,
    budgets: BudgetState,
    last_event_id: str | None = None,
) -> dict[str, Any]:
    observation = ToolObservation(
        observation_id=f"obs_guard_{uuid.uuid4().hex}",
        call_id="read-guard",
        owner=current.identity,
        source="read_guard",
        target="platform",
        transport_status=TransportStatus.ERROR,
        disposition=ObservationDisposition.READ_GUARD_REJECTED,
        data={"error_code": "INVALID_READ_DECISION", "reason": reason},
        trust="RUNTIME_STRUCTURED",
        error_code="INVALID_READ_DECISION",
        observed_at=datetime.now(timezone.utc),
        provenance=ObservationProvenance(
            source_tool="read_guard",
            arguments_fingerprint="",
            requested_scope=ObservationScope(ScopeKind.PLATFORM),
            observed_scope=ObservationScope(ScopeKind.UNKNOWN),
            requested_identity=None,
            observed_identity=None,
            identity_status=IdentityStatus.NOT_APPLICABLE,
            scope_status=ScopeStatus.UNKNOWN,
        ),
    )
    updated = replace(
        current,
        budgets=budgets,
        decision=None,
        final_candidate=None,
        observations=current.observations + (observation,),
        gate_feedback=(f"READ_GUARD_REJECTED: {reason}",),
        continue_after_read_guard=True,
    )
    event = _emit(
        state,
        dependencies,
        "ToolObservationRecorded",
        {
            "observation_id": observation.observation_id,
            "call_id": observation.call_id,
            "source": observation.source,
            "owner": asdict(observation.owner),
            "disposition": observation.disposition.value,
            "trust": observation.trust,
            "error_code": observation.error_code,
            "data": observation.data,
        },
        current=updated,
        causation_id=last_event_id or state.get("last_event_id"),
    )
    return {"current_request": updated, "last_event_id": event.event_id}


def _agent_decision_rejection_update(
    state: AgentState,
    dependencies: GraphDependencies,
    current: CurrentRequestContext,
    *,
    reason: str,
    budgets: BudgetState,
    proposal_type: str,
) -> dict[str, Any]:
    """Record malformed provider structure as a bounded recoverable observation."""

    observation = ToolObservation(
        observation_id=f"obs_decision_{uuid.uuid4().hex}",
        call_id="agent-decision-ingress",
        owner=current.identity,
        source="agent_decision_ingress",
        target="platform",
        transport_status=TransportStatus.ERROR,
        disposition=ObservationDisposition.AGENT_DECISION_REJECTED,
        data={
            "error_code": "MALFORMED_AGENT_DECISION",
            "proposal_type": proposal_type,
            "reason": reason,
        },
        trust="RUNTIME_STRUCTURED",
        error_code="MALFORMED_AGENT_DECISION",
        observed_at=datetime.now(timezone.utc),
        provenance=ObservationProvenance(
            source_tool="agent_decision_ingress",
            arguments_fingerprint="",
            requested_scope=ObservationScope(ScopeKind.PLATFORM),
            observed_scope=ObservationScope(ScopeKind.UNKNOWN),
            requested_identity=None,
            observed_identity=None,
            identity_status=IdentityStatus.NOT_APPLICABLE,
            scope_status=ScopeStatus.UNKNOWN,
        ),
    )
    updated = replace(
        current,
        budgets=budgets,
        step_count=current.step_count + 1,
        decision=None,
        final_candidate=None,
        observations=current.observations + (observation,),
        gate_feedback=(f"AGENT_DECISION_REJECTED: {reason}",),
        continue_after_read_guard=True,
    )
    rejection_event = _emit(
        state,
        dependencies,
        "AgentDecisionRejected",
        {
            "proposal_type": proposal_type,
            "reason": reason,
            "error_code": "MALFORMED_AGENT_DECISION",
        },
        current=updated,
        causation_id=state.get("last_event_id"),
    )
    observation_event = _emit(
        state,
        dependencies,
        "ToolObservationRecorded",
        {
            "observation_id": observation.observation_id,
            "call_id": observation.call_id,
            "source": observation.source,
            "owner": asdict(observation.owner),
            "disposition": observation.disposition.value,
            "trust": observation.trust,
            "error_code": observation.error_code,
            "data": observation.data,
        },
        current=updated,
        causation_id=rejection_event.event_id,
    )
    return {"current_request": updated, "last_event_id": observation_event.event_id}


def _event_provenance_for(
    state: AgentState,
    dependencies: GraphDependencies,
    *,
    current: CurrentRequestContext | None = None,
) -> EventProvenance:
    active = current or state["current_request"]
    snapshot: OperatingPrinciplesSnapshot = active.operating_principles_snapshot
    return EventProvenance(
        model_version=dependencies.model_version,
        prompt_version=dependencies.prompt_version,
        tool_catalog_hash=dependencies.tool_catalog_hash,
        operating_principles_version=snapshot.version,
        operating_principles_hash=snapshot.content_hash,
        policy_version=dependencies.policy_version,
    )


def _persist_projection_after_atomic_event(
    dependencies: GraphDependencies,
    state: AgentState,
    current: CurrentRequestContext,
    event_id: str,
) -> None:
    """Catch the checkpoint projection up after an atomic capability+event commit.

    A hard crash before this projection save is safe: the capability and audit
    event are already one durable fact and host recovery can reconstruct the
    lagging checkpoint from that event.
    """
    saver = getattr(dependencies.runtime_checkpointer, "save_consistent", None)
    if not callable(saver):
        return
    candidate = dict(state)
    candidate["current_request"] = current
    candidate["last_event_id"] = event_id
    saver(dependencies.event_store, AgentState(**candidate))


def _emit(
    state: AgentState,
    dependencies: GraphDependencies,
    event_type: str,
    payload: dict[str, Any],
    *,
    current: CurrentRequestContext | None = None,
    causation_id: str | None = None,
    event_id: str | None = None,
):
    active = current or state["current_request"]
    provenance = _event_provenance_for(state, dependencies, current=active)
    parent = causation_id if causation_id is not None else state.get("last_event_id")
    appender = getattr(dependencies.event_store, "append_with_checkpoint", None)
    if callable(appender) and dependencies.runtime_checkpointer is not None:
        candidate = dict(state)
        candidate["current_request"] = active
        event = appender(
            dependencies.runtime_checkpointer,
            AgentState(**candidate),
            event_type=event_type,
            request_id=active.identity.request_id,
            thread_id=active.identity.thread_id,
            payload=payload,
            provenance=provenance,
            causation_id=parent,
            event_id=event_id,
        )
    else:
        event = dependencies.event_store.append(
            event_type=event_type,
            request_id=active.identity.request_id,
            thread_id=active.identity.thread_id,
            payload=payload,
            provenance=provenance,
            causation_id=parent,
            event_id=event_id,
        )
    # Record every durable event boundary, not only successful node returns.
    _remember(
        dependencies,
        state,
        {"current_request": active, "last_event_id": event.event_id},
    )
    return event


def _terminal_update(
    state: AgentState,
    dependencies: GraphDependencies,
    outcome: ControlledTerminalOutcome,
    *,
    budgets: BudgetState | None = None,
    causation_id: str | None = None,
) -> dict[str, Any]:
    current = state["current_request"]
    updated = replace(
        current,
        terminal_state=outcome,
        termination_reason=outcome.code.value,
        decision=None,
        final_candidate=None,
        budgets=budgets or current.budgets,
    )
    event = _emit(
        state,
        dependencies,
        "ControlledTerminalOutcomeProduced",
        {
            "code": outcome.code.value,
            "safe_facts": outcome.safe_facts,
            "message_template": outcome.message_template,
        },
        current=updated,
        causation_id=causation_id,
    )
    result = {"current_request": updated, "last_event_id": event.event_id}
    _remember(dependencies, state, result)
    return result


def _remember(dependencies: GraphDependencies, state: AgentState, updates: dict[str, Any]) -> None:
    if dependencies.latest_state_holder is None:
        return
    merged = dict(state)
    merged.update(updates)
    dependencies.latest_state_holder.record(merged)
