"""Visible Phase B LangGraph Agent loop with one current request owner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any
import uuid

from .budgets import BudgetState
from .context import ContextBudgetExceeded, ContextBuilder
from .contracts import CompletionContractCompiler
from .decisions import AgentDecision, FinalCandidate, ReadToolBatch, SingleToolCall
from .evidence import (
    EvidenceTracker,
    ObservationDisposition,
    ToolObservation,
    TransportStatus,
)
from .events import EventProvenance, EventStore
from .gate import ResponseCompletionGate
from .identity import RequestIdentity
from .outcomes import ControlledTerminalOutcome, GoalOutcome, TerminalCode
from .principles import OperatingPrinciplesSnapshot
from .provenance import (
    IdentityStatus,
    ObservationProvenance,
    ObservationScope,
    ScopeKind,
    ScopeStatus,
)
from .state import AgentState, CurrentRequestContext, LatestStateHolder
from ..providers.model import ProviderUnavailable
from ..tools.runtime import ReadToolRuntime


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


def build_graph(dependencies: GraphDependencies):
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
        try:
            decision: AgentDecision = await dependencies.provider.generate(context)
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
            current = replace(
                current,
                goal_descriptor=descriptor,
                completion_contract=contract,
                goal_outcomes={goal.goal_id: GoalOutcome(goal.goal_id) for goal in descriptor.goals},
            )
            event = _emit(
                state,
                dependencies,
                event_type,
                {"descriptor": descriptor.to_dict()},
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
            contract_event = _emit(
                state,
                dependencies,
                "CompletionContractCompiled",
                {
                    "descriptor_version": contract.descriptor_version,
                    "contract_version": contract.contract_version,
                    "contract_fingerprint": contract.contract_fingerprint,
                },
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = contract_event.event_id

        guard_reason = descriptor_error or _read_guard_reason(
            decision,
            dependencies.read_runtime,
            current.budgets.limits.max_parallel_read_batch,
        )
        if guard_reason is None and current.goal_descriptor is None and descriptor is None:
            guard_reason = "Agent did not declare a structured GoalDescriptor"
        decision_event = _emit(
            state,
            dependencies,
            "AgentDecisionMade",
            {
                "decision_kind": getattr(decision, "kind", type(decision).__name__),
                "accepted": guard_reason is None,
                "rejection_reason": guard_reason,
            },
            current=current,
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

        current = replace(
            current,
            decision=decision,
            final_candidate=decision if isinstance(decision, FinalCandidate) else None,
        )
        if isinstance(decision, FinalCandidate):
            final_event = _emit(
                state,
                dependencies,
                "FinalCandidateProduced",
                {
                    "referenced_goal_ids": list(decision.referenced_goal_ids),
                    "response_length": len(decision.response),
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
        guard_reason = (
            _batch_guard_reason(decision, dependencies.read_runtime, budget.limits.max_parallel_read_batch)
            if isinstance(decision, ReadToolBatch)
            else _single_guard_reason(decision, dependencies.read_runtime)
        )
        if guard_reason is not None:
            return _read_guard_update(state, dependencies, current, guard_reason, budget)

        last_event_id = state.get("last_event_id")

        async def on_started(call, attempt):
            nonlocal last_event_id
            event = _emit(
                state,
                dependencies,
                "ToolCallStarted",
                {"call_id": call.call_id, "tool_name": call.tool_name, "attempt": attempt},
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
        current = replace(
            current,
            budgets=budget.with_read_calls(len(calls)).with_retries(
                sum(item.retry_count for item in observations)
            ),
            tool_call_count=current.tool_call_count + len(calls),
            observations=current.observations + observations,
            evidence=new_evidence,
            goal_outcomes=outcomes,
            decision=None,
        )
        for observation in observations:
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
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        for record in created:
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
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        for outcome in outcomes.values():
            event = _emit(
                state,
                dependencies,
                "GoalOutcomeUpdated",
                {
                    "goal_id": outcome.goal_id,
                    "status": outcome.status.value,
                    "evidence_refs": list(outcome.evidence_refs),
                },
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        result = {"current_request": current, "last_event_id": last_event_id}
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
            current=current,
            causation_id=last_event_id,
        )
        last_event_id = gate_event.event_id
        current = replace(
            current,
            goal_outcomes=evaluation.goal_outcomes,
            gate_passed=evaluation.passed,
        )
        for outcome in evaluation.goal_outcomes.values():
            event = _emit(
                state,
                dependencies,
                "GoalOutcomeUpdated",
                {
                    "goal_id": outcome.goal_id,
                    "status": outcome.status.value,
                    "evidence_refs": list(outcome.evidence_refs),
                },
                current=current,
                causation_id=last_event_id,
            )
            last_event_id = event.event_id
        if evaluation.passed:
            result = {"current_request": current, "last_event_id": last_event_id}
            _remember(dependencies, state, result)
            return result
        budget = current.budgets.with_gate_rejection()
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
        current = replace(
            current,
            budgets=budget,
            gate_feedback=tuple(evaluation.facts + evaluation.missing),
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
        if isinstance(current.decision, (SingleToolCall, ReadToolBatch)):
            return "read_executor"
        return "terminal"

    def after_gate(state: AgentState) -> str:
        current = state["current_request"]
        if current.terminal_state is not None:
            return "terminal"
        return "end" if current.gate_passed else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("read_executor", read_executor)
    builder.add_node("response_completion_gate", response_completion_gate)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        after_agent,
        {
            "agent": "agent",
            "read_executor": "read_executor",
            "response_completion_gate": "response_completion_gate",
            "terminal": END,
        },
    )
    builder.add_edge("read_executor", "agent")
    builder.add_conditional_edges(
        "response_completion_gate",
        after_gate,
        {"agent": "agent", "end": END, "terminal": END},
    )
    return builder.compile()


def _read_guard_reason(decision: object, runtime: ReadToolRuntime, max_batch: int) -> str | None:
    if isinstance(decision, SingleToolCall):
        return _single_guard_reason(decision, runtime)
    if isinstance(decision, ReadToolBatch):
        return _batch_guard_reason(decision, runtime, max_batch)
    if isinstance(decision, FinalCandidate):
        return None
    return "AgentDecision is not SingleToolCall, ReadToolBatch, or FinalCandidate"


def _single_guard_reason(decision: SingleToolCall, runtime: ReadToolRuntime) -> str | None:
    try:
        runtime.validate_single(decision.call)
    except (TypeError, ValueError) as exc:
        return f"SINGLE_READ_GUARD_REJECTED: {type(exc).__name__}: {exc}"
    return None


def _batch_guard_reason(decision: ReadToolBatch, runtime: ReadToolRuntime, max_batch: int) -> str | None:
    try:
        runtime.validate_batch(decision, max_batch)
    except (TypeError, ValueError) as exc:
        return f"BATCH_READ_GUARD_REJECTED: {type(exc).__name__}: {exc}"
    return None


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
        current=current,
        causation_id=last_event_id or state.get("last_event_id"),
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
    return {"current_request": updated, "last_event_id": event.event_id}


def _emit(
    state: AgentState,
    dependencies: GraphDependencies,
    event_type: str,
    payload: dict[str, Any],
    *,
    current: CurrentRequestContext | None = None,
    causation_id: str | None = None,
):
    active = current or state["current_request"]
    snapshot: OperatingPrinciplesSnapshot = active.operating_principles_snapshot
    provenance = EventProvenance(
        model_version=dependencies.model_version,
        prompt_version=dependencies.prompt_version,
        tool_catalog_hash=dependencies.tool_catalog_hash,
        operating_principles_version=snapshot.version,
        operating_principles_hash=snapshot.content_hash,
        policy_version=dependencies.policy_version,
    )
    event = dependencies.event_store.append(
        event_type=event_type,
        request_id=active.identity.request_id,
        thread_id=active.identity.thread_id,
        payload=payload,
        provenance=provenance,
        causation_id=causation_id or state.get("last_event_id"),
    )
    # Record every durable event boundary, not only successful node returns.
    # If a later operation in the same node raises, the exception path can
    # continue from the latest state that was paired with the last durable
    # event instead of rolling back to the beginning of the node.
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
