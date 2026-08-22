"""The actual Phase B LangGraph loop.

The graph is intentionally short and visible:

    START -> agent -> read_executor -> agent
    agent -> response_completion_gate -> agent / END
    runtime terminal -> END

Each Agent node performs exactly one provider generation.  There is no hidden
model/tool loop inside a node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .budgets import BudgetState
from .context import ContextBuilder
from .contracts import CompletionContractCompiler
from .decisions import AgentDecision, FinalCandidate, ReadToolBatch, SingleToolCall
from .evidence import EvidenceTracker
from .events import EventProvenance, EventStore
from .gate import ResponseCompletionGate
from .outcomes import ControlledTerminalOutcome, GoalOutcome, GoalStatus, TerminalCode
from .principles import OperatingPrinciplesSnapshot
from .state import AgentState
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


def build_graph(dependencies: GraphDependencies):
    """Build and compile a real LangGraph StateGraph with explicit loop edges."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise RuntimeError(
            "LangGraph is required for the V2 explicit Agent loop; install project dependencies"
        ) from exc

    async def agent(state: AgentState) -> dict[str, Any]:
        if state.get("terminal_state") is not None:
            return {"decision": None}
        budget = state["budgets"]
        if not budget.has_agent_step():
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={
                        "agent_steps_used": budget.agent_steps_used,
                        "max_agent_steps": budget.limits.max_agent_steps,
                    },
                    message_template=(
                        "The bounded execution budget was exhausted before all goals could be completed."
                    ),
                ),
                dependencies,
            )

        budget = budget.with_agent_step()
        context = dependencies.context_builder.build(
            {**state, "budgets": budget}, state["operating_principles_snapshot"]
        )
        try:
            decision: AgentDecision = await dependencies.provider.generate(context)
        except ProviderUnavailable as exc:
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.PROVIDER_UNAVAILABLE,
                    safe_facts={"provider_error_type": type(exc).__name__},
                    message_template=(
                        "The configured provider is unavailable, so this interaction cannot safely continue."
                    ),
                ),
                dependencies,
                budgets=budget,
            )
        except Exception as exc:
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"provider_error_type": type(exc).__name__},
                    message_template="The Agent provider returned an unusable decision.",
                ),
                dependencies,
                budgets=budget,
            )

        updates: dict[str, Any] = {
            "budgets": budget,
            "step_count": state.get("step_count", 0) + 1,
            "decision": decision,
            "final_candidate": decision if isinstance(decision, FinalCandidate) else None,
            "gate_passed": None,
            "new_turn": False,
            "messages": list(state.get("messages", ()))
            + [
                {
                    "role": "assistant",
                    "kind": getattr(decision, "kind", type(decision).__name__),
                    "content": (
                        decision.response
                        if isinstance(decision, FinalCandidate)
                        else getattr(decision, "kind", type(decision).__name__)
                    ),
                    "candidate": isinstance(decision, FinalCandidate),
                }
            ],
        }
        descriptor = getattr(decision, "proposed_goal_descriptor", None)
        current_descriptor = state.get("goal_descriptor")
        if current_descriptor is None and descriptor is None:
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "agent_did_not_declare_goal_descriptor"},
                    message_template="The Agent did not declare a structured GoalDescriptor.",
                ),
                dependencies,
                budgets=budget,
            )
        if descriptor is not None and descriptor != current_descriptor:
            if current_descriptor is not None:
                if descriptor.descriptor_version != current_descriptor.descriptor_version + 1:
                    return _terminal_update(
                        state,
                        ControlledTerminalOutcome(
                            code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                            safe_facts={"reason": "non_monotonic_goal_descriptor_revision"},
                            message_template="The Agent proposed an invalid GoalDescriptor revision.",
                        ),
                        dependencies,
                        budgets=budget,
                    )
                descriptor_event = "GoalDescriptorRevised"
            else:
                descriptor_event = "GoalDescriptorDeclared"
            contract = dependencies.compiler.compile(descriptor)
            updates.update(
                {
                    "goal_descriptor": descriptor,
                    "goal_descriptor_version": descriptor.descriptor_version,
                    "completion_contract": contract,
                    "goal_outcomes": {
                        goal.goal_id: GoalOutcome(goal.goal_id)
                        for goal in descriptor.goals
                    },
                }
            )
            event = _emit(
                state,
                dependencies,
                descriptor_event,
                {"descriptor": descriptor.to_dict()},
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
                causation_id=event.event_id,
            )
            updates["last_event_id"] = contract_event.event_id
        decision_event = _emit(
            state,
            dependencies,
            "AgentDecisionMade",
            {"decision_kind": getattr(decision, "kind", type(decision).__name__)},
            causation_id=updates.get("last_event_id", state.get("last_event_id")),
        )
        updates["last_event_id"] = decision_event.event_id
        if isinstance(decision, FinalCandidate):
            final_event = _emit(
                state,
                dependencies,
                "FinalCandidateProduced",
                {
                    "referenced_goal_ids": list(decision.referenced_goal_ids),
                    "response_length": len(decision.response),
                },
                causation_id=decision_event.event_id,
            )
            updates["last_event_id"] = final_event.event_id
        return updates

    async def read_executor(state: AgentState) -> dict[str, Any]:
        decision = state.get("decision")
        if not isinstance(decision, (SingleToolCall, ReadToolBatch)):
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "read_executor_received_non_read_decision"},
                    message_template="The read runtime received an invalid AgentDecision.",
                ),
                dependencies,
            )
        calls = (decision.call,) if isinstance(decision, SingleToolCall) else decision.calls
        budget = state["budgets"]
        if not budget.has_read_calls(len(calls)):
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={
                        "read_tool_calls_used": budget.read_tool_calls_used,
                        "requested_calls": len(calls),
                        "max_read_tool_calls": budget.limits.max_read_tool_calls,
                    },
                    message_template=(
                        "The bounded execution budget was exhausted before all goals could be completed."
                    ),
                ),
                dependencies,
            )
        if isinstance(decision, ReadToolBatch):
            try:
                dependencies.read_runtime.validate_batch(
                    decision, budget.limits.max_parallel_read_batch
                )
            except Exception as exc:
                return _terminal_update(
                    state,
                    ControlledTerminalOutcome(
                        code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                        safe_facts={"reason": "invalid_read_batch", "error_type": type(exc).__name__},
                        message_template="The Agent emitted a structurally invalid READ batch.",
                    ),
                    dependencies,
                    budgets=budget,
                )

        async def on_started(call, attempt):
            _emit(
                state,
                dependencies,
                "ToolCallStarted",
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "attempt": attempt,
                },
            )

        if isinstance(decision, SingleToolCall):
            observation = await dependencies.read_runtime.execute_single(
                decision.call,
                max_retries=budget.limits.max_runtime_read_retries,
                on_started=on_started,
            )
            observations = (observation,)
        else:
            result = await dependencies.read_runtime.execute_batch(
                decision,
                max_retries=budget.limits.max_runtime_read_retries,
                max_batch=budget.limits.max_parallel_read_batch,
                on_started=on_started,
            )
            observations = result.results

        new_evidence, created = dependencies.evidence_tracker.record_observations(
            state["evidence"], observations
        )
        outcomes = state.get("goal_outcomes", {})
        if state.get("goal_descriptor") is not None and state.get("completion_contract") is not None:
            outcomes = dependencies.evidence_tracker.refresh_goal_outcomes(
                state["goal_descriptor"], state["completion_contract"], new_evidence, outcomes
            )
        for observation in observations:
            _emit(
                state,
                dependencies,
                "ToolObservationRecorded",
                {
                    "observation_id": observation.observation_id,
                    "call_id": observation.call_id,
                    "source": observation.source,
                    "target": observation.target,
                    "status": observation.status,
                    "trust": observation.trust,
                    "error_code": observation.error_code,
                    "data": observation.data,
                },
            )
        for record in created:
            _emit(
                state,
                dependencies,
                "EvidenceRecorded",
                {
                    "evidence_id": record.evidence_id,
                    "kind": record.kind,
                    "target": record.target,
                    "observation_id": record.observation_id,
                    "provenance": record.provenance,
                    "status": record.status,
                },
            )
        for outcome in outcomes.values():
            _emit(
                state,
                dependencies,
                "GoalOutcomeUpdated",
                {
                    "goal_id": outcome.goal_id,
                    "status": outcome.status.value,
                    "evidence_refs": list(outcome.evidence_refs),
                },
            )
        return {
            "budgets": budget.with_read_calls(len(calls)).with_retries(
                sum(item.retry_count for item in observations)
            ),
            "tool_call_count": state.get("tool_call_count", 0) + len(calls),
            "observations": tuple(state.get("observations", ())) + observations,
            "evidence": new_evidence,
            "goal_outcomes": outcomes,
            "decision": None,
        }

    async def response_completion_gate(state: AgentState) -> dict[str, Any]:
        candidate = state.get("final_candidate")
        descriptor = state.get("goal_descriptor")
        contract = state.get("completion_contract")
        if not isinstance(candidate, FinalCandidate) or descriptor is None or contract is None:
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
                    safe_facts={"reason": "completion_gate_missing_structured_input"},
                    message_template="The completion gate could not evaluate the Agent candidate.",
                ),
                dependencies,
            )
        evaluation = dependencies.completion_gate.evaluate(
            candidate,
            descriptor,
            contract,
            state["evidence"],
            state.get("goal_outcomes", {}),
        )
        gate_event = _emit(
            state,
            dependencies,
            "CompletionGateEvaluated",
            {
                "passed": evaluation.passed,
                "facts": list(evaluation.facts),
                "missing": list(evaluation.missing),
            },
        )
        updates: dict[str, Any] = {
            "goal_outcomes": evaluation.goal_outcomes,
            "gate_passed": evaluation.passed,
            "last_event_id": gate_event.event_id,
        }
        for outcome in evaluation.goal_outcomes.values():
            _emit(
                state,
                dependencies,
                "GoalOutcomeUpdated",
                {
                    "goal_id": outcome.goal_id,
                    "status": outcome.status.value,
                    "evidence_refs": list(outcome.evidence_refs),
                },
                causation_id=gate_event.event_id,
            )
        if evaluation.passed:
            return updates
        budget = state["budgets"].with_gate_rejection()
        if budget.completion_gate_rejections > budget.limits.max_completion_gate_rejections:
            return _terminal_update(
                state,
                ControlledTerminalOutcome(
                    code=TerminalCode.BUDGET_EXHAUSTED,
                    safe_facts={
                        "completion_gate_rejections": budget.completion_gate_rejections,
                        "max_completion_gate_rejections": budget.limits.max_completion_gate_rejections,
                    },
                    message_template=(
                        "The bounded execution budget was exhausted before all goals could be completed."
                    ),
                ),
                dependencies,
                budgets=budget,
            )
        return {
            **updates,
            "budgets": budget,
            "gate_feedback": tuple(evaluation.facts + evaluation.missing),
        }

    def after_agent(state: AgentState) -> str:
        if state.get("terminal_state") is not None:
            return "terminal"
        decision = state.get("decision")
        if isinstance(decision, FinalCandidate):
            return "response_completion_gate"
        if isinstance(decision, (SingleToolCall, ReadToolBatch)):
            return "read_executor"
        return "terminal"

    def after_gate(state: AgentState) -> str:
        if state.get("terminal_state") is not None:
            return "terminal"
        return "end" if state.get("gate_passed") else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("read_executor", read_executor)
    builder.add_node("response_completion_gate", response_completion_gate)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        after_agent,
        {
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


def _emit(
    state: AgentState,
    dependencies: GraphDependencies,
    event_type: str,
    payload: dict[str, Any],
    *,
    causation_id: str | None = None,
):
    snapshot: OperatingPrinciplesSnapshot = state["operating_principles_snapshot"]
    provenance = EventProvenance(
        model_version=dependencies.model_version,
        prompt_version=dependencies.prompt_version,
        tool_catalog_hash=dependencies.tool_catalog_hash,
        operating_principles_version=snapshot.version,
        operating_principles_hash=snapshot.content_hash,
        policy_version=dependencies.policy_version,
    )
    return dependencies.event_store.append(
        event_type=event_type,
        request_id=state["request_id"],
        thread_id=state["thread_id"],
        payload=payload,
        provenance=provenance,
        causation_id=causation_id or state.get("last_event_id"),
    )


def _terminal_update(
    state: AgentState,
    outcome: ControlledTerminalOutcome,
    dependencies: GraphDependencies,
    *,
    budgets: BudgetState | None = None,
) -> dict[str, Any]:
    event = _emit(
        state,
        dependencies,
        "ControlledTerminalOutcomeProduced",
        {
            "code": outcome.code.value,
            "safe_facts": outcome.safe_facts,
            "message_template": outcome.message_template,
        },
    )
    return {
        "terminal_state": outcome,
        "termination_reason": outcome.code.value,
        "decision": None,
        "final_candidate": None,
        "budgets": budgets or state["budgets"],
        "last_event_id": event.event_id,
    }
