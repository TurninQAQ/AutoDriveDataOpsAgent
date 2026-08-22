"""Canonical LangGraph state and a small in-memory checkpointer."""

from __future__ import annotations

import copy
import uuid
from typing import Any, TypedDict

from .budgets import BudgetState, RuntimeBudgets
from .decisions import AgentDecision, FinalCandidate
from .evidence import EvidenceState, ToolObservation
from .goals import GoalDescriptor
from .contracts import CompletionContract
from .outcomes import ControlledTerminalOutcome, GoalOutcome


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_input: str
    request_id: str
    thread_id: str
    step_count: int
    tool_call_count: int
    goal_descriptor: GoalDescriptor | None
    goal_descriptor_version: int
    completion_contract: CompletionContract | None
    goal_outcomes: dict[str, GoalOutcome]
    evidence: EvidenceState
    observations: tuple[ToolObservation, ...]
    budgets: BudgetState
    terminal_state: ControlledTerminalOutcome | None
    termination_reason: str | None
    operating_principles_snapshot: Any
    decision: AgentDecision | None
    final_candidate: FinalCandidate | None
    gate_feedback: tuple[str, ...]
    gate_passed: bool | None
    new_turn: bool
    last_event_id: str | None
    continue_after_read_guard: bool


class InMemoryCheckpointer:
    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def load(self, thread_id: str) -> AgentState | None:
        state = self._states.get(thread_id)
        return copy.deepcopy(state) if state is not None else None

    def save(self, state: AgentState) -> None:
        self._states[state["thread_id"]] = copy.deepcopy(state)


def new_state(
    *,
    user_input: str,
    thread_id: str,
    snapshot: Any,
    budgets: RuntimeBudgets,
    prior: AgentState | None = None,
) -> AgentState:
    request_id = f"req_{uuid.uuid4().hex}"
    prior_messages = list(prior.get("messages", [])) if prior else []
    prior_descriptor = prior.get("goal_descriptor") if prior else None
    prior_contract = prior.get("completion_contract") if prior else None
    prior_outcomes = dict(prior.get("goal_outcomes", {})) if prior else {}
    prior_evidence = prior.get("evidence", EvidenceState()) if prior else EvidenceState()
    return AgentState(
        messages=prior_messages + [{"role": "user", "content": user_input}],
        user_input=user_input,
        request_id=request_id,
        thread_id=thread_id,
        step_count=0,
        tool_call_count=0,
        goal_descriptor=prior_descriptor,
        goal_descriptor_version=(prior.get("goal_descriptor_version", 0) if prior else 0),
        completion_contract=prior_contract,
        goal_outcomes=prior_outcomes,
        evidence=prior_evidence,
        observations=(),
        budgets=BudgetState(budgets),
        terminal_state=None,
        termination_reason=None,
        operating_principles_snapshot=snapshot,
        decision=None,
        final_candidate=None,
        gate_feedback=(),
        gate_passed=None,
        new_turn=True,
        continue_after_read_guard=False,
    )
