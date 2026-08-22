"""Canonical request state and explicitly separated thread history."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, replace
from typing import Any, TypedDict

from .budgets import BudgetState, RuntimeBudgets
from .contracts import CompletionContract
from .decisions import AgentDecision, FinalCandidate
from .evidence import EvidenceState, ToolObservation
from .goals import GoalDescriptor
from .identity import RequestIdentity
from .outcomes import ControlledTerminalOutcome, GoalOutcome


@dataclass(frozen=True)
class CurrentRequestContext:
    """The sole authoritative state for the active request/turn."""

    identity: RequestIdentity
    user_input: str
    messages: tuple[dict[str, Any], ...]
    step_count: int
    tool_call_count: int
    goal_descriptor: GoalDescriptor | None
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
    continue_after_read_guard: bool

    def __post_init__(self) -> None:
        if self.evidence.owner != self.identity:
            raise ValueError("CurrentRequestContext evidence owner does not match request identity")


@dataclass(frozen=True)
class RequestHistoryEntry:
    request_id: str
    turn_id: str
    user_input: str
    status: str
    goal_descriptor: GoalDescriptor | None
    goal_outcomes: tuple[GoalOutcome, ...]
    evidence_refs: tuple[str, ...]
    response: str | None
    termination_reason: str | None


@dataclass(frozen=True)
class ThreadHistory:
    """Historical request summaries; it never supplies active evidence."""

    requests: tuple[RequestHistoryEntry, ...] = ()

    def append(self, current: CurrentRequestContext) -> "ThreadHistory":
        candidate = current.final_candidate
        status = "COMPLETED" if current.gate_passed else (
            "CONTROLLED_TERMINAL" if current.terminal_state else "INCOMPLETE"
        )
        refs = tuple(
            ref
            for outcome in current.goal_outcomes.values()
            for ref in outcome.evidence_refs
        )
        entry = RequestHistoryEntry(
            request_id=current.identity.request_id,
            turn_id=current.identity.turn_id,
            user_input=current.user_input,
            status=status,
            goal_descriptor=current.goal_descriptor,
            goal_outcomes=tuple(current.goal_outcomes.values()),
            evidence_refs=tuple(dict.fromkeys(refs)),
            response=candidate.response if candidate is not None and current.gate_passed else None,
            termination_reason=current.termination_reason,
        )
        return replace(self, requests=self.requests + (entry,))


class AgentState(TypedDict, total=False):
    thread_id: str
    current_request: CurrentRequestContext
    thread_history: ThreadHistory
    last_event_id: str | None


class InMemoryCheckpointer:
    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def load(self, thread_id: str) -> AgentState | None:
        state = self._states.get(thread_id)
        return copy.deepcopy(state) if state is not None else None

    def save(self, state: AgentState) -> None:
        self._states[state["thread_id"]] = copy.deepcopy(state)


class LatestStateHolder:
    """Crash-path projection of the latest state returned by a graph node."""

    def __init__(self) -> None:
        self._state: AgentState | None = None

    def record(self, state: AgentState) -> None:
        self._state = copy.deepcopy(state)

    def current(self) -> AgentState | None:
        return copy.deepcopy(self._state) if self._state is not None else None


def new_state(
    *,
    user_input: str,
    thread_id: str,
    snapshot: Any,
    budgets: RuntimeBudgets,
    prior: AgentState | None = None,
) -> AgentState:
    request_id = f"req_{uuid.uuid4().hex}"
    identity = RequestIdentity(
        thread_id=thread_id,
        request_id=request_id,
        turn_id=f"turn_{uuid.uuid4().hex}",
    )
    history = prior["thread_history"] if prior is not None else ThreadHistory()
    if prior is not None:
        history = history.append(prior["current_request"])
    current = CurrentRequestContext(
        identity=identity,
        user_input=user_input,
        messages=({"role": "user", "content": user_input},),
        step_count=0,
        tool_call_count=0,
        goal_descriptor=None,
        completion_contract=None,
        goal_outcomes={},
        evidence=EvidenceState(owner=identity),
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
    return AgentState(
        thread_id=thread_id,
        current_request=current,
        thread_history=history,
        last_event_id=None,
    )
