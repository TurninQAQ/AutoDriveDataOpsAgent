"""Explicit separation of authoritative state, guidance, and untrusted data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .budgets import BudgetState
from .contracts import CompletionContract
from .evidence import (
    ContextBudgetExceeded,
    EvidenceProjection,
    EvidenceProjectionBuilder,
    EvidenceState,
    ToolObservation,
)
from .goals import GoalDescriptor
from .outcomes import ControlledTerminalOutcome, GoalOutcome
from .principles import OperatingPrinciplesSnapshot


@dataclass(frozen=True)
class RuntimeStructuredContext:
    request_id: str
    thread_id: str
    goal_descriptor: GoalDescriptor | None
    completion_contract: CompletionContract | None
    goal_outcomes: tuple[GoalOutcome, ...]
    evidence: EvidenceProjection
    budgets: BudgetState
    terminal_state: ControlledTerminalOutcome | None
    gate_feedback: tuple[str, ...]
    new_turn: bool


@dataclass(frozen=True)
class OperatingGuidanceContext:
    version: str
    content_hash: str
    principles: tuple[str, ...]


@dataclass(frozen=True)
class SemanticObservationContext:
    observations: tuple[ToolObservation, ...]
    trust_label: str = "UNTRUSTED_EXTERNAL_DATA"


@dataclass(frozen=True)
class AgentContext:
    user_input: str
    messages: tuple[dict[str, Any], ...]
    runtime_structured: RuntimeStructuredContext
    operating_guidance: OperatingGuidanceContext
    semantic_observations: SemanticObservationContext
    new_turn: bool
    estimated_context_chars: int


class ContextBuilder:
    """Build bounded typed projections without truncating runtime authority.

    ``max_context_tokens`` is an explicit character approximation (4 chars per
    token) for the entire Agent-facing projection. Canonical evidence remains
    complete in state; only a metadata projection enters this context.
    """

    def __init__(
        self,
        *,
        max_guidance_chars: int = 16_000,
        max_observations: int = 32,
        max_messages: int = 32,
        max_message_chars: int = 4_096,
        evidence_projection: EvidenceProjectionBuilder | None = None,
    ):
        self.max_guidance_chars = max_guidance_chars
        self.max_observations = max_observations
        self.max_messages = max_messages
        self.max_message_chars = max_message_chars
        self.evidence_projection = evidence_projection or EvidenceProjectionBuilder()

    def build(self, state: dict[str, Any], snapshot: OperatingPrinciplesSnapshot) -> AgentContext:
        max_context_chars = int(state["budgets"].limits.max_context_tokens) * 4
        if max_context_chars < 256:
            raise ContextBudgetExceeded("context budget is too small for critical structured state")

        projection_budget = max(256, max_context_chars // 2)
        evidence_projection = self.evidence_projection.build(
            state.get("evidence", EvidenceState()),
            state.get("goal_descriptor"),
            state.get("completion_contract"),
            max_records=min(64, max(1, projection_budget // 160)),
            max_chars=projection_budget,
        )
        critical_structured = {
            "request_id": state["request_id"],
            "thread_id": state["thread_id"],
            "goal_descriptor": state.get("goal_descriptor"),
            "completion_contract": state.get("completion_contract"),
            "goal_outcomes": tuple(state.get("goal_outcomes", {}).values()),
            "budgets": state["budgets"],
            "terminal_state": state.get("terminal_state"),
            "gate_feedback": tuple(state.get("gate_feedback", ()))[:8],
        }
        structured_cost = len(repr(critical_structured)) + len(repr(evidence_projection))
        if structured_cost >= max_context_chars:
            raise ContextBudgetExceeded("critical structured projection exceeds context budget")
        semantic_budget_total = max_context_chars - structured_cost

        # Guidance has priority over semantic payloads, but remains advisory.
        guidance_limit = min(semantic_budget_total // 3, self.max_guidance_chars)
        guidance: list[str] = []
        guidance_size = 0
        for item in snapshot.principles:
            text = f"{item.principle_id}: {item.title}\n{item.text}"
            remaining = guidance_limit - guidance_size
            if remaining <= 0:
                break
            bounded = text[:remaining]
            guidance.append(bounded)
            guidance_size += len(bounded)

        user_budget = max(32, semantic_budget_total // 4)
        user_input = _bound_text(str(state.get("user_input", "")), user_budget)
        semantic_budget = max(0, semantic_budget_total - guidance_size - len(user_input))
        observations, observation_size = self._latest_observations(
            state.get("observations", ()), min(self.max_observations, semantic_budget // 2), semantic_budget
        )

        message_budget = max(
            0, semantic_budget - observation_size
        )
        messages = self._latest_messages(state.get("messages", ()), message_budget)

        return AgentContext(
            user_input=user_input,
            messages=messages,
            runtime_structured=RuntimeStructuredContext(
                request_id=state["request_id"],
                thread_id=state["thread_id"],
                goal_descriptor=state.get("goal_descriptor"),
                completion_contract=state.get("completion_contract"),
                goal_outcomes=tuple(state.get("goal_outcomes", {}).values()),
                evidence=evidence_projection,
                budgets=state["budgets"],
                terminal_state=state.get("terminal_state"),
                gate_feedback=tuple(state.get("gate_feedback", ())),
                new_turn=bool(state.get("new_turn", False)),
            ),
            operating_guidance=OperatingGuidanceContext(
                version=snapshot.version,
                content_hash=snapshot.content_hash,
                principles=tuple(guidance),
            ),
            semantic_observations=SemanticObservationContext(observations=observations),
            new_turn=bool(state.get("new_turn", False)),
            estimated_context_chars=(
                structured_cost
                + guidance_size
                + len(user_input)
                + observation_size
                + sum(len(str(item.get("content", ""))) for item in messages)
            ),
        )

    def _latest_observations(
        self, observations: Any, limit: int, budget: int
    ) -> tuple[tuple[ToolObservation, ...], int]:
        selected = list(observations)[-self.max_observations :]
        if limit <= 0 or budget <= 0:
            return (), 0
        kept: list[ToolObservation] = []
        used = 0
        # Walk backwards so the newest observation is never pushed out by old
        # large logs. Restore chronological order for the Agent projection.
        for observation in reversed(selected):
            if len(kept) >= limit or used >= budget:
                break
            data_text = repr(observation.data)
            remaining = budget - used
            if len(data_text) > remaining:
                bounded_data = _with_marker(
                    data_text, remaining, " [semantic context truncated]"
                )
                kept.append(replace(observation, data=bounded_data, result=None))
                used += len(bounded_data)
                break
            # The normalized result can contain large knowledge/diagnostic
            # content. It is canonical Runtime data, not a prompt payload.
            kept.append(replace(observation, result=None))
            used += len(data_text)
        kept.reverse()
        return tuple(kept), used

    def _latest_messages(self, messages: Any, budget: int) -> tuple[dict[str, Any], ...]:
        if budget <= 0:
            return ()
        kept: list[dict[str, Any]] = []
        used = 0
        for message in reversed(list(messages)[-self.max_messages :]):
            if used >= budget:
                break
            row = dict(message) if isinstance(message, dict) else {"content": str(message)}
            content = str(row.get("content", ""))
            remaining = min(self.max_message_chars, budget - used)
            bounded = _with_marker(content, remaining, " [message context truncated]")
            row["content"] = bounded
            kept.append(row)
            used += len(bounded)
        kept.reverse()
        return tuple(kept)


def _bound_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - len("…"))] + "…"


def _with_marker(value: str, limit: int, marker: str) -> str:
    if len(value) <= limit:
        return value
    if limit <= len(marker):
        return _bound_text(value, limit)
    return _bound_text(value, limit - len(marker)) + marker
