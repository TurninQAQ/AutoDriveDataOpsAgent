"""Explicit separation of runtime state, advisory guidance, and untrusted data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .budgets import BudgetState
from .contracts import CompletionContract
from .evidence import EvidenceState, ToolObservation
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
    evidence: EvidenceState
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


class ContextBuilder:
    """Builds three typed projections; it never collapses them into one authority."""

    def __init__(self, *, max_guidance_chars: int = 16_000, max_observations: int = 32):
        self.max_guidance_chars = max_guidance_chars
        self.max_observations = max_observations

    def build(self, state: dict[str, Any], snapshot: OperatingPrinciplesSnapshot) -> AgentContext:
        max_context_chars = max(
            1_024,
            min(
                self.max_guidance_chars,
                int(state["budgets"].limits.max_context_tokens) * 4,
            ),
        )
        guidance_limit = max_context_chars // 2
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
        observations = tuple(state.get("observations", ()))[: self.max_observations]
        observation_budget = max_context_chars
        bounded_observations = []
        used_chars = guidance_size
        for observation in observations:
            data_text = repr(observation.data)
            remaining = observation_budget - used_chars
            if remaining <= 0:
                break
            if len(data_text) > remaining:
                bounded_observations.append(
                    replace(
                        observation,
                        data=(
                            data_text[: max(0, remaining)]
                            + " [semantic context truncated; observation remains canonical]"
                        ),
                    )
                )
                break
            bounded_observations.append(observation)
            used_chars += len(data_text)
        return AgentContext(
            user_input=str(state.get("user_input", "")),
            messages=tuple(state.get("messages", ())),
            runtime_structured=RuntimeStructuredContext(
                request_id=state["request_id"],
                thread_id=state["thread_id"],
                goal_descriptor=state.get("goal_descriptor"),
                completion_contract=state.get("completion_contract"),
                goal_outcomes=tuple(state.get("goal_outcomes", {}).values()),
                evidence=state.get("evidence", EvidenceState()),
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
            semantic_observations=SemanticObservationContext(
                observations=tuple(bounded_observations)
            ),
            new_turn=bool(state.get("new_turn", False)),
        )
