"""Bounded Agent projection over one current request and separate history."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .budgets import BudgetState
from .contracts import CompletionContract
from .evidence import ContextBudgetExceeded, EvidenceProjection, EvidenceProjectionBuilder
from .goals import GoalDescriptor
from .identity import RequestIdentity
from .outcomes import ControlledTerminalOutcome, GoalOutcome
from .principles import OperatingPrinciplesSnapshot
from .state import CurrentRequestContext, ThreadHistory


@dataclass(frozen=True)
class HistoricalRequestProjection:
    request_id: str
    turn_id: str
    user_input: str
    status: str
    goal_ids: tuple[str, ...]
    outcome_statuses: tuple[str, ...]
    response: str | None


@dataclass(frozen=True)
class RuntimeStructuredContext:
    identity: RequestIdentity
    goal_descriptor: GoalDescriptor | None
    completion_contract: CompletionContract | None
    goal_outcomes: tuple[GoalOutcome, ...]
    evidence: EvidenceProjection
    budgets: BudgetState
    terminal_state: ControlledTerminalOutcome | None
    gate_feedback: tuple[str, ...]
    new_turn: bool
    write_transaction: object | None = None


@dataclass(frozen=True)
class OperatingGuidanceContext:
    version: str
    content_hash: str
    principles: tuple[str, ...]


@dataclass(frozen=True)
class SemanticObservationContext:
    observations: tuple[Any, ...]
    trust_label: str = "UNTRUSTED_EXTERNAL_DATA"


@dataclass(frozen=True)
class AgentContext:
    user_input: str
    messages: tuple[dict[str, Any], ...]
    runtime_structured: RuntimeStructuredContext
    operating_guidance: OperatingGuidanceContext
    semantic_observations: SemanticObservationContext
    thread_history: tuple[HistoricalRequestProjection, ...]
    new_turn: bool

    def model_facing_payload(self) -> tuple[Any, ...]:
        """The exact immutable representation delivered to the provider."""
        return (
            self.user_input,
            self.messages,
            self.runtime_structured,
            self.operating_guidance,
            self.semantic_observations,
            self.thread_history,
            self.new_turn,
        )

    @property
    def estimated_context_chars(self) -> int:
        # Telemetry is derived from, and never added to, the model payload.
        return len(repr(self.model_facing_payload()))


class ContextBuilder:
    """Build a bounded, authority-preserving projection for one request."""

    def __init__(
        self,
        *,
        max_guidance_chars: int = 16_000,
        max_observations: int = 32,
        max_messages: int = 32,
        max_message_chars: int = 4_096,
        max_history_requests: int = 8,
        evidence_projection: EvidenceProjectionBuilder | None = None,
    ):
        self.max_guidance_chars = max_guidance_chars
        self.max_observations = max_observations
        self.max_messages = max_messages
        self.max_message_chars = max_message_chars
        self.max_history_requests = max_history_requests
        self.evidence_projection = evidence_projection or EvidenceProjectionBuilder()

    def build(self, current: CurrentRequestContext, history: ThreadHistory) -> AgentContext:
        max_context_chars = int(current.budgets.limits.max_context_tokens) * 4
        if max_context_chars < 256:
            raise ContextBudgetExceeded("context budget is too small for critical request state")
        projection_budget = max(256, max_context_chars // 2)
        evidence_projection = self.evidence_projection.build(
            current.evidence,
            current.goal_descriptor,
            current.completion_contract,
            max_records=min(64, max(1, projection_budget // 160)),
            max_chars=projection_budget,
        )
        bounded_gate_feedback = tuple(current.gate_feedback[:8])
        critical_structured = {
            "identity": current.identity,
            "goal_descriptor": current.goal_descriptor,
            "completion_contract": current.completion_contract,
            "goal_outcomes": tuple(current.goal_outcomes.values()),
            "evidence": evidence_projection,
            "budgets": current.budgets,
            "terminal_state": current.terminal_state,
            "gate_feedback": bounded_gate_feedback,
            "write_transaction": current.write_transaction.agent_projection() if current.write_transaction is not None else None,
        }
        structured_cost = len(repr(critical_structured))
        if structured_cost >= max_context_chars:
            raise ContextBudgetExceeded("critical current request projection exceeds context budget")
        history_projection = self._history_projection(history)
        history_cost = sum(len(repr(item)) for item in history_projection)
        semantic_budget_total = max_context_chars - structured_cost - history_cost
        if semantic_budget_total < 0:
            raise ContextBudgetExceeded(
                "critical current request plus bounded history exceeds context budget"
            )

        # Keep room for provider-facing dataclass/dict framing.  The budget is
        # for the whole projection, not merely the text bodies, so semantic
        # channels use conservative shares of the remaining characters.
        guidance_limit = min(max(0, semantic_budget_total // 6), self.max_guidance_chars)
        guidance: list[str] = []
        guidance_size = 0
        for item in current.operating_principles_snapshot.principles:
            text = f"{item.principle_id}: {item.title}\n{item.text}"
            remaining = guidance_limit - guidance_size
            if remaining <= 0:
                break
            bounded = text[:remaining]
            guidance.append(bounded)
            guidance_size += len(bounded)

        user_budget = max(32, max(0, semantic_budget_total - guidance_size) // 8)
        user_input = _bound_text(current.user_input, user_budget)
        semantic_budget = max(0, semantic_budget_total - guidance_size - len(user_input))
        observations, observation_size = self._latest_observations(
            current.observations,
            min(self.max_observations, semantic_budget // 4),
            semantic_budget // 4,
        )
        # Dataclass and container framing is part of the provider-facing
        # representation too.  Reserve a fixed envelope allowance so the
        # semantic channels cannot consume the entire remaining budget.
        message_budget = max(0, semantic_budget - observation_size - 1_024)
        messages = self._latest_messages(current.messages, message_budget)
        runtime_structured = RuntimeStructuredContext(
            identity=current.identity,
            goal_descriptor=current.goal_descriptor,
            completion_contract=current.completion_contract,
            goal_outcomes=tuple(current.goal_outcomes.values()),
            evidence=evidence_projection,
            budgets=current.budgets,
            terminal_state=current.terminal_state,
            gate_feedback=bounded_gate_feedback,
            new_turn=current.new_turn,
            write_transaction=current.write_transaction.agent_projection() if current.write_transaction is not None else None,
        )
        guidance_context = OperatingGuidanceContext(
            version=current.operating_principles_snapshot.version,
            content_hash=current.operating_principles_snapshot.content_hash,
            principles=tuple(guidance),
        )
        semantic_context = SemanticObservationContext(observations=observations)
        final_context = AgentContext(
            user_input=user_input,
            messages=messages,
            runtime_structured=runtime_structured,
            operating_guidance=guidance_context,
            semantic_observations=semantic_context,
            thread_history=history_projection,
            new_turn=current.new_turn,
        )
        # The budget applies to the exact object passed to the provider.  The
        # same immutable payload is measured here; no provisional object or
        # telemetry field can diverge from what the provider receives.
        if final_context.estimated_context_chars > max_context_chars:
            raise ContextBudgetExceeded("Agent-facing projection exceeded the full context budget")
        return final_context

    def _history_projection(self, history: ThreadHistory) -> tuple[HistoricalRequestProjection, ...]:
        selected = history.requests[-self.max_history_requests :]
        return tuple(
            HistoricalRequestProjection(
                request_id=item.request_id,
                turn_id=item.turn_id,
                user_input=_bound_text(item.user_input, 512),
                status=item.status,
                goal_ids=tuple(goal.goal_id for goal in item.goal_descriptor.goals)
                if item.goal_descriptor is not None
                else (),
                outcome_statuses=tuple(outcome.status.value for outcome in item.goal_outcomes),
                response=_bound_text(item.response, 1_024) if item.response else None,
            )
            for item in selected
        )

    def _latest_observations(
        self, observations: Any, limit: int, budget: int
    ) -> tuple[tuple[Any, ...], int]:
        selected = list(observations)[-self.max_observations :]
        if limit <= 0 or budget <= 0:
            return (), 0
        kept: list[Any] = []
        used = 0
        for observation in reversed(selected):
            if len(kept) >= limit or used >= budget:
                break
            remaining = budget - used
            base = _observation_projection(observation, "")
            overhead = len(repr(base))
            if overhead >= remaining:
                break
            data_text = repr(observation.data)
            available = remaining - overhead
            bounded_data = _with_marker(
                data_text, available, " [semantic context truncated]"
            )
            projected = _observation_projection(observation, bounded_data)
            cost = len(repr(projected))
            if cost > remaining:
                break
            kept.append(projected)
            used += cost
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
            remaining = budget - used
            # Only preserve the small conversational fields that are part of
            # the provider contract.  Arbitrary message metadata is not a
            # bounded context channel.
            prefix = {
                "role": _bound_text(row.get("role"), 32),
                "kind": _bound_text(row.get("kind"), 64),
                "candidate": bool(row.get("candidate", False)),
            }
            overhead = len(repr({**prefix, "content": ""}))
            if overhead >= remaining:
                break
            content_limit = min(
                self.max_message_chars,
                max(0, remaining - overhead),
            )
            bounded = _with_marker(content, content_limit, " [message context truncated]")
            projected = {**prefix, "content": bounded}
            cost = len(repr(projected))
            if cost > remaining:
                break
            kept.append(projected)
            used += cost
        kept.reverse()
        return tuple(kept)


def _observation_projection(observation: Any, bounded_data: str) -> dict[str, Any]:
    provenance = getattr(observation, "provenance", None)
    return {
        "observation_id": getattr(observation, "observation_id", ""),
        "source": getattr(observation, "source", ""),
        "target": getattr(observation, "target", ""),
        "transport_status": getattr(getattr(observation, "transport_status", None), "value", ""),
        "disposition": getattr(getattr(observation, "disposition", None), "value", ""),
        "trust": getattr(observation, "trust", "UNTRUSTED_EXTERNAL_DATA"),
        "error_code": getattr(observation, "error_code", None),
        "observed_at": getattr(observation, "observed_at", None),
        "provenance": {
            "requested_identity": getattr(provenance, "requested_identity", None),
            "observed_identity": getattr(provenance, "observed_identity", None),
            "identity_status": getattr(getattr(provenance, "identity_status", None), "value", ""),
            "scope_status": getattr(getattr(provenance, "scope_status", None), "value", ""),
        },
        "data": bounded_data,
    }


def _bound_text(value: str | None, limit: int) -> str:
    text = "" if value is None else str(value)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _with_marker(value: str, limit: int, marker: str) -> str:
    if len(value) <= limit:
        return value
    if limit <= len(marker):
        return _bound_text(value, limit)
    return _bound_text(value, limit - len(marker)) + marker
