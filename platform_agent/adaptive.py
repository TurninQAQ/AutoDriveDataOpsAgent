"""Shared controller for the read-only adaptive evidence loop.

The controller owns loop mechanics and safety invariants only.  Domain evidence
normalization and final answer synthesis remain in :mod:`platform_agent.workflow`.
This keeps Sequential and LangGraph runtimes on the same budget and validation
rules without creating a second policy engine.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from platform_mcp.server import READ_ONLY_TOOL_NAMES

from .models import (
    AgentIntent,
    AgentPlan,
    AgentStepAction,
    AgentStepDecision,
    ConversationTurn,
    KnowledgeObservation,
    ToolCallSpec,
    ToolObservation,
)


READ_ONLY_INTENTS = frozenset(
    {
        AgentIntent.PLATFORM_HEALTH,
        AgentIntent.LIST_TASKS,
        AgentIntent.TASK_STATUS,
        AgentIntent.TASK_DIAGNOSIS,
        AgentIntent.GPU_DIAGNOSIS,
        AgentIntent.STAGE_FAILURE,
        AgentIntent.GENERAL_READ,
        AgentIntent.PLATFORM_KNOWLEDGE,
    }
)


@dataclass(frozen=True)
class AdaptiveLimits:
    max_steps: int = 8
    max_tool_calls: int = 6
    max_identical_tool_calls: int = 2
    max_consecutive_tool_failures: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_steps", max(1, int(self.max_steps)))
        object.__setattr__(self, "max_tool_calls", max(1, int(self.max_tool_calls)))
        object.__setattr__(self, "max_identical_tool_calls", max(1, int(self.max_identical_tool_calls)))
        object.__setattr__(self, "max_consecutive_tool_failures", max(1, int(self.max_consecutive_tool_failures)))


@dataclass
class AdaptiveLoopResult:
    observations: list[ToolObservation] = field(default_factory=list)
    knowledge: list[KnowledgeObservation] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    current_intent: AgentIntent | None = None
    evidence_sufficient: bool = False
    termination_reason: str = "unknown"
    tool_call_count: int = 0
    errors: list[str] = field(default_factory=list)


def canonical_tool_signature(call: ToolCallSpec) -> str:
    """Return a stable signature for duplicate-call enforcement."""

    return json.dumps(
        {"tool": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _decision_summary(value: str) -> str:
    # Tracing must stay auditable without turning the field into a reasoning dump.
    return " ".join(str(value or "").split())[:500]


class AdaptiveLoopController:
    """Execute at most one validated read-only tool per model decision."""

    def __init__(self, model, policy, limits: AdaptiveLimits, trace_event: Callable[..., None] | None = None):
        self.model = model
        self.policy = policy
        self.limits = limits
        self.trace_event = trace_event

    def _trace(self, name: str, *, status: str = "ok", data: dict[str, Any] | None = None) -> None:
        if self.trace_event is not None:
            self.trace_event(name, status=status, data=data or {})

    def _validate_decision(
        self,
        decision: AgentStepDecision,
        available_tools: set[str],
    ) -> None:
        if decision.revised_intent is not None and decision.revised_intent not in READ_ONLY_INTENTS:
            raise PermissionError(
                "Adaptive read-only decision cannot revise intent to "
                f"{decision.revised_intent.value}"
            )
        if decision.action == AgentStepAction.FINISH:
            if decision.tool_call is not None:
                raise PermissionError("FINISH decision must not include a tool_call")
            return
        if decision.action != AgentStepAction.CALL_TOOL:
            raise PermissionError(f"Unsupported adaptive action: {decision.action}")
        call = decision.tool_call
        if call is None:
            raise PermissionError("CALL_TOOL decision must include exactly one tool_call")
        if call.name not in READ_ONLY_TOOL_NAMES:
            raise PermissionError(f"Adaptive decision attempted a non-read-only tool: {call.name}")
        if call.name not in available_tools:
            raise PermissionError(f"Adaptive decision selected unavailable tool: {call.name}")
        self.policy.validate_tool_name(call.name)

    async def run(
        self,
        *,
        user_text: str,
        initial_plan: AgentPlan,
        tool_descriptions: list[dict[str, Any]],
        observations: list[ToolObservation],
        knowledge: list[KnowledgeObservation],
        history: list[ConversationTurn],
        execute_tool: Callable[[ToolCallSpec], Awaitable[list[ToolObservation]]],
        normalize_observation: Callable[[ToolObservation], list[KnowledgeObservation]],
        initial_intent: AgentIntent | None = None,
    ) -> AdaptiveLoopResult:
        result = AdaptiveLoopResult(
            observations=list(observations),
            knowledge=list(knowledge),
            current_intent=initial_intent or initial_plan.intent,
        )
        available_tools = {
            str(item.get("name"))
            for item in tool_descriptions
            if isinstance(item, dict) and item.get("name")
        }
        successful_signatures: Counter[str] = Counter()
        consecutive_failures = 0

        for step_index in range(self.limits.max_steps):
            if result.tool_call_count >= self.limits.max_tool_calls:
                result.termination_reason = "tool_budget_exhausted"
                break

            remaining_steps = self.limits.max_steps - step_index
            remaining_tools = self.limits.max_tool_calls - result.tool_call_count
            try:
                decision = await self.model.decide_next(
                    user_text=user_text,
                    initial_plan=initial_plan,
                    tool_descriptions=tool_descriptions,
                    observations=list(result.observations),
                    knowledge=list(result.knowledge),
                    history=history,
                    step_index=step_index,
                    remaining_tool_calls=remaining_tools,
                    current_intent=result.current_intent,
                    adaptive_steps=list(result.steps[-8:]),
                )
                if not isinstance(decision, AgentStepDecision):
                    decision = AgentStepDecision.model_validate(decision)
            except Exception as exc:
                result.errors.append(f"adaptive decision failed: {exc}")
                result.termination_reason = "decision_error"
                self._trace(
                    "adaptive_decision",
                    status="error",
                    data={"step": step_index, "error": str(exc), "remaining_steps": remaining_steps, "remaining_tool_calls": remaining_tools},
                )
                break

            summary = _decision_summary(decision.decision_summary)
            decision_data: dict[str, Any] = {
                "step": step_index,
                "current_intent": result.current_intent.value if result.current_intent else None,
                "action": decision.action.value,
                "evidence_sufficient": decision.evidence_sufficient,
                "revised_intent": decision.revised_intent.value if decision.revised_intent else None,
                "decision_summary": summary,
                "remaining_steps": remaining_steps,
                "remaining_tool_calls": remaining_tools,
            }
            if decision.tool_call is not None:
                decision_data["tool"] = decision.tool_call.name
                decision_data["arguments"] = decision.tool_call.arguments
            result.steps.append(dict(decision_data))
            self._trace("adaptive_decision", data=decision_data)

            try:
                self._validate_decision(decision, available_tools)
            except Exception as exc:
                result.errors.append(str(exc))
                result.evidence_sufficient = False
                result.termination_reason = "unsafe_adaptive_decision"
                self._trace(
                    "adaptive_decision",
                    status="blocked",
                    data={**decision_data, "error": str(exc)},
                )
                break

            if decision.revised_intent is not None:
                result.current_intent = decision.revised_intent

            if decision.action == AgentStepAction.FINISH:
                result.evidence_sufficient = bool(decision.evidence_sufficient)
                result.termination_reason = "agent_finished"
                break

            call = decision.tool_call
            assert call is not None  # validated above
            signature = canonical_tool_signature(call)
            if successful_signatures[signature] >= self.limits.max_identical_tool_calls:
                result.errors.append(
                    f"Adaptive duplicate tool limit reached for {call.name}."
                )
                result.evidence_sufficient = False
                result.termination_reason = "duplicate_tool_limit"
                self._trace(
                    "adaptive_termination",
                    status="blocked",
                    data={"step": step_index, "reason": result.termination_reason, "tool": call.name},
                )
                break

            if result.tool_call_count >= self.limits.max_tool_calls:
                result.termination_reason = "tool_budget_exhausted"
                break

            result.tool_call_count += 1
            try:
                new_observations = await execute_tool(call)
            except Exception as exc:
                new_observations = [
                    ToolObservation(
                        tool_name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=str(exc),
                    )
                ]
            if not new_observations:
                new_observations = [
                    ToolObservation(
                        tool_name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error="Tool client returned no observation",
                    )
                ]

            result.observations.extend(new_observations)
            normalized: list[KnowledgeObservation] = []
            for observation in new_observations:
                normalized.extend(normalize_observation(observation))
            if normalized:
                seen = {item.chunk_id for item in result.knowledge}
                result.knowledge.extend(item for item in normalized if item.chunk_id not in seen)

            successful = all(item.ok for item in new_observations)
            if successful:
                successful_signatures[signature] += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                result.errors.extend(
                    f"{item.tool_name}: {item.error}"
                    for item in new_observations
                    if not item.ok and item.error
                )

            self._trace(
                "adaptive_tool_result",
                status="ok" if successful else "error",
                data={
                    "step": step_index,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "observation_status": ["ok" if item.ok else "error" for item in new_observations],
                    "remaining_steps": max(0, self.limits.max_steps - step_index - 1),
                    "remaining_tool_calls": self.limits.max_tool_calls - result.tool_call_count,
                },
            )
            if consecutive_failures >= self.limits.max_consecutive_tool_failures:
                result.evidence_sufficient = False
                result.termination_reason = "consecutive_tool_failures"
                break
        else:
            result.termination_reason = "step_budget_exhausted"

        if result.termination_reason == "unknown":
            result.termination_reason = "step_budget_exhausted"
        self._trace(
            "adaptive_termination",
            status="ok" if result.evidence_sufficient else "incomplete",
            data={
                "reason": result.termination_reason,
                "step_count": len(result.steps),
                "tool_call_count": result.tool_call_count,
                "evidence_sufficient": result.evidence_sufficient,
            },
        )
        return result


__all__ = [
    "AdaptiveLimits",
    "AdaptiveLoopController",
    "AdaptiveLoopResult",
    "READ_ONLY_INTENTS",
    "canonical_tool_signature",
]
