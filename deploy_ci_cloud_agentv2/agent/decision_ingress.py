"""Deterministic validation of untrusted provider AgentDecision proposals."""

from __future__ import annotations

from collections.abc import Mapping

from .decisions import AgentDecision, FinalCandidate, ReadToolBatch, SingleToolCall, ToolCall
from .goals import (
    DiagnoseTask,
    ExplainKnowledge,
    GoalDescriptor,
    InspectGPU,
    InspectQueue,
    ReadTaskState,
)
from .immutable import CanonicalizationError, canonical_snapshot


class AgentDecisionValidationError(ValueError):
    """A provider proposal failed deterministic structural ingress validation."""


class AgentDecisionIngressValidator:
    """Validate shape and canonicalize proposals before compiler/guard use.

    This component performs no intent inference and no tool selection.  It
    only proves that a provider proposal has one known variant, valid fields,
    a closed canonical argument value, and (when present) a valid goal
    descriptor.  The returned proposal is detached from provider-owned
    collections.
    """

    def validate(self, proposal: object) -> AgentDecision:
        if type(proposal) is SingleToolCall:
            if proposal.kind != "SINGLE_TOOL_CALL":
                raise AgentDecisionValidationError("SingleToolCall.kind is invalid")
            call = self._tool_call(proposal.call)
            descriptor = self._descriptor(proposal.proposed_goal_descriptor)
            return SingleToolCall(call, descriptor)
        if type(proposal) is ReadToolBatch:
            if proposal.kind != "READ_TOOL_BATCH":
                raise AgentDecisionValidationError("ReadToolBatch.kind is invalid")
            calls = proposal.calls
            if not isinstance(calls, (list, tuple)) or not calls:
                raise AgentDecisionValidationError("READ_TOOL_BATCH calls must be a non-empty sequence")
            if any(type(call) is not ToolCall for call in calls):
                raise AgentDecisionValidationError(
                    "READ_TOOL_BATCH may contain only unaccepted ToolCall proposals"
                )
            descriptor = self._descriptor(proposal.proposed_goal_descriptor)
            return ReadToolBatch(tuple(self._tool_call(call) for call in calls), descriptor)
        if type(proposal) is FinalCandidate:
            if proposal.kind != "FINAL_CANDIDATE":
                raise AgentDecisionValidationError("FinalCandidate.kind is invalid")
            if not isinstance(proposal.response, str) or not proposal.response.strip():
                raise AgentDecisionValidationError("FINAL_CANDIDATE response must be a non-empty string")
            refs = proposal.referenced_goal_ids
            if not isinstance(refs, (list, tuple)):
                raise AgentDecisionValidationError("FINAL_CANDIDATE referenced_goal_ids must be a sequence")
            normalized_refs: list[str] = []
            for ref in refs:
                if not isinstance(ref, str) or not ref.strip():
                    raise AgentDecisionValidationError(
                        "FINAL_CANDIDATE goal references must be non-empty strings"
                    )
                normalized_refs.append(ref.strip())
            if len(normalized_refs) != len(set(normalized_refs)):
                raise AgentDecisionValidationError("FINAL_CANDIDATE goal references must be unique")
            descriptor = self._descriptor(proposal.proposed_goal_descriptor)
            return FinalCandidate(
                proposal.response,
                descriptor,
                tuple(normalized_refs),
            )
        raise AgentDecisionValidationError(
            f"unsupported AgentDecision proposal: {type(proposal).__name__}"
        )

    def _tool_call(self, proposal: object) -> ToolCall:
        if type(proposal) is not ToolCall:
            raise AgentDecisionValidationError("tool decision must contain a ToolCall proposal")
        if not isinstance(proposal.call_id, str) or not proposal.call_id.strip():
            raise AgentDecisionValidationError("ToolCall.call_id must be a non-empty string")
        if not isinstance(proposal.tool_name, str) or not proposal.tool_name.strip():
            raise AgentDecisionValidationError("ToolCall.tool_name must be a non-empty string")
        if not isinstance(proposal.arguments, Mapping):
            raise AgentDecisionValidationError("ToolCall.arguments must be a mapping")
        try:
            arguments = canonical_snapshot(proposal.arguments)
        except CanonicalizationError as exc:
            raise AgentDecisionValidationError(f"ToolCall.arguments are not canonical: {exc}") from exc
        return ToolCall(proposal.call_id.strip(), proposal.tool_name.strip(), arguments)

    def _descriptor(self, proposal: object) -> GoalDescriptor | None:
        if proposal is None:
            return None
        if type(proposal) is not GoalDescriptor:
            raise AgentDecisionValidationError("proposed_goal_descriptor must be GoalDescriptor or None")
        if type(proposal.descriptor_version) is not int or proposal.descriptor_version < 1:
            raise AgentDecisionValidationError("descriptor_version must be a positive integer")
        goals = proposal.goals
        if not isinstance(goals, (list, tuple)) or not goals:
            raise AgentDecisionValidationError("GoalDescriptor.goals must be non-empty")
        normalized = []
        seen: set[str] = set()
        for goal in goals:
            if type(goal) not in {
                ReadTaskState,
                InspectGPU,
                InspectQueue,
                ExplainKnowledge,
                DiagnoseTask,
            }:
                raise AgentDecisionValidationError("GoalDescriptor contains an invalid goal")
            goal_id = self._text(getattr(goal, "goal_id", None), "goal_id")
            if goal_id in seen:
                raise AgentDecisionValidationError("GoalDescriptor goal_id values must be unique")
            seen.add(goal_id)
            try:
                if type(goal) is ReadTaskState:
                    normalized.append(ReadTaskState(goal_id, self._text(goal.target, "target")))
                elif type(goal) is DiagnoseTask:
                    normalized.append(DiagnoseTask(goal_id, self._text(goal.target, "target")))
                elif type(goal) is ExplainKnowledge:
                    normalized.append(ExplainKnowledge(goal_id, self._text(goal.topic, "topic")))
                elif type(goal) is InspectGPU:
                    normalized.append(InspectGPU(goal_id))
                else:
                    target = goal.target
                    if target is not None:
                        target = self._text(target, "target")
                    normalized.append(InspectQueue(goal_id, target))
            except (TypeError, ValueError, AttributeError) as exc:
                raise AgentDecisionValidationError(f"invalid GoalDescriptor goal: {exc}") from exc
        try:
            return GoalDescriptor(proposal.descriptor_version, tuple(normalized))
        except (TypeError, ValueError) as exc:
            raise AgentDecisionValidationError(f"invalid GoalDescriptor: {exc}") from exc

    @staticmethod
    def _text(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AgentDecisionValidationError(f"{field} must be a non-empty string")
        return value.strip()
