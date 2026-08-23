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
        try:
            if type(proposal) is SingleToolCall:
                self._expected_kind(proposal, "SINGLE_TOOL_CALL")
                call = self._tool_call(self._field(proposal, "call"))
                descriptor = self._descriptor(self._field(proposal, "proposed_goal_descriptor"))
                return SingleToolCall(call, descriptor)
            if type(proposal) is ReadToolBatch:
                self._expected_kind(proposal, "READ_TOOL_BATCH")
                calls = self._field(proposal, "calls")
                if type(calls) not in (list, tuple) or not calls:
                    raise AgentDecisionValidationError(
                        "READ_TOOL_BATCH calls must be a non-empty sequence"
                    )
                if any(type(call) is not ToolCall for call in calls):
                    raise AgentDecisionValidationError(
                        "READ_TOOL_BATCH may contain only unaccepted ToolCall proposals"
                    )
                normalized_calls = tuple(self._tool_call(call) for call in calls)
                if len({call.call_id for call in normalized_calls}) != len(normalized_calls):
                    raise AgentDecisionValidationError("READ_TOOL_BATCH call_id values must be unique")
                descriptor = self._descriptor(self._field(proposal, "proposed_goal_descriptor"))
                return ReadToolBatch(normalized_calls, descriptor)
            if type(proposal) is FinalCandidate:
                self._expected_kind(proposal, "FINAL_CANDIDATE")
                response = self._field(proposal, "response")
                if type(response) is not str or not response.strip():
                    raise AgentDecisionValidationError(
                        "FINAL_CANDIDATE response must be a non-empty string"
                    )
                refs = self._field(proposal, "referenced_goal_ids")
                if type(refs) not in (list, tuple):
                    raise AgentDecisionValidationError(
                        "FINAL_CANDIDATE referenced_goal_ids must be a sequence"
                    )
                normalized_refs: list[str] = []
                for ref in refs:
                    if type(ref) is not str or not ref.strip():
                        raise AgentDecisionValidationError(
                            "FINAL_CANDIDATE goal references must be non-empty strings"
                        )
                    normalized_refs.append(ref.strip())
                if len(normalized_refs) != len(set(normalized_refs)):
                    raise AgentDecisionValidationError(
                        "FINAL_CANDIDATE goal references must be unique"
                    )
                descriptor = self._descriptor(self._field(proposal, "proposed_goal_descriptor"))
                return FinalCandidate(response, descriptor, tuple(normalized_refs))
            raise AgentDecisionValidationError(
                f"unsupported AgentDecision proposal: {type(proposal).__name__}"
            )
        except AgentDecisionValidationError:
            raise
        except Exception as exc:
            # This is the untrusted provider boundary.  A malformed exact-type
            # proposal must never leak AttributeError/TypeError/etc. into the
            # graph.  Normalized proposals are built only after this block.
            raise AgentDecisionValidationError(
                f"malformed AgentDecision proposal: {type(proposal).__name__}"
            ) from exc

    def _tool_call(self, proposal: object) -> ToolCall:
        if type(proposal) is not ToolCall:
            raise AgentDecisionValidationError("tool decision must contain a ToolCall proposal")
        call_id = self._field(proposal, "call_id")
        tool_name = self._field(proposal, "tool_name")
        arguments = self._field(proposal, "arguments")
        if type(call_id) is not str or not call_id.strip():
            raise AgentDecisionValidationError("ToolCall.call_id must be a non-empty string")
        if type(tool_name) is not str or not tool_name.strip():
            raise AgentDecisionValidationError("ToolCall.tool_name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise AgentDecisionValidationError("ToolCall.arguments must be a mapping")
        try:
            canonical_arguments = canonical_snapshot(arguments)
        except CanonicalizationError as exc:
            raise AgentDecisionValidationError(f"ToolCall.arguments are not canonical: {exc}") from exc
        return ToolCall(call_id.strip(), tool_name.strip(), canonical_arguments)

    def _descriptor(self, proposal: object) -> GoalDescriptor | None:
        if proposal is None:
            return None
        if type(proposal) is not GoalDescriptor:
            raise AgentDecisionValidationError("proposed_goal_descriptor must be GoalDescriptor or None")
        descriptor_version = self._field(proposal, "descriptor_version")
        goals = self._field(proposal, "goals")
        if type(descriptor_version) is not int or descriptor_version < 1:
            raise AgentDecisionValidationError("descriptor_version must be a positive integer")
        if type(goals) not in (list, tuple) or not goals:
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
            goal_id = self._text(self._field(goal, "goal_id"), "goal_id")
            if goal_id in seen:
                raise AgentDecisionValidationError("GoalDescriptor goal_id values must be unique")
            seen.add(goal_id)
            try:
                if type(goal) is ReadTaskState:
                    self._expected_goal_kind(goal, "READ_TASK_STATE")
                    normalized.append(
                        ReadTaskState(goal_id, self._text(self._field(goal, "target"), "target"))
                    )
                elif type(goal) is DiagnoseTask:
                    self._expected_goal_kind(goal, "DIAGNOSE_TASK")
                    normalized.append(
                        DiagnoseTask(goal_id, self._text(self._field(goal, "target"), "target"))
                    )
                elif type(goal) is ExplainKnowledge:
                    self._expected_goal_kind(goal, "EXPLAIN_KNOWLEDGE")
                    normalized.append(
                        ExplainKnowledge(goal_id, self._text(self._field(goal, "topic"), "topic"))
                    )
                elif type(goal) is InspectGPU:
                    self._expected_goal_kind(goal, "INSPECT_GPU")
                    normalized.append(InspectGPU(goal_id))
                else:
                    self._expected_goal_kind(goal, "INSPECT_QUEUE")
                    target = self._field(goal, "target")
                    if target is not None:
                        target = self._text(target, "target")
                    normalized.append(InspectQueue(goal_id, target))
            except Exception as exc:
                raise AgentDecisionValidationError(f"invalid GoalDescriptor goal: {exc}") from exc
        try:
            return GoalDescriptor(descriptor_version, tuple(normalized))
        except Exception as exc:
            raise AgentDecisionValidationError(f"invalid GoalDescriptor: {exc}") from exc

    @staticmethod
    def _field(value: object, name: str) -> object:
        try:
            return getattr(value, name)
        except Exception as exc:
            raise AgentDecisionValidationError(
                f"malformed proposal is missing or cannot expose field {name!r}"
            ) from exc

    @classmethod
    def _expected_kind(cls, proposal: object, expected: str) -> None:
        value = cls._field(proposal, "kind")
        if type(value) is not str or value != expected:
            raise AgentDecisionValidationError(f"decision kind must be {expected}")

    @classmethod
    def _expected_goal_kind(cls, goal: object, expected: str) -> None:
        value = cls._field(goal, "kind")
        if type(value) is not str or value != expected:
            raise AgentDecisionValidationError(f"goal kind must be {expected}")

    @staticmethod
    def _text(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AgentDecisionValidationError(f"{field} must be a non-empty string")
        return value.strip()
