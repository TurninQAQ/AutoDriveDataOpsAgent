"""Deterministic completion requirements compiled from structured goals."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .goals import (
    DeleteTask, DiagnoseTask, ExplainKnowledge, Goal, GoalDescriptor, InspectGPU,
    InspectQueue, ReadTaskState, ResumeTask, SetTaskPriority, StopTask, SubmitTask,
)
from .immutable import FrozenMapping, register_canonical_dataclass


class RequirementKind(str, Enum):
    TARGET_BINDING = "TARGET_BINDING"
    LIVE_TASK = "LIVE_TASK"
    GPU_POOL = "GPU_POOL"
    QUEUE_STATE = "QUEUE_STATE"
    KNOWLEDGE = "KNOWLEDGE"
    DIAGNOSTIC_CONTEXT = "DIAGNOSTIC_CONTEXT"
    ACTION_VERIFIED = "ACTION_VERIFIED"
    OPERATIONAL_GOAL_VERIFIED = "OPERATIONAL_GOAL_VERIFIED"


@dataclass(frozen=True)
class CompletionRequirement:
    kind: RequirementKind
    target: str

register_canonical_dataclass(CompletionRequirement)


@dataclass(frozen=True)
class CompletionContract:
    descriptor_version: int
    contract_version: int
    contract_fingerprint: str
    requirements_by_goal: Mapping[str, tuple[CompletionRequirement, ...]]

    def __post_init__(self) -> None:
        normalized = FrozenMapping({
            str(goal_id): tuple(requirements)
            for goal_id, requirements in self.requirements_by_goal.items()
        })
        object.__setattr__(self, "requirements_by_goal", normalized)


class CompletionContractCompiler:
    """Map GoalDescriptor types to fixed completion requirements only."""
    def compile(self, descriptor: GoalDescriptor) -> CompletionContract:
        requirements = {goal.goal_id: self._requirements_for(goal) for goal in descriptor.goals}
        canonical = {
            "descriptor_version": descriptor.descriptor_version,
            "requirements": {
                goal_id: [{"kind": item.kind.value, "target": item.target} for item in requirements[goal_id]]
                for goal_id in sorted(requirements)
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CompletionContract(
            descriptor_version=descriptor.descriptor_version,
            contract_version=descriptor.descriptor_version,
            contract_fingerprint=fingerprint,
            requirements_by_goal=requirements,
        )

    @staticmethod
    def _requirements_for(goal: Goal) -> tuple[CompletionRequirement, ...]:
        if isinstance(goal, ReadTaskState):
            return (CompletionRequirement(RequirementKind.TARGET_BINDING, goal.target), CompletionRequirement(RequirementKind.LIVE_TASK, goal.target))
        if isinstance(goal, DiagnoseTask):
            return (
                CompletionRequirement(RequirementKind.TARGET_BINDING, goal.target),
                CompletionRequirement(RequirementKind.LIVE_TASK, goal.target),
                CompletionRequirement(RequirementKind.DIAGNOSTIC_CONTEXT, goal.target),
            )
        if isinstance(goal, ExplainKnowledge):
            return (CompletionRequirement(RequirementKind.KNOWLEDGE, goal.topic),)
        if isinstance(goal, InspectGPU):
            return (CompletionRequirement(RequirementKind.GPU_POOL, "platform"),)
        if isinstance(goal, InspectQueue):
            return (CompletionRequirement(RequirementKind.QUEUE_STATE, goal.target or "platform"),)
        if isinstance(goal, DeleteTask):
            return (
                CompletionRequirement(RequirementKind.TARGET_BINDING, goal.target),
                CompletionRequirement(RequirementKind.ACTION_VERIFIED, goal.target),
            )
        if isinstance(goal, (ResumeTask, StopTask, SetTaskPriority, SubmitTask)):
            return (
                CompletionRequirement(RequirementKind.TARGET_BINDING, goal.target),
                CompletionRequirement(RequirementKind.ACTION_VERIFIED, goal.target),
                CompletionRequirement(RequirementKind.OPERATIONAL_GOAL_VERIFIED, goal.target),
            )
        raise TypeError(f"Unsupported GoalDescriptor goal: {type(goal).__name__}")
