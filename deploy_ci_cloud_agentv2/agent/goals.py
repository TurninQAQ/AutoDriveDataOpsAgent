"""Agent-declared semantic goals.

The classes in this module describe user intent only.  They intentionally do not
contain evidence, workflow, verification, or completion-policy fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ReadTaskState:
    goal_id: str
    target: str
    kind: str = "READ_TASK_STATE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))


@dataclass(frozen=True)
class InspectGPU:
    goal_id: str
    kind: str = "INSPECT_GPU"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))


@dataclass(frozen=True)
class InspectQueue:
    goal_id: str
    target: str | None = None
    kind: str = "INSPECT_QUEUE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        if self.target is not None:
            object.__setattr__(self, "target", _required_text(self.target, "target"))


@dataclass(frozen=True)
class ExplainKnowledge:
    goal_id: str
    topic: str
    kind: str = "EXPLAIN_KNOWLEDGE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "topic", _required_text(self.topic, "topic"))


@dataclass(frozen=True)
class DiagnoseTask:
    goal_id: str
    target: str
    kind: str = "DIAGNOSE_TASK"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))


Goal = Union[ReadTaskState, InspectGPU, InspectQueue, ExplainKnowledge, DiagnoseTask]


@dataclass(frozen=True)
class GoalDescriptor:
    descriptor_version: int
    goals: tuple[Goal, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(self.goals))
        if self.descriptor_version < 1:
            raise ValueError("descriptor_version must be positive")
        if any(not isinstance(goal, (ReadTaskState, InspectGPU, InspectQueue, ExplainKnowledge, DiagnoseTask)) for goal in self.goals):
            raise TypeError("GoalDescriptor contains an unsupported goal type")
        ids = [goal.goal_id.strip() if isinstance(goal.goal_id, str) else goal.goal_id for goal in self.goals]
        if not ids or any(not item.strip() for item in ids):
            raise ValueError("GoalDescriptor requires at least one non-empty goal_id")
        if len(ids) != len(set(ids)):
            raise ValueError("GoalDescriptor goal_id values must be unique")
        for goal, goal_id in zip(self.goals, ids):
            if goal.goal_id != goal_id:
                object.__setattr__(goal, "goal_id", goal_id)

    def goal(self, goal_id: str) -> Goal:
        for item in self.goals:
            if item.goal_id == goal_id:
                return item
        raise KeyError(goal_id)

    def to_dict(self) -> dict:
        rows = []
        for goal in self.goals:
            row = {"goal_id": goal.goal_id, "kind": goal.kind}
            if isinstance(goal, (ReadTaskState, DiagnoseTask, InspectQueue)):
                row["target"] = goal.target
            if isinstance(goal, ExplainKnowledge):
                row["topic"] = goal.topic
            rows.append(row)
        return {"descriptor_version": self.descriptor_version, "goals": rows}


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 256:
        raise ValueError(f"{name} exceeds the maximum supported length")
    return normalized
