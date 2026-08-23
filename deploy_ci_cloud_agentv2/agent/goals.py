"""Agent-declared semantic goals.

Goal types describe user intent only.  They intentionally do not contain
completion, approval, verification, or workflow authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Union

from .immutable import FrozenMapping, canonical_snapshot


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


@dataclass(frozen=True)
class ResumeTask:
    goal_id: str
    target: str
    kind: str = "RESUME_TASK"
    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))


@dataclass(frozen=True)
class StopTask:
    goal_id: str
    target: str
    kind: str = "STOP_TASK"
    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))


@dataclass(frozen=True)
class DeleteTask:
    goal_id: str
    target: str
    kind: str = "DELETE_TASK"
    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))


@dataclass(frozen=True)
class SetTaskPriority:
    goal_id: str
    target: str
    priority: int
    kind: str = "SET_TASK_PRIORITY"
    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")


@dataclass(frozen=True)
class SubmitTask:
    goal_id: str
    target: str
    config: Mapping[str, object] = field(default_factory=FrozenMapping)
    kind: str = "SUBMIT_TASK"
    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _required_text(self.goal_id, "goal_id"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))
        if not isinstance(self.config, Mapping):
            raise ValueError("config must be a mapping")
        object.__setattr__(self, "config", canonical_snapshot(self.config))


Goal = Union[
    ReadTaskState, InspectGPU, InspectQueue, ExplainKnowledge, DiagnoseTask,
    ResumeTask, StopTask, DeleteTask, SetTaskPriority, SubmitTask,
]

_GOAL_TYPES = (
    ReadTaskState, InspectGPU, InspectQueue, ExplainKnowledge, DiagnoseTask,
    ResumeTask, StopTask, DeleteTask, SetTaskPriority, SubmitTask,
)


@dataclass(frozen=True)
class GoalDescriptor:
    descriptor_version: int
    goals: tuple[Goal, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(self.goals))
        if type(self.descriptor_version) is not int or self.descriptor_version < 1:
            raise ValueError("descriptor_version must be positive")
        if any(type(goal) not in _GOAL_TYPES for goal in self.goals):
            raise TypeError("GoalDescriptor contains an unsupported goal type")
        ids = [goal.goal_id.strip() if isinstance(goal.goal_id, str) else goal.goal_id for goal in self.goals]
        if not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ValueError("GoalDescriptor requires at least one non-empty goal_id")
        if len(ids) != len(set(ids)):
            raise ValueError("GoalDescriptor goal_id values must be unique")

    def goal(self, goal_id: str) -> Goal:
        for item in self.goals:
            if item.goal_id == goal_id:
                return item
        raise KeyError(goal_id)

    def to_dict(self) -> dict:
        rows = []
        for goal in self.goals:
            row = {"goal_id": goal.goal_id, "kind": goal.kind}
            if hasattr(goal, "target"):
                row["target"] = getattr(goal, "target")
            if isinstance(goal, ExplainKnowledge):
                row["topic"] = goal.topic
            if isinstance(goal, SetTaskPriority):
                row["priority"] = goal.priority
            if isinstance(goal, SubmitTask):
                row["config"] = goal.config
            rows.append(row)
        return {"descriptor_version": self.descriptor_version, "goals": rows}


def write_goal_tool_name(goal: Goal) -> str | None:
    return {
        ResumeTask: "resume_task",
        StopTask: "stop_task",
        DeleteTask: "delete_task",
        SetTaskPriority: "set_task_priority",
        SubmitTask: "submit_task",
    }.get(type(goal))


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 256:
        raise ValueError(f"{name} exceeds the maximum supported length")
    return normalized
