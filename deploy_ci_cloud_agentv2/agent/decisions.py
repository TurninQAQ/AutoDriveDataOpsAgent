"""Typed decisions emitted by the Agent provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .goals import GoalDescriptor


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("ToolCall requires call_id and tool_name")
        if not isinstance(self.arguments, dict):
            raise TypeError("ToolCall arguments must be a concrete dict")


@dataclass(frozen=True)
class SingleToolCall:
    call: ToolCall
    proposed_goal_descriptor: GoalDescriptor | None = None
    kind: str = "SINGLE_TOOL_CALL"


@dataclass(frozen=True)
class ReadToolBatch:
    calls: tuple[ToolCall, ...]
    proposed_goal_descriptor: GoalDescriptor | None = None
    kind: str = "READ_TOOL_BATCH"

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("ReadToolBatch requires at least one call")


@dataclass(frozen=True)
class FinalCandidate:
    response: str
    proposed_goal_descriptor: GoalDescriptor | None = None
    referenced_goal_ids: tuple[str, ...] = ()
    kind: str = "FINAL_CANDIDATE"

    def __post_init__(self) -> None:
        if not self.response.strip():
            raise ValueError("FinalCandidate response must not be empty")


AgentDecision = Union[SingleToolCall, ReadToolBatch, FinalCandidate]
