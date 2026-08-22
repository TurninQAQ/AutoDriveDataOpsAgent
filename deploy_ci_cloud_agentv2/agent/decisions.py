"""Typed decisions emitted by the Agent provider."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Union

from .goals import GoalDescriptor
from .immutable import FrozenMapping


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("ToolCall requires call_id and tool_name")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("ToolCall arguments must be a concrete mapping")
        object.__setattr__(self, "arguments", FrozenMapping(self.arguments))


@dataclass(frozen=True)
class AcceptedToolCall:
    """Runtime-owned normalized call accepted by the READ guard.

    A provider ``ToolCall`` is a proposal.  This type is created only after
    schema and semantic argument validation.  Execution accepts this type so
    it cannot silently re-read or reconstruct provider-owned arguments.
    """

    call_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("AcceptedToolCall requires call_id and tool_name")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("AcceptedToolCall arguments must be a mapping")
        object.__setattr__(self, "arguments", FrozenMapping(self.arguments))


@dataclass(frozen=True)
class SingleToolCall:
    call: ToolCall
    proposed_goal_descriptor: GoalDescriptor | None = None
    kind: str = "SINGLE_TOOL_CALL"


@dataclass(frozen=True)
class ReadToolBatch:
    calls: tuple[ToolCall | AcceptedToolCall, ...]
    proposed_goal_descriptor: GoalDescriptor | None = None
    kind: str = "READ_TOOL_BATCH"

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("ReadToolBatch requires at least one call")
        calls = tuple(self.calls)
        if not all(isinstance(call, (ToolCall, AcceptedToolCall)) for call in calls):
            raise TypeError("ReadToolBatch calls must be ToolCall values")
        object.__setattr__(self, "calls", calls)


@dataclass(frozen=True)
class FinalCandidate:
    response: str
    proposed_goal_descriptor: GoalDescriptor | None = None
    referenced_goal_ids: tuple[str, ...] = ()
    kind: str = "FINAL_CANDIDATE"

    def __post_init__(self) -> None:
        if not self.response.strip():
            raise ValueError("FinalCandidate response must not be empty")
        object.__setattr__(self, "referenced_goal_ids", tuple(self.referenced_goal_ids))


AgentDecision = Union[SingleToolCall, ReadToolBatch, FinalCandidate]
