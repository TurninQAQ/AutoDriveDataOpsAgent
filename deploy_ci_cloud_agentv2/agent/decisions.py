"""Typed decisions emitted by the Agent provider."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Union

from .goals import GoalDescriptor
from .immutable import CanonicalizationError, FrozenMapping, canonical_snapshot


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        # This is still an untrusted provider proposal.  Freeze ordinary
        # JSON-like arguments for the common path, but retain malformed input
        # only as proposal data so the ingress validator can reject it with a
        # bounded decision error instead of leaking a constructor TypeError.
        if isinstance(self.arguments, Mapping):
            try:
                object.__setattr__(self, "arguments", canonical_snapshot(self.arguments))
            except CanonicalizationError:
                pass


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
        if isinstance(self.arguments, Mapping):
            try:
                object.__setattr__(self, "arguments", canonical_snapshot(self.arguments))
            except CanonicalizationError:
                # A manually constructed object is not a capability.  The
                # executor performs the defensive assertion before use.
                pass


@dataclass(frozen=True)
class AcceptedWriteCall:
    """Runtime-owned normalized WRITE proposal; not an execution capability."""
    call_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if isinstance(self.arguments, Mapping):
            try:
                object.__setattr__(self, "arguments", canonical_snapshot(self.arguments))
            except CanonicalizationError:
                pass


@dataclass(frozen=True)
class SingleToolCall:
    call: ToolCall | AcceptedToolCall | AcceptedWriteCall
    proposed_goal_descriptor: GoalDescriptor | None = None
    kind: str = "SINGLE_TOOL_CALL"


@dataclass(frozen=True)
class ReadToolBatch:
    calls: tuple[ToolCall | AcceptedToolCall, ...]
    proposed_goal_descriptor: GoalDescriptor | None = None
    kind: str = "READ_TOOL_BATCH"

    def __post_init__(self) -> None:
        # Shape validation belongs to AgentDecisionIngressValidator.  Copy
        # ordinary proposal sequences, but do not let malformed provider data
        # fail in a Python constructor before the Runtime can classify it.
        if isinstance(self.calls, (list, tuple)):
            object.__setattr__(self, "calls", tuple(self.calls))


@dataclass(frozen=True)
class FinalCandidate:
    response: str
    proposed_goal_descriptor: GoalDescriptor | None = None
    referenced_goal_ids: tuple[str, ...] = ()
    kind: str = "FINAL_CANDIDATE"

    def __post_init__(self) -> None:
        # FinalCandidate is also an untrusted proposal.  The ingress validator
        # owns exact type/non-empty checks; preserving malformed fields here
        # allows the graph to emit a bounded rejection and continue.
        if isinstance(self.referenced_goal_ids, (list, tuple)):
            object.__setattr__(self, "referenced_goal_ids", tuple(self.referenced_goal_ids))


AgentDecision = Union[SingleToolCall, ReadToolBatch, FinalCandidate]
