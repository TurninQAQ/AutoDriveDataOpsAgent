"""Deterministic WRITE admission policy; never a semantic planner."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WriteAdmissionPolicy:
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    protected_targets: frozenset[str] = field(default_factory=frozenset)
    version: str = "human-approved-write-v2"

    def denial_reason(self, tool_name: str, target: str) -> str | None:
        if tool_name in self.denied_tools:
            return "TOOL_DENIED_BY_POLICY"
        if tool_name == "delete_task" and target in self.protected_targets:
            return "PROTECTED_TARGET_DELETION_DENIED"
        return None
