"""Deterministic Agent capability projection derived from the sealed catalog.

This is informational context for the semantic Agent.  It is generated from
the Runtime-owned ToolRegistry and never selects a tool, infers a goal, or
grants approval/execution authority.
"""

from __future__ import annotations

from ..tools.registry import ToolRegistry
from .immutable import FrozenMapping


def build_capability_projection(registry: ToolRegistry) -> FrozenMapping[str, object]:
    """Return a small immutable capability view of the sealed catalog."""
    if not registry.is_sealed:
        raise RuntimeError("capability projection requires a sealed ToolRegistry")
    tools = []
    for spec in registry.catalog():
        tools.append(
            FrozenMapping(
                {
                    "name": spec.name,
                    "kind": spec.kind.value,
                    "arguments": spec.schema,
                    "parallel_safe": spec.parallel_safe,
                }
            )
        )
    # This is a schema projection, not an intent classifier.  The goal
    # descriptor remains an Agent proposal and is validated by Runtime.
    goal_types = FrozenMapping(
        {
            "READ_TASK_STATE": ("goal_id", "target"),
            "INSPECT_GPU": ("goal_id",),
            "INSPECT_QUEUE": ("goal_id", "target?"),
            "EXPLAIN_KNOWLEDGE": ("goal_id", "topic"),
            "DIAGNOSE_TASK": ("goal_id", "target"),
            "RESUME_TASK": ("goal_id", "target"),
            "STOP_TASK": ("goal_id", "target"),
            "DELETE_TASK": ("goal_id", "target"),
            "SET_TASK_PRIORITY": ("goal_id", "target", "priority"),
            "SUBMIT_TASK": ("goal_id", "target", "config"),
        }
    )
    return FrozenMapping(
        {
            "available_tools": tuple(tools),
            "allowed_goal_types": goal_types,
        }
    )
