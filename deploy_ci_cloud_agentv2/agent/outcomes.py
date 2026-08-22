"""Per-goal outcomes and bounded runtime terminal outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GoalStatus(str, Enum):
    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class GoalOutcome:
    goal_id: str
    status: GoalStatus = GoalStatus.PENDING
    reason_code: str | None = None
    evidence_refs: tuple[str, ...] = ()
    write_transaction_id: str | None = None


class TerminalCode(str, Enum):
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    UNRECOVERABLE_RUNTIME_ERROR = "UNRECOVERABLE_RUNTIME_ERROR"
    CHECKPOINT_CORRUPTION = "CHECKPOINT_CORRUPTION"


@dataclass(frozen=True)
class ControlledTerminalOutcome:
    code: TerminalCode
    safe_facts: dict[str, Any]
    message_template: str
    retry_allowed: bool = False
    human_action_required: bool = False
