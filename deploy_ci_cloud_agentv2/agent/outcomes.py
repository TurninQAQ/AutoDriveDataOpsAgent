"""Per-goal outcomes and bounded runtime terminal outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .immutable import FrozenMapping


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.status is GoalStatus.SATISFIED and not self.evidence_refs:
            raise ValueError("SATISFIED GoalOutcome requires supporting evidence_refs")


class TerminalCode(str, Enum):
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    UNRECOVERABLE_RUNTIME_ERROR = "UNRECOVERABLE_RUNTIME_ERROR"
    CHECKPOINT_CORRUPTION = "CHECKPOINT_CORRUPTION"
    REQUIRES_RECONCILIATION = "REQUIRES_RECONCILIATION"


@dataclass(frozen=True)
class ControlledTerminalOutcome:
    code: TerminalCode
    safe_facts: FrozenMapping[str, Any] | dict[str, Any]
    message_template: str
    retry_allowed: bool = False
    human_action_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_facts", FrozenMapping(self.safe_facts))
