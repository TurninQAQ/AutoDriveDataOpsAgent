"""Frozen WRITE proposal and explicit lifecycle state."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from ..agent.immutable import FrozenMapping, canonical_snapshot


class WriteTransactionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    REJECTED = "REJECTED"
    APPROVED = "APPROVED"
    REVALIDATING = "REVALIDATING"
    INVALIDATED = "INVALIDATED"
    INVALIDATED_GOAL_CHANGED = "INVALIDATED_GOAL_CHANGED"
    EXECUTING = "EXECUTING"
    FAILED = "FAILED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    EXECUTED = "EXECUTED"
    VERIFYING = "VERIFYING"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    VERIFIED = "VERIFIED"


class MutationOutcome(str, Enum):
    FAILED_BEFORE_EFFECT = "FAILED_BEFORE_EFFECT"
    CONFIRMED_SUCCESS = "CONFIRMED_SUCCESS"
    CONFIRMED_FAILURE = "CONFIRMED_FAILURE"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


@dataclass(frozen=True)
class FrozenToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        object.__setattr__(self, "call_id", self.call_id.strip())
        object.__setattr__(self, "tool_name", self.tool_name.strip())
        object.__setattr__(self, "arguments", canonical_snapshot(self.arguments))


@dataclass(frozen=True)
class PreconditionSnapshot:
    target: str
    tool_name: str
    observed_at: datetime
    fingerprint: str
    entity_version: str | None
    state: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", canonical_snapshot(self.state))


@dataclass(frozen=True)
class MutationResult:
    outcome: MutationOutcome
    data: Mapping[str, object]
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", canonical_snapshot(self.data))


@dataclass(frozen=True)
class ReconciliationState:
    status: str
    checked_at: datetime | None = None
    detail: Mapping[str, object] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", canonical_snapshot(self.detail))


@dataclass(frozen=True)
class WriteTransaction:
    transaction_id: str
    proposal: FrozenToolCall
    fingerprint: str
    bound_goal_ids: tuple[str, ...]
    goal_descriptor_version: int
    completion_contract_fingerprint: str
    bound_goal_contract_fingerprint: str
    status: WriteTransactionStatus
    approval_request_id: str
    approval: object | None = None
    precondition: PreconditionSnapshot | None = None
    execution_claim: object | None = None
    execution_attempt_id: str | None = None
    mutation_result: MutationResult | None = None
    action_verification: object | None = None
    operational_goal_verification: object | None = None
    affected_entities: tuple[str, ...] = ()
    reconciliation: ReconciliationState | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bound_goal_ids", tuple(self.bound_goal_ids))
        object.__setattr__(self, "affected_entities", tuple(self.affected_entities))
        if not self.bound_goal_ids:
            raise ValueError("WriteTransaction must bind at least one goal")

    def transition(self, status: WriteTransactionStatus, **changes: Any) -> "WriteTransaction":
        return replace(self, status=status, **changes)

    def agent_projection(self) -> FrozenMapping[str, object]:
        """Safe provider-facing state: never expose approval/claim capabilities."""
        return FrozenMapping({
            "transaction_id": self.transaction_id,
            "tool_name": self.proposal.tool_name,
            "arguments": self.proposal.arguments,
            "fingerprint": self.fingerprint,
            "bound_goal_ids": self.bound_goal_ids,
            "goal_descriptor_version": self.goal_descriptor_version,
            "completion_contract_fingerprint": self.completion_contract_fingerprint,
            "bound_goal_contract_fingerprint": self.bound_goal_contract_fingerprint,
            "status": self.status.value,
            "affected_entities": self.affected_entities,
            "mutation_outcome": self.mutation_result.outcome.value if self.mutation_result else None,
            "action_verification_status": getattr(getattr(self.action_verification, "status", None), "value", None),
            "operational_goal_verification_status": getattr(getattr(self.operational_goal_verification, "status", None), "value", None),
            "reconciliation_status": self.reconciliation.status if self.reconciliation else None,
        })

    def audit_projection(self) -> FrozenMapping[str, object]:
        return FrozenMapping({
            **dict(self.agent_projection()),
            "approval_request_id": self.approval_request_id,
            "approval_id": getattr(self.approval, "approval_id", None),
            "execution_claim_id": getattr(self.execution_claim, "claim_id", None),
            "execution_attempt_id": self.execution_attempt_id,
            "precondition_fingerprint": self.precondition.fingerprint if self.precondition else None,
        })
