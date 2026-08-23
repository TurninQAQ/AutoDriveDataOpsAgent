"""Human approval binding and resume-input validation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import threading
import uuid

from .write_transaction import WriteTransaction, WriteTransactionStatus


class ApprovalDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ApprovalInterrupt:
    approval_request_id: str
    transaction_id: str
    fingerprint: str
    tool_name: str
    arguments: object
    bound_goal_ids: tuple[str, ...]
    risk: str

    @classmethod
    def from_transaction(cls, transaction: WriteTransaction, risk: str) -> "ApprovalInterrupt":
        return cls(
            approval_request_id=transaction.approval_request_id,
            transaction_id=transaction.transaction_id,
            fingerprint=transaction.fingerprint,
            tool_name=transaction.proposal.tool_name,
            arguments=transaction.proposal.arguments,
            bound_goal_ids=transaction.bound_goal_ids,
            risk=risk,
        )


@dataclass(frozen=True)
class ResumeInput:
    decision: ApprovalDecision | str
    approval_request_id: str
    transaction_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        decision = self.decision if isinstance(self.decision, ApprovalDecision) else ApprovalDecision(str(self.decision).upper())
        object.__setattr__(self, "decision", decision)
        for field_name in ("approval_request_id", "transaction_id", "fingerprint"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip())


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    approval_request_id: str
    transaction_id: str
    fingerprint: str
    operator_id: str
    trust_domain: str
    decision: ApprovalDecision
    approved_at: datetime




class ApprovalRecordConflict(RuntimeError):
    """The same approval request was replayed with different authorization facts."""


class ApprovalRecordStore:
    """In-memory idempotent approval authority keyed by approval_request_id."""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def record(self, candidate: ApprovalRecord) -> ApprovalRecord:
        with self._lock:
            existing = self._records.get(candidate.approval_request_id)
            if existing is None:
                self._records[candidate.approval_request_id] = candidate
                return candidate
            if not _same_approval_authority(existing, candidate):
                raise ApprovalRecordConflict(
                    "approval request was already resolved with different authorization facts"
                )
            return existing

    def get(self, approval_request_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(approval_request_id)


def _same_approval_authority(left: ApprovalRecord, right: ApprovalRecord) -> bool:
    return (
        left.approval_request_id == right.approval_request_id
        and left.transaction_id == right.transaction_id
        and left.fingerprint == right.fingerprint
        and left.operator_id == right.operator_id
        and left.trust_domain == right.trust_domain
        and left.decision is right.decision
    )


class ApprovalValidator:
    @staticmethod
    def validate_binding(
        transaction: WriteTransaction,
        pending: ApprovalInterrupt,
        resume_input: ResumeInput,
        *,
        operator_id: str,
        trust_domain: str,
    ) -> None:
        """Validate the host resume payload without minting an ApprovalRecord."""
        if transaction.status is not WriteTransactionStatus.PENDING_APPROVAL:
            raise ValueError("transaction is not pending approval")
        if pending.approval_request_id != transaction.approval_request_id:
            raise ValueError("pending interrupt does not match transaction approval request")
        if pending.transaction_id != transaction.transaction_id or pending.fingerprint != transaction.fingerprint:
            raise ValueError("pending interrupt does not match frozen transaction")
        if pending.tool_name != transaction.proposal.tool_name or pending.arguments != transaction.proposal.arguments:
            raise ValueError("pending interrupt display content does not match frozen transaction")
        if resume_input.approval_request_id != transaction.approval_request_id:
            raise ValueError("resume approval_request_id mismatch")
        if resume_input.transaction_id != transaction.transaction_id:
            raise ValueError("resume transaction_id mismatch")
        if resume_input.fingerprint != transaction.fingerprint:
            raise ValueError("resume fingerprint mismatch")
        if not operator_id or not trust_domain:
            raise ValueError("trusted operator identity is required")

    @staticmethod
    def validate_resume(
        transaction: WriteTransaction,
        pending: ApprovalInterrupt,
        resume_input: ResumeInput,
        *,
        operator_id: str,
        trust_domain: str,
    ) -> ApprovalRecord:
        ApprovalValidator.validate_binding(
            transaction, pending, resume_input, operator_id=operator_id, trust_domain=trust_domain
        )
        return ApprovalRecord(
            approval_id=f"apr_{uuid.uuid4().hex}",
            approval_request_id=transaction.approval_request_id,
            transaction_id=transaction.transaction_id,
            fingerprint=transaction.fingerprint,
            operator_id=operator_id,
            trust_domain=trust_domain,
            decision=resume_input.decision,
            approved_at=datetime.now(timezone.utc),
        )
