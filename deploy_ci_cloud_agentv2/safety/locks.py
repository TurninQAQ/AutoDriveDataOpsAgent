"""Single-use ExecutionClaim and mutation-attempt stores."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import uuid

from .approval import ApprovalRecord, ApprovalDecision
from .write_transaction import WriteTransaction


class ExecutionClaimAlreadyExists(RuntimeError):
    pass


class MutationAttemptAlreadyConsumed(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionClaim:
    claim_id: str
    transaction_id: str
    approval_id: str
    fingerprint: str
    claimed_at: datetime


class ExecutionClaimStore:
    def __init__(self) -> None:
        self._claims: dict[str, ExecutionClaim] = {}
        self._attempted: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, transaction: WriteTransaction, approval: ApprovalRecord) -> ExecutionClaim:
        if approval.decision is not ApprovalDecision.APPROVE:
            raise ValueError("only an approval grant may create an ExecutionClaim")
        if approval.transaction_id != transaction.transaction_id or approval.fingerprint != transaction.fingerprint:
            raise ValueError("approval does not authorize this transaction")
        with self._lock:
            if transaction.transaction_id in self._claims:
                raise ExecutionClaimAlreadyExists("execution claim already exists")
            claim = ExecutionClaim(
                claim_id=f"claim_{uuid.uuid4().hex}",
                transaction_id=transaction.transaction_id,
                approval_id=approval.approval_id,
                fingerprint=transaction.fingerprint,
                claimed_at=datetime.now(timezone.utc),
            )
            self._claims[transaction.transaction_id] = claim
            return claim

    def consume_attempt(self, claim: ExecutionClaim) -> str:
        """Atomically consume the single mutation attempt for a claim."""
        with self._lock:
            stored = self._claims.get(claim.transaction_id)
            if stored != claim:
                raise ValueError("execution claim is not authoritative")
            if claim.claim_id in self._attempted:
                raise MutationAttemptAlreadyConsumed("mutation attempt already consumed")
            self._attempted.add(claim.claim_id)
            return f"attempt_{uuid.uuid4().hex}"
