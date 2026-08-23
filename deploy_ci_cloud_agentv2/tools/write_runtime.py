"""Exactly-once-per-approval WRITE mutation boundary."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..agent.immutable import canonical_snapshot
from ..safety.locks import ExecutionClaim, ExecutionClaimStore
from ..safety.write_transaction import MutationOutcome, MutationResult, WriteTransaction
from .metadata import ToolKind
from .registry import ToolCatalogIntegrityError, ToolRegistry


class MutationOutcomeUnknown(RuntimeError):
    """The host cannot determine whether the external mutation took effect."""


class MutationFailedBeforeEffect(RuntimeError):
    """The host knows the mutation did not start/take effect."""


class WriteToolRuntime:
    def __init__(self, registry: ToolRegistry, claim_store: ExecutionClaimStore, expected_catalog_hash: str):
        self.registry = registry
        self.claim_store = claim_store
        self.expected_catalog_hash = expected_catalog_hash

    async def execute_once(self, transaction: WriteTransaction, claim: ExecutionClaim, *, on_started=None) -> tuple[str, MutationResult]:
        if self.registry.catalog_hash() != self.expected_catalog_hash:
            raise ToolCatalogIntegrityError("sealed tool catalog hash does not match Runtime context")
        spec = self.registry.spec(transaction.proposal.tool_name)
        if spec.kind is not ToolKind.WRITE:
            raise ValueError("mutation runtime requires WRITE ToolSpec")
        # This atomic consume is the final execution capability boundary. The
        # same claim cannot cross the external mutation boundary twice.
        attempt_id = self.claim_store.consume_attempt(claim)
        if on_started is not None:
            maybe = on_started(attempt_id)
            if hasattr(maybe, "__await__"):
                await maybe
        started_transaction = transaction.transition(
            transaction.status, execution_attempt_id=attempt_id
        )
        mutation = await self.execute_started_once(started_transaction, claim, attempt_id)
        return attempt_id, mutation

    async def execute_started_once(
        self,
        transaction: WriteTransaction,
        claim: ExecutionClaim,
        attempt_id: str,
    ) -> MutationResult:
        """Invoke one already-durably-started mutation attempt exactly once.

        The caller must have consumed ``attempt_id`` from the authoritative
        claim store and durably recorded ``MutationStarted`` before entering
        this method. This split lets SQLite commit capability consumption and
        the audit event in one transaction before any external side effect.
        """
        if self.registry.catalog_hash() != self.expected_catalog_hash:
            raise ToolCatalogIntegrityError("sealed tool catalog hash does not match Runtime context")
        spec = self.registry.spec(transaction.proposal.tool_name)
        if spec.kind is not ToolKind.WRITE:
            raise ValueError("mutation runtime requires WRITE ToolSpec")
        if transaction.execution_claim != claim:
            raise ValueError("transaction does not carry the authoritative ExecutionClaim")
        if transaction.execution_attempt_id != attempt_id:
            raise ValueError("transaction does not carry the authoritative mutation attempt")
        try:
            result = await self.registry.call(
                transaction.proposal,
                runtime_precondition=transaction.precondition,
            )
            snapshot = canonical_snapshot(result)
            ok = bool(snapshot.get("ok")) if hasattr(snapshot, "get") else False
            outcome = MutationOutcome.CONFIRMED_SUCCESS if ok else MutationOutcome.CONFIRMED_FAILURE
            error_code = snapshot.get("error_code") if hasattr(snapshot, "get") else None
            return MutationResult(outcome, snapshot if hasattr(snapshot, "items") else {}, error_code)
        except MutationFailedBeforeEffect as exc:
            return MutationResult(MutationOutcome.FAILED_BEFORE_EFFECT, {}, type(exc).__name__)
        except MutationOutcomeUnknown as exc:
            return MutationResult(MutationOutcome.OUTCOME_UNKNOWN, {}, type(exc).__name__)
        except Exception as exc:
            # An arbitrary exception after entering the mutation handler is
            # conservatively unknown; blindly retrying could duplicate effect.
            return MutationResult(MutationOutcome.OUTCOME_UNKNOWN, {}, type(exc).__name__)
