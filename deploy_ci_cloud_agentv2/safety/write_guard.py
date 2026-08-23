"""Deterministic WRITE structural admission and transaction creation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import uuid

from ..agent.contracts import CompletionContract
from ..agent.decisions import AcceptedWriteCall
from ..agent.goals import GoalDescriptor, SetTaskPriority, SubmitTask, write_goal_tool_name
from ..agent.outcomes import GoalOutcome, GoalStatus
from ..agent.provenance import canonical_tool_call_fingerprint
from ..tools.metadata import ToolKind
from ..tools.registry import ToolRegistry
from .policy import WriteAdmissionPolicy
from .precondition import PreconditionReader, target_for_write
from .write_transaction import FrozenToolCall, WriteTransaction, WriteTransactionStatus


class WriteAdmissionOutcome(str, Enum):
    INVALID = "INVALID"
    DENIED = "DENIED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"


@dataclass(frozen=True)
class WriteAdmissionResult:
    outcome: WriteAdmissionOutcome
    reason: str | None = None
    transaction: WriteTransaction | None = None
    bound_goal_ids: tuple[str, ...] = ()


class WriteGuard:
    def __init__(self, registry: ToolRegistry, read_facade, policy: WriteAdmissionPolicy, replay_blocker=None):
        self.registry = registry
        self.preconditions = PreconditionReader(read_facade)
        self.policy = policy
        self.replay_blocker = replay_blocker

    def normalize(self, call) -> AcceptedWriteCall:
        normalized = self.registry.normalize_call(call, require_read=False)
        spec = self.registry.spec(normalized.tool_name)
        if spec.kind is not ToolKind.WRITE:
            raise ValueError(f"{normalized.tool_name} is not a WRITE tool")
        if spec.parallel_safe:
            raise ValueError("WRITE tools must not be parallel-safe")
        return AcceptedWriteCall(normalized.call_id, normalized.tool_name, normalized.arguments)

    def assess(
        self,
        call: AcceptedWriteCall,
        descriptor: GoalDescriptor | None,
        contract: CompletionContract | None,
        outcomes: dict[str, GoalOutcome] | None = None,
    ) -> WriteAdmissionResult:
        if descriptor is None or contract is None:
            return WriteAdmissionResult(WriteAdmissionOutcome.INVALID, "WRITE requires GoalDescriptor and CompletionContract")
        frozen = FrozenToolCall(call.call_id, call.tool_name, call.arguments)
        try:
            target = target_for_write(frozen)
        except ValueError as exc:
            return WriteAdmissionResult(WriteAdmissionOutcome.INVALID, str(exc))
        bound = self._bound_goal_ids(frozen, descriptor)
        if not bound:
            return WriteAdmissionResult(WriteAdmissionOutcome.INVALID, "WRITE proposal does not match a declared write goal")
        if outcomes and any(outcomes.get(goal_id, GoalOutcome(goal_id)).status is GoalStatus.REJECTED for goal_id in bound):
            return WriteAdmissionResult(
                WriteAdmissionOutcome.INVALID,
                "WRITE_GOAL_ALREADY_REJECTED",
                bound_goal_ids=bound,
            )
        if self.replay_blocker is not None and self.replay_blocker(frozen.tool_name, target):
            return WriteAdmissionResult(
                WriteAdmissionOutcome.DENIED,
                "REPLAY_BLOCKED_AFTER_UNKNOWN_OUTCOME",
                bound_goal_ids=bound,
            )
        try:
            precondition = self.preconditions.capture(frozen)
        except Exception as exc:
            return WriteAdmissionResult(
                WriteAdmissionOutcome.INVALID,
                f"PRECONDITION_INVALID: {exc}",
                bound_goal_ids=bound,
            )
        denial = self.policy.denial_reason(frozen.tool_name, target)
        if denial:
            return WriteAdmissionResult(WriteAdmissionOutcome.DENIED, denial, bound_goal_ids=bound)
        fingerprint = canonical_tool_call_fingerprint(frozen.tool_name, frozen.arguments)
        tx = WriteTransaction(
            transaction_id=f"wtx_{uuid.uuid4().hex}",
            proposal=frozen,
            fingerprint=fingerprint,
            bound_goal_ids=bound,
            goal_descriptor_version=descriptor.descriptor_version,
            completion_contract_fingerprint=contract.contract_fingerprint,
            bound_goal_contract_fingerprint=self._bound_contract_fingerprint(bound, contract),
            status=WriteTransactionStatus.PENDING_APPROVAL,
            approval_request_id=f"apreq_{uuid.uuid4().hex}",
            precondition=precondition,
            affected_entities=(target,),
        )
        return WriteAdmissionResult(
            WriteAdmissionOutcome.APPROVAL_REQUIRED,
            transaction=tx,
            bound_goal_ids=bound,
        )

    def compatible(
        self,
        transaction: WriteTransaction,
        descriptor: GoalDescriptor | None,
        contract: CompletionContract | None,
    ) -> bool:
        if descriptor is None or contract is None:
            return False
        if self._bound_goal_ids(transaction.proposal, descriptor) != transaction.bound_goal_ids:
            return False
        return self._bound_contract_fingerprint(transaction.bound_goal_ids, contract) == transaction.bound_goal_contract_fingerprint

    @staticmethod
    def _bound_goal_ids(frozen: FrozenToolCall, descriptor: GoalDescriptor) -> tuple[str, ...]:
        target = target_for_write(frozen)
        bound = []
        for goal in descriptor.goals:
            if write_goal_tool_name(goal) != frozen.tool_name or getattr(goal, "target", None) != target:
                continue
            if isinstance(goal, SetTaskPriority) and frozen.arguments.get("priority") != goal.priority:
                continue
            if isinstance(goal, SubmitTask) and frozen.arguments.get("config", {}) != goal.config:
                continue
            bound.append(goal.goal_id)
        return tuple(bound)

    @staticmethod
    def _bound_contract_fingerprint(goal_ids: tuple[str, ...], contract: CompletionContract) -> str:
        payload = []
        for goal_id in sorted(goal_ids):
            requirements = contract.requirements_by_goal.get(goal_id, ())
            payload.append((goal_id, tuple((item.kind.value, item.target) for item in requirements)))
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
