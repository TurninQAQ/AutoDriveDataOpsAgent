"""Deterministic operational-goal verification over a predeclared post-state read."""
from __future__ import annotations

from collections.abc import Mapping

from ..agent.immutable import canonical_snapshot
from ..agent.results import TaskState, normalize_read_result
from ..safety.write_transaction import WriteTransaction
from .results import VerificationResult, VerificationStatus


class OperationalGoalVerifier:
    def __init__(self, read_facade, registry=None):
        self.read_facade = read_facade
        self.registry = registry

    def verify(self, transaction: WriteTransaction) -> VerificationResult:
        self._assert_contract(transaction)
        target = _verification_target(transaction)
        raw = canonical_snapshot(self.read_facade.get_task_detail(target))
        result = normalize_read_result("get_task_detail", {"task_name": target}, raw)
        tool = transaction.proposal.tool_name
        ok = False
        if result.is_valid:
            if tool == "resume_task":
                ok = result.state in {
                    TaskState.QUEUED,
                    TaskState.SUBMITTED,
                    TaskState.RUNNING,
                    TaskState.SUCCEEDED,
                    TaskState.COMPLETED,
                }
            elif tool == "stop_task":
                ok = result.state in {TaskState.STOPPED, TaskState.CANCELLED}
            elif tool == "set_task_priority":
                ok = (
                    result.qualifies_for_evidence()
                    and result.priority == transaction.proposal.arguments.get("priority")
                )
            elif tool == "submit_task":
                ok = result.qualifies_for_evidence()
            elif tool == "delete_task":
                ok = True
        return VerificationResult(
            VerificationStatus.VERIFIED if ok else VerificationStatus.FAILED,
            target,
            "OPERATIONAL_GOAL_CONFIRMED" if ok else "OPERATIONAL_GOAL_NOT_CONFIRMED",
        )

    def _assert_contract(self, transaction: WriteTransaction) -> None:
        if self.registry is None:
            return
        spec = self.registry.spec(transaction.proposal.tool_name)
        if spec.verification_reads != ("get_task_detail",):
            raise ValueError("OperationalGoalVerifier requires the predeclared get_task_detail verification read")


def _verification_target(transaction: WriteTransaction) -> str:
    """Use the platform-generated task identity only after confirmed success."""

    target = transaction.affected_entities[0]
    if transaction.proposal.tool_name != "submit_task" or transaction.mutation_result is None:
        return target
    result = transaction.mutation_result.data.get("result")
    if isinstance(result, Mapping):
        generated = result.get("task_name")
        if isinstance(generated, str) and generated.strip():
            return generated.strip()
    return target
