"""Deterministic direct-action verification; no semantic tool choice."""
from __future__ import annotations

from collections.abc import Mapping

from ..agent.immutable import canonical_snapshot
from ..agent.results import ResultStatus, TaskState, normalize_read_result
from ..safety.write_transaction import WriteTransaction
from .results import VerificationResult, VerificationStatus


class ActionVerifier:
    """Verify the direct expected effect of the frozen WRITE on its exact target."""

    def __init__(self, read_facade, registry=None):
        self.read_facade = read_facade
        self.registry = registry

    def verify(self, transaction: WriteTransaction) -> VerificationResult:
        self._assert_contract(transaction)
        target = _verification_target(transaction)
        raw = canonical_snapshot(self.read_facade.get_task_detail(target))
        result = normalize_read_result("get_task_detail", {"task_name": target}, raw)
        tool = transaction.proposal.tool_name

        if tool == "delete_task":
            ok = (
                result.envelope.status in {ResultStatus.NOT_FOUND, ResultStatus.NO_DATA}
                or result.exists is False
            )
        elif tool == "resume_task":
            ok = result.qualifies_for_evidence() and result.state in {
                TaskState.QUEUED,
                TaskState.SUBMITTED,
                TaskState.RUNNING,
                TaskState.SUCCEEDED,
                TaskState.COMPLETED,
            }
        elif tool == "stop_task":
            ok = result.qualifies_for_evidence() and result.state in {
                TaskState.STOPPED,
                TaskState.CANCELLED,
            }
        elif tool == "set_task_priority":
            ok = (
                result.qualifies_for_evidence()
                and result.metadata.get("priority") == transaction.proposal.arguments.get("priority")
            )
        elif tool == "submit_task":
            ok = result.qualifies_for_evidence() and result.task_name == target
        else:
            ok = False

        return VerificationResult(
            VerificationStatus.VERIFIED if ok else VerificationStatus.FAILED,
            target,
            "DIRECT_MUTATION_CONFIRMED" if ok else "DIRECT_MUTATION_NOT_CONFIRMED",
        )

    def _assert_contract(self, transaction: WriteTransaction) -> None:
        if self.registry is None:
            return
        spec = self.registry.spec(transaction.proposal.tool_name)
        if spec.verification_reads != ("get_task_detail",):
            raise ValueError("ActionVerifier requires the predeclared get_task_detail verification read")


def _verification_target(transaction: WriteTransaction) -> str:
    """Resolve a deterministic platform-generated target for submit_task.

    The V2 proposal carries a validated task prefix for submission, while the
    platform creates the concrete task identity.  The mutation result is the
    only accepted source for that generated identity; all other WRITE tools
    continue to verify their frozen affected entity.
    """

    target = transaction.affected_entities[0]
    if transaction.proposal.tool_name != "submit_task" or transaction.mutation_result is None:
        return target
    result = transaction.mutation_result.data.get("result")
    if isinstance(result, Mapping):
        generated = result.get("task_name")
        if isinstance(generated, str) and generated.strip():
            return generated.strip()
    return target
