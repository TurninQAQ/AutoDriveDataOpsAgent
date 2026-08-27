from __future__ import annotations

import pytest

from deploy_ci_cloud_agentv3.mcp.client import InProcessMCPClient
from deploy_ci_cloud_agentv3.mcp.factory import build_tooling
from deploy_ci_cloud_agentv3.mcp.profiles import RUNTIME_TOOLS
from deploy_ci_cloud_agentv3.models.pending_action import PendingAction, compute_pending_action_fingerprint
from deploy_ci_cloud_agentv3.services.write_service import WriteService
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


def pending(facade: FakeFacade, priority: int = 5) -> PendingAction:
    pre = facade.get_write_precondition("task_a")
    args = {"task_name": "task_a", "priority": priority}
    fingerprint = compute_pending_action_fingerprint(action="set_task_priority", args=args, artifact=None, precondition=pre)
    return PendingAction(
        proposal_id="p1", action="set_task_priority", args=args, reason="test",
        expected_effect="priority changes", before={"priority": facade.priority},
        precondition=pre, fingerprint=fingerprint,
    )


@pytest.mark.asyncio
async def test_wrong_fingerprint_cannot_execute():
    facade = FakeFacade(); tooling = build_tooling(facade)
    service = WriteService(InProcessMCPClient(tooling.registry, RUNTIME_TOOLS))
    with pytest.raises(PermissionError):
        await service.execute(pending(facade), "wrong")
    assert facade.mutations == []


@pytest.mark.asyncio
async def test_approved_action_tamper_priority_is_blocked_before_mutation():
    facade = FakeFacade(); tooling = build_tooling(facade)
    service = WriteService(InProcessMCPClient(tooling.registry, RUNTIME_TOOLS))
    action = pending(facade, 5); approved = action.fingerprint
    action.args["priority"] = 9
    with pytest.raises(PermissionError, match="modified"):
        await service.execute(action, approved)
    assert facade.mutations == []
    assert facade.priority == 3


@pytest.mark.asyncio
async def test_stale_approval_is_blocked():
    facade = FakeFacade(); action = pending(facade)
    facade.precondition["queue_sha256"] = "changed"
    tooling = build_tooling(facade)
    result = await WriteService(InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)).execute(action, action.fingerprint)
    assert result.status == "PRECONDITION_FAILED"
    assert facade.mutations == []


@pytest.mark.asyncio
async def test_write_executes_at_most_once_per_approval_and_verifies():
    facade = FakeFacade(); tooling = build_tooling(facade)
    service = WriteService(InProcessMCPClient(tooling.registry, RUNTIME_TOOLS))
    action = pending(facade)
    first = await service.execute(action, action.fingerprint)
    second = await service.execute(action, action.fingerprint)
    assert first.status == "VERIFIED" and first.verified is True
    assert second.id == first.id
    assert len(facade.mutations) == 1

@pytest.mark.asyncio
async def test_approved_submit_nested_yaml_tamper_is_blocked_before_mutation():
    facade = FakeFacade(); tooling = build_tooling(facade)
    runtime = InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)
    config = {"max_active_runs": 1, "datasets": [{"dataset_name": "d", "dataset_path": "/approved"}]}
    args = {"artifact_id": "a1", "task_prefix": "train", "config": config}
    artifact = {"artifact_id": "a1", "task_prefix": "train", "config": config, "sha256": "x", "yaml_text": "..."}
    pre = facade.get_write_precondition("")
    fp = compute_pending_action_fingerprint(action="submit_task", args=args, artifact=artifact, precondition=pre)
    action = PendingAction(proposal_id="p-submit", action="submit_task", args=args, reason="", expected_effect="", before={}, artifact=artifact, precondition=pre, fingerprint=fp)
    approved = action.fingerprint
    action.args["config"]["datasets"][0]["dataset_path"] = "/tampered"
    with pytest.raises(PermissionError, match="modified"):
        await WriteService(runtime).execute(action, approved)
    assert facade.mutations == []


class AlwaysFalseVerification:
    async def verify(self, action, args, raw_result, *, before=None):
        return False, {"task_exists": True, "priority": 3, "errors": {}}


@pytest.mark.asyncio
async def test_idempotency_key_tamper_cannot_enable_second_mutation():
    facade = FakeFacade(); tooling = build_tooling(facade)
    runtime = InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)
    service = WriteService(runtime, verification=AlwaysFalseVerification())
    action = pending(facade, 5)
    approved = action.fingerprint

    first = await service.execute(action, approved)
    assert first.status == "VERIFICATION_FAILED"
    assert len(facade.mutations) == 1

    # Even a hostile low-level state mutation cannot mint a second attempt key:
    # WriteService derives write_{recomputed_fingerprint} internally.
    object.__setattr__(action, "idempotency_key", "write_attacker_new_key")
    second = await service.execute(action, approved)

    assert second.id == first.id
    assert second.status == "VERIFICATION_FAILED"
    assert len(facade.mutations) == 1


class ResumeTOCTOURuntime:
    def __init__(self):
        self.precondition = {
            "queue_sha256": "q1",
            "task_name": "task_a",
            "task_config_sha256": "c1",
            "task_exists": True,
            "active_task_name": "task_a",
        }
        self.snapshot = {
            "task_name": "task_a",
            "task_exists": True,
            "available_datasets": ["A"],
            "airflow_runs": [
                {"run_id": "old-failed", "dataset_name": "A", "state": "failed"},
            ],
            "errors": {},
        }
        self.mutation_calls = 0

    async def call_tool(self, name, args):
        if name == "get_action_verification_snapshot":
            return self.snapshot
        if name == "capture_write_precondition":
            return dict(self.precondition)
        if name == "resume_task":
            self.mutation_calls += 1
            return {"ok": True}
        raise AssertionError(name)


@pytest.mark.asyncio
async def test_resume_stale_failed_dataset_revalidation_blocks_duplicate_mutation():
    """Approved failed dataset must still be failed immediately before mutation."""
    from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
    from deploy_ci_cloud_agentv3.services.pending_action import PendingActionFactory

    runtime = ResumeTOCTOURuntime()
    action = await PendingActionFactory(runtime).build(
        ProposalResult(
            action="resume_task",
            args={"task_name": "task_a", "datasets": None},
            reason="resume failed A",
            expected_effect="A continues",
        )
    )
    approved = action.fingerprint

    # Approval-time global precondition remains identical, but an external actor has
    # already resumed A. Keep the historical failed run to prove that checking for
    # "any failed run" is insufficient; the latest run is now running.
    runtime.snapshot = {
        "task_name": "task_a",
        "task_exists": True,
        "available_datasets": ["A"],
        "airflow_runs": [
            {"run_id": "external-running", "dataset_name": "A", "state": "running"},
            {"run_id": "old-failed", "dataset_name": "A", "state": "failed"},
        ],
        "errors": {},
    }

    result = await WriteService(runtime).execute(action, approved)

    assert result.status == "PRECONDITION_FAILED"
    assert result.verified is False
    assert runtime.mutation_calls == 0
    assert "stale" in (result.error or "")


@pytest.mark.asyncio
async def test_resume_verification_baseline_tamper_is_blocked_before_mutation():
    from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
    from deploy_ci_cloud_agentv3.services.pending_action import PendingActionFactory

    runtime = ResumeTOCTOURuntime()
    action = await PendingActionFactory(runtime).build(
        ProposalResult(
            action="resume_task",
            args={"task_name": "task_a", "datasets": None},
            reason="resume failed A",
            expected_effect="A continues",
        )
    )
    approved = action.fingerprint
    action.before["airflow_runs"] = []

    with pytest.raises(PermissionError, match="before snapshot"):
        await WriteService(runtime).execute(action, approved)
    assert runtime.mutation_calls == 0
