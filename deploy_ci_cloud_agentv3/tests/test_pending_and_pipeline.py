from __future__ import annotations

import pytest

from deploy_ci_cloud_agentv3.mcp.client import InProcessMCPClient
from deploy_ci_cloud_agentv3.mcp.factory import build_tooling
from deploy_ci_cloud_agentv3.mcp.profiles import AGENT_TOOLS, RUNTIME_TOOLS
from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
from deploy_ci_cloud_agentv3.services.pending_action import PendingActionFactory
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


@pytest.mark.asyncio
async def test_edit_invalidates_old_fingerprint():
    facade = FakeFacade()
    tooling = build_tooling(facade)
    runtime = InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)
    factory = PendingActionFactory(runtime, tooling.artifacts)
    original = await factory.build(
        ProposalResult(
            action="set_task_priority",
            args={"task_name": "task_a", "priority": 5},
            reason="raise priority",
            expected_effect="priority becomes 5",
        )
    )
    edited = await factory.rebuild_from_edit(original, {"task_name": "task_a", "priority": 7})
    assert edited.fingerprint != original.fingerprint
    assert edited.proposal_id != original.proposal_id
    assert edited.args["priority"] == 7


@pytest.mark.asyncio
async def test_prepare_then_submit_proposal_binds_exact_artifact_without_mutation():
    facade = FakeFacade()
    tooling = build_tooling(facade)
    agent = InProcessMCPClient(tooling.registry, AGENT_TOOLS)
    artifact = await agent.call_tool(
        "prepare_task_spec",
        {"task_prefix": "train", "dataset_path": "/data/clip_001"},
    )
    assert artifact["artifact_id"]
    assert "gpu_ids" in artifact["config"]
    assert facade.mutations == []

    proposal = await agent.call_tool(
        "propose_submit_task", {"artifact_id": artifact["artifact_id"]}
    )
    assert proposal["args"]["artifact_id"] == artifact["artifact_id"]
    assert facade.mutations == []


class ResumeFreezeRuntime:
    def __init__(self):
        self.snapshot = {
            "task_exists": True,
            "airflow_runs": [
                {"run_id": "a-old", "dataset_name": "A", "state": "failed"},
                {"run_id": "b-old", "dataset_name": "B", "state": "failed"},
                {"run_id": "c-old", "dataset_name": "C", "state": "success"},
            ],
            "errors": {},
        }
        self.precondition = {"task_name": "task_a", "queue_sha256": "q", "task_config_sha256": "c"}

    async def call_tool(self, name, args):
        if name == "get_action_verification_snapshot":
            return self.snapshot
        if name == "capture_write_precondition":
            return self.precondition
        raise AssertionError(name)


@pytest.mark.asyncio
async def test_resume_none_is_resolved_to_explicit_failed_datasets_before_review():
    factory = PendingActionFactory(ResumeFreezeRuntime())
    pending = await factory.build(
        ProposalResult(
            action="resume_task",
            args={"task_name": "task_a", "datasets": None},
            reason="resume failures",
            expected_effect="failed datasets continue",
        )
    )
    assert pending.args["datasets"] == ["A", "B"]
    assert pending.before["airflow_runs"][0]["run_id"] == "a-old"
    assert pending.recompute_fingerprint() == pending.fingerprint


@pytest.mark.asyncio
async def test_resume_none_with_no_failed_dataset_does_not_create_reviewable_action():
    runtime = ResumeFreezeRuntime()
    runtime.snapshot["airflow_runs"] = [
        {"run_id": "done", "dataset_name": "A", "state": "success"}
    ]
    factory = PendingActionFactory(runtime)
    with pytest.raises(ValueError, match="no currently failed datasets"):
        await factory.build(
            ProposalResult(
                action="resume_task",
                args={"task_name": "task_a", "datasets": None},
                reason="resume failures",
                expected_effect="failed datasets continue",
            )
        )


class ResumeSingleSnapshotRuntime(ResumeFreezeRuntime):
    def __init__(self):
        super().__init__()
        self.snapshot_calls = 0

    async def call_tool(self, name, args):
        if name == "get_action_verification_snapshot":
            self.snapshot_calls += 1
            if self.snapshot_calls > 1:
                raise RuntimeError("transient Airflow failure on forbidden second snapshot")
            return self.snapshot
        return await super().call_tool(name, args)


@pytest.mark.asyncio
async def test_resume_none_resolution_snapshot_is_reused_as_before_without_second_read():
    """Regression for the false-success race found during V3.3 review.

    The old implementation performed one snapshot to resolve datasets=None and a
    second snapshot for ``before``. If that second read failed it could fall back to
    task detail, erasing run-id baseline evidence. The factory must now use exactly
    one Airflow snapshot for both responsibilities.
    """
    runtime = ResumeSingleSnapshotRuntime()
    factory = PendingActionFactory(runtime)
    pending = await factory.build(
        ProposalResult(
            action="resume_task",
            args={"task_name": "task_a", "datasets": None},
            reason="resume failures",
            expected_effect="failed datasets continue",
        )
    )

    assert runtime.snapshot_calls == 1
    assert pending.args["datasets"] == ["A", "B"]
    assert pending.before == runtime.snapshot
    assert pending.action_precondition["kind"] == "resume_latest_state_failed"
    assert pending.action_precondition["datasets"] == ["A", "B"]
    assert pending.recompute_fingerprint() == pending.fingerprint


class ResumeSnapshotFailureRuntime(ResumeFreezeRuntime):
    async def call_tool(self, name, args):
        if name == "get_action_verification_snapshot":
            raise RuntimeError("Airflow unavailable")
        if name == "get_task_detail":
            raise AssertionError("resume must never degrade to task-detail baseline")
        return await super().call_tool(name, args)


@pytest.mark.asyncio
async def test_resume_before_snapshot_failure_is_fail_closed_without_task_detail_fallback():
    factory = PendingActionFactory(ResumeSnapshotFailureRuntime())
    with pytest.raises(RuntimeError, match="Airflow unavailable"):
        await factory.build(
            ProposalResult(
                action="resume_task",
                args={"task_name": "task_a", "datasets": ["A"]},
                reason="explicit rerun",
                expected_effect="A continues",
            )
        )


@pytest.mark.asyncio
async def test_resume_none_freezes_only_datasets_whose_latest_run_is_failed():
    runtime = ResumeFreezeRuntime()
    # Snapshot ordering is latest-first. A has an old failure but has already moved
    # to running, so only B is reviewable as a failed-dataset resume target.
    runtime.snapshot["airflow_runs"] = [
        {"run_id": "a-running", "dataset_name": "A", "state": "running"},
        {"run_id": "a-old-failed", "dataset_name": "A", "state": "failed"},
        {"run_id": "b-failed", "dataset_name": "B", "state": "failed"},
    ]
    pending = await PendingActionFactory(runtime).build(
        ProposalResult(
            action="resume_task",
            args={"task_name": "task_a", "datasets": None},
            reason="resume failures",
            expected_effect="failed datasets continue",
        )
    )
    assert pending.args["datasets"] == ["B"]
