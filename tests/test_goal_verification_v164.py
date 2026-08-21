from __future__ import annotations

import asyncio
import json
from pathlib import Path

from platform_agent.actions import WriteActionCoordinator
from platform_agent.approval import ApprovalStore
from platform_agent.goal_verification import GoalVerificationResult, GoalVerifier
from platform_agent.models import AgentIntent, AgentPlan, ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.verification import ActionVerificationResult


def run(coro):
    return asyncio.run(coro)


def action_verified():
    return ActionVerificationResult(action="resume_task", task_name="release_demo", status="verified")


def baseline():
    return {
        "task_name": "release_demo",
        "task_exists": True,
        "errors": {},
        "airflow_runs": [{"run_id": "old1", "dataset_name": "clip_001", "state": "failed"}],
    }


class SnapshotClient:
    def __init__(self, snapshot=None, *, error: str | None = None):
        self.snapshot = snapshot
        self.error = error
        self.calls = []

    async def execute(self, calls):
        self.calls.extend(calls)
        if self.error:
            return [ToolObservation(tool_name=calls[0].name, arguments=calls[0].arguments, ok=False, error=self.error)]
        return [ToolObservation(tool_name=calls[0].name, arguments=calls[0].arguments, ok=True, data=self.snapshot)]


def verify(snapshot, *, arguments=None, error=None):
    client = SnapshotClient(snapshot, error=error)
    result = run(
        GoalVerifier(client).verify_resume(
            arguments=arguments or {"task_name": "release_demo", "datasets": []},
            action_verification=action_verified(),
            baseline=baseline(),
        )
    )
    return result, client


def accepted_snapshot(state="queued", task_name="release_demo"):
    return {
        "task_name": task_name,
        "task_exists": True,
        "errors": {},
        "airflow_runs": [
            {"run_id": "old1", "dataset_name": "clip_001", "state": "failed"},
            {"run_id": "new1", "dataset_name": "clip_001", "state": state},
        ],
    }


def test_action_verified_and_new_accepted_execution_satisfies_goal():
    result, _ = verify(accepted_snapshot("running"))
    assert result.status == "satisfied"


def test_action_verified_but_execution_not_yet_accepted_is_in_progress():
    result, _ = verify(accepted_snapshot("starting"))
    assert result.status == "in_progress"


def test_action_verified_but_new_execution_failed_fails_goal():
    result, _ = verify(accepted_snapshot("failed"))
    assert result.status == "failed"


def test_post_action_evidence_unavailable_is_inconclusive():
    result, _ = verify(None, error="read backend unavailable")
    assert result.status == "inconclusive"
    assert result.status != "satisfied"


def test_wrong_target_post_action_evidence_cannot_satisfy_goal():
    result, _ = verify(accepted_snapshot("running", task_name="other_task"))
    assert result.status == "inconclusive"


def test_old_failed_run_without_new_execution_is_not_satisfied():
    snapshot = baseline()
    result, _ = verify(snapshot)
    assert result.status == "in_progress"


class FixedActionVerifier:
    def __init__(self, result):
        self.result = result

    async def verify(self, **kwargs):
        return self.result


class SpyGoalVerifier:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def verify_resume(self, **kwargs):
        self.calls += 1
        return self.result


class MutationClient:
    async def execute(self, calls):
        call = calls[0]
        if call.name == "resume_task":
            return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={"ok": True})]
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={})]


class FailedMutationClient(MutationClient):
    async def execute(self, calls):
        call = calls[0]
        if call.name == "resume_task":
            return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error="mutation rejected")]
        return await super().execute(calls)


def pending_resume(store: ApprovalStore):
    return store.create(
        thread_id="thread",
        user_request="恢复 release_demo",
        tool_name="resume_task",
        arguments={"task_name": "release_demo", "datasets": []},
        precondition={"task_exists": True},
        risk_level="high",
        impact_summary="resume",
        verification_baseline=baseline(),
    )


def test_action_verification_failure_never_calls_goal_verifier(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    goal = SpyGoalVerifier(GoalVerificationResult(action="resume_task", task_name="release_demo", status="satisfied"))
    coordinator = WriteActionCoordinator(
        MutationClient(),
        AgentPolicyEngine(),
        store,
        verifier=FixedActionVerifier(ActionVerificationResult(action="resume_task", task_name="release_demo", status="failed")),
        goal_verifier=goal,
    )
    item = run(coordinator.execute_approval(pending_resume(store).approval_id))
    assert item.status == "verification_failed"
    assert goal.calls == 0


def test_mutation_failure_never_calls_goal_verifier(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    goal = SpyGoalVerifier(GoalVerificationResult(action="resume_task", task_name="release_demo", status="satisfied"))
    coordinator = WriteActionCoordinator(
        FailedMutationClient(),
        AgentPolicyEngine(),
        store,
        goal_verifier=goal,
    )
    item = run(coordinator.execute_approval(pending_resume(store).approval_id))
    assert item.status == "failed"
    assert goal.calls == 0


def test_action_verified_goal_result_is_stored_separately(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    goal_result = GoalVerificationResult(action="resume_task", task_name="release_demo", status="in_progress")
    goal = SpyGoalVerifier(goal_result)
    coordinator = WriteActionCoordinator(
        MutationClient(),
        AgentPolicyEngine(),
        store,
        verifier=FixedActionVerifier(action_verified()),
        goal_verifier=goal,
    )
    item = run(coordinator.execute_approval(pending_resume(store).approval_id))
    assert item.status == "executed"
    assert item.verification_result["status"] == "verified"
    assert item.goal_verification_result["status"] == "in_progress"
    assert goal.calls == 1


def test_action_verified_and_goal_satisfied_still_uses_executed_mutation_status(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    goal_result = GoalVerificationResult(action="resume_task", task_name="release_demo", status="satisfied")
    goal = SpyGoalVerifier(goal_result)
    coordinator = WriteActionCoordinator(
        MutationClient(),
        AgentPolicyEngine(),
        store,
        verifier=FixedActionVerifier(action_verified()),
        goal_verifier=goal,
    )
    item = run(coordinator.execute_approval(pending_resume(store).approval_id))
    assert item.status == "executed"
    assert item.goal_verification_result["status"] == "satisfied"


def test_old_approval_json_without_goal_verification_field_loads(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = pending_resume(store)
    path = tmp_path / "approvals" / f"{item.approval_id}.json"
    payload = item.model_dump(mode="json")
    payload.pop("goal_verification_result", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = store.get(item.approval_id)
    assert loaded.goal_verification_result is None
