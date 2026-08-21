from __future__ import annotations

import asyncio
from pathlib import Path

from platform_agent.approval import ApprovalStore
from platform_agent.actions import WriteActionCoordinator
from platform_agent.goal_verification import GoalVerificationResult
from platform_agent.models import ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.verification import ActionVerificationResult


def run(coro):
    return asyncio.run(coro)


def auto_record(store: ApprovalStore, *, datasets=None):
    return store.create_auto_execution(
        thread_id="t",
        user_request="resume release_demo",
        tool_name="resume_task",
        arguments={"task_name": "release_demo", "datasets": list(datasets or ["A"])},
        precondition={"queue_sha256": "q", "task_name": "release_demo"},
        risk_level="low",
        impact_summary="resume",
        verification_baseline={"airflow_runs": []},
        trace_id="trace-failure",
        policy_decision={"mode": "AUTO", "policy_version": "v1.8.0"},
    )


class FailureClient:
    def __init__(self, *, exception: Exception | None = None, ok: bool = True):
        self.exception = exception
        self.ok = ok
        self.mutation_calls = 0

    async def execute(self, calls):
        call = calls[0]
        if call.name == "resume_task":
            self.mutation_calls += 1
            if self.exception:
                raise self.exception
            return [ToolObservation(
                tool_name=call.name,
                arguments=call.arguments,
                ok=self.ok,
                data={"ok": self.ok},
                error=None if self.ok else "mutation returned ok=false",
            )]
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={})]


class FixedActionVerifier:
    def __init__(self, status="verified"):
        self.status = status

    async def verify(self, **_kwargs):
        return ActionVerificationResult(action="resume_task", task_name="release_demo", status=self.status)


class FixedGoalVerifier:
    def __init__(self, status):
        self.status = status

    async def verify_resume(self, **_kwargs):
        return GoalVerificationResult(action="resume_task", task_name="release_demo", status=self.status)


def coordinator(store, client, *, action_status="verified", goal_status="satisfied"):
    return WriteActionCoordinator(
        client,
        AgentPolicyEngine(),
        store,
        verifier=FixedActionVerifier(action_status),
        goal_verifier=FixedGoalVerifier(goal_status),
    )


def test_mutation_exception_has_one_attempt_and_no_retry(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = auto_record(store)
    client = FailureClient(exception=RuntimeError("mutation transport failed"))
    result = run(coordinator(store, client).execute_approval(item.approval_id))
    assert result.status == "failed"
    assert client.mutation_calls == 1


def test_mutation_false_has_one_attempt_and_no_retry(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = auto_record(store)
    client = FailureClient(ok=False)
    result = run(coordinator(store, client).execute_approval(item.approval_id))
    assert result.status == "failed"
    assert client.mutation_calls == 1


def test_action_verification_failure_has_no_auto_retry(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = auto_record(store)
    client = FailureClient()
    result = run(coordinator(store, client, action_status="failed").execute_approval(item.approval_id))
    assert result.status == "verification_failed"
    assert client.mutation_calls == 1


def test_goal_failure_in_progress_and_inconclusive_never_retry(tmp_path: Path):
    for status in ("failed", "in_progress", "inconclusive"):
        store = ApprovalStore(tmp_path / status)
        item = auto_record(store)
        client = FailureClient()
        result = run(coordinator(store, client, goal_status=status).execute_approval(item.approval_id))
        assert result.status == "executed"
        assert result.goal_verification_result["status"] == status
        assert client.mutation_calls == 1


def test_frozen_action_fingerprint_mismatch_blocks_before_mutation(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = auto_record(store)
    item_path = store._path(item.approval_id)
    payload = item.model_dump(mode="json")
    payload["arguments"]["datasets"] = ["B"]
    item_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    client = FailureClient()
    result = run(coordinator(store, client).execute_approval(item.approval_id))
    assert result.status == "failed"
    assert "fingerprint mismatch" in (result.error or "")
    assert client.mutation_calls == 0
