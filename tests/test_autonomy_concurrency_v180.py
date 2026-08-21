from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from platform_agent.approval import ApprovalStore, AutoReservationResult, action_fingerprint
from platform_agent.actions import WriteActionCoordinator
from platform_agent.goal_verification import GoalVerificationResult
from platform_agent.models import ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.verification import ActionVerificationResult


def reserve(store: ApprovalStore, *, trace_id: str, arguments: dict) -> AutoReservationResult:
    return store.reserve_auto_execution(
        max_actions_per_request=1,
        thread_id="thread-1",
        user_request="resume release_demo",
        tool_name="resume_task",
        arguments=arguments,
        precondition={"queue_sha256": "queue-1", "task_name": "release_demo"},
        risk_level="low",
        impact_summary="resume",
        trace_id=trace_id,
        policy_decision={"mode": "AUTO", "policy_version": "v1.8.0"},
    )


def concurrent_reservations(store: ApprovalStore, arguments_factory) -> list[AutoReservationResult]:
    barrier = Barrier(2)

    def worker(index: int) -> AutoReservationResult:
        barrier.wait()
        return reserve(store, trace_id="trace-race", arguments=arguments_factory(index))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, index) for index in range(2)]
        return [future.result() for future in futures]


def test_same_trace_same_action_concurrent_reservation_creates_one_record(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    results = concurrent_reservations(
        store,
        lambda _index: {"task_name": "release_demo", "datasets": ["B", "A"]},
    )

    assert sorted(result.status for result in results) == ["duplicate_existing", "reserved"]
    assert store.count_auto_actions("trace-race") == 1
    assert sum(result.record is not None and result.status == "reserved" for result in results) == 1
    assert results[0].action_fingerprint == results[1].action_fingerprint


def test_same_trace_different_action_concurrent_reservation_obeys_budget(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    results = concurrent_reservations(
        store,
        lambda index: {"task_name": "release_demo", "datasets": [chr(ord("A") + index)]},
    )

    assert sorted(result.status for result in results) == ["budget_exhausted", "reserved"]
    assert store.count_auto_actions("trace-race") == 1


def test_sequential_duplicate_returns_existing_record(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    arguments = {"task_name": "release_demo", "datasets": ["A", "B"]}
    first = reserve(store, trace_id="trace-sequential", arguments=arguments)
    second = reserve(store, trace_id="trace-sequential", arguments={"task_name": "release_demo", "datasets": ["B", "A"]})

    assert first.status == "reserved"
    assert second.status == "duplicate_existing"
    assert second.existing_record is not None
    assert second.existing_record.approval_id == first.record.approval_id
    assert store.count_auto_actions("trace-sequential") == 1


class CountingMutationClient:
    def __init__(self):
        self.mutation_calls = 0

    async def execute(self, calls):
        call = calls[0]
        if call.name == "resume_task":
            self.mutation_calls += 1
            return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={"ok": True})]
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={})]


class FixedActionVerifier:
    async def verify(self, **_kwargs):
        return ActionVerificationResult(action="resume_task", task_name="release_demo", status="verified")


class FixedGoalVerifier:
    async def verify_resume(self, **_kwargs):
        return GoalVerificationResult(action="resume_task", task_name="release_demo", status="satisfied")


def test_same_record_concurrent_claim_has_one_mutation(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    result = reserve(store, trace_id="trace-claim", arguments={"task_name": "release_demo", "datasets": ["A"]})
    assert result.record is not None
    client = CountingMutationClient()
    coordinator = WriteActionCoordinator(
        client,
        AgentPolicyEngine(),
        store,
        verifier=FixedActionVerifier(),
        goal_verifier=FixedGoalVerifier(),
    )
    barrier = Barrier(2)

    def execute():
        barrier.wait()
        try:
            return asyncio.run(coordinator.execute_approval(result.record.approval_id))
        except RuntimeError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [future.result() for future in [executor.submit(execute), executor.submit(execute)]]

    assert sum(not isinstance(item, RuntimeError) for item in outcomes) == 1
    assert sum(isinstance(item, RuntimeError) for item in outcomes) == 1
    assert client.mutation_calls == 1
    rejected_message = next(item for item in outcomes if isinstance(item, RuntimeError)).args[0]
    assert "AUTO execution claim rejected" in rejected_message or "Approval is not pending" in rejected_message


def test_restart_can_claim_authorized_record_but_never_replays_executing_record(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    result = reserve(store, trace_id="trace-restart", arguments={"task_name": "release_demo", "datasets": ["A"]})
    assert result.record is not None

    restarted = ApprovalStore(tmp_path / "approvals")
    persisted = restarted.get(result.record.approval_id)
    assert persisted.status == "authorized"
    assert persisted.action_fingerprint == action_fingerprint(persisted.tool_name, persisted.arguments)
    assert persisted.policy_decision["reservation_status"] == "reserved"
    assert restarted.claim_for_execution(persisted.approval_id).status == "executing"

    with pytest.raises(RuntimeError, match="outcome is unknown"):
        ApprovalStore(tmp_path / "approvals").claim_for_execution(persisted.approval_id)


def test_authorization_persistence_failure_prevents_reservation(tmp_path: Path, monkeypatch):
    store = ApprovalStore(tmp_path / "approvals")

    def fail_write(_item):
        raise OSError("authorization persistence unavailable")

    monkeypatch.setattr(store, "_write_unlocked", fail_write)
    with pytest.raises(OSError, match="authorization persistence unavailable"):
        reserve(store, trace_id="trace-persist-failure", arguments={"task_name": "release_demo", "datasets": ["A"]})
    assert store.count_auto_actions("trace-persist-failure") == 0
