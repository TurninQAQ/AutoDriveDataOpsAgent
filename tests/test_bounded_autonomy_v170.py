import asyncio
from pathlib import Path

from platform_agent.approval import ApprovalStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import AgentIntent, ToolCallSpec, ToolObservation
from platform_agent.tool_client import InMemoryMCPToolClient
from platform_agent.verification import ActionVerificationResult
from platform_agent.workflow import build_agent_runtime
from platform_agent.goal_verification import GoalVerificationResult
from platform_agent.actions import WriteActionCoordinator
from platform_agent.policy import AgentPolicyEngine
from platform_agent.autonomy import BoundedAutonomyPolicy
from platform_agent.memory import ConversationStore


def run(coro):
    return asyncio.run(coro)


class AutoResumeClient:
    def __init__(self, *, mutation_error: str | None = None, drift_to: list[str] | None = None):
        self.calls: list[ToolCallSpec] = []
        self.mutation_error = mutation_error
        self.drift_to = drift_to or []
        self.snapshot_calls = 0
        self.mutation_calls = 0

    async def describe_tools(self):
        return [{"name": name, "description": name, "input_schema": {}} for name in (
            "get_task_detail", "get_queue_state", "get_write_precondition",
            "get_action_verification_snapshot", "resume_task",
        )]

    async def execute(self, calls):
        result = []
        for call in calls:
            self.calls.append(call)
            if call.name == "get_task_detail":
                data = {"task_name": call.arguments.get("task_name"), "datasets": ["A"]}
            elif call.name == "get_queue_state":
                data = {"version": 2, "active": None, "queue": []}
            elif call.name == "get_write_precondition":
                data = {
                    "queue_sha256": "queue-1",
                    "task_name": call.arguments.get("task_name", ""),
                    "task_config_sha256": "config-1",
                    "task_exists": True,
                    "active_task_name": "",
                }
            elif call.name == "get_action_verification_snapshot":
                self.snapshot_calls += 1
                runs = [
                    {"run_id": "old-a", "dataset_name": "A", "state": "failed"},
                ]
                if self.snapshot_calls >= 2:
                    runs.insert(0, {"run_id": "new-a", "dataset_name": "A", "state": "running"})
                data = {
                    "task_name": call.arguments.get("task_name"),
                    "task_exists": True,
                    "config_file_exists": True,
                    "dag_file_exists": True,
                    "airflow_dag_exists": True,
                    "available_datasets": ["A"],
                    "task_exclusive": False,
                    "queue": {"location": "not_found", "position": -1, "entry": None},
                    "containers": [],
                    "gpu_reservations": [],
                    "airflow_runs": runs,
                    "errors": {},
                }
            elif call.name == "resume_task":
                self.mutation_calls += 1
                if self.mutation_error:
                    result.append(ToolObservation(
                        tool_name=call.name, arguments=call.arguments, ok=False, error=self.mutation_error
                    ))
                    continue
                # A scope-drift fixture changes the platform's later candidate
                # set, but the frozen mutation arguments remain observable here.
                data = {"ok": True, "task_name": call.arguments.get("task_name"), "datasets": call.arguments.get("datasets")}
            else:
                data = {}
            result.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=data))
        return result


def make_agent(tmp_path: Path, client: AutoResumeClient, *, enabled: bool):
    store = ApprovalStore(tmp_path / "approvals")
    agent = build_agent_runtime(
        "sequential",
        HeuristicReadOnlyModel(),
        client,
        ConversationStore(tmp_path / "sessions"),
        approval_store=store,
        autonomy_enabled=enabled,
        auto_actions_per_request=1,
        auto_resume_max_datasets=3,
    )
    # The integration fixture has no waiting interval.
    agent.nodes.action_coordinator.verifier.attempts = 1
    agent.nodes.action_coordinator.verifier.interval_sec = 0
    agent.nodes.action_coordinator.goal_verifier.attempts = 1
    agent.nodes.action_coordinator.goal_verifier.interval_sec = 0
    return agent, store


def test_safe_resume_uses_auto_and_shared_guarded_execution(tmp_path: Path):
    client = AutoResumeClient()
    agent, store = make_agent(tmp_path, client, enabled=True)
    response = run(agent.run("恢复 release_demo", "trace-auto"))

    assert response.intent == AgentIntent.RESUME_TASK
    assert response.authorization_mode == "auto"
    assert response.approval_required is False
    assert response.goal_verification_result["status"] == "satisfied"
    assert response.goal_progress.value == "SATISFIED"
    assert response.blocked is False
    assert client.mutation_calls == 1
    mutation = next(call for call in client.calls if call.name == "resume_task")
    assert mutation.arguments["datasets"] == ["A"]
    item = store.get(response.approval_id)
    assert item.authorization_mode == "auto"
    assert item.status == "executed"
    assert item.policy_decision["mode"] == "AUTO"


def test_autonomy_disabled_keeps_resume_pending_hitl(tmp_path: Path):
    client = AutoResumeClient()
    agent, store = make_agent(tmp_path, client, enabled=False)
    response = run(agent.run("恢复 release_demo", "trace-hitl"))

    assert response.approval_required is True
    assert response.authorization_mode == "hitl"
    assert response.policy_decision["mode"] == "HITL"
    assert client.mutation_calls == 0
    assert store.get(response.approval_id).status == "pending"


def test_scope_drift_cannot_expand_frozen_auto_dataset_arguments(tmp_path: Path):
    client = AutoResumeClient(drift_to=["A", "B"])
    agent, _ = make_agent(tmp_path, client, enabled=True)
    response = run(agent.run("恢复 release_demo", "trace-drift"))

    assert response.authorization_mode == "auto"
    mutation = next(call for call in client.calls if call.name == "resume_task")
    assert mutation.arguments["datasets"] == ["A"]
    assert mutation.arguments["datasets"] != client.drift_to


def test_stale_precondition_stops_auto_mutation_without_retry(tmp_path: Path):
    client = AutoResumeClient(mutation_error="PRECONDITION_FAILED: queue_sha256")
    agent, store = make_agent(tmp_path, client, enabled=True)
    response = run(agent.run("恢复 release_demo", "trace-stale"))

    assert response.authorization_mode == "auto"
    assert response.blocked is True
    assert response.approval_required is False
    assert client.mutation_calls == 1
    item = store.get(response.approval_id)
    assert item.status == "failed"
    assert "PRECONDITION_FAILED" in (item.error or "")


def test_non_resume_write_never_becomes_auto(tmp_path: Path):
    client = AutoResumeClient()
    agent, store = make_agent(tmp_path, client, enabled=True)
    response = run(agent.run("停止 release_demo", "trace-stop"))

    assert response.intent == AgentIntent.STOP_TASK
    assert response.approval_required is True
    assert response.authorization_mode == "hitl"
    assert client.mutation_calls == 0
    assert store.get(response.approval_id).authorization_mode == "hitl"


def test_old_approval_record_without_autonomy_metadata_loads_as_hitl(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = store.create(
        thread_id="t",
        user_request="stop",
        tool_name="stop_task",
        arguments={"task_name": "release_demo"},
        precondition={},
        risk_level="high",
        impact_summary="stop",
    )
    path = tmp_path / "approvals" / f"{item.approval_id}.json"
    payload = item.model_dump(mode="json")
    payload.pop("authorization_mode", None)
    payload.pop("policy_decision", None)
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    loaded = store.get(item.approval_id)
    assert loaded.authorization_mode == "hitl"
    assert loaded.policy_decision is None


class FixedVerifier:
    def __init__(self, result):
        self.result = result

    async def verify(self, **kwargs):
        return self.result


class FixedGoalVerifier:
    def __init__(self, result):
        self.result = result

    async def verify_resume(self, **kwargs):
        return self.result


class MinimalMutationClient:
    def __init__(self):
        self.mutations = 0

    async def execute(self, calls):
        call = calls[0]
        if call.name == "resume_task":
            self.mutations += 1
            return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={"ok": True})]
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={})]


def auto_record(store):
    return store.create_auto_execution(
        thread_id="t",
        user_request="resume",
        tool_name="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition={"queue_sha256": "q", "task_name": "release_demo"},
        risk_level="low",
        impact_summary="resume",
        verification_baseline={},
        trace_id="trace",
        policy_decision={"mode": "AUTO"},
    )


def test_action_verified_goal_in_progress_does_not_become_success(tmp_path: Path):
    from platform_agent.goal_verification import GoalVerificationResult

    store = ApprovalStore(tmp_path / "approvals")
    client = MinimalMutationClient()
    coordinator = WriteActionCoordinator(
        client,
        AgentPolicyEngine(),
        store,
        verifier=FixedVerifier(ActionVerificationResult(action="resume_task", task_name="release_demo", status="verified")),
        goal_verifier=FixedGoalVerifier(GoalVerificationResult(action="resume_task", task_name="release_demo", status="in_progress")),
    )
    item = run(coordinator.execute_approval(auto_record(store).approval_id))
    assert item.status == "executed"
    assert item.verification_result["status"] == "verified"
    assert item.goal_verification_result["status"] == "in_progress"
    assert client.mutations == 1


def test_action_verified_goal_failed_does_not_trigger_retry(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    client = MinimalMutationClient()
    coordinator = WriteActionCoordinator(
        client,
        AgentPolicyEngine(),
        store,
        verifier=FixedVerifier(ActionVerificationResult(action="resume_task", task_name="release_demo", status="verified")),
        goal_verifier=FixedGoalVerifier(GoalVerificationResult(action="resume_task", task_name="release_demo", status="failed")),
    )
    item = run(coordinator.execute_approval(auto_record(store).approval_id))
    assert item.status == "executed"
    assert item.goal_verification_result["status"] == "failed"
    assert client.mutations == 1
