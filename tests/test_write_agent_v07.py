from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
import yaml

from platform_agent.actions import WriteActionCoordinator
from platform_agent.approval import ApprovalStore
from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec, ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.tool_client import FacadeToolClient
from platform_agent.workflow import SequentialReadOnlyAgent, build_agent_runtime
from platform_core.mutation import PreconditionFailed
from platform_core.services.mutation_service import PlatformMutationService
from platform_core.services.precondition_service import PreconditionService
from platform_core.services.queue_service import QueueService
from platform_mcp.server import (
    ALL_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    WRITE_PREP_TOOL_NAMES,
    WRITE_TOOL_NAMES,
    build_mcp_server,
)
from platform_planning.service import TaskPlanningService


ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def planning_service() -> TaskPlanningService:
    return TaskPlanningService(
        defaults_path=ROOT / "config" / "task_planning_defaults.yaml",
        scripts_dir=ROOT / "scripts",
    )


def valid_plan():
    result = planning_service().plan(
        "创建一个release任务，任务名release，数据 /data/record_001，完整流程，最多并发4个clip"
    )
    assert result.valid, result.issues
    return result


def write_task_config(root: Path, task_name: str = "release_20260819_120000") -> dict:
    result = valid_plan()
    task_dir = root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "datasets_config.yaml").write_text(
        yaml.safe_dump(result.config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return result.config


class FakeMutationGateway:
    def __init__(self):
        self.calls = []

    def submit(self, task_prefix, config):
        self.calls.append(("submit", task_prefix, config))
        return {"task_name": f"{task_prefix}_generated", "triggered": len(config["datasets"])}

    def set_priority(self, task_name, priority):
        self.calls.append(("priority", task_name, priority))
        return {"task_name": task_name, "priority": priority}

    def resume(self, task_name, datasets):
        self.calls.append(("resume", task_name, list(datasets)))
        return {"task_name": task_name, "datasets": list(datasets)}

    def stop(self, task_name, datasets):
        self.calls.append(("stop", task_name, list(datasets)))
        return {"task_name": task_name, "datasets": list(datasets)}

    def delete(self, task_name):
        self.calls.append(("delete", task_name))
        return {"task_name": task_name}


def make_mutation_service(tmp_path: Path):
    queue_file = tmp_path / "state" / "task_queue" / "queue.lock"
    queue_file.parent.mkdir(parents=True)
    queue_file.write_text(json.dumps({"version": 2, "active": None, "queue": []}), encoding="utf-8")
    task_root = tmp_path / "tasks"
    dags = tmp_path / "dags"
    task_root.mkdir()
    dags.mkdir()
    queue = QueueService(queue_file)
    pre = PreconditionService(queue, dags, task_root)
    gateway = FakeMutationGateway()
    service = PlatformMutationService(gateway, pre, scripts_dir=ROOT / "scripts")
    return service, gateway, queue_file, task_root


def test_v07_tool_contract_extends_but_does_not_change_v03_read_only_contract():
    assert READ_ONLY_TOOL_NAMES == (
        "get_platform_health",
        "list_tasks",
        "get_task_detail",
        "get_queue_state",
        "get_gpu_pool",
        "inspect_task_containers",
        "get_stage_logs",
        "diagnose_task",
        "search_knowledge",
    )
    assert WRITE_PREP_TOOL_NAMES == (
        "get_write_precondition", "validate_task_spec", "get_action_verification_snapshot"
    )
    assert WRITE_TOOL_NAMES == (
        "submit_task",
        "resume_task",
        "set_task_priority",
        "stop_task",
        "delete_task",
    )
    assert len(ALL_TOOL_NAMES) == 17


def test_mcp_server_registers_write_surface_only_when_enabled(monkeypatch):
    import sys
    import types

    class FakeFacade:
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"name": name, "args": args, "kwargs": kwargs}

    class FakeMCPServer:
        def __init__(self, name, instructions=""):
            self.tools = {}
        def tool(self):
            def register(fn):
                self.tools[fn.__name__] = fn
                return fn
            return register

    mcp = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    server_mod.MCPServer = FakeMCPServer
    mcp.server = server_mod
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)

    read_server = build_mcp_server(FakeFacade())
    assert tuple(read_server.tools) == READ_ONLY_TOOL_NAMES
    write_server = build_mcp_server(FakeFacade(), include_write_tools=True)
    assert tuple(write_server.tools) == ALL_TOOL_NAMES


def test_precondition_detects_queue_change(tmp_path: Path):
    service, _, queue_file, _ = make_mutation_service(tmp_path)
    expected = service.capture_precondition()
    queue_file.write_text(
        json.dumps({"version": 2, "active": {"task_name": "release_a"}, "queue": []}), encoding="utf-8"
    )
    with pytest.raises(PreconditionFailed, match="PRECONDITION_FAILED"):
        service.preconditions.assert_matches(expected)


def test_precondition_detects_task_config_change(tmp_path: Path):
    service, _, _, task_root = make_mutation_service(tmp_path)
    task_name = "release_20260819_120000"
    write_task_config(task_root, task_name)
    expected = service.capture_precondition(task_name)
    config_file = task_root / task_name / "datasets_config.yaml"
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config["priority"] = 99
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(PreconditionFailed, match="task_config_sha256"):
        service.preconditions.assert_matches(expected)


def test_submit_revalidates_config_and_checks_precondition_before_gateway(tmp_path: Path):
    service, gateway, queue_file, _ = make_mutation_service(tmp_path)
    plan = valid_plan()
    pre = service.capture_precondition()
    result = service.submit_task(plan.task_spec.task_prefix, plan.config, pre)
    assert result["ok"] is True
    assert gateway.calls[0][0] == "submit"

    bad = dict(plan.config)
    bad["pipeline_stages"] = ["not_a_real_stage"]
    with pytest.raises(Exception):
        service.submit_task(plan.task_spec.task_prefix, bad, service.capture_precondition())
    assert len(gateway.calls) == 1

    queue_file.write_text(json.dumps({"version": 2, "active": {"task_name": "x"}, "queue": []}), encoding="utf-8")
    with pytest.raises(PreconditionFailed):
        service.submit_task(plan.task_spec.task_prefix, plan.config, pre)
    assert len(gateway.calls) == 1


def test_task_mutations_validate_dataset_and_use_gateway(tmp_path: Path):
    service, gateway, _, task_root = make_mutation_service(tmp_path)
    task_name = "release_20260819_120000"
    config = write_task_config(task_root, task_name)
    ds = config["datasets"][0]["dataset_name"]

    service.set_task_priority(task_name, 5, service.capture_precondition(task_name))
    service.resume_task(task_name, [ds], service.capture_precondition(task_name))
    service.stop_task(task_name, [ds], service.capture_precondition(task_name))
    service.delete_task(task_name, service.capture_precondition(task_name))
    assert [call[0] for call in gateway.calls] == ["priority", "resume", "stop", "delete"]

    with pytest.raises(Exception, match="Unknown dataset_name"):
        service.stop_task(task_name, ["clip_does_not_exist"], service.capture_precondition(task_name))


def test_approval_store_is_persistent_and_rejectable(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals", ttl_sec=300)
    item = store.create(
        thread_id="t",
        user_request="delete task",
        tool_name="delete_task",
        arguments={"task_name": "release_a"},
        precondition={"queue_sha256": "abc"},
        risk_level="destructive",
        impact_summary="delete",
    )
    assert store.get(item.approval_id).status == "pending"
    rejected = store.reject(item.approval_id)
    assert rejected.status == "rejected"
    assert ApprovalStore(tmp_path / "approvals").get(item.approval_id).status == "rejected"


def test_approval_store_expires_pending_item(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals", ttl_sec=30)
    item = store.create(
        thread_id="t", user_request="x", tool_name="stop_task", arguments={}, precondition={},
        risk_level="high", impact_summary="stop",
    )
    path = tmp_path / "approvals" / f"{item.approval_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expires_at"] = time.time() - 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.get(item.approval_id).status == "expired"


class FakeWriteToolClient:
    def __init__(self, *, write_error: str | None = None):
        self.calls: list[ToolCallSpec] = []
        self.write_error = write_error
        self.current_priority = 30

    async def describe_tools(self):
        return [{"name": name, "description": name, "input_schema": {}} for name in READ_ONLY_TOOL_NAMES]

    async def execute(self, calls):
        self.calls.extend(calls)
        result = []
        for call in calls:
            if call.name == "get_queue_state":
                data = {"version": 2, "active": {"task_name": "reprocess_active", "priority": 20}, "queue": []}
            elif call.name == "get_task_detail":
                data = {"task_name": call.arguments["task_name"], "priority": 30, "datasets": ["clip_001"]}
            elif call.name == "get_write_precondition":
                data = {"queue_sha256": "q1", "task_name": call.arguments.get("task_name", ""), "task_config_sha256": "c1", "task_exists": True if call.arguments.get("task_name") else None, "active_task_name": "reprocess_active"}
            elif call.name == "validate_task_spec":
                data = {"valid": True}
            elif call.name == "get_action_verification_snapshot":
                data = {
                    "task_name": call.arguments.get("task_name", ""),
                    "task_exists": True, "dag_file_exists": True, "priority": self.current_priority,
                    "task_exclusive": True,
                    "queue": {"location": "not_found", "position": -1, "entry": None},
                    "containers": [], "gpu_reservations": [],
                    "airflow_dag_exists": True, "airflow_runs": [], "errors": {},
                }
            elif call.name in WRITE_TOOL_NAMES:
                if self.write_error:
                    result.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error=self.write_error))
                    continue
                if call.name == "set_task_priority":
                    self.current_priority = int(call.arguments["priority"])
                data = {"ok": True, "action": call.name, "arguments": call.arguments}
            else:
                data = {}
            result.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=data))
        return result


def make_write_agent(tmp_path: Path, client=None):
    client = client or FakeWriteToolClient()
    store = ApprovalStore(tmp_path / "approvals", ttl_sec=300)
    agent = build_agent_runtime(
        "sequential",
        HeuristicReadOnlyModel(),
        client,
        ConversationStore(tmp_path / "sessions"),
        task_planning_service=planning_service(),
        approval_store=store,
    )
    return agent, client, store


def test_stop_request_creates_hitl_approval_without_write_execution(tmp_path: Path):
    agent, client, store = make_write_agent(tmp_path)
    response = run(agent.run("把 release_20260819_a 停止掉", "t"))
    assert response.intent == AgentIntent.STOP_TASK
    assert response.approval_required is True
    assert response.approval_id
    assert response.blocked is True
    assert not any(call.name in WRITE_TOOL_NAMES for call in client.calls)
    item = store.get(response.approval_id)
    assert item.tool_name == "stop_task"
    assert item.risk_level == "high"
    assert item.arguments == {"task_name": "release_20260819_a", "datasets": []}


def test_delete_requires_destructive_approval(tmp_path: Path):
    agent, _, store = make_write_agent(tmp_path)
    response = run(agent.run("删除 release_20260819_a", "t"))
    item = store.get(response.approval_id)
    assert response.intent == AgentIntent.DELETE_TASK
    assert item.risk_level == "destructive"
    assert "cannot be undone" in " ".join(item.impact_details)


def test_priority_requires_explicit_numeric_value(tmp_path: Path):
    agent, client, store = make_write_agent(tmp_path)
    response = run(agent.run("让 release_20260819_a 先跑", "t"))
    assert response.intent == AgentIntent.SET_TASK_PRIORITY
    assert response.approval_required is False
    assert response.blocked is True
    assert "numeric priority" in response.summary
    assert store.list() == []
    assert not any(call.name in WRITE_TOOL_NAMES for call in client.calls)


def test_priority_approval_freezes_explicit_priority(tmp_path: Path):
    agent, _, store = make_write_agent(tmp_path)
    response = run(agent.run("把 release_20260819_a 优先级改成5", "t"))
    item = store.get(response.approval_id)
    assert item.tool_name == "set_task_priority"
    assert item.arguments["priority"] == 5
    assert item.precondition["queue_sha256"] == "q1"


def test_submit_consumes_valid_v06_plan_and_calls_validate_before_approval(tmp_path: Path):
    agent, client, store = make_write_agent(tmp_path)
    response = run(agent.run("创建一个release任务并提交，任务名release，数据 /data/record_001，完整流程", "t"))
    assert response.intent == AgentIntent.SUBMIT_TASK
    assert response.task_plan["valid"] is True
    assert response.approval_required is True
    names = [call.name for call in client.calls]
    assert "validate_task_spec" in names
    assert "get_write_precondition" in names
    assert not any(name in WRITE_TOOL_NAMES for name in names)
    item = store.get(response.approval_id)
    assert item.tool_name == "submit_task"
    assert item.arguments["config"] == response.task_plan["config"]


def test_invalid_submit_never_creates_approval(tmp_path: Path):
    agent, client, store = make_write_agent(tmp_path)
    response = run(agent.run("创建一个release任务并提交", "t"))
    assert response.intent == AgentIntent.SUBMIT_TASK
    assert response.task_plan["valid"] is False
    assert response.approval_required is False
    assert store.list() == []
    assert "validate_task_spec" not in [call.name for call in client.calls]


def test_approve_executes_exact_frozen_action_with_precondition(tmp_path: Path):
    agent, client, store = make_write_agent(tmp_path)
    response = run(agent.run("把 release_20260819_a 优先级改成5", "t"))
    item_before = store.get(response.approval_id)
    executed = run(agent.approve(response.approval_id))
    assert executed.status == "executed"
    write_calls = [call for call in client.calls if call.name == "set_task_priority"]
    assert len(write_calls) == 1
    assert write_calls[0].arguments["task_name"] == item_before.arguments["task_name"]
    assert write_calls[0].arguments["priority"] == item_before.arguments["priority"]
    assert write_calls[0].arguments["precondition"] == item_before.precondition


def test_write_tool_failure_marks_approval_failed(tmp_path: Path):
    client = FakeWriteToolClient(write_error="PRECONDITION_FAILED: queue_sha256")
    agent, _, store = make_write_agent(tmp_path, client)
    response = run(agent.run("删除 release_20260819_a", "t"))
    result = run(agent.approve(response.approval_id))
    assert result.status == "failed"
    assert "PRECONDITION_FAILED" in result.error
    assert store.get(response.approval_id).status == "failed"


def test_rejected_approval_cannot_execute(tmp_path: Path):
    agent, client, _ = make_write_agent(tmp_path)
    response = run(agent.run("删除 release_20260819_a", "t"))
    agent.reject(response.approval_id)
    with pytest.raises(RuntimeError, match="not pending"):
        run(agent.approve(response.approval_id))
    assert not any(call.name == "delete_task" for call in client.calls)


def test_model_cannot_put_write_tool_in_normal_tool_calls(tmp_path: Path):
    class MaliciousModel:
        requires_tool_descriptions = False
        async def plan(self, *_):
            return AgentPlan(
                intent=AgentIntent.DELETE_TASK,
                task_name="release_a",
                tool_calls=[ToolCallSpec(name="delete_task", arguments={"task_name": "release_a"})],
                write_action={"task_name": "release_a"},
            )
        async def synthesize(self, *args, **kwargs):
            raise AssertionError("must not synthesize")

    client = FakeWriteToolClient()
    agent = build_agent_runtime(
        "sequential", MaliciousModel(), client, ConversationStore(tmp_path / "s"),
        approval_store=ApprovalStore(tmp_path / "a"), task_planning_service=planning_service(),
    )
    with pytest.raises(PermissionError, match="before HITL approval"):
        run(agent.run("delete", "t"))
    assert client.calls == []


def test_approval_claim_is_exactly_once(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals")
    item = store.create(
        thread_id="t", user_request="x", tool_name="stop_task", arguments={}, precondition={},
        risk_level="high", impact_summary="stop",
    )
    claimed = store.claim_for_execution(item.approval_id)
    assert claimed.status == "executing"
    with pytest.raises(RuntimeError, match="not pending"):
        store.claim_for_execution(item.approval_id)


def test_real_precondition_change_between_approval_and_execution_fails(tmp_path: Path):
    mutation, gateway, queue_file, task_root = make_mutation_service(tmp_path)
    task_name = "release_20260819_120000"
    write_task_config(task_root, task_name)

    class MiniFacade:
        def get_task_detail(self, task_name, include_airflow_runs=True, run_limit=20):
            return {"task_name": task_name, "priority": 30, "datasets": ["record_001"]}
        def get_queue_state(self, task_name=""):
            return mutation.preconditions.queue_service.snapshot()
        def get_write_precondition(self, task_name=""):
            return mutation.capture_precondition(task_name)
        def validate_task_spec(self, task_prefix, config):
            return mutation.validate_task_spec(task_prefix, config)
        def set_task_priority(self, task_name, priority, precondition):
            return mutation.set_task_priority(task_name, priority, precondition)

    client = FacadeToolClient(MiniFacade())
    store = ApprovalStore(tmp_path / "approvals")
    agent = build_agent_runtime(
        "sequential", HeuristicReadOnlyModel(), client, ConversationStore(tmp_path / "sessions"),
        task_planning_service=planning_service(), approval_store=store,
    )
    response = run(agent.run(f"把 {task_name} 优先级改成5", "t"))
    assert response.approval_required is True
    # Simulate another scheduler/process changing global queue during human review.
    queue_file.write_text(
        json.dumps({"version": 2, "active": {"task_name": "another_task", "priority": 1}, "queue": []}),
        encoding="utf-8",
    )
    executed = run(agent.approve(response.approval_id))
    assert executed.status == "failed"
    assert "PRECONDITION_FAILED" in executed.error
    assert gateway.calls == []
