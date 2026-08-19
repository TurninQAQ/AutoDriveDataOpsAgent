from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

from platform_agent.approval import ApprovalStore
from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import ToolCallSpec, ToolObservation
from platform_agent.settings import AgentSettings
from platform_agent.verification import ActionVerificationResult, VerificationCheck
from platform_agent.workflow import build_agent_runtime
from platform_core.settings import PlatformSettings
from platform_eval import evaluate_agent_suite
from platform_observability import ObservedToolClient, TraceRecorder, TraceStore
from platform_observability.redaction import REDACTED, sanitize
from platform_planning.service import TaskPlanningService


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeToolClient:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls: list[ToolCallSpec] = []

    async def describe_tools(self):
        return []

    async def execute(self, calls):
        self.calls.extend(calls)
        out = []
        for call in calls:
            value = self.results.get(call.name, {})
            if callable(value):
                value = value(call)
            if isinstance(value, dict) and value.get("__error__"):
                out.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error=value["__error__"]))
            else:
                out.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=value))
        return out


class AlwaysVerified:
    async def verify(self, *, action, arguments, execution_result, baseline=None):
        return ActionVerificationResult(
            action=action,
            task_name=str(arguments.get("task_name") or "release_demo"),
            status="verified",
            attempts=1,
            checks=[VerificationCheck(name="fixture", passed=True, expected=True, actual=True)],
            snapshot={"fixture": True},
        )


def _recorder(tmp: Path):
    store = TraceStore(tmp / "traces", tmp / "audit" / "audit.jsonl")
    return TraceRecorder(store), store


def test_secret_redaction_is_recursive_and_persisted_trace_never_contains_secret():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        recorder, store = _recorder(root)
        trace_id = recorder.start_trace(kind="agent_request", user_request="test", thread_id="t")
        recorder.record(
            trace_id,
            "tool",
            "secret_tool",
            data={
                "authorization": "Bearer abcdefghijklmnop",
                "nested": {"password": "super-secret", "api_key": "sk-1234567890abcdef"},
                "message": "OPENAI_API_KEY=sk-abcdef1234567890",
            },
        )
        recorder.finish(
            trace_id, status="ok", intent="general_read",
            response_summary="OPENAI_API_KEY=sk-response123456789",
            errors=["AIRFLOW_API_TOKEN=runtime-token-123456"],
        )
        raw = store.trace_path(trace_id).read_text(encoding="utf-8") + store.audit_file.read_text(encoding="utf-8")
        assert "super-secret" not in raw
        assert "sk-1234567890abcdef" not in raw
        assert "sk-abcdef1234567890" not in raw
        assert "abcdefghijklmnop" not in raw
        assert "sk-response123456789" not in raw
        assert "runtime-token-123456" not in raw
        assert REDACTED in raw


def test_trace_store_records_request_plan_tool_response_and_audit():
    async def run_case(root: Path):
        recorder, store = _recorder(root)
        raw = FakeToolClient({
            "get_task_detail": {"task_name": "release_demo", "airflow_runs": [{"state": "running"}]},
            "get_queue_state": {"location": "active"},
        })
        client = ObservedToolClient(raw, recorder)
        agent = build_agent_runtime(
            "sequential",
            HeuristicReadOnlyModel(),
            client,
            ConversationStore(root / "memory"),
            trace_recorder=recorder,
        )
        response = await agent.run("release_demo 现在是什么状态？", thread_id="trace-test")
        assert response.trace_id
        events = store.load_events(response.trace_id)
        stages = [item.stage for item in events]
        assert stages[0] == "request"
        assert "plan" in stages
        assert "retrieval" in stages
        assert stages.count("tool") == 2
        assert stages[-1] == "response"
        audit = store.load_audit()[-1]
        assert audit.trace_id == response.trace_id
        assert audit.intent == "task_status"
        assert [item["tool"] for item in audit.tool_calls] == ["get_task_detail", "get_queue_state"]
        assert audit.status == "ok"
        assert audit.latency_ms >= 0

    with tempfile.TemporaryDirectory() as td:
        asyncio.run(run_case(Path(td)))


def test_approval_execution_creates_child_trace_linked_to_origin_and_records_mutation_verification():
    async def run_case(root: Path):
        recorder, store = _recorder(root)
        raw = FakeToolClient({
            "get_task_detail": {"task_name": "release_demo", "airflow_runs": [{"state": "running"}]},
            "get_queue_state": {"location": "active", "active": {"task_name": "release_demo", "priority": 20}},
            "get_write_precondition": {"queue_fingerprint": "q1", "task_config_fingerprint": "c1"},
            "get_action_verification_snapshot": {"task_exists": True, "queue": {"location": "active"}, "airflow_runs": []},
            "stop_task": {"ok": True, "result": {"task_name": "release_demo"}},
        })
        client = ObservedToolClient(raw, recorder)
        approvals = ApprovalStore(root / "approvals", ttl_sec=300)
        agent = build_agent_runtime(
            "sequential",
            HeuristicReadOnlyModel(),
            client,
            ConversationStore(root / "memory"),
            approval_store=approvals,
            action_verifier=AlwaysVerified(),
            trace_recorder=recorder,
        )
        prepared = await agent.run("停止 release_demo", thread_id="write-thread")
        assert prepared.approval_required is True
        assert prepared.trace_id
        pending = approvals.get(prepared.approval_id)
        assert pending.trace_id == prepared.trace_id
        item = await agent.approve(prepared.approval_id)
        assert item.status == "executed"
        assert item.execution_trace_id
        child_events = store.load_events(item.execution_trace_id)
        assert any(event.stage == "mutation" and event.name == "stop_task" for event in child_events)
        assert any(event.stage == "verification" and event.status == "verified" for event in child_events)
        child_audit = next(row for row in store.load_audit() if row.trace_id == item.execution_trace_id)
        assert child_audit.parent_trace_id == prepared.trace_id
        assert child_audit.status == "executed"
        assert child_audit.mutations and child_audit.verification

    with tempfile.TemporaryDirectory() as td:
        asyncio.run(run_case(Path(td)))


def test_agent_evaluation_suite_is_deterministic_and_all_metrics_pass():
    result = evaluate_agent_suite(
        REPO_ROOT / "eval" / "agent_cases.json",
        REPO_ROOT / "eval" / "task_planning_cases.json",
        TaskPlanningService.from_env(),
    )
    assert result["intent_accuracy"] == 1.0
    assert result["tool_selection_accuracy"] == 1.0
    assert result["diagnosis_accuracy"] == 1.0
    assert result["unsafe_action_rate"] == 0.0
    assert result["task_planning_accuracy"] == 1.0
    assert result["verification_accuracy"] == 1.0
    assert result["overall_score"] == 1.0


def test_agent_settings_observability_defaults_follow_state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRFLOW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("PLATFORM_AGENT_TRACE_DIR", raising=False)
    monkeypatch.delenv("PLATFORM_AGENT_AUDIT_FILE", raising=False)
    platform_settings = PlatformSettings.from_env()
    settings = AgentSettings.from_env(platform_settings)
    assert settings.trace_enabled is True
    assert settings.trace_dir == platform_settings.state_dir / "agent_traces"
    assert settings.audit_file == platform_settings.state_dir / "agent_audit" / "audit.jsonl"
    assert settings.trace_max_value_chars == 16000


def test_v09_cli_registers_trace_and_eval_commands():
    from platform_agent.cli import parser

    help_text = parser().format_help()
    assert "traces" in help_text
    assert "trace" in help_text
    assert "eval" in help_text
    assert "read-only Agent" in help_text


def test_deploy_copies_observability_and_eval_packages():
    with tempfile.TemporaryDirectory() as td:
        runtime = Path(td)
        env = os.environ.copy()
        env.update({
            "DEPLOY_SKIP_VERIFY": "1",
            "PLATFORM_HOME": str(runtime),
            "AIRFLOW_HOME": str(runtime / "airflow"),
            "AIRFLOW_BIN": "/bin/true",
            "AIRFLOW_DAGS_DIR": str(runtime / "airflow" / "dags" / "data_center"),
            "AIRFLOW_HOST_DATA_ROOT": str(runtime / "opt_airflow" / "data"),
            "AIRFLOW_CONFIG_DIR": str(runtime / "opt_airflow" / "config"),
            "AIRFLOW_SCRIPTS_DIR": str(runtime / "opt_airflow" / "scripts"),
            "AIRFLOW_TASK_CONFIG_ROOT": str(runtime / "opt_airflow" / "config" / "tasks"),
            "AIRFLOW_PLATFORM_OBSERVABILITY_DIR": str(runtime / "opt_airflow" / "platform_observability"),
            "AIRFLOW_PLATFORM_EVAL_DIR": str(runtime / "opt_airflow" / "platform_eval"),
        })
        result = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "deploy_ci_cloud.sh")], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (runtime / "opt_airflow" / "platform_observability" / "recorder.py").is_file()
        assert (runtime / "opt_airflow" / "platform_eval" / "service.py").is_file()
        assert (runtime / "opt_airflow" / "eval" / "agent_cases.json").is_file()


def test_platform_install_persists_v09_observability_environment_contract():
    text = (REPO_ROOT / "platform").read_text(encoding="utf-8")
    for name in (
        "PLATFORM_AGENT_TRACE_ENABLED",
        "PLATFORM_AGENT_TRACE_DIR",
        "PLATFORM_AGENT_AUDIT_FILE",
        "PLATFORM_AGENT_TRACE_MAX_VALUE_CHARS",
        "AIRFLOW_PLATFORM_OBSERVABILITY_DIR",
        "AIRFLOW_PLATFORM_EVAL_DIR",
    ):
        assert name in text


def test_redaction_helper_handles_secret_keys_without_mutating_normal_fields():
    value = sanitize({"token": "abc", "normal": "visible", "nested": {"password": "p"}})
    assert value["token"] == REDACTED
    assert value["nested"]["password"] == REDACTED
    assert value["normal"] == "visible"
