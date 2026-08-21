from platform_agent.autonomy import AutonomyMode, BoundedAutonomyPolicy, latest_dataset_states
from platform_agent.approval import ApprovalStore
from platform_agent.settings import AgentSettings


def precondition(active_task_name=""):
    return {
        "queue_sha256": "queue-1",
        "task_name": "release_demo",
        "task_config_sha256": "config-1",
        "task_exists": True,
        "active_task_name": active_task_name,
    }


def baseline(runs=None, *, task_name="release_demo", errors=None, available=None, exclusive=True):
    return {
        "task_name": task_name,
        "task_exists": True,
        "config_file_exists": True,
        "dag_file_exists": True,
        "airflow_dag_exists": True,
        "available_datasets": available or ["A", "B"],
        "task_exclusive": exclusive,
        "airflow_runs": runs if runs is not None else [
            {"run_id": "a-failed", "dataset_name": "A", "state": "failed"},
            {"run_id": "b-failed", "dataset_name": "B", "state": "failed"},
        ],
        "errors": errors or {},
    }


def enabled_policy(**kwargs):
    return BoundedAutonomyPolicy(enabled=True, **kwargs)


def test_safe_single_dataset_is_auto():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(),
        baseline=baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}]),
    )
    assert decision.mode == AutonomyMode.AUTO
    assert decision.frozen_arguments["datasets"] == ["A"]
    assert decision.risk_level == "low"


def test_explicit_multi_dataset_is_auto_when_all_currently_failed():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A", "B"]},
        precondition=precondition(),
        baseline=baseline(),
    )
    assert decision.mode == AutonomyMode.AUTO
    assert decision.frozen_arguments["datasets"] == ["A", "B"]


def test_empty_dataset_scope_is_frozen_from_current_failures():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": []},
        precondition=precondition(),
        baseline=baseline(),
    )
    assert decision.mode == AutonomyMode.AUTO
    assert decision.frozen_arguments["datasets"] == ["A", "B"]


def test_latest_success_removes_historical_failure_from_auto_scope():
    runs = [
        {"run_id": "a-success", "dataset_name": "A", "state": "success"},
        {"run_id": "a-failed", "dataset_name": "A", "state": "failed"},
    ]
    assert latest_dataset_states(runs) == {"A": "success"}
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": []},
        precondition=precondition(),
        baseline=baseline(runs=runs, available=["A"]),
    )
    assert decision.mode == AutonomyMode.DENY
    assert "no_currently_failed_dataset" in decision.reasons


def test_latest_failed_is_current_failure():
    runs = [
        {"run_id": "a-failed-new", "dataset_name": "A", "state": "failed"},
        {"run_id": "a-success-old", "dataset_name": "A", "state": "success"},
    ]
    assert latest_dataset_states(runs) == {"A": "failed"}


def test_requested_non_failed_dataset_falls_back_to_hitl():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A", "B"]},
        precondition=precondition(),
        baseline=baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}]),
    )
    assert decision.mode == AutonomyMode.HITL
    assert "requested_dataset_not_currently_failed" in decision.reasons


def test_dataset_budget_falls_back_to_hitl():
    policy = enabled_policy(max_resume_datasets=1)
    decision = policy.decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": []},
        precondition=precondition(),
        baseline=baseline(),
    )
    assert decision.mode == AutonomyMode.HITL
    assert "autonomy_dataset_budget_exceeded" in decision.reasons


def test_cross_task_preemption_falls_back_to_hitl():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(active_task_name="other_task"),
        baseline=baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}]),
    )
    assert decision.mode == AutonomyMode.HITL
    assert "cross_task_preemption_possible" in decision.reasons


def test_disabled_autonomy_is_hitl():
    decision = BoundedAutonomyPolicy(enabled=False).decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(),
        baseline=baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}]),
    )
    assert decision.mode == AutonomyMode.HITL


def test_non_resume_writes_are_never_auto():
    for action in ("submit_task", "set_task_priority", "stop_task", "delete_task"):
        decision = enabled_policy().decide(
            action=action,
            arguments={"task_name": "release_demo"},
            precondition=precondition(),
            baseline=baseline(),
        )
        assert decision.mode == AutonomyMode.HITL


def test_missing_target_is_denied():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"datasets": ["A"]},
        precondition=precondition(),
        baseline=baseline(),
    )
    assert decision.mode == AutonomyMode.DENY


def test_unknown_dataset_is_denied():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["unknown"]},
        precondition=precondition(),
        baseline=baseline(),
    )
    assert decision.mode == AutonomyMode.DENY
    assert "unknown_dataset" in decision.reasons


def test_task_missing_is_denied():
    data = baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}])
    data["task_exists"] = False
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(),
        baseline=data,
    )
    assert decision.mode == AutonomyMode.DENY


def test_critical_snapshot_error_is_denied():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(),
        baseline=baseline(
            runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}],
            errors={"airflow": "backend unavailable"},
        ),
    )
    assert decision.mode == AutonomyMode.DENY


def test_action_budget_prevents_duplicate_auto_authorization():
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(),
        baseline=baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}]),
        auto_actions_used=1,
    )
    assert decision.mode == AutonomyMode.HITL
    assert "autonomy_action_budget_exceeded" in decision.reasons


def test_persisted_auto_attempt_consumes_request_budget(tmp_path):
    store = ApprovalStore(tmp_path / "approvals")
    store.create_auto_execution(
        thread_id="t",
        user_request="resume",
        tool_name="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition={"queue_sha256": "q", "task_name": "release_demo"},
        risk_level="low",
        impact_summary="resume",
        trace_id="trace-1",
        policy_decision={"mode": "AUTO"},
    )
    assert store.count_auto_actions("trace-1") == 1
    decision = enabled_policy().decide(
        action="resume_task",
        arguments={"task_name": "release_demo", "datasets": ["A"]},
        precondition=precondition(),
        baseline=baseline(runs=[{"run_id": "a-failed", "dataset_name": "A", "state": "failed"}]),
        auto_actions_used=store.count_auto_actions("trace-1"),
    )
    assert decision.mode == AutonomyMode.HITL


def test_autonomy_settings_are_disabled_and_bounded_by_default(monkeypatch):
    for name in (
        "PLATFORM_AGENT_AUTONOMY_ENABLED",
        "PLATFORM_AGENT_AUTO_ACTIONS_PER_REQUEST",
        "PLATFORM_AGENT_AUTO_RESUME_MAX_DATASETS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = AgentSettings.from_env()
    assert settings.autonomy_enabled is False
    assert settings.auto_actions_per_request == 1
    assert settings.auto_resume_max_datasets == 3
