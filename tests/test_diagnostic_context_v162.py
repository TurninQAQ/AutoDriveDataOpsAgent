from __future__ import annotations

from platform_agent.evidence import EvidenceTracker
from platform_agent.goal import evaluate_goal_progress, finalize_goal_response, goal_for_intent, resolve_goal_contract
from platform_agent.models import AgentIntent, AgentResponse, EvidenceType, GoalProgress, GoalType, ToolObservation


def observation(tool: str, data, **arguments):
    return ToolObservation(tool_name=tool, arguments=arguments, ok=True, data=data)


def test_production_shaped_diagnose_task_creates_diagnostic_context_without_fake_reason():
    item = observation(
        "diagnose_task",
        {
            "task_name": "release_demo",
            "datasets": ["clip_001"],
            "queue": {"location": "draining"},
            "airflow": {"latest_run": {"state": "running"}, "task_instances": []},
            "containers": [],
            "gpu_reservations": [],
            "gpu_devices": [],
            "errors": [],
            "evidence_complete": True,
        },
        task_name="release_demo",
    )
    records = EvidenceTracker.from_observations([item]).records
    assert any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records)
    assert not any(record.type == EvidenceType.DIAGNOSIS for record in records)


def test_empty_stage_logs_do_not_create_diagnostic_context():
    item = observation("get_stage_logs", {"logs": []}, task_name="release_demo")
    records = EvidenceTracker.from_observations([item]).records
    assert any(record.type == EvidenceType.LIVE_LOG for record in records)
    assert not any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records)


def test_non_empty_target_stage_logs_can_create_diagnostic_context():
    item = observation("get_stage_logs", {"logs": [{"log": "validation failed"}]}, task_name="release_demo")
    records = EvidenceTracker.from_observations([item]).records
    assert any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records)


def test_diagnostic_context_requires_root_cause_at_finalization():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS)
    item = observation(
        "diagnose_task",
        {"task_name": "release_demo", "queue": {"location": "queued"}, "evidence_complete": True},
        task_name="release_demo",
    )
    records = EvidenceTracker.from_observations([item]).records
    evaluation = evaluate_goal_progress(goal, records, [item], goal_contract=contract)
    assert evaluation.state == GoalProgress.SATISFIED
    incomplete = finalize_goal_response(
        goal,
        contract,
        evaluation,
        AgentResponse(intent=AgentIntent.TASK_DIAGNOSIS, summary="facts only", root_cause=None),
    )
    assert incomplete.state == GoalProgress.IN_PROGRESS
    assert "ROOT_CAUSE_CONCLUSION" in incomplete.missing_conditions
    complete = finalize_goal_response(
        goal,
        contract,
        evaluation,
        AgentResponse(intent=AgentIntent.TASK_DIAGNOSIS, summary="diagnosed", root_cause="waiting for GPU"),
    )
    assert complete.state == GoalProgress.SATISFIED


def test_static_knowledge_alone_never_satisfies_task_diagnosis():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    rules = observation("search_knowledge", {"results": [{"content": "soft preemption"}]}, query="soft preemption")
    result = evaluate_goal_progress(
        goal,
        EvidenceTracker.from_observations([rules]).records,
        [rules],
        goal_contract=resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS),
    )
    assert result.state == GoalProgress.IN_PROGRESS
