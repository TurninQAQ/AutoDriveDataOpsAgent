from __future__ import annotations

from platform_agent.evidence import EvidenceTracker, is_diagnostic_context_payload
from platform_agent.goal import evaluate_goal_progress, goal_for_intent, resolve_goal_contract
from platform_agent.models import AgentIntent, EvidenceType, GoalProgress, ToolObservation


def observation(tool: str, data, **arguments):
    return ToolObservation(tool_name=tool, arguments=arguments, ok=True, data=data)


def records_for(*items):
    tracker = EvidenceTracker()
    for item in items:
        tracker.record_tool_observation(item)
    return tracker.records


def test_empty_or_non_contract_diagnosis_payload_is_not_context():
    for data in ({}, {"task_name": "release_demo"}, {"foo": "bar"}, {"message": "ok"}):
        assert is_diagnostic_context_payload(data) is False
        records = records_for(observation("diagnose_task", data, task_name="release_demo"))
        assert not any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records)


def test_production_shape_is_context_even_when_partial():
    for data in (
        {
            "task_name": "release_demo",
            "queue": {"state": "draining"},
            "errors": [],
            "evidence_complete": True,
        },
        {
            "task_name": "release_demo",
            "queue": {"state": "draining"},
            "airflow": None,
            "errors": [{"source": "airflow", "error": "unavailable"}],
            "evidence_complete": False,
        },
    ):
        records = records_for(observation("diagnose_task", data, task_name="release_demo"))
        assert any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records)


def test_wrong_target_context_does_not_complete_release_demo():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    other_task = observation(
        "diagnose_task",
        {"task_name": "other_task", "queue": {}, "evidence_complete": True},
        task_name="other_task",
    )
    result = evaluate_goal_progress(
        goal,
        records_for(other_task),
        [other_task],
        goal_contract=resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS),
    )
    assert result.state == GoalProgress.IN_PROGRESS
    assert result.missing_conditions == [EvidenceType.DIAGNOSTIC_CONTEXT.value]


def test_empty_and_non_empty_stage_logs_keep_existing_contract():
    empty = observation("get_stage_logs", {"task_name": "release_demo", "logs": []}, task_name="release_demo")
    non_empty = observation(
        "get_stage_logs",
        {"task_name": "release_demo", "logs": [{"line": "validation failed"}]},
        task_name="release_demo",
    )
    empty_records = records_for(empty)
    non_empty_records = records_for(non_empty)
    assert not any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in empty_records)
    assert any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in non_empty_records)
