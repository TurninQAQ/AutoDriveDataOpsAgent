from __future__ import annotations

from platform_agent.evidence import EvidenceTracker, evidence_entity_conflict, is_diagnostic_context_payload
from platform_agent.goal import evaluate_goal_progress, goal_for_intent, resolve_goal_contract
from platform_agent.models import AgentIntent, EvidenceType, GoalProgress, ToolObservation


def observation(tool: str, data, **arguments):
    return ToolObservation(tool_name=tool, arguments=arguments, ok=True, data=data)


def records_for(*items):
    tracker = EvidenceTracker()
    for item in items:
        tracker.record_tool_observation(item)
    return tracker.records


def test_argument_payload_conflict_does_not_create_target_bound_context():
    item = observation(
        "diagnose_task",
        {"task_name": "other_task", "queue": {"state": "running"}, "evidence_complete": True},
        task_name="release_demo",
    )
    records = records_for(item)
    conflict = evidence_entity_conflict(item)
    assert conflict["task_conflict"] is True
    assert conflict["requested_task"] == "release_demo"
    assert conflict["observed_task"] == "other_task"
    assert not any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records)

    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    result = evaluate_goal_progress(
        goal,
        records,
        [item],
        goal_contract=resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS),
    )
    assert result.state == GoalProgress.IN_PROGRESS


def test_matching_argument_and_payload_create_context():
    item = observation(
        "diagnose_task",
        {"task_name": "release_demo", "queue": {"state": "running"}, "evidence_complete": True},
        task_name="release_demo",
    )
    assert any(record.type == EvidenceType.DIAGNOSTIC_CONTEXT for record in records_for(item))


def test_argument_only_and_payload_only_subject_fallbacks_remain_valid():
    argument_only = observation(
        "diagnose_task",
        {"queue": {"state": "running"}, "evidence_complete": False},
        task_name="release_demo",
    )
    payload_only = observation(
        "diagnose_task",
        {"task_name": "release_demo", "queue": {"state": "running"}, "evidence_complete": False},
    )
    for item in (argument_only, payload_only):
        records = records_for(item)
        context = [record for record in records if record.type == EvidenceType.DIAGNOSTIC_CONTEXT]
        assert context and context[0].task_name == "release_demo"


def test_wrong_target_plus_knowledge_does_not_false_complete_goal():
    bad = observation(
        "diagnose_task",
        {"task_name": "other_task", "queue": {"state": "running"}, "evidence_complete": True},
        task_name="release_demo",
    )
    rules = observation("search_knowledge", {"results": [{"content": "soft preemption"}]}, query="soft preemption")
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    result = evaluate_goal_progress(
        goal,
        records_for(bad, rules),
        [bad, rules],
        goal_contract=resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS),
    )
    assert result.state == GoalProgress.IN_PROGRESS
    assert result.missing_conditions == [EvidenceType.DIAGNOSTIC_CONTEXT.value]


def test_diagnostic_fact_meaningfulness_invariant():
    negative = [
        {},
        {"task_name": "release_demo"},
        {"evidence_complete": False},
        {"evidence_complete": True},
        {"queue": None},
        {"queue": {}},
        {"errors": []},
        {"datasets": []},
    ]
    positive = [
        {"queue": {"location": "queued"}},
        {"errors": [{"source": "airflow", "error": "unavailable"}]},
        {"airflow": {"latest_run": {"state": "queued"}}},
        {"queue": {"location": "queued"}, "evidence_complete": False},
    ]
    assert all(not is_diagnostic_context_payload(item) for item in negative)
    assert all(is_diagnostic_context_payload(item) for item in positive)
