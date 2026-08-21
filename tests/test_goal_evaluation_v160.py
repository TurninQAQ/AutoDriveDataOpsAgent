from __future__ import annotations

from platform_agent.goal import evaluate_goal_progress, goal_for_intent
from platform_agent.models import (
    AgentIntent,
    EvidenceType,
    GoalProgress,
    GoalType,
    KnowledgeObservation,
    ToolObservation,
)
from platform_agent.evidence import EvidenceTracker


def observation(tool: str, data, **arguments):
    return ToolObservation(tool_name=tool, arguments=arguments, ok=True, data=data)


def records_for(*observations):
    tracker = EvidenceTracker()
    for item in observations:
        tracker.record_tool_observation(item)
    return tracker.records


def test_knowledge_goal_requires_static_knowledge():
    goal = goal_for_intent(AgentIntent.PLATFORM_KNOWLEDGE)
    before = evaluate_goal_progress(goal)
    after = evaluate_goal_progress(
        goal,
        records_for(observation("search_knowledge", {"results": [{"content": "rule"}]})),
    )
    assert before.state == GoalProgress.IN_PROGRESS
    assert after.state == GoalProgress.SATISFIED


def test_live_state_goal_does_not_require_diagnosis_or_knowledge():
    goal = goal_for_intent(AgentIntent.TASK_STATUS, target="release_demo")
    live = observation("get_task_detail", {"task_name": "release_demo", "state": "running"}, task_name="release_demo")
    result = evaluate_goal_progress(
        goal,
        records_for(live),
        [live],
    )
    assert result.state == GoalProgress.SATISFIED
    assert "STATIC_KNOWLEDGE" not in result.satisfied_conditions


def test_hybrid_goal_requires_both_live_and_static_evidence():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo").model_copy(
        update={
            "goal_type": GoalType.EXPLAIN_WITH_PLATFORM_RULES,
            "success_criteria": ["ignored"],
        }
    )
    live = observation(
        "diagnose_task",
        {
            "task_name": "release_demo",
            "queue": {"state": "draining"},
            "errors": [],
            "evidence_complete": True,
        },
        task_name="release_demo",
    )
    static = observation("search_knowledge", {"results": [{"content": "soft preemption"}]}, query="soft preemption")
    only_live = evaluate_goal_progress(goal, records_for(live), [live])
    complete = evaluate_goal_progress(goal, records_for(live, static), [live, static])
    assert only_live.state == GoalProgress.IN_PROGRESS
    assert only_live.missing_conditions == ["STATIC_KNOWLEDGE"]
    assert complete.state == GoalProgress.SATISFIED


def test_diagnosis_requires_target_bound_diagnostic_context():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    state_only = observation("get_task_detail", {"task_name": "release_demo", "current_stage": "segment"}, task_name="release_demo")
    diagnosis = observation("diagnose_task", {"task_name": "release_demo", "queue": {"location": "queued"}, "evidence_complete": True}, task_name="release_demo")
    assert evaluate_goal_progress(goal, records_for(state_only), [state_only]).state == GoalProgress.IN_PROGRESS
    result = evaluate_goal_progress(goal, records_for(diagnosis), [diagnosis])
    assert result.state == GoalProgress.SATISFIED
    assert EvidenceType.DIAGNOSTIC_CONTEXT.value in [item.type.value for item in records_for(diagnosis)]


def test_recovery_goal_requires_live_and_recovery_evidence():
    goal = goal_for_intent(AgentIntent.TASK_STATUS, target="release_demo").model_copy(
        update={"goal_type": GoalType.VERIFY_RECOVERY_STATE}
    )
    live = observation("get_task_detail", {"task_name": "release_demo", "state": "running"}, task_name="release_demo")
    recovery = observation(
        "diagnose_task",
        {"task_name": "release_demo", "recovery": {"checkpoint": "segment", "state": "running"}},
        task_name="release_demo",
    )
    assert evaluate_goal_progress(goal, records_for(live), [live]).state == GoalProgress.IN_PROGRESS
    assert evaluate_goal_progress(goal, records_for(recovery), [recovery]).state == GoalProgress.SATISFIED


def test_legacy_knowledge_observation_counts_as_static_evidence():
    goal = goal_for_intent(AgentIntent.PLATFORM_KNOWLEDGE)
    item = KnowledgeObservation(
        chunk_id="rule-1", source_path="rules.md", title="rules", content="soft preemption", score=1.0
    )
    assert evaluate_goal_progress(goal, knowledge=[item]).state == GoalProgress.SATISFIED
