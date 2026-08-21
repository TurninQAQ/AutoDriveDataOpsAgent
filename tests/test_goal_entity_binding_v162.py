from __future__ import annotations

from platform_agent.evidence import EvidenceTracker
from platform_agent.goal import evidence_matches_goal_target, evaluate_goal_progress, resolve_goal_contract
from platform_agent.models import AgentGoal, AgentIntent, EvidenceType, GoalProgress, GoalType, ToolObservation


def observation(tool: str, data, **arguments):
    return ToolObservation(tool_name=tool, arguments=arguments, ok=True, data=data)


def evaluate(goal, contract, observations):
    tracker = EvidenceTracker.from_observations(observations)
    return evaluate_goal_progress(goal, tracker.records, observations, goal_contract=contract)


def test_wrong_task_diagnostic_context_cannot_complete_target_goal():
    goal = AgentGoal(goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES, target="release_demo")
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS)
    other = observation(
        "diagnose_task",
        {"task_name": "other_task", "queue": {"location": "queued"}, "evidence_complete": True},
        task_name="other_task",
    )
    rules = observation("search_knowledge", {"results": [{"content": "soft preemption"}]}, query="soft preemption")
    result = evaluate(goal, contract, [other, rules])
    assert result.state == GoalProgress.IN_PROGRESS
    assert result.missing_conditions == [EvidenceType.DIAGNOSTIC_CONTEXT.value]


def test_wrong_task_recovery_evidence_cannot_complete_target_goal():
    goal = AgentGoal(goal_type=GoalType.VERIFY_RECOVERY_STATE, target="release_demo")
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_STATUS)
    target_state = observation(
        "get_task_detail", {"task_name": "release_demo", "state": "running"}, task_name="release_demo"
    )
    other_recovery = observation(
        "get_task_detail",
        {"task_name": "other_task", "recovery": {"checkpoint": "segment"}},
        task_name="other_task",
    )
    result = evaluate(goal, contract, [target_state, other_recovery])
    assert result.state == GoalProgress.IN_PROGRESS
    assert result.missing_conditions == [EvidenceType.RECOVERY_STATE.value]


def test_target_task_recovery_evidence_satisfies_recovery_goal():
    goal = AgentGoal(goal_type=GoalType.VERIFY_RECOVERY_STATE, target="release_demo")
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_STATUS)
    target = observation(
        "get_task_detail",
        {"task_name": "release_demo", "state": "running", "recovery": {"checkpoint": "segment"}},
        task_name="release_demo",
    )
    result = evaluate(goal, contract, [target])
    assert result.state == GoalProgress.SATISFIED


def test_global_gpu_evidence_remains_usable_for_targeted_gpu_goal():
    goal = AgentGoal(goal_type=GoalType.REPORT_LIVE_STATE, target="release_demo")
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.GPU_DIAGNOSIS)
    gpu = observation("get_gpu_pool", {"devices": [{"gpu_id": "0", "free_mb": 1024}]})
    result = evaluate(goal, contract, [gpu])
    assert result.state == GoalProgress.SATISFIED


def test_legacy_evidence_record_without_subject_remains_loadable_but_not_task_bound():
    tracker = EvidenceTracker.from_records(
        [{"type": "LIVE_TASK", "source_tool": "get_task_detail", "timestamp": 1.0, "summary": "legacy"}]
    )
    goal = AgentGoal(goal_type=GoalType.REPORT_LIVE_STATE, target="release_demo")
    assert tracker.records[0].task_name is None
    assert not evidence_matches_goal_target(tracker.records[0], goal)
