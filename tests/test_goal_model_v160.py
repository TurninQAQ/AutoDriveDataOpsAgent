from __future__ import annotations

import pytest

from platform_agent.goal import normalize_plan_goal
from platform_agent.models import (
    AgentGoal,
    AgentIntent,
    AgentPlan,
    GoalProgress,
    GoalType,
)


def test_goal_schema_serializes_and_uses_deterministic_criteria():
    goal = AgentGoal(
        goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES,
        target="release_demo",
        success_criteria=["provider supplied text is ignored"],
    )
    normalized = normalize_plan_goal(
        AgentPlan(intent=AgentIntent.TASK_DIAGNOSIS, task_name="release_demo", goal=goal)
    ).goal

    assert normalized is not None
    assert normalized.goal_type == GoalType.EXPLAIN_WITH_PLATFORM_RULES
    assert normalized.target == "release_demo"
    assert normalized.success_criteria == ["LIVE_OPERATIONAL_EVIDENCE", "STATIC_KNOWLEDGE"]
    assert normalized.completion_state == GoalProgress.NOT_STARTED
    assert "provider supplied text" not in normalized.model_dump_json()


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (AgentIntent.PLATFORM_KNOWLEDGE, GoalType.ANSWER_KNOWLEDGE),
        (AgentIntent.TASK_STATUS, GoalType.REPORT_LIVE_STATE),
        (AgentIntent.TASK_DIAGNOSIS, GoalType.DIAGNOSE_ROOT_CAUSE),
        (AgentIntent.TASK_PLANNING, GoalType.PREPARE_TASK_PLAN),
        (AgentIntent.STOP_TASK, GoalType.PREPARE_WRITE_ACTION),
    ],
)
def test_legacy_plan_without_goal_uses_intent_fallback(intent, expected):
    normalized = normalize_plan_goal(AgentPlan(intent=intent)).goal
    assert normalized is not None
    assert normalized.goal_type == expected


def test_invalid_goal_type_is_rejected():
    with pytest.raises(ValueError):
        AgentGoal.model_validate({"goal_type": "CALL_A_TOOL"})
