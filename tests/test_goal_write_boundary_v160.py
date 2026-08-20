from __future__ import annotations

from platform_agent.goal import goal_for_intent
from platform_agent.models import AgentIntent, GoalProgress, GoalType


def test_write_goal_means_prepare_and_never_mutation_completion():
    goal = goal_for_intent(AgentIntent.STOP_TASK, target="release_demo")
    assert goal.goal_type == GoalType.PREPARE_WRITE_ACTION
    assert "WRITE_PLAN_PREPARED" in goal.success_criteria
    assert goal.completion_state == GoalProgress.NOT_STARTED
