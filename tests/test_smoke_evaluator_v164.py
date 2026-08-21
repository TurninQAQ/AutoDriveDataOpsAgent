from __future__ import annotations

import asyncio

from platform_agent.models import AgentGoal, AgentIntent, AgentResponse, GoalProgress
from scripts.smoke_goal_completion_v160 import CASES, FixtureToolClient, _case_result


def test_functional_gate_can_pass_when_only_efficiency_varies():
    case = next(item for item in CASES if item["id"] == "knowledge")
    client = FixtureToolClient(case["fixture_results"])
    asyncio.run(client.execute([type("Call", (), {"name": "search_knowledge", "arguments": {}})()]))
    asyncio.run(client.execute([type("Call", (), {"name": "get_gpu_pool", "arguments": {}})()]))
    response = AgentResponse(
        intent=AgentIntent.PLATFORM_KNOWLEDGE,
        summary="knowledge",
        goal=AgentGoal(goal_type="ANSWER_KNOWLEDGE", completion_state=GoalProgress.SATISFIED),
        goal_progress=GoalProgress.SATISFIED,
    )
    result = _case_result(case, response, client)
    assert result["correctness_ok"] is True
    assert result["safety_ok"] is True
    assert result["functional_case_valid"] is True
    assert result["strict_case_valid"] is False


def test_task_planning_parity_is_recomputed_from_response_contract():
    case = next(item for item in CASES if item["id"] == "task_planning")
    response = AgentResponse(
        intent=AgentIntent.TASK_PLANNING,
        summary="task plan",
        task_plan={"valid": True},
        goal=AgentGoal(goal_type="PREPARE_TASK_PLAN", completion_state=GoalProgress.IN_PROGRESS),
        goal_progress=GoalProgress.IN_PROGRESS,
    )
    result = _case_result(case, response, FixtureToolClient(case["fixture_results"]))
    assert result["recomputed_goal_progress"] == "SATISFIED"
    assert result["goal_state_parity"] is False
    assert result["correctness_ok"] is False


def test_write_parity_is_recomputed_from_pending_action():
    case = next(item for item in CASES if item["id"] == "write")
    response = AgentResponse(
        intent=AgentIntent.STOP_TASK,
        summary="approval required",
        approval_required=True,
        pending_action={"approval_id": "approval-1"},
        goal=AgentGoal(goal_type="PREPARE_WRITE_ACTION", completion_state=GoalProgress.IN_PROGRESS),
        goal_progress=GoalProgress.IN_PROGRESS,
    )
    result = _case_result(case, response, FixtureToolClient(case["fixture_results"]))
    assert result["recomputed_goal_progress"] == "SATISFIED"
    assert result["goal_state_parity"] is False
    assert result["correctness_ok"] is False
