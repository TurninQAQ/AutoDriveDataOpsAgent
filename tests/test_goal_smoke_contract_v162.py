from __future__ import annotations

from platform_agent.models import AgentResponse, AgentIntent, AgentGoal, GoalProgress
from scripts.smoke_goal_completion_v160 import CASES, FixtureToolClient, _case_result


def test_smoke_uses_response_goal_progress_and_production_recomputed_state():
    case = next(item for item in CASES if item["id"] == "hybrid_draining")
    client = FixtureToolClient(case["fixture_results"])
    client.calls = []
    client.observations = []
    # The case helper evaluates actual execution; use production-shaped results
    # and a response that claims the synthesized root cause.
    import asyncio
    asyncio.run(client.execute([type("Call", (), {"name": "diagnose_task", "arguments": {"task_name": "release_demo"}})()]))
    asyncio.run(client.execute([type("Call", (), {"name": "search_knowledge", "arguments": {"query": "soft preemption"}})()]))
    response = AgentResponse(
        intent=AgentIntent.TASK_DIAGNOSIS,
        summary="draining is waiting for the Stage boundary",
        root_cause="the task is draining while soft preemption waits for a Stage boundary",
        goal=AgentGoal(goal_type="EXPLAIN_WITH_PLATFORM_RULES", target="release_demo", completion_state=GoalProgress.SATISFIED),
        goal_progress=GoalProgress.SATISFIED,
    )
    result = _case_result(case, response, client)
    assert result["response_goal_progress"] == "SATISFIED"
    assert result["recomputed_goal_progress"] == "SATISFIED"
    assert result["goal_state_parity"] is True
    assert result["unnecessary_tool_count"] == 0


def test_smoke_counts_extra_read_tool_as_unnecessary():
    case = next(item for item in CASES if item["id"] == "knowledge")
    client = FixtureToolClient(case["fixture_results"])
    import asyncio
    asyncio.run(client.execute([type("Call", (), {"name": "search_knowledge", "arguments": {"query": "rule"}})()]))
    asyncio.run(client.execute([type("Call", (), {"name": "get_gpu_pool", "arguments": {}})()]))
    response = AgentResponse(
        intent=AgentIntent.PLATFORM_KNOWLEDGE,
        summary="knowledge",
        goal=AgentGoal(goal_type="ANSWER_KNOWLEDGE", completion_state=GoalProgress.SATISFIED),
        goal_progress=GoalProgress.SATISFIED,
    )
    result = _case_result(case, response, client)
    assert result["unnecessary_tool_count"] == 1
    assert result["smoke_case_valid"] is False
