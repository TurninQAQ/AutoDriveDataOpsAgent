from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from platform_agent.memory import ConversationStore
from platform_agent.models import (
    AgentGoal,
    AgentIntent,
    AgentPlan,
    AgentResponse,
    AgentStepAction,
    AgentStepDecision,
    GoalProgress,
    GoalType,
    ToolCallSpec,
    ToolObservation,
)
from platform_agent.tool_catalog import build_read_only_tool_catalog
from platform_agent.workflow import build_agent_runtime


def call(name: str, **arguments):
    return AgentStepDecision(
        action=AgentStepAction.CALL_TOOL,
        tool_call=ToolCallSpec(name=name, arguments=arguments),
        decision_summary=f"collect {name}",
    )


def finish():
    return AgentStepDecision(
        action=AgentStepAction.FINISH,
        evidence_sufficient=True,
        decision_summary="goal check complete",
    )


class _Model:
    requires_tool_descriptions = True

    def __init__(self, plan, decisions):
        self.plan_value = plan
        self.decisions = list(decisions)
        self.goal_contracts = []

    async def plan(self, user_text, tool_descriptions, history):
        del user_text, tool_descriptions, history
        return self.plan_value

    async def decide_next(self, **kwargs):
        self.goal_contracts.append(kwargs.get("goal_contract"))
        return self.decisions.pop(0)

    async def synthesize(self, user_text, plan, observations, history, knowledge=None):
        del user_text, observations, history, knowledge
        return AgentResponse(intent=plan.intent, summary="synthesized", confidence="high")


class _Client:
    async def describe_tools(self):
        return build_read_only_tool_catalog()

    async def execute(self, calls):
        assert len(calls) == 1
        call_spec = calls[0]
        if call_spec.name == "diagnose_task":
            data = {"task_name": "release_demo", "reason": "waiting_gpu"}
        else:
            data = {"results": [{"content": "soft preemption rule"}]}
        return [ToolObservation(tool_name=call_spec.name, arguments=call_spec.arguments, ok=True, data=data)]


def _plan():
    return AgentPlan(
        intent=AgentIntent.TASK_DIAGNOSIS,
        task_name="release_demo",
        goal=AgentGoal(
            goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES,
            target="release_demo",
        ),
    )


@pytest.mark.parametrize("runtime", ["sequential", "langgraph"])
def test_final_goal_and_completion_state_are_consistent(runtime, tmp_path: Path):
    pytest.importorskip("langgraph") if runtime == "langgraph" else None
    model = _Model(
        _plan(),
        [
            call("diagnose_task", task_name="release_demo"),
            call("search_knowledge", query="soft preemption"),
            finish(),
        ],
    )
    agent = build_agent_runtime(
        runtime,
        model,
        _Client(),
        ConversationStore(tmp_path / runtime / "sessions"),
        max_tool_calls=3,
        max_steps=3,
    )

    response = asyncio.run(agent.run("release_demo 为什么没继续并结合规则解释", f"sync-{runtime}"))

    assert response.goal_progress == GoalProgress.SATISFIED
    assert response.goal is not None
    assert response.goal.completion_state == response.goal_progress
    assert response.termination_reason == "goal_satisfied"
    assert model.goal_contracts[0]["required_conditions"] == ["DIAGNOSIS", "STATIC_KNOWLEDGE"]


def test_incomplete_goal_state_is_consistent_after_finish_without_diagnosis(tmp_path: Path):
    model = _Model(
        _plan(),
        [
            call("get_task_detail", task_name="release_demo"),
            call("search_knowledge", query="soft preemption"),
            finish(),
        ],
    )
    agent = build_agent_runtime(
        "sequential",
        model,
        _Client(),
        ConversationStore(tmp_path / "sessions"),
        max_tool_calls=4,
        max_steps=4,
    )

    response = asyncio.run(agent.run("release_demo 为什么 draining 并结合规则解释", "incomplete"))

    assert response.termination_reason == "goal_incomplete"
    assert response.goal_progress == GoalProgress.IN_PROGRESS
    assert response.goal is not None
    assert response.goal.completion_state == GoalProgress.IN_PROGRESS


def test_blocked_goal_state_is_consistent_after_budget_termination(tmp_path: Path):
    model = _Model(_plan(), [call("get_task_detail", task_name="release_demo")])
    agent = build_agent_runtime(
        "sequential",
        model,
        _Client(),
        ConversationStore(tmp_path / "sessions"),
        max_tool_calls=1,
        max_steps=2,
    )

    response = asyncio.run(agent.run("release_demo 为什么 draining 并结合规则解释", "blocked"))

    assert response.termination_reason == "tool_budget_exhausted"
    assert response.goal_progress == GoalProgress.BLOCKED
    assert response.goal is not None
    assert response.goal.completion_state == GoalProgress.BLOCKED
