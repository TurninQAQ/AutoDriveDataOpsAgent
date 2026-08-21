from __future__ import annotations

import asyncio

from platform_agent.adaptive import AdaptiveLimits, AdaptiveLoopController
from platform_agent.goal import goal_for_intent
from platform_agent.model import build_adaptive_evidence_prompt
from platform_agent.models import (
    AgentGoal,
    AgentIntent,
    AgentPlan,
    AgentStepAction,
    AgentStepDecision,
    GoalProgress,
    GoalEvaluation,
    GoalType,
    ToolCallSpec,
    ToolObservation,
)
from platform_agent.policy import AgentPolicyEngine


def run(coro):
    return asyncio.run(coro)


class GoalAwareModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    async def decide_next(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0)


def call(name, **arguments):
    return AgentStepDecision(
        action=AgentStepAction.CALL_TOOL,
        tool_call=ToolCallSpec(name=name, arguments=arguments),
        decision_summary=f"Collect {name} evidence.",
    )


def finish():
    return AgentStepDecision(
        action=AgentStepAction.FINISH,
        evidence_sufficient=True,
        decision_summary="Goal evidence is sufficient.",
    )


def test_goal_progress_reaches_next_decision_and_finish_is_goal_satisfied():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    model = GoalAwareModel([call("diagnose_task", task_name="release_demo"), finish()])

    async def execute(call_spec):
        return [
            ToolObservation(
                tool_name=call_spec.name,
                arguments=call_spec.arguments,
                ok=True,
                data={
                    "task_name": "release_demo",
                    "queue": {"state": "waiting_gpu"},
                    "errors": [],
                    "evidence_complete": True,
                },
            )
        ]

    result = run(
        AdaptiveLoopController(
            model,
            AgentPolicyEngine(max_tool_calls=3),
            AdaptiveLimits(max_steps=3, max_tool_calls=3),
        ).run(
            user_text="release_demo 为什么没继续运行？",
            initial_plan=AgentPlan(intent=AgentIntent.TASK_DIAGNOSIS, task_name="release_demo", goal=goal),
            tool_descriptions=[{"name": "diagnose_task"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=execute,
            normalize_observation=lambda item: [],
            goal=goal,
        )
    )

    assert result.termination_reason == "goal_satisfied"
    assert result.goal_evaluation.state == GoalProgress.SATISFIED
    assert model.calls[1]["goal_evaluation"]["state"] == GoalProgress.SATISFIED.value
    assert model.calls[1]["goal"]["goal_type"] == GoalType.DIAGNOSE_ROOT_CAUSE.value
    assert result.steps[0]["goal_state_after"] == GoalProgress.SATISFIED.value


def test_finish_before_goal_completion_is_incomplete_not_success():
    goal = goal_for_intent(AgentIntent.PLATFORM_KNOWLEDGE)
    model = GoalAwareModel([finish()])
    result = run(
        AdaptiveLoopController(
            model,
            AgentPolicyEngine(max_tool_calls=1),
            AdaptiveLimits(max_steps=1, max_tool_calls=1),
        ).run(
            user_text="解释平台规则",
            initial_plan=AgentPlan(intent=AgentIntent.PLATFORM_KNOWLEDGE, goal=goal),
            tool_descriptions=[{"name": "search_knowledge"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=lambda call_spec: None,
            normalize_observation=lambda item: [],
            goal=goal,
        )
    )
    assert result.termination_reason == "goal_incomplete"
    assert result.evidence_sufficient is False
    assert result.goal_evaluation.state == GoalProgress.IN_PROGRESS


def test_adaptive_prompt_exposes_goal_progress_without_hidden_reasoning():
    goal = AgentGoal(
        goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES,
        target="release_demo",
        success_criteria=["LIVE_OPERATIONAL_EVIDENCE", "STATIC_KNOWLEDGE"],
    )
    prompt = build_adaptive_evidence_prompt(
        user_text="release_demo 为什么 draining？结合平台规则解释",
        initial_plan=AgentPlan(intent=AgentIntent.TASK_DIAGNOSIS, task_name="release_demo", goal=goal),
        current_intent=AgentIntent.TASK_DIAGNOSIS,
        tool_descriptions=[{"name": "diagnose_task"}, {"name": "search_knowledge"}],
        observations=[],
        knowledge=[],
        history=[],
        step_index=0,
        remaining_tool_calls=6,
        goal=goal,
        goal_evaluation=GoalEvaluation(
            state=GoalProgress.IN_PROGRESS,
            missing_conditions=["LIVE_OPERATIONAL_EVIDENCE", "STATIC_KNOWLEDGE"],
        ),
    )
    assert "REQUEST_GOAL" in prompt
    assert "EXPLAIN_WITH_PLATFORM_RULES" in prompt
    assert "GOAL_PROGRESS" in prompt
    assert "LIVE_OPERATIONAL_EVIDENCE" in prompt
    assert "chain-of-thought" in prompt
    assert "reasoning" not in prompt.lower().split("decision_summary", 1)[0]
