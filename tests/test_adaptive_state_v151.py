from __future__ import annotations

import asyncio

from platform_agent.adaptive import AdaptiveLimits, AdaptiveLoopController
from platform_agent.model import build_adaptive_evidence_prompt
from platform_agent.models import (
    AgentIntent,
    AgentPlan,
    AgentStepAction,
    AgentStepDecision,
    ToolCallSpec,
    ToolObservation,
)
from platform_agent.policy import AgentPolicyEngine


def run(coro):
    return asyncio.run(coro)


class CapturingAdaptiveModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls: list[dict] = []

    async def decide_next(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0)


def test_current_intent_and_previous_steps_feed_the_next_decision():
    model = CapturingAdaptiveModel(
        [
            AgentStepDecision(
                action=AgentStepAction.CALL_TOOL,
                tool_call=ToolCallSpec(name="search_knowledge", arguments={"query": "soft preemption"}),
                revised_intent=AgentIntent.TASK_DIAGNOSIS,
                decision_summary="Static rule evidence is available; task evidence remains relevant.",
            ),
            AgentStepDecision(
                action=AgentStepAction.FINISH,
                evidence_sufficient=True,
                decision_summary="Evidence is sufficient.",
            ),
        ]
    )

    async def execute(call):
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={"rule": "boundary"})]

    result = run(
        AdaptiveLoopController(
            model,
            AgentPolicyEngine(max_tool_calls=3),
            AdaptiveLimits(max_steps=3, max_tool_calls=3),
        ).run(
            user_text="release_demo 为什么 draining？结合软抢占解释",
            initial_plan=AgentPlan(intent=AgentIntent.PLATFORM_KNOWLEDGE),
            tool_descriptions=[{"name": "search_knowledge"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=execute,
            normalize_observation=lambda observation: [],
        )
    )

    assert result.current_intent == AgentIntent.TASK_DIAGNOSIS
    assert model.calls[0]["current_intent"] == AgentIntent.PLATFORM_KNOWLEDGE
    assert model.calls[1]["current_intent"] == AgentIntent.TASK_DIAGNOSIS
    assert len(model.calls[1]["adaptive_steps"]) == 1
    assert model.calls[1]["adaptive_steps"][0]["revised_intent"] == AgentIntent.TASK_DIAGNOSIS.value


def test_adaptive_prompt_exposes_state_without_hidden_reasoning_fields():
    prompt = build_adaptive_evidence_prompt(
        user_text="release_demo 当前状态",
        initial_plan=AgentPlan(intent=AgentIntent.PLATFORM_KNOWLEDGE),
        current_intent=AgentIntent.TASK_DIAGNOSIS,
        tool_descriptions=[],
        observations=[],
        knowledge=[],
        history=[],
        step_index=1,
        remaining_tool_calls=4,
        adaptive_steps=[
            {
                "step": 0,
                "action": "CALL_TOOL",
                "tool": "search_knowledge",
                "arguments": {"query": "soft preemption"},
                "revised_intent": "task_diagnosis",
                "evidence_sufficient": False,
                "decision_summary": "Task evidence remains missing.",
                "thought": "must not be forwarded",
            }
        ],
    )

    assert "INITIAL_INTENT:\nplatform_knowledge" in prompt
    assert "CURRENT_INTENT:\ntask_diagnosis" in prompt
    assert "PREVIOUS_ADAPTIVE_DECISIONS" in prompt
    assert "Task evidence remains missing." in prompt
    assert "must not be forwarded" not in prompt
    assert "chain-of-thought" in prompt
