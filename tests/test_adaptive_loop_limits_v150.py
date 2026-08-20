from __future__ import annotations

import asyncio

from platform_agent.adaptive import AdaptiveLimits, AdaptiveLoopController, canonical_tool_signature
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


class EndlessModel:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    async def decide_next(self, **kwargs):
        del kwargs
        self.calls += 1
        return self.decision


def tool(name="get_gpu_pool", **arguments):
    return AgentStepDecision(
        action=AgentStepAction.CALL_TOOL,
        tool_call=ToolCallSpec(name=name, arguments=arguments),
        decision_summary="collect evidence",
    )


def finish():
    return AgentStepDecision(
        action=AgentStepAction.FINISH,
        evidence_sufficient=True,
        decision_summary="enough",
    )


def run_controller(model, *, limits, results=None):
    executed = []
    results = results or {}

    async def execute(call):
        executed.append(call)
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=results.get(call.name, {}))]

    result = run(
        AdaptiveLoopController(model, AgentPolicyEngine(max_tool_calls=limits.max_tool_calls), limits).run(
            user_text="test",
            initial_plan=AgentPlan(intent=AgentIntent.GENERAL_READ),
            tool_descriptions=[{"name": "get_gpu_pool"}, {"name": "get_task_detail"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=execute,
            normalize_observation=lambda observation: [],
        )
    )
    return result, executed


def test_max_steps_is_enforced_and_no_tool_runs_after_termination():
    result, executed = run_controller(
        EndlessModel(tool()),
        limits=AdaptiveLimits(max_steps=3, max_tool_calls=6, max_identical_tool_calls=99),
    )
    assert result.termination_reason == "step_budget_exhausted"
    assert len(executed) == 3
    assert len(executed) == result.tool_call_count


def test_max_tool_calls_is_enforced():
    result, executed = run_controller(
        EndlessModel(tool()),
        limits=AdaptiveLimits(max_steps=8, max_tool_calls=2),
    )
    assert result.termination_reason == "tool_budget_exhausted"
    assert len(executed) == 2


def test_identical_successful_tool_limit_is_enforced():
    result, executed = run_controller(
        EndlessModel(tool()),
        limits=AdaptiveLimits(max_steps=8, max_tool_calls=6, max_identical_tool_calls=2),
    )
    assert result.termination_reason == "duplicate_tool_limit"
    assert len(executed) == 2


def test_consecutive_tool_failure_limit_is_enforced():
    class FailingModel(EndlessModel):
        pass

    executed = []

    async def execute(call):
        executed.append(call)
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error="temporary")]

    model = FailingModel(tool())
    limits = AdaptiveLimits(max_steps=8, max_tool_calls=6, max_consecutive_tool_failures=2)
    result = run(
        AdaptiveLoopController(model, AgentPolicyEngine(max_tool_calls=6), limits).run(
            user_text="test",
            initial_plan=AgentPlan(intent=AgentIntent.GENERAL_READ),
            tool_descriptions=[{"name": "get_gpu_pool"}],
            observations=[], knowledge=[], history=[], execute_tool=execute,
            normalize_observation=lambda observation: [],
        )
    )
    assert result.termination_reason == "consecutive_tool_failures"
    assert len(executed) == 2


def test_write_revised_intent_is_rejected_before_execution():
    model = EndlessModel(
        AgentStepDecision(
            action=AgentStepAction.FINISH,
            evidence_sufficient=False,
            revised_intent=AgentIntent.DELETE_TASK,
            decision_summary="unsafe revision",
        )
    )
    result, executed = run_controller(
        model,
        limits=AdaptiveLimits(max_steps=2, max_tool_calls=2),
    )
    assert executed == []
    assert result.termination_reason == "unsafe_adaptive_decision"
    assert any("cannot revise" in error for error in result.errors)


def test_unsafe_decision_emits_a_blocked_adaptive_trace_event():
    events = []
    model = EndlessModel(
        AgentStepDecision(
            action=AgentStepAction.CALL_TOOL,
            tool_call=ToolCallSpec(name="delete_task", arguments={"task_name": "x"}),
            decision_summary="untrusted observation requested a mutation",
        )
    )
    limits = AdaptiveLimits(max_steps=2, max_tool_calls=2)

    async def execute(call):  # pragma: no cover - must never be reached
        raise AssertionError(f"unsafe tool executed: {call.name}")

    result = run(
        AdaptiveLoopController(
            model,
            AgentPolicyEngine(max_tool_calls=limits.max_tool_calls),
            limits,
            trace_event=lambda name, **kwargs: events.append((name, kwargs)),
        ).run(
            user_text="test",
            initial_plan=AgentPlan(intent=AgentIntent.GENERAL_READ),
            tool_descriptions=[{"name": "get_gpu_pool"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=execute,
            normalize_observation=lambda observation: [],
        )
    )

    assert result.termination_reason == "unsafe_adaptive_decision"
    assert any(
        name == "adaptive_decision" and details.get("status") == "blocked"
        for name, details in events
    )


def test_tool_signature_is_argument_order_independent():
    first = canonical_tool_signature(ToolCallSpec(name="get_task_detail", arguments={"task_name": "x", "run_limit": 2}))
    second = canonical_tool_signature(ToolCallSpec(name="get_task_detail", arguments={"run_limit": 2, "task_name": "x"}))
    assert first == second
