from __future__ import annotations

import asyncio

from platform_agent.adaptive import AdaptiveLimits, AdaptiveLoopController
from platform_agent.evidence import EvidenceTracker, EvidenceType
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


def _call(name: str, **arguments) -> AgentStepDecision:
    return AgentStepDecision(
        action=AgentStepAction.CALL_TOOL,
        tool_call=ToolCallSpec(name=name, arguments=arguments),
        decision_summary=f"collect {name}",
    )


def _finish() -> AgentStepDecision:
    return AgentStepDecision(
        action=AgentStepAction.FINISH,
        evidence_sufficient=True,
        decision_summary="Evidence is sufficient.",
    )


class CapturingModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls: list[dict] = []

    async def decide_next(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0)


def test_evidence_tracker_maps_successful_tools_without_storing_results():
    tracker = EvidenceTracker()
    tracker.record_tool_observation(
        ToolObservation(
            tool_name="get_gpu_pool",
            arguments={},
            ok=True,
            data={"secret_runtime_payload": "must not be copied"},
        )
    )
    tracker.record_tool_observation(
        ToolObservation(
            tool_name="diagnose_task",
            arguments={"task_name": "release_demo"},
            ok=False,
            error="temporary failure",
        )
    )

    assert tracker.get_collected_types() == [EvidenceType.LIVE_GPU]
    assert tracker.has(EvidenceType.LIVE_GPU)
    assert not tracker.has(EvidenceType.LIVE_TASK)
    assert "secret_runtime_payload" not in str(tracker.summary())
    assert "temporary failure" not in str(tracker.summary())


def test_adaptive_evidence_coverage_reaches_the_next_decision():
    model = CapturingModel([_call("get_gpu_pool"), _call("search_knowledge", query="exclusive rule"), _finish()])
    executed: list[ToolCallSpec] = []

    async def execute(call):
        executed.append(call)
        data = {"results": [{"source_path": "knowledge/gpu.md", "content": "exclusive"}]} if call.name == "search_knowledge" else {"devices": []}
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=data)]

    result = run(
        AdaptiveLoopController(model, AgentPolicyEngine(max_tool_calls=4), AdaptiveLimits(max_steps=4, max_tool_calls=4)).run(
            user_text="检查当前 GPU 并解释独占规则",
            initial_plan=AgentPlan(intent=AgentIntent.GPU_DIAGNOSIS),
            tool_descriptions=[{"name": "get_gpu_pool"}, {"name": "search_knowledge"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=execute,
            normalize_observation=lambda observation: [],
        )
    )

    assert [item.name for item in executed] == ["get_gpu_pool", "search_knowledge"]
    assert model.calls[0]["evidence_records"] == []
    assert [item["type"] for item in model.calls[1]["evidence_records"]] == ["LIVE_GPU"]
    assert [item["type"] for item in model.calls[2]["evidence_records"]] == ["LIVE_GPU", "STATIC_KNOWLEDGE"]
    assert result.evidence_records[-1].type == EvidenceType.STATIC_KNOWLEDGE
    assert result.steps[0]["evidence_after"] == ["LIVE_GPU"]
    assert result.steps[1]["evidence_after"] == ["LIVE_GPU", "STATIC_KNOWLEDGE"]


def test_semantic_search_repetition_is_advisory_and_traced():
    model = CapturingModel(
        [
            _call("search_knowledge", query="soft preemption stage boundary"),
            _call("search_knowledge", query="soft preemption stage boundary rule"),
            _call("search_knowledge", query="soft preemption stage boundary mechanism"),
            _finish(),
        ]
    )
    events: list[tuple[str, dict]] = []

    async def execute(call):
        return [ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data={"results": []})]

    result = run(
        AdaptiveLoopController(
            model,
            AgentPolicyEngine(max_tool_calls=5),
            AdaptiveLimits(max_steps=5, max_tool_calls=5),
            trace_event=lambda name, **kwargs: events.append((name, kwargs)),
        ).run(
            user_text="解释软抢占",
            initial_plan=AgentPlan(intent=AgentIntent.PLATFORM_KNOWLEDGE),
            tool_descriptions=[{"name": "search_knowledge"}],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=execute,
            normalize_observation=lambda observation: [],
        )
    )

    assert result.termination_reason == "agent_finished"
    assert result.repetition_warnings
    assert any(name == "adaptive_repetition_warning" for name, _ in events)
    assert len(result.steps) == 4


def test_adaptive_prompt_exposes_coverage_but_not_raw_result_or_cot():
    prompt = build_adaptive_evidence_prompt(
        user_text="当前 GPU 并解释规则",
        initial_plan=AgentPlan(intent=AgentIntent.GPU_DIAGNOSIS),
        current_intent=AgentIntent.GPU_DIAGNOSIS,
        tool_descriptions=[],
        observations=[],
        knowledge=[],
        history=[],
        step_index=1,
        remaining_tool_calls=3,
        evidence_records=[
            {
                "type": "LIVE_GPU",
                "source_tool": "get_gpu_pool",
                "timestamp": 1.0,
                "summary": "GPU observation",
                "raw_result": "must not be included",
            }
        ],
        adaptive_steps=[
            {
                "step": 0,
                "action": "CALL_TOOL",
                "tool": "get_gpu_pool",
                "evidence_after": ["LIVE_GPU"],
                "decision_summary": "live evidence is available",
                "thought": "must not be forwarded",
            }
        ],
    )

    assert "CURRENT_EVIDENCE_COVERAGE" in prompt
    assert "LIVE_GPU" in prompt
    assert "must not be included" not in prompt
    assert "must not be forwarded" not in prompt
    assert "chain-of-thought" in prompt


def test_legacy_explicit_decide_next_signature_remains_compatible():
    class LegacyModel:
        def __init__(self):
            self.called = False

        async def decide_next(
            self,
            user_text,
            initial_plan,
            tool_descriptions,
            observations,
            knowledge,
            history,
            step_index,
            remaining_tool_calls,
            current_intent=None,
            adaptive_steps=None,
        ):
            del user_text, initial_plan, tool_descriptions, observations, knowledge
            del history, step_index, remaining_tool_calls, current_intent, adaptive_steps
            self.called = True
            return _finish()

    model = LegacyModel()
    result = run(
        AdaptiveLoopController(model, AgentPolicyEngine(max_tool_calls=1), AdaptiveLimits(max_steps=1, max_tool_calls=1)).run(
            user_text="谢谢",
            initial_plan=AgentPlan(intent=AgentIntent.GENERAL_READ),
            tool_descriptions=[],
            observations=[],
            knowledge=[],
            history=[],
            execute_tool=lambda call: None,
            normalize_observation=lambda observation: [],
        )
    )

    assert model.called
    assert result.termination_reason == "agent_finished"
