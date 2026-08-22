"""A tiny offline Agent implementation for smoke/demo scenarios.

This is the semantic Agent provider itself.  It is intentionally state-aware,
but it does not own runtime evidence, completion, budgets, or tool execution.
"""

from __future__ import annotations

import re

from ..agent.context import AgentContext
from ..agent.decisions import FinalCandidate, ReadToolBatch, SingleToolCall, ToolCall
from ..agent.evidence import EvidenceKind, ToolObservation
from ..agent.goals import (
    DiagnoseTask,
    ExplainKnowledge,
    GoalDescriptor,
    InspectGPU,
    InspectQueue,
    ReadTaskState,
)
from .model import AgentProvider


class DeterministicReadAgent:
    """Offline semantic provider, useful when no real LLM is configured."""

    model_version = "deterministic-read-agent-v2"
    prompt_version = "phase-b-deterministic-v1"

    async def generate(self, context: AgentContext):
        prompt = context.user_input.strip()
        descriptor = context.runtime_structured.goal_descriptor
        version = (descriptor.descriptor_version + 1) if descriptor else 1

        if descriptor is None or context.new_turn:
            descriptor = self._understand(prompt, version)
            return self._first_action(prompt, descriptor)

        if self._requirements_satisfied(context):
            return FinalCandidate(
                self._compose(prompt, descriptor, context),
                referenced_goal_ids=tuple(goal.goal_id for goal in descriptor.goals),
            )

        return self._next_action(prompt, descriptor, context)

    def _understand(self, prompt: str, version: int) -> GoalDescriptor:
        lower = prompt.lower()
        task = self._task_name(prompt) or "task_A"
        goals = []
        if (
            any(token in lower for token in ("gpu", "显卡"))
            and any(token in lower for token in ("queue", "队列"))
            and any(token in lower for token in ("task", "任务", "状态"))
        ):
            goals.extend(
                [ReadTaskState("g1", task), InspectQueue("g2", task), InspectGPU("g3")]
            )
        elif any(token in lower for token in ("顺便", "同时", "also", "and explain")) and (
            "exclusive" in lower or "task_exclusive" in lower or "解释" in lower
        ):
            goals.append(DiagnoseTask("g1", task))
            goals.append(ExplainKnowledge("g2", "task_exclusive"))
        elif any(token in lower for token in ("为什么失败", "why", "diagnos", "失败原因")):
            goals.append(DiagnoseTask("g1", task))
        elif any(token in lower for token in ("gpu", "显卡")):
            goals.append(InspectGPU("g1"))
        elif any(token in lower for token in ("queue", "队列")):
            goals.append(InspectQueue("g1", task if task != "task_A" else None))
        elif any(token in lower for token in ("什么意思", "meaning", "explain", "解释")):
            topic = "task_exclusive" if "exclusive" in lower else prompt
            goals.append(ExplainKnowledge("g1", topic))
        else:
            goals.append(ReadTaskState("g1", task))
        return GoalDescriptor(version, tuple(goals))

    def _first_action(self, prompt: str, descriptor: GoalDescriptor):
        goals = descriptor.goals
        if len(goals) > 1:
            calls = []
            for index, goal in enumerate(goals, start=1):
                single = self._call_for_goal(goal, descriptor)
                calls.append(
                    ToolCall(
                        f"c_initial_{index}",
                        single.call.tool_name,
                        dict(single.call.arguments),
                    )
                )
            return ReadToolBatch(tuple(calls), descriptor)
        goal = goals[0]
        return self._call_for_goal(goal, descriptor)

    def _next_action(
        self, prompt: str, descriptor: GoalDescriptor, context: AgentContext
    ):
        for goal in descriptor.goals:
            if isinstance(goal, DiagnoseTask):
                has_detail = self._has_evidence(context, EvidenceKind.LIVE_TASK, goal.target)
                has_diagnosis = self._has_evidence(context, EvidenceKind.DIAGNOSTIC_CONTEXT, goal.target)
                if has_detail and not has_diagnosis:
                    return SingleToolCall(
                        ToolCall("c_diagnosis", "diagnose_task", {"task_name": goal.target})
                    )
                if not has_detail:
                    return SingleToolCall(
                        ToolCall("c_task_retry", "get_task_detail", {"task_name": goal.target})
                    )
            elif isinstance(goal, ExplainKnowledge):
                if not self._has_evidence(context, EvidenceKind.KNOWLEDGE, goal.topic):
                    return SingleToolCall(
                        ToolCall("c_knowledge_retry", "search_knowledge", {"query": goal.topic})
                    )
            elif isinstance(goal, ReadTaskState):
                if not self._has_evidence(context, EvidenceKind.LIVE_TASK, goal.target):
                    return SingleToolCall(
                        ToolCall("c_task_retry", "get_task_detail", {"task_name": goal.target})
                    )
            elif isinstance(goal, InspectGPU):
                if not self._has_evidence(context, EvidenceKind.GPU_POOL, "platform"):
                    return SingleToolCall(ToolCall("c_gpu_retry", "get_gpu_pool", {}))
            elif isinstance(goal, InspectQueue):
                target = goal.target or "platform"
                if not self._has_evidence(context, EvidenceKind.QUEUE_STATE, target):
                    return SingleToolCall(
                        ToolCall("c_queue_retry", "get_queue_state", {"task_name": goal.target})
                    )
        return FinalCandidate("The currently available read evidence is inconclusive.")

    @staticmethod
    def _call_for_goal(goal, descriptor):
        if isinstance(goal, DiagnoseTask):
            return SingleToolCall(
                ToolCall("c_task", "get_task_detail", {"task_name": goal.target}), descriptor
            )
        if isinstance(goal, ReadTaskState):
            return SingleToolCall(
                ToolCall("c_task", "get_task_detail", {"task_name": goal.target}), descriptor
            )
        if isinstance(goal, InspectGPU):
            return SingleToolCall(ToolCall("c_gpu", "get_gpu_pool", {}), descriptor)
        if isinstance(goal, InspectQueue):
            return SingleToolCall(
                ToolCall("c_queue", "get_queue_state", {"task_name": goal.target}), descriptor
            )
        if isinstance(goal, ExplainKnowledge):
            return SingleToolCall(
                ToolCall("c_knowledge", "search_knowledge", {"query": goal.topic}), descriptor
            )
        raise TypeError(type(goal).__name__)

    @staticmethod
    def _requirements_satisfied(context: AgentContext) -> bool:
        outcomes = context.runtime_structured.goal_outcomes
        return bool(outcomes) and all(item.status.value == "SATISFIED" for item in outcomes)

    @staticmethod
    def _task_name(prompt: str) -> str | None:
        match = re.search(r"\b(task[_-][A-Za-z0-9]+|[A-Za-z]+_task)\b", prompt, re.I)
        return match.group(1) if match else None

    @staticmethod
    def _compose(prompt, descriptor, context):
        parts = []
        evidence = context.runtime_structured.evidence.records
        for goal in descriptor.goals:
            relevant = [
                {"kind": item.kind.value, "target": item.target, "source": item.source_tool}
                for item in evidence
                if _goal_matches_evidence(goal, item.kind, item.target)
            ]
            parts.append(f"{goal.goal_id} {goal.kind}: qualified_evidence={relevant}")
        return "\n".join(parts)

    @staticmethod
    def _has_evidence(context: AgentContext, kind: EvidenceKind, target: str) -> bool:
        return any(
            item.kind is kind
            and item.target == target
            and item.freshness.is_current()
            for item in context.runtime_structured.evidence.records
        )


def _goal_matches_evidence(goal, kind: EvidenceKind, target: str) -> bool:
    if isinstance(goal, InspectGPU):
        return kind is EvidenceKind.GPU_POOL and target == "platform"
    if isinstance(goal, ExplainKnowledge):
        return kind is EvidenceKind.KNOWLEDGE and target == goal.topic
    if isinstance(goal, DiagnoseTask):
        return kind in {EvidenceKind.LIVE_TASK, EvidenceKind.DIAGNOSTIC_CONTEXT} and target == goal.target
    if isinstance(goal, ReadTaskState):
        return kind is EvidenceKind.LIVE_TASK and target == goal.target
    if isinstance(goal, InspectQueue):
        return kind is EvidenceKind.QUEUE_STATE and target == (goal.target or "platform")
    return False
