"""A tiny offline Agent implementation for smoke/demo scenarios.

This is the semantic Agent provider itself.  It is intentionally state-aware,
but it does not own runtime evidence, completion, budgets, or tool execution.
"""

from __future__ import annotations

import re

from ..agent.context import AgentContext
from ..agent.decisions import FinalCandidate, ReadToolBatch, SingleToolCall, ToolCall
from ..agent.evidence import ToolObservation
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
        observations = context.semantic_observations.observations
        version = (descriptor.descriptor_version + 1) if descriptor else 1

        if descriptor is None or context.new_turn:
            descriptor = self._understand(prompt, version)
            return self._first_action(prompt, descriptor)

        if self._requirements_satisfied(context):
            return FinalCandidate(
                self._compose(prompt, descriptor, observations),
                referenced_goal_ids=tuple(goal.goal_id for goal in descriptor.goals),
            )

        return self._next_action(prompt, descriptor, observations)

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
            goals.append(InspectQueue("g1", task if task != "task_A" else ""))
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
        self, prompt: str, descriptor: GoalDescriptor, observations: tuple[ToolObservation, ...]
    ):
        for goal in descriptor.goals:
            if isinstance(goal, DiagnoseTask):
                has_detail = any(
                    item.source == "get_task_detail"
                    and item.target == goal.target
                    and item.status == "SUCCESS"
                    for item in observations
                )
                has_diagnosis = any(
                    item.source == "diagnose_task"
                    and item.target == goal.target
                    and item.status == "SUCCESS"
                    for item in observations
                )
                if has_detail and not has_diagnosis:
                    return SingleToolCall(
                        ToolCall("c_diagnosis", "diagnose_task", {"task_name": goal.target})
                    )
                if not has_detail:
                    return SingleToolCall(
                        ToolCall("c_task_retry", "get_task_detail", {"task_name": goal.target})
                    )
            elif isinstance(goal, ExplainKnowledge):
                if not any(
                    item.source == "search_knowledge"
                    and item.target == goal.topic
                    and item.status == "SUCCESS"
                    for item in observations
                ):
                    return SingleToolCall(
                        ToolCall("c_knowledge_retry", "search_knowledge", {"query": goal.topic})
                    )
            elif isinstance(goal, ReadTaskState):
                if not any(
                    item.source == "get_task_detail"
                    and item.target == goal.target
                    and item.status == "SUCCESS"
                    for item in observations
                ):
                    return SingleToolCall(
                        ToolCall("c_task_retry", "get_task_detail", {"task_name": goal.target})
                    )
            elif isinstance(goal, InspectGPU):
                if not any(item.source == "get_gpu_pool" and item.status == "SUCCESS" for item in observations):
                    return SingleToolCall(ToolCall("c_gpu_retry", "get_gpu_pool", {}))
            elif isinstance(goal, InspectQueue):
                if not any(
                    item.source == "get_queue_state"
                    and item.status == "SUCCESS"
                    and (not goal.target or item.target == goal.target)
                    for item in observations
                ):
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
    def _compose(prompt, descriptor, observations):
        parts = []
        for goal in descriptor.goals:
            relevant = [
                item.data
                for item in observations
                if item.status == "SUCCESS"
                and (
                    isinstance(goal, InspectGPU) and item.source == "get_gpu_pool"
                    or isinstance(goal, ExplainKnowledge)
                    and item.source == "search_knowledge"
                    and item.target == goal.topic
                    or isinstance(goal, DiagnoseTask)
                    and item.source in {"get_task_detail", "diagnose_task"}
                    and item.target == goal.target
                    or isinstance(goal, ReadTaskState)
                    and item.source == "get_task_detail"
                    and item.target == goal.target
                    or isinstance(goal, InspectQueue)
                    and item.source == "get_queue_state"
                    and (not goal.target or item.target == goal.target)
                )
            ]
            parts.append(f"{goal.goal_id} {goal.kind}: {relevant}")
        return "\n".join(parts)
