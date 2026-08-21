from __future__ import annotations

import asyncio
from pathlib import Path

from platform_agent.memory import ConversationStore
from platform_agent.models import AgentIntent, AgentPlan, AgentResponse
from platform_agent.policy import AgentPolicyEngine
from platform_agent.workflow import ReadOnlyAgentNodes, SequentialReadOnlyAgent
from platform_planning.service import TaskPlanningService, merge_task_drafts


DEFAULTS = Path(__file__).resolve().parents[1] / "config" / "task_planning_defaults.yaml"
REQUEST = "生成一个 release 任务配置，数据在 /data/test_a，优先级 4，不要执行。"


def test_explicit_literals_recover_missing_model_fields():
    service = TaskPlanningService(defaults_path=DEFAULTS)
    result = service.plan_from_draft(REQUEST, {"pipeline_stages": ["precheck"]})

    assert result.valid is True
    assert result.task_spec is not None
    assert result.task_spec.task_prefix == "release"
    assert result.task_spec.priority == 4
    assert result.task_spec.datasets[0].dataset_path == "/data/test_a"


def test_explicit_priority_overrides_conflicting_model_literal():
    service = TaskPlanningService(defaults_path=DEFAULTS)
    result = service.plan_from_draft(
        REQUEST,
        {
            "task_prefix": "model_prefix",
            "priority": 8,
            "dataset_paths": ["/data/model_path"],
        },
    )

    assert result.valid is True
    assert result.task_spec is not None
    assert result.task_spec.task_prefix == "release"
    assert result.task_spec.priority == 4
    assert result.task_spec.datasets[0].dataset_path == "/data/test_a"


def test_merge_does_not_invent_missing_semantic_fields():
    merged = merge_task_drafts(
        {"explicit_fields": [], "task_type": "release"},
        {"explicit_fields": []},
    )

    assert merged["task_type"] == "release"
    assert "priority" not in merged
    assert "dataset_paths" not in merged


class _PlanningModel:
    requires_tool_descriptions = False

    async def plan(self, user_text, tool_descriptions, history):
        del user_text, tool_descriptions, history
        return AgentPlan(
            intent=AgentIntent.TASK_PLANNING,
            task_draft={"pipeline_stages": ["precheck"]},
        )

    async def synthesize(self, user_text, plan, observations, history, knowledge=None):
        del user_text, observations, history, knowledge
        return AgentResponse(intent=plan.intent, summary="planning")


class _NoCallClient:
    async def describe_tools(self):
        return []

    async def execute(self, calls):
        raise AssertionError(f"task planning must not execute tools: {calls}")


def test_workflow_merges_explicit_fields_before_task_plan_completion(tmp_path: Path):
    nodes = ReadOnlyAgentNodes(
        _PlanningModel(),
        _NoCallClient(),
        AgentPolicyEngine(max_tool_calls=3),
        task_planning_service=TaskPlanningService(defaults_path=DEFAULTS),
    )
    agent = SequentialReadOnlyAgent(nodes, ConversationStore(tmp_path / "sessions"))
    response = asyncio.run(agent.run(REQUEST, "planning-merge"))

    assert response.goal_progress.value == "SATISFIED"
    assert response.goal is not None
    assert response.goal.completion_state.value == "SATISFIED"
    assert response.task_plan["valid"] is True
    assert response.task_plan["task_spec"]["task_prefix"] == "release"
    assert response.task_plan["task_spec"]["priority"] == 4
    assert response.task_plan["task_spec"]["datasets"][0]["dataset_path"] == "/data/test_a"
