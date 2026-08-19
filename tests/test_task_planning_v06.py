from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import AgentIntent
from platform_agent.policy import ReadOnlyPolicy
from platform_agent.workflow import ReadOnlyAgentNodes, SequentialReadOnlyAgent
from platform_core.config import normalize_task_priority_config, validate_config
from platform_core.errors import TaskConfigError
from platform_planning.service import TaskPlanningService


class NoCallToolClient:
    def __init__(self):
        self.describe_count = 0
        self.calls = []

    async def describe_tools(self):
        self.describe_count += 1
        return []

    async def execute(self, calls):
        self.calls.extend(calls)
        raise AssertionError("Task planning must not call MCP tools in V0.6")


def run(coro):
    return asyncio.run(coro)


def service() -> TaskPlanningService:
    return TaskPlanningService()


def test_full_release_plan_uses_platform_defaults_and_validates():
    result = service().plan(
        "创建一个release任务，把 /data/record_001 做完整流程，最多同时4个clip，"
        "segment和od独占GPU，occ共享GPU"
    )
    assert result.valid is True
    assert result.resolved_priority == 10
    assert result.priority_source == "task_type"
    assert result.task_spec.task_prefix == "release"
    assert result.task_spec.max_active_runs == 4
    assert result.task_spec.pipeline_stages == ["precheck", "parser", "segment", "map", ["od", "occ"], "coloration"]
    assert result.task_spec.gpu_stages == "segment,od,occ"
    assert result.task_spec.exclusive_gpu_stages == "segment,od"
    assert result.task_spec.datasets[0].dataset_name == "record_001"
    assert result.task_spec.datasets[0].images["segment"].startswith("172.16.201.100:5000/sam31:")
    assert "xxx" not in result.yaml_text
    validate_config(result.config, scripts_dir=Path(__file__).resolve().parents[1] / "scripts")


def test_only_precheck_generates_no_gpu_requirement():
    result = service().plan("创建一个test任务，只运行 precheck，数据 /tmp/record_a")
    assert result.valid is True
    assert result.resolved_priority == 50
    assert result.task_spec.pipeline_stages == ["precheck"]
    assert result.task_spec.gpu_ids == ""
    assert result.task_spec.gpu_stages == ""
    assert result.task_spec.gpu_stage_memory_mb == {}
    assert result.config["datasets"][0].get("image_precheck") is None


def test_explicit_priority_memory_and_single_stage_override_defaults():
    result = service().plan(
        "生成任务配置，任务名 debug_case，只运行 od，dataset /data/clip_001，"
        "优先级 30，od 24GB"
    )
    assert result.valid is True
    assert result.resolved_priority == 30
    assert result.priority_source == "explicit"
    assert result.task_spec.task_prefix == "debug_case"
    assert result.task_spec.pipeline_stages == ["od"]
    assert result.task_spec.gpu_stage_memory_mb == {"od": 24576}
    assert set(result.config["datasets"][0]) >= {"dataset_name", "dataset_path", "image_od"}
    assert "image_segment" not in result.config["datasets"][0]


def test_llm_style_structured_draft_is_revalidated_by_platform_core():
    draft = {
        "task_prefix": "release_hotfix",
        "task_type": "release",
        "pipeline_stages": ["parser", ["od", "occ"]],
        "max_active_runs": 2,
        "dataset_paths": ["/data/r1"],
        "dataset_names": ["clip_001"],
        "exclusive_gpu_stages": ["od"],
        "shared_gpu_stages": ["occ"],
        "explicit_fields": [
            "task_prefix", "task_type", "pipeline_stages", "max_active_runs",
            "datasets.dataset_path", "datasets.dataset_name", "exclusive_gpu_stages", "shared_gpu_stages",
        ],
    }
    result = service().plan_from_draft("structured draft", draft)
    assert result.valid is True
    assert result.task_spec.exclusive_gpu_stages == "od"
    assert result.task_spec.gpu_stages == "od,occ"
    assert result.config["datasets"][0]["image_parser"]
    assert result.config["datasets"][0]["image_od"]
    assert result.config["datasets"][0]["image_occ"]


def test_structured_llm_draft_can_derive_prefix_from_explicit_task_type():
    result = service().plan_from_draft(
        "创建一个release任务，数据 /data/r1",
        {"task_type": "release", "dataset_paths": ["/data/r1"], "explicit_fields": ["task_type", "datasets.dataset_path"]},
    )
    assert result.valid is True
    assert result.task_spec.task_prefix == "release"
    assert "task_prefix_from_task_type" in result.defaults_used


def test_missing_dataset_path_is_unresolved_and_cannot_be_written(tmp_path: Path):
    svc = service()
    result = svc.plan("创建一个release任务，完整流程")
    assert result.valid is False
    assert "datasets.dataset_path" in result.unresolved_fields
    assert any(item.code == "UNRESOLVED_FIELD" for item in result.issues)
    with pytest.raises(TaskConfigError):
        svc.write_yaml(result, tmp_path / "task.yaml")
    assert not (tmp_path / "task.yaml").exists()


def test_invalid_pipeline_cannot_bypass_existing_platform_validator():
    result = service().plan("生成任务配置，任务名 debug_case，pipeline=unknown_stage，数据 /data/r1")
    assert result.valid is False
    assert any(item.code == "PLATFORM_VALIDATION_FAILED" for item in result.issues)
    assert any("run_unknown_stage.sh" in item.message for item in result.issues)


def test_valid_yaml_write_is_atomic_and_loadable(tmp_path: Path):
    svc = service()
    result = svc.plan("创建一个debug任务，只运行 od，数据 /data/clip_001")
    assert result.valid
    target = svc.write_yaml(result, tmp_path / "planned" / "task.yaml")
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload == result.config
    assert normalize_task_priority_config(payload)["priority"] == 80
    assert list(target.parent.glob(".*.tmp")) == []


def test_task_prefix_uses_same_32_char_submission_limit():
    long_prefix = "a" * 33
    result = service().plan(f"生成任务配置，任务名 {long_prefix}，只运行 precheck，数据 /tmp/r1")
    assert result.valid is False
    assert any(item.code == "INVALID_TASK_PREFIX" for item in result.issues)


def test_policy_allows_local_task_planning_but_blocks_submit():
    policy = ReadOnlyPolicy()
    assert policy.is_write_request("创建一个release任务，生成YAML") is False
    assert policy.is_task_planning_request("创建一个release任务，生成YAML") is True
    assert policy.is_write_request("创建一个release任务并提交") is True
    assert policy.is_write_request("submit the release task") is True


def test_agent_task_planning_returns_structured_plan_without_mcp_calls(tmp_path: Path):
    client = NoCallToolClient()
    nodes = ReadOnlyAgentNodes(
        HeuristicReadOnlyModel(),
        client,
        ReadOnlyPolicy(max_tool_calls=6),
        knowledge_retriever=None,
        task_planning_service=service(),
    )
    agent = SequentialReadOnlyAgent(nodes, ConversationStore(tmp_path / "sessions"))
    response = run(
        agent.run(
            "创建一个release任务，把 /data/record_001 做完整流程，最多同时4个clip，segment和od独占GPU，occ共享GPU",
            "plan-1",
        )
    )
    assert response.intent == AgentIntent.TASK_PLANNING
    assert response.blocked is False
    assert response.task_plan["valid"] is True
    assert response.task_plan["resolved_priority"] == 10
    assert client.calls == []
    # Heuristic model does not need tool descriptions either.
    assert client.describe_count == 0


def test_agent_explicit_submit_is_still_blocked_before_planning_or_tools(tmp_path: Path):
    client = NoCallToolClient()
    nodes = ReadOnlyAgentNodes(
        HeuristicReadOnlyModel(),
        client,
        ReadOnlyPolicy(max_tool_calls=6),
        task_planning_service=service(),
    )
    agent = SequentialReadOnlyAgent(nodes, ConversationStore(tmp_path / "sessions"))
    response = run(agent.run("创建一个release任务并提交，数据 /data/r1", "blocked"))
    assert response.intent == AgentIntent.UNSUPPORTED_WRITE
    assert response.blocked is True
    assert response.task_plan is None
    assert client.calls == []


def test_cli_has_plan_task_and_does_not_expose_submit_action():
    from platform_agent.cli import parser

    args = parser().parse_args(["plan-task", "创建一个test任务，只运行precheck，数据 /tmp/r1"])
    assert args.command == "plan-task"
    assert args.output == ""
    eval_args = parser().parse_args(["plan-task-eval"])
    assert eval_args.command == "plan-task-eval"


def test_repository_task_planning_eval_is_full_accuracy():
    from platform_planning.evaluation import evaluate_task_planning

    root = Path(__file__).resolve().parents[1]
    metrics = evaluate_task_planning(service(), root / "eval" / "task_planning_cases.json")
    assert metrics["case_count"] == 8
    assert metrics["case_accuracy"] == 1.0


def test_deploy_and_platform_env_include_planning_package():
    root = Path(__file__).resolve().parents[1]
    deploy = (root / "scripts" / "deploy_ci_cloud.sh").read_text(encoding="utf-8")
    platform = (root / "platform").read_text(encoding="utf-8")
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    assert "platform_planning" in deploy
    assert "AIRFLOW_PLATFORM_PLANNING_DIR" in deploy
    assert "PLATFORM_TASK_PLANNING_DEFAULTS" in platform
    assert "task_planning_defaults.yaml" in platform
    assert "PLATFORM_TASK_PLANNING_DEFAULTS" in env_example


def test_defaults_file_contains_real_concrete_images_not_template_placeholders():
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((root / "config" / "task_planning_defaults.yaml").read_text(encoding="utf-8"))
    images = payload["image_defaults"]
    assert set(images) >= {"parser", "segment", "map", "od", "coloration", "occ"}
    assert all("xxx" not in value for value in images.values())
