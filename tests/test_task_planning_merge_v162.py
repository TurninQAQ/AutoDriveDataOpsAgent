from __future__ import annotations

from pathlib import Path

from platform_planning.heuristic import HeuristicTaskDraftParser
from platform_planning.service import TaskPlanningService, merge_task_drafts


DEFAULTS = Path(__file__).resolve().parents[1] / "config" / "task_planning_defaults.yaml"


def test_derived_task_prefix_does_not_override_model_semantic_prefix():
    deterministic = HeuristicTaskDraftParser().parse("生成一个 release 任务，数据在 /data/test_a")
    assert deterministic["task_prefix"] == "release"
    assert "task_prefix" not in deterministic["explicit_fields"]
    merged = merge_task_drafts(deterministic, {"task_prefix": "nightly_prod"})
    assert merged["task_prefix"] == "nightly_prod"


def test_derived_task_prefix_fills_missing_model_prefix():
    service = TaskPlanningService(defaults_path=DEFAULTS)
    result = service.plan_from_draft("生成一个 release 任务，数据在 /data/test_a", {})
    assert result.valid
    assert result.task_spec.task_prefix == "release"
    assert "task_prefix_from_task_type" in result.defaults_used


def test_explicit_task_prefix_overrides_model_semantic_prefix():
    service = TaskPlanningService(defaults_path=DEFAULTS)
    result = service.plan_from_draft(
        "生成一个 release 任务，任务名 release_hotfix，数据在 /data/test_a",
        {"task_prefix": "wrong_name"},
    )
    assert result.valid
    assert result.task_spec.task_prefix == "release_hotfix"


def test_dataset_output_path_is_not_treated_as_input_dataset():
    parsed = HeuristicTaskDraftParser().parse(
        "创建 release 任务，数据在 /data/test_a，输出 YAML 到 /tmp/out.yaml。"
    )
    assert parsed["dataset_paths"] == ["/data/test_a"]


def test_explicit_priority_overrides_model_and_derived_values():
    merged = merge_task_drafts(
        {"priority": 4, "explicit_fields": ["priority"]},
        {"priority": 8},
    )
    assert merged["priority"] == 4
