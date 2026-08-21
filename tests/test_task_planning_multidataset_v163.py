from __future__ import annotations

from pathlib import Path

from platform_planning.heuristic import HeuristicTaskDraftParser
from platform_planning.service import TaskPlanningService


DEFAULTS = Path(__file__).resolve().parents[1] / "config" / "task_planning_defaults.yaml"


def test_chinese_multi_dataset_group_is_complete():
    parsed = HeuristicTaskDraftParser().parse("创建一个 release 任务，数据在 /data/a 和 /data/b，只运行 precheck")
    assert parsed["dataset_paths"] == ["/data/a", "/data/b"]


def test_dataset_comma_group_is_complete():
    parsed = HeuristicTaskDraftParser().parse("创建 release 任务，dataset /data/a, /data/b")
    assert parsed["dataset_paths"] == ["/data/a", "/data/b"]


def test_multi_dataset_group_excludes_output_yaml_path():
    parsed = HeuristicTaskDraftParser().parse(
        "创建 release 任务，数据在 /data/a、/data/b，输出 YAML 到 /tmp/out.yaml"
    )
    assert parsed["dataset_paths"] == ["/data/a", "/data/b"]


def test_complete_explicit_group_overrides_model_collection_without_loss():
    service = TaskPlanningService(defaults_path=DEFAULTS)
    result = service.plan_from_draft(
        "创建一个 release 任务，数据在 /data/a 和 /data/b，只运行 precheck",
        {"dataset_paths": ["/data/a", "/data/b"]},
    )
    assert result.valid is True
    assert result.task_spec is not None
    assert len(result.task_spec.datasets) == 2
    assert [item.dataset_path for item in result.task_spec.datasets] == ["/data/a", "/data/b"]


def test_single_dataset_and_model_prefix_precedence_remain_unchanged():
    service = TaskPlanningService(defaults_path=DEFAULTS)
    result = service.plan_from_draft(
        "创建一个 release 任务，数据在 /data/test_a",
        {"task_prefix": "nightly_prod", "dataset_paths": ["/data/test_a"]},
    )
    assert result.valid is True
    assert result.task_spec is not None
    assert result.task_spec.task_prefix == "nightly_prod"
    assert [item.dataset_path for item in result.task_spec.datasets] == ["/data/test_a"]
