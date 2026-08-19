#!/usr/bin/env python3
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def task_config_yaml():
    return "\n".join(
        [
            "task_type: test",
            "pipeline_stages:",
            "  - precheck",
            "gpu_stages: ''",
            "datasets:",
            "  - dataset_name: clip_1",
            "    dataset_path: /tmp",
            "",
        ]
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        queue_dir = tmp_path / "queue"
        schedule_file = queue_dir / "scheduled_submits.lock"
        config_root = tmp_path / "tasks"
        dags_dir = tmp_path / "dags"
        task_types_config = tmp_path / "task_types.yaml"
        task_yaml = tmp_path / "task.yaml"

        task_types_config.write_text(
            "\n".join(
                [
                    "default_priority: 100",
                    "task_types:",
                    "  release:",
                    "    priority: 10",
                    "  test:",
                    "    priority: 50",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        task_yaml.write_text(task_config_yaml(), encoding="utf-8")

        os.environ["AIRFLOW_TASK_QUEUE_DIR"] = str(queue_dir)
        os.environ["AIRFLOW_TASK_SCHEDULE_FILE"] = str(schedule_file)
        os.environ["AIRFLOW_TASK_CONFIG_ROOT"] = str(config_root)
        os.environ["AIRFLOW_TASK_TYPES_CONFIG"] = str(task_types_config)
        os.environ["AIRFLOW_BIN"] = "/bin/true"

        tm = importlib.import_module("scripts.task_manager")

        parser_args = tm.build_parser().parse_args(["submit", "scheduler", "--scheduler-once"])
        assert tm.is_submit_scheduler_mode(parser_args)

        submit_args = SimpleNamespace(
            legacy_task_prefix=None,
            legacy_yaml_path=None,
            task_prefix="schedtask",
            yaml_path=str(task_yaml),
            dags_dir=str(dags_dir),
            task_config_root=str(config_root),
            parse_timeout_sec=1,
            no_trigger=True,
            schedule="2000-01-01 00:00",
            scheduler_once=False,
            scheduler_interval_sec=1,
        )

        scheduled = tm.submit(submit_args)
        assert scheduled["schedule_id"].startswith("sched_")
        assert schedule_file.is_file()
        assert not (dags_dir / "generated").exists()
        assert not config_root.exists()

        state = json.loads(schedule_file.read_text(encoding="utf-8"))
        assert len(state["items"]) == 1
        item = state["items"][0]
        assert item["status"] == "scheduled"
        assert item["task_prefix"] == "schedtask"
        assert item["priority"] == 50

        processed = tm.run_task_submit_scheduler_once(now_ts=time.time())
        assert processed == 1

        state = json.loads(schedule_file.read_text(encoding="utf-8"))
        item = state["items"][0]
        assert item["status"] == "submitted"
        assert item["result_task_name"].startswith("schedtask_")
        assert item["result_dag_id"].startswith("batch_pipeline_universal_schedtask_")

        dag_files = list((dags_dir / "generated").glob("batch_pipeline_universal_schedtask_*.py"))
        assert len(dag_files) == 1
        assert (config_root / item["result_task_name"] / "datasets_config.yaml").is_file()

        future_args = SimpleNamespace(
            legacy_task_prefix=None,
            legacy_yaml_path=None,
            task_prefix="futuretask",
            yaml_path=str(task_yaml),
            dags_dir=str(dags_dir),
            task_config_root=str(config_root),
            parse_timeout_sec=1,
            no_trigger=True,
            schedule="2099-01-01 00:00",
            scheduler_once=False,
            scheduler_interval_sec=1,
        )

        future = tm.submit(future_args)
        future_schedule_id = future["schedule_id"]

        pending = tm.list_scheduled_submits(
            SimpleNamespace(show_all=False, status=[], json=False)
        )
        assert [entry["schedule_id"] for entry in pending["items"]] == [future_schedule_id]

        dry_run_remove = tm.remove_scheduled_submit(
            SimpleNamespace(schedule_id=future_schedule_id, yes=False)
        )
        assert dry_run_remove["removed"] is False

        state = json.loads(schedule_file.read_text(encoding="utf-8"))
        future_item = [
            entry for entry in state["items"] if entry["schedule_id"] == future_schedule_id
        ][0]
        assert future_item["status"] == "scheduled"

        removed = tm.remove_scheduled_submit(
            SimpleNamespace(schedule_id=future_schedule_id, yes=True)
        )
        assert removed["removed"] is True

        pending = tm.list_scheduled_submits(
            SimpleNamespace(show_all=False, status=[], json=False)
        )
        assert pending["items"] == []

        all_items = tm.list_scheduled_submits(
            SimpleNamespace(show_all=True, status=[], json=False)
        )
        statuses = {entry["schedule_id"]: entry["status"] for entry in all_items["items"]}
        assert statuses[future_schedule_id] == "removed"
        assert statuses[scheduled["schedule_id"]] == "submitted"


if __name__ == "__main__":
    main()
