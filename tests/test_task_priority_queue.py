#!/usr/bin/env python3
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def write_task_config(root, task_name, content):
    task_dir = root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "datasets_config.yaml").write_text(content, encoding="utf-8")


def task_config_yaml(task_type=None, priority=None, dataset_name="clip_1"):
    lines = []
    if task_type:
        lines.append(f"task_type: {task_type}")
    if priority is not None:
        lines.append(f"priority: {priority}")
    lines.extend(
        [
            "pipeline_stages:",
            "  - parser",
            "gpu_stages: ''",
            "datasets:",
            f"  - dataset_name: {dataset_name}",
            "    dataset_path: /tmp",
            "    image_parser: parser:test",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        queue_dir = tmp_path / "queue"
        lock_dir = tmp_path / "locks"
        config_root = tmp_path / "tasks"
        task_types_config = tmp_path / "task_types.yaml"

        task_types_config.write_text(
            "\n".join(
                [
                    "default_priority: 100",
                    "task_types:",
                    "  release:",
                    "    priority: 10",
                    "  test:",
                    "    priority: 50",
                    "  debug:",
                    "    priority: 80",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        os.environ["AIRFLOW_TASK_QUEUE_DIR"] = str(queue_dir)
        os.environ["AIRFLOW_TASK_LOCK_DIR"] = str(lock_dir)
        os.environ["AIRFLOW_TASK_CONFIG_ROOT"] = str(config_root)
        os.environ["AIRFLOW_TASK_TYPES_CONFIG"] = str(task_types_config)
        os.environ["AIRFLOW_BIN"] = "/bin/true"

        tm = importlib.import_module("scripts.task_manager")

        write_task_config(config_root, "active_task", task_config_yaml(task_type="debug"))
        write_task_config(config_root, "low_task", task_config_yaml(dataset_name="clip_2"))
        write_task_config(config_root, "high_task", task_config_yaml(task_type="release"))

        assert tm.queue_action_name(tm.register_task_queue(
            "active_task",
            "dag_active",
            1,
            True,
            task_config={"task_type": "debug"},
            task_config_root=config_root,
        )) == "start"
        assert tm.queue_action_name(tm.register_task_queue(
            "low_task",
            "dag_low",
            1,
            True,
            task_config={},
            task_config_root=config_root,
        )) == "queued"
        lock_dir.mkdir(parents=True, exist_ok=True)
        (lock_dir / "active_task.lock").write_text(
            json.dumps(
                {
                    "task_name": "active_task",
                    "dag_id": "dag_active",
                    "active_runs": [
                        {"run_id": "run_active_1", "pid": os.getpid()},
                        {"run_id": "run_active_2", "pid": os.getpid()},
                    ],
                }
            ),
            encoding="utf-8",
        )
        preempt_event = tm.register_task_queue(
            "high_task",
            "dag_high",
            1,
            True,
            task_config={"task_type": "release"},
            task_config_root=config_root,
        )
        assert tm.queue_action_name(preempt_event) == "preempt_requested"
        assert preempt_event["preempted_task_name"] == "active_task"
        assert preempt_event["queued_task_name"] == "high_task"

        state = json.loads((queue_dir / "queue.lock").read_text(encoding="utf-8"))
        assert state["active"]["task_name"] == "active_task"
        assert state["active"]["status"] == "draining"
        assert state["active"]["preempt_requested"] is True
        assert state["active"]["preempt_requested_by"] == "high_task"
        assert state["active"]["drain_target_run_ids"] == ["run_active_1", "run_active_2"]
        assert [entry["task_name"] for entry in state["queue"]] == ["high_task", "low_task"]
        assert [entry["priority"] for entry in state["queue"]] == [10, 100]

        calls = []
        original_pause_dag_cli = tm.pause_dag_cli
        original_unpause_dag = tm.unpause_dag
        original_trigger_preempted_recovery_runs = tm.trigger_preempted_recovery_runs
        try:
            tm.pause_dag_cli = lambda dag_id: calls.append(("pause", dag_id))
            tm.unpause_dag = lambda dag_id: calls.append(("unpause", dag_id))

            def fake_trigger_recovery(dag_id, pending_run_confs, preempted_by=""):
                calls.append(
                    (
                        "recover",
                        dag_id,
                        [conf["dataset_name"] for conf in pending_run_confs],
                        preempted_by,
                    )
                )
                return len(pending_run_confs)

            tm.trigger_preempted_recovery_runs = fake_trigger_recovery
            tm.apply_queue_event_runtime_effects(preempt_event)
        finally:
            tm.pause_dag_cli = original_pause_dag_cli
            tm.unpause_dag = original_unpause_dag
            tm.trigger_preempted_recovery_runs = original_trigger_preempted_recovery_runs

        assert calls == []

        original_list_dag_runs_db = tm.list_dag_runs_db
        try:
            tm.list_dag_runs_db = lambda dag_id: [
                {"state": "running", "conf": {"dataset_name": "clip_old_active"}},
                {
                    "state": "queued",
                    "conf": {
                        "dataset_name": "clip_existing_recovery",
                        tm.PLATFORM_RECOVERY_CONF_KEY: tm.PLATFORM_RECOVERY_REASON_PREEMPTED,
                    },
                },
            ]
            recovery_plan = tm.preempted_recovery_trigger_plan(
                "dag_active",
                [
                    {"dataset_name": "clip_old_active"},
                    {"dataset_name": "clip_existing_recovery"},
                ],
            )
        finally:
            tm.list_dag_runs_db = original_list_dag_runs_db

        assert [conf["dataset_name"] for conf in recovery_plan["trigger_confs"]] == [
            "clip_old_active"
        ]
        assert recovery_plan["existing_recovery_datasets"] == ["clip_existing_recovery"]

        recovery_conf = {
            "task_name": "active_task",
            "dataset_name": "clip_1",
            "dataset_path": "/tmp",
            "pipeline_stages": [["parser"], ["segment"]],
            tm.PLATFORM_RECOVERY_CONF_KEY: tm.PLATFORM_RECOVERY_REASON_PREEMPTED,
            tm.PLATFORM_RESUME_FROM_STAGE_KEY: "segment",
            tm.PLATFORM_ORIGINAL_RUN_ID_KEY: "run_original",
        }
        queue_state = {
            "active": {
                "task_name": "high_task",
                "dag_id": "dag_high",
                "status": "active",
                "priority": 10,
                "total_runs": 1,
                "remaining_runs": 1,
            },
            "queue": [
                {
                    "task_name": "active_task",
                    "dag_id": "dag_active",
                    "status": "queued",
                    "preempted": True,
                    "priority": 80,
                    "total_runs": 1,
                    "remaining_runs": 1,
                    "pending_run_confs": [recovery_conf],
                }
            ],
            "version": 2,
        }
        (queue_dir / "queue.lock").write_text(json.dumps(queue_state), encoding="utf-8")

        original_list_dag_runs_if_present = tm.list_dag_runs_if_present
        try:
            tm.list_dag_runs_if_present = lambda api_base, token, dag_id: [
                {
                    "state": "success",
                    "conf": {"dataset_name": "clip_1"},
                }
            ]
            removed, next_dag_id, pending = tm.remove_task_from_queue(
                "high_task",
                True,
                advance_next=True,
                task_config_root=config_root,
                api_base="http://airflow.local",
                token="token",
            )
        finally:
            tm.list_dag_runs_if_present = original_list_dag_runs_if_present

        assert removed == 1
        assert next_dag_id == "dag_active"
        assert pending == [recovery_conf]
        state = json.loads((queue_dir / "queue.lock").read_text(encoding="utf-8"))
        assert state["active"]["task_name"] == "active_task"
        assert state["active"]["remaining_runs"] == 1
        assert state["queue"] == []

        queue_state["active"]["task_name"] = "high_task"
        queue_state["active"]["dag_id"] = "dag_high"
        queue_state["queue"][0]["pending_run_confs"] = [recovery_conf]
        (queue_dir / "queue.lock").write_text(json.dumps(queue_state), encoding="utf-8")
        try:
            tm.list_dag_runs_if_present = lambda api_base, token, dag_id: [
                {
                    "state": "queued",
                    "conf": recovery_conf,
                }
            ]
            removed, next_dag_id, pending = tm.remove_task_from_queue(
                "high_task",
                True,
                advance_next=True,
                task_config_root=config_root,
                api_base="http://airflow.local",
                token="token",
            )
        finally:
            tm.list_dag_runs_if_present = original_list_dag_runs_if_present

        assert removed == 1
        assert next_dag_id == "dag_active"
        assert pending == []
        state = json.loads((queue_dir / "queue.lock").read_text(encoding="utf-8"))
        assert state["active"]["task_name"] == "active_task"
        assert state["active"]["pending_run_confs"] == []
        assert state["active"]["remaining_runs"] == 1

        (queue_dir / "queue.lock").write_text(
            json.dumps(
                {
                    "active": {
                        "task_name": "active_task",
                        "dag_id": "dag_active",
                        "status": "draining",
                        "preempt_requested": True,
                        "preempt_requested_by": "high_task",
                        "priority": 80,
                        "total_runs": 1,
                        "remaining_runs": 1,
                        "completed_run_ids": [],
                    },
                    "queue": [
                        {
                            "task_name": "high_task",
                            "dag_id": "dag_high",
                            "status": "queued",
                            "priority": 10,
                            "total_runs": 1,
                            "remaining_runs": 1,
                        },
                        {
                            "task_name": "low_task",
                            "dag_id": "dag_low",
                            "status": "queued",
                            "priority": 100,
                            "total_runs": 1,
                            "remaining_runs": 1,
                        },
                    ],
                    "version": 2,
                }
            ),
            encoding="utf-8",
        )

        original_apply_queue_event_runtime_effects = tm.apply_queue_event_runtime_effects
        try:
            tm.apply_queue_event_runtime_effects = lambda event: None
            tm.set_task_priority(
                SimpleNamespace(
                    task_name="low_task",
                    priority="1",
                    task_config_root=config_root,
                )
            )
        finally:
            tm.apply_queue_event_runtime_effects = original_apply_queue_event_runtime_effects

        state = json.loads((queue_dir / "queue.lock").read_text(encoding="utf-8"))
        assert state["active"]["task_name"] == "active_task"
        assert state["active"]["status"] == "draining"
        assert state["active"]["preempt_requested_by"] == "low_task"
        assert [entry["task_name"] for entry in state["queue"]] == ["low_task", "high_task"]
        assert [entry["priority"] for entry in state["queue"]] == [1, 10]
        config_text = (config_root / "low_task" / "datasets_config.yaml").read_text(
            encoding="utf-8"
        )
        assert "priority: 1" in config_text


if __name__ == "__main__":
    main()
