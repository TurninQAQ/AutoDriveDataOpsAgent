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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        queue_dir = tmp_path / "queue"
        lock_dir = tmp_path / "task_locks"
        gpu_dir = tmp_path / "gpu_locks"
        schedule_file = queue_dir / "scheduled_submits.lock"
        dags_dir = tmp_path / "dags"
        generated_dir = dags_dir / "generated"
        config_root = tmp_path / "tasks"

        os.environ["AIRFLOW_TASK_QUEUE_DIR"] = str(queue_dir)
        os.environ["AIRFLOW_TASK_SCHEDULE_FILE"] = str(schedule_file)
        os.environ["AIRFLOW_TASK_LOCK_DIR"] = str(lock_dir)
        os.environ["AIRFLOW_GPU_LOCK_DIR"] = str(gpu_dir)
        os.environ["AIRFLOW_TASK_CONFIG_ROOT"] = str(config_root)

        tm = importlib.import_module("scripts.task_manager")

        write_json(
            queue_dir / "queue.lock",
            {
                "active": {
                    "task_name": "active_task",
                    "dag_id": "batch_pipeline_universal_active_task",
                },
                "queue": [
                    {
                        "task_name": "queued_task",
                        "dag_id": "batch_pipeline_universal_queued_task",
                    }
                ],
            },
        )
        write_json(
            schedule_file,
            {
                "items": [
                    {"schedule_id": "sched_1", "status": "scheduled"},
                    {"schedule_id": "sched_2", "status": "running"},
                    {"schedule_id": "sched_3", "status": "submitted"},
                ]
            },
        )
        write_json(lock_dir / "active_task.lock", {"task_name": "active_task"})
        write_json(
            gpu_dir / "gpu_0.lock",
            {"reservations": {"token_1": {"task_name": "active_task"}, "token_2": {}}},
        )

        (config_root / "config_task").mkdir(parents=True)
        (config_root / "config_task" / "datasets_config.yaml").write_text(
            "datasets: []\n",
            encoding="utf-8",
        )
        generated_dir.mkdir(parents=True)
        (generated_dir / "batch_pipeline_universal_generated_task.py").write_text(
            "# generated\n",
            encoding="utf-8",
        )

        seen_dag_ids = []

        def fake_stop_generated_airflow_runs(dag_ids, apply_changes):
            assert apply_changes is True
            seen_dag_ids.extend(dag_ids)
            return {
                "dag_ids": list(dag_ids),
                "dag_runs_failed": 3,
                "task_instances_failed": 4,
                "dags_paused": 5,
            }

        tm.stop_generated_airflow_runs = fake_stop_generated_airflow_runs
        tm.stop_all_task_containers = lambda apply_changes: 2

        result = tm.platform_restart_cleanup(
            SimpleNamespace(
                yes=True,
                dags_dir=str(dags_dir),
                task_config_root=str(config_root),
                stop_containers=True,
            )
        )

        assert result["scheduled_stopped"] == 2
        assert result["containers_stopped"] == 2
        assert result["gpu_reservations_cleared"] == 2
        assert result["task_lock_cleared"] == 1
        assert result["queue_entries_cleared"] == 2
        assert "batch_pipeline_universal_active_task" in seen_dag_ids
        assert "batch_pipeline_universal_queued_task" in seen_dag_ids
        assert "batch_pipeline_universal_config_task" in seen_dag_ids
        assert "batch_pipeline_universal_generated_task" in seen_dag_ids

        queue_state = json.loads((queue_dir / "queue.lock").read_text(encoding="utf-8"))
        assert queue_state["active"] is None
        assert queue_state["queue"] == []

        schedule_state = json.loads(schedule_file.read_text(encoding="utf-8"))
        statuses = {item["schedule_id"]: item["status"] for item in schedule_state["items"]}
        assert statuses == {
            "sched_1": "stopped",
            "sched_2": "stopped",
            "sched_3": "submitted",
        }

        assert (lock_dir / "active_task.lock").read_text(encoding="utf-8") == ""
        gpu_state = json.loads((gpu_dir / "gpu_0.lock").read_text(encoding="utf-8"))
        assert gpu_state["reservations"] == {}


if __name__ == "__main__":
    main()
