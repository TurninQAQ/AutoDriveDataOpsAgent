#!/usr/bin/env python3
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_local_settings(path):
    spec = importlib.util.spec_from_file_location("deploy_ci_airflow_local_settings_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_raises(expected_text, func):
    try:
        func()
    except Exception as exc:
        assert expected_text in str(exc), str(exc)
        return
    raise AssertionError("expected exception containing: {}".format(expected_text))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        airflow_home = tmp_path / "airflow"
        dags_dir = airflow_home / "dags" / "data_center"
        generated_dir = dags_dir / "generated"
        task_config_root = tmp_path / "opt_airflow" / "config" / "tasks"

        os.environ["AIRFLOW_HOME"] = str(airflow_home)
        os.environ["AIRFLOW_DAGS_DIR"] = str(dags_dir)
        os.environ["AIRFLOW_TASK_CONFIG_ROOT"] = str(task_config_root)
        os.environ["AIRFLOW_SCRIPTS_DIR"] = str(REPO_ROOT / "scripts")
        os.environ.pop("AIRFLOW_PLATFORM_DELETE_BYPASS", None)
        os.environ.pop("AIRFLOW_PLATFORM_DELETE_PATCH_DISABLED", None)

        if str(REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))

        from airflow.api.common import delete_dag as delete_dag_module

        original_marker = getattr(delete_dag_module, "_deploy_ci_cloud_original_delete_dag", None)
        original_delete = delete_dag_module.delete_dag
        original_marker_present = hasattr(delete_dag_module, "_deploy_ci_cloud_original_delete_dag")
        calls = []

        def fake_original_delete_dag(dag_id, keep_records_in_log=True, session=None):
            calls.append(("original", dag_id, keep_records_in_log, session))
            return 42

        try:
            if original_marker_present:
                delattr(delete_dag_module, "_deploy_ci_cloud_original_delete_dag")
            delete_dag_module.delete_dag = fake_original_delete_dag

            load_local_settings(REPO_ROOT / "config" / "airflow_local_settings.py")

            import task_manager

            task_name = "taska_20260729_000000"
            dag_id = "batch_pipeline_universal_{}".format(task_name)
            (task_config_root / task_name).mkdir(parents=True)
            (task_config_root / task_name / "datasets_config.yaml").write_text(
                "datasets: []\n",
                encoding="utf-8",
            )
            generated_dir.mkdir(parents=True)
            (generated_dir / "{}.py".format(dag_id)).write_text("# generated\n", encoding="utf-8")

            delete_calls = []
            original_delete_task_by_name = task_manager.delete_task_by_name

            def fake_delete_task_by_name(*args, **kwargs):
                delete_calls.append((args, kwargs))
                return {"dag_metadata_deleted": 7}

            task_manager.delete_task_by_name = fake_delete_task_by_name
            try:
                assert_raises(
                    "Platform shared DAG cannot be deleted",
                    lambda: delete_dag_module.delete_dag(
                        "batch_pipeline_universal",
                        session=object(),
                    ),
                )

                assert delete_dag_module.delete_dag(dag_id, session=object()) == 7
                assert delete_calls
                assert delete_calls[0][0][0] == task_name
                assert delete_calls[0][1]["use_api"] is False

                assert delete_dag_module.delete_dag("ordinary_dag", session=object()) == 42
                assert calls[-1][0:2] == ("original", "ordinary_dag")

                assert delete_dag_module.delete_dag(
                    "batch_pipeline_universal_test",
                    session=object(),
                ) == 42
                assert calls[-1][0:2] == ("original", "batch_pipeline_universal_test")

                orphan_task_name = "orphan_20260729_000000"
                orphan_dag_id = "batch_pipeline_universal_{}".format(orphan_task_name)
                (generated_dir / "{}.py".format(orphan_dag_id)).write_text(
                    "# generated\n",
                    encoding="utf-8",
                )
                assert_raises(
                    "Platform task DAG state is inconsistent",
                    lambda: delete_dag_module.delete_dag(orphan_dag_id, session=object()),
                )

                os.environ["AIRFLOW_PLATFORM_DELETE_BYPASS"] = "1"
                assert delete_dag_module.delete_dag(dag_id, session=object()) == 42
                assert calls[-1][0:2] == ("original", dag_id)
            finally:
                task_manager.delete_task_by_name = original_delete_task_by_name

            core_task_name = "coredelete_20260729_000000"
            core_dag_id = "batch_pipeline_universal_{}".format(core_task_name)
            core_task_dir = task_config_root / core_task_name
            core_task_dir.mkdir(parents=True)
            (core_task_dir / "datasets_config.yaml").write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  - dataset_name: clip_001",
                        "    dataset_path: /tmp",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            core_dag_file = generated_dir / "{}.py".format(core_dag_id)
            core_dag_file.write_text("# generated\n", encoding="utf-8")

            originals = {
                "pause_dag_db": task_manager.pause_dag_db,
                "fail_selected_runs_db": task_manager.fail_selected_runs_db,
                "stop_containers": task_manager.stop_containers,
                "wait_for_task_reservations": task_manager.wait_for_task_reservations,
                "remove_task_from_queue": task_manager.remove_task_from_queue,
                "clear_task_lock": task_manager.clear_task_lock,
            }
            original_delete_calls = []

            def fake_fail_selected_runs_db(dag_id, dataset_names, apply_changes, session=None):
                assert dag_id == core_dag_id
                assert dataset_names == ["clip_001"]
                assert apply_changes is True
                return (
                    [
                        {
                            "dag_run_id": "run_1",
                            "state": "running",
                            "conf": {"dataset_name": "clip_001"},
                        }
                    ],
                    1,
                    2,
                )

            def fake_stop_containers(task_name, config, dataset_names, apply_changes):
                assert task_name == core_task_name
                assert dataset_names == ["clip_001"]
                assert apply_changes is True
                return 3

            def fake_original_metadata_delete(dag_id, keep_records_in_log=True, session=None):
                original_delete_calls.append((dag_id, keep_records_in_log, session))
                return 4

            try:
                task_manager.pause_dag_db = lambda dag_id, paused=True, session=None: 1
                task_manager.fail_selected_runs_db = fake_fail_selected_runs_db
                task_manager.stop_containers = fake_stop_containers
                task_manager.wait_for_task_reservations = lambda task_name, dataset_names: 0
                task_manager.remove_task_from_queue = (
                    lambda task_name, apply_changes, advance_next=True, task_config_root=None, api_base=None, token=None: (1, "", [])
                )
                task_manager.clear_task_lock = lambda task_name, apply_changes: 1

                result = task_manager.delete_task_by_name(
                    core_task_name,
                    apply_changes=True,
                    stop_running_containers=True,
                    use_api=False,
                    original_delete_dag=fake_original_metadata_delete,
                    session=object(),
                    print_summary=False,
                )

                assert result["dag_runs_failed"] == 1
                assert result["task_instances_failed"] == 2
                assert result["containers_stopped"] == 3
                assert result["queue_removed"] == 1
                assert result["task_lock_cleared"] == 1
                assert result["dag_runs_deleted"] == 1
                assert result["dag_file_deleted"] == 1
                assert result["task_dir_deleted"] == 1
                assert result["dag_metadata_deleted"] == 1
                assert not core_dag_file.exists()
                assert not core_task_dir.exists()
                assert original_delete_calls[0][0] == core_dag_id
            finally:
                for name, value in originals.items():
                    setattr(task_manager, name, value)
        finally:
            delete_dag_module.delete_dag = original_delete
            if original_marker_present:
                setattr(delete_dag_module, "_deploy_ci_cloud_original_delete_dag", original_marker)
            elif hasattr(delete_dag_module, "_deploy_ci_cloud_original_delete_dag"):
                delattr(delete_dag_module, "_deploy_ci_cloud_original_delete_dag")


if __name__ == "__main__":
    main()
