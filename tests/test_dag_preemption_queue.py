#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dags"))

import batch_pipeline_universal as dag_runtime


def main():
    original_list_task_dag_runs = dag_runtime.list_task_dag_runs
    original_task_queue_file = dag_runtime.TASK_QUEUE_FILE
    original_task_lock_file = dag_runtime.TASK_LOCK_FILE
    original_task_lock_dir = dag_runtime.TASK_LOCK_DIR
    original_pause_task_dag = dag_runtime.pause_task_dag
    original_unpause_task_dag = dag_runtime.unpause_task_dag
    original_trigger_pending_task_runs = dag_runtime.trigger_pending_task_runs
    original_metadata_fetchall = dag_runtime.metadata_fetchall
    original_preempted_original_runs_still_active = (
        dag_runtime.preempted_original_runs_still_active
    )
    try:
        dag_runtime.list_task_dag_runs = lambda dag_id: [
                {
                    "state": "success",
                    "conf": {"dataset_name": "clip_success"},
                    "task_states": {"verify_pipeline_status": "success"},
                },
                {
                    "state": "success",
                    "conf": {
                        "dataset_name": "clip_recovery_success",
                        "pipeline_stages": [["parser"], ["segment"]],
                        dag_runtime.PLATFORM_RECOVERY_CONF_KEY: dag_runtime.PLATFORM_RECOVERY_REASON_PREEMPTED,
                        dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY: "segment",
                    },
                    "task_states": {
                        "run_parser": "skipped",
                        "validate_parser": "skipped",
                        "run_segment": "success",
                        "validate_segment": "success",
                        "verify_pipeline_status": "success",
                    },
                },
                {
                    "state": "success",
                    "conf": {
                        "dataset_name": "clip_bad_verify_success",
                        "pipeline_stages": [["parser"], ["segment"]],
                    },
                    "task_states": {
                        "run_parser": "skipped",
                        "verify_pipeline_status": "success",
                    },
                },
                {
                    "state": "success",
                    "conf": {"dataset_name": "clip_preempted_success"},
                    "task_states": {"run_segment": "skipped", "verify_pipeline_status": "skipped"},
                },
            {"state": "queued", "conf": {"dataset_name": "clip_active"}},
            {"state": "failed", "conf": {"dataset_name": "clip_failed"}},
        ]
        plan = dag_runtime.pending_activation_plan(
            "dag_low",
            [
                {"dataset_name": "clip_success"},
                {
                    "dataset_name": "clip_recovery_success",
                    "pipeline_stages": [["parser"], ["segment"]],
                    dag_runtime.PLATFORM_RECOVERY_CONF_KEY: dag_runtime.PLATFORM_RECOVERY_REASON_PREEMPTED,
                    dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY: "segment",
                },
                {
                    "dataset_name": "clip_bad_verify_success",
                    "pipeline_stages": [["parser"], ["segment"]],
                },
                {"dataset_name": "clip_preempted_success"},
                {"dataset_name": "clip_active"},
                {"dataset_name": "clip_failed"},
            ],
        )
        assert plan["expected_runs"] == 4
        assert [conf["dataset_name"] for conf in plan["trigger_confs"]] == [
            "clip_bad_verify_success",
            "clip_preempted_success",
            "clip_failed",
        ]
        assert plan["active_datasets"] == ["clip_active"]
        assert plan["skipped_success"] == ["clip_recovery_success", "clip_success"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            queue_file = tmp_path / "queue.lock"
            dag_runtime.TASK_QUEUE_FILE = str(queue_file)
            dag_runtime.TASK_LOCK_DIR = str(tmp_path / "task_locks")
            dag_runtime.TASK_LOCK_FILE = str(tmp_path / "task_locks" / "active_task.lock")
            queue_file.write_text(
                json.dumps(
                    {
                        "active": {"task_name": "high_task"},
                        "queue": [
                            {
                                "task_name": "low_task",
                                "preempted": True,
                                "preempted_at": time.time() - 61,
                                "priority": 80,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cleanup_plan = dag_runtime.preempt_cleanup_plan(
                "high_task",
                "low_task",
                {"active_runs": [{"pid": os.getpid(), "dataset_name": "clip_1"}]},
                1,
            )
            assert cleanup_plan is not None
            assert cleanup_plan["locked_task_name"] == "low_task"

            try:
                dag_runtime.ensure_task_still_active(
                    "low_task",
                    "dag_low",
                    "clip_1",
                    "segment",
                )
            except dag_runtime.TaskPreempted:
                pass
            else:
                raise AssertionError("expected TaskPreempted when another task is active")

            paused_dags = []
            dag_runtime.pause_task_dag = lambda dag_id: paused_dags.append(dag_id)
            dag_runtime.preempted_original_runs_still_active = (
                lambda dag_id, current_run_id: True
            )
            paused = dag_runtime.pause_task_dag_after_preempted_runs_drain(
                "low_task",
                "dag_low",
                "run_1",
            )
            assert paused is False
            assert paused_dags == []

            dag_runtime.preempted_original_runs_still_active = (
                lambda dag_id, current_run_id: False
            )
            paused = dag_runtime.pause_task_dag_after_preempted_runs_drain(
                "low_task",
                "dag_low",
                "run_1",
            )
            assert paused is True
            assert paused_dags == ["dag_low"]

            clip_dir = tmp_path / "clip_1"
            clip_dir.mkdir()
            (clip_dir / "results_parser.json").write_text("{}", encoding="utf-8")
            (clip_dir / "results_occ.json").write_text("{}", encoding="utf-8")
            (clip_dir / "keep.json").write_text("{}", encoding="utf-8")
            removed = dag_runtime.cleanup_result_jsons(tmp_path, "clip_1")
            assert removed == 2
            assert not (clip_dir / "results_parser.json").exists()
            assert not (clip_dir / "results_occ.json").exists()
            assert (clip_dir / "keep.json").exists()

            queue_file.write_text(
                json.dumps(
                    {
                        "active": {
                            "task_name": "task_a",
                            "dag_id": "dag_a",
                            "status": "draining",
                            "preempt_requested": True,
                            "preempt_requested_by": "task_b",
                            "priority": 80,
                            "total_runs": 1,
                            "remaining_runs": 1,
                            "completed_run_ids": [],
                        },
                        "queue": [
                            {
                                "task_name": "task_b",
                                "dag_id": "dag_b",
                                "status": "queued",
                                "priority": 5,
                                "total_runs": 1,
                                "remaining_runs": 1,
                                "pending_run_confs": [
                                    {
                                        "task_name": "task_b",
                                        "dataset_name": "clip_b",
                                        "dataset_path": "/tmp",
                                        "pipeline_stages": [["parser"], ["segment"]],
                                    }
                                ],
                            }
                        ],
                        "version": 2,
                    }
                ),
                encoding="utf-8",
            )
            context = {
                "params": {
                    "task_name": "task_a",
                    "dataset_name": "clip_a",
                    "dataset_path": "/tmp",
                    "task_exclusive": True,
                    "pipeline_stages": [["parser"], ["segment"], ["od"]],
                },
                "dag": SimpleNamespace(dag_id="dag_a"),
                "dag_run": SimpleNamespace(
                    dag_id="dag_a",
                    run_id="run_a_1",
                    conf={
                        "task_name": "task_a",
                        "dataset_name": "clip_a",
                        "dataset_path": "/tmp",
                        "pipeline_stages": [["parser"], ["segment"], ["od"]],
                    },
                ),
            }
            assert dag_runtime.stage_before_resume("parser", context) is False
            result = dag_runtime.record_stage_checkpoint_after_validate(
                "task_a",
                "dag_a",
                "run_a_1",
                "clip_a",
                "parser",
                context,
            )
            assert result == "drained"
            state = json.loads(queue_file.read_text(encoding="utf-8"))
            assert state["active"]["drained_run_ids"] == ["run_a_1"]
            drained_conf = state["active"]["drained_run_confs"][0]
            assert drained_conf["dataset_name"] == "clip_a"
            assert drained_conf[dag_runtime.PLATFORM_RECOVERY_CONF_KEY] == (
                dag_runtime.PLATFORM_RECOVERY_REASON_PREEMPTED
            )
            assert drained_conf[dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY] == "segment"
            assert dag_runtime.task_run_is_drained_for_preemption(
                "task_a",
                "dag_a",
                "run_a_1",
            )

            resume_context = {
                "params": context["params"],
                "dag_run": SimpleNamespace(
                    conf={
                        "pipeline_stages": [["parser"], ["segment"], ["od"]],
                        dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY: "segment",
                    }
                ),
            }
            assert dag_runtime.stage_before_resume("parser", resume_context) is True
            assert dag_runtime.stage_before_resume("segment", resume_context) is False

            runtime_calls = []
            dag_runtime.pause_task_dag = lambda dag_id: runtime_calls.append(
                ("pause", dag_id)
            )
            dag_runtime.unpause_task_dag = lambda dag_id: runtime_calls.append(
                ("unpause", dag_id)
            )
            dag_runtime.trigger_pending_task_runs = (
                lambda dag_id, confs: runtime_calls.append(
                    (
                        "trigger",
                        dag_id,
                        [conf["dataset_name"] for conf in confs],
                    )
                )
            )
            result = dag_runtime.finalize_drained_run_and_maybe_advance_queue(
                "task_a",
                "dag_a",
                "run_a_1",
            )
            assert result == "drained_advance"
            state = json.loads(queue_file.read_text(encoding="utf-8"))
            assert state["active"]["task_name"] == "task_b"
            assert state["queue"][0]["task_name"] == "task_a"
            assert state["queue"][0]["pending_run_confs"][0][
                dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY
            ] == "segment"
            assert ("pause", "dag_a") in runtime_calls
            assert ("trigger", "dag_a", ["clip_a"]) not in runtime_calls
            assert ("trigger", "dag_b", ["clip_b"]) in runtime_calls
            assert ("unpause", "dag_b") in runtime_calls

            queue_file.write_text(
                json.dumps(
                    {
                        "active": {
                            "task_name": "task_a",
                            "dag_id": "dag_a",
                            "status": "draining",
                            "preempt_requested": True,
                            "preempt_requested_by": "task_b",
                            "priority": 80,
                            "total_runs": 3,
                            "remaining_runs": 3,
                            "completed_run_ids": [],
                            "drain_target_run_ids": ["run_a_1", "run_a_2"],
                            "pending_run_confs": [
                                {
                                    "task_name": "task_a",
                                    "dataset_name": "clip_a1",
                                    "dataset_path": "/tmp",
                                    "pipeline_stages": [["parser"], ["segment"], ["od"]],
                                },
                                {
                                    "task_name": "task_a",
                                    "dataset_name": "clip_a2",
                                    "dataset_path": "/tmp",
                                    "pipeline_stages": [["parser"], ["segment"], ["od"]],
                                },
                                {
                                    "task_name": "task_a",
                                    "dataset_name": "clip_a3",
                                    "dataset_path": "/tmp",
                                    "pipeline_stages": [["parser"], ["segment"], ["od"]],
                                },
                            ],
                        },
                        "queue": [
                            {
                                "task_name": "task_b",
                                "dag_id": "dag_b",
                                "status": "queued",
                                "priority": 20,
                                "total_runs": 1,
                                "remaining_runs": 1,
                                "pending_run_confs": [
                                    {
                                        "task_name": "task_b",
                                        "dataset_name": "clip_b",
                                        "dataset_path": "/tmp",
                                        "pipeline_stages": [["parser"], ["segment"]],
                                    }
                                ],
                            },
                            {
                                "task_name": "task_c",
                                "dag_id": "dag_c",
                                "status": "queued",
                                "priority": 5,
                                "total_runs": 1,
                                "remaining_runs": 1,
                                "pending_run_confs": [
                                    {
                                        "task_name": "task_c",
                                        "dataset_name": "clip_c",
                                        "dataset_path": "/tmp",
                                        "pipeline_stages": [["parser"], ["segment"]],
                                    }
                                ],
                            },
                        ],
                        "version": 2,
                    }
                ),
                encoding="utf-8",
            )

            for run_id, dataset_name in (("run_a_1", "clip_a1"), ("run_a_2", "clip_a2")):
                drain_context = {
                    "params": {
                        "task_name": "task_a",
                        "dataset_name": dataset_name,
                        "dataset_path": "/tmp",
                        "task_exclusive": True,
                        "pipeline_stages": [["parser"], ["segment"], ["od"]],
                    },
                    "dag": SimpleNamespace(dag_id="dag_a"),
                    "dag_run": SimpleNamespace(
                        dag_id="dag_a",
                        run_id=run_id,
                        conf={
                            "task_name": "task_a",
                            "dataset_name": dataset_name,
                            "dataset_path": "/tmp",
                            "pipeline_stages": [["parser"], ["segment"], ["od"]],
                        },
                    ),
                }
                assert dag_runtime.record_stage_checkpoint_after_validate(
                    "task_a",
                    "dag_a",
                    run_id,
                    dataset_name,
                    "parser",
                    drain_context,
                ) == "drained"

            runtime_calls.clear()
            dag_runtime.list_task_dag_runs = lambda dag_id: [
                {
                    "run_id": "run_a_1",
                    "state": "success",
                    "conf": {"dataset_name": "clip_a1"},
                    "task_states": {},
                },
                {
                    "run_id": "run_a_2",
                    "state": "success",
                    "conf": {"dataset_name": "clip_a2"},
                    "task_states": {},
                },
                {
                    "run_id": "run_a_3",
                    "state": "queued",
                    "conf": {"dataset_name": "clip_a3"},
                    "task_states": {},
                },
            ]
            assert dag_runtime.finalize_drained_run_and_maybe_advance_queue(
                "task_a",
                "dag_a",
                "run_a_1",
            ) == "drained_waiting"
            assert runtime_calls == []

            assert dag_runtime.finalize_drained_run_and_maybe_advance_queue(
                "task_a",
                "dag_a",
                "run_a_2",
            ) == "drained_advance"
            state = json.loads(queue_file.read_text(encoding="utf-8"))
            assert state["active"]["task_name"] == "task_c"
            assert [entry["task_name"] for entry in state["queue"]] == ["task_b", "task_a"]
            recovered_a = state["queue"][1]
            recovered_by_dataset = {
                conf["dataset_name"]: conf
                for conf in recovered_a["pending_run_confs"]
            }
            assert recovered_by_dataset["clip_a1"][
                dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY
            ] == "segment"
            assert recovered_by_dataset["clip_a2"][
                dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY
            ] == "segment"
            assert dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY not in recovered_by_dataset["clip_a3"]
            assert recovered_by_dataset["clip_a3"][
                dag_runtime.PLATFORM_RECOVERY_CONF_KEY
            ] == dag_runtime.PLATFORM_RECOVERY_REASON_PREEMPTED
            assert recovered_a[dag_runtime.BLOCKED_ORIGINAL_RUN_IDS_KEY] == [
                "run_a_1",
                "run_a_2",
                "run_a_3",
            ]
            assert ("trigger", "dag_c", ["clip_c"]) in runtime_calls
            assert ("unpause", "dag_c") in runtime_calls

            dag_runtime.list_task_dag_runs = lambda dag_id: [
                {
                    "run_id": "old_original_clip_a3",
                    "state": "running",
                    "conf": {"dataset_name": "clip_a3"},
                    "task_states": {},
                }
            ]
            plan = dag_runtime.pending_activation_plan(
                "dag_a",
                [recovered_by_dataset["clip_a3"]],
            )
            assert plan["expected_runs"] == 1
            assert [conf["dataset_name"] for conf in plan["trigger_confs"]] == ["clip_a3"]
            assert plan["active_datasets"] == []

            active_recovered_a = dag_runtime.activate_queue_entry(recovered_a)
            assert active_recovered_a[dag_runtime.BLOCKED_ORIGINAL_RUN_IDS_KEY] == [
                "run_a_1",
                "run_a_2",
                "run_a_3",
            ]
            queue_file.write_text(
                json.dumps(
                    {
                        "active": active_recovered_a,
                        "queue": [],
                        "version": 2,
                    }
                ),
                encoding="utf-8",
            )
            assert dag_runtime.task_run_preemption_state(
                "task_a",
                "dag_a",
                "run_a_3",
            ) == "blocked_original"
            skipped_context = {
                "params": {
                    "task_name": "task_a",
                    "pipeline_stages": [["parser"], ["segment"], ["od"]],
                },
                "dag": SimpleNamespace(dag_id="dag_a"),
                "dag_run": SimpleNamespace(
                    dag_id="dag_a",
                    run_id="run_a_3",
                    conf={
                        "task_name": "task_a",
                        "dataset_name": "clip_a3",
                        "pipeline_stages": [["parser"], ["segment"], ["od"]],
                    },
                ),
            }
            dag_runtime.metadata_fetchall = lambda sql, params=(): [
                {"task_id": "run_parser", "state": "skipped"},
                {"task_id": "validate_parser", "state": "skipped"},
                {"task_id": "verify_pipeline_status", "state": "running"},
            ] if "select task_id, state from task_instance" in sql else []
            try:
                dag_runtime.verify_pipeline_terminal_state(**skipped_context)
            except dag_runtime.AirflowSkipException:
                pass
            else:
                raise AssertionError("expected skipped unfinished run to stay uncounted")

            dag_runtime.metadata_fetchall = lambda sql, params=(): [
                {"state": "skipped"}
            ] if "task_id = %s" in sql else []
            try:
                dag_runtime.finalize_task_queue_task(**skipped_context)
            except dag_runtime.AirflowSkipException:
                pass
            else:
                raise AssertionError("expected skipped finalizer to stay skipped")

            resume_verify_context = {
                "params": {
                    "task_name": "task_a",
                    "pipeline_stages": [["parser"], ["segment"], ["od"]],
                },
                "dag": SimpleNamespace(dag_id="dag_a"),
                "dag_run": SimpleNamespace(
                    dag_id="dag_a",
                    run_id="run_a_resume",
                    conf={
                        "task_name": "task_a",
                        "dataset_name": "clip_a1",
                        "pipeline_stages": [["parser"], ["segment"], ["od"]],
                        dag_runtime.PLATFORM_RESUME_FROM_STAGE_KEY: "segment",
                    },
                ),
            }
            dag_runtime.metadata_fetchall = lambda sql, params=(): [
                {"task_id": "run_parser", "state": "skipped"},
                {"task_id": "validate_parser", "state": "skipped"},
                {"task_id": "run_segment", "state": "success"},
                {"task_id": "validate_segment", "state": "success"},
                {"task_id": "verify_pipeline_status", "state": "running"},
            ] if "select task_id, state from task_instance" in sql else []
            assert dag_runtime.verify_pipeline_terminal_state(**resume_verify_context) == "success"

            queue_file.write_text(
                json.dumps(
                    {
                        "active": {
                            "task_name": "task_fail",
                            "dag_id": "dag_fail",
                            "status": "active",
                            "priority": 30,
                            "total_runs": 2,
                            "remaining_runs": 2,
                            "completed_run_ids": [],
                        },
                        "queue": [
                            {
                                "task_name": "task_next",
                                "dag_id": "dag_next",
                                "status": "queued",
                                "priority": 10,
                                "total_runs": 1,
                                "remaining_runs": 1,
                            }
                        ],
                        "version": 2,
                    }
                ),
                encoding="utf-8",
            )
            dag_runtime.metadata_fetchall = lambda sql, params=(): [
                {"state": "failed"}
            ] if "task_id = %s" in sql else []
            fail_context = {
                "params": {"task_name": "task_fail"},
                "dag": SimpleNamespace(dag_id="dag_fail"),
                "dag_run": SimpleNamespace(dag_id="dag_fail", run_id="run_fail_1"),
            }
            assert dag_runtime.finalize_task_queue_task(**fail_context) == "active_remaining_failed"
            state = json.loads(queue_file.read_text(encoding="utf-8"))
            assert state["active"]["task_name"] == "task_fail"
            assert state["active"]["remaining_runs"] == 1
            assert state["active"].get("completed_run_ids") == []
            assert state["active"].get("failed_run_ids") == ["run_fail_1"]
            assert state["queue"][0]["task_name"] == "task_next"
    finally:
        dag_runtime.list_task_dag_runs = original_list_task_dag_runs
        dag_runtime.TASK_QUEUE_FILE = original_task_queue_file
        dag_runtime.TASK_LOCK_FILE = original_task_lock_file
        dag_runtime.TASK_LOCK_DIR = original_task_lock_dir
        dag_runtime.pause_task_dag = original_pause_task_dag
        dag_runtime.unpause_task_dag = original_unpause_task_dag
        dag_runtime.trigger_pending_task_runs = original_trigger_pending_task_runs
        dag_runtime.metadata_fetchall = original_metadata_fetchall
        dag_runtime.preempted_original_runs_still_active = (
            original_preempted_original_runs_still_active
        )


if __name__ == "__main__":
    main()
