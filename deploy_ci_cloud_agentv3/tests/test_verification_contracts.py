from __future__ import annotations

import pytest

from deploy_ci_cloud_agentv3.services.verification import VerificationService


class SnapshotRuntime:
    def __init__(self, snapshot, config=None): self.snapshot=snapshot; self.config=config
    async def call_tool(self, name, args):
        if name == "get_action_verification_snapshot": return self.snapshot
        if name == "get_task_config_for_verification": return {"task_name": args["task_name"], "config": self.config}
        raise AssertionError(name)


@pytest.mark.asyncio
async def test_delete_does_not_verify_when_dag_queue_or_runs_remain():
    snapshot = {
        "task_exists": False, "config_file_exists": False, "dag_file_exists": False,
        "airflow_dag_exists": True, "containers": [], "gpu_reservations": [],
        "queue": {"location": "queued", "position": 1},
        "airflow_runs": [{"state": "running", "dataset_name": "d"}], "errors": {},
    }
    verified, _ = await VerificationService(SnapshotRuntime(snapshot)).verify("delete_task", {"task_name": "task_a"}, {})
    assert verified is False


@pytest.mark.asyncio
async def test_delete_observation_error_is_not_absence_evidence():
    snapshot = {
        "task_exists": False, "config_file_exists": False, "dag_file_exists": False,
        "airflow_dag_exists": False, "containers": [], "gpu_reservations": [],
        "queue": {"location": "not_found", "position": -1}, "airflow_runs": [],
        "errors": {"docker": "docker unavailable"},
    }
    verified, after = await VerificationService(SnapshotRuntime(snapshot)).verify("delete_task", {"task_name": "task_a"}, {})
    assert verified is False
    assert after["verification_errors"]["docker"]


@pytest.mark.asyncio
async def test_delete_airflow_not_found_is_valid_absence_evidence():
    snapshot = {
        "task_exists": False, "config_file_exists": False, "dag_file_exists": False,
        "airflow_dag_exists": False, "containers": [], "gpu_reservations": [],
        "queue": {"location": "not_found", "position": -1}, "airflow_runs": [],
        "errors": {"airflow": "HTTP 404 DAG not found"},
    }
    verified, _ = await VerificationService(SnapshotRuntime(snapshot)).verify("delete_task", {"task_name": "task_a"}, {})
    assert verified is True


@pytest.mark.asyncio
async def test_submit_requires_exact_reviewed_config_readback():
    expected = {"max_active_runs": 1, "datasets": [{"dataset_name": "d", "dataset_path": "/a"}]}
    snapshot = {
        "task_exists": True, "config_file_exists": True, "dag_file_exists": True,
        "airflow_dag_exists": True, "errors": {},
    }
    raw = {"result": {"task_name": "train_generated"}}
    verified, _ = await VerificationService(SnapshotRuntime(snapshot, config=expected)).verify("submit_task", {"task_prefix": "train", "config": expected}, raw)
    assert verified is True
    tampered_actual = {"max_active_runs": 1, "datasets": [{"dataset_name": "d", "dataset_path": "/other"}]}
    verified, _ = await VerificationService(SnapshotRuntime(snapshot, config=tampered_actual)).verify("submit_task", {"task_prefix": "train", "config": expected}, raw)
    assert verified is False


@pytest.mark.asyncio
async def test_resume_rejects_old_success_run_as_false_success():
    before = {
        "airflow_runs": [
            {"run_id": "old_success", "dataset_name": "target", "state": "success"}
        ]
    }
    after = {
        "task_exists": True,
        "airflow_runs": [
            {"run_id": "old_success", "dataset_name": "target", "state": "success"}
        ],
        "errors": {},
    }
    verified, _ = await VerificationService(SnapshotRuntime(after)).verify(
        "resume_task", {"task_name": "task_a", "datasets": ["target"]}, {}, before=before
    )
    assert verified is False


@pytest.mark.asyncio
async def test_resume_rejects_new_run_for_wrong_dataset():
    before = {"airflow_runs": [{"run_id": "r1", "dataset_name": "target", "state": "failed"}]}
    after = {
        "task_exists": True,
        "airflow_runs": [
            {"run_id": "r1", "dataset_name": "target", "state": "failed"},
            {"run_id": "r2", "dataset_name": "OTHER", "state": "running"},
        ],
        "errors": {},
    }
    verified, _ = await VerificationService(SnapshotRuntime(after)).verify(
        "resume_task", {"task_name": "task_a", "datasets": ["target"]}, {}, before=before
    )
    assert verified is False


@pytest.mark.asyncio
async def test_resume_requires_new_progress_run_for_every_approved_dataset():
    before = {
        "airflow_runs": [
            {"run_id": "a-old", "dataset_name": "A", "state": "failed"},
            {"run_id": "b-old", "dataset_name": "B", "state": "failed"},
        ]
    }
    after = {
        "task_exists": True,
        "airflow_runs": [
            *before["airflow_runs"],
            {"run_id": "a-new", "dataset_name": "A", "state": "running"},
            {"run_id": "b-new", "dataset_name": "B", "state": "queued"},
        ],
        "errors": {},
    }
    verified, _ = await VerificationService(SnapshotRuntime(after)).verify(
        "resume_task", {"task_name": "task_a", "datasets": ["A", "B"]}, {}, before=before
    )
    assert verified is True


@pytest.mark.asyncio
async def test_stop_entire_task_requires_gpu_release_and_queue_removal():
    snapshot = {
        "task_exists": True,
        "containers": [],
        "gpu_reservations": [{"gpu_id": "0", "task_name": "task_a", "dataset_name": "d1"}],
        "airflow_runs": [],
        "queue": {"location": "queued", "position": 1},
        "errors": {},
    }
    verified, _ = await VerificationService(SnapshotRuntime(snapshot)).verify(
        "stop_task", {"task_name": "task_a", "datasets": None}, {}
    )
    assert verified is False


@pytest.mark.asyncio
async def test_stop_selected_dataset_does_not_require_whole_task_queue_removal():
    snapshot = {
        "task_exists": True,
        "containers": [],
        "gpu_reservations": [],
        "airflow_runs": [
            {"run_id": "other", "dataset_name": "OTHER", "state": "running"},
            {"run_id": "target", "dataset_name": "target", "state": "failed"},
        ],
        "queue": {"location": "active", "position": 0},
        "errors": {},
    }
    verified, _ = await VerificationService(SnapshotRuntime(snapshot)).verify(
        "stop_task", {"task_name": "task_a", "datasets": ["target"]}, {}
    )
    assert verified is True
