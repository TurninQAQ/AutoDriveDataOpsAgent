from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from platform_core.config import normalize_gpu_config, normalize_pipeline_stages
from platform_core.gateways.airflow import AirflowGateway
from platform_core.gateways.docker import container_matches_dataset
from platform_core.gateways import gpu_reservations
from platform_core.services.queue_service import QueueService
from platform_core.services.task_service import TaskService


REPO_ROOT = Path(__file__).resolve().parents[1]


def minimal_task_config(dataset_path: Path):
    return {
        "pipeline_stages": ["precheck", "parser", ["od", "occ"]],
        "max_active_runs": 2,
        "task_type": "test",
        "gpu_ids": "0,1",
        "gpu_stages": "od,occ",
        "exclusive_gpu_stages": "od",
        "exclusive_gpu_idle_used_max_mb": 512,
        "gpu_stage_memory_mb": {"od": 24000, "occ": 4000},
        "gpu_wait_interval_sec": 1,
        "gpu_reservation_pending_sec": 1,
        "datasets": [
            {
                "dataset_name": "clip_001",
                "dataset_path": str(dataset_path),
                "timeout_min": 30,
                "image_parser": "parser:test",
                "image_od": "od:test",
                "image_occ": "occ:test",
            }
        ],
    }


def test_task_service_prepares_task_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dags_dir = tmp_path / "dags"
        tasks_dir = tmp_path / "tasks"
        dataset_dir = tmp_path / "data" / "clip_001"
        dataset_dir.mkdir(parents=True)
        dags_dir.mkdir()

        config = minimal_task_config(dataset_dir)
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        service = TaskService(dags_dir=dags_dir, task_config_root=tasks_dir)
        prepared = service.prepare_submission("agentcore", yaml_path)

        assert prepared.task_name.startswith("agentcore_")
        assert prepared.dag_id == f"batch_pipeline_universal_{prepared.task_name}"
        assert prepared.target_yaml.is_file()
        assert prepared.dag_path.is_file()
        assert prepared.priority_config["task_type"] == "test"
        assert prepared.priority_config["priority"] == 50
        assert prepared.stage_groups == [["precheck"], ["parser"], ["od", "occ"]]
        rendered = prepared.dag_path.read_text(encoding="utf-8")
        assert prepared.dag_id in rendered
        assert "__DAG_ID__" not in rendered


def test_queue_service_snapshot_and_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        queue_file = Path(tmp) / "queue.lock"
        queue_file.write_text(
            json.dumps(
                {
                    "version": 2,
                    "active": {"task_name": "task_a", "priority": 10},
                    "queue": [
                        {"task_name": "task_b", "priority": 20},
                        {"task_name": "task_c", "priority": 30},
                    ],
                }
            ),
            encoding="utf-8",
        )
        service = QueueService(queue_file)
        assert service.task_status("task_a")["location"] == "active"
        assert service.task_status("task_b")["position"] == 1
        assert service.task_status("missing")["location"] == "not_found"


def test_docker_dataset_matching_is_token_safe():
    container = {
        "Name": "/airflow-task-demo--segment--clip_0010--123-456",
        "Id": "abc",
        "Config": {
            "Env": ["DATASET_NAME=clip_0010"],
            "Cmd": ["run", "clip_0010"],
            "Entrypoint": [],
        },
        "HostConfig": {"Binds": ["/data/clip_0010:/input"]},
        "Mounts": [{"Source": "/data/clip_0010", "Destination": "/input"}],
    }
    assert container_matches_dataset(container, "clip_0010") is True
    assert container_matches_dataset(container, "clip_001") is False


def test_gpu_reservation_cleanup_dead_pid():
    with tempfile.TemporaryDirectory() as tmp:
        old_dir = gpu_reservations.GPU_LOCK_DIR
        try:
            lock_dir = Path(tmp)
            gpu_reservations.GPU_LOCK_DIR = lock_dir
            lock_path = lock_dir / "gpu_0.lock"
            lock_path.write_text(
                json.dumps(
                    {
                        "reservations": {
                            "dead": {
                                "pid": 999999999,
                                "task_name": "task_a",
                                "dataset_name": "clip_001",
                                "stage": "segment",
                                "required_mb": 24000,
                                "exclusive": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            matches = gpu_reservations.active_task_reservations(
                "task_a", ["clip_001"], cleanup_dead=True
            )
            assert matches == []
            state = json.loads(lock_path.read_text(encoding="utf-8"))
            assert state["reservations"] == {}
        finally:
            gpu_reservations.GPU_LOCK_DIR = old_dir


def test_gpu_config_still_uses_existing_rules():
    config = minimal_task_config(Path("/tmp/clip_001"))
    stages = [stage for group in normalize_pipeline_stages(config) for stage in group]
    normalized = normalize_gpu_config(config, stages=stages)
    assert normalized["exclusive_gpu_stages"] == "od"
    assert normalized["gpu_stage_memory_mb"]["od"] == 24000
    assert normalized["gpu_stage_memory_mb"]["occ"] == 4000


def test_airflow_gateway_cli_boundary():
    gateway = AirflowGateway(
        airflow_bin="/bin/echo",
        airflow_home="/tmp/airflow-home",
        run_home="/tmp/platform-home",
        api_timeout_sec=1,
    )
    result = gateway.run_cli(["dags", "list"])
    assert result.returncode == 0
    assert result.stdout.strip() == "dags list"


def test_deploy_copies_platform_core_package():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = Path(tmp)
        env = os.environ.copy()
        env.update(
            {
                "DEPLOY_SKIP_VERIFY": "1",
                "PLATFORM_HOME": str(runtime),
                "AIRFLOW_HOME": str(runtime / "airflow"),
                "AIRFLOW_BIN": "/bin/true",
                "AIRFLOW_DAGS_DIR": str(runtime / "airflow" / "dags" / "data_center"),
                "AIRFLOW_HOST_DATA_ROOT": str(runtime / "opt_airflow" / "data"),
                "AIRFLOW_CONFIG_DIR": str(runtime / "opt_airflow" / "config"),
                "AIRFLOW_SCRIPTS_DIR": str(runtime / "opt_airflow" / "scripts"),
                "AIRFLOW_TASK_CONFIG_ROOT": str(runtime / "opt_airflow" / "config" / "tasks"),
            }
        )
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "deploy_ci_cloud.sh")],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        deployed_core = runtime / "opt_airflow" / "platform_core"
        assert (deployed_core / "__init__.py").is_file()
        assert (deployed_core / "services" / "task_service.py").is_file()
        assert not any(deployed_core.rglob("__pycache__"))
        deployed_mcp = runtime / "opt_airflow" / "platform_mcp"
        assert (deployed_mcp / "__init__.py").is_file()
        assert (deployed_mcp / "server.py").is_file()
        assert not any(deployed_mcp.rglob("__pycache__"))
        deployed_agent = runtime / "opt_airflow" / "platform_agent"
        assert (deployed_agent / "__init__.py").is_file()
        assert (deployed_agent / "workflow.py").is_file()
        assert not any(deployed_agent.rglob("__pycache__"))

        help_result = subprocess.run(
            [
                os.environ.get("PYTHON", "python"),
                str(runtime / "opt_airflow" / "scripts" / "task_manager.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 0, help_result.stdout + "\n" + help_result.stderr
        assert "Manage generated Airflow batch tasks" in help_result.stdout

        agent_help = subprocess.run(
            [
                os.environ.get("PYTHON", "python"),
                str(runtime / "opt_airflow" / "scripts" / "dataops_agent.py"),
                "--help",
            ],
            env={**os.environ, "PYTHONPATH": str(runtime / "opt_airflow")},
            capture_output=True,
            text=True,
        )
        assert agent_help.returncode == 0, agent_help.stdout + "\n" + agent_help.stderr
        assert "read-only Agent" in agent_help.stdout


def test_diagnosis_service_aggregates_runtime_evidence():
    from platform_core.services.diagnosis_service import DiagnosisService

    class FakeQueue:
        def task_status(self, task_name):
            return {"location": "queued", "position": 2, "entry": {"task_name": task_name}}

    class FakeDocker:
        def matching_containers(self, task_name, config, dataset_names):
            return [
                {
                    "Id": "abcdef123456",
                    "Name": f"/airflow-task-{task_name}--segment--clip_001--1-2",
                    "Config": {"Image": "segment:test"},
                    "State": {"Status": "running"},
                }
            ]

    class FakeGPU:
        def task_reservations(self, task_name, dataset_names, cleanup_dead=False):
            assert cleanup_dead is True
            return [
                (
                    "gpu_0.lock",
                    "token-1",
                    {
                        "pid": 123,
                        "task_name": task_name,
                        "dataset_name": "clip_001",
                        "stage": "segment",
                        "required_mb": 24000,
                        "exclusive": True,
                    },
                )
            ]

    service = DiagnosisService(FakeQueue(), FakeDocker(), FakeGPU())
    snapshot = service.inspect_task("task_a", {}, ["clip_001"])
    assert snapshot["queue"]["position"] == 2
    assert snapshot["containers"][0]["state"] == "running"
    assert snapshot["gpu_reservations"][0]["required_mb"] == 24000
    assert snapshot["gpu_reservations"][0]["exclusive"] is True


def test_gpu_runtime_is_an_explicit_interface():
    from platform_core.gateways.gpu_runtime import GPURuntime

    assert GPURuntime.__abstractmethods__ == {"list_devices", "get_memory_info", "process_alive"}
