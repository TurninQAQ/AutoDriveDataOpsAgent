from __future__ import annotations

import json
from pathlib import Path

import yaml

from platform_core.gateways.gpu_runtime import SimulatedGPURuntime
from platform_core.services.diagnosis_service import DiagnosisService
from platform_core.services.docker_service import DockerService
from platform_core.services.gpu_service import GPUService
from platform_core.services.health_service import HealthService
from platform_core.services.queue_service import QueueService
from platform_core.services.task_query_service import TaskQueryService
from platform_core.settings import PlatformSettings
from platform_mcp.facade import PlatformMCPFacade
from platform_mcp.server import READ_ONLY_TOOL_NAMES


class FakeDockerGateway:
    def matching(self, task_name, config, dataset_names):
        return [
            {
                "Id": "container123",
                "Name": f"/airflow-task-{task_name}--segment--{dataset_names[0]}--1-2",
                "Config": {"Image": "segment:test"},
                "State": {"Status": "running", "Running": True, "ExitCode": 0},
            }
        ]

    def inspect_running(self):
        return []

    def managed(self):
        return []


class FakeAirflowService:
    def __init__(self):
        self.dag_id = None

    def health(self):
        return {
            "metadatabase": {"status": "healthy"},
            "scheduler": {"status": "healthy"},
        }

    def runs(self, dag_id, limit=50):
        self.dag_id = dag_id
        return [
            {
                "dag_run_id": "manual__clip001",
                "state": "running",
                "conf": {"dataset_name": "clip_001"},
            }
        ][:limit]

    def latest_run(self, dag_id, dataset_name=None):
        runs = self.runs(dag_id)
        if dataset_name and dataset_name != "clip_001":
            return None
        return runs[0]

    def task_instances(self, dag_id, run_id):
        return [
            {
                "task_id": "run_segment",
                "state": "failed",
                "try_number": 1,
                "map_index": -1,
            },
            {
                "task_id": "validate_segment",
                "state": "upstream_failed",
                "try_number": 1,
                "map_index": -1,
            },
        ]

    def run_evidence(self, dag_id, dataset_name=None):
        run = self.latest_run(dag_id, dataset_name)
        return {
            "dag_id": dag_id,
            "dataset_name": dataset_name,
            "latest_run": run,
            "task_instances": self.task_instances(dag_id, run["dag_run_id"]) if run else [],
        }

    def task_log(self, dag_id, run_id, task_id, try_number=1, map_index=-1, tail_lines=200):
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "task_id": task_id,
            "try_number": try_number,
            "map_index": map_index,
            "tail_lines": tail_lines,
            "log": "CUDA out of memory\nmock failure",
        }


def make_task(tmp_path: Path):
    task_name = "release_20260819_120000"
    tasks = tmp_path / "tasks"
    task_dir = tasks / task_name
    task_dir.mkdir(parents=True)
    config = {
        "pipeline_stages": ["precheck", "parser", "segment", ["od", "occ"]],
        "max_active_runs": 2,
        "task_type": "release",
        "priority": 10,
        "gpu_ids": "0,1",
        "gpu_stages": "segment,od,occ",
        "exclusive_gpu_stages": "segment,od",
        "gpu_stage_memory_mb": {"segment": 24000, "od": 24000, "occ": 4000},
        "datasets": [
            {
                "dataset_name": "clip_001",
                "dataset_path": "/data/clip_001",
                "timeout_min": 30,
                "image_segment": "segment:test",
            },
            {
                "dataset_name": "clip_002",
                "dataset_path": "/data/clip_002",
                "timeout_min": 30,
                "image_segment": "segment:test",
            },
        ],
    }
    (task_dir / "datasets_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    dags_dir = tmp_path / "dags"
    (dags_dir / "generated").mkdir(parents=True)
    dag_id = f"batch_pipeline_universal_{task_name}"
    (dags_dir / "generated" / f"{dag_id}.py").write_text("# generated\n", encoding="utf-8")
    return task_name, tasks, dags_dir, config


def make_facade(tmp_path: Path):
    task_name, tasks, dags_dir, config = make_task(tmp_path)
    state_dir = tmp_path / "state"
    queue_dir = state_dir / "task_queue"
    queue_dir.mkdir(parents=True)
    queue_file = queue_dir / "queue.lock"
    queue_file.write_text(
        json.dumps(
            {
                "version": 2,
                "active": {"task_name": "another_task", "priority": 20},
                "queue": [{"task_name": task_name, "priority": 10}],
            }
        ),
        encoding="utf-8",
    )

    runtime = SimulatedGPURuntime(
        state_dir / "gpu_simulator.json", fallback_to_os_processes=False
    )
    runtime.initialize(
        [
            {"id": 0, "total_memory_mb": 48000, "external_used_mb": 1000},
            {"id": 1, "total_memory_mb": 48000, "external_used_mb": 30000},
        ]
    )
    runtime.set_process_alive(333, True)
    gpu_locks = state_dir / "gpu_locks"
    gpu_locks.mkdir(parents=True)
    (gpu_locks / "gpu_0.lock").write_text(
        json.dumps(
            {
                "reservations": {
                    "res-1": {
                        "pid": 333,
                        "task_name": task_name,
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

    settings = PlatformSettings(
        platform_home=tmp_path,
        airflow_home=tmp_path / "airflow",
        dags_dir=dags_dir,
        task_config_root=tasks,
        state_dir=state_dir,
        queue_file=queue_file,
        gpu_lock_dir=gpu_locks,
        airflow_api_base="http://airflow.invalid",
        airflow_api_user="admin",
        airflow_api_password="",
        airflow_api_token="test",
        airflow_password_file=tmp_path / "passwords.json",
        airflow_bin="airflow",
    )
    queue = QueueService(queue_file)
    task_query = TaskQueryService(tasks, dags_dir, queue)
    gpu = GPUService(runtime=runtime, lock_dir=gpu_locks)
    docker = DockerService(FakeDockerGateway())
    airflow = FakeAirflowService()
    diagnosis = DiagnosisService(queue, docker, gpu, airflow)
    health = HealthService(queue, airflow, gpu, tasks)
    facade = PlatformMCPFacade(
        settings, task_query, queue, gpu, docker, airflow, diagnosis, health
    )
    return task_name, facade


def test_read_only_tool_contract_is_fixed():
    assert READ_ONLY_TOOL_NAMES == (
        "get_platform_health",
        "list_tasks",
        "get_task_detail",
        "get_queue_state",
        "get_gpu_pool",
        "inspect_task_containers",
        "get_stage_logs",
        "diagnose_task",
        "search_knowledge",
    )


def test_list_tasks_and_detail(tmp_path: Path):
    task_name, facade = make_facade(tmp_path)
    listed = facade.list_tasks()
    assert listed["count"] == 1
    assert listed["tasks"][0]["task_name"] == task_name
    assert listed["tasks"][0]["queue"]["location"] == "queued"

    detail = facade.get_task_detail(task_name)
    assert detail["dag_id"].endswith(task_name)
    assert detail["max_active_runs"] == 2
    assert detail["priority"]["priority"] == 10
    assert detail["airflow_runs"][0]["state"] == "running"


def test_queue_state_global_and_task(tmp_path: Path):
    task_name, facade = make_facade(tmp_path)
    state = facade.get_queue_state()
    assert state["active"]["task_name"] == "another_task"
    task = facade.get_queue_state(task_name)
    assert task["location"] == "queued"
    assert task["position"] == 1


def test_gpu_pool_contains_simulated_memory_and_reservations(tmp_path: Path):
    task_name, facade = make_facade(tmp_path)
    pool = facade.get_gpu_pool()
    assert len(pool["devices"]) == 2
    assert pool["devices"][0]["free_mb"] == 47000
    assert pool["reservations"][0]["task_name"] == task_name
    assert pool["reservations"][0]["exclusive"] is True


def test_container_inspection_is_dataset_scoped(tmp_path: Path):
    task_name, facade = make_facade(tmp_path)
    result = facade.inspect_task_containers(task_name, ["clip_001"])
    assert result["datasets"] == ["clip_001"]
    assert result["containers"][0]["state"] == "running"


def test_stage_logs_fetch_failed_stage_evidence(tmp_path: Path):
    task_name, facade = make_facade(tmp_path)
    result = facade.get_stage_logs(task_name, "clip_001", "segment", 50)
    assert result["run_id"] == "manual__clip001"
    assert result["logs"][0]["task_id"] == "run_segment"
    assert "out of memory" in result["logs"][0]["log"].lower()


def test_diagnose_task_aggregates_all_domains(tmp_path: Path):
    task_name, facade = make_facade(tmp_path)
    result = facade.diagnose_task(task_name, "clip_001")
    assert result["queue"]["location"] == "queued"
    assert result["airflow"]["latest_run"]["state"] == "running"
    assert result["containers"][0]["running"] is True
    assert result["gpu_reservations"][0]["stage"] == "segment"
    assert result["gpu_devices"][1]["free_mb"] == 18000
    assert result["evidence_complete"] is True


def test_health_is_structured_and_does_not_raise(tmp_path: Path):
    _, facade = make_facade(tmp_path)
    result = facade.get_platform_health()
    assert result["airflow"]["ok"] is True
    assert result["gpu"]["ok"] is True
    assert result["queue"]["ok"] is True


class RecordingAirflowGateway:
    def __init__(self):
        self.calls = []

    def _request_json(self, method, path, payload=None, use_auth=True):
        self.calls.append((method, path))
        if "/taskInstances" in path:
            return {"task_instances": [{"task_id": "run_segment", "state": "running"}], "total_entries": 1}
        if "/dagRuns" in path:
            return {"dag_runs": [{"dag_run_id": "run1", "state": "running"}], "total_entries": 1}
        return {"scheduler": {"status": "healthy"}}

    def _request_raw(self, method, path, payload=None, use_auth=True):
        self.calls.append((method, path))
        return b"line1\nline2\n", "text/plain"


def test_airflow_read_gateway_uses_airflow3_v2_routes():
    from platform_core.gateways.airflow_read import AirflowReadGateway

    gateway = RecordingAirflowGateway()
    # Bind concrete methods to the recorder so the test validates the generated paths
    gateway.list_dag_runs = AirflowReadGateway.list_dag_runs.__get__(gateway, RecordingAirflowGateway)
    gateway.list_task_instances = AirflowReadGateway.list_task_instances.__get__(gateway, RecordingAirflowGateway)
    gateway.get_task_log = AirflowReadGateway.get_task_log.__get__(gateway, RecordingAirflowGateway)
    gateway._quote = AirflowReadGateway._quote

    assert gateway.list_dag_runs("dag/a", 10)[0]["dag_run_id"] == "run1"
    assert "/api/v2/dags/dag%2Fa/dagRuns" in gateway.calls[0][1]

    gateway.list_task_instances("dag/a", "run/1", 10)
    assert "/api/v2/dags/dag%2Fa/dagRuns/run%2F1/taskInstances" in gateway.calls[1][1]

    text = gateway.get_task_log("dag/a", "run/1", "run_segment", 2)
    assert text.endswith("line2\n")
    assert "/taskInstances/run_segment/logs/2" in gateway.calls[2][1]


def test_mcp_server_missing_dependency_fails_cleanly(monkeypatch, capsys):
    import builtins
    from platform_mcp import server as server_module

    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "mcp.server":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        server_module.build_mcp_server()
    except RuntimeError as exc:
        assert "requirements-mcp.txt" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_mcp_registration_layer_registers_all_read_only_tools(tmp_path, monkeypatch):
    import sys
    import types
    from platform_mcp.server import build_mcp_server

    _, facade = make_facade(tmp_path)
    # This test exercises the enabled read-only surface; the V1.4.2 disabled
    # capability contract is covered by test_rag_agent_tool_v142.py.
    facade.knowledge_service = object()

    class FakeMCPServer:
        def __init__(self, name, instructions=""):
            self.name = name
            self.instructions = instructions
            self.tools = {}

        def tool(self):
            def register(fn):
                self.tools[fn.__name__] = fn
                return fn
            return register

        def run(self, transport="stdio"):
            self.transport = transport

    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    server_module.MCPServer = FakeMCPServer
    mcp_module.server = server_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)

    server = build_mcp_server(facade)
    assert tuple(server.tools) == READ_ONLY_TOOL_NAMES
    queue = server.tools["get_queue_state"]()
    assert queue["active"]["task_name"] == "another_task"
