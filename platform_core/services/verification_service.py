from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import normalize_task_lock_config, normalize_task_priority_config
from ..gateways.docker import DockerGateway
from ..task_store import dataset_map, load_task_config, task_paths
from .airflow_read_service import AirflowReadService
from .gpu_service import GPUService
from .queue_service import QueueService


def _container_summary(container: dict[str, Any]) -> dict[str, Any]:
    config = container.get("Config") or {}
    state = container.get("State") or {}
    return {
        "id": str(container.get("Id") or ""),
        "name": str(container.get("Name") or "").lstrip("/"),
        "image": str(config.get("Image") or ""),
        "state": str(state.get("Status") or ""),
        "running": bool(state.get("Running", False)),
        "exit_code": state.get("ExitCode"),
    }


def _reservation_summary(lock_name: str, token: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "gpu_id": lock_name.removeprefix("gpu_").removesuffix(".lock"),
        "token": token,
        "pid": item.get("pid"),
        "task_name": item.get("task_name"),
        "dataset_name": item.get("dataset_name"),
        "stage": item.get("stage"),
        "required_mb": item.get("required_mb"),
        "exclusive": bool(item.get("exclusive", False)),
    }


def _run_id(run: dict[str, Any]) -> str:
    return str(run.get("dag_run_id") or run.get("run_id") or "")


def _run_dataset(run: dict[str, Any]) -> str:
    conf = run.get("conf") or {}
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except json.JSONDecodeError:
            conf = {}
    return str(conf.get("dataset_name") or "") if isinstance(conf, dict) else ""


class ActionVerificationSnapshotService:
    """Collect deterministic post-mutation evidence without requiring task YAML."""

    def __init__(
        self,
        *,
        task_config_root: str | Path,
        dags_dir: str | Path,
        queue_service: QueueService,
        docker_gateway: DockerGateway,
        gpu_service: GPUService,
        airflow_service: AirflowReadService | None = None,
    ):
        self.task_config_root = Path(task_config_root)
        self.dags_dir = Path(dags_dir)
        self.queue_service = queue_service
        self.docker_gateway = docker_gateway
        self.gpu_service = gpu_service
        self.airflow_service = airflow_service

    @staticmethod
    def _airflow_not_found(error: str) -> bool:
        text = error.lower()
        return "http 404" in text or "not found" in text or "dagnotfound" in text

    def snapshot(self, task_name: str, datasets: list[str] | None = None, airflow_limit: int = 100) -> dict[str, Any]:
        selected = list(datasets or [])
        paths = task_paths(task_name, self.dags_dir, self.task_config_root)
        task_exists = Path(paths["config_file"]).is_file()
        priority = None
        available_datasets: list[str] = []
        task_exclusive = None
        config_error = None
        if task_exists:
            try:
                _, config = load_task_config(task_name, self.task_config_root)
                priority = normalize_task_priority_config(config).get("priority")
                available_datasets = list(dataset_map(config))
                task_exclusive = bool(normalize_task_lock_config(config).get("task_exclusive", True))
            except Exception as exc:
                config_error = str(exc)

        containers_error = None
        try:
            containers = [_container_summary(item) for item in self.docker_gateway.task_containers(task_name, selected)]
        except Exception as exc:
            containers = []
            containers_error = str(exc)

        reservations_error = None
        try:
            reservations = []
            for lock_name, token, item in self.gpu_service.reservations(cleanup_dead=True):
                if str(item.get("task_name") or "") != task_name:
                    continue
                if selected and str(item.get("dataset_name") or "") not in selected:
                    continue
                reservations.append(_reservation_summary(lock_name, token, item))
        except Exception as exc:
            reservations = []
            reservations_error = str(exc)

        dag_exists = None
        airflow_runs: list[dict[str, Any]] = []
        airflow_error = None
        if self.airflow_service is not None:
            try:
                self.airflow_service.gateway.get_dag(str(paths["dag_id"]))
                dag_exists = True
                airflow_runs = self.airflow_service.runs(str(paths["dag_id"]), limit=max(1, min(int(airflow_limit), 200)))
            except Exception as exc:
                airflow_error = str(exc)
                if self._airflow_not_found(airflow_error):
                    dag_exists = False
                else:
                    dag_exists = None

        normalized_runs = [
            {
                "run_id": _run_id(run),
                "dataset_name": _run_dataset(run),
                "state": str(run.get("state") or "").lower(),
                "conf": run.get("conf") or {},
            }
            for run in airflow_runs
        ]
        return {
            "task_name": task_name,
            "captured_at": time.time(),
            "task_exists": task_exists,
            "config_file_exists": Path(paths["config_file"]).is_file(),
            "dag_file_exists": Path(paths["dag_file"]).is_file(),
            "dag_id": str(paths["dag_id"]),
            "priority": priority,
            "task_exclusive": task_exclusive,
            "available_datasets": available_datasets,
            "selected_datasets": selected,
            "queue": self.queue_service.task_status(task_name),
            "containers": containers,
            "gpu_reservations": reservations,
            "airflow_dag_exists": dag_exists,
            "airflow_runs": normalized_runs,
            "errors": {
                key: value for key, value in {
                    "config": config_error,
                    "docker": containers_error,
                    "gpu_reservations": reservations_error,
                    "airflow": airflow_error,
                }.items() if value
            },
        }
