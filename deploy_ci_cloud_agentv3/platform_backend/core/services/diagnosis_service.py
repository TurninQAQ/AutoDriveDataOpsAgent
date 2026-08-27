from __future__ import annotations

from typing import Any

from .airflow_read_service import AirflowReadService
from .docker_service import DockerService
from .gpu_service import GPUService
from .queue_service import QueueService


class DiagnosisService:
    """Aggregate deterministic evidence without LLM reasoning.

    The service deliberately returns facts plus collection errors. It does not infer
    a root cause; that responsibility belongs to the Agent layer in later versions.
    """

    def __init__(
        self,
        queue_service: QueueService,
        docker_service: DockerService | None = None,
        gpu_service: GPUService | None = None,
        airflow_service: AirflowReadService | None = None,
    ):
        self.queue_service = queue_service
        self.docker_service = docker_service or DockerService()
        self.gpu_service = gpu_service or GPUService()
        self.airflow_service = airflow_service

    @staticmethod
    def _container_summary(container: dict[str, Any]) -> dict[str, Any]:
        config = container.get("Config") or {}
        state = container.get("State") or {}
        return {
            "id": str(container.get("Id") or ""),
            "name": str(container.get("Name") or "").lstrip("/"),
            "image": str(config.get("Image") or ""),
            "state": state.get("Status", ""),
            "running": bool(state.get("Running", False)),
            "exit_code": state.get("ExitCode"),
        }

    @staticmethod
    def _reservation_summary(item) -> dict[str, Any]:
        lock_name, token, reservation = item
        return {
            "lock": lock_name,
            "token": token,
            "pid": reservation.get("pid"),
            "task_name": reservation.get("task_name"),
            "dataset_name": reservation.get("dataset_name"),
            "stage": reservation.get("stage"),
            "required_mb": reservation.get("required_mb"),
            "exclusive": bool(reservation.get("exclusive", False)),
        }

    @staticmethod
    def _capture(errors: list[dict[str, str]], source: str, fn, fallback):
        try:
            return fn()
        except Exception as exc:
            errors.append({"source": source, "error": str(exc)})
            return fallback

    def inspect_task(
        self,
        task_name: str,
        config: dict,
        dataset_names: list[str],
        dag_id: str | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        queue = self._capture(
            errors, "queue", lambda: self.queue_service.task_status(task_name), {}
        )
        containers = self._capture(
            errors,
            "docker",
            lambda: self.docker_service.matching_containers(task_name, config, dataset_names),
            [],
        )
        reservations = self._capture(
            errors,
            "gpu_reservations",
            lambda: self.gpu_service.task_reservations(
                task_name, dataset_names, cleanup_dead=True
            ),
            [],
        )
        gpu_devices = []
        if getattr(self.gpu_service, "runtime", None) is not None:
            gpu_devices = self._capture(
                errors, "gpu_runtime", self.gpu_service.device_snapshot, []
            )

        airflow = None
        if self.airflow_service is not None and dag_id:
            airflow = self._capture(
                errors,
                "airflow",
                lambda: self.airflow_service.run_evidence(dag_id, dataset_name),
                {"dag_id": dag_id, "dataset_name": dataset_name, "latest_run": None, "task_instances": []},
            )

        return {
            "task_name": task_name,
            "datasets": list(dataset_names),
            "queue": queue,
            "airflow": airflow,
            "containers": [self._container_summary(item) for item in containers],
            "gpu_reservations": [self._reservation_summary(item) for item in reservations],
            "gpu_devices": gpu_devices,
            "errors": errors,
            "evidence_complete": not errors,
        }
