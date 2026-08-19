from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .airflow_read_service import AirflowReadService
from .gpu_service import GPUService
from .queue_service import QueueService


class HealthService:
    def __init__(
        self,
        queue_service: QueueService,
        airflow_service: AirflowReadService | None = None,
        gpu_service: GPUService | None = None,
        task_config_root: str | Path | None = None,
    ):
        self.queue_service = queue_service
        self.airflow_service = airflow_service
        self.gpu_service = gpu_service
        self.task_config_root = Path(task_config_root) if task_config_root else None

    @staticmethod
    def _capture(fn):
        try:
            return {"ok": True, "data": fn()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "queue": self._capture(self.queue_service.snapshot),
            "docker": {
                "ok": shutil.which("docker") is not None,
                "binary": shutil.which("docker"),
            },
        }
        if self.task_config_root is not None:
            result["task_config_root"] = {
                "ok": self.task_config_root.is_dir(),
                "path": str(self.task_config_root),
            }
        if self.airflow_service is not None:
            result["airflow"] = self._capture(self.airflow_service.health)
        if self.gpu_service is not None and self.gpu_service.runtime is not None:
            result["gpu"] = self._capture(self.gpu_service.device_snapshot)
        components = [item for item in result.values() if isinstance(item, dict) and "ok" in item]
        result["ok"] = all(bool(item.get("ok")) for item in components)
        return result
