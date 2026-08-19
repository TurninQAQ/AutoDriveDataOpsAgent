from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import normalize_pipeline_stages, normalize_task_priority_config
from ..errors import TaskConfigError
from ..task_store import dataset_map, load_task_config, task_paths
from .queue_service import QueueService


class TaskQueryService:
    """Read-only task catalog used by MCP/Agent layers."""

    def __init__(
        self,
        task_config_root: str | Path,
        dags_dir: str | Path,
        queue_service: QueueService | None = None,
    ):
        self.task_config_root = Path(task_config_root)
        self.dags_dir = Path(dags_dir)
        self.queue_service = queue_service

    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        if not self.task_config_root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for task_dir in sorted(self.task_config_root.iterdir(), key=lambda p: p.name, reverse=True):
            if not task_dir.is_dir() or not (task_dir / "datasets_config.yaml").is_file():
                continue
            task_name = task_dir.name
            try:
                _, config = load_task_config(task_name, self.task_config_root)
                priority = normalize_task_priority_config(config)
                datasets = list(dataset_map(config))
                stage_groups = normalize_pipeline_stages(config)
                queue = self.queue_service.task_status(task_name) if self.queue_service else None
                paths = task_paths(task_name, self.dags_dir, self.task_config_root)
                items.append(
                    {
                        "task_name": task_name,
                        "dag_id": paths["dag_id"],
                        "priority": priority.get("priority"),
                        "task_type": priority.get("task_type"),
                        "dataset_count": len(datasets),
                        "datasets": datasets,
                        "pipeline_stages": stage_groups,
                        "queue": queue,
                    }
                )
            except Exception as exc:
                items.append({"task_name": task_name, "error": str(exc)})
            if len(items) >= limit:
                break
        return items

    def get_task_detail(self, task_name: str) -> dict[str, Any]:
        config_file, config = load_task_config(task_name, self.task_config_root)
        paths = task_paths(task_name, self.dags_dir, self.task_config_root)
        priority = normalize_task_priority_config(config)
        stage_groups = normalize_pipeline_stages(config)
        datasets = dataset_map(config)
        return {
            "task_name": task_name,
            "dag_id": paths["dag_id"],
            "config_file": str(config_file),
            "dag_file": str(paths["dag_file"]),
            "dag_file_exists": Path(paths["dag_file"]).is_file(),
            "priority": priority,
            "max_active_runs": int(config.get("max_active_runs", 1)),
            "pipeline_stages": stage_groups,
            "datasets": [
                {
                    "dataset_name": name,
                    "data_dir": item.get("data_dir") or item.get("dataset_path"),
                    "config": item,
                }
                for name, item in datasets.items()
            ],
            "gpu_ids": config.get("gpu_ids"),
            "gpu_stages": config.get("gpu_stages"),
            "exclusive_gpu_stages": config.get("exclusive_gpu_stages"),
            "gpu_stage_memory_mb": config.get("gpu_stage_memory_mb") or {},
            "queue": self.queue_service.task_status(task_name) if self.queue_service else None,
        }

    def dataset_names(self, task_name: str) -> list[str]:
        _, config = load_task_config(task_name, self.task_config_root)
        return list(dataset_map(config))

    def load_config(self, task_name: str) -> tuple[Path, dict[str, Any]]:
        try:
            return load_task_config(task_name, self.task_config_root)
        except TaskConfigError:
            raise
