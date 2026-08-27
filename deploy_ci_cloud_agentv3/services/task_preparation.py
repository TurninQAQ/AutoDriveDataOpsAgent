from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from deploy_ci_cloud_agentv3.models.artifact import PreparedArtifact
from deploy_ci_cloud_agentv3.models.common import sha256_json
from deploy_ci_cloud_agentv3.services.artifacts import ArtifactStore


class TaskPreparationService:
    """Deterministic NL-draft -> validated TaskSpec/YAML preparation.

    Only fields explicitly supplied by the model/user are accepted as draft input.
    Platform defaults and image tags are loaded from the repository-owned defaults.
    """

    def __init__(self, facade: Any, artifact_store: ArtifactStore, defaults_path: str | Path | None = None) -> None:
        self.facade = facade
        self.artifact_store = artifact_store
        self.defaults_path = Path(defaults_path or Path(__file__).resolve().parents[2] / "deploy_ci_cloud_agentv3" / "platform_backend" / "config" / "task_planning_defaults.yaml")

    def prepare(
        self,
        task_prefix: str,
        dataset_path: str,
        dataset_name: str | None = None,
        task_type: str | None = None,
        pipeline_stages: list[Any] | None = None,
        max_active_runs: int | None = None,
    ) -> PreparedArtifact:
        defaults = yaml.safe_load(self.defaults_path.read_text(encoding="utf-8"))
        task_defaults = dict(defaults.get("task_defaults") or {})
        dataset_defaults = dict(defaults.get("dataset_defaults") or {})
        image_defaults = dict(defaults.get("image_defaults") or {})

        stages = pipeline_stages if pipeline_stages is not None else task_defaults["pipeline_stages"]
        flat_stages = self._flatten_stages(stages)
        name = dataset_name or self._derive_dataset_name(dataset_path)

        dataset: dict[str, Any] = {
            "dataset_name": name,
            "dataset_path": dataset_path,
            "tier": dataset_defaults["tier"],
            "pool": dataset_defaults["pool"],
            "timeout_min": dataset_defaults["timeout_min"],
        }
        for stage in flat_stages:
            image = image_defaults.get(stage)
            if image:
                dataset[f"image_{stage}"] = image

        gpu = dict(task_defaults.get("gpu") or {})
        config: dict[str, Any] = {
            "pipeline_stages": stages,
            "max_active_runs": max_active_runs or task_defaults["max_active_runs"],
            "task_exclusive": bool(task_defaults["task_exclusive"]),
            "task_lock_wait_interval_sec": int(task_defaults["task_lock_wait_interval_sec"]),
            "preempt_grace_timeout_min": int(task_defaults["preempt_grace_timeout_min"]),
            "gpu_ids": gpu["gpu_ids"],
            "gpu_stages": ",".join(gpu["gpu_stages"]),
            "exclusive_gpu_stages": ",".join(gpu["exclusive_gpu_stages"]),
            "exclusive_gpu_idle_used_max_mb": int(gpu["exclusive_gpu_idle_used_max_mb"]),
            "gpu_stage_memory_mb": dict(gpu["gpu_stage_memory_mb"]),
            "gpu_wait_interval_sec": int(gpu["gpu_wait_interval_sec"]),
            "gpu_reservation_pending_sec": int(gpu["gpu_reservation_pending_sec"]),
            "datasets": [dataset],
        }
        effective_task_type = task_type if task_type is not None else str(task_defaults.get("task_type") or "")
        if effective_task_type:
            config["task_type"] = effective_task_type

        validated = self.facade.validate_task_spec(task_prefix, config)
        normalized = dict(validated["config"])
        yaml_text = yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False)
        digest = sha256_json({"task_prefix": task_prefix, "config": normalized})
        artifact = PreparedArtifact(
            artifact_id=f"artifact_{uuid.uuid4().hex}",
            sha256=digest,
            task_prefix=task_prefix,
            config=normalized,
            yaml_text=yaml_text,
        )
        return self.artifact_store.put(artifact)

    @staticmethod
    def _flatten_stages(stages: list[Any]) -> list[str]:
        out: list[str] = []
        for item in stages:
            if isinstance(item, list):
                out.extend(str(x) for x in item)
            else:
                out.append(str(item))
        return out

    @staticmethod
    def _derive_dataset_name(dataset_path: str) -> str:
        raw = Path(dataset_path.rstrip("/")).name or "dataset"
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
        return safe or "dataset"
