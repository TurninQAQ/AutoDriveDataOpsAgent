from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import yaml

from platform_core.errors import TaskConfigError


DEFAULT_PLANNING_CONFIG = Path(
    os.environ.get("PLATFORM_TASK_PLANNING_DEFAULTS")
    or str(Path(__file__).resolve().parents[1] / "config" / "task_planning_defaults.yaml")
)


class TaskPlanningDefaults:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_PLANNING_CONFIG).expanduser().resolve()

    def load(self) -> dict:
        if not self.path.is_file():
            raise TaskConfigError(f"Task planning defaults not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict):
            raise TaskConfigError(f"Task planning defaults root must be a mapping: {self.path}")
        if not isinstance(payload.get("task_defaults"), dict):
            raise TaskConfigError("task_planning_defaults.yaml missing task_defaults mapping")
        if not isinstance(payload.get("dataset_defaults"), dict):
            raise TaskConfigError("task_planning_defaults.yaml missing dataset_defaults mapping")
        if not isinstance(payload.get("image_defaults"), dict):
            raise TaskConfigError("task_planning_defaults.yaml missing image_defaults mapping")
        return deepcopy(payload)
