from __future__ import annotations

import copy
from typing import Any

from ..config import priority_int, validate_config, validate_task_name
from ..errors import TaskConfigError
from ..mutation import MutationPrecondition
from ..task_store import dataset_map, load_task_config
from .precondition_service import PreconditionService


class PlatformMutationService:
    """Validated state-changing operations with optimistic precondition checks."""

    def __init__(self, gateway, preconditions: PreconditionService, scripts_dir=None):
        self.gateway = gateway
        self.preconditions = preconditions
        self.scripts_dir = scripts_dir

    def capture_precondition(self, task_name: str = "") -> dict[str, Any]:
        return self.preconditions.capture(task_name).to_dict()

    def validate_task_spec(self, task_prefix: str, config: dict[str, Any]) -> dict[str, Any]:
        # Re-run the exact deterministic platform validator at the write boundary.
        normalized = copy.deepcopy(config or {})
        validate_task_name(task_prefix)
        datasets, max_active_runs, stage_groups = validate_config(
            normalized, scripts_dir=self.scripts_dir
        )
        return {
            "valid": True,
            "task_prefix": task_prefix,
            "dataset_count": len(datasets),
            "max_active_runs": max_active_runs,
            "stage_groups": stage_groups,
            "config": normalized,
        }

    def submit_task(
        self,
        task_prefix: str,
        config: dict[str, Any],
        precondition: dict[str, Any] | MutationPrecondition | None,
    ) -> dict[str, Any]:
        validated = self.validate_task_spec(task_prefix, config)
        self.preconditions.assert_matches(precondition)
        result = self.gateway.submit(task_prefix, validated["config"])
        return {"ok": True, "action": "submit_task", "result": result}

    def set_task_priority(
        self,
        task_name: str,
        priority: int | str,
        precondition: dict[str, Any] | MutationPrecondition | None,
    ) -> dict[str, Any]:
        validate_task_name(task_name)
        new_priority = priority_int(priority, "priority")
        load_task_config(task_name, self.preconditions.task_config_root)
        self.preconditions.assert_matches(precondition)
        result = self.gateway.set_priority(task_name, new_priority)
        return {"ok": True, "action": "set_task_priority", "result": result}

    def resume_task(
        self,
        task_name: str,
        datasets: list[str] | None,
        precondition: dict[str, Any] | MutationPrecondition | None,
    ) -> dict[str, Any]:
        validate_task_name(task_name)
        _, config = load_task_config(task_name, self.preconditions.task_config_root)
        available = dataset_map(config)
        selected = list(datasets or [])
        unknown = [name for name in selected if name not in available]
        if unknown:
            raise TaskConfigError("Unknown dataset_name: " + ",".join(unknown))
        self.preconditions.assert_matches(precondition)
        result = self.gateway.resume(task_name, selected)
        return {"ok": True, "action": "resume_task", "result": result}

    def stop_task(
        self,
        task_name: str,
        datasets: list[str] | None,
        precondition: dict[str, Any] | MutationPrecondition | None,
    ) -> dict[str, Any]:
        validate_task_name(task_name)
        _, config = load_task_config(task_name, self.preconditions.task_config_root)
        available = dataset_map(config)
        selected = list(datasets or [])
        unknown = [name for name in selected if name not in available]
        if unknown:
            raise TaskConfigError("Unknown dataset_name: " + ",".join(unknown))
        self.preconditions.assert_matches(precondition)
        result = self.gateway.stop(task_name, selected)
        return {"ok": True, "action": "stop_task", "result": result}

    def delete_task(
        self,
        task_name: str,
        precondition: dict[str, Any] | MutationPrecondition | None,
    ) -> dict[str, Any]:
        validate_task_name(task_name)
        load_task_config(task_name, self.preconditions.task_config_root)
        self.preconditions.assert_matches(precondition)
        result = self.gateway.delete(task_name)
        return {"ok": True, "action": "delete_task", "result": result}
