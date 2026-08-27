from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class PreparedTaskSubmission:
    task_prefix: str
    task_name: str
    source_yaml: Path
    config: dict[str, Any]
    datasets: list[dict[str, Any]]
    max_active_runs: int
    stage_groups: list[list[str]]
    priority_config: dict[str, Any]
    dag_id: str
    dag_path: Path
    target_yaml: Path

@dataclass(frozen=True)
class PriorityUpdate:
    task_name: str
    config_file: Path
    old_priority_config: dict[str, Any]
    new_priority_config: dict[str, Any]
