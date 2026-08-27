from __future__ import annotations

import shutil
from pathlib import Path

from ..config import (
    build_task_name,
    load_yaml,
    normalize_task_priority_config,
    validate_config,
)
from ..errors import TaskConfigError
from ..models import PreparedTaskSubmission, PriorityUpdate
from ..task_store import (
    install_task_config,
    render_dag,
    task_paths,
    update_task_priority_config,
)

class TaskService:
    """Application service for task configuration and generated task artifacts.

    Airflow triggering, queue activation and runtime side effects intentionally stay
    outside this service in V0.1. This keeps artifact creation deterministic and
    independently testable while preserving the existing CLI behavior.
    """

    def __init__(self, dags_dir: Path, task_config_root: Path):
        self.dags_dir = Path(dags_dir)
        self.task_config_root = Path(task_config_root)

    def prepare_submission(self, task_prefix: str, yaml_path: str | Path) -> PreparedTaskSubmission:
        task_name = build_task_name(task_prefix)
        source_yaml, config = load_yaml(yaml_path)
        datasets, max_active_runs, stage_groups = validate_config(config)
        priority_config = normalize_task_priority_config(config)
        paths = task_paths(task_name, dags_dir=self.dags_dir, task_config_root=self.task_config_root)
        existing = [str(path) for key, path in paths.items() if key != "dag_id" and Path(path).exists()]
        if existing:
            raise TaskConfigError(
                "Generated task already exists for this second; wait one second and submit again: "
                + ", ".join(existing)
            )

        target_yaml = None
        dag_path = None
        try:
            target_yaml = install_task_config(task_name, source_yaml, self.task_config_root)
            dag_id, dag_path = render_dag(task_name, max_active_runs, stage_groups, self.dags_dir)
        except Exception:
            if dag_path and Path(dag_path).exists():
                Path(dag_path).unlink()
            task_dir = Path(paths["task_dir"])
            if task_dir.exists():
                shutil.rmtree(task_dir)
            raise

        return PreparedTaskSubmission(
            task_prefix=task_prefix,
            task_name=task_name,
            source_yaml=source_yaml,
            config=config,
            datasets=datasets,
            max_active_runs=max_active_runs,
            stage_groups=stage_groups,
            priority_config=priority_config,
            dag_id=dag_id,
            dag_path=Path(dag_path),
            target_yaml=Path(target_yaml),
        )

    def update_priority(self, task_name: str, priority: int | str) -> PriorityUpdate:
        config_file, old_config, new_config = update_task_priority_config(
            task_name, priority, task_config_root=self.task_config_root
        )
        return PriorityUpdate(
            task_name=task_name,
            config_file=Path(config_file),
            old_priority_config=old_config,
            new_priority_config=new_config,
        )
