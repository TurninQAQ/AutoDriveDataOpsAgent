from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml

from .config import (
    flatten_stage_groups,
    normalize_pipeline_stages,
    normalize_task_priority_config,
    priority_int,
    validate_task_name,
)
from .errors import TaskConfigError

DEFAULT_DAGS_DIR = Path(os.environ.get("AIRFLOW_DAGS_DIR", "/home/cidi/airflow/dags/data_center"))
DEFAULT_TASK_CONFIG_ROOT = Path(os.environ.get("AIRFLOW_TASK_CONFIG_ROOT", "/opt/airflow/config/tasks"))

def repo_root_from_script():
    return Path(__file__).resolve().parents[1]

def template_candidates(dags_dir):
    repo_root = repo_root_from_script()
    return [
        dags_dir / "templates" / "batch_pipeline_universal_template.py",
        repo_root / "dags" / "templates" / "batch_pipeline_universal_template.py",
    ]

def find_template(dags_dir):
    for candidate in template_candidates(dags_dir):
        if candidate.is_file():
            return candidate
    raise TaskConfigError(
        "Template not found. Expected one of: "
        + ", ".join(str(path) for path in template_candidates(dags_dir))
    )

def render_dag(task_name, max_active_runs, stage_groups, dags_dir):
    dag_id = f"batch_pipeline_universal_{task_name}"
    generated_dir = dags_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    template_path = find_template(dags_dir)
    rendered = template_path.read_text(encoding="utf-8")
    rendered = rendered.replace("__DAG_ID__", dag_id)
    rendered = rendered.replace("__TASK_NAME__", task_name)
    rendered = rendered.replace("__MAX_ACTIVE_RUNS__", str(max_active_runs))
    rendered = rendered.replace(
        "__PIPELINE_STAGE_GROUPS__",
        json.dumps(stage_groups, ensure_ascii=False),
    )

    dag_path = generated_dir / f"{dag_id}.py"
    if dag_path.exists():
        raise TaskConfigError(f"Generated DAG already exists: {dag_path}")
    temp_path = generated_dir / f".{dag_path.name}.{os.getpid()}.tmp"
    temp_path.write_text(rendered, encoding="utf-8")
    os.replace(temp_path, dag_path)
    return dag_id, dag_path

def install_task_config(task_name, source_yaml, task_config_root):
    task_dir = task_config_root / task_name
    task_dir.mkdir(parents=True, exist_ok=False)
    target_yaml = task_dir / "datasets_config.yaml"
    shutil.copy2(source_yaml, target_yaml)
    return target_yaml

def task_paths(task_name, dags_dir=None, task_config_root=None):
    dags_root = Path(dags_dir or DEFAULT_DAGS_DIR)
    config_root = Path(task_config_root or DEFAULT_TASK_CONFIG_ROOT)
    dag_id = f"batch_pipeline_universal_{task_name}"
    return {
        "dag_id": dag_id,
        "dag_file": dags_root / "generated" / f"{dag_id}.py",
        "task_dir": config_root / task_name,
        "config_file": config_root / task_name / "datasets_config.yaml",
    }

def load_task_config(task_name, task_config_root=None):
    paths = task_paths(task_name, task_config_root=task_config_root)
    config_file = paths["config_file"]
    if not config_file.is_file():
        raise TaskConfigError(f"Task config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TaskConfigError(f"Task config root must be a mapping: {config_file}")
    return config_file, config

def save_task_config(config_file, config):
    with Path(config_file).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

def update_task_priority_config(task_name, priority, task_config_root=None):
    validate_task_name(task_name)
    new_priority = priority_int(priority, "priority")
    config_file, config = load_task_config(task_name, task_config_root=task_config_root)
    old_priority_config = normalize_task_priority_config(config)
    config["priority"] = new_priority
    new_priority_config = normalize_task_priority_config(config)
    save_task_config(config_file, config)
    return config_file, old_priority_config, new_priority_config

def dataset_map(config):
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise TaskConfigError("Task config has no datasets")
    result = {}
    for ds in datasets:
        if isinstance(ds, dict) and ds.get("dataset_name"):
            result[ds["dataset_name"]] = ds
    return result

def selected_dataset_names(config, requested):
    ds_map = dataset_map(config)
    if not requested:
        return list(ds_map.keys())

    missing = [name for name in requested if name not in ds_map]
    if missing:
        raise TaskConfigError("Unknown dataset_name in task config: " + ",".join(missing))
    return list(requested)

def stages_from_config(config):
    return flatten_stage_groups(normalize_pipeline_stages(config))

def image_set_for_datasets(config, dataset_names):
    stages = stages_from_config(config)
    ds_map = dataset_map(config)
    images = set()
    for dataset_name in dataset_names:
        ds = ds_map[dataset_name]
        for stage in stages:
            image = ds.get(f"image_{stage}")
            if image:
                images.add(str(image))
    return images

