from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .errors import TaskConfigError

TASK_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TASK_TYPE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
MAX_TASK_PREFIX_LENGTH = 32
MAX_TASK_NAME_LENGTH = 64
DEFAULT_TASK_TYPES_CONFIG = Path(
    os.environ.get(
        "AIRFLOW_TASK_TYPES_CONFIG",
        str(Path(__file__).resolve().parents[1] / "config" / "task_types.yaml"),
    )
)
DEFAULT_HOST_DATA_ROOT = Path(os.environ.get("AIRFLOW_HOST_DATA_ROOT", "/opt/airflow/data"))
DEFAULT_TASK_EXCLUSIVE = True
DEFAULT_TASK_LOCK_WAIT_INTERVAL_SEC = 10
DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN = int(os.environ.get("AIRFLOW_PREEMPT_GRACE_TIMEOUT_MIN", "60"))
DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB = int(os.environ.get("AIRFLOW_EXCLUSIVE_GPU_IDLE_USED_MAX_MB", "512"))
LOCAL_IMAGE_OPTIONAL_STAGES = {"precheck"}
DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

def warn(message):
    print(f"[WARN] {message}", flush=True)

def validate_task_name(task_name):
    if not TASK_NAME_RE.match(task_name or ""):
        raise TaskConfigError(
            "task_name must start with a lowercase letter and only contain "
            "lowercase letters, numbers and underscores: "
            f"{task_name!r}"
        )
    if len(task_name) > MAX_TASK_NAME_LENGTH:
        raise TaskConfigError(
            f"task_name must not exceed {MAX_TASK_NAME_LENGTH} characters: {task_name!r}"
        )

def build_task_name(task_prefix):
    validate_task_name(task_prefix)
    if len(task_prefix) > MAX_TASK_PREFIX_LENGTH:
        raise TaskConfigError(
            f"task prefix must not exceed {MAX_TASK_PREFIX_LENGTH} characters: {task_prefix!r}"
        )
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    task_name = f"{task_prefix}_{timestamp}"
    validate_task_name(task_name)
    return task_name

def local_time_text(timestamp=None):
    dt = datetime.fromtimestamp(time.time() if timestamp is None else timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def load_yaml(path):
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise TaskConfigError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TaskConfigError("YAML root must be a mapping")
    return yaml_path, config

def parse_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace("，", ",").split(",")
    return [str(item).strip() for item in items if str(item).strip()]

def positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise TaskConfigError(f"{name} must be a positive integer: {value}") from None
    if parsed <= 0:
        raise TaskConfigError(f"{name} must be a positive integer: {value}")
    return parsed

def priority_int(value, name):
    if isinstance(value, bool):
        raise TaskConfigError(f"{name} must be a non-negative integer: {value}")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise TaskConfigError(f"{name} must be a non-negative integer: {value}") from None
    if parsed < 0:
        raise TaskConfigError(f"{name} must be a non-negative integer: {value}")
    return parsed

def load_task_types_config(path=None):
    config_path = Path(path or DEFAULT_TASK_TYPES_CONFIG).expanduser()
    if not config_path.is_file():
        raise TaskConfigError(f"Task types config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TaskConfigError(f"Task types config root must be a mapping: {config_path}")

    if "default_priority" not in config:
        raise TaskConfigError(f"Task types config missing default_priority: {config_path}")
    default_priority = priority_int(config.get("default_priority"), "default_priority")

    raw_task_types = config.get("task_types")
    if not isinstance(raw_task_types, dict):
        raise TaskConfigError(f"Task types config missing task_types mapping: {config_path}")

    task_types = {}
    for raw_name, raw_item in raw_task_types.items():
        task_type = str(raw_name).strip()
        if not TASK_TYPE_RE.match(task_type):
            raise TaskConfigError(f"Invalid task type name in {config_path}: {raw_name!r}")
        if not isinstance(raw_item, dict):
            raise TaskConfigError(f"task_types.{task_type} must be a mapping")
        if "priority" not in raw_item:
            raise TaskConfigError(f"task_types.{task_type} missing priority")
        task_types[task_type] = {
            "priority": priority_int(raw_item.get("priority"), f"task_types.{task_type}.priority")
        }

    return {"default_priority": default_priority, "task_types": task_types}

def normalize_task_priority_config(config, task_types_config=None):
    if not isinstance(config, dict):
        raise TaskConfigError("Task config root must be a mapping")

    task_types_config = task_types_config or load_task_types_config()
    raw_task_type = config.get("task_type", "")
    task_type = "" if raw_task_type is None else str(raw_task_type).strip()
    if task_type and not TASK_TYPE_RE.match(task_type):
        raise TaskConfigError(f"task_type contains unsupported characters: {task_type!r}")
    if task_type and task_type not in task_types_config["task_types"]:
        raise TaskConfigError(f"Unknown task_type in task YAML: {task_type}")

    raw_priority = config.get("priority", None)
    if raw_priority not in (None, ""):
        priority = priority_int(raw_priority, "priority")
        priority_source = "explicit"
    elif task_type:
        task_type_config = task_types_config["task_types"][task_type]
        priority = int(task_type_config["priority"])
        priority_source = "task_type"
    else:
        priority = int(task_types_config["default_priority"])
        priority_source = "default"

    return {
        "task_type": task_type,
        "priority": priority,
        "priority_source": priority_source,
    }

def parse_bool_config(value, name):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off", ""}:
        return False
    raise TaskConfigError(f"{name} must be a boolean: {value}")

def require_config(config, name, allow_empty=False):
    if name not in config:
        raise TaskConfigError(f"Missing required config: {name}")
    value = config.get(name)
    if value is None:
        if allow_empty:
            return ""
        raise TaskConfigError(f"Missing required config: {name}")
    if value == "" and not allow_empty:
        raise TaskConfigError(f"Missing required config: {name}")
    return value

def parse_stage_memory_map(value):
    if not isinstance(value, dict):
        raise TaskConfigError("gpu_stage_memory_mb must be a mapping")
    stage_memory = {}
    for stage, memory_mb in value.items():
        stage_name = str(stage).strip()
        if not stage_name:
            raise TaskConfigError("gpu_stage_memory_mb contains an empty stage name")
        stage_memory[stage_name] = positive_int(memory_mb, f"gpu_stage_memory_mb.{stage_name}")
    if not stage_memory:
        raise TaskConfigError("gpu_stage_memory_mb must not be empty")
    return stage_memory

def validate_stage_name(stage):
    if not isinstance(stage, str):
        raise TaskConfigError(f"pipeline stage must be a string: {stage!r}")
    stage = stage.strip()
    if not stage:
        raise TaskConfigError("pipeline stage must not be empty")
    if not STAGE_NAME_RE.match(stage):
        raise TaskConfigError(
            "pipeline stage may only contain letters, numbers and underscores: "
            f"{stage!r}"
        )
    return stage

def normalize_pipeline_stages(config):
    raw_stages = require_config(config, "pipeline_stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise TaskConfigError("pipeline_stages must be a non-empty list")

    groups = []
    seen = set()
    for index, item in enumerate(raw_stages):
        raw_group = item if isinstance(item, list) else [item]
        if not raw_group:
            raise TaskConfigError(f"pipeline_stages[{index}] must not be an empty group")

        group = []
        group_seen = set()
        for raw_stage in raw_group:
            stage = validate_stage_name(raw_stage)
            if stage in group_seen:
                raise TaskConfigError(f"Duplicate stage [{stage}] in pipeline group {index}")
            if stage in seen:
                raise TaskConfigError(f"Duplicate stage [{stage}] in pipeline_stages")
            group_seen.add(stage)
            seen.add(stage)
            group.append(stage)
        groups.append(group)

    return groups

def flatten_stage_groups(stage_groups):
    return [stage for group in stage_groups for stage in group]

def script_path_for_stage(stage, scripts_dir=None):
    scripts_root = Path(scripts_dir) if scripts_dir else DEFAULT_SCRIPTS_DIR
    return scripts_root / f"run_{stage}.sh"

def script_requires_gpu(stage, scripts_dir=None):
    script_path = script_path_for_stage(stage, scripts_dir=scripts_dir)
    try:
        content = script_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = script_path.read_text(errors="ignore")
    return "GPU_IDS:?ENV MISSING" in content or "GPU_IDS?ENV MISSING" in content

def validate_stage_scripts(stages, scripts_dir=None):
    missing = [
        str(script_path_for_stage(stage, scripts_dir=scripts_dir))
        for stage in stages
        if not script_path_for_stage(stage, scripts_dir=scripts_dir).is_file()
    ]
    if missing:
        raise TaskConfigError("Missing stage scripts: " + ", ".join(missing))

def normalize_task_lock_config(config):
    return {
        "task_exclusive": parse_bool_config(
            config.get("task_exclusive", DEFAULT_TASK_EXCLUSIVE),
            "task_exclusive",
        ),
        "task_lock_wait_interval_sec": positive_int(
            config.get("task_lock_wait_interval_sec", DEFAULT_TASK_LOCK_WAIT_INTERVAL_SEC),
            "task_lock_wait_interval_sec",
        ),
    }

def normalize_preempt_config(config):
    return {
        "preempt_grace_timeout_min": priority_int(
            config.get("preempt_grace_timeout_min", DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN),
            "preempt_grace_timeout_min",
        )
    }

def unique_stage_list(stages):
    result = []
    seen = set()
    for raw_stage in stages:
        stage = validate_stage_name(str(raw_stage).strip())
        if stage in seen:
            continue
        seen.add(stage)
        result.append(stage)
    return result

def normalize_exclusive_gpu_config(config, gpu_stages, stages=None):
    gpu_stages = unique_stage_list(gpu_stages)
    stage_set = set(stages or [])
    explicit_exclusive = "exclusive_gpu_stages" in config
    if explicit_exclusive:
        exclusive_gpu_stages = unique_stage_list(parse_csv(config.get("exclusive_gpu_stages")))
    else:
        exclusive_gpu_stages = list(gpu_stages)

    missing_from_gpu = [stage for stage in exclusive_gpu_stages if stage not in set(gpu_stages)]
    if missing_from_gpu:
        raise TaskConfigError(
            "exclusive_gpu_stages contains stages not listed in gpu_stages: "
            + ",".join(missing_from_gpu)
        )

    if explicit_exclusive and stage_set:
        missing_from_pipeline = [stage for stage in exclusive_gpu_stages if stage not in stage_set]
        if missing_from_pipeline:
            raise TaskConfigError(
                "exclusive_gpu_stages contains stages not listed in pipeline_stages: "
                + ",".join(missing_from_pipeline)
            )

    idle_used_max_mb = DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB
    if exclusive_gpu_stages:
        idle_used_max_mb = priority_int(
            config.get(
                "exclusive_gpu_idle_used_max_mb",
                DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB,
            ),
            "exclusive_gpu_idle_used_max_mb",
        )

    return {
        "exclusive_gpu_stages": ",".join(exclusive_gpu_stages),
        "exclusive_gpu_idle_used_max_mb": idle_used_max_mb,
    }

def normalize_gpu_config(config, stages=None, scripts_dir=None):
    gpu_stages = parse_csv(require_config(config, "gpu_stages", allow_empty=True))
    stages = list(stages or [])
    stage_set = set(stages)

    missing_required_gpu = [
        stage
        for stage in stages
        if script_requires_gpu(stage, scripts_dir=scripts_dir) and stage not in gpu_stages
    ]
    if missing_required_gpu:
        raise TaskConfigError(
            "These pipeline stages require GPU_IDS in their run script, "
            "but are not listed in gpu_stages: "
            + ",".join(missing_required_gpu)
        )

    unused_gpu_stages = [stage for stage in gpu_stages if stage not in stage_set]
    if unused_gpu_stages:
        warn(
            "gpu_stages contains stages not used by pipeline_stages: "
            + ",".join(unused_gpu_stages)
        )

    if not gpu_stages:
        return {
            "gpu_ids": "",
            "gpu_stages": "",
            "gpu_stage_memory_mb": {},
            "gpu_wait_interval_sec": None,
            "gpu_reservation_pending_sec": None,
            **normalize_exclusive_gpu_config(config, [], stages=stages),
        }

    gpu_ids = require_config(config, "gpu_ids")
    if not parse_csv(gpu_ids):
        raise TaskConfigError("gpu_ids must contain at least one GPU id")

    stage_memory = parse_stage_memory_map(require_config(config, "gpu_stage_memory_mb"))
    missing_memory = [stage for stage in gpu_stages if stage not in stage_memory]
    if missing_memory:
        raise TaskConfigError(
            "gpu_stage_memory_mb missing memory config for stages: "
            + ",".join(missing_memory)
        )

    return {
        "gpu_ids": str(gpu_ids),
        "gpu_stages": ",".join(gpu_stages),
        "gpu_stage_memory_mb": stage_memory,
        "gpu_wait_interval_sec": positive_int(
            require_config(config, "gpu_wait_interval_sec"),
            "gpu_wait_interval_sec",
        ),
        "gpu_reservation_pending_sec": positive_int(
            require_config(config, "gpu_reservation_pending_sec"),
            "gpu_reservation_pending_sec",
        ),
        **normalize_exclusive_gpu_config(config, gpu_stages, stages=stages),
    }

def dataset_path_for(ds):
    ds_name = ds.get("dataset_name")
    custom_path = ds.get("dataset_path")
    data_path = Path(custom_path) if custom_path else DEFAULT_HOST_DATA_ROOT / ds_name
    if not data_path.is_absolute():
        raise TaskConfigError(f"Dataset [{ds_name}] dataset_path must be absolute: {data_path}")
    return str(data_path)

def validate_datasets(config, stages):
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise TaskConfigError("datasets must be a non-empty list")

    seen = set()
    for index, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            raise TaskConfigError(f"datasets[{index}] must be a mapping")
        ds_name = ds.get("dataset_name")
        if not ds_name:
            raise TaskConfigError(f"datasets[{index}] missing dataset_name")
        if ds_name in seen:
            raise TaskConfigError(f"Duplicate dataset_name: {ds_name}")
        seen.add(ds_name)
        dataset_path_for(ds)
        positive_int(ds.get("timeout_min", 60), f"datasets[{index}].timeout_min")

        for stage in stages:
            image_key = f"image_{stage}"
            if stage in LOCAL_IMAGE_OPTIONAL_STAGES:
                continue
            if not ds.get(image_key):
                raise TaskConfigError(
                    f"Dataset [{ds_name}] missing {image_key} for pipeline stage [{stage}]"
                )

    return datasets

def validate_config(config, scripts_dir=None):
    stage_groups = normalize_pipeline_stages(config)
    stages = flatten_stage_groups(stage_groups)
    validate_stage_scripts(stages, scripts_dir=scripts_dir)
    normalize_task_lock_config(config)
    normalize_task_priority_config(config)
    normalize_preempt_config(config)
    normalize_gpu_config(config, stages=stages, scripts_dir=scripts_dir)
    datasets = validate_datasets(config, stages)
    max_active_runs = positive_int(config.get("max_active_runs", 5), "max_active_runs")
    return datasets, max_active_runs, stage_groups

def pool_for_dataset(ds):
    explicit_pool = ds.get("pool")
    if explicit_pool:
        return explicit_pool

    tier = ds.get("tier", "small")
    pool_map = {"large": "pool_large", "medium": "pool_medium", "small": "pool_small"}
    return pool_map.get(tier, "pool_small")

def build_trigger_conf(task_name, config, ds, stage_groups):
    stages = flatten_stage_groups(stage_groups)
    gpu_config = normalize_gpu_config(config, stages=stages)
    ds_name = ds["dataset_name"]
    conf = {
        "task_name": task_name,
        "pipeline_stages": stage_groups,
        "dataset_name": ds_name,
        "dataset_path": dataset_path_for(ds),
        "pool": pool_for_dataset(ds),
        "tier": ds.get("tier", "small"),
        "timeout_min": ds.get("timeout_min", 60),
    }

    for stage in stages:
        if stage in LOCAL_IMAGE_OPTIONAL_STAGES:
            conf[f"image_{stage}"] = ds.get(f"image_{stage}", "local")
        else:
            conf[f"image_{stage}"] = ds[f"image_{stage}"]
    conf["image_qc"] = ds.get("image_qc", "")
    conf.update(normalize_task_lock_config(config))
    conf.update(normalize_preempt_config(config))
    conf.update(gpu_config)
    return conf

