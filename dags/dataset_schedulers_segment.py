import os
import yaml
from datetime import datetime
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

LATEST_VERSION="v1.1.0  2027-07-15:(new:None base:None)"
print(LATEST_VERSION)

CONFIG_PATH = "/opt/airflow/config/datasets_config_segment.yaml"
HOST_DATA_ROOT = "/opt/airflow/data"


with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

datasets_cfg = config["datasets"]

default_args = {
    "owner": "data-team",
    "retries": 0,
}


def require_config(name, allow_empty=False):
    if name not in config:
        raise RuntimeError(f"Missing required config: {name}")
    value = config.get(name)
    if value is None:
        if allow_empty:
            return ""
        raise RuntimeError(f"Missing required config: {name}")
    if value == "" and not allow_empty:
        raise RuntimeError(f"Missing required config: {name}")
    return value


def parse_csv(value):
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace("，", ",").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Config [{name}] must be a positive integer: {value}") from None
    if parsed <= 0:
        raise RuntimeError(f"Config [{name}] must be a positive integer: {value}")
    return parsed


def parse_stage_memory_map(value):
    if not isinstance(value, dict):
        raise RuntimeError("Config [gpu_stage_memory_mb] must be a mapping")
    stage_memory = {}
    for stage, memory_mb in value.items():
        stage_name = str(stage).strip()
        if not stage_name:
            raise RuntimeError("Config [gpu_stage_memory_mb] contains an empty stage name")
        stage_memory[stage_name] = positive_int(memory_mb, f"gpu_stage_memory_mb.{stage_name}")
    if not stage_memory:
        raise RuntimeError("Config [gpu_stage_memory_mb] must not be empty")
    return stage_memory


gpu_stages = parse_csv(require_config("gpu_stages", allow_empty=True))
global_gpu_ids = ""
gpu_stage_memory_mb = {}
gpu_wait_interval_sec = None
gpu_reservation_pending_sec = None

if gpu_stages:
    global_gpu_ids = require_config("gpu_ids")
    gpu_stage_memory_mb = parse_stage_memory_map(require_config("gpu_stage_memory_mb"))
    gpu_wait_interval_sec = positive_int(
        require_config("gpu_wait_interval_sec"),
        "gpu_wait_interval_sec",
    )
    gpu_reservation_pending_sec = positive_int(
        require_config("gpu_reservation_pending_sec"),
        "gpu_reservation_pending_sec",
    )
    if not parse_csv(global_gpu_ids):
        raise RuntimeError("Config [gpu_ids] must contain at least one GPU id")
    missing_memory = [stage for stage in gpu_stages if stage not in gpu_stage_memory_mb]
    if missing_memory:
        raise RuntimeError(
            "Config [gpu_stage_memory_mb] missing memory config for stages: "
            + ",".join(missing_memory)
        )


def build_dataset_context(ds):
    ds_name = ds["dataset_name"]
    tier = ds.get("tier", "small")
    pool_map = {"large": "pool_large", "medium": "pool_medium", "small": "pool_small"}
    pool = pool_map.get(tier, "pool_small")

    custom_path = ds.get("dataset_path")
    data_path = custom_path if custom_path else os.path.join(HOST_DATA_ROOT, ds_name)

    if not os.path.isabs(data_path):
        raise ValueError(f"Dataset [{ds_name}] data_path must be absolute: {data_path}")

    return ds_name, tier, pool, data_path


def build_trigger_conf(ds):
    ds_name, tier, pool, data_path = build_dataset_context(ds)
    conf = {
        "dataset_name": ds_name,
        "dataset_path": data_path,
        "pool": pool,
        "tier": tier,
        "image_parser": ds.get("image_parser", f"batch-processor:{ds_name}:parser"),
        "image_segment": ds.get("image_segment", f"batch-processor:{ds_name}:segment"),
        "image_od": ds.get("image_od", f"batch-processor:{ds_name}:od"),
        "image_map": ds.get("image_map", f"batch-processor:{ds_name}:map"),
        "image_coloration": ds.get("image_coloration", f"batch-processor:{ds_name}:coloration"),
        "image_occ": ds.get("image_occ", f"batch-processor:{ds_name}:occ"),
        "image_qc": ds.get("image_qc", f"batch-processor:{ds_name}:qc"),
        "timeout_min": ds.get("timeout_min", 60),
        "gpu_ids": global_gpu_ids,
        "gpu_stages": ",".join(gpu_stages),
        "gpu_stage_memory_mb": gpu_stage_memory_mb,
        "gpu_wait_interval_sec": gpu_wait_interval_sec,
        "gpu_reservation_pending_sec": gpu_reservation_pending_sec,
    }
    return conf


scheduler_all_segment = DAG(
    dag_id="scheduler_all_segment",
    default_args=default_args,
    start_date=datetime(2026, 6, 26),
    schedule=None,
    catchup=False,
    tags=["flywheel_scheduler"],
    max_active_runs=1,
    is_paused_upon_creation=False,
)

with scheduler_all_segment:
    for ds in datasets_cfg:
        ds_name = ds["dataset_name"]
        TriggerDagRunOperator(
            task_id=f"trigger_{ds_name}",
            trigger_dag_id="batch_pipeline_universal_segment",
            conf=build_trigger_conf(ds),
            wait_for_completion=False,
            pool="global_dataset_pool",
        )


for ds in datasets_cfg:
    ds_name = ds["dataset_name"]

    scheduler_dag = DAG(
        dag_id=f"scheduler_{ds_name}_segment",
        default_args=default_args,
        start_date=datetime(2026, 6, 26),
        schedule=None,
        catchup=False,
        tags=["flywheel_scheduler"],
        max_active_runs=1,
        is_paused_upon_creation=False,
    )

    with scheduler_dag:
        TriggerDagRunOperator(
            task_id=f"trigger_{ds_name}",
            trigger_dag_id="batch_pipeline_universal_segment",
            conf=build_trigger_conf(ds),
            wait_for_completion=False,
            pool="global_dataset_pool",
        )
