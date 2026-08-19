import os
import fcntl
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator

VALIDATE_IMG = "python:3.11-slim"
SCRIPT_DIR = "/opt/airflow/scripts"
DEFAULT_TIMEOUT = 60
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/home/cidi/airflow")
AIRFLOW_RUN_HOME = os.environ.get("PLATFORM_HOME", os.path.dirname(AIRFLOW_HOME.rstrip("/")))
AIRFLOW_STATE_DIR = os.environ.get("AIRFLOW_STATE_DIR", os.path.join(AIRFLOW_RUN_HOME, "state"))
GPU_LOCK_DIR = os.environ.get("AIRFLOW_GPU_LOCK_DIR", os.path.join(AIRFLOW_STATE_DIR, "gpu_locks"))
GPU_QUERY_TIMEOUT_SEC = 30
EXCLUSIVE_GPU_STAGES = {"segment"}
GPU_IDLE_USED_MAX_MB = 512


def parse_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace("，", ",").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def stage_from_script(script_name):
    parts = os.path.basename(script_name).split("_")
    if len(parts) >= 2:
        return parts[1].replace(".sh", "")
    return os.path.basename(script_name).replace(".sh", "")


def require_param(params, name, allow_empty=False):
    if name not in params:
        raise RuntimeError(f"Missing required parameter: {name}")
    value = params.get(name)
    if value is None:
        if allow_empty:
            return ""
        raise RuntimeError(f"Missing required parameter: {name}")
    if value == "" and not allow_empty:
        raise RuntimeError(f"Missing required parameter: {name}")
    return value


def positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Parameter [{name}] must be a positive integer: {value}") from None
    if parsed <= 0:
        raise RuntimeError(f"Parameter [{name}] must be a positive integer: {value}")
    return parsed


def parse_stage_memory_map(value):
    if value is None:
        raise RuntimeError("Missing required parameter: gpu_stage_memory_mb")
    if isinstance(value, dict):
        items = value.items()
    else:
        parsed_items = []
        for item in parse_csv(value):
            if ":" not in item:
                raise RuntimeError(
                    "Parameter [gpu_stage_memory_mb] must be a map or comma separated "
                    f"stage:memory list, got item: {item}"
                )
            stage, memory_mb = item.split(":", 1)
            parsed_items.append((stage.strip(), memory_mb.strip()))
        items = parsed_items

    stage_memory = {}
    for stage, memory_mb in items:
        stage_name = str(stage).strip()
        if not stage_name:
            raise RuntimeError("Parameter [gpu_stage_memory_mb] contains an empty stage name")
        stage_memory[stage_name] = positive_int(memory_mb, f"gpu_stage_memory_mb.{stage_name}")
    if not stage_memory:
        raise RuntimeError("Parameter [gpu_stage_memory_mb] must not be empty")
    return stage_memory


def validate_gpu_config(gpu_ids, gpu_stages, gpu_stage_memory_mb):
    gpu_pool = parse_csv(gpu_ids)
    if not gpu_pool:
        raise RuntimeError("Parameter [gpu_ids] must contain at least one GPU id")
    missing = [stage for stage in gpu_stages if stage not in gpu_stage_memory_mb]
    if missing:
        raise RuntimeError(
            "Parameter [gpu_stage_memory_mb] missing memory config for stages: "
            + ",".join(missing)
        )


def query_gpu_memory_mb(gpu_id):
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_id),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=GPU_QUERY_TIMEOUT_SEC)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed for GPU {gpu_id}: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"nvidia-smi returned empty memory output for GPU {gpu_id}")
    values = [int(part.strip()) for part in lines[0].split(",")]
    if len(values) != 2:
        raise RuntimeError(f"nvidia-smi returned invalid memory output for GPU {gpu_id}: {lines[0]}")
    total_mb, free_mb = values
    return total_mb, free_mb


def is_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def read_gpu_state(lock_file):
    lock_file.seek(0)
    raw = lock_file.read().strip()
    if not raw:
        return {"reservations": {}}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        state = {"reservations": {}}
    state.setdefault("reservations", {})
    return state


def write_gpu_state(lock_file, state):
    lock_file.seek(0)
    lock_file.truncate()
    json.dump(state, lock_file, sort_keys=True)
    lock_file.write("\n")
    lock_file.flush()


def prune_dead_reservations(state):
    reservations = state.get("reservations", {})
    alive = {
        token: item
        for token, item in reservations.items()
        if is_pid_alive(item.get("pid"))
    }
    state["reservations"] = alive


def pending_reserved_mb(state, now, pending_sec):
    total = 0
    for item in state.get("reservations", {}).values():
        age_sec = now - float(item.get("ts", 0))
        if age_sec < pending_sec:
            total += int(item.get("required_mb", 0))
    return total


def is_exclusive_reservation(item):
    return bool(item.get("exclusive")) or item.get("stage") in EXCLUSIVE_GPU_STAGES


def has_exclusive_reservation(state):
    return any(
        is_exclusive_reservation(item)
        for item in state.get("reservations", {}).values()
    )


def release_gpu_reservation(gpu_id, reservation_token):
    if gpu_id is None or reservation_token is None:
        return
    lock_path = os.path.join(GPU_LOCK_DIR, f"gpu_{gpu_id}.lock")
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = read_gpu_state(lock_file)
        item = state.get("reservations", {}).pop(reservation_token, {})
        write_gpu_state(lock_file, state)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    print(
        f"dbg, release_gpu={gpu_id} release_memory_mb={item.get('required_mb')}",
        flush=True,
    )


def acquire_gpu_from_pool(
    gpu_ids,
    stage,
    required_mb,
    wait_interval_sec,
    pending_sec,
    dataset_name="",
):
    gpu_pool = parse_csv(gpu_ids)
    if not gpu_pool:
        raise RuntimeError("GPU stage requires non-empty gpu_ids pool")

    required_mb = int(required_mb or 0)
    if required_mb <= 0:
        raise RuntimeError(f"GPU stage [{stage}] requires positive memory reservation")

    wait_interval_sec = positive_int(wait_interval_sec, "gpu_wait_interval_sec")
    pending_sec = positive_int(pending_sec, "gpu_reservation_pending_sec")
    os.makedirs(GPU_LOCK_DIR, exist_ok=True)

    print(
        f"dbg, gpu_pool={','.join(gpu_pool)} gpu_stage={stage} required_memory_mb={required_mb}",
        flush=True,
    )
    while True:
        for gpu_id in gpu_pool:
            lock_path = os.path.join(GPU_LOCK_DIR, f"gpu_{gpu_id}.lock")
            lock_file = open(lock_path, "a+")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                state = read_gpu_state(lock_file)
                prune_dead_reservations(state)
                now = time.time()
                reservations = state.get("reservations", {})
                exclusive_request = stage in EXCLUSIVE_GPU_STAGES

                if exclusive_request and reservations:
                    active_stages = ",".join(
                        sorted(str(item.get("stage", "unknown")) for item in reservations.values())
                    )
                    print(
                        f"dbg, trying_gpu={gpu_id} "
                        "skip_reason=segment_requires_empty_reservations "
                        f"active_stages={active_stages}",
                        flush=True,
                    )
                    write_gpu_state(lock_file, state)
                    continue

                if not exclusive_request and has_exclusive_reservation(state):
                    print(
                        f"dbg, trying_gpu={gpu_id} "
                        "skip_reason=segment_exclusive_active",
                        flush=True,
                    )
                    write_gpu_state(lock_file, state)
                    continue

                pending_mb = pending_reserved_mb(state, now, pending_sec)
                try:
                    total_mb, free_mb = query_gpu_memory_mb(gpu_id)
                except Exception as exc:
                    print(
                        f"dbg, trying_gpu={gpu_id} skip_reason=query_failed error={exc}",
                        flush=True,
                    )
                    write_gpu_state(lock_file, state)
                    continue
                used_mb = total_mb - free_mb
                if exclusive_request and used_mb > GPU_IDLE_USED_MAX_MB:
                    print(
                        f"dbg, trying_gpu={gpu_id} "
                        "skip_reason=segment_requires_idle_gpu "
                        f"used_mb={used_mb} idle_used_max_mb={GPU_IDLE_USED_MAX_MB}",
                        flush=True,
                    )
                    write_gpu_state(lock_file, state)
                    continue
                effective_free_mb = free_mb - pending_mb
                print(
                    "dbg, trying_gpu={} total_mb={} free_mb={} pending_reserved_mb={} "
                    "effective_free_mb={} required_memory_mb={}".format(
                        gpu_id,
                        total_mb,
                        free_mb,
                        pending_mb,
                        effective_free_mb,
                        required_mb,
                    ),
                    flush=True,
                )

                if effective_free_mb < required_mb:
                    print(f"dbg, trying_gpu={gpu_id} skip_reason=insufficient_memory", flush=True)
                    write_gpu_state(lock_file, state)
                    continue

                token = f"{os.getpid()}-{time.time_ns()}"
                state["reservations"][token] = {
                    "pid": os.getpid(),
                    "stage": stage,
                    "exclusive": exclusive_request,
                    "dataset_name": dataset_name,
                    "required_mb": required_mb,
                    "ts": now,
                }
                write_gpu_state(lock_file, state)
                print(
                    f"dbg, assigned_gpu={gpu_id} stage={stage} "
                    f"mode={'exclusive' if exclusive_request else 'shared'} "
                    f"reserved_memory_mb={required_mb} token={token}",
                    flush=True,
                )
                return gpu_id, token
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

        print(
            f"dbg, gpu_wait interval_sec={wait_interval_sec} reason=no_available_gpu",
            flush=True,
        )
        time.sleep(wait_interval_sec)


def run_shell_script(script_name, **context):
    params = context.get("params", {})
    dataset_name = params.get("dataset_name")
    dataset_path = params.get("dataset_path")
    gpu_stages = parse_csv(require_param(params, "gpu_stages", allow_empty=True))
    gpu_ids = ""
    gpu_stage_memory_mb = {}
    gpu_wait_interval_sec = None
    gpu_reservation_pending_sec = None

    if gpu_stages:
        gpu_ids = require_param(params, "gpu_ids")
        gpu_stage_memory_mb = parse_stage_memory_map(require_param(params, "gpu_stage_memory_mb"))
        gpu_wait_interval_sec = positive_int(
            require_param(params, "gpu_wait_interval_sec"),
            "gpu_wait_interval_sec",
        )
        gpu_reservation_pending_sec = positive_int(
            require_param(params, "gpu_reservation_pending_sec"),
            "gpu_reservation_pending_sec",
        )
        validate_gpu_config(gpu_ids, gpu_stages, gpu_stage_memory_mb)

    if not dataset_name or not dataset_path:
        raise RuntimeError("Missing required parameters: dataset_name or dataset_path")

    dataset_dir = os.path.join(dataset_path, dataset_name)
    image_tag_key = f"image_{script_name.split('_')[1].replace('.sh', '')}"
    image_tag = params.get(image_tag_key, "python:3.11-slim")

    print("dbg, dataset_name:", dataset_name)
    print("dbg, dataset_path:", dataset_path)
    print("dbg, dataset_dir:", dataset_dir)
    print("dbg, image_tag:", image_tag)
    print("dbg, gpu_ids:", gpu_ids)
    stage = stage_from_script(script_name)
    print("dbg, gpu_stage:", stage)
    print("dbg, gpu_stages:", ",".join(gpu_stages))
    print("dbg, gpu_stage_memory_mb:", gpu_stage_memory_mb)

    cmd = f"bash {SCRIPT_DIR}/{script_name}"
    env = os.environ.copy()
    env.update({
        "DATASET_NAME": dataset_name,
        "DATASET_PATH": dataset_path,
        "DATA_DIR": dataset_dir,
        "IMAGE_TAG": image_tag,
    })

    assigned_gpu = None
    reservation_token = None
    try:
        if stage in gpu_stages:
            required_mb = gpu_stage_memory_mb[stage]
            assigned_gpu, reservation_token = acquire_gpu_from_pool(
                gpu_ids,
                stage=stage,
                required_mb=required_mb,
                wait_interval_sec=gpu_wait_interval_sec,
                pending_sec=gpu_reservation_pending_sec,
                dataset_name=dataset_name,
            )
            env["GPU_IDS"] = assigned_gpu
        result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    finally:
        release_gpu_reservation(assigned_gpu, reservation_token)

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {result.stderr}")
    return result.stdout


def run_validate(task_suffix, **context):
    params = context.get("params", {})
    dataset_name = params.get("dataset_name")
    dataset_path = params.get("dataset_path")

    print("qzc:", dataset_name, dataset_path)
    
    if not dataset_name or not dataset_path:
        raise RuntimeError("Missing required parameters: dataset_name or dataset_path")
    
    dataset_dir = os.path.join(dataset_path, dataset_name)
    ds = resolve_validation_min_date(context)
    
    cmd = (
        f"docker run --rm "
        f"-v {dataset_path}:/data:ro "
        f"-v {SCRIPT_DIR}/validate_json.py:/v.py:ro "
        f"{VALIDATE_IMG} python /v.py "
        f"--root-dir /data "
        f"--dataset {dataset_name} "
        f"--task-suffix {task_suffix} "
        f"--min-date {ds}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Validation failed: {result.stderr}")
    return result.stdout


def resolve_validation_min_date(context):
    for key in ("ds", "logical_date", "data_interval_start", "ts"):
        value = context.get(key)
        if value:
            return format_date(value)

    dag_run = context.get("dag_run")
    for attr in ("logical_date", "run_after", "start_date"):
        value = getattr(dag_run, attr, None) if dag_run is not None else None
        if value:
            return format_date(value)

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def format_date(value):
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def cleanup_dataset(**context):
    params = context.get("params", {})
    dataset_name = params.get("dataset_name")
    dataset_path = params.get("dataset_path")
    
    if not dataset_name or not dataset_path:
        raise RuntimeError("Missing required parameters: dataset_name or dataset_path")
    
    dataset_dir = os.path.join(dataset_path, dataset_name)
    
    cmd = f"bash {SCRIPT_DIR}/cleanup_dataset.sh"
    env = os.environ.copy()
    env.update({
        "DATASET_NAME": dataset_name,
        "DATASET_PATH": dataset_path,
        "DATA_DIR": dataset_dir,
    })
    result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Cleanup failed: {result.stderr}")
    return result.stdout


with DAG(
    dag_id="batch_pipeline_universal_segment",
    schedule=None,
    catchup=False,
    max_active_runs=4,
    params={
        "dataset_name": Param(type="string", description="Target dataset name"),
        "dataset_path": Param(type="string", description="Dataset data path"),
        "pool": Param(type="string", default="pool_small", description="Pool name"),
        "tier": Param(type="string", default="small", description="Pool tier"),
        "image_parser": Param(type="string", description="Image for stage PARSER"),
        "image_segment": Param(type="string", description="Image for stage SEGMENT"),
        "image_od": Param(type="string", description="Image for stage OD"),
        "image_occ": Param(type="string", description="Image for stage OCC"),
        "image_map": Param(type="string", description="Image for stage MAP"),
        "image_coloration": Param(type="string", description="Image for stage COLORATION"),
        "image_qc": Param(type="string", description="Image for stage QC"),
        "timeout_min": Param(type="integer", default=60, description="Timeout in minutes"),
        "gpu_ids": Param(type="string", description="GPU pool"),
        "gpu_stages": Param(type="string", description="Stages requiring GPU"),
        "gpu_stage_memory_mb": Param(
            type="object",
            description="GPU memory reservation per stage in MiB",
        ),
        "gpu_wait_interval_sec": Param(
            type=["null", "integer"],
            description="GPU wait retry interval",
        ),
        "gpu_reservation_pending_sec": Param(
            type=["null", "integer"],
            description="Reservation time counted against free memory before GPU allocation is visible",
        ),
    },
    tags=["flywheel-batch"],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
) as dag:
    run_segment = PythonOperator(
        task_id="run_segment",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_segment_segment.sh"},
        pool="default_pool",
        execution_timeout=timedelta(minutes=DEFAULT_TIMEOUT),
    )

    run_od = PythonOperator(
        task_id="run_od",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_od.sh"},
        pool="default_pool",
        execution_timeout=timedelta(minutes=DEFAULT_TIMEOUT),
    )

    run_occ = PythonOperator(
        task_id="run_occ",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_occ.sh"},
        pool="default_pool",
        execution_timeout=timedelta(minutes=DEFAULT_TIMEOUT),
    )

    run_segment >> run_od >> run_occ
