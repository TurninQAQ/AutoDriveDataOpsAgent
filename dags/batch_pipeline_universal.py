import os
import fcntl
import json
import re
import signal
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk.exceptions import AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule

SCRIPT_DIR = os.environ.get("AIRFLOW_SCRIPTS_DIR", "/opt/airflow/scripts")
AIRFLOW_BIN = os.environ.get("AIRFLOW_BIN", "/home/cidi/miniforge3/envs/airflow/bin/airflow")
AIRFLOW_PYTHON = os.environ.get("AIRFLOW_PYTHON", "python3")
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/home/cidi/airflow")
AIRFLOW_RUN_HOME = os.environ.get("PLATFORM_HOME", os.path.dirname(AIRFLOW_HOME.rstrip("/")))
AIRFLOW_STATE_DIR = os.environ.get("AIRFLOW_STATE_DIR", os.path.join(AIRFLOW_RUN_HOME, "state"))
GPU_LOCK_DIR = os.environ.get("AIRFLOW_GPU_LOCK_DIR", os.path.join(AIRFLOW_STATE_DIR, "gpu_locks"))
GPU_QUERY_TIMEOUT_SEC = 30

# V0.2: keep GPU scheduling rules shared between real NVIDIA hosts and the
# local simulator. platform_core is deployed next to AIRFLOW_SCRIPTS_DIR.
_PLATFORM_CORE_DIR = os.environ.get(
    "AIRFLOW_PLATFORM_CORE_DIR",
    os.path.join(os.path.dirname(SCRIPT_DIR.rstrip("/")), "platform_core"),
)
_PLATFORM_CORE_CANDIDATE_PARENTS = [
    os.environ.get("AIRFLOW_PLATFORM_CORE_PARENT", ""),
    os.path.dirname(_PLATFORM_CORE_DIR.rstrip("/")),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
]
for _candidate in _PLATFORM_CORE_CANDIDATE_PARENTS:
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from platform_core.gateways.gpu_runtime import create_gpu_runtime_from_env
from platform_core.services.gpu_allocator import GPUAllocator

_GPU_RUNTIME_INSTANCE = None
_GPU_ALLOCATOR_INSTANCE = None

def platform_gpu_runtime():
    global _GPU_RUNTIME_INSTANCE
    if _GPU_RUNTIME_INSTANCE is None:
        _GPU_RUNTIME_INSTANCE = create_gpu_runtime_from_env()
    return _GPU_RUNTIME_INSTANCE

def platform_gpu_allocator():
    global _GPU_ALLOCATOR_INSTANCE
    if _GPU_ALLOCATOR_INSTANCE is None:
        _GPU_ALLOCATOR_INSTANCE = GPUAllocator(
            platform_gpu_runtime(),
            GPU_LOCK_DIR,
            logger=lambda message: print(f"dbg, {message}", flush=True),
        )
    return _GPU_ALLOCATOR_INSTANCE

CONTAINER_STOP_TIMEOUT_SEC = 10
CONTAINER_CHECK_INTERVAL_SEC = 2
STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB = int(
    os.environ.get("AIRFLOW_EXCLUSIVE_GPU_IDLE_USED_MAX_MB", "512")
)
TASK_LOCK_DIR = os.environ.get("AIRFLOW_TASK_LOCK_DIR", os.path.join(AIRFLOW_STATE_DIR, "task_locks"))
TASK_LOCK_FILE = os.path.join(TASK_LOCK_DIR, "active_task.lock")
TASK_LOCK_STALE_AFTER_SEC = 30 * 60
TASK_QUEUE_DIR = os.environ.get("AIRFLOW_TASK_QUEUE_DIR", os.path.join(AIRFLOW_STATE_DIR, "task_queue"))
TASK_QUEUE_FILE = os.path.join(TASK_QUEUE_DIR, "queue.lock")
TASK_QUEUE_SCHEMA_VERSION = 2
DEFAULT_TASK_PRIORITY = 100
DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN = int(os.environ.get("AIRFLOW_PREEMPT_GRACE_TIMEOUT_MIN", "60"))
DAGRUN_ACTIVE_STATES = {"queued", "running", "scheduled"}
PLATFORM_RECOVERY_CONF_KEY = "_platform_recovery"
PLATFORM_RECOVERY_REASON_PREEMPTED = "preempted"
PLATFORM_RESUME_FROM_STAGE_KEY = "_platform_resume_from_stage"
PLATFORM_PREEMPTED_BY_KEY = "_platform_preempted_by"
PLATFORM_ORIGINAL_RUN_ID_KEY = "_platform_original_run_id"
BLOCKED_ORIGINAL_RUN_IDS_KEY = "blocked_original_run_ids"


class TaskPreempted(AirflowSkipException):
    """Task run was intentionally interrupted by platform priority scheduling."""

def parse_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace("，", ",").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def validate_stage_name(stage, param_name):
    stage = str(stage).strip()
    if not stage or not STAGE_NAME_RE.match(stage):
        raise RuntimeError(f"Parameter [{param_name}] contains invalid stage name: {stage!r}")
    return stage


def unique_stage_list(stages, param_name):
    result = []
    seen = set()
    for raw_stage in stages:
        stage = validate_stage_name(raw_stage, param_name)
        if stage in seen:
            continue
        seen.add(stage)
        result.append(stage)
    return result


def flatten_stage_groups(stage_groups):
    result = []
    for group in stage_groups or []:
        raw_group = group if isinstance(group, (list, tuple)) else [group]
        result.extend(unique_stage_list(raw_group, "pipeline_stages"))
    return result


def context_dag_run_conf(context):
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", {}) if dag_run is not None else {}
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except json.JSONDecodeError:
            conf = {}
    return conf if isinstance(conf, dict) else {}


def stage_names_from_conf_params(conf=None, params=None):
    conf = conf if isinstance(conf, dict) else {}
    params = params if isinstance(params, dict) else {}
    return flatten_stage_groups(
        conf.get("pipeline_stages") or params.get("pipeline_stages") or []
    )


def stage_names_from_context(context):
    return stage_names_from_conf_params(
        context_dag_run_conf(context),
        context.get("params", {}),
    )


def next_pipeline_stage(current_stage, stage_names):
    try:
        index = stage_names.index(current_stage)
    except ValueError:
        return None
    if index + 1 >= len(stage_names):
        return None
    return stage_names[index + 1]


def stage_before_resume_from_conf_params(stage, conf=None, params=None):
    conf = conf if isinstance(conf, dict) else {}
    resume_from_stage = conf.get(PLATFORM_RESUME_FROM_STAGE_KEY)
    if not resume_from_stage:
        return False
    stage_names = stage_names_from_conf_params(conf, params)
    try:
        return stage_names.index(stage) < stage_names.index(str(resume_from_stage))
    except ValueError:
        return False


def stage_before_resume(stage, context):
    return stage_before_resume_from_conf_params(
        stage,
        context_dag_run_conf(context),
        context.get("params", {}),
    )


def stage_task_id_stage(task_id):
    task_id = str(task_id or "")
    for prefix in ("run_", "validate_"):
        if task_id.startswith(prefix):
            stage = task_id[len(prefix):]
            return validate_stage_name(stage, "task_id") if stage else ""
    return ""


def unexpected_skipped_stage_tasks(task_instances, context=None, conf=None, params=None):
    if context is not None:
        conf = context_dag_run_conf(context)
        params = context.get("params", {})
    skipped_tasks = []
    for row in task_instances or []:
        task_id = row.get("task_id")
        state_value = airflow_state_text(row.get("state")).lower()
        if state_value != "skipped":
            continue
        stage = stage_task_id_stage(task_id)
        if not stage:
            continue
        if stage_before_resume_from_conf_params(stage, conf, params):
            continue
        skipped_tasks.append(str(task_id))
    return skipped_tasks


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


def non_negative_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Parameter [{name}] must be a non-negative integer: {value}") from None
    if parsed < 0:
        raise RuntimeError(f"Parameter [{name}] must be a non-negative integer: {value}")
    return parsed


def parse_bool(value, name):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off", ""}:
        return False
    raise RuntimeError(f"Parameter [{name}] must be a boolean: {value}")


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


def validate_gpu_config(
    gpu_ids,
    gpu_stages,
    gpu_stage_memory_mb,
    exclusive_gpu_stages=None,
    pipeline_stages=None,
    exclusive_gpu_stages_explicit=False,
):
    gpu_pool = parse_csv(gpu_ids)
    if not gpu_pool:
        raise RuntimeError("Parameter [gpu_ids] must contain at least one GPU id")
    missing = [stage for stage in gpu_stages if stage not in gpu_stage_memory_mb]
    if missing:
        raise RuntimeError(
            "Parameter [gpu_stage_memory_mb] missing memory config for stages: "
            + ",".join(missing)
        )
    exclusive_gpu_stages = exclusive_gpu_stages or []
    missing_exclusive = [stage for stage in exclusive_gpu_stages if stage not in set(gpu_stages)]
    if missing_exclusive:
        raise RuntimeError(
            "Parameter [exclusive_gpu_stages] contains stages not listed in gpu_stages: "
            + ",".join(missing_exclusive)
        )
    if exclusive_gpu_stages_explicit and pipeline_stages:
        missing_pipeline = [
            stage for stage in exclusive_gpu_stages if stage not in set(pipeline_stages)
        ]
        if missing_pipeline:
            raise RuntimeError(
                "Parameter [exclusive_gpu_stages] contains stages not listed in pipeline_stages: "
                + ",".join(missing_pipeline)
            )


def query_gpu_memory_mb(gpu_id):
    memory = platform_gpu_runtime().get_memory_info(str(gpu_id))
    return memory.total_mb, memory.free_mb


def is_pid_alive(pid):
    return platform_gpu_runtime().process_alive(pid)

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


def read_task_lock_state(lock_file):
    lock_file.seek(0)
    raw = lock_file.read().strip()
    if not raw:
        return {}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(state, dict):
        return {}
    return state


def write_task_lock_state(lock_file, state):
    lock_file.seek(0)
    lock_file.truncate()
    if state:
        json.dump(state, lock_file, sort_keys=True)
        lock_file.write("\n")
    lock_file.flush()


def task_lock_runs(state):
    raw_runs = state.get("active_runs") or []
    if isinstance(raw_runs, dict):
        raw_runs = list(raw_runs.values())
    runs = []
    if isinstance(raw_runs, list):
        for item in raw_runs:
            if isinstance(item, dict):
                runs.append(dict(item))

    if not runs:
        for raw_pid in list(state.get("active_pids") or []):
            runs.append({"pid": raw_pid})
        if state.get("pid") is not None:
            runs.append({"pid": state.get("pid")})

    deduped = []
    seen = set()
    for run in runs:
        key = str(run.get("pid") or run.get("script_pid") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(run)
    return deduped


def task_lock_run_alive(run):
    return is_pid_alive(run.get("pid")) or is_pid_alive(run.get("script_pid"))


def task_lock_alive_runs(state):
    return [run for run in task_lock_runs(state) if task_lock_run_alive(run)]


def task_lock_alive_pids(state):
    raw_pids = []
    for run in task_lock_runs(state):
        raw_pids.extend([run.get("pid"), run.get("script_pid")])

    alive = []
    seen = set()
    for raw_pid in raw_pids:
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid in seen:
            continue
        seen.add(pid)
        if is_pid_alive(pid):
            alive.append(pid)
    return alive


def task_lock_is_reusable(state, current_task_name):
    locked_task_name = state.get("task_name")
    if not locked_task_name:
        return True
    if locked_task_name == current_task_name:
        return True

    alive_runs = task_lock_alive_runs(state)
    alive_pids = task_lock_alive_pids({"active_runs": alive_runs})
    state["active_runs"] = alive_runs
    state["active_pids"] = alive_pids
    if alive_pids:
        return False

    locked_ts = float(state.get("ts") or 0)
    age_sec = time.time() - locked_ts if locked_ts else TASK_LOCK_STALE_AFTER_SEC
    if age_sec >= TASK_LOCK_STALE_AFTER_SEC:
        print(
            f"dbg, task_lock_stale active_task={locked_task_name} "
            f"active_pids=none age_sec={int(age_sec)}",
            flush=True,
        )
        return True

    return False


def release_task_lock(task_name, dag_id="", reason=""):
    if not os.path.exists(TASK_LOCK_FILE):
        print(f"dbg, task_lock_release_skip reason=no_lock task_name={task_name}", flush=True)
        return 0

    with open(TASK_LOCK_FILE, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = read_task_lock_state(lock_file)
        locked_task_name = state.get("task_name")
        locked_dag_id = state.get("dag_id")
        if locked_task_name != task_name:
            print(
                f"dbg, task_lock_release_skip reason=owner_mismatch "
                f"task_name={task_name} active_task={locked_task_name}",
                flush=True,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return 0
        if dag_id and locked_dag_id and locked_dag_id != dag_id:
            print(
                f"dbg, task_lock_release_skip reason=dag_mismatch "
                f"task_name={task_name} dag_id={dag_id} active_dag_id={locked_dag_id}",
                flush=True,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return 0

        alive_runs = task_lock_alive_runs(state)
        if alive_runs:
            alive_pids = task_lock_alive_pids({"active_runs": alive_runs})
            state["active_runs"] = alive_runs
            state["active_pids"] = [run.get("pid") for run in alive_runs if run.get("pid")]
            state["pid"] = state["active_pids"][-1] if state["active_pids"] else alive_pids[-1]
            state["ts"] = time.time()
            write_task_lock_state(lock_file, state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(
                f"dbg, task_lock_release_deferred task_name={task_name} "
                f"dag_id={dag_id} active_pids={alive_pids} reason={reason}",
                flush=True,
            )
            return 0

        write_task_lock_state(lock_file, {})
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    print(
        f"dbg, task_lock_released task_name={task_name} dag_id={dag_id} reason={reason}",
        flush=True,
    )
    return 1


def acquire_task_lock(
    task_name,
    dag_id,
    run_id,
    wait_interval_sec,
    dataset_name="",
    dataset_path="",
    stage="",
    container_name="",
    preempt_grace_timeout_min=DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN,
):
    wait_interval_sec = positive_int(wait_interval_sec, "task_lock_wait_interval_sec")
    os.makedirs(TASK_LOCK_DIR, exist_ok=True)

    while True:
        with open(TASK_LOCK_FILE, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = read_task_lock_state(lock_file)
            if task_lock_is_reusable(state, task_name):
                previous_task = state.get("task_name")
                active_runs = task_lock_alive_runs(state) if previous_task == task_name else []
                active_runs.append(
                    {
                        "pid": os.getpid(),
                        "dag_id": dag_id,
                        "run_id": run_id,
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "stage": stage,
                        "container_name": container_name,
                        "ts": time.time(),
                    }
                )
                active_pids = []
                for run in active_runs:
                    try:
                        active_pids.append(int(run.get("pid")))
                    except (TypeError, ValueError):
                        continue
                active_pids = sorted(set(active_pids))
                new_state = {
                    "task_name": task_name,
                    "dag_id": dag_id,
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "active_pids": active_pids,
                    "active_runs": active_runs,
                    "ts": time.time(),
                }
                write_task_lock_state(lock_file, new_state)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                if previous_task == task_name:
                    print(
                        f"dbg, task_lock_continue task_name={task_name} dag_id={dag_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"dbg, task_lock_acquired task_name={task_name} dag_id={dag_id}",
                        flush=True,
                    )
                return

            locked_task_name = state.get("task_name", "")
            locked_dag_id = state.get("dag_id", "")
            cleanup_plan = preempt_cleanup_plan(
                task_name,
                locked_task_name,
                state,
                preempt_grace_timeout_min,
            )
            write_task_lock_state(lock_file, state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        if cleanup_plan:
            hard_cleanup_preempted_task(cleanup_plan)
            continue

        print(
            f"dbg, task_lock_wait active_task={locked_task_name} "
            f"active_dag_id={locked_dag_id} interval_sec={wait_interval_sec}",
            flush=True,
        )
        time.sleep(wait_interval_sec)


def update_task_lock_run(task_name, pid, **updates):
    if not os.path.exists(TASK_LOCK_FILE):
        return 0
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return 0
    with open(TASK_LOCK_FILE, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = read_task_lock_state(lock_file)
        if state.get("task_name") != task_name:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return 0
        active_runs = task_lock_runs(state)
        changed = 0
        for run in active_runs:
            if int(run.get("pid") or -1) != pid:
                continue
            run.update(updates)
            run["ts"] = time.time()
            changed = 1
            break
        if changed:
            state["active_runs"] = active_runs
            state["active_pids"] = [run.get("pid") for run in active_runs if run.get("pid")]
            state["ts"] = time.time()
            write_task_lock_state(lock_file, state)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return changed


def queue_entry_priority(entry):
    try:
        return int(entry.get("priority"))
    except (TypeError, ValueError):
        return DEFAULT_TASK_PRIORITY


def queue_entry_time(entry):
    for key in ("queued_at", "submitted_at", "updated_at"):
        try:
            return float(entry.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def queue_entry_sort_key(entry):
    return (
        queue_entry_priority(entry),
        queue_entry_time(entry),
        str(entry.get("task_name") or ""),
    )


def sort_queued_tasks(queue):
    return sorted(
        [entry for entry in (queue or []) if isinstance(entry, dict)],
        key=queue_entry_sort_key,
    )


def conf_dataset_key(conf):
    if not isinstance(conf, dict):
        return ""
    return str(conf.get("dataset_name") or conf.get(PLATFORM_ORIGINAL_RUN_ID_KEY) or "")


def merged_preempted_recovery_confs(active):
    by_key = {}
    order = []
    preempted_by = active.get("preempt_requested_by") or active.get("preempted_by") or ""

    for conf in active.get("pending_run_confs") or []:
        if not isinstance(conf, dict):
            continue
        key = conf_dataset_key(conf)
        if not key:
            continue
        if key not in by_key:
            order.append(key)
        recovery_conf = dict(conf)
        recovery_conf.setdefault(PLATFORM_RECOVERY_CONF_KEY, PLATFORM_RECOVERY_REASON_PREEMPTED)
        if preempted_by:
            recovery_conf.setdefault(PLATFORM_PREEMPTED_BY_KEY, preempted_by)
        by_key[key] = recovery_conf

    for conf in active.get("drained_run_confs") or []:
        if not isinstance(conf, dict):
            continue
        key = conf_dataset_key(conf)
        if not key:
            continue
        if key not in by_key:
            order.append(key)
        by_key[key] = dict(conf)

    return [by_key[key] for key in order if key in by_key]


def activate_queue_entry(entry):
    entry = dict(entry)
    entry["status"] = "active"
    entry["remaining_runs"] = int(entry.get("remaining_runs") or entry.get("total_runs") or 1)
    entry["completed_run_ids"] = []
    entry["failed_run_ids"] = []
    entry["preempted"] = False
    entry.pop("preempted_by", None)
    entry.pop("preempted_at", None)
    entry.pop("hard_preempted", None)
    entry.pop("hard_preempted_by", None)
    entry.pop("hard_preempted_at", None)
    entry.pop("hard_preempt_cleanup", None)
    entry.pop("preempt_requested", None)
    entry.pop("preempt_requested_by", None)
    entry.pop("preempt_requested_at", None)
    entry.pop("drain_target_run_ids", None)
    entry.pop("drained_run_ids", None)
    entry.pop("verified_drained_run_ids", None)
    entry.pop("preempt_terminal_run_ids", None)
    entry.pop("drained_run_confs", None)
    entry.pop("drained_datasets", None)
    entry.pop("stage_checkpoints", None)
    entry["started_at"] = time.time()
    entry["updated_at"] = time.time()
    return entry


def read_task_queue_state(queue_file):
    queue_file.seek(0)
    raw = queue_file.read().strip()
    if not raw:
        return {"version": TASK_QUEUE_SCHEMA_VERSION, "active": None, "queue": []}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        state = {"active": None, "queue": []}
    if not isinstance(state, dict):
        state = {"active": None, "queue": []}
    if state.get("active") is not None and not isinstance(state.get("active"), dict):
        state["active"] = None
    if not isinstance(state.get("queue"), list):
        state["queue"] = []
    state["version"] = TASK_QUEUE_SCHEMA_VERSION
    state["queue"] = sort_queued_tasks(state.get("queue") or [])
    return state


def write_task_queue_state(queue_file, state):
    state["version"] = TASK_QUEUE_SCHEMA_VERSION
    queue_file.seek(0)
    queue_file.truncate()
    json.dump(state, queue_file, ensure_ascii=False, sort_keys=True)
    queue_file.write("\n")
    queue_file.flush()


def active_task_from_queue():
    if not os.path.exists(TASK_QUEUE_FILE):
        return ""
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_SH)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}
        active_task = str(active.get("task_name") or "")
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
    return active_task


def queued_entry_for_task(task_name, dag_id=""):
    if not task_name or not os.path.exists(TASK_QUEUE_FILE):
        return None
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_SH)
        state = read_task_queue_state(queue_file)
        queue = state.get("queue") or []
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    for entry in queue:
        if entry.get("task_name") != task_name:
            continue
        if dag_id and entry.get("dag_id") and entry.get("dag_id") != dag_id:
            continue
        return entry
    return None


def task_is_preempted(task_name, dag_id=""):
    entry = queued_entry_for_task(task_name, dag_id=dag_id)
    return bool(entry and (entry.get("preempted") or entry.get("hard_preempted")))


def active_entry_for_task(task_name, dag_id=""):
    if not task_name or not os.path.exists(TASK_QUEUE_FILE):
        return None
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_SH)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    if active.get("task_name") != task_name:
        return None
    if dag_id and active.get("dag_id") and active.get("dag_id") != dag_id:
        return None
    return active


def task_run_preemption_state(task_name, dag_id, run_id):
    active = active_entry_for_task(task_name, dag_id=dag_id)
    if active:
        blocked_original_run_ids = set(active.get(BLOCKED_ORIGINAL_RUN_IDS_KEY) or [])
        if run_id in blocked_original_run_ids:
            return "blocked_original"
        if active.get("status") == "draining":
            drained_run_ids = set(active.get("drained_run_ids") or [])
            if run_id in drained_run_ids:
                return "drained"
            drain_target_run_ids = set(active.get("drain_target_run_ids") or [])
            if drain_target_run_ids and run_id not in drain_target_run_ids:
                return "not_drain_target"
    if task_is_preempted(task_name, dag_id=dag_id):
        return "queued_preempted"
    return ""


def task_run_is_drained_for_preemption(task_name, dag_id, run_id):
    if not task_name or not run_id or not os.path.exists(TASK_QUEUE_FILE):
        return False
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_SH)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    if active.get("task_name") != task_name:
        return False
    if dag_id and active.get("dag_id") and active.get("dag_id") != dag_id:
        return False
    return run_id in set(active.get("drained_run_ids") or [])


def preempt_requested_by_from_queue(task_name, dag_id=""):
    if not task_name or not os.path.exists(TASK_QUEUE_FILE):
        return ""
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_SH)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
    if active.get("task_name") != task_name:
        return ""
    if dag_id and active.get("dag_id") and active.get("dag_id") != dag_id:
        return ""
    return str(active.get("preempt_requested_by") or "")


def raise_task_preempted(task_name, dag_id, dataset_name, stage, active_task=""):
    active_task = active_task or active_task_from_queue()
    print(
        f"dbg, task_preempted task_name={task_name} dag_id={dag_id} "
        f"dataset={dataset_name} stage={stage} active_task={active_task}",
        flush=True,
    )
    raise TaskPreempted(
        f"Task [{task_name}] was preempted by active task [{active_task}] "
        f"before or during stage [{stage}] for dataset [{dataset_name}]"
    )


def ensure_task_still_active(task_name, dag_id, dataset_name, stage):
    active_task = active_task_from_queue()
    if not active_task or active_task == task_name:
        return
    raise_task_preempted(task_name, dag_id, dataset_name, stage, active_task=active_task)


def preempted_queue_entry_for_locked_task(current_task_name, locked_task_name):
    if not os.path.exists(TASK_QUEUE_FILE):
        return None
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_SH)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}
        queue = state.get("queue") or []
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    if active.get("task_name") != current_task_name:
        return None
    for entry in queue:
        if entry.get("task_name") == locked_task_name and entry.get("preempted"):
            return entry
    return None


def preempt_cleanup_plan(current_task_name, locked_task_name, lock_state, grace_timeout_min):
    entry = preempted_queue_entry_for_locked_task(current_task_name, locked_task_name)
    if not entry:
        return None
    preempted_at = float(entry.get("preempted_at") or 0)
    if not preempted_at:
        return None
    grace_timeout_min = non_negative_int(
        grace_timeout_min if grace_timeout_min is not None else entry.get(
            "preempt_grace_timeout_min",
            DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN,
        ),
        "preempt_grace_timeout_min",
    )
    deadline = preempted_at + grace_timeout_min * 60
    if time.time() < deadline:
        return None
    return {
        "locked_task_name": locked_task_name,
        "current_task_name": current_task_name,
        "preempted_at": preempted_at,
        "grace_timeout_min": grace_timeout_min,
        "runs": task_lock_alive_runs(lock_state),
    }


def run_airflow_cli(args, timeout_sec=120):
    env = os.environ.copy()
    env["HOME"] = AIRFLOW_RUN_HOME
    env["AIRFLOW_HOME"] = AIRFLOW_HOME
    db_uri = airflow_metadata_db_uri(required=False)
    if db_uri:
        env["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = db_uri
        env["AIRFLOW__CORE__SQL_ALCHEMY_CONN"] = db_uri
    else:
        for key in (
            "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
            "AIRFLOW__CORE__SQL_ALCHEMY_CONN",
        ):
            env.pop(key, None)
    cmd = [AIRFLOW_BIN] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Airflow command failed: {}\nstdout:\n{}\nstderr:\n{}".format(
                " ".join(cmd),
                result.stdout.strip(),
                result.stderr.strip(),
            )
    )
    return result


def is_airflow_db_uri_allowed(uri):
    return bool(uri) and not str(uri).startswith("airflow-db-not-allowed:")


def read_platform_env_value(name):
    env_file = os.environ.get(
        "PLATFORM_ENV_FILE",
        os.path.join(AIRFLOW_RUN_HOME, "config", "platform.env"),
    )
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if not line.startswith(f"{name}="):
                    continue
                try:
                    parts = shlex.split(line, posix=True)
                except ValueError:
                    parts = [line]
                if not parts:
                    continue
                key_value = parts[0]
                if "=" not in key_value:
                    continue
                return key_value.split("=", 1)[1]
    except OSError:
        return ""
    return ""


def airflow_metadata_db_uri(required=True):
    for uri in (
        os.environ.get("AIRFLOW_DB_URI"),
        os.environ.get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"),
        os.environ.get("AIRFLOW__CORE__SQL_ALCHEMY_CONN"),
        read_platform_env_value("AIRFLOW_DB_URI"),
    ):
        if is_airflow_db_uri_allowed(uri):
            return uri
    if required:
        raise RuntimeError("Airflow metadata DB URI is not configured")
    return ""


def normalize_metadata_db_uri(uri):
    if uri.startswith("postgresql+psycopg2://"):
        return "postgresql://" + uri[len("postgresql+psycopg2://") :]
    if uri.startswith("postgres+psycopg2://"):
        return "postgresql://" + uri[len("postgres+psycopg2://") :]
    return uri


def metadata_db_connection():
    uri = normalize_metadata_db_uri(airflow_metadata_db_uri())
    if uri.startswith("sqlite:///"):
        path = uri[len("sqlite:///") :]
        if path.startswith("/"):
            db_path = path
        else:
            db_path = os.path.join(AIRFLOW_HOME, path)
        return "sqlite", sqlite3.connect(db_path)
    if uri.startswith("postgresql://") or uri.startswith("postgres://"):
        import psycopg2

        return "postgres", psycopg2.connect(uri)
    raise RuntimeError(f"Unsupported Airflow metadata DB URI: {uri}")


def metadata_sql(sql, backend):
    if backend == "sqlite":
        return sql.replace("%s", "?")
    return sql


def metadata_fetchall(sql, params=()):
    backend, conn = metadata_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(metadata_sql(sql, backend), tuple(params))
        columns = [item[0] for item in cursor.description or []]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


def metadata_execute(sql, params=()):
    backend, conn = metadata_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(metadata_sql(sql, backend), tuple(params))
        rowcount = cursor.rowcount
        conn.commit()
        return rowcount
    finally:
        conn.close()


def airflow_state_text(value):
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "")


def unpause_task_dag(dag_id):
    if not dag_id:
        return
    print(f"dbg, task_queue_unpause_next dag_id={dag_id}", flush=True)
    updated = metadata_execute(
        "update dag set is_paused = %s where dag_id = %s",
        (False, dag_id),
    )
    if not updated:
        print(f"dbg, task_queue_unpause_skip reason=dag_missing dag_id={dag_id}", flush=True)


def pause_task_dag(dag_id):
    if not dag_id:
        return
    print(f"dbg, task_queue_pause_preempted dag_id={dag_id}", flush=True)
    try:
        updated = metadata_execute(
            "update dag set is_paused = %s where dag_id = %s",
            (True, dag_id),
        )
        if not updated:
            print(
                f"dbg, task_queue_pause_skip reason=dag_missing dag_id={dag_id}",
                flush=True,
            )
    except Exception as exc:
        print(f"dbg, task_queue_pause_failed dag_id={dag_id} error={exc}", flush=True)


def is_preempted_recovery_conf(conf):
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except json.JSONDecodeError:
            conf = {}
    if not isinstance(conf, dict):
        return False
    return conf.get(PLATFORM_RECOVERY_CONF_KEY) == PLATFORM_RECOVERY_REASON_PREEMPTED


def preempted_original_runs_still_active(dag_id, current_run_id):
    if not dag_id:
        return False
    rows = metadata_fetchall(
        "select run_id, state, conf from dag_run where dag_id = %s",
        (dag_id,),
    )

    for row in rows:
        run_id = row.get("run_id")
        state = row.get("state")
        conf = row.get("conf")
        if run_id == current_run_id:
            continue
        if is_preempted_recovery_conf(conf):
            continue
        if airflow_state_text(state).lower() in DAGRUN_ACTIVE_STATES:
            return True
    return False


def pause_task_dag_after_preempted_runs_drain(task_name, dag_id, current_run_id):
    if preempted_original_runs_still_active(dag_id, current_run_id):
        print(
            f"dbg, task_queue_pause_deferred task_name={task_name} "
            f"dag_id={dag_id} run_id={current_run_id}",
            flush=True,
        )
        return False
    pause_task_dag(dag_id)
    return True


def dag_run_id(run):
    return run.get("dag_run_id") or run.get("run_id")


def dag_run_conf(run):
    conf = run.get("conf") or {}
    if isinstance(conf, str):
        try:
            return json.loads(conf)
        except json.JSONDecodeError:
            return {}
    return conf if isinstance(conf, dict) else {}


def dag_run_dataset_name(run):
    return dag_run_conf(run).get("dataset_name")


def list_task_dag_runs(dag_id):
    run_rows = metadata_fetchall(
        "select run_id, state, conf from dag_run where dag_id = %s",
        (dag_id,),
    )
    task_rows = metadata_fetchall(
        "select run_id, task_id, state from task_instance where dag_id = %s",
        (dag_id,),
    )

    task_states_by_run = {}
    for row in task_rows:
        run_id = row.get("run_id")
        task_id = row.get("task_id")
        state = row.get("state")
        task_states_by_run.setdefault(run_id, {})[task_id] = airflow_state_text(state)

    return [
        {
            "run_id": row.get("run_id"),
            "dag_run_id": row.get("run_id"),
            "state": airflow_state_text(row.get("state")),
            "conf": row.get("conf") or {},
            "task_states": task_states_by_run.get(row.get("run_id"), {}),
        }
        for row in run_rows
    ]


def dag_run_completed_pipeline(run):
    if str(run.get("state") or "").lower() != "success":
        return False
    run_conf = dag_run_conf(run)
    task_states = {
        task_id: str(state or "").lower()
        for task_id, state in (run.get("task_states") or {}).items()
    }
    if not task_states:
        return True
    verify_state = task_states.get("verify_pipeline_status")
    if verify_state == "success":
        task_rows = [
            {"task_id": task_id, "state": state}
            for task_id, state in task_states.items()
        ]
        if unexpected_skipped_stage_tasks(task_rows, conf=run_conf):
            return False
        return True
    if any(state in {"failed", "upstream_failed", "skipped"} for state in task_states.values()):
        return False
    return verify_state is None


def conf_is_preempted_recovery(conf):
    return (
        isinstance(conf, dict)
        and conf.get(PLATFORM_RECOVERY_CONF_KEY) == PLATFORM_RECOVERY_REASON_PREEMPTED
    )


def original_run_ids_for_recovery(dag_id, recovery_confs):
    if not dag_id or not recovery_confs:
        return []

    recovery_datasets = {
        str(conf.get("dataset_name"))
        for conf in recovery_confs
        if isinstance(conf, dict) and conf.get("dataset_name")
    }
    try:
        dag_runs = list_task_dag_runs(dag_id)
    except Exception as exc:
        print(
            f"dbg, preempt_original_run_scan_failed dag_id={dag_id} error={exc}",
            flush=True,
        )
        return []

    blocked_run_ids = []
    for run in dag_runs:
        run_id = str(dag_run_id(run) or "")
        if not run_id:
            continue
        run_conf = dag_run_conf(run)
        if conf_is_preempted_recovery(run_conf):
            continue
        dataset_name = str(run_conf.get("dataset_name") or "")
        if recovery_datasets and dataset_name not in recovery_datasets:
            continue
        blocked_run_ids.append(run_id)
    return sorted(set(blocked_run_ids))


def run_matches_pending_conf(run, pending_conf):
    if not isinstance(pending_conf, dict):
        return False

    pending_dataset = pending_conf.get("dataset_name")
    if not pending_dataset or dag_run_dataset_name(run) != pending_dataset:
        return False

    run_conf = dag_run_conf(run)
    pending_is_recovery = conf_is_preempted_recovery(pending_conf)
    run_is_recovery = conf_is_preempted_recovery(run_conf)
    if pending_is_recovery != run_is_recovery:
        return False

    if not pending_is_recovery:
        return True

    pending_original_run_id = pending_conf.get(PLATFORM_ORIGINAL_RUN_ID_KEY)
    run_original_run_id = run_conf.get(PLATFORM_ORIGINAL_RUN_ID_KEY)
    if (
        pending_original_run_id
        and run_original_run_id
        and pending_original_run_id != run_original_run_id
    ):
        return False

    pending_resume_from_stage = pending_conf.get(PLATFORM_RESUME_FROM_STAGE_KEY)
    run_resume_from_stage = run_conf.get(PLATFORM_RESUME_FROM_STAGE_KEY)
    if (
        pending_resume_from_stage
        and run_resume_from_stage
        and pending_resume_from_stage != run_resume_from_stage
    ):
        return False

    return True


def pending_activation_plan(dag_id, pending_run_confs):
    if not pending_run_confs:
        return {
            "expected_runs": None,
            "trigger_confs": [],
            "active_datasets": [],
            "skipped_success": [],
        }

    dag_runs = list_task_dag_runs(dag_id)

    trigger_confs = []
    skipped_success = []
    active_wait = []
    for conf in pending_run_confs:
        if not isinstance(conf, dict):
            continue
        dataset_name = conf.get("dataset_name")
        if not dataset_name:
            continue
        matching_runs = [
            run
            for run in dag_runs
            if run_matches_pending_conf(run, conf)
        ]
        if any(dag_run_completed_pipeline(run) for run in matching_runs):
            skipped_success.append(dataset_name)
        elif any(str(run.get("state") or "").lower() in DAGRUN_ACTIVE_STATES for run in matching_runs):
            active_wait.append(dataset_name)
        else:
            trigger_confs.append(conf)

    return {
        "expected_runs": len(trigger_confs) + len(set(active_wait)),
        "trigger_confs": trigger_confs,
        "active_datasets": sorted(set(active_wait)),
        "skipped_success": sorted(set(skipped_success)),
    }


def trigger_task_dag_run(dag_id, conf):
    conf_json = json.dumps(conf, ensure_ascii=False, separators=(",", ":"))
    last_error = None
    for flag in ("--conf", "-c"):
        try:
            run_airflow_cli(["dags", "trigger", dag_id, flag, conf_json])
            print(
                f"dbg, task_queue_trigger_pending dag_id={dag_id} "
                f"dataset={conf.get('dataset_name')}",
                flush=True,
            )
            return
        except RuntimeError as exc:
            last_error = exc
    raise last_error


def trigger_pending_task_runs(dag_id, pending_run_confs):
    for conf in pending_run_confs:
        trigger_task_dag_run(dag_id, conf)


def prepare_queued_task_activation(entry):
    next_task = activate_queue_entry(entry)
    pending_run_confs = next_task.get("pending_run_confs") or []
    plan = pending_activation_plan(next_task.get("dag_id"), pending_run_confs)
    if plan["expected_runs"] is not None:
        expected_runs = int(plan["expected_runs"])
        next_task["pending_run_confs"] = plan["trigger_confs"]
        next_task["total_runs"] = expected_runs
        next_task["remaining_runs"] = expected_runs
        print(
            f"dbg, task_queue_prepare_preempted task_name={next_task.get('task_name')} "
            f"expected_runs={expected_runs} trigger_runs={len(plan['trigger_confs'])} "
            f"active_existing={len(plan['active_datasets'])} "
            f"skipped_success={len(plan['skipped_success'])}",
            flush=True,
        )
        if expected_runs == 0:
            return None, plan
    return next_task, plan


def json_safe_conf_from_params(params):
    conf = {}
    for key, value in (params or {}).items():
        try:
            json.dumps(value)
        except TypeError:
            continue
        conf[key] = value
    return conf


def build_stage_resume_conf(context, resume_from_stage, preempted_by):
    conf = dict(context_dag_run_conf(context) or {})
    if not conf:
        conf = json_safe_conf_from_params(context.get("params") or {})
    conf[PLATFORM_RECOVERY_CONF_KEY] = PLATFORM_RECOVERY_REASON_PREEMPTED
    conf[PLATFORM_RESUME_FROM_STAGE_KEY] = resume_from_stage
    if preempted_by:
        conf[PLATFORM_PREEMPTED_BY_KEY] = preempted_by
    return conf


def upsert_drained_run_conf(existing_confs, new_conf, run_id):
    dataset_name = new_conf.get("dataset_name") if isinstance(new_conf, dict) else ""
    key = dataset_name or run_id
    result = []
    replaced = False
    for conf in existing_confs or []:
        if not isinstance(conf, dict):
            continue
        conf_key = conf.get("dataset_name") or conf.get("_platform_original_run_id")
        if conf_key == key:
            result.append(new_conf)
            replaced = True
        else:
            result.append(conf)
    if not replaced:
        result.append(new_conf)
    return result


def record_stage_checkpoint_after_validate(
    task_name,
    dag_id,
    run_id,
    dataset_name,
    completed_stage,
    context,
):
    stage_names = stage_names_from_context(context)
    resume_from_stage = next_pipeline_stage(completed_stage, stage_names)
    if not resume_from_stage:
        return "final_stage"
    if not task_name or not run_id or not os.path.exists(TASK_QUEUE_FILE):
        return "no_queue"

    os.makedirs(os.path.dirname(TASK_QUEUE_FILE), exist_ok=True)
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}
        if active.get("task_name") != task_name:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "not_active"
        if dag_id and active.get("dag_id") and active.get("dag_id") != dag_id:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "dag_mismatch"
        if not active.get("preempt_requested") and active.get("status") != "draining":
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "not_draining"

        now = time.time()
        active["status"] = "draining"
        active["preempt_requested"] = True
        active["updated_at"] = now

        preempted_by = str(active.get("preempt_requested_by") or "")
        recovery_conf = build_stage_resume_conf(context, resume_from_stage, preempted_by)
        recovery_conf[PLATFORM_ORIGINAL_RUN_ID_KEY] = run_id

        drained_run_ids = set(active.get("drained_run_ids") or [])
        drained_run_ids.add(run_id)
        active["drained_run_ids"] = sorted(drained_run_ids)

        stage_checkpoints = active.get("stage_checkpoints") or {}
        stage_checkpoints[dataset_name or run_id] = completed_stage
        active["stage_checkpoints"] = stage_checkpoints

        drained_datasets = active.get("drained_datasets") or {}
        drained_datasets[dataset_name or run_id] = {
            "run_id": run_id,
            "completed_stage": completed_stage,
            "resume_from_stage": resume_from_stage,
            "updated_at": now,
        }
        active["drained_datasets"] = drained_datasets
        active["drained_run_confs"] = upsert_drained_run_conf(
            active.get("drained_run_confs") or [],
            recovery_conf,
            run_id,
        )
        state["active"] = active
        write_task_queue_state(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    print(
        f"dbg, preempt_stage_boundary task_name={task_name} run_id={run_id} "
        f"dataset={dataset_name} completed_stage={completed_stage} "
        f"resume_from_stage={resume_from_stage}",
        flush=True,
    )
    return "drained"


def queued_recovery_entry(active, pending_run_confs, blocked_original_run_ids=None):
    now = time.time()
    entry = dict(active)
    preempted_by = entry.get("preempt_requested_by") or entry.get("preempted_by") or ""
    entry.update(
        {
            "status": "queued",
            "preempted": True,
            "preempted_by": preempted_by,
            "preempted_at": now,
            "queued_at": now,
            "updated_at": now,
            "pending_run_confs": pending_run_confs,
            "total_runs": len(pending_run_confs),
            "remaining_runs": len(pending_run_confs),
            "completed_run_ids": [],
            "failed_run_ids": [],
        }
    )
    blocked_original_run_ids = [
        str(run_id) for run_id in (blocked_original_run_ids or []) if run_id
    ]
    if blocked_original_run_ids:
        entry[BLOCKED_ORIGINAL_RUN_IDS_KEY] = sorted(set(blocked_original_run_ids))
    else:
        entry.pop(BLOCKED_ORIGINAL_RUN_IDS_KEY, None)
    entry.pop("started_at", None)
    entry.pop("preempt_requested", None)
    entry.pop("preempt_requested_by", None)
    entry.pop("preempt_requested_at", None)
    entry.pop("drain_target_run_ids", None)
    entry.pop("drained_run_ids", None)
    entry.pop("verified_drained_run_ids", None)
    entry.pop("preempt_terminal_run_ids", None)
    entry.pop("drained_run_confs", None)
    entry.pop("drained_datasets", None)
    entry.pop("stage_checkpoints", None)
    return entry


def advance_queue_after_active_terminal(state, task_name):
    next_dag_id = ""
    pending_run_confs_to_trigger = []
    queued_tasks = sort_queued_tasks(state.get("queue") or [])
    next_task = None
    while queued_tasks:
        candidate = queued_tasks.pop(0)
        prepared_task, plan = prepare_queued_task_activation(candidate)
        if prepared_task is None:
            print(
                f"dbg, task_queue_skip_next reason=no_pending_runs "
                f"task_name={candidate.get('task_name')}",
                flush=True,
            )
            continue
        next_task = prepared_task
        pending_run_confs_to_trigger = plan["trigger_confs"]
        break

    if next_task:
        state["active"] = next_task
        next_dag_id = str(next_task.get("dag_id") or "")
        print(
            f"dbg, task_queue_advance finished_task={task_name} "
            f"next_task={next_task.get('task_name')} next_dag_id={next_dag_id}",
            flush=True,
        )
        queue_action = "advance"
    else:
        state["active"] = None
        print(f"dbg, task_queue_empty finished_task={task_name}", flush=True)
        queue_action = "empty"

    state["queue"] = queued_tasks
    return queue_action, next_dag_id, pending_run_confs_to_trigger


def run_id_set(active, *keys):
    result = set()
    for key in keys:
        result.update(str(item) for item in (active.get(key) or []) if item)
    return result


def preempt_drain_complete(active):
    target_run_ids = run_id_set(active, "drain_target_run_ids")
    verified_run_ids = run_id_set(active, "verified_drained_run_ids")
    terminal_run_ids = run_id_set(active, "preempt_terminal_run_ids")
    all_terminal = verified_run_ids | terminal_run_ids
    if target_run_ids:
        return target_run_ids.issubset(all_terminal), len(target_run_ids), len(all_terminal)

    drain_target = int(active.get("remaining_runs") or len(active.get("drained_run_ids") or []) or 1)
    return len(all_terminal) >= drain_target, drain_target, len(all_terminal)


def finalize_preempt_drain_and_maybe_advance_queue(
    task_name,
    dag_id,
    run_id,
    mark_drained_verified=False,
    mark_preempt_terminal=False,
):
    if not task_name or not run_id or not os.path.exists(TASK_QUEUE_FILE):
        return "not_drained"

    next_dag_id = ""
    pending_run_confs_to_trigger = []
    recovery_confs = []
    queue_action = "drained_waiting"

    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}

        if active.get("task_name") != task_name:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            print(
                f"dbg, task_queue_complete_skip reason=not_active "
                f"task_name={task_name} active_task={active.get('task_name')}",
                flush=True,
            )
            return "not_active"
        if dag_id and active.get("dag_id") and active.get("dag_id") != dag_id:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "dag_mismatch"
        if active.get("status") != "draining":
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "not_draining"

        if mark_drained_verified:
            drained_run_ids = set(active.get("drained_run_ids") or [])
            if run_id not in drained_run_ids:
                write_task_queue_state(queue_file, state)
                fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
                return "not_drained"
            verified_run_ids = set(active.get("verified_drained_run_ids") or [])
            verified_run_ids.add(run_id)
            active["verified_drained_run_ids"] = sorted(verified_run_ids)

        if mark_preempt_terminal:
            target_run_ids = set(active.get("drain_target_run_ids") or [])
            if target_run_ids and run_id not in target_run_ids:
                write_task_queue_state(queue_file, state)
                fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
                return "preempted_not_target"
            terminal_run_ids = set(active.get("preempt_terminal_run_ids") or [])
            terminal_run_ids.add(run_id)
            active["preempt_terminal_run_ids"] = sorted(terminal_run_ids)

        active["updated_at"] = time.time()
        complete, drain_target, terminal_count = preempt_drain_complete(active)
        if not complete:
            state["active"] = active
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            print(
                f"dbg, preempt_drain_waiting task_name={task_name} "
                f"terminal={terminal_count} target={drain_target}",
                flush=True,
            )
            return "drained_waiting"

        recovery_confs = merged_preempted_recovery_confs(active)
        blocked_original_run_ids = original_run_ids_for_recovery(dag_id, recovery_confs)
        queue = [
            entry
            for entry in (state.get("queue") or [])
            if entry.get("task_name") != task_name
        ]
        if recovery_confs:
            queue.append(
                queued_recovery_entry(
                    active,
                    recovery_confs,
                    blocked_original_run_ids=blocked_original_run_ids,
                )
            )
        state["queue"] = sort_queued_tasks(queue)
        queue_action, next_dag_id, pending_run_confs_to_trigger = (
            advance_queue_after_active_terminal(state, task_name)
        )
        if queue_action == "advance":
            queue_action = "drained_advance"
        elif queue_action == "empty":
            queue_action = "drained_empty"
        write_task_queue_state(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    if recovery_confs:
        pause_task_dag(dag_id)
    release_task_lock(task_name, dag_id=dag_id, reason="task_queue_preempted")
    if next_dag_id:
        if pending_run_confs_to_trigger:
            trigger_pending_task_runs(next_dag_id, pending_run_confs_to_trigger)
        unpause_task_dag(next_dag_id)
    return queue_action


def finalize_drained_run_and_maybe_advance_queue(task_name, dag_id, run_id):
    return finalize_preempt_drain_and_maybe_advance_queue(
        task_name,
        dag_id,
        run_id,
        mark_drained_verified=True,
    )


def mark_preempted_run_terminal_and_maybe_advance_queue(task_name, dag_id, run_id):
    return finalize_preempt_drain_and_maybe_advance_queue(
        task_name,
        dag_id,
        run_id,
        mark_preempt_terminal=True,
    )


def finish_task_run_and_advance_queue(task_name, dag_id, run_id, outcome):
    os.makedirs(os.path.dirname(TASK_QUEUE_FILE), exist_ok=True)
    next_dag_id = ""
    queue_action = "none"
    release_lock = False
    pending_run_confs_to_trigger = []

    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = read_task_queue_state(queue_file)
        active = state.get("active") or {}

        if active.get("task_name") != task_name:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            print(
                f"dbg, task_queue_complete_skip reason=not_active "
                f"task_name={task_name} active_task={active.get('task_name')}",
                flush=True,
            )
            return "not_active"
        if dag_id and active.get("dag_id") and active.get("dag_id") != dag_id:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "dag_mismatch"
        if active.get("status") == "draining":
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return "draining"

        completed_run_ids = set(active.get("completed_run_ids") or [])
        failed_run_ids = set(active.get("failed_run_ids") or [])
        terminal_run_ids = completed_run_ids | failed_run_ids
        if run_id in terminal_run_ids:
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            print(
                f"dbg, task_queue_complete_skip reason=already_counted "
                f"task_name={task_name} run_id={run_id}",
                flush=True,
            )
            return "already_counted"

        if outcome == "success":
            completed_run_ids.add(run_id)
            active["completed_run_ids"] = sorted(completed_run_ids)
        else:
            failed_run_ids.add(run_id)
            active["failed_run_ids"] = sorted(failed_run_ids)

        remaining_runs = int(active.get("remaining_runs") or 1)
        remaining_runs = max(remaining_runs - 1, 0)
        active["remaining_runs"] = remaining_runs
        active["updated_at"] = time.time()

        print(
            f"dbg, task_queue_{outcome} task_name={task_name} run_id={run_id} "
            f"remaining_runs={remaining_runs}",
            flush=True,
        )

        if remaining_runs > 0:
            state["active"] = active
            write_task_queue_state(queue_file, state)
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return f"active_remaining_{outcome}"

        release_lock = True
        queue_action, next_dag_id, pending_run_confs_to_trigger = (
            advance_queue_after_active_terminal(state, task_name)
        )
        write_task_queue_state(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    if release_lock:
        release_task_lock(task_name, dag_id=dag_id, reason=f"task_queue_{outcome}")
    if next_dag_id:
        if pending_run_confs_to_trigger:
            trigger_pending_task_runs(next_dag_id, pending_run_confs_to_trigger)
        unpause_task_dag(next_dag_id)
    return queue_action


def complete_task_run_and_advance_queue(task_name, dag_id, run_id):
    return finish_task_run_and_advance_queue(task_name, dag_id, run_id, "success")


def fail_task_run_and_maybe_advance_queue(task_name, dag_id, run_id):
    return finish_task_run_and_advance_queue(task_name, dag_id, run_id, "failed")


def prune_dead_reservations(state):
    reservations = state.get("reservations", {})
    alive = {
        token: item
        for token, item in reservations.items()
        if is_pid_alive(item.get("pid"))
    }
    state["reservations"] = alive


def active_reserved_mb(state):
    return sum(
        int(item.get("required_mb", 0))
        for item in state.get("reservations", {}).values()
    )


def is_exclusive_reservation(item):
    return bool(item.get("exclusive"))


def has_exclusive_reservation(state):
    return any(
        is_exclusive_reservation(item)
        for item in state.get("reservations", {}).values()
    )


def sanitize_container_part(value, fallback):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-._") or fallback


def build_container_name(task_name, stage, dataset_name):
    safe_task = sanitize_container_part(task_name, "task")
    safe_stage = sanitize_container_part(stage, "stage")
    safe_dataset = sanitize_container_part(dataset_name, "dataset")
    suffix = f"{os.getpid()}-{time.time_ns()}"
    prefix = f"airflow-task-{safe_task}--{safe_stage}--"
    max_dataset_len = max(1, 127 - len(prefix) - len(suffix) - 2)
    return f"{prefix}{safe_dataset[:max_dataset_len]}--{suffix}"


def container_exists(container_name):
    result = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True

    message = f"{result.stdout}\n{result.stderr}"
    message_lower = message.lower()
    if "no such object" in message_lower or "no such container" in message_lower:
        return False

    raise RuntimeError(
        f"Unable to confirm container state for [{container_name}]: {message.strip()}"
    )


def ensure_container_removed(container_name):
    if not container_name:
        return

    while True:
        try:
            if not container_exists(container_name):
                print(f"dbg, container_removed={container_name}", flush=True)
                return
        except Exception as exc:
            print(
                f"dbg, container_state_unknown={container_name} error={exc} retry_sec={CONTAINER_CHECK_INTERVAL_SEC}",
                flush=True,
            )
            time.sleep(CONTAINER_CHECK_INTERVAL_SEC)
            continue

        print(f"dbg, stopping_container={container_name}", flush=True)
        stop_result = subprocess.run(
            ["docker", "stop", "-t", str(CONTAINER_STOP_TIMEOUT_SEC), container_name],
            capture_output=True,
            text=True,
        )
        if stop_result.returncode != 0:
            print(
                f"dbg, docker_stop_failed={container_name} error={stop_result.stderr.strip()}",
                flush=True,
            )

        remove_result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
        )
        if remove_result.returncode != 0:
            message = f"{remove_result.stdout}\n{remove_result.stderr}"
            if "No such object" not in message and "No such container" not in message:
                print(
                    f"dbg, docker_rm_failed={container_name} error={message.strip()}",
                    flush=True,
                )
        time.sleep(CONTAINER_CHECK_INTERVAL_SEC)


def terminate_process_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=CONTAINER_STOP_TIMEOUT_SEC)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def terminate_process_group_pid(pid, label=""):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return 0
    if not is_pid_alive(pid):
        return 0
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 0
    except OSError as exc:
        print(f"dbg, preempt_killpg_term_failed pid={pid} label={label} error={exc}", flush=True)
    time.sleep(2)
    if not is_pid_alive(pid):
        return 1
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return 1
    except OSError as exc:
        print(f"dbg, preempt_killpg_kill_failed pid={pid} label={label} error={exc}", flush=True)
        return 0
    return 1


def terminate_pid(pid, label=""):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return 0
    if pid == os.getpid() or not is_pid_alive(pid):
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 0
    except OSError as exc:
        print(f"dbg, preempt_pid_term_failed pid={pid} label={label} error={exc}", flush=True)
    time.sleep(2)
    if not is_pid_alive(pid):
        return 1
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return 1
    except OSError as exc:
        print(f"dbg, preempt_pid_kill_failed pid={pid} label={label} error={exc}", flush=True)
        return 0
    return 1


def remove_container_best_effort(container_name):
    if not container_name:
        return 0
    deadline = time.time() + 30
    attempted = 0
    while time.time() < deadline:
        try:
            if not container_exists(container_name):
                return attempted
        except Exception as exc:
            print(f"dbg, preempt_container_state_unknown={container_name} error={exc}", flush=True)
            return attempted
        attempted = 1
        subprocess.run(
            ["docker", "stop", "-t", str(CONTAINER_STOP_TIMEOUT_SEC), container_name],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
        )
        time.sleep(CONTAINER_CHECK_INTERVAL_SEC)
    print(f"dbg, preempt_container_remove_timeout={container_name}", flush=True)
    return attempted


def cleanup_result_jsons(dataset_path, dataset_name):
    if not dataset_path or not dataset_name:
        return 0
    clip_dir = os.path.join(str(dataset_path), str(dataset_name))
    if not os.path.isdir(clip_dir):
        print(f"dbg, preempt_result_cleanup_skip missing_dir={clip_dir}", flush=True)
        return 0
    removed = 0
    for filename in os.listdir(clip_dir):
        if not filename.startswith("results_") or not filename.endswith(".json"):
            continue
        path = os.path.join(clip_dir, filename)
        if not os.path.isfile(path) and not os.path.islink(path):
            continue
        try:
            os.unlink(path)
            removed += 1
            print(f"dbg, preempt_result_removed={path}", flush=True)
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"dbg, preempt_result_remove_failed={path} error={exc}", flush=True)
    return removed


def remove_gpu_reservations_for_task(task_name, dataset_names):
    if not os.path.isdir(GPU_LOCK_DIR):
        return 0
    selected = {name for name in dataset_names if name}
    removed = 0
    for filename in sorted(os.listdir(GPU_LOCK_DIR)):
        if not filename.startswith("gpu_") or not filename.endswith(".lock"):
            continue
        lock_path = os.path.join(GPU_LOCK_DIR, filename)
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = read_gpu_state(lock_file)
            reservations = state.setdefault("reservations", {})
            changed = False
            for token, item in list(reservations.items()):
                if item.get("task_name") != task_name:
                    continue
                if selected and item.get("dataset_name") not in selected:
                    continue
                reservations.pop(token, None)
                removed += 1
                changed = True
            if changed:
                write_gpu_state(lock_file, state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    if removed:
        print(f"dbg, preempt_gpu_reservations_removed task_name={task_name} count={removed}", flush=True)
    return removed


def clear_task_lock_after_hard_preempt(locked_task_name, current_task_name):
    with open(TASK_LOCK_FILE, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = read_task_lock_state(lock_file)
        if state.get("task_name") != locked_task_name:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return 0
        write_task_lock_state(lock_file, {})
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    print(
        f"dbg, preempt_task_lock_cleared task_name={locked_task_name} by={current_task_name}",
        flush=True,
    )
    return 1


def mark_queue_entry_hard_preempted(locked_task_name, current_task_name, stats):
    if not os.path.exists(TASK_QUEUE_FILE):
        return 0
    with open(TASK_QUEUE_FILE, "a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = read_task_queue_state(queue_file)
        changed = 0
        for entry in state.get("queue") or []:
            if entry.get("task_name") != locked_task_name:
                continue
            entry["hard_preempted"] = True
            entry["hard_preempted_at"] = time.time()
            entry["hard_preempted_by"] = current_task_name
            entry["hard_preempt_cleanup"] = stats
            changed = 1
            break
        if changed:
            write_task_queue_state(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
    return changed


def hard_cleanup_preempted_task(plan):
    locked_task_name = plan["locked_task_name"]
    current_task_name = plan["current_task_name"]
    runs = plan.get("runs") or []
    dataset_names = sorted({run.get("dataset_name") for run in runs if run.get("dataset_name")})
    stats = {
        "containers_stopped": 0,
        "processes_killed": 0,
        "results_removed": 0,
        "gpu_reservations_removed": 0,
    }
    print(
        f"dbg, hard_preempt_start task_name={locked_task_name} by={current_task_name} "
        f"runs={len(runs)} grace_timeout_min={plan.get('grace_timeout_min')}",
        flush=True,
    )
    for run in runs:
        stats["containers_stopped"] += remove_container_best_effort(run.get("container_name"))
        stats["processes_killed"] += terminate_process_group_pid(
            run.get("script_pid"),
            label=run.get("container_name") or run.get("dataset_name") or "",
        )
        stats["results_removed"] += cleanup_result_jsons(
            run.get("dataset_path"),
            run.get("dataset_name"),
        )
        stats["processes_killed"] += terminate_pid(
            run.get("pid"),
            label=run.get("dataset_name") or "",
        )
    stats["gpu_reservations_removed"] = remove_gpu_reservations_for_task(
        locked_task_name,
        dataset_names,
    )
    clear_task_lock_after_hard_preempt(locked_task_name, current_task_name)
    mark_queue_entry_hard_preempted(locked_task_name, current_task_name, stats)
    print(
        f"dbg, hard_preempt_done task_name={locked_task_name} by={current_task_name} stats={stats}",
        flush=True,
    )
    return stats


def print_process_output(output):
    if not output:
        return
    print(output, end="" if output.endswith("\n") else "\n", flush=True)


def ensure_stage_result_readable(stage, dataset_path, dataset_name, image_tag, task_name=""):
    result_path = os.path.join(dataset_path, dataset_name, f"results_{stage}.json")
    if not os.path.exists(result_path) or os.access(result_path, os.R_OK):
        return

    try:
        mode = os.stat(result_path).st_mode
        os.chmod(result_path, (mode & 0o7777) | 0o444)
    except OSError as exc:
        print(
            f"dbg, result_permission_host_failed={result_path} error={exc}",
            flush=True,
        )

    if os.access(result_path, os.R_OK):
        print(f"dbg, result_permission_fixed={result_path} method=chmod", flush=True)
        return

    if not image_tag or image_tag == "local":
        raise RuntimeError(f"Result JSON is not readable and no docker image is available: {result_path}")

    container_result_path = f"/data_pipeline/{dataset_name}/results_{stage}.json"
    helper_name = build_container_name(task_name or "task", f"{stage}_chmod", dataset_name)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        helper_name,
        "--user",
        "0:0",
        "--entrypoint",
        "sh",
        "-v",
        f"{dataset_path}:/data_pipeline",
        image_tag,
        "-c",
        "chmod a+r -- {}".format(shlex.quote(container_result_path)),
    ]
    print(f"dbg, fixing_result_permission_with_container={helper_name}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to fix result JSON permissions: {}\nstdout={}\nstderr={}".format(
                result_path,
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )
    if not os.access(result_path, os.R_OK):
        raise RuntimeError(f"Result JSON is still not readable after permission fix: {result_path}")
    print(f"dbg, result_permission_fixed={result_path} method=docker_chmod", flush=True)


def release_gpu_reservation(gpu_id, reservation_token):
    if gpu_id is None or reservation_token is None:
        return
    item = platform_gpu_allocator().release(str(gpu_id), str(reservation_token))
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
    exclusive_gpu_stages=None,
    exclusive_gpu_idle_used_max_mb=DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB,
    dataset_name="",
    task_name="",
    dag_id="",
    run_id="",
):
    gpu_pool = parse_csv(gpu_ids)
    if not gpu_pool:
        raise RuntimeError("GPU stage requires non-empty gpu_ids pool")

    required_mb = int(required_mb or 0)
    if required_mb <= 0:
        raise RuntimeError(f"GPU stage [{stage}] requires positive memory reservation")

    wait_interval_sec = positive_int(wait_interval_sec, "gpu_wait_interval_sec")
    positive_int(pending_sec, "gpu_reservation_pending_sec")
    exclusive_gpu_stages = set(exclusive_gpu_stages or [])
    exclusive_gpu_idle_used_max_mb = non_negative_int(
        exclusive_gpu_idle_used_max_mb,
        "exclusive_gpu_idle_used_max_mb",
    )
    print(
        f"dbg, gpu_pool={','.join(gpu_pool)} gpu_stage={stage} "
        f"required_memory_mb={required_mb} "
        f"exclusive_gpu_stages={','.join(sorted(exclusive_gpu_stages))} "
        f"exclusive_gpu_idle_used_max_mb={exclusive_gpu_idle_used_max_mb}",
        flush=True,
    )

    before_scan = None
    if task_name:
        before_scan = lambda: ensure_task_still_active(
            task_name, dag_id, dataset_name, stage
        )

    allocation = platform_gpu_allocator().acquire(
        gpu_pool,
        stage=stage,
        required_mb=required_mb,
        exclusive=stage in exclusive_gpu_stages,
        exclusive_idle_used_max_mb=exclusive_gpu_idle_used_max_mb,
        wait_interval_sec=wait_interval_sec,
        task_name=task_name,
        dag_id=dag_id,
        run_id=run_id,
        dataset_name=dataset_name,
        before_scan=before_scan,
    )
    print(
        f"dbg, assigned_gpu={allocation.gpu_id} stage={stage} "
        f"mode={'exclusive' if allocation.exclusive else 'shared'} "
        f"reserved_memory_mb={allocation.required_mb} token={allocation.token}",
        flush=True,
    )
    return allocation.gpu_id, allocation.token

def run_shell_script(script_name, **context):
    params = context.get("params", {})
    dag = context.get("dag")
    dag_run = context.get("dag_run")
    dag_id = getattr(dag, "dag_id", "") or getattr(dag_run, "dag_id", "")
    run_id = getattr(dag_run, "run_id", "")
    task_name = str(params.get("task_name") or dag_id or "legacy")
    dataset_name = params.get("dataset_name")
    dataset_path = params.get("dataset_path")
    runtime_timeout_min = positive_int(
        require_param(params, "timeout_min"),
        "timeout_min",
    )
    task_exclusive = parse_bool(params.get("task_exclusive", False), "task_exclusive")
    task_lock_wait_interval_sec = positive_int(
        params.get("task_lock_wait_interval_sec", 10),
        "task_lock_wait_interval_sec",
    )
    preempt_grace_timeout_min = non_negative_int(
        params.get("preempt_grace_timeout_min", DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN),
        "preempt_grace_timeout_min",
    )
    gpu_stages = unique_stage_list(
        parse_csv(require_param(params, "gpu_stages", allow_empty=True)),
        "gpu_stages",
    )
    pipeline_stage_names = flatten_stage_groups(params.get("pipeline_stages") or [])
    raw_exclusive_gpu_stages = params.get("exclusive_gpu_stages", None)
    exclusive_gpu_stages = []
    exclusive_gpu_idle_used_max_mb = DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB
    gpu_ids = ""
    gpu_stage_memory_mb = {}
    gpu_wait_interval_sec = None
    gpu_reservation_pending_sec = None

    if gpu_stages:
        if raw_exclusive_gpu_stages is None:
            exclusive_gpu_stages = list(gpu_stages)
        else:
            exclusive_gpu_stages = unique_stage_list(
                parse_csv(raw_exclusive_gpu_stages),
                "exclusive_gpu_stages",
            )
        if exclusive_gpu_stages:
            exclusive_gpu_idle_used_max_mb = non_negative_int(
                params.get(
                    "exclusive_gpu_idle_used_max_mb",
                    DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB,
                ),
                "exclusive_gpu_idle_used_max_mb",
            )
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
        validate_gpu_config(
            gpu_ids,
            gpu_stages,
            gpu_stage_memory_mb,
            exclusive_gpu_stages,
            pipeline_stages=pipeline_stage_names,
            exclusive_gpu_stages_explicit=raw_exclusive_gpu_stages is not None,
        )
    elif raw_exclusive_gpu_stages is not None and parse_csv(raw_exclusive_gpu_stages):
        raise RuntimeError(
            "Parameter [exclusive_gpu_stages] cannot be set when gpu_stages is empty"
        )

    if not dataset_name or not dataset_path:
        raise RuntimeError("Missing required parameters: dataset_name or dataset_path")

    dataset_dir = os.path.join(dataset_path, dataset_name)
    image_tag_key = f"image_{script_name.split('_')[1].replace('.sh', '')}"
    image_tag = params.get(image_tag_key, "python:3.11-slim")

    print("dbg, task_name:", task_name)
    print("dbg, dataset_name:", dataset_name)
    print("dbg, dataset_path:", dataset_path)
    print("dbg, dataset_dir:", dataset_dir)
    print("dbg, image_tag:", image_tag)
    print("dbg, gpu_ids:", gpu_ids)
    stage = stage_from_script(script_name)
    print("dbg, gpu_stage:", stage)
    print("dbg, gpu_stages:", ",".join(gpu_stages))
    print("dbg, gpu_stage_memory_mb:", gpu_stage_memory_mb)
    print("dbg, exclusive_gpu_stages:", ",".join(exclusive_gpu_stages))
    print("dbg, exclusive_gpu_idle_used_max_mb:", exclusive_gpu_idle_used_max_mb)
    print("dbg, task_exclusive:", task_exclusive)
    print("dbg, task_lock_wait_interval_sec:", task_lock_wait_interval_sec)
    print("dbg, preempt_grace_timeout_min:", preempt_grace_timeout_min)

    if stage_before_resume(stage, context):
        resume_from_stage = context_dag_run_conf(context).get(PLATFORM_RESUME_FROM_STAGE_KEY)
        print(
            f"dbg, resume_skip_run stage={stage} resume_from_stage={resume_from_stage} "
            f"task_name={task_name} dataset={dataset_name}",
            flush=True,
        )
        return f"skipped before resume stage {resume_from_stage}"

    container_name = build_container_name(task_name, stage, dataset_name)
    if task_exclusive:
        preemption_state = task_run_preemption_state(task_name, dag_id, run_id)
        if preemption_state:
            raise_task_preempted(
                task_name,
                dag_id,
                dataset_name,
                stage,
                active_task=preempt_requested_by_from_queue(task_name, dag_id),
            )
        ensure_task_still_active(task_name, dag_id, dataset_name, stage)
        acquire_task_lock(
            task_name,
            dag_id,
            run_id,
            task_lock_wait_interval_sec,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            stage=stage,
            container_name=container_name,
            preempt_grace_timeout_min=preempt_grace_timeout_min,
        )

    cmd = ["bash", f"{SCRIPT_DIR}/{script_name}"]
    env = os.environ.copy()
    env.update({
        "DATASET_NAME": dataset_name,
        "DATASET_PATH": dataset_path,
        "DATA_DIR": dataset_dir,
        "IMAGE_TAG": image_tag,
    })

    assigned_gpu = None
    reservation_token = None
    process = None
    try:
        if stage in gpu_stages:
            required_mb = gpu_stage_memory_mb[stage]
            assigned_gpu, reservation_token = acquire_gpu_from_pool(
                gpu_ids,
                stage=stage,
                required_mb=required_mb,
                wait_interval_sec=gpu_wait_interval_sec,
                pending_sec=gpu_reservation_pending_sec,
                exclusive_gpu_stages=exclusive_gpu_stages,
                exclusive_gpu_idle_used_max_mb=exclusive_gpu_idle_used_max_mb,
                dataset_name=dataset_name,
                task_name=task_name,
                dag_id=dag_id,
                run_id=run_id,
            )
            env["GPU_IDS"] = assigned_gpu

        env["CONTAINER_NAME"] = container_name
        print(f"dbg, container_name={container_name}", flush=True)

        print(
            f"dbg, container_runtime_timeout_min={runtime_timeout_min} timeout_start=after_gpu_assignment",
            flush=True,
        )
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        if task_exclusive:
            update_task_lock_run(
                task_name,
                os.getpid(),
                script_pid=process.pid,
                container_name=container_name,
            )
        try:
            output, _ = process.communicate(timeout=runtime_timeout_min * 60)
        except subprocess.TimeoutExpired:
            print(
                f"dbg, stage_timeout={stage} timeout_min={runtime_timeout_min} container={container_name}",
                flush=True,
            )
            if container_name:
                ensure_container_removed(container_name)
            terminate_process_group(process)
            output, _ = process.communicate()
            print_process_output(output)
            if task_exclusive and task_is_preempted(task_name, dag_id=dag_id):
                raise_task_preempted(task_name, dag_id, dataset_name, stage)
            raise RuntimeError(
                f"Stage [{stage}] timed out after {runtime_timeout_min} minutes "
                f"for dataset [{dataset_name}]"
            )

        print_process_output(output)
        if process.returncode != 0:
            raise RuntimeError(
                f"Script failed with exit code {process.returncode}: {script_name}"
            )
        ensure_stage_result_readable(
            stage,
            dataset_path,
            dataset_name,
            image_tag,
            task_name=task_name,
        )
        return output
    finally:
        terminate_process_group(process)
        if container_name:
            ensure_container_removed(container_name)
        release_gpu_reservation(assigned_gpu, reservation_token)


def run_validate(task_suffix, **context):
    params = context.get("params", {})
    dag = context.get("dag")
    dag_run = context.get("dag_run")
    dag_id = getattr(dag, "dag_id", "") or getattr(dag_run, "dag_id", "")
    run_id = getattr(dag_run, "run_id", "")
    task_name = str(params.get("task_name") or dag_id or "legacy")
    dataset_name = params.get("dataset_name")
    dataset_path = params.get("dataset_path")
    task_exclusive = parse_bool(params.get("task_exclusive", False), "task_exclusive")

    print("qzc:", dataset_name, dataset_path)
    
    if not dataset_name or not dataset_path:
        raise RuntimeError("Missing required parameters: dataset_name or dataset_path")

    if stage_before_resume(task_suffix, context):
        resume_from_stage = context_dag_run_conf(context).get(PLATFORM_RESUME_FROM_STAGE_KEY)
        print(
            f"dbg, resume_skip_validate stage={task_suffix} "
            f"resume_from_stage={resume_from_stage} task_name={task_name} "
            f"dataset={dataset_name}",
            flush=True,
        )
        return f"skipped validation before resume stage {resume_from_stage}"
    
    ds = resolve_validation_min_date(context)

    validate_script = os.path.join(SCRIPT_DIR, "validate_json.py")
    cmd = (
        AIRFLOW_PYTHON,
        validate_script,
        "--root-dir",
        dataset_path,
        "--dataset",
        dataset_name,
        "--task-suffix",
        task_suffix,
        "--min-date",
        ds,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Validation failed: stdout={}\nstderr={}".format(
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )
    if task_exclusive:
        record_stage_checkpoint_after_validate(
            task_name,
            dag_id,
            run_id,
            dataset_name,
            task_suffix,
            context,
        )
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


def task_instance_state(dag_id, run_id, task_id):
    rows = metadata_fetchall(
        "select state from task_instance where dag_id = %s and run_id = %s and task_id = %s",
        (dag_id, run_id, task_id),
    )
    if not rows:
        return ""
    return airflow_state_text(rows[0].get("state")).lower()


def release_task_lock_task(**context):
    params = context.get("params", {})
    dag = context.get("dag")
    dag_run = context.get("dag_run")
    dag_id = getattr(dag, "dag_id", "") or getattr(dag_run, "dag_id", "")
    task_name = str(params.get("task_name") or dag_id or "legacy")
    print(
        f"dbg, release_task_lock_task_deprecated_noop task_name={task_name} dag_id={dag_id}",
        flush=True,
    )
    return "deprecated_noop"


def finalize_task_queue_task(**context):
    params = context.get("params", {})
    dag = context.get("dag")
    dag_run = context.get("dag_run")
    dag_id = getattr(dag, "dag_id", "") or getattr(dag_run, "dag_id", "")
    run_id = getattr(dag_run, "run_id", "")
    task_name = str(params.get("task_name") or dag_id or "legacy")
    if not dag_id or not run_id:
        return "no_dag_run"

    verify_state = task_instance_state(dag_id, run_id, "verify_pipeline_status")
    preemption_state = task_run_preemption_state(task_name, dag_id, run_id)
    print(
        f"dbg, finalize_task_queue task_name={task_name} dag_id={dag_id} "
        f"run_id={run_id} verify_state={verify_state} preemption_state={preemption_state}",
        flush=True,
    )

    if verify_state == "success":
        active = active_entry_for_task(task_name, dag_id=dag_id)
        if active and active.get("status") == "draining":
            return mark_preempted_run_terminal_and_maybe_advance_queue(task_name, dag_id, run_id)
        return complete_task_run_and_advance_queue(task_name, dag_id, run_id)

    if verify_state == "skipped":
        if preemption_state == "drained" or task_run_is_drained_for_preemption(
            task_name,
            dag_id,
            run_id,
        ):
            return finalize_drained_run_and_maybe_advance_queue(task_name, dag_id, run_id)
        if preemption_state in {"not_drain_target", "queued_preempted", "blocked_original"}:
            raise AirflowSkipException(
                f"Task [{task_name}] run [{run_id}] was skipped by preemption and is not counted"
            )
        raise AirflowSkipException(
            f"Task [{task_name}] run [{run_id}] was skipped and is not counted"
        )

    if verify_state in {"failed", "upstream_failed"}:
        active = active_entry_for_task(task_name, dag_id=dag_id)
        if active and active.get("status") == "draining":
            return mark_preempted_run_terminal_and_maybe_advance_queue(task_name, dag_id, run_id)
        return fail_task_run_and_maybe_advance_queue(task_name, dag_id, run_id)

    return f"verify_state_{verify_state or 'missing'}"


def verify_pipeline_terminal_state(**context):
    params = context.get("params", {})
    dag = context.get("dag")
    dag_run = context.get("dag_run")
    dag_id = getattr(dag, "dag_id", "") or getattr(dag_run, "dag_id", "")
    run_id = getattr(dag_run, "run_id", "")
    task_name = str(params.get("task_name") or dag_id or "legacy")
    if not dag_id or not run_id:
        return "no_dag_run"

    preemption_state = task_run_preemption_state(task_name, dag_id, run_id)
    if preemption_state:
        print(
            f"dbg, verify_skip_preempted_stage_boundary task_name={task_name} "
            f"dag_id={dag_id} run_id={run_id} state={preemption_state}",
            flush=True,
        )
        raise AirflowSkipException(
            f"Task [{task_name}] reached a preemption stage boundary and is queued"
        )

    if task_is_preempted(task_name, dag_id=dag_id):
        print(
            f"dbg, verify_skip_preempted task_name={task_name} dag_id={dag_id} run_id={run_id}",
            flush=True,
        )
        pause_task_dag_after_preempted_runs_drain(task_name, dag_id, run_id)
        raise AirflowSkipException(
            f"Task [{task_name}] is preempted and queued for later recovery"
        )

    failed_tasks = []
    task_instances = metadata_fetchall(
        "select task_id, state from task_instance where dag_id = %s and run_id = %s",
        (dag_id, run_id),
    )

    for row in task_instances:
        task_id = row.get("task_id")
        state = row.get("state")
        if task_id == "verify_pipeline_status":
            continue
        state_value = airflow_state_text(state).lower()
        if state_value in {"failed", "upstream_failed"}:
            failed_tasks.append(f"{task_id}:{state_value}")

    if failed_tasks:
        raise RuntimeError("Pipeline failed tasks: " + ", ".join(sorted(failed_tasks)))

    skipped_tasks = unexpected_skipped_stage_tasks(task_instances, context)
    if skipped_tasks:
        print(
            f"dbg, verify_skip_unfinished task_name={task_name} dag_id={dag_id} "
            f"run_id={run_id} skipped_tasks={','.join(sorted(skipped_tasks))}",
            flush=True,
        )
        raise AirflowSkipException(
            "Pipeline did not finish because stage tasks were skipped: "
            + ", ".join(sorted(skipped_tasks))
        )

    return "success"


with DAG(
    dag_id="batch_pipeline_universal",
    schedule=None,
    catchup=False,
    # Dynamic task DAGs import the runtime functions from this module. Disable
    # import-side auto registration so each generated file only registers its
    # own DAG; direct DagBag parsing of this file still discovers this DAG.
    auto_register=False,
    max_active_runs=5,
    params={
        "task_name": Param(type="string", default="legacy", description="Submitted task name"),
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
        "timeout_min": Param(
            type="integer",
            description="Container runtime timeout in minutes",
        ),
        "gpu_ids": Param(type="string", description="GPU pool"),
        "gpu_stages": Param(type="string", description="Stages requiring GPU"),
        "exclusive_gpu_stages": Param(
            type=["null", "string"],
            default=None,
            description="GPU stages that require exclusive GPU access; null defaults to gpu_stages",
        ),
        "exclusive_gpu_idle_used_max_mb": Param(
            type="integer",
            default=512,
            description="Maximum used GPU memory allowed before assigning an exclusive GPU stage",
        ),
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
            description="Compatibility field; active reservations are held for the full stage runtime",
        ),
        "task_exclusive": Param(
            type="boolean",
            default=False,
            description="If true, only one submitted task_name may execute stages platform-wide at a time",
        ),
        "task_lock_wait_interval_sec": Param(
            type="integer",
            default=10,
            description="Wait interval while another task_name owns the platform task lock",
        ),
        "preempt_grace_timeout_min": Param(
            type="integer",
            default=60,
            description="Minutes to wait before hard-cleaning a preempted task",
        ),
    },
    tags=["flywheel-batch"],
    default_args={"retries": 1, "retry_delay": timedelta(seconds=60)},
) as dag:
    # 1.PARSER
    run_parser = PythonOperator(
        task_id="run_parser",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_parser.sh"},
        pool="default_pool",
    )

    validate_parser = PythonOperator(
        task_id="validate_parser",
        python_callable=run_validate,
        op_kwargs={"task_suffix": "parser"},
        pool="default_pool",
    )

    # 2 segment
    run_segment = PythonOperator(
        task_id="run_segment",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_segment.sh"},
        pool="default_pool",
    )

    validate_segment = PythonOperator(
        task_id="validate_segment",
        python_callable=run_validate,
        op_kwargs={"task_suffix": "segment"},
        pool="default_pool",
    )

    # 3.MAPPING
    run_map = PythonOperator(
        task_id="run_map",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_map.sh"},
        pool="default_pool",
    )

    validate_map = PythonOperator(
        task_id="validate_map",
        python_callable=run_validate,
        op_kwargs={"task_suffix": "map"},
        pool="default_pool",
    )

    #  4 od
    run_od = PythonOperator(
        task_id="run_od",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_od.sh"},
        pool="default_pool",
    )

    validate_od = PythonOperator(
        task_id="validate_od",
        python_callable=run_validate,
        op_kwargs={"task_suffix": "od"},
        pool="default_pool",
    )


    #  5 coloration
    run_coloration = PythonOperator(
        task_id="run_coloration",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_coloration.sh"},
        pool="default_pool",
    )

    validate_coloration = PythonOperator(
        task_id="validate_coloration",
        python_callable=run_validate,
        op_kwargs={"task_suffix": "coloration"},
        pool="default_pool",
    )

    # 6.OCC
    run_occ = PythonOperator(
        task_id="run_occ",
        python_callable=run_shell_script,
        op_kwargs={"script_name": "run_occ.sh"},
        pool="default_pool",
    )

    validate_occ = PythonOperator(
        task_id="validate_occ",
        python_callable=run_validate,
        op_kwargs={"task_suffix": "occ"},
        pool="default_pool",
    )

    verify_status = PythonOperator(
        task_id="verify_pipeline_status",
        python_callable=verify_pipeline_terminal_state,
        trigger_rule=TriggerRule.ALL_DONE,
        pool="default_pool",
    )

    finalize_queue = PythonOperator(
        task_id="finalize_task_queue",
        python_callable=finalize_task_queue_task,
        trigger_rule=TriggerRule.ALL_DONE,
        pool="default_pool",
    )

    # run_all without segment
    run_parser >> validate_parser >> run_segment >> validate_segment >> run_map >> validate_map >> run_od >> validate_od >> run_coloration >> validate_coloration >> run_occ >> validate_occ >> verify_status >> finalize_queue
    # run_parser >> validate_parser >> run_map >> validate_map >> run_od >> validate_od >> run_coloration >> validate_coloration >> run_occ >> validate_occ
    # run_occ >> validate_occ
