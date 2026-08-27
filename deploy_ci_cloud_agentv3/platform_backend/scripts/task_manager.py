#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

# Make the repository root importable when this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy_ci_cloud_agentv3.platform_backend.core.config import (
    build_task_name, build_trigger_conf, dataset_path_for, flatten_stage_groups,
    load_task_types_config, load_yaml, local_time_text, normalize_exclusive_gpu_config,
    normalize_gpu_config, normalize_pipeline_stages, normalize_preempt_config,
    normalize_task_lock_config, normalize_task_priority_config, parse_bool_config,
    parse_csv, pool_for_dataset, positive_int, priority_int, require_config,
    script_path_for_stage, script_requires_gpu,
    unique_stage_list, validate_config, validate_datasets, validate_stage_name,
    validate_stage_scripts, validate_task_name, parse_stage_memory_map,
)
from deploy_ci_cloud_agentv3.platform_backend.core.errors import TaskConfigError
from deploy_ci_cloud_agentv3.platform_backend.core.gateways.airflow import AirflowGateway
from deploy_ci_cloud_agentv3.platform_backend.core.gateways.docker import (
    container_absent, container_matches_dataset, container_text, inspect_running_containers,
    managed_task_containers, matching_containers, safe_container_part, stop_all_task_containers,
    stop_container_objects, stop_containers, task_container_prefix,
)
from deploy_ci_cloud_agentv3.platform_backend.core.gateways.gpu_reservations import (
    active_task_reservations, pid_alive, wait_for_task_reservations,
)
from deploy_ci_cloud_agentv3.platform_backend.core.services.task_service import TaskService
from deploy_ci_cloud_agentv3.platform_backend.core.queue_state import (
    apply_priority_config_to_queue_entry, queue_entry_priority, queue_entry_sort_key,
    queue_entry_time, read_task_queue_file, refresh_queue_entry_priority,
    refresh_task_queue_priorities, sort_queued_tasks, write_task_queue_file,
)
from deploy_ci_cloud_agentv3.platform_backend.core.task_store import (
    dataset_map, find_template, image_set_for_datasets, install_task_config, load_task_config, render_dag,
    repo_root_from_script, save_task_config, selected_dataset_names, stages_from_config, task_paths, template_candidates,
    update_task_priority_config,
)


TASK_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TASK_TYPE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
MAX_TASK_PREFIX_LENGTH = 32
MAX_TASK_NAME_LENGTH = 64
DEFAULT_DAGS_DIR = Path(os.environ.get("AIRFLOW_DAGS_DIR", "/home/cidi/airflow/dags/data_center"))
DEFAULT_TASK_CONFIG_ROOT = Path(os.environ.get("AIRFLOW_TASK_CONFIG_ROOT", "/opt/airflow/config/tasks"))
DEFAULT_TASK_TYPES_CONFIG = Path(
    os.environ.get(
        "AIRFLOW_TASK_TYPES_CONFIG",
        str(Path(__file__).resolve().parents[1] / "config" / "task_types.yaml"),
    )
)
DEFAULT_HOST_DATA_ROOT = Path(os.environ.get("AIRFLOW_HOST_DATA_ROOT", "/opt/airflow/data"))
DEFAULT_PARSE_TIMEOUT_SEC = int(os.environ.get("AIRFLOW_DAG_PARSE_TIMEOUT_SEC", "300"))
DEFAULT_API_BASE = os.environ.get(
    "AIRFLOW_API_BASE",
    "http://127.0.0.1:{}".format(os.environ.get("AIRFLOW_PORT", "8080")),
)
DEFAULT_API_TIMEOUT_SEC = int(os.environ.get("AIRFLOW_API_TIMEOUT_SEC", "10"))
AIRFLOW_BIN = os.environ.get("AIRFLOW_BIN", "/home/cidi/miniforge3/envs/airflow/bin/airflow")
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/home/cidi/airflow")
AIRFLOW_RUN_HOME = Path(os.environ.get("PLATFORM_HOME", str(Path(AIRFLOW_HOME).parent)))
AIRFLOW_STATE_DIR = Path(os.environ.get("AIRFLOW_STATE_DIR", str(AIRFLOW_RUN_HOME / "state")))
DEFAULT_TASK_EXCLUSIVE = True
DEFAULT_TASK_LOCK_WAIT_INTERVAL_SEC = 10
DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN = int(
    os.environ.get("AIRFLOW_PREEMPT_GRACE_TIMEOUT_MIN", "60")
)
DEFAULT_EXCLUSIVE_GPU_IDLE_USED_MAX_MB = int(
    os.environ.get("AIRFLOW_EXCLUSIVE_GPU_IDLE_USED_MAX_MB", "512")
)
LOCAL_IMAGE_OPTIONAL_STAGES = {"precheck"}
TASK_COMMAND = os.environ.get("AIRFLOW_TASK_COMMAND", "./task")
PASSWORD_FILE = Path(
    os.environ.get(
        "AIRFLOW_PASSWORD_FILE",
        str(Path(AIRFLOW_HOME) / "simple_auth_manager_passwords.json.generated"),
    )
)
GPU_LOCK_DIR = Path(os.environ.get("AIRFLOW_GPU_LOCK_DIR", str(AIRFLOW_STATE_DIR / "gpu_locks")))
TASK_LOCK_DIR = Path(os.environ.get("AIRFLOW_TASK_LOCK_DIR", str(AIRFLOW_STATE_DIR / "task_locks")))
TASK_LOCK_FILE = TASK_LOCK_DIR / "active_task.lock"
TASK_QUEUE_DIR = Path(os.environ.get("AIRFLOW_TASK_QUEUE_DIR", str(AIRFLOW_STATE_DIR / "task_queue")))
TASK_QUEUE_FILE = TASK_QUEUE_DIR / "queue.lock"
TASK_QUEUE_SCHEMA_VERSION = 2
TASK_SCHEDULE_FILE = Path(
    os.environ.get(
        "AIRFLOW_TASK_SCHEDULE_FILE",
        str(TASK_QUEUE_DIR / "scheduled_submits.lock"),
    )
)
TASK_SCHEDULE_SCHEMA_VERSION = 1
DEFAULT_SUBMIT_SCHEDULER_INTERVAL_SEC = int(
    os.environ.get("AIRFLOW_SUBMIT_SCHEDULER_INTERVAL_SEC", "30")
)
SCHEDULE_PENDING_STATUSES = {"scheduled", "running"}
SCHEDULE_REMOVABLE_STATUSES = {"scheduled"}
DEFAULT_TASK_PRIORITY = 100
GENERATED_DAG_PREFIX = "batch_pipeline_universal_"
PROTECTED_PLATFORM_DAG_IDS = {"batch_pipeline_universal"}
PLATFORM_DELETE_BYPASS_ENV = "AIRFLOW_PLATFORM_DELETE_BYPASS"
PLATFORM_RECOVERY_CONF_KEY = "_platform_recovery"
PLATFORM_RECOVERY_REASON_PREEMPTED = "preempted"
PLATFORM_RECOVERY_PREEMPTED_BY_KEY = "_platform_preempted_by"
PLATFORM_RECOVERY_CREATED_AT_KEY = "_platform_recovery_created_at"
PLATFORM_RESUME_FROM_STAGE_KEY = "_platform_resume_from_stage"
PLATFORM_ORIGINAL_RUN_ID_KEY = "_platform_original_run_id"
ACTIVE_STATES = {
    "queued",
    "running",
    "scheduled",
    "up_for_retry",
    "up_for_reschedule",
    "deferred",
}




def info(message):
    print(f"[INFO] {message}", flush=True)


def warn(message):
    print(f"[WARN] {message}", flush=True)


def fail(message, code=1):
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)
















































































































def read_task_lock_file(lock_file):
    lock_file.seek(0)
    raw = lock_file.read().strip()
    if not raw:
        return {}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return state if isinstance(state, dict) else {}


def active_task_lock_run_ids(task_name, dag_id=""):
    if not TASK_LOCK_FILE.exists():
        return []
    with TASK_LOCK_FILE.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        state = read_task_lock_file(lock_file)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    if state.get("task_name") != task_name:
        return []
    if dag_id and state.get("dag_id") and state.get("dag_id") != dag_id:
        return []

    result = []
    seen = set()
    for run in state.get("active_runs") or []:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        result.append(run_id)
    return result


def clear_task_lock(task_name, apply_changes):
    if not TASK_LOCK_FILE.exists():
        info("No matching task lock found")
        return 0
    with TASK_LOCK_FILE.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = read_task_lock_file(lock_file)
        locked_task_name = state.get("task_name")
        if locked_task_name != task_name:
            if locked_task_name:
                info(f"Task lock belongs to another task: {locked_task_name}")
            else:
                info("No matching task lock found")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return 0

        info(f"Clear task lock: {task_name}")
        if apply_changes:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return 1




















def queue_event(action, task_name=None, dag_id="", **extra):
    event = {
        "action": action,
        "task_name": task_name or "",
        "dag_id": dag_id or "",
        "started_task_name": "",
        "started_dag_id": "",
        "preempted_task_name": "",
        "preempted_dag_id": "",
        "queued_task_name": "",
        "queued_dag_id": "",
    }
    if action == "start":
        event["started_task_name"] = event["task_name"]
        event["started_dag_id"] = event["dag_id"]
    event.update(extra)
    return event


def queue_action_name(event):
    if isinstance(event, dict):
        return str(event.get("action") or "")
    return str(event)


def pending_run_confs_for_task(task_name, task_config_root=None):
    _, config = load_task_config(task_name, task_config_root=task_config_root)
    stage_groups = normalize_pipeline_stages(config)
    return [
        build_trigger_conf(task_name, config, ds, stage_groups)
        for ds in config.get("datasets", [])
        if isinstance(ds, dict) and ds.get("dataset_name")
    ]


def build_preempted_queue_entry(active, preempted_by, task_config_root=None):
    task_name = active.get("task_name")
    pending_run_confs = pending_run_confs_for_task(
        task_name,
        task_config_root=task_config_root,
    )
    now = time.time()
    entry = dict(active)
    entry.update(
        {
            "status": "queued",
            "preempted": True,
            "preempted_by": preempted_by,
            "preempted_at": now,
            "preempt_grace_timeout_min": active.get(
                "preempt_grace_timeout_min",
                DEFAULT_PREEMPT_GRACE_TIMEOUT_MIN,
            ),
            "queued_at": now,
            "updated_at": now,
            "pending_run_confs": pending_run_confs,
            "total_runs": len(pending_run_confs),
            "remaining_runs": len(pending_run_confs),
            "completed_run_ids": [],
            "failed_run_ids": [],
        }
    )
    entry.pop("started_at", None)
    entry.pop("hard_preempted", None)
    entry.pop("hard_preempted_by", None)
    entry.pop("hard_preempted_at", None)
    entry.pop("hard_preempt_cleanup", None)
    entry.pop("drain_target_run_ids", None)
    entry.pop("drained_run_ids", None)
    entry.pop("verified_drained_run_ids", None)
    entry.pop("preempt_terminal_run_ids", None)
    entry.pop("drained_run_confs", None)
    entry.pop("drained_datasets", None)
    entry.pop("stage_checkpoints", None)
    return entry


def activate_queue_entry(entry):
    now = time.time()
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
    entry["started_at"] = now
    entry["updated_at"] = now
    return entry


def apply_soft_preemption_if_needed(state, preempted_by="", task_config_root=None):
    active = state.get("active") or None
    queue = sort_queued_tasks(state.get("queue") or [])
    state["queue"] = queue
    if not active or not queue:
        return queue_event("none")

    top_queued = queue[0]
    if queue_entry_priority(top_queued) >= queue_entry_priority(active):
        return queue_event("none")

    now = time.time()
    active = dict(active)
    active["status"] = "draining"
    active["preempt_requested"] = True
    active["preempt_requested_by"] = preempted_by or top_queued.get("task_name") or ""
    active["preempt_requested_at"] = active.get("preempt_requested_at") or now
    active["updated_at"] = now
    if not active.get("pending_run_confs"):
        active["pending_run_confs"] = pending_run_confs_for_task(
            active.get("task_name"),
            task_config_root=task_config_root,
        )
    drain_target_run_ids = active.get("drain_target_run_ids") or active_task_lock_run_ids(
        active.get("task_name"),
        active.get("dag_id") or "",
    )
    if drain_target_run_ids:
        active["drain_target_run_ids"] = sorted(set(str(run_id) for run_id in drain_target_run_ids))
    active.setdefault("drained_run_ids", [])
    active.setdefault("verified_drained_run_ids", [])
    active.setdefault("preempt_terminal_run_ids", [])
    active.setdefault("drained_run_confs", [])
    active.setdefault("drained_datasets", {})
    active.setdefault("stage_checkpoints", {})
    state["active"] = active
    state["queue"] = queue
    return queue_event(
        "preempt_requested",
        task_name=top_queued.get("task_name"),
        dag_id=top_queued.get("dag_id"),
        started_task_name=active.get("task_name") or "",
        started_dag_id=active.get("dag_id") or "",
        preempted_task_name=active.get("task_name") or "",
        preempted_dag_id=active.get("dag_id") or "",
        queued_task_name=top_queued.get("task_name") or "",
        queued_dag_id=top_queued.get("dag_id") or "",
        pending_run_confs=top_queued.get("pending_run_confs") or [],
    )


def describe_task_queue_status(state, task_name):
    active = state.get("active") or None
    if active and active.get("task_name") == task_name:
        return {
            "queue_status": "draining" if active.get("status") == "draining" else "active",
            "queue_position": 0,
            "dag_id": str(active.get("dag_id") or ""),
            "priority": queue_entry_priority(active),
        }

    for index, entry in enumerate(sort_queued_tasks(state.get("queue") or []), start=1):
        if entry.get("task_name") == task_name:
            return {
                "queue_status": "queued",
                "queue_position": index,
                "dag_id": str(entry.get("dag_id") or ""),
                "priority": queue_entry_priority(entry),
            }

    return {
        "queue_status": "not_found",
        "queue_position": -1,
        "dag_id": "",
        "priority": "",
    }


def refresh_task_queue_file(task_name=None, task_config_root=None, allow_preempt=False):
    if not TASK_QUEUE_FILE.exists():
        return {
            "action": "none",
            "queue_status": "not_found",
            "queue_position": -1,
            "dag_id": "",
            "priority": "",
        }

    with TASK_QUEUE_FILE.open("a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = refresh_task_queue_priorities(
            read_task_queue_file(queue_file),
            task_config_root=task_config_root,
        )
        event = queue_event("none")
        if allow_preempt:
            event = apply_soft_preemption_if_needed(
                state,
                preempted_by=task_name or "",
                task_config_root=task_config_root,
            )
        write_task_queue_file(queue_file, state)
        status = describe_task_queue_status(state, task_name) if task_name else {}
        result = dict(event)
        result.update(status)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    return result


def build_queue_entry(task_name, dag_id, run_count, status, task_config=None):
    run_count = positive_int(run_count, "run_count")
    now = time.time()
    priority_config = normalize_task_priority_config(task_config or {})
    preempt_config = normalize_preempt_config(task_config or {})
    return {
        "task_name": task_name,
        "dag_id": dag_id,
        "total_runs": run_count,
        "remaining_runs": run_count,
        "completed_run_ids": [],
        "failed_run_ids": [],
        "status": status,
        "submitted_at": now,
        "updated_at": now,
        **priority_config,
        **preempt_config,
    }


def register_task_queue(
    task_name,
    dag_id,
    run_count,
    task_exclusive,
    task_config=None,
    task_config_root=None,
):
    if not task_exclusive:
        return queue_event("start", task_name=task_name, dag_id=dag_id)

    run_count = positive_int(run_count, "run_count")
    priority_config = normalize_task_priority_config(task_config or {})
    preempt_config = normalize_preempt_config(task_config or {})
    TASK_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with TASK_QUEUE_FILE.open("a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = refresh_task_queue_priorities(
            read_task_queue_file(queue_file),
            task_config_root=task_config_root,
        )
        active = state.get("active") or None
        queue = sort_queued_tasks(state.get("queue") or [])

        if active and active.get("task_name") == task_name:
            active["total_runs"] = int(active.get("total_runs") or 0) + run_count
            active["remaining_runs"] = int(active.get("remaining_runs") or 0) + run_count
            active["updated_at"] = time.time()
            active.update(priority_config)
            active.update(preempt_config)
            state["active"] = active
            event = queue_event("start", task_name=task_name, dag_id=dag_id)
        elif not active:
            entry = build_queue_entry(
                task_name,
                dag_id,
                run_count,
                "active",
                task_config=task_config,
            )
            entry["started_at"] = time.time()
            state["active"] = entry
            event = queue_event("start", task_name=task_name, dag_id=dag_id)
        else:
            for entry in queue:
                if entry.get("task_name") == task_name:
                    entry["total_runs"] = int(entry.get("total_runs") or 0) + run_count
                    entry["remaining_runs"] = int(entry.get("remaining_runs") or 0) + run_count
                    entry["updated_at"] = time.time()
                    entry.update(priority_config)
                    entry.update(preempt_config)
                    event = queue_event("queued", task_name=task_name, dag_id=dag_id)
                    break
            else:
                entry = build_queue_entry(
                    task_name,
                    dag_id,
                    run_count,
                    "queued",
                    task_config=task_config,
                )
                entry["queued_at"] = time.time()
                queue.append(entry)
                event = queue_event("queued", task_name=task_name, dag_id=dag_id)
            state["queue"] = sort_queued_tasks(queue)
            preempt_event = apply_soft_preemption_if_needed(
                state,
                preempted_by=task_name,
                task_config_root=task_config_root,
            )
            if queue_action_name(preempt_event) in {"preempt", "preempt_requested"}:
                event = preempt_event
                if event.get("queued_task_name") != task_name and event.get("started_task_name") != task_name:
                    event["action"] = "queued"

        write_task_queue_file(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    return event


def remove_task_from_queue(
    task_name,
    apply_changes,
    advance_next=True,
    task_config_root=None,
    api_base=None,
    token=None,
):
    if not TASK_QUEUE_FILE.exists():
        info("No matching task queue state found")
        return 0, "", []

    removed = 0
    next_dag_id = ""
    pending_run_confs_to_trigger = []
    with TASK_QUEUE_FILE.open("a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = refresh_task_queue_priorities(
            read_task_queue_file(queue_file),
            task_config_root=task_config_root,
        )
        active = state.get("active") or None
        queue = sort_queued_tasks(state.get("queue") or [])

        if active and active.get("task_name") == task_name:
            removed += 1
            kept_queue = []
            for entry in queue:
                if entry.get("task_name") == task_name:
                    removed += 1
                else:
                    kept_queue.append(entry)
            queue = sort_queued_tasks(kept_queue)
            if advance_next and queue:
                next_task = None
                while queue:
                    candidate = queue.pop(0)
                    prepared_task, plan = prepare_queued_task_activation(
                        candidate,
                        api_base=api_base,
                        token=token,
                    )
                    if prepared_task is None:
                        continue
                    next_task = prepared_task
                    pending_run_confs_to_trigger = plan["trigger_confs"]
                    break
            else:
                next_task = None
            if next_task:
                state["active"] = next_task
                next_dag_id = str(next_task.get("dag_id") or "")
            else:
                state["active"] = None
            state["queue"] = queue
        else:
            kept_queue = []
            for entry in queue:
                if entry.get("task_name") == task_name:
                    removed += 1
                else:
                    kept_queue.append(entry)
            state["queue"] = sort_queued_tasks(kept_queue)

        if apply_changes:
            write_task_queue_file(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    if removed:
        info(f"Removed task queue entry: {task_name}")
    else:
        info("No matching task queue entry found")
    return removed, next_dag_id, pending_run_confs_to_trigger


def mark_runs_completed_in_queue(
    task_name,
    run_ids,
    apply_changes,
    task_config_root=None,
    api_base=None,
    token=None,
):
    run_ids = [run_id for run_id in run_ids if run_id]
    if not run_ids or not TASK_QUEUE_FILE.exists():
        return 0, "", []

    completed = 0
    next_dag_id = ""
    pending_run_confs_to_trigger = []
    with TASK_QUEUE_FILE.open("a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = refresh_task_queue_priorities(
            read_task_queue_file(queue_file),
            task_config_root=task_config_root,
        )
        active = state.get("active") or None
        queue = sort_queued_tasks(state.get("queue") or [])

        target = None
        target_location = None
        if active and active.get("task_name") == task_name:
            target = active
            target_location = "active"
        else:
            for entry in queue:
                if entry.get("task_name") == task_name:
                    target = entry
                    target_location = "queued"
                    break

        if not target:
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
            return 0, "", []

        completed_ids = set(target.get("completed_run_ids") or [])
        for run_id in run_ids:
            if run_id in completed_ids:
                continue
            completed_ids.add(run_id)
            completed += 1
        if completed:
            target["completed_run_ids"] = sorted(completed_ids)
            target["remaining_runs"] = max(int(target.get("remaining_runs") or 0) - completed, 0)
            target["updated_at"] = time.time()

        if target.get("remaining_runs") == 0:
            if target_location == "active":
                if queue:
                    queue = sort_queued_tasks(queue)
                    next_task = None
                    while queue:
                        candidate = queue.pop(0)
                        prepared_task, plan = prepare_queued_task_activation(
                            candidate,
                            api_base=api_base,
                            token=token,
                        )
                        if prepared_task is None:
                            continue
                        next_task = prepared_task
                        pending_run_confs_to_trigger = plan["trigger_confs"]
                        break
                else:
                    next_task = None
                if next_task:
                    state["active"] = next_task
                    next_dag_id = str(next_task.get("dag_id") or "")
                else:
                    state["active"] = None
                state["queue"] = queue
            else:
                state["queue"] = [
                    entry for entry in queue if entry.get("task_name") != task_name
                ]
        else:
            state["queue"] = queue

        if apply_changes:
            write_task_queue_file(queue_file, state)
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)

    return completed, next_dag_id, pending_run_confs_to_trigger


def airflow_gateway():
    return AirflowGateway(
        airflow_bin=AIRFLOW_BIN,
        airflow_home=AIRFLOW_HOME,
        run_home=AIRFLOW_RUN_HOME,
        api_timeout_sec=DEFAULT_API_TIMEOUT_SEC,
    )


def run_airflow(args, check=True, extra_env=None):
    return airflow_gateway().run_cli(args, check=check, extra_env=extra_env)


def quote(value):
    return urllib.parse.quote(str(value), safe="")


def load_password(user):
    password = os.environ.get("AIRFLOW_API_PASSWORD")
    if password:
        return password
    try:
        with PASSWORD_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(user)
    except FileNotFoundError:
        return None


def request_json(method, api_base, path, payload=None, token=None, ok=(200, 201, 204)):
    return airflow_gateway().request_json(
        method, api_base, path, payload=payload, token=token, ok=ok
    )


def get_token(api_base):
    token = os.environ.get("AIRFLOW_API_TOKEN")
    if token:
        return token
    user = (
        os.environ.get("AIRFLOW_API_USER")
        or os.environ.get("AIRFLOW_ADMIN_USER")
        or "chang.fy"
    )
    password = load_password(user)
    if not password:
        raise RuntimeError(
            "Airflow API password not found. Set AIRFLOW_API_PASSWORD or AIRFLOW_API_TOKEN."
        )
    data = request_json(
        "POST",
        api_base,
        "/auth/token",
        payload={"username": user, "password": password},
        token=None,
    )
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Airflow token response did not contain access_token")
    return token


def list_dag_runs(api_base, token, dag_id):
    runs = []
    limit = 100
    offset = 0
    while True:
        path = f"/api/v2/dags/{quote(dag_id)}/dagRuns?limit={limit}&offset={offset}"
        data = request_json("GET", api_base, path, token=token)
        batch = data.get("dag_runs", [])
        runs.extend(batch)
        total = data.get("total_entries")
        if total is None:
            if len(batch) < limit:
                break
        elif len(runs) >= int(total):
            break
        offset += limit
    return runs


def list_dag_runs_if_present(api_base, token, dag_id):
    try:
        return list_dag_runs(api_base, token, dag_id)
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP 404" in message or "was not found" in message:
            info(f"DAG metadata already absent: {dag_id}")
            return []
        raise


def dag_run_id(run):
    return run.get("dag_run_id") or run.get("run_id")


def get_run_conf(run):
    conf = run.get("conf") or {}
    if isinstance(conf, str):
        try:
            return json.loads(conf)
        except json.JSONDecodeError:
            return {}
    return conf if isinstance(conf, dict) else {}


def run_dataset_name(run):
    return get_run_conf(run).get("dataset_name")


def conf_is_preempted_recovery(conf):
    return (
        isinstance(conf, dict)
        and conf.get(PLATFORM_RECOVERY_CONF_KEY) == PLATFORM_RECOVERY_REASON_PREEMPTED
    )


def run_matches_pending_conf(run, pending_conf):
    if not isinstance(pending_conf, dict):
        return False

    pending_dataset = pending_conf.get("dataset_name")
    if not pending_dataset or run_dataset_name(run) != pending_dataset:
        return False

    run_conf = get_run_conf(run)
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


def dag_run_completed_pipeline(run):
    if str(run.get("state") or "").lower() != "success":
        return False
    task_states = {
        task_id: str(state or "").lower()
        for task_id, state in (run.get("task_states") or {}).items()
    }
    if not task_states:
        return True
    verify_state = task_states.get("verify_pipeline_status")
    if verify_state == "success":
        return True
    if any(state in {"failed", "upstream_failed", "skipped"} for state in task_states.values()):
        return False
    return verify_state is None


def filter_runs_by_datasets(runs, datasets):
    dataset_set = set(datasets)
    return [run for run in runs if run_dataset_name(run) in dataset_set]


def filter_runs_by_states(runs, states):
    states = set(states)
    return [run for run in runs if (run.get("state") or "").lower() in states]


def run_ids_for_runs(runs):
    return [run_id for run_id in (dag_run_id(run) for run in runs) if run_id]


def pending_activation_plan(dag_id, pending_run_confs, api_base=None, token=None):
    if not pending_run_confs:
        return {
            "expected_runs": None,
            "trigger_confs": [],
            "active_datasets": [],
            "skipped_success": [],
        }
    if not api_base or not token:
        return {
            "expected_runs": len(pending_run_confs),
            "trigger_confs": pending_run_confs,
            "active_datasets": [],
            "skipped_success": [],
        }

    dag_runs = list_dag_runs_if_present(api_base, token, dag_id)

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
        elif any(str(run.get("state") or "").lower() in ACTIVE_STATES for run in matching_runs):
            active_wait.append(dataset_name)
        else:
            trigger_confs.append(conf)

    return {
        "expected_runs": len(trigger_confs) + len(set(active_wait)),
        "trigger_confs": trigger_confs,
        "active_datasets": sorted(set(active_wait)),
        "skipped_success": sorted(set(skipped_success)),
    }


def build_preempted_recovery_conf(conf, preempted_by=""):
    recovery_conf = dict(conf)
    recovery_conf[PLATFORM_RECOVERY_CONF_KEY] = PLATFORM_RECOVERY_REASON_PREEMPTED
    if preempted_by:
        recovery_conf[PLATFORM_RECOVERY_PREEMPTED_BY_KEY] = preempted_by
    recovery_conf[PLATFORM_RECOVERY_CREATED_AT_KEY] = local_time_text()
    return recovery_conf


def preempted_recovery_trigger_plan(dag_id, pending_run_confs):
    pending_confs = [
        conf
        for conf in pending_run_confs or []
        if isinstance(conf, dict) and conf.get("dataset_name")
    ]
    if not pending_confs:
        return {
            "trigger_confs": [],
            "existing_recovery_datasets": [],
            "db_checked": False,
        }

    try:
        existing_runs = list_dag_runs_db(dag_id)
        db_checked = True
    except Exception as exc:
        warn(f"Could not inspect existing recovery DagRuns for {dag_id}; triggering all. Detail: {exc}")
        existing_runs = []
        db_checked = False

    existing_recovery_datasets = set()
    for run in existing_runs:
        conf = get_run_conf(run)
        if conf.get(PLATFORM_RECOVERY_CONF_KEY) != PLATFORM_RECOVERY_REASON_PREEMPTED:
            continue
        dataset_name = conf.get("dataset_name")
        if not dataset_name:
            continue
        state = str(run.get("state") or "").lower()
        if state in ACTIVE_STATES or state == "success":
            existing_recovery_datasets.add(dataset_name)

    trigger_confs = [
        conf
        for conf in pending_confs
        if conf.get("dataset_name") not in existing_recovery_datasets
    ]
    return {
        "trigger_confs": trigger_confs,
        "existing_recovery_datasets": sorted(existing_recovery_datasets),
        "db_checked": db_checked,
    }


def trigger_preempted_recovery_runs(dag_id, pending_run_confs, preempted_by=""):
    plan = preempted_recovery_trigger_plan(dag_id, pending_run_confs)
    skipped = plan["existing_recovery_datasets"]
    if skipped:
        info(
            "Preempted recovery DagRuns already exist: "
            + ",".join(str(item) for item in skipped)
        )

    triggered = 0
    for conf in plan["trigger_confs"]:
        recovery_conf = build_preempted_recovery_conf(
            conf,
            preempted_by=preempted_by,
        )
        trigger_dag(dag_id, recovery_conf)
        triggered += 1
        info(f"triggered preempted recovery dataset={conf.get('dataset_name')}")
    return triggered


def prepare_queued_task_activation(entry, api_base=None, token=None):
    next_task = activate_queue_entry(entry)
    pending_run_confs = next_task.get("pending_run_confs") or []
    plan = pending_activation_plan(
        next_task.get("dag_id"),
        pending_run_confs,
        api_base=api_base,
        token=token,
    )
    if plan["expected_runs"] is not None:
        expected_runs = int(plan["expected_runs"])
        next_task["pending_run_confs"] = plan["trigger_confs"]
        next_task["total_runs"] = expected_runs
        next_task["remaining_runs"] = expected_runs
        if expected_runs == 0:
            return None, plan
    return next_task, plan


def trigger_pending_task_runs(dag_id, pending_run_confs):
    for conf in pending_run_confs:
        trigger_dag(dag_id, conf)
        info(f"triggered pending dataset={conf.get('dataset_name')}")


def patch_dag_run_state(api_base, token, dag_id, run_id, state):
    request_json(
        "PATCH",
        api_base,
        f"/api/v2/dags/{quote(dag_id)}/dagRuns/{quote(run_id)}",
        {"state": state},
        token,
    )


def pause_dag(api_base, token, dag_id, paused=True):
    request_json(
        "PATCH",
        api_base,
        f"/api/v2/dags/{quote(dag_id)}",
        {"is_paused": bool(paused)},
        token,
    )


def pause_dag_if_present(api_base, token, dag_id, paused=True):
    try:
        pause_dag(api_base, token, dag_id, paused=paused)
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP 404" in message or "was not found" in message:
            info(f"DAG metadata already absent: {dag_id}")
            return 0
        raise


def delete_dag_run(api_base, token, dag_id, run_id):
    request_json(
        "DELETE",
        api_base,
        f"/api/v2/dags/{quote(dag_id)}/dagRuns/{quote(run_id)}",
        token=token,
    )


def delete_dag_metadata(api_base, token, dag_id):
    try:
        request_json(
            "DELETE",
            api_base,
            f"/api/v2/dags/{quote(dag_id)}",
            token=token,
            ok=(200, 204),
        )
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP 404" in message or "was not found" in message:
            info(f"DAG metadata already absent: {dag_id}")
            return 0
        raise


def dag_exists(dag_id):
    result = run_airflow(["dags", "list"], check=False)
    if result.returncode != 0:
        warn(f"Could not list DAGs yet: {result.stderr.strip()}")
        return False
    for line in result.stdout.splitlines():
        listed_dag_id = line.split("|", 1)[0].strip()
        if listed_dag_id == dag_id:
            return True
    return False


def wait_for_dag(dag_id, timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if dag_exists(dag_id):
            return
        info(f"Waiting for Airflow to parse DAG {dag_id} ...")
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for Airflow DAG: {dag_id}")


def trigger_dag(dag_id, conf):
    conf_json = json.dumps(conf, ensure_ascii=False, separators=(",", ":"))
    result = run_airflow(["dags", "trigger", dag_id, "--conf", conf_json], check=False)
    if result.returncode != 0:
        result = run_airflow(["dags", "trigger", dag_id, "-c", conf_json], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to trigger {} for dataset {}.\nstdout:\n{}\nstderr:\n{}".format(
                dag_id,
                conf.get("dataset_name"),
                result.stdout,
                result.stderr,
            )
        )
    return result.stdout


def unpause_dag(dag_id):
    result = run_airflow(["dags", "unpause", dag_id], check=False)
    if result.returncode != 0:
        warn(
            "Could not unpause DAG {} before triggering.\nstdout:\n{}\nstderr:\n{}".format(
                dag_id,
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )


def pause_dag_cli(dag_id):
    result = run_airflow(["dags", "pause", dag_id], check=False)
    if result.returncode != 0:
        warn(
            "Could not pause DAG {} before triggering.\nstdout:\n{}\nstderr:\n{}".format(
                dag_id,
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )


def state_value(state):
    return getattr(state, "value", state)


def list_dag_runs_db(dag_id, session=None):
    def query(db_session):
        from airflow.models.dagrun import DagRun

        return [
            {
                "dag_run_id": run.run_id,
                "run_id": run.run_id,
                "state": state_value(run.state),
                "conf": run.conf or {},
            }
            for run in db_session.query(DagRun).filter(DagRun.dag_id == dag_id).all()
        ]

    if session is not None:
        return query(session)

    from airflow.utils.session import create_session

    with create_session() as new_session:
        return query(new_session)


def pause_dag_db(dag_id, paused=True, session=None):
    def apply(db_session):
        from airflow.models.dag import DagModel

        dag_model = (
            db_session.query(DagModel)
            .filter(DagModel.dag_id == dag_id)
            .one_or_none()
        )
        if dag_model is None:
            info(f"DAG metadata already absent: {dag_id}")
            return 0
        dag_model.is_paused = bool(paused)
        return 1

    if session is not None:
        return apply(session)

    from airflow.utils.session import create_session

    with create_session() as new_session:
        return apply(new_session)


def fail_selected_runs_db(dag_id, dataset_names, apply_changes, session=None):
    def apply(db_session):
        from airflow.models.dagrun import DagRun
        from airflow.models.taskinstance import TaskInstance
        from airflow.utils.state import DagRunState, TaskInstanceState

        runs = list_dag_runs_db(dag_id, session=db_session)
        active_runs = filter_runs_by_states(runs, ACTIVE_STATES)
        active_runs = filter_runs_by_datasets(active_runs, dataset_names)
        print_runs("DagRuns to mark failed", active_runs)
        if not apply_changes or not active_runs:
            return runs, len(active_runs), 0

        run_ids = run_ids_for_runs(active_runs)
        task_instances_failed = (
            db_session.query(TaskInstance)
            .filter(
                TaskInstance.dag_id == dag_id,
                TaskInstance.run_id.in_(run_ids),
                TaskInstance.state.in_(ACTIVE_STATES),
            )
            .update({TaskInstance.state: TaskInstanceState.FAILED}, synchronize_session=False)
        )
        db_runs = (
            db_session.query(DagRun)
            .filter(DagRun.dag_id == dag_id, DagRun.run_id.in_(run_ids))
            .all()
        )
        for run in db_runs:
            run.set_state(DagRunState.FAILED)
        return runs, len(db_runs), int(task_instances_failed or 0)

    if session is not None:
        return apply(session)

    from airflow.utils.session import create_session

    with create_session() as new_session:
        return apply(new_session)


def trigger_dag_internal(dag_id, conf, session=None):
    from airflow.api.common.trigger_dag import trigger_dag as airflow_trigger_dag
    from airflow.utils.types import DagRunTriggeredByType

    kwargs = {
        "triggered_by": DagRunTriggeredByType.UI,
        "conf": conf,
    }
    if session is not None:
        kwargs["session"] = session
    airflow_trigger_dag(dag_id, **kwargs)
    info(f"triggered pending dataset={conf.get('dataset_name')}")


def trigger_pending_task_runs_internal(dag_id, pending_run_confs, session=None):
    for conf in pending_run_confs:
        trigger_dag_internal(dag_id, conf, session=session)


def apply_queue_event_runtime_effects(event):
    action = queue_action_name(event)
    preempted_task_name = event.get("preempted_task_name") or ""
    preempted_dag_id = event.get("preempted_dag_id") or ""
    if action == "preempt_requested":
        info(
            "Graceful preempt requested: active_task={} active_dag_id={} "
            "queued_task={} queued_dag_id={}. Active task will stop at the next "
            "validated stage boundary.".format(
                preempted_task_name,
                preempted_dag_id,
                event.get("queued_task_name") or "",
                event.get("queued_dag_id") or "",
            )
        )
        return

    if preempted_task_name:
        info(
            "Soft preempt task={} dag_id={} -> started_task={} started_dag_id={} queued_task={}".format(
                preempted_task_name,
                preempted_dag_id,
                event.get("started_task_name"),
                event.get("started_dag_id"),
                event.get("queued_task_name") or preempted_task_name,
            )
        )
    if preempted_dag_id and action == "preempt":
        recovery_triggered = trigger_preempted_recovery_runs(
            preempted_dag_id,
            event.get("preempted_pending_run_confs") or [],
            preempted_by=event.get("started_task_name") or "",
        )
        info(
            f"Preempted recovery DagRuns triggered: "
            f"dag_id={preempted_dag_id} count={recovery_triggered}"
        )
        info(
            f"Preempted DAG will be paused by its cleanup task after active runs drain: "
            f"{preempted_dag_id}"
        )
    started_dag_id = event.get("started_dag_id") or ""
    if started_dag_id and action in {"start", "preempt"}:
        unpause_dag(started_dag_id)


def resolve_submit_inputs(args):
    legacy_task_prefix = getattr(args, "legacy_task_prefix", None)
    legacy_yaml_path = getattr(args, "legacy_yaml_path", None)
    task_prefix = args.task_prefix or legacy_task_prefix
    yaml_path = args.yaml_path or legacy_yaml_path
    if not task_prefix:
        raise TaskConfigError("submit requires --name <task_name_prefix>")
    if not yaml_path:
        raise TaskConfigError("submit requires --yaml <task.yaml>")
    if args.task_prefix and legacy_task_prefix and args.task_prefix != legacy_task_prefix:
        raise TaskConfigError("submit task name was provided twice with different values")
    if args.yaml_path and legacy_yaml_path and str(args.yaml_path) != str(legacy_yaml_path):
        raise TaskConfigError("submit YAML path was provided twice with different values")
    return task_prefix, yaml_path


def is_submit_scheduler_mode(args):
    return (
        getattr(args, "legacy_task_prefix", None) == "scheduler"
        and not getattr(args, "legacy_yaml_path", None)
        and not getattr(args, "task_prefix", None)
        and not getattr(args, "yaml_path", None)
        and not getattr(args, "schedule", None)
    )


def parse_schedule_time(value):
    text = str(value or "").strip()
    if not text:
        raise TaskConfigError("--schedule requires a time value")

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    )
    for fmt in formats:
        try:
            scheduled_at = datetime.strptime(text, fmt)
            return scheduled_at.timestamp(), scheduled_at.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    try:
        scheduled_at = datetime.fromisoformat(text)
    except ValueError:
        raise TaskConfigError(
            "Invalid --schedule time. Use 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS'."
        ) from None
    if scheduled_at.tzinfo is not None:
        timestamp = scheduled_at.timestamp()
        return timestamp, local_time_text(timestamp)
    return scheduled_at.timestamp(), scheduled_at.strftime("%Y-%m-%d %H:%M:%S")


def safe_schedule_part(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")[:24] or "task"


def build_schedule_id(task_prefix):
    suffix = str(time.time_ns())[-9:]
    return "sched_{}_{}_{}_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        os.getpid(),
        safe_schedule_part(task_prefix),
        suffix,
    )


def read_task_schedule_file(schedule_file):
    schedule_file.seek(0)
    raw = schedule_file.read().strip()
    if not raw:
        return {"version": TASK_SCHEDULE_SCHEMA_VERSION, "items": []}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        state = {"items": []}
    if isinstance(state, list):
        state = {"items": state}
    if not isinstance(state, dict):
        state = {"items": []}
    if not isinstance(state.get("items"), list):
        state["items"] = []
    state["version"] = TASK_SCHEDULE_SCHEMA_VERSION
    return state


def write_task_schedule_file(schedule_file, state):
    state["version"] = TASK_SCHEDULE_SCHEMA_VERSION
    schedule_file.seek(0)
    schedule_file.truncate()
    json.dump(state, schedule_file, ensure_ascii=False, sort_keys=True)
    schedule_file.write("\n")
    schedule_file.flush()


def schedule_item_run_after_ts(item):
    try:
        return float(item.get("run_after_ts"))
    except (TypeError, ValueError):
        return 0.0


def register_scheduled_submit(args):
    task_prefix, yaml_path = resolve_submit_inputs(args)
    validate_task_name(task_prefix)
    source_yaml, config = load_yaml(yaml_path)
    validate_config(config)
    priority_config = normalize_task_priority_config(config)
    run_after_ts, run_after_text = parse_schedule_time(args.schedule)
    schedule_id = build_schedule_id(task_prefix)
    now = time.time()
    item = {
        "schedule_id": schedule_id,
        "status": "scheduled",
        "task_prefix": task_prefix,
        "yaml_path": str(source_yaml),
        "run_after": run_after_text,
        "run_after_ts": run_after_ts,
        "created_at": now,
        "created_at_text": local_time_text(now),
        "no_trigger": bool(args.no_trigger),
        "dags_dir": str(Path(args.dags_dir or DEFAULT_DAGS_DIR).expanduser().resolve()),
        "task_config_root": str(
            Path(args.task_config_root or DEFAULT_TASK_CONFIG_ROOT).expanduser().resolve()
        ),
        "parse_timeout_sec": int(args.parse_timeout_sec or DEFAULT_PARSE_TIMEOUT_SEC),
        "task_type": priority_config["task_type"],
        "priority": priority_config["priority"],
        "priority_source": priority_config["priority_source"],
    }

    TASK_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TASK_SCHEDULE_FILE.open("a+") as schedule_file:
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_EX)
        state = read_task_schedule_file(schedule_file)
        state["items"].append(item)
        state["items"] = sorted(
            [entry for entry in state["items"] if isinstance(entry, dict)],
            key=lambda entry: (
                str(entry.get("status") or ""),
                schedule_item_run_after_ts(entry),
                str(entry.get("schedule_id") or ""),
            ),
        )
        write_task_schedule_file(schedule_file, state)
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)

    info("Scheduled submit registered")
    info(
        "task_type={} priority={} priority_source={}".format(
            priority_config["task_type"] or "(default)",
            priority_config["priority"],
            priority_config["priority_source"],
        )
    )
    if run_after_ts <= time.time():
        warn("Schedule time is already due; task submit scheduler will pick it up on next scan.")
    print(
        "schedule_id={} action=schedule_submit task_prefix={} run_after={} yaml={}".format(
            schedule_id,
            task_prefix,
            run_after_text,
            source_yaml,
        )
    )
    return {"schedule_id": schedule_id, "run_after": run_after_text}


def claim_due_scheduled_submits(now_ts=None):
    now_ts = time.time() if now_ts is None else now_ts
    due_items = []
    TASK_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TASK_SCHEDULE_FILE.open("a+") as schedule_file:
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_EX)
        state = read_task_schedule_file(schedule_file)
        for item in state["items"]:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "scheduled":
                continue
            if schedule_item_run_after_ts(item) > now_ts:
                continue
            item["status"] = "running"
            item["started_at"] = now_ts
            item["started_at_text"] = local_time_text(now_ts)
            item["scheduler_pid"] = os.getpid()
            item["attempts"] = int(item.get("attempts") or 0) + 1
            due_items.append(dict(item))
        write_task_schedule_file(schedule_file, state)
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
    return due_items


def update_scheduled_submit(schedule_id, updates):
    if not schedule_id:
        return False
    TASK_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TASK_SCHEDULE_FILE.open("a+") as schedule_file:
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_EX)
        state = read_task_schedule_file(schedule_file)
        updated = False
        for item in state["items"]:
            if isinstance(item, dict) and item.get("schedule_id") == schedule_id:
                item.update(updates)
                updated = True
                break
        if updated:
            write_task_schedule_file(schedule_file, state)
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
    return updated


def normalize_schedule_status_filters(values):
    statuses = []
    for value in values or []:
        for part in str(value).split(","):
            status = part.strip()
            if status:
                statuses.append(status)
    return statuses


def schedule_item_sort_key(item):
    return (
        schedule_item_run_after_ts(item),
        str(item.get("schedule_id") or ""),
    )


def quote_schedule_value(value):
    if value is None:
        return ""
    return shlex.quote(str(value))


def schedule_item_summary_fields(item):
    return {
        "schedule_id": item.get("schedule_id") or "",
        "status": item.get("status") or "",
        "run_after": item.get("run_after") or "",
        "task_prefix": item.get("task_prefix") or "",
        "priority": item.get("priority", ""),
        "task_type": item.get("task_type") or "",
        "yaml": item.get("yaml_path") or "",
        "result_task_name": item.get("result_task_name") or "",
        "error": item.get("error") or "",
    }


def print_schedule_item(item):
    fields = schedule_item_summary_fields(item)
    print(
        " ".join(
            f"{key}={quote_schedule_value(value)}"
            for key, value in fields.items()
            if value != ""
        )
    )


def read_scheduled_submit_state():
    TASK_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TASK_SCHEDULE_FILE.open("a+") as schedule_file:
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_EX)
        state = read_task_schedule_file(schedule_file)
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
    return state


def list_scheduled_submits(args):
    state = read_scheduled_submit_state()
    status_filters = normalize_schedule_status_filters(getattr(args, "status", []))
    if status_filters:
        status_filter_set = set(status_filters)
        filter_name = ",".join(status_filters)
    elif getattr(args, "show_all", False):
        status_filter_set = None
        filter_name = "all"
    else:
        status_filter_set = set(SCHEDULE_PENDING_STATUSES)
        filter_name = "pending"

    items = []
    for item in state["items"]:
        if not isinstance(item, dict):
            continue
        if status_filter_set is not None and item.get("status") not in status_filter_set:
            continue
        items.append(dict(item))
    items = sorted(items, key=schedule_item_sort_key)

    result = {
        "action": "schedule_list",
        "filter": filter_name,
        "count": len(items),
        "items": items,
    }
    if getattr(args, "json", False):
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"schedule_count={len(items)} action=schedule_list filter={filter_name}")
        for item in items:
            print_schedule_item(item)
    return result


def remove_scheduled_submit(args):
    schedule_id = str(getattr(args, "schedule_id", "") or "").strip()
    if not schedule_id:
        raise TaskConfigError("schedule remove requires a schedule_id")
    apply_changes = bool(getattr(args, "yes", False))

    TASK_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TASK_SCHEDULE_FILE.open("a+") as schedule_file:
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_EX)
        state = read_task_schedule_file(schedule_file)
        target = None
        for item in state["items"]:
            if isinstance(item, dict) and item.get("schedule_id") == schedule_id:
                target = item
                break
        if target is None:
            fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
            raise TaskConfigError(f"Scheduled submit not found: {schedule_id}")

        status = target.get("status") or ""
        if status == "removed":
            fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
            print(
                "schedule_id={} action=schedule_remove status=already_removed".format(
                    quote_schedule_value(schedule_id)
                )
            )
            return {"schedule_id": schedule_id, "removed": False, "status": status}
        if status not in SCHEDULE_REMOVABLE_STATUSES:
            fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
            raise TaskConfigError(
                "Cannot remove scheduled submit {} with status={}. "
                "Only status=scheduled can be removed safely.".format(schedule_id, status)
            )

        if not apply_changes:
            fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
            warn("Dry-run only. Re-run with --yes to remove this scheduled submit.")
            print_schedule_item(target)
            return {"schedule_id": schedule_id, "removed": False, "status": status}

        now = time.time()
        target["status"] = "removed"
        target["removed_at"] = now
        target["removed_at_text"] = local_time_text(now)
        target["remove_reason"] = "manual"
        write_task_schedule_file(schedule_file, state)
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)

    print(
        "schedule_id={} action=schedule_remove old_status={} new_status=removed".format(
            quote_schedule_value(schedule_id),
            quote_schedule_value(status),
        )
    )
    return {"schedule_id": schedule_id, "removed": True, "status": "removed"}


def submit_scheduled_item(item):
    submit_args = argparse.Namespace(
        legacy_task_prefix=None,
        legacy_yaml_path=None,
        task_prefix=item.get("task_prefix"),
        yaml_path=item.get("yaml_path"),
        dags_dir=item.get("dags_dir") or str(DEFAULT_DAGS_DIR),
        task_config_root=item.get("task_config_root") or str(DEFAULT_TASK_CONFIG_ROOT),
        parse_timeout_sec=int(item.get("parse_timeout_sec") or DEFAULT_PARSE_TIMEOUT_SEC),
        no_trigger=bool(item.get("no_trigger")),
        schedule=None,
        scheduler_once=False,
        scheduler_interval_sec=DEFAULT_SUBMIT_SCHEDULER_INTERVAL_SEC,
    )
    return submit(submit_args)


def run_task_submit_scheduler_once(now_ts=None):
    due_items = claim_due_scheduled_submits(now_ts=now_ts)
    processed = 0
    for item in due_items:
        schedule_id = item.get("schedule_id") or ""
        try:
            info(
                "Scheduled submit due: schedule_id={} task_prefix={} run_after={}".format(
                    schedule_id,
                    item.get("task_prefix") or "",
                    item.get("run_after") or "",
                )
            )
            result = submit_scheduled_item(item) or {}
            now = time.time()
            update_scheduled_submit(
                schedule_id,
                {
                    "status": "submitted",
                    "submitted_at": now,
                    "submitted_at_text": local_time_text(now),
                    "result_task_name": result.get("task_name", ""),
                    "result_dag_id": result.get("dag_id", ""),
                },
            )
            processed += 1
        except Exception as exc:
            now = time.time()
            update_scheduled_submit(
                schedule_id,
                {
                    "status": "failed",
                    "failed_at": now,
                    "failed_at_text": local_time_text(now),
                    "error": str(exc),
                },
            )
            warn(f"Scheduled submit failed: schedule_id={schedule_id} error={exc}")
    return processed


def run_task_submit_scheduler(args):
    interval = positive_int(
        getattr(args, "scheduler_interval_sec", DEFAULT_SUBMIT_SCHEDULER_INTERVAL_SEC),
        "--scheduler-interval-sec",
    )
    info(
        "Task submit scheduler started: schedule_file={} interval_sec={}".format(
            TASK_SCHEDULE_FILE,
            interval,
        )
    )
    while True:
        processed = run_task_submit_scheduler_once()
        if getattr(args, "scheduler_once", False):
            print(f"scheduled_processed={processed}")
            return {"processed": processed}
        time.sleep(interval)


def task_name_from_generated_dag_id(dag_id):
    dag_id = str(dag_id or "")
    if not dag_id.startswith(GENERATED_DAG_PREFIX):
        return ""
    task_name = dag_id[len(GENERATED_DAG_PREFIX):]
    return task_name if TASK_NAME_RE.match(task_name) else ""


def generated_dag_id_from_file(path):
    name = Path(path).stem
    return name if name.startswith(GENERATED_DAG_PREFIX) else ""


def read_restart_queue_refs():
    refs = []
    if not TASK_QUEUE_FILE.exists():
        return refs
    with TASK_QUEUE_FILE.open("a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = read_task_queue_file(queue_file)
        entries = []
        active = state.get("active") or None
        if active:
            entries.append(active)
        entries.extend(state.get("queue") or [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            task_name = str(entry.get("task_name") or "")
            dag_id = str(entry.get("dag_id") or "")
            if task_name or dag_id:
                refs.append({"task_name": task_name, "dag_id": dag_id, "source": "queue"})
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
    return refs


def discover_restart_task_refs(dags_dir=None, task_config_root=None):
    refs = read_restart_queue_refs()
    config_root = Path(task_config_root or DEFAULT_TASK_CONFIG_ROOT)
    if config_root.is_dir():
        for task_dir in sorted(config_root.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            if not TASK_NAME_RE.match(task_name):
                continue
            if not (task_dir / "datasets_config.yaml").is_file():
                continue
            refs.append(
                {
                    "task_name": task_name,
                    "dag_id": f"{GENERATED_DAG_PREFIX}{task_name}",
                    "source": "task_config",
                }
            )

    generated_dir = Path(dags_dir or DEFAULT_DAGS_DIR) / "generated"
    if generated_dir.is_dir():
        for dag_file in sorted(generated_dir.glob(f"{GENERATED_DAG_PREFIX}*.py")):
            dag_id = generated_dag_id_from_file(dag_file)
            task_name = task_name_from_generated_dag_id(dag_id)
            if dag_id and task_name:
                refs.append(
                    {
                        "task_name": task_name,
                        "dag_id": dag_id,
                        "source": "generated_dag",
                    }
                )

    task_names = sorted({ref["task_name"] for ref in refs if ref.get("task_name")})
    dag_ids = sorted({ref["dag_id"] for ref in refs if ref.get("dag_id")})
    return {"refs": refs, "task_names": task_names, "dag_ids": dag_ids}


def stop_scheduled_submit_state(apply_changes):
    if not TASK_SCHEDULE_FILE.exists():
        info("No scheduled submit state found")
        return 0
    stopped = 0
    with TASK_SCHEDULE_FILE.open("a+") as schedule_file:
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_EX)
        state = read_task_schedule_file(schedule_file)
        now = time.time()
        for item in state["items"]:
            if not isinstance(item, dict):
                continue
            if item.get("status") not in {"scheduled", "running"}:
                continue
            stopped += 1
            if apply_changes:
                item["status"] = "stopped"
                item["stopped_at"] = now
                item["stopped_at_text"] = local_time_text(now)
                item["stop_reason"] = "platform_restart"
        if apply_changes and stopped:
            write_task_schedule_file(schedule_file, state)
        fcntl.flock(schedule_file.fileno(), fcntl.LOCK_UN)
    info(f"Scheduled submits stopped: {stopped}")
    return stopped


def clear_task_queue_state(apply_changes):
    if not TASK_QUEUE_FILE.exists():
        info("No task queue state found")
        return 0
    with TASK_QUEUE_FILE.open("a+") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        state = read_task_queue_file(queue_file)
        active_count = 1 if state.get("active") else 0
        queued_count = len([entry for entry in state.get("queue") or [] if isinstance(entry, dict)])
        total = active_count + queued_count
        if apply_changes:
            write_task_queue_file(
                queue_file,
                {"version": TASK_QUEUE_SCHEMA_VERSION, "active": None, "queue": []},
            )
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)
    info(f"Task queue entries cleared: {total}")
    return total


def clear_all_task_locks(apply_changes):
    if not TASK_LOCK_FILE.exists():
        info("No task lock state found")
        return 0
    with TASK_LOCK_FILE.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = read_task_lock_file(lock_file)
        cleared = 1 if state else 0
        if apply_changes:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    info(f"Task lock cleared: {cleared}")
    return cleared


def clear_all_gpu_locks(apply_changes):
    if not GPU_LOCK_DIR.is_dir():
        info("No GPU lock directory found")
        return 0

    cleared = 0
    for lock_path in sorted(GPU_LOCK_DIR.glob("gpu_*.lock")):
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            lock_file.seek(0)
            raw = lock_file.read().strip()
            try:
                state = json.loads(raw or "{}")
            except json.JSONDecodeError:
                state = {}
                if raw:
                    cleared += 1
            reservations = state.get("reservations")
            if isinstance(reservations, dict):
                cleared += len(reservations)
                if apply_changes:
                    state["reservations"] = {}
            if apply_changes:
                lock_file.seek(0)
                lock_file.truncate()
                json.dump(state, lock_file, ensure_ascii=False, sort_keys=True)
                lock_file.write("\n")
                lock_file.flush()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    info(f"GPU reservations cleared: {cleared}")
    return cleared


def stop_generated_airflow_runs(dag_ids, apply_changes):
    try:
        from airflow.models.dag import DagModel
        from airflow.models.dagrun import DagRun
        from airflow.models.taskinstance import TaskInstance
        from airflow.utils.session import create_session
        from airflow.utils.state import DagRunState, TaskInstanceState
    except Exception as exc:
        warn(f"Could not import Airflow ORM for restart cleanup: {exc}")
        return {"dag_ids": [], "dag_runs_failed": 0, "task_instances_failed": 0, "dags_paused": 0}

    target_dag_ids = set(str(dag_id) for dag_id in dag_ids if dag_id)
    with create_session() as session:
        db_dag_ids = {
            row[0]
            for row in session.query(DagModel.dag_id)
            .filter(DagModel.dag_id.like(f"{GENERATED_DAG_PREFIX}%"))
            .all()
        }
        db_dag_ids.update(
            row[0]
            for row in session.query(DagRun.dag_id)
            .filter(DagRun.dag_id.like(f"{GENERATED_DAG_PREFIX}%"))
            .distinct()
            .all()
        )
        target_dag_ids.update(dag_id for dag_id in db_dag_ids if dag_id)
        target_dag_ids = sorted(target_dag_ids)
        if not target_dag_ids:
            info("No generated task DAGs found in Airflow DB")
            return {
                "dag_ids": [],
                "dag_runs_failed": 0,
                "task_instances_failed": 0,
                "dags_paused": 0,
            }

        info(f"Generated task DAGs considered for restart cleanup: {len(target_dag_ids)}")
        if not apply_changes:
            dag_run_count = (
                session.query(DagRun)
                .filter(DagRun.dag_id.in_(target_dag_ids), DagRun.state.in_(ACTIVE_STATES))
                .count()
            )
            task_instance_count = (
                session.query(TaskInstance)
                .filter(
                    TaskInstance.dag_id.in_(target_dag_ids),
                    TaskInstance.state.in_(ACTIVE_STATES),
                )
                .count()
            )
            dag_pause_count = (
                session.query(DagModel)
                .filter(DagModel.dag_id.in_(target_dag_ids), DagModel.is_paused.is_(False))
                .count()
            )
            return {
                "dag_ids": target_dag_ids,
                "dag_runs_failed": dag_run_count,
                "task_instances_failed": task_instance_count,
                "dags_paused": dag_pause_count,
            }

        dags_paused = (
            session.query(DagModel)
            .filter(DagModel.dag_id.in_(target_dag_ids), DagModel.is_paused.is_(False))
            .update({DagModel.is_paused: True}, synchronize_session=False)
        )
        task_instances_failed = (
            session.query(TaskInstance)
            .filter(
                TaskInstance.dag_id.in_(target_dag_ids),
                TaskInstance.state.in_(ACTIVE_STATES),
            )
            .update({TaskInstance.state: TaskInstanceState.FAILED}, synchronize_session=False)
        )
        active_dag_runs = (
            session.query(DagRun)
            .filter(DagRun.dag_id.in_(target_dag_ids), DagRun.state.in_(ACTIVE_STATES))
            .all()
        )
        for dag_run in active_dag_runs:
            dag_run.set_state(DagRunState.FAILED)
        return {
            "dag_ids": target_dag_ids,
            "dag_runs_failed": len(active_dag_runs),
            "task_instances_failed": int(task_instances_failed or 0),
            "dags_paused": int(dags_paused or 0),
        }


def platform_restart_cleanup(args):
    apply_changes = bool(args.yes)
    dags_dir = Path(args.dags_dir or DEFAULT_DAGS_DIR)
    task_config_root = Path(args.task_config_root or DEFAULT_TASK_CONFIG_ROOT)
    info(f"mode={'apply' if apply_changes else 'dry-run'}")
    if not apply_changes:
        warn("Dry-run only. Re-run with --yes to apply restart cleanup.")

    scheduled_stopped = stop_scheduled_submit_state(apply_changes)
    refs = discover_restart_task_refs(dags_dir=dags_dir, task_config_root=task_config_root)
    info(
        "Discovered restart task refs: tasks={} dag_ids={}".format(
            len(refs["task_names"]),
            len(refs["dag_ids"]),
        )
    )
    db_result = stop_generated_airflow_runs(refs["dag_ids"], apply_changes)

    if args.stop_containers:
        containers_stopped = stop_all_task_containers(apply_changes)
    else:
        info("Docker container stop disabled")
        containers_stopped = 0

    gpu_reservations_cleared = clear_all_gpu_locks(apply_changes)
    task_lock_cleared = clear_all_task_locks(apply_changes)
    queue_entries_cleared = clear_task_queue_state(apply_changes)

    print(
        "action=restart_cleanup scheduled_stopped={} dag_runs_failed={} "
        "task_instances_failed={} dags_paused={} containers_stopped={} "
        "gpu_reservations_cleared={} task_lock_cleared={} queue_entries_cleared={} "
        "dag_ids_considered={}".format(
            scheduled_stopped,
            db_result["dag_runs_failed"],
            db_result["task_instances_failed"],
            db_result["dags_paused"],
            containers_stopped,
            gpu_reservations_cleared,
            task_lock_cleared,
            queue_entries_cleared,
            len(db_result["dag_ids"]),
        )
    )
    return {
        "scheduled_stopped": scheduled_stopped,
        "dag_runs_failed": db_result["dag_runs_failed"],
        "task_instances_failed": db_result["task_instances_failed"],
        "dags_paused": db_result["dags_paused"],
        "containers_stopped": containers_stopped,
        "gpu_reservations_cleared": gpu_reservations_cleared,
        "task_lock_cleared": task_lock_cleared,
        "queue_entries_cleared": queue_entries_cleared,
        "dag_ids_considered": len(db_result["dag_ids"]),
    }


def print_manage_commands(task_name=None, task_deleted=False):
    task_command = shlex.quote(TASK_COMMAND)
    quoted_task_name = "<完整任务名>"
    if task_name and not task_deleted:
        quoted_task_name = shlex.quote(task_name)
    print("")
    print("常用任务命令:")
    print(f"提交新任务: {task_command} submit --name <任务名前缀> --yaml <任务yaml>")
    print(f"提交定时任务: {task_command} submit --name <任务名前缀> --yaml <任务yaml> --schedule \"YYYY-MM-DD HH:MM\"")
    print(f"查看定时任务: {task_command} schedule list")
    print(f"取消定时任务: {task_command} schedule remove <schedule_id> --yes")
    print(f"停止整个任务: {task_command} stop {quoted_task_name} --yes")
    print(f"停止指定 clip: {task_command} stop {quoted_task_name} <clip_1> <clip_2> --yes")
    print(f"恢复失败 clip: {task_command} resume {quoted_task_name}")
    print(f"恢复指定 clip: {task_command} resume {quoted_task_name} <clip_1> <clip_2>")
    print(f"调整任务优先级: {task_command} priority {quoted_task_name} --priority <数字>")
    print(f"删除整个任务: {task_command} delete {quoted_task_name} --yes")


def submit(args):
    if is_submit_scheduler_mode(args):
        return run_task_submit_scheduler(args)
    if getattr(args, "schedule", None):
        return register_scheduled_submit(args)

    task_prefix, yaml_path = resolve_submit_inputs(args)
    dags_dir = Path(args.dags_dir or DEFAULT_DAGS_DIR)
    task_config_root = Path(args.task_config_root or DEFAULT_TASK_CONFIG_ROOT)
    task_service = TaskService(dags_dir=dags_dir, task_config_root=task_config_root)
    prepared = task_service.prepare_submission(task_prefix, yaml_path)

    task_name = prepared.task_name
    config = prepared.config
    datasets = prepared.datasets
    stage_groups = prepared.stage_groups
    priority_config = prepared.priority_config
    target_yaml = prepared.target_yaml
    dag_id = prepared.dag_id
    dag_path = prepared.dag_path

    info(f"name={task_prefix}")
    info(f"task_name={task_name}")
    info(f"dag_id={dag_id}")
    info(f"config={target_yaml}")
    info(f"dag_file={dag_path}")
    info(
        "task_type={} priority={} priority_source={}".format(
            priority_config["task_type"] or "(default)",
            priority_config["priority"],
            priority_config["priority_source"],
        )
    )

    if args.no_trigger:
        info("Trigger skipped by --no-trigger")
        print(f"task_name={task_name} dag_id={dag_id} triggered=0")
        print_manage_commands(task_name)
        return {
            "task_name": task_name,
            "dag_id": dag_id,
            "triggered": 0,
            "queue_state": "skipped",
            "priority": priority_config["priority"],
        }

    wait_for_dag(dag_id, args.parse_timeout_sec)
    task_lock_config = normalize_task_lock_config(config)
    task_exclusive = task_lock_config["task_exclusive"]
    if task_exclusive:
        pause_dag_cli(dag_id)
    queue_event_result = register_task_queue(
        task_name,
        dag_id,
        len(datasets),
        task_exclusive=task_exclusive,
        task_config=config,
        task_config_root=task_config_root,
    )
    queue_action = queue_action_name(queue_event_result)
    if queue_action == "start":
        apply_queue_event_runtime_effects(queue_event_result)
    elif queue_action == "preempt":
        apply_queue_event_runtime_effects(queue_event_result)
    else:
        info(f"Task queued behind active task; DAG remains paused: {dag_id}")
        apply_queue_event_runtime_effects(queue_event_result)

    triggered = 0
    for ds in datasets:
        conf = build_trigger_conf(task_name, config, ds, stage_groups)
        trigger_dag(dag_id, conf)
        triggered += 1
        info(f"triggered dataset={ds['dataset_name']}")

    print(
        f"task_name={task_name} dag_id={dag_id} triggered={triggered} "
        f"queue_state={queue_action} priority={priority_config['priority']} "
        f"preempted_task={queue_event_result.get('preempted_task_name', '')} "
        f"started_task={queue_event_result.get('started_task_name', '')} "
        f"queued_task={queue_event_result.get('queued_task_name', '')}"
    )
    print_manage_commands(task_name)
    return {
        "task_name": task_name,
        "dag_id": dag_id,
        "triggered": triggered,
        "queue_state": queue_action,
        "priority": priority_config["priority"],
        "preempted_task": queue_event_result.get("preempted_task_name", ""),
        "started_task": queue_event_result.get("started_task_name", ""),
        "queued_task": queue_event_result.get("queued_task_name", ""),
    }


def api_context(args):
    api_base = args.api_base.rstrip("/")
    token = get_token(api_base)
    return api_base, token


def optional_api_context(args):
    try:
        return api_context(args)
    except Exception as exc:
        warn(f"Airflow API unavailable; continuing with CLI fallback. Detail: {exc}")
        return None, None


def print_runs(title, runs):
    info(f"{title}: {len(runs)}")
    for run in runs[:20]:
        info(
            "  {} dataset={} state={}".format(
                dag_run_id(run),
                run_dataset_name(run),
                run.get("state"),
            )
        )
    if len(runs) > 20:
        info(f"  ... and {len(runs) - 20} more")


def fail_selected_runs(api_base, token, dag_id, runs, apply_changes):
    print_runs("DagRuns to mark failed", runs)
    if not apply_changes:
        return 0
    count = 0
    for run in runs:
        run_id = dag_run_id(run)
        if not run_id:
            warn(f"Skip DagRun without run id: {run}")
            continue
        patch_dag_run_state(api_base, token, dag_id, run_id, "failed")
        count += 1
    return count


def stop_task(args):
    validate_task_name(args.task_name)
    config_file, config = load_task_config(args.task_name, args.task_config_root)
    dataset_names = selected_dataset_names(config, args.datasets)
    paths = task_paths(args.task_name, dags_dir=args.dags_dir, task_config_root=args.task_config_root)
    dag_id = paths["dag_id"]
    api_base, token = api_context(args)

    info(f"task_name={args.task_name}")
    info(f"dag_id={dag_id}")
    info(f"config={config_file}")
    info(f"datasets={','.join(dataset_names)}")
    info(f"mode={'apply' if args.yes else 'dry-run'}")
    if not args.yes:
        warn("Dry-run only. Re-run with --yes to apply changes.")

    stop_entire_task = not args.datasets
    if args.pause_dag and stop_entire_task:
        info(f"Pause DAG: {dag_id}")
        if args.yes:
            pause_dag(api_base, token, dag_id, paused=True)
    elif args.pause_dag:
        info("Selected-clip stop: DAG remains unpaused so other clips can continue")

    runs = filter_runs_by_states(list_dag_runs(api_base, token, dag_id), ACTIVE_STATES)
    runs = filter_runs_by_datasets(runs, dataset_names)
    run_ids_to_fail = run_ids_for_runs(runs)
    failed = fail_selected_runs(api_base, token, dag_id, runs, args.yes)

    stopped = 0
    if args.stop_containers:
        stopped = stop_containers(args.task_name, config, dataset_names, args.yes)
    else:
        info("Docker container stop disabled")

    remaining_reservations = (
        wait_for_task_reservations(args.task_name, dataset_names)
        if args.yes
        else len(active_task_reservations(args.task_name, dataset_names))
    )
    queue_removed = 0
    queue_completed = 0
    next_dag_id = ""
    pending_run_confs_to_trigger = []
    if stop_entire_task:
        queue_removed, next_dag_id, pending_run_confs_to_trigger = remove_task_from_queue(
            args.task_name,
            args.yes,
            advance_next=True,
            task_config_root=args.task_config_root,
            api_base=api_base,
            token=token,
        )
        task_lock_cleared = clear_task_lock(args.task_name, args.yes)
    else:
        queue_completed, next_dag_id, pending_run_confs_to_trigger = mark_runs_completed_in_queue(
            args.task_name,
            run_ids_to_fail,
            args.yes,
            task_config_root=args.task_config_root,
            api_base=api_base,
            token=token,
        )
        task_lock_cleared = clear_task_lock(args.task_name, args.yes) if next_dag_id else 0
    if args.yes and next_dag_id:
        if pending_run_confs_to_trigger:
            trigger_pending_task_runs(next_dag_id, pending_run_confs_to_trigger)
        unpause_dag(next_dag_id)

    print(
        f"task_name={args.task_name} action=stop dag_runs_failed={failed} "
        f"containers_stopped={stopped} reservations_remaining={remaining_reservations} "
        f"task_lock_cleared={task_lock_cleared} queue_removed={queue_removed} "
        f"queue_completed={queue_completed} next_dag_id={next_dag_id}"
    )
    print_manage_commands(args.task_name)


def trigger_datasets_for_task(task_name, config, dataset_names, dag_id):
    stage_groups = normalize_pipeline_stages(config)
    ds_map = dataset_map(config)
    triggered = 0
    for dataset_name in dataset_names:
        conf = build_trigger_conf(task_name, config, ds_map[dataset_name], stage_groups)
        trigger_dag(dag_id, conf)
        triggered += 1
        info(f"triggered dataset={dataset_name}")
    return triggered


def resume_task(args):
    validate_task_name(args.task_name)
    config_file, config = load_task_config(args.task_name, args.task_config_root)
    paths = task_paths(args.task_name, dags_dir=args.dags_dir, task_config_root=args.task_config_root)
    dag_id = paths["dag_id"]

    if args.datasets:
        dataset_names = selected_dataset_names(config, args.datasets)
    else:
        api_base, token = api_context(args)
        failed_runs = filter_runs_by_states(list_dag_runs(api_base, token, dag_id), {"failed"})
        names = []
        seen = set()
        for run in failed_runs:
            dataset_name = run_dataset_name(run)
            if dataset_name and dataset_name not in seen:
                seen.add(dataset_name)
                names.append(dataset_name)
        # An empty requested list normally means "all datasets" for stop/delete,
        # but resume without arguments must only re-submit failed datasets.
        dataset_names = selected_dataset_names(config, names) if names else []

    info(f"task_name={args.task_name}")
    info(f"dag_id={dag_id}")
    info(f"config={config_file}")
    info(f"datasets={','.join(dataset_names) if dataset_names else '(none)'}")
    priority_config = normalize_task_priority_config(config)
    info(
        "task_type={} priority={} priority_source={}".format(
            priority_config["task_type"] or "(default)",
            priority_config["priority"],
            priority_config["priority_source"],
        )
    )
    if not dataset_names:
        print(f"task_name={args.task_name} action=resume triggered=0")
        print_manage_commands(args.task_name)
        return

    task_lock_config = normalize_task_lock_config(config)
    task_exclusive = task_lock_config["task_exclusive"]
    queue_event_result = queue_event("start", task_name=args.task_name, dag_id=dag_id)
    queue_action = "start"
    if task_exclusive:
        pause_dag_cli(dag_id)
        queue_event_result = register_task_queue(
            args.task_name,
            dag_id,
            len(dataset_names),
            task_exclusive=True,
            task_config=config,
            task_config_root=args.task_config_root,
        )
        queue_action = queue_action_name(queue_event_result)
        if queue_action == "start":
            if args.unpause:
                info(f"Unpause DAG: {dag_id}")
                apply_queue_event_runtime_effects(queue_event_result)
            else:
                info("Resume requested with --no-unpause; DAG remains paused")
        elif queue_action == "preempt":
            if args.unpause:
                apply_queue_event_runtime_effects(queue_event_result)
            else:
                info("Resume requested with --no-unpause; DAG remains paused")
        else:
            info(f"Task queued behind active task; DAG remains paused: {dag_id}")
            if args.unpause:
                apply_queue_event_runtime_effects(queue_event_result)
    elif args.unpause:
        api_base, token = api_context(args)
        info(f"Unpause DAG: {dag_id}")
        pause_dag(api_base, token, dag_id, paused=False)

    triggered = trigger_datasets_for_task(args.task_name, config, dataset_names, dag_id)
    print(
        f"task_name={args.task_name} action=resume triggered={triggered} "
        f"queue_state={queue_action} "
        f"preempted_task={queue_event_result.get('preempted_task_name', '')} "
        f"started_task={queue_event_result.get('started_task_name', '')} "
        f"queued_task={queue_event_result.get('queued_task_name', '')}"
    )
    print_manage_commands(args.task_name)


def set_task_priority(args):
    task_config_root = Path(args.task_config_root or DEFAULT_TASK_CONFIG_ROOT)
    task_service = TaskService(
        dags_dir=Path(getattr(args, "dags_dir", None) or DEFAULT_DAGS_DIR),
        task_config_root=task_config_root,
    )
    priority_update = task_service.update_priority(args.task_name, args.priority)
    config_file = priority_update.config_file
    old_priority_config = priority_update.old_priority_config
    new_priority_config = priority_update.new_priority_config
    queue_status = refresh_task_queue_file(
        task_name=args.task_name,
        task_config_root=args.task_config_root,
        allow_preempt=True,
    )
    apply_queue_event_runtime_effects(queue_status)

    info(f"task_name={args.task_name}")
    info(f"config={config_file}")
    info(
        "task_type={} old_priority={} new_priority={}".format(
            new_priority_config["task_type"] or "(default)",
            old_priority_config["priority"],
            new_priority_config["priority"],
        )
    )
    print(
        "task_name={} action=priority old_priority={} new_priority={} "
        "queue_status={} queue_position={} dag_id={} "
        "preempted_task={} started_task={} queued_task={}".format(
            args.task_name,
            old_priority_config["priority"],
            new_priority_config["priority"],
            queue_status.get("queue_status", "not_found"),
            queue_status.get("queue_position", -1),
            queue_status.get("dag_id", ""),
            queue_status.get("preempted_task_name", ""),
            queue_status.get("started_task_name", ""),
            queue_status.get("queued_task_name", ""),
        )
    )
    print_manage_commands(args.task_name)


def delete_path(path, apply_changes):
    path = Path(path)
    if not path.exists():
        info(f"Path already absent: {path}")
        return False
    info(f"Delete path: {path}")
    if apply_changes:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return True


def delete_dag_metadata_cli(dag_id, apply_changes):
    if not apply_changes:
        return 0
    result = run_airflow(
        ["dags", "delete", dag_id, "-y"],
        check=False,
        extra_env={PLATFORM_DELETE_BYPASS_ENV: "1"},
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode == 0:
        if output:
            info(output)
        return 1
    lowered = output.lower()
    if "not found" in lowered or "no data found" in lowered:
        info(f"DAG metadata already absent: {dag_id}")
        return 0
    raise RuntimeError(
        "Failed to delete DAG metadata with airflow CLI: {}\n{}".format(
            dag_id,
            output,
        )
    )


def delete_dag_metadata_internal(
    dag_id,
    apply_changes,
    original_delete_dag=None,
    session=None,
    keep_records_in_log=True,
):
    if not apply_changes:
        return 0
    if original_delete_dag is None:
        return delete_dag_metadata_cli(dag_id, apply_changes)

    kwargs = {"keep_records_in_log": keep_records_in_log}
    if session is not None:
        kwargs["session"] = session
    try:
        original_delete_dag(dag_id, **kwargs)
        info(f"Deleted DAG metadata with Airflow original delete: {dag_id}")
        return 1
    except Exception as exc:
        if exc.__class__.__name__ == "DagNotFound":
            info(f"DAG metadata already absent: {dag_id}")
            return 0
        raise


def delete_task_by_name(
    task_name,
    apply_changes=False,
    dags_dir=None,
    task_config_root=None,
    stop_running_containers=True,
    api_base_arg=None,
    use_api=True,
    original_delete_dag=None,
    session=None,
    keep_records_in_log=True,
    print_summary=True,
):
    validate_task_name(task_name)
    config_file, config = load_task_config(task_name, task_config_root)
    paths = task_paths(task_name, dags_dir=dags_dir, task_config_root=task_config_root)
    dag_id = paths["dag_id"]
    all_dataset_names = selected_dataset_names(config, [])

    info(f"task_name={task_name}")
    info(f"dag_id={dag_id}")
    info(f"config={config_file}")
    info(f"dag_file={paths['dag_file']}")
    info(f"task_dir={paths['task_dir']}")
    info(f"mode={'apply' if apply_changes else 'dry-run'}")
    if not apply_changes:
        warn("Dry-run only. Re-run with --yes to delete the task.")

    info(f"Pause DAG: {dag_id}")
    if apply_changes:
        if use_api:
            pause_dag_cli(dag_id)
        else:
            pause_dag_db(dag_id, paused=True, session=session)

    api_base = None
    token = None
    task_instances_failed = 0
    failed = 0
    runs = []
    if use_api:
        api_args = argparse.Namespace(api_base=api_base_arg or DEFAULT_API_BASE)
        if apply_changes:
            api_base, token = optional_api_context(api_args)

        if api_base and token:
            try:
                if apply_changes:
                    pause_dag_if_present(api_base, token, dag_id, paused=True)
                runs = list_dag_runs_if_present(api_base, token, dag_id)
                active_runs = filter_runs_by_states(runs, ACTIVE_STATES)
                active_runs = filter_runs_by_datasets(active_runs, all_dataset_names)
                failed = fail_selected_runs(api_base, token, dag_id, active_runs, apply_changes)
            except Exception as exc:
                warn(f"Airflow API DagRun state changes failed; continuing with CLI fallback. Detail: {exc}")
                api_base = None
                token = None
                runs = []
        else:
            warn("Skipping API DagRun state changes; airflow CLI metadata delete will remove DB records.")

        if not apply_changes and api_base_arg:
            api_base, token = optional_api_context(api_args)
            if api_base and token:
                runs = list_dag_runs_if_present(api_base, token, dag_id)
                active_runs = filter_runs_by_states(runs, ACTIVE_STATES)
                active_runs = filter_runs_by_datasets(active_runs, all_dataset_names)
                failed = fail_selected_runs(api_base, token, dag_id, active_runs, apply_changes)
    else:
        runs, failed, task_instances_failed = fail_selected_runs_db(
            dag_id,
            all_dataset_names,
            apply_changes,
            session=session,
        )

    stopped = 0
    if stop_running_containers:
        stopped = stop_containers(task_name, config, all_dataset_names, apply_changes)
    else:
        info("Docker container stop disabled")

    remaining_reservations = (
        wait_for_task_reservations(task_name, all_dataset_names)
        if apply_changes
        else len(active_task_reservations(task_name, all_dataset_names))
    )
    queue_removed, next_dag_id, pending_run_confs_to_trigger = remove_task_from_queue(
        task_name,
        apply_changes,
        advance_next=True,
        task_config_root=task_config_root,
        api_base=api_base if use_api else None,
        token=token if use_api else None,
    )
    task_lock_cleared = clear_task_lock(task_name, apply_changes)
    if apply_changes and next_dag_id:
        if pending_run_confs_to_trigger:
            if use_api:
                trigger_pending_task_runs(next_dag_id, pending_run_confs_to_trigger)
            else:
                trigger_pending_task_runs_internal(
                    next_dag_id,
                    pending_run_confs_to_trigger,
                    session=session,
                )
        if use_api:
            unpause_dag(next_dag_id)
        else:
            pause_dag_db(next_dag_id, paused=False, session=session)

    runs_to_delete = filter_runs_by_datasets(runs, all_dataset_names)
    print_runs("DagRuns to delete", runs_to_delete)
    deleted_runs = 0
    if apply_changes and api_base and token:
        try:
            for run in runs_to_delete:
                run_id = dag_run_id(run)
                if not run_id:
                    continue
                delete_dag_run(api_base, token, dag_id, run_id)
                deleted_runs += 1
        except Exception as exc:
            warn(f"Airflow API DagRun delete failed; continuing with CLI metadata delete. Detail: {exc}")
    elif apply_changes and not use_api:
        deleted_runs = len(runs_to_delete)

    deleted_dag_file = delete_path(paths["dag_file"], apply_changes)
    deleted_task_dir = delete_path(paths["task_dir"], apply_changes)
    deleted_dag_metadata = delete_dag_metadata_internal(
        dag_id,
        apply_changes,
        original_delete_dag=original_delete_dag,
        session=session,
        keep_records_in_log=keep_records_in_log,
    )

    result = {
        "task_name": task_name,
        "action": "delete",
        "dag_runs_failed": failed,
        "task_instances_failed": task_instances_failed,
        "containers_stopped": stopped,
        "reservations_remaining": remaining_reservations,
        "task_lock_cleared": task_lock_cleared,
        "dag_runs_deleted": deleted_runs,
        "dag_file_deleted": int(bool(deleted_dag_file)),
        "task_dir_deleted": int(bool(deleted_task_dir)),
        "dag_metadata_deleted": deleted_dag_metadata,
        "queue_removed": queue_removed,
        "next_dag_id": next_dag_id,
    }

    if print_summary:
        print(
            "task_name={} action=delete dag_runs_failed={} containers_stopped={} "
            "reservations_remaining={} task_lock_cleared={} dag_runs_deleted={} "
            "dag_file_deleted={} task_dir_deleted={} dag_metadata_deleted={} "
            "queue_removed={} next_dag_id={}".format(
                task_name,
                failed,
                stopped,
                remaining_reservations,
                task_lock_cleared,
                deleted_runs,
                int(bool(deleted_dag_file)),
                int(bool(deleted_task_dir)),
                deleted_dag_metadata,
                queue_removed,
                next_dag_id,
            )
        )
    return result


def delete_task(args):
    delete_task_by_name(
        args.task_name,
        apply_changes=args.yes,
        dags_dir=args.dags_dir,
        task_config_root=args.task_config_root,
        stop_running_containers=args.stop_containers,
        api_base_arg=args.api_base,
        use_api=True,
        print_summary=True,
    )
    print_manage_commands(args.task_name, task_deleted=bool(args.yes))


def build_parser():
    parser = argparse.ArgumentParser(description="Manage generated Airflow batch tasks")
    subparsers = parser.add_subparsers(dest="command")

    submit_parser = subparsers.add_parser("submit", help="Submit a task YAML")
    submit_parser.add_argument("legacy_task_prefix", nargs="?", default=None, help=argparse.SUPPRESS)
    submit_parser.add_argument("legacy_yaml_path", nargs="?", default=None, help=argparse.SUPPRESS)
    submit_parser.add_argument("--name", dest="task_prefix", default=None, required=False)
    submit_parser.add_argument("--yaml", dest="yaml_path", default=None, required=False)
    submit_parser.add_argument("--dags-dir", default=str(DEFAULT_DAGS_DIR))
    submit_parser.add_argument("--task-config-root", default=str(DEFAULT_TASK_CONFIG_ROOT))
    submit_parser.add_argument("--parse-timeout-sec", type=int, default=DEFAULT_PARSE_TIMEOUT_SEC)
    submit_parser.add_argument("--no-trigger", action="store_true")
    submit_parser.add_argument(
        "--schedule",
        default=None,
        help="Schedule submit at local system time, e.g. '2026-07-28 02:00'",
    )
    submit_parser.add_argument(
        "--scheduler-interval-sec",
        type=int,
        default=DEFAULT_SUBMIT_SCHEDULER_INTERVAL_SEC,
        help=argparse.SUPPRESS,
    )
    submit_parser.add_argument(
        "--scheduler-once",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    submit_parser.set_defaults(func=submit)

    def add_common_manage_options(subparser):
        subparser.add_argument("task_name")
        subparser.add_argument("--dags-dir", default=str(DEFAULT_DAGS_DIR))
        subparser.add_argument("--task-config-root", default=str(DEFAULT_TASK_CONFIG_ROOT))
        subparser.add_argument("--api-base", default=DEFAULT_API_BASE)
        return subparser

    stop_parser = subparsers.add_parser("stop", help="Stop all or selected active datasets")
    add_common_manage_options(stop_parser)
    stop_parser.add_argument("datasets", nargs="*")
    stop_parser.add_argument("--yes", action="store_true")
    stop_parser.add_argument("--no-stop-containers", dest="stop_containers", action="store_false")
    stop_parser.add_argument("--no-pause", dest="pause_dag", action="store_false")
    stop_parser.set_defaults(func=stop_task, stop_containers=True, pause_dag=True)

    resume_parser = subparsers.add_parser("resume", help="Resume failed or selected datasets")
    add_common_manage_options(resume_parser)
    resume_parser.add_argument("datasets", nargs="*")
    resume_parser.add_argument("--no-unpause", dest="unpause", action="store_false")
    resume_parser.set_defaults(func=resume_task, unpause=True)

    priority_parser = subparsers.add_parser("priority", help="Update submitted task priority")
    priority_parser.add_argument("task_name")
    priority_parser.add_argument("--priority", required=True)
    priority_parser.add_argument("--task-config-root", default=str(DEFAULT_TASK_CONFIG_ROOT))
    priority_parser.set_defaults(func=set_task_priority)

    schedule_parser = subparsers.add_parser("schedule", help="List or remove scheduled submits")
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command")
    schedule_subparsers.required = True

    schedule_list_parser = schedule_subparsers.add_parser(
        "list",
        help="List scheduled submits",
    )
    schedule_list_parser.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        help="Show historical submitted/failed/stopped/removed records too",
    )
    schedule_list_parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Filter by status. Can be repeated or comma-separated.",
    )
    schedule_list_parser.add_argument("--json", action="store_true")
    schedule_list_parser.set_defaults(func=list_scheduled_submits)

    schedule_remove_parser = schedule_subparsers.add_parser(
        "remove",
        help="Remove a pending scheduled submit",
    )
    schedule_remove_parser.add_argument("schedule_id")
    schedule_remove_parser.add_argument("--yes", action="store_true")
    schedule_remove_parser.set_defaults(func=remove_scheduled_submit)

    delete_parser = subparsers.add_parser("delete", help="Delete a submitted task completely")
    add_common_manage_options(delete_parser)
    delete_parser.add_argument("--yes", action="store_true")
    delete_parser.add_argument("--no-stop-containers", dest="stop_containers", action="store_false")
    delete_parser.set_defaults(func=delete_task, stop_containers=True)

    restart_cleanup_parser = subparsers.add_parser("restart-cleanup", help=argparse.SUPPRESS)
    restart_cleanup_parser.add_argument("--yes", action="store_true")
    restart_cleanup_parser.add_argument("--dags-dir", default=str(DEFAULT_DAGS_DIR))
    restart_cleanup_parser.add_argument("--task-config-root", default=str(DEFAULT_TASK_CONFIG_ROOT))
    restart_cleanup_parser.add_argument(
        "--no-stop-containers",
        dest="stop_containers",
        action="store_false",
    )
    restart_cleanup_parser.set_defaults(func=platform_restart_cleanup, stop_containers=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    try:
        args.func(args)
    except TaskConfigError as exc:
        fail(str(exc), code=2)
    except Exception as exc:
        fail(str(exc), code=1)


if __name__ == "__main__":
    main()
