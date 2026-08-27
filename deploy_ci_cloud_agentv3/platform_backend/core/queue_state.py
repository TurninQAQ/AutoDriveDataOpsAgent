from __future__ import annotations

import json

from .config import normalize_preempt_config, normalize_task_priority_config
from .errors import TaskConfigError
from .task_store import load_task_config

TASK_QUEUE_SCHEMA_VERSION = 2
DEFAULT_TASK_PRIORITY = 100

def warn(message):
    print(f"[WARN] {message}", flush=True)

def read_task_queue_file(queue_file):
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
    return state

def write_task_queue_file(queue_file, state):
    state["version"] = TASK_QUEUE_SCHEMA_VERSION
    queue_file.seek(0)
    queue_file.truncate()
    json.dump(state, queue_file, ensure_ascii=False, sort_keys=True)
    queue_file.write("\n")
    queue_file.flush()

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

def apply_priority_config_to_queue_entry(entry, config):
    priority_config = normalize_task_priority_config(config)
    entry.update(priority_config)
    entry.update(normalize_preempt_config(config))
    return entry

def refresh_queue_entry_priority(entry, task_config_root=None):
    task_name = entry.get("task_name")
    if not task_name:
        return entry
    try:
        _, config = load_task_config(task_name, task_config_root=task_config_root)
        apply_priority_config_to_queue_entry(entry, config)
    except TaskConfigError as exc:
        warn(f"Skip priority refresh for {task_name}: {exc}")
    return entry

def refresh_task_queue_priorities(state, task_config_root=None):
    active = state.get("active") or None
    if active:
        state["active"] = refresh_queue_entry_priority(
            active,
            task_config_root=task_config_root,
        )
    state["queue"] = sort_queued_tasks(
        [
            refresh_queue_entry_priority(entry, task_config_root=task_config_root)
            for entry in state.get("queue") or []
            if isinstance(entry, dict)
        ]
    )
    return state

