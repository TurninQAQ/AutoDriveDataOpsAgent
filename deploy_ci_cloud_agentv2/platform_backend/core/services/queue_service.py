from __future__ import annotations

import fcntl
from pathlib import Path

from ..queue_state import read_task_queue_file


class QueueService:
    """Read-side service for the task-level global queue.

    V0.1 keeps queue transition orchestration in the existing runtime path to avoid
    changing preemption/recovery semantics. This service provides a stable read
    boundary that MCP/Agent layers can consume in later versions.
    """

    def __init__(self, queue_file: str | Path):
        self.queue_file = Path(queue_file)

    def snapshot(self) -> dict:
        if not self.queue_file.exists():
            return {"version": 2, "active": None, "queue": []}
        with self.queue_file.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return read_task_queue_file(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def task_status(self, task_name: str) -> dict:
        state = self.snapshot()
        active = state.get("active") or {}
        if active.get("task_name") == task_name:
            return {"location": "active", "position": 0, "entry": active}
        for index, entry in enumerate(state.get("queue") or [], start=1):
            if isinstance(entry, dict) and entry.get("task_name") == task_name:
                return {"location": "queued", "position": index, "entry": entry}
        return {"location": "not_found", "position": -1, "entry": None}
