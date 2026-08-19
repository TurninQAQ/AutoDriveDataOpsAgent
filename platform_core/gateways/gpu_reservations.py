from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/home/cidi/airflow")
AIRFLOW_RUN_HOME = Path(os.environ.get("PLATFORM_HOME", str(Path(AIRFLOW_HOME).parent)))
AIRFLOW_STATE_DIR = Path(os.environ.get("AIRFLOW_STATE_DIR", str(AIRFLOW_RUN_HOME / "state")))
GPU_LOCK_DIR = Path(os.environ.get("AIRFLOW_GPU_LOCK_DIR", str(AIRFLOW_STATE_DIR / "gpu_locks")))


def info(message):
    print(f"[INFO] {message}", flush=True)


def warn(message):
    print(f"[WARN] {message}", flush=True)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


class GPUReservationStore:
    """Read/cleanup boundary for the existing gpu_*.lock reservation format."""

    def __init__(self, lock_dir: str | Path, runtime=None):
        self.lock_dir = Path(lock_dir)
        self.runtime = runtime

    def _process_alive(self, pid) -> bool:
        if self.runtime is not None:
            return self.runtime.process_alive(pid)
        return pid_alive(pid)

    def active_task_reservations(self, task_name, dataset_names, cleanup_dead=False):
        matches = []
        if not self.lock_dir.is_dir():
            return matches

        selected = set(dataset_names)
        for lock_path in sorted(self.lock_dir.glob("gpu_*.lock")):
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                lock_file.seek(0)
                try:
                    state = json.loads(lock_file.read().strip() or "{}")
                except json.JSONDecodeError:
                    state = {}
                reservations = state.setdefault("reservations", {})
                changed = False
                for token, item in list(reservations.items()):
                    if cleanup_dead and not self._process_alive(item.get("pid")):
                        reservations.pop(token, None)
                        changed = True
                        continue
                    if item.get("task_name") != task_name:
                        continue
                    if selected and item.get("dataset_name") not in selected:
                        continue
                    matches.append((lock_path.name, token, item))
                if changed:
                    lock_file.seek(0)
                    lock_file.truncate()
                    json.dump(state, lock_file, sort_keys=True)
                    lock_file.write("\n")
                    lock_file.flush()
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return matches

    def list_reservations(self, cleanup_dead=False):
        """Return all active reservations as (lock_name, token, item) tuples."""
        matches = []
        if not self.lock_dir.is_dir():
            return matches

        for lock_path in sorted(self.lock_dir.glob("gpu_*.lock")):
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                lock_file.seek(0)
                try:
                    state = json.loads(lock_file.read().strip() or "{}")
                except json.JSONDecodeError:
                    state = {}
                reservations = state.setdefault("reservations", {})
                changed = False
                for token, item in list(reservations.items()):
                    if cleanup_dead and not self._process_alive(item.get("pid")):
                        reservations.pop(token, None)
                        changed = True
                        continue
                    matches.append((lock_path.name, token, item))
                if changed:
                    lock_file.seek(0)
                    lock_file.truncate()
                    json.dump(state, lock_file, sort_keys=True)
                    lock_file.write("\n")
                    lock_file.flush()
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return matches

    def wait_for_task_reservations(self, task_name, dataset_names, timeout_sec=60):
        deadline = time.time() + timeout_sec
        while True:
            matches = self.active_task_reservations(
                task_name, dataset_names, cleanup_dead=True
            )
            if not matches:
                info("No matching GPU reservations remain")
                return 0
            if time.time() >= deadline:
                warn(
                    "GPU reservations still active after container cleanup: "
                    + ", ".join(f"{lock}:{token}" for lock, token, _ in matches)
                )
                return len(matches)
            time.sleep(2)


def active_task_reservations(task_name, dataset_names, cleanup_dead=False, runtime=None):
    return GPUReservationStore(GPU_LOCK_DIR, runtime=runtime).active_task_reservations(
        task_name, dataset_names, cleanup_dead=cleanup_dead
    )


def wait_for_task_reservations(task_name, dataset_names, timeout_sec=60, runtime=None):
    return GPUReservationStore(GPU_LOCK_DIR, runtime=runtime).wait_for_task_reservations(
        task_name, dataset_names, timeout_sec=timeout_sec
    )
