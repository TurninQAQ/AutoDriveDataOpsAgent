from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..gateways.gpu_runtime import GPURuntime


@dataclass(frozen=True)
class GPUAllocation:
    gpu_id: str
    token: str
    required_mb: int
    exclusive: bool


class GPUAllocationTimeout(RuntimeError):
    pass


class GPUAllocator:
    """Shared GPU reservation/scheduling algorithm.

    The allocator owns reservation file semantics while GPURuntime only reports
    hardware/process state. This keeps real and simulated execution on exactly the
    same allocation rules.
    """

    def __init__(
        self,
        runtime: GPURuntime,
        lock_dir: str | Path,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
        time_ns_fn: Callable[[], int] = time.time_ns,
        pid_provider: Callable[[], int] = os.getpid,
        logger: Callable[[str], None] | None = None,
    ):
        self.runtime = runtime
        self.lock_dir = Path(lock_dir)
        self.sleep_fn = sleep_fn
        self.now_fn = now_fn
        self.time_ns_fn = time_ns_fn
        self.pid_provider = pid_provider
        self.logger = logger or (lambda _message: None)

    @staticmethod
    def _read_state(handle) -> dict[str, Any]:
        handle.seek(0)
        raw = handle.read().strip()
        if not raw:
            return {"reservations": {}}
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = {"reservations": {}}
        if not isinstance(state, dict):
            state = {"reservations": {}}
        state.setdefault("reservations", {})
        if not isinstance(state["reservations"], dict):
            state["reservations"] = {}
        return state

    @staticmethod
    def _write_state(handle, state: dict[str, Any]) -> None:
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()

    def _lock_path(self, gpu_id: str) -> Path:
        return self.lock_dir / f"gpu_{gpu_id}.lock"

    def _prune_dead(self, state: dict[str, Any]) -> int:
        reservations = state.get("reservations") or {}
        removed = 0
        for token, item in list(reservations.items()):
            if not self.runtime.process_alive(item.get("pid")):
                reservations.pop(token, None)
                removed += 1
        return removed

    @staticmethod
    def _reserved_mb(state: dict[str, Any]) -> int:
        return sum(
            max(0, int(item.get("required_mb", 0)))
            for item in (state.get("reservations") or {}).values()
        )

    @staticmethod
    def _has_exclusive(state: dict[str, Any]) -> bool:
        return any(bool(item.get("exclusive")) for item in (state.get("reservations") or {}).values())

    def try_acquire(
        self,
        gpu_ids: list[str],
        *,
        stage: str,
        required_mb: int,
        exclusive: bool,
        exclusive_idle_used_max_mb: int,
        task_name: str = "",
        dag_id: str = "",
        run_id: str = "",
        dataset_name: str = "",
    ) -> GPUAllocation | None:
        if not gpu_ids:
            raise ValueError("GPU pool must not be empty")
        required_mb = int(required_mb)
        if required_mb <= 0:
            raise ValueError("required_mb must be positive")
        exclusive_idle_used_max_mb = int(exclusive_idle_used_max_mb)
        if exclusive_idle_used_max_mb < 0:
            raise ValueError("exclusive_idle_used_max_mb must be non-negative")
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        for gpu_id in [str(item) for item in gpu_ids]:
            lock_path = self._lock_path(gpu_id)
            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                state = self._read_state(handle)
                removed = self._prune_dead(state)
                if removed:
                    self.logger(f"gpu={gpu_id} pruned_dead_reservations={removed}")
                reservations = state["reservations"]

                if exclusive and reservations:
                    self.logger(
                        f"gpu={gpu_id} skip=exclusive_requires_empty active={len(reservations)}"
                    )
                    self._write_state(handle, state)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    continue

                if not exclusive and self._has_exclusive(state):
                    self.logger(f"gpu={gpu_id} skip=exclusive_active")
                    self._write_state(handle, state)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    continue

                try:
                    memory = self.runtime.get_memory_info(gpu_id)
                except Exception as exc:
                    self.logger(f"gpu={gpu_id} skip=query_failed error={exc}")
                    self._write_state(handle, state)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    continue

                if exclusive and memory.used_mb > exclusive_idle_used_max_mb:
                    self.logger(
                        f"gpu={gpu_id} skip=exclusive_requires_idle used_mb={memory.used_mb} "
                        f"idle_used_max_mb={exclusive_idle_used_max_mb}"
                    )
                    self._write_state(handle, state)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    continue

                reserved_mb = self._reserved_mb(state)
                effective_free_mb = memory.free_mb - reserved_mb
                self.logger(
                    f"gpu={gpu_id} total_mb={memory.total_mb} free_mb={memory.free_mb} "
                    f"reserved_mb={reserved_mb} effective_free_mb={effective_free_mb} "
                    f"required_mb={required_mb}"
                )
                if effective_free_mb < required_mb:
                    self.logger(f"gpu={gpu_id} skip=insufficient_memory")
                    self._write_state(handle, state)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    continue

                pid = int(self.pid_provider())
                token = f"{pid}-{self.time_ns_fn()}"
                reservations[token] = {
                    "pid": pid,
                    "stage": stage,
                    "exclusive": bool(exclusive),
                    "task_name": task_name,
                    "dag_id": dag_id,
                    "run_id": run_id,
                    "dataset_name": dataset_name,
                    "required_mb": required_mb,
                    "ts": self.now_fn(),
                }
                self._write_state(handle, state)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                self.logger(
                    f"gpu={gpu_id} assigned stage={stage} mode={'exclusive' if exclusive else 'shared'} "
                    f"required_mb={required_mb} token={token}"
                )
                return GPUAllocation(gpu_id, token, required_mb, bool(exclusive))
        return None

    def acquire(
        self,
        gpu_ids: list[str],
        *,
        stage: str,
        required_mb: int,
        exclusive: bool,
        exclusive_idle_used_max_mb: int,
        wait_interval_sec: float,
        task_name: str = "",
        dag_id: str = "",
        run_id: str = "",
        dataset_name: str = "",
        before_scan: Callable[[], None] | None = None,
        max_wait_sec: float | None = None,
    ) -> GPUAllocation:
        wait_interval_sec = float(wait_interval_sec)
        if wait_interval_sec <= 0:
            raise ValueError("wait_interval_sec must be positive")
        start = self.now_fn()
        while True:
            if before_scan is not None:
                before_scan()
            allocation = self.try_acquire(
                gpu_ids,
                stage=stage,
                required_mb=required_mb,
                exclusive=exclusive,
                exclusive_idle_used_max_mb=exclusive_idle_used_max_mb,
                task_name=task_name,
                dag_id=dag_id,
                run_id=run_id,
                dataset_name=dataset_name,
            )
            if allocation is not None:
                return allocation
            if max_wait_sec is not None and self.now_fn() - start >= float(max_wait_sec):
                raise GPUAllocationTimeout(
                    f"No GPU became available for stage={stage} within {max_wait_sec}s"
                )
            self.logger(f"gpu_wait interval_sec={wait_interval_sec} reason=no_available_gpu")
            self.sleep_fn(wait_interval_sec)

    def release(self, gpu_id: str | None, token: str | None) -> dict[str, Any]:
        if gpu_id is None or token is None:
            return {}
        lock_path = self._lock_path(str(gpu_id))
        if not lock_path.exists():
            return {}
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state = self._read_state(handle)
            item = state["reservations"].pop(str(token), {})
            self._write_state(handle, state)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self.logger(f"gpu={gpu_id} released token={token} required_mb={item.get('required_mb')}")
        return item

    def reservations(self, gpu_id: str, cleanup_dead: bool = False) -> dict[str, dict[str, Any]]:
        lock_path = self._lock_path(str(gpu_id))
        if not lock_path.exists():
            return {}
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state = self._read_state(handle)
            if cleanup_dead:
                self._prune_dead(state)
                self._write_state(handle, state)
            result = json.loads(json.dumps(state["reservations"]))
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return result

    def all_reservations(self, cleanup_dead: bool = False) -> dict[str, dict[str, dict[str, Any]]]:
        result: dict[str, dict[str, dict[str, Any]]] = {}
        if not self.lock_dir.is_dir():
            return result
        for path in sorted(self.lock_dir.glob("gpu_*.lock")):
            gpu_id = path.stem[len("gpu_") :]
            result[gpu_id] = self.reservations(gpu_id, cleanup_dead=cleanup_dead)
        return result

    def remove_for_task(self, task_name: str, dataset_names: list[str] | None = None) -> int:
        selected = set(dataset_names or [])
        removed = 0
        if not self.lock_dir.is_dir():
            return removed
        for path in sorted(self.lock_dir.glob("gpu_*.lock")):
            with path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                state = self._read_state(handle)
                for token, item in list(state["reservations"].items()):
                    if item.get("task_name") != task_name:
                        continue
                    if selected and item.get("dataset_name") not in selected:
                        continue
                    state["reservations"].pop(token, None)
                    removed += 1
                self._write_state(handle, state)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return removed
