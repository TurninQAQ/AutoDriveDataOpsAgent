from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..gateways import gpu_reservations
from ..gateways.gpu_reservations import GPUReservationStore
from .gpu_allocator import GPUAllocator


class GPUService:
    def __init__(self, runtime=None, lock_dir: str | Path | None = None):
        self.runtime = runtime
        self.lock_dir = Path(lock_dir) if lock_dir is not None else None

    def _store(self) -> GPUReservationStore:
        return GPUReservationStore(
            self.lock_dir or gpu_reservations.GPU_LOCK_DIR,
            runtime=self.runtime,
        )

    def reservations(self, cleanup_dead=False):
        return self._store().list_reservations(cleanup_dead=cleanup_dead)

    def task_reservations(self, task_name, dataset_names, cleanup_dead=False):
        return self._store().active_task_reservations(
            task_name, dataset_names, cleanup_dead=cleanup_dead
        )

    def wait_for_task_reservations(self, task_name, dataset_names, timeout_sec=60):
        return self._store().wait_for_task_reservations(
            task_name, dataset_names, timeout_sec=timeout_sec
        )

    def allocator(self) -> GPUAllocator:
        if self.runtime is None or self.lock_dir is None:
            raise RuntimeError("GPUService allocator requires runtime and lock_dir")
        return GPUAllocator(self.runtime, self.lock_dir)

    def device_snapshot(self):
        if self.runtime is None:
            raise RuntimeError("GPUService device_snapshot requires runtime")
        return [asdict(self.runtime.get_memory_info(gpu_id)) for gpu_id in self.runtime.list_devices()]
