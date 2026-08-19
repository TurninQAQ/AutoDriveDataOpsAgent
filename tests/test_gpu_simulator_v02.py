from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from platform_core.gateways.gpu_runtime import (
    GPUDeviceMemory,
    NvidiaSMIRuntime,
    SimulatedGPURuntime,
    create_gpu_runtime_from_env,
)
from platform_core.services.gpu_allocator import GPUAllocationTimeout, GPUAllocator


def make_runtime(tmp_path: Path, devices=None, fallback=False):
    devices = devices or [
        {"id": 0, "total_memory_mb": 48000, "external_used_mb": 0},
        {"id": 1, "total_memory_mb": 48000, "external_used_mb": 0},
    ]
    runtime = SimulatedGPURuntime(
        tmp_path / "sim_state.json",
        fallback_to_os_processes=fallback,
    )
    runtime.initialize(devices)
    return runtime


def make_allocator(tmp_path: Path, runtime: SimulatedGPURuntime, pid=1001):
    runtime.set_process_alive(pid, True)
    return GPUAllocator(
        runtime,
        tmp_path / "gpu_locks",
        pid_provider=lambda: pid,
        sleep_fn=lambda _seconds: None,
    )


def test_simulated_runtime_memory_and_persistence(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    assert runtime.list_devices() == ["0", "1"]
    assert runtime.get_memory_info("0") == GPUDeviceMemory("0", 48000, 0, 48000)

    runtime.set_external_used_mb("0", 12345)
    reloaded = SimulatedGPURuntime(tmp_path / "sim_state.json", fallback_to_os_processes=False)
    assert reloaded.get_memory_info("0") == GPUDeviceMemory("0", 48000, 12345, 35655)


def test_shared_gpu_reservations_use_same_real_algorithm(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    allocator = make_allocator(tmp_path, runtime)

    first = allocator.try_acquire(
        ["0"], stage="occ", required_mb=4000, exclusive=False,
        exclusive_idle_used_max_mb=512, task_name="task_a", dataset_name="clip_1"
    )
    second = allocator.try_acquire(
        ["0"], stage="occ", required_mb=4000, exclusive=False,
        exclusive_idle_used_max_mb=512, task_name="task_b", dataset_name="clip_2"
    )

    assert first is not None and second is not None
    assert first.gpu_id == second.gpu_id == "0"
    reservations = allocator.reservations("0")
    assert sum(item["required_mb"] for item in reservations.values()) == 8000
    assert all(item["exclusive"] is False for item in reservations.values())


def test_exclusive_request_rejects_existing_shared_reservation(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    allocator = make_allocator(tmp_path, runtime)
    shared = allocator.try_acquire(
        ["0"], stage="occ", required_mb=4000, exclusive=False,
        exclusive_idle_used_max_mb=512
    )
    assert shared is not None
    exclusive = allocator.try_acquire(
        ["0"], stage="segment", required_mb=24000, exclusive=True,
        exclusive_idle_used_max_mb=512
    )
    assert exclusive is None


def test_shared_request_rejects_existing_exclusive_reservation(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    allocator = make_allocator(tmp_path, runtime)
    exclusive = allocator.try_acquire(
        ["0"], stage="segment", required_mb=24000, exclusive=True,
        exclusive_idle_used_max_mb=512
    )
    assert exclusive is not None
    shared = allocator.try_acquire(
        ["0"], stage="occ", required_mb=4000, exclusive=False,
        exclusive_idle_used_max_mb=512
    )
    assert shared is None


def test_insufficient_memory_rejects_allocation(tmp_path: Path):
    runtime = make_runtime(
        tmp_path,
        [{"id": 0, "total_memory_mb": 48000, "external_used_mb": 30000}],
    )
    allocator = make_allocator(tmp_path, runtime)
    result = allocator.try_acquire(
        ["0"], stage="od", required_mb=24000, exclusive=False,
        exclusive_idle_used_max_mb=512
    )
    assert result is None


def test_stale_reservation_is_pruned_using_simulated_pid_state(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    allocator = make_allocator(tmp_path, runtime, pid=111)
    stale = allocator.try_acquire(
        ["0"], stage="occ", required_mb=4000, exclusive=False,
        exclusive_idle_used_max_mb=512, task_name="old_task"
    )
    assert stale is not None
    runtime.set_process_alive(111, False)

    runtime.set_process_alive(222, True)
    fresh_allocator = GPUAllocator(
        runtime,
        tmp_path / "gpu_locks",
        pid_provider=lambda: 222,
        sleep_fn=lambda _seconds: None,
    )
    fresh = fresh_allocator.try_acquire(
        ["0"], stage="segment", required_mb=24000, exclusive=True,
        exclusive_idle_used_max_mb=512, task_name="new_task"
    )
    assert fresh is not None
    reservations = fresh_allocator.reservations("0")
    assert stale.token not in reservations
    assert fresh.token in reservations


def test_multi_gpu_selection_skips_insufficient_gpu(tmp_path: Path):
    runtime = make_runtime(
        tmp_path,
        [
            {"id": 0, "total_memory_mb": 48000, "external_used_mb": 40000},
            {"id": 1, "total_memory_mb": 48000, "external_used_mb": 1000},
        ],
    )
    allocator = make_allocator(tmp_path, runtime)
    result = allocator.try_acquire(
        ["0", "1"], stage="segment", required_mb=24000, exclusive=False,
        exclusive_idle_used_max_mb=512
    )
    assert result is not None
    assert result.gpu_id == "1"


def test_exclusive_gpu_requires_hardware_idle_threshold(tmp_path: Path):
    runtime = make_runtime(
        tmp_path,
        [{"id": 0, "total_memory_mb": 48000, "external_used_mb": 1024}],
    )
    allocator = make_allocator(tmp_path, runtime)
    assert allocator.try_acquire(
        ["0"], stage="segment", required_mb=24000, exclusive=True,
        exclusive_idle_used_max_mb=512
    ) is None

    runtime.set_external_used_mb("0", 256)
    assert allocator.try_acquire(
        ["0"], stage="segment", required_mb=24000, exclusive=True,
        exclusive_idle_used_max_mb=512
    ) is not None


def test_gpu_wait_has_deterministic_timeout(tmp_path: Path):
    runtime = make_runtime(
        tmp_path,
        [{"id": 0, "total_memory_mb": 24000, "external_used_mb": 23000}],
    )
    # Drive virtual time forward without sleeping in real time.
    clock = {"now": 0.0}
    def sleep_fn(seconds):
        clock["now"] += seconds

    runtime.set_process_alive(123, True)
    allocator = GPUAllocator(
        runtime,
        tmp_path / "gpu_locks",
        pid_provider=lambda: 123,
        now_fn=lambda: clock["now"],
        sleep_fn=sleep_fn,
    )
    with pytest.raises(GPUAllocationTimeout):
        allocator.acquire(
            ["0"], stage="segment", required_mb=24000, exclusive=True,
            exclusive_idle_used_max_mb=512, wait_interval_sec=1, max_wait_sec=3
        )
    assert clock["now"] == 3


def test_release_removes_reservation(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    allocator = make_allocator(tmp_path, runtime)
    allocation = allocator.try_acquire(
        ["0"], stage="occ", required_mb=4000, exclusive=False,
        exclusive_idle_used_max_mb=512
    )
    assert allocation is not None
    removed = allocator.release(allocation.gpu_id, allocation.token)
    assert removed["required_mb"] == 4000
    assert allocator.reservations("0") == {}


def test_runtime_factory_defaults_to_nvidia_and_supports_simulated(tmp_path: Path):
    default = create_gpu_runtime_from_env({})
    assert isinstance(default, NvidiaSMIRuntime)

    config = tmp_path / "gpus.yaml"
    config.write_text(
        "gpus:\n  - id: 0\n    total_memory_mb: 48000\n    external_used_mb: 1000\n",
        encoding="utf-8",
    )
    state = tmp_path / "factory_state.json"
    simulated = create_gpu_runtime_from_env(
        {
            "PLATFORM_GPU_RUNTIME": "simulated",
            "PLATFORM_GPU_SIM_STATE": str(state),
            "PLATFORM_GPU_SIM_CONFIG": str(config),
            "PLATFORM_GPU_SIM_FALLBACK_OS_PROCESS": "0",
        }
    )
    assert isinstance(simulated, SimulatedGPURuntime)
    assert simulated.get_memory_info("0").free_mb == 47000


def test_allocator_state_is_compatible_with_existing_lock_format(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    allocator = make_allocator(tmp_path, runtime)
    allocation = allocator.try_acquire(
        ["0"], stage="segment", required_mb=24000, exclusive=True,
        exclusive_idle_used_max_mb=512,
        task_name="task_a", dag_id="dag_a", run_id="run_a", dataset_name="clip_001"
    )
    assert allocation is not None
    raw = json.loads((tmp_path / "gpu_locks" / "gpu_0.lock").read_text(encoding="utf-8"))
    item = raw["reservations"][allocation.token]
    assert item["task_name"] == "task_a"
    assert item["dataset_name"] == "clip_001"
    assert item["stage"] == "segment"
    assert item["required_mb"] == 24000
    assert item["exclusive"] is True
