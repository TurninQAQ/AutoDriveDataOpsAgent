from __future__ import annotations

import fcntl
import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class GPUDeviceMemory:
    gpu_id: str
    total_mb: int
    used_mb: int
    free_mb: int


class GPURuntime(ABC):
    """Hardware-state boundary used by GPU scheduling.

    Reservation, exclusive/shared scheduling and queueing are intentionally not
    implemented here. A runtime only reports hardware/process state so the same
    scheduling algorithm can run against a real NVIDIA host or a deterministic
    local simulator.
    """

    @abstractmethod
    def list_devices(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_memory_info(self, gpu_id: str) -> GPUDeviceMemory:
        raise NotImplementedError

    @abstractmethod
    def process_alive(self, pid: int) -> bool:
        raise NotImplementedError


class NvidiaSMIRuntime(GPURuntime):
    """Real GPU runtime backed by nvidia-smi."""

    def __init__(
        self,
        nvidia_smi_bin: str = "nvidia-smi",
        timeout_sec: int = 30,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.nvidia_smi_bin = nvidia_smi_bin
        self.timeout_sec = int(timeout_sec)
        self.runner = runner

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        result = self.runner(
            [self.nvidia_smi_bin, *args],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi failed: {(result.stderr or result.stdout or '').strip()}"
            )
        return result

    def list_devices(self) -> list[str]:
        result = self._run(["--query-gpu=index", "--format=csv,noheader,nounits"])
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not devices:
            raise RuntimeError("nvidia-smi returned no GPU devices")
        return devices

    def get_memory_info(self, gpu_id: str) -> GPUDeviceMemory:
        result = self._run(
            [
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ]
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"nvidia-smi returned empty memory output for GPU {gpu_id}")
        parts = [part.strip() for part in lines[0].split(",")]
        if len(parts) != 3:
            raise RuntimeError(
                f"nvidia-smi returned invalid memory output for GPU {gpu_id}: {lines[0]}"
            )
        try:
            total_mb, used_mb, free_mb = (int(value) for value in parts)
        except ValueError as exc:
            raise RuntimeError(
                f"nvidia-smi returned non-integer memory output for GPU {gpu_id}: {lines[0]}"
            ) from exc
        return GPUDeviceMemory(str(gpu_id), total_mb, used_mb, free_mb)

    def process_alive(self, pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False


class SimulatedGPURuntime(GPURuntime):
    """File-backed deterministic GPU simulator.

    State is intentionally process-safe so Airflow tasks or local test workers can
    share one simulated hardware view. ``external_used_mb`` represents memory
    consumed outside the platform reservation system. Platform reservations remain
    stored in the existing gpu_*.lock files and are subtracted by GPUAllocator.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        state_path: str | Path,
        devices: list[dict[str, Any]] | None = None,
        fallback_to_os_processes: bool = True,
    ):
        self.state_path = Path(state_path)
        self.fallback_to_os_processes = bool(fallback_to_os_processes)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if devices is not None and not self.state_path.exists():
            self.initialize(devices, overwrite=False)

    @staticmethod
    def _normalize_device(item: dict[str, Any]) -> dict[str, int]:
        gpu_id = str(item.get("id", item.get("gpu_id", ""))).strip()
        if not gpu_id:
            raise ValueError("Simulated GPU device requires id")
        total_mb = int(item.get("total_memory_mb", item.get("total_mb", 0)))
        used_mb = int(item.get("external_used_mb", item.get("used_mb", 0)))
        if total_mb <= 0:
            raise ValueError(f"Simulated GPU {gpu_id} total memory must be positive")
        if used_mb < 0 or used_mb > total_mb:
            raise ValueError(
                f"Simulated GPU {gpu_id} external_used_mb must be within [0, {total_mb}]"
            )
        return {"gpu_id": gpu_id, "total_mb": total_mb, "external_used_mb": used_mb}

    def _default_state(self) -> dict[str, Any]:
        return {"version": self.STATE_VERSION, "devices": {}, "processes": {}}

    def _locked_state(self, write: bool, mutate=None):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            try:
                state = json.loads(raw) if raw else self._default_state()
            except json.JSONDecodeError:
                state = self._default_state()
            if not isinstance(state, dict):
                state = self._default_state()
            state.setdefault("version", self.STATE_VERSION)
            state.setdefault("devices", {})
            state.setdefault("processes", {})
            result = mutate(state) if mutate is not None else state
            if write:
                handle.seek(0)
                handle.truncate()
                json.dump(state, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return result

    def initialize(self, devices: list[dict[str, Any]], overwrite: bool = True) -> None:
        normalized = [self._normalize_device(item) for item in devices]

        def mutate(state):
            if state.get("devices") and not overwrite:
                return None
            state.clear()
            state.update(self._default_state())
            state["devices"] = {
                item["gpu_id"]: {
                    "total_mb": item["total_mb"],
                    "external_used_mb": item["external_used_mb"],
                }
                for item in normalized
            }
            return None

        self._locked_state(write=True, mutate=mutate)

    def list_devices(self) -> list[str]:
        state = self._locked_state(write=False)
        return sorted((str(key) for key in state["devices"]), key=self._sort_key)

    @staticmethod
    def _sort_key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    def get_memory_info(self, gpu_id: str) -> GPUDeviceMemory:
        gpu_id = str(gpu_id)
        state = self._locked_state(write=False)
        item = state["devices"].get(gpu_id)
        if item is None:
            raise RuntimeError(f"Simulated GPU does not exist: {gpu_id}")
        total_mb = int(item.get("total_mb", 0))
        used_mb = int(item.get("external_used_mb", 0))
        free_mb = max(0, total_mb - used_mb)
        return GPUDeviceMemory(gpu_id, total_mb, used_mb, free_mb)

    def process_alive(self, pid: int) -> bool:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return False
        state = self._locked_state(write=False)
        key = str(pid_int)
        if key in state["processes"]:
            return bool(state["processes"][key])
        if not self.fallback_to_os_processes:
            return False
        try:
            os.kill(pid_int, 0)
            return True
        except OSError:
            return False

    def set_external_used_mb(self, gpu_id: str, used_mb: int) -> None:
        gpu_id = str(gpu_id)
        used_mb = int(used_mb)

        def mutate(state):
            item = state["devices"].get(gpu_id)
            if item is None:
                raise RuntimeError(f"Simulated GPU does not exist: {gpu_id}")
            total_mb = int(item.get("total_mb", 0))
            if used_mb < 0 or used_mb > total_mb:
                raise ValueError(f"used_mb must be within [0, {total_mb}]")
            item["external_used_mb"] = used_mb

        self._locked_state(write=True, mutate=mutate)

    def set_process_alive(self, pid: int, alive: bool) -> None:
        pid = int(pid)

        def mutate(state):
            state["processes"][str(pid)] = bool(alive)

        self._locked_state(write=True, mutate=mutate)

    def clear_process_override(self, pid: int) -> None:
        pid = int(pid)

        def mutate(state):
            state["processes"].pop(str(pid), None)

        self._locked_state(write=True, mutate=mutate)

    def snapshot(self) -> dict[str, Any]:
        state = self._locked_state(write=False)
        devices = []
        for gpu_id in self.list_devices():
            devices.append(asdict(self.get_memory_info(gpu_id)))
        return {
            "version": state.get("version", self.STATE_VERSION),
            "state_path": str(self.state_path),
            "devices": devices,
            "processes": dict(state.get("processes") or {}),
        }


def load_simulated_devices_from_yaml(config_path: str | Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project already depends on PyYAML
        raise RuntimeError("PyYAML is required to load simulated GPU config") from exc
    path = Path(config_path)
    if not path.is_file():
        raise RuntimeError(f"Simulated GPU config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    devices = payload.get("gpus") if isinstance(payload, dict) else None
    if not isinstance(devices, list) or not devices:
        raise RuntimeError(f"Simulated GPU config requires non-empty 'gpus' list: {path}")
    return devices


def create_gpu_runtime_from_env(env: dict[str, str] | None = None) -> GPURuntime:
    env = env or os.environ
    mode = str(env.get("PLATFORM_GPU_RUNTIME", "nvidia")).strip().lower()
    if mode in {"nvidia", "nvidia-smi", "real"}:
        return NvidiaSMIRuntime(
            nvidia_smi_bin=env.get("NVIDIA_SMI_BIN", "nvidia-smi"),
            timeout_sec=int(env.get("PLATFORM_GPU_QUERY_TIMEOUT_SEC", "30")),
        )
    if mode in {"sim", "simulated", "fake"}:
        state_path = env.get("PLATFORM_GPU_SIM_STATE")
        if not state_path:
            platform_home = env.get("PLATFORM_HOME") or str(Path(env.get("AIRFLOW_HOME", "/home/cidi/airflow")).parent)
            state_path = str(Path(platform_home) / "state" / "gpu_simulator.json")
        runtime = SimulatedGPURuntime(
            state_path,
            fallback_to_os_processes=str(
                env.get("PLATFORM_GPU_SIM_FALLBACK_OS_PROCESS", "1")
            ).strip().lower()
            not in {"0", "false", "no", "off"},
        )
        if not runtime.list_devices():
            config_path = env.get("PLATFORM_GPU_SIM_CONFIG")
            if not config_path:
                raise RuntimeError(
                    "Simulated GPU state is empty. Set PLATFORM_GPU_SIM_CONFIG or initialize the state first."
                )
            runtime.initialize(load_simulated_devices_from_yaml(config_path), overwrite=True)
        return runtime
    raise RuntimeError(
        f"Unsupported PLATFORM_GPU_RUNTIME={mode!r}; expected 'nvidia' or 'simulated'"
    )
