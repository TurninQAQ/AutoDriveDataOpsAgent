from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformSettings:
    platform_home: Path
    airflow_home: Path
    dags_dir: Path
    task_config_root: Path
    state_dir: Path
    queue_file: Path
    gpu_lock_dir: Path
    airflow_api_base: str
    airflow_api_user: str
    airflow_api_password: str
    airflow_api_token: str
    airflow_password_file: Path
    airflow_bin: str
    api_timeout_sec: int = 10

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PlatformSettings":
        env = env or os.environ
        airflow_home = Path(env.get("AIRFLOW_HOME", "/home/cidi/airflow"))
        platform_home = Path(env.get("PLATFORM_HOME", str(airflow_home.parent)))
        state_dir = Path(env.get("AIRFLOW_STATE_DIR", str(platform_home / "state")))
        task_queue_dir = Path(
            env.get("AIRFLOW_TASK_QUEUE_DIR", str(state_dir / "task_queue"))
        )
        return cls(
            platform_home=platform_home,
            airflow_home=airflow_home,
            dags_dir=Path(
                env.get("AIRFLOW_DAGS_DIR", str(airflow_home / "dags" / "data_center"))
            ),
            task_config_root=Path(
                env.get("AIRFLOW_TASK_CONFIG_ROOT", str(platform_home / "opt_airflow" / "config" / "tasks"))
            ),
            state_dir=state_dir,
            queue_file=task_queue_dir / "queue.lock",
            gpu_lock_dir=Path(
                env.get("AIRFLOW_GPU_LOCK_DIR", str(state_dir / "gpu_locks"))
            ),
            airflow_api_base=env.get("AIRFLOW_API_BASE", "http://127.0.0.1:8080").rstrip("/"),
            airflow_api_user=(
                env.get("AIRFLOW_API_USER")
                or env.get("AIRFLOW_ADMIN_USER")
                or "admin"
            ),
            airflow_api_password=(
                env.get("AIRFLOW_API_PASSWORD")
                or env.get("AIRFLOW_ADMIN_PASSWORD")
                or ""
            ),
            airflow_api_token=env.get("AIRFLOW_API_TOKEN", ""),
            airflow_password_file=Path(
                env.get(
                    "AIRFLOW_PASSWORD_FILE",
                    str(airflow_home / "simple_auth_manager_passwords.json.generated"),
                )
            ),
            airflow_bin=env.get("AIRFLOW_BIN", "airflow"),
            api_timeout_sec=int(env.get("AIRFLOW_API_TIMEOUT_SEC", "10")),
        )
