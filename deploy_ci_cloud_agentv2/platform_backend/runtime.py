"""Construction of the V2-owned platform runtime.

The builder composes the existing platform execution services inside the V2
package.  It does not select actions, decide whether evidence is sufficient,
or authorize writes; those responsibilities remain in the V2 Agent/Runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

from .core.settings import PlatformSettings
from .mcp.facade import PlatformMCPFacade, build_default_facade
from .rag.service import KnowledgeService


DEFAULT_RUNTIME_ROOT = "/home/ubuntu/project/autodrive_dataops_runtimev2"


def _platform_environment() -> dict[str, str]:
    """Return platform defaults without reading or copying secret values.

    Explicit deployment environment variables always win.  The defaults are
    filesystem locations only; credentials remain owned by the lower-level
    adapter and are never projected into Agent context or audit payloads.
    """

    env = dict(os.environ)
    runtime_root = Path(env.get("AUTODRIVE_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT))
    env.setdefault("PLATFORM_HOME", str(runtime_root))
    env.setdefault("AIRFLOW_HOME", str(runtime_root / "airflow"))
    env.setdefault("AIRFLOW_DAGS_DIR", str(runtime_root / "airflow" / "dags" / "data_center"))
    env.setdefault("AIRFLOW_TASK_CONFIG_ROOT", str(runtime_root / "opt_airflow" / "config" / "tasks"))
    env.setdefault("AIRFLOW_STATE_DIR", str(runtime_root / "state"))
    env.setdefault("AIRFLOW_TASK_QUEUE_DIR", str(runtime_root / "state" / "task_queue"))
    env.setdefault("AIRFLOW_GPU_LOCK_DIR", str(runtime_root / "state" / "gpu_locks"))
    # The migrated task manager reads AIRFLOW_BIN at module import time.  Keep
    # that legacy integration detail inside the V2-owned platform runtime and
    # prefer the active runtime venv when it exists; never fall back to a
    # host-specific source-tree path.
    airflow_candidate = runtime_root / "venv" / "bin" / "airflow"
    if "AIRFLOW_BIN" not in env and airflow_candidate.is_file():
        env["AIRFLOW_BIN"] = str(airflow_candidate)
    return env


def build_platform_facade() -> PlatformMCPFacade:
    """Build the concrete platform facade used by the V2 in-process gateway."""

    platform_env = _platform_environment()
    # The canonical task-manager module resolves a few non-secret deployment
    # settings from os.environ when it is lazily imported by a WRITE.  Apply
    # only the already-normalized platform paths/binary, never credentials.
    for key in (
        "AIRFLOW_BIN",
        "PLATFORM_HOME",
        "AIRFLOW_HOME",
        "AIRFLOW_DAGS_DIR",
        "AIRFLOW_TASK_CONFIG_ROOT",
        "AIRFLOW_STATE_DIR",
        "AIRFLOW_TASK_QUEUE_DIR",
        "AIRFLOW_GPU_LOCK_DIR",
    ):
        if key in platform_env:
            os.environ.setdefault(key, platform_env[key])
    settings = PlatformSettings.from_env(platform_env)
    source_dir = Path(
        os.environ.get(
            "AUTODRIVE_PLATFORM_KNOWLEDGE_DIR",
            str(Path(__file__).resolve().parent / "knowledge"),
        )
    )
    index_file = Path(
        os.environ.get(
            "AUTODRIVE_PLATFORM_KNOWLEDGE_INDEX",
            str(settings.state_dir / "v2_knowledge" / "index.json"),
        )
    )
    knowledge = KnowledgeService(source_dir=source_dir, index_file=index_file)
    return build_default_facade(settings=settings, knowledge_service=knowledge)
