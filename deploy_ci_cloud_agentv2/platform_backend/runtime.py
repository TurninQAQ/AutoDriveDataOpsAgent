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
    return env


def build_platform_facade() -> PlatformMCPFacade:
    """Build the concrete platform facade used by the V2 in-process gateway."""

    settings = PlatformSettings.from_env(_platform_environment())
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

