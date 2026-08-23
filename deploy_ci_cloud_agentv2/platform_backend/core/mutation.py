from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .task_store import task_paths


class PreconditionFailed(RuntimeError):
    """Raised when platform state changed after impact analysis/approval creation."""


@dataclass(frozen=True)
class MutationPrecondition:
    queue_sha256: str
    task_name: str = ""
    task_config_sha256: str = ""
    task_exists: bool | None = None
    active_task_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MutationPrecondition":
        value = dict(value or {})
        return cls(
            queue_sha256=str(value.get("queue_sha256") or ""),
            task_name=str(value.get("task_name") or ""),
            task_config_sha256=str(value.get("task_config_sha256") or ""),
            task_exists=value.get("task_exists"),
            active_task_name=str(value.get("active_task_name") or ""),
        )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def task_config_fingerprint(task_name: str, dags_dir: Path, task_config_root: Path) -> tuple[bool, str]:
    config_path = Path(task_paths(task_name, dags_dir=dags_dir, task_config_root=task_config_root)["config_file"])
    return config_path.is_file(), file_sha256(config_path)
