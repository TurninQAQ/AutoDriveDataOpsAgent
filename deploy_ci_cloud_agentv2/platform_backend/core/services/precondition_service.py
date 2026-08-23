from __future__ import annotations

from pathlib import Path

from ..mutation import (
    MutationPrecondition,
    PreconditionFailed,
    canonical_sha256,
    task_config_fingerprint,
)
from .queue_service import QueueService


class PreconditionService:
    """Capture and validate deterministic state fingerprints for write operations."""

    def __init__(self, queue_service: QueueService, dags_dir: str | Path, task_config_root: str | Path):
        self.queue_service = queue_service
        self.dags_dir = Path(dags_dir)
        self.task_config_root = Path(task_config_root)

    def capture(self, task_name: str = "") -> MutationPrecondition:
        queue = self.queue_service.snapshot()
        active = queue.get("active") or {}
        exists = None
        config_hash = ""
        if task_name:
            exists, config_hash = task_config_fingerprint(
                task_name, self.dags_dir, self.task_config_root
            )
        return MutationPrecondition(
            queue_sha256=canonical_sha256(queue),
            task_name=task_name,
            task_config_sha256=config_hash,
            task_exists=exists,
            active_task_name=str(active.get("task_name") or ""),
        )

    def assert_matches(self, expected: MutationPrecondition | dict | None) -> MutationPrecondition:
        expected_obj = expected if isinstance(expected, MutationPrecondition) else MutationPrecondition.from_dict(expected)
        actual = self.capture(expected_obj.task_name)
        mismatches: list[str] = []
        if expected_obj.queue_sha256 and expected_obj.queue_sha256 != actual.queue_sha256:
            mismatches.append("queue_sha256")
        if expected_obj.task_exists is not None and expected_obj.task_exists != actual.task_exists:
            mismatches.append("task_exists")
        if expected_obj.task_config_sha256 and expected_obj.task_config_sha256 != actual.task_config_sha256:
            mismatches.append("task_config_sha256")
        if expected_obj.active_task_name and expected_obj.active_task_name != actual.active_task_name:
            mismatches.append("active_task_name")
        if mismatches:
            raise PreconditionFailed(
                "PRECONDITION_FAILED: platform state changed after approval analysis; mismatched="
                + ",".join(mismatches)
            )
        return actual
