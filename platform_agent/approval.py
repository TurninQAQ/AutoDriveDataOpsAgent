from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


STATUSES = "pending|executing|rejected|executed|failed|verification_failed|expired"


class PendingApproval(BaseModel):
    approval_id: str
    status: str = Field(default="pending", pattern=rf"^({STATUSES})$")
    created_at: float
    expires_at: float
    thread_id: str = "default"
    trace_id: str = ""
    execution_trace_id: str = ""
    user_request: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    precondition: dict[str, Any] = Field(default_factory=dict)
    risk_level: str
    impact_summary: str
    impact_details: list[str] = Field(default_factory=list)
    execution_result: dict[str, Any] | None = None
    verification_baseline: dict[str, Any] = Field(default_factory=dict)
    verification_result: dict[str, Any] | None = None
    # Optional user-goal lifecycle result.  Approval status continues to model
    # mutation lifecycle; this field separately records post-action Goal state.
    goal_verification_result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


class ApprovalStore:
    """Atomic file-backed HITL approval store with per-approval process locks."""

    def __init__(self, root: str | Path, ttl_sec: int = 900):
        self.root = Path(root)
        self.ttl_sec = max(30, int(ttl_sec))

    def _path(self, approval_id: str) -> Path:
        safe = "".join(ch for ch in approval_id if ch.isalnum() or ch in "-_")
        if safe != approval_id or not safe:
            raise ValueError("Invalid approval id")
        return self.root / f"{safe}.json"

    def _lock_path(self, approval_id: str) -> Path:
        return self.root / f".{approval_id}.lock"

    @contextlib.contextmanager
    def _locked(self, approval_id: str, exclusive: bool = True):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path(approval_id)
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_unlocked(self, item: PendingApproval) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(item.approval_id)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(item.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _read_unlocked(self, approval_id: str) -> PendingApproval:
        path = self._path(approval_id)
        if not path.is_file():
            raise FileNotFoundError(f"Approval not found: {approval_id}")
        return PendingApproval.model_validate_json(path.read_text(encoding="utf-8"))

    def create(
        self,
        *,
        thread_id: str,
        user_request: str,
        tool_name: str,
        arguments: dict[str, Any],
        precondition: dict[str, Any],
        risk_level: str,
        impact_summary: str,
        impact_details: list[str] | None = None,
        verification_baseline: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> PendingApproval:
        now = time.time()
        item = PendingApproval(
            approval_id=uuid.uuid4().hex[:16],
            created_at=now,
            expires_at=now + self.ttl_sec,
            thread_id=thread_id,
            trace_id=trace_id,
            user_request=user_request,
            tool_name=tool_name,
            arguments=dict(arguments),
            precondition=dict(precondition),
            risk_level=risk_level,
            impact_summary=impact_summary,
            impact_details=list(impact_details or []),
            verification_baseline=dict(verification_baseline or {}),
        )
        with self._locked(item.approval_id):
            self._write_unlocked(item)
        return item

    def get(self, approval_id: str) -> PendingApproval:
        # Pending expiration is a mutation, so take an exclusive lock.
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status == "pending" and item.expired:
                item.status = "expired"
                self._write_unlocked(item)
            return item

    def list(self, status: str = "pending") -> list[PendingApproval]:
        if not self.root.is_dir():
            return []
        items: list[PendingApproval] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                item = self.get(path.stem)
            except Exception:
                continue
            if status and item.status != status:
                continue
            items.append(item)
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def reject(self, approval_id: str, reason: str = "Rejected by user") -> PendingApproval:
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status == "pending" and item.expired:
                item.status = "expired"
                self._write_unlocked(item)
                raise RuntimeError("Approval is expired")
            if item.status != "pending":
                raise RuntimeError(f"Approval is not pending: {item.status}")
            item.status = "rejected"
            item.error = reason
            self._write_unlocked(item)
            return item

    def claim_for_execution(self, approval_id: str, execution_trace_id: str = "") -> PendingApproval:
        """Atomically claim a pending approval so only one process can execute it."""
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status == "pending" and item.expired:
                item.status = "expired"
                self._write_unlocked(item)
                raise RuntimeError("Approval is expired")
            if item.status != "pending":
                raise RuntimeError(f"Approval is not pending: {item.status}")
            item.status = "executing"
            if execution_trace_id:
                item.execution_trace_id = execution_trace_id
            self._write_unlocked(item)
            return item

    def mark_executed(
        self,
        approval_id: str,
        result: dict[str, Any],
        verification_result: dict[str, Any] | None = None,
        goal_verification_result: dict[str, Any] | None = None,
    ) -> PendingApproval:
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status != "executing":
                raise RuntimeError(f"Approval cannot be executed from status: {item.status}")
            item.status = "executed"
            item.execution_result = result
            item.verification_result = dict(verification_result or {}) or None
            item.goal_verification_result = dict(goal_verification_result or {}) or None
            item.error = None
            self._write_unlocked(item)
            return item

    def mark_verification_failed(self, approval_id: str, result: dict[str, Any], verification_result: dict[str, Any], error: str) -> PendingApproval:
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status != "executing":
                raise RuntimeError(f"Approval cannot fail verification from status: {item.status}")
            item.status = "verification_failed"
            item.execution_result = result
            item.verification_result = dict(verification_result)
            item.error = error
            self._write_unlocked(item)
            return item

    def mark_failed(self, approval_id: str, error: str) -> PendingApproval:
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status != "executing":
                raise RuntimeError(f"Approval cannot fail from status: {item.status}")
            item.status = "failed"
            item.error = error
            self._write_unlocked(item)
            return item
