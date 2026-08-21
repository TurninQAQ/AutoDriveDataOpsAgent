from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


STATUSES = "pending|authorized|executing|rejected|executed|failed|verification_failed|execution_unknown|expired"


def _canonical_arguments(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(name): _canonical_arguments(item, key=str(name))
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        items = [_canonical_arguments(item, key=key) for item in value]
        # Dataset scope is a set semantically.  Canonicalizing its order makes
        # retries with the same frozen scope deduplicate deterministically.
        if key == "datasets":
            return sorted({json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in items})
        return items
    return value


def action_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a stable fingerprint for a frozen mutation action."""

    payload = {
        "tool_name": str(tool_name),
        "arguments": _canonical_arguments(dict(arguments or {})),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PendingApproval(BaseModel):
    approval_id: str
    status: str = Field(default="pending", pattern=rf"^({STATUSES})$")
    created_at: float
    expires_at: float
    thread_id: str = "default"
    trace_id: str = ""
    execution_trace_id: str = ""
    authorization_mode: str = Field(default="hitl", pattern="^(hitl|auto)$")
    policy_decision: dict[str, Any] | None = None
    action_fingerprint: str | None = None
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


@dataclass(frozen=True)
class AutoReservationResult:
    status: str
    record: PendingApproval | None = None
    existing_record: PendingApproval | None = None
    action_fingerprint: str = ""
    auto_actions_used: int = 0


class ApprovalStore:
    """Atomic file-backed approval store with scoped AUTO reservations."""

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

    def _trace_lock_path(self, trace_id: str) -> Path:
        digest = hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()
        return self.root / f".autonomy-trace-{digest}.lock"

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

    @contextlib.contextmanager
    def _trace_locked(self, trace_id: str):
        if not str(trace_id):
            raise ValueError("AUTO reservation requires a non-empty trace_id")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._trace_lock_path(trace_id).open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_unlocked(self, item: PendingApproval) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(item.approval_id)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(item.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Atomic replace remains the portability fallback on filesystems
            # that do not expose a directory descriptor.
            pass

    def _read_unlocked(self, approval_id: str) -> PendingApproval:
        path = self._path(approval_id)
        if not path.is_file():
            raise FileNotFoundError(f"Approval not found: {approval_id}")
        return PendingApproval.model_validate_json(path.read_text(encoding="utf-8"))

    def _records_unlocked(self) -> list[PendingApproval]:
        if not self.root.is_dir():
            return []
        records: list[PendingApproval] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                records.append(PendingApproval.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records

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
        authorization_mode: str = "hitl",
        policy_decision: dict[str, Any] | None = None,
        initial_status: str = "pending",
        action_fingerprint_value: str | None = None,
    ) -> PendingApproval:
        now = time.time()
        item = PendingApproval(
            approval_id=uuid.uuid4().hex[:16],
            created_at=now,
            expires_at=now + self.ttl_sec,
            thread_id=thread_id,
            trace_id=trace_id,
            authorization_mode=authorization_mode,
            policy_decision=dict(policy_decision or {}) or None,
            action_fingerprint=(
                action_fingerprint_value
                or (action_fingerprint(tool_name, arguments) if authorization_mode == "auto" else None)
            ),
            user_request=user_request,
            tool_name=tool_name,
            arguments=dict(arguments),
            precondition=dict(precondition),
            risk_level=risk_level,
            impact_summary=impact_summary,
            impact_details=list(impact_details or []),
            verification_baseline=dict(verification_baseline or {}),
            status=initial_status,
        )
        with self._locked(item.approval_id):
            self._write_unlocked(item)
        return item

    def create_auto_execution(
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
        policy_decision: dict[str, Any] | None = None,
        action_fingerprint_value: str | None = None,
    ) -> PendingApproval:
        """Persist a policy-authorized action awaiting its execution claim.

        This is not a fake human approval.  The record explicitly carries
        ``authorization_mode=auto`` and uses the same execution path as HITL.
        """

        return self.create(
            thread_id=thread_id,
            user_request=user_request,
            tool_name=tool_name,
            arguments=arguments,
            precondition=precondition,
            risk_level=risk_level,
            impact_summary=impact_summary,
            impact_details=impact_details,
            verification_baseline=verification_baseline,
            trace_id=trace_id,
            authorization_mode="auto",
            policy_decision=policy_decision,
            initial_status="authorized",
            action_fingerprint_value=action_fingerprint_value,
        )

    def reserve_auto_execution(
        self,
        *,
        max_actions_per_request: int,
        thread_id: str,
        user_request: str,
        tool_name: str,
        arguments: dict[str, Any],
        precondition: dict[str, Any],
        risk_level: str,
        impact_summary: str,
        impact_details: list[str] | None = None,
        verification_baseline: dict[str, Any] | None = None,
        trace_id: str,
        policy_decision: dict[str, Any] | None = None,
    ) -> AutoReservationResult:
        """Atomically reserve one AUTO slot and persist its authorization.

        The trace-scoped lock covers duplicate detection, budget accounting and
        record creation.  It is intentionally separate from the per-record
        execution lock so a second worker can safely claim a pre-existing
        authorization without reserving another mutation.
        """

        fingerprint = action_fingerprint(tool_name, arguments)
        limit = max(1, int(max_actions_per_request))
        with self._trace_locked(trace_id):
            records = [item for item in self._records_unlocked() if item.authorization_mode == "auto" and item.trace_id == trace_id]
            for item in records:
                existing_fingerprint = item.action_fingerprint or action_fingerprint(item.tool_name, item.arguments)
                if existing_fingerprint == fingerprint:
                    return AutoReservationResult(
                        status="duplicate_existing",
                        record=item,
                        existing_record=item,
                        action_fingerprint=fingerprint,
                        auto_actions_used=len(records),
                    )
            if len(records) >= limit:
                return AutoReservationResult(
                    status="budget_exhausted",
                    action_fingerprint=fingerprint,
                    auto_actions_used=len(records),
                )
            payload = dict(policy_decision or {})
            payload["reservation_status"] = "reserved"
            budget = dict(payload.get("budget") or {})
            budget["actions_used"] = len(records) + 1
            payload["budget"] = budget
            item = self.create(
                thread_id=thread_id,
                user_request=user_request,
                tool_name=tool_name,
                arguments=arguments,
                precondition=precondition,
                risk_level=risk_level,
                impact_summary=impact_summary,
                impact_details=impact_details,
                verification_baseline=verification_baseline,
                trace_id=trace_id,
                authorization_mode="auto",
                policy_decision=payload,
                initial_status="authorized",
                action_fingerprint_value=fingerprint,
            )
            return AutoReservationResult(
                status="reserved",
                record=item,
                action_fingerprint=fingerprint,
                auto_actions_used=len(records) + 1,
            )

    def count_auto_actions(self, trace_id: str) -> int:
        if not trace_id:
            return 0
        return sum(
            1
            for item in self.list(status="")
            if item.authorization_mode == "auto" and item.trace_id == trace_id
        )

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
        """Atomically claim one HITL or AUTO authorization for execution."""
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status == "pending" and item.expired:
                item.status = "expired"
                self._write_unlocked(item)
                raise RuntimeError("Approval is expired")
            expected_status = "authorized" if item.authorization_mode == "auto" else "pending"
            if item.status != expected_status:
                if item.authorization_mode == "auto" and item.status == "executing":
                    raise RuntimeError("AUTO execution claim rejected: execution is already claimed or its outcome is unknown")
                raise RuntimeError(f"Approval is not pending: {item.status}")
            item.status = "executing"
            if execution_trace_id:
                item.execution_trace_id = execution_trace_id
            self._write_unlocked(item)
            return item

    def mark_execution_unknown(self, approval_id: str, reason: str = "External mutation outcome is unknown") -> PendingApproval:
        with self._locked(approval_id):
            item = self._read_unlocked(approval_id)
            if item.status != "executing":
                raise RuntimeError(f"Approval cannot become execution_unknown from status: {item.status}")
            item.status = "execution_unknown"
            item.error = reason
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
