"""Deterministic safety precondition capture and TOCTOU revalidation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from ..agent.immutable import canonical_snapshot, thaw_value
from ..agent.results import ResultStatus, TaskDetailResult, normalize_read_result
from .write_transaction import FrozenToolCall, PreconditionSnapshot


def target_for_write(call: FrozenToolCall) -> str:
    value = call.arguments.get("task_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("WRITE requires concrete task_name target")
    return value.strip()


class PreconditionReader:
    def __init__(self, read_facade: Any):
        self.read_facade = read_facade

    def capture(self, call: FrozenToolCall) -> PreconditionSnapshot:
        target = target_for_write(call)
        raw = canonical_snapshot(self.read_facade.get_task_detail(target))
        result = normalize_read_result("get_task_detail", {"task_name": target}, raw)
        self._validate_for_tool(call.tool_name, result)
        canonical = {
            "target": target,
            "tool_name": call.tool_name,
            "result": thaw_value(raw),
            "entity_version": result.entity_version,
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return PreconditionSnapshot(
            target=target,
            tool_name=call.tool_name,
            observed_at=datetime.now(timezone.utc),
            fingerprint=fingerprint,
            entity_version=result.entity_version,
            state=raw if isinstance(raw, Mapping) else {},
        )

    def matches(self, original: PreconditionSnapshot, call: FrozenToolCall) -> bool:
        try:
            current = self.capture(call)
        except Exception:
            return False
        return current.fingerprint == original.fingerprint

    @staticmethod
    def _validate_for_tool(tool_name: str, result: TaskDetailResult) -> None:
        if result.validation_errors or result.envelope.status is ResultStatus.MALFORMED:
            raise ValueError("safety precondition response is malformed")
        if tool_name == "submit_task":
            if result.envelope.status not in {ResultStatus.NOT_FOUND, ResultStatus.NO_DATA} and result.exists is not False:
                raise ValueError("submit_task target already exists")
            return
        if not result.qualifies_for_evidence():
            raise ValueError("WRITE target is not a valid live task")
