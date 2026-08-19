from __future__ import annotations

import contextlib
import contextvars
import time
import uuid
from typing import Any, Iterator

from .models import AuditRecord, TraceEvent
from .redaction import sanitize
from .store import TraceStore


_CURRENT_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("platform_trace_id", default="")


def current_trace_id() -> str:
    return _CURRENT_TRACE_ID.get()


class TraceRecorder:
    def __init__(self, store: TraceStore, enabled: bool = True, max_value_chars: int = 16000, maintenance_kwargs: dict[str, int] | None = None):
        self.store = store
        self.enabled = bool(enabled)
        self.max_value_chars = max(1024, int(max_value_chars))
        self.maintenance_kwargs = dict(maintenance_kwargs or {})

    def _clean(self, data: Any) -> Any:
        return sanitize(data, max_chars=self.max_value_chars)

    def start_trace(
        self,
        *,
        kind: str,
        user_request: str,
        thread_id: str = "default",
        parent_trace_id: str | None = None,
    ) -> str:
        trace_id = uuid.uuid4().hex
        if not self.enabled:
            return trace_id
        self.record(
            trace_id,
            "request",
            kind,
            data={
                "kind": kind,
                "thread_id": thread_id,
                "user_request": user_request,
            },
            parent_trace_id=parent_trace_id,
        )
        return trace_id

    @contextlib.contextmanager
    def activate(self, trace_id: str) -> Iterator[None]:
        token = _CURRENT_TRACE_ID.set(trace_id)
        try:
            yield
        finally:
            _CURRENT_TRACE_ID.reset(token)

    def record(
        self,
        trace_id: str,
        stage: str,
        name: str,
        *,
        status: str = "ok",
        duration_ms: float | None = None,
        data: dict[str, Any] | None = None,
        parent_trace_id: str | None = None,
    ) -> None:
        if not self.enabled or not trace_id:
            return
        event = TraceEvent(
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            event_id=uuid.uuid4().hex[:16],
            timestamp=time.time(),
            stage=stage,
            name=name,
            status=status,
            duration_ms=None if duration_ms is None else round(float(duration_ms), 3),
            data=self._clean(data or {}),
        )
        self.store.append_event(event)

    def record_current(self, stage: str, name: str, **kwargs) -> None:
        self.record(current_trace_id(), stage, name, **kwargs)

    def finish(
        self,
        trace_id: str,
        *,
        status: str,
        intent: str | None = None,
        response_summary: str = "",
        errors: list[str] | None = None,
    ) -> AuditRecord | None:
        if not self.enabled or not trace_id:
            return None
        self.record(
            trace_id,
            "response",
            "trace_finished",
            status=status,
            data={"intent": intent, "summary": response_summary, "errors": errors or []},
        )
        events = self.store.load_events(trace_id)
        if not events:
            return None
        request = next((item for item in events if item.stage == "request"), events[0])
        ended_at = max(item.timestamp for item in events)
        started_at = request.timestamp
        error_statuses = {"error", "failed", "verification_failed", "inconclusive", "invalid"}
        event_errors = [
            str(item.data.get("error") or item.data.get("message") or item.name)
            for item in events
            if item.status in error_statuses
        ]
        safe_errors = self._clean(errors or [])
        event_errors.extend(safe_errors if isinstance(safe_errors, list) else [str(safe_errors)])
        safe_summary = self._clean(response_summary)
        tool_calls = [item.data for item in events if item.stage == "tool"]
        approvals = [item.data for item in events if item.stage == "approval"]
        mutations = [item.data for item in events if item.stage == "mutation"]
        verification = [item.data for item in events if item.stage == "verification"]
        parent = next((item.parent_trace_id for item in events if item.parent_trace_id), None)
        record = AuditRecord(
            trace_id=trace_id,
            parent_trace_id=parent,
            kind=str(request.data.get("kind") or request.name),
            thread_id=str(request.data.get("thread_id") or "default"),
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=round(max(0.0, (ended_at - started_at) * 1000.0), 3),
            status=status,
            user_request=str(request.data.get("user_request") or ""),
            intent=intent,
            tool_calls=tool_calls,
            approvals=approvals,
            mutations=mutations,
            verification=verification,
            errors=list(dict.fromkeys(str(item) for item in event_errors if item)),
            response_summary=str(safe_summary),
        )
        self.store.append_audit(record)
        if self.maintenance_kwargs:
            self.store.maintenance(**self.maintenance_kwargs)
        return record
