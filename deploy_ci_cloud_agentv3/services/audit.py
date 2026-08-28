from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore as SQLiteAuditStore
from deploy_ci_cloud_agentv3.persistence.redaction import redact


class AuditStore:
    """Business audit facade.

    Tests may keep it in memory. Production passes a SQLite path, preserving the
    append/query semantics across process restarts without making AuditStore an
    AgentState source of truth.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._lock = Lock()
        self._sqlite = SQLiteAuditStore(self.path) if self.path else None

    def append(self, event_type: str, payload: dict[str, Any], *, thread_id: str | None = None, run_id: str | None = None) -> None:
        record = {"ts": time.time(), "event_type": event_type, "payload": redact(payload), "thread_id": thread_id, "run_id": run_id}
        with self._lock:
            self.records.append(record)
        if self._sqlite is not None:
            self._sqlite.append(event_type, payload, thread_id=thread_id, run_id=run_id)

    def query(self, *, thread_id: str | None = None, run_id: str | None = None, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if self._sqlite is not None:
            return self._sqlite.query(thread_id=thread_id, run_id=run_id, event_type=event_type, limit=limit)
        rows = self.records
        if thread_id is not None:
            rows = [row for row in rows if row.get("thread_id") == thread_id]
        if run_id is not None:
            rows = [row for row in rows if row.get("run_id") == run_id]
        if event_type is not None:
            rows = [row for row in rows if row.get("event_type") == event_type]
        return list(rows[-limit:])
