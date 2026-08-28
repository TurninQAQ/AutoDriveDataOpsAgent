from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .database import connect, initialize
from .redaction import redact


class AuditStore:
    """Append-only durable business audit; never an AgentState source of truth."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        initialize(self.path)

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "thread_id": thread_id,
            "run_id": run_id,
            "event_type": str(event_type),
            "timestamp": time.time(),
            "payload": redact(payload),
        }
        with connect(self.path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO audit_events(event_id, thread_id, run_id, event_type, timestamp, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["event_id"], thread_id, run_id, record["event_type"], record["timestamp"],
                    json.dumps(record["payload"], ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
            conn.execute("COMMIT")
        return record

    def query(
        self,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            where.append("thread_id = ?"); params.append(thread_id)
        if run_id is not None:
            where.append("run_id = ?"); params.append(run_id)
        if event_type is not None:
            where.append("event_type = ?"); params.append(event_type)
        sql = "SELECT * FROM audit_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(max(1, min(int(limit), 10000)))
        with connect(self.path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "event_id": row["event_id"], "thread_id": row["thread_id"], "run_id": row["run_id"],
                "event_type": row["event_type"], "timestamp": row["timestamp"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]
