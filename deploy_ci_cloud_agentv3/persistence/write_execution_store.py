from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database import connect, initialize


@dataclass(frozen=True)
class WriteExecutionRecord:
    idempotency_key: str
    fingerprint: str
    thread_id: str | None
    run_id: str | None
    action: str
    status: str
    mutation_attempted: bool
    result: dict[str, Any] | None
    created_at: float
    updated_at: float


class InMemoryWriteExecutionStore:
    def __init__(self) -> None:
        self._records: dict[str, WriteExecutionRecord] = {}
        self._lock = asyncio.Lock()

    async def claim(self, key: str, *, fingerprint: str, action: str, thread_id: str | None = None, run_id: str | None = None) -> tuple[bool, WriteExecutionRecord]:
        async with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return False, existing
            now = time.time()
            record = WriteExecutionRecord(key, fingerprint, thread_id, run_id, action, "DISPATCHING", True, None, now, now)
            self._records[key] = record
            return True, record

    async def get(self, key: str) -> WriteExecutionRecord | None:
        return self._records.get(key)

    async def save_result(self, key: str, *, status: str, result: dict[str, Any]) -> WriteExecutionRecord:
        async with self._lock:
            old = self._records[key]
            record = WriteExecutionRecord(old.idempotency_key, old.fingerprint, old.thread_id, old.run_id, old.action, status, True, result, old.created_at, time.time())
            self._records[key] = record
            return record


class SQLiteWriteExecutionStore:
    """Persistent at-most-one mutation claim per runtime-derived idempotency key."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        initialize(self.path)

    async def claim(self, key: str, *, fingerprint: str, action: str, thread_id: str | None = None, run_id: str | None = None) -> tuple[bool, WriteExecutionRecord]:
        return await asyncio.to_thread(self._claim_sync, key, fingerprint, action, thread_id, run_id)

    def _claim_sync(self, key: str, fingerprint: str, action: str, thread_id: str | None, run_id: str | None) -> tuple[bool, WriteExecutionRecord]:
        now = time.time()
        with connect(self.path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM write_executions WHERE idempotency_key=?", (key,)).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                return False, self._row(row)
            conn.execute(
                "INSERT INTO write_executions(idempotency_key,fingerprint,thread_id,run_id,action,status,mutation_attempted,result_json,created_at,updated_at) VALUES(?,?,?,?,?,'DISPATCHING',1,NULL,?,?)",
                (key, fingerprint, thread_id, run_id, action, now, now),
            )
            row = conn.execute("SELECT * FROM write_executions WHERE idempotency_key=?", (key,)).fetchone()
            conn.execute("COMMIT")
        return True, self._row(row)

    async def get(self, key: str) -> WriteExecutionRecord | None:
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> WriteExecutionRecord | None:
        with connect(self.path) as conn:
            row = conn.execute("SELECT * FROM write_executions WHERE idempotency_key=?", (key,)).fetchone()
        return self._row(row) if row else None

    async def save_result(self, key: str, *, status: str, result: dict[str, Any]) -> WriteExecutionRecord:
        return await asyncio.to_thread(self._save_sync, key, status, result)

    def _save_sync(self, key: str, status: str, result: dict[str, Any]) -> WriteExecutionRecord:
        now = time.time()
        with connect(self.path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE write_executions SET status=?, result_json=?, updated_at=? WHERE idempotency_key=?",
                (status, json.dumps(result, ensure_ascii=False, sort_keys=True, default=str), now, key),
            )
            row = conn.execute("SELECT * FROM write_executions WHERE idempotency_key=?", (key,)).fetchone()
            conn.execute("COMMIT")
        if row is None:
            raise KeyError(key)
        return self._row(row)

    @staticmethod
    def _row(row: Any) -> WriteExecutionRecord:
        return WriteExecutionRecord(
            idempotency_key=row["idempotency_key"], fingerprint=row["fingerprint"], thread_id=row["thread_id"], run_id=row["run_id"],
            action=row["action"], status=row["status"], mutation_attempted=bool(row["mutation_attempted"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        )
