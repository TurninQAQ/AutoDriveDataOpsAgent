from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def connect(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def initialize(path: str | Path) -> None:
    with connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                thread_id TEXT,
                run_id TEXT,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_thread_ts ON audit_events(thread_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_run_ts ON audit_events(run_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_type_ts ON audit_events(event_type, timestamp)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS write_executions (
                idempotency_key TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                thread_id TEXT,
                run_id TEXT,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                mutation_attempted INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                final_response_json TEXT,
                pending_action_json TEXT,
                error TEXT
            )
            """
        )
