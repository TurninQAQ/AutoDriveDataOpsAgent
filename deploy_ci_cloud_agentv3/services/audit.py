from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any


class AuditStore:
    """Lightweight JSONL business audit: proposal, approval, write and verification."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._lock = Lock()

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"ts": time.time(), "event_type": event_type, "payload": payload}
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
