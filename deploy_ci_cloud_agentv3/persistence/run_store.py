from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_UNSET = object()

from .database import connect, initialize


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        initialize(self.path)

    def create(self, run_id: str, thread_id: str, status: str = "CREATED") -> dict[str, Any]:
        now = time.time()
        with connect(self.path) as conn:
            conn.execute(
                "INSERT INTO runs(run_id,thread_id,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                (run_id, thread_id, status, now, now),
            )
        return self.get(run_id)

    def update(self, run_id: str, *, status: str | None = None, final_response: Any = _UNSET, pending_action: Any = _UNSET, error: Any = _UNSET) -> dict[str, Any]:
        current = self.get(run_id)
        final_json = current.get("_final_response_json") if final_response is _UNSET else (json.dumps(final_response, ensure_ascii=False, default=str) if final_response is not None else None)
        pending_json = current.get("_pending_action_json") if pending_action is _UNSET else (json.dumps(pending_action, ensure_ascii=False, default=str) if pending_action is not None else None)
        resolved_error = current.get("error") if error is _UNSET else error
        with connect(self.path) as conn:
            conn.execute(
                "UPDATE runs SET status=?, updated_at=?, final_response_json=?, pending_action_json=?, error=? WHERE run_id=?",
                (status or current["status"], time.time(), final_json, pending_json, resolved_error, run_id),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with connect(self.path) as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {
            "run_id": row["run_id"], "thread_id": row["thread_id"], "status": row["status"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "final_response": json.loads(row["final_response_json"]) if row["final_response_json"] else None,
            "pending_action": json.loads(row["pending_action_json"]) if row["pending_action_json"] else None,
            "error": row["error"], "_final_response_json": row["final_response_json"], "_pending_action_json": row["pending_action_json"],
        }
