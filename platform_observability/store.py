from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Iterable

from .models import AuditRecord, TraceEvent, TraceSummary


class TraceStore:
    """Append-only JSONL trace/audit store using process-safe file locks."""

    def __init__(self, trace_dir: str | Path, audit_file: str | Path):
        self.trace_dir = Path(trace_dir)
        self.audit_file = Path(audit_file)

    def trace_path(self, trace_id: str) -> Path:
        safe = "".join(ch for ch in trace_id if ch.isalnum() or ch in "-_")
        if not safe or safe != trace_id:
            raise ValueError("Invalid trace_id")
        return self.trace_dir / f"{safe}.jsonl"

    @contextlib.contextmanager
    def _locked_file(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_name(f".{path.name}.lock")
        with lock.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append_event(self, event: TraceEvent) -> None:
        path = self.trace_path(event.trace_id)
        with self._locked_file(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def load_events(self, trace_id: str) -> list[TraceEvent]:
        path = self.trace_path(trace_id)
        if not path.is_file():
            return []
        items: list[TraceEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                items.append(TraceEvent.model_validate_json(line))
            except Exception:
                continue
        return items

    def append_audit(self, record: AuditRecord) -> None:
        with self._locked_file(self.audit_file):
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_file.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def load_audit(self) -> list[AuditRecord]:
        result: list[AuditRecord] = []
        backups: list[tuple[int, Path]] = []
        prefix = f"{self.audit_file.name}."
        if self.audit_file.parent.is_dir():
            for path in self.audit_file.parent.glob(f"{self.audit_file.name}.*"):
                suffix = path.name[len(prefix):]
                if suffix.isdigit() and path.is_file():
                    backups.append((int(suffix), path))
        # Higher suffix is older; load oldest -> newest -> current so callers get
        # one chronological stream across rotations.
        paths = [path for _, path in sorted(backups, key=lambda item: item[0], reverse=True)]
        if self.audit_file.is_file():
            paths.append(self.audit_file)
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    result.append(AuditRecord.model_validate_json(line))
                except Exception:
                    continue
        return result


    def prune_traces(self, retention_days: int = 14, max_files: int = 5000, now: float | None = None) -> dict[str, int]:
        """Delete old trace JSONL files while preserving newest traces.

        Both limits are optional: 0 disables that limit. Lock files are removed only
        when their matching trace file is removed and are never counted as traces.
        """
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        now = time.time() if now is None else float(now)
        files = [p for p in self.trace_dir.glob("*.jsonl") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        remove: set[Path] = set()
        if retention_days > 0:
            cutoff = now - int(retention_days) * 86400
            remove.update(p for p in files if p.stat().st_mtime < cutoff)
        if max_files > 0:
            keep_candidates = [p for p in files if p not in remove]
            remove.update(keep_candidates[max_files:])
        deleted = 0
        for path in remove:
            with self._locked_file(path):
                try:
                    path.unlink()
                    deleted += 1
                except FileNotFoundError:
                    pass
            lock = path.with_name(f".{path.name}.lock")
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
        return {"trace_files_seen": len(files), "trace_files_deleted": deleted}

    def rotate_audit(self, max_bytes: int = 20 * 1024 * 1024, backup_count: int = 5) -> dict[str, int | bool]:
        """Size-based audit JSONL rotation under the same process lock used for writes."""
        max_bytes = max(0, int(max_bytes))
        backup_count = max(0, int(backup_count))
        if max_bytes == 0 or not self.audit_file.is_file():
            return {"rotated": False, "size_bytes": self.audit_file.stat().st_size if self.audit_file.is_file() else 0}
        with self._locked_file(self.audit_file):
            if not self.audit_file.is_file():
                return {"rotated": False, "size_bytes": 0}
            size = self.audit_file.stat().st_size
            if size <= max_bytes:
                return {"rotated": False, "size_bytes": size}
            if backup_count <= 0:
                self.audit_file.write_text("", encoding="utf-8")
                return {"rotated": True, "size_bytes": size}
            oldest = self.audit_file.with_name(f"{self.audit_file.name}.{backup_count}")
            try:
                oldest.unlink()
            except FileNotFoundError:
                pass
            for idx in range(backup_count - 1, 0, -1):
                src = self.audit_file.with_name(f"{self.audit_file.name}.{idx}")
                dst = self.audit_file.with_name(f"{self.audit_file.name}.{idx + 1}")
                if src.exists():
                    os.replace(src, dst)
            os.replace(self.audit_file, self.audit_file.with_name(f"{self.audit_file.name}.1"))
            self.audit_file.touch()
            return {"rotated": True, "size_bytes": size}

    def maintenance(
        self, *, retention_days: int = 14, max_trace_files: int = 5000,
        audit_max_bytes: int = 20 * 1024 * 1024, audit_backup_count: int = 5, now: float | None = None,
    ) -> dict[str, object]:
        return {
            "traces": self.prune_traces(retention_days=retention_days, max_files=max_trace_files, now=now),
            "audit": self.rotate_audit(max_bytes=audit_max_bytes, backup_count=audit_backup_count),
        }

    def summaries(self, limit: int = 50) -> list[TraceSummary]:
        records = self.load_audit()[-max(1, int(limit)) :]
        records.reverse()
        return [
            TraceSummary(
                trace_id=item.trace_id,
                parent_trace_id=item.parent_trace_id,
                kind=item.kind,
                status=item.status,
                started_at=item.started_at,
                ended_at=item.ended_at,
                latency_ms=item.latency_ms,
                thread_id=item.thread_id,
                intent=item.intent,
                user_request=item.user_request,
                response_summary=item.response_summary,
                error_count=len(item.errors),
            )
            for item in records
        ]
