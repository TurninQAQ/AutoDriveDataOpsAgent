from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import KnowledgeChunk, KnowledgeIndexStats
from .sources import KnowledgeSourceLoader


SCHEMA_VERSION = 1


class KnowledgeIndex:
    def __init__(self, index_file: Path, loader: KnowledgeSourceLoader):
        self.index_file = Path(index_file)
        self.loader = loader
        self.lock_file = self.index_file.with_suffix(self.index_file.suffix + ".lock")

    @contextmanager
    def _exclusive_lock(self):
        """Serialize index rebuilds across local Agent/CLI processes.

        The platform already targets Linux and uses file locks elsewhere, so fcntl
        keeps this dependency-free and consistent with the existing runtime.
        Readers remain lock-free because publication uses atomic rename.
        """
        import fcntl

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _build_unlocked(self, source_fingerprint: str) -> KnowledgeIndexStats:
        chunks = self.loader.load()
        built_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_fingerprint": source_fingerprint,
            "built_at": built_at,
            "document_count": len({chunk.source_path for chunk in chunks}),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        # PID-specific temp path prevents stale temp collisions even if another
        # process crashed before acquiring/releasing the rebuild lock.
        import os

        tmp = self.index_file.with_suffix(self.index_file.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.index_file)
        return KnowledgeIndexStats(
            schema_version=SCHEMA_VERSION,
            source_fingerprint=source_fingerprint,
            document_count=payload["document_count"],
            chunk_count=len(chunks),
            built_at=built_at,
        )

    def build(self, force: bool = False) -> KnowledgeIndexStats:
        source_fingerprint = self.loader.fingerprint()
        with self._exclusive_lock():
            if not force:
                current = self.stats()
                if current and current.schema_version == SCHEMA_VERSION and current.source_fingerprint == source_fingerprint:
                    return current
            return self._build_unlocked(source_fingerprint)

    def ensure(self) -> KnowledgeIndexStats:
        fingerprint = self.loader.fingerprint()
        current = self.stats()
        if current and current.schema_version == SCHEMA_VERSION and current.source_fingerprint == fingerprint:
            return current
        with self._exclusive_lock():
            # Another process may have rebuilt while we were waiting.
            current = self.stats()
            fingerprint = self.loader.fingerprint()
            if current and current.schema_version == SCHEMA_VERSION and current.source_fingerprint == fingerprint:
                return current
            return self._build_unlocked(fingerprint)

    def load_chunks(self, ensure_fresh: bool = True) -> list[KnowledgeChunk]:
        if ensure_fresh:
            self.ensure()
        if not self.index_file.exists():
            return []
        payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        return [KnowledgeChunk.model_validate(item) for item in payload.get("chunks", [])]

    def stats(self) -> KnowledgeIndexStats | None:
        if not self.index_file.exists():
            return None
        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
            return KnowledgeIndexStats(
                schema_version=int(payload.get("schema_version", 0)),
                source_fingerprint=str(payload.get("source_fingerprint", "")),
                document_count=int(payload.get("document_count", 0)),
                chunk_count=len(payload.get("chunks", [])),
                built_at=str(payload.get("built_at", "")),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
