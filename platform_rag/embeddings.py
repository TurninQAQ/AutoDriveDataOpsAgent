from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from platform_integrations.gemini_retry import retry_sync

from .models import KnowledgeChunk


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimension: int

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> dict[str, list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm <= 0:
        return [float(v) for v in values]
    return [float(v) / norm for v in values]


def cosine_dense(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class GeminiEmbeddingProvider:
    """Gemini Embedding 2 adapter for asymmetric RAG retrieval.

    Google recommends task instructions in text for gemini-embedding-2 instead
    of the task_type field. Documents and queries are therefore formatted with
    different retrieval prefixes before embedding.
    """

    provider_name = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-embedding-2",
        dimension: int = 768,
        batch_size: int = 32,
        client=None,
    ):
        self.model_name = model_name
        self.dimension = max(128, min(3072, int(dimension)))
        self.batch_size = max(1, int(batch_size))
        if client is not None:
            self.client = client
            self._types = None
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "google-genai is not installed. Install requirements-agent.txt first."
            ) from exc
        api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini embeddings")
        self.client = genai.Client(api_key=api_key)
        self._types = types

    @staticmethod
    def _document_text(chunk: KnowledgeChunk) -> str:
        title = chunk.title or chunk.section or "none"
        section = f"\nsection: {chunk.section}" if chunk.section else ""
        return f"title: {title} | text: {section}\n{chunk.content}"

    @staticmethod
    def _query_text(query: str) -> str:
        return f"task: question answering | query: {query.strip()}"

    def _embed_strings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        # For the real google-genai client, multiple Content objects produce
        # separate embeddings. Fake clients used in tests may accept raw strings.
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self._types is not None:
                contents = [
                    self._types.Content(parts=[self._types.Part.from_text(text=text)])
                    for text in batch
                ]
                config = self._types.EmbedContentConfig(output_dimensionality=self.dimension)
            else:
                contents = batch
                config = {"output_dimensionality": self.dimension}
            response = retry_sync(
                lambda: self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                ),
                operation_name=f"embed_content:{self.model_name}",
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Gemini embedding response count mismatch: expected {len(batch)}, got {len(embeddings)}"
                )
            for item in embeddings:
                values = list(getattr(item, "values", item))
                vectors.append(_normalize([float(v) for v in values]))
        return vectors

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> dict[str, list[float]]:
        texts = [self._document_text(chunk) for chunk in chunks]
        vectors = self._embed_strings(texts)
        return {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}

    def embed_query(self, query: str) -> list[float]:
        vectors = self._embed_strings([self._query_text(query)])
        return vectors[0] if vectors else []


class DenseEmbeddingIndex:
    """Incremental, resumable file-backed dense vector sidecar."""

    schema_version = 2

    def __init__(self, path: Path, provider: EmbeddingProvider):
        self.path = Path(path)
        self.provider = provider
        self.lock_file = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _lock(self):
        import fcntl
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_payload(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def is_fresh(self, source_fingerprint: str, chunk_ids: list[str]) -> bool:
        payload = self._load_payload()
        if not payload:
            return False
        return (
            int(payload.get("schema_version", 0)) == self.schema_version
            and payload.get("source_fingerprint") == source_fingerprint
            and payload.get("provider") == self.provider.provider_name
            and payload.get("model") == self.provider.model_name
            and int(payload.get("dimension", 0)) == self.provider.dimension
            and bool(payload.get("complete"))
            and int(payload.get("expected_chunk_count", 0)) == len(chunk_ids)
            and set(payload.get("vectors", {}).keys()) == set(chunk_ids)
        )

    def _provider_matches(self, payload: dict | None) -> bool:
        return bool(payload) and (
            int(payload.get("schema_version", 0)) == self.schema_version
            and payload.get("provider") == self.provider.provider_name
            and payload.get("model") == self.provider.model_name
            and int(payload.get("dimension", 0)) == self.provider.dimension
        )

    def _atomic_write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)

    def _checkpoint(
        self,
        *,
        source_fingerprint: str,
        chunks: list[KnowledgeChunk],
        vectors: dict[str, list[float]],
        complete: bool,
        built_at: str | None,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": self.schema_version,
            "source_fingerprint": source_fingerprint,
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "dimension": self.provider.dimension,
            "expected_chunk_count": len(chunks),
            "complete": complete,
            "content_hashes": {chunk.chunk_id: chunk.content_hash for chunk in chunks},
            "vectors": vectors,
            "updated_at": now,
            "built_at": built_at if complete else None,
        }
        self._atomic_write(payload)
        return payload

    def ensure(
        self,
        source_fingerprint: str,
        chunks: list[KnowledgeChunk],
        force: bool = False,
        reset: bool = False,
    ) -> dict[str, list[float]]:
        """Build or resume vectors, reusing matching chunk content.

        ``force`` refreshes the lexical index but intentionally does not discard
        matching dense vectors. ``reset`` is the explicit full dense rebuild.
        """
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if not reset and self.is_fresh(source_fingerprint, chunk_ids):
            return self.load_vectors()
        with self._lock():
            if not reset and self.is_fresh(source_fingerprint, chunk_ids):
                return self.load_vectors()
            existing = self._load_payload()
            old_vectors = (existing or {}).get("vectors", {}) if self._provider_matches(existing) and not reset else {}
            old_hashes = (existing or {}).get("content_hashes", {}) if self._provider_matches(existing) and not reset else {}
            vectors = {
                chunk.chunk_id: old_vectors[chunk.chunk_id]
                for chunk in chunks
                if chunk.chunk_id in old_vectors
                and old_hashes.get(chunk.chunk_id) == chunk.content_hash
                and old_vectors[chunk.chunk_id]
            }
            missing = [chunk for chunk in chunks if chunk.chunk_id not in vectors]
            self._checkpoint(
                source_fingerprint=source_fingerprint,
                chunks=chunks,
                vectors=vectors,
                complete=not missing,
                built_at=(datetime.now(timezone.utc).isoformat() if not missing else None),
            )
            if not missing:
                return vectors

            batch_size = max(1, int(getattr(self.provider, "batch_size", len(missing)) or len(missing)))
            try:
                for start in range(0, len(missing), batch_size):
                    batch = missing[start : start + batch_size]
                    batch_vectors = self.provider.embed_documents(batch)
                    expected = {chunk.chunk_id for chunk in batch}
                    if set(batch_vectors) != expected:
                        raise RuntimeError(
                            "Gemini embedding response ids mismatch while building dense sidecar"
                        )
                    vectors.update({str(key): list(value) for key, value in batch_vectors.items()})
                    self._checkpoint(
                        source_fingerprint=source_fingerprint,
                        chunks=chunks,
                        vectors=vectors,
                        complete=False,
                        built_at=None,
                    )
            except Exception:
                # The last successful checkpoint remains resumable and the
                # explicit incomplete marker is published even on batch zero.
                self._checkpoint(
                    source_fingerprint=source_fingerprint,
                    chunks=chunks,
                    vectors=vectors,
                    complete=False,
                    built_at=None,
                )
                raise

            self._checkpoint(
                source_fingerprint=source_fingerprint,
                chunks=chunks,
                vectors=vectors,
                complete=True,
                built_at=datetime.now(timezone.utc).isoformat(),
            )
            return vectors

    def load_vectors(self) -> dict[str, list[float]]:
        payload = self._load_payload() or {}
        return {
            str(key): [float(v) for v in value]
            for key, value in (payload.get("vectors") or {}).items()
        }

    def status(self) -> dict:
        payload = self._load_payload()
        return {
            "enabled": True,
            "index_file": str(self.path),
            "exists": bool(payload),
            "provider": (payload or {}).get("provider", self.provider.provider_name),
            "model": (payload or {}).get("model", self.provider.model_name),
            "dimension": int((payload or {}).get("dimension", self.provider.dimension)),
            "vector_count": len((payload or {}).get("vectors", {})),
            "expected_vector_count": int((payload or {}).get("expected_chunk_count", 0)),
            "complete": bool((payload or {}).get("complete", False)),
            "missing_vector_count": max(
                0,
                int((payload or {}).get("expected_chunk_count", 0))
                - len((payload or {}).get("vectors", {})),
            ),
            "resumable": bool(payload) and not bool((payload or {}).get("complete", False)),
            "updated_at": (payload or {}).get("updated_at"),
            "built_at": (payload or {}).get("built_at"),
        }
