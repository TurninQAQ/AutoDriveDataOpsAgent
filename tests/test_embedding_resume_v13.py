from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from platform_rag.embeddings import DenseEmbeddingIndex
from platform_rag.models import KnowledgeChunk
from platform_rag.service import AsyncKnowledgeRetriever


def _chunk(index: int, content_suffix: str = "") -> KnowledgeChunk:
    chunk_id = f"chunk-{index}"
    content = f"content-{index}{content_suffix}"
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_path=f"doc-{index}.md",
        title=f"Doc {index}",
        content=content,
        content_hash=f"hash-{index}{content_suffix}",
    )


class _FakeProvider:
    provider_name = "fake"
    model_name = "fake-embedding"
    dimension = 3
    batch_size = 2

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def embed_documents(self, chunks):
        ids = [chunk.chunk_id for chunk in chunks]
        self.calls.append(ids)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated transient provider failure")
        return {chunk.chunk_id: [float(index + 1), 0.0, 0.0] for index, chunk in enumerate(chunks)}

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_partial_embedding_checkpoint_and_resume(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    chunks = [_chunk(index) for index in range(5)]
    provider = _FakeProvider(fail_on_call=3)
    store = DenseEmbeddingIndex(path, provider)

    with pytest.raises(RuntimeError):
        store.ensure("fingerprint-1", chunks)
    partial = _payload(path)
    assert partial["schema_version"] == 2
    assert partial["complete"] is False
    assert partial["expected_chunk_count"] == 5
    assert set(partial["vectors"]) == {"chunk-0", "chunk-1", "chunk-2", "chunk-3"}
    assert set(partial["content_hashes"]) == {f"chunk-{index}" for index in range(5)}

    provider.fail_on_call = None
    result = store.ensure("fingerprint-1", chunks)
    assert set(result) == {f"chunk-{index}" for index in range(5)}
    assert provider.calls[-1] == ["chunk-4"]
    finished = _payload(path)
    assert finished["complete"] is True
    assert len(finished["vectors"]) == 5
    assert store.is_fresh("fingerprint-1", [chunk.chunk_id for chunk in chunks])


def test_content_hash_change_reembeds_only_one_chunk(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    original = [_chunk(index) for index in range(5)]
    provider = _FakeProvider()
    store = DenseEmbeddingIndex(path, provider)
    store.ensure("fingerprint-1", original)
    calls_before = len(provider.calls)

    changed = [_chunk(index, "-changed") if index == 2 else _chunk(index) for index in range(5)]
    store.ensure("fingerprint-2", changed)
    assert len(provider.calls) == calls_before + 1
    assert provider.calls[-1] == ["chunk-2"]


def test_removed_chunk_is_dropped_from_sidecar(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    provider = _FakeProvider()
    store = DenseEmbeddingIndex(path, provider)
    store.ensure("fingerprint-1", [_chunk(index) for index in range(3)])
    store.ensure("fingerprint-2", [_chunk(index) for index in range(2)])
    payload = _payload(path)
    assert set(payload["vectors"]) == {"chunk-0", "chunk-1"}
    assert payload["expected_chunk_count"] == 2
    assert payload["complete"] is True


def test_provider_or_dimension_change_invalidates_all_vectors(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    chunks = [_chunk(index) for index in range(3)]
    provider_a = _FakeProvider()
    DenseEmbeddingIndex(path, provider_a).ensure("fingerprint", chunks)

    provider_b = _FakeProvider()
    provider_b.model_name = "other-model"
    DenseEmbeddingIndex(path, provider_b).ensure("fingerprint", chunks)
    assert provider_b.calls == [["chunk-0", "chunk-1"], ["chunk-2"]]

    provider_c = _FakeProvider()
    provider_c.dimension = 6
    DenseEmbeddingIndex(path, provider_c).ensure("fingerprint", chunks)
    assert provider_c.calls == [["chunk-0", "chunk-1"], ["chunk-2"]]


def test_corrupt_partial_sidecar_rebuilds_safely(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    path.write_text("{not-json", encoding="utf-8")
    provider = _FakeProvider()
    chunks = [_chunk(index) for index in range(2)]
    result = DenseEmbeddingIndex(path, provider).ensure("fingerprint", chunks)
    assert set(result) == {"chunk-0", "chunk-1"}
    assert _payload(path)["complete"] is True


def test_force_reuses_matching_dense_vectors_and_reset_rebuilds(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    chunks = [_chunk(index) for index in range(3)]
    provider = _FakeProvider()
    store = DenseEmbeddingIndex(path, provider)
    store.ensure("fingerprint-1", chunks)
    calls_before = len(provider.calls)
    store.ensure("fingerprint-2", chunks, force=True)
    assert len(provider.calls) == calls_before
    store.ensure("fingerprint-2", chunks, reset=True)
    assert len(provider.calls) == calls_before + 2


def test_status_reports_resume_progress(tmp_path: Path):
    path = tmp_path / "embeddings.json"
    provider = _FakeProvider(fail_on_call=2)
    chunks = [_chunk(index) for index in range(3)]
    with pytest.raises(RuntimeError):
        DenseEmbeddingIndex(path, provider).ensure("fingerprint", chunks)
    status = DenseEmbeddingIndex(path, provider).status()
    assert status["vector_count"] == 2
    assert status["expected_vector_count"] == 3
    assert status["missing_vector_count"] == 1
    assert status["complete"] is False
    assert status["resumable"] is True


def test_async_retriever_moves_sync_search_to_worker_thread():
    main_thread = threading.get_ident()
    thread_ids: list[int] = []

    class Service:
        def search(self, query, top_k=None):
            thread_ids.append(threading.get_ident())
            return type("Result", (), {"results": []})()

    result = asyncio.run(AsyncKnowledgeRetriever(Service()).retrieve("query"))
    assert result == []
    assert thread_ids and thread_ids[0] != main_thread
