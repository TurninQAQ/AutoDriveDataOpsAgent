from __future__ import annotations

import math
from pathlib import Path

import pytest

from deploy_ci_cloud_agentv2.platform_backend import runtime
from deploy_ci_cloud_agentv2.platform_backend.rag import embeddings
from deploy_ci_cloud_agentv2.platform_backend.rag.embeddings import (
    DenseEmbeddingIndex,
    EmbeddingConfigurationError,
    EmbeddingResponseError,
    build_embedding_provider,
)
from deploy_ci_cloud_agentv2.platform_backend.rag.evaluation import evaluate_retrieval
from deploy_ci_cloud_agentv2.platform_backend.rag.models import KnowledgeChunk
from deploy_ci_cloud_agentv2.platform_backend.rag.retriever import HybridRetriever
from deploy_ci_cloud_agentv2.platform_backend.rag.service import KnowledgeService


class _FakeProvider:
    provider_name = "fake"
    model_name = "fake-model"
    dimension = 3
    batch_size = 8

    def __init__(self, vectors: dict[str, list[float]] | None = None, query: list[float] | None = None):
        self.vectors = vectors or {}
        self.query = query or [1.0, 0.0, 0.0]
        self.document_calls: list[list[str]] = []

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> dict[str, list[float]]:
        self.document_calls.append([chunk.chunk_id for chunk in chunks])
        return {
            chunk.chunk_id: self.vectors.get(chunk.chunk_id, [1.0, 0.0, 0.0])
            for chunk in chunks
        }

    def embed_query(self, query: str) -> list[float]:
        return self.query


def _chunk(chunk_id: str, content_hash: str = "hash") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_path=f"{chunk_id}.md",
        title=chunk_id,
        section="section",
        content=f"content for {chunk_id}",
        content_hash=content_hash,
    )


def test_embedding_factory_defaults_to_local_and_preserves_hash_alias():
    assert build_embedding_provider({}) is None
    assert build_embedding_provider({"PLATFORM_RAG_EMBED_PROVIDER": "local"}) is None
    assert build_embedding_provider({"PLATFORM_RAG_EMBED_PROVIDER": "hash"}) is None


def test_embedding_factory_fails_closed_for_invalid_explicit_configuration():
    with pytest.raises(EmbeddingConfigurationError, match="DASHSCOPE_API_KEY"):
        build_embedding_provider({"PLATFORM_RAG_EMBED_PROVIDER": "qwen"})
    with pytest.raises(EmbeddingConfigurationError, match="must be one of"):
        build_embedding_provider({"PLATFORM_RAG_EMBED_PROVIDER": "unknown"})
    with pytest.raises(EmbeddingConfigurationError, match="must be positive"):
        build_embedding_provider(
            {
                "PLATFORM_RAG_EMBED_PROVIDER": "qwen",
                "PLATFORM_RAG_EMBED_DIM": "0",
                "DASHSCOPE_API_KEY": "test-only-key",
                "DASHSCOPE_API_BASE_URL": "https://embedding.example.test/api/v1",
            }
        )
    with pytest.raises(EmbeddingConfigurationError, match="must be 1024"):
        build_embedding_provider(
            {
                "PLATFORM_RAG_EMBED_PROVIDER": "qwen",
                "PLATFORM_RAG_EMBED_DIM": "768",
                "DASHSCOPE_API_KEY": "test-only-key",
                "DASHSCOPE_API_BASE_URL": "https://embedding.example.test/api/v1",
            }
        )


def test_embedding_factory_constructs_selected_qwen_and_gemini(monkeypatch):
    captured: dict[str, dict] = {}

    class FakeQwen:
        def __init__(self, **kwargs):
            captured["qwen"] = kwargs

    class FakeGemini:
        def __init__(self, **kwargs):
            captured["gemini"] = kwargs

    monkeypatch.setattr(embeddings, "QwenEmbeddingProvider", FakeQwen)
    monkeypatch.setattr(embeddings, "GeminiEmbeddingProvider", FakeGemini)

    qwen = build_embedding_provider(
        {
            "PLATFORM_RAG_EMBED_PROVIDER": "qwen",
            "PLATFORM_RAG_EMBED_MODEL": "qwen-test-embedding",
            "PLATFORM_RAG_EMBED_DIM": "1024",
            "PLATFORM_RAG_EMBED_BATCH_SIZE": "7",
            "DASHSCOPE_API_KEY": "test-only-key",
            "DASHSCOPE_API_BASE_URL": "https://embedding.example.test/api/v1",
        }
    )
    gemini = build_embedding_provider(
        {
            "PLATFORM_RAG_EMBED_PROVIDER": "gemini",
            "PLATFORM_RAG_EMBED_MODEL": "gemini-test-embedding",
            "PLATFORM_RAG_EMBED_DIM": "768",
            "PLATFORM_RAG_EMBED_BATCH_SIZE": "9",
            "GEMINI_API_KEY": "test-only-key",
        }
    )

    assert isinstance(qwen, FakeQwen)
    assert captured["qwen"] == {
        "model_name": "qwen-test-embedding",
        "dimension": 1024,
        "batch_size": 7,
        "base_url": "https://embedding.example.test/api/v1",
        "api_key": "test-only-key",
    }
    assert isinstance(gemini, FakeGemini)
    assert captured["gemini"] == {
        "model_name": "gemini-test-embedding",
        "dimension": 768,
        "batch_size": 9,
        "api_key": "test-only-key",
    }


def test_embedding_factory_builds_the_real_qwen_adapter_without_a_network_call():
    provider = build_embedding_provider(
        {
            "PLATFORM_RAG_EMBED_PROVIDER": "qwen",
            "DASHSCOPE_API_KEY": "test-only-key",
            "DASHSCOPE_API_BASE_URL": "https://embedding.example.test/api/v1",
        }
    )
    assert isinstance(provider, embeddings.QwenEmbeddingProvider)
    assert provider.provider_name == "qwen"
    assert provider.model_name == "qwen3.7-text-embedding"
    assert provider.dimension == 1024


def test_runtime_injects_factory_provider_into_knowledge_service(monkeypatch, tmp_path):
    provider = _FakeProvider()
    captured: dict[str, object] = {}
    monkeypatch.setenv("AUTODRIVE_PLATFORM_KNOWLEDGE_DIR", str(tmp_path / "knowledge"))
    monkeypatch.setenv("AUTODRIVE_PLATFORM_KNOWLEDGE_INDEX", str(tmp_path / "index.json"))
    monkeypatch.setattr(runtime, "build_embedding_provider", lambda env: provider)

    def fake_facade(*, settings, knowledge_service):
        captured["settings"] = settings
        captured["knowledge"] = knowledge_service
        return knowledge_service

    monkeypatch.setattr(runtime, "build_default_facade", fake_facade)
    service = runtime.build_platform_facade()

    assert service is captured["knowledge"]
    assert service.embedding_provider is provider
    assert service.embedding_index is not None
    assert service.embedding_index.path == (tmp_path / "index.embeddings.json")


def test_dense_index_reuses_unchanged_chunks_and_rebuilds_changed_or_incompatible(tmp_path):
    chunks = [_chunk("one", "hash-one"), _chunk("two", "hash-two")]
    provider = _FakeProvider()
    index_path = tmp_path / "dense.json"
    index = DenseEmbeddingIndex(index_path, provider)

    assert set(index.ensure("source-a", chunks)) == {"one", "two"}
    assert provider.document_calls == [["one", "two"]]
    assert set(index.ensure("source-a", chunks)) == {"one", "two"}
    assert provider.document_calls == [["one", "two"]]

    changed = [_chunk("one", "hash-one"), _chunk("two", "hash-two-changed")]
    index.ensure("source-b", changed)
    assert provider.document_calls[-1] == ["two"]

    class DifferentModelProvider(_FakeProvider):
        model_name = "different-model"

    incompatible = DifferentModelProvider()
    DenseEmbeddingIndex(index_path, incompatible).ensure("source-b", changed)
    assert incompatible.document_calls == [["one", "two"]]

    class DifferentProvider(_FakeProvider):
        provider_name = "different-provider"

    provider_changed = DifferentProvider()
    DenseEmbeddingIndex(index_path, provider_changed).ensure("source-b", changed)
    assert provider_changed.document_calls == [["one", "two"]]

    class DifferentDimensionProvider(_FakeProvider):
        dimension = 4

        def embed_documents(self, chunks: list[KnowledgeChunk]) -> dict[str, list[float]]:
            self.document_calls.append([chunk.chunk_id for chunk in chunks])
            return {chunk.chunk_id: [1.0, 0.0, 0.0, 0.0] for chunk in chunks}

    dimension_changed = DifferentDimensionProvider()
    DenseEmbeddingIndex(index_path, dimension_changed).ensure("source-b", changed)
    assert dimension_changed.document_calls == [["one", "two"]]

    # Deleted chunks are absent from the next published sidecar; unchanged
    # surviving chunks retain their existing embedding without another call.
    reduced = [_chunk("one", "hash-one")]
    assert set(DenseEmbeddingIndex(index_path, dimension_changed).ensure("source-c", reduced)) == {"one"}
    assert dimension_changed.document_calls == [["one", "two"]]


@pytest.mark.parametrize(
    "vector",
    [
        [1.0, 0.0],
        [math.nan, 0.0, 0.0],
        [math.inf, 0.0, 0.0],
    ],
)
def test_dense_index_rejects_invalid_provider_vectors(tmp_path, vector):
    provider = _FakeProvider(vectors={"one": vector})
    index = DenseEmbeddingIndex(tmp_path / "dense.json", provider)
    with pytest.raises(EmbeddingResponseError):
        index.ensure("source", [_chunk("one")])


def test_dense_retriever_rejects_invalid_query_vector_instead_of_falling_back():
    provider = _FakeProvider(query=[1.0, 0.0])
    retriever = HybridRetriever(
        [_chunk("one")],
        dense_vectors={"one": [1.0, 0.0, 0.0]},
        embedding_provider=provider,
    )
    with pytest.raises(EmbeddingResponseError, match="dense embedding query"):
        retriever.search("query")


def test_local_golden_set_remains_offline_and_hits_all_expected_sources(tmp_path):
    knowledge_root = Path(__file__).parents[2] / "platform_backend" / "knowledge"
    service = KnowledgeService(knowledge_root, tmp_path / "index.json")
    report = evaluate_retrieval(service, knowledge_root / "retrieval_golden.json")

    assert service.status()["retrieval_mode"] == "hash_hybrid"
    assert report["case_count"] == 5
    assert report["hit_at_k"] == 1.0
    assert report["mrr"] >= 0.8
