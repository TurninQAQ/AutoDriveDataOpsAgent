from __future__ import annotations

import asyncio
from pathlib import Path

from .embeddings import DenseEmbeddingIndex, EmbeddingProvider
from .index import KnowledgeIndex
from .models import KnowledgeIndexStats, KnowledgeSearchResult, RetrievedKnowledge
from .retriever import HybridRetriever, HybridRetrieverConfig
from .sources import KnowledgeSourceConfig, KnowledgeSourceLoader


class KnowledgeService:
    def __init__(
        self,
        source_dir: Path,
        index_file: Path,
        *,
        top_k: int = 5,
        min_score: float = 0.08,
        max_chunk_chars: int = 1800,
        overlap_chars: int = 180,
        lexical_weight: float = 0.65,
        vector_weight: float = 0.35,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_index_file: Path | None = None,
    ):
        self.source_dir = Path(source_dir)
        self.index_file = Path(index_file)
        self.retriever_config = HybridRetrieverConfig(
            top_k=max(1, top_k),
            min_score=max(0.0, min_score),
            lexical_weight=max(0.0, float(lexical_weight)),
            vector_weight=max(0.0, float(vector_weight)),
        )
        self.loader = KnowledgeSourceLoader(
            KnowledgeSourceConfig(
                source_dir=self.source_dir,
                max_chunk_chars=max_chunk_chars,
                overlap_chars=overlap_chars,
            )
        )
        self.index = KnowledgeIndex(self.index_file, self.loader)
        self.embedding_provider = embedding_provider
        self.embedding_index = None
        if embedding_provider is not None:
            sidecar = Path(embedding_index_file) if embedding_index_file else self.index_file.with_suffix('.embeddings.json')
            self.embedding_index = DenseEmbeddingIndex(sidecar, embedding_provider)
        self._fingerprint: str | None = None
        self._retriever: HybridRetriever | None = None

    def build(self, force: bool = False, reset_embeddings: bool = False) -> KnowledgeIndexStats:
        stats = self.index.build(force=force)
        if self.embedding_index is not None:
            chunks = self.index.load_chunks(ensure_fresh=False)
            self.embedding_index.ensure(
                stats.source_fingerprint,
                chunks,
                force=force,
                reset=reset_embeddings,
            )
        self._fingerprint = None
        self._retriever = None
        return stats

    def status(self) -> dict:
        stats = self.index.stats()
        current_fingerprint = self.loader.fingerprint()
        result = {
            "source_dir": str(self.source_dir),
            "index_file": str(self.index_file),
            "source_exists": self.source_dir.exists(),
            "index_exists": self.index_file.exists(),
            "current_source_fingerprint": current_fingerprint,
            "index_fresh": bool(stats and stats.source_fingerprint == current_fingerprint),
            "retrieval_mode": "gemini_hybrid" if self.embedding_provider is not None else "hash_hybrid",
            "lexical_weight": self.retriever_config.lexical_weight,
            "vector_weight": self.retriever_config.vector_weight,
            "stats": stats.model_dump(mode="json") if stats else None,
        }
        if self.embedding_index is not None:
            result["embedding"] = self.embedding_index.status()
        else:
            result["embedding"] = {"enabled": False, "provider": "hash", "dimension": self.retriever_config.vector_dimension}
        return result

    def _get_retriever(self) -> HybridRetriever:
        stats = self.index.ensure()
        if self._retriever is None or self._fingerprint != stats.source_fingerprint:
            chunks = self.index.load_chunks(ensure_fresh=False)
            dense_vectors = None
            if self.embedding_index is not None:
                dense_vectors = self.embedding_index.ensure(stats.source_fingerprint, chunks)
            self._retriever = HybridRetriever(
                chunks,
                self.retriever_config,
                dense_vectors=dense_vectors,
                embedding_provider=self.embedding_provider,
            )
            self._fingerprint = stats.source_fingerprint
        return self._retriever

    def search(self, query: str, top_k: int | None = None) -> KnowledgeSearchResult:
        retriever = self._get_retriever()
        results = retriever.search(query, top_k=top_k)
        return KnowledgeSearchResult(query=query, results=results, index_stats=self.index.stats())


class AsyncKnowledgeRetriever:
    """Small async adapter used by LangGraph/Sequential Agent nodes."""

    def __init__(self, service: KnowledgeService, enabled: bool = True):
        self.service = service
        self.enabled = enabled

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedKnowledge]:
        if not self.enabled:
            return []
        return (await asyncio.to_thread(self.service.search, query, top_k)).results
