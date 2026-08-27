from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .models import KnowledgeChunk, RetrievedKnowledge
from .text import cosine_sparse, hashed_vector, tokenize
from .embeddings import EmbeddingProvider, cosine_dense, validate_embedding_vector


@dataclass(frozen=True)
class HybridRetrieverConfig:
    top_k: int = 5
    min_score: float = 0.08
    lexical_weight: float = 0.65
    vector_weight: float = 0.35
    vector_dimension: int = 384


class HybridRetriever:
    """BM25 + vector hybrid retriever with a deterministic local fallback.

    The lexical side always uses BM25. When an EmbeddingProvider and dense
    sidecar vectors are supplied, the vector side uses real dense cosine
    similarity (V1.2 Gemini mode). Otherwise it falls back to feature hashing +
    cosine so dependency-light development and CI still work without an API key
    or GPU. The Agent-facing retrieval contract is identical in both modes.
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        config: HybridRetrieverConfig | None = None,
        *,
        dense_vectors: dict[str, list[float]] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.chunks = chunks
        self.config = config or HybridRetrieverConfig()
        self.dense_vectors = dense_vectors or {}
        self.embedding_provider = embedding_provider
        self._tokens = [tokenize(self._search_text(chunk), expand=False) for chunk in chunks]
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._avgdl = (sum(self._doc_lengths) / len(self._doc_lengths)) if self._doc_lengths else 0.0
        self._df: Counter[str] = Counter()
        for tokens in self._tokens:
            self._df.update(set(tokens))
        self._vectors = [hashed_vector(tokens, self.config.vector_dimension) for tokens in self._tokens]

    @staticmethod
    def _search_text(chunk: KnowledgeChunk) -> str:
        return f"{chunk.title} {chunk.title} {chunk.title}\n{chunk.section} {chunk.section}\n{chunk.content}"

    def _bm25(self, query_tokens: list[str], index: int) -> float:
        if not query_tokens or not self.chunks:
            return 0.0
        tf = Counter(self._tokens[index])
        dl = self._doc_lengths[index] or 1
        avgdl = self._avgdl or 1.0
        k1 = 1.5
        b = 0.75
        score = 0.0
        n_docs = len(self.chunks)
        for token in set(query_tokens):
            freq = tf.get(token, 0)
            if not freq:
                continue
            df = self._df.get(token, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf * (freq * (k1 + 1) / denom)
        return score

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[RetrievedKnowledge]:
        if not query.strip() or not self.chunks:
            return []
        query_tokens = tokenize(query, expand=True)
        query_vector = hashed_vector(query_tokens, self.config.vector_dimension)
        dense_query = []
        if self.embedding_provider is not None:
            dense_query = validate_embedding_vector(
                self.embedding_provider.embed_query(query),
                self.embedding_provider.dimension,
                context="dense embedding query",
            )
        raw: list[tuple[int, float, float]] = []
        for index in range(len(self.chunks)):
            lexical = self._bm25(query_tokens, index)
            if dense_query and self.chunks[index].chunk_id in self.dense_vectors:
                vector = max(0.0, cosine_dense(dense_query, self.dense_vectors[self.chunks[index].chunk_id]))
            else:
                vector = max(0.0, cosine_sparse(query_vector, self._vectors[index]))
            raw.append((index, lexical, vector))
        max_lexical = max((item[1] for item in raw), default=0.0)
        scored: list[RetrievedKnowledge] = []
        threshold = self.config.min_score if min_score is None else min_score
        query_lower = query.lower()
        title_phrases = (
            "软抢占", "draining", "recovery", "checkpoint", "gpu", "显存",
            "reservation", "container", "容器", "validate", "airflow", "postgresql", "sqlite",
        )
        for index, lexical, vector in raw:
            lexical_norm = lexical / max_lexical if max_lexical > 0 else 0.0
            score = self.config.lexical_weight * lexical_norm + self.config.vector_weight * vector
            chunk = self.chunks[index]
            heading = (chunk.section or chunk.title).lower()
            heading_bonus = sum(0.15 for phrase in title_phrases if phrase in query_lower and phrase in heading)
            score = min(1.0, score + min(0.30, heading_bonus))
            if score < threshold:
                continue
            scored.append(
                RetrievedKnowledge(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    title=chunk.title,
                    section=chunk.section,
                    content=chunk.content,
                    score=round(score, 6),
                    lexical_score=round(lexical_norm, 6),
                    vector_score=round(vector, 6),
                    metadata=chunk.metadata,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.source_path, item.chunk_id))
        return scored[: max(1, top_k or self.config.top_k)]
