from __future__ import annotations

import asyncio
import hashlib
import math
import os
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm else values


class DeterministicEmbeddingProvider:
    """Stable test-only embedding. Never advertised as a production Dense model."""
    model_name = "deterministic-test-embedding"

    def __init__(self, dimension: int = 64):
        self.dimension = int(dimension)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        lowered = text.lower()
        for token in lowered.replace("/", " ").replace("_", " ").split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            values[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        return _normalize(values)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbeddingProvider:
    """google-genai adapter for gemini-embedding-2 retrieval embeddings.

    Documents are embedded one-by-one intentionally. gemini-embedding-2 can aggregate
    multiple contents into one representation, which is not a document-index batch.
    """
    def __init__(self, model_name: str = "gemini-embedding-2", dimension: int = 768, api_key: str | None = None, client=None):
        self.model_name = model_name
        self.dimension = int(dimension)
        if client is not None:
            self.client = client
            self._types = None
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai is required for Gemini dense retrieval") from exc
        key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini dense retrieval")
        self.client = genai.Client(api_key=key)
        self._types = types

    def _sync_embed(self, text: str) -> list[float]:
        formatted = text
        if self._types is not None:
            config = self._types.EmbedContentConfig(output_dimensionality=self.dimension)
        else:
            config = {"output_dimensionality": self.dimension}
        response = self.client.models.embed_content(model=self.model_name, contents=formatted, config=config)
        embeddings = getattr(response, "embeddings", None) or []
        if len(embeddings) != 1:
            raise RuntimeError(f"expected one embedding, got {len(embeddings)}")
        raw = getattr(embeddings[0], "values", embeddings[0])
        values = [float(v) for v in raw]
        if len(values) != self.dimension or not all(math.isfinite(v) for v in values):
            raise RuntimeError("invalid Gemini embedding response")
        return _normalize(values)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(await asyncio.to_thread(self._sync_embed, f"task: search result | {text}"))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._sync_embed, f"task: question answering | query: {text}")
