from __future__ import annotations

from pathlib import Path

from deploy_ci_cloud_agentv3.config import Settings
from .embeddings import DeterministicEmbeddingProvider, GeminiEmbeddingProvider
from .service import RAGService


def build_embedding_provider(settings: Settings, *, test_deterministic: bool = False):
    if test_deterministic: return DeterministicEmbeddingProvider(settings.rag_embedding_dim)
    if settings.rag_dense_provider == "disabled": return None
    if settings.rag_dense_provider == "gemini": return GeminiEmbeddingProvider(settings.rag_dense_model, settings.rag_embedding_dim)
    raise ValueError(f"unsupported RAG_DENSE_PROVIDER: {settings.rag_dense_provider}")


def build_rag_service(*, settings: Settings | None = None, source_dir: Path | None = None, test_deterministic: bool = False) -> RAGService:
    settings=settings or Settings.from_env(); settings.ensure_dirs()
    source_dir=source_dir or (Path(__file__).resolve().parents[1]/"platform_backend"/"knowledge")
    provider=build_embedding_provider(settings,test_deterministic=test_deterministic)
    return RAGService(source_dir, settings.state_dir/"knowledge_index", mode=settings.rag_mode, embedding_provider=provider)
