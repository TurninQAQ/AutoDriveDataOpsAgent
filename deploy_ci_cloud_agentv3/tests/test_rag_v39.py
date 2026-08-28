from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy_ci_cloud_agentv3.rag.embeddings import DeterministicEmbeddingProvider, GeminiEmbeddingProvider
from deploy_ci_cloud_agentv3.rag.index import DenseIndex
from deploy_ci_cloud_agentv3.rag.service import RAGService


@pytest.fixture
def knowledge_dir(tmp_path):
    (tmp_path/"gpu.md").write_text("# GPU Scheduling\nGPU reservation priority queue memory scheduling policy",encoding="utf-8")
    (tmp_path/"airflow.md").write_text("# Airflow\nAirflow DagRun scheduler failure diagnosis",encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_bm25_mode(knowledge_dir, tmp_path):
    service=RAGService(knowledge_dir,tmp_path/"idx",mode="hybrid",embedding_provider=None)
    result=await service.search("GPU priority",top_k=2)
    assert result["mode"] == "bm25"
    assert result["results"][0]["source"] == "gpu.md"


@pytest.mark.asyncio
async def test_dense_mode_with_fake_embedding(knowledge_dir, tmp_path):
    provider=DeterministicEmbeddingProvider(32); index=DenseIndex(tmp_path/"idx")
    await index.build(knowledge_dir,provider)
    service=RAGService(knowledge_dir,tmp_path/"idx",mode="dense",embedding_provider=provider)
    result=await service.search("Airflow scheduler",top_k=2)
    assert result["mode"] == "dense"
    assert all(row["dense_rank"] is not None for row in result["results"])


@pytest.mark.asyncio
async def test_hybrid_rrf(knowledge_dir, tmp_path):
    provider=DeterministicEmbeddingProvider(32); await DenseIndex(tmp_path/"idx").build(knowledge_dir,provider)
    result=await RAGService(knowledge_dir,tmp_path/"idx",mode="hybrid",embedding_provider=provider).search("GPU scheduling",top_k=2)
    assert result["mode"] == "hybrid"
    assert result["results"][0]["fusion_score"] is not None


@pytest.mark.asyncio
async def test_index_invalidates_when_knowledge_changes(knowledge_dir, tmp_path):
    provider=DeterministicEmbeddingProvider(32); index=DenseIndex(tmp_path/"idx"); await index.build(knowledge_dir,provider)
    assert index.is_fresh(knowledge_dir,model=provider.model_name,dimension=32)
    (knowledge_dir/"gpu.md").write_text("changed content",encoding="utf-8")
    assert not index.is_fresh(knowledge_dir,model=provider.model_name,dimension=32)


@pytest.mark.asyncio
async def test_index_invalidates_when_embedding_model_changes(knowledge_dir, tmp_path):
    provider=DeterministicEmbeddingProvider(32); index=DenseIndex(tmp_path/"idx"); await index.build(knowledge_dir,provider)
    assert not index.is_fresh(knowledge_dir,model="other",dimension=32)


@pytest.mark.asyncio
async def test_search_knowledge_returns_sources(knowledge_dir, tmp_path):
    result=await RAGService(knowledge_dir,tmp_path/"idx",mode="bm25").search("Airflow",top_k=1)
    assert result["results"][0]["source"] == "airflow.md"


def test_dense_disabled_is_not_reported_as_dense(knowledge_dir, tmp_path):
    service=RAGService(knowledge_dir,tmp_path/"idx",mode="dense",embedding_provider=None)
    assert service.effective_mode == "bm25"


@pytest.mark.asyncio
async def test_gemini_documents_are_embedded_one_at_a_time():
    calls=[]
    class Models:
        def embed_content(self, **kwargs):
            calls.append(kwargs["contents"])
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0,0.0])])
    provider=GeminiEmbeddingProvider(dimension=2,client=SimpleNamespace(models=Models()))
    vectors=await provider.embed_documents(["doc one","doc two"])
    assert len(vectors)==2 and len(calls)==2
