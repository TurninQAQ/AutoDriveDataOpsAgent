from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    chunk_id: str
    source_path: str
    title: str
    section: str = ""
    content: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedKnowledge(BaseModel):
    chunk_id: str
    source_path: str
    title: str
    section: str = ""
    content: str
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        suffix = f"#{self.section}" if self.section else ""
        return f"{self.source_path}{suffix}"


class KnowledgeIndexStats(BaseModel):
    schema_version: int
    source_fingerprint: str
    document_count: int
    chunk_count: int
    built_at: str


class KnowledgeSearchResult(BaseModel):
    query: str
    results: list[RetrievedKnowledge] = Field(default_factory=list)
    index_stats: KnowledgeIndexStats | None = None
