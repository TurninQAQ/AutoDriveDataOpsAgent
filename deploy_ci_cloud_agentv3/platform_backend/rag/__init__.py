"""Local RAG subsystem for platform documentation and operational runbooks."""

from .models import KnowledgeChunk, RetrievedKnowledge, KnowledgeSearchResult
from .service import KnowledgeService

__all__ = ["KnowledgeChunk", "RetrievedKnowledge", "KnowledgeSearchResult", "KnowledgeService"]
