"""V3 keeps RAG as a normal MCP READ tool and reuses the V2 platform backend retrieval stack."""

from deploy_ci_cloud_agentv2.platform_backend.rag.service import KnowledgeService

__all__ = ["KnowledgeService"]
