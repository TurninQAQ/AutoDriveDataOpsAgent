from __future__ import annotations

from platform_core.settings import PlatformSettings
from platform_mcp.facade import build_default_facade
from platform_rag.service import KnowledgeService
from platform_rag.embeddings import GeminiEmbeddingProvider, QwenEmbeddingProvider
from platform_planning.service import TaskPlanningService
from platform_observability import ObservedToolClient, TraceRecorder, TraceStore

from .memory import ConversationStore
from .approval import ApprovalStore
from .model import build_model_from_env
from .settings import AgentSettings
from .tool_client import InMemoryMCPToolClient
from .verification import ActionVerifier
from .workflow import build_agent_runtime


def build_knowledge_service(agent_settings: AgentSettings) -> KnowledgeService:
    embedding_provider = None
    provider = agent_settings.knowledge_embedding_provider
    if provider in {"gemini", "google", "google-genai", "google_genai"}:
        embedding_provider = GeminiEmbeddingProvider(
            model_name=agent_settings.knowledge_embedding_model,
            dimension=agent_settings.knowledge_embedding_dimension,
            batch_size=agent_settings.knowledge_embedding_batch_size,
        )
    elif provider in {"qwen", "dashscope", "aliyun", "alibaba"}:
        embedding_provider = QwenEmbeddingProvider(
            model_name=agent_settings.knowledge_embedding_model,
            dimension=agent_settings.knowledge_embedding_dimension,
            batch_size=agent_settings.knowledge_embedding_batch_size,
            instruct=agent_settings.knowledge_qwen_instruct,
        )
    elif provider not in {"hash", "hashed", "feature-hash", "feature_hash", "none", "off", "disabled"}:
        raise ValueError(f"Unsupported PLATFORM_RAG_EMBED_PROVIDER: {provider}")
    return KnowledgeService(
        source_dir=agent_settings.knowledge_source_dir,
        index_file=agent_settings.knowledge_index_file,
        top_k=agent_settings.knowledge_top_k,
        min_score=agent_settings.knowledge_min_score,
        lexical_weight=agent_settings.knowledge_lexical_weight,
        vector_weight=agent_settings.knowledge_vector_weight,
        embedding_provider=embedding_provider,
        embedding_index_file=agent_settings.knowledge_embedding_index_file,
    )


def build_agent_knowledge_service(agent_settings: AgentSettings) -> KnowledgeService | None:
    """Return the knowledge capability only when it is enabled for the Agent."""
    if not agent_settings.knowledge_enabled:
        return None
    return build_knowledge_service(agent_settings)



def build_trace_recorder(agent_settings: AgentSettings) -> TraceRecorder:
    store = TraceStore(agent_settings.trace_dir, agent_settings.audit_file)
    store.maintenance(
        retention_days=agent_settings.trace_retention_days,
        max_trace_files=agent_settings.trace_max_files,
        audit_max_bytes=agent_settings.audit_max_bytes,
        audit_backup_count=agent_settings.audit_backup_count,
    )
    return TraceRecorder(
        store,
        enabled=agent_settings.trace_enabled,
        max_value_chars=agent_settings.trace_max_value_chars,
        maintenance_kwargs={
            "retention_days": agent_settings.trace_retention_days,
            "max_trace_files": agent_settings.trace_max_files,
            "audit_max_bytes": agent_settings.audit_max_bytes,
            "audit_backup_count": agent_settings.audit_backup_count,
        },
    )


def build_default_agent():
    platform_settings = PlatformSettings.from_env()
    agent_settings = AgentSettings.from_env(platform_settings)
    knowledge_service = build_agent_knowledge_service(agent_settings)
    facade = build_default_facade(
        platform_settings,
        knowledge_service=knowledge_service,
    )
    raw_tool_client = InMemoryMCPToolClient(facade)
    trace_recorder = build_trace_recorder(agent_settings)
    tool_client = ObservedToolClient(raw_tool_client, trace_recorder)
    model = build_model_from_env(
        agent_settings.provider,
        agent_settings.model,
        agent_settings.temperature,
    )
    memory = ConversationStore(agent_settings.session_dir)
    approval_store = ApprovalStore(agent_settings.approval_dir, ttl_sec=agent_settings.approval_ttl_sec)
    task_planning_service = TaskPlanningService.from_env()
    return build_agent_runtime(
        agent_settings.runtime,
        model,
        tool_client,
        memory,
        max_tool_calls=agent_settings.max_tool_calls,
        max_steps=agent_settings.max_steps,
        max_identical_tool_calls=agent_settings.max_identical_tool_calls,
        max_consecutive_tool_failures=agent_settings.max_consecutive_tool_failures,
        # Production RAG is now selected through the read-only MCP
        # search_knowledge Tool. The workflow parameter remains optional for
        # legacy V0.5 tests and historical evaluation collectors.
        knowledge_retriever=None,
        knowledge_top_k=agent_settings.knowledge_top_k,
        task_planning_service=task_planning_service,
        approval_store=approval_store,
        action_verifier=ActionVerifier(tool_client, attempts=agent_settings.verification_attempts, interval_sec=agent_settings.verification_interval_sec),
        trace_recorder=trace_recorder,
    )
