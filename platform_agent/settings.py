from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platform_core.settings import PlatformSettings


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True)
class AgentSettings:
    provider: str
    model: str
    runtime: str
    max_tool_calls: int
    session_dir: Path
    temperature: float
    base_url: str | None
    knowledge_enabled: bool
    knowledge_source_dir: Path
    knowledge_index_file: Path
    knowledge_top_k: int
    knowledge_min_score: float
    knowledge_embedding_provider: str
    knowledge_embedding_model: str
    knowledge_embedding_dimension: int
    knowledge_embedding_batch_size: int
    knowledge_embedding_index_file: Path
    knowledge_qwen_instruct: str
    knowledge_lexical_weight: float
    knowledge_vector_weight: float
    approval_dir: Path
    approval_ttl_sec: int
    verification_attempts: int
    verification_interval_sec: float
    trace_enabled: bool
    trace_dir: Path
    audit_file: Path
    trace_max_value_chars: int
    trace_retention_days: int
    trace_max_files: int
    audit_max_bytes: int
    audit_backup_count: int

    @classmethod
    def from_env(cls, platform_settings: PlatformSettings | None = None) -> "AgentSettings":
        platform_settings = platform_settings or PlatformSettings.from_env()
        provider = os.environ.get("PLATFORM_AGENT_PROVIDER", "auto").strip().lower() or "auto"
        qwen_provider = provider in {"qwen", "dashscope", "aliyun", "alibaba"}
        model_raw = os.environ.get("PLATFORM_AGENT_MODEL", "").strip()
        if model_raw:
            model = model_raw
        else:
            gemini_configured = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            qwen_configured = bool(
                os.environ.get("DASHSCOPE_API_KEY") and os.environ.get("DASHSCOPE_OPENAI_BASE_URL")
            )
            gemini_provider = provider in {"gemini", "google", "google-genai", "google_genai"}
            model = (
                "qwen3.7-flash"
                if (provider == "auto" and qwen_configured) or qwen_provider
                else "gemini-3.7-flash"
                if (gemini_provider or (provider == "auto" and gemini_configured))
                else "gpt-5.4-mini"
            )
        runtime = os.environ.get("PLATFORM_AGENT_RUNTIME", "langgraph").strip().lower() or "langgraph"
        max_tool_calls = int(os.environ.get("PLATFORM_AGENT_MAX_TOOL_CALLS", "6"))
        temperature = float(os.environ.get("PLATFORM_AGENT_TEMPERATURE", "0"))
        session_dir_raw = os.environ.get("PLATFORM_AGENT_SESSION_DIR", "").strip()
        session_dir = Path(session_dir_raw) if session_dir_raw else platform_settings.state_dir / "agent_sessions"
        base_url = (
            os.environ.get("OPENAI_BASE_URL", "").strip()
            or os.environ.get("DASHSCOPE_OPENAI_BASE_URL", "").strip()
            or None
        )

        # This default works both in source checkout and deployed runtime:
        #   <repo>/platform_agent/settings.py         -> <repo>/knowledge
        #   <runtime>/opt_airflow/platform_agent/... -> <runtime>/opt_airflow/knowledge
        default_knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge"
        knowledge_source_raw = os.environ.get("PLATFORM_AGENT_KNOWLEDGE_DIR", "").strip()
        knowledge_source_dir = Path(knowledge_source_raw) if knowledge_source_raw else default_knowledge_dir
        knowledge_index_raw = os.environ.get("PLATFORM_AGENT_KNOWLEDGE_INDEX", "").strip()
        knowledge_index_file = (
            Path(knowledge_index_raw)
            if knowledge_index_raw
            else platform_settings.state_dir / "agent_knowledge" / "index.json"
        )
        knowledge_top_k = int(os.environ.get("PLATFORM_AGENT_KNOWLEDGE_TOP_K", "5"))
        knowledge_min_score = float(os.environ.get("PLATFORM_AGENT_KNOWLEDGE_MIN_SCORE", "0.08"))
        knowledge_embedding_provider = os.environ.get("PLATFORM_RAG_EMBED_PROVIDER", "hash").strip().lower() or "hash"
        qwen_embedding_provider = knowledge_embedding_provider in {"qwen", "dashscope", "aliyun", "alibaba"}
        default_embedding_model = "qwen3.7-text-embedding" if qwen_embedding_provider else "gemini-embedding-2"
        default_embedding_dimension = "1024" if qwen_embedding_provider else "768"
        default_embedding_batch_size = "20" if qwen_embedding_provider else "32"
        knowledge_embedding_model = os.environ.get("PLATFORM_RAG_EMBED_MODEL", default_embedding_model).strip() or default_embedding_model
        knowledge_embedding_dimension = int(os.environ.get("PLATFORM_RAG_EMBED_DIM", default_embedding_dimension))
        knowledge_embedding_batch_size = int(os.environ.get("PLATFORM_RAG_EMBED_BATCH_SIZE", default_embedding_batch_size))
        knowledge_embedding_index_raw = os.environ.get("PLATFORM_RAG_EMBED_INDEX", "").strip()
        knowledge_embedding_index_file = (
            Path(knowledge_embedding_index_raw)
            if knowledge_embedding_index_raw
            else platform_settings.state_dir / "agent_knowledge" / "embeddings.json"
        )
        default_lexical = "0.50" if knowledge_embedding_provider in {"gemini", "qwen", "google", "dashscope", "aliyun", "alibaba"} else "0.65"
        default_vector = "0.50" if knowledge_embedding_provider in {"gemini", "qwen", "google", "dashscope", "aliyun", "alibaba"} else "0.35"
        lexical_raw = os.environ.get("PLATFORM_RAG_LEXICAL_WEIGHT", "").strip() or default_lexical
        vector_raw = os.environ.get("PLATFORM_RAG_VECTOR_WEIGHT", "").strip() or default_vector
        knowledge_lexical_weight = float(lexical_raw)
        knowledge_vector_weight = float(vector_raw)
        weight_sum = knowledge_lexical_weight + knowledge_vector_weight
        if weight_sum <= 0:
            knowledge_lexical_weight, knowledge_vector_weight = 1.0, 0.0
        else:
            knowledge_lexical_weight /= weight_sum
            knowledge_vector_weight /= weight_sum
        approval_dir_raw = os.environ.get("PLATFORM_AGENT_APPROVAL_DIR", "").strip()
        approval_dir = Path(approval_dir_raw) if approval_dir_raw else platform_settings.state_dir / "agent_approvals"
        approval_ttl_sec = int(os.environ.get("PLATFORM_AGENT_APPROVAL_TTL_SEC", "900"))
        verification_attempts = int(os.environ.get("PLATFORM_AGENT_VERIFY_ATTEMPTS", "5"))
        verification_interval_sec = float(os.environ.get("PLATFORM_AGENT_VERIFY_INTERVAL_SEC", "1.0"))
        trace_dir_raw = os.environ.get("PLATFORM_AGENT_TRACE_DIR", "").strip()
        trace_dir = Path(trace_dir_raw) if trace_dir_raw else platform_settings.state_dir / "agent_traces"
        audit_file_raw = os.environ.get("PLATFORM_AGENT_AUDIT_FILE", "").strip()
        audit_file = Path(audit_file_raw) if audit_file_raw else platform_settings.state_dir / "agent_audit" / "audit.jsonl"
        trace_max_value_chars = int(os.environ.get("PLATFORM_AGENT_TRACE_MAX_VALUE_CHARS", "16000"))
        trace_retention_days = int(os.environ.get("PLATFORM_AGENT_TRACE_RETENTION_DAYS", "14"))
        trace_max_files = int(os.environ.get("PLATFORM_AGENT_TRACE_MAX_FILES", "5000"))
        audit_max_bytes = int(os.environ.get("PLATFORM_AGENT_AUDIT_MAX_BYTES", str(20 * 1024 * 1024)))
        audit_backup_count = int(os.environ.get("PLATFORM_AGENT_AUDIT_BACKUP_COUNT", "5"))

        return cls(
            provider=provider,
            model=model,
            runtime=runtime,
            max_tool_calls=max(1, max_tool_calls),
            session_dir=session_dir,
            temperature=temperature,
            base_url=base_url,
            knowledge_enabled=_env_bool("PLATFORM_AGENT_KNOWLEDGE_ENABLED", True),
            knowledge_source_dir=knowledge_source_dir,
            knowledge_index_file=knowledge_index_file,
            knowledge_top_k=max(1, knowledge_top_k),
            knowledge_min_score=max(0.0, knowledge_min_score),
            knowledge_embedding_provider=knowledge_embedding_provider,
            knowledge_embedding_model=knowledge_embedding_model,
            knowledge_embedding_dimension=max(128, min(3072, knowledge_embedding_dimension)),
            knowledge_embedding_batch_size=max(1, knowledge_embedding_batch_size),
            knowledge_embedding_index_file=knowledge_embedding_index_file,
            knowledge_qwen_instruct=os.environ.get("PLATFORM_RAG_QWEN_INSTRUCT", "").strip(),
            knowledge_lexical_weight=knowledge_lexical_weight,
            knowledge_vector_weight=knowledge_vector_weight,
            approval_dir=approval_dir,
            approval_ttl_sec=max(30, approval_ttl_sec),
            verification_attempts=max(1, verification_attempts),
            verification_interval_sec=max(0.0, verification_interval_sec),
            trace_enabled=_env_bool("PLATFORM_AGENT_TRACE_ENABLED", True),
            trace_dir=trace_dir,
            audit_file=audit_file,
            trace_max_value_chars=max(1024, trace_max_value_chars),
            trace_retention_days=max(0, trace_retention_days),
            trace_max_files=max(0, trace_max_files),
            audit_max_bytes=max(0, audit_max_bytes),
            audit_backup_count=max(0, audit_backup_count),
        )
