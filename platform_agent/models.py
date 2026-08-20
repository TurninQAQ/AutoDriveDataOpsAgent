from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentIntent(str, Enum):
    PLATFORM_HEALTH = "platform_health"
    LIST_TASKS = "list_tasks"
    TASK_STATUS = "task_status"
    TASK_DIAGNOSIS = "task_diagnosis"
    GPU_DIAGNOSIS = "gpu_diagnosis"
    STAGE_FAILURE = "stage_failure"
    UNSUPPORTED_WRITE = "unsupported_write"
    GENERAL_READ = "general_read"
    PLATFORM_KNOWLEDGE = "platform_knowledge"
    TASK_PLANNING = "task_planning"
    SUBMIT_TASK = "submit_task"
    RESUME_TASK = "resume_task"
    SET_TASK_PRIORITY = "set_task_priority"
    STOP_TASK = "stop_task"
    DELETE_TASK = "delete_task"


class ToolCallSpec(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    intent: AgentIntent = Field(
        description=(
            "Evidence-routing intent: platform_knowledge is static platform "
            "mechanisms/docs; gpu_diagnosis is live GPU/resource state; "
            "task_status is current state for a named task; task_diagnosis "
            "requires a concrete task identity; task_planning "
            "generates configuration without execution; submit_task means an "
            "explicit request to submit/start execution."
        )
    )
    task_name: str | None = None
    dataset_name: str | None = None
    stage: str | None = None
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    decision_summary: str = ""
    task_draft: dict[str, Any] | None = None
    write_action: dict[str, Any] | None = None


class ToolObservation(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    data: Any = None
    error: str | None = None


class KnowledgeObservation(BaseModel):
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


class AgentResponse(BaseModel):
    intent: AgentIntent
    trace_id: str | None = None
    summary: str
    root_cause: str | None = None
    evidence: list[str] = Field(default_factory=list)
    knowledge_sources: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    confidence: str = Field(default="medium", pattern="^(low|medium|high)$")
    blocked: bool = False
    errors: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)
    task_plan: dict[str, Any] | None = None
    approval_required: bool = False
    approval_id: str | None = None
    pending_action: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None


class ConversationTurn(BaseModel):
    user: str
    assistant_summary: str
    intent: AgentIntent
