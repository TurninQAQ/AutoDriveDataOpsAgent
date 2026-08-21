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


class AgentStepAction(str, Enum):
    """The only actions an adaptive read-only step may request."""

    CALL_TOOL = "CALL_TOOL"
    FINISH = "FINISH"


class EvidenceType(str, Enum):
    LIVE_TASK = "LIVE_TASK"
    LIVE_GPU = "LIVE_GPU"
    STATIC_KNOWLEDGE = "STATIC_KNOWLEDGE"
    LIVE_QUEUE = "LIVE_QUEUE"
    LIVE_LOG = "LIVE_LOG"
    LIVE_CONTAINER = "LIVE_CONTAINER"
    PLATFORM_HEALTH = "PLATFORM_HEALTH"
    DIAGNOSTIC_CONTEXT = "DIAGNOSTIC_CONTEXT"
    DIAGNOSIS = "DIAGNOSIS"
    RECOVERY_STATE = "RECOVERY_STATE"


class EvidenceRecord(BaseModel):
    """Bounded evidence coverage metadata; never a full tool result."""

    type: EvidenceType
    source_tool: str
    timestamp: float
    summary: str = Field(default="", max_length=500)
    # Optional provenance keeps V1.5.x artifacts loadable while preventing
    # task-scoped evidence from satisfying a different task's goal.
    task_name: str | None = None
    dataset_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GoalType(str, Enum):
    ANSWER_KNOWLEDGE = "ANSWER_KNOWLEDGE"
    REPORT_LIVE_STATE = "REPORT_LIVE_STATE"
    DIAGNOSE_ROOT_CAUSE = "DIAGNOSE_ROOT_CAUSE"
    EXPLAIN_WITH_PLATFORM_RULES = "EXPLAIN_WITH_PLATFORM_RULES"
    VERIFY_RECOVERY_STATE = "VERIFY_RECOVERY_STATE"
    PREPARE_TASK_PLAN = "PREPARE_TASK_PLAN"
    PREPARE_WRITE_ACTION = "PREPARE_WRITE_ACTION"
    GENERAL_ASSISTANCE = "GENERAL_ASSISTANCE"


class GoalProgress(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SATISFIED = "SATISFIED"
    BLOCKED = "BLOCKED"


class GoalContract(BaseModel):
    """Frozen, evidence-level completion contract for one request."""

    goal_type: GoalType
    domain_intent: AgentIntent
    required_conditions: list[str] = Field(default_factory=list)
    schema_version: str = "v1.6.2"


class AgentGoal(BaseModel):
    """A request-level user outcome, not a tool-routing instruction."""

    goal_type: GoalType
    target: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    completion_state: GoalProgress = GoalProgress.NOT_STARTED


class GoalEvaluation(BaseModel):
    """Deterministic, bounded goal progress produced from observed evidence."""

    state: GoalProgress
    satisfied_conditions: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=500)


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
    goal: AgentGoal | None = None
    task_name: str | None = None
    dataset_name: str | None = None
    stage: str | None = None
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    decision_summary: str = ""
    task_draft: dict[str, Any] | None = None
    write_action: dict[str, Any] | None = None


class AgentStepDecision(BaseModel):
    """A short, auditable next-evidence decision.

    The model is deliberately not a reasoning transcript.  The workflow validates
    the action, intent revision and tool boundary before any tool is executed.
    """

    action: AgentStepAction
    tool_call: ToolCallSpec | None = None
    evidence_sufficient: bool = False
    revised_intent: AgentIntent | None = None
    decision_summary: str = Field(default="", max_length=1000)


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
    termination_reason: str | None = None
    adaptive_step_count: int = 0
    evidence_sufficient: bool | None = None
    initial_plan: dict[str, Any] | None = None
    adaptive_steps: list[dict[str, Any]] = Field(default_factory=list)
    goal: AgentGoal | None = None
    goal_progress: GoalProgress | None = None


class ConversationTurn(BaseModel):
    user: str
    assistant_summary: str
    intent: AgentIntent
