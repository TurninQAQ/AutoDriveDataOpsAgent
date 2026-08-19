from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


PipelineItem = str | list[str]


class DatasetSpec(BaseModel):
    dataset_name: str
    dataset_path: str
    tier: str = "small"
    pool: str = "default_pool"
    timeout_min: int = 60
    images: dict[str, str] = Field(default_factory=dict)
    image_qc: str = ""

    @field_validator("dataset_name", "dataset_path", "tier", "pool")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class TaskSpec(BaseModel):
    """Agent-facing task specification before the timestamped task_name exists."""

    task_prefix: str
    task_type: str = ""
    priority: int | None = None
    pipeline_stages: list[PipelineItem]
    max_active_runs: int = 5
    task_exclusive: bool = True
    task_lock_wait_interval_sec: int = 10
    preempt_grace_timeout_min: int = 60
    gpu_ids: str = ""
    gpu_stages: str = ""
    exclusive_gpu_stages: str = ""
    exclusive_gpu_idle_used_max_mb: int = 512
    gpu_stage_memory_mb: dict[str, int] = Field(default_factory=dict)
    gpu_wait_interval_sec: int = 10
    gpu_reservation_pending_sec: int = 60
    datasets: list[DatasetSpec] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    code: str
    path: str
    message: str
    severity: str = Field(default="error", pattern="^(error|warning)$")


class TaskPlanningResult(BaseModel):
    user_text: str
    valid: bool
    task_spec: TaskSpec | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    yaml_text: str = ""
    resolved_priority: int | None = None
    priority_source: str = ""
    defaults_used: list[str] = Field(default_factory=list)
    explicit_fields: list[str] = Field(default_factory=list)
    unresolved_fields: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "warning"]
