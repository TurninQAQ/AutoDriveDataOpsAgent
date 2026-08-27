from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class EmptyArgs(BaseModel):
    pass


class GetTaskDetailArgs(BaseModel):
    task_name: str
    include_airflow_runs: bool = True
    run_limit: int = Field(default=20, ge=1, le=100)


class GetGpuPoolArgs(BaseModel):
    cleanup_dead: bool = True


class GetQueueStateArgs(BaseModel):
    task_name: str = ""


class DiagnoseTaskArgs(BaseModel):
    task_name: str
    dataset_name: str = ""


class SearchKnowledgeArgs(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)


class TaskDraftArgs(BaseModel):
    task_prefix: str
    dataset_path: str
    dataset_name: str | None = None
    task_type: str | None = None
    pipeline_stages: list[Any] | None = None
    max_active_runs: int | None = Field(default=None, ge=1)


class SetTaskPriorityArgs(BaseModel):
    task_name: str
    priority: int = Field(ge=0, le=100)


class ResumeTaskArgs(BaseModel):
    task_name: str
    datasets: list[str] | None = None


class StopTaskArgs(BaseModel):
    task_name: str
    datasets: list[str] | None = None


class DeleteTaskArgs(BaseModel):
    task_name: str


class SubmitArtifactArgs(BaseModel):
    artifact_id: str


class RuntimeSubmitTaskArgs(BaseModel):
    task_prefix: str
    config: dict[str, Any]
    precondition: dict[str, Any]


class RuntimeSetTaskPriorityArgs(SetTaskPriorityArgs):
    precondition: dict[str, Any]


class RuntimeResumeTaskArgs(ResumeTaskArgs):
    precondition: dict[str, Any]


class RuntimeStopTaskArgs(StopTaskArgs):
    precondition: dict[str, Any]


class RuntimeDeleteTaskArgs(DeleteTaskArgs):
    precondition: dict[str, Any]


class CapturePreconditionArgs(BaseModel):
    task_name: str = ""


class GetTaskConfigForVerificationArgs(BaseModel):
    task_name: str


class VerificationSnapshotArgs(BaseModel):
    task_name: str
    datasets: list[str] | None = None
    airflow_limit: int = Field(default=100, ge=1, le=200)
