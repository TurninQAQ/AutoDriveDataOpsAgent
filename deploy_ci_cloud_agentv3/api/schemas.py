from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class EditReviewRequest(BaseModel):
    args: dict[str, Any]
    fingerprint: str | None = None


class ApproveRequest(BaseModel):
    fingerprint: str | None = None


class RejectRequest(BaseModel):
    reason: str = "rejected by reviewer"
    fingerprint: str | None = None


class RunResponse(BaseModel):
    run_id: str
    thread_id: str
    status: Literal["CREATED","RUNNING","WAITING_FOR_REVIEW","COMPLETED","FAILED","UNCERTAIN"]
    created_at: float
    updated_at: float
    final_response: dict[str, Any] | None = None
    pending_action: dict[str, Any] | None = None
    error: str | None = None
