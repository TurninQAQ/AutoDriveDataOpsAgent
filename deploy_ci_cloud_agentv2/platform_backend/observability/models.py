from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    trace_id: str
    parent_trace_id: str | None = None
    event_id: str
    timestamp: float
    stage: str
    name: str
    status: str = "ok"
    duration_ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class AuditRecord(BaseModel):
    trace_id: str
    parent_trace_id: str | None = None
    kind: str
    thread_id: str = "default"
    started_at: float
    ended_at: float
    latency_ms: float
    status: str
    user_request: str = ""
    intent: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    mutations: list[dict[str, Any]] = Field(default_factory=list)
    verification: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    response_summary: str = ""


class TraceSummary(BaseModel):
    trace_id: str
    parent_trace_id: str | None = None
    kind: str
    status: str
    started_at: float
    ended_at: float
    latency_ms: float
    thread_id: str
    intent: str | None = None
    user_request: str = ""
    response_summary: str = ""
    error_count: int = 0
