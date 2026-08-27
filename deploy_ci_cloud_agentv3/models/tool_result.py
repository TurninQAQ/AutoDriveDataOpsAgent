from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


ToolResultKind = Literal[
    "OBSERVATION",
    "PREPARED_ARTIFACT",
    "ACTION_PROPOSAL",
    "TOOL_ERROR",
    "WRITE_RESULT",
    "REVIEW_REJECTED",
]


class ToolResult(BaseModel):
    kind: ToolResultKind
    tool_name: str
    call_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
