from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


WriteStatus = Literal[
    "VERIFIED",
    "FAILED",
    "PRECONDITION_FAILED",
    "VERIFICATION_FAILED",
    "UNKNOWN_OUTCOME",
    "REJECTED",
]


class WriteResult(BaseModel):
    id: str
    action: str
    status: WriteStatus
    verified: bool = False
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
