from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ProposalResult(BaseModel):
    kind: Literal["ACTION_PROPOSAL"] = "ACTION_PROPOSAL"
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str
    expected_effect: str
