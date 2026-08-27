from __future__ import annotations

from typing import Literal
from pydantic import BaseModel

FinalStatus = Literal["informational", "write_verified", "write_failed", "write_not_executed", "write_uncertain"]


class FinalCandidate(BaseModel):
    status: FinalStatus
    message: str


class FinalResponse(BaseModel):
    status: FinalStatus
    write_result_id: str | None = None
    message: str
