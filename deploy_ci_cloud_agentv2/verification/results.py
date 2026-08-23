"""Deterministic WRITE verification result models."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    target: str
    reason_code: str
    observation_id: str | None = None
