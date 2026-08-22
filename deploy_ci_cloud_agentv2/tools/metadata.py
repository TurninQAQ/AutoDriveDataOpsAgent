"""ToolSpec metadata used for deterministic structural runtime checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any

from ..agent.immutable import FrozenMapping


class ToolKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Idempotency(str, Enum):
    SAFE_RETRY = "SAFE_RETRY"
    RECONCILE_BEFORE_RETRY = "RECONCILE_BEFORE_RETRY"
    NO_RETRY = "NO_RETRY"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    risk: RiskLevel
    schema: Mapping[str, Any]
    parallel_safe: bool = False
    requires_precondition: bool = False
    verification: str = "NONE"
    idempotency: Idempotency = Idempotency.NO_RETRY

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolSpec.name must not be empty")
        if not isinstance(self.schema, Mapping):
            raise TypeError("ToolSpec.schema must be a mapping")
        object.__setattr__(self, "schema", FrozenMapping(self.schema))
        if self.kind is ToolKind.READ and self.verification != "NONE":
            raise ValueError("Phase B READ tools cannot require WRITE verification")
        if self.kind is ToolKind.READ and self.idempotency is not Idempotency.SAFE_RETRY:
            raise ValueError("Phase B READ tools must be SAFE_RETRY")
