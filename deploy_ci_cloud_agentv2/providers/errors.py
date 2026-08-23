"""Provider boundary error taxonomy."""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.decision_ingress import AgentDecisionValidationError
from .model import ProviderUnavailable


class ProviderResponseInvalid(AgentDecisionValidationError):
    """The provider returned no structurally usable AgentDecision proposal."""


@dataclass(frozen=True)
class ProviderTransportFailure(ProviderUnavailable):
    error_code: str
    retryable: bool
    status_code: int | None = None

    def __str__(self) -> str:
        return self.error_code
