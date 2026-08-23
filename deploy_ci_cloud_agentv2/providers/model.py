"""Provider boundary. Real network providers are optional and never used by correctness tests."""

from __future__ import annotations

from typing import Protocol

from ..agent.context import AgentContext
from ..agent.decisions import AgentDecision


class ProviderUnavailable(RuntimeError):
    """The semantic provider cannot safely produce the next AgentDecision."""


class AgentProvider(Protocol):
    model_version: str
    prompt_version: str

    async def generate(self, context: AgentContext) -> AgentDecision: ...
