"""Reproducible fake Agent provider used by unit and integration tests."""

from __future__ import annotations

from collections.abc import Iterable

from ..agent.context import AgentContext
from ..agent.decisions import AgentDecision
from .model import ProviderUnavailable


class ScriptedProvider:
    model_version = "fake-scripted-v2"
    prompt_version = "phase-b-test-prompt-v1"

    def __init__(self, decisions: Iterable[AgentDecision], *, repeat_last: bool = False):
        self.decisions = list(decisions)
        self.repeat_last = repeat_last
        self.contexts: list[AgentContext] = []
        self.calls = 0

    async def generate(self, context: AgentContext) -> AgentDecision:
        self.contexts.append(context)
        index = self.calls
        self.calls += 1
        if index < len(self.decisions):
            return self.decisions[index]
        if self.repeat_last and self.decisions:
            return self.decisions[-1]
        raise ProviderUnavailable("scripted provider has no further decision")
