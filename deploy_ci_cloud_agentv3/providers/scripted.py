from __future__ import annotations

from collections import deque
from typing import Iterable
from deploy_ci_cloud_agentv3.providers.base import AssistantMessage


class ScriptedProvider:
    def __init__(self, responses: Iterable[AssistantMessage | dict]) -> None:
        self.responses = deque(
            item if isinstance(item, AssistantMessage) else AssistantMessage.model_validate(item)
            for item in responses
        )

    async def invoke(self, messages, tools) -> AssistantMessage:
        if not self.responses:
            return AssistantMessage(content="No scripted response remains.")
        return self.responses.popleft()
