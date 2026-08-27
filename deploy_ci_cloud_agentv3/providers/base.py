from __future__ import annotations

from typing import Any, Protocol
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelProvider(Protocol):
    async def invoke(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantMessage: ...
