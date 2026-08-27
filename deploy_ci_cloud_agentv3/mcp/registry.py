from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Type
from pydantic import BaseModel

ToolKind = Literal["READ", "PREPARE", "PROPOSAL", "WRITE", "RUNTIME_INTERNAL"]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    kind: ToolKind
    args_model: Type[BaseModel]
    handler: Callable[..., Any]

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        parsed = self.args_model.model_validate(arguments or {})
        result = self.handler(**parsed.model_dump())
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json")
        return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def list(self, allowed: set[str] | None = None) -> list[ToolDefinition]:
        values = self._tools.values()
        if allowed is not None:
            values = (tool for tool in values if tool.name in allowed)
        return sorted(values, key=lambda item: item.name)
