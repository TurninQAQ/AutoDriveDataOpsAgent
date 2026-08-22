"""Explicit V2-local tool catalog; handlers are injected by the host."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..agent.decisions import ToolCall
from ..agent.events import catalog_fingerprint
from ..agent.results import normalize_tool_arguments
from .metadata import ToolKind, ToolSpec


Handler = Callable[..., Any]


class ToolCatalogIntegrityError(RuntimeError):
    """The effective sealed catalog differs from audited Runtime metadata."""


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}
        self._sealed = False
        self._sealed_hash: str | None = None

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if self._sealed:
            raise RuntimeError("ToolRegistry is sealed")
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def seal(self) -> None:
        if self._sealed:
            return
        self._sealed_hash = self._compute_catalog_hash()
        self._sealed = True

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def spec(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(f"unknown tool: {name}") from exc

    def handler(self, name: str) -> Handler:
        return self._handlers[name]

    def catalog(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def catalog_hash(self) -> str:
        if not self._sealed or self._sealed_hash is None:
            raise RuntimeError("ToolRegistry must be sealed before catalog hashing")
        return self._sealed_hash

    def _compute_catalog_hash(self) -> str:
        return catalog_fingerprint(
            [
                {
                    "name": spec.name,
                    "kind": spec.kind.value,
                    "risk": spec.risk.value,
                    "schema": spec.schema,
                    "parallel_safe": spec.parallel_safe,
                    "requires_precondition": spec.requires_precondition,
                    "verification": spec.verification,
                    "idempotency": spec.idempotency.value,
                }
                for spec in self.catalog()
            ]
        )

    def validate_call(self, call: ToolCall, *, require_read: bool = True) -> ToolSpec:
        if not self._sealed:
            raise RuntimeError("ToolRegistry must be sealed before validation")
        spec = self.spec(call.tool_name)
        if require_read and spec.kind is not ToolKind.READ:
            raise ValueError(f"{call.tool_name} is not a READ tool")
        _validate_arguments(spec.schema, call.arguments)
        normalize_tool_arguments(call.tool_name, call.arguments)
        return spec

    def normalize_call(self, call: ToolCall, *, require_read: bool = True) -> ToolCall:
        spec = self.validate_call(call, require_read=require_read)
        return ToolCall(
            call_id=call.call_id,
            tool_name=call.tool_name,
            arguments=normalize_tool_arguments(spec.name, call.arguments),
        )

    async def call(self, call: ToolCall) -> Any:
        if not self._sealed:
            raise RuntimeError("ToolRegistry must be sealed before execution")
        handler = self.handler(call.tool_name)
        result = handler(**call.arguments)
        if inspect.isawaitable(result):
            return await result
        return result


def _validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be a mapping")
    required = tuple(schema.get("required", ()))
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"missing required tool arguments: {', '.join(missing)}")
    properties = schema.get("properties", {})
    unknown = [name for name in arguments if name not in properties]
    if unknown:
        raise ValueError(f"unknown tool arguments: {', '.join(unknown)}")
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type")
        if value is None and properties.get(name, {}).get("nullable"):
            continue
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{name} must be an integer")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        if expected == "array" and not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array")
