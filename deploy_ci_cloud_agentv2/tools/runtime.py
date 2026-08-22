"""Structural READ validation, bounded retries, and concurrent partial-failure handling."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from ..agent.decisions import ReadToolBatch, ToolCall
from ..agent.evidence import ToolObservation
from ..agent.outcomes import TerminalCode
from .metadata import ToolKind
from .registry import ToolRegistry


class ReadFailure(Exception):
    def __init__(self, error_code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True)
class ReadBatchObservation:
    results: tuple[ToolObservation, ...]


class ReadToolRuntime:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def validate_batch(self, batch: ReadToolBatch, max_batch: int) -> None:
        if len(batch.calls) > max_batch:
            raise ValueError(f"READ batch exceeds configured maximum {max_batch}")
        for call in batch.calls:
            spec = self.registry.validate_call(call, require_read=True)
            if spec.kind is not ToolKind.READ or not spec.parallel_safe:
                raise ValueError(f"tool is not parallel-safe READ: {call.tool_name}")
            if not _arguments_are_concrete(call.arguments):
                raise ValueError("all READ batch arguments must be concrete")
        call_ids = {call.call_id for call in batch.calls}
        if len(call_ids) != len(batch.calls):
            raise ValueError("READ batch call_id values must be unique")

    async def execute_single(
        self,
        call: ToolCall,
        *,
        max_retries: int,
        on_started=None,
    ) -> ToolObservation:
        spec = self.registry.validate_call(call, require_read=True)
        if not _arguments_are_concrete(call.arguments):
            raise ValueError("tool arguments must be concrete before execution")
        retries = 0
        while True:
            if on_started is not None:
                maybe = on_started(call, retries)
                if inspect.isawaitable(maybe):
                    await maybe
            try:
                data = await self.registry.call(call)
                return ToolObservation(
                    observation_id=f"obs_{uuid.uuid4().hex}",
                    call_id=call.call_id,
                    source=spec.name,
                    target=_target_for(call),
                    status="SUCCESS",
                    data=data,
                    retry_count=retries,
                    observed_at=datetime.now(timezone.utc),
                )
            except ReadFailure as exc:
                if exc.retryable and retries < max_retries:
                    retries += 1
                    continue
                return ToolObservation(
                    observation_id=f"obs_{uuid.uuid4().hex}",
                    call_id=call.call_id,
                    source=spec.name,
                    target=_target_for(call),
                    status="READ_FAILURE",
                    data=None,
                    error_code=exc.error_code,
                    retryable=exc.retryable,
                    retry_count=retries,
                    observed_at=datetime.now(timezone.utc),
                )
            except TimeoutError as exc:
                if retries < max_retries:
                    retries += 1
                    continue
                return ToolObservation(
                    observation_id=f"obs_{uuid.uuid4().hex}",
                    call_id=call.call_id,
                    source=spec.name,
                    target=_target_for(call),
                    status="READ_FAILURE",
                    data=None,
                    error_code="READ_TIMEOUT",
                    retryable=True,
                    retry_count=retries,
                    observed_at=datetime.now(timezone.utc),
                )
            except Exception as exc:  # external read errors become data, not authority
                return ToolObservation(
                    observation_id=f"obs_{uuid.uuid4().hex}",
                    call_id=call.call_id,
                    source=spec.name,
                    target=_target_for(call),
                    status="READ_FAILURE",
                    data=None,
                    error_code="READ_ERROR",
                    retryable=False,
                    retry_count=retries,
                    observed_at=datetime.now(timezone.utc),
                )

    async def execute_batch(
        self,
        batch: ReadToolBatch,
        *,
        max_retries: int,
        max_batch: int | None = None,
        on_started=None,
    ) -> ReadBatchObservation:
        self.validate_batch(batch, max_batch=max_batch or len(batch.calls))
        results = await asyncio.gather(
            *(
                self.execute_single(call, max_retries=max_retries, on_started=on_started)
                for call in batch.calls
            )
        )
        return ReadBatchObservation(tuple(results))


def _arguments_are_concrete(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) == {"$ref"} or set(value) == {"from_call"}:
            return False
        return all(_arguments_are_concrete(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_arguments_are_concrete(item) for item in value)
    if isinstance(value, str) and (value.startswith("$call.") or value.startswith("${")):
        return False
    return True


def _target_for(call: ToolCall) -> str:
    arguments = call.arguments
    if call.tool_name in {"get_task_detail", "get_queue_state", "diagnose_task"}:
        return str(arguments.get("task_name", "")) or "platform"
    if call.tool_name == "search_knowledge":
        return str(arguments.get("query", ""))
    return "platform"
