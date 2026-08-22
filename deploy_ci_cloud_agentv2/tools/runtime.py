"""Structural READ validation, bounded retries, and concurrent partial-failure handling."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from ..agent.decisions import ReadToolBatch, ToolCall
from ..agent.evidence import (
    ObservationDisposition,
    ToolObservation,
    TransportStatus,
)
from ..agent.identity import RequestIdentity
from ..agent.provenance import build_provenance
from ..agent.results import ResultStatus, normalize_read_result
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
    def __init__(self, registry: ToolRegistry, owner: RequestIdentity):
        self.registry = registry
        self.owner = owner

    def validate_single(self, call: ToolCall) -> ToolCall:
        """Validate one proposed READ using the same guard as batch calls."""
        normalized = self.registry.normalize_call(call, require_read=True)
        if not _arguments_are_concrete(normalized.arguments):
            raise ValueError("tool arguments must be concrete before execution")
        return normalized

    def validate_batch(self, batch: ReadToolBatch, max_batch: int) -> ReadToolBatch:
        if not batch.calls:
            raise ValueError("READ batch must contain at least one call")
        if len(batch.calls) > max_batch:
            raise ValueError(f"READ batch exceeds configured maximum {max_batch}")
        normalized_calls = []
        for call in batch.calls:
            normalized = self.registry.normalize_call(call, require_read=True)
            spec = self.registry.spec(normalized.tool_name)
            if spec.kind is not ToolKind.READ or not spec.parallel_safe:
                raise ValueError(f"tool is not parallel-safe READ: {normalized.tool_name}")
            if not _arguments_are_concrete(normalized.arguments):
                raise ValueError("all READ batch arguments must be concrete")
            normalized_calls.append(normalized)
        call_ids = {call.call_id for call in batch.calls}
        if len(call_ids) != len(batch.calls):
            raise ValueError("READ batch call_id values must be unique")
        return ReadToolBatch(
            calls=tuple(normalized_calls),
            proposed_goal_descriptor=batch.proposed_goal_descriptor,
        )

    async def execute_single(
        self,
        call: ToolCall,
        *,
        max_retries: int,
        on_started=None,
    ) -> ToolObservation:
        if max_retries < 0:
            raise ValueError("max_retries is a non-negative per-call retry allowance")
        call = self.validate_single(call)
        spec = self.registry.spec(call.tool_name)
        retries = 0
        while True:
            if on_started is not None:
                maybe = on_started(call, retries)
                if inspect.isawaitable(maybe):
                    await maybe
            try:
                data = await self.registry.call(call)
                return _observation(
                    call,
                    spec.name,
                    owner=self.owner,
                    transport_status=TransportStatus.SUCCESS,
                    data=data,
                    retry_count=retries,
                )
            except ReadFailure as exc:
                if exc.retryable and retries < max_retries:
                    retries += 1
                    continue
                return _observation(
                    call,
                    spec.name,
                    owner=self.owner,
                    transport_status=(
                        TransportStatus.TIMEOUT
                        if exc.error_code == "READ_TIMEOUT"
                        else TransportStatus.ERROR
                    ),
                    data=None,
                    error_code=exc.error_code,
                    retryable=exc.retryable,
                    retry_count=retries,
                )
            except TimeoutError as exc:
                if retries < max_retries:
                    retries += 1
                    continue
                return _observation(
                    call,
                    spec.name,
                    owner=self.owner,
                    transport_status=TransportStatus.TIMEOUT,
                    data=None,
                    error_code="READ_TIMEOUT",
                    retryable=True,
                    retry_count=retries,
                )
            except Exception as exc:  # external read errors become data, not authority
                return _observation(
                    call,
                    spec.name,
                    owner=self.owner,
                    transport_status=TransportStatus.ERROR,
                    data=None,
                    error_code="READ_ERROR",
                    retryable=False,
                    retry_count=retries,
                )

    async def execute_batch(
        self,
        batch: ReadToolBatch,
        *,
        max_retries: int,
        max_batch: int | None = None,
        on_started=None,
    ) -> ReadBatchObservation:
        if max_retries < 0:
            raise ValueError("max_retries is a non-negative per-call retry allowance")
        batch = self.validate_batch(
            batch,
            max_batch=max_batch if max_batch is not None else len(batch.calls),
        )
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
    if call.tool_name in {"get_task_detail", "diagnose_task"}:
        return str(arguments.get("task_name", ""))
    if call.tool_name == "get_queue_state":
        return str(arguments["task_name"]) if arguments.get("task_name") is not None else "platform"
    if call.tool_name == "search_knowledge":
        return str(arguments.get("query", ""))
    return "platform"


def _observation(
    call: ToolCall,
    source: str,
    *,
    owner: RequestIdentity,
    transport_status: TransportStatus,
    data: object | None,
    error_code: str | None = None,
    retryable: bool = False,
    retry_count: int = 0,
) -> ToolObservation:
    result = normalize_read_result(source, call.arguments, data)
    if transport_status is not TransportStatus.SUCCESS:
        result = None
    provenance = build_provenance(source, call.arguments, result)
    if transport_status is not TransportStatus.SUCCESS:
        disposition = ObservationDisposition.TRANSPORT_FAILURE
    elif result is None:
        disposition = ObservationDisposition.MALFORMED
    elif result.envelope.status in {
        ResultStatus.NOT_FOUND,
        ResultStatus.NO_DATA,
        ResultStatus.UNAVAILABLE,
        ResultStatus.EMPTY,
    }:
        disposition = ObservationDisposition.ABSENT
    elif result.envelope.status is ResultStatus.ERROR:
        disposition = ObservationDisposition.EXTERNAL_ERROR
    elif result.validation_errors or not result.is_valid:
        disposition = ObservationDisposition.MALFORMED
    elif result.qualifies_for_evidence():
        disposition = ObservationDisposition.NORMALIZED
    else:
        disposition = ObservationDisposition.NORMALIZED_NO_QUALIFIED_EVIDENCE
    validation_error = "RESULT_VALIDATION_FAILED" if result is not None and result.validation_errors else error_code
    return ToolObservation(
        observation_id=f"obs_{uuid.uuid4().hex}",
        call_id=call.call_id,
        source=source,
        target=_target_for(call),
        owner=owner,
        transport_status=transport_status,
        disposition=disposition,
        data=data,
        error_code=validation_error,
        retryable=retryable,
        retry_count=retry_count,
        observed_at=datetime.now(timezone.utc),
        provenance=provenance,
        result=result,
    )
