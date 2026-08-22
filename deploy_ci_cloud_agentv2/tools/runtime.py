"""Structural READ validation, bounded retries, and concurrent partial-failure handling."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from ..agent.decisions import AcceptedToolCall, ReadToolBatch, ToolCall
from ..agent.evidence import (
    ObservationDisposition,
    ToolObservation,
    TransportStatus,
)
from ..agent.identity import RequestIdentity
from ..agent.immutable import canonical_snapshot
from ..agent.provenance import build_provenance, canonical_tool_call_fingerprint
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

    def validate_single(self, call: ToolCall) -> AcceptedToolCall:
        """Validate one proposed READ using the same guard as batch calls."""
        normalized = self.registry.normalize_call(call, require_read=True)
        if not _arguments_are_concrete(normalized.arguments):
            raise ValueError("tool arguments must be concrete before execution")
        return AcceptedToolCall(
            call_id=normalized.call_id,
            tool_name=normalized.tool_name,
            arguments=normalized.arguments,
        )

    def validate_batch(self, batch: ReadToolBatch, max_batch: int) -> ReadToolBatch:
        if not batch.calls:
            raise ValueError("READ batch must contain at least one call")
        if len(batch.calls) > max_batch:
            raise ValueError(f"READ batch exceeds configured maximum {max_batch}")
        normalized_calls: list[AcceptedToolCall] = []
        for call in batch.calls:
            normalized = self.registry.normalize_call(call, require_read=True)
            spec = self.registry.spec(normalized.tool_name)
            if spec.kind is not ToolKind.READ or not spec.parallel_safe:
                raise ValueError(f"tool is not parallel-safe READ: {normalized.tool_name}")
            if not _arguments_are_concrete(normalized.arguments):
                raise ValueError("all READ batch arguments must be concrete")
            normalized_calls.append(
                AcceptedToolCall(
                    call_id=normalized.call_id,
                    tool_name=normalized.tool_name,
                    arguments=normalized.arguments,
                )
            )
        call_ids = {call.call_id for call in normalized_calls}
        if len(call_ids) != len(batch.calls):
            raise ValueError("READ batch call_id values must be unique")
        return ReadToolBatch(
            calls=tuple(normalized_calls),
            proposed_goal_descriptor=batch.proposed_goal_descriptor,
        )

    async def execute_single(
        self,
        call: ToolCall | AcceptedToolCall,
        *,
        max_retries: int,
        on_started=None,
    ) -> ToolObservation:
        if max_retries < 0:
            raise ValueError("max_retries is a non-negative per-call retry allowance")
        accepted_call = call if isinstance(call, AcceptedToolCall) else self.validate_single(call)
        call = accepted_call
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
        if not all(isinstance(call, AcceptedToolCall) for call in batch.calls):
            batch = self.validate_batch(
                batch,
                max_batch=max_batch if max_batch is not None else len(batch.calls),
            )
        elif max_batch is not None and len(batch.calls) > max_batch:
            raise ValueError(f"READ batch exceeds configured maximum {max_batch}")
        results = await asyncio.gather(
            *(
                self.execute_single(call, max_retries=max_retries, on_started=on_started)
                for call in batch.calls
            )
        )
        return ReadBatchObservation(tuple(results))


def _arguments_are_concrete(value: object) -> bool:
    if isinstance(value, dict) or hasattr(value, "items") and not isinstance(value, (str, bytes)):
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
    # Snapshot exactly once at the Runtime ingress.  Normalization, evidence,
    # audit payloads, and Agent projections all consume this same immutable
    # representation; no later facade mutation can change the observation.
    snapshot = canonical_snapshot(data)
    result = normalize_read_result(source, call.arguments, snapshot)
    if transport_status is not TransportStatus.SUCCESS:
        result = None
    provenance = build_provenance(source, call.arguments, result)
    disposition = classify_normalized_result(result, transport_status)
    validation_error = (
        "; ".join(result.validation_errors)
        if result is not None and result.validation_errors
        else error_code
    )
    return ToolObservation(
        observation_id=f"obs_{uuid.uuid4().hex}",
        call_id=call.call_id,
        source=source,
        target=_target_for(call),
        owner=owner,
        transport_status=transport_status,
        disposition=disposition,
        data=snapshot,
        error_code=validation_error,
        retryable=retryable,
        retry_count=retry_count,
        observed_at=datetime.now(timezone.utc),
        provenance=provenance,
        result=result,
    )


def classify_normalized_result(result, transport_status: TransportStatus) -> ObservationDisposition:
    """Map transport and validated-result state to an observation disposition.

    Validation errors are checked before semantic envelope status.  Therefore a
    malformed field in a nominal ``NO_DATA``/``ERROR`` response cannot be
    downgraded to an ordinary absence/error envelope.
    """
    if transport_status is not TransportStatus.SUCCESS:
        return ObservationDisposition.TRANSPORT_FAILURE
    if result is None or result.validation_errors or result.envelope.status is ResultStatus.MALFORMED:
        return ObservationDisposition.MALFORMED
    if result.envelope.status is ResultStatus.ERROR:
        return ObservationDisposition.EXTERNAL_ERROR
    if result.envelope.status in {
        ResultStatus.NOT_FOUND,
        ResultStatus.NO_DATA,
        ResultStatus.UNAVAILABLE,
        ResultStatus.EMPTY,
    }:
        return ObservationDisposition.ABSENT
    if result.qualifies_for_evidence():
        return ObservationDisposition.NORMALIZED
    return ObservationDisposition.NORMALIZED_NO_QUALIFIED_EVIDENCE
