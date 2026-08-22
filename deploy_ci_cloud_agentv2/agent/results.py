"""Typed deterministic response contracts for the five Phase B READ tools.

This module is the boundary between an external mapping and Runtime-owned
meaning.  A normalized result may preserve small external fields for semantic
context, but evidence qualification consumes only these result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .provenance import ObservationScope, ScopeKind


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    NO_DATA = "NO_DATA"
    UNAVAILABLE = "UNAVAILABLE"
    EMPTY = "EMPTY"
    ERROR = "ERROR"
    MALFORMED = "MALFORMED"


@dataclass(frozen=True)
class ResultEnvelope:
    status: ResultStatus
    error_code: str | None = None
    message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status is ResultStatus.SUCCESS


@dataclass(frozen=True)
class NormalizedReadResult:
    envelope: ResultEnvelope
    observed_scope: ObservationScope
    validation_errors: tuple[str, ...] = ()
    entity_version: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.envelope.is_success and not self.validation_errors

    def qualifies_for_evidence(self) -> bool:
        return False


class TaskState(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"
    UNKNOWN_EXTERNAL_STATE = "UNKNOWN_EXTERNAL_STATE"


@dataclass(frozen=True)
class TaskDetailResult(NormalizedReadResult):
    task_name: str | None = None
    exists: bool | None = None
    state: TaskState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def qualifies_for_evidence(self) -> bool:
        return (
            self.is_valid
            and self.observed_scope.kind is ScopeKind.TASK
            and self.task_name is not None
            and self.exists is True
            and self.state is not None
            and self.state is not TaskState.UNKNOWN_EXTERNAL_STATE
        )


@dataclass(frozen=True)
class GpuRecord:
    identifier: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class GpuPoolResult(NormalizedReadResult):
    devices: tuple[GpuRecord, ...] = ()
    reservations: tuple[GpuRecord, ...] = ()

    def qualifies_for_evidence(self) -> bool:
        return self.is_valid and self.observed_scope.kind is ScopeKind.PLATFORM


@dataclass(frozen=True)
class KnowledgeHit:
    title: str | None
    content: str | None
    source: str | None
    url: str | None

    @property
    def has_explanatory_content(self) -> bool:
        return isinstance(self.content, str) and bool(self.content.strip())


@dataclass(frozen=True)
class KnowledgeResult(NormalizedReadResult):
    query: str | None = None
    hits: tuple[KnowledgeHit, ...] = ()

    def qualifies_for_evidence(self) -> bool:
        return (
            self.is_valid
            and self.observed_scope.kind is ScopeKind.QUERY
            and self.query is not None
            and any(hit.has_explanatory_content for hit in self.hits)
        )


@dataclass(frozen=True)
class QueueEntry:
    task_name: str | None
    position: int | None
    state: str | None


@dataclass(frozen=True)
class QueueResult(NormalizedReadResult):
    task_name: str | None = None
    position: int | None = None
    state: str | None = None
    entries: tuple[QueueEntry, ...] = ()

    def qualifies_for_evidence(self) -> bool:
        return self.is_valid and self.observed_scope.kind in {
            ScopeKind.PLATFORM,
            ScopeKind.TASK,
        }


@dataclass(frozen=True)
class DiagnosticFinding:
    message: str
    code: str | None = None
    category: str | None = None
    supporting_fact_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticResult(NormalizedReadResult):
    task_name: str | None = None
    findings: tuple[DiagnosticFinding, ...] = ()

    def qualifies_for_evidence(self) -> bool:
        return (
            self.is_valid
            and self.observed_scope.kind is ScopeKind.TASK
            and self.task_name is not None
            and bool(self.findings)
        )


_ERROR_MARKERS = {"ERROR", "FAILURE", "FAILED", "UNAVAILABLE"}
_NO_DATA_MARKERS = {"NO_DATA", "NOT_FOUND", "EMPTY_RESULT", "EMPTY"}
_TASK_STATE_ALIASES = {
    "SUCCESS": "SUCCEEDED",
    "SUCCEEDED": "SUCCEEDED",
    "DONE": "COMPLETED",
    "COMPLETED": "COMPLETED",
    "CANCELED": "CANCELLED",
    "CANCELLED": "CANCELLED",
    "UNKNOWN": "UNKNOWN_EXTERNAL_STATE",
    "UNKNOWN_EXTERNAL_STATE": "UNKNOWN_EXTERNAL_STATE",
}
_TASK_STATES = {item.value for item in TaskState}
_EMPTY_DIAGNOSTIC_TEXT = {"no diagnostic facts", "no diagnostic finding", "none"}


def normalize_read_result(
    tool_name: str, arguments: Mapping[str, Any], raw: object
) -> NormalizedReadResult:
    """Validate and normalize one known tool response without semantic inference."""

    envelope = _envelope(raw)
    payload = raw if isinstance(raw, Mapping) else {}
    if tool_name == "get_task_detail":
        return _task_detail(envelope, payload)
    if tool_name == "get_gpu_pool":
        return _gpu_pool(envelope, payload)
    if tool_name == "search_knowledge":
        return _knowledge(envelope, payload)
    if tool_name == "get_queue_state":
        return _queue(envelope, payload)
    if tool_name == "diagnose_task":
        return _diagnosis(envelope, payload)
    return NormalizedReadResult(
        ResultEnvelope(ResultStatus.MALFORMED, error_code="UNKNOWN_TOOL_RESULT"),
        ObservationScope(ScopeKind.UNKNOWN),
        ("unknown tool result contract",),
    )


def _envelope(raw: object) -> ResultEnvelope:
    if not isinstance(raw, Mapping):
        return ResultEnvelope(ResultStatus.MALFORMED, error_code="RESPONSE_NOT_OBJECT")
    error_code = _text(raw.get("error_code"))
    error_value = raw.get("error")
    if raw.get("success") is False or error_code or error_value:
        return ResultEnvelope(
            ResultStatus.ERROR,
            error_code=error_code or "EXTERNAL_ERROR",
            message=_text(raw.get("message")) or _text(error_value),
        )
    marker = _text(raw.get("status"))
    if marker:
        marker = marker.upper()
        if marker in _ERROR_MARKERS:
            return ResultEnvelope(ResultStatus.ERROR, error_code=marker)
        if marker in _NO_DATA_MARKERS:
            status = ResultStatus.NOT_FOUND if marker == "NOT_FOUND" else (
                ResultStatus.EMPTY if marker in {"EMPTY", "EMPTY_RESULT"} else ResultStatus.NO_DATA
            )
            return ResultEnvelope(status)
        if marker == "OK" or marker == "SUCCESS":
            return ResultEnvelope(ResultStatus.SUCCESS)
        return ResultEnvelope(ResultStatus.MALFORMED, error_code="UNKNOWN_RESULT_STATUS")
    if raw.get("not_found") is True:
        return ResultEnvelope(ResultStatus.NOT_FOUND)
    if raw.get("available") is False:
        return ResultEnvelope(ResultStatus.UNAVAILABLE)
    return ResultEnvelope(ResultStatus.SUCCESS)


def _task_detail(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> TaskDetailResult:
    errors: list[str] = []
    task_name = _text(raw.get("task_name"))
    state: TaskState | None = None
    exists = raw.get("exists")
    if envelope.is_success:
        if task_name is None:
            errors.append("task_name is required")
        if "exists" in raw and not isinstance(exists, bool):
            errors.append("exists must be boolean")
        if "state" not in raw:
            errors.append("state is required")
        else:
            state = _task_state(raw.get("state"), errors)
        if exists is None and task_name is not None and state is not None:
            exists = True
        if exists is False:
            envelope = ResultEnvelope(ResultStatus.NOT_FOUND)
    return TaskDetailResult(
        envelope=envelope,
        observed_scope=(
            ObservationScope(ScopeKind.TASK, task_name)
            if task_name is not None
            else ObservationScope(ScopeKind.UNKNOWN)
        ),
        validation_errors=tuple(errors),
        entity_version=_text(raw.get("entity_version")),
        task_name=task_name,
        exists=exists if isinstance(exists, bool) else None,
        state=state,
        metadata={
            key: value
            for key, value in raw.items()
            if key not in {"task_name", "exists", "state", "status", "success", "error_code", "error"}
        },
    )


def _task_state(value: object, errors: list[str]) -> TaskState | None:
    marker = _text(value)
    if marker is None:
        errors.append("state must be a non-empty string")
        return None
    normalized = _TASK_STATE_ALIASES.get(marker.upper(), marker.upper())
    if normalized not in _TASK_STATES:
        errors.append(f"unknown task state: {marker}")
        return None
    return TaskState(normalized)


def _gpu_pool(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> GpuPoolResult:
    errors: list[str] = []
    devices = _gpu_records(raw.get("devices"), "devices", errors)
    reservations = _gpu_records(raw.get("reservations"), "reservations", errors)
    if envelope.is_success and "devices" not in raw and "reservations" not in raw:
        errors.append("devices or reservations is required")
    scope, scope_errors = _observed_scope(raw, default=ScopeKind.PLATFORM)
    errors.extend(scope_errors)
    return GpuPoolResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=_text(raw.get("entity_version")),
        devices=devices,
        reservations=reservations,
    )


def _gpu_records(value: object, field: str, errors: list[str]) -> tuple[GpuRecord, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return ()
    records: list[GpuRecord] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"{field}[{index}] must be an object")
            continue
        identifier = _text(item.get("gpu_id")) or _text(item.get("device_id")) or _text(
            item.get("reservation_id")
        )
        if identifier is None:
            errors.append(f"{field}[{index}] requires an identifier")
            continue
        records.append(GpuRecord(identifier, dict(item)))
    return tuple(records)


def _knowledge(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> KnowledgeResult:
    errors: list[str] = []
    query = _text(raw.get("query"))
    if envelope.is_success and query is None:
        errors.append("query is required")
    raw_results = raw.get("results")
    hits: list[KnowledgeHit] = []
    if raw_results is None:
        raw_results = []
    if not isinstance(raw_results, list):
        errors.append("results must be a list")
    else:
        for index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                errors.append(f"results[{index}] must be an object")
                continue
            hits.append(
                KnowledgeHit(
                    title=_text(item.get("title")),
                    content=(
                        _text(item.get("content"))
                        or _text(item.get("body"))
                        or _text(item.get("text"))
                        or _text(item.get("snippet"))
                        or _text(item.get("summary"))
                    ),
                    source=_text(item.get("source")),
                    url=_text(item.get("url")),
                )
            )
    if query is None:
        scope = ObservationScope(ScopeKind.UNKNOWN)
    else:
        scope = ObservationScope(ScopeKind.QUERY, query)
    return KnowledgeResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=_text(raw.get("entity_version")),
        query=query,
        hits=tuple(hits),
    )


def _queue(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> QueueResult:
    errors: list[str] = []
    task_name = _text(raw.get("task_name"))
    position = raw.get("position")
    if position is not None and (not isinstance(position, int) or isinstance(position, bool) or position < 0):
        errors.append("position must be a non-negative integer")
        position = None
    state = _text(raw.get("state")) or _text(raw.get("queue_state"))
    entries: list[QueueEntry] = []
    if "queue" in raw:
        queue = raw["queue"]
        if not isinstance(queue, list):
            errors.append("queue must be a list")
        else:
            for index, item in enumerate(queue):
                if not isinstance(item, Mapping):
                    errors.append(f"queue[{index}] must be an object")
                    continue
                entry_position = item.get("position")
                if entry_position is not None and (
                    not isinstance(entry_position, int)
                    or isinstance(entry_position, bool)
                    or entry_position < 0
                ):
                    errors.append(f"queue[{index}].position must be non-negative integer")
                    continue
                entries.append(
                    QueueEntry(_text(item.get("task_name")), entry_position, _text(item.get("state")))
                )
    if envelope.is_success and not any(
        key in raw for key in ("queue", "position", "state", "queue_state", "active", "pending", "running")
    ):
        errors.append("queue result has no recognized state fields")
    scope, scope_errors = _observed_scope(raw, default=None)
    errors.extend(scope_errors)
    if scope.kind is ScopeKind.TASK and scope.identity is None:
        errors.append("task queue scope requires task_name")
    return QueueResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=_text(raw.get("entity_version")),
        task_name=task_name,
        position=position,
        state=state,
        entries=tuple(entries),
    )


def _observed_scope(
    raw: Mapping[str, Any], default: ScopeKind | None
) -> tuple[ObservationScope, list[str]]:
    errors: list[str] = []
    task_name = _text(raw.get("task_name"))
    declared = _text(raw.get("scope"))
    declared_kind: ScopeKind | None = None
    if declared is not None:
        declared_kind = {
            "PLATFORM": ScopeKind.PLATFORM,
            "GLOBAL": ScopeKind.PLATFORM,
            "TASK": ScopeKind.TASK,
        }.get(declared.upper())
        if declared_kind is None:
            errors.append(f"unknown scope: {declared}")
    if task_name is not None:
        observed = ObservationScope(ScopeKind.TASK, task_name)
        if declared_kind is ScopeKind.PLATFORM:
            errors.append("scope declares PLATFORM but task_name is present")
        return observed, errors
    if declared_kind is ScopeKind.TASK:
        return ObservationScope(ScopeKind.UNKNOWN), errors + ["TASK scope missing task_name"]
    if declared_kind is ScopeKind.PLATFORM:
        return ObservationScope(ScopeKind.PLATFORM), errors
    if default is not None:
        return ObservationScope(default), errors
    if "queue" in raw or any(key in raw for key in ("active", "pending", "running")):
        return ObservationScope(ScopeKind.PLATFORM), errors
    # A bare position is task-like but has no identity; fail closed rather than
    # silently promoting it to platform scope.
    return ObservationScope(ScopeKind.UNKNOWN), errors


def _diagnosis(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> DiagnosticResult:
    errors: list[str] = []
    task_name = _text(raw.get("task_name"))
    if envelope.is_success and task_name is None:
        errors.append("task_name is required")
    findings: list[DiagnosticFinding] = []
    for key in ("root_cause", "reason", "diagnosis", "anomaly", "error_condition", "finding"):
        finding = _finding_from_value(raw.get(key))
        if finding is not None:
            findings.append(finding)
    for key in ("findings", "diagnostic_findings", "facts"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for index, item in enumerate(value):
                finding = _finding_from_value(item)
                if finding is None:
                    errors.append(f"{key}[{index}] is not a diagnostic finding")
                else:
                    findings.append(finding)
        elif isinstance(value, Mapping):
            finding = _finding_from_value(value)
            if finding is not None:
                findings.append(finding)
            elif key in {"findings", "diagnostic_findings"}:
                errors.append(f"{key} is not a diagnostic finding")
        else:
            errors.append(f"{key} must be an object or list")
    scope = (
        ObservationScope(ScopeKind.TASK, task_name)
        if task_name is not None
        else ObservationScope(ScopeKind.UNKNOWN)
    )
    return DiagnosticResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=_text(raw.get("entity_version")),
        task_name=task_name,
        findings=tuple(findings),
    )


def _finding_from_value(value: object) -> DiagnosticFinding | None:
    if isinstance(value, str):
        message = value.strip()
        if not message or message.lower() in _EMPTY_DIAGNOSTIC_TEXT:
            return None
        return DiagnosticFinding(message=message)
    if not isinstance(value, Mapping):
        return None
    message = (
        _text(value.get("message"))
        or _text(value.get("description"))
        or _text(value.get("finding"))
        or _text(value.get("root_cause"))
        or _text(value.get("reason"))
    )
    if message is None or message.lower() in _EMPTY_DIAGNOSTIC_TEXT:
        return None
    refs = value.get("supporting_fact_refs", ())
    if not isinstance(refs, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in refs):
        refs = ()
    return DiagnosticFinding(
        message=message,
        code=_text(value.get("code")),
        category=_text(value.get("category")),
        supporting_fact_refs=tuple(item.strip() for item in refs),
    )


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def normalize_tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and enforce semantic constraints before execution/fingerprints.

    This is intentionally an explicit Phase B READ boundary, not a generic
    schema inference engine.  Whitespace-only identifiers are invalid and
    ``None`` is the sole representation of a global queue request.
    """

    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be a mapping")
    args = dict(arguments)
    if tool_name in {"get_task_detail", "diagnose_task"}:
        args["task_name"] = _required_argument(args, "task_name")
    elif tool_name == "search_knowledge":
        args["query"] = _required_argument(args, "query")
        top_k = args.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0 or top_k > 50:
            raise ValueError("top_k must be an integer between 1 and 50")
        args["top_k"] = top_k
    elif tool_name == "get_queue_state":
        task_name = args.get("task_name")
        if task_name is None:
            args["task_name"] = None
        else:
            args["task_name"] = _required_argument(args, "task_name")
    elif tool_name == "get_gpu_pool":
        if args:
            raise ValueError("get_gpu_pool does not accept arguments")
    return args


def _required_argument(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    maximum = 2_048 if name == "query" else 256
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds the maximum supported length")
    return normalized
