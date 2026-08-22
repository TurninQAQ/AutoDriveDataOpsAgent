"""Strict deterministic response contracts for the five Phase B READ tools.

External mappings cross this boundary exactly once.  The parsers below use the
shared field-contract primitives so a known malformed field can never be
silently treated as absent.  EvidenceTracker consumes these typed results; it
does not inspect the raw payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from .field_contract import (
    collect_invalid,
    read_optional_bool,
    read_optional_enum,
    read_optional_int,
    read_optional_mapping,
    read_optional_sequence,
    read_optional_string,
)
from .immutable import FrozenMapping
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
        return self.envelope.status is ResultStatus.SUCCESS and not self.validation_errors

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
    metadata: Mapping[str, Any] = field(default_factory=FrozenMapping)

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", FrozenMapping(self.attributes))


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


class QueueState(str, Enum):
    QUEUED = "QUEUED"
    PENDING = "PENDING"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN_EXTERNAL_STATE = "UNKNOWN_EXTERNAL_STATE"


@dataclass(frozen=True)
class QueueEntry:
    task_name: str | None
    position: int | None
    state: QueueState | None

    @property
    def is_meaningful(self) -> bool:
        """A queue entry needs an independently useful queue fact.

        Identity alone is not treated as a queue status.  A position or a
        known (non-UNKNOWN) state is the minimum evidence; an UNKNOWN state
        may accompany those facts without erasing their usefulness.
        """

        return self.position is not None or (
            self.state is not None and self.state is not QueueState.UNKNOWN_EXTERNAL_STATE
        )


@dataclass(frozen=True)
class QueueResult(NormalizedReadResult):
    task_name: str | None = None
    position: int | None = None
    state: QueueState | None = None
    entries: tuple[QueueEntry, ...] = ()
    meaningful: bool = False

    def qualifies_for_evidence(self) -> bool:
        return (
            self.is_valid
            and self.meaningful
            and self.observed_scope.kind in {ScopeKind.PLATFORM, ScopeKind.TASK}
        )


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
_TASK_STATE_VALUES = {item.value: item for item in TaskState}
_TASK_STATE_VALUES.update({name: TaskState(value) for name, value in _TASK_STATE_ALIASES.items()})
_QUEUE_STATES = {item.value: item for item in QueueState}
_EMPTY_DIAGNOSTIC_TEXT = {"no diagnostic facts", "no diagnostic finding", "none"}
_SCOPE_VALUES = {
    "PLATFORM": ScopeKind.PLATFORM,
    "GLOBAL": ScopeKind.PLATFORM,
    "TASK": ScopeKind.TASK,
}
_KNOWLEDGE_SCOPE_VALUES = {"QUERY": ScopeKind.QUERY, "KNOWLEDGE": ScopeKind.QUERY}
_KNOWN_ENVELOPE_FIELDS = {
    "success", "status", "available", "not_found", "error", "error_code", "message", "scope",
}
_COMMON_KNOWN_FIELDS = _KNOWN_ENVELOPE_FIELDS | {"entity_version"}


def normalize_read_result(
    tool_name: str, arguments: Mapping[str, Any], raw: object
) -> NormalizedReadResult:
    """Validate and normalize one known tool response."""

    payload = raw if isinstance(raw, Mapping) else {}
    envelope, envelope_errors = _envelope(raw)
    if tool_name == "get_task_detail":
        result = _task_detail(envelope, payload)
    elif tool_name == "get_gpu_pool":
        result = _gpu_pool(envelope, payload)
    elif tool_name == "search_knowledge":
        result = _knowledge(envelope, payload)
    elif tool_name == "get_queue_state":
        result = _queue(envelope, payload)
    elif tool_name == "diagnose_task":
        result = _diagnosis(envelope, payload)
    else:
        return NormalizedReadResult(
            ResultEnvelope(ResultStatus.MALFORMED, error_code="UNKNOWN_TOOL_RESULT"),
            ObservationScope(ScopeKind.UNKNOWN),
            ("unknown tool result contract",),
        )
    if envelope_errors:
        result = replace(
            result,
            envelope=ResultEnvelope(ResultStatus.MALFORMED, error_code="MALFORMED_ENVELOPE"),
            validation_errors=tuple(envelope_errors) + result.validation_errors,
        )
    return result


def _envelope(raw: object) -> tuple[ResultEnvelope, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return ResultEnvelope(ResultStatus.MALFORMED, error_code="RESPONSE_NOT_OBJECT"), (
            "response must be an object",
        )
    fields = {
        "success": read_optional_bool(raw, "success"),
        "status": read_optional_string(raw, "status"),
        "available": read_optional_bool(raw, "available"),
        "not_found": read_optional_bool(raw, "not_found"),
        "error": read_optional_string(raw, "error"),
        "error_code": read_optional_string(raw, "error_code"),
        "message": read_optional_string(raw, "message"),
    }
    errors = collect_invalid(*fields.values())
    if errors:
        return ResultEnvelope(ResultStatus.MALFORMED, error_code="MALFORMED_ENVELOPE"), tuple(errors)
    success = fields["success"].value if fields["success"].is_valid else None
    status_marker = fields["status"].value if fields["status"].is_valid else None
    error_code = fields["error_code"].value if fields["error_code"].is_valid else None
    error_message = fields["error"].value if fields["error"].is_valid else None
    message = fields["message"].value if fields["message"].is_valid else None
    if success is False or error_code is not None or error_message is not None:
        return ResultEnvelope(ResultStatus.ERROR, error_code=error_code or "EXTERNAL_ERROR", message=message or error_message), ()
    if status_marker is not None:
        marker = status_marker.upper()
        if marker in _ERROR_MARKERS:
            return ResultEnvelope(ResultStatus.ERROR, error_code=marker, message=message), ()
        if marker in _NO_DATA_MARKERS:
            status = ResultStatus.NOT_FOUND if marker == "NOT_FOUND" else (
                ResultStatus.EMPTY if marker in {"EMPTY", "EMPTY_RESULT"} else ResultStatus.NO_DATA
            )
            return ResultEnvelope(status, message=message), ()
        if marker not in {"OK", "SUCCESS"}:
            return ResultEnvelope(ResultStatus.MALFORMED, error_code="UNKNOWN_RESULT_STATUS"), (
                f"unknown result status: {status_marker}",
            )
    if fields["not_found"].is_valid and fields["not_found"].value is True:
        return ResultEnvelope(ResultStatus.NOT_FOUND, message=message), ()
    if fields["available"].is_valid and fields["available"].value is False:
        return ResultEnvelope(ResultStatus.UNAVAILABLE, message=message), ()
    return ResultEnvelope(ResultStatus.SUCCESS, message=message), ()


def _task_detail(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> TaskDetailResult:
    errors: list[str] = []
    task_field = read_optional_string(raw, "task_name")
    state_field = read_optional_enum(raw, "state", _TASK_STATE_VALUES)
    exists_field = read_optional_bool(raw, "exists")
    version_field = read_optional_string(raw, "entity_version")
    errors.extend(collect_invalid(task_field, state_field, exists_field, version_field))
    scope, scope_errors, _ = _observed_scope(raw, default=None, infer_platform=False)
    errors.extend(scope_errors)
    task_name = task_field.value if task_field.is_valid else None
    state = state_field.value if state_field.is_valid else None
    exists = exists_field.value if exists_field.is_valid else None
    if envelope.is_success:
        if task_field.is_absent:
            errors.append("task_name is required")
        if state_field.is_absent:
            errors.append("state is required")
        if exists is None and task_name is not None and state is not None:
            exists = True
        if exists is False and not errors:
            envelope = ResultEnvelope(ResultStatus.NOT_FOUND)
    return TaskDetailResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=version_field.value if version_field.is_valid else None,
        task_name=task_name,
        exists=exists,
        state=state,
        metadata=FrozenMapping({key: value for key, value in raw.items() if key not in _COMMON_KNOWN_FIELDS | {"task_name", "state", "exists"}}),
    )


def _gpu_pool(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> GpuPoolResult:
    errors: list[str] = []
    devices_field = read_optional_sequence(raw, "devices")
    reservations_field = read_optional_sequence(raw, "reservations")
    version_field = read_optional_string(raw, "entity_version")
    errors.extend(collect_invalid(devices_field, reservations_field, version_field))
    devices = _gpu_records(devices_field.value, "devices", errors) if devices_field.is_valid else ()
    reservations = _gpu_records(reservations_field.value, "reservations", errors) if reservations_field.is_valid else ()
    if envelope.is_success and devices_field.is_absent and reservations_field.is_absent:
        errors.append("devices or reservations is required")
    scope, scope_errors, _ = _observed_scope(raw, default=ScopeKind.PLATFORM, infer_platform=False)
    errors.extend(scope_errors)
    return GpuPoolResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=version_field.value if version_field.is_valid else None,
        devices=devices,
        reservations=reservations,
    )


def _gpu_records(value: object, field: str, errors: list[str]) -> tuple[GpuRecord, ...]:
    if not isinstance(value, (list, tuple)):
        errors.append(f"{field} must be a list")
        return ()
    records: list[GpuRecord] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"{field}[{index}] must be an object")
            continue
        names = ("gpu_id", "device_id", "reservation_id")
        fields = {name: read_optional_string(item, name) for name in names}
        errors.extend(
            f"{field}[{index}].{name}: {result.error or 'invalid field'}"
            for name, result in fields.items()
            if result.is_invalid
        )
        identifiers = [result.value for result in fields.values() if result.is_valid and result.value is not None]
        if not identifiers:
            errors.append(f"{field}[{index}] requires an identifier")
            continue
        records.append(GpuRecord(identifiers[0], dict(item)))
    return tuple(records)


def _knowledge(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> KnowledgeResult:
    errors: list[str] = []
    query_field = read_optional_string(raw, "query")
    results_field = read_optional_sequence(raw, "results")
    version_field = read_optional_string(raw, "entity_version")
    scope_field = read_optional_enum(raw, "scope", _KNOWLEDGE_SCOPE_VALUES)
    errors.extend(collect_invalid(query_field, results_field, version_field, scope_field))
    if envelope.is_success:
        if query_field.is_absent:
            errors.append("query is required")
        if results_field.is_absent:
            errors.append("results is required")
    hits: list[KnowledgeHit] = []
    if results_field.is_valid and results_field.value is not None:
        for index, item in enumerate(results_field.value):
            if not isinstance(item, Mapping):
                errors.append(f"results[{index}] must be an object")
                continue
            hit, hit_errors = _knowledge_hit(item, index)
            errors.extend(hit_errors)
            hits.append(hit)
    query = query_field.value if query_field.is_valid else None
    declared_scope = scope_field.value if scope_field.is_valid else None
    if query is not None and declared_scope not in (None, ScopeKind.QUERY):
        errors.append("knowledge response scope conflicts with QUERY")
    if query is None or (declared_scope is not None and declared_scope is not ScopeKind.QUERY):
        scope = ObservationScope(ScopeKind.UNKNOWN)
    else:
        scope = ObservationScope(ScopeKind.QUERY, query)
    return KnowledgeResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=version_field.value if version_field.is_valid else None,
        query=query,
        hits=tuple(hits),
    )


def _knowledge_hit(raw: Mapping[str, Any], index: int) -> tuple[KnowledgeHit, list[str]]:
    errors: list[str] = []
    fields = {
        name: read_optional_string(raw, name)
        for name in ("title", "source", "url", "content", "body", "text", "snippet", "summary")
    }
    errors.extend(
        f"results[{index}].{name}: {result.error or 'invalid field'}"
        for name, result in fields.items()
        if result.is_invalid
    )
    content = None
    for name in ("content", "body", "text", "snippet", "summary"):
        result = fields[name]
        if result.is_valid and result.value is not None:
            content = result.value
            break
    return KnowledgeHit(
        title=fields["title"].value if fields["title"].is_valid else None,
        content=content,
        source=fields["source"].value if fields["source"].is_valid else None,
        url=fields["url"].value if fields["url"].is_valid else None,
    ), errors


def _queue(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> QueueResult:
    errors: list[str] = []
    task_field = read_optional_string(raw, "task_name")
    position_field = read_optional_int(raw, "position", minimum=0)
    version_field = read_optional_string(raw, "entity_version")
    errors.extend(collect_invalid(task_field, position_field, version_field))
    state, state_errors = _read_queue_state_fields(raw)
    errors.extend(state_errors)
    count_values: dict[str, int] = {}
    for name in ("active", "waiting", "queued", "pending", "running"):
        field_result = read_optional_int(raw, name, minimum=0)
        errors.extend(collect_invalid(field_result))
        if field_result.is_valid and field_result.value is not None:
            count_values[name] = field_result.value
    entries: list[QueueEntry] = []
    queue_field = read_optional_sequence(raw, "queue")
    errors.extend(collect_invalid(queue_field))
    if queue_field.is_valid and queue_field.value is not None:
        for index, item in enumerate(queue_field.value):
            entry, entry_errors = _queue_entry(item, index)
            errors.extend(entry_errors)
            if entry is not None:
                entries.append(entry)
    scope, scope_errors, _ = _observed_scope(raw, default=None, infer_platform=True)
    errors.extend(scope_errors)
    # UNKNOWN_EXTERNAL_STATE is not a queue answer by itself.  It only becomes
    # useful when another independently validated queue fact is present.  An
    # explicit queue collection or numeric position/count is such a fact.
    meaningful_entries = tuple(entry for entry in entries if entry.is_meaningful)
    meaningful = (
        position_field.is_valid
        or bool(count_values)
        or bool(meaningful_entries)
        # An explicitly present empty queue is an independently meaningful
        # aggregate fact (the platform queue is known to be empty). A
        # non-empty collection of UNKNOWN-only entries is not.
        or (queue_field.is_valid and queue_field.value is not None and len(queue_field.value) == 0)
        or (state is not None and state is not QueueState.UNKNOWN_EXTERNAL_STATE)
    )
    if envelope.is_success and not meaningful:
        errors.append("queue result has no meaningful state value")
    task_name = task_field.value if task_field.is_valid else None
    return QueueResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=version_field.value if version_field.is_valid else None,
        task_name=task_name,
        position=position_field.value if position_field.is_valid else None,
        state=state,
        entries=tuple(entries),
        meaningful=meaningful,
    )


def _queue_entry(value: object, index: int) -> tuple[QueueEntry | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return None, [f"queue[{index}] must be an object"]
    task = read_optional_string(value, "task_name")
    position = read_optional_int(value, "position", minimum=0)
    state = read_optional_enum(value, "state", _QUEUE_STATES)
    errors.extend(collect_invalid(task, position, state))
    if not any(field.is_valid for field in (task, position, state)):
        errors.append(f"queue[{index}] has no meaningful fields")
    return QueueEntry(
        task.value if task.is_valid else None,
        position.value if position.is_valid else None,
        state.value if state.is_valid else None,
    ), errors


def _read_queue_state_fields(raw: Mapping[str, Any]) -> tuple[QueueState | None, list[str]]:
    state_field = read_optional_enum(raw, "state", _QUEUE_STATES)
    queue_state_field = read_optional_enum(raw, "queue_state", _QUEUE_STATES)
    local_errors = collect_invalid(state_field, queue_state_field)
    values = [field.value for field in (state_field, queue_state_field) if field.is_valid and field.value is not None]
    if len(values) == 2 and values[0] is not values[1]:
        local_errors.append("state and queue_state conflict")
    return (values[0] if values else None), local_errors


def _observed_scope(
    raw: Mapping[str, Any], *, default: ScopeKind | None, infer_platform: bool
) -> tuple[ObservationScope, list[str], bool]:
    errors: list[str] = []
    task_field = read_optional_string(raw, "task_name")
    scope_field = read_optional_enum(raw, "scope", _SCOPE_VALUES)
    errors.extend(collect_invalid(task_field, scope_field))
    if task_field.is_invalid or scope_field.is_invalid:
        return ObservationScope(ScopeKind.UNKNOWN), errors, False
    task_name = task_field.value if task_field.is_valid else None
    declared = scope_field.value if scope_field.is_valid else None
    if task_name is not None:
        if declared is ScopeKind.PLATFORM:
            errors.append("scope declares PLATFORM but task_name is present")
        return ObservationScope(ScopeKind.TASK, task_name), errors, True
    if declared is ScopeKind.TASK:
        errors.append("TASK scope missing task_name")
        return ObservationScope(ScopeKind.UNKNOWN), errors, True
    if declared is ScopeKind.PLATFORM:
        return ObservationScope(ScopeKind.PLATFORM), errors, True
    if default is not None:
        return ObservationScope(default), errors, True
    if infer_platform and ("queue" in raw or any(name in raw for name in ("active", "waiting", "queued", "pending", "running"))):
        return ObservationScope(ScopeKind.PLATFORM), errors, True
    return ObservationScope(ScopeKind.UNKNOWN), errors, True


def _diagnosis(envelope: ResultEnvelope, raw: Mapping[str, Any]) -> DiagnosticResult:
    errors: list[str] = []
    task_field = read_optional_string(raw, "task_name")
    version_field = read_optional_string(raw, "entity_version")
    errors.extend(collect_invalid(task_field, version_field))
    scope, scope_errors, _ = _observed_scope(raw, default=None, infer_platform=False)
    errors.extend(scope_errors)
    if envelope.is_success and task_field.is_absent:
        errors.append("task_name is required")
    findings: list[DiagnosticFinding] = []
    for key in ("root_cause", "reason", "diagnosis", "anomaly", "error_condition", "finding"):
        if key not in raw:
            continue
        finding, finding_errors = _diagnostic_value(raw[key], key)
        errors.extend(finding_errors)
        if finding is not None:
            findings.append(finding)
    for key in ("findings", "diagnostic_findings", "facts"):
        if key not in raw:
            continue
        sequence_field = read_optional_sequence(raw, key)
        if sequence_field.is_valid and sequence_field.value is not None:
            for index, item in enumerate(sequence_field.value):
                finding, finding_errors = _diagnostic_value(item, f"{key}[{index}]")
                errors.extend(finding_errors)
                if finding is not None:
                    findings.append(finding)
            continue
        mapping_field = read_optional_mapping(raw, key)
        if mapping_field.is_invalid:
            errors.append(mapping_field.error or f"{key} must be an object or list")
        elif mapping_field.is_valid and mapping_field.value is not None:
            finding, finding_errors = _diagnostic_mapping(mapping_field.value, key)
            errors.extend(finding_errors)
            if finding is not None:
                findings.append(finding)
    task_name = task_field.value if task_field.is_valid else None
    return DiagnosticResult(
        envelope=envelope,
        observed_scope=scope,
        validation_errors=tuple(errors),
        entity_version=version_field.value if version_field.is_valid else None,
        task_name=task_name,
        findings=tuple(findings),
    )


def _diagnostic_value(value: object, name: str) -> tuple[DiagnosticFinding | None, list[str]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, [f"{name} must be non-empty"]
        if text.lower() in _EMPTY_DIAGNOSTIC_TEXT:
            return None, []
        return DiagnosticFinding(message=text), []
    if isinstance(value, Mapping):
        return _diagnostic_mapping(value, name)
    return None, [f"{name} must be a string or object"]


def _diagnostic_mapping(value: Mapping[str, Any], name: str) -> tuple[DiagnosticFinding | None, list[str]]:
    errors: list[str] = []
    string_names = ("message", "description", "finding", "root_cause", "reason", "code", "category")
    fields = {field_name: read_optional_string(value, field_name) for field_name in string_names}
    errors.extend(
        f"{name}.{field_name}: {field.error or 'invalid field'}"
        for field_name, field in fields.items()
        if field.is_invalid
    )
    refs = read_optional_sequence(value, "supporting_fact_refs")
    if refs.is_invalid:
        errors.append(refs.error or f"{name}.supporting_fact_refs is invalid")
    refs_value: tuple[str, ...] = ()
    if refs.is_valid and refs.value is not None:
        converted: list[str] = []
        for index, item in enumerate(refs.value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{name}.supporting_fact_refs[{index}] must be a non-empty string")
            else:
                converted.append(item.strip())
        refs_value = tuple(converted)
    if errors:
        return None, errors
    message = next(
        (fields[field_name].value for field_name in ("message", "description", "finding", "root_cause", "reason") if fields[field_name].is_valid and fields[field_name].value is not None),
        None,
    )
    if message is None:
        return None, []
    if message.lower() in _EMPTY_DIAGNOSTIC_TEXT:
        return None, []
    return DiagnosticFinding(
        message=message,
        code=fields["code"].value if fields["code"].is_valid else None,
        category=fields["category"].value if fields["category"].is_valid else None,
        supporting_fact_refs=refs_value,
    ), []


def normalize_tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and enforce semantic constraints before execution/fingerprints."""

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
