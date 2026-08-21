"""Small, provider-neutral evidence coverage state for the adaptive loop.

Evidence tracking deliberately describes what has been observed; it does not
select tools or turn a missing evidence type into a forced tool call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .models import EvidenceRecord, EvidenceType, ToolObservation


TOOL_EVIDENCE_TYPES: dict[str, tuple[EvidenceType, ...]] = {
    "get_platform_health": (EvidenceType.PLATFORM_HEALTH,),
    "list_tasks": (EvidenceType.LIVE_TASK,),
    "get_task_detail": (EvidenceType.LIVE_TASK,),
    "get_queue_state": (EvidenceType.LIVE_QUEUE,),
    "get_gpu_pool": (EvidenceType.LIVE_GPU,),
    "inspect_task_containers": (EvidenceType.LIVE_CONTAINER,),
    "get_stage_logs": (EvidenceType.LIVE_LOG,),
    "diagnose_task": (EvidenceType.LIVE_TASK,),
    "search_knowledge": (EvidenceType.STATIC_KNOWLEDGE,),
}

TASK_SCOPED_TOOLS = frozenset(
    {
        "get_task_detail",
        "diagnose_task",
        "get_stage_logs",
        "inspect_task_containers",
        "get_queue_state",
    }
)

TARGET_AWARE_EVIDENCE_TYPES = frozenset(
    {
        EvidenceType.LIVE_TASK,
        EvidenceType.LIVE_LOG,
        EvidenceType.LIVE_CONTAINER,
        EvidenceType.LIVE_QUEUE,
        EvidenceType.DIAGNOSTIC_CONTEXT,
        EvidenceType.DIAGNOSIS,
        EvidenceType.RECOVERY_STATE,
    }
)

# These are fields from DiagnosisService.inspect_task().  A successful call
# with only task_name (or an arbitrary status/message field) is not enough to
# claim that diagnostic facts were collected.
DIAGNOSTIC_FACT_KEYS = frozenset(
    {
        "queue",
        "airflow",
        "containers",
        "gpu_reservations",
        "gpu_devices",
        "errors",
        "evidence_complete",
        "datasets",
    }
)


def evidence_types_for_tool(tool_name: str) -> tuple[EvidenceType, ...]:
    """Return metadata for a tool without making a routing decision."""

    return TOOL_EVIDENCE_TYPES.get(str(tool_name), ())


def _subject_value(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def evidence_subject_from_observation(observation: ToolObservation) -> dict[str, str | None]:
    """Extract exact entity provenance from tool arguments/result metadata.

    This helper intentionally does not infer identity from free-form text.  Tool
    arguments are preferred because they are the request's structured subject;
    result fields are only a fallback for production tool payloads that echo it.
    """

    data = observation.data if isinstance(observation.data, dict) else {}
    task_name = _subject_value(observation.arguments.get("task_name"))
    if task_name is None:
        task_name = _subject_value(data.get("task_name"))
    dataset_name = _subject_value(observation.arguments.get("dataset_name"))
    if dataset_name is None:
        dataset_name = _subject_value(data.get("dataset_name"))
    return {"task_name": task_name, "dataset_name": dataset_name}


def _observation_summary(observation: ToolObservation) -> str:
    """Create a bounded, non-result-dump audit summary."""

    if not observation.ok:
        return f"{observation.tool_name} failed; no evidence was collected."
    data = observation.data
    if isinstance(data, dict):
        if observation.tool_name == "search_knowledge":
            results = data.get("results")
            count = len(results) if isinstance(results, list) else 0
            return f"search_knowledge returned {count} static result(s)."
        return f"{observation.tool_name} returned a successful structured observation."
    if isinstance(data, list):
        return f"{observation.tool_name} returned a successful list ({len(data)} item(s))."
    return f"{observation.tool_name} returned a successful observation."


def _has_non_empty_key(data: Any, keys: set[str]) -> bool:
    if not isinstance(data, dict):
        return False
    return any(
        value not in (None, "", [], {}, ())
        for key, value in data.items()
        if str(key).lower() in keys
    )


def is_diagnostic_context_payload(data: Any) -> bool:
    """Return whether *data* contains a real diagnosis facts field.

    DiagnosisService is deliberately a deterministic facts plane.  Presence of
    at least one field from its structured contract is sufficient, including
    partial values and ``evidence_complete=False``.  A task name alone,
    arbitrary metadata, or a success message is not diagnostic context.
    """

    if not isinstance(data, dict):
        return False
    return any(str(key).lower() in DIAGNOSTIC_FACT_KEYS for key in data)


@dataclass
class EvidenceTracker:
    """Accumulate abstract evidence coverage without routing policy."""

    records: list[EvidenceRecord] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: Iterable[EvidenceRecord | dict[str, Any]] = ()) -> "EvidenceTracker":
        normalized: list[EvidenceRecord] = []
        for item in records:
            if isinstance(item, EvidenceRecord):
                normalized.append(item)
                continue
            if not isinstance(item, dict):
                continue
            try:
                normalized.append(
                    EvidenceRecord(
                        type=EvidenceType(str(item["type"])),
                        source_tool=str(item["source_tool"]),
                        timestamp=float(item.get("timestamp", time.time())),
                        summary=str(item.get("summary", ""))[:500],
                        task_name=_subject_value(item.get("task_name")),
                        dataset_name=_subject_value(item.get("dataset_name")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return cls(normalized)

    @classmethod
    def from_observations(cls, observations: Iterable[ToolObservation]) -> "EvidenceTracker":
        tracker = cls()
        for observation in observations:
            tracker.record_tool_observation(observation)
        return tracker

    def record_tool_observation(self, observation: ToolObservation) -> list[EvidenceRecord]:
        """Record successful evidence-producing observations only."""

        if not observation.ok:
            return []
        created: list[EvidenceRecord] = []
        subject = evidence_subject_from_observation(observation)
        task_name = subject["task_name"] if observation.tool_name in TASK_SCOPED_TOOLS else None
        dataset_name = subject["dataset_name"] if task_name else None
        for evidence_type in evidence_types_for_tool(observation.tool_name):
            record = EvidenceRecord(
                type=evidence_type,
                source_tool=observation.tool_name,
                timestamp=time.time(),
                summary=_observation_summary(observation),
                task_name=task_name,
                dataset_name=dataset_name,
            )
            self.records.append(record)
            created.append(record)
        # A successful production diagnose_task is a deterministic facts bundle,
        # not a root-cause conclusion.  It still provides diagnostic context for
        # the Agent synthesis layer.  Stage logs provide the same context only
        # when a non-empty log payload is actually present.
        diagnostic_context = (
            observation.tool_name == "diagnose_task"
            and isinstance(observation.data, dict)
            and task_name is not None
            and is_diagnostic_context_payload(observation.data)
        )
        if observation.tool_name == "get_stage_logs":
            logs = observation.data.get("logs") if isinstance(observation.data, dict) else None
            diagnostic_context = logs not in (None, "", [], {}, ())
        if diagnostic_context:
            complete = observation.data.get("evidence_complete") if isinstance(observation.data, dict) else None
            detail = "complete" if complete is True else "partial" if complete is False else "available"
            record = EvidenceRecord(
                type=EvidenceType.DIAGNOSTIC_CONTEXT,
                source_tool=observation.tool_name,
                timestamp=time.time(),
                summary=f"{observation.tool_name} returned target-bound diagnostic context ({detail}).",
                task_name=task_name,
                dataset_name=dataset_name,
            )
            self.records.append(record)
            created.append(record)

        # Keep DIAGNOSIS records for V1.5.x artifact/test compatibility.  New
        # production contracts use DIAGNOSTIC_CONTEXT and final root-cause
        # validation happens after synthesis.
        if observation.tool_name in {"diagnose_task", "get_stage_logs"} and _has_non_empty_key(
            observation.data,
            {"diagnosis", "root_cause", "rootcause", "reason", "failure_reason", "cause"},
        ):
            record = EvidenceRecord(
                type=EvidenceType.DIAGNOSIS,
                source_tool=observation.tool_name,
                timestamp=time.time(),
                summary=f"{observation.tool_name} returned explicit diagnosis evidence.",
                task_name=task_name,
                dataset_name=dataset_name,
            )
            self.records.append(record)
            created.append(record)
        if observation.tool_name in {
            "diagnose_task",
            "get_task_detail",
            "get_queue_state",
            "get_stage_logs",
            "inspect_task_containers",
        } and _has_non_empty_key(
            observation.data,
            {
                "recovery",
                "recovery_state",
                "recovery_runs",
                "checkpoint",
                "checkpoint_state",
                "resume",
                "resumed",
                "recovered",
            },
        ):
            record = EvidenceRecord(
                type=EvidenceType.RECOVERY_STATE,
                source_tool=observation.tool_name,
                timestamp=time.time(),
                summary=f"{observation.tool_name} returned recovery/checkpoint evidence.",
                task_name=task_name,
                dataset_name=dataset_name,
            )
            self.records.append(record)
            created.append(record)
        return created

    def get_collected_types(self) -> list[EvidenceType]:
        """Return distinct types in first-seen order for stable prompts/traces."""

        result: list[EvidenceType] = []
        seen: set[EvidenceType] = set()
        for record in self.records:
            if record.type not in seen:
                seen.add(record.type)
                result.append(record.type)
        return result

    def has(self, evidence_type: EvidenceType | str) -> bool:
        try:
            candidate = evidence_type if isinstance(evidence_type, EvidenceType) else EvidenceType(str(evidence_type))
        except ValueError:
            return False
        return candidate in set(self.get_collected_types())

    def summary(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records]

    def coverage(self) -> list[str]:
        return [item.value for item in self.get_collected_types()]


__all__ = [
    "EvidenceRecord",
    "EvidenceTracker",
    "EvidenceType",
    "TOOL_EVIDENCE_TYPES",
    "TASK_SCOPED_TOOLS",
    "TARGET_AWARE_EVIDENCE_TYPES",
    "DIAGNOSTIC_FACT_KEYS",
    "evidence_subject_from_observation",
    "is_diagnostic_context_payload",
    "evidence_types_for_tool",
]
