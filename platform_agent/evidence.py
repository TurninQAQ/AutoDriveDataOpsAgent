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


def evidence_types_for_tool(tool_name: str) -> tuple[EvidenceType, ...]:
    """Return metadata for a tool without making a routing decision."""

    return TOOL_EVIDENCE_TYPES.get(str(tool_name), ())


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
        for evidence_type in evidence_types_for_tool(observation.tool_name):
            record = EvidenceRecord(
                type=evidence_type,
                source_tool=observation.tool_name,
                timestamp=time.time(),
                summary=_observation_summary(observation),
            )
            self.records.append(record)
            created.append(record)
        if observation.tool_name in {"diagnose_task", "get_stage_logs"} and _has_non_empty_key(
            observation.data,
            {"diagnosis", "root_cause", "rootcause", "reason", "failure_reason", "cause"},
        ):
            record = EvidenceRecord(
                type=EvidenceType.DIAGNOSIS,
                source_tool=observation.tool_name,
                timestamp=time.time(),
                summary=f"{observation.tool_name} returned explicit diagnosis evidence.",
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
    "evidence_types_for_tool",
]
