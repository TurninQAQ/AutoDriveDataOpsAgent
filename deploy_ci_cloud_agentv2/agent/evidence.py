"""Runtime-owned observations, entity provenance, and qualified evidence.

An observation is a report from a READ boundary. It is not evidence merely
because the transport returned successfully. This module is the deterministic
boundary that decides whether a report is correctly bound to its queried
identity and meaningful for a completion requirement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Mapping

from .outcomes import GoalOutcome, GoalStatus


class IdentityStatus(str, Enum):
    MATCHED = "MATCHED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class ObservationProvenance:
    """Identity facts used to bind an external observation."""

    requested_target: str
    observed_target: str | None
    source_tool: str
    arguments_fingerprint: str
    identity_status: IdentityStatus


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    call_id: str
    source: str
    # Display/compatibility field: this is the requested identity, not an
    # authority-bearing evidence target. Evidence uses provenance below.
    target: str
    status: str
    data: object | None
    trust: str = "UNTRUSTED_EXTERNAL_DATA"
    error_code: str | None = None
    retryable: bool = False
    retry_count: int = 0
    observed_at: datetime = datetime.min.replace(tzinfo=timezone.utc)
    provenance: ObservationProvenance | None = None

    @property
    def requested_target(self) -> str:
        return self.provenance.requested_target if self.provenance else self.target

    @property
    def observed_target(self) -> str | None:
        return self.provenance.observed_target if self.provenance else None


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    kind: str
    # Authoritative only after qualification. For an entity-bound record this
    # is the observed identity, never merely the request argument.
    target: str
    observation_id: str
    provenance: ObservationProvenance
    observed_at: datetime
    entity_version: str | None = None
    valid_until: datetime | None = None
    status: str = "VALID"
    invalidated_by: str | None = None

    @property
    def source_tool(self) -> str:
        return self.provenance.source_tool

    @property
    def requested_target(self) -> str:
        return self.provenance.requested_target

    @property
    def observed_target(self) -> str | None:
        return self.provenance.observed_target

    def is_current(self, now: datetime | None = None) -> bool:
        if self.status != "VALID":
            return False
        check_time = now or datetime.now(timezone.utc)
        return self.valid_until is None or self.valid_until > check_time


@dataclass(frozen=True)
class EvidenceState:
    records: tuple[EvidenceRecord, ...] = ()

    def current(self, now: datetime | None = None) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.is_current(now))


def build_observation_provenance(
    source_tool: str, arguments: Mapping[str, Any], data: object | None
) -> ObservationProvenance:
    """Derive identity provenance from request and response deterministically."""

    requested = _requested_identity(source_tool, arguments)
    observed = _observed_identity(source_tool, data)
    if source_tool == "get_gpu_pool" or (
        source_tool == "get_queue_state" and not requested
    ):
        status = IdentityStatus.NOT_APPLICABLE
    elif observed is None:
        status = IdentityStatus.MISSING
    elif requested == observed:
        status = IdentityStatus.MATCHED
    else:
        status = IdentityStatus.CONFLICT
    encoded = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"), default=str)
    return ObservationProvenance(
        requested_target=requested,
        observed_target=observed,
        source_tool=source_tool,
        arguments_fingerprint=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        identity_status=status,
    )


class EvidenceTracker:
    """Runtime-owned evidence writer; the Agent cannot manufacture records."""

    def __init__(self, freshness_seconds: int = 300):
        self.freshness_seconds = freshness_seconds

    def record_observations(
        self, state: EvidenceState, observations: Iterable[ToolObservation]
    ) -> tuple[EvidenceState, tuple[EvidenceRecord, ...]]:
        created: list[EvidenceRecord] = []
        records = list(state.records)
        for observation in observations:
            record = self._qualify(observation)
            if record is None:
                continue
            records.append(record)
            created.append(record)
        return EvidenceState(tuple(records)), tuple(created)

    def refresh_goal_outcomes(
        self,
        descriptor,
        contract,
        evidence: EvidenceState,
        existing: dict[str, GoalOutcome],
    ) -> dict[str, GoalOutcome]:
        current = evidence.current()
        result: dict[str, GoalOutcome] = {}
        for goal in descriptor.goals:
            prior = existing.get(goal.goal_id, GoalOutcome(goal.goal_id))
            if prior.status in {
                GoalStatus.DENIED,
                GoalStatus.REJECTED,
                GoalStatus.FAILED,
                GoalStatus.INCONCLUSIVE,
                GoalStatus.BLOCKED,
            }:
                result[goal.goal_id] = prior
                continue
            requirements = contract.requirements_by_goal[goal.goal_id]
            refs = tuple(
                record.evidence_id
                for requirement in requirements
                if requirement.kind.value != "TARGET_BINDING"
                for record in current
                if record.kind == requirement.kind.value
                and _target_matches(requirement.target, record.target, requirement.kind.value)
            )
            complete = all(
                requirement.kind.value == "TARGET_BINDING"
                or any(
                    record.kind == requirement.kind.value
                    and _target_matches(
                        requirement.target, record.target, requirement.kind.value
                    )
                    for record in current
                )
                for requirement in requirements
            )
            result[goal.goal_id] = GoalOutcome(
                goal_id=goal.goal_id,
                status=GoalStatus.SATISFIED if complete else GoalStatus.PENDING,
                reason_code=None if complete else "REQUIRED_EVIDENCE_MISSING",
                evidence_refs=tuple(dict.fromkeys(refs)),
            )
        return result

    def _qualify(self, observation: ToolObservation) -> EvidenceRecord | None:
        if observation.status != "SUCCESS" or observation.data is None:
            return None
        provenance = observation.provenance
        if provenance is None:
            # No explicit provenance means no target-bound evidence. Fail
            # closed for all tools so a future tool cannot bypass this gate.
            return None
        kind = self._kind_for(observation.source)
        if kind is None or not _identity_is_valid(observation.source, provenance):
            return None
        if not _meaningful_for(kind, observation.data):
            return None
        observed_at = observation.observed_at
        target = _evidence_target(provenance)
        return EvidenceRecord(
            evidence_id=f"ev_{observation.observation_id}",
            kind=kind,
            target=target,
            observation_id=observation.observation_id,
            provenance=provenance,
            observed_at=observed_at,
            entity_version=self._entity_version(observation.data),
            valid_until=observed_at + timedelta(seconds=self.freshness_seconds),
        )

    @staticmethod
    def _kind_for(source: str) -> str | None:
        return {
            "get_task_detail": "LIVE_TASK",
            "get_gpu_pool": "GPU_POOL",
            "search_knowledge": "KNOWLEDGE",
            "get_queue_state": "QUEUE_STATE",
            "diagnose_task": "DIAGNOSTIC_CONTEXT",
        }.get(source)

    @staticmethod
    def _entity_version(data: object) -> str | None:
        if isinstance(data, Mapping):
            for key in ("entity_version", "generation", "revision", "etag", "version"):
                if data.get(key) is not None:
                    return str(data[key])
        return None


def _requested_identity(source_tool: str, arguments: Mapping[str, Any]) -> str:
    if source_tool in {"get_task_detail", "get_queue_state", "diagnose_task"}:
        return str(arguments.get("task_name", ""))
    if source_tool == "search_knowledge":
        return str(arguments.get("query", ""))
    return "platform"


def _observed_identity(source_tool: str, data: object | None) -> str | None:
    if not isinstance(data, Mapping):
        return None
    if source_tool in {"get_task_detail", "get_queue_state", "diagnose_task"}:
        value = data.get("task_name")
    elif source_tool == "search_knowledge":
        value = data.get("query")
    else:
        return None
    return str(value) if isinstance(value, str) and value.strip() else None


def _identity_is_valid(source: str, provenance: ObservationProvenance) -> bool:
    if source == "get_gpu_pool":
        return provenance.identity_status is IdentityStatus.NOT_APPLICABLE
    if source == "get_queue_state" and not provenance.requested_target:
        return provenance.identity_status is IdentityStatus.NOT_APPLICABLE
    return provenance.identity_status is IdentityStatus.MATCHED


def _evidence_target(provenance: ObservationProvenance) -> str:
    if provenance.identity_status is IdentityStatus.NOT_APPLICABLE:
        return "platform"
    # This function is reached only after MATCHED qualification.
    return str(provenance.observed_target)


_PLACEHOLDER_TEXT = {
    "",
    "unknown",
    "no data",
    "no_data",
    "not found",
    "not_found",
    "unavailable",
    "no diagnostic facts",
    "no_diagnostic_facts",
    "placeholder",
}


def _meaningful_value(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _PLACEHOLDER_TEXT
    if isinstance(value, Mapping):
        return bool(value) and any(_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return bool(value) and any(_meaningful_value(item) for item in value)
    return True


def _not_data_payload(data: Mapping[str, Any]) -> bool:
    marker = data.get("status")
    return isinstance(marker, str) and marker.strip().lower() in {
        "no_data",
        "not_found",
        "unavailable",
        "empty_result",
    }


def _meaningful_for(kind: str, data: object) -> bool:
    if not isinstance(data, Mapping) or _not_data_payload(data):
        return False
    if kind == "LIVE_TASK":
        if data.get("exists") is False:
            return False
        return isinstance(data.get("state"), str) and _meaningful_value(data.get("state"))
    if kind == "KNOWLEDGE":
        results = data.get("results")
        if not isinstance(results, list) or not results:
            return False
        for item in results:
            if isinstance(item, str) and _meaningful_value(item):
                return True
            if isinstance(item, Mapping) and any(
                key in item and _meaningful_value(item[key])
                for key in ("content", "text", "title", "answer", "snippet", "description")
            ):
                return True
        return False
    if kind == "DIAGNOSTIC_CONTEXT":
        return any(
            _meaningful_value(data.get(key))
            for key in ("diagnosis", "root_cause", "reason", "facts", "details", "summary")
        )
    if kind == "GPU_POOL":
        return isinstance(data.get("devices"), list) or isinstance(
            data.get("reservations"), list
        )
    if kind == "QUEUE_STATE":
        for key in ("state", "status", "position", "queue", "active", "pending", "running"):
            if key in data and (
                isinstance(data[key], list) or _meaningful_value(data[key])
            ):
                return True
        return False
    return False


def _target_matches(required: str, actual: str, kind: str) -> bool:
    if kind in {"GPU_POOL", "QUEUE_STATE"} and required == "platform":
        return actual == "platform"
    return required == actual
