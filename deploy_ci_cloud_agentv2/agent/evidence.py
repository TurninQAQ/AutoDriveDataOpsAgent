"""Runtime-owned observations, qualified evidence, and bounded projections.

Raw tool payloads remain untrusted external data.  Only normalized result
contracts can cross into an :class:`EvidenceRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping

from .contracts import CompletionContract, RequirementKind
from .goals import GoalDescriptor
from .outcomes import GoalOutcome, GoalStatus
from .provenance import (
    IdentityStatus,
    ObservationProvenance,
    ObservationScope,
    ScopeKind,
    ScopeStatus,
    build_provenance,
)
from .results import (
    DiagnosticResult,
    GpuPoolResult,
    KnowledgeResult,
    NormalizedReadResult,
    QueueResult,
    TaskDetailResult,
    normalize_read_result,
)


class EvidenceKind(str, Enum):
    TARGET_BINDING = "TARGET_BINDING"
    LIVE_TASK = "LIVE_TASK"
    GPU_POOL = "GPU_POOL"
    QUEUE_STATE = "QUEUE_STATE"
    KNOWLEDGE = "KNOWLEDGE"
    DIAGNOSTIC_CONTEXT = "DIAGNOSTIC_CONTEXT"


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    call_id: str
    source: str
    target: str
    status: str
    data: object | None
    trust: str = "UNTRUSTED_EXTERNAL_DATA"
    error_code: str | None = None
    retryable: bool = False
    retry_count: int = 0
    observed_at: datetime = datetime.min.replace(tzinfo=timezone.utc)
    provenance: ObservationProvenance | None = None
    result: NormalizedReadResult | None = None

    @property
    def requested_target(self) -> str:
        return self.provenance.requested_target if self.provenance else self.target

    @property
    def observed_target(self) -> str | None:
        return self.provenance.observed_target if self.provenance else None


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    kind: EvidenceKind
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

    @property
    def requested_scope(self) -> ObservationScope:
        return self.provenance.requested_scope

    @property
    def observed_scope(self) -> ObservationScope:
        return self.provenance.observed_scope

    def is_current(self, now: datetime | None = None) -> bool:
        if self.status != "VALID":
            return False
        if self.valid_until is None:
            return True
        return (now or datetime.now(timezone.utc)) <= self.valid_until


@dataclass(frozen=True)
class EvidenceState:
    records: tuple[EvidenceRecord, ...] = ()

    def current(self, now: datetime | None = None) -> tuple[EvidenceRecord, ...]:
        moment = now or datetime.now(timezone.utc)
        return tuple(record for record in self.records if record.is_current(moment))


@dataclass(frozen=True)
class EvidenceProjectionRecord:
    """Small, deterministic metadata only; never contains raw payload."""

    evidence_id: str
    kind: EvidenceKind
    target: str
    observation_id: str
    source_tool: str
    observed_at: datetime
    status: str
    valid_until: datetime | None
    entity_version: str | None
    requested_scope: ObservationScope
    observed_scope: ObservationScope
    requested_identity: str | None
    observed_identity: str | None
    identity_status: IdentityStatus
    scope_status: ScopeStatus


@dataclass(frozen=True)
class EvidenceProjection:
    records: tuple[EvidenceProjectionRecord, ...]
    total_records: int
    omitted_records: int
    estimated_chars: int


class ContextBudgetExceeded(RuntimeError):
    """The critical structured projection cannot fit the explicit budget."""


def build_observation_provenance(
    source_tool: str, arguments: Mapping[str, Any], data: object
) -> ObservationProvenance:
    """Compatibility helper for callers constructing test observations.

    Production Runtime calls normalization before this helper.  It is kept
    narrow and deterministic for V2-local tests and host integrations.
    """

    result = normalize_read_result(source_tool, arguments, data)
    return build_provenance(source_tool, arguments, result)


class EvidenceTracker:
    """The only component allowed to create evidence records from reads."""

    _tool_kinds = {
        "get_task_detail": (TaskDetailResult, EvidenceKind.LIVE_TASK),
        "get_gpu_pool": (GpuPoolResult, EvidenceKind.GPU_POOL),
        "search_knowledge": (KnowledgeResult, EvidenceKind.KNOWLEDGE),
        "get_queue_state": (QueueResult, EvidenceKind.QUEUE_STATE),
        "diagnose_task": (DiagnosticResult, EvidenceKind.DIAGNOSTIC_CONTEXT),
    }

    def __init__(self, *, freshness_seconds: int | None = None) -> None:
        if freshness_seconds is not None and freshness_seconds < 0:
            raise ValueError("freshness_seconds must not be negative")
        self.freshness_seconds = freshness_seconds

    def record_observations(
        self, state: EvidenceState, observations: tuple[ToolObservation, ...] | list[ToolObservation]
    ) -> tuple[EvidenceState, tuple[EvidenceRecord, ...]]:
        records = list(state.records)
        created: list[EvidenceRecord] = []
        for observation in observations:
            record = self._qualify(observation)
            if record is not None:
                if self.freshness_seconds is not None:
                    record = EvidenceRecord(
                        **{
                            **record.__dict__,
                            "valid_until": record.observed_at
                            + timedelta(seconds=self.freshness_seconds),
                        }
                    )
                records.append(record)
                created.append(record)
        return EvidenceState(tuple(records)), tuple(created)

    def refresh_goal_outcomes(
        self,
        descriptor: GoalDescriptor,
        contract: CompletionContract,
        evidence: EvidenceState,
        outcomes: Mapping[str, GoalOutcome],
    ) -> dict[str, GoalOutcome]:
        current = evidence.current()
        refreshed: dict[str, GoalOutcome] = {}
        for goal in descriptor.goals:
            previous = outcomes.get(goal.goal_id)
            if previous and previous.status in {
                GoalStatus.DENIED,
                GoalStatus.REJECTED,
                GoalStatus.FAILED,
                GoalStatus.INCONCLUSIVE,
                GoalStatus.BLOCKED,
            }:
                refreshed[goal.goal_id] = previous
                continue
            requirements = contract.requirements_by_goal[goal.goal_id]
            satisfied = all(
                requirement.kind is RequirementKind.TARGET_BINDING
                or self._has_requirement(current, requirement.kind, requirement.target)
                for requirement in requirements
            )
            refreshed[goal.goal_id] = GoalOutcome(
                goal_id=goal.goal_id,
                status=GoalStatus.SATISFIED if satisfied else GoalStatus.PENDING,
                reason_code="QUALIFIED_EVIDENCE_SATISFIED" if satisfied else "REQUIRED_EVIDENCE_MISSING",
            )
        return refreshed

    def _qualify(self, observation: ToolObservation) -> EvidenceRecord | None:
        if observation.status != "SUCCESS" or observation.result is None:
            return None
        result = observation.result
        expected = self._tool_kinds.get(observation.source)
        if expected is None or not isinstance(result, expected[0]):
            return None
        provenance = observation.provenance
        if provenance is None or not result.is_valid or not result.qualifies_for_evidence():
            return None
        kind = expected[1]
        if provenance.scope_status is not ScopeStatus.MATCHED:
            return None
        if kind is EvidenceKind.GPU_POOL:
            if provenance.identity_status is not IdentityStatus.NOT_APPLICABLE:
                return None
            target = "platform"
        elif kind is EvidenceKind.QUEUE_STATE and provenance.requested_scope.kind is ScopeKind.PLATFORM:
            if provenance.identity_status is not IdentityStatus.NOT_APPLICABLE:
                return None
            target = "platform"
        else:
            if provenance.identity_status is not IdentityStatus.MATCHED:
                return None
            target = provenance.observed_identity
            if target is None:
                return None
        return EvidenceRecord(
            evidence_id=f"ev_{observation.observation_id}",
            kind=kind,
            target=target,
            observation_id=observation.observation_id,
            provenance=provenance,
            observed_at=observation.observed_at,
            entity_version=result.entity_version,
        )

    @staticmethod
    def _has_requirement(records: tuple[EvidenceRecord, ...], kind: RequirementKind, target: str) -> bool:
        try:
            evidence_kind = EvidenceKind(kind.value)
        except ValueError:
            return False
        return any(record.kind is evidence_kind and record.target == target for record in records)


class EvidenceProjectionBuilder:
    """Project complete canonical evidence into bounded Agent-facing metadata."""

    def build(
        self,
        state: EvidenceState,
        descriptor: GoalDescriptor | None,
        contract: CompletionContract | None,
        *,
        max_records: int = 64,
        max_chars: int = 8_000,
    ) -> EvidenceProjection:
        all_records = list(state.records)
        current = [record for record in all_records if record.status == "VALID"]
        required: set[tuple[EvidenceKind, str]] = set()
        if contract is not None:
            for requirements in contract.requirements_by_goal.values():
                for requirement in requirements:
                    if requirement.kind is not RequirementKind.TARGET_BINDING:
                        required.add((EvidenceKind(requirement.kind.value), requirement.target))

        critical: list[EvidenceRecord] = []
        for kind, target in sorted(required, key=lambda item: (item[0].value, item[1])):
            matches = [record for record in current if record.kind is kind and record.target == target]
            if matches:
                critical.append(max(matches, key=lambda record: record.observed_at))

        if len(critical) > max_records:
            raise ContextBudgetExceeded("critical evidence record count exceeds context budget")
        selected = list(critical)
        selected_ids = {record.evidence_id for record in selected}
        targets = {
            goal.target if hasattr(goal, "target") and getattr(goal, "target") else getattr(goal, "topic", None)
            for goal in (descriptor.goals if descriptor else ())
        }
        ranked = sorted(
            [record for record in current if record.evidence_id not in selected_ids],
            key=lambda record: (
                0 if record.target in targets else 1,
                -record.observed_at.timestamp(),
            ),
        )
        selected.extend(ranked)
        projected: list[EvidenceProjectionRecord] = []
        used = 0
        for record in selected:
            if len(projected) >= max_records:
                break
            item = _project_record(record)
            cost = _projection_cost(item)
            if used + cost > max_chars:
                if record in critical:
                    raise ContextBudgetExceeded("critical evidence metadata exceeds context budget")
                continue
            projected.append(item)
            used += cost
        return EvidenceProjection(
            records=tuple(projected),
            total_records=len(all_records),
            omitted_records=max(0, len(all_records) - len(projected)),
            estimated_chars=used,
        )


def _project_record(record: EvidenceRecord) -> EvidenceProjectionRecord:
    return EvidenceProjectionRecord(
        evidence_id=record.evidence_id,
        kind=record.kind,
        target=record.target,
        observation_id=record.observation_id,
        source_tool=record.source_tool,
        observed_at=record.observed_at,
        status=record.status,
        valid_until=record.valid_until,
        entity_version=record.entity_version,
        requested_scope=record.requested_scope,
        observed_scope=record.observed_scope,
        requested_identity=record.provenance.requested_identity,
        observed_identity=record.provenance.observed_identity,
        identity_status=record.provenance.identity_status,
        scope_status=record.provenance.scope_status,
    )


def _projection_cost(item: EvidenceProjectionRecord) -> int:
    return len(repr(item))
