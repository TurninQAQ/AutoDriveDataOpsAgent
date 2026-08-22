"""Owned observations, fresh qualified evidence, and bounded projections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .contracts import CompletionContract, RequirementKind
from .immutable import canonical_snapshot
from .goals import GoalDescriptor
from .identity import RequestIdentity
from .outcomes import GoalOutcome, GoalStatus
from .provenance import (
    IdentityStatus,
    ObservationProvenance,
    ObservationScope,
    ScopeKind,
    ScopeStatus,
)
from .results import (
    DiagnosticResult,
    GpuPoolResult,
    KnowledgeResult,
    NormalizedReadResult,
    QueueResult,
    ResultStatus,
    TaskDetailResult,
)


class EvidenceKind(str, Enum):
    TARGET_BINDING = "TARGET_BINDING"
    LIVE_TASK = "LIVE_TASK"
    GPU_POOL = "GPU_POOL"
    QUEUE_STATE = "QUEUE_STATE"
    KNOWLEDGE = "KNOWLEDGE"
    DIAGNOSTIC_CONTEXT = "DIAGNOSTIC_CONTEXT"


class TransportStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class ObservationDisposition(str, Enum):
    NORMALIZED = "NORMALIZED"
    NORMALIZED_NO_QUALIFIED_EVIDENCE = "NORMALIZED_NO_QUALIFIED_EVIDENCE"
    ABSENT = "ABSENT"
    EXTERNAL_ERROR = "EXTERNAL_ERROR"
    MALFORMED = "MALFORMED"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    READ_GUARD_REJECTED = "READ_GUARD_REJECTED"


class EvidenceValidity(str, Enum):
    CURRENT = "CURRENT"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True)
class EvidenceFreshness:
    observed_at: datetime
    expires_at: datetime | None
    validity: EvidenceValidity = EvidenceValidity.CURRENT

    def is_current(self, now: datetime | None = None) -> bool:
        if self.validity is not EvidenceValidity.CURRENT:
            return False
        moment = now or datetime.now(timezone.utc)
        return self.expires_at is None or moment <= self.expires_at


@dataclass(frozen=True)
class EvidenceFreshnessPolicy:
    """Explicit default freshness policy for mutable READ observations."""

    default_ttl: timedelta | None = timedelta(minutes=5)

    def expiration(self, observed_at: datetime) -> datetime | None:
        return None if self.default_ttl is None else observed_at + self.default_ttl


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    call_id: str
    owner: RequestIdentity
    source: str
    target: str
    transport_status: TransportStatus
    disposition: ObservationDisposition
    data: object | None
    trust: str = "UNTRUSTED_EXTERNAL_DATA"
    error_code: str | None = None
    retryable: bool = False
    retry_count: int = 0
    observed_at: datetime = datetime.min.replace(tzinfo=timezone.utc)
    provenance: ObservationProvenance | None = None
    result: NormalizedReadResult | None = None

    def __post_init__(self) -> None:
        # Direct test hosts and future adapters may construct an observation
        # without going through ReadToolRuntime.  Keep the same authority
        # boundary in that path as well: observation data is never a mutable
        # handle to a caller-owned payload.
        object.__setattr__(self, "data", canonical_snapshot(self.data))

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
    owner: RequestIdentity
    provenance: ObservationProvenance
    freshness: EvidenceFreshness
    entity_version: str | None = None

    @property
    def source_tool(self) -> str:
        return self.provenance.source_tool

    @property
    def requested_scope(self) -> ObservationScope:
        return self.provenance.requested_scope

    @property
    def observed_scope(self) -> ObservationScope:
        return self.provenance.observed_scope

    @property
    def observed_target(self) -> str | None:
        return self.provenance.observed_target

    def is_current(self, now: datetime | None = None) -> bool:
        return self.freshness.is_current(now)


@dataclass(frozen=True)
class EvidenceState:
    owner: RequestIdentity
    records: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if any(record.owner != self.owner for record in self.records):
            raise ValueError("EvidenceState contains a record owned by another request")

    def current(self, now: datetime | None = None) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.is_current(now))


@dataclass(frozen=True)
class EvidenceProjectionRecord:
    """Bounded metadata; it contains no raw payload or semantic content."""

    evidence_id: str
    kind: EvidenceKind
    target: str
    observation_id: str
    request_id: str
    turn_id: str
    source_tool: str
    freshness: EvidenceFreshness
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))


class ContextBudgetExceeded(RuntimeError):
    """The critical structured projection cannot fit the explicit budget."""


class EvidenceTracker:
    """The only component allowed to create evidence records from reads."""

    _tool_kinds = {
        "get_task_detail": (TaskDetailResult, EvidenceKind.LIVE_TASK),
        "get_gpu_pool": (GpuPoolResult, EvidenceKind.GPU_POOL),
        "search_knowledge": (KnowledgeResult, EvidenceKind.KNOWLEDGE),
        "get_queue_state": (QueueResult, EvidenceKind.QUEUE_STATE),
        "diagnose_task": (DiagnosticResult, EvidenceKind.DIAGNOSTIC_CONTEXT),
    }

    def __init__(self, freshness_policy: EvidenceFreshnessPolicy | None = None) -> None:
        self.freshness_policy = freshness_policy or EvidenceFreshnessPolicy()

    def record_observations(
        self,
        state: EvidenceState,
        observations: tuple[ToolObservation, ...] | list[ToolObservation],
        owner: RequestIdentity,
    ) -> tuple[EvidenceState, tuple[EvidenceRecord, ...]]:
        if state.owner != owner:
            raise ValueError("EvidenceTracker owner does not match EvidenceState owner")
        records = list(state.records)
        created: list[EvidenceRecord] = []
        for observation in observations:
            record = self._qualify(observation, owner)
            if record is not None:
                records.append(record)
                created.append(record)
        return EvidenceState(owner=owner, records=tuple(records)), tuple(created)

    def refresh_goal_outcomes(
        self,
        descriptor: GoalDescriptor,
        contract: CompletionContract,
        evidence: EvidenceState,
        outcomes: dict[str, GoalOutcome],
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
            supporting_refs: list[str] = []
            satisfied = True
            for requirement in requirements:
                if requirement.kind is RequirementKind.TARGET_BINDING:
                    continue
                matching = self._matching_records(
                    current, requirement.kind, requirement.target
                )
                if not matching:
                    satisfied = False
                    continue
                supporting_refs.append(
                    max(matching, key=lambda record: record.freshness.observed_at).evidence_id
                )
            refreshed[goal.goal_id] = GoalOutcome(
                goal_id=goal.goal_id,
                status=GoalStatus.SATISFIED if satisfied else GoalStatus.PENDING,
                reason_code=(
                    "QUALIFIED_EVIDENCE_SATISFIED"
                    if satisfied
                    else "REQUIRED_EVIDENCE_MISSING"
                ),
                evidence_refs=tuple(dict.fromkeys(supporting_refs)) if satisfied else (),
            )
        return refreshed

    def _qualify(
        self, observation: ToolObservation, owner: RequestIdentity
    ) -> EvidenceRecord | None:
        if observation.owner != owner:
            return None
        if (
            observation.transport_status is not TransportStatus.SUCCESS
            or observation.disposition is not ObservationDisposition.NORMALIZED
            or observation.result is None
        ):
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
            owner=owner,
            provenance=provenance,
            freshness=EvidenceFreshness(
                observed_at=observation.observed_at,
                expires_at=self.freshness_policy.expiration(observation.observed_at),
            ),
            entity_version=result.entity_version,
        )

    @staticmethod
    def _has_requirement(
        records: tuple[EvidenceRecord, ...], kind: RequirementKind, target: str
    ) -> bool:
        return bool(EvidenceTracker._matching_records(records, kind, target))

    @staticmethod
    def _matching_records(
        records: tuple[EvidenceRecord, ...], kind: RequirementKind, target: str
    ) -> tuple[EvidenceRecord, ...]:
        try:
            evidence_kind = EvidenceKind(kind.value)
        except ValueError:
            return ()
        return tuple(
            record
            for record in records
            if record.kind is evidence_kind and record.target == target
        )


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
        current = list(state.current())
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
                critical.append(max(matches, key=lambda record: record.freshness.observed_at))
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
                -record.freshness.observed_at.timestamp(),
            ),
        )
        selected.extend(ranked)
        projected: list[EvidenceProjectionRecord] = []
        used = 0
        for record in selected:
            if len(projected) >= max_records:
                break
            item = _project_record(record)
            cost = len(repr(item))
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
        request_id=record.owner.request_id,
        turn_id=record.owner.turn_id,
        source_tool=record.source_tool,
        freshness=record.freshness,
        entity_version=record.entity_version,
        requested_scope=record.requested_scope,
        observed_scope=record.observed_scope,
        requested_identity=record.provenance.requested_identity,
        observed_identity=record.provenance.observed_identity,
        identity_status=record.provenance.identity_status,
        scope_status=record.provenance.scope_status,
    )
