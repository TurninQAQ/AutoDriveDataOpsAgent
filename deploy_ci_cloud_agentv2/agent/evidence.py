"""Canonical evidence records with provenance and basic freshness."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .outcomes import GoalOutcome, GoalStatus


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


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    kind: str
    target: str
    observation_id: str
    provenance: str
    observed_at: datetime
    entity_version: str | None = None
    valid_until: datetime | None = None
    status: str = "VALID"
    invalidated_by: str | None = None

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
            if observation.status != "SUCCESS" or observation.data is None:
                continue
            kind = self._kind_for(observation.source)
            if kind is None:
                continue
            observed_at = observation.observed_at
            record = EvidenceRecord(
                evidence_id=f"ev_{observation.observation_id}",
                kind=kind,
                target=observation.target,
                observation_id=observation.observation_id,
                provenance=observation.source,
                observed_at=observed_at,
                entity_version=self._entity_version(observation.data),
                valid_until=observed_at + timedelta(seconds=self.freshness_seconds),
            )
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
                and self._target_matches(requirement.target, record.target, requirement.kind.value)
            )
            complete = all(
                requirement.kind.value == "TARGET_BINDING"
                or any(
                    record.kind == requirement.kind.value
                    and self._target_matches(
                        requirement.target, record.target, requirement.kind.value
                    )
                    for record in current
                )
                for requirement in requirements
            )
            result[goal.goal_id] = GoalOutcome(
                goal_id=goal.goal_id,
                status=GoalStatus.SATISFIED if complete else GoalStatus.PENDING,
                evidence_refs=refs,
            )
        return result

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
    def _target_matches(required: str, actual: str, kind: str) -> bool:
        if kind in {"GPU_POOL", "QUEUE_STATE"} and required == "platform":
            return actual in {"", "platform"}
        return required == actual

    @staticmethod
    def _entity_version(data: object) -> str | None:
        if isinstance(data, dict):
            for key in ("entity_version", "generation", "revision", "etag", "version"):
                if data.get(key) is not None:
                    return str(data[key])
        return None
