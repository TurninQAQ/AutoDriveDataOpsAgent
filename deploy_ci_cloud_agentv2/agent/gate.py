"""Deterministic normal-completion gate."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import CompletionContract, RequirementKind
from .decisions import FinalCandidate
from .evidence import EvidenceState
from .goals import GoalDescriptor
from .outcomes import GoalOutcome, GoalStatus


@dataclass(frozen=True)
class GateEvaluation:
    passed: bool
    facts: tuple[str, ...]
    missing: tuple[str, ...]
    goal_outcomes: dict[str, GoalOutcome]


class ResponseCompletionGate:
    """Own eligibility for normal completion; it never selects a next tool."""

    def evaluate(
        self,
        candidate: FinalCandidate,
        descriptor: GoalDescriptor,
        contract: CompletionContract,
        evidence: EvidenceState,
        current_outcomes: dict[str, GoalOutcome],
    ) -> GateEvaluation:
        now = evidence.current()
        outcomes: dict[str, GoalOutcome] = {}
        missing: list[str] = []
        facts: list[str] = []
        for goal in descriptor.goals:
            prior = current_outcomes.get(goal.goal_id, GoalOutcome(goal.goal_id))
            if prior.status in {
                GoalStatus.DENIED,
                GoalStatus.REJECTED,
                GoalStatus.FAILED,
                GoalStatus.INCONCLUSIVE,
                GoalStatus.BLOCKED,
            }:
                outcomes[goal.goal_id] = prior
                facts.append(f"goal {goal.goal_id} is {prior.status.value}")
                continue
            requirements = contract.requirements_by_goal[goal.goal_id]
            refs: list[str] = []
            goal_missing: list[str] = []
            for requirement in requirements:
                if requirement.kind is RequirementKind.TARGET_BINDING:
                    if not requirement.target.strip():
                        goal_missing.append("target binding")
                    continue
                matching = [
                    record
                    for record in now
                    if record.kind == requirement.kind.value
                    and self._target_matches(
                        requirement.target, record.target, requirement.kind.value
                    )
                ]
                if matching:
                    refs.extend(record.evidence_id for record in matching)
                else:
                    goal_missing.append(
                        f"{requirement.kind.value} evidence for {requirement.target}"
                    )
            if goal_missing:
                outcomes[goal.goal_id] = GoalOutcome(
                    goal_id=goal.goal_id,
                    status=GoalStatus.PENDING,
                    reason_code="REQUIRED_EVIDENCE_MISSING",
                    evidence_refs=tuple(refs),
                )
                missing.extend(goal_missing)
                facts.append(f"goal {goal.goal_id} remains PENDING")
            else:
                outcomes[goal.goal_id] = GoalOutcome(
                    goal_id=goal.goal_id,
                    status=GoalStatus.SATISFIED,
                    evidence_refs=tuple(dict.fromkeys(refs)),
                )
                facts.append(f"goal {goal.goal_id} has required current evidence")
        return GateEvaluation(not missing, tuple(facts), tuple(missing), outcomes)

    @staticmethod
    def _target_matches(required: str, actual: str, kind: str) -> bool:
        if kind in {"GPU_POOL", "QUEUE_STATE"} and required == "platform":
            return actual in {"", "platform"}
        return required == actual
