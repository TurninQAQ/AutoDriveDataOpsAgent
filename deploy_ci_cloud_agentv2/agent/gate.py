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
    """Own normal completion eligibility; it never selects a next tool.

    The gate performs structural closure only. It does not judge the truth of
    free-form prose and does not call another model.
    """

    def evaluate(
        self,
        candidate: FinalCandidate,
        descriptor: GoalDescriptor,
        contract: CompletionContract,
        evidence: EvidenceState,
        current_outcomes: dict[str, GoalOutcome],
    ) -> GateEvaluation:
        goal_ids = tuple(goal.goal_id for goal in descriptor.goals)
        candidate_ids = tuple(candidate.referenced_goal_ids)
        facts: list[str] = []
        missing: list[str] = []

        if len(candidate_ids) != len(set(candidate_ids)):
            missing.append("FinalCandidate references duplicate goal ids")
        unknown = sorted(set(candidate_ids) - set(goal_ids))
        if unknown:
            missing.append(f"FinalCandidate references unknown goal ids: {', '.join(unknown)}")
        omitted = [goal_id for goal_id in goal_ids if goal_id not in candidate_ids]
        if omitted:
            missing.append(
                "FinalCandidate does not reference required goals: " + ", ".join(omitted)
            )

        current = evidence.current()
        outcomes: dict[str, GoalOutcome] = {}
        for goal in descriptor.goals:
            prior = current_outcomes.get(goal.goal_id, GoalOutcome(goal.goal_id))
            if prior.status in {
                GoalStatus.DENIED,
                GoalStatus.REJECTED,
                GoalStatus.FAILED,
                GoalStatus.INCONCLUSIVE,
                GoalStatus.BLOCKED,
            }:
                # A terminal non-success outcome can be reported honestly only
                # when the Agent references that goal in the candidate.
                outcomes[goal.goal_id] = prior
                facts.append(f"goal {goal.goal_id} is {prior.status.value}")
                continue

            requirements = contract.requirements_by_goal.get(goal.goal_id)
            if requirements is None:
                missing.append(f"completion contract missing goal {goal.goal_id}")
                outcomes[goal.goal_id] = GoalOutcome(
                    goal.goal_id, status=GoalStatus.PENDING, reason_code="CONTRACT_MISMATCH"
                )
                continue

            refs: list[str] = []
            goal_missing: list[str] = []
            for requirement in requirements:
                if requirement.kind is RequirementKind.TARGET_BINDING:
                    if not requirement.target.strip():
                        goal_missing.append("target binding")
                    continue
                matching = [
                    record
                    for record in current
                    if record.kind == requirement.kind.value
                    and _target_matches(
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
                    evidence_refs=tuple(dict.fromkeys(refs)),
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

        for goal_id, outcome in outcomes.items():
            if outcome.status is GoalStatus.PENDING:
                continue
            if goal_id not in candidate_ids:
                missing.append(f"resolved goal {goal_id} is not referenced by FinalCandidate")

        # A candidate can pass only when every declared goal is structurally
        # accounted for and no goal remains recoverably unresolved.
        passed = not missing and all(
            outcome.status is not GoalStatus.PENDING for outcome in outcomes.values()
        )
        return GateEvaluation(passed, tuple(dict.fromkeys(facts)), tuple(dict.fromkeys(missing)), outcomes)


def _target_matches(required: str, actual: str, kind: str) -> bool:
    if kind in {"GPU_POOL", "QUEUE_STATE"} and required == "platform":
        return actual == "platform"
    return required == actual
