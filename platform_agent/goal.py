"""Deterministic request-goal interpretation and completion checks.

Goals describe the result requested by the user.  This module deliberately does
not choose tools: it only derives bounded completion criteria from a GoalType and
checks those criteria against actual observations/evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import (
    AgentGoal,
    AgentIntent,
    AgentPlan,
    EvidenceRecord,
    EvidenceType,
    GoalEvaluation,
    GoalProgress,
    GoalType,
    KnowledgeObservation,
    ToolObservation,
)


_GOAL_CRITERIA: dict[GoalType, tuple[str, ...]] = {
    GoalType.ANSWER_KNOWLEDGE: ("STATIC_KNOWLEDGE",),
    GoalType.REPORT_LIVE_STATE: ("LIVE_OPERATIONAL_EVIDENCE",),
    GoalType.DIAGNOSE_ROOT_CAUSE: ("DIAGNOSIS",),
    GoalType.EXPLAIN_WITH_PLATFORM_RULES: (
        "LIVE_OPERATIONAL_EVIDENCE",
        "STATIC_KNOWLEDGE",
    ),
    GoalType.VERIFY_RECOVERY_STATE: (
        "LIVE_OPERATIONAL_EVIDENCE",
        "RECOVERY_STATE",
    ),
    GoalType.PREPARE_TASK_PLAN: ("TASK_PLAN_VALIDATED",),
    GoalType.PREPARE_WRITE_ACTION: ("WRITE_PLAN_PREPARED",),
    GoalType.GENERAL_ASSISTANCE: (),
}

_INTENT_TO_GOAL: dict[AgentIntent, GoalType] = {
    AgentIntent.PLATFORM_HEALTH: GoalType.REPORT_LIVE_STATE,
    AgentIntent.LIST_TASKS: GoalType.REPORT_LIVE_STATE,
    AgentIntent.TASK_STATUS: GoalType.REPORT_LIVE_STATE,
    AgentIntent.TASK_DIAGNOSIS: GoalType.DIAGNOSE_ROOT_CAUSE,
    AgentIntent.GPU_DIAGNOSIS: GoalType.REPORT_LIVE_STATE,
    AgentIntent.STAGE_FAILURE: GoalType.DIAGNOSE_ROOT_CAUSE,
    AgentIntent.GENERAL_READ: GoalType.GENERAL_ASSISTANCE,
    AgentIntent.PLATFORM_KNOWLEDGE: GoalType.ANSWER_KNOWLEDGE,
    AgentIntent.TASK_PLANNING: GoalType.PREPARE_TASK_PLAN,
    AgentIntent.SUBMIT_TASK: GoalType.PREPARE_WRITE_ACTION,
    AgentIntent.RESUME_TASK: GoalType.PREPARE_WRITE_ACTION,
    AgentIntent.SET_TASK_PRIORITY: GoalType.PREPARE_WRITE_ACTION,
    AgentIntent.STOP_TASK: GoalType.PREPARE_WRITE_ACTION,
    AgentIntent.DELETE_TASK: GoalType.PREPARE_WRITE_ACTION,
    AgentIntent.UNSUPPORTED_WRITE: GoalType.PREPARE_WRITE_ACTION,
}

_LIVE_EVIDENCE = frozenset(
    {
        EvidenceType.LIVE_TASK,
        EvidenceType.LIVE_GPU,
        EvidenceType.LIVE_QUEUE,
        EvidenceType.LIVE_LOG,
        EvidenceType.LIVE_CONTAINER,
        EvidenceType.PLATFORM_HEALTH,
    }
)
_TASK_TOOLS = frozenset({"diagnose_task", "get_task_detail", "get_queue_state", "get_stage_logs", "inspect_task_containers"})


def criteria_for_goal(goal_type: GoalType) -> list[str]:
    return list(_GOAL_CRITERIA[goal_type])


def goal_for_intent(
    intent: AgentIntent,
    *,
    target: str | None = None,
) -> AgentGoal:
    goal_type = _INTENT_TO_GOAL.get(intent, GoalType.GENERAL_ASSISTANCE)
    return AgentGoal(
        goal_type=goal_type,
        target=target,
        success_criteria=criteria_for_goal(goal_type),
    )


def normalize_goal(
    goal: AgentGoal | dict[str, Any] | None,
    intent: AgentIntent,
    *,
    target: str | None = None,
) -> AgentGoal:
    """Normalize model/fake-model output to deterministic criteria.

    A provider may return a GoalType, but its free-form success criteria are not
    trusted as policy.  Criteria always come from this module.
    """

    if goal is None:
        return goal_for_intent(intent, target=target)
    if not isinstance(goal, AgentGoal):
        goal = AgentGoal.model_validate(goal)
    return goal.model_copy(
        update={
            "target": goal.target or target,
            "success_criteria": criteria_for_goal(goal.goal_type),
            "completion_state": GoalProgress.NOT_STARTED,
        }
    )


def normalize_plan_goal(plan: AgentPlan) -> AgentPlan:
    return plan.model_copy(
        update={
            "goal": normalize_goal(
                plan.goal,
                plan.intent,
                target=plan.task_name,
            )
        }
    )


def _record_types(records: Iterable[EvidenceRecord | dict[str, Any]]) -> set[EvidenceType]:
    types: set[EvidenceType] = set()
    for item in records:
        value = item.type if isinstance(item, EvidenceRecord) else item.get("type") if isinstance(item, dict) else None
        try:
            types.add(value if isinstance(value, EvidenceType) else EvidenceType(str(value)))
        except (TypeError, ValueError):
            continue
    return types


def _non_empty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _has_explicit_diagnosis(observations: Iterable[ToolObservation]) -> bool:
    diagnosis_keys = {
        "diagnosis",
        "root_cause",
        "rootcause",
        "reason",
        "failure_reason",
        "cause",
    }
    for observation in observations:
        if not observation.ok or observation.tool_name not in {"diagnose_task", "get_stage_logs"}:
            continue
        data = observation.data
        if not isinstance(data, dict):
            continue
        if any(_non_empty(data.get(key)) for key in diagnosis_keys):
            return True
    return False


def _has_recovery_evidence(observations: Iterable[ToolObservation]) -> bool:
    recovery_keys = {
        "recovery",
        "recovery_state",
        "recovery_runs",
        "checkpoint",
        "checkpoint_state",
        "resume",
        "resumed",
        "recovered",
    }
    for observation in observations:
        if not observation.ok or observation.tool_name not in _TASK_TOOLS:
            continue
        data = observation.data
        if isinstance(data, dict) and any(_non_empty(data.get(key)) for key in recovery_keys):
            return True
    return False


def _has_relevant_live_evidence(
    goal: AgentGoal,
    records: set[EvidenceType],
    observations: Iterable[ToolObservation],
) -> bool:
    if not records.intersection(_LIVE_EVIDENCE):
        return False
    if not goal.target:
        return True
    for observation in observations:
        if not observation.ok or observation.tool_name not in _TASK_TOOLS:
            continue
        argument_target = observation.arguments.get("task_name")
        data_target = observation.data.get("task_name") if isinstance(observation.data, dict) else None
        if goal.target in {str(argument_target or ""), str(data_target or "")}:
            return True
    return False


def evaluate_goal_progress(
    goal: AgentGoal | dict[str, Any],
    evidence_records: Iterable[EvidenceRecord | dict[str, Any]] = (),
    observations: Iterable[ToolObservation] = (),
    knowledge: Iterable[KnowledgeObservation] = (),
) -> GoalEvaluation:
    """Evaluate a request goal from actual observations, never from model claims."""

    if not isinstance(goal, AgentGoal):
        goal = AgentGoal.model_validate(goal)
    observation_list = list(observations)
    types = _record_types(evidence_records)
    if any(item.ok and item.tool_name == "search_knowledge" for item in observation_list):
        types.add(EvidenceType.STATIC_KNOWLEDGE)
    if any(item.content.strip() and not item.metadata.get("error") for item in knowledge):
        types.add(EvidenceType.STATIC_KNOWLEDGE)
    if _has_explicit_diagnosis(observation_list):
        types.add(EvidenceType.DIAGNOSIS)
    if _has_recovery_evidence(observation_list):
        types.add(EvidenceType.RECOVERY_STATE)

    satisfied: list[str] = []
    missing: list[str] = []
    for condition in _GOAL_CRITERIA[goal.goal_type]:
        if condition == "STATIC_KNOWLEDGE":
            present = EvidenceType.STATIC_KNOWLEDGE in types
        elif condition == "LIVE_OPERATIONAL_EVIDENCE":
            present = _has_relevant_live_evidence(goal, types, observation_list)
        elif condition == "DIAGNOSIS":
            present = EvidenceType.DIAGNOSIS in types
        elif condition == "RECOVERY_STATE":
            present = EvidenceType.RECOVERY_STATE in types
        else:
            # Task-plan and write-plan completion is finalized by their guarded
            # deterministic services, not by read observations.
            present = False
        (satisfied if present else missing).append(condition)

    if not missing:
        state = GoalProgress.SATISFIED
        summary = "Goal completion criteria are supported by observed evidence."
    elif satisfied:
        state = GoalProgress.IN_PROGRESS
        summary = "Some goal completion criteria are supported; more evidence is missing."
    elif not _GOAL_CRITERIA[goal.goal_type]:
        state = GoalProgress.SATISFIED
        summary = "No platform evidence is required for this request goal."
    else:
        state = GoalProgress.IN_PROGRESS
        summary = "No goal completion criteria are currently supported by observations."
    return GoalEvaluation(
        state=state,
        satisfied_conditions=satisfied,
        missing_conditions=missing,
        summary=summary,
    )


__all__ = [
    "criteria_for_goal",
    "evaluate_goal_progress",
    "goal_for_intent",
    "normalize_goal",
    "normalize_plan_goal",
]
