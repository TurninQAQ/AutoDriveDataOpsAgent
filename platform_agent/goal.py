"""Deterministic request-goal interpretation and completion checks.

Goals describe the result requested by the user.  This module deliberately does
not choose tools: it only derives bounded completion criteria from a GoalType and
checks those criteria against actual observations/evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .evidence import TARGET_AWARE_EVIDENCE_TYPES, EvidenceTracker
from .models import (
    AgentGoal,
    AgentIntent,
    AgentPlan,
    EvidenceRecord,
    EvidenceType,
    GoalContract,
    GoalEvaluation,
    GoalProgress,
    GoalType,
    KnowledgeObservation,
    ToolObservation,
)


_GOAL_CRITERIA: dict[GoalType, tuple[str, ...]] = {
    GoalType.ANSWER_KNOWLEDGE: ("STATIC_KNOWLEDGE",),
    GoalType.REPORT_LIVE_STATE: ("LIVE_OPERATIONAL_EVIDENCE",),
    GoalType.DIAGNOSE_ROOT_CAUSE: ("DIAGNOSTIC_CONTEXT",),
    GoalType.EXPLAIN_WITH_PLATFORM_RULES: (
        "DIAGNOSTIC_CONTEXT",
        "STATIC_KNOWLEDGE",
    ),
    GoalType.VERIFY_RECOVERY_STATE: ("LIVE_TASK", "RECOVERY_STATE"),
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


def resolve_goal_contract(goal_type: GoalType, intent: AgentIntent) -> GoalContract:
    """Resolve and freeze domain-specific completion conditions.

    The contract describes evidence classes only.  It never names a concrete
    Tool, so the Adaptive Agent retains responsibility for selecting a suitable
    read-only source.
    """

    try:
        intent = intent if isinstance(intent, AgentIntent) else AgentIntent(str(intent))
    except (TypeError, ValueError):
        intent = AgentIntent.GENERAL_READ

    conditions: tuple[str, ...]
    if goal_type == GoalType.ANSWER_KNOWLEDGE:
        conditions = (EvidenceType.STATIC_KNOWLEDGE.value,)
    elif goal_type == GoalType.REPORT_LIVE_STATE:
        conditions = {
            AgentIntent.TASK_STATUS: (EvidenceType.LIVE_TASK.value,),
            AgentIntent.LIST_TASKS: (EvidenceType.LIVE_TASK.value,),
            AgentIntent.GPU_DIAGNOSIS: (EvidenceType.LIVE_GPU.value,),
            AgentIntent.PLATFORM_HEALTH: (EvidenceType.PLATFORM_HEALTH.value,),
        }.get(intent, ("LIVE_OPERATIONAL_EVIDENCE",))
    elif goal_type == GoalType.DIAGNOSE_ROOT_CAUSE:
        conditions = (EvidenceType.DIAGNOSTIC_CONTEXT.value,)
    elif goal_type == GoalType.EXPLAIN_WITH_PLATFORM_RULES:
        conditions = {
            AgentIntent.TASK_DIAGNOSIS: (
                EvidenceType.DIAGNOSTIC_CONTEXT.value,
                EvidenceType.STATIC_KNOWLEDGE.value,
            ),
            AgentIntent.STAGE_FAILURE: (
                EvidenceType.DIAGNOSTIC_CONTEXT.value,
                EvidenceType.STATIC_KNOWLEDGE.value,
            ),
            AgentIntent.GPU_DIAGNOSIS: (
                EvidenceType.LIVE_GPU.value,
                EvidenceType.STATIC_KNOWLEDGE.value,
            ),
            AgentIntent.TASK_STATUS: (
                EvidenceType.LIVE_TASK.value,
                EvidenceType.STATIC_KNOWLEDGE.value,
            ),
            AgentIntent.PLATFORM_HEALTH: (
                EvidenceType.PLATFORM_HEALTH.value,
                EvidenceType.STATIC_KNOWLEDGE.value,
            ),
        }.get(intent, ("LIVE_OPERATIONAL_EVIDENCE", EvidenceType.STATIC_KNOWLEDGE.value))
    elif goal_type == GoalType.VERIFY_RECOVERY_STATE:
        conditions = (EvidenceType.LIVE_TASK.value, EvidenceType.RECOVERY_STATE.value)
    else:
        conditions = _GOAL_CRITERIA[goal_type]
    return GoalContract(
        goal_type=goal_type,
        domain_intent=intent,
        required_conditions=list(conditions),
        schema_version="v1.6.2",
    )


def evidence_matches_goal_target(
    record: EvidenceRecord | dict[str, Any],
    goal: AgentGoal | dict[str, Any],
) -> bool:
    """Return whether a record can satisfy this goal's entity scope.

    Task-scoped evidence requires exact task identity.  Global evidence remains
    usable for target-bound goals when its evidence domain is intentionally global
    (for example LIVE_GPU or STATIC_KNOWLEDGE).
    """

    if not isinstance(record, EvidenceRecord):
        try:
            record = EvidenceRecord.model_validate(record)
        except Exception:
            return False
    if not isinstance(goal, AgentGoal):
        try:
            goal = AgentGoal.model_validate(goal)
        except Exception:
            return False
    target = goal.target.strip() if isinstance(goal.target, str) else None
    if not target or record.type not in TARGET_AWARE_EVIDENCE_TYPES:
        return True
    return bool(record.task_name and record.task_name.strip() == target)


def criteria_for_goal(goal_type: GoalType, intent: AgentIntent | None = None) -> list[str]:
    if intent is None:
        return list(_GOAL_CRITERIA[goal_type])
    return resolve_goal_contract(goal_type, intent).required_conditions


def goal_for_intent(
    intent: AgentIntent,
    *,
    target: str | None = None,
) -> AgentGoal:
    goal_type = _INTENT_TO_GOAL.get(intent, GoalType.GENERAL_ASSISTANCE)
    contract = resolve_goal_contract(goal_type, intent)
    return AgentGoal(
        goal_type=goal_type,
        target=target,
        success_criteria=list(contract.required_conditions),
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
    contract = resolve_goal_contract(goal.goal_type, intent)
    return goal.model_copy(
        update={
            "target": goal.target or target,
            "success_criteria": list(contract.required_conditions),
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


def _normalized_records(
    evidence_records: Iterable[EvidenceRecord | dict[str, Any]],
    observations: list[ToolObservation],
) -> list[EvidenceRecord]:
    """Load saved records and derive missing records from current observations."""

    tracker = EvidenceTracker.from_records(evidence_records)
    derived = EvidenceTracker.from_observations(observations)
    records: list[EvidenceRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for item in [*tracker.records, *derived.records]:
        key = (item.type, item.source_tool, item.task_name, item.dataset_name, item.summary)
        if key in seen:
            continue
        seen.add(key)
        records.append(item)
    return records


def _non_empty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _matching_records(
    records: Iterable[EvidenceRecord],
    goal: AgentGoal,
    evidence_type: EvidenceType,
) -> list[EvidenceRecord]:
    return [
        item
        for item in records
        if item.type == evidence_type and evidence_matches_goal_target(item, goal)
    ]


def _condition_satisfied(
    condition: str,
    goal: AgentGoal,
    records: list[EvidenceRecord],
) -> bool:
    if condition == "STATIC_KNOWLEDGE":
        return bool(_matching_records(records, goal, EvidenceType.STATIC_KNOWLEDGE))
    if condition == "LIVE_OPERATIONAL_EVIDENCE":
        return any(
            _matching_records(records, goal, candidate)
            for candidate in _LIVE_EVIDENCE
        )
    if condition == EvidenceType.LIVE_TASK.value:
        return bool(_matching_records(records, goal, EvidenceType.LIVE_TASK))
    if condition == EvidenceType.LIVE_GPU.value:
        return bool(_matching_records(records, goal, EvidenceType.LIVE_GPU))
    if condition == EvidenceType.PLATFORM_HEALTH.value:
        return bool(_matching_records(records, goal, EvidenceType.PLATFORM_HEALTH))
    if condition == EvidenceType.DIAGNOSTIC_CONTEXT.value:
        return bool(_matching_records(records, goal, EvidenceType.DIAGNOSTIC_CONTEXT))
    if condition == EvidenceType.DIAGNOSIS.value:
        return bool(_matching_records(records, goal, EvidenceType.DIAGNOSIS))
    if condition == EvidenceType.RECOVERY_STATE.value:
        return bool(_matching_records(records, goal, EvidenceType.RECOVERY_STATE))
    if condition == "TASK_PLAN_VALIDATED" or condition == "WRITE_PLAN_PREPARED":
        return False
    return False


def evaluate_goal_progress(
    goal: AgentGoal | dict[str, Any],
    evidence_records: Iterable[EvidenceRecord | dict[str, Any]] = (),
    observations: Iterable[ToolObservation] = (),
    knowledge: Iterable[KnowledgeObservation] = (),
    current_intent: AgentIntent | None = None,
    goal_contract: GoalContract | dict[str, Any] | None = None,
) -> GoalEvaluation:
    """Evaluate a request goal from actual observations, never from model claims."""

    if not isinstance(goal, AgentGoal):
        goal = AgentGoal.model_validate(goal)
    if goal_contract is not None:
        contract = (
            goal_contract
            if isinstance(goal_contract, GoalContract)
            else GoalContract.model_validate(goal_contract)
        )
    elif current_intent is not None:
        contract = resolve_goal_contract(goal.goal_type, current_intent)
    else:
        # Compatibility for direct callers that predate frozen contracts. The
        # production workflow always supplies the initial-plan contract.
        contract = GoalContract(
            goal_type=goal.goal_type,
            domain_intent=AgentIntent.GENERAL_READ,
            required_conditions=list(_GOAL_CRITERIA[goal.goal_type]),
        )
    observation_list = list(observations)
    records = _normalized_records(evidence_records, observation_list)
    if any(item.ok and item.tool_name == "search_knowledge" for item in observation_list):
        if not any(item.type == EvidenceType.STATIC_KNOWLEDGE for item in records):
            records.append(
                EvidenceRecord(
                    type=EvidenceType.STATIC_KNOWLEDGE,
                    source_tool="search_knowledge",
                    timestamp=0.0,
                    summary="search_knowledge returned static knowledge.",
                )
            )
    if any(item.content.strip() and not item.metadata.get("error") for item in knowledge):
        if not any(item.type == EvidenceType.STATIC_KNOWLEDGE for item in records):
            records.append(
                EvidenceRecord(
                    type=EvidenceType.STATIC_KNOWLEDGE,
                    source_tool="knowledge_retriever",
                    timestamp=0.0,
                    summary="Static knowledge is available.",
                )
            )

    satisfied: list[str] = []
    missing: list[str] = []
    for condition in contract.required_conditions:
        present = _condition_satisfied(condition, goal, records)
        (satisfied if present else missing).append(condition)

    if not missing:
        state = GoalProgress.SATISFIED
        summary = "Goal completion criteria are supported by observed evidence."
    elif satisfied:
        state = GoalProgress.IN_PROGRESS
        summary = "Some goal completion criteria are supported; more evidence is missing."
    elif not contract.required_conditions:
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


def finalize_goal_response(
    goal: AgentGoal | dict[str, Any],
    goal_contract: GoalContract | dict[str, Any],
    evidence_evaluation: GoalEvaluation,
    response: Any,
) -> GoalEvaluation:
    """Apply deterministic post-synthesis checks to a goal evaluation.

    Evidence readiness is not itself a natural-language diagnosis.  Diagnosis
    goals require the synthesis response to contain a non-empty root_cause; the
    finalizer validates that shape but never invents or edits the conclusion.
    """

    if not isinstance(goal, AgentGoal):
        goal = AgentGoal.model_validate(goal)
    if not isinstance(goal_contract, GoalContract):
        goal_contract = GoalContract.model_validate(goal_contract)

    diagnosis_goal = goal.goal_type in {
        GoalType.DIAGNOSE_ROOT_CAUSE,
        GoalType.EXPLAIN_WITH_PLATFORM_RULES,
    } and EvidenceType.DIAGNOSTIC_CONTEXT.value in goal_contract.required_conditions
    if not diagnosis_goal or evidence_evaluation.state != GoalProgress.SATISFIED:
        return evidence_evaluation

    root_cause = getattr(response, "root_cause", None)
    if _non_empty(root_cause):
        return evidence_evaluation

    missing = list(evidence_evaluation.missing_conditions)
    if "ROOT_CAUSE_CONCLUSION" not in missing:
        missing.append("ROOT_CAUSE_CONCLUSION")
    return evidence_evaluation.model_copy(
        update={
            "state": GoalProgress.IN_PROGRESS,
            "missing_conditions": missing,
            "summary": "Diagnostic evidence is available, but synthesis did not provide a root-cause conclusion.",
        }
    )


__all__ = [
    "criteria_for_goal",
    "evaluate_goal_progress",
    "evidence_matches_goal_target",
    "finalize_goal_response",
    "goal_for_intent",
    "normalize_goal",
    "normalize_plan_goal",
    "resolve_goal_contract",
]
