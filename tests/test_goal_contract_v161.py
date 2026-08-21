from __future__ import annotations

from platform_agent.evidence import EvidenceTracker
from platform_agent.goal import evaluate_goal_progress, goal_for_intent, resolve_goal_contract
from platform_agent.models import (
    AgentGoal,
    AgentIntent,
    EvidenceType,
    GoalProgress,
    GoalType,
    ToolObservation,
)


def observation(tool: str, data, **arguments):
    return ToolObservation(tool_name=tool, arguments=arguments, ok=True, data=data)


def records_for(*items):
    tracker = EvidenceTracker()
    for item in items:
        tracker.record_tool_observation(item)
    return tracker.records


def test_goal_contract_matrix_is_domain_specific():
    assert resolve_goal_contract(
        GoalType.EXPLAIN_WITH_PLATFORM_RULES, AgentIntent.TASK_DIAGNOSIS
    ).required_conditions == ["DIAGNOSTIC_CONTEXT", "STATIC_KNOWLEDGE"]
    assert resolve_goal_contract(
        GoalType.EXPLAIN_WITH_PLATFORM_RULES, AgentIntent.GPU_DIAGNOSIS
    ).required_conditions == ["LIVE_GPU", "STATIC_KNOWLEDGE"]
    assert resolve_goal_contract(
        GoalType.REPORT_LIVE_STATE, AgentIntent.TASK_STATUS
    ).required_conditions == ["LIVE_TASK"]
    assert resolve_goal_contract(
        GoalType.REPORT_LIVE_STATE, AgentIntent.GPU_DIAGNOSIS
    ).required_conditions == ["LIVE_GPU"]
    assert resolve_goal_contract(
        GoalType.REPORT_LIVE_STATE, AgentIntent.PLATFORM_HEALTH
    ).required_conditions == ["PLATFORM_HEALTH"]


def test_task_state_and_rules_without_diagnosis_remain_incomplete():
    goal = AgentGoal(
        goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES,
        target="release_demo",
    )
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS)
    task_state = observation(
        "get_task_detail",
        {"task_name": "release_demo", "state": "draining"},
        task_name="release_demo",
    )
    rules = observation(
        "search_knowledge",
        {"results": [{"content": "soft preemption waits for a Stage boundary"}]},
        query="soft preemption",
    )
    result = evaluate_goal_progress(
        goal,
        records_for(task_state, rules),
        [task_state, rules],
        goal_contract=contract,
    )

    assert result.state == GoalProgress.IN_PROGRESS
    assert result.missing_conditions == [EvidenceType.DIAGNOSTIC_CONTEXT.value]


def test_task_diagnosis_and_rules_satisfy_frozen_hybrid_contract():
    goal = AgentGoal(goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES, target="release_demo")
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS)
    diagnosis = observation(
        "diagnose_task",
        {"task_name": "release_demo", "queue": {"location": "draining"}, "evidence_complete": True},
        task_name="release_demo",
    )
    rules = observation(
        "search_knowledge",
        {"results": [{"content": "soft preemption waits for a Stage boundary"}]},
        query="soft preemption",
    )
    result = evaluate_goal_progress(
        goal,
        records_for(diagnosis, rules),
        [diagnosis, rules],
        goal_contract=contract,
    )

    assert result.state == GoalProgress.SATISFIED


def test_platform_health_does_not_substitute_for_live_gpu_evidence():
    goal = AgentGoal(goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES)
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.GPU_DIAGNOSIS)
    health = observation("get_platform_health", {"status": "healthy"})
    rules = observation(
        "search_knowledge",
        {"results": [{"content": "exclusive reservations block sharing"}]},
        query="exclusive reservation",
    )
    result = evaluate_goal_progress(
        goal,
        records_for(health, rules),
        [health, rules],
        goal_contract=contract,
    )

    assert result.state == GoalProgress.IN_PROGRESS
    assert result.missing_conditions == [EvidenceType.LIVE_GPU.value]


def test_gpu_evidence_and_rules_satisfy_gpu_hybrid_contract():
    goal = AgentGoal(goal_type=GoalType.EXPLAIN_WITH_PLATFORM_RULES)
    contract = resolve_goal_contract(goal.goal_type, AgentIntent.GPU_DIAGNOSIS)
    gpu = observation("get_gpu_pool", {"devices": [{"gpu_id": "0", "free_mb": 0}]})
    rules = observation(
        "search_knowledge",
        {"results": [{"content": "exclusive reservations block sharing"}]},
        query="exclusive reservation",
    )
    result = evaluate_goal_progress(
        goal,
        records_for(gpu, rules),
        [gpu, rules],
        goal_contract=contract,
    )

    assert result.state == GoalProgress.SATISFIED


def test_frozen_contract_survives_adaptive_intent_revision():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo").model_copy(
        update={"goal_type": GoalType.EXPLAIN_WITH_PLATFORM_RULES}
    )
    frozen = resolve_goal_contract(goal.goal_type, AgentIntent.TASK_DIAGNOSIS)
    static = observation(
        "search_knowledge",
        {"results": [{"content": "soft preemption rule"}]},
        query="soft preemption",
    )
    revised_intent_result = evaluate_goal_progress(
        goal,
        records_for(static),
        [static],
        current_intent=AgentIntent.PLATFORM_KNOWLEDGE,
        goal_contract=frozen,
    )

    assert frozen.required_conditions == ["DIAGNOSTIC_CONTEXT", "STATIC_KNOWLEDGE"]
    assert revised_intent_result.state == GoalProgress.IN_PROGRESS
    assert revised_intent_result.missing_conditions == ["DIAGNOSTIC_CONTEXT"]


def test_production_diagnose_context_requires_a_real_fact_field():
    goal = goal_for_intent(AgentIntent.TASK_DIAGNOSIS, target="release_demo")
    state_only = observation(
        "diagnose_task",
        {"task_name": "release_demo", "evidence_complete": True},
        task_name="release_demo",
    )

    result = evaluate_goal_progress(goal, records_for(state_only), [state_only])

    # ``task_name`` and ``evidence_complete`` are metadata only.  The
    # production-shaped contract requires at least one fact-bearing field.
    assert result.state == GoalProgress.IN_PROGRESS
    assert EvidenceType.DIAGNOSTIC_CONTEXT.value in result.missing_conditions
