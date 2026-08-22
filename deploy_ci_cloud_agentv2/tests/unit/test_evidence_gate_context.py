from datetime import timedelta
from dataclasses import replace
from pathlib import Path

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate
from deploy_ci_cloud_agentv2.agent.evidence import (
    EvidenceFreshnessPolicy,
    EvidenceState,
    EvidenceTracker,
)
from deploy_ci_cloud_agentv2.agent.gate import ResponseCompletionGate
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome, GoalStatus
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles
from deploy_ci_cloud_agentv2.agent.state import CurrentRequestContext, ThreadHistory, new_state
from deploy_ci_cloud_agentv2.tests.helpers import identity, observation


def test_evidence_has_explicit_owner_and_freshness():
    owner = identity()
    item = observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "task_A", "state": "FAILED"},
        owner=owner,
    )
    state, records = EvidenceTracker(
        EvidenceFreshnessPolicy(default_ttl=timedelta(seconds=1))
    ).record_observations(EvidenceState(owner), [item], owner)
    assert records[0].owner == owner
    assert records[0].freshness.observed_at == item.observed_at
    assert records[0].is_current(item.observed_at)
    assert not records[0].is_current(item.observed_at + timedelta(seconds=2))
    assert state.owner == owner


def test_tracker_rejects_cross_request_observations():
    owner = identity()
    other = identity(request_id="other")
    item = observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "task_A", "state": "RUNNING"},
        owner=other,
    )
    state, created = EvidenceTracker().record_observations(EvidenceState(owner), [item], owner)
    assert state.records == ()
    assert created == ()


def test_premature_final_is_rejected_without_prescribing_a_tool():
    owner = identity()
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    evaluation = ResponseCompletionGate().evaluate(
        FinalCandidate("done"),
        descriptor,
        contract,
        EvidenceState(owner),
        {},
    )
    assert not evaluation.passed
    assert "LIVE_TASK evidence for task_A" in evaluation.missing
    assert all("get_task_detail" not in item for item in evaluation.missing)
    assert evaluation.goal_outcomes["g1"].status is GoalStatus.PENDING


def test_context_projection_preserves_current_structured_state_and_is_bounded(tmp_path):
    principles = tmp_path / "principles.md"
    principles.write_text("## Principle P01 — Test\nAdvisory only.\n", encoding="utf-8")
    snapshot = load_operating_principles(principles)
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    owner = identity()
    current = CurrentRequestContext(
        identity=owner,
        user_input="task_A",
        messages=({"role": "user", "content": "x" * 100_000},),
        step_count=0,
        tool_call_count=0,
        goal_descriptor=descriptor,
        completion_contract=contract,
        goal_outcomes={"g1": GoalOutcome("g1")},
        evidence=EvidenceState(owner),
        observations=(),
        budgets=BudgetState(RuntimeBudgets(max_context_tokens=1_000)),
        terminal_state=None,
        termination_reason=None,
        operating_principles_snapshot=snapshot,
        decision=None,
        final_candidate=None,
        gate_feedback=(),
        gate_passed=None,
        new_turn=False,
        continue_after_read_guard=False,
    )
    context = ContextBuilder().build(current, ThreadHistory())
    assert context.runtime_structured.identity == owner
    assert context.runtime_structured.goal_descriptor == descriptor
    assert sum(len(item.get("content", "")) for item in context.messages) < 100_000
    assert context.estimated_context_chars <= 4_000


def test_new_turn_has_fresh_current_context_and_history_only_keeps_refs(tmp_path):
    principles = tmp_path / "principles.md"
    principles.write_text("## Principle P01 — Test\nAdvisory only.\n", encoding="utf-8")
    snapshot = load_operating_principles(principles)
    first = new_state(
        user_input="task_A status",
        thread_id="thread-ownership",
        snapshot=snapshot,
        budgets=RuntimeBudgets(),
    )
    first_current = first["current_request"]
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    item = observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "task_A", "state": "RUNNING"},
        owner=first_current.identity,
        observation_id="first-observation",
    )
    evidence, records = EvidenceTracker().record_observations(
        first_current.evidence, [item], first_current.identity
    )
    completed_current = replace(
        first_current,
        goal_descriptor=descriptor,
        completion_contract=contract,
        goal_outcomes={
            "g1": GoalOutcome("g1", GoalStatus.SATISFIED, evidence_refs=(records[0].evidence_id,))
        },
        evidence=evidence,
        final_candidate=FinalCandidate("done", referenced_goal_ids=("g1",)),
        gate_passed=True,
    )
    prior = {**first, "current_request": completed_current}

    second = new_state(
        user_input="what is task_B doing?",
        thread_id="thread-ownership",
        snapshot=snapshot,
        budgets=RuntimeBudgets(),
        prior=prior,
    )
    current = second["current_request"]
    assert current.identity.thread_id == first_current.identity.thread_id
    assert current.identity.request_id != first_current.identity.request_id
    assert current.identity.turn_id != first_current.identity.turn_id
    assert current.goal_descriptor is None
    assert current.completion_contract is None
    assert current.goal_outcomes == {}
    assert current.evidence.records == ()
    assert len(second["thread_history"].requests) == 1
    assert second["thread_history"].requests[0].evidence_refs == (records[0].evidence_id,)


def test_context_projection_keeps_latest_observations_only(tmp_path):
    principles = tmp_path / "principles.md"
    principles.write_text("## Principle P01 — Test\nAdvisory only.\n", encoding="utf-8")
    snapshot = load_operating_principles(principles)
    owner = identity()
    observations = tuple(
        observation(
            "get_task_detail",
            {"task_name": "task_A"},
            {"task_name": "task_A", "state": "RUNNING", "seq": index},
            owner=owner,
            observation_id=f"obs-{index}",
        )
        for index in range(33)
    )
    current = CurrentRequestContext(
        identity=owner,
        user_input="task_A",
        messages=(),
        step_count=0,
        tool_call_count=33,
        goal_descriptor=None,
        completion_contract=None,
        goal_outcomes={},
        evidence=EvidenceState(owner),
        observations=observations,
        budgets=BudgetState(RuntimeBudgets(max_context_tokens=2_000)),
        terminal_state=None,
        termination_reason=None,
        operating_principles_snapshot=snapshot,
        decision=None,
        final_candidate=None,
        gate_feedback=(),
        gate_passed=None,
        new_turn=False,
        continue_after_read_guard=False,
    )
    context = ContextBuilder(max_observations=32).build(current, ThreadHistory())
    ids = {item["observation_id"] for item in context.semantic_observations.observations}
    assert "obs-32" in ids
    assert "obs-0" not in ids
