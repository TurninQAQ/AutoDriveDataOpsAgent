from datetime import datetime, timedelta, timezone

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate
from deploy_ci_cloud_agentv2.agent.evidence import (
    EvidenceState,
    EvidenceTracker,
    ToolObservation,
    build_observation_provenance,
)
from deploy_ci_cloud_agentv2.agent.provenance import build_provenance
from deploy_ci_cloud_agentv2.agent.results import normalize_read_result
from deploy_ci_cloud_agentv2.agent.gate import ResponseCompletionGate
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import GoalStatus
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles


def test_evidence_records_provenance_and_freshness():
    now = datetime.now(timezone.utc)
    result = normalize_read_result(
        "get_task_detail", {"task_name": "task_A"}, {"task_name": "task_A", "state": "failed"}
    )
    observation = ToolObservation(
        "obs1",
        "call1",
        "get_task_detail",
        "task_A",
        "SUCCESS",
        {"task_name": "task_A", "state": "failed"},
        observed_at=now,
        provenance=build_provenance("get_task_detail", {"task_name": "task_A"}, result),
        result=result,
    )
    state, records = EvidenceTracker(freshness_seconds=1).record_observations(
        EvidenceState(), [observation]
    )
    assert records[0].kind == "LIVE_TASK"
    assert records[0].provenance.source_tool == "get_task_detail"
    assert records[0].observation_id == "obs1"
    assert records[0].is_current(now)
    assert not records[0].is_current(now + timedelta(seconds=2))
    assert state.records[0].status == "VALID"


def test_premature_final_is_rejected_without_prescribing_a_tool():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    gate = ResponseCompletionGate()
    evaluation = gate.evaluate(
        FinalCandidate("done"), descriptor, contract, EvidenceState(), {}
    )
    assert evaluation.passed is False
    assert "LIVE_TASK evidence for task_A" in evaluation.missing
    assert all("get_task_detail" not in item for item in evaluation.missing)
    assert evaluation.goal_outcomes["g1"].status is GoalStatus.PENDING


def test_context_builder_keeps_three_contexts_distinct(tmp_path):
    principles_path = tmp_path / "principles.md"
    principles_path.write_text("## Principle P01 — Test\nAdvisory only.\n", encoding="utf-8")
    snapshot = load_operating_principles(principles_path)
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    state = {
        "request_id": "r",
        "thread_id": "t",
        "user_input": "task_A",
        "messages": [],
        "goal_descriptor": descriptor,
        "completion_contract": CompletionContractCompiler().compile(descriptor),
        "goal_outcomes": {},
        "evidence": EvidenceState(),
        "budgets": BudgetState(RuntimeBudgets()),
        "observations": (),
        "gate_feedback": (),
    }
    context = ContextBuilder().build(state, snapshot)
    assert context.runtime_structured.goal_descriptor == descriptor
    assert context.operating_guidance.version == snapshot.version
    assert context.semantic_observations.trust_label == "UNTRUSTED_EXTERNAL_DATA"
