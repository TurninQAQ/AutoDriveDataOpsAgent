from pathlib import Path

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, ToolCall
from deploy_ci_cloud_agentv2.agent.evidence import EvidenceKind, EvidenceState, EvidenceTracker
from deploy_ci_cloud_agentv2.agent.gate import ResponseCompletionGate
from deploy_ci_cloud_agentv2.agent.goals import ExplainKnowledge, GoalDescriptor, ReadTaskState, SubmitTask
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome, GoalStatus
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles
from deploy_ci_cloud_agentv2.agent.results import DiagnosticResult, TaskDetailResult
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.tests.helpers import identity, observation
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.runtime import ReadToolRuntime


def test_entity_mismatch_missing_identity_and_unknown_state_fail_closed():
    owner = identity()
    tracker = EvidenceTracker()
    observations = [
        observation(
            "get_task_detail",
            {"task_name": "task_A"},
            {"task_name": "other_task", "state": "RUNNING"},
            owner=owner,
            observation_id="mismatch",
        ),
        observation(
            "get_task_detail",
            {"task_name": "task_A"},
            {"state": "RUNNING"},
            owner=owner,
            observation_id="missing",
        ),
        observation(
            "get_task_detail",
            {"task_name": "task_A"},
            {"task_name": "task_A", "state": "UNKNOWN"},
            owner=owner,
            observation_id="unknown",
        ),
    ]
    state, records = tracker.record_observations(EvidenceState(owner), observations, owner)
    assert records == ()
    assert state.records == ()


def test_error_envelope_cannot_be_overridden_by_task_fields():
    owner = identity()
    item = observation(
        "get_task_detail",
        {"task_name": "A"},
        {"status": "ERROR", "task_name": "A", "state": "RUNNING"},
        owner=owner,
    )
    assert isinstance(item.result, TaskDetailResult)
    assert not item.result.is_valid
    assert item.disposition.value == "EXTERNAL_ERROR"


def test_empty_knowledge_and_diagnosis_are_normalized_but_not_evidence():
    owner = identity()
    items = [
        observation(
            "search_knowledge",
            {"query": "task_exclusive", "top_k": 5},
            {"query": "task_exclusive", "results": []},
            owner=owner,
            observation_id="knowledge-empty",
        ),
        observation(
            "diagnose_task",
            {"task_name": "task_A"},
            {"task_name": "task_A", "facts": {"task_name": "task_A"}},
            owner=owner,
            observation_id="diagnosis-empty",
        ),
    ]
    state, records = EvidenceTracker().record_observations(EvidenceState(owner), items, owner)
    assert all(item.disposition.value == "NORMALIZED_NO_QUALIFIED_EVIDENCE" for item in items)
    assert state.records == ()
    assert records == ()


def test_meaningful_knowledge_and_diagnosis_qualify():
    owner = identity()
    state, records = EvidenceTracker().record_observations(
        EvidenceState(owner),
        [
            observation(
                "search_knowledge",
                {"query": "task_exclusive"},
                {"query": "task_exclusive", "results": [{"content": "sharing rule"}]},
                owner=owner,
                observation_id="knowledge",
            ),
            observation(
                "diagnose_task",
                {"task_name": "task_A"},
                {"task_name": "task_A", "root_cause": "CUDA OOM"},
                owner=owner,
                observation_id="diagnosis",
            ),
        ],
        owner,
    )
    assert {record.kind.value for record in records} == {"KNOWLEDGE", "DIAGNOSTIC_CONTEXT"}
    assert {record.target for record in state.records} == {"task_exclusive", "task_A"}


def test_final_candidate_must_cover_all_goals():
    owner = identity()
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"), ExplainKnowledge("g2", "topic")))
    contract = CompletionContractCompiler().compile(descriptor)
    state, _ = EvidenceTracker().record_observations(
        EvidenceState(owner),
        [
            observation("get_task_detail", {"task_name": "task_A"}, {"task_name": "task_A", "state": "RUNNING"}, owner=owner, observation_id="task"),
            observation("search_knowledge", {"query": "topic"}, {"query": "topic", "results": [{"content": "meaning"}]}, owner=owner, observation_id="topic"),
        ],
        owner,
    )
    gate = ResponseCompletionGate()
    assert not gate.evaluate(FinalCandidate("one", referenced_goal_ids=("g1",)), descriptor, contract, state, {}).passed
    assert not gate.evaluate(FinalCandidate("unknown", referenced_goal_ids=("g1", "nope")), descriptor, contract, state, {}).passed
    assert not gate.evaluate(FinalCandidate("pending", referenced_goal_ids=("g1", "g2")), descriptor, contract, EvidenceState(owner), {}).passed


def test_submit_task_generated_identity_matches_only_its_deterministic_prefix():
    owner = identity()
    descriptor = GoalDescriptor(1, (SubmitTask("g1", "autodrive_v2_bootstrap", {}),))
    contract = CompletionContractCompiler().compile(descriptor)
    actual = "autodrive_v2_bootstrap_20260824_010203"
    tracker = EvidenceTracker()
    evidence, _ = tracker.record_verification(
        EvidenceState(owner),
        kind=EvidenceKind.ACTION_VERIFIED,
        target=actual,
        source="action_verifier",
        observation_id="generated-action",
        owner=owner,
    )
    evidence, _ = tracker.record_verification(
        evidence,
        kind=EvidenceKind.OPERATIONAL_GOAL_VERIFIED,
        target=actual,
        source="operational_goal_verifier",
        observation_id="generated-goal",
        owner=owner,
    )
    refreshed = tracker.refresh_goal_outcomes(
        descriptor,
        contract,
        evidence,
        {"g1": GoalOutcome("g1")},
    )
    evaluation = ResponseCompletionGate().evaluate(
        FinalCandidate("created", referenced_goal_ids=("g1",)),
        descriptor,
        contract,
        evidence,
        refreshed,
    )
    assert evaluation.passed
    assert evaluation.goal_outcomes["g1"].status is GoalStatus.SATISFIED


def test_submit_task_does_not_accept_arbitrary_observed_identity():
    owner = identity()
    descriptor = GoalDescriptor(1, (SubmitTask("g1", "autodrive_v2_bootstrap", {}),))
    contract = CompletionContractCompiler().compile(descriptor)
    evidence, _ = EvidenceTracker().record_verification(
        EvidenceState(owner),
        kind=EvidenceKind.ACTION_VERIFIED,
        target="unrelated_task_20260824_010203",
        source="action_verifier",
        observation_id="unrelated-action",
        owner=owner,
    )
    evaluation = ResponseCompletionGate().evaluate(
        FinalCandidate("created", referenced_goal_ids=("g1",)),
        descriptor,
        contract,
        evidence,
        {"g1": GoalOutcome("g1")},
    )
    assert not evaluation.passed


def test_operating_principles_parser_has_clean_p01_to_p18():
    snapshot = load_operating_principles(
        Path(__file__).resolve().parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md"
    )
    assert len(snapshot.principles) == 18
    assert all(section not in snapshot.principles[-1].text for section in ("Runtime Integration", "Anti-Drift Rules", "Final Operating Rule"))


def test_goal_and_tool_arguments_are_structurally_rejected():
    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()), identity())
    import pytest
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "get_task_detail", {"task_name": ""}))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "search_knowledge", {"query": "", "top_k": 5}))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "search_knowledge", {"query": "abc", "top_k": -1}))
