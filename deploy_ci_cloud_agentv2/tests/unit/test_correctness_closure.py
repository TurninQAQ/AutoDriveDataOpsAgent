from datetime import datetime, timezone

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, ToolCall
from deploy_ci_cloud_agentv2.agent.evidence import (
    EvidenceState,
    EvidenceTracker,
    IdentityStatus,
    ToolObservation,
    build_observation_provenance,
)
from deploy_ci_cloud_agentv2.agent.gate import ResponseCompletionGate
from deploy_ci_cloud_agentv2.agent.goals import (
    ExplainKnowledge,
    GoalDescriptor,
    ReadTaskState,
)
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome, GoalStatus
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.runtime import ReadToolRuntime


def _observation(source, arguments, data, *, status="SUCCESS"):
    return ToolObservation(
        observation_id=f"obs-{source}",
        call_id=f"call-{source}",
        source=source,
        target=str(arguments.get("task_name", arguments.get("query", "platform"))) or "platform",
        status=status,
        data=data,
        observed_at=datetime.now(timezone.utc),
        provenance=build_observation_provenance(source, arguments, data),
    )


def test_entity_mismatch_and_missing_identity_fail_closed():
    tracker = EvidenceTracker()
    mismatch = _observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "other_task", "state": "RUNNING"},
    )
    missing = _observation(
        "get_task_detail", {"task_name": "task_A"}, {"state": "RUNNING"}
    )
    assert mismatch.provenance.identity_status is IdentityStatus.CONFLICT
    assert missing.provenance.identity_status is IdentityStatus.MISSING
    state, records = tracker.record_observations(EvidenceState(), [mismatch, missing])
    assert records == ()
    assert state.records == ()


def test_unknown_placeholder_and_default_facade_are_not_live_task_evidence():
    tracker = EvidenceTracker()
    unknown = _observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "task_A", "state": "UNKNOWN"},
    )
    state, records = tracker.record_observations(EvidenceState(), [unknown])
    assert records == ()
    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()))
    import asyncio

    observation = asyncio.run(
        runtime.execute_single(
            ToolCall("task", "get_task_detail", {"task_name": "task_A"}),
            max_retries=0,
        )
    )
    assert observation.status == "SUCCESS"
    assert tracker.record_observations(EvidenceState(), [observation])[1] == ()


def test_knowledge_and_diagnostic_qualification_requires_meaningful_bound_payloads():
    tracker = EvidenceTracker()
    observations = [
        _observation(
            "search_knowledge",
            {"query": "task_exclusive"},
            {"query": "task_exclusive", "results": []},
        ),
        _observation(
            "diagnose_task",
            {"task_name": "task_A"},
            {"task_name": "task_A", "diagnosis": "no diagnostic facts"},
        ),
        *[
            _observation(
                "diagnose_task",
                {"task_name": "task_A"},
                {"task_name": "task_A", "diagnosis": value},
            )
            for value in (None, "", [], "no diagnostic facts")
        ],
        _observation(
            "diagnose_task",
            {"task_name": "task_A"},
            {"task_name": "task_B", "root_cause": "CUDA OOM"},
        ),
    ]
    state, records = tracker.record_observations(EvidenceState(), observations)
    assert state.records == ()
    assert records == ()


def test_meaningful_knowledge_and_diagnosis_can_qualify():
    tracker = EvidenceTracker()
    state, records = tracker.record_observations(
        EvidenceState(),
        [
            _observation(
                "search_knowledge",
                {"query": "task_exclusive"},
                {"query": "task_exclusive", "results": [{"content": "sharing rule"}]},
            ),
            _observation(
                "diagnose_task",
                {"task_name": "task_A"},
                {"task_name": "task_A", "root_cause": "CUDA OOM"},
            ),
        ],
    )
    assert {record.kind for record in records} == {"KNOWLEDGE", "DIAGNOSTIC_CONTEXT"}
    assert {record.target for record in state.records} == {"task_exclusive", "task_A"}


def test_resource_and_queue_evidence_require_explicit_payloads():
    tracker = EvidenceTracker()
    state, records = tracker.record_observations(
        EvidenceState(),
        [
            _observation(
                "get_gpu_pool",
                {},
                {"devices": [{"gpu_id": "0", "free_mb": 1024}]},
            ),
            _observation(
                "get_queue_state",
                {"task_name": "task_A"},
                {"task_name": "task_A", "position": 2},
            ),
        ],
    )
    assert {record.kind for record in records} == {"GPU_POOL", "QUEUE_STATE"}
    assert {record.target for record in state.records} == {"platform", "task_A"}

    state, records = tracker.record_observations(
        EvidenceState(),
        [
            _observation("get_gpu_pool", {}, {"status": "NO_DATA", "devices": []}),
            _observation(
                "get_queue_state",
                {"task_name": "task_A"},
                {"task_name": "task_A", "queue": []},
            ),
        ],
    )
    assert [record.kind for record in records] == ["QUEUE_STATE"]


def test_completion_gate_requires_all_known_goals_and_candidate_references():
    descriptor = GoalDescriptor(
        1,
        (ReadTaskState("g1", "task_A"), ExplainKnowledge("g2", "topic")),
    )
    contract = CompletionContractCompiler().compile(descriptor)
    tracker = EvidenceTracker()
    evidence, _ = tracker.record_observations(
        EvidenceState(),
        [
            _observation(
                "get_task_detail",
                {"task_name": "task_A"},
                {"task_name": "task_A", "state": "RUNNING"},
            ),
            _observation(
                "search_knowledge",
                {"query": "topic"},
                {"query": "topic", "results": [{"content": "meaning"}]},
            ),
        ],
    )
    gate = ResponseCompletionGate()
    outcomes = {
        "g1": GoalOutcome("g1", GoalStatus.SATISFIED),
        "g2": GoalOutcome("g2", GoalStatus.SATISFIED),
    }
    missing_goal = gate.evaluate(
        FinalCandidate("only one", referenced_goal_ids=("g1",)),
        descriptor,
        contract,
        evidence,
        outcomes,
    )
    assert not missing_goal.passed
    assert any("g2" in item for item in missing_goal.missing)

    unknown_goal = gate.evaluate(
        FinalCandidate("unknown", referenced_goal_ids=("g1", "nonexistent")),
        descriptor,
        contract,
        evidence,
        outcomes,
    )
    assert not unknown_goal.passed
    assert any("unknown" in item for item in unknown_goal.missing)

    pending = gate.evaluate(
        FinalCandidate("pending", referenced_goal_ids=("g1", "g2")),
        descriptor,
        contract,
        EvidenceState(),
        {},
    )
    assert not pending.passed
    assert pending.goal_outcomes["g1"].status is GoalStatus.PENDING

    meaningless = gate.evaluate(
        FinalCandidate("meaningless", referenced_goal_ids=()),
        descriptor,
        contract,
        evidence,
        outcomes,
    )
    assert not meaningless.passed


def test_principles_parser_stops_at_top_level_sections():
    snapshot = load_operating_principles(
        "/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agentv2/doc/Luna_OPERATING_PRINCIPLES.md"
    )
    assert len(snapshot.principles) == 18
    assert snapshot.principles[-1].principle_id == "P18"
    assert all(
        section not in snapshot.principles[-1].text
        for section in ("Runtime Integration", "Anti-Drift Rules", "Final Operating Rule")
    )


def test_context_builder_keeps_latest_observations_and_bounds_messages():
    snapshot = load_operating_principles(
        "/home/ubuntu/project/AutoDriveDataOpsAgent/deploy_ci_cloud_agentv2/doc/Luna_OPERATING_PRINCIPLES.md"
    )
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    state = {
        "request_id": "r",
        "thread_id": "t",
        "user_input": "task_A",
        "messages": [{"role": "user", "content": "x" * 100_000}],
        "goal_descriptor": descriptor,
        "completion_contract": CompletionContractCompiler().compile(descriptor),
        "goal_outcomes": {"g1": GoalOutcome("g1")},
        "evidence": EvidenceState(),
        "budgets": BudgetState(RuntimeBudgets(max_context_tokens=200)),
        "observations": tuple(
            _observation("read_guard", {}, {"sequence": index})
            for index in range(33)
        ),
        "gate_feedback": (),
    }
    context = ContextBuilder(max_observations=32).build(state, snapshot)
    assert len(context.semantic_observations.observations) == 32
    assert context.semantic_observations.observations[-1].data == {"sequence": 32}
    assert context.semantic_observations.observations[0].data == {"sequence": 1}
    assert sum(len(str(item.get("content", ""))) for item in context.messages) <= 800
    assert context.runtime_structured.goal_descriptor == descriptor
    assert context.runtime_structured.completion_contract is not None
