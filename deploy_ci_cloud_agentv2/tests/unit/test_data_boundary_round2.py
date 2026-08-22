from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceState,
    EvidenceTracker,
    ToolObservation,
)
from deploy_ci_cloud_agentv2.agent.gate import ResponseCompletionGate
from deploy_ci_cloud_agentv2.agent.goals import (
    DiagnoseTask,
    ExplainKnowledge,
    GoalDescriptor,
    InspectQueue,
    ReadTaskState,
)
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome, GoalStatus
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles
from deploy_ci_cloud_agentv2.agent.provenance import build_provenance
from deploy_ci_cloud_agentv2.agent.results import (
    DiagnosticResult,
    GpuPoolResult,
    KnowledgeResult,
    ResultStatus,
    TaskDetailResult,
    TaskState,
    normalize_read_result,
)
from deploy_ci_cloud_agentv2.agent.runtime import build_system_context, invoke
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers.deterministic import DeterministicReadAgent
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.runtime import ReadToolRuntime


def _observation(tool, arguments, payload, *, observation_id="obs-round2", status="SUCCESS"):
    result = normalize_read_result(tool, arguments, payload) if status == "SUCCESS" else None
    return ToolObservation(
        observation_id=observation_id,
        call_id=f"call-{observation_id}",
        source=tool,
        target=(
            str(arguments.get("task_name"))
            if arguments.get("task_name") is not None
            else str(arguments.get("query", "platform"))
        ),
        status=status,
        data=payload,
        observed_at=datetime.now(timezone.utc),
        provenance=build_provenance(tool, arguments, result),
        result=result,
    )


def test_global_queue_cannot_accept_task_scoped_response():
    observation = _observation(
        "get_queue_state",
        {"task_name": None},
        {"task_name": "task_A", "position": 1},
    )
    assert observation.provenance.scope_status.value == "CONFLICT"
    state, created = EvidenceTracker().record_observations(EvidenceState(), [observation])
    assert created == ()
    assert state.records == ()


def test_task_queue_cannot_accept_global_response_or_wrong_identity():
    tracker = EvidenceTracker()
    global_response = _observation(
        "get_queue_state",
        {"task_name": "task_A"},
        {"scope": "PLATFORM", "queue": []},
        observation_id="global",
    )
    wrong_task = _observation(
        "get_queue_state",
        {"task_name": "task_A"},
        {"task_name": "task_B", "position": 1},
        observation_id="wrong",
    )
    state, created = tracker.record_observations(EvidenceState(), [global_response, wrong_task])
    assert created == ()
    assert state.records == ()


def test_error_envelope_and_unknown_task_state_fail_closed():
    tracker = EvidenceTracker()
    error = _observation(
        "get_task_detail",
        {"task_name": "A"},
        {"status": "ERROR", "task_name": "A", "state": "RUNNING"},
        observation_id="error",
    )
    banana = _observation(
        "get_task_detail",
        {"task_name": "A"},
        {"task_name": "A", "state": "BANANA"},
        observation_id="banana",
    )
    assert error.result is not None and error.result.envelope.status is ResultStatus.ERROR
    assert isinstance(banana.result, TaskDetailResult)
    assert banana.result.validation_errors
    assert tracker.record_observations(EvidenceState(), [error, banana])[1] == ()


@pytest.mark.parametrize(
    "payload",
    [
        {"task_name": "A", "facts": {"task_name": "A"}},
        {"task_name": "A", "facts": {"task_name": "A", "timestamp": "now", "dataset": "d"}},
        {"task_name": "A", "diagnosis": None},
        {"task_name": "A", "diagnosis": "no diagnostic facts"},
    ],
)
def test_metadata_only_or_empty_diagnosis_is_not_evidence(payload):
    observation = _observation("diagnose_task", {"task_name": "A"}, payload)
    assert isinstance(observation.result, DiagnosticResult)
    assert EvidenceTracker().record_observations(EvidenceState(), [observation])[1] == ()


def test_title_or_url_only_knowledge_is_not_evidence_but_content_is():
    tracker = EvidenceTracker()
    title_only = _observation(
        "search_knowledge",
        {"query": "task_exclusive", "top_k": 5},
        {"query": "task_exclusive", "results": [{"title": "task_exclusive"}]},
        observation_id="title",
    )
    url_only = _observation(
        "search_knowledge",
        {"query": "task_exclusive", "top_k": 5},
        {"query": "task_exclusive", "results": [{"url": "https://example.invalid"}]},
        observation_id="url",
    )
    content = _observation(
        "search_knowledge",
        {"query": "task_exclusive", "top_k": 5},
        {"query": "task_exclusive", "results": [{"title": "x", "content": "meaningful explanation"}]},
        observation_id="content",
    )
    state, created = tracker.record_observations(EvidenceState(), [title_only, url_only, content])
    assert [record.kind for record in created] == [EvidenceKind.KNOWLEDGE]
    assert state.records[0].target == "task_exclusive"


def test_context_projection_keeps_canonical_history_and_required_current_evidence():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    base = _observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "task_A", "state": "RUNNING"},
        observation_id="current",
    )
    provenance = base.provenance
    assert provenance is not None
    now = datetime.now(timezone.utc)
    records = tuple(
        EvidenceRecord(
            evidence_id=f"ev-{index}",
            kind=EvidenceKind.LIVE_TASK,
            target="task_A",
            observation_id=f"obs-{index}",
            provenance=provenance,
            observed_at=now + timedelta(seconds=index),
        )
        for index in range(10_000)
    )
    evidence = EvidenceState(records)
    snapshot = load_operating_principles(
        Path(__file__).resolve().parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md"
    )
    state = {
        "request_id": "r",
        "thread_id": "t",
        "user_input": "task_A",
        "messages": [{"role": "user", "content": "x" * 100_000}],
        "goal_descriptor": descriptor,
        "completion_contract": contract,
        "goal_outcomes": {"g1": GoalOutcome("g1", GoalStatus.SATISFIED)},
        "evidence": evidence,
        "budgets": BudgetState(RuntimeBudgets(max_context_tokens=2_000)),
        "observations": (),
        "gate_feedback": (),
    }
    context = ContextBuilder().build(state, snapshot)
    assert len(evidence.records) == 10_000
    assert context.runtime_structured.evidence.total_records == 10_000
    assert len(context.runtime_structured.evidence.records) <= 64
    assert any(
        record.kind is EvidenceKind.LIVE_TASK and record.target == "task_A"
        for record in context.runtime_structured.evidence.records
    )
    assert context.estimated_context_chars <= 2_000 * 4
    assert len(context.messages[0]["content"]) < 100_000


def test_goal_and_tool_input_semantics_are_rejected_early():
    with pytest.raises(ValueError):
        ReadTaskState("g", "")
    with pytest.raises(ValueError):
        DiagnoseTask("g", "   ")
    with pytest.raises(ValueError):
        ExplainKnowledge("g", "")
    assert InspectQueue("g", None).target is None

    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "get_task_detail", {"task_name": ""}))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "search_knowledge", {"query": "", "top_k": 5}))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "search_knowledge", {"query": "abc", "top_k": -1}))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("x", "get_queue_state", {"task_name": "   "}))
    assert runtime.validate_single(ToolCall("x", "get_queue_state", {"task_name": None}))


def test_provider_reasons_from_qualified_projection_not_transport_success():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    invalid = _observation(
        "get_task_detail",
        {"task_name": "task_A"},
        {"task_name": "task_A", "state": "BANANA"},
    )
    snapshot = load_operating_principles(
        Path(__file__).resolve().parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md"
    )
    state = {
        "request_id": "r",
        "thread_id": "t",
        "user_input": "task_A",
        "messages": [],
        "goal_descriptor": descriptor,
        "completion_contract": contract,
        "goal_outcomes": {"g1": GoalOutcome("g1")},
        "evidence": EvidenceState(),
        "budgets": BudgetState(RuntimeBudgets()),
        "observations": (replace(invalid, status="SUCCESS"),),
        "gate_feedback": (),
    }
    context = ContextBuilder().build(state, snapshot)
    decision = __import__("asyncio").run(DeterministicReadAgent().generate(context))
    assert isinstance(decision, SingleToolCall)
    assert decision.call.tool_name == "get_task_detail"


class _FailAfterFirstRead(ContextBuilder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def build(self, state, snapshot):
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("simulated internal failure after successful READ")
        return super().build(state, snapshot)


def test_exception_path_keeps_read_projection_and_terminal_event_consistent():
    context = build_system_context(
        provider=DeterministicReadAgent(),
        context_builder=_FailAfterFirstRead(),
        read_facade=InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
        ),
    )
    result = __import__("asyncio").run(invoke("task_A status", thread_id="exception-path", system_context=context))
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.state is not None and result.state["evidence"].records
    checkpoint = context.checkpointer.load("exception-path")
    assert checkpoint is not None
    events = context.event_store.for_thread("exception-path")
    assert checkpoint["last_event_id"] == events[-1].event_id
    assert result.state["last_event_id"] == events[-1].event_id
