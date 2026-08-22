from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.decisions import SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.evidence import (
    EvidenceFreshness,
    EvidenceFreshnessPolicy,
    EvidenceKind,
    EvidenceProjectionBuilder,
    EvidenceRecord,
    EvidenceState,
    EvidenceTracker,
)
from deploy_ci_cloud_agentv2.agent.goals import DiagnoseTask, ExplainKnowledge, GoalDescriptor, InspectQueue, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles
from deploy_ci_cloud_agentv2.agent.provenance import build_provenance
from deploy_ci_cloud_agentv2.agent.results import DiagnosticResult, KnowledgeResult, TaskDetailResult
from deploy_ci_cloud_agentv2.agent.runtime import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.state import CurrentRequestContext, ThreadHistory
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers.deterministic import DeterministicReadAgent
from deploy_ci_cloud_agentv2.tests.helpers import identity, observation
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.runtime import ReadToolRuntime


def test_scope_provenance_rejects_global_task_confusion():
    owner = identity()
    tracker = EvidenceTracker()
    items = [
        observation("get_queue_state", {"task_name": None}, {"task_name": "task_A", "position": 1}, owner=owner, observation_id="global-task"),
        observation("get_queue_state", {"task_name": "task_A"}, {"scope": "PLATFORM", "queue": []}, owner=owner, observation_id="task-global"),
        observation("get_queue_state", {"task_name": "task_A"}, {"task_name": "task_B", "position": 1}, owner=owner, observation_id="wrong-task"),
    ]
    state, created = tracker.record_observations(EvidenceState(owner), items, owner)
    assert state.records == ()
    assert created == ()


@pytest.mark.parametrize(
    "payload",
    [
        {"task_name": "A", "facts": {"task_name": "A"}},
        {"task_name": "A", "facts": {"task_name": "A", "timestamp": "now", "dataset": "d"}},
        {"task_name": "A", "diagnosis": None},
        {"task_name": "A", "diagnosis": "no diagnostic facts"},
    ],
)
def test_diagnostic_metadata_only_never_qualifies(payload):
    owner = identity()
    item = observation("diagnose_task", {"task_name": "A"}, payload, owner=owner)
    assert isinstance(item.result, DiagnosticResult)
    assert EvidenceTracker().record_observations(EvidenceState(owner), [item], owner)[1] == ()


def test_title_url_only_knowledge_is_not_evidence():
    owner = identity()
    items = [
        observation("search_knowledge", {"query": "x"}, {"query": "x", "results": [{"title": "x"}]}, owner=owner, observation_id="title"),
        observation("search_knowledge", {"query": "x"}, {"query": "x", "results": [{"url": "https://example.invalid"}]}, owner=owner, observation_id="url"),
    ]
    assert EvidenceTracker().record_observations(EvidenceState(owner), items, owner)[1] == ()


def test_typed_result_contracts_distinguish_malformed_and_absent():
    owner = identity()
    malformed = observation("get_task_detail", {"task_name": "A"}, {"task_name": "A", "state": "BANANA"}, owner=owner, observation_id="malformed")
    absent = observation("get_task_detail", {"task_name": "A"}, {"status": "NO_DATA", "task_name": "A"}, owner=owner, observation_id="absent")
    assert isinstance(malformed.result, TaskDetailResult)
    assert malformed.disposition.value == "MALFORMED"
    assert absent.disposition.value == "ABSENT"


def test_canonical_evidence_is_complete_but_projection_is_bounded():
    owner = identity()
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    base = observation("get_task_detail", {"task_name": "task_A"}, {"task_name": "task_A", "state": "RUNNING"}, owner=owner)
    assert base.provenance is not None
    now = datetime.now(timezone.utc)
    records = tuple(
        EvidenceRecord(
            evidence_id=f"ev-{i}", kind=EvidenceKind.LIVE_TASK, target="task_A",
            observation_id=f"obs-{i}", owner=owner, provenance=base.provenance,
            freshness=EvidenceFreshness(now + timedelta(seconds=i), now + timedelta(hours=1)),
        )
        for i in range(10_000)
    )
    evidence = EvidenceState(owner, records)
    projection = EvidenceProjectionBuilder().build(evidence, descriptor, contract, max_records=4, max_chars=2_000)
    assert len(evidence.records) == 10_000
    assert len(projection.records) <= 4
    assert projection.total_records == 10_000
    assert any(item.target == "task_A" for item in projection.records)


def test_context_budget_covers_current_state_and_projection():
    owner = identity()
    snapshot = load_operating_principles(Path(__file__).resolve().parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md")
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    current = CurrentRequestContext(
        identity=owner, user_input="task_A", messages=({"role": "user", "content": "x" * 100_000},),
        step_count=0, tool_call_count=0, goal_descriptor=descriptor,
        completion_contract=CompletionContractCompiler().compile(descriptor), goal_outcomes={"g1": GoalOutcome("g1")},
        evidence=EvidenceState(owner), observations=(), budgets=BudgetState(RuntimeBudgets(max_context_tokens=1_000)),
        terminal_state=None, termination_reason=None, operating_principles_snapshot=snapshot,
        decision=None, final_candidate=None, gate_feedback=(), gate_passed=None, new_turn=False,
        continue_after_read_guard=False,
    )
    context = ContextBuilder().build(current, ThreadHistory())
    assert context.runtime_structured.identity == owner
    assert context.estimated_context_chars <= 4_000
    assert len(context.messages[0]["content"]) < 100_000


def test_none_is_only_global_queue_argument():
    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()), identity())
    assert runtime.validate_single(ToolCall("global", "get_queue_state", {"task_name": None}))
    with pytest.raises(ValueError):
        runtime.validate_single(ToolCall("blank", "get_queue_state", {"task_name": "   "}))


def test_provider_uses_qualified_projection_not_transport_success():
    owner = identity()
    snapshot = load_operating_principles(Path(__file__).resolve().parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md")
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    invalid = observation("get_task_detail", {"task_name": "task_A"}, {"task_name": "task_A", "state": "BANANA"}, owner=owner)
    current = CurrentRequestContext(
        identity=owner, user_input="task_A", messages=(), step_count=0, tool_call_count=1,
        goal_descriptor=descriptor, completion_contract=CompletionContractCompiler().compile(descriptor), goal_outcomes={"g1": GoalOutcome("g1")},
        evidence=EvidenceState(owner), observations=(replace(invalid, transport_status=invalid.transport_status),),
        budgets=BudgetState(RuntimeBudgets()), terminal_state=None, termination_reason=None,
        operating_principles_snapshot=snapshot, decision=None, final_candidate=None, gate_feedback=(),
        gate_passed=None, new_turn=False, continue_after_read_guard=False,
    )
    context = ContextBuilder().build(current, ThreadHistory())
    decision = __import__("asyncio").run(DeterministicReadAgent().generate(context))
    assert isinstance(decision, SingleToolCall)
    assert decision.call.tool_name == "get_task_detail"


class FailAfterFirstContext(ContextBuilder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def build(self, current, history):
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("internal failure after successful READ")
        return super().build(current, history)


def test_exception_path_keeps_current_request_and_event_tail_consistent():
    context = build_system_context(
        provider=DeterministicReadAgent(), context_builder=FailAfterFirstContext(),
        read_facade=InMemoryReadFacade(responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}),
    )
    result = __import__("asyncio").run(invoke("task_A status", thread_id="exception-round", system_context=context))
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.state["current_request"].evidence.records
    checkpoint = context.checkpointer.load("exception-round")
    events = context.event_store.for_thread("exception-round")
    assert checkpoint["last_event_id"] == events[-1].event_id
    assert result.state["last_event_id"] == events[-1].event_id
