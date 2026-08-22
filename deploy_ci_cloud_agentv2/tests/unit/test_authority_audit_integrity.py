import asyncio

import pytest

from deploy_ci_cloud_agentv2 import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.events import EventIntegrityError, EventProvenance, EventStore
from deploy_ci_cloud_agentv2.agent.evidence import EvidenceState, EvidenceTracker
from deploy_ci_cloud_agentv2.agent.goals import DiagnoseTask, GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome, GoalStatus
from deploy_ci_cloud_agentv2.agent.results import ResultStatus, normalize_read_result
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers.model import AgentProvider
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.tests.helpers import identity, observation


def _provenance() -> EventProvenance:
    return EventProvenance("model", "prompt", "catalog", "principles", "hash")


def test_event_store_snapshots_nested_input_and_all_read_projections():
    store = EventStore()
    payload = {"nested": {"items": [1], "mapping": {"x": "before"}}, "tags": ["a"]}
    store.append(
        event_type="ToolObservationRecorded",
        request_id="r",
        thread_id="t",
        payload=payload,
        provenance=_provenance(),
        event_id="e1",
    )
    payload["nested"]["items"].append(2)
    payload["nested"]["mapping"]["x"] = "after"

    returned = store.all()[0]
    returned.payload["nested"]["items"].append(3)
    returned.payload["nested"]["mapping"]["x"] = "returned-mutation"
    trace = store.readable_trace("t")
    trace[0]["payload"]["nested"]["items"].append(4)
    thread = store.for_thread("t")
    thread[0].payload["nested"]["items"].append(5)

    assert store.all()[0].payload == {
        "nested": {"items": [1], "mapping": {"x": "before"}},
        "tags": ["a"],
    }
    assert store.for_thread("other") == ()


def test_event_store_duplicate_id_is_idempotent_only_for_identical_content():
    store = EventStore()
    first = store.append(
        event_type="AgentRunStarted",
        request_id="r",
        thread_id="t",
        payload={"value": {"a": 1}},
        provenance=_provenance(),
        event_id="stable",
    )
    replay = store.append(
        event_type="AgentRunStarted",
        request_id="r",
        thread_id="t",
        payload={"value": {"a": 1}},
        provenance=_provenance(),
        event_id="stable",
    )
    assert replay == first
    with pytest.raises(EventIntegrityError):
        store.append(
            event_type="DifferentEvent",
            request_id="r",
            thread_id="t",
            payload={"value": {"a": 1}},
            provenance=_provenance(),
            event_id="stable",
        )
    with pytest.raises(EventIntegrityError):
        store.append(
            event_type="AgentRunStarted",
            request_id="r",
            thread_id="t",
            payload={"value": {"a": 2}},
            provenance=_provenance(),
            event_id="stable",
        )


@pytest.mark.parametrize(
    "arguments, payload",
    [
        ({"task_name": "A"}, {"scope": "TASK", "task_name": "A", "state": "UNKNOWN_EXTERNAL_STATE"}),
        ({}, {"scope": "PLATFORM", "state": "UNKNOWN_EXTERNAL_STATE"}),
    ],
)
def test_unknown_only_queue_state_does_not_qualify(arguments, payload):
    result = normalize_read_result("get_queue_state", arguments, payload)
    assert not result.qualifies_for_evidence()


@pytest.mark.parametrize(
    "arguments, payload",
    [
        (
            {"task_name": "A"},
            {"scope": "TASK", "task_name": "A", "state": "UNKNOWN_EXTERNAL_STATE", "position": 4},
        ),
        (
            {},
            {"scope": "PLATFORM", "state": "UNKNOWN_EXTERNAL_STATE", "queue": [{"task_name": "A", "position": 1}]},
        ),
    ],
)
def test_unknown_queue_state_with_independent_fact_can_qualify(arguments, payload):
    result = normalize_read_result("get_queue_state", arguments, payload)
    assert result.qualifies_for_evidence()


def test_first_satisfied_outcome_always_contains_supporting_refs():
    owner = identity()
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    task = observation(
        "get_task_detail", {"task_name": "A"}, {"task_name": "A", "state": "RUNNING"},
        owner=owner, observation_id="task",
    )
    evidence, _ = EvidenceTracker().record_observations(EvidenceState(owner), [task], owner)
    outcomes = EvidenceTracker().refresh_goal_outcomes(
        descriptor, contract, evidence, {"g1": GoalOutcome("g1")}
    )
    assert outcomes["g1"].status is GoalStatus.SATISFIED
    assert outcomes["g1"].evidence_refs == ("ev_task",)
    with pytest.raises(ValueError):
        GoalOutcome("g1", GoalStatus.SATISFIED)


def test_diagnosis_satisfied_outcome_contains_all_supporting_refs():
    owner = identity()
    descriptor = GoalDescriptor(1, (DiagnoseTask("g1", "A"),))
    contract = CompletionContractCompiler().compile(descriptor)
    items = [
        observation("get_task_detail", {"task_name": "A"}, {"task_name": "A", "state": "FAILED"}, owner=owner, observation_id="task"),
        observation("diagnose_task", {"task_name": "A"}, {"task_name": "A", "root_cause": "OOM"}, owner=owner, observation_id="diagnosis"),
    ]
    evidence, _ = EvidenceTracker().record_observations(EvidenceState(owner), items, owner)
    outcome = EvidenceTracker().refresh_goal_outcomes(
        descriptor, contract, evidence, {"g1": GoalOutcome("g1")}
    )["g1"]
    assert outcome.status is GoalStatus.SATISFIED
    assert set(outcome.evidence_refs) == {"ev_task", "ev_diagnosis"}


def test_default_facade_no_data_shapes_are_strictly_valid_absence():
    facade = InMemoryReadFacade()
    calls = [
        ("get_task_detail", {"task_name": "A"}, facade.get_task_detail("A")),
        ("get_gpu_pool", {}, facade.get_gpu_pool()),
        ("search_knowledge", {"query": "x", "top_k": 5}, facade.search_knowledge("x")),
        ("get_queue_state", {"task_name": None}, facade.get_queue_state()),
        ("diagnose_task", {"task_name": "A"}, facade.diagnose_task("A")),
    ]
    for tool, arguments, payload in calls:
        result = normalize_read_result(tool, arguments, payload)
        assert result.validation_errors == (), (tool, result.validation_errors)
        assert result.envelope.status is not ResultStatus.MALFORMED


class MaliciousProvider:
    model_version = "malicious-test-provider"
    prompt_version = "authority-test"

    def __init__(self) -> None:
        self.calls = 0
        self.mutation_errors: list[str] = []
        self.descriptor = GoalDescriptor(1, (ReadTaskState("g1", "A"),))

    async def generate(self, context):
        self.calls += 1
        self._attempt_mutations(context)
        if self.calls == 1:
            return SingleToolCall(ToolCall("task", "get_task_detail", {"task_name": "A"}), self.descriptor)
        return FinalCandidate("done", referenced_goal_ids=("g1",))

    def _attempt_mutations(self, context) -> None:
        runtime = context.runtime_structured
        attempts = [
            ("contract", lambda: runtime.completion_contract.requirements_by_goal.__setitem__("g1", ())),
            ("goals", lambda: runtime.goal_descriptor.goals.__setitem__(0, None)),
            ("outcomes", lambda: runtime.goal_outcomes.__setitem__(0, None)),
            ("evidence", lambda: runtime.evidence.records.__setitem__(0, None)),
            ("feedback", lambda: runtime.gate_feedback.__setitem__(0, "weakened")),
            ("budget", lambda: setattr(runtime.budgets, "agent_steps_used", 0)),
            ("principles", lambda: runtime.terminal_state.safe_facts.__setitem__("x", "y") if runtime.terminal_state else runtime.gate_feedback.__setitem__(0, "x")),
        ]
        for name, attempt in attempts:
            try:
                attempt()
            except Exception as exc:
                self.mutation_errors.append(f"{name}:{type(exc).__name__}")


def test_malicious_provider_cannot_delete_contract_requirements_end_to_end():
    provider = MaliciousProvider()
    context = build_system_context(provider, read_facade=InMemoryReadFacade())
    result = asyncio.run(invoke("A status", thread_id="authority-e2e", system_context=context))
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.state["current_request"].completion_contract.requirements_by_goal["g1"]
    assert result.state["current_request"].goal_descriptor == provider.descriptor
    assert result.state["current_request"].goal_outcomes["g1"].status is GoalStatus.PENDING
    assert any(item.startswith("contract:") for item in provider.mutation_errors)
