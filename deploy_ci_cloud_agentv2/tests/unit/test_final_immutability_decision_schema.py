import asyncio
from dataclasses import dataclass
from math import inf, nan

import pytest

from deploy_ci_cloud_agentv2 import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.decision_ingress import (
    AgentDecisionIngressValidator,
    AgentDecisionValidationError,
)
from deploy_ci_cloud_agentv2.agent.decisions import (
    AcceptedToolCall,
    FinalCandidate,
    ReadToolBatch,
    SingleToolCall,
    ToolCall,
)
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.immutable import CanonicalizationError, FrozenMapping, canonical_snapshot
from deploy_ci_cloud_agentv2.agent.results import normalize_read_result
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers.model import AgentProvider
from deploy_ci_cloud_agentv2.tests.helpers import identity
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.runtime import ReadToolRuntime


class Box:
    def __init__(self, value):
        self.value = value


@pytest.mark.parametrize(
    "value",
    [bytearray(b"abc"), memoryview(b"abc"), Box(1), object(), lambda: None],
)
def test_closed_canonical_domain_rejects_unknown_leaves(value):
    with pytest.raises(CanonicalizationError):
        canonical_snapshot({"value": value})


def test_closed_canonical_domain_rejects_nested_unknown_non_string_key_and_non_finite_float():
    with pytest.raises(CanonicalizationError):
        canonical_snapshot({"a": [{"b": bytearray(b"x")}]})
    with pytest.raises(CanonicalizationError):
        canonical_snapshot({1: "not a canonical string key"})
    with pytest.raises(CanonicalizationError):
        canonical_snapshot({"value": nan})
    with pytest.raises(CanonicalizationError):
        canonical_snapshot({"value": inf})
    with pytest.raises(CanonicalizationError):
        FrozenMapping({"value": Box(1)})


def test_canonical_values_have_no_mutable_reachable_containers():
    source = {"nested": [{"items": [1, 2]}], "text": "safe"}
    snapshot = canonical_snapshot(source)
    assert isinstance(snapshot, FrozenMapping)
    assert isinstance(snapshot["nested"], tuple)
    assert isinstance(snapshot["nested"][0], FrozenMapping)
    assert isinstance(snapshot["nested"][0]["items"], tuple)
    source["nested"][0]["items"].append(3)
    assert snapshot["nested"][0]["items"] == (1, 2)


class UnsupportedPayloadFacade(InMemoryReadFacade):
    def __init__(self):
        super().__init__()
        self.payload = {"task_name": "A", "state": "RUNNING", "bad": Box(1)}

    def get_task_detail(self, task_name: str):
        return self.payload


def test_tool_ingress_rejects_unsupported_external_payload_without_canonical_data():
    facade = UnsupportedPayloadFacade()
    runtime = ReadToolRuntime(build_read_registry(facade), identity())
    observation = asyncio.run(
        runtime.execute_single(
            ToolCall("task", "get_task_detail", {"task_name": "A"}),
            max_retries=0,
        )
    )
    assert observation.data is None
    assert observation.error_code == "UNSUPPORTED_EXTERNAL_PAYLOAD"
    assert observation.result is None


def test_recursive_queue_entry_unknown_state_is_not_meaningful():
    platform_unknown = normalize_read_result(
        "get_queue_state",
        {"task_name": None},
        {"scope": "PLATFORM", "queue": [{"state": "UNKNOWN_EXTERNAL_STATE"}]},
    )
    task_unknown = normalize_read_result(
        "get_queue_state",
        {"task_name": "A"},
        {
            "scope": "TASK",
            "task_name": "A",
            "queue": [{"task_name": "A", "state": "UNKNOWN_EXTERNAL_STATE"}],
        },
    )
    positioned = normalize_read_result(
        "get_queue_state",
        {"task_name": None},
        {
            "scope": "PLATFORM",
            "queue": [{"state": "UNKNOWN_EXTERNAL_STATE", "position": 4}],
        },
    )
    mixed = normalize_read_result(
        "get_queue_state",
        {"task_name": None},
        {
            "scope": "PLATFORM",
            "queue": [
                {"state": "UNKNOWN_EXTERNAL_STATE"},
                {"state": "UNKNOWN_EXTERNAL_STATE"},
                {"task_name": "A", "state": "QUEUED"},
            ],
        },
    )
    assert not platform_unknown.qualifies_for_evidence()
    assert not task_unknown.qualifies_for_evidence()
    assert positioned.qualifies_for_evidence()
    assert mixed.qualifies_for_evidence()


def test_agent_decision_ingress_rejects_malformed_non_tool_fields_before_compiler():
    validator = AgentDecisionIngressValidator()
    cases = [
        FinalCandidate(123, referenced_goal_ids=("g1",)),
        FinalCandidate("done", referenced_goal_ids=("g1", 123)),
        FinalCandidate("done", referenced_goal_ids=(None,)),
        SingleToolCall(ToolCall("c", "get_task_detail", {"task_name": "A"}), "bad"),
        ReadToolBatch(("not-a-tool-call",)),
        SingleToolCall(
            ToolCall("c", "get_task_detail", {"task_name": Box("A")}),
            None,
        ),
    ]
    for proposal in cases:
        with pytest.raises(AgentDecisionValidationError):
            validator.validate(proposal)

    forged_descriptor = object.__new__(GoalDescriptor)
    object.__setattr__(forged_descriptor, "descriptor_version", 1)
    object.__setattr__(forged_descriptor, "goals", ("bad-goal",))
    with pytest.raises(AgentDecisionValidationError):
        validator.validate(
            SingleToolCall(
                ToolCall("c", "get_task_detail", {"task_name": "A"}),
                forged_descriptor,
            )
        )


class MalformedThenValidProvider(AgentProvider):
    model_version = "malformed-decision-provider"
    prompt_version = "malformed-decision-test"

    def __init__(self):
        self.calls = 0
        self.descriptor = GoalDescriptor(1, (ReadTaskState("g1", "A"),))

    async def generate(self, context):
        self.calls += 1
        if self.calls == 1:
            return FinalCandidate(123, referenced_goal_ids=("g1",))
        if self.calls == 2:
            return SingleToolCall(
                ToolCall("task", "get_task_detail", {"task_name": "A"}),
                self.descriptor,
            )
        return FinalCandidate("grounded", referenced_goal_ids=("g1",))


def test_graph_recovers_after_malformed_agent_decision():
    provider = MalformedThenValidProvider()
    context = build_system_context(
        provider,
        read_facade=InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "A", "state": "RUNNING"}}
        ),
    )
    result = asyncio.run(invoke("A status", thread_id="malformed-decision", system_context=context))
    assert result.status == "COMPLETED"
    assert any(event.event_type == "AgentDecisionRejected" for event in context.event_store.all())
    assert result.state["current_request"].terminal_state is None


def test_manual_accepted_tool_call_cannot_bypass_executor_parallel_guard():
    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()), identity())
    manual = AcceptedToolCall("diagnose", "diagnose_task", {"task_name": "A"})
    with pytest.raises(ValueError, match="parallel-safe"):
        asyncio.run(
            runtime.execute_batch(
                ReadToolBatch((manual,)), max_retries=0, max_batch=3
            )
        )


def test_manual_accepted_tool_call_unknown_or_malformed_is_rejected():
    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()), identity())
    with pytest.raises(ValueError, match="unknown tool"):
        asyncio.run(
            runtime.execute_single(
                AcceptedToolCall("unknown", "missing_tool", {}), max_retries=0
            )
        )
    with pytest.raises(ValueError, match="arguments"):
        asyncio.run(
            runtime.execute_single(
                AcceptedToolCall("bad", "get_task_detail", {"task_name": Box("A")}),
                max_retries=0,
            )
        )


def test_valid_runtime_accepted_call_still_executes():
    facade = InMemoryReadFacade(
        responses={"get_task_detail": {"task_name": "A", "state": "RUNNING"}}
    )
    runtime = ReadToolRuntime(build_read_registry(facade), identity())
    accepted = runtime.validate_single(ToolCall("task", "get_task_detail", {"task_name": " A "}))
    observation = asyncio.run(runtime.execute_single(accepted, max_retries=0))
    assert observation.data["task_name"] == "A"
    assert observation.result.task_name == "A"
