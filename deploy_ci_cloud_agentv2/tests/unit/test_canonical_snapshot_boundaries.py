import asyncio
from dataclasses import dataclass, replace

import pytest

from deploy_ci_cloud_agentv2 import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.decisions import (
    AcceptedToolCall,
    FinalCandidate,
    ReadToolBatch,
    SingleToolCall,
    ToolCall,
)
from deploy_ci_cloud_agentv2.agent.events import EventStore
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.immutable import canonical_snapshot
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers.model import AgentProvider
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.metadata import Idempotency, RiskLevel, ToolKind, ToolSpec
from deploy_ci_cloud_agentv2.tools.registry import ToolRegistry
from deploy_ci_cloud_agentv2.tools.runtime import ReadToolRuntime
from deploy_ci_cloud_agentv2.tests.helpers import identity


class AliasingFacade(InMemoryReadFacade):
    def __init__(self):
        super().__init__()
        self.payload = {
            "task_name": "task_A",
            "state": "RUNNING",
            "meta": {"log": ["safe"]},
        }

    def get_task_detail(self, task_name: str):
        return self.payload


class SnapshotProvider(AgentProvider):
    model_version = "snapshot-test-provider"
    prompt_version = "snapshot-test"

    def __init__(self, facade: AliasingFacade):
        self.facade = facade
        self.calls = 0
        self.contexts = []
        self.descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))

    async def generate(self, context):
        self.calls += 1
        self.contexts.append(context)
        if self.calls == 1:
            return SingleToolCall(
                ToolCall("task", "get_task_detail", {"task_name": " task_A "}),
                self.descriptor,
            )
        # This mutation occurs after the Runtime has accepted and observed the
        # payload.  It must not change canonical observation/evidence/audit.
        self.facade.payload["state"] = "FAILED"
        self.facade.payload["meta"]["log"][0] = "INJECTED"
        return FinalCandidate("done", referenced_goal_ids=("g1",))


def test_external_tool_payload_is_one_detached_snapshot_across_all_paths():
    facade = AliasingFacade()
    provider = SnapshotProvider(facade)
    context = build_system_context(provider, read_facade=facade)
    result = asyncio.run(invoke("task_A status", thread_id="snapshot", system_context=context))

    assert result.status == "COMPLETED"
    observation = result.state["current_request"].observations[0]
    assert observation.data["state"] == "RUNNING"
    assert observation.data["meta"]["log"][0] == "safe"
    assert observation.result.state.value == "RUNNING"
    assert observation.result.metadata["meta"]["log"][0] == "safe"
    assert result.state["current_request"].evidence.records

    observation_events = [
        event
        for event in context.event_store.all()
        if event.event_type == "ToolObservationRecorded"
    ]
    assert observation_events[0].payload["data"]["state"] == "RUNNING"
    assert observation_events[0].payload["data"]["meta"]["log"][0] == "safe"
    # The projection was built before the provider changed its source object.
    assert "INJECTED" not in repr(provider.contexts[-1].semantic_observations)


def test_parallel_sibling_payloads_are_snapshotted_independently():
    class ParallelFacade(InMemoryReadFacade):
        def __init__(self):
            super().__init__()
            self.payloads = {
                "get_task_detail": {"task_name": "A", "state": "RUNNING", "meta": {"v": [1]}},
                "get_gpu_pool": {"devices": [{"gpu_id": "0", "meta": {"v": [2]}}]},
                "get_queue_state": {"scope": "PLATFORM", "queue": [{"task_name": "A", "position": 1}]},
            }

        def get_task_detail(self, task_name):
            return self.payloads["get_task_detail"]

        def get_gpu_pool(self):
            return self.payloads["get_gpu_pool"]

        def get_queue_state(self, task_name=None):
            return self.payloads["get_queue_state"]

    facade = ParallelFacade()
    runtime = ReadToolRuntime(build_read_registry(facade), identity())
    batch = ReadToolBatch(
        (
            ToolCall("task", "get_task_detail", {"task_name": "A"}),
            ToolCall("gpu", "get_gpu_pool", {}),
            ToolCall("queue", "get_queue_state", {"task_name": None}),
        )
    )
    result = asyncio.run(runtime.execute_batch(batch, max_retries=0, max_batch=3))
    facade.payloads["get_task_detail"]["state"] = "FAILED"
    facade.payloads["get_task_detail"]["meta"]["v"].append(9)
    facade.payloads["get_gpu_pool"]["devices"][0]["meta"]["v"].append(9)
    facade.payloads["get_queue_state"]["queue"][0]["position"] = 99

    assert result.results[0].data["state"] == "RUNNING"
    assert result.results[0].data["meta"]["v"] == (1,)
    assert result.results[1].data["devices"][0]["meta"]["v"] == (2,)
    assert result.results[2].data["queue"][0]["position"] == 1


def test_tool_call_arguments_are_detached_and_accepted_call_is_frozen():
    arguments = {"task_name": " task_A "}
    proposal = ToolCall("task", "get_task_detail", arguments)
    arguments["task_name"] = "task_B"
    assert proposal.arguments["task_name"] == " task_A "

    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()), identity())
    accepted = runtime.validate_single(proposal)
    assert isinstance(accepted, AcceptedToolCall)
    assert accepted.arguments == {"task_name": "task_A"}
    with pytest.raises(TypeError):
        accepted.arguments["task_name"] = "task_B"

    nested = {"filters": {"datasets": ["A"]}}
    nested_accepted = AcceptedToolCall("nested", "test", nested)
    nested["filters"]["datasets"].append("B")
    assert nested_accepted.arguments["filters"]["datasets"] == ("A",)


def test_canonical_snapshot_detaches_dataclass_and_set_like_payloads():
    @dataclass(frozen=True)
    class ExternalPayload:
        metadata: dict
        tags: set

    source = ExternalPayload({"items": ["safe"]}, {"one"})
    snapshot = canonical_snapshot(source)
    source.metadata["items"].append("changed")
    source.tags.add("changed")
    assert snapshot["metadata"]["items"] == ("safe",)
    assert snapshot["tags"] == {"one"}


def test_batch_and_final_candidate_collections_are_canonicalized():
    calls = [ToolCall("task", "get_task_detail", {"task_name": "A"})]
    batch = ReadToolBatch(calls)
    calls.append(ToolCall("queue", "get_queue_state", {"task_name": None}))
    assert len(batch.calls) == 1

    refs = ["g1"]
    candidate = FinalCandidate("done", referenced_goal_ids=refs)
    refs.append("attacker-goal")
    assert candidate.referenced_goal_ids == ("g1",)

    source_goals = [ReadTaskState("g1", "A")]
    descriptor = GoalDescriptor(1, source_goals)
    source_goals.append(ReadTaskState("attacker", "B"))
    assert tuple(goal.goal_id for goal in descriptor.goals) == ("g1",)


def test_guard_and_executor_audit_the_same_normalized_arguments():
    class Provider:
        model_version = "audit-provider"
        prompt_version = "audit-prompt"

        def __init__(self):
            self.calls = 0
            self.args = {"task_name": " task_A "}
            self.descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))

        async def generate(self, context):
            self.calls += 1
            if self.calls == 1:
                proposal = ToolCall("task", "get_task_detail", self.args)
                self.args["task_name"] = "task_B"
                return SingleToolCall(proposal, self.descriptor)
            return FinalCandidate("done", referenced_goal_ids=("g1",))

    provider = Provider()
    facade = InMemoryReadFacade(
        responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
    )
    context = build_system_context(provider, read_facade=facade)
    result = asyncio.run(invoke("task_A status", thread_id="toctou", system_context=context))
    assert result.status == "COMPLETED"
    decision_event = next(
        event for event in context.event_store.all() if event.event_type == "AgentDecisionMade"
    )
    started = next(event for event in context.event_store.all() if event.event_type == "ToolCallStarted")
    observation = result.state["current_request"].observations[0]
    assert started.payload["arguments"] == {"task_name": "task_A"}
    assert decision_event.payload["calls"][0]["arguments"] == {"task_name": "task_A"}
    assert started.payload["arguments_fingerprint"] == observation.provenance.arguments_fingerprint
    assert observation.provenance.requested_identity == "task_A"


def _simple_spec(name="test_read"):
    return ToolSpec(
        name=name,
        kind=ToolKind.READ,
        risk=RiskLevel.LOW,
        schema={"type": "object", "properties": {}, "required": []},
        parallel_safe=True,
        idempotency=Idempotency.SAFE_RETRY,
    )


def test_tool_registry_seal_freezes_specs_and_hash():
    registry = ToolRegistry()
    registry.register(_simple_spec(), lambda: {"devices": []})
    assert not registry.is_sealed
    registry.seal()
    assert registry.is_sealed
    first_hash = registry.catalog_hash()
    assert registry.catalog_hash() == first_hash
    with pytest.raises(RuntimeError):
        registry.register(_simple_spec("later"), lambda: {})
    with pytest.raises(TypeError):
        registry.spec("test_read").schema["properties"] = {"x": {}}
    with pytest.raises(TypeError):
        registry.spec("test_read").schema["properties"]["x"] = {}
    assert registry.catalog_hash() == first_hash


def test_invoke_fails_closed_on_catalog_hash_mismatch_without_provider_execution():
    class NeverCalledProvider:
        model_version = "never"
        prompt_version = "never"
        calls = 0

        async def generate(self, context):
            self.calls += 1
            raise AssertionError("provider must not run with a mismatched catalog")

    provider = NeverCalledProvider()
    good = build_system_context(provider)
    tampered = replace(good, tool_catalog_hash="tampered-catalog-hash")
    result = asyncio.run(invoke("task_A status", thread_id="catalog-drift", system_context=tampered))
    assert result.status == "CONTROLLED_TERMINAL"
    assert provider.calls == 0
    assert result.terminal_outcome.safe_facts["reason"] == "TOOL_CATALOG_INTEGRITY_ERROR"
