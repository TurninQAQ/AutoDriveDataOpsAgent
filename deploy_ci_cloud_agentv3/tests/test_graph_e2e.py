from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("mcp")

from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
from deploy_ci_cloud_agentv3.providers.base import AssistantMessage, ToolCall
from deploy_ci_cloud_agentv3.providers.scripted import ScriptedProvider
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


def _interrupt_value(state):
    items = state.get("__interrupt__") or []
    assert items
    return getattr(items[0], "value", items[0])


@pytest.mark.asyncio
async def test_read_react_loop_reaches_structured_final():
    facade = FakeFacade()
    provider = ScriptedProvider([
        AssistantMessage(tool_calls=[ToolCall(id="r1", name="get_task_detail", arguments={"task_name": "task_a"})]),
        AssistantMessage(content='{"status":"informational","message":"task observed"}'),
    ])
    state = await AgentRuntime.in_process(provider, facade=facade).start("read-1", "what is task_a doing?")
    assert state["final_response"]["status"] == "informational"
    assert any(item.get("tool_name") == "get_task_detail" for item in state["tool_results"])


@pytest.mark.asyncio
async def test_proposal_interrupt_approve_executes_exact_frozen_write():
    facade = FakeFacade()
    provider = ScriptedProvider([
        AssistantMessage(tool_calls=[ToolCall(id="p1", name="propose_set_task_priority", arguments={"task_name": "task_a", "priority": 5})]),
        AssistantMessage(content='{"status":"write_verified","message":"done"}'),
    ])
    runtime = AgentRuntime.in_process(provider, facade=facade)
    state = await runtime.start("approve-1", "raise priority")
    review = _interrupt_value(state)
    assert review["action"] == "set_task_priority"
    state = await runtime.review("approve-1", {"decision": "approve", "fingerprint": review["fingerprint"]})
    assert state["final_response"]["status"] == "write_verified"
    assert facade.priority == 5
    assert len(facade.mutations) == 1


@pytest.mark.asyncio
async def test_proposal_reject_never_mutates():
    facade = FakeFacade()
    provider = ScriptedProvider([
        AssistantMessage(tool_calls=[ToolCall(id="p1", name="propose_delete_task", arguments={"task_name": "task_a"})]),
        AssistantMessage(content='{"status":"write_not_executed","message":"rejected"}'),
    ])
    runtime = AgentRuntime.in_process(provider, facade=facade)
    state = await runtime.start("reject-1", "delete task")
    _interrupt_value(state)
    state = await runtime.review("reject-1", {"decision": "reject", "reason": "no"})
    assert state["final_response"]["status"] == "write_not_executed"
    assert facade.mutations == []


@pytest.mark.asyncio
async def test_edit_rebuilds_fingerprint_and_reinterrupts_before_execution():
    facade = FakeFacade()
    provider = ScriptedProvider([
        AssistantMessage(tool_calls=[ToolCall(id="p1", name="propose_set_task_priority", arguments={"task_name": "task_a", "priority": 5})]),
        AssistantMessage(content='{"status":"write_verified","message":"done"}'),
    ])
    runtime = AgentRuntime.in_process(provider, facade=facade)
    state = await runtime.start("edit-1", "raise priority")
    first = _interrupt_value(state)
    state = await runtime.review("edit-1", {"decision": "edit", "args": {"task_name": "task_a", "priority": 7}})
    second = _interrupt_value(state)
    assert second["fingerprint"] != first["fingerprint"]
    assert second["args"]["priority"] == 7
    assert facade.mutations == []
    state = await runtime.review("edit-1", {"decision": "approve", "fingerprint": second["fingerprint"]})
    assert state["final_response"]["status"] == "write_verified"
    assert facade.priority == 7
