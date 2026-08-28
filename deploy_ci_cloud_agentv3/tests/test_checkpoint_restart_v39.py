from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("mcp")

pytestmark = pytest.mark.real_langgraph
pytest.importorskip("langgraph.checkpoint.sqlite.aio")

from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
from deploy_ci_cloud_agentv3.persistence.checkpoint import CheckpointerFactory
from deploy_ci_cloud_agentv3.providers.base import AssistantMessage, ToolCall
from deploy_ci_cloud_agentv3.providers.scripted import ScriptedProvider
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


def interrupt(state):
    item=(state.get("__interrupt__") or [])[0]
    return getattr(item,"value",item)


@pytest.mark.asyncio
async def test_interrupt_survives_runtime_restart(tmp_path,monkeypatch):
    monkeypatch.setenv("AUTODRIVE_STATE_DIR",str(tmp_path))
    monkeypatch.setenv("AUTODRIVE_DB_PATH",str(tmp_path/"state.sqlite"))
    facade=FakeFacade(); checkpoint=tmp_path/"checkpoints.sqlite"
    async with CheckpointerFactory.open("sqlite",path=checkpoint) as saver:
        runtime=AgentRuntime.local(ScriptedProvider([AssistantMessage(tool_calls=[ToolCall(id="p",name="propose_set_task_priority",arguments={"task_name":"task_a","priority":5})])]),facade=facade,checkpointer=saver)
        state=await runtime.start("restart-approve","raise priority",run_id="r1"); review=interrupt(state)
    async with CheckpointerFactory.open("sqlite",path=checkpoint) as saver:
        runtime=AgentRuntime.local(ScriptedProvider([AssistantMessage(content='{"status":"write_verified","message":"done"}')]),facade=facade,checkpointer=saver)
        state=await runtime.review("restart-approve",{"decision":"approve","fingerprint":review["fingerprint"]})
    assert state["final_response"]["status"]=="write_verified" and len(facade.mutations)==1


@pytest.mark.asyncio
async def test_reject_survives_restart_zero_mutation(tmp_path,monkeypatch):
    monkeypatch.setenv("AUTODRIVE_STATE_DIR",str(tmp_path)); monkeypatch.setenv("AUTODRIVE_DB_PATH",str(tmp_path/"state.sqlite"))
    facade=FakeFacade(); checkpoint=tmp_path/"checkpoints.sqlite"
    async with CheckpointerFactory.open("sqlite",path=checkpoint) as saver:
        runtime=AgentRuntime.local(ScriptedProvider([AssistantMessage(tool_calls=[ToolCall(id="p",name="propose_delete_task",arguments={"task_name":"task_a"})])]),facade=facade,checkpointer=saver)
        state=await runtime.start("restart-reject","delete",run_id="r2"); interrupt(state)
    async with CheckpointerFactory.open("sqlite",path=checkpoint) as saver:
        runtime=AgentRuntime.local(ScriptedProvider([AssistantMessage(content='{"status":"write_not_executed","message":"rejected"}')]),facade=facade,checkpointer=saver)
        state=await runtime.review("restart-reject",{"decision":"reject","reason":"no"})
    assert state["final_response"]["status"]=="write_not_executed" and facade.mutations==[]


@pytest.mark.asyncio
async def test_edit_restart_invalidates_old_fingerprint(tmp_path,monkeypatch):
    monkeypatch.setenv("AUTODRIVE_STATE_DIR",str(tmp_path)); monkeypatch.setenv("AUTODRIVE_DB_PATH",str(tmp_path/"state.sqlite"))
    facade=FakeFacade(); checkpoint=tmp_path/"checkpoints.sqlite"
    async with CheckpointerFactory.open("sqlite",path=checkpoint) as saver:
        runtime=AgentRuntime.local(ScriptedProvider([AssistantMessage(tool_calls=[ToolCall(id="p",name="propose_set_task_priority",arguments={"task_name":"task_a","priority":5})])]),facade=facade,checkpointer=saver)
        first=interrupt(await runtime.start("restart-edit","priority",run_id="r3"))
    async with CheckpointerFactory.open("sqlite",path=checkpoint) as saver:
        runtime=AgentRuntime.local(ScriptedProvider([]),facade=facade,checkpointer=saver)
        second=interrupt(await runtime.review("restart-edit",{"decision":"edit","args":{"task_name":"task_a","priority":7}}))
        assert second["fingerprint"]!=first["fingerprint"]
        state=await runtime.review("restart-edit",{"decision":"approve","fingerprint":first["fingerprint"]})
        # Wrong old fingerprint keeps the graph at review instead of executing.
        assert (state.get("__interrupt__") or []) and facade.mutations==[]
