from __future__ import annotations

import pytest

from deploy_ci_cloud_agentv3.mcp.client import InProcessMCPClient
from deploy_ci_cloud_agentv3.mcp.factory import build_tooling
from deploy_ci_cloud_agentv3.mcp.profiles import AGENT_TOOLS, RUNTIME_TOOLS, WRITE_TOOLS
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


@pytest.mark.asyncio
async def test_agent_mcp_never_exposes_real_write_tools():
    facade = FakeFacade()
    tooling = build_tooling(facade)
    client = InProcessMCPClient(tooling.registry, AGENT_TOOLS)
    names = {tool.name for tool in await client.list_tools()}
    assert WRITE_TOOLS.isdisjoint(names)
    assert {"get_task_detail", "prepare_task_spec", "propose_set_task_priority"} <= names


@pytest.mark.asyncio
async def test_runtime_mcp_exposes_write_tools():
    tooling = build_tooling(FakeFacade())
    client = InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)
    names = {tool.name for tool in await client.list_tools()}
    assert WRITE_TOOLS <= names
    assert "propose_delete_task" not in names


@pytest.mark.asyncio
async def test_proposal_tool_has_zero_platform_side_effect():
    facade = FakeFacade()
    tooling = build_tooling(facade)
    client = InProcessMCPClient(tooling.registry, AGENT_TOOLS)
    result = await client.call_tool("propose_set_task_priority", {"task_name": "task_a", "priority": 5})
    assert result["kind"] == "ACTION_PROPOSAL"
    assert result["action"] == "set_task_priority"
    assert facade.priority == 3
    assert facade.mutations == []
