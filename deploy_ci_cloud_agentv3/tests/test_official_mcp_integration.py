from __future__ import annotations

import os
import socket

import pytest

pytest.importorskip("mcp")

from deploy_ci_cloud_agentv3.mcp.client import OfficialMCPClient
from deploy_ci_cloud_agentv3.mcp.profiles import AGENT_TOOLS, RUNTIME_TOOLS, WRITE_TOOLS
from deploy_ci_cloud_agentv3.mcp.server import build_mcp_servers, create_app
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


@pytest.mark.asyncio
async def test_official_mcp_inprocess_profiles_and_proposal_zero_side_effect():
    facade = FakeFacade()
    agent_server, runtime_server, _ = build_mcp_servers(facade)
    agent = OfficialMCPClient(agent_server)
    runtime = OfficialMCPClient(runtime_server)
    agent_names = {item.name for item in await agent.list_tools()}
    runtime_names = {item.name for item in await runtime.list_tools()}
    assert agent_names == AGENT_TOOLS
    assert runtime_names == RUNTIME_TOOLS
    assert WRITE_TOOLS.isdisjoint(agent_names)
    assert WRITE_TOOLS <= runtime_names
    result = await agent.call_tool("propose_delete_task", {"task_name": "task_a"})
    assert result["action"] == "delete_task"
    assert facade.mutations == []
    priority_schema = next(item.input_schema for item in await runtime.list_tools() if item.name == "set_task_priority")
    assert priority_schema["properties"]["priority"]["minimum"] == 0
    assert priority_schema["properties"]["priority"]["maximum"] == 100


@pytest.mark.asyncio
async def test_streamable_http_smoke_tools_list_and_call():
    if os.environ.get("SKIP_MCP_HTTP_SMOKE") == "1":
        pytest.skip("explicitly disabled")
    uvicorn = pytest.importorskip("uvicorn")
    facade = FakeFacade()
    app = create_app(facade)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    import asyncio
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.02)
        assert server.started
        client = OfficialMCPClient(f"http://127.0.0.1:{port}/mcp/agent/")
        names = {item.name for item in await client.list_tools()}
        assert "get_task_detail" in names
        result = await client.call_tool("propose_set_task_priority", {"task_name": "task_a", "priority": 5})
        assert result["action"] == "set_task_priority"
        assert facade.mutations == []
    finally:
        server.should_exit = True
        await task
