from __future__ import annotations

import pytest
pytest.importorskip("mcp")

from types import SimpleNamespace

import deploy_ci_cloud_agentv3.agent.runtime as runtime_module
from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
from deploy_ci_cloud_agentv3.mcp.client import OfficialMCPClient


def test_local_mainline_uses_official_mcp_server_objects(monkeypatch):
    agent_server = object()
    runtime_server = object()
    monkeypatch.setattr(runtime_module, "build_mcp_servers", lambda facade: (agent_server, runtime_server, object()))

    captured = {}

    def fake_build_runtime(cls, provider, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(provider=provider)

    monkeypatch.setattr(runtime_module, "_build_runtime", fake_build_runtime)
    result = AgentRuntime.local(object(), facade=object())

    assert result is not None
    assert isinstance(captured["agent_mcp"], OfficialMCPClient)
    assert isinstance(captured["runtime_mcp"], OfficialMCPClient)
    assert captured["agent_mcp"].source is agent_server
    assert captured["runtime_mcp"].source is runtime_server


def test_remote_mainline_uses_official_mcp_urls(monkeypatch):
    captured = {}

    def fake_build_runtime(cls, provider, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(provider=provider)

    monkeypatch.setattr(runtime_module, "_build_runtime", fake_build_runtime)
    AgentRuntime.remote(
        object(),
        agent_mcp_url="http://127.0.0.1:8000/mcp/agent",
        runtime_mcp_url="http://127.0.0.1:8000/mcp/runtime",
    )
    assert isinstance(captured["agent_mcp"], OfficialMCPClient)
    assert isinstance(captured["runtime_mcp"], OfficialMCPClient)
    assert captured["agent_mcp"].source.endswith("/mcp/agent")
    assert captured["runtime_mcp"].source.endswith("/mcp/runtime")
