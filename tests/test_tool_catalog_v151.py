from __future__ import annotations

import asyncio
import sys
import types

from platform_agent.tool_catalog import build_read_only_tool_catalog
from platform_agent.tool_client import FacadeToolClient
from platform_mcp.facade import PlatformMCPFacade
from platform_mcp.server import READ_ONLY_TOOL_NAMES, build_mcp_server


def run(coro):
    return asyncio.run(coro)


def _catalog(knowledge_enabled: bool = True):
    return {item["name"]: item for item in build_read_only_tool_catalog(knowledge_enabled=knowledge_enabled)}


def test_catalog_names_match_read_only_policy_surface():
    assert tuple(_catalog()) == READ_ONLY_TOOL_NAMES


def test_catalog_uses_semantic_descriptions_and_required_identity_fields():
    catalog = _catalog()
    assert "current/live GPU runtime state" in catalog["get_gpu_pool"]["description"]
    assert "static platform documentation" in catalog["search_knowledge"]["description"]
    assert catalog["get_task_detail"]["input_schema"]["required"] == ["task_name"]
    assert catalog["diagnose_task"]["input_schema"]["required"] == ["task_name"]
    assert catalog["get_stage_logs"]["input_schema"]["required"] == ["task_name"]
    assert catalog["search_knowledge"]["input_schema"]["required"] == ["query"]


def test_knowledge_disabled_hides_search_knowledge():
    catalog = _catalog(knowledge_enabled=False)
    assert "search_knowledge" not in catalog
    assert len(catalog) == len(READ_ONLY_TOOL_NAMES) - 1


def test_facade_fixture_uses_canonical_catalog():
    facade = PlatformMCPFacade(None, None, None, None, None, None, None, None)
    facade.knowledge_service = object()
    actual = {item["name"]: item for item in run(FacadeToolClient(facade).describe_tools())}
    expected = _catalog()
    assert actual == expected


def test_mcp_registered_descriptions_match_canonical_catalog(monkeypatch):
    class FakeMCPServer:
        def __init__(self, name, instructions=""):
            self.tools = {}

        def tool(self):
            def register(fn):
                self.tools[fn.__name__] = fn
                return fn

            return register

    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    server_module.MCPServer = FakeMCPServer
    mcp_module.server = server_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)

    facade = PlatformMCPFacade(None, None, None, None, None, None, None, None)
    facade.knowledge_service = object()
    server = build_mcp_server(facade)
    catalog = _catalog()
    assert tuple(server.tools) == READ_ONLY_TOOL_NAMES
    assert {name: server.tools[name].__doc__ for name in READ_ONLY_TOOL_NAMES} == {
        name: catalog[name]["description"] for name in READ_ONLY_TOOL_NAMES
    }
