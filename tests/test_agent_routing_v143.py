from __future__ import annotations

import asyncio
import sys
import types

from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import AgentIntent, AgentPlan
from platform_agent.policy import AgentPolicyEngine
from platform_agent.prompt_contract import EVIDENCE_ROUTING_CONTRACT
from platform_agent.tool_client import FacadeToolClient
from platform_mcp.facade import PlatformMCPFacade
from platform_mcp.server import MCP_TOOL_DESCRIPTIONS
from platform_mcp.server import build_mcp_server


def run(coro):
    return asyncio.run(coro)


def plan(query: str, names: tuple[str, ...] = ("get_gpu_pool", "get_task_detail", "get_queue_state", "diagnose_task", "search_knowledge")):
    tools = [{"name": name, "description": MCP_TOOL_DESCRIPTIONS.get(name, name), "input_schema": {}} for name in names]
    return run(HeuristicReadOnlyModel().plan(query, tools, []))


def test_gpu_and_knowledge_tool_descriptions_are_semantically_distinct_and_shared():
    gpu = MCP_TOOL_DESCRIPTIONS["get_gpu_pool"]
    knowledge = MCP_TOOL_DESCRIPTIONS["search_knowledge"]
    assert gpu != knowledge
    assert "current/live" in gpu
    assert "Do not use it to explain platform concepts" in gpu
    assert "static platform documentation" in knowledge
    assert "does not return current runtime state" in knowledge

    facade = PlatformMCPFacade(None, None, None, None, None, None, None, None)
    facade.knowledge_service = object()
    catalog = {item["name"]: item for item in run(FacadeToolClient(facade).describe_tools())}
    assert catalog["get_gpu_pool"]["description"] == gpu
    assert catalog["search_knowledge"]["description"] == knowledge


def test_production_mcp_server_uses_the_same_canonical_descriptions(monkeypatch):
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
    assert server.tools["get_gpu_pool"].__doc__ == MCP_TOOL_DESCRIPTIONS["get_gpu_pool"]
    assert server.tools["search_knowledge"].__doc__ == MCP_TOOL_DESCRIPTIONS["search_knowledge"]


def test_agent_plan_intent_schema_documents_routing_taxonomy():
    description = AgentPlan.model_json_schema()["properties"]["intent"]["description"]
    assert "platform_knowledge is static platform" in description
    assert "gpu_diagnosis is live GPU/resource state" in description
    assert "task_status is current state for a named task" in description


def test_shared_provider_routing_contract_names_all_evidence_classes():
    for marker in ("STATIC_KNOWLEDGE", "LIVE_GPU_STATE", "LIVE_TASK_STATE", "NAMED_TASK_DIAGNOSIS", "TASK_PLANNING", "WRITE_OPERATION", "NO_TOOL"):
        assert marker in EVIDENCE_ROUTING_CONTRACT
    assert "GPU Reservation 是什么" in EVIDENCE_ROUTING_CONTRACT
    assert "现在 GPU0 上有哪些 Reservation" in EVIDENCE_ROUTING_CONTRACT


def test_static_knowledge_routes_to_search_not_gpu_state():
    result = plan("平台的 GPU Reservation 是什么？")
    assert result.intent == AgentIntent.PLATFORM_KNOWLEDGE
    assert [item.name for item in result.tool_calls] == ["search_knowledge"]


def test_live_gpu_routes_to_gpu_pool():
    result = plan("现在 GPU0 上有哪些 Reservation？")
    assert result.intent == AgentIntent.GPU_DIAGNOSIS
    assert [item.name for item in result.tool_calls] == ["get_gpu_pool"]


def test_generic_gpu_diagnosis_does_not_fabricate_task_identity():
    result = plan("Segment 一直拿不到 GPU，帮我分析原因。")
    assert result.intent == AgentIntent.GPU_DIAGNOSIS
    assert result.task_name is None
    assert [item.name for item in result.tool_calls] == ["get_gpu_pool"]


def test_named_task_diagnosis_uses_concrete_task_name():
    result = plan("release_demo 一直没跑起来，帮我看看原因。")
    assert result.intent == AgentIntent.TASK_DIAGNOSIS
    assert result.task_name == "release_demo"
    assert result.tool_calls[0].name == "diagnose_task"
    assert result.tool_calls[0].arguments["task_name"] == "release_demo"


def test_hybrid_gpu_diagnosis_keeps_live_evidence_and_may_add_rules():
    result = plan("现在 Segment 为什么拿不到 GPU？结合平台的 Reservation 规则解释。")
    assert result.intent == AgentIntent.GPU_DIAGNOSIS
    names = [item.name for item in result.tool_calls]
    assert "get_gpu_pool" in names
    assert names.index("get_gpu_pool") < names.index("search_knowledge")


def test_task_planning_and_explicit_submit_are_distinct():
    planning = plan("帮我生成一个 release 任务配置，处理 /data/release_0819，跑完整流程，优先级 5。")
    assert planning.intent == AgentIntent.TASK_PLANNING
    assert planning.tool_calls == []

    submit = plan("帮我创建并提交 release 任务，处理 /data/release_0819，跑完整流程，优先级 5。")
    assert submit.intent == AgentIntent.SUBMIT_TASK
    assert all(item.name not in {"submit_task", "resume_task", "set_task_priority", "stop_task", "delete_task"} for item in submit.tool_calls)
    assert all(item.name in AgentPolicyEngine().allowed_read_tools for item in submit.tool_calls)


def test_greeting_has_no_tool():
    result = plan("你好")
    assert result.intent == AgentIntent.GENERAL_READ
    assert result.tool_calls == []


def test_live_task_status_does_not_use_knowledge_only():
    result = plan("release_demo 当前状态？")
    assert result.intent == AgentIntent.TASK_STATUS
    names = [item.name for item in result.tool_calls]
    assert "get_task_detail" in names
    assert "search_knowledge" not in names
