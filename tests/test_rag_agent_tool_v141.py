from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

from platform_agent.memory import ConversationStore
from platform_agent.models import AgentIntent, AgentPlan, AgentResponse, ToolCallSpec, ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.tool_client import FacadeToolClient
from platform_agent.workflow import ReadOnlyAgentNodes, SequentialReadOnlyAgent
from platform_mcp.facade import PlatformMCPFacade
from platform_mcp.server import READ_ONLY_TOOL_NAMES, WRITE_TOOL_NAMES, build_mcp_server
from platform_rag.models import KnowledgeSearchResult, RetrievedKnowledge


class FakeKnowledgeService:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int | None = None):
        self.calls.append((query, int(top_k or 0)))
        return KnowledgeSearchResult(
            query=query,
            results=[
                RetrievedKnowledge(
                    chunk_id="gpu-1",
                    source_path="runbooks/gpu.md",
                    title="GPU Reservation",
                    section="独占",
                    content="独占 GPU 需要检查 Reservation 和可用显存。",
                    score=0.91,
                    lexical_score=0.9,
                    vector_score=0.92,
                    metadata={"kind": "runbook"},
                )
            ],
        )


def make_facade(knowledge_service=None):
    return PlatformMCPFacade(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        knowledge_service=knowledge_service,
    )


def test_search_knowledge_is_registered_with_read_only_schema(monkeypatch):
    class FakeMCPServer:
        def __init__(self, name, instructions=""):
            self.name = name
            self.instructions = instructions
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

    service = FakeKnowledgeService()
    server = build_mcp_server(make_facade(service))

    assert "search_knowledge" in READ_ONLY_TOOL_NAMES
    assert "search_knowledge" not in WRITE_TOOL_NAMES
    assert "search_knowledge" in server.tools
    result = server.tools["search_knowledge"]("GPU Reservation", 3)
    assert result["results"][0]["rank"] == 1
    assert result["results"][0]["source"] == "runbooks/gpu.md#独占"
    assert result["results"][0]["content"]
    assert result["results"][0]["score"] == 0.91
    assert service.calls == [("GPU Reservation", 3)]


def test_facade_tool_client_exposes_search_schema_and_executes_read_only_tool():
    service = FakeKnowledgeService()
    client = FacadeToolClient(make_facade(service))
    tools = asyncio.run(client.describe_tools())
    search = next(item for item in tools if item["name"] == "search_knowledge")

    assert search["input_schema"]["required"] == ["query"]
    assert search["input_schema"]["properties"]["top_k"]["default"] == 5

    observations = asyncio.run(
        client.execute([ToolCallSpec(name="search_knowledge", arguments={"query": "GPU Reservation", "top_k": 2})])
    )
    assert observations[0].ok is True
    assert observations[0].data["results"][0]["rank"] == 1


class PlannerModel:
    requires_tool_descriptions = True

    def __init__(self, use_search: bool):
        self.use_search = use_search
        self.catalogs: list[list[dict]] = []

    async def plan(self, user_text, tool_descriptions, history):
        del history
        self.catalogs.append(tool_descriptions)
        if self.use_search:
            return AgentPlan(
                intent=AgentIntent.PLATFORM_KNOWLEDGE,
                tool_calls=[
                    ToolCallSpec(
                        name="search_knowledge",
                        arguments={"query": user_text, "top_k": 3},
                    )
                ],
                decision_summary="Use the knowledge search tool.",
            )
        return AgentPlan(
            intent=AgentIntent.GENERAL_READ,
            tool_calls=[],
            decision_summary="No tool is needed.",
        )

    async def synthesize(self, user_text, plan, observations, history, knowledge=None):
        del user_text, history, knowledge
        return AgentResponse(
            intent=plan.intent,
            summary="; ".join(item.tool_name for item in observations) or "direct answer",
            confidence="high",
            tool_trace=[
                {"tool": item.tool_name, "arguments": item.arguments, "ok": item.ok}
                for item in observations
            ],
        )


class AgentToolClient:
    def __init__(self):
        self.calls: list[ToolCallSpec] = []

    async def describe_tools(self):
        return [
            {
                "name": "search_knowledge",
                "description": "Search the static platform knowledge base.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
            }
        ]

    async def execute(self, calls):
        self.calls.extend(calls)
        return [
            ToolObservation(
                tool_name=call.name,
                arguments=call.arguments,
                ok=True,
                data={"results": [{"rank": 1, "source": "runbooks/gpu.md", "content": "evidence"}]},
            )
            for call in calls
        ]


def test_agent_can_choose_search_knowledge_without_forced_retrieval(tmp_path: Path):
    model = PlannerModel(use_search=True)
    client = AgentToolClient()
    nodes = ReadOnlyAgentNodes(model, client, policy=AgentPolicyEngine())
    agent = SequentialReadOnlyAgent(nodes, ConversationStore(tmp_path / "sessions"))

    response = asyncio.run(agent.run("平台的 GPU Reservation 是什么？", "search"))

    assert [call.name for call in client.calls] == ["search_knowledge"]
    assert model.catalogs[0][0]["name"] == "search_knowledge"
    assert response.summary == "search_knowledge"


def test_agent_can_answer_without_search_knowledge(tmp_path: Path):
    model = PlannerModel(use_search=False)
    client = AgentToolClient()
    nodes = ReadOnlyAgentNodes(model, client, policy=AgentPolicyEngine())
    agent = SequentialReadOnlyAgent(nodes, ConversationStore(tmp_path / "sessions"))

    response = asyncio.run(agent.run("你好", "hello"))

    assert client.calls == []
    assert response.summary == "direct answer"
