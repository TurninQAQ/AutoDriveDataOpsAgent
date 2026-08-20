from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import AgentIntent, KnowledgeObservation, ToolCallSpec, ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.runtime import build_agent_knowledge_service
from platform_agent.settings import AgentSettings
from platform_agent.tool_client import FacadeToolClient
from platform_agent.workflow import ReadOnlyAgentNodes, SequentialReadOnlyAgent, merge_knowledge_observations, normalize_search_knowledge
from platform_core.settings import PlatformSettings
from platform_mcp.facade import PlatformMCPFacade
from platform_mcp.server import READ_ONLY_TOOL_NAMES, build_mcp_server


class SearchAndPlatformClient:
    def __init__(self, *, include_search: bool = True):
        self.include_search = include_search
        self.calls: list[ToolCallSpec] = []

    async def describe_tools(self):
        names = list(READ_ONLY_TOOL_NAMES)
        if not self.include_search:
            names.remove("search_knowledge")
        return [
            {"name": name, "description": name, "input_schema": {"type": "object"}}
            for name in names
        ]

    async def execute(self, calls):
        self.calls.extend(calls)
        result = []
        for call in calls:
            if call.name == "search_knowledge":
                data = {
                    "query": call.arguments["query"],
                    "count": 1,
                    "results": [
                        {
                            "rank": 1,
                            "source": "runbooks/gpu.md#独占",
                            "source_path": "runbooks/gpu.md",
                            "chunk_id": "gpu-exclusive",
                            "title": "GPU Reservation",
                            "section": "独占",
                            "content": "GPU Reservation 独占模式要求检查显存和活动 Reservation。",
                            "score": 0.91,
                            "lexical_score": 0.9,
                            "vector_score": 0.92,
                            "metadata": {"kind": "runbook"},
                        }
                    ],
                }
            elif call.name == "get_task_detail":
                data = {"airflow_runs": [{"state": "running"}], "priority": {"priority": 10}}
            elif call.name == "get_queue_state":
                data = {"location": "active", "position": 0}
            elif call.name == "get_gpu_pool":
                data = {"devices": [{"gpu_id": 0, "free_mb": 18000}], "reservations": []}
            else:
                data = {}
            result.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=data))
        return result


def run(coro):
    return asyncio.run(coro)


def make_agent(tmp_path: Path, client: SearchAndPlatformClient):
    nodes = ReadOnlyAgentNodes(
        HeuristicReadOnlyModel(),
        client,
        AgentPolicyEngine(max_tool_calls=6),
    )
    return SequentialReadOnlyAgent(nodes, ConversationStore(tmp_path / "sessions"))


def test_static_knowledge_uses_search_tool_not_gpu_state(tmp_path: Path):
    client = SearchAndPlatformClient()
    response = run(make_agent(tmp_path, client).run("平台的 GPU Reservation 是什么？", "knowledge"))

    assert response.intent == AgentIntent.PLATFORM_KNOWLEDGE
    assert [call.name for call in client.calls] == ["search_knowledge"]
    assert client.calls[0].arguments == {"query": "平台的 GPU Reservation 是什么？"}
    assert response.knowledge_sources == ["runbooks/gpu.md#独占"]
    assert response.retrieval_trace[0]["chunk_id"] == "gpu-exclusive"
    assert response.tool_trace[0]["tool"] == "search_knowledge"
    assert "GPU Reservation" in response.summary


def test_search_tool_result_is_normalized_into_knowledge_provenance(tmp_path: Path):
    client = SearchAndPlatformClient()
    response = run(make_agent(tmp_path, client).run("平台的 GPU Reservation 是什么？", "provenance"))

    assert response.knowledge_sources
    assert response.retrieval_trace
    assert any(item["tool"] == "search_knowledge" for item in response.tool_trace)
    assert "独占模式" in response.summary


def test_legacy_and_tool_knowledge_merge_by_chunk_id():
    legacy = KnowledgeObservation(
        chunk_id="gpu-exclusive",
        source_path="runbooks/gpu.md",
        title="GPU Reservation",
        section="独占",
        content="legacy evidence",
        score=0.8,
    )
    observation = ToolObservation(
        tool_name="search_knowledge",
        arguments={"query": "GPU"},
        ok=True,
        data={
            "results": [
                {
                    "chunk_id": "gpu-exclusive",
                    "source": "runbooks/gpu.md#独占",
                    "content": "duplicate evidence",
                    "score": 0.9,
                },
                {
                    "chunk_id": "gpu-shared",
                    "source": "platform/gpu.md#共享",
                    "content": "shared evidence",
                    "score": 0.7,
                },
            ]
        },
    )

    merged = merge_knowledge_observations([legacy], normalize_search_knowledge(observation))

    assert [item.chunk_id for item in merged] == ["gpu-exclusive", "gpu-shared"]
    assert merged[0].content == "legacy evidence"
    assert merged[1].citation == "platform/gpu.md#共享"


def test_live_task_status_prefers_operational_tools(tmp_path: Path):
    client = SearchAndPlatformClient()
    response = run(make_agent(tmp_path, client).run("release_demo 当前状态？", "status"))

    assert response.intent == AgentIntent.TASK_STATUS
    assert [call.name for call in client.calls] == ["get_task_detail", "get_queue_state"]
    assert "search_knowledge" not in [call.name for call in client.calls]


def test_gpu_diagnosis_uses_live_gpu_evidence_without_forced_rag(tmp_path: Path):
    client = SearchAndPlatformClient()
    response = run(make_agent(tmp_path, client).run("Segment 一直拿不到 GPU，帮我分析原因。", "gpu"))

    assert response.intent == AgentIntent.GPU_DIAGNOSIS
    assert [call.name for call in client.calls] == ["get_gpu_pool"]
    assert response.knowledge_sources == []


def test_greeting_has_no_tool_call(tmp_path: Path):
    client = SearchAndPlatformClient()
    response = run(make_agent(tmp_path, client).run("你好", "hello"))

    assert response.intent == AgentIntent.GENERAL_READ
    assert client.calls == []
    assert response.knowledge_sources == []


def test_write_planning_does_not_use_search_or_write_tools(tmp_path: Path):
    client = SearchAndPlatformClient()
    response = run(
        make_agent(tmp_path, client).run(
            "帮我创建 release 任务，处理 /data/release_0819，跑完整流程，优先级 5。",
            "write-boundary",
        )
    )

    assert response.intent == AgentIntent.TASK_PLANNING
    assert client.calls == []


def test_heuristic_knowledge_disabled_does_not_invent_search_tool():
    plan = run(HeuristicReadOnlyModel().plan("平台的 GPU Reservation 是什么？", [], []))

    assert plan.intent == AgentIntent.PLATFORM_KNOWLEDGE
    assert plan.tool_calls == []


def test_knowledge_enabled_flag_disables_runtime_capability(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLATFORM_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path / "runtime" / "airflow"))
    monkeypatch.setenv("AIRFLOW_STATE_DIR", str(tmp_path / "runtime" / "state"))
    monkeypatch.setenv("PLATFORM_AGENT_KNOWLEDGE_ENABLED", "0")

    settings = AgentSettings.from_env(PlatformSettings.from_env())
    assert settings.knowledge_enabled is False
    assert build_agent_knowledge_service(settings) is None


def test_facade_catalog_hides_unavailable_knowledge_tool():
    client = FacadeToolClient(PlatformMCPFacade(None, None, None, None, None, None, None, None))
    tools = run(client.describe_tools())

    assert "search_knowledge" not in {item["name"] for item in tools}


def test_mcp_server_hides_unavailable_knowledge_tool(monkeypatch):
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
    server = build_mcp_server(facade)

    assert "search_knowledge" not in server.tools


def test_v142_contract_has_no_optional_forbidden_overlap():
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / "eval" / "v1_4_2" / "rag_as_tool_cases.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 5
    for row in rows:
        assert not set(row["optional_tools"]) & set(row["forbidden_tools"])
    write_case = next(row for row in rows if row["id"] == "rag_tool_write_boundary")
    assert "search_knowledge" in write_case["forbidden_tools"]
    assert "search_knowledge" not in write_case["optional_tools"]
