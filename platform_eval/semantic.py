from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from platform_agent.memory import ConversationStore
from platform_agent.model import build_model_from_env
from platform_agent.settings import AgentSettings
from platform_agent.tool_client import InMemoryMCPToolClient
from platform_agent.workflow import build_agent_runtime
from platform_planning.service import TaskPlanningService
from platform_rag.service import AsyncKnowledgeRetriever, KnowledgeService

from .aligned import FixtureToolClient, load_jsonl
from .deepeval_adapter import run_deepeval_tool_metrics
from .ragas_adapter import run_ragas_judge


async def _collect_rag_samples_async(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> list[dict[str, Any]]:
    cases = load_jsonl(cases_path)
    model = build_model_from_env(settings.provider, settings.model, settings.temperature, settings.base_url)
    samples = []
    with tempfile.TemporaryDirectory(prefix="v11_rag_judge_memory_") as td:
        runtime = build_agent_runtime(
            "sequential",
            model,
            FixtureToolClient({}),
            ConversationStore(td),
            max_tool_calls=settings.max_tool_calls,
            knowledge_retriever=AsyncKnowledgeRetriever(service, enabled=True),
            knowledge_top_k=settings.knowledge_top_k,
            task_planning_service=TaskPlanningService.from_env(),
        )
        for case in cases:
            query = str(case["query"])
            response = await runtime.run(query, thread_id=f"rag-judge-{case['id']}")
            retrieval = service.search(query, top_k=int(case.get("top_k") or settings.knowledge_top_k)).results
            text = response.summary
            if response.root_cause:
                text += "\n" + response.root_cause
            samples.append({
                "id": case.get("id"),
                "user_input": query,
                "response": text,
                "reference": str(case.get("reference_answer") or ""),
                "retrieved_contexts": [item.content for item in retrieval],
                "retrieved_context_ids": [item.citation for item in retrieval],
            })
    return samples


def collect_rag_judge_samples(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> list[dict[str, Any]]:
    return asyncio.run(_collect_rag_samples_async(service, cases_path, settings))


def run_ragas_on_agent(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> dict[str, Any]:
    samples = collect_rag_judge_samples(service, cases_path, settings)
    result = run_ragas_judge(samples)
    result["sample_count"] = len(samples)
    return result


async def _collect_tool_samples_async(cases_path: str | Path, settings: AgentSettings) -> list[dict[str, Any]]:
    cases = load_jsonl(cases_path)
    model = build_model_from_env(settings.provider, settings.model, settings.temperature, settings.base_url)
    # Use the same MCP server/client boundary as the production Agent.  The client
    # filters the server's definitions down to READ_ONLY_TOOL_NAMES, so evaluation
    # can ground planning without exposing write tools or executing any tool call.
    tool_descriptions = await InMemoryMCPToolClient().describe_tools()
    samples = []
    for case in cases:
        plan = await model.plan(str(case["query"]), tool_descriptions, [])
        expected_arguments = case.get("expected_arguments") or {}
        actual_arguments = [
            {"name": call.name, "arguments": call.arguments}
            for call in plan.tool_calls
        ]
        samples.append({
            "id": case.get("id"),
            "case_id": case.get("id"),
            "input": case["query"],
            "query": case["query"],
            "actual_output": plan.decision_summary,
            "tools_called": actual_arguments,
            "actual_tools": [item["name"] for item in actual_arguments],
            "actual_arguments": actual_arguments,
            "required_tools": list(case.get("required_tools") or []),
            "optional_tools": list(case.get("optional_tools") or []),
            "forbidden_tools": list(case.get("forbidden_tools") or []),
            "expected_tools": [
                {"name": name, "arguments": expected_arguments.get(name)}
                for name in case.get("required_tools") or []
            ],
            "consider_ordering": bool(case.get("required_order")),
        })
    return samples


def collect_deepeval_tool_samples(cases_path: str | Path, settings: AgentSettings) -> list[dict[str, Any]]:
    return asyncio.run(_collect_tool_samples_async(cases_path, settings))


def run_deepeval_on_agent(cases_path: str | Path, settings: AgentSettings) -> dict[str, Any]:
    return run_deepeval_tool_metrics(collect_deepeval_tool_samples(cases_path, settings))
