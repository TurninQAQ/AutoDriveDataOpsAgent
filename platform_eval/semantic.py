from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from platform_agent.memory import ConversationStore
from platform_agent.model import build_model_from_env
from platform_agent.models import AgentPlan
from platform_agent.settings import AgentSettings
from platform_agent.tool_client import InMemoryMCPToolClient
from platform_agent.workflow import build_agent_runtime
from platform_mcp.server import WRITE_TOOL_NAMES
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


def _tool_case_seed(case: dict[str, Any]) -> dict[str, Any]:
    """Create a stable output row before planner execution.

    A row is kept even when the harness cannot collect a plan.  This makes a
    collection failure observable without confusing it with a valid empty plan.
    """
    expected_arguments = case.get("expected_arguments") if isinstance(case.get("expected_arguments"), dict) else {}
    required_tools = list(case.get("required_tools") or []) if isinstance(case.get("required_tools") or [], list) else []
    actual_tools: list[str] = []
    forbidden_tools = list(case.get("forbidden_tools") or []) if isinstance(case.get("forbidden_tools") or [], list) else []
    return {
        "id": case.get("id"),
        "case_id": case.get("id"),
        "input": case.get("query", ""),
        "query": case.get("query", ""),
        "actual_output": "",
        "tools_called": [],
        "actual_tools": actual_tools,
        "actual_arguments": [],
        "required_tools": required_tools,
        "optional_tools": list(case.get("optional_tools") or []) if isinstance(case.get("optional_tools") or [], list) else [],
        "forbidden_tools": forbidden_tools,
        "forbidden_tools_called": [],
        "expected_tools": [
            {"name": name, "arguments": expected_arguments.get(name)}
            for name in required_tools
        ],
        "expected_arguments": expected_arguments,
        "expected_intent": str(case.get("expected_intent") or ""),
        "actual_intent": "",
        "intent_ok": None,
        "write_case": case.get("category") == "write",
        "write_action": None,
        "consider_ordering": bool(case.get("required_order")),
        "collection_valid": False,
        "catalog_valid": False,
        "planner_valid": False,
        "collection_error": None,
        "model_tool_miss": bool(required_tools),
    }


def _set_collection_error(sample: dict[str, Any], error: str) -> None:
    sample["collection_valid"] = False
    sample["planner_valid"] = False
    sample["collection_error"] = error


def _catalog_is_valid(tool_descriptions: Any) -> bool:
    if not isinstance(tool_descriptions, list) or not tool_descriptions:
        return False
    for tool in tool_descriptions:
        if not isinstance(tool, dict):
            return False
        if not str(tool.get("name") or "").strip():
            return False
        if not str(tool.get("description") or "").strip():
            return False
        if not isinstance(tool.get("input_schema"), dict):
            return False
    return True


def _case_schema_error(case: dict[str, Any]) -> str | None:
    query = case.get("query")
    if not isinstance(query, str) or not query.strip():
        return "case_missing_query"
    for field in ("required_tools", "optional_tools", "forbidden_tools", "required_order"):
        value = case.get(field)
        if value is not None and not isinstance(value, list):
            return f"case_invalid_{field}"
    expected_arguments = case.get("expected_arguments")
    if expected_arguments is not None and not isinstance(expected_arguments, dict):
        return "case_invalid_expected_arguments"
    expected_intent = case.get("expected_intent")
    if expected_intent is not None and not isinstance(expected_intent, str):
        return "case_invalid_expected_intent"
    return None


async def _collect_tool_samples_async(cases_path: str | Path, settings: AgentSettings) -> list[dict[str, Any]]:
    cases = load_jsonl(cases_path)
    samples = [_tool_case_seed(case) for case in cases]

    try:
        model = build_model_from_env(settings.provider, settings.model, settings.temperature, settings.base_url)
    except Exception:
        for sample in samples:
            _set_collection_error(sample, "model_build_failed")
        return samples

    # Use the same MCP server/client boundary as the production Agent.  The client
    # filters the server's definitions down to READ_ONLY_TOOL_NAMES, so evaluation
    # can ground planning without exposing write tools or executing any tool call.
    try:
        tool_descriptions = await InMemoryMCPToolClient().describe_tools()
    except Exception:
        for sample in samples:
            _set_collection_error(sample, "tool_catalog_loading_failed")
        return samples
    if not _catalog_is_valid(tool_descriptions):
        for sample in samples:
            _set_collection_error(sample, "tool_catalog_invalid")
        return samples

    catalog_names = {str(item["name"]) for item in tool_descriptions}
    for sample, case in zip(samples, cases):
        schema_error = _case_schema_error(case)
        if schema_error:
            _set_collection_error(sample, schema_error)
            continue
        query = case["query"]
        try:
            plan = await model.plan(query, tool_descriptions, [])
            if not isinstance(plan, AgentPlan):
                raise TypeError("model_plan_not_agent_plan")
            actual_arguments = [
                {"name": call.name, "arguments": dict(call.arguments or {})}
                for call in plan.tool_calls
            ]
            actual_tools = [item["name"] for item in actual_arguments]
            required_tools = set(sample["required_tools"])
            actual_tool_set = set(actual_tools)
            forbidden_tools = set(sample["forbidden_tools"]) | set(WRITE_TOOL_NAMES)
            sample.update({
                "input": query,
                "query": query,
                "actual_output": plan.decision_summary,
                "tools_called": actual_arguments,
                "actual_tools": actual_tools,
                "actual_arguments": actual_arguments,
                "forbidden_tools_called": sorted(actual_tool_set & forbidden_tools),
                "actual_intent": plan.intent.value,
                "intent_ok": not sample["expected_intent"] or plan.intent.value == sample["expected_intent"],
                "write_action": plan.write_action,
                "collection_valid": True,
                "catalog_valid": True,
                "planner_valid": True,
                "collection_error": None,
                # An empty tool_calls list is a valid planner result.  This flag
                # is a model outcome, never a collection-health judgment.
                "model_tool_miss": bool(required_tools - actual_tool_set),
            })
            # Keep the catalog available in debug output without treating a model
            # selecting an unknown name as a harness exception.
            sample["catalog_tool_names"] = sorted(catalog_names)
        except Exception:
            _set_collection_error(sample, "model_plan_failed")
    return samples


def collect_deepeval_tool_samples(cases_path: str | Path, settings: AgentSettings) -> list[dict[str, Any]]:
    return asyncio.run(_collect_tool_samples_async(cases_path, settings))


def run_deepeval_on_agent(cases_path: str | Path, settings: AgentSettings) -> dict[str, Any]:
    return run_deepeval_tool_metrics(collect_deepeval_tool_samples(cases_path, settings))
