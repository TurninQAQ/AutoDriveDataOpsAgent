from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
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
from platform_observability.redaction import redact_text

from .aligned import FixtureToolClient, load_jsonl
from .deepeval_adapter import PRE_CONTRACT_AUDIT_BASELINE, run_deepeval_tool_metrics
from .ragas_adapter import GENERATION_METRIC_NAMES, run_ragas_judge


def _rag_judge_model_from_env() -> str:
    explicit = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "").strip()
    if explicit:
        return explicit
    provider = os.getenv("PLATFORM_EVAL_PROVIDER", "").strip().lower()
    if provider in {"qwen", "dashscope", "aliyun", "alibaba"}:
        return "qwen-plus"
    return ""


def _safe_rag_error(exc: BaseException) -> str:
    text = redact_text(str(exc)).replace("\n", " ").strip()
    if len(text) > 240:
        text = text[:240] + "..."
    return "; ".join(item for item in (type(exc).__name__, text) if item)


def _rag_case_seed(case: dict[str, Any], settings: AgentSettings, judge_model: str) -> dict[str, Any]:
    query = str(case.get("query") or "")
    return {
        "id": case.get("id"),
        "case_id": case.get("id"),
        "query": query,
        "agent_model": settings.model,
        "judge_model": judge_model,
        "embedding_model": settings.knowledge_embedding_model,
        "retrieved_contexts": [],
        "retrieved_sources": [],
        "final_answer": "",
        "reference_answer": str(case.get("reference_answer") or ""),
        "agent_status": "PENDING",
        "evaluation_status": "PENDING",
        "failure_reason": None,
        "latency_sec": None,
        "token_usage": None,
        "agent_api_request_count": None,
    }


async def _collect_rag_samples_detailed_async(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path)
    judge_model = _rag_judge_model_from_env()
    case_rows = [_rag_case_seed(case, settings, judge_model) for case in cases]
    samples: list[dict[str, Any]] = []
    try:
        model = build_model_from_env(settings.provider, settings.model, settings.temperature)
    except Exception as exc:
        for row in case_rows:
            row.update({
                "agent_status": "AGENT_PROVIDER_FAILED",
                "evaluation_status": "BLOCKED_NOT_VALIDATED",
                "failure_reason": _safe_rag_error(exc),
            })
        return {"samples": samples, "cases": case_rows}

    try:
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
            for case, row in zip(cases, case_rows):
                query = row["query"]
                started = time.perf_counter()
                try:
                    if not query:
                        raise ValueError("RAG case query is empty")
                    response = await runtime.run(query, thread_id=f"rag-judge-{case['id']}")
                    row["agent_status"] = "PASS"
                    row["latency_sec"] = time.perf_counter() - started
                except Exception as exc:
                    row.update({
                        "agent_status": "AGENT_ERROR",
                        "evaluation_status": "BLOCKED_NOT_VALIDATED",
                        "failure_reason": _safe_rag_error(exc),
                        "latency_sec": time.perf_counter() - started,
                    })
                    continue

                try:
                    retrieval = service.search(
                        query,
                        top_k=int(case.get("top_k") or settings.knowledge_top_k),
                    ).results
                except Exception as exc:
                    row.update({
                        "agent_status": "RETRIEVAL_ERROR",
                        "evaluation_status": "BLOCKED_NOT_VALIDATED",
                        "failure_reason": _safe_rag_error(exc),
                    })
                    continue

                text = response.summary
                if response.root_cause:
                    text += "\n" + response.root_cause
                contexts = [item.content for item in retrieval]
                sources = [item.citation for item in retrieval]
                row.update({
                    "retrieved_contexts": contexts,
                    "retrieved_sources": sources,
                    "final_answer": text,
                    "evaluation_status": "COLLECTED",
                })
                samples.append({
                    "id": row["id"],
                    "case_id": row["case_id"],
                    "user_input": query,
                    "query": query,
                    "response": text,
                    "final_answer": text,
                    "reference": row["reference_answer"],
                    "reference_answer": row["reference_answer"],
                    "retrieved_contexts": contexts,
                    "retrieved_context_ids": sources,
                    "retrieved_sources": sources,
                    "agent_model": settings.model,
                    "judge_model": judge_model,
                    "embedding_model": settings.knowledge_embedding_model,
                    "agent_status": "PASS",
                    "agent_latency_sec": row["latency_sec"],
                    "token_usage": None,
                    "agent_api_request_count": None,
                })
    except Exception as exc:
        # Runtime construction or an unexpected collector failure is retained
        # per case rather than discarding already collected samples.
        for row in case_rows:
            if row["evaluation_status"] == "PENDING":
                row.update({
                    "agent_status": "AGENT_ERROR",
                    "evaluation_status": "BLOCKED_NOT_VALIDATED",
                    "failure_reason": _safe_rag_error(exc),
                })
    return {"samples": samples, "cases": case_rows}


async def _collect_rag_samples_async(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> list[dict[str, Any]]:
    detailed = await _collect_rag_samples_detailed_async(service, cases_path, settings)
    return detailed["samples"]


def collect_rag_judge_samples(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> list[dict[str, Any]]:
    return asyncio.run(_collect_rag_samples_async(service, cases_path, settings))


def collect_rag_judge_samples_detailed(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> dict[str, Any]:
    return asyncio.run(_collect_rag_samples_detailed_async(service, cases_path, settings))


def _persist_rag_collection_checkpoint(collected: dict[str, Any]) -> None:
    target = os.getenv("PLATFORM_EVAL_COLLECTION_ARTIFACT", "").strip()
    if not target:
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "COLLECTION_COMPLETE",
        "sample_count": len(collected.get("samples") or []),
        "case_count": len(collected.get("cases") or []),
        "cases": collected.get("cases") or [],
        "samples": collected.get("samples") or [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_ragas_on_agent(
    service: KnowledgeService,
    cases_path: str | Path,
    settings: AgentSettings,
) -> dict[str, Any]:
    collected = collect_rag_judge_samples_detailed(service, cases_path, settings)
    _persist_rag_collection_checkpoint(collected)
    samples = collected["samples"]
    # Retrieval quality is reported by the deterministic aligned evaluator.
    # The formal Ragas generation evaluation is limited to semantic metrics;
    # LLM-judged context metrics remain available through run_ragas_judge(...)
    # for explicitly requested diagnostics.
    result = run_ragas_judge(samples, metric_names=GENERATION_METRIC_NAMES)
    judge_rows = {str(row.get("case_id", row.get("id"))): row for row in result.get("cases", [])}
    merged_cases: list[dict[str, Any]] = []
    for row in collected["cases"]:
        case_id = str(row.get("case_id"))
        merged = dict(row)
        judged = judge_rows.get(case_id)
        if judged:
            merged.update(judged)
            merged["agent_status"] = row["agent_status"]
            merged["agent_model"] = row["agent_model"]
            merged["judge_model"] = row["judge_model"]
            merged["embedding_model"] = row["embedding_model"]
        else:
            merged.setdefault("scores", {})
            merged.setdefault("metrics", [])
            merged["status"] = row["evaluation_status"]
        merged_cases.append(merged)
    result["cases"] = merged_cases
    result["case_count"] = len(collected["cases"])
    result["sample_count"] = len(samples)
    result["agent_model"] = settings.model
    result["judge_model"] = _rag_judge_model_from_env()
    result["embedding_model"] = settings.knowledge_embedding_model
    result["self_model_evaluation"] = bool(
        result["agent_model"] and result["agent_model"] == result["judge_model"]
    )
    result["agent_success_count"] = sum(row["agent_status"] == "PASS" for row in collected["cases"])
    result["agent_failure_count"] = len(collected["cases"]) - result["agent_success_count"]
    result["complete_case_count"] = sum(row.get("status") == "PASS" for row in merged_cases)
    result["partial_case_count"] = sum(row.get("status") == "PARTIAL" for row in merged_cases)
    result["blocked_case_count"] = len(merged_cases) - result["complete_case_count"] - result["partial_case_count"]
    if result["agent_failure_count"] and result.get("status") == "PASS":
        result["status"] = "PARTIAL"
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
        "category": str(case.get("category") or ""),
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
        model = build_model_from_env(settings.provider, settings.model, settings.temperature)
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
    result = run_deepeval_tool_metrics(collect_deepeval_tool_samples(cases_path, settings))
    contract_version = Path(cases_path).parent.name
    result["contract_version"] = contract_version
    result["case_file"] = str(cases_path)
    if contract_version == "v1_3_2":
        result["pre_contract_audit_baseline"] = dict(PRE_CONTRACT_AUDIT_BASELINE)
    return result
