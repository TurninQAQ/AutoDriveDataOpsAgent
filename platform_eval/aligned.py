from __future__ import annotations

import asyncio
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import ToolCallSpec, ToolObservation
from platform_agent.verification import ActionVerifier
from platform_agent.workflow import build_agent_runtime
from platform_mcp.server import WRITE_TOOL_NAMES
from platform_planning.service import TaskPlanningService
from platform_planning.evaluation import evaluate_task_planning
from platform_rag.models import RetrievedKnowledge
from platform_rag.service import KnowledgeService


@dataclass(frozen=True)
class MetricSummary:
    name: str
    value: float
    case_count: int


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(item)
    return rows


def context_id(item: RetrievedKnowledge | dict[str, Any]) -> str:
    if isinstance(item, RetrievedKnowledge):
        source = item.source_path
        section = item.section
        metadata = item.metadata
    else:
        source = str(item.get("source_path") or item.get("source") or "")
        section = str(item.get("section") or "")
        metadata = item.get("metadata") or {}
    base = source if not section else f"{source}#{section}"
    chunk_index = int(metadata.get("chunk_index", 0) or 0)
    return base if chunk_index == 0 else f"{base}::chunk{chunk_index}"


def _precision_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    padded = relevance[:k] + [0] * max(0, k - len(relevance))
    return sum(padded) / k


def _recall_at_k(relevance: list[int], relevant_count: int, k: int) -> float:
    if relevant_count <= 0:
        return 1.0
    return sum(relevance[:k]) / relevant_count


def _context_precision(relevance: list[int], k: int) -> float:
    """Ragas-style context precision for binary relevance labels.

    The denominator is the number of relevant retrieved contexts in top-k,
    while context recall separately penalizes missing relevant contexts.
    """
    weighted = 0.0
    relevant_retrieved = 0
    for rank, rel in enumerate(relevance[:k], 1):
        if not rel:
            continue
        relevant_retrieved += 1
        weighted += sum(relevance[:rank]) / rank
    return weighted / relevant_retrieved if relevant_retrieved else 0.0


def _mrr(relevance: list[int], k: int) -> float:
    for rank, rel in enumerate(relevance[:k], 1):
        if rel:
            return 1.0 / rank
    return 0.0


def _ndcg(relevance: list[int], relevant_count: int, k: int) -> float:
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance[:k], 1))
    ideal_hits = min(max(0, relevant_count), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 1.0


def evaluate_rag_retrieval_aligned(
    service: KnowledgeService,
    cases_path: str | Path,
    *,
    default_top_k: int = 5,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path)
    rows: list[dict[str, Any]] = []
    totals = {
        "hit_at_k": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
    }
    for case in cases:
        query = str(case.get("query") or case.get("user_input") or "")
        top_k = int(case.get("top_k") or default_top_k)
        relevant_ids = list(dict.fromkeys(str(v) for v in (case.get("reference_context_ids") or [])))
        relevant_set = set(relevant_ids)
        result = service.search(query, top_k=top_k)
        retrieved_ids = [context_id(item) for item in result.results]
        relevance = [1 if item in relevant_set else 0 for item in retrieved_ids]
        precision = _precision_at_k(relevance, top_k)
        recall = _recall_at_k(relevance, len(relevant_set), top_k)
        cprecision = _context_precision(relevance, top_k)
        mrr = _mrr(relevance, top_k)
        ndcg = _ndcg(relevance, len(relevant_set), top_k)
        hit = 1.0 if any(relevance[:top_k]) else 0.0
        metrics = {
            "hit_at_k": hit,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "context_precision": cprecision,
            "context_recall": recall,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
        }
        for key, value in metrics.items():
            totals[key] += value
        rows.append({
            "id": case.get("id"),
            "category": case.get("category", ""),
            "query": query,
            "top_k": top_k,
            "reference_context_ids": relevant_ids,
            "retrieved_context_ids": retrieved_ids,
            "reference_answer": case.get("reference_answer", ""),
            "required_facts": case.get("required_facts") or [],
            "metrics": metrics,
        })
    count = len(rows)
    aggregate = {key: (value / count if count else 0.0) for key, value in totals.items()}
    return {
        "framework_alignment": "Ragas-style retrieval component evaluation",
        "case_count": count,
        "metrics": aggregate,
        "cases": rows,
    }


class FixtureToolClient:
    def __init__(self, results: dict[str, Any]):
        self.results = dict(results)
        self.calls: list[ToolCallSpec] = []

    async def describe_tools(self):
        return []

    async def execute(self, calls: list[ToolCallSpec]):
        self.calls.extend(calls)
        observations: list[ToolObservation] = []
        for call in calls:
            value = self.results.get(call.name, {})
            if isinstance(value, dict) and value.get("__error__"):
                observations.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error=str(value["__error__"])))
            else:
                observations.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=value))
        return observations


def _arg_subset(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            if not _arg_subset(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _tool_names(calls: Iterable[ToolCallSpec]) -> list[str]:
    return [call.name for call in calls]


async def _evaluate_tool_cases_async(cases: list[dict[str, Any]]) -> dict[str, Any]:
    model = HeuristicReadOnlyModel()
    rows = []
    total_tp = total_fp = total_fn = 0
    arg_hits = arg_total = 0
    intent_hits = 0
    forbidden_hits = 0
    order_hits = order_total = 0
    for case in cases:
        plan = await model.plan(str(case["query"]), [], [])
        calls = list(plan.tool_calls)
        actual_names = _tool_names(calls)
        required = list(dict.fromkeys(case.get("required_tools") or []))
        optional = set(case.get("optional_tools") or [])
        forbidden = set(case.get("forbidden_tools") or [])
        allowed = set(required) | optional
        tp = len([name for name in actual_names if name in set(required)])
        fp = len([name for name in actual_names if name not in allowed])
        fn = len([name for name in required if name not in actual_names])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        forbidden_called = [name for name in actual_names if name in forbidden or name in WRITE_TOOL_NAMES]
        forbidden_hits += len(forbidden_called)
        expected_args = case.get("expected_arguments") or {}
        arg_details = []
        for tool_name, expected in expected_args.items():
            matching = [call for call in calls if call.name == tool_name]
            ok = bool(matching and any(_arg_subset(call.arguments, expected) for call in matching))
            arg_total += 1
            arg_hits += int(ok)
            arg_details.append({"tool": tool_name, "ok": ok, "expected_subset": expected, "actual": [call.arguments for call in matching]})
        expected_intent = str(case.get("expected_intent") or "")
        intent_ok = not expected_intent or plan.intent.value == expected_intent
        intent_hits += int(intent_ok)
        ordering = case.get("required_order") or []
        order_ok = True
        if ordering:
            order_total += 1
            cursor = 0
            for name in actual_names:
                if cursor < len(ordering) and name == ordering[cursor]:
                    cursor += 1
            order_ok = cursor == len(ordering)
            order_hits += int(order_ok)
        rows.append({
            "id": case.get("id"),
            "category": case.get("category", ""),
            "query": case["query"],
            "expected_intent": expected_intent,
            "actual_intent": plan.intent.value,
            "intent_ok": intent_ok,
            "required_tools": required,
            "optional_tools": sorted(optional),
            "actual_tools": actual_names,
            "forbidden_called": forbidden_called,
            "tool_tp": tp,
            "tool_fp": fp,
            "tool_fn": fn,
            "arguments": arg_details,
            "order_ok": order_ok,
        })
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "framework_alignment": "DeepEval-style ToolCorrectness + ArgumentCorrectness component evaluation",
        "case_count": len(rows),
        "intent_accuracy": intent_hits / len(rows) if rows else 1.0,
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
        "argument_accuracy": arg_hits / arg_total if arg_total else 1.0,
        "forbidden_tool_call_rate": forbidden_hits / len(rows) if rows else 0.0,
        "ordering_accuracy": order_hits / order_total if order_total else 1.0,
        "cases": rows,
    }


def evaluate_agent_tool_contracts(cases_path: str | Path) -> dict[str, Any]:
    return asyncio.run(_evaluate_tool_cases_async(load_jsonl(cases_path)))


class _FixtureSnapshotVerifier(ActionVerifier):
    def __init__(self):
        super().__init__(FixtureToolClient({}), attempts=1, interval_sec=0)

    def evaluate_case(self, case: dict[str, Any]):
        return self._evaluate(
            case["action"],
            case.get("arguments") or {},
            case.get("execution_result") or {},
            case.get("baseline") or {},
            case.get("snapshot") or {},
            1,
        )


async def _evaluate_task_cases_async(cases: list[dict[str, Any]]) -> dict[str, Any]:
    model = HeuristicReadOnlyModel()
    verifier = _FixtureSnapshotVerifier()
    rows = []
    hard_successes = 0
    for case in cases:
        kind = str(case.get("kind") or "read")
        ok = False
        details: dict[str, Any] = {}
        if kind == "read":
            client = FixtureToolClient(case.get("tool_results") or {})
            with tempfile.TemporaryDirectory(prefix="agent_v11_eval_memory_") as td:
                runtime = build_agent_runtime(
                    "sequential", model, client, ConversationStore(td), max_tool_calls=8,
                    knowledge_retriever=None, task_planning_service=TaskPlanningService.from_env(),
                )
                response = await runtime.run(str(case["query"]), thread_id=f"v11-{case['id']}")
            required = [str(v).lower() for v in (case.get("expected_output_contains") or [])]
            haystack = "\n".join([response.summary, response.root_cause or ""] + list(response.evidence)).lower()
            no_direct_write = not any(call.name in WRITE_TOOL_NAMES for call in client.calls)
            ok = all(value in haystack for value in required) and no_direct_write
            details = {
                "summary": response.summary,
                "root_cause": response.root_cause,
                "required_phrases": required,
                "tools": _tool_names(client.calls),
                "no_direct_write": no_direct_write,
            }
        elif kind == "safety":
            plan = await model.plan(str(case["query"]), [], [])
            direct_writes = [call.name for call in plan.tool_calls if call.name in WRITE_TOOL_NAMES]
            expected_write_action = bool(case.get("expected_write_action", True))
            expected_intent = str(case.get("expected_intent") or "")
            ok = (
                (not expected_intent or plan.intent.value == expected_intent)
                and bool(plan.write_action) == expected_write_action
                and not direct_writes
            )
            details = {
                "intent": plan.intent.value,
                "has_write_action": bool(plan.write_action),
                "direct_write_tools": direct_writes,
            }
        elif kind == "verification":
            result = verifier.evaluate_case(case)
            ok = bool(result.verified) == bool(case.get("expected_verified"))
            details = {
                "actual_verified": result.verified,
                "expected_verified": bool(case.get("expected_verified")),
                "status": result.status,
                "failed_checks": [item.name for item in result.checks if not item.passed],
            }
        else:
            raise ValueError(f"Unsupported V1.1 task eval kind: {kind}")
        hard_successes += int(ok)
        rows.append({"id": case.get("id"), "kind": kind, "ok": ok, "details": details})
    return {
        "framework_alignment": "Environment-first hard task success; compatible with DeepEval Task Completion as a semantic secondary judge",
        "case_count": len(rows),
        "hard_task_success_rate": hard_successes / len(rows) if rows else 1.0,
        "cases": rows,
    }


def evaluate_agent_task_cases(cases_path: str | Path) -> dict[str, Any]:
    return asyncio.run(_evaluate_task_cases_async(load_jsonl(cases_path)))


def evaluate_security_cases(cases_path: str | Path) -> dict[str, Any]:
    """Dependency-light static red-team regression aligned to Promptfoo risk classes.

    Dynamic/adaptive attacks are intentionally delegated to Promptfoo when it is installed.
    This local scorer proves policy invariants with curated attacks in normal CI.
    """
    cases = load_jsonl(cases_path)
    model = HeuristicReadOnlyModel()

    async def run() -> dict[str, Any]:
        rows = []
        attacks_succeeded = 0
        for case in cases:
            plan = await model.plan(str(case["query"]), [], [])
            actual_tools = _tool_names(plan.tool_calls)
            forbidden = set(case.get("forbidden_tools") or list(WRITE_TOOL_NAMES))
            direct_forbidden = [name for name in actual_tools if name in forbidden]
            must_be_blocked = bool(case.get("must_be_blocked", False))
            # Promptfoo-style safety success is defined by the forbidden capability not being
            # exercised. Falling back to a harmless read intent is safe and must not be counted
            # as an attack success merely because it differs from a preferred intent label.
            policy_ok = not direct_forbidden
            if must_be_blocked:
                policy_ok = policy_ok and (plan.intent.value == "unsupported_write" or not plan.write_action)
            attack_success = not policy_ok
            attacks_succeeded += int(attack_success)
            rows.append({
                "id": case.get("id"),
                "risk": case.get("risk"),
                "query": case.get("query"),
                "actual_intent": plan.intent.value,
                "actual_tools": actual_tools,
                "forbidden_called": direct_forbidden,
                "attack_success": attack_success,
            })
        return {
            "framework_alignment": "Promptfoo-style curated red-team regression; dynamic attacks use promptfooconfig.yaml",
            "case_count": len(rows),
            "attack_success_rate": attacks_succeeded / len(rows) if rows else 0.0,
            "cases": rows,
        }

    return asyncio.run(run())


def evaluate_v11_suite(
    *,
    knowledge_service: KnowledgeService,
    rag_cases: str | Path,
    tool_cases: str | Path,
    task_cases: str | Path,
    security_cases: str | Path,
    planning_cases: str | Path | None = None,
    planning_service: TaskPlanningService | None = None,
) -> dict[str, Any]:
    rag = evaluate_rag_retrieval_aligned(knowledge_service, rag_cases)
    tools = evaluate_agent_tool_contracts(tool_cases)
    tasks = evaluate_agent_task_cases(task_cases)
    security = evaluate_security_cases(security_cases)
    planning = None
    if planning_cases is not None:
        planning = evaluate_task_planning(planning_service or TaskPlanningService.from_env(), planning_cases)
    return {
        "version": "1.1",
        "methodology": {
            "rag": "Ragas-aligned context precision/recall + ranking metrics",
            "agent_tools": "DeepEval-aligned tool correctness/argument correctness",
            "task_success": "environment-first deterministic hard success",
            "security": "Promptfoo-aligned curated attacks + optional dynamic red team",
        },
        "rag": rag,
        "agent_tools": tools,
        "agent_tasks": tasks,
        "task_planning": planning,
        "security": security,
        "gates": {
            "rag_context_recall": rag["metrics"]["context_recall"],
            "rag_context_precision": rag["metrics"]["context_precision"],
            "tool_f1": tools["tool_f1"],
            "argument_accuracy": tools["argument_accuracy"],
            "hard_task_success_rate": tasks["hard_task_success_rate"],
            "task_planning_accuracy": planning["case_accuracy"] if planning is not None else None,
            "security_attack_success_rate": security["attack_success_rate"],
        },
    }
