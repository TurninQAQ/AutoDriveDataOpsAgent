from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import ToolCallSpec, ToolObservation
from platform_agent.verification import ActionVerifier
from platform_agent.workflow import build_agent_runtime
from platform_mcp.server import WRITE_TOOL_NAMES
from platform_planning.evaluation import evaluate_task_planning
from platform_planning.service import TaskPlanningService


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


class SnapshotVerifier(ActionVerifier):
    def __init__(self):
        # _evaluate is dependency-light and does not use tool_client.
        super().__init__(FixtureToolClient({}), attempts=1, interval_sec=0)

    def evaluate_fixture(self, case: dict[str, Any]):
        return self._evaluate(
            case["action"],
            case.get("arguments") or {},
            case.get("execution_result") or {},
            case.get("baseline") or {},
            case.get("snapshot") or {},
            1,
        )


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tools(plan) -> list[str]:
    return [item.name for item in plan.tool_calls]


async def _evaluate_async(cases: dict[str, Any]) -> dict[str, Any]:
    model = HeuristicReadOnlyModel()

    intent_rows = []
    intent_hits = 0
    tool_hits = 0
    for case in cases.get("intent_tool_cases") or []:
        plan = await model.plan(case["query"], [], [])
        expected_tools = list(case.get("expected_tools") or [])
        actual_tools = _tools(plan)
        intent_ok = plan.intent.value == case["expected_intent"]
        tool_ok = actual_tools == expected_tools
        intent_hits += int(intent_ok)
        tool_hits += int(tool_ok)
        intent_rows.append({
            "id": case["id"], "intent_ok": intent_ok, "tool_ok": tool_ok,
            "expected_intent": case["expected_intent"], "actual_intent": plan.intent.value,
            "expected_tools": expected_tools, "actual_tools": actual_tools,
        })

    diagnosis_rows = []
    diagnosis_hits = 0
    for case in cases.get("diagnosis_cases") or []:
        client = FixtureToolClient(case.get("tool_results") or {})
        with tempfile.TemporaryDirectory(prefix="agent_eval_memory_") as td:
            agent = build_agent_runtime(
                "sequential", model, client, ConversationStore(td), max_tool_calls=6,
                knowledge_retriever=None, task_planning_service=TaskPlanningService.from_env(),
            )
            response = await agent.run(case["query"], thread_id=f"eval-{case['id']}")
        root = response.root_cause or ""
        needles = [str(item) for item in case.get("expected_root_cause_contains") or []]
        ok = all(item.lower() in root.lower() for item in needles)
        diagnosis_hits += int(ok)
        diagnosis_rows.append({"id": case["id"], "ok": ok, "root_cause": root, "expected_contains": needles})

    safety_rows = []
    unsafe = 0
    for case in cases.get("safety_cases") or []:
        plan = await model.plan(case["query"], [], [])
        direct_write_tools = [name for name in _tools(plan) if name in WRITE_TOOL_NAMES]
        expected_intent_ok = plan.intent.value == case["expected_intent"]
        write_action_ok = bool(plan.write_action) == bool(case.get("expected_write_action"))
        unsafe_case = bool(direct_write_tools)
        unsafe += int(unsafe_case)
        safety_rows.append({
            "id": case["id"], "ok": expected_intent_ok and write_action_ok and not unsafe_case,
            "intent": plan.intent.value, "direct_write_tools": direct_write_tools,
            "has_write_action": bool(plan.write_action),
        })

    verifier = SnapshotVerifier()
    verification_rows = []
    verification_hits = 0
    for case in cases.get("verification_cases") or []:
        result = verifier.evaluate_fixture(case)
        actual = result.verified
        expected = bool(case["expected_verified"])
        ok = actual == expected
        verification_hits += int(ok)
        verification_rows.append({
            "id": case["id"], "ok": ok, "expected_verified": expected,
            "actual_verified": actual, "status": result.status,
            "failed_checks": [item.name for item in result.checks if not item.passed],
        })

    intent_count = len(intent_rows)
    diagnosis_count = len(diagnosis_rows)
    safety_count = len(safety_rows)
    verification_count = len(verification_rows)
    return {
        "intent_accuracy": intent_hits / intent_count if intent_count else 1.0,
        "tool_selection_accuracy": tool_hits / intent_count if intent_count else 1.0,
        "diagnosis_accuracy": diagnosis_hits / diagnosis_count if diagnosis_count else 1.0,
        "unsafe_action_rate": unsafe / safety_count if safety_count else 0.0,
        "verification_accuracy": verification_hits / verification_count if verification_count else 1.0,
        "rows": {
            "intent_tool": intent_rows,
            "diagnosis": diagnosis_rows,
            "safety": safety_rows,
            "verification": verification_rows,
        },
    }


def evaluate_agent_suite(
    cases_path: str | Path,
    planning_cases_path: str | Path,
    planning_service: TaskPlanningService | None = None,
) -> dict[str, Any]:
    cases = _load(cases_path)
    result = asyncio.run(_evaluate_async(cases))
    planning = evaluate_task_planning(planning_service or TaskPlanningService.from_env(), planning_cases_path)
    result["task_planning_accuracy"] = float(planning["case_accuracy"])
    result["planning"] = planning
    metrics = [
        result["intent_accuracy"],
        result["tool_selection_accuracy"],
        result["diagnosis_accuracy"],
        1.0 - result["unsafe_action_rate"],
        result["task_planning_accuracy"],
        result["verification_accuracy"],
    ]
    result["overall_score"] = sum(metrics) / len(metrics)
    return result
