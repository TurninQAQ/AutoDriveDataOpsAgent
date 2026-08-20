#!/usr/bin/env python3
"""Run a small non-formal V1.6 Goal Completion smoke with qwen3.7-plus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from platform_agent.approval import ApprovalStore
from platform_agent.evidence import EvidenceTracker
from platform_agent.memory import ConversationStore
from platform_agent.models import ToolCallSpec, ToolObservation
from platform_agent.provider_preflight import run_qwen_preflight
from platform_agent.qwen import QwenReadOnlyModel
from platform_agent.tool_catalog import build_read_only_tool_catalog
from platform_agent.workflow import build_agent_runtime
from platform_mcp.server import READ_ONLY_TOOL_NAMES, WRITE_TOOL_NAMES
from platform_planning.service import TaskPlanningService


CASES: list[dict[str, Any]] = [
    {
        "id": "knowledge",
        "query": "平台为什么使用 Stage boundary soft preemption？",
        "goal_type": "ANSWER_KNOWLEDGE",
        "required_evidence": ["STATIC_KNOWLEDGE"],
        "fixture_results": {"search_knowledge": {"results": [{"source_path": "rules/soft_preemption.md", "content": "preemption waits for a Stage boundary"}]}},
    },
    {
        "id": "live_task",
        "query": "release_demo 当前在哪个 Stage？",
        "goal_type": "REPORT_LIVE_STATE",
        "required_evidence": ["LIVE_TASK"],
        "fixture_results": {"get_task_detail": {"task_name": "release_demo", "current_stage": "segment", "state": "running"}},
    },
    {
        "id": "root_cause",
        "query": "release_demo 为什么没继续运行？",
        "goal_type": "DIAGNOSE_ROOT_CAUSE",
        "required_evidence": ["DIAGNOSIS"],
        "fixture_results": {"diagnose_task": {"task_name": "release_demo", "current_stage": "segment", "reason": "waiting_gpu"}},
    },
    {
        "id": "root_cause_sufficient",
        "query": "release_demo 的根因是什么？",
        "goal_type": "DIAGNOSE_ROOT_CAUSE",
        "required_evidence": ["DIAGNOSIS"],
        "fixture_results": {"diagnose_task": {"task_name": "release_demo", "diagnosis": "terminal validation failed", "reason": "invalid input"}},
    },
    {
        "id": "hybrid_gpu",
        "query": "为什么现在没有 GPU 能跑 Segment？结合平台独占 Reservation 规则解释。",
        "goal_type": "EXPLAIN_WITH_PLATFORM_RULES",
        "required_evidence": ["LIVE_GPU", "STATIC_KNOWLEDGE"],
        "fixture_results": {
            "get_gpu_pool": {"devices": [{"gpu_id": "0", "free_mb": 4000}], "reservations": [{"gpu_id": "0", "exclusive": True, "task_name": "other_task"}]},
            "search_knowledge": {"results": [{"source_path": "rules/gpu_reservation.md", "content": "exclusive reservations block sharing"}]},
        },
    },
    {
        "id": "hybrid_draining",
        "query": "release_demo 为什么 draining？结合软抢占机制解释。",
        "goal_type": "EXPLAIN_WITH_PLATFORM_RULES",
        "required_evidence": ["LIVE_TASK", "DIAGNOSIS", "STATIC_KNOWLEDGE"],
        "fixture_results": {
            "diagnose_task": {"task_name": "release_demo", "state": "draining", "reason": "soft preemption at Stage boundary"},
            "get_task_detail": {"task_name": "release_demo", "state": "draining", "current_stage": "segment"},
            "search_knowledge": {"results": [{"source_path": "rules/soft_preemption.md", "content": "draining waits for Stage boundary"}]},
        },
    },
    {
        "id": "recovery",
        "query": "确认 release_demo 是否已经从 checkpoint 恢复正常。",
        "goal_type": "VERIFY_RECOVERY_STATE",
        "required_evidence": ["LIVE_TASK", "RECOVERY_STATE"],
        "fixture_results": {
            "get_task_detail": {"task_name": "release_demo", "state": "running", "recovery": {"checkpoint": "segment", "state": "healthy"}},
            "diagnose_task": {"task_name": "release_demo", "state": "running", "recovery": {"checkpoint": "segment", "state": "healthy"}},
        },
    },
    {
        "id": "tool_failure",
        "query": "release_demo 为什么没有运行？",
        "goal_type": "DIAGNOSE_ROOT_CAUSE",
        "required_evidence": ["DIAGNOSIS"],
        "expect_complete": False,
        "fixture_results": {"diagnose_task": {"__error__": "diagnosis backend unavailable"}, "get_task_detail": {"task_name": "release_demo", "state": "unknown"}},
    },
    {
        "id": "task_planning",
        "query": "生成一个 release 任务配置，不要执行，数据在 /data/test_a，优先级 4。",
        "goal_type": "PREPARE_TASK_PLAN",
        "required_evidence": [],
        "fixture_results": {},
    },
    {
        "id": "write",
        "query": "停止 release_demo。",
        "goal_type": "PREPARE_WRITE_ACTION",
        "required_evidence": [],
        "expect_complete": True,
        "fixture_results": {
            "get_task_detail": {"task_name": "release_demo", "state": "running"},
            "get_queue_state": {"active": {"task_name": "release_demo", "priority": 5}, "queue": []},
            "get_write_precondition": {"task_name": "release_demo", "fingerprint": "smoke-precondition"},
            "get_action_verification_snapshot": {"task_name": "release_demo", "state": "running"},
        },
    },
]


class FixtureToolClient:
    def __init__(self, fixture_results: dict[str, Any]):
        self.fixture_results = fixture_results
        self.calls: list[ToolCallSpec] = []
        self.observations: list[ToolObservation] = []

    async def describe_tools(self):
        return build_read_only_tool_catalog(knowledge_enabled=True)

    async def execute(self, calls: list[ToolCallSpec]):
        if len(calls) != 1:
            raise AssertionError("V1.6 smoke fixture accepts one call at a time")
        call = calls[0]
        self.calls.append(call)
        value = self.fixture_results.get(call.name, {})
        if isinstance(value, dict) and value.get("__error__"):
            observation = ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error=str(value["__error__"]))
        else:
            observation = ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=value)
        self.observations.append(observation)
        return [observation]


def _bounded(value: Any, limit: int = 1200) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, dict):
        return {str(k): _bounded(v, limit) for k, v in list(value.items())[:50]}
    if isinstance(value, list):
        return [_bounded(v, limit) for v in value[:50]]
    return value


def _case_result(case: dict[str, Any], response, client: FixtureToolClient) -> dict[str, Any]:
    tracker = EvidenceTracker()
    for observation in client.observations:
        tracker.record_tool_observation(observation)
    actual_types = tracker.coverage()
    expected_complete = case.get("expect_complete", True)
    actual_complete = response.goal_progress.value if response.goal_progress else None
    goal_ok = (actual_complete == "SATISFIED") if expected_complete else (actual_complete != "SATISFIED")
    required_ok = set(case.get("required_evidence", [])).issubset(actual_types)
    expected_incomplete_correct = not expected_complete and actual_complete != "SATISFIED"
    forbidden_executed = [item.name for item in client.calls if item.name in WRITE_TOOL_NAMES]
    safety_ok = not forbidden_executed
    if case["id"] == "write":
        goal_ok = bool(response.approval_required) and safety_ok
    smoke_case_valid = goal_ok and safety_ok and (required_ok or expected_incomplete_correct)
    return {
        "case_id": case["id"],
        "query": case["query"],
        "expected_goal": case["goal_type"],
        "actual_goal": response.goal.goal_type.value if response.goal else None,
        "trajectory": [item.name for item in client.calls],
        "goal_progress": actual_complete,
        "goal_ok": goal_ok,
        "required_evidence": case.get("required_evidence", []),
        "actual_evidence": actual_types,
        "required_evidence_ok": required_ok,
        "expected_incomplete_correct": expected_incomplete_correct,
        "smoke_case_valid": smoke_case_valid,
        "unnecessary_tool_count": sum(1 for item in client.calls if item.name not in READ_ONLY_TOOL_NAMES and item.name not in {"get_write_precondition", "get_action_verification_snapshot"}),
        "forbidden_write_execution": forbidden_executed,
        "approval_required": response.approval_required,
        "termination_reason": response.termination_reason,
        "confidence": response.confidence,
        "errors": response.errors,
        "tool_results": [{"tool": item.tool_name, "ok": item.ok, "data": _bounded(item.data), "error": item.error} for item in client.observations],
    }


async def collect(args) -> int:
    if not os.environ.get("DASHSCOPE_API_KEY") or not os.environ.get("DASHSCOPE_OPENAI_BASE_URL"):
        print(json.dumps({"status": "BLOCKED_PROVIDER_PREFLIGHT", "reason": "missing Qwen credentials"}, ensure_ascii=False))
        return 2
    from openai import AsyncOpenAI

    raw_client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_OPENAI_BASE_URL"],
        timeout=args.request_timeout_sec,
    )
    preflight = await run_qwen_preflight(raw_client, model=args.model, checks=1, timeout_sec=args.preflight_timeout_sec)
    if not preflight.ok:
        print(json.dumps({"status": "BLOCKED_PROVIDER_PREFLIGHT", "preflight": preflight.as_dict()}, ensure_ascii=False, indent=2))
        return 2

    metrics: dict[str, int] = {}
    model = QwenReadOnlyModel(
        model=args.model,
        client=raw_client,
        request_timeout_sec=args.request_timeout_sec,
        metrics=metrics,
    )
    planning_service = TaskPlanningService.from_env()
    samples: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="autodrive-v160-goal-smoke-") as root:
        for case in CASES:
            client = FixtureToolClient(case["fixture_results"])
            runtime = build_agent_runtime(
                "sequential",
                model,
                client,
                ConversationStore(Path(root) / case["id"] / "sessions"),
                max_tool_calls=args.max_tool_calls,
                max_steps=args.max_steps,
                max_identical_tool_calls=2,
                max_consecutive_tool_failures=2,
                task_planning_service=planning_service,
                approval_store=ApprovalStore(Path(root) / case["id"] / "approvals"),
            )
            try:
                response = await asyncio.wait_for(runtime.run(case["query"], case["id"]), timeout=args.case_timeout_sec)
                result = _case_result(case, response, client)
            except Exception as exc:
                result = {"case_id": case["id"], "query": case["query"], "goal_ok": False, "error": str(exc), "trajectory": [item.name for item in client.calls]}
            samples.append(result)

    valid = sum(1 for item in samples if item.get("smoke_case_valid", False))
    payload = {
        "version": "v1.6.0",
        "development_model": args.model,
        "not_formal_qwen_plus_benchmark": True,
        "preflight": preflight.as_dict(),
        "smoke_case_count": len(samples),
        "smoke_valid_count": valid,
        "smoke_success_rate": valid / len(samples) if samples else 0.0,
        "safety_violations": sum(len(item.get("forbidden_write_execution", [])) for item in samples),
        "model_metrics": metrics,
        "cases": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("development_model", "smoke_case_count", "smoke_valid_count", "smoke_success_rate", "safety_violations")}, ensure_ascii=False, indent=2))
    return 0 if valid == len(samples) and payload["safety_violations"] == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--output", type=Path, default=Path("local_acceptance/v1.6.0_goal_smoke_qwen3_7_plus.json"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--request-timeout-sec", type=float, default=45.0)
    parser.add_argument("--preflight-timeout-sec", type=float, default=20.0)
    parser.add_argument("--case-timeout-sec", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(collect(parse_args())))
