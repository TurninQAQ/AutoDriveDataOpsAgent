#!/usr/bin/env python3
"""Collect and score the V1.5 adaptive scenarios with a real Qwen model.

The tool world is deterministic and local.  Only Agent planning/step decisions
and synthesis use the configured provider.  Samples are saved before aggregate
metrics are written so a later evaluator never has to collect another trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from platform_agent.memory import ConversationStore
from platform_agent.models import ToolCallSpec, ToolObservation
from platform_agent.qwen import QwenReadOnlyModel
from platform_agent.tool_catalog import build_read_only_tool_catalog
from platform_agent.workflow import build_agent_runtime
from platform_eval.adaptive import (
    aggregate_adaptive_results,
    evaluate_adaptive_trajectory,
    load_adaptive_cases,
)
from platform_mcp.server import READ_ONLY_TOOL_NAMES


def _bounded(value: Any, limit: int = 3000) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, dict):
        return {str(key): _bounded(item, limit) for key, item in list(value.items())[:100]}
    if isinstance(value, list):
        return [_bounded(item, limit) for item in value[:100]]
    return value


class ScenarioFixtureToolClient:
    def __init__(self, fixture_results: dict[str, Any]):
        self.fixture_results = fixture_results
        self.calls: list[ToolCallSpec] = []
        self.results: list[dict[str, Any]] = []

    async def describe_tools(self):
        return build_read_only_tool_catalog(knowledge_enabled=True)

    async def execute(self, calls: list[ToolCallSpec]):
        if len(calls) != 1:
            raise AssertionError("V1.5 ScenarioFixtureToolClient only accepts one tool per adaptive step")
        call = calls[0]
        self.calls.append(call)
        fixture = self.fixture_results.get(call.name, {})
        if isinstance(fixture, dict) and fixture.get("__error__"):
            observation = ToolObservation(
                tool_name=call.name,
                arguments=call.arguments,
                ok=False,
                error=str(fixture["__error__"]),
            )
        else:
            observation = ToolObservation(
                tool_name=call.name,
                arguments=call.arguments,
                ok=True,
                data=fixture,
            )
        self.results.append({
            "tool": call.name,
            "arguments": call.arguments,
            "ok": observation.ok,
            "error": observation.error,
            "data_summary": _bounded(observation.data),
        })
        return [observation]


class CountingCompletions:
    def __init__(self, delegate, counter: dict[str, int]):
        self.delegate = delegate
        self.counter = counter

    async def create(self, **kwargs):
        self.counter["requests"] += 1
        response = await self.delegate.create(**kwargs)
        self.counter["completed"] = self.counter.get("completed", 0) + 1
        return response


class CountingChat:
    def __init__(self, delegate, counter: dict[str, int]):
        self.completions = CountingCompletions(delegate.completions, counter)


class CountingClient:
    def __init__(self, delegate, counter: dict[str, int]):
        self.chat = CountingChat(delegate.chat, counter)


def _trajectory(case: dict[str, Any], response, client: ScenarioFixtureToolClient) -> dict[str, Any]:
    initial_plan = response.initial_plan or {}
    return {
        "case_id": case["id"],
        "query": case["query"],
        "initial_intent": initial_plan.get("intent"),
        "initial_plan": initial_plan,
        "adaptive_steps": response.adaptive_steps,
        "adaptive_step_count": response.adaptive_step_count,
        "tool_calls": response.tool_trace,
        "tool_results_summary": client.results,
        "final_intent": response.intent.value,
        "termination_reason": response.termination_reason or "legacy_path",
        "evidence_sufficient": response.evidence_sufficient,
        "response_confidence": response.confidence,
        "errors": response.errors,
        "required_evidence": case.get("required_tools", []),
        "safety_invariant": all(item.get("tool") in READ_ONLY_TOOL_NAMES for item in response.tool_trace),
    }


async def _collect(
    cases: list[dict[str, Any]],
    model,
    root: Path,
    limits: dict[str, int],
    checkpoint_prefix: str = "v1.5.0_adaptive_qwen_plus",
):
    samples = []
    eval_rows = []
    root.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        (root / f"{checkpoint_prefix}_samples.partial.json").write_text(
            json.dumps({"samples": samples}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (root / f"{checkpoint_prefix}_eval.partial.json").write_text(
            json.dumps({"cases": eval_rows, "aggregate": aggregate_adaptive_results(eval_rows)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with tempfile.TemporaryDirectory(prefix="autodrive-v150-sessions-") as session_root:
        for case in cases:
            fixture = ScenarioFixtureToolClient(case.get("fixture_results", {}))
            runtime = build_agent_runtime(
                "sequential",
                model,
                fixture,
                ConversationStore(Path(session_root) / case["id"]),
                max_tool_calls=limits["max_tool_calls"],
                max_steps=limits["max_steps"],
                max_identical_tool_calls=limits["max_identical_tool_calls"],
                max_consecutive_tool_failures=limits["max_consecutive_tool_failures"],
            )
            try:
                response = await asyncio.wait_for(
                    runtime.run(case["query"], case["id"]),
                    timeout=limits["case_timeout_sec"],
                )
                sample = _trajectory(case, response, fixture)
            except Exception as exc:
                sample = {
                    "case_id": case["id"],
                    "query": case["query"],
                    "initial_intent": None,
                    "initial_plan": None,
                    "adaptive_steps": [],
                    "adaptive_step_count": 0,
                    "tool_calls": fixture.results,
                    "tool_results_summary": fixture.results,
                    "final_intent": "",
                    "termination_reason": "collection_error",
                    "evidence_sufficient": False,
                    "response_confidence": "low",
                    "errors": [str(exc)],
                    "required_evidence": case.get("required_tools", []),
                    "safety_invariant": all(item.get("tool") in READ_ONLY_TOOL_NAMES for item in fixture.results),
                    "collection_error": str(exc),
                }
            evaluation = evaluate_adaptive_trajectory(case, sample)
            evaluation["actual_intent"] = sample.get("final_intent", "")
            evaluation["initial_intent"] = sample.get("initial_intent", "")
            samples.append(sample)
            eval_rows.append(evaluation)
            checkpoint()
    return samples, eval_rows


async def main(args) -> int:
    if not os.environ.get("DASHSCOPE_API_KEY") or not os.environ.get("DASHSCOPE_OPENAI_BASE_URL"):
        print("BLOCKED_NOT_VALIDATED: DASHSCOPE_API_KEY and DASHSCOPE_OPENAI_BASE_URL are required for real qwen-plus collection.")
        return 2
    cases = load_adaptive_cases(args.cases)
    counter = {"requests": 0}
    from openai import AsyncOpenAI

    raw_client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_OPENAI_BASE_URL"],
        timeout=args.request_timeout_sec,
    )
    model = QwenReadOnlyModel(model=args.model, client=CountingClient(raw_client, counter))
    limits = {
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_identical_tool_calls": args.max_identical_tool_calls,
        "max_consecutive_tool_failures": args.max_consecutive_tool_failures,
        "case_timeout_sec": args.case_timeout_sec,
    }
    samples, rows = await _collect(cases, model, args.output_dir, limits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v1.5.0_adaptive_qwen_plus_samples.json").write_text(
        json.dumps({"model": args.model, "limits": limits, "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unseen = [row for row, case in zip(rows, cases) if not case.get("known_regression_probe")]
    probes = [row for row, case in zip(rows, cases) if case.get("known_regression_probe")]
    evaluation = {
        "version": "v1.5.0",
        "agent_model": args.model,
        "judge_model": None,
        "self_model_evaluation": False,
        "api_request_count": counter["requests"],
        "case_count": len(cases),
        "unseen_case_count": len(unseen),
        "known_regression_probe_count": len(probes),
        "aggregate": aggregate_adaptive_results(unseen),
        "known_regression_probes": probes,
        "cases": rows,
    }
    (args.output_dir / "v1.5.0_adaptive_qwen_plus_eval.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"api_request_count": counter["requests"], "aggregate": evaluation["aggregate"], "known_regression_probes": probes}, ensure_ascii=False, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("eval/v1_5_0/adaptive_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_acceptance"))
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-identical-tool-calls", type=int, default=2)
    parser.add_argument("--max-consecutive-tool-failures", type=int, default=2)
    parser.add_argument("--case-timeout-sec", type=float, default=120.0)
    parser.add_argument("--request-timeout-sec", type=float, default=45.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
