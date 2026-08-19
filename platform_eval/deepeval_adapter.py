from __future__ import annotations

import os
from typing import Any


def run_deepeval_tool_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional native DeepEval component metrics.

    This adapter intentionally covers tool/argument correctness only. DeepEval's
    TaskCompletionMetric is trajectory-based and should be attached to a real
    model run with DeepEval tracing instead of being faked from our deterministic
    trace JSON.
    """
    try:
        from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric
        from deepeval.test_case import LLMTestCase, ToolCall
    except ImportError as exc:
        raise RuntimeError("Install requirements-eval.txt before running DeepEval metrics") from exc

    judge_model = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "gpt-5-mini").strip()
    rows = []
    tool_scores = []
    arg_scores = []
    for sample in samples:
        tools_called = [ToolCall(name=item["name"], input_parameters=item.get("arguments")) for item in sample.get("tools_called", [])]
        expected_tools = [ToolCall(name=item["name"], input_parameters=item.get("arguments")) for item in sample.get("expected_tools", [])]
        case = LLMTestCase(
            input=str(sample["input"]),
            actual_output=str(sample.get("actual_output") or ""),
            tools_called=tools_called,
            expected_tools=expected_tools,
        )
        tool_metric = ToolCorrectnessMetric(should_consider_ordering=bool(sample.get("consider_ordering", False)))
        tool_metric.measure(case)
        argument_metric = ArgumentCorrectnessMetric(model=judge_model)
        argument_metric.measure(case)
        tool_score = float(tool_metric.score or 0.0)
        arg_score = float(argument_metric.score or 0.0)
        tool_scores.append(tool_score)
        arg_scores.append(arg_score)
        rows.append({"id": sample.get("id"), "tool_correctness": tool_score, "argument_correctness": arg_score})
    return {
        "framework": "deepeval",
        "case_count": len(rows),
        "tool_correctness": sum(tool_scores) / len(tool_scores) if tool_scores else 0.0,
        "argument_correctness": sum(arg_scores) / len(arg_scores) if arg_scores else 0.0,
        "cases": rows,
        "task_completion_note": "Run TaskCompletionMetric on a real traced Agent execution; V1.1 does not fabricate a trajectory judge from fixture traces.",
    }
