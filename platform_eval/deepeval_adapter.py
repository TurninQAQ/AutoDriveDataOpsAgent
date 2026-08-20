from __future__ import annotations

import os
import asyncio
from typing import Any


COLLECTION_INVALID = "COLLECTION_INVALID"


def _tool_names(sample: dict[str, Any], field: str) -> list[str]:
    if field in sample:
        return [str(name) for name in sample.get(field) or []]
    return [str(item.get("name")) for item in sample.get("tools_called", []) if item.get("name")]


def _collection_invalid_cases(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid = []
    for sample in samples:
        required = [str(name) for name in sample.get("required_tools") or []]
        if not required:
            required = [str(item.get("name")) for item in sample.get("expected_tools", []) if item.get("name")]
        actual = _tool_names(sample, "actual_tools")
        if required and not actual:
            invalid.append(
                {
                    "case_id": sample.get("case_id", sample.get("id")),
                    "query": sample.get("query", sample.get("input", "")),
                    "required_tools": required,
                    "actual_tools": actual,
                    "reason": "required_tools_nonempty_but_actual_tools_empty",
                }
            )
    return invalid


def _case_metadata(
    sample: dict[str, Any],
    *,
    tool_score: float | None = None,
    argument_score: float | None = None,
    collection_valid: bool = True,
    collection_error: str | None = None,
) -> dict[str, Any]:
    actual_arguments = sample.get("actual_arguments") or sample.get("tools_called", [])
    row = {
        "id": sample.get("id"),
        "case_id": sample.get("case_id", sample.get("id")),
        "query": sample.get("query", sample.get("input", "")),
        "required_tools": list(sample.get("required_tools") or []),
        "optional_tools": list(sample.get("optional_tools") or []),
        "forbidden_tools": list(sample.get("forbidden_tools") or []),
        "actual_tools": _tool_names(sample, "actual_tools"),
        "actual_arguments": actual_arguments,
        "tool_correctness": tool_score,
        "argument_correctness": argument_score,
        "collection_valid": collection_valid,
    }
    if collection_error:
        row["collection_error"] = collection_error
    return row


def run_deepeval_tool_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional native DeepEval component metrics.

    This adapter intentionally covers tool/argument correctness only. DeepEval's
    TaskCompletionMetric is trajectory-based and should be attached to a real
    model run with DeepEval tracing instead of being faked from our deterministic
    trace JSON.
    """
    invalid_cases = _collection_invalid_cases(samples)
    if invalid_cases:
        invalid_by_id = {
            item.get("case_id"): item.get("reason") for item in invalid_cases
        }
        return {
            "framework": "deepeval",
            "status": COLLECTION_INVALID,
            "collection_status": COLLECTION_INVALID,
            "provider": os.getenv("PLATFORM_EVAL_PROVIDER", "").strip().lower() or "default",
            "judge_model": os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "gpt-5-mini").strip(),
            "case_count": len(samples),
            "tool_correctness": None,
            "argument_correctness": None,
            "invalid_cases": invalid_cases,
            "cases": [
                _case_metadata(
                    sample,
                    collection_valid=sample.get("case_id", sample.get("id")) not in invalid_by_id,
                    collection_error=invalid_by_id.get(sample.get("case_id", sample.get("id"))),
                )
                for sample in samples
            ],
            "task_completion_note": "DeepEval metrics were not run because tool collection was invalid.",
        }

    try:
        from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric
        from deepeval.test_case import LLMTestCase, ToolCall
    except ImportError as exc:
        raise RuntimeError("Install requirements-eval.txt before running DeepEval metrics") from exc

    provider = os.getenv("PLATFORM_EVAL_PROVIDER", "").strip().lower()
    judge_model = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "gpt-5-mini").strip()
    judge = _build_qwen_judge(judge_model) if provider in {"qwen", "dashscope", "aliyun", "alibaba"} else judge_model
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
        tool_metric = ToolCorrectnessMetric(
            model=judge,
            should_consider_ordering=bool(sample.get("consider_ordering", False)),
        )
        tool_metric.measure(case)
        argument_metric = ArgumentCorrectnessMetric(model=judge)
        argument_metric.measure(case)
        tool_score = float(tool_metric.score or 0.0)
        arg_score = float(argument_metric.score or 0.0)
        tool_scores.append(tool_score)
        arg_scores.append(arg_score)
        rows.append(
            _case_metadata(
                sample,
                tool_score=tool_score,
                argument_score=arg_score,
            )
        )
    return {
        "framework": "deepeval",
        "status": "PASS",
        "collection_status": "VALID",
        "provider": provider or "default",
        "judge_model": judge_model,
        "case_count": len(rows),
        "tool_correctness": sum(tool_scores) / len(tool_scores) if tool_scores else 0.0,
        "argument_correctness": sum(arg_scores) / len(arg_scores) if arg_scores else 0.0,
        "cases": rows,
        "task_completion_note": "Run TaskCompletionMetric on a real traced Agent execution; V1.1 does not fabricate a trajectory judge from fixture traces.",
    }


def _build_qwen_judge(model_name: str):
    """Build DeepEval's custom-model interface over the native Qwen adapter."""
    try:
        from deepeval.models import DeepEvalBaseLLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install requirements-eval.txt before running DeepEval metrics") from exc

    from platform_agent.qwen import QwenReadOnlyModel

    class QwenDeepEvalModel(DeepEvalBaseLLM):
        def __init__(self, name: str):
            self._qwen = QwenReadOnlyModel(model=name, temperature=0.0)
            super().__init__(model=name)

        def load_model(self):
            return self

        def get_model_name(self, *args, **kwargs) -> str:
            return self.name

        async def a_generate(self, prompt, schema=None, **kwargs):
            if schema is None:
                raise RuntimeError("QwenDeepEvalModel requires a structured DeepEval schema")
            return await self._qwen._structured(str(prompt), schema)

        def generate(self, prompt, schema=None, **kwargs):
            return asyncio.run(self.a_generate(prompt, schema=schema, **kwargs))

        def supports_structured_outputs(self):
            return True

        def supports_json_mode(self):
            return True

    return QwenDeepEvalModel(model_name)
