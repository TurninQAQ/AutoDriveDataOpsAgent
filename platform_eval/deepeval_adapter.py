from __future__ import annotations

import os
import asyncio
from typing import Any

from platform_mcp.server import WRITE_TOOL_NAMES


COLLECTION_INVALID = "COLLECTION_INVALID"

SUPPORTED_WRITE_INTENTS = {
    "submit_task",
    "resume_task",
    "set_task_priority",
    "stop_task",
    "delete_task",
}

PRE_CONTRACT_AUDIT_BASELINE = {
    "label": "PRE-CONTRACT-AUDIT BASELINE",
    "contract_version": "v1_1",
    "tool_correctness": 0.5,
    "tool_precision": 0.857143,
    "tool_recall": 0.363636,
    "tool_f1": 0.510638,
    "argument_requirement_coverage": 0.2,
}


def _is_write_case(sample: dict[str, Any]) -> bool:
    return (
        str(sample.get("category") or "") == "write"
        and str(sample.get("expected_intent") or "") in SUPPORTED_WRITE_INTENTS
    )


def _is_read_case(sample: dict[str, Any]) -> bool:
    return str(sample.get("category") or "") != "write"


def _intent_accuracy(samples: list[dict[str, Any]]) -> float:
    rows = [sample for sample in samples if sample.get("expected_intent")]
    if not rows:
        return 1.0
    hits = sum(
        int(str(sample.get("expected_intent")) == str(sample.get("actual_intent")))
        for sample in rows
    )
    return hits / len(rows)


def _tool_names(sample: dict[str, Any], field: str) -> list[str]:
    if field in sample:
        return [str(name) for name in sample.get(field) or []]
    return [str(item.get("name")) for item in sample.get("tools_called", []) if item.get("name")]


def _collection_invalid_cases(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only failures in the collector/harness itself.

    An empty ``actual_tools`` value is a valid AgentPlan outcome.  Whether it
    misses a required tool is a model-selection metric, never a collection
    health signal.
    """
    invalid = []
    for sample in samples:
        collection_error = sample.get("collection_error")
        if sample.get("collection_valid") is False or collection_error:
            invalid.append(
                {
                    "case_id": sample.get("case_id", sample.get("id")),
                    "query": sample.get("query", sample.get("input", "")),
                    "required_tools": list(sample.get("required_tools") or []),
                    "actual_tools": _tool_names(sample, "actual_tools"),
                    "reason": collection_error or "collection_valid_false",
                }
            )
    return invalid


def _expected_arguments(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = sample.get("expected_arguments")
    if isinstance(expected, dict):
        return {
            str(name): value if isinstance(value, dict) else {}
            for name, value in expected.items()
        }
    derived: dict[str, dict[str, Any]] = {}
    for item in sample.get("expected_tools") or []:
        if item.get("name"):
            value = item.get("arguments")
            derived[str(item["name"])] = value if isinstance(value, dict) else {}
    return derived


def _argument_requirements(sample: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    expected = _expected_arguments(sample)
    actual = sample.get("actual_arguments") or sample.get("tools_called", [])
    details: list[dict[str, Any]] = []
    hits = 0
    for tool_name, expected_subset in expected.items():
        matching = [
            item for item in actual
            if str(item.get("name") or "") == tool_name
        ]
        if expected_subset:
            ok = any(_arg_subset(item.get("arguments") or {}, expected_subset) for item in matching)
        else:
            # An empty Golden subset still requires the expected tool to be
            # present, while imposing no additional argument fields.
            ok = bool(matching)
        hits += int(ok)
        details.append({
            "tool": tool_name,
            "required_arguments": expected_subset,
            "actual": [item.get("arguments") or {} for item in matching],
            "ok": ok,
        })
    return hits, len(expected), details


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


def _deterministic_tool_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = total_fp = total_fn = 0
    forbidden_case_count = 0
    argument_hits = argument_total = 0
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for sample in samples:
        required = set(str(name) for name in sample.get("required_tools") or [])
        optional = set(str(name) for name in sample.get("optional_tools") or [])
        acceptable = required | optional
        actual = set(_tool_names(sample, "actual_tools"))
        forbidden = set(str(name) for name in sample.get("forbidden_tools") or []) | set(WRITE_TOOL_NAMES)
        forbidden_called = sorted(actual & forbidden)
        tp = len(actual & acceptable)
        fp = len(actual - acceptable)
        fn = len(required - actual)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        forbidden_case_count += int(bool(forbidden_called))

        arg_hits, arg_total, arg_details = _argument_requirements(sample)
        argument_hits += arg_hits
        argument_total += arg_total
        row = {
            "case_id": sample.get("case_id", sample.get("id")),
            "tool_tp": tp,
            "tool_fp": fp,
            "tool_fn": fn,
            "tool_precision": tp / (tp + fp) if tp + fp else 1.0,
            "tool_recall": tp / (tp + fn) if tp + fn else 1.0,
            "tool_f1": (
                2 * (tp / (tp + fp)) * (tp / (tp + fn))
                / ((tp / (tp + fp)) + (tp / (tp + fn)))
                if tp + fp and tp + fn and (tp / (tp + fp)) + (tp / (tp + fn))
                else 0.0
            ),
            "forbidden_tools_called": forbidden_called,
            "argument_requirement_hits": arg_hits,
            "argument_requirement_total": arg_total,
            "argument_requirement_coverage": arg_hits / arg_total if arg_total else 1.0,
            "model_tool_miss": bool(sample.get("model_tool_miss", bool(fn))),
        }
        cases.append(row)

        collection_error = sample.get("collection_error")
        if sample.get("collection_valid") is False or collection_error:
            failure_type = "PLANNER_ERROR" if collection_error == "model_plan_failed" else "HARNESS_ERROR"
            failures.append({"case_id": str(row["case_id"]), "failure_type": failure_type})
            continue
        if not sample.get("planner_valid", True):
            failures.append({"case_id": str(row["case_id"]), "failure_type": "PLANNER_ERROR"})
            continue
        if forbidden_called:
            failures.append({"case_id": str(row["case_id"]), "failure_type": "FORBIDDEN_TOOL"})
        if fn:
            failures.append({"case_id": str(row["case_id"]), "failure_type": "TOOL_MISSING"})
        if fp:
            failures.append({"case_id": str(row["case_id"]), "failure_type": "TOOL_EXTRA"})
        if arg_total and arg_hits < arg_total:
            failures.append({"case_id": str(row["case_id"]), "failure_type": "ARGUMENT_WRONG"})
        expected_intent = str(sample.get("expected_intent") or "")
        actual_intent = str(sample.get("actual_intent") or "")
        if expected_intent and actual_intent and expected_intent != actual_intent:
            failures.append({"case_id": str(row["case_id"]), "failure_type": "PLANNER_ERROR"})

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
        "forbidden_tool_call_rate": forbidden_case_count / len(samples) if samples else 0.0,
        "forbidden_tool_call_count": forbidden_case_count,
        "argument_requirement_coverage": argument_hits / argument_total if argument_total else 1.0,
        "argument_requirement_hits": argument_hits,
        "argument_requirement_total": argument_total,
        "deterministic_cases": cases,
        "failures": failures,
    }


def _contract_metrics(
    samples: list[dict[str, Any]],
    deterministic_read: dict[str, Any],
    deterministic_write: dict[str, Any],
    read_tool_scores: list[float],
    read_argument_scores: list[float],
) -> dict[str, Any]:
    read_samples = [sample for sample in samples if _is_read_case(sample)]
    write_samples = [sample for sample in samples if _is_write_case(sample)]
    all_write_samples = [sample for sample in samples if str(sample.get("category") or "") == "write"]
    observed_write_cases = sum(int(bool(_tool_names(sample, "actual_tools"))) for sample in write_samples)
    forbidden_write_cases = sum(
        int(bool(set(_tool_names(sample, "actual_tools")) & (set(sample.get("forbidden_tools") or []) | set(WRITE_TOOL_NAMES))))
        for sample in all_write_samples
    )
    return {
        "read_metrics": {
            "case_count": len(read_samples),
            "intent_accuracy": _intent_accuracy(read_samples),
            "deep_eval_tool_correctness": sum(read_tool_scores) / len(read_tool_scores) if read_tool_scores else 0.0,
            "deep_eval_argument_correctness": sum(read_argument_scores) / len(read_argument_scores) if read_argument_scores else 0.0,
            "tool_precision": deterministic_read["tool_precision"],
            "tool_recall": deterministic_read["tool_recall"],
            "tool_f1": deterministic_read["tool_f1"],
            "argument_requirement_coverage": deterministic_read["argument_requirement_coverage"],
        },
        "write_metrics": {
            "case_count": len(write_samples),
            "intent_accuracy": _intent_accuracy(write_samples),
            "write_action_accuracy": {
                "value": None,
                "status": "NOT_AVAILABLE_FROM_CURRENT_GOLDEN_SCHEMA",
            },
            "pre_action_observation_rate": observed_write_cases / len(write_samples) if write_samples else 0.0,
            "observed_case_count": observed_write_cases,
            "forbidden_write_tool_rate": forbidden_write_cases / len(all_write_samples) if all_write_samples else 0.0,
            "forbidden_write_case_count": forbidden_write_cases,
            "forbidden_write_denominator": len(all_write_samples),
            "deterministic_tool_precision": deterministic_write["tool_precision"],
            "deterministic_tool_recall": deterministic_write["tool_recall"],
            "deterministic_tool_f1": deterministic_write["tool_f1"],
        },
        "safety_metrics": {
            "hitl_enforcement": {
                "value": None,
                "status": "COVERED_BY_DETERMINISTIC_TESTS",
                "tests": ["tests/test_write_agent_v07.py", "tests/test_hardening_v10.py"],
            },
            "precondition_enforcement": {
                "value": None,
                "status": "COVERED_BY_DETERMINISTIC_TESTS",
                "tests": ["tests/test_write_agent_v07.py", "tests/test_action_verification_v08.py"],
            },
            "verification": {
                "value": None,
                "status": "COVERED_BY_DETERMINISTIC_TESTS",
                "tests": ["tests/test_action_verification_v08.py"],
            },
            "hard_task_success": {
                "value": None,
                "status": "NOT_EVALUATED_IN_AGENT_TOOL_CASES",
                "source": "environment/task cases and dependency-light E2E",
            },
        },
    }


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
        "category": sample.get("category", ""),
        "query": sample.get("query", sample.get("input", "")),
        "required_tools": list(sample.get("required_tools") or []),
        "optional_tools": list(sample.get("optional_tools") or []),
        "forbidden_tools": list(sample.get("forbidden_tools") or []),
        "actual_tools": _tool_names(sample, "actual_tools"),
        "actual_arguments": actual_arguments,
        "tool_correctness": tool_score,
        "argument_correctness": argument_score,
        "collection_valid": sample.get("collection_valid", collection_valid),
        "collection_error": sample.get("collection_error", collection_error),
        "planner_valid": sample.get("planner_valid", True),
        "expected_intent": sample.get("expected_intent", ""),
        "actual_intent": sample.get("actual_intent", ""),
        "intent_ok": sample.get("intent_ok"),
        "model_tool_miss": sample.get("model_tool_miss"),
        "write_action": sample.get("write_action"),
        "forbidden_tools_called": sample.get("forbidden_tools_called", []),
    }
    return row


def run_deepeval_tool_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional native DeepEval component metrics.

    This adapter intentionally covers tool/argument correctness only. DeepEval's
    TaskCompletionMetric is trajectory-based and should be attached to a real
    model run with DeepEval tracing instead of being faked from our deterministic
    trace JSON.
    """
    invalid_cases = _collection_invalid_cases(samples)
    deterministic = _deterministic_tool_metrics(samples)
    read_samples = [sample for sample in samples if _is_read_case(sample)]
    write_samples = [sample for sample in samples if _is_write_case(sample)]
    deterministic_read = _deterministic_tool_metrics(read_samples)
    deterministic_write = _deterministic_tool_metrics(write_samples)
    collection_valid_count = sum(
        int(sample.get("collection_valid", True) is not False and not sample.get("collection_error"))
        for sample in samples
    )
    collection_invalid_count = len(samples) - collection_valid_count
    if invalid_cases:
        contract_metrics = _contract_metrics(samples, deterministic_read, deterministic_write, [], [])
        return {
            "framework": "deepeval",
            "status": COLLECTION_INVALID,
            "collection_status": COLLECTION_INVALID,
            "provider": os.getenv("PLATFORM_EVAL_PROVIDER", "").strip().lower() or "default",
            "judge_model": os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "gpt-5-mini").strip(),
            "case_count": len(samples),
            "collection_valid_count": collection_valid_count,
            "collection_invalid_count": collection_invalid_count,
            "tool_correctness": None,
            "argument_correctness": None,
            "invalid_cases": invalid_cases,
            "cases": [
                _case_metadata(
                    sample,
                    collection_valid=sample.get("collection_valid", True),
                    collection_error=sample.get("collection_error"),
                )
                for sample in samples
            ],
            **{key: value for key, value in deterministic.items() if key != "deterministic_cases"},
            **contract_metrics,
            "task_completion_note": "DeepEval metrics were not run because the collector reported an explicit harness failure.",
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
    read_tool_scores = []
    read_argument_scores = []
    for sample in samples:
        # Empty tools_called is valid input to DeepEval.  It represents a
        # planner output with no selected tools, and must receive a real metric
        # score rather than invalidating the collection.
        tools_called = [ToolCall(name=item["name"], input_parameters=item.get("arguments") or {}) for item in sample.get("tools_called", [])]
        expected_tools = [ToolCall(name=item["name"], input_parameters=item.get("arguments") or {}) for item in sample.get("expected_tools", [])]
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
        if _is_read_case(sample):
            read_tool_scores.append(tool_score)
            read_argument_scores.append(arg_score)
        row = _case_metadata(sample, tool_score=tool_score, argument_score=arg_score)
        deterministic_row = next(
            item for item in deterministic["deterministic_cases"]
            if item["case_id"] == row["case_id"]
        )
        row.update({key: value for key, value in deterministic_row.items() if key != "case_id"})
        rows.append(row)
    contract_metrics = _contract_metrics(
        samples,
        deterministic_read,
        deterministic_write,
        read_tool_scores,
        read_argument_scores,
    )
    return {
        "framework": "deepeval",
        "status": "PASS",
        "collection_status": "VALID",
        "provider": provider or "default",
        "judge_model": judge_model,
        "case_count": len(rows),
        "collection_valid_count": collection_valid_count,
        "collection_invalid_count": collection_invalid_count,
        "tool_correctness": sum(tool_scores) / len(tool_scores) if tool_scores else 0.0,
        "argument_correctness": sum(arg_scores) / len(arg_scores) if arg_scores else 0.0,
        **{key: value for key, value in deterministic.items() if key != "deterministic_cases"},
        **contract_metrics,
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
