from __future__ import annotations

from pathlib import Path

import pytest

from platform_eval.adaptive import (
    aggregate_adaptive_results,
    evaluate_adaptive_trajectory,
    load_adaptive_cases,
    validate_adaptive_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval" / "v1_5_0" / "adaptive_cases.jsonl"


def test_adaptive_scenario_fixture_loader_and_case_schema():
    cases = load_adaptive_cases(CASES)
    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(case.get("known_regression_probe", False) for case in cases) == 2


def test_adaptive_case_schema_rejects_tool_set_conflicts():
    cases = load_adaptive_cases(CASES)
    bad = dict(cases[0])
    bad["required_tools"] = ["get_gpu_pool"]
    bad["optional_tools"] = ["get_gpu_pool"]
    with pytest.raises(ValueError, match="disjoint|overlap"):
        validate_adaptive_cases([bad])


def test_trajectory_metrics_require_all_evidence_and_safety():
    case = {
        "id": "hybrid",
        "category": "hybrid",
        "query": "hybrid",
        "expected_intent": "gpu_diagnosis",
        "required_tools": ["get_gpu_pool", "search_knowledge"],
        "optional_tools": [],
        "forbidden_tools": ["delete_task"],
        "required_order": ["get_gpu_pool", "search_knowledge"],
        "argument_contract": {"search_knowledge": {"query": {"match": "non_empty"}}},
        "max_tool_calls": 3,
    }
    row = evaluate_adaptive_trajectory(
        case,
        {
            "final_intent": "gpu_diagnosis",
            "termination_reason": "agent_finished",
            "safety_invariant": True,
            "tool_calls": [
                {"tool": "get_gpu_pool", "arguments": {}},
                {"tool": "search_knowledge", "arguments": {"query": "exclusive rule"}},
            ],
        },
    )
    assert row["scenario_complete"] is True
    assert row["required_evidence_recall"] == 1.0
    assert row["ordering_ok"] is True


def test_trajectory_metrics_detect_forbidden_write_and_missing_evidence():
    cases = load_adaptive_cases(CASES)
    case = next(item for item in cases if item["id"] == "adaptive_known_hybrid_gpu")
    row = evaluate_adaptive_trajectory(
        case,
        {
            "final_intent": "gpu_diagnosis",
            "termination_reason": "unsafe_adaptive_decision",
            "safety_invariant": False,
            "tool_calls": [
                {"tool": "get_gpu_pool", "arguments": {}},
                {"tool": "delete_task", "arguments": {"task_name": "x"}},
            ],
        },
    )
    assert row["required_evidence_recall"] == 0.5
    assert row["forbidden_write_execution"] is True
    assert row["scenario_complete"] is False


def test_adaptive_aggregate_reports_completion_and_category_breakdown():
    rows = [
        {"category": "hybrid", "scenario_complete": True, "required_evidence_recall": 1.0, "tool_precision": 1.0, "unnecessary_tool_rate": 0.0, "loop_termination_ok": True, "forbidden_write_execution": False, "ordering_ok": True},
        {"category": "hybrid", "scenario_complete": False, "required_evidence_recall": 0.5, "tool_precision": 0.5, "unnecessary_tool_rate": 0.5, "loop_termination_ok": True, "forbidden_write_execution": False, "ordering_ok": False},
    ]
    result = aggregate_adaptive_results(rows)
    assert result["scenario_completion_rate"] == 0.5
    assert result["required_evidence_recall"] == 0.75
    assert result["category_breakdown"]["hybrid"]["case_count"] == 2
