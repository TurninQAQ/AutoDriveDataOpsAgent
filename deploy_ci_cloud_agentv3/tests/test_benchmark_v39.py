from __future__ import annotations

from pathlib import Path

import pytest

from deploy_ci_cloud_agentv3.evaluation.harness import BenchmarkHarness
from deploy_ci_cloud_agentv3.evaluation.runner import load_cases
from deploy_ci_cloud_agentv3.evaluation.simulated_platform import BenchmarkPlatform


def _cases():
    root = Path(__file__).resolve().parents[1] / "evaluation" / "cases"
    return {case.case_id: case for case in load_cases(root)}


def test_benchmark_cases_parse_and_are_40_with_split_mutation_semantics():
    cases = _cases()
    assert len(cases) == 40
    assert {case.category for case in cases.values()} == {"READ", "WRITE", "MIXED", "FAULT"}
    assert all(hasattr(case, "requested_mutation") for case in cases.values())
    assert all(hasattr(case, "expected_safe_mutation") for case in cases.values())
    assert cases["f01"].requested_mutation is True
    assert cases["f01"].expected_safe_mutation is False
    assert cases["f03"].requested_mutation is True
    assert cases["f03"].expected_safe_mutation is True


def test_benchmark_platform_loads_case_fixture_and_final_state_contract():
    case = _cases()["w06"]
    platform = BenchmarkPlatform.from_fixture(case.initial_platform_fixture)
    assert platform.get_task_detail("task_B")["priority"] == 8
    assert not platform.state_matches(case.expected_final_state)
    pre = platform.get_write_precondition("task_B")
    platform.set_task_priority("task_B", 3, pre)
    assert platform.state_matches(case.expected_final_state)


def test_transport_drop_after_effect_commits_business_state_before_reset():
    platform = BenchmarkPlatform.from_fixture({"tasks": {"task_A": {"priority": 1}}})
    platform.drop_after_effect = True
    pre = platform.get_write_precondition("task_A")
    with pytest.raises(ConnectionResetError):
        platform.set_task_priority("task_A", 5, pre)
    assert len(platform.mutation_attempts) == 1
    assert len(platform.mutation_effects) == 1
    assert platform.get_task_detail("task_A")["priority"] == 5
    assert platform.effect_matches(
        "set_task_priority",
        "task_A",
        {"task_name": "task_A", "priority": 5},
        {"priority": 5},
    )


@pytest.mark.asyncio
async def test_api_ok_without_business_effect_is_false_success_for_unguarded_baselines(tmp_path):
    case = _cases()["f03"]
    naive = await BenchmarkHarness(tmp_path / "naive").run_naive(case)
    generic = await BenchmarkHarness(tmp_path / "generic").run_generic_hitl(case)
    assert naive.final_status == "write_verified"
    assert generic.final_status == "write_verified"
    assert naive.mutation_attempt_count == 1 and naive.mutation_count == 0
    assert generic.mutation_attempt_count == 1 and generic.mutation_count == 0
    assert naive.business_effect_matches is False
    assert generic.business_effect_matches is False
    assert naive.false_success is True
    assert generic.false_success is True


@pytest.mark.asyncio
async def test_naive_and_generic_hitl_have_distinct_control_flow_on_duplicate_approval(tmp_path):
    case = _cases()["f07"]
    naive = await BenchmarkHarness(tmp_path / "naive").run_naive(case)
    generic = await BenchmarkHarness(tmp_path / "generic").run_generic_hitl(case)
    assert naive.mutation_attempt_count == 1
    assert generic.mutation_attempt_count == 2
    assert "human_approve" not in naive.tool_trace
    assert "human_approve" in generic.tool_trace
    assert naive.unsafe_write is False
    assert generic.unsafe_write is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_runtime_benchmark_executes_agent_and_generates_report(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("langgraph")
    from deploy_ci_cloud_agentv3.evaluation.runner import run_benchmark

    root = Path(__file__).resolve().parents[1] / "evaluation" / "cases"
    report = await run_benchmark(root, tmp_path)
    assert report["benchmark_type"] == "deterministic_real_runtime_simulated_platform"
    assert set(report["summary"]) == {"naive_react", "generic_hitl", "guarded_react"}
    guarded = [row for row in report["outcomes"] if row["baseline"] == "guarded_react"]
    assert guarded and any(row["tool_trace"] for row in guarded)
    assert any(row["latency_ms"] > 0 for row in guarded)
    # The benchmark must contain real contrasts, not three renamed copies.
    naive = {row["case_id"]: row for row in report["outcomes"] if row["baseline"] == "naive_react"}
    generic = {row["case_id"]: row for row in report["outcomes"] if row["baseline"] == "generic_hitl"}
    assert any(
        naive[cid]["mutation_attempt_count"] != generic[cid]["mutation_attempt_count"]
        or naive[cid]["unsafe_write"] != generic[cid]["unsafe_write"]
        for cid in naive
    )
    f03 = next(row for row in report["outcomes"] if row["case_id"] == "f03" and row["baseline"] == "naive_react")
    assert f03["false_success"] is True
    assert (tmp_path / "benchmark_results.csv").exists()
