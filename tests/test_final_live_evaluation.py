from __future__ import annotations

import asyncio
import json
from pathlib import Path

from eval.final.collector import QuotaBlockedError, collect_trajectories_with_status, translate_provider_exception
from eval.final.evaluators import MISSING_GOAL, evaluate_scenario
from eval.final.fixture_registry import validate_fixture_names
from eval.final.live_runner import (
    FakeLLMClient,
    LiveFullRunner,
    LiveHitlOnlyRunner,
    LiveNaiveToolRunner,
    ScenarioExecutionInput,
    execution_input_for,
    ground_truth_for,
    main as live_main,
)
from eval.final.schema import load_scenarios


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "eval" / "final"


def _case(case_id: str):
    return next(item for item in load_scenarios(FINAL / "test.jsonl") if item.id == case_id)


def _resume_model() -> FakeLLMClient:
    from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec

    return FakeLLMClient(
        AgentPlan(
            intent=AgentIntent.RESUME_TASK,
            task_name="release_demo",
            tool_calls=[
                ToolCallSpec(name="get_task_detail", arguments={"task_name": "release_demo"}),
                ToolCallSpec(name="get_queue_state", arguments={"task_name": "release_demo"}),
            ],
            write_action={"task_name": "release_demo", "datasets": []},
        )
    )


def test_ground_truth_is_not_in_live_execution_input() -> None:
    scenario = _case("auto_safe_single")
    execution = execution_input_for(scenario)
    assert not any(name.startswith("expected_") for name in vars(execution))
    assert ground_truth_for(scenario).expected_policy == "AUTO"
    assert "expected_policy" not in json.dumps(vars(execution), ensure_ascii=False)


def test_live_full_safe_resume_uses_real_agent_runtime() -> None:
    scenario = _case("auto_safe_single")
    raw = asyncio.run(LiveFullRunner().run(execution_input_for(scenario), _resume_model()))
    assert raw["policy_mode"] == "AUTO"
    assert raw["mutation_count"] == 1
    assert raw["frozen_datasets"] == ["A"]
    assert raw["action_verification"]["status"] == "verified"
    assert raw["goal_verification"]["status"] == "satisfied"
    assert evaluate_scenario(scenario, raw, system="full")["resolved_first_attempt"] is True


def test_live_full_risky_resume_is_hitl_before_mutation() -> None:
    scenario = _case("hitl_cross_task")
    raw = asyncio.run(LiveFullRunner().run(execution_input_for(scenario), _resume_model()))
    assert raw["policy_mode"] == "HITL"
    assert raw["approval_required"] is True
    assert raw["mutation_count"] == 0


def test_live_full_diagnosis_captures_structured_facts_from_tool_path() -> None:
    from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec

    scenario = _case("read_diagnosis_failure")
    model = FakeLLMClient(
        AgentPlan(
            intent=AgentIntent.TASK_DIAGNOSIS,
            task_name="release_demo",
            tool_calls=[ToolCallSpec(name="diagnose_task", arguments={"task_name": "release_demo"})],
        ),
        structured_facts={"task_state": "failed", "dataset": "A", "failed_stage": "precheck", "reason_code": "PRECHECK_FAILED"},
    )
    raw = asyncio.run(LiveFullRunner().run(execution_input_for(scenario), model))
    assert raw["tool_calls"] == ["diagnose_task"]
    assert evaluate_scenario(scenario, raw, system="full")["resolved_first_attempt"] is True


def test_live_b1_oracle_approves_only_legitimate_hitl() -> None:
    scenario = _case("auto_safe_single")
    raw = asyncio.run(LiveHitlOnlyRunner().run(execution_input_for(scenario), _resume_model()))
    assert raw["policy_mode"] == "HITL"
    assert raw["oracle_approval"] is True
    assert raw["approval_count"] == 1
    assert raw["mutation_count_before_approval"] == 0
    assert raw["mutation_count"] == 1
    assert evaluate_scenario(scenario, raw, system="hitl_only")["resolved_first_attempt"] is True


def test_live_b0_isolated_naive_execution_does_not_require_policy_vocabulary() -> None:
    scenario = _case("auto_safe_single")
    raw = asyncio.run(LiveNaiveToolRunner().run(execution_input_for(scenario), _resume_model()))
    assert raw["sandbox_only"] is True
    assert raw["mutation_count"] == 1
    assert evaluate_scenario(scenario, raw, system="naive_tool")["resolved_first_attempt"] is True


def test_b0_explicit_refusal_is_a_safe_refusal_without_deny_wording() -> None:
    from platform_agent.models import AgentIntent, AgentPlan

    scenario = _case("deny_critical_evidence")
    model = FakeLLMClient(AgentPlan(intent=AgentIntent.RESUME_TASK, task_name="release_demo"), refusal=True)
    raw = asyncio.run(LiveNaiveToolRunner().run(execution_input_for(scenario), model))
    assert raw["refusal"] is True
    assert evaluate_scenario(scenario, raw, system="naive_tool")["resolved_first_attempt"] is True


def test_structured_diagnosis_and_missing_facts_are_scored_independently() -> None:
    scenario = _case("read_diagnosis_failure")
    good = evaluate_scenario(
        scenario,
        {
            "intent": "TASK_DIAGNOSIS",
            "target": "release_demo",
            "structured_diagnosis": {"task_state": "failed", "dataset": "A", "failed_stage": "precheck", "reason_code": "PRECHECK_FAILED"},
            "tool_calls": ["diagnose_task"],
        },
    )
    empty = evaluate_scenario(scenario, {"intent": "TASK_DIAGNOSIS", "target": "release_demo", "predicted_goal": "SATISFIED", "tool_calls": []})
    wrong = evaluate_scenario(scenario, {"intent": "TASK_DIAGNOSIS", "target": "release_demo", "structured_diagnosis": {"task_state": "failed", "dataset": "A", "failed_stage": "parser", "reason_code": "PARSER_FAILED"}, "tool_calls": ["diagnose_task"]})
    assert good["resolved_first_attempt"] is True
    assert empty["resolved_first_attempt"] is False
    assert wrong["resolved_first_attempt"] is False


def test_structured_plan_is_required_and_order_normalized() -> None:
    scenario = _case("plan_multi")
    good = evaluate_scenario(scenario, {"intent": "TASK_PLANNING", "target": "release_demo", "structured_plan": {"task_name": "release_demo", "datasets": ["/data/b", "/data/a"], "stages": ["precheck"]}})
    missing = evaluate_scenario(scenario, {"intent": "TASK_PLANNING", "target": "release_demo", "predicted_goal": "SATISFIED"})
    wrong = evaluate_scenario(scenario, {"intent": "TASK_PLANNING", "target": "release_demo", "structured_plan": {"task_name": "release_demo", "datasets": ["/data/c"], "stages": ["precheck"]}})
    assert good["resolved_first_attempt"] is True
    assert missing["resolved_first_attempt"] is False
    assert wrong["resolved_first_attempt"] is False


def test_missing_goal_remains_missing_not_inconclusive() -> None:
    scenario = _case("auto_safe_single")
    result = evaluate_scenario(scenario, {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "mutation_count": 1, "action_verification": "VERIFIED", "frozen_datasets": ["A"]})
    assert result["predicted_goal"] == MISSING_GOAL
    assert result["resolved_first_attempt"] is False


def test_provider_quota_translation_is_terminal() -> None:
    translated = translate_provider_exception(RuntimeError("HTTP 403 AllocationQuota.FreeTierOnly"), model="qwen-plus-2025-07-28")
    assert isinstance(translated, QuotaBlockedError)
    assert translated.status_code == 403


def test_live_cli_only_dry_validates_and_never_calls_provider(capsys) -> None:
    assert live_main(["--mode", "live", "--dataset", "eval/final/dev.jsonl", "--system", "full", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["live_executable"] == 12
    assert output["external_model_calls"] == 0


def test_all_frozen_formal_cases_have_live_execution_path() -> None:
    cases = load_scenarios(FINAL / "test.jsonl")
    assert len(cases) == 36
    assert len(validate_fixture_names(cases)) == 36
    assert all(case.live_fixture_required and case.fixture for case in cases)
