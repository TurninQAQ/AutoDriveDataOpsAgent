from __future__ import annotations

import asyncio
import json
from pathlib import Path

from eval.final.collector import CollectorConfig, QuotaBlockedError, adapter_for, collect_trajectories_with_status, translate_provider_exception
from eval.final.evaluators import MISSING_GOAL, evaluate_scenario
from eval.final.fixture_registry import validate_fixture_names
from eval.final.live_runner import (
    FakeLLMClient,
    LiveFullRunner,
    LiveHitlOnlyRunner,
    LiveNaiveToolRunner,
    FixtureToolClient,
    ScenarioExecutionInput,
    execution_input_for,
    ground_truth_for,
    run_live_dataset,
    main as live_main,
)
from eval.final.schema import load_scenarios
from platform_integrations.model_retry import ModelRequestError


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


def _dev_model(case_id: str) -> FakeLLMClient:
    from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec

    if "diagnosis" in case_id:
        plan = AgentPlan(intent=AgentIntent.TASK_DIAGNOSIS, task_name="dev_release_demo", tool_calls=[ToolCallSpec(name="diagnose_task", arguments={"task_name": "dev_release_demo"})])
    elif "partial" in case_id:
        plan = AgentPlan(intent=AgentIntent.TASK_STATUS, task_name="dev_release_demo", tool_calls=[ToolCallSpec(name="get_task_detail", arguments={"task_name": "dev_release_demo"})])
    elif "plan" in case_id:
        plan = AgentPlan(intent=AgentIntent.TASK_PLANNING, task_name="dev_release_demo", task_draft={"task_prefix": "dev_release_demo", "dataset_paths": ["/data/dev_a"], "pipeline_stages": ["precheck"]})
    elif "stop" in case_id or "adversarial" in case_id:
        plan = AgentPlan(intent=AgentIntent.STOP_TASK, task_name="dev_release_demo", tool_calls=[ToolCallSpec(name="get_task_detail", arguments={"task_name": "dev_release_demo"}), ToolCallSpec(name="get_queue_state", arguments={"task_name": "dev_release_demo"})], write_action={"task_name": "dev_release_demo", "datasets": []})
    elif "missing" in case_id:
        plan = AgentPlan(intent=AgentIntent.RESUME_TASK, task_name=None, write_action={"task_name": None, "datasets": []})
    else:
        plan = AgentPlan(intent=AgentIntent.RESUME_TASK, task_name="dev_release_demo", tool_calls=[ToolCallSpec(name="get_task_detail", arguments={"task_name": "dev_release_demo"}), ToolCallSpec(name="get_queue_state", arguments={"task_name": "dev_release_demo"})], write_action={"task_name": "dev_release_demo", "datasets": []})
    return FakeLLMClient(plan)


class _ProviderLikeModel:
    """Real-provider-shaped fake: deliberately has no scripted fact fields."""

    requires_tool_descriptions = True
    supports_adaptive = True

    def __init__(self, plan):
        self._plan = plan

    async def plan(self, _user_text, _tool_descriptions, _history):
        return self._plan

    async def synthesize(self, _user_text, plan, _observations, _history, knowledge=None):
        from platform_agent.models import AgentResponse
        return AgentResponse(intent=plan.intent, summary="provider-like response", confidence="high")


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


def test_live_full_risky_resume_uses_oracle_then_guarded_mutation() -> None:
    scenario = _case("hitl_cross_task")
    raw = asyncio.run(LiveFullRunner().run(execution_input_for(scenario), _resume_model()))
    assert raw["policy_mode"] == "HITL"
    assert raw["approval_required"] is True
    assert raw["oracle_approval"] is True
    assert raw["mutation_count_before_approval"] == 0
    assert raw["mutation_count"] == 1
    assert raw["action_verification"]["status"] == "verified"
    assert raw["goal_verification"]["status"] in {"satisfied", "in_progress"}


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


def test_real_provider_like_model_without_scripted_attributes_is_safe() -> None:
    from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec

    scenario = _case("read_diagnosis_failure")
    plan = AgentPlan(intent=AgentIntent.TASK_DIAGNOSIS, task_name="release_demo", tool_calls=[ToolCallSpec(name="diagnose_task", arguments={"task_name": "release_demo"})])
    model = _ProviderLikeModel(plan)
    assert not hasattr(model, "structured_facts")
    raw = asyncio.run(LiveFullRunner().run(execution_input_for(scenario), model))
    assert raw["structured_facts"]["reason_code"] == "PRECHECK_FAILED"
    assert raw["tool_calls"] == ["diagnose_task"]


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


def test_b0_executes_read_diagnosis_and_planning_tools() -> None:
    diagnosis = _case("read_diagnosis_failure")
    diagnosis_model = _dev_model("diagnosis")
    raw_diagnosis = asyncio.run(LiveNaiveToolRunner().run(execution_input_for(diagnosis), diagnosis_model))
    assert "diagnose_task" in raw_diagnosis["tool_calls"]
    assert raw_diagnosis["structured_diagnosis"]["reason_code"] == "PRECHECK_FAILED"

    planning = _case("plan_multi")
    plan_model = FakeLLMClient(_dev_model("plan").plan_result)
    raw_plan = asyncio.run(LiveNaiveToolRunner().run(execution_input_for(planning), plan_model))
    assert raw_plan["structured_plan"]["stages"] == ["precheck"]


def test_fixture_write_mutations_have_real_post_state_verification() -> None:
    from platform_agent.models import ToolCallSpec
    from platform_agent.verification import ActionVerifier

    async def verify(action, fixture_name, arguments):
        client = FixtureToolClient(__import__("eval.final.fixture_registry", fromlist=["resolve_fixture"]).resolve_fixture(fixture_name))
        baseline = (await client.execute([ToolCallSpec(name="get_action_verification_snapshot", arguments={"task_name": "release_demo", "datasets": arguments.get("datasets", [])})]))[0].data
        observation = (await client.execute([ToolCallSpec(name=action, arguments=arguments)]))[0]
        result = await ActionVerifier(client, attempts=1, interval_sec=0).verify(action=action, arguments=arguments, execution_result=observation.data, baseline=baseline)
        return client, result

    async def run_all():
        return [
            await verify("resume_task", "safe_single_failed_dataset", {"task_name": "release_demo", "datasets": ["A"]}),
            await verify("stop_task", "stop_write", {"task_name": "release_demo", "datasets": []}),
            await verify("submit_task", "submit_write", {"task_name": "release_demo", "config": {"datasets": [{"dataset_name": "A"}], "priority": 3, "task_exclusive": True}}),
            await verify("delete_task", "adversarial_delete_bypass", {"task_name": "release_demo"}),
            await verify("set_task_priority", "stop_write", {"task_name": "release_demo", "priority": 5}),
        ]

    results = asyncio.run(run_all())
    assert all(client.mutation_calls == 1 and result.status == "verified" for client, result in results)


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


def test_model_request_error_preserves_free_tier_identity() -> None:
    error = ModelRequestError("qwen:chat", 1, 403, False, provider_error_code="AllocationQuota.FreeTierOnly")
    translated = translate_provider_exception(error, model="qwen-plus-2025-07-28")
    assert isinstance(translated, QuotaBlockedError)


def test_collector_stops_after_provider_quota_block() -> None:
    cases = load_scenarios(FINAL / "dev.jsonl")[:5]
    calls = []

    def runner(_case, _repetition, _model):
        calls.append(len(calls) + 1)
        if len(calls) == 3:
            raise ModelRequestError("qwen:chat", 1, 403, False, provider_error_code="AllocationQuota.FreeTierOnly")
        return {"intent": "TASK_STATUS"}

    records, summary = collect_trajectories_with_status(cases, CollectorConfig(model="qwen", system="full", repetitions=1), adapter_for("full", runner))
    assert len(calls) == len(records) == 3
    assert summary["status"] == "INCOMPLETE_QUOTA_BLOCKED"
    assert summary["remaining_attempts"] == 2


def test_live_cli_only_dry_validates_and_never_calls_provider(capsys) -> None:
    assert live_main(["--mode", "live", "--dataset", "eval/final/dev.jsonl", "--system", "full", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["live_executable"] == 12
    assert output["external_model_calls"] == 0


def test_live_cli_preflight_is_non_executing(capsys) -> None:
    assert live_main(["--preflight", "--model", "qwen-plus-2025-07-28"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PROVIDER_PREFLIGHT_NOT_RUN"
    assert output["external_model_calls"] == 0


def test_live_cli_rejects_frozen_formal_test_without_explicit_guard() -> None:
    try:
        live_main(["--dataset", "eval/final/test.jsonl", "--run-id", "must-not-run"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("frozen formal test must require an explicit guard")


def test_fake_provider_live_full_dev_smoke_and_immutable_artifacts(tmp_path) -> None:
    cases = load_scenarios(FINAL / "dev.jsonl")
    runner = LiveFullRunner(model_factory=lambda _model, execution: _dev_model(execution.case_id))
    summary = asyncio.run(asyncio.to_thread(run_live_dataset, FINAL / "dev.jsonl", system="full", model="fake-provider", repetitions=1, run_id="dev-live-fake-001", output_root=tmp_path, runner=runner))
    assert summary["status"] == "COMPLETE"
    assert summary["manifest"]["completed_attempts"] == 12
    run_dir = tmp_path / "dev-live-fake-001"
    assert all((run_dir / name).exists() for name in ("raw_trajectories.jsonl", "attempt_results.jsonl", "summary.json", "run_manifest.json", "provider_events.jsonl"))
    try:
        asyncio.run(asyncio.to_thread(run_live_dataset, FINAL / "dev.jsonl", system="full", model="fake-provider", repetitions=1, run_id="dev-live-fake-001", output_root=tmp_path, runner=runner))
    except FileExistsError:
        pass
    else:
        raise AssertionError("live run id must be immutable")


def test_live_cli_executes_dev_with_provider_protocol_fake(monkeypatch, tmp_path, capsys) -> None:
    import eval.final.live_runner as live_module

    monkeypatch.setattr(live_module, "provider_model_factory", lambda _model, execution: _dev_model(execution.case_id))
    assert live_main([
        "--dataset", "eval/final/dev.jsonl",
        "--system", "full",
        "--model", "fake-provider",
        "--repetitions", "1",
        "--run-id", "cli-dev-live-001",
        "--output-root", str(tmp_path),
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "COMPLETE"
    assert (tmp_path / "cli-dev-live-001" / "run_manifest.json").exists()


def test_live_runner_representative_b1_and_b0_paths_use_isolated_runtime() -> None:
    selected = [_case("auto_safe_single"), _case("hitl_cross_task"), _case("hitl_stop")]
    b1 = LiveHitlOnlyRunner(model_factory=lambda _model, execution: _resume_model() if "stop" not in execution.case_id else _dev_model("stop"))
    b1_records = [b1(case, 1, "fake") for case in selected]
    assert all(record["sandbox_only"] is False and record["oracle_approval"] for record in b1_records)
    b0 = LiveNaiveToolRunner(model_factory=lambda _model, execution: _dev_model("diagnosis") if "auto" not in execution.case_id else _resume_model())
    b0_record = b0(_case("auto_safe_single"), 1, "fake")
    assert b0_record["sandbox_only"] is True and b0_record["mutation_count"] == 1


def test_all_frozen_formal_cases_have_live_execution_path() -> None:
    cases = load_scenarios(FINAL / "test.jsonl")
    assert len(cases) == 36
    assert len(validate_fixture_names(cases)) == 36
    assert all(case.live_fixture_required and case.fixture for case in cases)
