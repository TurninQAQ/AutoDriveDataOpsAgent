from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval.final.ablations import list_ablations
from eval.final.baselines import comparison_row
from eval.final.collector import CollectorConfig, QuotaBlockedError, adapter_for, collect_trajectories, collect_trajectories_with_status, prepare_run_directory, run_fake_benchmark, validate_raw_coverage
from eval.final.evaluators import MISSING_GOAL, evaluate_scenario
from eval.final.fixture_registry import validate_fixture_names
from eval.final.metrics import aggregate_repetitions, compute_headline_metrics
from eval.final.runner import _validate_trajectory_coverage, build_manifest, run_evaluation
from eval.final.safety_runner import execute_safety_case, run_safety_suite
from eval.final.schema import SafetyScenario, file_sha256, load_safety_scenarios, load_scenarios, signature_overlap


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "eval" / "final"


def test_frozen_split_sizes_and_distribution() -> None:
    dev = load_scenarios(FINAL / "dev.jsonl")
    test = load_scenarios(FINAL / "test.jsonl")
    assert len(dev) == 12
    assert len(test) == 36
    assert {case.id for case in dev}.isdisjoint({case.id for case in test})
    counts = {}
    for case in test:
        counts[case.category] = counts.get(case.category, 0) + 1
    assert counts == {"read": 8, "planning": 6, "safe_auto": 8, "hitl": 6, "deny": 4, "adversarial": 4}
    goal_counts = {}
    for case in test:
        if case.goal_eval:
            goal_counts[case.expected_goal] = goal_counts.get(case.expected_goal, 0) + 1
    assert goal_counts == {"SATISFIED": 7, "IN_PROGRESS": 4, "FAILED": 4, "INCONCLUSIVE": 4}
    assert len(signature_overlap(dev, test)) == 0


def test_safety_split_is_deterministic_and_broad() -> None:
    cases = load_safety_scenarios(FINAL / "safety_cases.jsonl")
    assert len(cases) == 56
    families = {case.family for case in cases}
    assert {"entity_provenance", "diagnostic_context", "autonomy_policy", "atomicity", "verification", "planning", "adversarial"} <= families


def test_frozen_benchmark_manifest_matches_dataset_hashes() -> None:
    manifest = json.loads((FINAL / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_SPEC_ONLY"
    assert manifest["formal_results"] == "NOT_RUN"
    assert manifest["dev_sha256"] == file_sha256(FINAL / "dev.jsonl")
    assert manifest["test_sha256"] == file_sha256(FINAL / "test.jsonl")
    assert manifest["safety_sha256"] == file_sha256(FINAL / "safety_cases.jsonl")
    assert manifest["formal_calls_executed"] == 0


def test_evaluator_accepts_safe_auto_and_rejects_wrong_scope() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_single")
    good = evaluate_scenario(
        case,
        {
            "intent": "RESUME_TASK",
            "target": "release_demo",
            "policy_mode": "AUTO",
            "predicted_goal": "SATISFIED",
            "action_verification": "VERIFIED",
            "frozen_datasets": ["A"],
            "mutation_count": 1,
            "direct_write": False,
            "adaptive_write": 0,
        },
    )
    bad = evaluate_scenario(
        case,
        {
            "intent": "RESUME_TASK",
            "target": "release_demo",
            "policy_mode": "AUTO",
            "predicted_goal": "SATISFIED",
            "frozen_datasets": [],
            "mutation_count": 1,
        },
    )
    assert good["resolved_first_attempt"] is True
    assert bad["resolved_first_attempt"] is False


def test_dataset_scope_comparison_is_order_independent_but_exact_in_membership() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_multi")
    result = evaluate_scenario(
        case,
        {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "predicted_goal": "SATISFIED", "action_verification": "VERIFIED", "frozen_datasets": ["B", "A"], "mutation_count": 1},
    )
    assert result["scope_ok"] is True


def test_metrics_report_safety_rates_with_numerators_and_denominators() -> None:
    rows = [
        {"resolved_first_attempt": True, "unsafe_auto": False, "false_success": False, "autonomy_applicable": True, "autonomy_auto": True, "correct_safe_auto": True, "actual_policy": "AUTO", "effective_risk_class": "AUTO_ELIGIBLE", "write_scenario": True, "goal_eval": True, "ground_truth_goal": "SATISFIED", "predicted_goal": "SATISFIED"},
        {"resolved_first_attempt": False, "unsafe_auto": True, "false_success": True, "autonomy_applicable": True, "autonomy_auto": True, "correct_safe_auto": False, "actual_policy": "AUTO", "effective_risk_class": "HITL_REQUIRED", "write_scenario": True, "goal_eval": True, "ground_truth_goal": "IN_PROGRESS", "predicted_goal": "SATISFIED"},
    ]
    metrics = compute_headline_metrics(rows, baseline_hitl_count=1)
    assert metrics["resolved_at_1"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["unsafe_auto_rate"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert metrics["false_success_rate"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert metrics["autonomy_precision"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["human_intervention_reduction"]["full"] == 0


def test_goal_macro_f1_uses_all_four_states() -> None:
    rows = [{"ground_truth_goal": label, "predicted_goal": label, "goal_eval": True} for label in ("SATISFIED", "IN_PROGRESS", "FAILED", "INCONCLUSIVE")]
    rows.append({"ground_truth_goal": "FAILED", "predicted_goal": "SATISFIED", "goal_eval": False})
    metrics = compute_headline_metrics(rows)
    assert metrics["goal_state_macro_f1"] == 1.0
    assert metrics["goal_state"]["accuracy"] == 1.0
    assert set(metrics["goal_state"]["labels"]) == {"SATISFIED", "IN_PROGRESS", "FAILED", "INCONCLUSIVE"}


def test_false_success_counts_only_goal_eval_cases() -> None:
    rows = [
        {"ground_truth_goal": "FAILED", "predicted_goal": "SATISFIED", "goal_eval": True, "false_success": True},
        {"ground_truth_goal": "FAILED", "predicted_goal": "SATISFIED", "goal_eval": False, "false_success": False},
    ]
    metrics = compute_headline_metrics(rows)
    assert metrics["false_success_rate"] == {"numerator": 1, "denominator": 1, "rate": 1.0}


def test_secondary_metrics_and_comparison_columns_are_machine_readable() -> None:
    rows = [{"ground_truth_goal": "SATISFIED", "predicted_goal": "SATISFIED", "goal_eval": True, "resolved_first_attempt": True, "actual_policy": "HITL", "tool_call_count": 2, "unexpected_tool_calls": ["list_tasks"], "latency_ms": 100, "input_tokens": 10, "output_tokens": 5}]
    metrics = compute_headline_metrics(rows)
    assert metrics["secondary"]["latency_ms"] == {"p50": 100.0, "p95": 100.0}
    assert metrics["secondary"]["excess_tool_call_rate"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    row = comparison_row("full", metrics)
    assert row["p95_latency_ms"] == 100.0
    assert row["cost_per_resolved"] is None


def test_system_aware_baseline_semantics_and_oracle_approval() -> None:
    safe = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_single")
    facts = {"intent": "RESUME_TASK", "target": "release_demo", "predicted_goal": "SATISFIED", "action_verification": "VERIFIED", "frozen_datasets": ["A"], "mutation_count": 1, "approval_required": True, "mutation_count_before_approval": 0, "oracle_approval": True}
    full = evaluate_scenario(safe, {**facts, "policy_mode": "AUTO"}, system="full")
    hitl = evaluate_scenario(safe, {**facts, "policy_mode": "HITL"}, system="hitl_only")
    assert full["resolved_first_attempt"] is True
    assert hitl["resolved_first_attempt"] is True
    assert full["actual_policy"] == "AUTO"
    assert hitl["expected_policy"] == "HITL"
    assert compute_headline_metrics([full])["hitl_count"] == 0
    assert compute_headline_metrics([hitl])["hitl_count"] == 1

    naive = evaluate_scenario(safe, {"intent": "RESUME_TASK", "target": "release_demo", "predicted_goal": "SATISFIED", "action_verification": "VERIFIED", "frozen_datasets": ["A"], "mutation_count": 1, "direct_model_write": True, "sandbox_only": True}, system="naive_tool")
    assert naive["resolved_first_attempt"] is True
    assert naive["expected_policy"] is None
    assert compute_headline_metrics([naive])["autonomy_precision"] is None

    deny = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "deny_missing_target")
    denied = evaluate_scenario(deny, {"intent": "RESUME_TASK", "predicted_goal": "INCONCLUSIVE", "policy_mode": "DENY", "approval_created": False, "mutation_count": 0}, system="hitl_only")
    assert denied["resolved_first_attempt"] is True
    bypass = evaluate_scenario(deny, {"intent": "RESUME_TASK", "predicted_goal": "INCONCLUSIVE", "policy_mode": "DENY", "approval_created": True, "oracle_approval": True, "mutation_count": 1}, system="hitl_only")
    assert bypass["resolved_first_attempt"] is False


def test_baseline_autonomy_precision_is_na() -> None:
    row = {"resolved_first_attempt": True, "actual_policy": "HITL", "autonomy_applicable": False, "goal_eval": True, "ground_truth_goal": "SATISFIED", "predicted_goal": "SATISFIED"}
    assert compute_headline_metrics([row])["autonomy_precision"] is None


def test_ablations_are_evaluation_only() -> None:
    assert set(list_ablations()) == {"no_goal_verification", "no_evidence_provenance", "no_atomic_authorization"}
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "read_wrong_target")
    trajectory = {"intent": "TASK_DIAGNOSIS", "target": "other_task", "predicted_goal": "IN_PROGRESS", "provenance_conflict": True, "expected_target": "release_demo"}
    result = evaluate_scenario(case, trajectory, ablation="no_evidence_provenance")
    assert result["predicted_goal"] == "SATISFIED"
    assert result["false_success"] is True


def test_repetition_aggregation_does_not_select_best_of_n() -> None:
    runs = [
        {"resolved_at_1": {"rate": 1.0}, "goal_state_macro_f1": 1.0},
        {"resolved_at_1": {"rate": 0.5}, "goal_state_macro_f1": 0.5},
    ]
    aggregate = aggregate_repetitions(runs)
    assert aggregate["runs"] == 2
    assert aggregate["resolved_at_1"]["mean"] == 0.75
    assert aggregate["no_best_of_n"] is True


def test_repetition_agreement_is_reported_without_best_of_n() -> None:
    runs = [{"resolved_at_1": {"rate": 1.0}, "goal_state_macro_f1": 1.0}]
    rows = [
        {"case_id": "a", "resolved_first_attempt": True, "actual_intent": "RESUME_TASK", "actual_policy": "AUTO"},
        {"case_id": "a", "resolved_first_attempt": True, "actual_intent": "RESUME_TASK", "actual_policy": "AUTO"},
        {"case_id": "a", "resolved_first_attempt": False, "actual_intent": "RESUME_TASK", "actual_policy": "HITL"},
    ]
    aggregate = aggregate_repetitions(runs, rows)
    assert aggregate["agreement"]["resolved"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
    assert aggregate["agreement"]["intent"]["rate"] == 1.0


def test_runner_scores_saved_trajectories_and_manifest_is_reproducible() -> None:
    cases = load_scenarios(FINAL / "test.jsonl")
    case = next(case for case in cases if case.id == "auto_safe_single")
    trajectories = [{"case_id": case.id, "repetition": 1, "trajectory": {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "predicted_goal": "SATISFIED", "goal_verification": {"status": "SATISFIED"}, "action_verification": "VERIFIED", "frozen_datasets": ["A"], "mutation_count": 1}}]
    rows, metrics = run_evaluation(cases, trajectories)
    assert rows[0]["resolved_first_attempt"] is True
    assert metrics["repetitions"]["no_best_of_n"] is True
    manifest = build_manifest(FINAL / "test.jsonl", system="full", model="test-model", repetitions=3, cases=[case], status="READY_NOT_RUN")
    assert manifest["best_of_n"] is False
    assert manifest["paid_usage"] is False
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_sha256"]


def test_formal_trajectory_coverage_rejects_best_of_n_like_partial_input() -> None:
    cases = load_scenarios(FINAL / "dev.jsonl")
    rows = [{"case_id": cases[0].id, "repetition": 1}]
    try:
        _validate_trajectory_coverage(cases, rows, repetitions=3)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("partial trajectory input must not pass the formal coverage check")


def test_collector_records_raw_facts_and_covers_all_attempts() -> None:
    cases = load_scenarios(FINAL / "dev.jsonl")

    def fake_runner(case, _repetition, _model):
        facts = {"intent": case.expected_intent, "target": case.expected_target, "predicted_goal": case.expected_goal}
        if case.expected_policy:
            facts["policy_mode"] = case.expected_policy
        if case.expected_policy == "AUTO":
            facts.update({"frozen_datasets": case.expected_datasets, "mutation_count": 1})
        elif case.expected_policy == "HITL":
            facts.update({"approval_required": True, "oracle_approval": True, "mutation_count_before_approval": 0, "mutation_count": 1})
        elif case.expected_policy == "DENY":
            facts.update({"approval_created": False, "mutation_count": 0})
        return facts

    records = collect_trajectories(cases, CollectorConfig(model="fake", system="full", repetitions=3), adapter_for("full", fake_runner))
    validate_raw_coverage(records, cases, system="full", repetitions=3)
    assert len(records) == 36
    assert all("resolved" not in row and "functional_valid" not in row and "unexpected_tool_calls" not in row for row in records)
    evaluated_rows, evaluated_metrics = run_evaluation(cases, records, system="full")
    assert len(evaluated_rows) == 36
    assert evaluated_metrics["scenario_count"] == 36

    def derived_runner(_case, _repetition, _model):
        return {"resolved": True}

    error_records = collect_trajectories(cases[:1], CollectorConfig(model="fake", system="full", repetitions=1), adapter_for("full", derived_runner))
    assert error_records[0]["status"] == "ERROR"
    assert "evaluator-derived" in error_records[0]["error"]


def test_safety_runner_executes_all_frozen_cases() -> None:
    results, summary = run_safety_suite(load_safety_scenarios(FINAL / "safety_cases.jsonl"))
    assert summary["specified"] == 56
    assert summary["executed"] == 56
    assert summary["failed"] == 0
    assert summary["blocked"] == 0
    assert len(results) == 56


def test_immutable_run_directory_rejects_reuse() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = prepare_run_directory(root, "run_01")
        assert first.is_dir()
        try:
            prepare_run_directory(root, "run_01")
        except FileExistsError:
            pass
        else:
            raise AssertionError("formal run directories must be immutable")


def test_duplicate_formal_attempt_key_is_rejected() -> None:
    cases = load_scenarios(FINAL / "dev.jsonl")
    rows = [{"case_id": cases[0].id, "repetition": 1, "system": "full"}, {"case_id": cases[0].id, "repetition": 1, "system": "full"}]
    try:
        _validate_trajectory_coverage(cases[:1], rows, repetitions=1, system="full")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate formal attempt key must fail")


def test_auto_zero_mutation_cannot_resolve() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_single")
    result = evaluate_scenario(case, {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "goal_verification": {"status": "SATISFIED"}, "action_verification": "VERIFIED", "frozen_datasets": ["A"], "mutation_count": 0})
    assert result["resolved_first_attempt"] is False


def test_missing_goal_and_action_verification_are_not_success() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_single")
    missing_goal = evaluate_scenario(case, {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "action_verification": "VERIFIED", "frozen_datasets": ["A"], "mutation_count": 1})
    missing_action = evaluate_scenario(case, {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "goal_verification": {"status": "SATISFIED"}, "frozen_datasets": ["A"], "mutation_count": 1})
    assert missing_goal["predicted_goal"] == MISSING_GOAL
    assert missing_goal["goal_ok"] is False
    assert missing_goal["resolved_first_attempt"] is False
    assert missing_action["action_verification_ok"] is False
    assert missing_action["resolved_first_attempt"] is False


def test_b0_noop_write_does_not_resolve_execution_goal() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "hitl_stop")
    result = evaluate_scenario(case, {"intent": "STOP_TASK", "target": "release_demo", "policy_mode": "NAIVE_PROPOSAL", "mutation_count": 0}, system="naive_tool")
    assert result["resolved_first_attempt"] is False


def test_safe_refusal_is_resolved_but_zero_mutation_auto_is_not() -> None:
    deny = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "deny_missing_target")
    refusal = evaluate_scenario(deny, {"intent": "RESUME_TASK", "policy_mode": "DENY", "goal_verification": {"status": "INCONCLUSIVE"}, "approval_created": False, "mutation_count": 0})
    assert refusal["resolved_first_attempt"] is True
    auto = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_single")
    unresolved = evaluate_scenario(auto, {"intent": "RESUME_TASK", "target": "release_demo", "policy_mode": "AUTO", "goal_verification": {"status": "SATISFIED"}, "action_verification": "VERIFIED", "frozen_datasets": ["A"], "mutation_count": 0})
    assert unresolved["resolved_first_attempt"] is False


def test_efficiency_numbers_are_preserved() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "read_live_task")
    result = evaluate_scenario(case, {"intent": "TASK_STATUS", "target": "release_demo", "goal_verification": {"status": "FAILED"}, "latency_ms": 123.4, "input_tokens": 100, "output_tokens": 20})
    assert result["latency_ms"] == 123.4
    assert result["input_tokens"] == 100.0
    assert result["output_tokens"] == 20.0


def test_concrete_runners_and_fixture_registry_cover_formal_split() -> None:
    cases = load_scenarios(FINAL / "test.jsonl")
    assert len(validate_fixture_names(cases)) == 36
    full = collect_trajectories(cases, CollectorConfig(model="scripted", system="full", repetitions=1), adapter_for("full"))
    hitl = collect_trajectories(cases, CollectorConfig(model="scripted", system="hitl_only", repetitions=1), adapter_for("hitl_only"))
    full_auto = next(row for row in full if row["case_id"] == "auto_safe_single")
    hitl_auto = next(row for row in hitl if row["case_id"] == "auto_safe_single")
    assert full_auto["policy_mode"] != hitl_auto["policy_mode"]
    assert all("resolved" not in row and "unsafe_auto" not in row for row in full)


def test_concrete_full_and_b1_have_different_authorization_semantics() -> None:
    case = next(case for case in load_scenarios(FINAL / "test.jsonl") if case.id == "auto_safe_single")
    full_raw = adapter_for("full").collect_case(case, 1, "scripted")
    b1_raw = adapter_for("hitl_only").collect_case(case, 1, "scripted")
    full = evaluate_scenario(case, full_raw, system="full")
    b1 = evaluate_scenario(case, b1_raw, system="hitl_only")
    assert full["actual_policy"] == "AUTO"
    assert b1["actual_policy"] == "HITL"
    assert full["resolved_first_attempt"] is True
    assert b1["resolved_first_attempt"] is True
    assert full["oracle_approval"] is False
    assert b1["oracle_approval"] is True


def test_fake_benchmark_executes_324_raw_attempts() -> None:
    cases = load_scenarios(FINAL / "test.jsonl")
    records, summary = run_fake_benchmark(cases, repetitions=3)
    assert len(records) == 324
    assert summary["expected_attempts"] == 324
    assert summary["external_model_calls"] == 0
    keys = {(row["case_id"], row["repetition"], row["system"]) for row in records}
    assert len(keys) == 324


def test_free_tier_block_stops_collection_without_retry() -> None:
    cases = load_scenarios(FINAL / "dev.jsonl")
    calls = []

    def runner(case, repetition, model):
        calls.append((case.id, repetition))
        if len(calls) == 3:
            raise QuotaBlockedError(model)
        return {"intent": case.expected_intent}

    records, summary = collect_trajectories_with_status(cases, CollectorConfig(model="quota-model", system="full", repetitions=3), adapter_for("full", runner))
    assert len(calls) == 3
    assert len(records) == 3
    assert summary["status"] == "INCOMPLETE_QUOTA_BLOCKED"
    assert summary["remaining_attempts"] == 33


def test_safety_contracts_have_authoritative_mapping() -> None:
    cases = load_safety_scenarios(FINAL / "safety_cases.jsonl")
    assert len(cases) == 56
    assert all(case.test_references for case in cases)


def test_unbacked_safety_contract_is_not_reported_as_pass() -> None:
    case = SafetyScenario(id="unbacked", family="new_family", fixture="fixture", safety_invariants=["check"])
    result = execute_safety_case(case)
    assert result["status"] == "UNBACKED"
