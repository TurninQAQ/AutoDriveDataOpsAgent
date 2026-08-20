from __future__ import annotations

import asyncio
import json

import pytest

from platform_eval.resumable_collection import (
    CollectionAttemptError,
    CollectionCompatibilityError,
    CollectionSafetyError,
    FrozenCaseHashMismatch,
    atomic_write_json,
    build_manifest,
    finalize_collection,
    read_json,
    run_resumable_cases,
    summarize_manifest,
    validate_frozen_case_file,
    validate_manifest_compatibility,
)


def run(coro):
    return asyncio.run(coro)


def case(case_id: str):
    return {"id": case_id, "query": f"query-{case_id}"}


def manifest_for(tmp_path, cases):
    return build_manifest(
        version="v1.5.2",
        model="qwen-plus",
        provider="qwen",
        endpoint_host="dashscope.aliyuncs.com",
        frozen_case_file=str(tmp_path / "cases.jsonl"),
        frozen_case_sha256="frozen",
        production_base_commit="commit",
        request_timeout_sec=45.0,
        formal_eval_retry_attempts=2,
        max_steps=8,
        max_tool_calls=6,
        max_identical_tool_calls=2,
        max_consecutive_tool_failures=2,
        max_case_attempts=2,
        case_timeout_sec=120.0,
        tool_catalog_sha256="catalog",
        cases=cases,
    )


def success_result(case_id: str):
    return {
        "sample": {"case_id": case_id, "tool_calls": [], "known_regression_probe": False},
        "evaluation": {"case_id": case_id, "scenario_complete": True},
        "provider_stats": {"requests_attempted": 1, "requests_completed": 1, "timeouts": 0, "retries": 0, "errors": 0},
    }


def provider_failure(case_id: str):
    return CollectionAttemptError(
        f"provider timeout for {case_id}",
        failure_type="provider_timeout",
        provider_failure=True,
        operation_type="decide_next",
        step_index=1,
        details={"provider_stats": {"requests_attempted": 1, "requests_completed": 0, "timeouts": 1, "retries": 1, "errors": 1}},
    )


def test_atomic_write_and_case_manifest_checkpoint(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"status": "PENDING", "value": 1})
    assert read_json(path)["value"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_provider_failure_does_not_stop_later_cases_and_partial_metrics_are_labeled(tmp_path):
    cases = [case("a"), case("b"), case("c")]
    manifest = manifest_for(tmp_path, cases)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    async def execute(item, attempt):
        calls.append((item["id"], attempt))
        if item["id"] == "b":
            raise provider_failure("b")
        return success_result(item["id"])

    run(
        run_resumable_cases(
            cases=cases,
            manifest=manifest,
            manifest_path=manifest_path,
            artifact_root=tmp_path,
            execute_case=execute,
            max_case_attempts=1,
        )
    )

    assert calls == [("a", 1), ("b", 1), ("c", 1)]
    assert manifest["cases"]["a"]["status"] == "COMPLETE"
    assert manifest["cases"]["b"]["status"] == "PROVIDER_ERROR"
    assert manifest["cases"]["c"]["status"] == "COMPLETE"
    assert (tmp_path / "failed_attempts/b.attempt_1.json").exists()
    partial = finalize_collection(
        manifest=manifest,
        artifact_root=tmp_path,
        output_dir=tmp_path,
        allow_partial=True,
    )
    assert partial["FULL_BASELINE"] is False
    assert partial["completed_case_count"] == 2
    assert partial["aggregate"] is None
    assert partial["aggregate_completed_cases"]["case_count"] == 2


def test_resume_skips_complete_cases_and_restarts_failed_case_from_initial_attempt(tmp_path):
    cases = [case("a"), case("b")]
    manifest = manifest_for(tmp_path, cases)
    manifest_path = tmp_path / "manifest.json"
    first_calls = []

    async def first_execute(item, attempt):
        first_calls.append((item["id"], attempt))
        if item["id"] == "b":
            raise provider_failure("b")
        return success_result(item["id"])

    run(run_resumable_cases(cases=cases, manifest=manifest, manifest_path=manifest_path, artifact_root=tmp_path, execute_case=first_execute, max_case_attempts=1))

    resumed_calls = []

    async def resumed_execute(item, attempt):
        resumed_calls.append((item["id"], attempt))
        return success_result(item["id"])

    run(run_resumable_cases(cases=cases, manifest=manifest, manifest_path=manifest_path, artifact_root=tmp_path, execute_case=resumed_execute, max_case_attempts=1))
    assert first_calls == [("a", 1), ("b", 1)]
    assert resumed_calls == [("b", 2)]
    assert manifest["cases"]["a"]["status"] == "COMPLETE"
    assert manifest["cases"]["b"]["status"] == "COMPLETE"
    assert manifest["cases"]["b"]["completed_attempts"] == 1


def test_case_attempt_budget_and_failed_attempt_history(tmp_path):
    cases = [case("retry")]
    manifest = manifest_for(tmp_path, cases)
    manifest_path = tmp_path / "manifest.json"
    attempts = []

    async def execute(item, attempt):
        attempts.append(attempt)
        raise provider_failure(item["id"])

    run(run_resumable_cases(cases=cases, manifest=manifest, manifest_path=manifest_path, artifact_root=tmp_path, execute_case=execute, max_case_attempts=2))
    assert attempts == [1, 2]
    assert manifest["cases"]["retry"]["status"] == "PROVIDER_ERROR"
    assert manifest["cases"]["retry"]["provider_failures"] == 2
    assert len(manifest["cases"]["retry"]["failed_attempt_paths"]) == 2


def test_full_finalize_requires_all_cases_complete(tmp_path):
    cases = [case("a"), case("b")]
    manifest = manifest_for(tmp_path, cases)
    manifest_path = tmp_path / "manifest.json"

    async def execute(item, attempt):
        return success_result(item["id"]) if item["id"] == "a" else (_ for _ in ()).throw(provider_failure("b"))

    run(run_resumable_cases(cases=cases, manifest=manifest, manifest_path=manifest_path, artifact_root=tmp_path, execute_case=execute, max_case_attempts=1))
    with pytest.raises(CollectionCompatibilityError):
        finalize_collection(manifest=manifest, artifact_root=tmp_path, output_dir=tmp_path, allow_partial=False)


def test_resume_compatibility_and_frozen_hash_guards(tmp_path):
    case_file = tmp_path / "frozen.jsonl"
    case_file.write_text('{"id":"x"}\n', encoding="utf-8")
    with pytest.raises(FrozenCaseHashMismatch):
        validate_frozen_case_file(case_file, expected_sha256="different")

    cases = [case("x")]
    manifest = manifest_for(tmp_path, cases)
    expected = dict(manifest)
    expected["request_timeout_sec"] = 30.0
    with pytest.raises(CollectionCompatibilityError):
        validate_manifest_compatibility(manifest, expected, case_ids=["x"])


def test_unsafe_write_execution_aborts_collection(tmp_path):
    cases = [case("unsafe"), case("later")]
    manifest = manifest_for(tmp_path, cases)
    manifest_path = tmp_path / "manifest.json"
    calls = []

    async def execute(item, attempt):
        calls.append(item["id"])
        return {"unsafe_write_executed": True, "operation_type": "decide_next", "safety_event": "delete_task executed"}

    with pytest.raises(CollectionSafetyError):
        run(run_resumable_cases(cases=cases, manifest=manifest, manifest_path=manifest_path, artifact_root=tmp_path, execute_case=execute, max_case_attempts=1))
    assert calls == ["unsafe"]
    assert manifest["status"] == "ABORTED_SAFETY"
    assert manifest["cases"]["later"]["status"] == "PENDING"


def test_manifest_summary_counts_resume_and_statuses(tmp_path):
    cases = [case("a")]
    manifest = manifest_for(tmp_path, cases)
    assert summarize_manifest(manifest)["statuses"] == {"PENDING": 1}
    assert json.loads(json.dumps(manifest))["cases"]["a"]["status"] == "PENDING"


def test_partial_finalize_distinguishes_forbidden_read_route_from_write_execution(tmp_path):
    cases = [case("read_route")]
    manifest = manifest_for(tmp_path, cases)
    manifest["cases"]["read_route"]["status"] = "COMPLETE"
    manifest["cases"]["read_route"]["artifact_path"] = "cases/read_route.json"
    atomic_write_json(
        tmp_path / "cases/read_route.json",
        {
            "status": "COMPLETE",
            "case_id": "read_route",
            "sample": {"case_id": "read_route", "tool_calls": [{"tool": "get_task_detail", "arguments": {}}]},
            "evaluation": {
                "case_id": "read_route",
                "category": "live_only",
                "scenario_complete": False,
                "forbidden_write_execution": True,
                "intent_ok": False,
                "loop_termination_ok": True,
            },
        },
    )
    result = finalize_collection(manifest=manifest, artifact_root=tmp_path, output_dir=tmp_path, allow_partial=True)
    assert result["aggregate"]["forbidden_write_execution_rate"] == 0.0
    assert result["cases"][0]["forbidden_tool_attempt"] is True
    assert result["cases"][0]["forbidden_write_tools_executed"] == []
