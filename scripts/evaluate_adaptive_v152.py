#!/usr/bin/env python3
"""Run the frozen V1.5 Adaptive scenarios with case-level resume support."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from platform_agent.memory import ConversationStore
from platform_agent.provider_preflight import classify_provider_failure, run_qwen_preflight
from platform_agent.qwen import QwenReadOnlyModel
from platform_agent.tool_catalog import build_read_only_tool_catalog
from platform_agent.workflow import build_agent_runtime
from platform_eval.adaptive import evaluate_adaptive_trajectory, load_adaptive_cases
from platform_eval.resumable_collection import (
    EXPECTED_V1_5_0_CASES_SHA256,
    CollectionAttemptError,
    CollectionCompatibilityError,
    CollectionSafetyError,
    FrozenCaseHashMismatch,
    atomic_write_json,
    build_manifest,
    file_sha256,
    finalize_collection,
    read_json,
    run_resumable_cases,
    summarize_manifest,
    utc_now,
    validate_frozen_case_file,
    validate_manifest_compatibility,
)
from platform_mcp.server import READ_ONLY_TOOL_NAMES, WRITE_TOOL_NAMES
from platform_integrations.model_retry import ModelRequestError, ModelRetryPolicy
from platform_observability.redaction import redact_text

from evaluate_adaptive_v150 import CountingClient, ScenarioFixtureToolClient, _trajectory


def _endpoint_host(value: str) -> str:
    parsed = urlparse(value)
    return parsed.netloc or parsed.path.split("/", 1)[0]


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _catalog_sha256() -> str:
    payload = json.dumps(build_read_only_tool_catalog(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bounded_error(exc: BaseException) -> str:
    return redact_text(str(exc))[:1000]


def _provider_stats(
    counter: dict[str, int], metrics: dict[str, int], baseline: dict[str, int] | None = None
) -> dict[str, int]:
    baseline = baseline or {"requests": 0, "completed": 0}
    return {
        "requests_attempted": max(0, int(counter.get("requests", 0)) - int(baseline.get("requests", 0))),
        "requests_completed": max(0, int(counter.get("completed", 0)) - int(baseline.get("completed", 0))),
        "timeouts": int(metrics.get("provider_timeout", 0)),
        "retries": int(metrics.get("retries", 0)),
        "errors": int(metrics.get("provider_errors", 0)),
    }


class OperationTrackingModel:
    """Track operation type/step without recording prompts or private reasoning."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.current_operation = "unknown"
        self.current_step: int | None = None
        self.supports_adaptive = getattr(delegate, "supports_adaptive", False)
        self.requires_tool_descriptions = getattr(delegate, "requires_tool_descriptions", True)

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    async def plan(self, *args, **kwargs):
        self.current_operation = "plan"
        self.current_step = None
        return await self.delegate.plan(*args, **kwargs)

    async def decide_next(self, *args, **kwargs):
        self.current_operation = "decide_next"
        self.current_step = kwargs.get("step_index")
        return await self.delegate.decide_next(*args, **kwargs)

    async def synthesize(self, *args, **kwargs):
        self.current_operation = "synthesize"
        self.current_step = None
        return await self.delegate.synthesize(*args, **kwargs)


def _as_attempt_error(exc: BaseException, tracker: OperationTrackingModel, stats: dict[str, int]) -> CollectionAttemptError:
    if isinstance(exc, ModelRequestError):
        failure_type = exc.failure_type
        provider_failure = True
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        failure_type = classify_provider_failure(exc)
        provider_failure = True
    else:
        failure_type = "agent_error"
        provider_failure = False
    return CollectionAttemptError(
        _bounded_error(exc),
        failure_type=failure_type,
        provider_failure=provider_failure,
        operation_type=tracker.current_operation,
        step_index=tracker.current_step,
        details={"provider_stats": stats},
    )


def _case_executor(
    model,
    counter: dict[str, int],
    metrics: dict[str, int],
    session_root: Path,
    provider_baseline: dict[str, int],
    *,
    max_steps: int,
    max_tool_calls: int,
    max_identical_tool_calls: int,
    max_consecutive_tool_failures: int,
    case_timeout_sec: float,
):
    async def execute(case: dict[str, Any], attempt: int) -> dict[str, Any]:
        fixture = ScenarioFixtureToolClient(case.get("fixture_results", {}))
        tracker = OperationTrackingModel(model)
        thread_root = session_root / f"{case['id']}.attempt_{attempt}"
        thread_root.mkdir(parents=True, exist_ok=True)
        runtime = build_agent_runtime(
            "sequential",
            tracker,
            fixture,
            ConversationStore(thread_root),
            max_tool_calls=max_tool_calls,
            max_steps=max_steps,
            max_identical_tool_calls=max_identical_tool_calls,
            max_consecutive_tool_failures=max_consecutive_tool_failures,
        )
        try:
            response = await asyncio.wait_for(runtime.run(case["query"], case["id"]), timeout=case_timeout_sec)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except BaseException as exc:
            raise _as_attempt_error(exc, tracker, _provider_stats(counter, metrics, provider_baseline)) from None

        unsafe_executed = any(call.name in WRITE_TOOL_NAMES for call in fixture.calls)
        if unsafe_executed:
            return {
                "unsafe_write_executed": True,
                "operation_type": tracker.current_operation,
                "step_index": tracker.current_step,
                "safety_event": "write tool was executed by the fixture client",
                "provider_stats": _provider_stats(counter, metrics, provider_baseline),
            }
        sample = _trajectory(case, response, fixture)
        sample["known_regression_probe"] = bool(case.get("known_regression_probe"))
        evaluation = evaluate_adaptive_trajectory(case, sample)
        evaluation["actual_intent"] = sample.get("final_intent", "")
        evaluation["initial_intent"] = sample.get("initial_intent", "")
        return {
            "sample": sample,
            "evaluation": evaluation,
            "provider_stats": _provider_stats(counter, metrics),
        }

    return execute


def _manifest_expected(args, case_file: Path, case_hash: str) -> dict[str, Any]:
    retry_attempts = int(args.model_retry_attempts)
    return {
        "version": "v1.5.2",
        "agent_model": args.model,
        "provider": "qwen",
        "endpoint_host": _endpoint_host(os.environ.get("DASHSCOPE_OPENAI_BASE_URL", "")),
        "frozen_case_sha256": case_hash,
        "production_base_commit": _git_head(),
        "request_timeout_sec": float(args.request_timeout_sec),
        "formal_eval_retry_attempts": retry_attempts,
        "max_steps": int(args.max_steps),
        "max_tool_calls": int(args.max_tool_calls),
        "max_identical_tool_calls": int(args.max_identical_tool_calls),
        "max_consecutive_tool_failures": int(args.max_consecutive_tool_failures),
        "max_case_attempts": int(args.max_case_attempts),
        "case_timeout_sec": float(args.case_timeout_sec),
        "tool_catalog_sha256": _catalog_sha256(),
        "frozen_case_file": str(case_file),
    }


def _create_or_load_manifest(args, cases: list[dict[str, Any]], case_file: Path, case_hash: str, root: Path):
    manifest_path = root / "manifest.json"
    expected = _manifest_expected(args, case_file, case_hash)
    if manifest_path.exists():
        if not args.resume and not (args.finalize or args.finalize_partial):
            raise CollectionCompatibilityError(f"manifest already exists; use --resume: {manifest_path}")
        manifest = read_json(manifest_path)
        validate_manifest_compatibility(manifest, expected, case_ids=[str(case["id"]) for case in cases])
        if args.resume and not (args.finalize or args.finalize_partial):
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest["collection_invocations"] = int(manifest.get("collection_invocations", 0)) + 1
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
        return manifest, manifest_path, False
    if args.resume or args.finalize or args.finalize_partial:
        raise CollectionCompatibilityError(f"--resume requested but manifest is missing: {manifest_path}")
    manifest = build_manifest(
        version="v1.5.2",
        model=args.model,
        provider="qwen",
        endpoint_host=expected["endpoint_host"],
        frozen_case_file=str(case_file),
        frozen_case_sha256=case_hash,
        production_base_commit=expected["production_base_commit"],
        request_timeout_sec=expected["request_timeout_sec"],
        formal_eval_retry_attempts=expected["formal_eval_retry_attempts"],
        max_steps=expected["max_steps"],
        max_tool_calls=expected["max_tool_calls"],
        max_identical_tool_calls=expected["max_identical_tool_calls"],
        max_consecutive_tool_failures=expected["max_consecutive_tool_failures"],
        max_case_attempts=expected["max_case_attempts"],
        case_timeout_sec=expected["case_timeout_sec"],
        tool_catalog_sha256=expected["tool_catalog_sha256"],
        cases=cases,
    )
    atomic_write_json(manifest_path, manifest)
    return manifest, manifest_path, True


async def _run(args) -> int:
    case_file = args.cases.resolve()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_hash = validate_frozen_case_file(case_file)
    cases = load_adaptive_cases(case_file)
    manifest, manifest_path, new_manifest = _create_or_load_manifest(args, cases, case_file, case_hash, root)

    if args.finalize or args.finalize_partial:
        result = finalize_collection(
            manifest=manifest,
            artifact_root=root,
            output_dir=root,
            allow_partial=args.finalize_partial,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["FULL_BASELINE"] or args.finalize_partial else 2

    if args.rerun_all:
        force_case_ids = {str(case["id"]) for case in cases}
    else:
        force_case_ids = set(args.rerun_case or [])
    unknown_rerun = force_case_ids - {str(case["id"]) for case in cases}
    if unknown_rerun:
        raise CollectionCompatibilityError(f"unknown --rerun-case IDs: {sorted(unknown_rerun)}")
    if force_case_ids and not args.resume:
        raise CollectionCompatibilityError("--rerun-case/--rerun-all requires --resume")

    # Formal evaluation retry policy is process-local and does not modify .env
    # or production defaults.
    os.environ["PLATFORM_MODEL_RETRY_ATTEMPTS"] = str(args.model_retry_attempts)
    if not os.environ.get("DASHSCOPE_API_KEY") or not os.environ.get("DASHSCOPE_OPENAI_BASE_URL"):
        manifest["status"] = "BLOCKED_PROVIDER_PREFLIGHT"
        manifest["provider_preflight"] = {"status": "FAIL", "failure_types": ["missing_credentials_or_endpoint"]}
        atomic_write_json(manifest_path, manifest)
        print("BLOCKED_PROVIDER_PREFLIGHT: missing DASHSCOPE credentials or endpoint")
        return 2

    from openai import AsyncOpenAI

    counter = {"requests": 0, "completed": 0}
    metrics: dict[str, int] = {}
    raw_client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_OPENAI_BASE_URL"],
        timeout=float(args.request_timeout_sec),
    )
    client = CountingClient(raw_client, counter)
    preflight = await run_qwen_preflight(
        client,
        model=args.model,
        checks=args.preflight_checks,
        timeout_sec=args.preflight_timeout_sec,
    )
    manifest["provider_preflight"] = preflight.as_dict()
    atomic_write_json(manifest_path, manifest)
    if not preflight.ok:
        manifest["status"] = "BLOCKED_PROVIDER_PREFLIGHT"
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({"status": "BLOCKED_PROVIDER_PREFLIGHT", "provider_preflight": preflight.as_dict()}, indent=2))
        return 2

    model = QwenReadOnlyModel(
        model=args.model,
        client=client,
        request_timeout_sec=args.request_timeout_sec,
        metrics=metrics,
    )
    formal_provider_baseline = {"requests": counter["requests"], "completed": counter["completed"]}
    with tempfile.TemporaryDirectory(prefix="autodrive-v152-sessions-") as session_root:
        try:
            await run_resumable_cases(
                cases=cases,
                manifest=manifest,
                manifest_path=manifest_path,
                artifact_root=root,
                execute_case=_case_executor(
                    model,
                    counter,
                    metrics,
                    Path(session_root),
                    provider_baseline=formal_provider_baseline,
                    max_steps=args.max_steps,
                    max_tool_calls=args.max_tool_calls,
                    max_identical_tool_calls=args.max_identical_tool_calls,
                    max_consecutive_tool_failures=args.max_consecutive_tool_failures,
                    case_timeout_sec=args.case_timeout_sec,
                ),
                max_case_attempts=args.max_case_attempts,
                force_case_ids=force_case_ids,
            )
        except CollectionSafetyError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("COLLECTION_INTERRUPTED: manifest preserved")
            return 130
    manifest["provider_stats"] = _provider_stats(counter, metrics, formal_provider_baseline)
    manifest["provider_stats_scope"] = "formal_collection"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest.get("status"), "summary": summarize_manifest(manifest), "provider_stats": manifest["provider_stats"]}, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("eval/v1_5_0/adaptive_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_acceptance/v1.5.2_adaptive"))
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-case-attempts", type=int, default=2)
    parser.add_argument("--model-retry-attempts", type=int, default=2)
    parser.add_argument("--request-timeout-sec", type=float, default=45.0)
    parser.add_argument("--case-timeout-sec", type=float, default=120.0)
    parser.add_argument("--preflight-checks", type=int, default=2)
    parser.add_argument("--preflight-timeout-sec", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-identical-tool-calls", type=int, default=2)
    parser.add_argument("--max-consecutive-tool-failures", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-case", action="append")
    parser.add_argument("--rerun-all", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--finalize-partial", action="store_true")
    return parser.parse_args()


async def main(args) -> int:
    try:
        return await _run(args)
    except FrozenCaseHashMismatch as exc:
        print(f"FROZEN_CASE_HASH_MISMATCH: {exc}")
        return 2
    except (CollectionCompatibilityError, CollectionSafetyError) as exc:
        print(f"COLLECTION_BLOCKED: {redact_text(str(exc))}")
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
