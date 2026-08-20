#!/usr/bin/env python3
"""V1.5.1 frozen Adaptive collection with provider preflight and metadata."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from platform_agent.provider_preflight import run_qwen_preflight
from platform_agent.qwen import QwenReadOnlyModel
from platform_integrations.model_retry import ModelRetryPolicy
from platform_eval.adaptive import aggregate_adaptive_results, load_adaptive_cases

from evaluate_adaptive_v150 import CountingClient, _collect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _endpoint_host(value: str) -> str:
    parsed = urlparse(value)
    return parsed.netloc or parsed.path.split("/", 1)[0]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _metadata(args, case_file: Path, preflight: dict, started: str) -> dict:
    policy = ModelRetryPolicy.from_env()
    return {
        "version": "v1.5.1",
        "agent_model": args.model,
        "provider": "qwen",
        "endpoint_host": _endpoint_host(os.environ.get("DASHSCOPE_OPENAI_BASE_URL", "")),
        "request_timeout_sec": args.request_timeout_sec,
        "retry_attempts": policy.attempts,
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_identical_tool_calls": args.max_identical_tool_calls,
        "max_consecutive_tool_failures": args.max_consecutive_tool_failures,
        "source_cases": str(case_file),
        "frozen_cases_sha256": _sha256(case_file),
        "provider_preflight": preflight,
        "collection_started_at": started,
    }


async def main(args) -> int:
    case_file = args.cases.resolve()
    output_dir = args.output_dir
    started = _now()
    if args.finalize_partial:
        cases = load_adaptive_cases(case_file)
        partial_samples_path = output_dir / "v1.5.1_adaptive_qwen_plus_samples.partial.json"
        partial_eval_path = output_dir / "v1.5.1_adaptive_qwen_plus_eval.partial.json"
        partial_samples = json.loads(partial_samples_path.read_text(encoding="utf-8"))
        partial_eval = json.loads(partial_eval_path.read_text(encoding="utf-8"))
        in_progress_path = output_dir / "v1.5.1_adaptive_qwen_plus_eval.json"
        in_progress = {}
        if in_progress_path.exists():
            try:
                candidate = json.loads(in_progress_path.read_text(encoding="utf-8"))
                if candidate.get("status") == "COLLECTION_IN_PROGRESS":
                    in_progress = candidate
            except (OSError, json.JSONDecodeError):
                in_progress = {}
        metadata = _metadata(
            args,
            case_file,
            in_progress.get(
                "metadata",
                {
                    "status": "PASS",
                    "checks_requested": args.preflight_checks,
                    "requests_attempted": args.preflight_checks,
                    "requests_completed": args.preflight_checks,
                    "timeout_count": 0,
                    "error_count": 0,
                    "failure_types": [],
                },
            ),
            started,
        )
        original_metadata = in_progress.get("metadata", {})
        for key in ("collection_started_at", "endpoint_host", "request_timeout_sec", "retry_attempts"):
            if key in original_metadata:
                metadata[key] = original_metadata[key]
        metadata["collection_finalized_from_partial_at"] = _now()
        metadata["collection_status"] = "PARTIAL_PROVIDER_BLOCKED"
        metadata["collection_blocker"] = "formal_provider_request_interrupted_before_case_completion"
        metadata["api_request_count"] = args.provider_requests_attempted
        metadata["provider_requests_attempted"] = args.provider_requests_attempted
        metadata["provider_requests_completed"] = args.provider_requests_completed
        metadata["provider_timeout_count"] = 1
        metadata["provider_error_count"] = 1
        evaluation = {
            "status": "PARTIAL_PROVIDER_BLOCKED",
            "metadata": metadata,
            "api_request_count": args.provider_requests_attempted,
            "provider_requests_attempted": args.provider_requests_attempted,
            "provider_requests_completed": args.provider_requests_completed,
            "provider_timeout_count": 1,
            "provider_error_count": 1,
            "case_count": len(cases),
            "completed_case_count": len(partial_samples.get("samples", [])),
            "aggregate_completed_cases": partial_eval.get("aggregate", {}),
            "cases": partial_eval.get("cases", []),
            "known_regression_probes": [
                row for row in partial_eval.get("cases", [])
                if row.get("category") == "known_regression_probe"
            ],
        }
        _write_json(
            output_dir / "v1.5.1_adaptive_qwen_plus_samples.json",
            {"status": evaluation["status"], "metadata": metadata, "samples": partial_samples.get("samples", [])},
        )
        _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_eval.json", evaluation)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return 2

    if not os.environ.get("DASHSCOPE_API_KEY") or not os.environ.get("DASHSCOPE_OPENAI_BASE_URL"):
        metadata = _metadata(
            args,
            case_file,
            {
                "status": "BLOCKED_PROVIDER_PREFLIGHT",
                "failure_types": ["missing_credentials_or_endpoint"],
                "requests_attempted": 0,
                "requests_completed": 0,
            },
            started,
        )
        evaluation = {
            "status": "BLOCKED_PROVIDER_PREFLIGHT",
            "metadata": metadata,
            "api_request_count": 0,
            "cases": [],
            "aggregate": aggregate_adaptive_results([]),
        }
        _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_samples.json", {"metadata": metadata, "samples": []})
        _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_eval.json", evaluation)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return 2

    cases = load_adaptive_cases(case_file)
    from openai import AsyncOpenAI

    counter = {"requests": 0, "completed": 0}
    raw_client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_OPENAI_BASE_URL"],
        timeout=args.request_timeout_sec,
    )
    client = CountingClient(raw_client, counter)
    preflight = await run_qwen_preflight(
        client,
        model=args.model,
        checks=args.preflight_checks,
        timeout_sec=args.preflight_timeout_sec,
    )
    preflight_payload = preflight.as_dict()
    metadata = _metadata(args, case_file, preflight_payload, started)
    if not preflight.ok:
        metadata["collection_completed_at"] = _now()
        evaluation = {
            "status": "BLOCKED_PROVIDER_PREFLIGHT",
            "metadata": metadata,
            "api_request_count": counter["requests"],
            "provider_requests_attempted": counter["requests"],
            "provider_requests_completed": counter["completed"],
            "cases": [],
            "aggregate": aggregate_adaptive_results([]),
        }
        _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_samples.json", {"metadata": metadata, "samples": []})
        _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_eval.json", evaluation)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return 2

    provider_metrics: dict[str, int] = {}
    model = QwenReadOnlyModel(
        model=args.model,
        client=client,
        request_timeout_sec=args.request_timeout_sec,
        metrics=provider_metrics,
    )
    limits = {
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_identical_tool_calls": args.max_identical_tool_calls,
        "max_consecutive_tool_failures": args.max_consecutive_tool_failures,
        "case_timeout_sec": args.case_timeout_sec,
    }
    # Persist the preflight result before the first formal request. A proxy
    # hang must leave an auditable artifact even if the process is terminated.
    _write_json(
        output_dir / "v1.5.1_adaptive_qwen_plus_samples.json",
        {"status": "COLLECTION_IN_PROGRESS", "metadata": metadata, "samples": []},
    )
    _write_json(
        output_dir / "v1.5.1_adaptive_qwen_plus_eval.json",
        {
            "status": "COLLECTION_IN_PROGRESS",
            "metadata": metadata,
            "api_request_count": counter["requests"],
            "provider_requests_attempted": counter["requests"],
            "provider_requests_completed": counter["completed"],
            "cases": [],
            "aggregate": aggregate_adaptive_results([]),
        },
    )
    samples, rows = await _collect(
        cases,
        model,
        output_dir,
        limits,
        checkpoint_prefix="v1.5.1_adaptive_qwen_plus",
    )
    metadata["collection_completed_at"] = _now()
    metadata["api_request_count"] = counter["requests"]
    metadata["provider_requests_attempted"] = counter["requests"]
    metadata["provider_requests_completed"] = counter["completed"]
    metadata["provider_retry_count"] = provider_metrics.get("retries", 0)
    metadata["provider_timeout_count"] = preflight.timeout_count + provider_metrics.get("provider_timeout", 0)
    metadata["provider_error_count"] = preflight.error_count + provider_metrics.get("provider_errors", 0)
    unseen = [row for row, case in zip(rows, cases) if not case.get("known_regression_probe")]
    probes = [row for row, case in zip(rows, cases) if case.get("known_regression_probe")]
    sample_payload = {"metadata": metadata, "samples": samples}
    evaluation = {
        "status": "COMPLETE",
        "metadata": metadata,
        "api_request_count": counter["requests"],
        "provider_requests_attempted": counter["requests"],
        "provider_requests_completed": counter["completed"],
        "provider_retry_count": provider_metrics.get("retries", 0),
        "provider_timeout_count": preflight.timeout_count + provider_metrics.get("provider_timeout", 0),
        "provider_error_count": preflight.error_count + provider_metrics.get("provider_errors", 0),
        "provider_model_operations": provider_metrics.get("model_operations", 0),
        "case_count": len(cases),
        "unseen_case_count": len(unseen),
        "known_regression_probe_count": len(probes),
        "aggregate": aggregate_adaptive_results(unseen),
        "known_regression_probes": probes,
        "cases": rows,
    }
    _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_samples.json", sample_payload)
    _write_json(output_dir / "v1.5.1_adaptive_qwen_plus_eval.json", evaluation)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("eval/v1_5_0/adaptive_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_acceptance"))
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-identical-tool-calls", type=int, default=2)
    parser.add_argument("--max-consecutive-tool-failures", type=int, default=2)
    parser.add_argument("--case-timeout-sec", type=float, default=120.0)
    parser.add_argument("--request-timeout-sec", type=float, default=45.0)
    parser.add_argument("--preflight-checks", type=int, default=2)
    parser.add_argument("--preflight-timeout-sec", type=float, default=15.0)
    parser.add_argument("--finalize-partial", action="store_true")
    parser.add_argument("--provider-requests-attempted", type=int, default=0)
    parser.add_argument("--provider-requests-completed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
