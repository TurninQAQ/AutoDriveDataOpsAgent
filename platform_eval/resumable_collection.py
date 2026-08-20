"""Case-level resumable collection primitives for deterministic evaluations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from platform_mcp.server import WRITE_TOOL_NAMES

from .adaptive import aggregate_adaptive_results


EXPECTED_V1_5_0_CASES_SHA256 = "dbd338133139da7785722b0efa1a5718461e62c4df6f888bb133c0ea78199e42"
COLLECTION_STATUSES = frozenset(
    {"PENDING", "RUNNING", "COMPLETE", "PROVIDER_ERROR", "AGENT_ERROR", "INVALID_RESULT", "INTERRUPTED"}
)


class FrozenCaseHashMismatch(ValueError):
    """The requested case file is not the frozen baseline."""


class CollectionCompatibilityError(ValueError):
    """A resume manifest is incompatible with the requested run."""


class CollectionSafetyError(RuntimeError):
    """A case attempted an unsafe executed mutation."""


class CollectionAttemptError(RuntimeError):
    """A complete case attempt failed before producing a valid trajectory."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        provider_failure: bool,
        operation_type: str,
        step_index: int | None = None,
        details: Mapping[str, Any] | None = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.provider_failure = provider_failure
        self.operation_type = operation_type
        self.step_index = step_index
        self.details = dict(details or {})
        self.fatal = fatal


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def build_manifest(
    *,
    version: str,
    model: str,
    provider: str,
    endpoint_host: str,
    frozen_case_file: str,
    frozen_case_sha256: str,
    production_base_commit: str,
    request_timeout_sec: float,
    formal_eval_retry_attempts: int,
    max_steps: int,
    max_tool_calls: int,
    max_identical_tool_calls: int,
    max_consecutive_tool_failures: int,
    max_case_attempts: int,
    case_timeout_sec: float,
    tool_catalog_sha256: str,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    now = utc_now()
    return {
        "version": version,
        "agent_model": model,
        "provider": provider,
        "endpoint_host": endpoint_host,
        "frozen_case_file": frozen_case_file,
        "frozen_case_sha256": frozen_case_sha256,
        "production_base_commit": production_base_commit,
        "request_timeout_sec": request_timeout_sec,
        "formal_eval_retry_attempts": formal_eval_retry_attempts,
        "max_steps": max_steps,
        "max_tool_calls": max_tool_calls,
        "max_identical_tool_calls": max_identical_tool_calls,
        "max_consecutive_tool_failures": max_consecutive_tool_failures,
        "max_case_attempts": max_case_attempts,
        "case_timeout_sec": case_timeout_sec,
        "tool_catalog_sha256": tool_catalog_sha256,
        "created_at": now,
        "updated_at": now,
        "status": "PENDING",
        "collection_invocations": 1,
        "resume_count": 0,
        "provider_preflight": None,
        "provider_stats": {
            "requests_attempted": 0,
            "requests_completed": 0,
            "timeouts": 0,
            "retries": 0,
            "errors": 0,
        },
        "provider_stats_scope": "formal_collection",
        "cases": {
            str(case["id"]): {
                "status": "PENDING",
                "attempts": 0,
                "completed_attempts": 0,
                "provider_failures": 0,
                "agent_failures": 0,
                "first_attempt_complete": False,
                "last_failure_type": None,
                "artifact_path": None,
                "failed_attempt_paths": [],
            }
            for case in cases
        },
    }


def manifest_compatibility_keys() -> tuple[str, ...]:
    return (
        "version",
        "agent_model",
        "provider",
        "endpoint_host",
        "frozen_case_sha256",
        "production_base_commit",
        "request_timeout_sec",
        "formal_eval_retry_attempts",
        "max_steps",
        "max_tool_calls",
        "max_identical_tool_calls",
        "max_consecutive_tool_failures",
        "max_case_attempts",
        "case_timeout_sec",
        "tool_catalog_sha256",
    )


def validate_manifest_compatibility(
    manifest: Mapping[str, Any], expected: Mapping[str, Any], *, case_ids: Sequence[str]
) -> None:
    for key in manifest_compatibility_keys():
        if manifest.get(key) != expected.get(key):
            raise CollectionCompatibilityError(
                f"resume metadata mismatch for {key}: "
                f"existing={manifest.get(key)!r} requested={expected.get(key)!r}"
            )
    entries = manifest.get("cases")
    if not isinstance(entries, Mapping) or set(entries) != set(case_ids):
        raise CollectionCompatibilityError("resume case IDs do not match the frozen case file")
    for case_id, entry in entries.items():
        if not isinstance(entry, Mapping) or entry.get("status") not in COLLECTION_STATUSES:
            raise CollectionCompatibilityError(f"invalid case manifest status for {case_id}")


def validate_frozen_case_file(path: str | Path, *, expected_sha256: str = EXPECTED_V1_5_0_CASES_SHA256) -> str:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise FrozenCaseHashMismatch(
            f"frozen case hash mismatch: expected={expected_sha256} actual={actual}"
        )
    return actual


def _merge_provider_stats(manifest: dict[str, Any], stats: Mapping[str, Any] | None) -> None:
    if not stats:
        return
    current = manifest.setdefault("provider_stats", {})
    for key, value in stats.items():
        if isinstance(value, int | float):
            current[key] = value


def _failure_payload(case_id: str, attempt: int, error: CollectionAttemptError) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_attempt": attempt,
        "operation_type": error.operation_type,
        "step_index": error.step_index,
        "failure_type": error.failure_type,
        "provider_failure": error.provider_failure,
        "error": str(error)[:1000],
        "details": error.details,
        "timestamp": utc_now(),
    }


def _augment_trajectory_metrics(
    rows: list[dict[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    max_identical = int(manifest.get("max_identical_tool_calls", 2))
    for row, sample in zip(rows, samples):
        row["final_intent_accuracy"] = float(bool(row.get("intent_ok")))
        row["budget_violation"] = float(not bool(row.get("loop_termination_ok")))
        signatures: dict[str, int] = {}
        for item in sample.get("tool_calls") or []:
            if not isinstance(item, Mapping):
                continue
            signature = json.dumps(
                {"tool": item.get("tool") or item.get("name"), "arguments": item.get("arguments") or {}},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            signatures[signature] = signatures.get(signature, 0) + 1
        row["duplicate_tool_violation"] = float(any(value > max_identical for value in signatures.values()))


def _augment_aggregate(aggregate: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        aggregate.update(
            {
                "final_intent_accuracy": 0.0,
                "hybrid_completion_rate": 0.0,
                "budget_violation_rate": 0.0,
                "duplicate_tool_violation_rate": 0.0,
            }
        )
        return aggregate
    aggregate["final_intent_accuracy"] = sum(float(row.get("final_intent_accuracy", row.get("intent_ok", False))) for row in rows) / count
    aggregate["budget_violation_rate"] = sum(float(row.get("budget_violation", 0.0)) for row in rows) / count
    aggregate["duplicate_tool_violation_rate"] = sum(float(row.get("duplicate_tool_violation", 0.0)) for row in rows) / count
    hybrid = [row for row in rows if row.get("category") == "hybrid"]
    aggregate["hybrid_completion_rate"] = (
        sum(float(bool(row.get("scenario_complete"))) for row in hybrid) / len(hybrid) if hybrid else 0.0
    )
    return aggregate


async def run_resumable_cases(
    *,
    cases: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    manifest_path: str | Path,
    artifact_root: str | Path,
    execute_case: Callable[[Mapping[str, Any], int], Awaitable[Mapping[str, Any]]],
    max_case_attempts: int,
    force_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Execute incomplete cases independently and persist each transition."""

    if max_case_attempts < 1:
        raise ValueError("max_case_attempts must be positive")
    force_case_ids = force_case_ids or set()
    root = Path(artifact_root)
    cases_dir = root / "cases"
    failures_dir = root / "failed_attempts"
    entries = manifest["cases"]
    case_by_id = {str(case["id"]): case for case in cases}
    current_case_id: str | None = None

    def persist() -> None:
        manifest["updated_at"] = utc_now()
        statuses = [str(item.get("status")) for item in entries.values()]
        if manifest.get("status") != "ABORTED_SAFETY":
            manifest["status"] = "COMPLETE" if statuses and all(item == "COMPLETE" for item in statuses) else "PARTIAL"
        atomic_write_json(manifest_path, manifest)

    try:
        for case_id, case in case_by_id.items():
            entry = entries[case_id]
            if entry.get("status") == "COMPLETE" and case_id not in force_case_ids:
                continue
            if case_id in force_case_ids:
                entry["status"] = "PENDING"
            current_case_id = case_id
            completed = False
            for _ in range(max_case_attempts):
                attempt = int(entry.get("attempts", 0)) + 1
                entry["attempts"] = attempt
                entry["status"] = "RUNNING"
                persist()
                try:
                    result = dict(await execute_case(case, attempt))
                    if result.get("unsafe_write_executed"):
                        raise CollectionAttemptError(
                            "unsafe write execution detected; collection aborted",
                            failure_type="unsafe_write_execution",
                            provider_failure=False,
                            operation_type=str(result.get("operation_type") or "unknown"),
                            step_index=result.get("step_index"),
                            details={"safety_event": result.get("safety_event", "unsafe mutation")},
                            fatal=True,
                        )
                    if not result.get("sample") or not result.get("evaluation"):
                        raise CollectionAttemptError(
                            "case executor returned an incomplete result",
                            failure_type="invalid_result",
                            provider_failure=False,
                            operation_type="collection",
                        )
                    case_artifact = {
                        "status": "COMPLETE",
                        "case_id": case_id,
                        "case_attempt": attempt,
                        "sample": result["sample"],
                        "evaluation": result["evaluation"],
                        "completed_at": utc_now(),
                    }
                    case_path = cases_dir / f"{case_id}.json"
                    atomic_write_json(case_path, case_artifact)
                    entry["status"] = "COMPLETE"
                    entry["completed_attempts"] = int(entry.get("completed_attempts", 0)) + 1
                    entry["first_attempt_complete"] = bool(attempt == 1)
                    entry["artifact_path"] = str(case_path.relative_to(root))
                    entry["last_failure_type"] = None
                    _merge_provider_stats(manifest, result.get("provider_stats"))
                    persist()
                    completed = True
                    break
                except CollectionAttemptError as error:
                    _merge_provider_stats(manifest, error.details.get("provider_stats"))
                    failure_path = failures_dir / f"{case_id}.attempt_{attempt}.json"
                    atomic_write_json(failure_path, _failure_payload(case_id, attempt, error))
                    entry["failed_attempt_paths"].append(str(failure_path.relative_to(root)))
                    entry["last_failure_type"] = error.failure_type
                    if error.provider_failure:
                        entry["provider_failures"] = int(entry.get("provider_failures", 0)) + 1
                        entry["status"] = "PROVIDER_ERROR"
                    else:
                        entry["agent_failures"] = int(entry.get("agent_failures", 0)) + 1
                        entry["status"] = "INVALID_RESULT" if error.failure_type == "invalid_result" else "AGENT_ERROR"
                    persist()
                    if error.fatal:
                        manifest["status"] = "ABORTED_SAFETY"
                        persist()
                        raise CollectionSafetyError(str(error)) from error
                except (asyncio.CancelledError, KeyboardInterrupt) as error:
                    interrupted = CollectionAttemptError(
                        "collection interrupted before case completion",
                        failure_type="interrupted",
                        provider_failure=False,
                        operation_type="unknown",
                    )
                    failure_path = failures_dir / f"{case_id}.attempt_{attempt}.json"
                    atomic_write_json(failure_path, _failure_payload(case_id, attempt, interrupted))
                    entry["failed_attempt_paths"].append(str(failure_path.relative_to(root)))
                    entry["status"] = "INTERRUPTED"
                    entry["last_failure_type"] = "interrupted"
                    persist()
                    raise error
            if not completed:
                persist()
            current_case_id = None
    except (asyncio.CancelledError, KeyboardInterrupt):
        if current_case_id and manifest["cases"][current_case_id].get("status") == "RUNNING":
            manifest["cases"][current_case_id]["status"] = "INTERRUPTED"
            persist()
        raise
    return manifest


def summarize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = list((manifest.get("cases") or {}).values())
    statuses = {status: sum(1 for item in entries if item.get("status") == status) for status in COLLECTION_STATUSES}
    return {
        "status": manifest.get("status"),
        "total_cases": len(entries),
        "statuses": {key: value for key, value in statuses.items() if value},
        "total_case_attempts": sum(int(item.get("attempts", 0)) for item in entries),
        "resume_count": manifest.get("resume_count", 0),
    }


def finalize_collection(
    *,
    manifest: Mapping[str, Any],
    artifact_root: str | Path,
    output_dir: str | Path,
    allow_partial: bool,
) -> dict[str, Any]:
    """Build final samples/eval artifacts only from atomic COMPLETE case files."""

    entries = manifest.get("cases") or {}
    missing = [case_id for case_id, item in entries.items() if item.get("status") != "COMPLETE"]
    if missing and not allow_partial:
        raise CollectionCompatibilityError(f"cannot finalize incomplete collection; missing cases: {missing}")
    root = Path(artifact_root)
    samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for case_id, entry in entries.items():
        if entry.get("status") != "COMPLETE":
            continue
        path = root / str(entry["artifact_path"])
        artifact = read_json(path)
        samples.append(artifact["sample"])
        row = dict(artifact["evaluation"])
        row.setdefault("actual_intent", artifact["sample"].get("final_intent", ""))
        row.setdefault("initial_intent", artifact["sample"].get("initial_intent", ""))
        tool_values = artifact["sample"].get("tool_calls") or []
        actual_tools = [
            str(item.get("tool") or item.get("name") or "")
            for item in tool_values
            if isinstance(item, Mapping)
        ]
        # Historical V1.5 evaluator rows call every contract-forbidden tool a
        # "forbidden_write_execution". Preserve that route metric separately,
        # but make the V1.5.2 safety metric mean actual executed WRITE tools.
        row["forbidden_tool_attempt"] = bool(row.get("forbidden_write_execution"))
        row["forbidden_write_execution"] = any(tool in WRITE_TOOL_NAMES for tool in actual_tools)
        row["forbidden_write_tools_executed"] = [tool for tool in actual_tools if tool in WRITE_TOOL_NAMES]
        rows.append(row)
    _augment_trajectory_metrics(rows, samples, manifest)
    complete = not missing
    unseen = [row for row, sample in zip(rows, samples) if not sample.get("known_regression_probe")]
    probes = [row for row, sample in zip(rows, samples) if sample.get("known_regression_probe")]
    status = "FULL_BASELINE" if complete else "PARTIAL_BASELINE"
    metadata = dict(manifest)
    metadata["finalized_at"] = utc_now()
    metadata["full_baseline"] = complete
    metadata["completed_case_count"] = len(samples)
    metadata["missing_cases"] = missing
    evaluation = {
        "status": status,
        "FULL_BASELINE": complete,
        "metadata": metadata,
        "case_count": len(entries),
        "completed_case_count": len(samples),
        "missing_cases": missing,
        "provider_failed_cases": [case_id for case_id, item in entries.items() if item.get("status") == "PROVIDER_ERROR"],
        "agent_failed_cases": [case_id for case_id, item in entries.items() if item.get("status") in {"AGENT_ERROR", "INVALID_RESULT"}],
        "aggregate": _augment_aggregate(aggregate_adaptive_results(unseen), unseen) if complete else None,
        "aggregate_completed_cases": _augment_aggregate(aggregate_adaptive_results(unseen), unseen) if not complete else None,
        "known_regression_probes": probes,
        "cases": rows,
        "completed_cases_only": not complete,
    }
    total_attempts = len(entries)
    completed_entries = [item for item in entries.values() if item.get("status") == "COMPLETE"]
    provider_stats = dict(manifest.get("provider_stats") or {})
    provider_stats_scope = manifest.get("provider_stats_scope") or "all_requests_including_preflight"
    metadata["provider_stats"] = provider_stats
    metadata["provider_stats_scope"] = provider_stats_scope
    attempted_requests = int(provider_stats.get("requests_attempted", 0))
    completed_requests = int(provider_stats.get("requests_completed", 0))
    evaluation["provider_availability"] = {
        "operation_success_rate": completed_requests / attempted_requests if attempted_requests else 0.0,
        "operation_timeout_rate": int(provider_stats.get("timeouts", 0)) / attempted_requests if attempted_requests else 0.0,
        "retry_count": int(provider_stats.get("retries", 0)),
        "error_count": int(provider_stats.get("errors", 0)),
        "case_first_attempt_completion_rate": sum(bool(item.get("first_attempt_complete")) for item in completed_entries) / total_attempts if total_attempts else 0.0,
        "case_completion_after_retry_rate": sum(int(item.get("attempts", 0)) > 1 for item in completed_entries) / len(completed_entries) if completed_entries else 0.0,
        "cases_requiring_retry": [case_id for case_id, item in entries.items() if int(item.get("attempts", 0)) > 1],
        "total_model_operations": attempted_requests,
        "total_completed_operations": completed_requests,
    }
    output = Path(output_dir)
    atomic_write_json(output / "final_samples.json", {"metadata": metadata, "samples": samples})
    atomic_write_json(output / "final_eval.json", evaluation)
    return evaluation


__all__ = [
    "COLLECTION_STATUSES",
    "EXPECTED_V1_5_0_CASES_SHA256",
    "CollectionAttemptError",
    "CollectionCompatibilityError",
    "CollectionSafetyError",
    "FrozenCaseHashMismatch",
    "atomic_write_json",
    "build_manifest",
    "file_sha256",
    "finalize_collection",
    "read_json",
    "run_resumable_cases",
    "summarize_manifest",
    "utc_now",
    "validate_frozen_case_file",
    "validate_manifest_compatibility",
]
