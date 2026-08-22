"""Raw trajectory collection for the final benchmark.

The collector is deliberately separate from scoring.  An adapter records
observed facts from a model/system run; deterministic evaluators decide
whether those facts resolved the scenario.  The default adapters use the
evaluation-only scripted runners and never call a provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


COLLECTOR_VERSION = "a-plus-final-collector-v2"
DERIVED_FIELDS = frozenset(
    {
        "resolved",
        "resolved_first_attempt",
        "functional_valid",
        "unsafe_auto",
        "false_success",
        "unnecessary_tool_calls",
        "unexpected_tool_calls",
        "required_tool_recall",
        "goal_state_macro_f1",
        "safety_pass",
        "goal_ok",
    }
)


class CaseRunner(Protocol):
    def __call__(self, scenario: Any, repetition: int, model: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CollectorConfig:
    model: str
    system: str
    repetitions: int = 3


class QuotaBlockedError(RuntimeError):
    """Free-tier stop signal; the collector must not retry or continue."""

    def __init__(self, model: str, provider: str = "Alibaba Bailian", status_code: int = 403, error_code: str = "AllocationQuota.FreeTierOnly") -> None:
        self.model = model
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"{error_code} ({status_code}) for {model}")


class Adapter:
    system: str

    def __init__(self, runner: CaseRunner | None = None):
        if runner is None:
            from .formal_runners import formal_runner_for
            runner = formal_runner_for(self.system)
        self._runner = runner

    def collect_case(self, scenario: Any, repetition: int, model: str) -> Mapping[str, Any]:
        return self._runner(scenario, repetition, model)


class NaiveToolAdapter(Adapter):
    system = "naive_tool"


class HitlOnlyAdapter(Adapter):
    system = "hitl_only"


class FullAgentAdapter(Adapter):
    system = "full"


def adapter_for(system: str, runner: CaseRunner | None = None) -> Adapter:
    adapters = {"naive_tool": NaiveToolAdapter, "hitl_only": HitlOnlyAdapter, "full": FullAgentAdapter}
    try:
        return adapters[system](runner)
    except KeyError as exc:
        raise ValueError(f"Unknown collector system: {system}") from exc


def _raw_record(*, case_id: str, repetition: int, system: str, model: str, facts: Mapping[str, Any], status: str = "OK", error: str | None = None) -> dict[str, Any]:
    forbidden = DERIVED_FIELDS & set(facts)
    if forbidden:
        raise ValueError(f"Collector received evaluator-derived fields: {sorted(forbidden)}")
    record = dict(facts)
    record.update({"case_id": case_id, "repetition": repetition, "system": system, "model": model, "status": status})
    if error:
        record["error"] = error
    return record


def collect_trajectories(cases: list[Any], config: CollectorConfig, adapter: Adapter) -> list[dict[str, Any]]:
    if adapter.system != config.system:
        raise ValueError(f"Adapter/system mismatch: {adapter.system} != {config.system}")
    if config.repetitions < 1:
        raise ValueError("repetitions must be positive")
    records: list[dict[str, Any]] = []
    for repetition in range(1, config.repetitions + 1):
        for scenario in cases:
            try:
                facts = adapter.collect_case(scenario, repetition, config.model)
                if not isinstance(facts, Mapping):
                    raise TypeError("adapter must return a mapping of raw facts")
                records.append(_raw_record(case_id=scenario.id, repetition=repetition, system=config.system, model=config.model, facts=facts))
            except Exception as exc:
                records.append(
                    _raw_record(
                        case_id=scenario.id,
                        repetition=repetition,
                        system=config.system,
                        model=config.model,
                        facts={},
                        status="ERROR",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return records


def collect_trajectories_with_status(cases: list[Any], config: CollectorConfig, adapter: Adapter) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect one run, stopping immediately on ``FreeTierOnly``.

    Ordinary attempt failures are retained as ``ERROR`` records.  A quota
    block is terminal for this model/run, so no later case is invoked.
    """
    if adapter.system != config.system:
        raise ValueError(f"Adapter/system mismatch: {adapter.system} != {config.system}")
    if config.repetitions < 1:
        raise ValueError("repetitions must be positive")
    records: list[dict[str, Any]] = []
    total = len(cases) * config.repetitions
    for repetition in range(1, config.repetitions + 1):
        for scenario in cases:
            try:
                facts = adapter.collect_case(scenario, repetition, config.model)
                if not isinstance(facts, Mapping):
                    raise TypeError("adapter must return a mapping of raw facts")
                records.append(_raw_record(case_id=scenario.id, repetition=repetition, system=config.system, model=config.model, facts=facts))
            except QuotaBlockedError as exc:
                records.append(_raw_record(case_id=scenario.id, repetition=repetition, system=config.system, model=config.model, facts={}, status="BLOCKED", error=str(exc)))
                completed = len(records)
                return records, {
                    "status": "INCOMPLETE_QUOTA_BLOCKED",
                    "completed_attempts": completed,
                    "remaining_attempts": max(0, total - completed),
                    "quota_blocked": True,
                    "model": exc.model,
                    "provider": exc.provider,
                    "http_status": exc.status_code,
                    "error_code": exc.error_code,
                }
            except Exception as exc:
                records.append(_raw_record(case_id=scenario.id, repetition=repetition, system=config.system, model=config.model, facts={}, status="ERROR", error=f"{type(exc).__name__}: {exc}"))
    return records, {"status": "COMPLETE", "completed_attempts": len(records), "remaining_attempts": 0, "quota_blocked": False, "model": config.model}


def run_fake_benchmark(cases: list[Any], *, repetitions: int = 3, model: str = "scripted-model") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute all systems against deterministic runners; never calls a model provider."""
    from .fixture_registry import validate_fixture_names
    validate_fixture_names(cases)
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for system in ("full", "hitl_only", "naive_tool"):
        config = CollectorConfig(model=model, system=system, repetitions=repetitions)
        records, summary = collect_trajectories_with_status(cases, config, adapter_for(system))
        if summary["status"] == "COMPLETE":
            validate_raw_coverage(records, cases, system=system, repetitions=repetitions)
        all_records.extend(records)
        summaries[system] = summary
    return all_records, {"systems": summaries, "expected_attempts": len(cases) * repetitions * 3, "external_model_calls": 0}


def write_raw_trajectories(records: list[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in records) + "\n", encoding="utf-8")


def prepare_run_directory(root: str | Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single safe directory name")
    destination = Path(root) / run_id
    if destination.exists():
        raise FileExistsError(f"Run directory already exists and is immutable: {destination}")
    destination.mkdir(parents=True)
    return destination


def validate_raw_coverage(records: list[Mapping[str, Any]], cases: list[Any], *, system: str, repetitions: int) -> None:
    expected = {(case.id, repetition, system) for case in cases for repetition in range(1, repetitions + 1)}
    actual = {(str(row.get("case_id")), int(row.get("repetition", 0)), str(row.get("system"))) for row in records}
    if len(actual) != len(records):
        raise ValueError("duplicate raw trajectory key: case_id + repetition + system")
    if actual != expected:
        raise ValueError(f"raw trajectory coverage mismatch: missing={sorted(expected - actual)[:5]} unexpected={sorted(actual - expected)[:5]}")
