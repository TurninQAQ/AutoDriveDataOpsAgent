from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from platform_agent.provider_preflight import run_qwen_preflight
from platform_integrations.model_retry import ModelRequestError
from eval.final.collector import CollectorConfig, QuotaBlockedError, adapter_for, collect_trajectories_with_status
from eval.final.metrics import compute_headline_metrics
from eval.final.schema import load_scenarios
from eval.final.telemetry import InstrumentedModelClient, ModelTelemetry
from platform_agent.models import AgentPlan
from platform_agent.qwen import QwenReadOnlyModel


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "eval" / "final"


def _response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(outcomes):
    completions = _Completions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class _SdkQuotaError(Exception):
    status_code = 403
    code = "AllocationQuota.FreeTierOnly"

    def __str__(self):
        return "Forbidden"


class _Generic403(Exception):
    status_code = 403

    def __str__(self):
        return "Forbidden"


class _Transient503(Exception):
    status_code = 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("HTTP 403 AllocationQuota.FreeTierOnly"),
        _SdkQuotaError("quota"),
        ModelRequestError("qwen:chat", 1, 403, False, provider_error_code="AllocationQuota.FreeTierOnly"),
    ],
)
async def test_free_tier_forms_stop_on_first_preflight_request(error):
    fake, completions = _client([error, _response()])
    result = await run_qwen_preflight(fake, checks=2, timeout_sec=0.1)
    assert result.requests_attempted == 1
    assert result.quota_blocked is True
    assert result.terminal is True
    assert completions.calls == 1
    assert result.failure_types == ["provider_quota_error"]


@pytest.mark.asyncio
async def test_generic_403_stops_free_tier_preflight_without_quota_claim():
    fake, completions = _client([_Generic403("forbidden"), _response()])
    result = await run_qwen_preflight(fake, checks=2, timeout_sec=0.1)
    assert result.requests_attempted == 1
    assert result.quota_blocked is False
    assert result.terminal is True
    assert result.failure_types == ["provider_auth_error"]
    assert completions.calls == 1


@pytest.mark.asyncio
async def test_successful_multi_check_preflight_is_unchanged():
    fake, completions = _client([_response(), _response()])
    result = await run_qwen_preflight(fake, checks=2, timeout_sec=0.1)
    assert result.ok is True
    assert result.requests_attempted == 2
    assert result.requests_completed == 2
    assert result.terminal is False
    assert completions.calls == 2


@pytest.mark.asyncio
async def test_nonterminal_5xx_keeps_existing_multi_check_protocol():
    error = _Transient503("temporarily unavailable")
    fake, completions = _client([error, _response()])
    result = await run_qwen_preflight(fake, checks=2, timeout_sec=0.1)
    assert result.requests_attempted == 2
    assert result.requests_completed == 1
    assert result.terminal is False
    assert completions.calls == 2


class _MetricModel:
    def __init__(self, with_usage: bool = True):
        self.metrics = {}
        self.with_usage = with_usage

    async def plan(self, *_args, **_kwargs):
        self._record_call()
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120) if self.with_usage else None)

    async def synthesize(self, *_args, **_kwargs):
        self._record_call()
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60) if self.with_usage else None)

    def _record_call(self):
        self.metrics["attempts"] = self.metrics.get("attempts", 0) + 1
        if self.with_usage:
            self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + (100 if self.metrics["attempts"] == 1 else 50)
            self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + (20 if self.metrics["attempts"] == 1 else 10)
            self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + (120 if self.metrics["attempts"] == 1 else 60)


@pytest.mark.asyncio
async def test_live_telemetry_counts_real_model_invocations_and_usage():
    base = _MetricModel()
    telemetry = ModelTelemetry()
    client = InstrumentedModelClient(base, telemetry)
    await client.plan("prompt", [], [])
    await client.synthesize("prompt", None, [], [])
    assert telemetry.as_dict()["llm_call_count"] == 2
    assert telemetry.as_dict()["input_tokens"] == 150
    assert telemetry.as_dict()["output_tokens"] == 30
    assert telemetry.as_dict()["total_tokens"] == 180
    assert telemetry.as_dict()["llm_latency_ms_total"] >= 0


@pytest.mark.asyncio
async def test_live_telemetry_keeps_missing_usage_unavailable():
    base = _MetricModel(with_usage=False)
    telemetry = ModelTelemetry()
    await InstrumentedModelClient(base, telemetry).plan("prompt", [], [])
    facts = telemetry.as_dict()
    assert facts["llm_call_count"] == 1
    assert facts["input_tokens"] is None
    assert facts["output_tokens"] is None
    assert facts["total_tokens"] is None


@pytest.mark.asyncio
async def test_qwen_adapter_exports_provider_usage_for_live_telemetry():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"intent":"platform_health","decision_summary":"ok","tool_calls":[]}'))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )

    class Completions:
        async def create(self, **_kwargs):
            return response

    model = QwenReadOnlyModel(client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    await model._structured("Return JSON", AgentPlan)
    assert model.metrics["input_tokens"] == 100
    assert model.metrics["output_tokens"] == 20
    assert model.metrics["total_tokens"] == 120


def test_collector_preserves_quota_attempt_telemetry_and_stops():
    cases = load_scenarios(FINAL / "dev.jsonl")[:3]
    calls = []

    def runner(_case, _repetition, model):
        calls.append(1)
        error = QuotaBlockedError(model)
        error.telemetry = {"llm_call_count": 1, "attempt_wall_latency_ms": 12.5, "provider_error_codes": [error.error_code]}
        raise error

    records, summary = collect_trajectories_with_status(
        cases,
        CollectorConfig(model="quota-model", system="full", repetitions=1),
        adapter_for("full", runner),
    )
    assert len(calls) == 1
    assert len(records) == 1
    assert records[0]["llm_call_count"] == 1
    assert records[0]["attempt_wall_latency_ms"] == 12.5
    assert summary["status"] == "INCOMPLETE_QUOTA_BLOCKED"


def test_live_telemetry_metrics_aggregate_latency_calls_and_tokens():
    rows = [
        {
            "resolved_first_attempt": True,
            "autonomy_applicable": False,
            "goal_eval": False,
            "attempt_wall_latency_ms": 100.0,
            "llm_call_count": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "tool_call_count": 0,
        },
        {
            "resolved_first_attempt": False,
            "autonomy_applicable": False,
            "goal_eval": False,
            "attempt_wall_latency_ms": 200.0,
            "llm_call_count": 1,
            "input_tokens": 50,
            "output_tokens": 10,
            "tool_call_count": 0,
        },
    ]
    metrics = compute_headline_metrics(rows)
    assert metrics["secondary"]["latency_ms"] == {"p50": 100.0, "p95": 200.0}
    assert metrics["secondary"]["llm_calls_mean"] == 1.5
    assert metrics["secondary"]["total_input_tokens"] == 150.0
    assert metrics["secondary"]["total_output_tokens"] == 30.0
    assert metrics["secondary"]["tokens_per_resolved"] == 180.0

    partial = compute_headline_metrics([rows[0], {**rows[1], "input_tokens": None}])
    assert partial["secondary"]["total_input_tokens"] is None
    assert partial["secondary"]["tokens_per_resolved"] is None
