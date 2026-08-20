from __future__ import annotations

import asyncio
import time

import pytest

from platform_integrations.model_retry import ModelRequestError, ModelRetryPolicy, retry_async


def test_model_request_timeout_is_configurable_and_provider_neutral(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_REQUEST_TIMEOUT_SEC", "0.25")
    assert ModelRetryPolicy.from_env().request_timeout_sec == 0.25


@pytest.mark.asyncio
async def test_timeout_is_retryable_and_success_after_one_timeout():
    calls = 0
    metrics: dict[str, int] = {}

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.03)
        return "ok"

    result = await retry_async(
        operation,
        operation_name="qwen:synthetic-timeout",
        policy=ModelRetryPolicy(attempts=2, base_sec=0, max_sec=0, jitter_sec=0, request_timeout_sec=0.01),
        attempt_metrics=metrics,
    )
    assert result == "ok"
    assert calls == 2
    assert metrics["provider_timeout"] == 1
    assert metrics["retries"] == 1


@pytest.mark.asyncio
async def test_timeout_stops_at_bounded_attempts_without_sensitive_text():
    async def operation():
        await asyncio.sleep(0.03)
        raise RuntimeError("secret prompt and API key should not escape")

    started = time.perf_counter()
    with pytest.raises(ModelRequestError) as caught:
        await retry_async(
            operation,
            operation_name="provider:synthetic",
            policy=ModelRetryPolicy(attempts=2, base_sec=0, max_sec=0, jitter_sec=0, request_timeout_sec=0.01),
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 0.15
    assert caught.value.failure_type == "provider_timeout"
    assert "secret prompt" not in str(caught.value)
    assert "API key" not in str(caught.value)


@pytest.mark.asyncio
async def test_non_timeout_exception_remains_non_retryable():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise ValueError("invalid structured response")

    with pytest.raises(ModelRequestError) as caught:
        await retry_async(
            operation,
            operation_name="provider:invalid-json",
            policy=ModelRetryPolicy(attempts=3, base_sec=0, max_sec=0, jitter_sec=0, request_timeout_sec=0.1),
        )
    assert calls == 1
    assert caught.value.failure_type == "unknown_provider_error"
