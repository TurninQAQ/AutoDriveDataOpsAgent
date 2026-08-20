from __future__ import annotations

import asyncio

import pytest

from platform_integrations.gemini_retry import (
    GeminiRequestError,
    GeminiRetryPolicy,
    classify_exception as gemini_classify_exception,
    retry_sync as gemini_retry_sync,
)
from platform_integrations.model_retry import (
    ModelRequestError,
    ModelRetryPolicy,
    RETRYABLE_STATUS_CODES,
    classify_exception,
    retry_async,
    retry_sync,
)


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "transient"):
        super().__init__(message)
        self.status_code = status_code


def _clear_retry_env(monkeypatch):
    for name in (
        "PLATFORM_MODEL_RETRY_ATTEMPTS",
        "PLATFORM_MODEL_RETRY_BASE_SEC",
        "PLATFORM_MODEL_RETRY_MAX_SEC",
        "PLATFORM_MODEL_RETRY_JITTER_SEC",
        "PLATFORM_GEMINI_RETRY_ATTEMPTS",
        "PLATFORM_GEMINI_RETRY_BASE_SEC",
        "PLATFORM_GEMINI_RETRY_MAX_SEC",
        "PLATFORM_GEMINI_RETRY_JITTER_SEC",
    ):
        monkeypatch.delenv(name, raising=False)


def test_model_retry_policy_defaults(monkeypatch):
    _clear_retry_env(monkeypatch)
    assert ModelRetryPolicy.from_env() == ModelRetryPolicy(5, 1.0, 20.0, 0.25)


def test_model_retry_env_has_priority_over_legacy(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_BASE_SEC", "3")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_MAX_SEC", "4")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_JITTER_SEC", "5")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_ATTEMPTS", "9")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_BASE_SEC", "9")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_MAX_SEC", "9")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_JITTER_SEC", "9")

    assert ModelRetryPolicy.from_env() == ModelRetryPolicy(2, 3.0, 4.0, 5.0)


def test_legacy_retry_env_is_supported(monkeypatch):
    _clear_retry_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_MAX_SEC", "7")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_JITTER_SEC", "0")

    assert ModelRetryPolicy.from_env() == ModelRetryPolicy(3, 0.0, 7.0, 0.0)


@pytest.mark.parametrize("status_code", [429, 503])
def test_retry_sync_retries_429_and_503(status_code: int):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _StatusError(status_code)
        return "ok"

    assert retry_sync(
        operation,
        operation_name=f"qwen:chat:status-{status_code}",
        policy=ModelRetryPolicy(attempts=2, base_sec=0, max_sec=0, jitter_sec=0),
        sleep=lambda _: None,
        jitter=lambda _: 0,
    ) == "ok"
    assert calls == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_non_retryable_4xx_does_not_retry(status_code: int):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise _StatusError(status_code)

    with pytest.raises(ModelRequestError) as caught:
        retry_sync(
            operation,
            operation_name="qwen:chat:non-retryable",
            policy=ModelRetryPolicy(attempts=5, base_sec=0, max_sec=0, jitter_sec=0),
            sleep=lambda _: None,
            jitter=lambda _: 0,
        )
    assert caught.value.status_code == status_code
    assert calls == 1


@pytest.mark.parametrize("error_type", [TimeoutError, ConnectionError])
def test_timeout_and_connection_errors_retry(error_type):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("temporary transport error")
        return "ok"

    assert retry_sync(
        operation,
        operation_name="qwen:embedding:transport",
        policy=ModelRetryPolicy(attempts=2, base_sec=0, max_sec=0, jitter_sec=0),
        sleep=lambda _: None,
        jitter=lambda _: 0,
    ) == "ok"
    assert calls == 2


def test_retry_after_is_used_and_capped_by_max_delay():
    calls = 0
    delays: list[float] = []

    class RetryAfterError(_StatusError):
        response = type("Response", (), {"headers": {"Retry-After": "3"}})()

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryAfterError(429)
        return "ok"

    assert retry_sync(
        operation,
        operation_name="gemini:generate_content:test",
        policy=ModelRetryPolicy(attempts=2, base_sec=1, max_sec=2, jitter_sec=0),
        sleep=delays.append,
        jitter=lambda _: 0,
    ) == "ok"
    assert delays == [2.0]


def test_max_attempts_is_preserved():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise _StatusError(503)

    with pytest.raises(ModelRequestError) as caught:
        retry_sync(
            operation,
            operation_name="model:always-fails",
            policy=ModelRetryPolicy(attempts=3, base_sec=0, max_sec=0, jitter_sec=0),
            sleep=lambda _: None,
            jitter=lambda _: 0,
        )
    assert calls == 3
    assert caught.value.attempts == 3


def test_async_non_awaitable_error_is_provider_neutral():
    with pytest.raises(ModelRequestError) as caught:
        asyncio.run(
            retry_async(
                lambda: "not-awaitable",
                operation_name="qwen:chat:test",
                policy=ModelRetryPolicy(attempts=1, base_sec=0, max_sec=0, jitter_sec=0),
            )
        )

    assert "Gemini" not in str(caught.value)
    assert "Model operation" in str(caught.value)


def test_gemini_compatibility_aliases_and_exports():
    assert GeminiRetryPolicy is ModelRetryPolicy
    assert GeminiRequestError is ModelRequestError
    assert gemini_classify_exception is classify_exception
    assert gemini_retry_sync is retry_sync
    assert 429 in RETRYABLE_STATUS_CODES


def test_secret_redaction_and_provider_neutral_error_text():
    with pytest.raises(ModelRequestError) as caught:
        retry_sync(
            lambda: (_ for _ in ()).throw(_StatusError(400, "GEMINI_API_KEY=secret-value")),
            operation_name="GEMINI_API_KEY=secret-value",
            policy=ModelRetryPolicy(attempts=1),
        )

    message = str(caught.value)
    assert "secret-value" not in message
    assert "Gemini operation" not in message
    assert "Model operation" in message
