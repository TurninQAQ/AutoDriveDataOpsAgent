from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from platform_integrations.gemini_retry import GeminiRequestError, GeminiRetryPolicy, retry_sync
from platform_rag.embeddings import GeminiEmbeddingProvider


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "transient"):
        super().__init__(message)
        self.status_code = status_code


class _Response:
    text = '{"value":"ok"}'


class _Result(BaseModel):
    value: str


@dataclass
class _Config:
    temperature: float
    response_mime_type: str
    response_json_schema: dict


class _Types:
    GenerateContentConfig = _Config


class _AsyncModels:
    def __init__(self, failures: list[Exception]):
        self.failures = failures
        self.calls = 0

    async def generate_content(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _Response()


class _AsyncClient:
    def __init__(self, failures: list[Exception]):
        self.aio = type("Aio", (), {"models": _AsyncModels(failures)})()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 429, 503])
async def test_async_generation_retries_transient_status(monkeypatch, status_code: int):
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_MAX_SEC", "0")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_JITTER_SEC", "0")
    from platform_agent.gemini import GeminiReadOnlyModel

    model = object.__new__(GeminiReadOnlyModel)
    model._types = _Types
    model.client = _AsyncClient([_StatusError(status_code)])
    model.model = "test-model"
    model.temperature = 0.0
    result = await model._structured("safe prompt", _Result)
    assert result.value == "ok"
    assert model.client.aio.models.calls == 2


def test_embedding_batch_retries_503(monkeypatch):
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_MAX_SEC", "0")
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_JITTER_SEC", "0")

    class Models:
        def __init__(self):
            self.calls = 0

        def embed_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _StatusError(503)
            return type("EmbeddingResponse", (), {"embeddings": [type("Embedding", (), {"values": [3.0, 4.0]})()]})()

    client = type("Client", (), {"models": Models()})()
    provider = GeminiEmbeddingProvider(client=client, dimension=768)
    assert provider._embed_strings(["one"]) == [[0.6, 0.8]]
    assert client.models.calls == 2


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_non_retryable_status_fails_once(monkeypatch, status_code: int):
    monkeypatch.setenv("PLATFORM_GEMINI_RETRY_ATTEMPTS", "5")
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise _StatusError(status_code, "GEMINI_API_KEY=secret-value-must-not-escape")

    with pytest.raises(GeminiRequestError) as caught:
        retry_sync(operation, operation_name="test-operation", sleep=lambda _: None, jitter=lambda _: 0)
    assert caught.value.status_code == status_code
    assert calls == 1
    assert "secret-value" not in str(caught.value)


def test_repeated_429_stops_at_max_attempts():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise _StatusError(429)

    with pytest.raises(GeminiRequestError) as caught:
        retry_sync(
            operation,
            operation_name="test-operation",
            policy=GeminiRetryPolicy(attempts=3, base_sec=1, max_sec=20, jitter_sec=0),
            sleep=lambda _: None,
            jitter=lambda _: 0,
        )
    assert calls == 3
    assert caught.value.attempts == 3


def test_retry_after_is_preferred_over_exponential_delay():
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

    result = retry_sync(
        operation,
        operation_name="test-operation",
        policy=GeminiRetryPolicy(attempts=2, base_sec=1, max_sec=20, jitter_sec=0),
        sleep=delays.append,
        jitter=lambda _: 0,
    )
    assert result == "ok"
    assert delays == [3.0]


def test_async_retry_does_not_block_with_injected_sleep():
    events: list[str] = []

    async def operation():
        events.append("call")
        if events.count("call") == 1:
            raise _StatusError(429)
        return "ok"

    async def fake_sleep(delay: float):
        events.append(f"sleep:{delay}")
        await asyncio.sleep(0)

    from platform_integrations.gemini_retry import retry_async

    result = asyncio.run(
        retry_async(
            operation,
            operation_name="test-operation",
            policy=GeminiRetryPolicy(attempts=2, base_sec=0, max_sec=0, jitter_sec=0),
            sleep=fake_sleep,
            jitter=lambda _: 0,
        )
    )
    assert result == "ok"
    assert events == ["call", "sleep:0.0", "call"]
