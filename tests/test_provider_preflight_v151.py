from __future__ import annotations

from types import SimpleNamespace

import pytest

from platform_agent.provider_preflight import run_qwen_preflight


def response(payload: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def client(outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.mark.asyncio
async def test_provider_preflight_requires_structured_successes():
    fake, completions = client([response('{"ok":true}'), response('{"ok":true}')])
    result = await run_qwen_preflight(fake, checks=2, timeout_sec=0.1)
    assert result.ok is True
    assert result.requests_attempted == 2
    assert result.requests_completed == 2
    assert len(completions.calls) == 2
    assert completions.calls[0]["messages"][0]["content"].startswith("Provider preflight")


@pytest.mark.asyncio
async def test_provider_preflight_classifies_timeout_and_blocks_collection():
    fake, _ = client([TimeoutError("proxy timeout")])
    result = await run_qwen_preflight(fake, checks=1, timeout_sec=0.1)
    assert result.ok is False
    assert result.status == "FAIL"
    assert result.failure_types == ["provider_timeout"]
    assert result.timeout_count == 1


@pytest.mark.asyncio
async def test_provider_preflight_classifies_connection_and_invalid_json():
    fake, _ = client([ConnectionError("network unavailable")])
    result = await run_qwen_preflight(fake, checks=1, timeout_sec=0.1)
    assert result.failure_types == ["provider_connection_error"]

    fake, _ = client([response('{"ok":false}')])
    result = await run_qwen_preflight(fake, checks=1, timeout_sec=0.1)
    assert result.failure_types == ["provider_invalid_json"]
