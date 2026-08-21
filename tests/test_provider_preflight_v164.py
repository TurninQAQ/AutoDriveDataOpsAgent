from __future__ import annotations

from types import SimpleNamespace

import pytest

from platform_agent.provider_preflight import run_qwen_preflight


class FakeCompletions:
    async def create(self, **kwargs):
        raise RuntimeError("403 AllocationQuota.FreeTierOnly")


@pytest.mark.asyncio
async def test_free_tier_quota_is_classified_without_recording_provider_body():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = await run_qwen_preflight(client, checks=1, timeout_sec=0.1)
    assert result.ok is False
    assert result.quota_blocked is True
    assert result.as_dict()["quota_blocked"] is True
    assert "AllocationQuota.FreeTierOnly" not in result.as_dict()["failure_types"]
