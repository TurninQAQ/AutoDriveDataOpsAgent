from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from platform_agent.models import AgentIntent, AgentPlan
from platform_agent.prompt_contract import ADAPTIVE_EVIDENCE_CONTRACT


def _chat_response(payload: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _chat_response(self.payload)


@pytest.mark.asyncio
async def test_qwen_decide_next_uses_shared_adaptive_contract_and_step_schema():
    from platform_agent.qwen import QwenReadOnlyModel

    completions = FakeCompletions(
        '{"action":"CALL_TOOL","tool_call":{"name":"get_gpu_pool","arguments":{}},"evidence_sufficient":false,"revised_intent":"gpu_diagnosis","decision_summary":"Live GPU evidence is still missing."}'
    )
    model = QwenReadOnlyModel(
        model="qwen-plus",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    decision = await model.decide_next(
        user_text="当前 GPU 怎么样？",
        initial_plan=AgentPlan(intent=AgentIntent.GPU_DIAGNOSIS),
        tool_descriptions=[{"name": "get_gpu_pool"}],
        observations=[],
        knowledge=[],
        history=[],
        step_index=0,
        remaining_tool_calls=6,
    )
    assert decision.tool_call.name == "get_gpu_pool"
    prompt = completions.calls[0]["messages"][1]["content"]
    assert ADAPTIVE_EVIDENCE_CONTRACT in prompt
    assert "one read-only ToolCallSpec" in prompt
    assert "chain-of-thought" in prompt


def _install_fake_google(monkeypatch, payload: str):
    class FakeResponse:
        text = payload

    class AsyncModels:
        async def generate_content(self, **kwargs):
            assert kwargs["config"].response_json_schema
            return FakeResponse()

    class FakeAio:
        models = AsyncModels()

    class FakeClient:
        def __init__(self, api_key=None):
            assert api_key == "test-key"
            self.aio = FakeAio()

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    types_mod = types.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = GenerateContentConfig
    genai_mod.types = types_mod
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)


@pytest.mark.asyncio
async def test_gemini_decide_next_uses_the_same_structured_contract(monkeypatch):
    _install_fake_google(
        monkeypatch,
        '{"action":"FINISH","tool_call":null,"evidence_sufficient":true,"decision_summary":"Evidence is sufficient."}',
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from platform_agent.gemini import GeminiReadOnlyModel

    model = GeminiReadOnlyModel("gemini-3.7-flash")
    decision = await model.decide_next(
        user_text="你好",
        initial_plan=AgentPlan(intent=AgentIntent.GENERAL_READ),
        tool_descriptions=[], observations=[], knowledge=[], history=[],
        step_index=0, remaining_tool_calls=6,
    )
    assert decision.evidence_sufficient is True
