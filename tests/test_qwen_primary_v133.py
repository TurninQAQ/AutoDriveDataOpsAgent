from __future__ import annotations

import sys
import types

import platform_agent.qwen as qwen_module
from platform_agent.model import build_model_from_env
from platform_agent.settings import AgentSettings


def test_qwen_is_primary_default_when_provider_is_explicit(monkeypatch):
    monkeypatch.setenv("PLATFORM_AGENT_PROVIDER", "qwen")
    monkeypatch.delenv("PLATFORM_AGENT_MODEL", raising=False)
    settings = AgentSettings.from_env()

    assert settings.provider == "qwen"
    assert settings.model == "qwen-plus"


def test_auto_qwen_configuration_selects_qwen_plus(monkeypatch):
    monkeypatch.setenv("PLATFORM_AGENT_PROVIDER", "auto")
    monkeypatch.delenv("PLATFORM_AGENT_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://dashscope.example/v1")

    settings = AgentSettings.from_env()

    assert settings.provider == "auto"
    assert settings.model == "qwen-plus"


def test_qwen_provider_receives_primary_model_without_changing_endpoint(monkeypatch):
    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://correct.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example/v1")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))

    model = build_model_from_env("qwen", "qwen-plus", 0.0)

    assert isinstance(model, qwen_module.QwenReadOnlyModel)
    assert model.model == "qwen-plus"
    assert captured["base_url"] == "https://correct.example/v1"


def test_qwen_adapter_direct_default_is_qwen_plus():
    model = qwen_module.QwenReadOnlyModel(client=object())

    assert model.model == "qwen-plus"
