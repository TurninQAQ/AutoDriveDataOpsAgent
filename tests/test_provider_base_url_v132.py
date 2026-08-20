from __future__ import annotations

import sys
import types

import platform_agent.gemini as gemini_module
import platform_agent.model as model_module
import platform_agent.qwen as qwen_module
from platform_agent.model import HeuristicReadOnlyModel, build_model_from_env
from platform_agent.settings import AgentSettings


class _CapturedModel:
    def __init__(self, model, temperature, base_url=None):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url


class _CapturedGemini:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature


def test_settings_keep_openai_and_qwen_base_urls_separate(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://correct.example")

    settings = AgentSettings.from_env()

    assert settings.openai_base_url == "https://wrong.example"
    assert settings.qwen_base_url == "https://correct.example"
    assert not hasattr(settings, "base_url")


def test_qwen_uses_dashscope_base_url_when_both_endpoints_exist(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://correct.example")
    monkeypatch.setattr(qwen_module, "QwenReadOnlyModel", _CapturedModel)

    result = build_model_from_env(
        "qwen",
        "qwen3.7-flash",
        0.0,
        "https://legacy-settings-value.example",
    )

    assert result.base_url == "https://correct.example"


def test_qwen_adapter_prefers_dashscope_environment_endpoint(monkeypatch):
    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://correct.example")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))

    qwen_module.QwenReadOnlyModel(
        model="qwen3.7-flash",
        base_url="https://legacy-settings-value.example",
    )

    assert captured["base_url"] == "https://correct.example"


def test_openai_uses_openai_base_url_when_both_endpoints_exist(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://correct-openai.example")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://wrong-qwen.example")
    monkeypatch.setattr(model_module, "OpenAIReadOnlyModel", _CapturedModel)

    result = build_model_from_env(
        "openai",
        "gpt-test",
        0.0,
        "https://legacy-settings-value.example",
    )

    assert result.base_url == "https://correct-openai.example"


def test_gemini_and_heuristic_do_not_receive_provider_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://dashscope.example")
    monkeypatch.setattr(gemini_module, "GeminiReadOnlyModel", _CapturedGemini)

    gemini = build_model_from_env("gemini", "gemini-test", 0.0, "https://wrong.example")
    heuristic = build_model_from_env("heuristic", "ignored", 0.0, "https://wrong.example")

    assert isinstance(gemini, _CapturedGemini)
    assert not hasattr(gemini, "base_url")
    assert isinstance(heuristic, HeuristicReadOnlyModel)
