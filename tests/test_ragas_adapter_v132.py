from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import platform_eval.ragas_adapter as adapter


def _sample(case_id: str = "rag_case"):
    return {
        "id": case_id,
        "user_input": "GPU 为什么等待？",
        "response": "GPU Reservation 当前没有可用资源。",
        "reference": "GPU Reservation 可能导致等待。",
        "retrieved_contexts": ["Reservation 会限制并发 GPU 分配。"],
    }


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.closed = False
        type(self).instances.append(self)

    async def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
        )

    async def close(self):
        self.closed = True


class _FakeEmbeddings:
    def __init__(self, capture):
        self.capture = capture

    async def aembed_texts(self, texts):
        return [[float(index)] * 4 for index, _ in enumerate(texts)]


def _fake_dependencies(monkeypatch, *, values=None, delays=None, errors=None):
    values = values or {}
    delays = delays or {}
    errors = errors or {}
    capture = {}

    def embedding_factory(provider, model, client):
        capture["embedding_provider"] = provider
        capture["embedding_model"] = model
        capture["embedding_client"] = client
        return _FakeEmbeddings(capture)

    def llm_factory(model, client, **kwargs):
        capture["judge_model"] = model
        capture["judge_client"] = client
        capture["judge_kwargs"] = kwargs
        return object()

    class _Metric:
        name = "metric"

        def __init__(self, **kwargs):
            del kwargs

        async def ascore(self, **kwargs):
            case_id = str(kwargs.get("user_input") or "")
            delay = delays.get(self.name, 0.0)
            if delay:
                await asyncio.sleep(delay)
            error = errors.get(self.name)
            if error:
                raise error
            return SimpleNamespace(value=values.get(self.name, 0.75))

    metric_classes = {}
    for metric_name in adapter.RAGAS_METRIC_NAMES:
        metric_classes[metric_name] = type(
            f"Fake{metric_name}",
            (_Metric,),
            {"name": metric_name},
        )

    monkeypatch.setattr(
        adapter,
        "_load_ragas_dependencies",
        lambda: {
            "AsyncOpenAI": _FakeClient,
            "embedding_factory": embedding_factory,
            "llm_factory": llm_factory,
            "metrics": metric_classes,
        },
    )
    monkeypatch.setattr(
        adapter,
        "_judge_config",
        lambda: (
            "qwen",
            "test-key",
            "https://correct.example/v1",
            "qwen3.7-flash",
            "qwen3.7-text-embedding",
        ),
    )
    return capture


def test_qwen_judge_and_embedding_config_are_provider_specific(monkeypatch):
    monkeypatch.setenv("PLATFORM_EVAL_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("DASHSCOPE_OPENAI_BASE_URL", "https://correct.example/v1")
    monkeypatch.delenv("PLATFORM_EVAL_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_EMBED_MODEL", raising=False)

    provider, _, base_url, model, embedding_model = adapter._judge_config()

    assert provider == "qwen"
    assert base_url == "https://correct.example/v1"
    assert model == "qwen-plus"
    assert embedding_model == "qwen3.7-text-embedding"


@pytest.mark.parametrize(
    ("provider", "api_key", "base_url", "model", "embedding_model"),
    [
        ("gemini", "GEMINI_API_KEY", adapter.GEMINI_OPENAI_BASE_URL, "gemini-3.7-flash", "gemini-embedding-2"),
        ("openai", "OPENAI_API_KEY", "https://openai.example/v1", "gpt-5-mini", "text-embedding-3-small"),
    ],
)
def test_gemini_and_openai_judge_config_remains_compatible(
    monkeypatch,
    provider,
    api_key,
    base_url,
    model,
    embedding_model,
):
    monkeypatch.setenv("PLATFORM_EVAL_PROVIDER", provider)
    monkeypatch.setenv(api_key, "test-key")
    if provider == "openai":
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    else:
        monkeypatch.delenv("PLATFORM_EVAL_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_EMBED_MODEL", raising=False)

    result = adapter._judge_config()

    assert result == (provider, "test-key", base_url, model, embedding_model)


def test_provider_smoke_uses_ragas_compatible_embedding_path(monkeypatch):
    _FakeClient.instances.clear()
    capture = _fake_dependencies(monkeypatch)

    result = adapter.run_ragas_provider_smoke()

    assert result["status"] == "PASS"
    assert len(result["judge"]) == 3
    assert all(item["status"] == "PASS" for item in result["judge"])
    assert result["embedding"][0]["count"] == 1
    assert result["embedding"][1]["count"] == 3
    assert capture["embedding_provider"] == "openai"
    assert capture["embedding_model"] == "qwen3.7-text-embedding"
    assert capture["embedding_client"] is _FakeClient.instances[0]
    assert _FakeClient.instances[0].closed is True


def test_metric_instrumentation_and_single_metric_success(monkeypatch):
    capture = _fake_dependencies(monkeypatch, values={"faithfulness": 0.8})
    result = adapter.run_ragas_judge([_sample()], metric_names=["faithfulness"])

    assert capture["judge_kwargs"]["max_retries"] == 1
    assert result["status"] == "PASS"
    assert result["metrics"] == {"faithfulness": 0.8}
    timing = result["timings"][0]
    assert timing["case_id"] == "rag_case"
    assert timing["metric_name"] == "faithfulness"
    assert timing["status"] == "PASS"
    assert timing["started_at"] and timing["finished_at"]
    assert timing["latency_sec"] >= 0
    assert result["metric_summary"]["faithfulness"]["status"] == "COMPLETE"


def test_metric_result_preserves_rag_case_metadata(monkeypatch):
    _fake_dependencies(monkeypatch, values={"faithfulness": 0.8})
    sample = _sample()
    sample.update({
        "query": sample["user_input"],
        "final_answer": sample["response"],
        "reference_answer": sample["reference"],
        "retrieved_sources": ["runbook/gpu.md#reservation"],
        "agent_model": "qwen-plus",
        "judge_model": "qwen-plus",
        "embedding_model": "qwen3.7-text-embedding",
        "agent_api_request_count": None,
        "token_usage": None,
    })

    result = adapter.run_ragas_judge([sample], metric_names=["faithfulness"])

    row = result["cases"][0]
    assert row["query"] == sample["query"]
    assert row["retrieved_sources"] == ["runbook/gpu.md#reservation"]
    assert row["agent_model"] == "qwen-plus"
    assert row["judge_model"] == "qwen-plus"
    assert row["embedding_model"] == "qwen3.7-text-embedding"
    assert row["judge_api_request_count"] == 0


def test_metric_instrumentation_counts_safe_api_operations(monkeypatch):
    capture = _fake_dependencies(monkeypatch)

    class CallingMetric:
        def __init__(self, **kwargs):
            del kwargs

        async def ascore(self, **kwargs):
            del kwargs
            await capture["judge_client"].chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "redacted"}],
            )
            return SimpleNamespace(value=0.6)

    deps = adapter._load_ragas_dependencies()
    deps["metrics"]["faithfulness"] = CallingMetric
    monkeypatch.setattr(adapter, "_load_ragas_dependencies", lambda: deps)

    result = adapter.run_ragas_judge([_sample()], metric_names=["faithfulness"])

    calls = result["timings"][0]["api_calls"]
    assert len(calls) == 1
    assert calls[0]["operation"] == "chat.completions.create"
    assert calls[0]["status"] == "PASS"
    assert calls[0]["latency_sec"] >= 0
    assert "messages" not in calls[0]
    assert "test-model" not in json.dumps(calls)


def test_metric_timeout_is_explicit(monkeypatch):
    monkeypatch.setenv("PLATFORM_EVAL_METRIC_TIMEOUT_SEC", "0.01")
    _fake_dependencies(monkeypatch, delays={"faithfulness": 0.05})

    result = adapter.run_ragas_judge([_sample()], metric_names=["faithfulness"])

    assert result["status"] == "BLOCKED_NOT_VALIDATED"
    assert result["timings"][0]["status"] == "METRIC_TIMEOUT"
    assert result["metric_summary"]["faithfulness"]["success_count"] == 0
    assert result["metrics"] == {}


def test_one_metric_failure_keeps_other_metric_result(monkeypatch):
    _fake_dependencies(
        monkeypatch,
        values={"answer_relevancy": 0.7},
        errors={"faithfulness": RuntimeError("judge unavailable")},
    )

    result = adapter.run_ragas_judge(
        [_sample()],
        metric_names=["faithfulness", "answer_relevancy"],
    )

    assert result["status"] == "PARTIAL"
    assert result["metrics"] == {"answer_relevancy": 0.7}
    assert result["metric_summary"]["faithfulness"]["status"] == "BLOCKED_NOT_VALIDATED"
    assert result["metric_summary"]["answer_relevancy"]["status"] == "COMPLETE"
    assert {item["status"] for item in result["timings"]} == {"METRIC_ERROR", "PASS"}


def test_aggregate_only_counts_successful_cases(monkeypatch):
    class CaseAwareMetric:
        def __init__(self, **kwargs):
            del kwargs

        async def ascore(self, **kwargs):
            if kwargs.get("user_input") == "bad":
                raise RuntimeError("one case failed")
            return SimpleNamespace(value=0.9)

    _fake_dependencies(monkeypatch)
    deps = adapter._load_ragas_dependencies()
    deps["metrics"]["faithfulness"] = CaseAwareMetric
    monkeypatch.setattr(adapter, "_load_ragas_dependencies", lambda: deps)

    good = _sample("good")
    bad = _sample("bad")
    bad["user_input"] = "bad"
    result = adapter.run_ragas_judge([good, bad], metric_names=["faithfulness"])

    assert result["status"] == "PARTIAL"
    assert result["metrics"] == {"faithfulness": 0.9}
    assert result["metric_summary"]["faithfulness"]["success_count"] == 1
    assert result["metric_summary"]["faithfulness"]["failure_count"] == 1


def test_provider_failure_has_clear_status_and_secret_redaction(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "_judge_config",
        lambda: ("qwen", "test-key", "https://correct.example/v1", "qwen3.7-flash", "qwen3.7-text-embedding"),
    )

    def broken_dependencies():
        raise RuntimeError("DASHSCOPE_API_KEY=secret-value provider unavailable")

    monkeypatch.setattr(adapter, "_load_ragas_dependencies", broken_dependencies)
    result = adapter.run_ragas_judge([_sample()], metric_names=["faithfulness"])

    assert result["status"] == "BLOCKED_NOT_VALIDATED"
    assert result["provider_error"]["status"] == "PROVIDER_PRIMITIVE_FAILED"
    assert "secret-value" not in result["provider_error"]["error_summary"]
