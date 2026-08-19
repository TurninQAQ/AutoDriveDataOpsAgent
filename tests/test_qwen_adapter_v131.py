from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from platform_agent.models import AgentIntent, AgentPlan
from platform_agent.qwen import QwenReadOnlyModel
from platform_integrations.model_retry import ModelRequestError
from platform_rag.embeddings import DenseEmbeddingIndex, GeminiEmbeddingProvider, QwenEmbeddingProvider
from platform_rag.models import KnowledgeChunk


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _chat_response(payload: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])


class _FakeCompletions:
    def __init__(self, failures=None, response=None):
        self.failures = list(failures or [])
        self.response = response or _chat_response('{"intent":"platform_health","decision_summary":"ok","tool_calls":[]}')
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return self.response


def _qwen_model(completions: _FakeCompletions) -> QwenReadOnlyModel:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return QwenReadOnlyModel(model="qwen3.7-flash", client=client)


@pytest.mark.asyncio
async def test_qwen_plan_uses_json_object_and_disables_thinking(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_MAX_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_JITTER_SEC", "0")
    completions = _FakeCompletions()
    plan = await _qwen_model(completions).plan("平台健康吗？", [{"name": "get_platform_health"}], [])
    assert plan.intent == AgentIntent.PLATFORM_HEALTH
    request = completions.calls[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"enable_thinking": False}
    assert "JSON" in request["messages"][0]["content"]


@pytest.mark.asyncio
async def test_qwen_503_retries_and_synthesizes_valid_response(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_MAX_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_JITTER_SEC", "0")
    response = _chat_response('{"intent":"platform_health","summary":"healthy"}')
    completions = _FakeCompletions(failures=[_StatusError(503)], response=response)
    model = _qwen_model(completions)
    plan = AgentPlan(intent=AgentIntent.PLATFORM_HEALTH, decision_summary="check", tool_calls=[])
    answer = await model.synthesize("health", plan, [], [], [])
    assert answer.summary == "healthy"
    assert len(completions.calls) == 2


@pytest.mark.asyncio
async def test_qwen_401_fails_without_retry(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_ATTEMPTS", "5")
    completions = _FakeCompletions(failures=[_StatusError(401)])
    model = _qwen_model(completions)
    with pytest.raises(ModelRequestError) as caught:
        await model._structured("Return JSON", AgentPlan)
    assert caught.value.status_code == 401
    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_qwen_invalid_json_is_explicit_failure():
    completions = _FakeCompletions(response=_chat_response("not-json"))
    with pytest.raises(RuntimeError, match="invalid AgentPlan JSON"):
        await _qwen_model(completions)._structured("Return JSON", AgentPlan)


def _vector(index: int) -> list[float]:
    values = [0.0] * 1024
    values[index] = 1.0
    return values


class _EmbeddingResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        raise _StatusError(self.status_code)


class _FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _embedding_payload(order=(0, 1)):
    return {
        "output": {
            "embeddings": [
                {"text_index": index, "embedding": _vector(index)} for index in order
            ]
        }
    }


def test_qwen_embedding_document_query_batch_and_reorders_indices(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_MAX_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_JITTER_SEC", "0")
    http = _FakeHTTP([_EmbeddingResponse(200, _embedding_payload((1, 0))), _EmbeddingResponse(200, _embedding_payload((0,)))])
    provider = QwenEmbeddingProvider(
        client=http,
        api_key="test-key",
        base_url="https://example.invalid/api/v1",
        batch_size=32,
    )
    chunks = [
        KnowledgeChunk(chunk_id="a", source_path="a.md", title="A", content="a", content_hash="a"),
        KnowledgeChunk(chunk_id="b", source_path="b.md", title="B", content="b", content_hash="b"),
    ]
    vectors = provider.embed_documents(chunks)
    assert provider.batch_size == 20
    assert vectors["a"][0] == pytest.approx(1.0)
    assert vectors["b"][1] == pytest.approx(1.0)
    request = http.calls[0][1]
    assert request["json"]["parameters"] == {"text_type": "document", "dimension": 1024, "output_type": "dense"}
    provider.embed_query("question")
    assert http.calls[1][1]["json"]["parameters"]["text_type"] == "query"


def test_qwen_embedding_retries_429_and_rejects_400(monkeypatch):
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_BASE_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_MAX_SEC", "0")
    monkeypatch.setenv("PLATFORM_MODEL_RETRY_JITTER_SEC", "0")
    http = _FakeHTTP([_StatusError(503), _EmbeddingResponse(200, _embedding_payload((0,)))])
    provider = QwenEmbeddingProvider(client=http, api_key="test-key", base_url="https://example.invalid/api/v1")
    assert len(provider.embed_query("q")) == 1024
    assert len(http.calls) == 2

    bad = _FakeHTTP([_EmbeddingResponse(400, {})])
    with pytest.raises(ModelRequestError) as caught:
        QwenEmbeddingProvider(client=bad, api_key="test-key", base_url="https://example.invalid/api/v1").embed_query("q")
    assert caught.value.status_code == 400
    assert len(bad.calls) == 1


def test_qwen_embedding_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="dimension=1024"):
        QwenEmbeddingProvider(client=SimpleNamespace(post=None), dimension=768)


def test_gemini_sidecar_is_not_reused_by_qwen(tmp_path: Path):
    class FakeGemini:
        provider_name = "gemini"
        model_name = "gemini-embedding-2"
        dimension = 768
        batch_size = 2

        def embed_documents(self, chunks):
            return {chunk.chunk_id: [1.0] for chunk in chunks}

    class FakeQwen:
        provider_name = "qwen"
        model_name = "qwen3.7-text-embedding"
        dimension = 1024
        batch_size = 2

        def embed_documents(self, chunks):
            return {chunk.chunk_id: [1.0] for chunk in chunks}

    chunks = [KnowledgeChunk(chunk_id="a", source_path="a", title="a", content="a", content_hash="a")]
    path = tmp_path / "embeddings.json"
    DenseEmbeddingIndex(path, FakeGemini()).ensure("fp", chunks)
    qwen = DenseEmbeddingIndex(path, FakeQwen())
    assert not qwen.is_fresh("fp", ["a"])
    qwen.ensure("fp", chunks)
    assert json.loads(path.read_text(encoding="utf-8"))["provider"] == "qwen"
