from __future__ import annotations

import json
import sys
import types as pytypes
from pathlib import Path

import pytest

from platform_agent.models import AgentIntent, AgentPlan, AgentResponse
from platform_agent.settings import AgentSettings
from platform_core.settings import PlatformSettings
from platform_rag.embeddings import DenseEmbeddingIndex, GeminiEmbeddingProvider
from platform_rag.models import KnowledgeChunk
from platform_rag.service import KnowledgeService


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbeddingResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbedding(v) for v in vectors]


class _FakeEmbeddingModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append((model, list(contents), config))
        vectors = []
        for text in contents:
            if not isinstance(text, str):
                text = str(text)
            lower = text.lower()
            if "gpu" in lower or "显存" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "recovery" in lower or "恢复" in lower:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return _FakeEmbeddingResponse(vectors)


class _FakeEmbeddingClient:
    def __init__(self):
        self.models = _FakeEmbeddingModels()


def _chunk(cid: str, title: str, content: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=cid,
        source_path=f"{cid}.md",
        title=title,
        section=title,
        content=content,
        content_hash=cid,
    )


def test_gemini_embedding_provider_formats_and_normalizes():
    client = _FakeEmbeddingClient()
    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-2", dimension=768, client=client)
    chunks = [_chunk("gpu", "GPU", "GPU显存与Reservation"), _chunk("rec", "Recovery", "断点恢复")]
    vectors = provider.embed_documents(chunks)
    assert vectors["gpu"] == [1.0, 0.0, 0.0]
    assert vectors["rec"] == [0.0, 1.0, 0.0]
    query = provider.embed_query("为什么GPU显存不足")
    assert query == [1.0, 0.0, 0.0]
    # fake-client path still receives asymmetric prefixes as strings
    first_batch = client.models.calls[0][1]
    assert "title: GPU" in first_batch[0]
    last_batch = client.models.calls[-1][1]
    assert "task: question answering" in last_batch[0]


def test_dense_embedding_index_cache_avoids_reembedding(tmp_path: Path):
    client = _FakeEmbeddingClient()
    provider = GeminiEmbeddingProvider(client=client, dimension=768)
    store = DenseEmbeddingIndex(tmp_path / "embeddings.json", provider)
    chunks = [_chunk("gpu", "GPU", "gpu memory")]
    first = store.ensure("fp-1", chunks)
    calls_after_first = len(client.models.calls)
    second = store.ensure("fp-1", chunks)
    assert first == second
    assert len(client.models.calls) == calls_after_first
    store.ensure("fp-2", chunks)
    # V1.3 reuses dense vectors by chunk content hash even when the lexical
    # source fingerprint changes.
    assert len(client.models.calls) == calls_after_first


def test_knowledge_service_uses_dense_embedding_sidecar(tmp_path: Path):
    src = tmp_path / "knowledge"
    src.mkdir()
    (src / "gpu.md").write_text("# GPU\n\nSegment显存与独占Reservation。", encoding="utf-8")
    (src / "recovery.md").write_text("# Recovery\n\n低优任务从checkpoint断点恢复。", encoding="utf-8")
    client = _FakeEmbeddingClient()
    provider = GeminiEmbeddingProvider(client=client, dimension=768)
    service = KnowledgeService(
        src,
        tmp_path / "index.json",
        top_k=2,
        min_score=0.0,
        lexical_weight=0.0,
        vector_weight=1.0,
        embedding_provider=provider,
        embedding_index_file=tmp_path / "dense.json",
    )
    service.build(force=True)
    result = service.search("GPU显存不足", top_k=1)
    assert result.results
    assert result.results[0].source_path == "gpu.md"
    status = service.status()
    assert status["retrieval_mode"] == "gemini_hybrid"
    assert status["embedding"]["vector_count"] == 2


def test_agent_settings_auto_prefers_gemini_and_provider_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path / "runtime" / "opt_airflow"))
    monkeypatch.setenv("AIRFLOW_STATE_DIR", str(tmp_path / "runtime" / "state"))
    monkeypatch.setenv("AIRFLOW_TASK_CONFIG_ROOT", str(tmp_path / "runtime" / "opt_airflow" / "config" / "tasks"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("PLATFORM_AGENT_MODEL", raising=False)
    monkeypatch.setenv("PLATFORM_AGENT_PROVIDER", "auto")
    monkeypatch.setenv("PLATFORM_RAG_EMBED_PROVIDER", "gemini")
    settings = AgentSettings.from_env(PlatformSettings.from_env())
    assert settings.model == "gemini-3.7-flash"
    assert settings.knowledge_embedding_provider == "gemini"
    assert settings.knowledge_embedding_model == "gemini-embedding-2"
    assert settings.knowledge_embedding_dimension == 768
    assert settings.knowledge_lexical_weight == pytest.approx(0.5)
    assert settings.knowledge_vector_weight == pytest.approx(0.5)


def test_agent_settings_accepts_empty_optional_hybrid_weights(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path / "runtime" / "opt_airflow"))
    monkeypatch.setenv("AIRFLOW_STATE_DIR", str(tmp_path / "runtime" / "state"))
    monkeypatch.setenv("AIRFLOW_TASK_CONFIG_ROOT", str(tmp_path / "runtime" / "opt_airflow" / "config" / "tasks"))
    monkeypatch.setenv("PLATFORM_RAG_EMBED_PROVIDER", "gemini")
    monkeypatch.setenv("PLATFORM_RAG_LEXICAL_WEIGHT", "")
    monkeypatch.setenv("PLATFORM_RAG_VECTOR_WEIGHT", "")
    settings = AgentSettings.from_env(PlatformSettings.from_env())
    assert settings.knowledge_lexical_weight == pytest.approx(0.5)
    assert settings.knowledge_vector_weight == pytest.approx(0.5)


def _install_fake_google(monkeypatch, plan_json: str, answer_json: str):
    queue = [plan_json, answer_json]

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class AsyncModels:
        async def generate_content(self, **kwargs):
            assert kwargs["config"].response_mime_type == "application/json"
            assert kwargs["config"].response_json_schema
            return FakeResponse(queue.pop(0))

    class FakeAio:
        models = AsyncModels()

    class FakeClient:
        def __init__(self, api_key=None):
            assert api_key == "test-key"
            self.aio = FakeAio()

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    google_mod = pytypes.ModuleType("google")
    genai_mod = pytypes.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    types_mod = pytypes.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = GenerateContentConfig
    genai_mod.types = types_mod
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)


@pytest.mark.asyncio
async def test_native_gemini_model_returns_valid_plan_and_answer(monkeypatch):
    plan = AgentPlan(intent=AgentIntent.PLATFORM_HEALTH, decision_summary="check", tool_calls=[])
    answer = AgentResponse(intent=AgentIntent.PLATFORM_HEALTH, summary="healthy")
    _install_fake_google(monkeypatch, plan.model_dump_json(), answer.model_dump_json())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from platform_agent.gemini import GeminiReadOnlyModel

    model = GeminiReadOnlyModel("gemini-3.7-flash")
    got_plan = await model.plan("平台健康吗", [], [])
    assert got_plan.intent == AgentIntent.PLATFORM_HEALTH
    got_answer = await model.synthesize("平台健康吗", got_plan, [], [], [])
    assert got_answer.summary == "healthy"


def test_requirements_and_env_document_gemini_support():
    root = Path(__file__).resolve().parents[1]
    assert "google-genai==2.18.1" in (root / "requirements-agent.txt").read_text()
    env = (root / ".env.example").read_text()
    assert "GEMINI_API_KEY" in env
    assert "PLATFORM_RAG_EMBED_PROVIDER" in env
    assert "gemini-embedding-2" in env

def test_ragas_judge_config_auto_uses_gemini(monkeypatch):
    from platform_eval.ragas_adapter import GEMINI_OPENAI_BASE_URL, _judge_config
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_PROVIDER", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("PLATFORM_EVAL_EMBED_MODEL", raising=False)
    provider, key, base_url, model, embedding = _judge_config()
    assert provider == "gemini"
    assert key == "test-key"
    assert base_url == GEMINI_OPENAI_BASE_URL
    assert model == "gemini-3.7-flash"
    assert embedding == "gemini-embedding-2"


def test_v12_deploy_contract_and_version_marker():
    root = Path(__file__).resolve().parents[1]
    deploy = (root / "scripts" / "deploy_ci_cloud.sh").read_text(encoding="utf-8")
    platform = (root / "platform").read_text(encoding="utf-8")
    assert "agent-v1.2.0" in deploy
    assert "V1.2_GEMINI_PROVIDER.md" in deploy
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "PLATFORM_GPU_RUNTIME",
        "PLATFORM_GPU_SIM_CONFIG",
        "PLATFORM_GPU_SIM_STATE",
        "PLATFORM_GPU_SIM_FALLBACK_OS_PROCESS",
        "PLATFORM_STAGE_RUNTIME",
        "MOCK_STAGE_RESULT",
        "MOCK_STAGE_DURATION_SEC",
        "PLATFORM_RAG_EMBED_PROVIDER",
        "PLATFORM_RAG_EMBED_MODEL",
        "PLATFORM_RAG_EMBED_DIM",
        "PLATFORM_RAG_EMBED_INDEX",
    ):
        assert name in platform


def test_dense_embedding_index_invalidates_on_model_change(tmp_path: Path):
    client_a = _FakeEmbeddingClient()
    provider_a = GeminiEmbeddingProvider(model_name="gemini-embedding-2", client=client_a, dimension=768)
    store_a = DenseEmbeddingIndex(tmp_path / "embeddings.json", provider_a)
    chunks = [_chunk("gpu", "GPU", "gpu memory")]
    store_a.ensure("same-fingerprint", chunks)
    assert store_a.is_fresh("same-fingerprint", ["gpu"])

    client_b = _FakeEmbeddingClient()
    provider_b = GeminiEmbeddingProvider(model_name="another-embedding-model", client=client_b, dimension=768)
    store_b = DenseEmbeddingIndex(tmp_path / "embeddings.json", provider_b)
    assert not store_b.is_fresh("same-fingerprint", ["gpu"])
    store_b.ensure("same-fingerprint", chunks)
    assert client_b.models.calls


def test_gemini_key_headers_are_redacted_from_free_text():
    from platform_observability.redaction import REDACTED, redact_text

    samples = [
        "GEMINI_API_KEY=secret-value-123",
        "GOOGLE_API_KEY: secret-value-456",
        "curl -H 'X-goog-api-key: secret-value-789' https://example.invalid",
    ]
    for sample in samples:
        redacted = redact_text(sample)
        assert "secret-value" not in redacted
        assert REDACTED in redacted
