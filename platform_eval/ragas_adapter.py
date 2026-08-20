from __future__ import annotations

import asyncio
import math
import os
import sys
import time
import types
from datetime import datetime, timezone
from statistics import median
from typing import Any, Mapping

from platform_observability.redaction import redact_text


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
)
GENERATION_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)


def _require(value: str, message: str) -> str:
    value = value.strip()
    if not value:
        raise RuntimeError(message)
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return value if math.isfinite(value) and value > 0 else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _judge_config() -> tuple[str, str, str | None, str, str]:
    provider = os.getenv("PLATFORM_EVAL_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "openai"
    if provider in {"gemini", "google"}:
        api_key = _require(
            os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini Ragas evaluation",
        )
        base_url = os.getenv("PLATFORM_EVAL_JUDGE_BASE_URL", "").strip() or GEMINI_OPENAI_BASE_URL
        model = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "").strip() or "gemini-3.7-flash"
        embedding_model = os.getenv("PLATFORM_EVAL_EMBED_MODEL", "").strip() or "gemini-embedding-2"
        return provider, api_key, base_url, model, embedding_model
    if provider in {"qwen", "dashscope", "aliyun", "alibaba"}:
        api_key = _require(
            os.getenv("DASHSCOPE_API_KEY", ""),
            "DASHSCOPE_API_KEY is required for Qwen Ragas evaluation",
        )
        base_url = (
            os.getenv("PLATFORM_EVAL_JUDGE_BASE_URL", "").strip()
            or os.getenv("DASHSCOPE_OPENAI_BASE_URL", "").strip()
            or None
        )
        if not base_url:
            raise RuntimeError(
                "DASHSCOPE_OPENAI_BASE_URL (or PLATFORM_EVAL_JUDGE_BASE_URL) is required for Qwen Ragas evaluation"
            )
        model = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "").strip() or "qwen3.7-flash"
        embedding_model = os.getenv("PLATFORM_EVAL_EMBED_MODEL", "").strip() or "qwen3.7-text-embedding"
        return provider, api_key, base_url, model, embedding_model
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        api_key = _require(os.getenv("OPENAI_API_KEY", ""), "OPENAI_API_KEY is required for OpenAI Ragas evaluation")
        base_url = os.getenv("PLATFORM_EVAL_JUDGE_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip() or None
        model = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "").strip() or "gpt-5-mini"
        embedding_model = os.getenv("PLATFORM_EVAL_EMBED_MODEL", "").strip() or "text-embedding-3-small"
        return provider, api_key, base_url, model, embedding_model
    raise RuntimeError(f"Unsupported PLATFORM_EVAL_PROVIDER: {provider}")


def _load_ragas_dependencies() -> dict[str, Any]:
    try:
        # Ragas 0.4.3 imports this legacy optional Vertex adapter eagerly, while
        # the current langchain-community package removed that module. Qwen's
        # OpenAI-compatible path never uses Vertex; provide the smallest import
        # shim so the supported Ragas OpenAI factory remains usable.
        try:
            import langchain_community.chat_models.vertexai  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            if exc.name != "langchain_community.chat_models.vertexai":
                raise
            vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
            vertexai_shim.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[vertexai_shim.__name__] = vertexai_shim
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements-eval.txt before running Ragas judge metrics") from exc
    return {
        "AsyncOpenAI": AsyncOpenAI,
        "embedding_factory": embedding_factory,
        "llm_factory": llm_factory,
        "metrics": {
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "answer_correctness": AnswerCorrectness,
        },
    }


def _request_timeout() -> float:
    return _env_float("PLATFORM_EVAL_REQUEST_TIMEOUT_SEC", 45.0)


def _metric_timeout() -> float:
    # The measured Qwen single-case fan-out is below 80 seconds for the
    # slowest diagnostic metric; keep a finite headroom without masking hangs.
    return _env_float("PLATFORM_EVAL_METRIC_TIMEOUT_SEC", 90.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _error_summary(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    details = f"status_code={status}" if status is not None else ""
    # Keep diagnostics short and redacted; never persist prompt, auth header or
    # the full provider exception payload.
    text = redact_text(str(exc)).replace("\n", " ").strip()
    if len(text) > 240:
        text = text[:240] + "..."
    return "; ".join(item for item in (type(exc).__name__, details, text) if item)


def _instrument_client(client: Any) -> list[dict[str, Any]]:
    """Count safe request metadata without changing the OpenAI client type."""

    calls: list[dict[str, Any]] = []
    for operation, endpoint in (
        ("chat.completions.create", getattr(getattr(client, "chat", None), "completions", None)),
        ("embeddings.create", getattr(client, "embeddings", None)),
    ):
        if endpoint is None:
            continue
        original = getattr(endpoint, "create", None)
        if original is None:
            continue

        async def instrumented_create(
            *args: Any,
            _original: Any = original,
            _operation: str = operation,
            **kwargs: Any,
        ) -> Any:
            started = time.perf_counter()
            call: dict[str, Any] = {
                "operation": _operation,
                "status": "ERROR",
                "latency_sec": None,
                "error_type": None,
            }
            try:
                result = await _original(*args, **kwargs)
                call["status"] = "PASS"
                return result
            except BaseException as exc:
                call["error_type"] = type(exc).__name__
                raise
            finally:
                call["latency_sec"] = time.perf_counter() - started
                calls.append(call)

        setattr(endpoint, "create", instrumented_create)
    return calls


def _metric_kwargs(name: str, sample: Mapping[str, Any]) -> dict[str, Any]:
    user_input = str(sample.get("user_input") or "")
    response = str(sample.get("response") or "")
    reference = str(sample.get("reference") or "")
    contexts = list(sample.get("retrieved_contexts") or [])
    if name == "faithfulness":
        return {"user_input": user_input, "response": response, "retrieved_contexts": contexts}
    if name == "answer_relevancy":
        return {"user_input": user_input, "response": response}
    if name == "answer_correctness":
        return {"user_input": user_input, "response": response, "reference": reference}
    if name == "context_recall":
        return {"user_input": user_input, "retrieved_contexts": contexts, "reference": reference}
    if name == "context_precision":
        return {"user_input": user_input, "retrieved_contexts": contexts, "reference": reference}
    raise ValueError(f"Unsupported Ragas metric: {name}")


def _metric_stats(timings: list[dict[str, Any]], case_count: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in RAGAS_METRIC_NAMES}
    for item in timings:
        grouped.setdefault(str(item["metric_name"]), []).append(item)
    summary: dict[str, Any] = {}
    for name, items in grouped.items():
        passed = [item for item in items if item.get("status") == "PASS" and item.get("score") is not None]
        failed = [item for item in items if item.get("status") != "PASS"]
        latencies = [float(item["latency_sec"]) for item in items]
        scores = [float(item["score"]) for item in passed]
        if len(passed) == case_count and case_count > 0:
            status = "COMPLETE"
        elif passed:
            status = "PARTIAL"
        else:
            status = "BLOCKED_NOT_VALIDATED"
        summary[name] = {
            "status": status,
            "success_count": len(passed),
            "failure_count": len(failed),
            "score": sum(scores) / len(scores) if scores else None,
            "mean": sum(scores) / len(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "p50": median(scores) if scores else None,
            "latency_mean_sec": sum(latencies) / len(latencies) if latencies else None,
            "latency_p50_sec": median(latencies) if latencies else None,
        }
    return summary


async def _provider_smoke_async() -> dict[str, Any]:
    provider, api_key, base_url, model, embedding_model = _judge_config()
    deps = _load_ragas_dependencies()
    client = deps["AsyncOpenAI"](
        api_key=api_key,
        base_url=base_url,
        timeout=_request_timeout(),
        max_retries=0,
    )
    result: dict[str, Any] = {
        "status": "PROVIDER_PRIMITIVE_FAILED",
        "provider": provider,
        "judge_model": model,
        "embedding_model": embedding_model,
        "base_url_configured": bool(base_url),
        "judge": [],
        "embedding": [],
    }
    try:
        for index in range(3):
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": 'Reply with JSON: {"ok":true}'}],
                        temperature=0,
                        response_format={"type": "json_object"},
                    ),
                    _request_timeout(),
                )
                content = getattr(response.choices[0].message, "content", "") if response.choices else ""
                if not str(content).strip():
                    raise RuntimeError("judge response was empty")
                result["judge"].append({"attempt": index + 1, "status": "PASS", "latency_sec": time.perf_counter() - started})
            except Exception as exc:
                result["judge"].append({
                    "attempt": index + 1,
                    "status": "FAIL",
                    "latency_sec": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error_summary": _error_summary(exc),
                })
        embeddings = deps["embedding_factory"]("openai", model=embedding_model, client=client)
        for label, texts in (("one_text", ["hello"]), ("three_texts", ["hello", "world", "test"])):
            started = time.perf_counter()
            try:
                vectors = await asyncio.wait_for(embeddings.aembed_texts(texts), _request_timeout())
                dimensions = sorted({len(vector) for vector in vectors})
                if len(vectors) != len(texts) or not dimensions:
                    raise RuntimeError("embedding response shape was invalid")
                result["embedding"].append({
                    "input": label,
                    "status": "PASS",
                    "latency_sec": time.perf_counter() - started,
                    "count": len(vectors),
                    "dimensions": dimensions,
                })
            except Exception as exc:
                result["embedding"].append({
                    "input": label,
                    "status": "FAIL",
                    "latency_sec": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error_summary": _error_summary(exc),
                })
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            closed = close()
            if hasattr(closed, "__await__"):
                await closed
    result["status"] = (
        "PASS"
        if len(result["judge"]) == 3
        and all(item["status"] == "PASS" for item in result["judge"])
        and len(result["embedding"]) == 2
        and all(item["status"] == "PASS" for item in result["embedding"])
        else "PROVIDER_PRIMITIVE_FAILED"
    )
    return result


def run_ragas_provider_smoke() -> dict[str, Any]:
    try:
        return asyncio.run(_provider_smoke_async())
    except Exception as exc:
        return {
            "status": "PROVIDER_PRIMITIVE_FAILED",
            "error_type": type(exc).__name__,
            "error_summary": _error_summary(exc),
        }


async def _run(samples: list[dict[str, Any]], metric_names: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    selected_metrics = tuple(metric_names or RAGAS_METRIC_NAMES)
    unknown = sorted(set(selected_metrics) - set(RAGAS_METRIC_NAMES))
    if unknown:
        raise ValueError(f"Unsupported Ragas metrics: {', '.join(unknown)}")
    provider, api_key, base_url, model, embedding_model = _judge_config()
    request_timeout = _request_timeout()
    metric_timeout = _metric_timeout()
    try:
        deps = _load_ragas_dependencies()
        client = deps["AsyncOpenAI"](
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=0,
        )
        client_calls = _instrument_client(client)
        llm = deps["llm_factory"](model, client=client)
        embeddings = deps["embedding_factory"]("openai", model=embedding_model, client=client)
        metric_classes = deps["metrics"]
        metrics = {
            "context_precision": metric_classes["context_precision"](llm=llm),
            "context_recall": metric_classes["context_recall"](llm=llm),
            "faithfulness": metric_classes["faithfulness"](llm=llm),
            "answer_relevancy": metric_classes["answer_relevancy"](llm=llm, embeddings=embeddings),
            "answer_correctness": metric_classes["answer_correctness"](llm=llm, embeddings=embeddings),
        }
    except Exception as exc:
        return {
            "framework": "ragas",
            "status": "BLOCKED_NOT_VALIDATED",
            "provider": provider,
            "judge_model": model,
            "embedding_model": embedding_model,
            "requested_metrics": list(selected_metrics),
            "case_count": len(samples),
            "metrics": {},
            "metric_summary": {},
            "timings": [],
            "cases": [],
            "provider_error": {
                "status": "PROVIDER_PRIMITIVE_FAILED",
                "error_type": type(exc).__name__,
                "error_summary": _error_summary(exc),
            },
        }

    timings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        for sample in samples:
            case_id = sample.get("case_id", sample.get("id"))
            scores: dict[str, float] = {}
            case_timings: list[dict[str, Any]] = []
            case_started = time.perf_counter()
            for name in selected_metrics:
                started_at = _utc_now()
                started = time.perf_counter()
                timing: dict[str, Any] = {
                    "case_id": case_id,
                    "metric_name": name,
                    "started_at": started_at,
                    "finished_at": None,
                    "latency_sec": None,
                    "status": "METRIC_ERROR",
                    "error_type": None,
                    "error_summary": None,
                    "score": None,
                    "api_calls": [],
                }
                call_start = len(client_calls)
                try:
                    raw = await asyncio.wait_for(
                        metrics[name].ascore(**_metric_kwargs(name, sample)),
                        metric_timeout,
                    )
                    score = float(raw.value)
                    if not math.isfinite(score):
                        raise ValueError("metric score was not finite")
                    scores[name] = score
                    timing["score"] = score
                    timing["status"] = "PASS"
                except asyncio.TimeoutError as exc:
                    timing["status"] = "METRIC_TIMEOUT"
                    timing["error_type"] = type(exc).__name__
                    timing["error_summary"] = f"metric exceeded timeout_sec={metric_timeout}"
                except Exception as exc:
                    timing["status"] = "METRIC_ERROR"
                    timing["error_type"] = type(exc).__name__
                    timing["error_summary"] = _error_summary(exc)
                finally:
                    timing["finished_at"] = _utc_now()
                    timing["latency_sec"] = time.perf_counter() - started
                    timing["api_calls"] = client_calls[call_start:]
                    timings.append(timing)
                    case_timings.append(timing)
            rows.append({
                "id": sample.get("id", case_id),
                "case_id": case_id,
                "scores": scores,
                "latency_sec": time.perf_counter() - case_started,
                "status": "PASS" if len(scores) == len(selected_metrics) else ("PARTIAL" if scores else "BLOCKED_NOT_VALIDATED"),
                "metrics": case_timings,
            })
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            closed = close()
            if hasattr(closed, "__await__"):
                await closed

    metric_summary = _metric_stats(timings, len(samples))
    aggregate = {
        name: metric_summary[name]["score"]
        for name in selected_metrics
        if metric_summary[name]["score"] is not None
    }
    selected_summaries = [metric_summary[name]["status"] for name in selected_metrics]
    if not timings or not any(item.get("status") == "PASS" for item in timings):
        status = "BLOCKED_NOT_VALIDATED"
    elif all(item == "COMPLETE" for item in selected_summaries):
        status = "PASS"
    else:
        status = "PARTIAL"
    return {
        "framework": "ragas",
        "status": status,
        "provider": provider,
        "judge_model": model,
        "embedding_model": embedding_model,
        "base_url_configured": bool(base_url),
        "request_timeout_sec": request_timeout,
        "metric_timeout_sec": metric_timeout,
        "concurrency": 1,
        "requested_metrics": list(selected_metrics),
        "case_count": len(rows),
        "metrics": aggregate,
        "metric_summary": {name: metric_summary[name] for name in selected_metrics},
        "timings": timings,
        "api_calls": client_calls,
        "cases": rows,
    }


def run_ragas_judge(
    samples: list[dict[str, Any]],
    metric_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return asyncio.run(_run(samples, metric_names=metric_names))


__all__ = [
    "GENERATION_METRIC_NAMES",
    "GEMINI_OPENAI_BASE_URL",
    "RAGAS_METRIC_NAMES",
    "_judge_config",
    "run_ragas_judge",
    "run_ragas_provider_smoke",
]
