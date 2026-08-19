from __future__ import annotations

import asyncio
import os
from typing import Any

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _require(value: str, message: str) -> str:
    value = value.strip()
    if not value:
        raise RuntimeError(message)
    return value


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
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        api_key = _require(os.getenv("OPENAI_API_KEY", ""), "OPENAI_API_KEY is required for OpenAI Ragas evaluation")
        base_url = os.getenv("PLATFORM_EVAL_JUDGE_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip() or None
        model = os.getenv("PLATFORM_EVAL_JUDGE_MODEL", "").strip() or "gpt-5-mini"
        embedding_model = os.getenv("PLATFORM_EVAL_EMBED_MODEL", "").strip() or "text-embedding-3-small"
        return provider, api_key, base_url, model, embedding_model
    raise RuntimeError(f"Unsupported PLATFORM_EVAL_PROVIDER: {provider}")


async def _run(samples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
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

    provider, api_key, base_url, model, embedding_model = _judge_config()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    llm = llm_factory(model, client=client)
    # Gemini's official OpenAI-compat endpoint exposes /embeddings, so the same
    # Ragas OpenAI adapter can drive gemini-embedding-2 without a separate judge stack.
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    metrics = {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }
    rows = []
    totals = {name: 0.0 for name in metrics}
    counts = {name: 0 for name in metrics}
    for sample in samples:
        user_input = str(sample["user_input"])
        response = str(sample["response"])
        reference = str(sample["reference"])
        contexts = list(sample.get("retrieved_contexts") or [])
        scores: dict[str, float] = {}
        for name, metric in metrics.items():
            kwargs = {"user_input": user_input, "response": response, "reference": reference, "retrieved_contexts": contexts}
            if name == "faithfulness":
                kwargs = {"user_input": user_input, "response": response, "retrieved_contexts": contexts}
            elif name == "answer_relevancy":
                kwargs = {"user_input": user_input, "response": response}
            elif name == "answer_correctness":
                kwargs = {"user_input": user_input, "response": response, "reference": reference}
            elif name == "context_recall":
                kwargs = {"user_input": user_input, "retrieved_contexts": contexts, "reference": reference}
            elif name == "context_precision":
                kwargs = {"user_input": user_input, "retrieved_contexts": contexts, "reference": reference}
            result = await metric.ascore(**kwargs)
            value = float(result.value)
            scores[name] = value
            totals[name] += value
            counts[name] += 1
        rows.append({"id": sample.get("id"), "scores": scores})
    aggregate = {name: (totals[name] / counts[name] if counts[name] else 0.0) for name in metrics}
    return {
        "framework": "ragas",
        "provider": provider,
        "judge_model": model,
        "embedding_model": embedding_model,
        "case_count": len(rows),
        "metrics": aggregate,
        "cases": rows,
    }


def run_ragas_judge(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return asyncio.run(_run(samples))
