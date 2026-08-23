from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from .service import KnowledgeService


V1_1_RAG_CASE_COUNT = 30
V1_1_RAG_RETRIEVAL_SHA256 = "7cb32e25fbb35a62274732558ed00f42aa98f20c871c7281127247efcb19f7ed"
V1_1_RAG_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "tier",
        "category",
        "query",
        "top_k",
        "reference_context_ids",
        "reference_answer",
        "required_facts",
    }
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("RAG eval file must contain a JSON list")
    return payload


def _validate_context_id(context_id: object, *, case_id: str) -> str:
    if not isinstance(context_id, str) or not context_id.strip():
        raise ValueError(f"case {case_id!r} has an invalid reference_context_ids item")
    value = context_id.strip()
    if "#" not in value:
        raise ValueError(f"case {case_id!r} context must use source#section: {value!r}")
    source, section = value.split("#", 1)
    if not source or not section:
        raise ValueError(f"case {case_id!r} has an incomplete context id: {value!r}")
    if "::chunk" in section:
        base, chunk = section.rsplit("::chunk", 1)
        if not base or not chunk.isdigit():
            raise ValueError(f"case {case_id!r} has an invalid chunk context id: {value!r}")
    return value


def load_v1_1_rag_cases(path: Path, *, source_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load and integrity-check the canonical V1.1 chunk-level RAG set.

    This validator owns dataset integrity and schema only.  Ranking remains in
    ``KnowledgeService``/``HybridRetriever`` so evaluation cannot drift into a
    second retrieval implementation.
    """

    path = Path(path)
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != V1_1_RAG_RETRIEVAL_SHA256:
        raise ValueError(
            "canonical V1.1 RAG dataset hash mismatch; labels must not be tuned silently"
        )
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"canonical V1.1 RAG dataset has a blank line at {line_number}")
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at canonical RAG line {line_number}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"canonical RAG line {line_number} must be an object")
        missing = V1_1_RAG_REQUIRED_FIELDS - set(case)
        if missing:
            raise ValueError(f"canonical RAG line {line_number} missing fields: {sorted(missing)}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError(f"canonical RAG line {line_number} has a missing/duplicate id")
        if not isinstance(case["query"], str) or not case["query"].strip():
            raise ValueError(f"canonical RAG case {case_id!r} has an empty query")
        if not isinstance(case["category"], str) or not case["category"].strip():
            raise ValueError(f"canonical RAG case {case_id!r} has no category")
        if not isinstance(case["tier"], str) or not case["tier"].strip():
            raise ValueError(f"canonical RAG case {case_id!r} has no tier")
        if not isinstance(case["top_k"], int) or case["top_k"] <= 0:
            raise ValueError(f"canonical RAG case {case_id!r} has an invalid top_k")
        contexts = case["reference_context_ids"]
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"canonical RAG case {case_id!r} has no reference contexts")
        case["reference_context_ids"] = [
            _validate_context_id(context, case_id=case_id) for context in contexts
        ]
        if not isinstance(case["reference_answer"], str) or not case["reference_answer"].strip():
            raise ValueError(f"canonical RAG case {case_id!r} has no reference answer")
        facts = case["required_facts"]
        if not isinstance(facts, list) or not facts or not all(
            isinstance(fact, str) and fact.strip() for fact in facts
        ):
            raise ValueError(f"canonical RAG case {case_id!r} has invalid required_facts")
        ids.add(case_id)
        rows.append(case)
    if len(rows) != V1_1_RAG_CASE_COUNT:
        raise ValueError(
            f"canonical V1.1 RAG case count mismatch: expected {V1_1_RAG_CASE_COUNT}, got {len(rows)}"
        )

    if source_dir is not None:
        source_dir = Path(source_dir)
        chunks = KnowledgeService(
            source_dir=source_dir,
            index_file=source_dir / ".evaluation-index.json",
        ).loader.load()
        available = {chunk.source_path for chunk in chunks}
        context_keys = {f"{chunk.source_path}#{chunk.section}" for chunk in chunks}
        for case in rows:
            for context_id in case["reference_context_ids"]:
                source, section = context_id.split("#", 1)
                if source not in available:
                    raise ValueError(f"canonical RAG references missing source: {source}")
                base_section = section.split("::chunk", 1)[0]
                if f"{source}#{base_section}" not in context_keys:
                    raise ValueError(f"canonical RAG references missing section: {context_id}")
    return rows


def _retrieved_context_identity(item: Any) -> tuple[str, int]:
    base = f"{item.source_path}#{item.section}" if item.section else item.source_path
    chunk_index = int(item.metadata.get("chunk_index", 0) or 0)
    return base, chunk_index


def _retrieved_context_id(item: Any) -> str:
    base, chunk_index = _retrieved_context_identity(item)
    return base if chunk_index == 0 else f"{base}::chunk{chunk_index}"


def _context_matches(expected: str, item: Any) -> bool:
    if "::chunk" in expected:
        expected_base, chunk_text = expected.rsplit("::chunk", 1)
        try:
            expected_chunk = int(chunk_text)
        except ValueError:
            return False
    else:
        expected_base = expected
        expected_chunk = 0
    actual_base, actual_chunk = _retrieved_context_identity(item)
    return expected_base == actual_base and expected_chunk == actual_chunk


def _case_metrics(ranked: list[Any], expected: list[str], top_k: int) -> dict[str, Any]:
    expected_ids = list(dict.fromkeys(expected))
    relevance = [
        1 if any(_context_matches(target, item) for target in expected_ids) else 0
        for item in ranked
    ]
    first_rank = next(
        (rank for rank, relevant in enumerate(relevance[:top_k], 1) if relevant),
        None,
    )
    padded = relevance[:top_k] + [0] * max(0, top_k - len(relevance))
    relevant_at_k = sum(padded)
    dcg = sum(
        relevant / math.log2(rank + 1)
        for rank, relevant in enumerate(relevance[:top_k], 1)
    )
    ideal_count = min(len(expected_ids), top_k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    context_precision_hits = 0
    context_precision_total = 0.0
    for rank, relevant in enumerate(relevance[:top_k], 1):
        if relevant:
            context_precision_hits += 1
            context_precision_total += sum(relevance[:rank]) / rank
    return {
        "first_relevant_rank": first_rank,
        "hit": first_rank is not None,
        "relevant_count": relevant_at_k,
        "recall": relevant_at_k / len(expected_ids) if expected_ids else 1.0,
        "precision": sum(padded) / top_k if top_k > 0 else 0.0,
        "context_precision": (
            context_precision_total / context_precision_hits
            if context_precision_hits
            else 0.0
        ),
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def evaluate_v1_1_retrieval(service: KnowledgeService, cases_path: Path) -> dict[str, Any]:
    """Evaluate the canonical 30-case set using the production retriever."""

    cases = load_v1_1_rag_cases(cases_path, source_dir=service.source_dir)
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = service.search(case["query"], top_k=case["top_k"])
        ranked = result.results
        expected = case["reference_context_ids"]
        metrics = _case_metrics(ranked, expected, case["top_k"])
        rows.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "tier": case["tier"],
                "query": case["query"],
                "expected_context_ids": expected,
                "ranked_context_ids": [_retrieved_context_id(item) for item in ranked],
                **metrics,
            }
        )
    count = len(rows)
    return {
        "case_count": count,
        "hit_at_1": sum(row["first_relevant_rank"] == 1 for row in rows) / count,
        "hit_at_3": sum(row["hit"] and row["first_relevant_rank"] <= 3 for row in rows) / count,
        "hit_at_5": sum(row["hit"] and row["first_relevant_rank"] <= 5 for row in rows) / count,
        "mrr": sum(1.0 / row["first_relevant_rank"] for row in rows if row["first_relevant_rank"]) / count,
        "ndcg_at_3": sum(_case_metrics_from_row(row, 3) for row in rows) / count,
        "ndcg_at_5": sum(_case_metrics_from_row(row, 5) for row in rows) / count,
        "recall_at_5": sum(row["recall"] for row in rows) / count,
        "precision_at_5": sum(row["precision"] for row in rows) / count,
        "context_precision_at_5": sum(row["context_precision"] for row in rows) / count,
        "cases": rows,
    }


def _case_metrics_from_row(row: dict[str, Any], top_k: int) -> float:
    ranked = row["ranked_context_ids"][:top_k]
    expected = row["expected_context_ids"]
    dcg = sum(
        int(any(context == target for target in expected)) / math.log2(index + 2)
        for index, context in enumerate(ranked)
    )
    ideal_dcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(min(len(set(expected)), top_k))
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(service: KnowledgeService, cases_path: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    rows = []
    hits = 0
    reciprocal_rank_total = 0.0
    for case in cases:
        top_k = int(case.get("top_k", 5))
        expected = set(case.get("expected_sources") or [])
        result = service.search(str(case.get("query", "")), top_k=top_k)
        sources = [item.source_path for item in result.results]
        rank = None
        for idx, source in enumerate(sources, 1):
            if source in expected:
                rank = idx
                break
        hit = rank is not None
        if hit:
            hits += 1
            reciprocal_rank_total += 1.0 / rank
        rows.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "expected_sources": sorted(expected),
                "retrieved_sources": sources,
                "hit": hit,
                "first_relevant_rank": rank,
            }
        )
    count = len(cases)
    return {
        "case_count": count,
        "hit_at_k": (hits / count) if count else 0.0,
        "mrr": (reciprocal_rank_total / count) if count else 0.0,
        "cases": rows,
    }
