from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service import KnowledgeService


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("RAG eval file must contain a JSON list")
    return payload


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
