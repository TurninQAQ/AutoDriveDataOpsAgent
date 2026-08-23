from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy_ci_cloud_agentv2.platform_backend.rag.evaluation import (
    V1_1_RAG_CASE_COUNT,
    V1_1_RAG_RETRIEVAL_SHA256,
    _case_metrics,
    evaluate_v1_1_retrieval,
    load_v1_1_rag_cases,
)
from deploy_ci_cloud_agentv2.platform_backend.rag.service import KnowledgeService


KNOWLEDGE_ROOT = Path(__file__).parents[2] / "platform_backend" / "knowledge"
DATASET = KNOWLEDGE_ROOT / "eval" / "v1_1" / "rag_retrieval.jsonl"


def test_canonical_v1_1_dataset_is_30_cases_and_source_validated():
    cases = load_v1_1_rag_cases(DATASET, source_dir=KNOWLEDGE_ROOT)
    assert len(cases) == V1_1_RAG_CASE_COUNT == 30
    assert len({case["id"] for case in cases}) == 30
    assert {case["category"] for case in cases} == {
        "gpu",
        "priority",
        "preemption",
        "recovery",
        "stage",
        "docker",
        "airflow",
        "architecture",
        "grounding",
        "security",
    }


def test_canonical_dataset_hash_is_locked_and_changes_are_rejected(tmp_path):
    copied = tmp_path / "rag_retrieval.jsonl"
    copied.write_bytes(DATASET.read_bytes())
    assert __import__("hashlib").sha256(copied.read_bytes()).hexdigest() == V1_1_RAG_RETRIEVAL_SHA256
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_v1_1_rag_cases(copied)


def test_canonical_metric_example_uses_padded_precision_and_ndcg():
    item = SimpleNamespace(
        source_path="platform/gpu_scheduling.md",
        section="Reservation",
        metadata={"chunk_index": 0},
    )
    metrics = _case_metrics([item], ["platform/gpu_scheduling.md#Reservation"], 5)
    assert metrics["first_relevant_rank"] == 1
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.2
    assert metrics["context_precision"] == 1.0
    assert metrics["ndcg"] == 1.0


def test_local_30_case_baseline_matches_historical_v1_1_alignment(tmp_path):
    result = evaluate_v1_1_retrieval(
        KnowledgeService(KNOWLEDGE_ROOT, tmp_path / "index.json"),
        DATASET,
    )
    assert result["case_count"] == 30
    assert result["hit_at_1"] == pytest.approx(0.6666666667)
    assert result["hit_at_3"] == pytest.approx(0.8333333333)
    assert result["hit_at_5"] == pytest.approx(0.8666666667)
    assert result["mrr"] == pytest.approx(0.7416666667)
    assert result["ndcg_at_5"] == pytest.approx(0.7464999292)
    assert result["recall_at_5"] == pytest.approx(0.8166666667)
    assert result["context_precision_at_5"] == pytest.approx(0.7416666667)


def test_dataset_is_jsonl_objects_not_legacy_runtime_code():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    assert all(isinstance(row, dict) for row in rows)
    assert not any("platform_agent" in json.dumps(row) for row in rows)
