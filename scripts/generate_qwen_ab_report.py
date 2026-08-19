from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "local_acceptance" / "v1.3.1_after"


def first_relevant(row: dict) -> int | None:
    relevant = set(row.get("reference_context_ids") or [])
    for rank, context_id in enumerate(row.get("retrieved_context_ids") or [], 1):
        if context_id in relevant:
            return rank
    return None


def classification(hash_row: dict, qwen_row: dict) -> str:
    keys = ("context_recall", "context_precision", "mrr", "ndcg_at_k")
    hash_metrics = hash_row["metrics"]
    qwen_metrics = qwen_row["metrics"]
    comparisons = [qwen_metrics[key] - hash_metrics[key] for key in keys]
    if all(abs(delta) < 1e-12 for delta in comparisons):
        return "unchanged"
    if all(delta >= -1e-12 for delta in comparisons) and any(delta > 1e-12 for delta in comparisons):
        return "improved"
    return "regressed"


def compact_contexts(contexts: list[str]) -> str:
    return "<br>".join(contexts) if contexts else "-"


def main() -> None:
    hash_eval = json.loads((ACCEPTANCE / "hash_eval.json").read_text(encoding="utf-8"))
    qwen_eval = json.loads((ACCEPTANCE / "qwen_eval.json").read_text(encoding="utf-8"))
    hash_metrics = hash_eval["rag"]["metrics"]
    qwen_metrics = qwen_eval["rag"]["metrics"]
    lines = [
        "# V1.3.1 Hash vs Qwen Golden A/B",
        "",
        "本报告使用同一份未修改的 V1.1 Golden Set、top-k=5、BM25/lexical=0.50、dense=0.50、Qwen 1024 维、instruct disabled、无 reranker。`improved/regressed` 按每 case 的 Context Recall、Context Precision、MRR、nDCG 逐项比较：Qwen 无回退且至少一项提高为 improved；出现任一回退为 regressed；全部相同为 unchanged。",
        "",
        "## Overall",
        "",
        "| Metric | Hash | Qwen | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Context Recall", "context_recall"), ("Context Precision", "context_precision"), ("MRR", "mrr"), ("nDCG", "ndcg_at_k")):
        delta = qwen_metrics[key] - hash_metrics[key]
        lines.append(f"| {label} | {hash_metrics[key]:.6f} | {qwen_metrics[key]:.6f} | {delta:+.6f} |")
    lines.extend([
        "",
        f"- Hash gate: `{hash_eval['passed']}`; Qwen gate: `{qwen_eval['passed']}`.",
        f"- Qwen case classification: improved={sum(classification(h, q) == 'improved' for h, q in zip(hash_eval['rag']['cases'], qwen_eval['rag']['cases']))}, unchanged={sum(classification(h, q) == 'unchanged' for h, q in zip(hash_eval['rag']['cases'], qwen_eval['rag']['cases']))}, regressed={sum(classification(h, q) == 'regressed' for h, q in zip(hash_eval['rag']['cases'], qwen_eval['rag']['cases']))}.",
        "- 这是一轮客观基线记录；未据此调权重、增加 instruct 或引入 reranker。",
        "",
        "## Per case",
        "",
        "| Case | Hash top-k | Qwen top-k | Hash first relevant | Qwen first relevant | Result |",
        "|---|---|---|---:|---:|---|",
    ])
    for hash_row, qwen_row in zip(hash_eval["rag"]["cases"], qwen_eval["rag"]["cases"]):
        hash_first = first_relevant(hash_row)
        qwen_first = first_relevant(qwen_row)
        lines.append(
            f"| `{hash_row['id']}` | {compact_contexts(hash_row.get('retrieved_context_ids') or [])} | {compact_contexts(qwen_row.get('retrieved_context_ids') or [])} | {hash_first if hash_first is not None else '-'} | {qwen_first if qwen_first is not None else '-'} | {classification(hash_row, qwen_row)} |"
        )
    (ACCEPTANCE / "rag_hash_vs_qwen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
