from __future__ import annotations

import json
import os
from pathlib import Path

from platform_agent.model import build_model_from_env
from platform_agent.settings import AgentSettings
from platform_core.settings import PlatformSettings
from platform_eval.aligned import evaluate_v11_suite
from platform_rag.service import KnowledgeService
from platform_agent.runtime import build_knowledge_service


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_acceptance" / "v1.3.1_after"
EVAL = ROOT / "eval" / "v1_1"


def evaluate(provider: str) -> dict:
    if provider == "hash":
        os.environ["PLATFORM_RAG_EMBED_PROVIDER"] = "hash"
        os.environ["PLATFORM_RAG_EMBED_INDEX"] = ""
    else:
        os.environ["PLATFORM_RAG_EMBED_PROVIDER"] = "qwen"
        os.environ["PLATFORM_RAG_EMBED_MODEL"] = "qwen3.7-text-embedding"
        os.environ["PLATFORM_RAG_EMBED_DIM"] = "1024"
        os.environ["PLATFORM_RAG_EMBED_BATCH_SIZE"] = "20"
        os.environ["PLATFORM_RAG_EMBED_INDEX"] = "/home/ubuntu/project/autodrive_dataops_runtime/state/agent_knowledge/embeddings_qwen.json"
    settings = AgentSettings.from_env(PlatformSettings.from_env())
    result = evaluate_v11_suite(
        knowledge_service=build_knowledge_service(settings),
        rag_cases=EVAL / "rag_retrieval.jsonl",
        tool_cases=EVAL / "agent_tool_cases.jsonl",
        task_cases=EVAL / "agent_task_cases.jsonl",
        security_cases=EVAL / "security" / "curated_attacks.jsonl",
        planning_cases=ROOT / "eval" / "task_planning_cases.json",
    )
    thresholds_path = EVAL / "thresholds.json"
    result["provider"] = provider
    result["model"] = settings.knowledge_embedding_model if provider == "qwen" else "hash"
    result["embedding_dimension"] = settings.knowledge_embedding_dimension if provider == "qwen" else None
    result["embedding_batch_size"] = settings.knowledge_embedding_batch_size if provider == "qwen" else None
    result["thresholds"] = json.loads(thresholds_path.read_text(encoding="utf-8"))
    limits = result["thresholds"]
    gates = result["gates"]
    result["passed"] = (
        gates["rag_context_recall"] >= limits["rag_context_recall_min"]
        and gates["rag_context_precision"] >= limits["rag_context_precision_min"]
        and gates["tool_f1"] >= limits["tool_f1_min"]
        and gates["argument_accuracy"] >= limits["argument_accuracy_min"]
        and gates["hard_task_success_rate"] >= limits["hard_task_success_rate_min"]
        and gates["task_planning_accuracy"] >= limits["task_planning_accuracy_min"]
        and gates["security_attack_success_rate"] <= limits["security_attack_success_rate_max"]
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for provider in ("hash", "qwen"):
        result = evaluate(provider)
        (OUT / f"{provider}_eval.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
