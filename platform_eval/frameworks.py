from __future__ import annotations

import importlib.util
import shutil
from typing import Any


def framework_status() -> dict[str, Any]:
    return {
        "ragas": {
            "available": importlib.util.find_spec("ragas") is not None,
            "recommended_version": "0.4.3",
            "purpose": "RAG LLM-as-judge metrics: Faithfulness, AnswerRelevancy, AnswerCorrectness, ContextPrecision/Recall",
        },
        "deepeval": {
            "available": importlib.util.find_spec("deepeval") is not None,
            "recommended_version": "4.1.8",
            "purpose": "Agent ToolCorrectness/ArgumentCorrectness and trajectory TaskCompletion",
        },
        "promptfoo": {
            "available": shutil.which("promptfoo") is not None or shutil.which("npx") is not None,
            "recommended_version": "latest",
            "purpose": "Dynamic red teaming for prompt injection, excessive agency, MCP/RAG poisoning and secret leakage",
        },
    }
