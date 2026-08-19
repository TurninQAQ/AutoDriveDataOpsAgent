from __future__ import annotations

import asyncio
import json
import os

from platform_agent.model import HeuristicReadOnlyModel, build_model_from_env
from platform_mcp.server import WRITE_TOOL_NAMES


def _model():
    if os.getenv("PLATFORM_PROMPTFOO_REAL_MODEL", "0").strip() in {"1", "true", "yes"}:
        return build_model_from_env(
            os.getenv("PLATFORM_AGENT_PROVIDER", "openai"),
            os.getenv("PLATFORM_AGENT_MODEL", "gpt-5-mini"),
            float(os.getenv("PLATFORM_AGENT_TEMPERATURE", "0")),
            os.getenv("OPENAI_BASE_URL", "") or None,
        )
    return HeuristicReadOnlyModel()


async def _plan(prompt: str):
    model = _model()
    return await model.plan(prompt, [], [])


def call_api(prompt, options, context):
    del options, context
    plan = asyncio.run(_plan(str(prompt)))
    direct_write_tools = [call.name for call in plan.tool_calls if call.name in WRITE_TOOL_NAMES]
    payload = {
        "intent": plan.intent.value,
        "tools": [call.model_dump(mode="json") for call in plan.tool_calls],
        "direct_write_tools": direct_write_tools,
        "has_write_action": bool(plan.write_action),
        "write_action": plan.write_action,
        "decision_summary": plan.decision_summary,
    }
    return {"output": json.dumps(payload, ensure_ascii=False)}
