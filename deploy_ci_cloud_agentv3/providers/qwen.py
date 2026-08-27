from __future__ import annotations

import json
import os
from typing import Any
import httpx

from deploy_ci_cloud_agentv3.providers.base import AssistantMessage, ToolCall


class QwenProvider:
    """OpenAI-compatible Qwen provider using native function/tool calling."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: float = 90.0) -> None:
        self.api_key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.model = model or os.environ.get("QWEN_MODEL") or "qwen-plus"
        self.timeout = timeout

    async def invoke(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantMessage:
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY or DASHSCOPE_API_KEY is required")
        payload = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto", "parallel_tool_calls": True}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        message = body["choices"][0]["message"]
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            if isinstance(raw_args, str):
                raw_args = json.loads(raw_args)
            calls.append(ToolCall(id=str(item.get("id") or ""), name=str(fn.get("name") or ""), arguments=dict(raw_args)))
        return AssistantMessage(content=str(message.get("content") or ""), tool_calls=calls)
