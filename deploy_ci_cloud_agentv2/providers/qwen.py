"""Qwen-compatible production provider."""

from __future__ import annotations

from .http_structured import HTTPStructuredProvider


class QwenProvider(HTTPStructuredProvider):
    """OpenAI-compatible Qwen/DashScope provider using strict V2 decisions."""

    pass
