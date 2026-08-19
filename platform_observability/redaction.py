from __future__ import annotations

import re
from typing import Any


_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret|authorization|cookie|session[_-]?key|private[_-]?key)($|[_-])",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]{8,}")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_ASSIGNMENT = re.compile(
    r"(?i)\b(OPENAI_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|X-GOOG-API-KEY|AIRFLOW_API_TOKEN|AIRFLOW_API_PASSWORD|PASSWORD|TOKEN|SECRET)\s*[=:]\s*([^\s,;'\"]+)"
)


REDACTED = "[REDACTED]"


def redact_text(value: str, max_chars: int = 16000) -> str:
    text = _BEARER.sub(f"Bearer {REDACTED}", str(value))
    text = _OPENAI_KEY.sub(REDACTED, text)
    text = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"
    return text


def sanitize(value: Any, *, key: str = "", max_chars: int = 16000, depth: int = 0) -> Any:
    if _SECRET_KEY.search(str(key)):
        return REDACTED
    if depth > 12:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value, max_chars=max_chars)
    if isinstance(value, dict):
        return {
            str(k): sanitize(v, key=str(k), max_chars=max_chars, depth=depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, max_chars=max_chars, depth=depth + 1) for item in value]
    try:
        if hasattr(value, "model_dump"):
            return sanitize(value.model_dump(mode="json"), max_chars=max_chars, depth=depth + 1)
    except Exception:
        pass
    return redact_text(str(value), max_chars=max_chars)
