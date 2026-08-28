from __future__ import annotations

from typing import Any

_SENSITIVE = ("authorization", "api_key", "apikey", "secret", "token", "password", "credential")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            out[str(key)] = "[REDACTED]" if any(word in lowered for word in _SENSITIVE) else redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value
