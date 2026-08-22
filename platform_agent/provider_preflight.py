"""Small, non-evaluation provider preflight for structured Qwen requests."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from platform_integrations.model_retry import classify_exception


FREE_TIER_QUOTA_CODE = "AllocationQuota.FreeTierOnly"


def is_free_tier_quota_block(failure_or_exc: Any) -> bool:
    """Recognize the normalized quota identity, with a safe legacy fallback."""
    for name in ("provider_error_code", "error_code", "code"):
        try:
            if str(getattr(failure_or_exc, name, "") or "") == FREE_TIER_QUOTA_CODE:
                return True
        except Exception:
            continue
    if str(getattr(failure_or_exc, "failure_type", "") or "") == "provider_quota_error":
        return True
    return FREE_TIER_QUOTA_CODE in str(failure_or_exc)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(getattr(item, "text", ""))
            for item in content
        )
    return str(content or "")


def classify_provider_failure(exc: BaseException) -> str:
    """Classify a provider failure without exposing response bodies or secrets."""

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "provider_timeout"
    failure = classify_exception(exc)
    if is_free_tier_quota_block(failure) or is_free_tier_quota_block(exc):
        return "provider_quota_error"
    status = failure.status_code
    if status in {401, 403} or (status is None and "403" in str(exc)):
        return "provider_auth_error"
    if status == 429:
        return "provider_rate_limit"
    if status is not None:
        return "provider_http_error"
    if isinstance(exc, (ConnectionError, OSError)):
        return "provider_connection_error"
    return "unknown_provider_error"


@dataclass
class ProviderPreflightResult:
    status: str
    model: str
    checks_requested: int
    requests_attempted: int = 0
    requests_completed: int = 0
    timeout_count: int = 0
    error_count: int = 0
    failure_types: list[str] = field(default_factory=list)
    latencies_sec: list[float] = field(default_factory=list)
    http_statuses: list[int] = field(default_factory=list)
    quota_blocked: bool = False
    terminal: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "checks_requested": self.checks_requested,
            "requests_attempted": self.requests_attempted,
            "requests_completed": self.requests_completed,
            "timeout_count": self.timeout_count,
            "error_count": self.error_count,
            "failure_types": list(self.failure_types),
            "latencies_sec": [round(item, 4) for item in self.latencies_sec],
            "http_statuses": list(self.http_statuses),
            "quota_blocked": self.quota_blocked,
            "terminal": self.terminal,
        }


async def run_qwen_preflight(
    client: Any,
    *,
    model: str = "qwen-plus",
    checks: int = 1,
    timeout_sec: float = 15.0,
) -> ProviderPreflightResult:
    """Verify credentials/endpoint/structured JSON using synthetic requests only."""

    checks = max(1, int(checks))
    timeout_sec = max(0.001, float(timeout_sec))
    result = ProviderPreflightResult(status="FAIL", model=model, checks_requested=checks)
    for _ in range(checks):
        started = time.perf_counter()
        result.requests_attempted += 1
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Provider preflight only. Return exactly the JSON object {\"ok\":true}.",
                        },
                        {"role": "user", "content": "Return {\"ok\":true}."},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                ),
                timeout=timeout_sec,
            )
            payload = json.loads(_response_text(response))
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                raise ValueError("provider returned invalid preflight JSON")
            result.requests_completed += 1
            result.latencies_sec.append(time.perf_counter() - started)
        except ValueError:
            result.error_count += 1
            result.failure_types.append("provider_invalid_json")
        except Exception as exc:
            failure_info = classify_exception(exc)
            failure_type = classify_provider_failure(exc)
            result.error_count += 1
            result.failure_types.append(failure_type)
            if failure_info.status_code is not None:
                result.http_statuses.append(int(failure_info.status_code))
            quota_block = is_free_tier_quota_block(failure_info) or is_free_tier_quota_block(exc)
            generic_403 = failure_info.status_code == 403 or "403" in str(exc)
            if quota_block:
                result.quota_blocked = True
            if quota_block or generic_403:
                result.terminal = True
                break
            if failure_type == "provider_timeout":
                result.timeout_count += 1
    result.status = "PASS" if result.requests_completed == checks else "FAIL"
    return result


__all__ = ["ProviderPreflightResult", "classify_provider_failure", "run_qwen_preflight"]
