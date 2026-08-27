"""Provider-neutral bounded retry primitives for model operations."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, TypeVar

from deploy_ci_cloud_agentv3.platform_backend.observability.redaction import redact_text


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
STATUS_PATTERN = re.compile(r"(?<!\d)(408|429|500|502|503|504)(?!\d)")


@dataclass(frozen=True)
class ModelRetryPolicy:
    """Bounded retry settings shared by all model providers."""

    attempts: int = 5
    base_sec: float = 1.0
    max_sec: float = 20.0
    jitter_sec: float = 0.25
    request_timeout_sec: float = 45.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ModelRetryPolicy":
        env = environ or os.environ

        def configured(primary: str, legacy: str, default: str) -> str:
            value = env.get(primary)
            if value is None or not str(value).strip():
                value = env.get(legacy)
            return str(value if value is not None else default)

        return cls(
            attempts=max(1, _int_env({"value": configured("PLATFORM_MODEL_RETRY_ATTEMPTS", "PLATFORM_GEMINI_RETRY_ATTEMPTS", "5")}, "value", 5)),
            base_sec=max(0.0, _float_env({"value": configured("PLATFORM_MODEL_RETRY_BASE_SEC", "PLATFORM_GEMINI_RETRY_BASE_SEC", "1")}, "value", 1.0)),
            max_sec=max(0.0, _float_env({"value": configured("PLATFORM_MODEL_RETRY_MAX_SEC", "PLATFORM_GEMINI_RETRY_MAX_SEC", "20")}, "value", 20.0)),
            jitter_sec=max(0.0, _float_env({"value": configured("PLATFORM_MODEL_RETRY_JITTER_SEC", "PLATFORM_GEMINI_RETRY_JITTER_SEC", "0.25")}, "value", 0.25)),
            request_timeout_sec=max(0.001, _float_env(os.environ if environ is None else environ, "PLATFORM_MODEL_REQUEST_TIMEOUT_SEC", 45.0)),
        )


class ModelRequestError(RuntimeError):
    """Safe, classified model failure without prompt or credential content."""

    def __init__(
        self,
        operation: str,
        attempts: int,
        status_code: int | None,
        retryable: bool,
        failure_type: str = "unknown_provider_error",
        provider_error_code: str | None = None,
    ) -> None:
        self.operation = operation
        self.attempts = attempts
        self.status_code = status_code
        self.retryable = retryable
        self.failure_type = failure_type
        self.provider_error_code = provider_error_code
        status = str(status_code) if status_code is not None else "unknown"
        super().__init__(
            f"Model operation {redact_text(operation)} failed after {attempts} attempt(s) "
            f"(status_code={status}, retryable={str(retryable).lower()})"
        )


@dataclass(frozen=True)
class _Failure:
    status_code: int | None
    retry_after_sec: float | None
    retryable: bool
    failure_type: str
    provider_error_code: str | None = None


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(env.get(name, str(default)))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _status_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    return code if 100 <= code <= 599 else None


def _status_from_exception(exc: BaseException) -> int | None:
    candidates: list[Any] = []
    for source in (exc, getattr(exc, "response", None), getattr(exc, "http_response", None)):
        if source is None:
            continue
        for name in ("status_code", "http_status", "status", "code"):
            try:
                candidates.append(getattr(source, name, None))
            except Exception:
                continue
    for candidate in candidates:
        status = _status_code(candidate)
        if status is not None:
            return status
    match = STATUS_PATTERN.search(str(exc))
    return int(match.group(1)) if match else None


def _provider_error_code(exc: BaseException) -> str | None:
    """Keep only a safe provider code, never provider payloads or credentials."""
    for source in (exc, getattr(exc, "response", None), getattr(exc, "http_response", None)):
        if source is None:
            continue
        for name in ("provider_error_code", "error_code", "code"):
            try:
                value = getattr(source, name, None)
            except Exception:
                value = None
            if str(value or "") == "AllocationQuota.FreeTierOnly":
                return "AllocationQuota.FreeTierOnly"
    if "AllocationQuota.FreeTierOnly" in str(exc):
        return "AllocationQuota.FreeTierOnly"
    return None


def _retry_after_value(source: Any) -> float | None:
    if source is None:
        return None
    for name in ("retry_after", "retry_after_sec"):
        try:
            raw = getattr(source, name, None)
        except Exception:
            raw = None
        if raw is not None:
            try:
                value = float(raw)
                if math.isfinite(value) and value >= 0:
                    return value
            except (TypeError, ValueError):
                pass
    for name in ("headers", "response_headers"):
        try:
            headers = getattr(source, name, None)
        except Exception:
            headers = None
        if not isinstance(headers, Mapping):
            continue
        raw = next((value for key, value in headers.items() if str(key).lower() == "retry-after"), None)
        if raw is None:
            continue
        try:
            value = float(raw)
            if math.isfinite(value) and value >= 0:
                return value
        except (TypeError, ValueError):
            try:
                parsed = parsedate_to_datetime(str(raw))
                value = parsed.timestamp() - time.time()
                return max(0.0, value)
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def classify_exception(exc: BaseException) -> _Failure:
    status = _status_from_exception(exc)
    provider_error_code = _provider_error_code(exc)
    retry_after = _retry_after_value(exc)
    if retry_after is None:
        retry_after = _retry_after_value(getattr(exc, "response", None))
    temporary_transport = isinstance(exc, (TimeoutError, ConnectionError, OSError))
    if isinstance(exc, TimeoutError):
        failure_type = "provider_timeout"
    elif provider_error_code == "AllocationQuota.FreeTierOnly":
        failure_type = "provider_quota_error"
    elif status in {401, 403}:
        failure_type = "provider_auth_error"
    elif status == 429:
        failure_type = "provider_rate_limit"
    elif status is not None:
        failure_type = "provider_http_error"
    elif isinstance(exc, (ConnectionError, OSError)):
        failure_type = "provider_connection_error"
    else:
        failure_type = "unknown_provider_error"
    return _Failure(
        status_code=status,
        retry_after_sec=retry_after,
        retryable=status in RETRYABLE_STATUS_CODES or (status is None and temporary_transport),
        failure_type=failure_type,
        provider_error_code=provider_error_code,
    )


def _delay(policy: ModelRetryPolicy, attempt: int, failure: _Failure, jitter: Callable[[float], float]) -> float:
    exponential = min(policy.max_sec, policy.base_sec * (2 ** (attempt - 1)))
    selected = failure.retry_after_sec if failure.retry_after_sec is not None else exponential
    selected = min(policy.max_sec, max(0.0, selected))
    extra = max(0.0, float(jitter(policy.jitter_sec))) if policy.jitter_sec else 0.0
    return selected + extra


def _safe_failure(operation: str, attempt: int, policy: ModelRetryPolicy, exc: BaseException) -> ModelRequestError:
    failure = classify_exception(exc)
    LOGGER.warning(
        "Model operation failed operation=%s attempt=%d status_code=%s retryable=%s",
        redact_text(operation),
        attempt,
        failure.status_code,
        failure.retryable,
    )
    return ModelRequestError(
        operation,
        attempt,
        failure.status_code,
        failure.retryable,
        failure.failure_type,
        failure.provider_error_code,
    )


def retry_sync(
    operation: Callable[[], T],
    *,
    operation_name: str,
    policy: ModelRetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float], float] = lambda amount: random.uniform(0.0, amount),
) -> T:
    active_policy = policy or ModelRetryPolicy.from_env()
    for attempt in range(1, active_policy.attempts + 1):
        try:
            return operation()
        except Exception as exc:
            failure = classify_exception(exc)
            if not failure.retryable or attempt >= active_policy.attempts:
                raise _safe_failure(operation_name, attempt, active_policy, exc) from None
            delay = _delay(active_policy, attempt, failure, jitter)
            LOGGER.warning(
                "Model operation retrying operation=%s attempt=%d status_code=%s retryable=true delay_ms=%d",
                redact_text(operation_name),
                attempt,
                failure.status_code,
                round(delay * 1000),
            )
            sleep(delay)
    raise AssertionError("unreachable")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    policy: ModelRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float], float] = lambda amount: random.uniform(0.0, amount),
    attempt_metrics: MutableMapping[str, int] | None = None,
) -> T:
    active_policy = policy or ModelRetryPolicy.from_env()
    for attempt in range(1, active_policy.attempts + 1):
        if attempt_metrics is not None:
            attempt_metrics["attempts"] = int(attempt_metrics.get("attempts", 0)) + 1
        try:
            result = operation()
            if not inspect.isawaitable(result):
                raise TypeError("async model operation did not return an awaitable")
            return await asyncio.wait_for(result, timeout=active_policy.request_timeout_sec)
        except Exception as exc:
            failure = classify_exception(exc)
            if attempt_metrics is not None:
                attempt_metrics["errors"] = int(attempt_metrics.get("errors", 0)) + 1
                attempt_metrics[failure.failure_type] = int(attempt_metrics.get(failure.failure_type, 0)) + 1
            if not failure.retryable or attempt >= active_policy.attempts:
                raise _safe_failure(operation_name, attempt, active_policy, exc) from None
            if attempt_metrics is not None:
                attempt_metrics["retries"] = int(attempt_metrics.get("retries", 0)) + 1
            delay = _delay(active_policy, attempt, failure, jitter)
            LOGGER.warning(
                "Model operation retrying operation=%s attempt=%d status_code=%s retryable=true delay_ms=%d",
                redact_text(operation_name),
                attempt,
                failure.status_code,
                round(delay * 1000),
            )
            await sleep(delay)
    raise AssertionError("unreachable")


__all__ = [
    "ModelRequestError",
    "ModelRetryPolicy",
    "RETRYABLE_STATUS_CODES",
    "classify_exception",
    "retry_async",
    "retry_sync",
]
