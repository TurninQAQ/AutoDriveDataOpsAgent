"""Evaluation-only model invocation telemetry.

The collector records these facts; scoring remains in ``metrics.py``.  This
wrapper never stores prompts, responses, credentials, or authorization data.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Mapping


def _number(source: Any, *names: str) -> int | None:
    if source is None:
        return None
    for name in names:
        value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass
class ModelTelemetry:
    llm_call_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_requests: int = 0
    provider_errors: int = 0
    llm_latency_ms_total: float = 0.0
    provider_error_codes: list[str] = field(default_factory=list)
    _last_attempts: int = 0
    _last_errors: int = 0
    _last_input_tokens: int = 0
    _last_output_tokens: int = 0
    _last_total_tokens: int = 0

    @staticmethod
    def _metrics(base: Any) -> Mapping[str, Any]:
        value = getattr(base, "metrics", None)
        return value if isinstance(value, Mapping) else {}

    def _usage_from(self, base: Any, result: Any) -> tuple[int | None, int | None, int | None]:
        usage = getattr(result, "usage", None)
        input_value = _number(usage, "prompt_tokens", "input_tokens")
        output_value = _number(usage, "completion_tokens", "output_tokens")
        total_value = _number(usage, "total_tokens")
        metrics = self._metrics(base)
        metric_input = _number(metrics, "input_tokens", "prompt_tokens")
        metric_output = _number(metrics, "output_tokens", "completion_tokens")
        metric_total = _number(metrics, "total_tokens")
        if metric_input is not None:
            input_value = max(0, metric_input - self._last_input_tokens)
            self._last_input_tokens = metric_input
        if metric_output is not None:
            output_value = max(0, metric_output - self._last_output_tokens)
            self._last_output_tokens = metric_output
        if metric_total is not None:
            total_value = max(0, metric_total - self._last_total_tokens)
            self._last_total_tokens = metric_total
        return input_value, output_value, total_value

    def record(self, base: Any, result: Any = None, error: BaseException | None = None, elapsed_ms: float = 0.0) -> None:
        metrics = self._metrics(base)
        attempts = _number(metrics, "attempts")
        errors = _number(metrics, "errors")
        calls = max(1, (attempts - self._last_attempts) if attempts is not None else 1)
        self._last_attempts = attempts if attempts is not None else self._last_attempts
        self.llm_call_count += calls
        self.provider_requests += calls
        if errors is not None:
            self.provider_errors += max(0, errors - self._last_errors)
            self._last_errors = errors
        elif error is not None:
            self.provider_errors += 1
        if error is not None:
            code = getattr(error, "provider_error_code", None) or getattr(error, "error_code", None)
            if code and str(code) not in self.provider_error_codes:
                self.provider_error_codes.append(str(code))
        input_value, output_value, total_value = self._usage_from(base, result)
        if input_value is not None:
            self.input_tokens = (self.input_tokens or 0) + input_value
        if output_value is not None:
            self.output_tokens = (self.output_tokens or 0) + output_value
        if total_value is not None:
            self.total_tokens = (self.total_tokens or 0) + total_value
        elif self.input_tokens is not None and self.output_tokens is not None:
            self.total_tokens = self.input_tokens + self.output_tokens
        self.llm_latency_ms_total += max(0.0, float(elapsed_ms))

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_call_count": self.llm_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "provider_requests": self.provider_requests,
            "provider_errors": self.provider_errors,
            "llm_latency_ms_total": round(self.llm_latency_ms_total, 3),
            "provider_error_codes": list(self.provider_error_codes),
        }


class InstrumentedModelClient:
    """Transparent model-protocol wrapper for one scenario attempt."""

    def __init__(self, base: Any, telemetry: ModelTelemetry):
        self._base = base
        self.telemetry = telemetry
        self.requires_tool_descriptions = getattr(base, "requires_tool_descriptions", True)
        self.supports_adaptive = getattr(base, "supports_adaptive", False)

    async def _invoke(self, name: str, *args, **kwargs):
        method = getattr(self._base, name)
        started = time.perf_counter()
        try:
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            self.telemetry.record(self._base, result=result, elapsed_ms=(time.perf_counter() - started) * 1000)
            return result
        except Exception as exc:
            self.telemetry.record(self._base, error=exc, elapsed_ms=(time.perf_counter() - started) * 1000)
            raise

    async def plan(self, *args, **kwargs):
        return await self._invoke("plan", *args, **kwargs)

    async def synthesize(self, *args, **kwargs):
        return await self._invoke("synthesize", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "decide_next":
            # Preserve the production capability probe: a model without an
            # adaptive method must not look adaptive merely because it is
            # wrapped for telemetry.
            method = getattr(self._base, name)
            if not callable(method):
                raise AttributeError(name)

            async def invoke(*args, **kwargs):
                return await self._invoke(name, *args, **kwargs)

            return invoke
        return getattr(self._base, name)


__all__ = ["InstrumentedModelClient", "ModelTelemetry"]
