"""Backward-compatible Gemini retry names.

The implementation is provider-neutral and owned by ``model_retry``. These
aliases preserve the V1.3 Gemini import surface for existing callers.
"""

from .model_retry import (
    RETRYABLE_STATUS_CODES,
    ModelRequestError,
    ModelRetryPolicy,
    classify_exception,
    retry_async,
    retry_sync,
)


GeminiRetryPolicy = ModelRetryPolicy
GeminiRequestError = ModelRequestError


__all__ = [
    "GeminiRequestError",
    "GeminiRetryPolicy",
    "RETRYABLE_STATUS_CODES",
    "classify_exception",
    "retry_async",
    "retry_sync",
]
