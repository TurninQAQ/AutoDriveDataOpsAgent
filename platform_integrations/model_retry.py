"""Provider-agnostic bounded retry facade.

The implementation remains in ``gemini_retry`` for V1.3 source compatibility;
this module is the provider-neutral public boundary used by new providers.
"""

from .gemini_retry import (
    GeminiRequestError as ModelRequestError,
    GeminiRetryPolicy as ModelRetryPolicy,
    classify_exception,
    retry_async,
    retry_sync,
)

__all__ = [
    "ModelRequestError",
    "ModelRetryPolicy",
    "classify_exception",
    "retry_async",
    "retry_sync",
]
