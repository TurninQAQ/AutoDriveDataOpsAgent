"""Platform transport errors mapped at the Tool Runtime boundary."""

from __future__ import annotations


class MCPPlatformError(RuntimeError):
    def __init__(self, error_code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
