"""V2-owned platform execution layer.

This package contains transport-independent platform mechanics only.  It is
deliberately separate from the V2 Agent so that the Agent remains the sole
semantic decision authority while the Runtime controls admission and safety.
"""

from .client import InProcessPlatformClient, PlatformBackendError
from .runtime import build_platform_facade

__all__ = [
    "InProcessPlatformClient",
    "PlatformBackendError",
    "build_platform_facade",
]
