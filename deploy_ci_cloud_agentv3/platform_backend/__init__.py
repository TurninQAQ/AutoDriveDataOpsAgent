"""V3 platform execution layer.

The backend contains transport-independent simulated AutoDrive mechanics.
Semantic decisions remain in the V3 Agent; this package only exposes the
deterministic platform facade used by the MCP adapter.
"""

from .runtime import build_platform_facade

__all__ = ["build_platform_facade"]
