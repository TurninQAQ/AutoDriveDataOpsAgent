"""Platform facades: deterministic offline and concrete MCP transport."""

from .errors import MCPPlatformError
from .facade import InMemoryReadFacade, ReadFacade
from .mcp import MCPPlatformFacade

__all__ = ["InMemoryReadFacade", "ReadFacade", "MCPPlatformFacade", "MCPPlatformError"]
