from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from deploy_ci_cloud_agentv3.mcp.registry import ToolRegistry


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient(Protocol):
    async def list_tools(self) -> list[MCPTool]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class InProcessMCPClient:
    """Registry-only unit-test adapter. Production/local AgentRuntime never uses it."""

    def __init__(self, registry: ToolRegistry, allowed_tools: set[str]) -> None:
        self.registry = registry
        self.allowed_tools = set(allowed_tools)

    async def list_tools(self) -> list[MCPTool]:
        return [MCPTool(item.name, item.description, item.input_schema) for item in self.registry.list(self.allowed_tools)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.allowed_tools:
            raise PermissionError(f"tool is not exposed by this capability profile: {name}")
        return await self.registry.get(name).invoke(arguments)


class OfficialMCPClient:
    """Official MCP Python SDK client for either an MCPServer object or Streamable HTTP URL."""

    def __init__(self, source: Any) -> None:
        self.source = source

    async def list_tools(self) -> list[MCPTool]:
        try:
            from mcp import Client
        except ImportError:
            return await self._list_tools_v1()
        async with Client(self.source) as client:
            result = await client.list_tools()
            tools = getattr(result, "tools", result)
            return [MCPTool(t.name, t.description or "", getattr(t, "input_schema", getattr(t, "inputSchema", {}))) for t in tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import Client
        except ImportError:
            return await self._call_tool_v1(name, arguments)
        async with Client(self.source) as client:
            result = await client.call_tool(name, arguments)
            return self._decode_result(name, result)

    async def _list_tools_v1(self) -> list[MCPTool]:
        if not isinstance(self.source, str):
            raise RuntimeError("Official in-process MCP Client requires MCP SDK v2")
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the official 'mcp' package to use Streamable HTTP") from exc
        async with streamable_http_client(self.source) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return [MCPTool(t.name, t.description or "", getattr(t, "inputSchema", getattr(t, "input_schema", {}))) for t in result.tools]

    async def _call_tool_v1(self, name: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(self.source, str):
            raise RuntimeError("Official in-process MCP Client requires MCP SDK v2")
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the official 'mcp' package to use Streamable HTTP") from exc
        async with streamable_http_client(self.source) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                return self._decode_result(name, result)

    def _decode_result(self, name: str, result: Any) -> Any:
        if getattr(result, "is_error", getattr(result, "isError", False)):
            text = self._text_content(result)
            raise RuntimeError(text or f"MCP tool failed: {name}")
        structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
        if structured is not None:
            return structured
        text = self._text_content(result)
        try:
            return json.loads(text)
        except Exception:
            return {"content": text}

    @staticmethod
    def _text_content(result: Any) -> str:
        chunks: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
