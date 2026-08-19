from __future__ import annotations

from typing import Any, Protocol

from platform_mcp.facade import PlatformMCPFacade
from platform_mcp.server import READ_ONLY_TOOL_NAMES, build_mcp_server

from .models import ToolCallSpec, ToolObservation


class PlatformToolClient(Protocol):
    async def describe_tools(self) -> list[dict[str, Any]]:
        ...

    async def execute(self, calls: list[ToolCallSpec]) -> list[ToolObservation]:
        ...


class InMemoryMCPToolClient:
    """Use the official MCP v2 Client against the Platform MCP server in-process.

    This keeps the Agent behind the MCP protocol boundary without requiring a
    subprocess or HTTP server. The MCP SDK is imported lazily so core tests can
    inject a fake client in dependency-limited environments.
    """

    def __init__(self, facade: PlatformMCPFacade | None = None):
        self.facade = facade

    def _server(self):
        return build_mcp_server(self.facade, include_write_tools=True)

    @staticmethod
    def _import_client():
        try:
            from mcp import Client
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "MCP Python SDK is not installed. Install requirements-mcp.txt first."
            ) from exc
        return Client

    async def describe_tools(self) -> list[dict[str, Any]]:
        Client = self._import_client()
        server = self._server()
        async with Client(server) as client:
            page = await client.list_tools()
            return [
                {
                    "name": tool.name,
                    "title": getattr(tool, "title", None),
                    "description": tool.description or "",
                    "input_schema": tool.input_schema,
                }
                for tool in page.tools
                if tool.name in READ_ONLY_TOOL_NAMES
            ]

    async def execute(self, calls: list[ToolCallSpec]) -> list[ToolObservation]:
        if not calls:
            return []
        Client = self._import_client()
        server = self._server()
        observations: list[ToolObservation] = []
        async with Client(server) as client:
            for call in calls:
                try:
                    result = await client.call_tool(call.name, call.arguments)
                    if result.is_error:
                        text = "\n".join(
                            str(getattr(block, "text", ""))
                            for block in result.content
                            if getattr(block, "text", None)
                        ).strip()
                        observations.append(
                            ToolObservation(
                                tool_name=call.name,
                                arguments=call.arguments,
                                ok=False,
                                error=text or f"MCP tool returned error: {call.name}",
                            )
                        )
                    else:
                        observations.append(
                            ToolObservation(
                                tool_name=call.name,
                                arguments=call.arguments,
                                ok=True,
                                data=result.structured_content,
                            )
                        )
                except Exception as exc:
                    observations.append(
                        ToolObservation(
                            tool_name=call.name,
                            arguments=call.arguments,
                            ok=False,
                            error=str(exc),
                        )
                    )
        return observations


class FacadeToolClient:
    """Dependency-light adapter used by unit tests and emergency local fallback.

    Production/default Agent runtime uses InMemoryMCPToolClient. This adapter keeps
    the exact V0.3 tool names/contracts while avoiding the MCP dependency in tests.
    """

    def __init__(self, facade: PlatformMCPFacade):
        self.facade = facade

    async def describe_tools(self) -> list[dict[str, Any]]:
        return [{"name": name, "description": name, "input_schema": {}} for name in READ_ONLY_TOOL_NAMES]

    async def execute(self, calls: list[ToolCallSpec]) -> list[ToolObservation]:
        results: list[ToolObservation] = []
        for call in calls:
            try:
                fn = getattr(self.facade, call.name)
                data = fn(**call.arguments)
                results.append(
                    ToolObservation(
                        tool_name=call.name,
                        arguments=call.arguments,
                        ok=True,
                        data=data,
                    )
                )
            except Exception as exc:
                results.append(
                    ToolObservation(
                        tool_name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=str(exc),
                    )
                )
        return results
