from __future__ import annotations

import contextlib
import inspect
from collections.abc import AsyncIterator
from typing import Any

from deploy_ci_cloud_agentv3.platform_backend.runtime import build_platform_facade
from deploy_ci_cloud_agentv3.mcp.profiles import AGENT_TOOLS, RUNTIME_TOOLS
from deploy_ci_cloud_agentv3.mcp.registry import ToolDefinition


def _server_class():
    try:
        from mcp.server import MCPServer
        return MCPServer
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer
            return MCPServer
        except ImportError:
            try:
                from mcp.server.fastmcp import FastMCP
                return FastMCP
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the official 'mcp' Python SDK") from exc


def _tool_handler(tool: ToolDefinition):
    """Create one SDK handler from the shared Pydantic ToolDefinition.

    __signature__ is synthesized from the args model so MCP tools/list and the
    in-process/native-function schemas share the same constraints.
    """
    async def handler(**kwargs: Any):
        return await tool.invoke(kwargs)

    params: list[inspect.Parameter] = []
    for name, field in tool.args_model.model_fields.items():
        annotation = field.rebuild_annotation()
        default = inspect.Parameter.empty if field.is_required() else field.default
        params.append(
            inspect.Parameter(
                name,
                # MCP arguments arrive as a named mapping and the handler
                # already accepts **kwargs.  Keyword-only parameters keep the
                # generated Python signature valid for any Pydantic field
                # order, including required fields after optional fields.
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )
    handler.__name__ = tool.name
    handler.__qualname__ = tool.name
    handler.__doc__ = tool.description
    handler.__signature__ = inspect.Signature(params, return_annotation=Any)  # type: ignore[attr-defined]
    return handler


def _register_profile(server: Any, registry: Any, allowed: set[str]) -> None:
    for tool in registry.list(allowed):
        handler = _tool_handler(tool)
        if hasattr(server, "add_tool"):
            server.add_tool(handler, name=tool.name, description=tool.description)
        else:  # pragma: no cover - legacy FastMCP compatibility
            server.tool(name=tool.name, description=tool.description)(handler)


def build_mcp_servers(facade: Any | None = None):
    """Build Agent/Runtime MCP profiles from one shared ToolDefinition registry."""
    from deploy_ci_cloud_agentv3.mcp.factory import build_tooling, assert_profile_boundaries

    facade = facade or build_platform_facade()
    tooling = build_tooling(facade)
    assert_profile_boundaries(tooling.registry)
    Server = _server_class()
    agent = Server("AutoDrive Agent MCP")
    runtime = Server("AutoDrive Runtime MCP")
    _register_profile(agent, tooling.registry, AGENT_TOOLS)
    _register_profile(runtime, tooling.registry, RUNTIME_TOOLS)
    return agent, runtime, tooling


def create_app(facade: Any | None = None):
    """Mount standard Streamable HTTP profiles at /mcp/agent and /mcp/runtime."""
    try:
        from starlette.applications import Starlette
        from starlette.routing import Mount
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("starlette is required for HTTP MCP serving") from exc

    agent, runtime, _ = build_mcp_servers(facade)

    def asgi(server):
        try:
            return server.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True)
        except TypeError:
            if hasattr(server, "settings"):
                server.settings.streamable_http_path = "/"
                server.settings.stateless_http = True
                server.settings.json_response = True
            return server.streamable_http_app()

    # streamable_http_app() must be called before session_manager is accessed.
    # Mounted Starlette sub-app lifespans are not run by the parent application,
    # so the host lifespan owns every mounted MCP session manager.
    agent_app = asgi(agent)
    runtime_app = asgi(runtime)

    @contextlib.asynccontextmanager
    async def lifespan(_app) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for server in (agent, runtime):
                try:
                    manager = server.session_manager
                except (AttributeError, RuntimeError):  # pragma: no cover - legacy SDK fallback
                    manager = None
                if manager is not None and hasattr(manager, "run"):
                    await stack.enter_async_context(manager.run())
            yield

    return Starlette(
        routes=[
            Mount("/mcp/agent", app=agent_app),
            Mount("/mcp/runtime", app=runtime_app),
        ],
        lifespan=lifespan,
    )


app = create_app()


def main():
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve MCP over HTTP") from exc
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
