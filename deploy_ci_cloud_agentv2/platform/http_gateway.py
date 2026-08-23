"""Local HTTP transport bridge for the V2 platform facade.

The V2 Runtime speaks a deliberately small custom HTTP JSON-RPC contract.  The
existing AutoDrive platform, however, exposes its canonical tool implementation
through an MCP stdio server.  This module bridges those transports only:

    V2 Runtime -> HTTP JSON-RPC -> this bridge -> canonical MCP stdio server

It does not select tools, authorize writes, qualify evidence, or decide goal
completion.  The stdio command is supplied by the host through
``AUTODRIVE_STDIO_MCP_COMMAND`` (default: ``mcp-server``), so this package does
not depend on the legacy platform source tree or copy its business logic.
"""

from __future__ import annotations

import json
import os
import selectors
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_STDIO_COMMAND = "mcp-server"
DEFAULT_TIMEOUT_SECONDS = 45.0
MAX_REQUEST_BYTES = 1024 * 1024

V2_TOOL_NAMES = frozenset(
    {
        "get_task_detail",
        "get_gpu_pool",
        "search_knowledge",
        "get_queue_state",
        "diagnose_task",
        "resume_task",
        "submit_task",
        "stop_task",
        "delete_task",
        "set_task_priority",
    }
)


class GatewayError(Exception):
    """A safe, deterministic protocol or downstream transport error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(request_id: Any, code: str, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _success(request_id: Any, result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"structuredContent": result},
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GatewayError("INVALID_PARAMS", f"{label} must be an object")
    return value


def _stdio_command(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        value = os.environ.get("AUTODRIVE_STDIO_MCP_COMMAND", DEFAULT_STDIO_COMMAND)
    if isinstance(value, str):
        command = tuple(shlex.split(value))
    else:
        command = tuple(str(item) for item in value)
    if not command or not command[0].strip():
        raise GatewayError("CONFIGURATION_ERROR", "stdio MCP command is empty")
    return command


class StdioMCPClient:
    """One bounded MCP stdio session per call.

    A fresh process per request keeps the bridge stateless and avoids sharing a
    mutable protocol session across concurrent HTTP requests.  The canonical
    platform server remains the owner of tool behavior and platform state.
    """

    def __init__(
        self,
        command: str | Sequence[str] | None = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.command = _stdio_command(command)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        if tool_name not in V2_TOOL_NAMES:
            raise GatewayError("TOOL_NOT_FOUND", f"tool is not exposed: {tool_name}")
        _mapping(arguments, "arguments")
        forwarded_arguments = _transport_arguments(tool_name, arguments)

        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            raise GatewayError("PLATFORM_UNAVAILABLE", "stdio MCP server is unavailable") from exc

        try:
            initialize_id = uuid.uuid4().hex
            self._send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": initialize_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "autodrive-dataops-agent-v2", "version": "2.0.0"},
                    },
                },
            )
            self._read_response(process, initialize_id)
            self._send(
                process,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )
            call_id = uuid.uuid4().hex
            self._send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": forwarded_arguments},
                },
            )
            response = self._read_response(process, call_id)
            if "error" in response:
                raise GatewayError("PLATFORM_TOOL_ERROR", "platform tool call failed")
            return _unwrap_mcp_result(response.get("result"))
        except GatewayError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise GatewayError("PLATFORM_PROTOCOL_ERROR", "invalid stdio MCP response") from exc
        finally:
            self._close_process(process)

    def _send(self, process: subprocess.Popen[bytes], message: Mapping[str, Any]) -> None:
        if process.stdin is None:
            raise GatewayError("PLATFORM_PROTOCOL_ERROR", "stdio MCP input is unavailable")
        process.stdin.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
        process.stdin.flush()

    def _read_response(self, process: subprocess.Popen[bytes], expected_id: str) -> dict[str, Any]:
        if process.stdout is None:
            raise GatewayError("PLATFORM_PROTOCOL_ERROR", "stdio MCP output is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GatewayError("PLATFORM_TIMEOUT", "stdio MCP request timed out")
                if not selector.select(remaining):
                    raise GatewayError("PLATFORM_TIMEOUT", "stdio MCP request timed out")
                line = process.stdout.readline()
                if not line:
                    raise GatewayError("PLATFORM_UNAVAILABLE", "stdio MCP server closed its output")
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, Mapping):
                    raise GatewayError("PLATFORM_PROTOCOL_ERROR", "stdio MCP response is not an object")
                # Notifications and progress messages do not answer this request.
                if message.get("id") != expected_id:
                    continue
                return dict(message)
        finally:
            selector.close()

    @staticmethod
    def _close_process(process: subprocess.Popen[bytes]) -> None:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def _unwrap_mcp_result(result: Any) -> Any:
    if not isinstance(result, Mapping):
        return result
    structured = result.get("structuredContent")
    if structured is not None:
        return structured
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                try:
                    return json.loads(item["text"])
                except json.JSONDecodeError:
                    continue
    return result


def _transport_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt only transport-shape differences in the canonical MCP schema.

    V2 represents an unscoped queue read as ``task_name=None``.  The existing
    stdio tool declares a string default instead of a nullable field, so the
    bridge uses its documented empty-string form.  No semantic value is
    inferred and no WRITE argument is synthesized here.
    """

    forwarded = dict(arguments)
    if tool_name == "get_queue_state" and forwarded.get("task_name") is None:
        forwarded["task_name"] = ""
    if tool_name == "get_gpu_pool" and "cleanup_dead" not in forwarded:
        # The legacy MCP tool defaults this maintenance flag to true. V2
        # exposes this operation as a READ, so observation must not clean
        # stale reservation state.
        forwarded["cleanup_dead"] = False
    return forwarded


class GatewayDispatcher:
    def __init__(self, client: Any) -> None:
        self.client = client

    def dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            return _error(None, "INVALID_REQUEST", "request must be an object")
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return _error(request_id, "INVALID_REQUEST", "jsonrpc must be '2.0'")
        if request.get("method") != "tools/call":
            return _error(request_id, "METHOD_NOT_FOUND", "method is not supported")
        try:
            params = _mapping(request.get("params"), "params")
            name = params.get("name")
            if not isinstance(name, str) or not name.strip():
                raise GatewayError("INVALID_PARAMS", "params.name must be a non-empty string")
            if name not in V2_TOOL_NAMES:
                raise GatewayError("TOOL_NOT_FOUND", f"tool is not exposed: {name}")
            arguments = _mapping(params.get("arguments", {}), "params.arguments")
            return _success(request_id, self.client.call(name, _transport_arguments(name, arguments)))
        except GatewayError as exc:
            return _error(request_id, exc.code, exc.message)
        except Exception:
            # Do not expose provider/platform stack traces, paths, or secrets.
            return _error(request_id, "PLATFORM_TOOL_ERROR", "platform tool call failed")


def _handler_for(dispatcher: GatewayDispatcher) -> type[BaseHTTPRequestHandler]:
    class GatewayRequestHandler(BaseHTTPRequestHandler):
        server_version = "AutoDriveV2PlatformGateway/1.0"
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path != "/mcp":
                self._send_json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except (TypeError, ValueError):
                self._send_json(400, _error(None, "INVALID_REQUEST", "Content-Length is invalid"))
                return
            if content_length < 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, _error(None, "REQUEST_TOO_LARGE", "request body is too large"))
                return
            try:
                request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(200, _error(None, "PARSE_ERROR", "request body is not valid JSON"))
                return
            self._send_json(200, dispatcher.dispatch(request))

        def log_message(self, fmt: str, *args: Any) -> None:
            # Never log request bodies or authorization headers.
            sys.stderr.write("[autodrive-v2-gateway] " + (fmt % args) + "\n")

    return GatewayRequestHandler


def create_server(host: str, port: int, client: Any) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler_for(GatewayDispatcher(client)))


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    command: str | Sequence[str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    if host != DEFAULT_HOST and os.environ.get("AUTODRIVE_ALLOW_NONLOCAL_GATEWAY") != "1":
        raise RuntimeError("gateway is localhost-only by default")
    server = create_server(host, port, StdioMCPClient(command, timeout_seconds=timeout_seconds))
    print(f"AutoDrive V2 stdio-to-HTTP gateway listening on http://{host}:{port}/mcp", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    host = os.environ.get("AUTODRIVE_GATEWAY_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    try:
        port = int(os.environ.get("AUTODRIVE_GATEWAY_PORT", str(DEFAULT_PORT)))
        timeout = float(os.environ.get("AUTODRIVE_STDIO_MCP_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError as exc:
        raise SystemExit("gateway port and timeout must be numeric") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("AUTODRIVE_GATEWAY_PORT must be between 1 and 65535")
    serve(host, port, timeout_seconds=timeout)


if __name__ == "__main__":
    main()
