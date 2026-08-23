"""MCP-over-HTTP platform facade for the frozen V2 ToolCatalog.

The adapter owns transport concerns only.  It never chooses a semantic action
and never normalizes a response into evidence; Tool Runtime ingress performs
the canonical snapshot, typed result validation, provenance, and qualification.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx

from ..config import PlatformConfig
from ..tools.runtime import ReadFailure
from ..tools.write_runtime import MutationFailedBeforeEffect, MutationOutcomeUnknown
from .errors import MCPPlatformError


_READ_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class MCPPlatformFacade:
    """Concrete synchronous facade using JSON-RPC ``tools/call`` over HTTP.

    The frozen READ/WRITE facade protocol is synchronous because deterministic
    precondition and verification readers are shared by both paths. The
    Runtime still bounds every call and treats the adapter as an untrusted
    external transport boundary.
    """

    def __init__(
        self,
        config: PlatformConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._api_key = api_key or _resolve_optional_secret(config.api_key_env)

    def get_task_detail(self, task_name: str) -> Any:
        return self._read_call("get_task_detail", {"task_name": task_name})

    def get_gpu_pool(self) -> Any:
        return self._read_call("get_gpu_pool", {})

    def search_knowledge(self, query: str, top_k: int = 5) -> Any:
        return self._read_call("search_knowledge", {"query": query, "top_k": top_k})

    def get_queue_state(self, task_name: str | None = None) -> Any:
        return self._read_call("get_queue_state", {"task_name": task_name})

    def diagnose_task(self, task_name: str) -> Any:
        return self._read_call("diagnose_task", {"task_name": task_name})

    def resume_task(self, task_name: str) -> Any:
        return self._write_call("resume_task", {"task_name": task_name})

    def submit_task(self, task_name: str, config: Mapping[str, Any]) -> Any:
        return self._write_call("submit_task", {"task_name": task_name, "config": dict(config)})

    def stop_task(self, task_name: str) -> Any:
        return self._write_call("stop_task", {"task_name": task_name})

    def delete_task(self, task_name: str) -> Any:
        return self._write_call("delete_task", {"task_name": task_name})

    def set_task_priority(self, task_name: str, priority: int) -> Any:
        return self._write_call("set_task_priority", {"task_name": task_name, "priority": priority})

    def _read_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self._call(tool_name, arguments, is_write=False)

    def _write_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self._call(tool_name, arguments, is_write=True)

    def _call(self, tool_name: str, arguments: dict[str, Any], *, is_write: bool) -> Any:
        attempts = 0
        while True:
            try:
                response = self._post(tool_name, arguments)
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
                if is_write:
                    raise MutationOutcomeUnknown(f"platform transport outcome unknown for {tool_name}") from exc
                if attempts < self.config.max_retries:
                    attempts += 1
                    self._backoff(attempts)
                    continue
                raise ReadFailure("PLATFORM_TIMEOUT", "platform read transport timed out", retryable=True) from exc
            status = response.status_code
            if status in _READ_RETRYABLE_STATUS and not is_write and attempts < self.config.max_retries:
                attempts += 1
                self._backoff(attempts)
                continue
            if status >= 400:
                if is_write:
                    if status in _READ_RETRYABLE_STATUS:
                        raise MutationOutcomeUnknown(f"platform write response uncertain: HTTP {status}")
                    raise MutationFailedBeforeEffect(f"platform rejected write: HTTP {status}")
                raise ReadFailure(f"PLATFORM_HTTP_{status}", f"platform returned HTTP {status}", retryable=status in _READ_RETRYABLE_STATUS)
            try:
                payload = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                if is_write:
                    raise MutationOutcomeUnknown("platform returned invalid JSON after write") from exc
                raise ReadFailure("PLATFORM_MALFORMED_JSON", "platform returned invalid JSON") from exc
            if not isinstance(payload, Mapping):
                if is_write:
                    raise MutationOutcomeUnknown("platform returned a non-object response after write")
                raise ReadFailure("PLATFORM_MALFORMED_RESPONSE", "platform response must be an object")
            if payload.get("error") is not None:
                message = _safe_error_message(payload["error"])
                if is_write:
                    raise MutationFailedBeforeEffect(message)
                raise ReadFailure("PLATFORM_TOOL_ERROR", message, retryable=False)
            if "result" not in payload:
                if is_write:
                    raise MutationOutcomeUnknown("platform response has no result after write")
                raise ReadFailure("PLATFORM_MISSING_RESULT", "platform response has no result")
            return _unwrap_result(payload["result"])

    def _post(self, tool_name: str, arguments: dict[str, Any]) -> httpx.Response:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        timeout = httpx.Timeout(
            timeout=self.config.overall_timeout_seconds,
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
        )
        with httpx.Client(timeout=timeout, transport=self.transport) as client:
            return client.post(self.config.endpoint, headers=headers, json=request)

    def _backoff(self, attempt: int) -> None:
        if self.config.retry_backoff_seconds:
            time.sleep(self.config.retry_backoff_seconds * attempt)


def _unwrap_result(result: Any) -> Any:
    if not isinstance(result, Mapping):
        return result
    structured = result.get("structuredContent")
    if structured is not None:
        return structured
    if set(result) == {"data"}:
        return result["data"]
    # A JSON-RPC server may return the normalized business object directly;
    # leave all other mappings untouched for Result Contract validation.
    return result


def _safe_error_message(value: Any) -> str:
    if isinstance(value, Mapping):
        code = value.get("code", "PLATFORM_TOOL_ERROR")
        return f"{code}: platform tool call failed"
    return "platform tool call failed"


def _resolve_optional_secret(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        raise MCPPlatformError("platform_api_key_missing", "configured platform API key is unavailable")
    return value
