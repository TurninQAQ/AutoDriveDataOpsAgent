from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    thread_id: str
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    pending_action: dict[str, Any] | None
    last_write_result: dict[str, Any] | None
    prepared_artifact: dict[str, Any] | None
    approved_fingerprint: str | None
    review_route: str | None
    final_response: dict[str, Any] | None
    step_count: int
