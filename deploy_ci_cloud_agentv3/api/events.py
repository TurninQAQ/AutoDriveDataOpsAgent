from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, AsyncIterator

from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore

_TERMINAL = {"FINAL_RESPONSE", "ERROR"}


class EventBroker:
    def __init__(self) -> None:
        self._queues: dict[str, set[asyncio.Queue]] = defaultdict(set)

    async def publish(self, run_id: str, event: dict[str, Any]) -> None:
        for queue in list(self._queues.get(run_id, set())):
            await queue.put(event)

    async def subscribe(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[run_id].add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("event_type") in _TERMINAL:
                    return
        finally:
            self._queues[run_id].discard(queue)


def encode_sse(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "event").lower()
    return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
