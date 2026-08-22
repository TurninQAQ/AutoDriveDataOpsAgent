"""Minimal immutable trace store for Phase B."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class EventProvenance:
    model_version: str
    prompt_version: str
    tool_catalog_hash: str
    operating_principles_version: str
    operating_principles_hash: str
    policy_version: str = "read-only-v2"


@dataclass(frozen=True)
class Event:
    event_id: str
    sequence_no: int
    request_id: str
    thread_id: str
    causation_id: str | None
    timestamp: datetime
    event_type: str
    payload: dict[str, Any]
    provenance: EventProvenance


class EventStore:
    """In-memory idempotent append store; suitable for the Phase B test host."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._by_id: dict[str, Event] = {}
        self._lock = threading.Lock()

    def append(
        self,
        *,
        event_type: str,
        request_id: str,
        thread_id: str,
        payload: dict[str, Any],
        provenance: EventProvenance,
        causation_id: str | None = None,
        event_id: str | None = None,
    ) -> Event:
        stable_id = event_id or f"evt_{uuid.uuid4().hex}"
        with self._lock:
            existing = self._by_id.get(stable_id)
            if existing is not None:
                return existing
            event = Event(
                event_id=stable_id,
                sequence_no=len(self._events) + 1,
                request_id=request_id,
                thread_id=thread_id,
                causation_id=causation_id,
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                payload=payload,
                provenance=provenance,
            )
            self._events.append(event)
            self._by_id[stable_id] = event
            return event

    def for_thread(self, thread_id: str) -> tuple[Event, ...]:
        with self._lock:
            return tuple(event for event in self._events if event.thread_id == thread_id)

    def all(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    def readable_trace(self, thread_id: str) -> list[dict[str, Any]]:
        rows = []
        for event in self.for_thread(thread_id):
            rows.append(
                {
                    "sequence_no": event.sequence_no,
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "causation_id": event.causation_id,
                    "timestamp": event.timestamp.isoformat(),
                    "payload": event.payload,
                    "provenance": asdict(event.provenance),
                }
            )
        return rows


def catalog_fingerprint(catalog: list[dict[str, Any]]) -> str:
    encoded = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
