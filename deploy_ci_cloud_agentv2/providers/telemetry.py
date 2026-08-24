"""Provider telemetry that never stores prompts, secrets, or raw payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderTelemetryEvent:
    provider_name: str
    model_version: str
    request_id: str
    latency_ms: int
    input_chars: int
    output_chars: int
    retry_count: int
    error_class: str | None = None
    status_code: int | None = None
    regeneration_count: int = 0


class TelemetrySink(Protocol):
    def record(self, event: ProviderTelemetryEvent) -> None: ...


class InMemoryTelemetrySink:
    """Small test/diagnostic sink; production hosts may inject a metrics sink."""

    def __init__(self) -> None:
        self.events: list[ProviderTelemetryEvent] = []

    def record(self, event: ProviderTelemetryEvent) -> None:
        self.events.append(event)
