"""Immutable ownership identity for one request execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestIdentity:
    thread_id: str
    request_id: str
    turn_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("thread_id", self.thread_id),
            ("request_id", self.request_id),
            ("turn_id", self.turn_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

