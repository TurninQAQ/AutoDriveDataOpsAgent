"""Loader and immutable snapshot for Luna's advisory operating guidance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class OperatingPrinciple:
    principle_id: str
    title: str
    text: str
    category: str = "advisory"


@dataclass(frozen=True)
class OperatingPrinciplesSnapshot:
    version: str
    content_hash: str
    loaded_at: datetime
    principles: tuple[OperatingPrinciple, ...]
    source_path: str


_HEADING = re.compile(
    r"^##\s+(?:\d+\.\s+)?Principle\s+(P\d+)\s+—\s+(.+?)\s*$"
)


def load_operating_principles(path: str | Path) -> OperatingPrinciplesSnapshot:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Operating Principles file is unavailable: {source}")
    content = source.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("Operating Principles file is empty")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    principles: list[OperatingPrinciple] = []
    current: tuple[str, str] | None = None
    body: list[str] = []
    for line in content.splitlines():
        match = _HEADING.match(line)
        if match:
            if current is not None:
                principles.append(
                    OperatingPrinciple(current[0], current[1], "\n".join(body).strip())
                )
            current = (match.group(1), match.group(2))
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        principles.append(
            OperatingPrinciple(current[0], current[1], "\n".join(body).strip())
        )
    if not principles:
        raise ValueError("Operating Principles file contains no principles")
    return OperatingPrinciplesSnapshot(
        version=f"sha256:{digest[:16]}",
        content_hash=digest,
        loaded_at=datetime.now(timezone.utc),
        principles=tuple(principles),
        source_path=str(source),
    )
