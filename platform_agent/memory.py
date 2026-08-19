from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import ConversationTurn


class ConversationStore:
    """Small file-backed thread memory independent from the LLM/runtime provider."""

    def __init__(self, root: str | Path, max_turns: int = 12):
        self.root = Path(root)
        self.max_turns = max(1, int(max_turns))

    def _path(self, thread_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in thread_id.strip())
        if not safe:
            safe = "default"
        if len(safe) > 96:
            digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:16]
            safe = f"{safe[:72]}-{digest}"
        return self.root / f"{safe}.jsonl"

    def load(self, thread_id: str) -> list[ConversationTurn]:
        path = self._path(thread_id)
        if not path.is_file():
            return []
        turns: list[ConversationTurn] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                turns.append(ConversationTurn.model_validate_json(line))
            except Exception:
                continue
        return turns[-self.max_turns :]

    def append(self, thread_id: str, turn: ConversationTurn) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(thread_id)
        existing = self.load(thread_id)
        existing.append(turn)
        existing = existing[-self.max_turns :]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            "\n".join(item.model_dump_json() for item in existing) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
