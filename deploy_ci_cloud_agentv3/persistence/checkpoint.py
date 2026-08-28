from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any


class CheckpointerFactory:
    """Factory for official LangGraph checkpointers; does not implement checkpoint protocol."""

    @staticmethod
    @contextlib.asynccontextmanager
    async def open(backend: str, *, path: str | Path | None = None) -> AsyncIterator[Any]:
        backend = backend.strip().lower()
        if backend == "memory":
            try:
                from langgraph.checkpoint.memory import InMemorySaver
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("langgraph is required for memory checkpoints") from exc
            yield InMemorySaver()
            return
        if backend != "sqlite":
            raise ValueError("AUTODRIVE_CHECKPOINT_BACKEND must be memory or sqlite")
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install langgraph-checkpoint-sqlite for durable checkpoints") from exc
        if path is None:
            raise ValueError("sqlite checkpoint path is required")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Official saver owns serialization/checkpoint protocol. Strict msgpack is
        # requested through environment/configuration; no pickle fallback is introduced here.
        async with AsyncSqliteSaver.from_conn_string(str(target)) as saver:
            yield saver
