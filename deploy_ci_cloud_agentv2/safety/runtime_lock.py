"""Cross-process ownership for the single-instance Runtime deployment model."""

from __future__ import annotations

from contextlib import nullcontext
import errno
import fcntl
from pathlib import Path
from typing import Any


class RuntimeInstanceAlreadyActive(RuntimeError, ValueError):
    """Another OS process currently owns this Runtime instance."""

    code = "RUNTIME_INSTANCE_ALREADY_ACTIVE"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(
            f"{self.code}: another Runtime process owns {self.path}; "
            "a mutation may be in flight"
        )


class RuntimeInstanceLock:
    """Non-blocking, process-scoped advisory lock for one runtime root."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> "RuntimeInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeInstanceAlreadyActive(self.path) from exc
            raise
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def runtime_ownership(path: str | Path | None, *, enabled: bool = True):
    """Return the operation ownership context for one Runtime instance."""
    if not enabled or path is None:
        return nullcontext()
    return RuntimeInstanceLock(path)
