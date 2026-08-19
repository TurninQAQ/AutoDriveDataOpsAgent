#!/usr/bin/env python3
"""Create and load Airflow runtime secrets without storing them in Git.

The file written by this module is a shell environment fragment with mode 0600.
It intentionally never prints secret values. Existing files are preserved so a
repeated install does not rotate encryption keys implicitly.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
import stat
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet


SECRET_NAMES = (
    "AIRFLOW_FERNET_KEY",
    "AIRFLOW_API_SECRET_KEY",
    "AIRFLOW_JWT_SECRET",
)


def _generate(name: str) -> str:
    if name == "AIRFLOW_FERNET_KEY":
        return Fernet.generate_key().decode("ascii")
    return secrets.token_urlsafe(48)


def _read(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in SECRET_NAMES:
            values[name] = value.strip().strip("'\"")
    return values


def _validate(values: dict[str, str]) -> dict[str, str]:
    missing = [name for name in SECRET_NAMES if not values.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"runtime secret file is incomplete: {', '.join(missing)}")
    return {name: values[name].strip() for name in SECRET_NAMES}


def load(path: Path) -> dict[str, str]:
    return _validate(_read(path))


def _atomic_write(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for name in SECRET_NAMES:
                handle.write(f"export {name}={shlex.quote(values[name])}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def ensure(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Load an existing secret file or create it once from explicit env values."""

    path = Path(path)
    if path.exists():
        return load(path)
    environ = environ or os.environ
    values = {
        name: str(environ.get(name, "")).strip() or _generate(name)
        for name in SECRET_NAMES
    }
    _atomic_write(path, values)
    return load(path)


def rotate(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Explicitly create a new secret set; callers must handle Fernet migration."""

    environ = environ or os.environ
    values = {
        name: str(environ.get(name, "")).strip() or _generate(name)
        for name in SECRET_NAMES
    }
    _atomic_write(Path(path), values)
    return load(Path(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ensure", "rotate"))
    parser.add_argument("--path", required=True, type=Path)
    args = parser.parse_args()
    (ensure if args.command == "ensure" else rotate)(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
