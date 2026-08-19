from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

from cryptography.fernet import Fernet

from platform_observability.redaction import REDACTED, redact_text
from scripts import runtime_secrets


ROOT = Path(__file__).resolve().parents[1]
SECRET_KEYS = ("fernet_key", "sql_alchemy_conn", "secret_key", "jwt_secret")
RUNTIME_SECRET_KEYS = runtime_secrets.SECRET_NAMES


def test_repository_does_not_track_runtime_recovery_secrets():
    tracked = subprocess.check_output(
        ["git", "ls-files", "recover/airflow.cfg", "recover/simple_auth_manager_passwords.json.generated"],
        cwd=ROOT,
        text=True,
    )
    assert tracked.strip() == ""
    assert "recover/README.md" in subprocess.check_output(
        ["git", "ls-files", "recover/README.md"], cwd=ROOT, text=True
    )


def test_airflow_template_has_no_committed_secret_values():
    text = (ROOT / "config" / "airflow.cfg.base").read_text(encoding="utf-8")
    for key in SECRET_KEYS:
        matches = re.findall(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=(.*)$", text)
        assert matches, key
        assert all(not value.strip() for value in matches), key


def test_gitignore_covers_runtime_secret_artifacts():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for expected in ("**/platform.env", "**/runtime_secrets.env", "local_acceptance/", "recover/*", "!recover/README.md", "*.generated"):
        assert expected in text


def test_runtime_secret_generation_is_atomic_private_and_stable(tmp_path: Path):
    path = tmp_path / "config" / "runtime_secrets.env"
    first = runtime_secrets.ensure(path)
    assert set(first) == set(RUNTIME_SECRET_KEYS)
    assert all(first[name] for name in RUNTIME_SECRET_KEYS)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first_text = path.read_text(encoding="utf-8")

    second = runtime_secrets.ensure(path)
    assert second == first
    assert path.read_text(encoding="utf-8") == first_text


def test_runtime_secret_generation_honors_explicit_values_and_rotate(tmp_path: Path):
    path = tmp_path / "runtime_secrets.env"
    explicit = {
        "AIRFLOW_FERNET_KEY": Fernet.generate_key().decode("ascii"),
        "AIRFLOW_API_SECRET_KEY": os.urandom(24).hex(),
        "AIRFLOW_JWT_SECRET": os.urandom(24).hex(),
    }
    assert runtime_secrets.ensure(path, explicit) == explicit
    rotated = runtime_secrets.rotate(path)
    assert any(rotated[name] != explicit[name] for name in RUNTIME_SECRET_KEYS)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_redaction_covers_provider_credentials_and_common_secrets():
    samples = (
        "GEMINI_API_KEY=secret-value-1",
        "GOOGLE_API_KEY: secret-value-2",
        "DASHSCOPE_API_KEY=secret-value-2b",
        "X-goog-api-key=secret-value-3",
        "password=secret-value-4 secret=secret-value-5 token=secret-value-6",
    )
    for sample in samples:
        redacted = redact_text(sample)
        assert "secret-value" not in redacted
        assert REDACTED in redacted
