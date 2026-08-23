from __future__ import annotations

from pathlib import Path

from deploy_ci_cloud_agentv2.config import ConfigurationError, RuntimeConfig
from deploy_ci_cloud_agentv2.host import health, readiness


def test_runtime_root_environment_controls_default_sqlite_location(tmp_path):
    runtime_root = tmp_path / "runtime"
    config = RuntimeConfig.from_env(
        runtime_root="/ignored/default",
        environ={
            "AUTODRIVE_RUNTIME_ROOT": str(runtime_root),
            "AUTODRIVE_PRINCIPLES_PATH": str(Path(__file__).parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md"),
        },
    )
    assert config.persistence.runtime_root == runtime_root.resolve()
    assert config.persistence.sqlite_path == (runtime_root / "state" / "autodrive.sqlite3").resolve()


def test_invalid_nested_config_is_configuration_error(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"provider": []}', encoding="utf-8")
    try:
        RuntimeConfig.from_env(config_path=config_file)
    except ConfigurationError as exc:
        assert "provider" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("malformed nested config was accepted")


def test_health_and_readiness_are_local_deterministic_checks(tmp_path):
    config = RuntimeConfig.from_env(
        runtime_root=tmp_path,
        environ={
            "AUTODRIVE_PRINCIPLES_PATH": str(Path(__file__).parents[2] / "doc" / "Luna_OPERATING_PRINCIPLES.md"),
        },
    )
    assert health(config)["status"] == "ok"
    ready = readiness(config)
    assert ready["status"] == "ready"
    assert ready["checks"]["tool_catalog_hash_stable"] is True
